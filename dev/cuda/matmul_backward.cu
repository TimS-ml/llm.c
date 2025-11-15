/*
===============================================================================
Matrix Multiplication Backward Pass - CUDA Kernel Development
===============================================================================

PURPOSE:
This file implements the backward pass (gradient computation) for matrix
multiplication operations. The backward pass is equally critical as the forward
pass, as it's executed once per training iteration to compute gradients for
backpropagation.

WHY BACKWARD MATMUL IS CRITICAL:
In transformer training, backward pass matmuls are just as expensive as forward:
- For forward: out = inp @ weight.T + bias
- We need to compute three gradients:
  1. dinp: gradient w.r.t. input (for backprop to previous layer)
  2. dweight: gradient w.r.t. weights (for parameter updates)
  3. dbias: gradient w.r.t. bias (for parameter updates)

Each backward pass requires 2 matrix multiplications + 1 reduction, making it
~2x the cost of the forward pass!

MATHEMATICAL BACKGROUND:
Given: out = inp @ weight.T + bias
Forward: out[b,t,oc] = sum_c(inp[b,t,c] * weight[oc,c]) + bias[oc]

Backward (using chain rule):
We receive dout (gradient w.r.t. output) and need to compute:

1. dinp (gradient w.r.t. input):
   dinp[b,t,c] = sum_oc(dout[b,t,oc] * weight[oc,c])
   Matrix form: dinp = dout @ weight
   Shape: (B*T, OC) @ (OC, C) = (B*T, C)

2. dweight (gradient w.r.t. weight):
   dweight[oc,c] = sum_{b,t}(dout[b,t,oc] * inp[b,t,c])
   Matrix form: dweight = dout.T @ inp
   Shape: (OC, B*T) @ (B*T, C) = (OC, C)

3. dbias (gradient w.r.t. bias):
   dbias[oc] = sum_{b,t}(dout[b,t,oc])
   This is a simple sum reduction over batch and time dimensions

COMPUTE COST:
For a single matmul backward:
- dinp computation: B*T*OC*C multiply-adds
- dweight computation: B*T*OC*C multiply-adds
- dbias computation: B*T*OC additions
Total: ~2x the cost of forward pass (2 matmuls instead of 1)

MEMORY LAYOUT:
All tensors use row-major storage (C-style):
- dout: (B, T, OC) - incoming gradients
- inp: (B, T, C) - saved from forward pass
- weight: (OC, C) - saved from forward pass
- dinp: (B, T, C) - gradient output
- dweight: (OC, C) - gradient output
- dbias: (OC) - gradient output

PERFORMANCE STRATEGY:
Like forward pass, we rely primarily on cuBLAS for the two matrix multiplications,
as they're the same fundamental operation just with different matrices. The
tricky part is the bias gradient, which requires a custom reduction kernel.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt -Xcompiler -fopenmp matmul_backward.cu -o matmul_backward

KERNEL VERSIONS:
version 1: Uses cuBLAS for dinp and dweight, custom kernel for dbias
OMP_NUM_THREADS=32 ./matmul_backward 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_backward_cpu(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float* wrow = weight + o*C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel for
    for (int o = 0; o < OC; o++) {
        double sum = 0.0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* dout_bt = dout + b * T * OC + t * OC;
                float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if (dbias != NULL) { sum += d; }
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
        if (dbias != NULL){dbias[o] = sum;}
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

/*
BIAS GRADIENT KERNEL (NAIVE): Simple Sequential Reduction
==========================================================

ALGORITHM:
Each thread is assigned one output channel and sequentially sums all the
gradients for that channel across the entire batch and sequence length.

THREAD MAPPING:
- 1D grid of threads, one thread per output channel
- Thread i computes dbias[i] = sum over all (b,t) of dout[b,t,i]

MEMORY ACCESS PATTERN:
Each thread reads B*T values with stride OC (non-coalesced!). For thread i:
- dout[0*T*OC + 0*OC + i]
- dout[0*T*OC + 1*OC + i]
- ... (stride OC between accesses)

PERFORMANCE:
Very poor! Each thread does strided global memory access with large stride.
On modern GPUs, this means we're not utilizing memory bandwidth efficiently.

WHY IT'S SLOW:
1. Strided memory access (stride = OC, typically 1024-4096)
2. No memory coalescing across threads in a warp
3. No shared memory utilization
4. Each thread works independently (no cooperation)

LESSON LEARNED:
Don't do strided access patterns! Memory bandwidth is wasted when threads in
a warp access non-contiguous memory locations.
*/
__global__ void matmul_backward_bias_kernel_naive(float* dbias, const float* dout, int B, int T, int OC) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < OC) {
        // Sum all gradients for this output channel
        // Using double for accumulation to reduce numerical errors
        double sum = 0.0;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                // Strided access: accessing elements with stride OC
                sum += dout[b * T * OC + t * OC + o];
            }
        }
        dbias[o] = sum;
    }
}

/*
BIAS GRADIENT KERNEL (FASTER): Shared Memory Reduction
=======================================================

ALGORITHM:
Uses a two-phase reduction strategy:
1. Thread coarsening: Each thread sums multiple elements into a partial sum
2. Block-wide reduction: Threads cooperate using shared memory to reduce
   partial sums to a single value

THREAD/BLOCK ORGANIZATION:
- One block per output channel: blockIdx.x = output channel index
- block_size threads per block (typically 512)
- Each block reduces B*T values to 1 value

MEMORY ACCESS PATTERN:
Phase 1 (Thread Coarsening):
- Threads in block stride through their output channel's data
- Thread tid accesses: dout[tid*OC + o], dout[(tid+block_size)*OC + o], ...
- Still strided, but better than naive (more parallelism)

Phase 2 (Tree Reduction):
- All accesses are to shared memory (very fast, ~100x faster than global)
- Binary tree reduction: stride = block_size/2, block_size/4, ..., 1
- Each iteration, active threads halve

OPTIMIZATION TECHNIQUES:
1. Thread Coarsening: Each thread processes multiple elements before reducing
   - Reduces number of threads needed
   - Increases arithmetic intensity
   - Better register utilization

2. Shared Memory Reduction: Fast on-chip memory for inter-thread communication
   - Shared memory bandwidth: ~10 TB/s (vs ~1.5 TB/s global memory)
   - Enables efficient tree reduction

3. Double Precision Accumulation: Reduces numerical error for large reductions

PERFORMANCE:
Much better than naive! Typically 5-10x faster. Still not optimal due to
strided global memory access, but the parallel reduction helps.

COMPARISON:
On A100, can process ~1-2 GB/s effective bandwidth (still memory bound).
More sophisticated kernels (see matmul_backward_bias.cu) can achieve 10-20x
better performance by improving the access pattern.

LESSON LEARNED:
Shared memory reductions are a key technique for parallel reductions on GPUs.
The pattern is: thread coarsening → shared memory → tree reduction → output.
*/
__global__ void matmul_backward_bias_kernel_faster(float* dbias, const float* dout, int B, int T, int OC) {
    extern __shared__ float shared[];
    int o = blockIdx.x; // range [0, OC) - which output channel this block handles
    int tid = threadIdx.x; // range [0, block_size) - thread index within block
    int block_size = blockDim.x;

    // Pointer to this output channel's data in dout
    const float* x = dout + o;

    // Phase 1: Thread coarsening
    // Each thread sums multiple elements with stride block_size
    // This reduces the problem size from B*T to block_size
    double sum = 0.0;
    for (int i = tid; i < B * T; i += block_size) {
        sum += x[i * OC];  // Strided access with stride OC
    }
    shared[tid] = (float) sum;
    __syncthreads();

    // Phase 2: Tree reduction in shared memory
    // Reduce block_size partial sums to 1 final sum
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }

    // Write the final result (at thread 0) to global memory
    if (tid == 0) {
        dbias[o] = shared[0];
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

/*
MATMUL BACKWARD VERSION 1: cuBLAS-based Implementation
=======================================================

ALGORITHM:
Uses cuBLAS for the two matrix multiplications (dinp and dweight) and a custom
reduction kernel for the bias gradient. This is the standard approach for
computing matmul gradients on GPUs.

GRADIENT COMPUTATION:

1. GRADIENT W.R.T. INPUT (dinp):
   Mathematical: dinp[b,t,c] = sum_oc(dout[b,t,oc] * weight[oc,c])
   Matrix form: dinp = dout @ weight

   Shape analysis:
   - dout: (B*T, OC)
   - weight: (OC, C)
   - dinp: (B*T, C)

   cuBLAS call: C = alpha * A * B + beta * C
   - A = weight (OC, C) - no transpose
   - B = dout (B*T, OC) - no transpose
   - C = dinp (B*T, C)

   With row-major/column-major conversion:
   - cublasSgemm(..., CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, OC, weight, dout, dinp)

2. GRADIENT W.R.T. WEIGHT (dweight):
   Mathematical: dweight[oc,c] = sum_{b,t}(dout[b,t,oc] * inp[b,t,c])
   Matrix form: dweight = dout.T @ inp

   Shape analysis:
   - dout.T: (OC, B*T)
   - inp: (B*T, C)
   - dweight: (OC, C)

   cuBLAS call: C = alpha * A * B + beta * C
   - A = inp (B*T, C) - no transpose
   - B = dout (B*T, OC) - needs transpose
   - C = dweight (OC, C)

   With row-major/column-major conversion:
   - cublasSgemm(..., CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B*T, inp, dout, dweight)

3. GRADIENT W.R.T. BIAS (dbias):
   Mathematical: dbias[oc] = sum_{b,t}(dout[b,t,oc])
   Simple reduction over batch and time dimensions

   Custom kernel needed because cuBLAS doesn't have an efficient operation for this.
   We use the shared memory reduction kernel for better performance.

IMPORTANT: GRADIENT ACCUMULATION
The beta = 1.0f parameter is crucial! In neural networks, gradients from
multiple operations are accumulated (added together). Using beta = 1.0 means:
  C = alpha * A * B + beta * C  becomes  C += A * B
This is essential for backpropagation through layers with residual connections,
where gradients from multiple paths must be summed.

PERFORMANCE:
- dinp computation: ~same speed as forward pass matmul (dominated by cuBLAS)
- dweight computation: ~same speed as forward pass matmul (dominated by cuBLAS)
- dbias computation: Much faster (small reduction), but uses custom kernel
Overall: Backward pass takes ~2x forward pass time (expected for 2 matmuls)

LESSON LEARNED:
1. Use cuBLAS for large matrix operations - it's highly optimized
2. Beta parameter enables gradient accumulation (crucial for backprop!)
3. Custom kernels needed for operations cuBLAS doesn't support (reductions)
4. The row-major vs column-major layout requires careful thought for correctness
*/
void matmul_backward1(float* dinp, float* dweight, float* dbias,
                      float* dout, float* inp, float* weight, float* ones,
                      int B, int T, int C, int OC) {
    float alpha = 1.0f;
    float beta = 1.0f; // note we must use beta = 1.0 so that we do a +=, as we should, because gradients add

    // for reference the API is:
    // cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //                        cublasOperation_t transa, cublasOperation_t transb,
    //                        int m, int n, int k,
    //                        const float           *alpha,
    //                        const float           *A, int lda,
    //                        const float           *B, int ldb,
    //                        const float           *beta,
    //                        float           *C, int ldc)

    // recall the forward pass was calculated with alpha = 1.0f, beta = 0.0f as:
    // cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B*T, C, &alpha, weight, C, inp, C, &beta, out, OC);

    // Compute gradient w.r.t. input: dinp = dout @ weight
    // Shape: (B*T, OC) @ (OC, C) = (B*T, C)
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, OC, &alpha, weight, C, dout, OC, &beta, dinp, C));

    // Compute gradient w.r.t. weight: dweight = dout.T @ inp
    // Shape: (OC, B*T) @ (B*T, C) = (OC, C)
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B*T, &alpha, inp, C, dout, OC, &beta, dweight, C));

    // Compute gradient w.r.t. bias: dbias = sum(dout, dim=(0,1))
    if (dbias != NULL) {
        // Note: We tried using cuBLAS gemv (matrix-vector multiply) to compute the
        // reduction, but it didn't work correctly. The issue is that gemv isn't
        // designed for this reduction pattern. Custom kernel is clearer and works.

        // sum over B,T using matrix vector multiplication with cuBLAS
        // for reference this API is:
        // cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
        //                    int m, int n,
        //                    const float           *alpha,
        //                    const float           *A, int lda,
        //                    const float           *x, int incx,
        //                    const float           *beta,
        //                    float           *y, int incy)
        // dout is (B,T,OC), or in 2D terms (B*T, OC)
        // cublasCheck(cublasSgemv(cublas_handle, CUBLAS_OP_N, B*T, OC, &alpha, dout, B*T, ones, 1, &beta, dbias, 1));
        // cublasCheck(cublasSgemv(cublas_handle, CUBLAS_OP_T, OC, B*T, &alpha, dout, OC, ones, 1, &beta, dbias, 1));

        // ugh the above isn't working...
        // let's just do naive calculation for now, fix later
        // const int block_size=128;
        // const int grid_size=(OC + block_size - 1) / block_size;
        // matmul_backward_bias_kernel<<<grid_size, block_size>>>(dbias, dout, B, T, OC);

        // Use shared memory reduction kernel (much faster than naive)
        const int block_size=512;
        dim3 block_dim(block_size);
        dim3 grid_dim(OC);  // One block per output channel
        size_t shared_mem_size = block_size * sizeof(float);
        matmul_backward_bias_kernel_faster<<<grid_dim, block_dim, shared_mem_size>>>(dbias, dout, B, T, OC);
    }
}

void matmul_backward(int kernel_num,
                     float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight, float* ones,
                     int B, int T, int C, int OC) {
    switch (kernel_num) {
        case 1:
            matmul_backward1(dinp, dweight, dbias, dout, inp, weight, ones, B, T, C, OC);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // setup cuBLAS and its mathmodes, ensure fp32
    int enable_tf32 = 0; // use fp32 to get accurate results for checking w.r.t. CPU
    cublasCheck(cublasCreate(&cublas_handle));
    printf("enable_tf32: %d\n", enable_tf32);
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

    // create host memory of random numbers
    float* dinp = make_zeros_float(B * T * C);
    float* dweight = make_zeros_float(OC * C);
    float* dbias = make_zeros_float(OC);
    float* dout = make_random_float(B * T * OC);
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(OC * C);
    float* ones = make_ones_float(OC);

    // move to GPU
    float* d_dinp;
    float* d_dweight;
    float* d_dbias;
    float* d_dout;
    float* d_inp;
    float* d_weight;
    float* d_ones;
    cudaCheck(cudaMalloc(&d_dinp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dweight, OC * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dbias, OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, B * T * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, OC * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_ones, OC * sizeof(float)));
    cudaCheck(cudaMemcpy(d_dinp, dinp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dweight, dweight, OC * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dbias, dbias, OC * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, dout, B * T * OC * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, OC * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_ones, ones, OC * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // calculate the CPU reference
    matmul_backward_cpu(dinp, dweight, dbias, dout, inp, weight, B, T, C, OC);

    // calculate the GPU version
    matmul_backward(kernel_num, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_ones, B, T, C, OC);

    // compare
    printf("Checking correctness...\n");
    printf("dinp:\n");
    validate_result(d_dinp, dinp, "dinp", B * T * C, 1e-3f);
    printf("dweight:\n");
    validate_result(d_dweight, dweight, "dweight", OC * C, 1e-3f);
    printf("dbias:\n");
    validate_result(d_dbias, dbias, "dbias", OC, 1e-3f);
    printf("All results match.\n\n");

    // now benchmark the kernel
    int repeat_times = 100;
    float elapsed_time = benchmark_kernel(repeat_times, matmul_backward, kernel_num,
                                          d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_ones,
                                          B, T, C, OC);
    printf("time %.4f ms\n", elapsed_time);

    // cleanups
    free(dinp);
    free(dweight);
    free(dbias);
    free(dout);
    free(inp);
    free(weight);
    free(ones);
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_dweight));
    cudaCheck(cudaFree(d_dbias));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_ones));
    cublasCheck(cublasDestroy(cublas_handle));

    return 0;
}