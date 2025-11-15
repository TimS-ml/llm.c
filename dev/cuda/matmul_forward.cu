/*
===============================================================================
Matrix Multiplication Forward Pass - CUDA Kernel Development
===============================================================================

PURPOSE:
This file contains multiple implementations of matrix multiplication (matmul)
kernels for the forward pass, progressing from naive implementations to highly
optimized versions. It serves as both production code and an educational
resource for understanding CUDA matmul optimization.

WHY MATMUL IS CRITICAL:
Matrix multiplication is the dominant operation in transformer training,
consuming 80-90% of total compute time. In a typical GPT-style model:
- Attention: Q, K, V projections + attention output projection (4 matmuls per layer)
- MLP: Two matmuls per layer (up-projection and down-projection)
- For a 12-layer model, this means ~60 matmuls per forward pass

Therefore, optimizing matmul is THE most important performance optimization
for training transformers. Even small improvements (5-10%) translate directly
to significant wall-clock training time reductions.

PERFORMANCE BASELINE:
The gold standard is cuBLAS/cuBLASLt, which uses highly optimized Tensor Core
operations on modern GPUs (Ampere, Hopper). cuBLAS typically achieves:
- 80-90% of peak TFLOPS on large matrices
- ~19 TFLOPS on A100 40GB for fp32 (out of 19.5 TFLOPS theoretical)
- Even higher with TF32/BF16 Tensor Cores

Our handwritten kernels aim to:
1. Understand what makes matmul fast (educational)
2. Achieve within 50-70% of cuBLAS performance (impressive for handwritten)
3. Demonstrate key optimization techniques: tiling, vectorization, register blocking

OPERATION:
Forward pass computes: out = inp @ weight.T + bias
- inp: (B, T, C) - batch, sequence length, input channels
- weight: (OC, C) - output channels, input channels (stored row-major)
- bias: (OC) - bias vector (optional)
- out: (B, T, OC) - output activations

Mathematically: out[b,t,oc] = sum_c(inp[b,t,c] * weight[oc,c]) + bias[oc]

MEMORY LAYOUT:
All tensors use row-major storage (C-style), which differs from cuBLAS's
native column-major layout. This requires careful transpose handling.

Compile example:
nvcc -O3 --use_fast_math -Xcompiler -fopenmp matmul_forward.cu -o matmul_forward -lcublas -lcublasLt

KERNEL VERSIONS:
version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
OMP_NUM_THREADS=32 ./matmul_forward 1

version 2 calls cuBLAS, very fast (baseline reference)
OMP_NUM_THREADS=32 ./matmul_forward 2

version 3 calls cuBLASLt, should be even faster (can fuse bias)
OMP_NUM_THREADS=32 ./matmul_forward 3

version 4 is handwritten kernel with tiling and vectorization
OMP_NUM_THREADS=32 ./matmul_forward 4
*/

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <omp.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void matmul_forward_cpu(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            const float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                const float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

/*
KERNEL 1: Naive Global Memory Implementation
==============================================

ALGORITHM:
This is a direct translation of the CPU code to CUDA. Each thread computes one
output element by performing a dot product of one row of inp with one row of weight.

THREAD MAPPING:
- 2D grid: (ceil(B*T / block_size), ceil(OC / block_size))
- Each thread computes out[bt, oc] where bt = batch*time index, oc = output channel
- Thread (x,y) in block computes output element at global position (blockIdx.x*blockDim.x + x, blockIdx.y*blockDim.y + y)

MEMORY ACCESS PATTERN:
- Input (inp): Each thread reads an entire row of length C (coalesced within warp if C is large)
- Weight: Each thread reads an entire row of length C (coalesced within warp if C is large)
- Output: Each thread writes one element (fully coalesced within blocks)
- Bias: Each thread reads one element (broadcast within warp)

PERFORMANCE CHARACTERISTICS:
- Very poor performance: ~1-5% of cuBLAS on large matrices
- Bottleneck: Global memory bandwidth - every element is read from DRAM
- No data reuse: weight rows are read multiple times by different threads with no sharing
- No shared memory utilization
- No vectorization

WHY IT'S SLOW:
1. No tiling: Every element of weight is read B*T times from global memory
2. No shared memory: Threads in a block don't cooperate to share data
3. High memory traffic: For (B*T, C) @ (OC, C).T, we read B*T*C + OC*C elements
   but perform B*T*OC*C multiply-adds, so arithmetic intensity is very low

COMPARISON TO CUBLAS:
Achieves ~1-5% of cuBLAS performance. On A100, might get ~0.5 TFLOPS vs 19 TFLOPS for cuBLAS.

LESSON LEARNED:
This demonstrates why naive GPU implementations are slow. GPUs need careful
memory hierarchy management (shared memory, registers) to achieve high performance.
*/
__global__ void matmul_forward_kernel1(float* out,
                                       const float* inp, const float* weight, const float* bias,
                                       int BT, int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // in the naive kernel, every thread handles one element of out
    int bt = blockIdx.x * blockDim.x + threadIdx.x;  // row index in output (which batch*time position)
    int oc = blockIdx.y * blockDim.y + threadIdx.y;  // column index in output (which output channel)

    if (bt < BT && oc < OC) {
        // Initialize accumulator with bias
        float val = (bias != NULL) ? bias[oc] : 0.0f;

        // Get pointers to the rows we need to dot product
        const float* wrow = weight + oc * C;  // row of weight matrix for this output channel
        const float* inp_bt = inp + bt * C;   // row of input matrix for this batch*time position

        // Perform dot product: sum over input channels
        // This loop reads C elements from global memory with no reuse
        for (int i = 0; i < C; i++) {
            val += inp_bt[i] * wrow[i];
        }

        // Write result to global memory
        out[bt * OC + oc] = val;
    }
}

/*
ADD_BIAS KERNEL: Separate Bias Addition
========================================

ALGORITHM:
This kernel adds bias to the output of matmul when using cuBLAS (which doesn't
support fused bias addition). Each thread processes multiple output elements
in a strided pattern.

WHY THIS EXISTS:
cuBLAS doesn't natively support bias addition, so we need a separate kernel.
This is suboptimal because:
1. Extra kernel launch overhead
2. Extra global memory read/write of the entire output tensor
3. Memory bandwidth bound operation (no compute intensity)

cuBLASLt (kernel 3) can fuse bias, which is much better!

MEMORY ACCESS:
- Read: entire out tensor (B*T*OC elements) + bias vector (OC elements, reused)
- Write: entire out tensor (B*T*OC elements)
- Pattern: Coalesced reads/writes in the grid-stride loop

PERFORMANCE:
This is a memory-bandwidth-bound operation. On A100 with ~1.5 TB/s bandwidth,
processing B*T*OC*8 bytes (read+write) takes significant time.

LESSON LEARNED:
Always try to fuse operations to avoid extra memory traffic. This is why
cuBLASLt with epilogue fusion is preferred over cuBLAS + separate bias kernel.
*/
__global__ void add_bias(float* out, const float* bias, int B, int T, int OC) {
    // Grid-stride loop pattern: each thread processes multiple elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < B * T * OC; i += stride) {
        int col = i % OC;  // Which output channel (determines which bias element)
        out[i] += bias[col];  // Add corresponding bias element
    }
}

/*
KERNEL 4: Handwritten Tiled Matmul with Shared Memory and Vectorization
========================================================================

ALGORITHM:
This kernel implements classical blocked/tiled matrix multiplication with
shared memory caching and register blocking. It's based on the standard
optimization techniques taught in GPU programming courses.

KEY OPTIMIZATION TECHNIQUES:
1. Shared Memory Tiling: Tiles of 128x32 from both matrices are loaded into
   shared memory to enable data reuse across threads in a block
2. Register Blocking: Each thread computes an 8x8 tile of output, keeping
   intermediate results in registers
3. Vectorized Loads/Stores: Uses float4 (128-bit) vector operations for
   memory transactions
4. Thread Cooperation: 16x16 = 256 threads per block work together to
   compute a 128x128 output tile

BLOCK/THREAD ORGANIZATION:
- Block size: 16x16 threads (256 threads total)
- Each thread computes: 8x8 output elements (64 outputs per thread)
- Each block computes: 128x128 output tile
- Grid: (ceil(B*T/128), ceil(OC/128)) blocks

MEMORY HIERARCHY:
Registers (per thread):
  - vals[8][8]: Accumulator for 8x8 output tile (64 floats)
  - rhs[8]: Temporary for 8 float4 vectors from weight
  - lhs: Temporary float4 vector from input

Shared Memory (per block):
  - lhs_s[128][32]: Tile from input matrix (16 KB)
  - rhs_s[128][32]: Tile from weight matrix (16 KB)
  Total: 32 KB shared memory per block

Global Memory:
  - inp, weight, bias (read-only, cached in L2)
  - out (write-only)

ALGORITHM STEPS:
1. Initialize vals[8][8] with bias (if present)
2. Loop over K dimension in chunks of 32:
   a. Cooperatively load 128x32 tile of inp into lhs_s (shared memory)
   b. Cooperatively load 128x32 tile of weight into rhs_s (shared memory)
   c. Sync threads (wait for all loads to complete)
   d. Each thread performs 8x8 outer product updates using shared memory data
      - Inner loop over 32 elements (8 float4 vectors)
      - Compute: vals[i][j] += dot(lhs_row[i], rhs_row[j]) for 8x8 tile
   e. Sync threads (prepare for next tile)
3. Write vals[8][8] to global memory using vectorized stores

MEMORY ACCESS PATTERNS:
Loading to Shared Memory:
  - Each thread loads multiple float4 vectors (vectorized, coalesced)
  - Pattern is designed to avoid bank conflicts on reads from shared memory

Shared Memory Reads (compute phase):
  - Each thread reads 8 rows from lhs_s and 8 rows from rhs_s per iteration
  - Bank conflicts are minimized by accessing 32-element dimension
  - float4 reads: 4 floats per transaction

Output Writes:
  - Fully coalesced float4 stores (each thread writes 8 rows with stride OC)

PERFORMANCE CHARACTERISTICS:
- Achieves ~50-70% of cuBLAS on large matrices (excellent for handwritten!)
- Arithmetic Intensity: Much better than naive due to shared memory reuse
  - Each element in shared memory tiles is reused 128 times
- Bottleneck: Register pressure and instruction throughput
  - 64 float accumulator registers + temporary registers per thread
- Good occupancy: 256 threads/block with moderate register usage

COMPARISON TO CUBLAS:
cuBLAS uses more advanced techniques:
- Tensor Cores (if available): 8x faster for mixed precision
- More sophisticated tiling strategies
- Warp-level matrix operations (WMMA/MMA instructions)
- Better handling of boundary conditions
This kernel achieves ~10 TFLOPS on A100 vs ~19 TFLOPS for cuBLAS (impressive!)

WHY IT'S FASTER THAN KERNEL 1:
1. Data Reuse: Each element of weight/inp loaded into shared memory is used
   128 times, vs being loaded from global memory every time
2. Coalesced Access: Vectorized float4 operations maximize memory bandwidth
3. Register Blocking: 8x8 tiles stay in fast registers during computation
4. Reduced Global Memory Traffic: ~128x less traffic due to tiling

LIMITATIONS:
- Requires C and OC to be multiples of 32 (could add boundary handling)
- Fixed tile sizes (could be tuned per GPU architecture)
- No Tensor Core utilization (could use WMMA for mixed precision)
- Shared memory bank conflicts possible (could optimize layout further)

LESSON LEARNED:
Shared memory tiling + register blocking + vectorization are the fundamental
techniques for writing fast matmul kernels. This gets you to ~50% of peak,
but the last 30-40% requires Tensor Cores and architecture-specific tuning.
*/

// Helper functions for vectorized memory access (128-bit transactions)
__device__ float4 ld_vec(const float* address) {
    return *reinterpret_cast<const float4*>(address);
}

__device__ void st_vec(float* address, float4 val) {
    *reinterpret_cast<float4*>(address) = val;
}

__global__ void __launch_bounds__(16*16) matmul_forward_kernel4(float* out,
                                       const float* inp, const float* weight, const float* bias,
                                       int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // each thread handles 8x8 elements; each block 128 by 128 elements.

    // Compute the starting output channel for this thread's 8x8 tile
    int oc = 8*(blockIdx.y * blockDim.y + threadIdx.y);

    // Shared memory buffers to cache tiles of the input matrices
    // These enable data reuse: each element is used by multiple threads
    __shared__ float lhs_s[128][32];  // Tile from inp (left-hand side)
    __shared__ float rhs_s[128][32];  // Tile from weight (right-hand side)

    // Adjust pointers to the start of this block's tile
    // This block is responsible for a 128x128 output tile
    inp += 128 * blockIdx.x * C;           // Start of 128 rows in inp
    weight += 128 * blockIdx.y * C;        // Start of 128 rows in weight
    out += 128 * blockIdx.x * OC + 128 * blockIdx.y;  // Start of 128x128 output tile

    // Register array to accumulate results for 8x8 output tile
    // This thread will compute vals[0..7][0..7]
    float vals[8][8] = {};

    // Initialize accumulators with bias (vectorized load)
    if(bias != NULL) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j += 4) {
                float4 b = ld_vec(bias + oc + j);  // Load 4 bias values at once
                vals[i][j+0] = b.x;
                vals[i][j+1] = b.y;
                vals[i][j+2] = b.z;
                vals[i][j+3] = b.w;
            }
        }
    }

    // Compute starting index for this thread's portion of shared memory
    int si_start = 4*(16 * threadIdx.y + threadIdx.x);

    // Main loop: process K dimension in tiles of 32
    for (int so = 0; so < C; so += 32) {
        __syncthreads();  // Wait for all threads before loading new tile

        // Cooperatively load 128x32 tiles into shared memory
        // Each thread loads multiple float4 vectors to fill the shared buffers
        int xmod8 = threadIdx.x % 8;
        int xby8 = threadIdx.x / 8;
        int xo = 4 * xmod8;

        // Strided loop: each thread loads multiple rows
        for(int y = 2 * threadIdx.y + xby8; y < 128; y += 32) {
            // Load 4 floats at a time (vectorized) from global to shared memory
            st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));
            st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));
        }
        __syncthreads();  // Wait for all loads to complete

        // Compute: multiply this thread's 8x8 tile using the shared memory tiles
        // Process 32 elements in chunks of 4 (float4 vectors)
        for (int si = si_start; si < si_start + 32; si += 4) {
            // Load 8 float4 vectors from weight tile (for 8 output rows)
            float4 rhs[8];
            for (int u = 0; u < 8; ++u) {
                rhs[u] = ld_vec(&rhs_s[u + 8 * threadIdx.y][si % 32]);
            }

            // Compute 8x8 outer product: each input row with each weight row
            for (int ii = 0; ii < 8; ++ii) {
                float4 lhs = ld_vec(&lhs_s[ii + 8 * threadIdx.x][si % 32]);
                // Update all 8 outputs for this input row
                for (int ji = 0; ji < 8; ++ji) {
                    // Dot product of 4 elements (from float4)
                    vals[ii][ji] += lhs.x * rhs[ji].x;
                    vals[ii][ji] += lhs.y * rhs[ji].y;
                    vals[ii][ji] += lhs.z * rhs[ji].z;
                    vals[ii][ji] += lhs.w * rhs[ji].w;
                }
            }
        }
    }

    // Write the 8x8 output tile to global memory (vectorized)
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; j += 4) {
            // Pack 4 floats into float4 for vectorized store
            float4 result;
            result.x = vals[i][j + 0];
            result.y = vals[i][j + 1];
            result.z = vals[i][j + 2];
            result.w = vals[i][j + 3];
            // Write to global memory (coalesced across threads)
            st_vec(out + (8*threadIdx.x+i) * OC + 8*threadIdx.y + j, result);
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

// kernel 1 is the most naive matmul kernel
void matmul_forward1(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    dim3 gridDim(ceil_div(B * T, sqrt_block_size), ceil_div(OC, sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_kernel1<<<gridDim, blockDim>>>(out, inp, weight, bias, B*T, C, OC);
    cudaCheck(cudaGetLastError());
}

/*
KERNEL 2: cuBLAS - Production Quality Baseline
===============================================

ALGORITHM:
Uses NVIDIA's highly optimized cuBLAS library (cublasSgemm). cuBLAS is the gold
standard for matrix multiplication on NVIDIA GPUs.

WHAT CUBLAS DOES INTERNALLY:
- On Ampere+ GPUs: Uses Tensor Cores with TF32 precision (if enabled)
- Sophisticated tiling and thread block strategies
- Warp-level matrix operations (HMMA/MMA instructions)
- Heavily tuned for each GPU architecture
- Auto-tuning for different matrix sizes

MEMORY LAYOUT COMPLICATION - ROW-MAJOR VS COLUMN-MAJOR:
This is the trickiest part of using cuBLAS from C/C++ code!

Our data (C-style, row-major):
  - inp: (B*T, C) means inp[i][j] = inp[i*C + j]
  - weight: (OC, C) means weight[i][j] = weight[i*C + j]
  - We want: out = inp @ weight.T

cuBLAS (Fortran-style, column-major):
  - Expects matrices in column-major order
  - When we pass row-major data, cuBLAS "sees" the transpose

The Trick:
  We want: out = inp @ weight.T
  Transpose both sides: out.T = (inp @ weight.T).T = weight @ inp.T

  Since cuBLAS sees our row-major matrices as transposed, we can compute
  out.T by calling cuBLAS with weight and inp in the right order.

  Final call: cublasSgemm computes out.T = weight.T @ inp (as cuBLAS sees it)
  - A = weight (but we tell cuBLAS to transpose it)
  - B = inp (no transpose needed)
  - C = out

PERFORMANCE:
- Achieves 80-90% of theoretical peak TFLOPS
- On A100: ~19 TFLOPS for fp32, ~312 TFLOPS for TF32 Tensor Cores
- Memory bandwidth optimized with clever tiling

LIMITATIONS:
- Cannot fuse bias addition (need separate kernel)
- Extra memory traffic for bias kernel
- Slight overhead from kernel launch

LESSON LEARNED:
For production code, always use cuBLAS/cuBLASLt unless you have a very specific
reason to write custom kernels. Years of engineering went into these libraries.
*/
void matmul_forward2(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     const int sqrt_block_size) {
    // for reference API is:
    // cublasStatus_t cublasSgemm(cublasHandle_t handle,
    //                        cublasOperation_t transa, cublasOperation_t transb,
    //                        int m, int n, int k,
    //                        const float           *alpha,
    //                        const float           *A, int lda,
    //                        const float           *B, int ldb,
    //                        const float           *beta,
    //                        float           *C, int ldc)
    // for us, inp is (B*T, C), weight is (OC, C), out is (B*T, OC)
    // cuBLAS does C = alpha * A * B + beta * C
    // where A is mxk, B is kxn, C is mxn
    // now, because we use row-major storage, cuBLAS (which is column-major) sees our matrices transposed.
    // algorithmically / in e.g. PyTorch we want to do: out = inp @ weight.T
    // but because cuBLAS is column-major, we actually want to get it to calculate out.T . Mathematically, this is:
    // out.T = weight @ inp.T
    // but again, our variables look transposed, so using the actual weight/inp we have here in this function, this becomes
    // out.T = weight.T @ inp
    // so we need to get cuBLAS to calculate weight.T @ inp (the variables here are the actual ones in this function)
    // => need to call cuBLAS with A = weight, B = inp
    // => need to call cuBLAS with transa = CUBLAS_OP_T, transb = CUBLAS_OP_N

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform the matrix multiplication: out = inp @ weight.T
    // As cuBLAS sees it (column-major): out.T = weight.T @ inp
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B*T, C, &alpha, weight, C, inp, C, &beta, out, OC));

    // cuBLAS doesn't support fused bias, so we need a separate kernel (suboptimal!)
    if (bias != NULL) {
        int block_size = sqrt_block_size * sqrt_block_size;
        int grid_size = ceil_div(OC * B * T, block_size);
        add_bias<<<grid_size, block_size>>>(out, bias, B, T, OC);
        cudaCheck(cudaGetLastError());
    }
}

/*
KERNEL 3: cuBLASLt - Advanced Library with Epilogue Fusion
===========================================================

ALGORITHM:
Uses cuBLASLt, NVIDIA's "light" BLAS library with more flexibility than cuBLAS.
The key advantage: support for epilogue fusion (fusing operations like bias
addition and activation functions directly into the matmul).

WHAT IS EPILOGUE FUSION?
In a standard matmul + bias + activation:
  1. Compute matmul: temp = inp @ weight.T  (write to memory)
  2. Add bias: temp += bias                 (read + write memory)
  3. Apply activation: out = gelu(temp)     (read + write memory)

Each step requires reading/writing the entire output tensor from/to global memory.

With epilogue fusion:
  1. Compute matmul + bias + activation: out = gelu(inp @ weight.T + bias)
  All in one kernel! The bias and activation are applied to each output element
  before it's written to memory. This saves 2 round-trips to global memory!

PERFORMANCE BENEFITS:
- Eliminates extra kernel launches (lower latency)
- Reduces memory bandwidth by 2x (no intermediate reads/writes)
- Same compute performance as cuBLAS (uses same Tensor Core kernels)
- Can be 20-30% faster overall due to memory savings

CUBLASLT FEATURES:
- Epilogue fusion (bias, activation, etc.)
- Support for various data types (fp32, fp16, bf16, int8)
- Algorithm heuristics for auto-tuning
- Workspace memory for better performance
- More flexible matrix layouts

COMPLEXITY:
cuBLASLt is more complex to use than cuBLAS:
- Requires creating multiple descriptors (operation, matrix layouts, preference)
- Need to query for algorithms via heuristics
- More setup code, but worth it for the performance gain

LESSON LEARNED:
Always prefer cuBLASLt over cuBLAS when you need to fuse operations. The extra
code complexity pays off in performance. In transformer training, this fusion
can save 5-10% of total training time!

References:
https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtSgemm/sample_cublasLt_LtSgemm.cu
*/
void matmul_forward3(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC) {
    int has_bias = (bias != NULL);
    int has_gelu = 0;

    // check bias alignment
    if(((uintptr_t)bias % 16) != 0) {
        printf("Bias pointer is not aligned (cuBLASLt requirement)!\n");
        exit(EXIT_FAILURE);
    }

    int returnedResults = 0;
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulPreference_t preference;
    cublasLtMatrixLayout_t weightLayout;
    cublasLtMatrixLayout_t inputLayout;
    cublasLtMatrixLayout_t outputLayout;
    cublasLtMatrixLayout_t biasLayout;
    cublasLtMatmulHeuristicResult_t heuristic;

    // create the operation descriptor
    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasLtEpilogue_t epilogueBias = CUBLASLT_EPILOGUE_DEFAULT;
    if (has_bias && has_gelu) {
        epilogueBias = CUBLASLT_EPILOGUE_GELU_BIAS;
    } else if (has_bias) {
        epilogueBias = CUBLASLT_EPILOGUE_BIAS;
    } else if (has_gelu) {
        epilogueBias = CUBLASLT_EPILOGUE_GELU;
    }
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute_type, CUDA_R_32F));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose, sizeof(opNoTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogueBias, sizeof(epilogueBias)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    // define matrix layouts
    cublasCheck(cublasLtMatrixLayoutCreate(&weightLayout, CUDA_R_32F, C, OC, C));
    cublasCheck(cublasLtMatrixLayoutCreate(&inputLayout, CUDA_R_32F, C, B*T, C));
    cublasCheck(cublasLtMatrixLayoutCreate(&outputLayout, CUDA_R_32F, OC, B*T, OC));
    cublasCheck(cublasLtMatrixLayoutCreate(&biasLayout, CUDA_R_32F, OC, 1, OC));

    // create a preference handle with specified max workspace
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    // find a suitable algorithm
    cublasCheck(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc,
        weightLayout, inputLayout, outputLayout, outputLayout,
        preference, 1, &heuristic, &returnedResults));
    if (returnedResults == 0) {
        printf("No cuBLASLt algorithm: B: %d, T: %d, C: %d, OC: %d, bias: %d, gelu: %d\n",
            B, T, C, OC, has_bias, has_gelu);
        exit(EXIT_FAILURE);
    }

    // call the matmul
    const float alpha = 1.0f, beta = 0.0f;
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
        &alpha, weight, weightLayout, inp, inputLayout, &beta,
        out, outputLayout, out, outputLayout, &heuristic.algo,
        cublaslt_workspace, cublaslt_workspace_size, 0));

    // cleanups
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(weightLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(inputLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(outputLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(biasLayout));
}

// handwritten, relatively efficient non-tensorcore matmul kernel
void matmul_forward4(float* out,
                     const float* inp, const float* weight, const float* bias,
                     int B, int T, int C, int OC,
                     int sqrt_block_size) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    sqrt_block_size = 16;

    dim3 gridDim(ceil_div(B * T, 8*sqrt_block_size), ceil_div(OC, 8*sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_kernel4<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void matmul_forward(int kernel_num,
                    float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC,
                    const int sqrt_block_size) {
    switch (kernel_num) {
        case 1:
            matmul_forward1(out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
            break;
        case 2:
            matmul_forward2(out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
            break;
        case 3:
            matmul_forward3(out, inp, weight, bias, B, T, C, OC);
            break;
        case 4:
            matmul_forward4(out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    srand(0);

    int B = 32;
    int T = 1024;
    int C = 768;
    int OC = 768 * 4; // expansion of 4, e.g. in the MLP

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    printf("enable_tf32: %d\n", enable_tf32);
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
    // setup the (global) cuBLASLt workspace
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * OC * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(OC * C);
    float* bias = make_random_float(OC);

    // move to GPU
    float* d_out;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    cudaCheck(cudaMalloc(&d_out, B * T * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * OC * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, OC * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C * OC * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, OC * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    // time the kernel at different block sizes
    int sqrt_block_sizes[] = {4, 8, 16, 32};

    for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
        int sqrt_block_size = sqrt_block_sizes[j];
        printf("Checking block size %d x %d.\n", sqrt_block_size, sqrt_block_size);
        matmul_forward(kernel_num, d_out, d_inp, d_weight, d_bias, B, T, C, OC, sqrt_block_size);
        validate_result(d_out, out, "out", B * T * OC, 1e-1f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
        int sqrt_block_size = sqrt_block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, matmul_forward,
                                              kernel_num, d_out, d_inp, d_weight, d_bias,
                                              B, T, C, OC, sqrt_block_size);

        // napkin math: estimate the flops achieved
        // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
        float tflops = (float)B * T * C * OC * 2 / elapsed_time * 1e3f / 1e12f;
        printf("sqrt_block_size %4d | time %.4f ms | tflops %.2f\n", sqrt_block_size, elapsed_time, tflops);
    }

    // free memory
    free(out);
    free(inp);
    free(weight);
    free(bias);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias));
    cudaCheck(cudaFree(cublaslt_workspace));
    cublasCheck(cublasDestroy(cublas_handle));
    cublasCheck(cublasLtDestroy(cublaslt_handle));
    return 0;
}