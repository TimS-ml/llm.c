/*
Kernels for GELU (Gaussian Error Linear Unit) backward pass.

OPERATION OVERVIEW:
This computes the gradient of the GELU activation function for backpropagation.
Given the input (inp) and the gradient from the next layer (dout), we compute
the gradient with respect to the input (dinp).

The GELU function is: y = GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

The derivative requires the chain rule:
  d(GELU)/dx = ∂GELU/∂x

This involves derivatives of tanh, which gives us sech² terms:
  dinp[i] = local_grad * dout[i]
where local_grad is the derivative of GELU evaluated at inp[i]

ROLE IN TRANSFORMER:
During backpropagation, gradients flow backward through the network. After computing
gradients at the output of the feedforward network, we need to propagate them through
the GELU activation. The GELU backward pass:
1. Takes gradients from the layer above (dout)
2. Multiplies by the local gradient (GELU's derivative)
3. Passes the result to the layer below (dinp)

This is a critical step in training - it determines how much each input value should
change to reduce the loss. The smooth derivative of GELU (compared to ReLU's discontinuity)
can help gradients flow more smoothly through deep networks.

WHY MULTIPLE KERNEL VERSIONS:
Like the forward pass, this is an element-wise operation:

- Version 1: Element-per-thread approach
             Each thread handles one element
             Straightforward implementation of the derivative

- Version 2: Vectorized 128-bit memory operations
             Each thread processes 4-8 elements
             Better memory bandwidth utilization
             Similar pattern to forward pass optimization

The backward pass is also memory-bound, so vectorization provides similar speedups.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt gelu_backward.cu -o gelu_backward

If encountering "error: identifier "M_PI" is undefined", add the following lines to the top of the file:

#define _USE_MATH_DEFINES
#include <math.h>  OR  #include <cmath>

version 1 is naive port from CPU code to kernel
./gelu_backward 1

version 2 uses the Packed128 data structure
./gelu_backward 2
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ENABLE_BF16
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

void gelu_backward_cpu(float* dinp, const float* inp, const float* dout, const int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = (floatX)(local_grad * (float)dout[i]);
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// KERNEL 1: Element-wise GELU backward pass
//
// ALGORITHM:
// - Each thread computes the gradient for one element
// - Implements the derivative of GELU using the chain rule
//
// MATHEMATICAL DERIVATION:
// Given: y = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715*x³)))
// Let: arg = √(2/π) * (x + 0.044715*x³)
//
// Using product rule and chain rule:
// dy/dx = 0.5 * (1 + tanh(arg)) + 0.5 * x * sech²(arg) * d(arg)/dx
//
// where:
// - sech²(arg) = 1/cosh²(arg) = 1 - tanh²(arg)
// - d(arg)/dx = √(2/π) * (1 + 3*0.044715*x²)
//
// Combined: local_grad = 0.5*(1 + tanh(arg)) + 0.5*x*sech²(arg)*√(2/π)*(1 + 3*0.044715*x²)
// Final gradient: dinp = local_grad * dout
//
// PARALLELIZATION:
// - Launch N threads (one per element)
// - Each thread independently computes its gradient
// - No inter-thread communication needed
//
// MEMORY ACCESS PATTERN:
// - Reads: inp[i] and dout[i]
// - Writes: dinp[i]
// - Coalesced when consecutive threads access consecutive memory
//
// PERFORMANCE CHARACTERISTICS:
// - Compute intensity: Moderate (more complex than forward - includes cosh, tanh, divisions)
// - Memory bandwidth: Moderate (3 memory ops: 2 reads, 1 write)
// - More compute-bound than forward pass due to sech² calculation
// - Best for: Small to medium tensors
//
// NUMERICAL NOTES:
// - sech²(x) = 1/cosh²(x) is computed instead of 1-tanh²(x) for numerical stability
// - All intermediate calculations in float32 even if using bfloat16 storage
// - This ensures gradient accuracy during training
//
__global__ void gelu_backward1(floatX* dinp, const floatX* inp, const floatX* dout, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = (float)inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = (floatX)(local_grad * (float)dout[i]);
    }
}

// KERNEL 2: Vectorized GELU backward with 128-bit memory operations
//
// ALGORITHM:
// - Same gradient computation as kernel1
// - Each thread processes x128::size elements (typically 4-8) using vectorized loads/stores
//
// OPTIMIZATION TECHNIQUE - VECTORIZED MEMORY ACCESS:
// - load128cs(): Loads 128 bits from both inp and dout
//   * Cache streaming hint (bypasses L1) since we read data only once
//   * Reduces memory transactions by 4-8x
// - store128(): Writes 128 bits to dinp
//   * Regular store (not streaming) to keep gradients in cache
//   * Next backward pass layer may need this data
//
// PARALLELIZATION:
// - Launch N/x128::size threads
// - Each thread computes gradients for x128::size elements
// - Maintains high GPU occupancy
//
// MEMORY ACCESS PATTERN:
// - Coalesced 128-bit reads from inp and dout
// - Coalesced 128-bit writes to dinp
// - Memory controller processes fewer, larger transactions
//
// PERFORMANCE CHARACTERISTICS:
// - Memory bandwidth: Excellent (~1.5-2x better than kernel1)
// - Compute per thread: Higher (x128::size gradient calculations)
// - Still memory-bound overall (3 memory ops per element)
// - Best for: Large tensors (typical in transformer training)
// - Speedup vs kernel1: ~1.5-2x
//
// COMPUTE vs MEMORY BALANCE:
// Each thread does x128::size iterations of:
// - Multiple FLOPs (cube, tanh, cosh, divisions, multiplications)
// - This higher compute per memory transaction improves arithmetic intensity
// - Helps hide memory latency with useful computation
//
// KEY INSIGHT:
// Backward passes typically have higher arithmetic intensity than forward passes
// (they compute derivatives which involve more operations). Vectorization still
// helps because we're memory-bound, but the benefit might be slightly less than
// forward pass due to more balanced compute/memory ratio.
//
__global__ void gelu_backward2(floatX* dinp, const floatX* inp, const floatX* dout, const int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (i < N) {
        x128 packed_dinp;
        x128 packed_inp = load128cs(inp + i);
        x128 packed_dout = load128cs(dout + i);
        for (int k = 0; k < packed_inp.size; ++k) {
            float x = (float)packed_inp[k];
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = tanhf(tanh_arg);
            float coshf_out = coshf(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            packed_dinp[k] = (floatX)(local_grad * (float)packed_dout[k]);
        }

        store128(dinp + i, packed_dinp);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void gelu_backward1(floatX* dinp, const floatX* inp, const floatX* dout, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    gelu_backward1<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

void gelu_backward2(floatX* dinp, const floatX* inp, const floatX* dout, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size * x128::size);
    gelu_backward2<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void gelu_backward(int kernel_num,
                  floatX* dinp, 
                  const floatX* inp, 
                  const floatX* dout,
                  int B, int T, int C,
                  int block_size) {
    switch (kernel_num) {
        case 1:
            gelu_backward1(dinp, inp, dout, B * T * C, block_size);
            break;
        case 2:
            gelu_backward2(dinp, inp, dout, B * T * C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 768;

    // create host memory of random numbers
    float* dinp = (float*)malloc(B * T * C * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* dout = make_random_float(B * T * C);

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    gelu_backward_cpu(dinp, inp, dout, B * T * C);

    // move to GPU
    floatX* d_dinp;
    floatX* d_inp;
    floatX* d_dout;
    cudaCheck(cudaMalloc(&d_dinp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(floatX)));

    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));
    cudaCheck(memcpy_convert(d_dout, dout, B * T * C));

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        gelu_backward(kernel_num, d_dinp, d_inp, d_dout, B, T, C, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_dinp, dinp, "dinp", B * T * C, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, gelu_backward,
                                              kernel_num, d_dinp, d_inp, d_dout,
                                              B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 1 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 2 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(dinp);
    free(inp);
    free(dout);
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_dout));
    return 0;
}
