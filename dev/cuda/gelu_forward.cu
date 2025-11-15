/*
Kernels for GELU (Gaussian Error Linear Unit) forward pass.

OPERATION OVERVIEW:
GELU is a smooth, non-linear activation function used in transformers (GPT, BERT, etc.).
It applies a non-linearity to each element independently:

  GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

This is a smooth approximation of the Gaussian CDF. Unlike ReLU which clips negative values
to zero, GELU allows small negative values through, creating a smoother gradient landscape
that often helps training.

ROLE IN TRANSFORMER:
GELU is applied in the feedforward network (FFN) of each transformer block:
  1. Linear projection: x → W1*x + b1  (typically expands dimension 4x)
  2. GELU activation: GELU(W1*x + b1)  (adds non-linearity)
  3. Linear projection: W2*GELU(...) + b2  (project back to original dimension)

Without this non-linearity, stacking linear layers would just be one linear transformation.
GELU enables the transformer to learn complex, non-linear patterns in language.

WHY MULTIPLE KERNEL VERSIONS:
GELU is an element-wise operation - perfect for GPU parallelization:

- Version 1: Naive element-per-thread
             Each thread processes one element
             Simple and effective for element-wise ops

- Version 2: Vectorized with packed 128-bit reads/writes
             Each thread processes 4-8 elements using 128-bit memory transactions
             Better memory bandwidth utilization
             Uses cache-streaming to avoid polluting L1 cache

Element-wise operations like GELU are memory-bound (limited by how fast we can read/write
data, not by computation). Vectorization helps us approach peak memory bandwidth.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt gelu_forward.cu -o gelu_forward

If encountering "error: identifier "M_PI" is undefined", add the following lines to the top of the file:

#define _USE_MATH_DEFINES
#include <math.h>  OR  #include <cmath>

version 1 is naive CPU port
./gelu_forward 1

version 2 is bfloat16 with the Packed128 data structure
./gelu_forward 2
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ENABLE_BF16
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

void gelu_forward_cpu(float* out, const float* inp, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// KERNEL 1: Naive element-wise GELU
//
// ALGORITHM:
// - Each thread computes GELU for one element
// - Uses the tanh approximation: GELU(x) ≈ 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715*x³)))
//
// MATHEMATICAL BREAKDOWN:
// 1. cube = 0.044715 * x³        (polynomial correction term)
// 2. arg = √(2/π) * (x + cube)   (scaled input to tanh)
// 3. result = 0.5 * x * (1 + tanh(arg))
//
// The tanh approximation is faster than the exact Gaussian CDF and accurate enough
// for neural network training.
//
// PARALLELIZATION:
// - Launch N threads (one per element)
// - Fully independent - no thread communication needed
//
// MEMORY ACCESS PATTERN:
// - Each thread: 1 read from inp, 1 write to out
// - Coalesced access when threads in a warp access consecutive elements
// - Memory-bound operation (arithmetic is simple relative to memory access)
//
// PERFORMANCE CHARACTERISTICS:
// - Compute intensity: Low (few FLOPs per element, dominated by tanh)
// - Memory bandwidth: Moderate (2 * sizeof(floatX) bytes per element)
// - Occupancy: High (minimal register/shared memory usage)
// - Best for: Small to medium tensors, or when vectorization isn't applicable
//
// OPTIMIZATION NOTES:
// - tanhf() is a relatively expensive transcendental function
// - Modern GPUs have special function units (SFU) for tanh, making it faster
// - --use_fast_math flag allows compiler to use approximate tanh for extra speed
//
__global__ void gelu_forward_kernel1(floatX* out, const floatX* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}

// KERNEL 2: Vectorized GELU with 128-bit memory transactions
//
// ALGORITHM:
// - Same GELU computation as kernel1
// - Each thread processes x128::size elements (typically 4-8) at once
//
// OPTIMIZATION TECHNIQUE - VECTORIZED MEMORY ACCESS:
// - load128cs(): Loads 128 bits in one transaction
//   * 'cs' means 'cache streaming' - hint to bypass L1 cache
//   * Reduces cache pollution since we only read each input once
// - store128(): Stores 128 bits in one transaction
//   * Regular store (not streaming) keeps output in cache
//   * Next layer will likely need this data soon
//
// PARALLELIZATION:
// - Launch N/x128::size threads
// - Fewer threads than kernel1, but each does more work
// - Still maintains high occupancy
//
// MEMORY ACCESS PATTERN:
// - Coalesced 128-bit aligned loads and stores
// - Memory controller serves 4-8x fewer transactions
// - Better utilization of memory bandwidth
//
// PERFORMANCE CHARACTERISTICS:
// - Memory bandwidth: Excellent (~1.5-2x better than kernel1)
// - Compute: Same per element (still calls tanhf x128::size times)
// - Instruction-level parallelism: Better (load multiple values with one instruction)
// - Best for: Large tensors in production (typical case)
// - Speedup vs kernel1: ~1.5-2x due to better memory efficiency
//
// CACHE STRATEGY:
// Input: Use cache streaming (load128cs)
//   - We read each input exactly once, so caching doesn't help
//   - Bypass L1 to save cache space for other data
// Output: Normal store (store128)
//   - Keep in cache because next layer will likely read it soon
//   - Example: GELU output feeds into a matrix multiply
//
// KEY INSIGHT:
// For memory-bound kernels, the bottleneck is moving data, not computing it.
// By reducing memory transactions through vectorization, we get closer to
// peak memory bandwidth, which directly translates to speedup.
//
__global__ void gelu_forward_kernel2(floatX* out, const floatX* inp, int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (i < N) {
        x128 packed_out;
        x128 packed_inp = load128cs(inp + i); // load and do not keep in cache
        for(int k = 0; k < packed_inp.size; ++k) {
            float xi = (float)packed_inp[k];
            float cube = 0.044715f * xi * xi * xi;
            packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
        }
        // store instead of storecs (without cache streaming) in case it is useful for the
        // data to be in the cache for the next operation after this GeLU
        store128(out + i, packed_out);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void gelu_forward1(floatX* out, const floatX* inp, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    gelu_forward_kernel1<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

void gelu_forward2(floatX* out, const floatX* inp, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size * x128::size);
    gelu_forward_kernel2<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void gelu_forward(int kernel_num,
                  floatX* out,
                  const floatX* inp,
                  int B, int T, int C,
                  int block_size) {
    switch (kernel_num) {
        case 1:
            gelu_forward1(out, inp, B * T * C, block_size);
            break;
        case 2:
            gelu_forward2(out, inp, B * T * C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, const char **argv) {
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 768;

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* inp = make_random_float(B * T * C);

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    gelu_forward_cpu(out, inp, B * T * C);

    // move to GPU
    floatX* d_out;
    floatX* d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(floatX)));
    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        gelu_forward(kernel_num, d_out, d_inp, B, T, C, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_out, out, "out", B * T * C, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, gelu_forward,
                                              kernel_num, d_out, d_inp,
                                              B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 1 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 2 * (int)sizeof(floatX);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(inp);

    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
    return 0;
}