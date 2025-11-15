/*
Kernels for residual connection forward pass.

OPERATION OVERVIEW:
Residual connections (skip connections) are a fundamental building block of transformers.
They add the input of a layer directly to its output:

  out = inp1 + inp2

Where typically:
- inp1: Output from a sublayer (e.g., attention or feedforward network)
- inp2: Input to that sublayer (the "residual" or "skip" connection)

Example in transformer block:
  x = embedding
  attn_out = attention(x)
  x = x + attn_out          # First residual connection
  ffn_out = feedforward(x)
  x = x + ffn_out           # Second residual connection

ROLE IN TRANSFORMER:
Residual connections are critical for training deep networks:

1. Gradient Flow: Allow gradients to flow directly backward through the network
   - Without residuals: Gradients must flow through many layers (can vanish)
   - With residuals: Gradients have a "highway" straight through
   - Enables training networks with 100+ layers

2. Identity Mapping: Network can learn to keep or modify features
   - If sublayer learns nothing useful, it can output zeros (keeping input)
   - If sublayer learns something useful, it adds to the input
   - This makes optimization easier - network starts with identity function

3. Ensemble Effect: Can be viewed as ensemble of shorter networks
   - Each residual path creates an implicit shorter network
   - Network learns multiple representations at different depths

WHY MULTIPLE KERNEL VERSIONS:
Residual addition is the simplest possible operation: out = a + b
Yet we still optimize it because it's called frequently in transformers.

- Version 1: Element-per-thread
             Each thread processes one element
             Simple but memory-bound

- Version 2: Vectorized 128-bit reads/writes
             Each thread processes 4-8 elements
             Better memory bandwidth (~1.5-2x speedup)
             Production-ready version

Even though it's just addition, memory bandwidth is the bottleneck.
Vectorization helps us approach peak bandwidth.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt residual_forward.cu -o residual_forward

version 1 is naive port from CPU code to kernel
./residual_forward 1
version 2 packs input into 128 bit memory reads
./residual_forward 2
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ENABLE_BF16
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// KERNEL 1: Element-wise residual addition
//
// ALGORITHM:
// - Each thread processes one element
// - Performs: out[i] = inp1[i] + inp2[i]
// - Converts to float32 for addition, then back to floatX (handles bfloat16/fp16)
//
// WHY CONVERT TO FLOAT32:
// Even though inp1 and inp2 might be bfloat16, we compute in float32:
// - Ensures numerical accuracy (bfloat16 has limited precision)
// - Modern GPUs have fast float32 ALUs
// - Only storage is in bfloat16 (saves memory bandwidth)
//
// PARALLELIZATION:
// - Launch N threads (one per element)
// - Fully independent - no thread communication
// - Perfect parallelism for element-wise operations
//
// MEMORY ACCESS PATTERN:
// - Each thread: 2 reads (inp1, inp2), 1 write (out)
// - Coalesced access when consecutive threads access consecutive elements
// - Memory bandwidth: 3 * sizeof(floatX) bytes per element
//
// PERFORMANCE CHARACTERISTICS:
// - Compute intensity: Very low (single addition)
// - Memory-bound: Limited by how fast we can read/write data
// - Arithmetic intensity: 1 FLOP / 3 memory ops ≈ 0.33 FLOP/byte
// - GPU utilization: Good (N threads, typically millions for transformers)
// - Best for: Small to medium tensors, or when vectorization not applicable
//
// WHY SO SIMPLE YET IMPORTANT:
// Transformers call this operation multiple times per layer, and have many layers.
// For GPT-3 scale (175B params, ~100 layers): billions of residual additions!
// Every microsecond saved here multiplies across the entire training run.
//
__global__ void residual_forward_kernel1(floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = (floatX)((float)inp1[idx] + (float)inp2[idx]);
    }
}

// KERNEL 2: Vectorized residual addition with 128-bit memory transactions
//
// ALGORITHM:
// - Same addition as kernel1: out = inp1 + inp2
// - Each thread processes x128::size elements (typically 4-8) at once
// - Uses vectorized loads and stores
//
// OPTIMIZATION TECHNIQUE - VECTORIZED MEMORY ACCESS:
// - load128cs(): Loads 128 bits from both inp1 and inp2
//   * 'cs' = cache streaming (bypass L1 cache)
//   * Good for data read only once
//   * Reduces cache pollution
// - store128(): Stores 128 bits to out
//   * Regular store (not streaming) to keep result in cache
//   * Next operation will likely need this data
//
// PARALLELIZATION:
// - Launch N/x128::size threads
// - Fewer threads than kernel1, but each does more work
// - Still maintains high GPU occupancy
//
// MEMORY ACCESS PATTERN:
// - Coalesced 128-bit aligned reads and writes
// - Memory controller serves 4-8x fewer transactions
// - Better utilization of memory bus width
// - Each thread: 2 vector reads, 1 vector write
//
// PERFORMANCE CHARACTERISTICS:
// - Memory bandwidth: Excellent (~1.5-2x better than kernel1)
// - Compute: Still minimal (just addition)
// - Arithmetic intensity: Slightly better (more FLOPs per memory transaction)
// - Best for: Large tensors in production (typical case)
// - Speedup vs kernel1: ~1.5-2x due to better memory efficiency
//
// WHY VECTORIZATION HELPS FOR ADDITION:
// Even though addition is trivial, memory is the bottleneck:
// - GPU can do billions of additions per second
// - GPU can only move hundreds of GB/s from memory
// - By loading 4-8 elements at once, we reduce memory transactions
// - This gets us closer to peak memory bandwidth
//
// LOOP INSIDE VECTORIZATION:
// The loop over k processes each element in the packed vector:
// - Unrolled by compiler (#pragma unroll could be added)
// - All additions can happen in parallel in registers
// - Only memory ops are the load128cs and store128 calls
//
__global__ void residual_forward_kernel2(floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx < N) {
        x128 packed_out;
        x128 packed_inp1 = load128cs(inp1 + idx);
        x128 packed_inp2 = load128cs(inp2 + idx);
        for (int k = 0; k < packed_inp1.size; ++k)
        {
            packed_out[k] = (floatX)((float)packed_inp1[k] + (float)packed_inp2[k]);
        }
        store128(out + idx, packed_out);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void residual_forward1(floatX* out, const floatX* inp1, const floatX* inp2, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    residual_forward_kernel1<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

void residual_forward2(floatX* out, const floatX* inp1, const floatX* inp2, int N, const int block_size) {
    const int grid_size = ceil_div(N, (int)(block_size * x128::size));
    residual_forward_kernel2<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void residual_forward(int kernel_num,
                  floatX* out,
                  const floatX* inp1,
                  const floatX* inp2,
                  int N,
                  int block_size) {
    switch (kernel_num) {
        case 1:
            residual_forward1(out, inp1, inp2, N, block_size);
            break;
        case 2:
            residual_forward2(out, inp1, inp2, N, block_size);
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
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* inp1 = make_random_float(B * T * C);
    float* inp2 = make_random_float(B * T * C);

    // move to GPU
    floatX* d_out;
    floatX* d_inp1;
    floatX* d_inp2;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp1, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp2, B * T * C * sizeof(floatX)));
    cudaCheck(memcpy_convert(d_inp1, inp1, B * T * C));
    cudaCheck(memcpy_convert(d_inp2, inp2, B * T * C));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    residual_forward_cpu(out, inp1, inp2, B * T * C);


    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        residual_forward(kernel_num, d_out, d_inp1, d_inp2, B * T * C, block_size);
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
        float elapsed_time = benchmark_kernel(repeat_times, residual_forward,
                                              kernel_num, d_out, d_inp1, d_inp2, B * T * C, block_size
                                              );

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 2 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 3 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(inp1);
    free(inp2);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp1));
    cudaCheck(cudaFree(d_inp2));

    return 0;
}
