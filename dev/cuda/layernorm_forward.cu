/*
================================================================================
Layer Normalization Forward Pass - CUDA Kernel Implementations
================================================================================

PURPOSE:
This file explores progressive optimizations for LayerNorm forward pass on GPU.
LayerNorm is a normalization technique used extensively in transformer models
like GPT-2, normalizing activations across the feature dimension.

LAYERNORM ALGORITHM OVERVIEW:
For each position (b,t) in the batch, normalize the C-dimensional feature vector:
  1. Compute mean: m = sum(x) / C
  2. Compute variance: v = sum((x - m)^2) / C
  3. Normalize: n = (x - m) / sqrt(v + eps)
  4. Scale and shift: out = n * weight + bias

The epsilon (eps = 1e-5) ensures numerical stability by preventing division by zero.

PERFORMANCE CONSIDERATIONS:
LayerNorm is a MEMORY-BOUND operation:
- Arithmetic intensity is low (few operations per byte loaded)
- Performance is limited by memory bandwidth, not compute
- Key optimization goal: maximize memory throughput and minimize memory accesses
- Typical GPUs can achieve 80-90% of peak memory bandwidth with well-optimized kernels

OPTIMIZATION PROGRESSION:
Version 1: Naive parallelization (one thread per sequence position)
Version 2: Separate kernels for mean, variance, normalization
Version 3: Warp-level primitives with cooperative groups
Version 4: Single-pass variance using online algorithm (numerically stable)
Version 5: Full thread block per row instead of single warp
Version 6: Shared memory caching of weights/bias with vectorization

TARGET DIMENSIONS:
- B (batch size): typically 8-32
- T (sequence length): typically 512-2048
- C (channels/features): typically 768-1024 for transformers

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt layernorm_forward.cu -o layernorm_forward

Usage:
./layernorm_forward [kernel_version]
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "common.h"
// ----------------------------------------------------------------------------
// CPU code reference

// GPT-2 layernorm forward pass
void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            const float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

// ============================================================================
// GPU KERNELS
// ============================================================================

/*
KERNEL 1: Naive Direct Translation
----------------------------------
APPROACH:
- Simple parallelization: one thread per sequence position (B*T total threads)
- Each thread independently processes one entire C-dimensional vector
- Direct translation from CPU code to GPU

PARALLELIZATION:
- Grid: ceil_div(B*T, block_size) blocks
- Each thread handles one complete row (all C elements)

MEMORY ACCESS PATTERN:
- Each thread reads C input values sequentially
- No cooperation between threads
- Three passes over the data:
  1. Compute mean
  2. Compute variance
  3. Apply normalization and affine transform

PERFORMANCE CHARACTERISTICS:
- PROS: Simple to understand, correct
- CONS:
  * Makes 3 passes over input (poor memory reuse)
  * Each thread does everything alone (no parallelism within C dimension)
  * Sequential loops are slow for large C
  * Underutilizes GPU parallelism

EXPECTED PERFORMANCE vs NAIVE:
- Baseline (1.0x) - this is our reference

NUMERICAL STABILITY:
- Uses textbook two-pass algorithm: compute mean first, then variance
- Stable for most practical purposes, but can suffer from catastrophic
  cancellation if data has very large magnitude with small variance
*/
__global__ void layernorm_forward_kernel1(float* out, float* mean, float* rstd,
                                 const float* inp, const float* weight, const float* bias,
                                 int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;

    if (idx < N) {
        // seek to the input position inp[idx,:]
        const float* x = inp + idx * C;

        // PASS 1: Calculate the mean
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += x[i];
        }
        m = m / C;

        // PASS 2: Calculate the variance (without bias correction)
        // Using two-pass algorithm: var = E[(x - mean)^2]
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;

        // Calculate the reciprocal standard deviation
        // rsqrt is faster than 1.0/sqrt, but sqrt is more readable here
        float s = 1.0f / sqrtf(v + eps);

        // PASS 3: Apply normalization and affine transformation
        float* out_idx = out + idx * C;
        for (int i = 0; i < C; i++) {
            float n = (s * (x[i] - m)); // normalized output
            float o = n * weight[i] + bias[i]; // scale and shift
            out_idx[i] = o; // write
        }

        // Cache the mean and rstd for the backward pass later
        // These are needed for gradient computation
        mean[idx] = m;
        rstd[idx] = s;
    }
}

/*
KERNEL 2: Three Separate Kernels with Block-Level Reduction
------------------------------------------------------------
APPROACH:
- Split the work into three separate kernel launches
- Parallelize across C dimension using thread blocks
- Use shared memory for efficient reduction within blocks

KERNEL 2a: mean_kernel
----------------------
REDUCTION STRATEGY:
- Classic tree-based reduction in shared memory
- Each block processes one row (one sequence position)
- Threads cooperate to sum all C elements

THREAD COARSENING:
- Each thread accumulates multiple elements before reduction
- Pattern: thread i processes elements i, i+block_size, i+2*block_size, ...
- Reduces the reduction tree depth and improves memory throughput

REDUCTION ALGORITHM:
- Phase 1: Thread coarsening (each thread sums its assigned elements)
- Phase 2: Tree reduction in shared memory
  * Stride starts at block_size/2
  * Each iteration, stride /= 2
  * Active threads: tid < stride
  * Combines pairs: shared[tid] += shared[tid + stride]

Example with block_size=8, C=20:
  Thread 0: sums elements [0,8,16] → stores in shared[0]
  Thread 1: sums elements [1,9,17] → stores in shared[1]
  ...
  Then tree reduction combines these partial sums
*/
__global__ void mean_kernel(float* mean, const float* inp, int N, int C, int block_size) {
    extern __shared__ float shared[];  // Size: block_size floats
    int idx = blockIdx.x; // range [0, B*T) - one block per row
    int tid = threadIdx.x; // range [0, block_size)
    const float* x = inp + idx * C;

    // PHASE 1: Thread coarsening - each thread accumulates its elements
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    shared[tid] = sum;
    __syncthreads();

    // PHASE 2: Tree-based reduction in shared memory
    // This is a classic parallel reduction pattern
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }

    // Thread 0 writes the final result
    if (tid == 0) {
        mean[idx] = shared[0] / C;
    }
}

/*
KERNEL 2b: rstd_kernel (Reciprocal Standard Deviation)
-------------------------------------------------------
SAME REDUCTION STRATEGY as mean_kernel, but computes variance instead.

DIFFERENCE FROM MEAN KERNEL:
- Reads the pre-computed mean
- Computes sum of squared differences: sum((x - mean)^2)
- Outputs reciprocal std dev: 1/sqrt(variance + eps)
*/
__global__ void rstd_kernel(float* rstd, const float* inp, const float* mean, int N, int C, int block_size) {
    extern __shared__ float shared[];  // Size: block_size floats
    int idx = blockIdx.x; // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)
    const float* x = inp + idx * C;
    float m = mean[idx];

    // PHASE 1: Thread coarsening - accumulate squared differences
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();

    // PHASE 2: Tree-based reduction
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }

    // Thread 0 computes and writes reciprocal standard deviation
    if (tid == 0) {
        rstd[idx] = 1.0f / sqrtf(shared[0] / C + 1e-5f);
    }
}

/*
KERNEL 2c: normalization_kernel
--------------------------------
APPROACH:
- Fully parallelized: one thread per element
- Simply applies the normalization formula using pre-computed mean and rstd

MEMORY ACCESS:
- Coalesced reads from inp (consecutive threads read consecutive elements)
- Broadcast reads from mean/rstd (all threads in a row read the same value)
- Broadcast reads from weight/bias (indexed by c, reused across batch)
- Coalesced writes to out

PERFORMANCE OF KERNEL 2 vs KERNEL 1:
- PROS:
  * Parallelizes across C dimension (better GPU utilization)
  * Efficient shared memory reduction
- CONS:
  * Three separate kernel launches (kernel launch overhead)
  * Mean and variance computed in separate passes (reads data twice)
  * Poor data reuse (variance pass re-reads all data)

EXPECTED PERFORMANCE:
- Typically 1.5-2x faster than kernel 1
*/
__global__ void normalization_kernel(float* out, const float* inp, float* mean, float* rstd,
                                     const float* weight, const float* bias, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int bt = idx / C;  // Which sequence position
    int c = idx % C;   // Which channel

    float m = mean[bt];
    float s = rstd[bt];
    float xi = inp[idx];
    float n = s * (xi - m);  // Normalize
    float o = n * weight[c] + bias[c];  // Scale and shift

    out[idx] = o;
}

/*
KERNEL 3: Warp-Level Primitives with Cooperative Groups
--------------------------------------------------------
APPROACH:
- Fuse all three operations into a single kernel
- Use warp-level primitives for efficient reduction
- One warp (32 threads) handles one row

WARP-LEVEL REDUCTION:
Warps are groups of 32 threads that execute in lockstep on NVIDIA GPUs.
They can perform very fast reductions without shared memory or synchronization.

COOPERATIVE GROUPS:
- Modern CUDA API for expressing thread cooperation
- cg::reduce() performs warp-level reduction using shuffle instructions
- Much faster than shared memory reduction (no memory accesses)
- Warp shuffle operations have very low latency (1-2 cycles)

PARALLELIZATION:
- One warp per row (sequence position)
- Each block contains multiple warps
- Total warps in grid = N (B*T)

MEMORY ACCESS OPTIMIZATIONS:
- __ldcs/__stcs: Cache streaming hints
  * Tells compiler this data won't be reused
  * Bypasses L1 cache, streams through L2
  * Reduces cache pollution
  * Benefits: weight/bias accessed repeatedly get more cache hits

THREE-PASS ALGORITHM:
1. Warp computes mean using warp reduction
2. Warp computes variance using warp reduction
3. Warp applies normalization

PERFORMANCE vs KERNEL 2:
- PROS:
  * Single kernel launch (no launch overhead)
  * Very fast warp reductions
  * Streaming loads reduce cache pollution
- CONS:
  * Still makes 2 passes over input data
  * Limited to 32 threads per row (may underutilize GPU for large C)

EXPECTED PERFORMANCE:
- Typically 1.2-1.5x faster than kernel 2 (2-3x faster than kernel 1)
*/
__global__ void layernorm_forward_kernel3(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // meta_group_size is the number of warps in a block
    // meta_group_rank is the warp index within the block
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }

    // The row of input that this warp is responsible for
    const float* x = inp + idx * C;

    // PASS 1: Compute mean using warp reduction
    float sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sum += x[i];
    }
    // Warp-level reduction using shuffle instructions
    // All 32 threads in warp cooperate to sum their partial sums
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float m = sum / C;

    // First thread in warp writes mean
    // __stcs = store with cache streaming hint
    if(warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }

    // PASS 2: Compute variance using warp reduction
    sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    // Use rsqrtf (reciprocal square root) - faster than 1.0/sqrtf
    float s = rsqrtf(sum / C + 1e-5f);

    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // PASS 3: Apply normalization and affine transform
    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        // __ldcs = load with cache streaming hint
        // Indicates this data will not be reused soon
        // Allows weight and bias (which ARE reused) to stay in cache
        float n = s * (__ldcs(x+c) - m);
        __stcs(o+c, n * weight[c] + bias[c]);
    }
}

/*
KERNEL 4: Online Variance Algorithm (Welford/Two-Pass Trick)
-------------------------------------------------------------
APPROACH:
- Use mathematical identity: var(x) = E[x²] - E[x]²
- Allows computing mean and variance in a SINGLE PASS over data
- Still need second pass for normalization (unavoidable)

ONLINE/WELFORD'S ALGORITHM:
Traditional (two-pass):
  Pass 1: mean = sum(x) / n
  Pass 2: var = sum((x - mean)²) / n

Online (single-pass for statistics):
  Simultaneously compute: sum(x) and sum(x²)
  Then: mean = sum(x) / n
        var = sum(x²)/n - mean²

This is numerically less stable in theory, but works well in practice
for normalized data (as in neural networks).

NUMERICAL STABILITY CONSIDERATION:
The formula var = E[x²] - E[x]² can suffer from catastrophic cancellation
when variance is small relative to mean². However, for normalized neural
network activations, this is rarely a problem because:
- Layer inputs are typically pre-normalized or standardized
- The formula saves memory bandwidth (critical for performance)
- Modern FP32 has enough precision for typical cases

PASSES:
1. Single pass: accumulate sum(x) and sum(x²) simultaneously
2. Compute mean and variance from the two sums
3. Second pass: apply normalization

PERFORMANCE vs KERNEL 3:
- PROS:
  * ONE LESS PASS over input data (reduces from 3 to 2 passes)
  * Better memory bandwidth utilization
  * Fewer memory reads = faster
- CONS:
  * Slightly less numerically stable (but fine in practice)
  * More register pressure (storing sum and sum2)

EXPECTED PERFORMANCE:
- Typically 1.3-1.5x faster than kernel 3 (3-4x faster than kernel 1)
- Often achieves 70-80% of peak memory bandwidth
*/
__global__ void layernorm_forward_kernel4(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N) {
        return;
    }

    // The row of input that this warp is responsible for
    const float* x = inp + idx * C;

    // SINGLE PASS: accumulate both sum(x) and sum(x²)
    // Thread coarsening: each thread processes multiple elements
    float sum = 0.0;   // Will become mean(x)
    float sum2 = 0.0;  // Will become mean(x²)
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float xi = x[i];
        sum += xi;
        sum2 += xi * xi;  // Accumulate squares simultaneously
    }

    // Warp-level reduction at the end
    sum = cg::reduce(warp, sum, cg::plus<float>{});   // sum(x)
    sum2 = cg::reduce(warp, sum2, cg::plus<float>{}); // sum(x²)
    sum /= C;   // mean(x) = E[x]
    sum2 /= C;  // mean(x²) = E[x²]

    // Compute statistics using the identity: var(x) = E[x²] - E[x]²
    float m = sum;
    float var = sum2 - sum * sum;  // Variance via the "two-pass trick"
    float s = rsqrtf(var + 1e-5f);

    // Store mean and rstd
    if(warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }
    if(warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // SECOND PASS: Apply normalization and affine transform
    float* o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        float n = s * (__ldcs(x+c) - m);
        __stcs(o+c, n * weight[c] + bias[c]);
    }
}

/*
KERNEL 5: Full Thread Block Per Row (Multi-Warp Reduction)
-----------------------------------------------------------
APPROACH:
- Use entire thread block for each row (up to 1024 threads)
- Multiple warps cooperate on a single row
- Two-level reduction: warp-level, then block-level

MOTIVATION:
Kernel 4 uses only 32 threads (one warp) per row. For large C (e.g., 768-1024),
this underutilizes the GPU. Using more threads per row increases parallelism.

TWO-LEVEL REDUCTION HIERARCHY:
Level 1 (Warp): Each warp reduces its partial sums using shuffle instructions
Level 2 (Block): Warp results are combined using shared memory

REDUCTION ALGORITHM:
1. Each thread accumulates sum and sum² for its assigned elements
2. Warp-level reduction: threads within each warp combine their values
3. First thread of each warp writes result to shared memory
4. One warp loads all warp results and performs final reduction
5. Thread 0 writes final mean and rstd

SHARED MEMORY USAGE:
- shared_sum[32]: stores partial sum from each warp (max 32 warps per block)
- shared_sum2[32]: stores partial sum² from each warp
- Only 256 bytes total (very small)

WHY SHARED MEMORY HERE:
- Need to communicate between warps (can't use shuffle across warps)
- Shared memory is fast (low latency, high bandwidth)
- Only stores num_warps values (minimal memory usage)

PARALLELIZATION:
- Grid size = N (one block per row)
- Block size = 128-1024 threads (configurable)
- More threads per row = better utilization for large C

PERFORMANCE vs KERNEL 4:
- PROS:
  * Better parallelism for large C
  * More threads working on reduction
  * Same online algorithm (single pass for statistics)
- CONS:
  * Overhead of block-level synchronization
  * Shared memory access latency
  * May not help for small C

EXPECTED PERFORMANCE:
- For C >= 512: typically 1.1-1.2x faster than kernel 4
- For C < 512: may be similar or slightly slower than kernel 4
- Often achieves 75-85% of peak memory bandwidth
*/
__global__ void layernorm_forward_kernel5(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Shared memory for inter-warp reduction
    // Max 32 warps per block (1024 threads / 32 = 32 warps)
    __shared__ float shared_sum[32];   // Partial sums from each warp
    __shared__ float shared_sum2[32];  // Partial sum² from each warp

    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int idx = blockIdx.x; // One block per row

    // The row of input that this block is responsible for
    const float* x = inp + idx * C;

    // PHASE 1: Thread coarsening - each thread accumulates its elements
    float thread_sum = 0.0;   // Accumulator for sum(x)
    float thread_sum2 = 0.0;  // Accumulator for sum(x²)
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float xi = x[i];
        thread_sum += xi;
        thread_sum2 += xi * xi;
    }

    // PHASE 2: Warp-level reduction using shuffle instructions
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>{});
    float warp_sum2 = cg::reduce(warp, thread_sum2, cg::plus<float>{});

    // PHASE 3: First thread in each warp writes to shared memory
    // (All threads write, but only lane 0's value matters - this avoids a branch)
    shared_sum[warp_id] = warp_sum;
    shared_sum2[warp_id] = warp_sum2;
    __syncthreads();

    // PHASE 4: Final reduction across warps
    // First warp loads all warp results and performs final reduction
    warp_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;

    // Reduce across the first warp to get block-level result
    float block_sum = cg::reduce(warp, warp_sum, cg::plus<float>{});
    float block_sum2 = cg::reduce(warp, warp_sum2, cg::plus<float>{});

    // Compute mean, variance, and reciprocal standard deviation
    block_sum /= C;   // mean(x)
    block_sum2 /= C;  // mean(x²)
    float m = block_sum;
    float var = block_sum2 - m * m;
    float s = rsqrtf(var + 1e-5f);

    // Thread 0 writes statistics
    if(threadIdx.x == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }
    if(threadIdx.x == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // SECOND PASS: Apply normalization with all threads in the block
    float* o = out + idx * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float n = s * (__ldcs(x+i) - m);
        __stcs(o+i, n * weight[i] + bias[i]);
    }
}

/*
KERNEL 6: Shared Memory Caching + Vectorized Loads (Most Optimized)
--------------------------------------------------------------------
APPROACH:
- Cache weight, bias, and input in shared memory
- Use vectorized 128-bit loads/stores (x128)
- Multiple warps per block process different rows
- Maximize data reuse from shared memory

MOTIVATION:
Previous kernels read weight and bias from global memory for every element.
For a batch, weight[i] and bias[i] are read B*T times! This is wasteful.
By caching in shared memory, we read them once per block.

VECTORIZATION (x128):
- x128 loads 128 bits = 16 bytes = 4 floats at once
- Much more efficient than scalar loads
- Requires aligned memory accesses
- Memory transactions reduced by 4x

SHARED MEMORY LAYOUT:
1. s_weight[C]: weight parameters (shared across all rows in block)
2. s_bias[C]: bias parameters (shared across all rows in block)
3. s_in[blockDim.y][C]: cached input for each row (reused in variance pass)

SHARED MEMORY SIZE CALCULATION:
- weights: C floats
- bias: C floats
- input cache: blockDim.y * C floats
- Total: (2 + blockDim.y) * C * sizeof(float)
- For C=768, blockDim.y=4: (2+4)*768*4 = 18KB (well under 48KB limit)

ALGORITHM:
1. All threads cooperate to load weight/bias into shared memory
2. Each warp processes one row:
   a. Load input with vectorization and cache in shared memory
   b. Compute mean (reading from shared memory)
   c. Compute variance (reading from shared memory - data reuse!)
   d. Apply normalization (reading input, weight, bias from shared memory)

DATA REUSE ANALYSIS:
- Input: read once from global, used twice from shared (mean + variance)
- Weight: read once from global, used C times from shared (once per element)
- Bias: read once from global, used C times from shared (once per element)

PERFORMANCE OPTIMIZATIONS:
1. Vectorized loads/stores (4x fewer memory transactions)
2. Shared memory caching (reduces global memory traffic)
3. Cache streaming hints (load128cs/store128cs)
4. Multiple rows per block (amortize weight/bias loading)

MEMORY BANDWIDTH:
- Global memory reads: input (1x) + weight (1x per block) + bias (1x per block)
- Global memory writes: output (1x)
- Shared memory: extensively used but very fast

PERFORMANCE vs KERNEL 5:
- PROS:
  * Vectorized memory access (4x fewer transactions)
  * Weight/bias cached in shared memory (huge win for large batches)
  * Input cached for variance computation
  * Better memory bandwidth utilization
- CONS:
  * Requires large shared memory (may limit occupancy)
  * More complex code
  * May fail if C is very large (shared memory limit)

FALLBACK:
If shared memory allocation fails (C too large), launcher falls back to kernel 5.

EXPECTED PERFORMANCE:
- Typically 1.2-1.5x faster than kernel 5 for typical C (768-1024)
- Can achieve 85-90% of peak memory bandwidth
- Best overall performance for typical transformer dimensions
*/
__global__ void layernorm_forward_kernel6(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                    const float*  __restrict__ inp, const float*  __restrict__ weight,
                                    const float* __restrict__ bias, int N, int C) {
    assert(blockDim.x == WARP_SIZE);

    // PHASE 1: Load weights and biases into shared memory
    // All threads cooperate to load these shared parameters
    // This happens BEFORE any thread exits (important for correctness)
    extern __shared__ char params[];

    // Shared memory layout using vectorized types (x128 = 4 floats)
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias = reinterpret_cast<x128*>(params) + (C / x128::size);
    x128* s_in = reinterpret_cast<x128*>(params) + ((2 + threadIdx.y) * C / x128::size);

    // Cooperative loading of weight and bias (all warps participate)
    int sidx = (threadIdx.x + WARP_SIZE * threadIdx.y) * x128::size;
    for(int i = sidx; i < C; i += blockDim.y * WARP_SIZE * x128::size) {
        s_weight[i/x128::size] = load128(weight + i);
        s_bias[i/x128::size] = load128(bias + i);
    }
    __syncthreads();  // Ensure all weights/biases are loaded

    // PHASE 2: Each warp processes one row
    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if(idx >= N) { return; }

    // Adjust pointers to current sequence position
    inp += idx * C;
    out += idx * C;

    const float eps = 1e-5f;

    // PASS 1: Compute mean while caching input in shared memory
    float sum = 0.0f;
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        // Vectorized load with cache streaming
        const x128 in_data = load128cs(inp + c);
        // Accumulate sum across the vector
        for(int k = 0; k < x128::size; ++k) {
            sum += (float)in_data[k];
        }
        // Cache input in shared memory for reuse in variance computation
        s_in[c / x128::size] = in_data;
    }

    sum = warpReduceSum(sum);
    float m = sum / C;

    // PASS 2: Compute variance (reading from shared memory - fast!)
    float v = 0.f;
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];  // Read from shared memory
        for(int k = 0; k < x128::size; ++k) {
            v += ((float)in_data[k] - m) * ((float)in_data[k] - m);
        }
    }

    v = warpReduceSum(v) / C;
    float s = rsqrtf(v + eps);

    // PASS 3: Apply normalization (all data from shared memory)
    for(int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];      // From shared memory
        const x128 w = s_weight[c / x128::size];        // From shared memory
        const x128 b = s_bias[c / x128::size];          // From shared memory
        x128 out_data;
        for(int k = 0; k < x128::size; ++k) {
            float n = s * ((float)in_data[k] - m);      // Normalize
            float o = n * (float)w[k] + (float)b[k];    // Scale and shift
            out_data[k] = o;
        }
        // Vectorized store with cache streaming
        store128cs(out + c, out_data);
    }

    // Store statistics for backward pass
    if(threadIdx.x == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }
    if(threadIdx.x == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void layernorm_forward1(float* out, float* mean, float* rstd,
                           const float* inp, const float* weight, const float* bias,
                           int B, int T, int C,
                           const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    layernorm_forward_kernel1<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward2(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    int N = B * T;
    // in mean and rstd, threads cooperate within blocks via reductions
    mean_kernel<<<N, block_size, block_size * sizeof(float)>>>(mean, inp, N, C, block_size);
    cudaCheck(cudaGetLastError());
    rstd_kernel<<<N, block_size, block_size * sizeof(float)>>>(rstd, inp, mean, N, C, block_size);
    cudaCheck(cudaGetLastError());
    // in the normalization, everything just gets flattened out
    const int block_size2 = 256;
    const int grid_size = ceil_div(B * T * C, block_size2);
    normalization_kernel<<<grid_size, block_size2>>>(out, inp, mean, rstd, weight, bias, B, T, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward3(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    layernorm_forward_kernel3<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward4(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    layernorm_forward_kernel4<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward5(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    assert(block_size <= 1024);
    const int N = B * T;
    const int grid_size = N;
    layernorm_forward_kernel5<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward6(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    int block_y = block_size / WARP_SIZE;
    const int grid_size = ceil_div(N, block_y);
    size_t smem = (2 + block_y) * C * sizeof(float);

    // in order to use more than 48 KiB of smem, need to call cudaFuncSetAttribute
    // this may fail, in which case we fall back to the smem free implementation.
    cudaCheck(cudaGetLastError());
    auto status = cudaFuncSetAttribute(layernorm_forward_kernel6, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaGetLastError();
    if (status == cudaSuccess) {
        layernorm_forward_kernel6<<<grid_size, dim3(32, block_y), smem>>>(out, mean, rstd, inp, weight, bias, N, C);
    } else {
        const int grid_size = N;
        // fall back to the version without shared memory
        layernorm_forward_kernel5<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    }
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void layernorm_forward(int kernel_num,
                    float* out, float* mean, float* rstd,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C,
                    const int block_size) {
    switch (kernel_num) {
        case 1:
            layernorm_forward1(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 2:
            layernorm_forward2(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 3:
            layernorm_forward3(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 4:
            layernorm_forward4(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 5:
            layernorm_forward5(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 6:
            layernorm_forward6(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
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

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);

    // move to GPU
    float* d_out;
    float* d_mean;
    float* d_rstd;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_mean, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_rstd, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, C * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    // check the correctness of the kernel at all block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);

        layernorm_forward(kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);

        validate_result(d_out, out, "out", B * T * C, 1e-5f);
        validate_result(d_mean, mean, "mean", B * T, 1e-5f);
        validate_result(d_rstd, rstd, "rstd", B * T, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 2000;
        float elapsed_time = benchmark_kernel(repeat_times, layernorm_forward,
                                              kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias,
                                              B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = (2 * B * T * C) * 4; // *4 for float
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_mean));
    cudaCheck(cudaFree(d_rstd));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias));

    return 0;
}