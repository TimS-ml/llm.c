/*
================================================================================
ATTENTION FORWARD PASS - MULTI-KERNEL PERFORMANCE EXPLORATION
================================================================================

PURPOSE:
This file implements multiple versions of the multi-head attention forward pass
for educational and benchmarking purposes. Each kernel version explores different
optimization strategies and performance tradeoffs in CUDA programming.

WHY MULTIPLE VERSIONS EXIST:
Attention is a critical bottleneck in transformer models, and optimizing it requires
understanding various CUDA programming techniques. This file demonstrates the
evolution from naive implementations to highly optimized kernels, showing how
different approaches impact performance:
- Memory access patterns (coalesced vs scattered)
- Computation patterns (parallel vs sequential)
- Use of hardware features (tensor cores, warp primitives, shared memory)
- Numerical precision tradeoffs (FP32 vs FP16/BF16)
- Fusion opportunities (combining operations to reduce memory traffic)

ATTENTION ALGORITHM OVERVIEW:
Multi-head attention computes weighted combinations of values based on query-key
similarity scores. For each position in a sequence:

1. SCORE COMPUTATION (QK^T):
   - Compute dot products between query at current position and all keys
   - Scale by 1/sqrt(head_dimension) for numerical stability
   - Result: attention scores (logits) for each position pair

2. SOFTMAX NORMALIZATION:
   - Apply softmax to scores to get attention weights (probabilities)
   - For autoregressive (causal) attention, mask future positions
   - Uses numerically stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))

3. WEIGHTED VALUE AGGREGATION (Attention @ V):
   - Multiply attention weights by corresponding value vectors
   - Sum weighted values to produce output for current position

4. MULTI-HEAD OPERATION:
   - Split input into multiple heads (different representation subspaces)
   - Run attention independently on each head
   - Concatenate head outputs

Input shape: (B, T, 3C) where B=batch, T=sequence_length, C=channels
- Contains Q, K, V packed together (3C = C for Q + C for K + C for V)
Output shape: (B, T, C)
Intermediate shapes: preatt/att are (B, NH, T, T) where NH=num_heads

COMPILATION:
With cuDNN:
nvcc -I/PATH/TO/cudnn-frontend/include -DENABLE_CUDNN -O3 --use_fast_math --lcublas -lcublasLt -lcudnn attention_forward.cu -o attention_forward

Without cuDNN:
nvcc -O3 --use_fast_math -lcublas -lcublasLt attention_forward.cu -o attention_forward

KERNEL VERSIONS:

VERSION 1: Naive GPU Port
./attention_forward 1
- Direct translation of CPU code to CUDA
- Parallelizes only over (batch, time, heads)
- Inner loops over head_size remain sequential within each thread
- Memory-bound: Poor memory access patterns, no coalescing
- Performance: Baseline (slowest)
- Use case: Understanding basic GPU parallelization

VERSION 2: Flash Attention (Minimal)
./attention_forward 2
- Implements tiling to keep data in SRAM (shared memory)
- Uses online softmax algorithm to avoid materializing full attention matrix
- Reduces HBM (global memory) accesses significantly
- UNFORTUNATELY: This implementation is ~3X slower than naive version
- Reason: Overhead from complex indexing and synchronization outweighs benefits
- Based on: https://github.com/tspeterkim/flash-attention-minimal
- Use case: Educational - shows that algorithmic improvements need careful implementation

VERSION 3: cuBLAS + Custom Softmax
./attention_forward 3
- Uses highly optimized cuBLAS library for matrix multiplications
- Custom kernel for softmax (efficient warp-level reductions)
- Pipeline: permute -> cuBLAS(QK^T) -> scale -> softmax -> cuBLAS(Att@V) -> unpermute
- Memory-bound but with optimized memory access patterns
- Performance: ~20X faster than version 1
- Use case: Production baseline - leverages vendor-optimized libraries

VERSION 4: Fused Operations + Online Softmax
./attention_forward 4
- Fuses scaling into softmax kernel (one less memory roundtrip)
- Uses online softmax algorithm (updates max/sum incrementally)
- Autoregressive masking built directly into softmax
- Reduces kernel launches and memory traffic
- Performance: Further improvement over version 3
- Use case: Understanding kernel fusion benefits

VERSION 5: FP16 Mixed Precision
./attention_forward 5
- Same algorithm as version 4 but uses FP16/BF16 for storage and compute
- Enables use of Tensor Cores on modern GPUs (Volta+)
- Reduces memory bandwidth requirements by 2X
- Includes FP32<->FP16 conversions for validation
- Trade-off: Slight accuracy loss for significant speedup
- Performance: Significantly faster on Tensor Core GPUs
- Use case: Production on modern hardware

VERSION 6: FP16 Without Permutes (Unrealistic)
./attention_forward 6
- Same as version 5 but skips permute/unpermute on perf runs
- Useful for isolating permutation overhead
- Not realistic for actual usage
- Use case: Performance analysis only

VERSION 10: cuDNN Flash Attention
./attention_forward 10
- Uses NVIDIA's cuDNN library flash attention implementation
- Highly optimized by NVIDIA engineers for specific GPU architectures
- Handles FP16/BF16 natively with custom strides (no explicit permute needed)
- May use special hardware features not accessible via CUDA
- Performance: Typically fastest on supported hardware
- Use case: Production when cuDNN is available
- Reference: https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md

VERSION 11: cuDNN Full Low Precision
./attention_forward 11
- Same as version 10 but assumes entire network is FP16/BF16
- Skips FP32<->FP16 conversions
- Use case: End-to-end low precision networks
*/
//#define ENABLE_CUDNN // can be enabled via nvcc "-DENABLE_CUDNN"

// ----------------------------------------------------------------------------
// Standard library includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>              // For FLT_MAX constant

// ----------------------------------------------------------------------------
// CUDA library includes
#include <cublas_v2.h>          // cuBLAS: NVIDIA's BLAS (Basic Linear Algebra Subprograms) library
                                // Provides highly optimized matrix multiplication (GEMM) routines
#include <cuda_runtime.h>       // CUDA runtime API for memory management, kernel launches, etc.
#include <cuda_bf16.h>          // BFloat16 data type support (16-bit floating point)
#include <cooperative_groups.h> // Cooperative Groups API for warp-level and block-level primitives
#include <cooperative_groups/reduce.h> // Warp-level reduction operations (sum, max, etc.)

// ----------------------------------------------------------------------------
// Project-specific includes
#define ENABLE_BF16             // Enable BFloat16 support in common.h
#include "common.h"             // Common utilities: error checking, memory helpers, benchmarking
                                // Defines floatX as either __nv_bfloat16 or half depending on config
                                // Provides cudaCheck, cublasCheck macros for error handling
                                // Includes ceil_div, make_random_float, validate_result utilities

// ----------------------------------------------------------------------------
// CUDA & cuDNN setup
static bool first_run_validation = true; // Always run validation steps (e.g. permute) on first run
                                           // to ensure correctness, then skip on subsequent runs

#ifdef ENABLE_CUDNN
#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;
#if CUBLAS_LOWP == CUDA_R_16BF
#define CUDNN_16BIT fe::DataType_t::BFLOAT16
#else
#define CUDNN_16BIT fe::DataType_t::HALF
#endif

static cudnnHandle_t cudnn_handle;
static size_t cudnn_workspace_size = 0; // dynamically allocated as needed (up to 256MiB!)
static void* cudnn_workspace = NULL;

#define checkCudaErr(err) assert((int)err == 0);
#define checkCudnnErr(err) assert((int)err == 0);
#endif // ENABLE_CUDNN
// ----------------------------------------------------------------------------
// CPU code reference
// This reference implementation is used for correctness validation of GPU kernels
// It implements the standard multi-head attention algorithm in a straightforward way

void attention_forward_cpu(float* out, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH) {
    // Input shapes:
    //   inp: (B, T, 3C) - batch, sequence length, 3*channels (Q,K,V concatenated)
    //   preatt: (B, NH, T, T) - pre-softmax attention scores (for debugging/backward)
    //   att: (B, NH, T, T) - post-softmax attention weights
    //   out: (B, T, C) - output after applying attention to values
    //
    // B = batch size, T = sequence length, C = channels (embedding dimension)
    // NH = number of attention heads, hs = head size = C / NH

    int C3 = C*3;            // Total size including Q, K, V
    int hs = C / NH;         // head size - dimension of each attention head
    float scale = 1.0 / sqrtf(hs);  // Scaling factor for attention scores
                                     // Prevents softmax from saturating for large head dimensions

    // Outer loops iterate over batch, sequence positions, and attention heads
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                // Locate query vector for current position t and head h
                // inp layout: [batch][time][qkv=0..2][head][head_dim]
                const float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // Pointers to attention score arrays for this (batch, head, time) position
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // PASS 1: Compute Query @ Key^T (attention scores)
                // For autoregressive/causal attention, only attend to current and previous positions
                // Also track maximum value for numerically stable softmax
                float maxval = -FLT_MAX;
                for (int t2 = 0; t2 <= t; t2++) {  // Only up to current position (causal mask)
                    // Locate key vector for position t2 and head h
                    // +C offset because keys come after queries in the concatenated layout
                    const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;

                    // Compute dot product: query_t · key_t2
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;  // Scale by 1/sqrt(head_size) for numerical stability

                    // Track maximum for stable softmax computation
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;  // Store pre-softmax attention score
                }

                // Pad with -INFINITY for positions beyond current (future positions)
                // Not strictly necessary but makes debugging easier and matches PyTorch output
                for (int t2 = t+1; t2 < T; t2++) {
                    preatt_bth[t2] = -INFINITY;
                }

                // PASS 2: Compute exp and accumulate sum (numerically stable softmax part 1)
                // Softmax formula: exp(x - max(x)) / sum(exp(x - max(x)))
                // Subtracting max prevents overflow and improves numerical stability
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;  // Store unnormalized exp values
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // PASS 3: Normalize to get final softmax probabilities
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;  // Divide by sum to get probabilities
                    } else {
                        // Causal attention mask - future positions get zero attention weight
                        // This is redundant (already -INFINITY in preatt) but explicit for clarity
                        att_bth[t2] = 0.0f;
                    }
                }

                // PASS 4: Compute weighted sum of values (Attention @ V)
                // out = sum_t2( att[t, t2] * value[t2] )
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }  // Initialize output

                for (int t2 = 0; t2 <= t; t2++) {
                    // Locate value vector for position t2 and head h
                    // +C*2 offset because values come after queries and keys
                    const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    float att_btht2 = att_bth[t2];  // Attention weight for this position

                    // Accumulate weighted value into output
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels - VERSION 1 (Naive)
// These kernels are direct ports of the CPU code to GPU, with minimal optimization.
// Each kernel handles one stage of the attention pipeline.

/*
KERNEL: attention_query_key_kernel1
PURPOSE: Compute Q @ K^T (attention scores before softmax)
APPROACH: Naive parallelization - one thread per element of output matrix
PARALLELIZATION: Over (batch, head, query_pos, key_pos)
PERFORMANCE CHARACTERISTICS:
  - Memory-bound: Poor memory access patterns
  - No coalescing: Each thread reads different scattered locations
  - Sequential inner loop over head_size dimension
  - Inefficient: Each query vector is loaded T times (once per key position)
OPTIMIZATION OPPORTUNITIES:
  - Use shared memory to cache query/key vectors
  - Coalesce memory accesses by reorganizing thread mapping
  - Use warp-level primitives for dot product reduction
  - Fuse with subsequent operations to reduce memory traffic
*/
__global__ void attention_query_key_kernel1(float* preatt, const float* inp,
                                           int B, int T, int C, int NH) {
    // Global thread index - each thread computes one element of the (B, NH, T, T) output
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * NH * T * T;

    if (idx < total_threads) {
        // Decode linear index into 4D coordinates (b, h, t, t2)
        // where t is query position, t2 is key position
        int t2 = idx % T;                    // Key position
        int t = (idx / T) % T;               // Query position

        // Apply causal mask: prevent attending to future positions
        if (t2 > t) {
            preatt[idx] = -INFINITY;
            return;
        }

        int h = (idx / (T * T)) % NH;        // Head index
        int b = idx / (NH * T * T);          // Batch index

        int C3 = C*3;
        int hs = C / NH; // head size

        // Locate query and key vectors
        // PERFORMANCE NOTE: These pointer calculations involve scattered memory access
        const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
        const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

        // Compute dot product: query_t · key_t2
        // PERFORMANCE NOTE: This loop is sequential - no parallelization within thread
        // Better approach would use warp shuffle or shared memory reductions
        float val = 0.0f;
        for (int i = 0; i < hs; i++) {
            val += query_t[i] * key_t2[i];
        }
        val *= 1.0 / sqrtf(hs);  // Scale for numerical stability

        preatt[idx] = val;
    }
}

/*
KERNEL: attention_softmax_kernel1
PURPOSE: Apply softmax to attention scores (convert scores to probabilities)
APPROACH: Naive - one thread per row of attention matrix
PARALLELIZATION: Over (batch, head, query_position)
ALGORITHM: Three-pass numerically stable softmax
  1. Find maximum value (for numerical stability)
  2. Compute exp(x - max) and sum
  3. Normalize by dividing by sum
PERFORMANCE CHARACTERISTICS:
  - Memory-bound: Multiple passes over same data
  - Sequential: Each thread processes entire row alone
  - No parallelization within row
OPTIMIZATION OPPORTUNITIES:
  - Use warp-level reductions for max/sum (see softmax_forward_kernel4)
  - Combine passes to reduce memory traffic
  - Use online softmax algorithm (see softmax_forward_kernel5)
*/
__global__ void attention_softmax_kernel1(float* att, const float* preatt,
                                         int B, int T, int NH) {
    // One thread per (batch, time, head) - each processes one row of attention matrix
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * T * NH;

    if (idx < total_threads) {
        // Decode linear index
        int h = idx % NH;           // Head index
        int t = (idx / NH) % T;     // Query position (row of attention matrix)
        int b = idx / (NH * T);     // Batch index

        // Pointers to input/output rows
        const float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
        float* att_bth = att + b*NH*T*T + h*T*T + t*T;

        // PASS 1: Find maximum value for numerical stability
        // Prevents overflow in exp() by computing exp(x - max) instead of exp(x)
        float maxval = -FLT_MAX;
        for (int t2 = 0; t2 <= t; t2++) {  // Only up to current position (causal)
            if (preatt_bth[t2] > maxval) {
                maxval = preatt_bth[t2];
            }
        }

        // PASS 2: Calculate exp(x - max) and sum for normalization
        float expsum = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) {
            float expv = expf(preatt_bth[t2] - maxval);
            expsum += expv;
            att_bth[t2] = expv;  // Store unnormalized values
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        // PASS 3: Normalize to get final softmax probabilities
        for (int t2 = 0; t2 < T; t2++) {
            if (t2 <= t) {
                att_bth[t2] *= expsum_inv;  // Divide by sum
            } else {
                // Causal attention mask - zero out future positions
                // Not strictly necessary but explicit for debugging
                att_bth[t2] = 0.0f;
            }
        }
    }
}

/*
DEVICE FUNCTION: warpReduceMax
PURPOSE: Efficiently find maximum value across all threads in a warp
APPROACH: Tree reduction using warp shuffle instructions
OPTIMIZATION TECHNIQUES:
  - Warp shuffle (__shfl_down_sync): Exchange data between threads without shared memory
  - Tree reduction: O(log n) steps instead of O(n)
  - No synchronization needed: Warp executes in lockstep (SIMT model)
EXPLANATION:
  A warp has 32 threads. We use shuffle to compare values:
  Step 1 (offset=16): Thread 0 compares with Thread 16, Thread 1 with 17, etc.
  Step 2 (offset=8):  Thread 0 compares with Thread 8, Thread 1 with 9, etc.
  ...continuing until offset=1
  After 5 steps (log2(32)), Thread 0 holds the maximum of all 32 values
*/
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        // __shfl_down_sync: Get value from thread (threadIdx.x + offset) in same warp
        // 0xFFFFFFFF mask means all threads participate
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;  // Only thread 0 has the true maximum, but all threads return a value
}

/*
KERNEL: softmax_forward_kernel4
PURPOSE: Optimized softmax with warp-level reductions and flexible block sizes
APPROACH: Two-level reduction (warp-level then block-level)
PARALLELIZATION: One block per row, threads cooperate within block
ALGORITHM:
  1. Thread coarsening: Each thread processes multiple elements (stride loop)
  2. Warp-level reduction: Use shuffle for fast intra-warp reduction
  3. Block-level reduction: Use shared memory for inter-warp reduction
PERFORMANCE CHARACTERISTICS:
  - Much faster than kernel1: Exploits warp-level parallelism
  - Flexible: Works with any block size (multiple of 32)
  - Memory efficient: Minimizes shared memory usage
OPTIMIZATION TECHNIQUES:
  - Warp shuffle instructions (no shared memory for intra-warp)
  - Thread coarsening (each thread handles C/blockDim.x elements)
  - Shared memory for inter-warp communication only
*/
__global__ void softmax_forward_kernel4(float* out, const float* inp, int N, int C) {
    // Input/output shape: (N, C) where N = number of rows, C = row length
    // Each row gets softmax applied independently
    // One block handles one row, using block_size threads cooperatively
    // Threads are organized into warps of 32

    // Shared memory layout: [maxvals for each warp | sumvals for each warp]
    extern __shared__ float shared[];

    int idx = blockIdx.x;              // Which row this block is processing
    int tid = threadIdx.x;             // Thread index within block
    int warpId = threadIdx.x / 32;     // Which warp this thread belongs to (0 to warpsPerBlock-1)
    int laneId = threadIdx.x % 32;     // Position within warp (0 to 31)

    // Calculate number of warps in this block
    int warpsPerBlock = blockDim.x / 32;

    // Partition shared memory: first half for max values, second half for sums
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // Pointer to the row this block is processing
    const float* x = inp + idx * C;

    // STEP 1: Find maximum value in row (for numerical stability)
    // Thread coarsening: each thread handles multiple elements with stride
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {  // Stride by blockDim.x
        maxval = fmaxf(maxval, x[i]);
    }

    // Reduce within warp: Each thread has a partial max, combine them
    maxval = warpReduceMax(maxval);

    // After warp reduction, lane 0 of each warp has that warp's maximum
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();  // Ensure all warps have written their maxvals

    // Reduce across warps: Thread 0 combines all warp maxima
    if (tid == 0) {
        float val = maxvals[0];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        maxvals[0] = val;  // Store global maximum
    }
    __syncthreads();

    // Broadcast global maximum to all threads
    float offset = maxvals[0];

    // STEP 2: Compute exp(x - max) and write to output
    // This is numerically stable: avoids overflow in exp()
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = expf(x[i] - offset);
    }

    // STEP 3: Sum all exp values for normalization
    // Now read from output (which contains exp values)
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }

    // Reduce sum within warp
    sumval = warpReduceSum(sumval);

    // Lane 0 of each warp writes to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    // Reduce across warps: Thread 0 computes total sum
    if (tid == 0) {
        float val = sumvals[0];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();

    // Broadcast total sum to all threads
    float sum = sumvals[0];

    // STEP 4: Normalize by dividing by sum
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sum;
    }
}


/*
DEVICE HELPER FUNCTIONS: vec_at
PURPOSE: Access individual float elements within a float4 vector
WHY NEEDED: float4 is a vectorized type for memory coalescing, but we need element access
USAGE: vec_at(vec, 0) gets first element, vec_at(vec, 3) gets fourth element
*/
__device__ float& vec_at(float4& vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}

__device__ float vec_at(const float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

/*
KERNEL: softmax_forward_kernel5
PURPOSE: Highly optimized softmax using online (streaming) algorithm
APPROACH: Online softmax - updates max and sum incrementally in one pass
PARALLELIZATION: Warp-level - one warp per row
KEY INNOVATIONS:
  1. Online algorithm: Computes softmax in single pass (vs 3 passes in kernel4)
  2. Fused scaling: Incorporates inv_temperature directly (for attention)
  3. Autoregressive-aware: Only processes valid positions (lower triangle)
  4. Vectorized loads: Uses float4 for memory bandwidth efficiency
ALGORITHM (Online Softmax):
  Traditional softmax requires:
    1. max = max(x)
    2. sum = sum(exp(x - max))
    3. out = exp(x - max) / sum
  Online softmax maintains running max and sum:
    When seeing new value x[i]:
      if x[i] > maxval:
        sumval = sumval * exp(old_max - new_max) + exp(x[i] - new_max)
        maxval = x[i]
  This allows single-pass computation without storing intermediate values!
PERFORMANCE CHARACTERISTICS:
  - Memory efficient: Single pass over data
  - Compute-bound: More exp() calls but fewer memory accesses
  - Bandwidth efficient: Vectorized float4 loads
  - Warp-level parallelism: Natural fit for 32-thread warps
WHEN TO USE:
  - When memory bandwidth is the bottleneck
  - For autoregressive attention (triangular matrices)
  - When fusing with other operations (like scaling)
*/
__global__ void softmax_forward_kernel5(float* out, float inv_temperature, const float* inp, int N, int T) {
    // Input/output shape: (N, T, T) where N = B * NH (batch * num_heads)
    // Each row is a separate softmax operation
    // inv_temperature = scaling factor (1/sqrt(head_size) for attention)
    // Only compute lower triangle due to causal masking

    assert(T % 4  == 0);  // Required for float4 vectorization
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Calculate which row this warp is processing
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N * T) {
        return;
    }

    int own_pos = idx % T;        // Position within sequence (row index)
    int pos_by_4 = own_pos / 4;   // Number of float4 vectors to process

    // Pointer to row of input
    const float* x = inp + idx * T;

    // Initialize running max and sum for online algorithm
    // Use -FLT_MAX (not -INF) to avoid NaN when subtracting infinities
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    // ONLINE SOFTMAX ALGORITHM - Main loop with vectorized loads
    // Process input in chunks of 4 floats for memory bandwidth efficiency
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = warp.thread_rank(); i < pos_by_4; i += warp.size()) {
        float4 v = x_vec[i];  // Load 4 values at once (coalesced access)

        // Update running max and sum for each of the 4 values
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, vec_at(v, k));
        }

        // Online softmax update: rescale previous sum when max changes
        // If max increased, previous exp values need to be scaled down
        sumval *= expf(inv_temperature * (old_maxval - maxval));

        // Add contributions from current values
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (vec_at(v, k) - maxval));
        }
    }

    // Handle remaining elements that don't fit in float4 vectors
    // Each warp thread processes one element if within bounds
    if(4*pos_by_4 + warp.thread_rank() <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, x[4*pos_by_4 + warp.thread_rank()]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * (x[4*pos_by_4 + warp.thread_rank()] - maxval));
    }

    // WARP-LEVEL REDUCTIONS
    // Each thread has computed partial max/sum, now combine across warp
    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});

    // Adjust sum for global max (in case thread's local max wasn't global max)
    sumval *= expf(inv_temperature * (maxval - global_maxval));

    // Combine sums from all threads in warp
    float sum = cg::reduce(warp, sumval, cg::plus<float>{});
    float norm = 1.f / sum;  // Normalization factor

    // WRITE OUTPUT - Compute and store softmax values
    // Recalculate exp values rather than storing them (saves memory bandwidth)
    for (int i = warp.thread_rank(); i <= own_pos; i += warp.size()) {
        // __ldcs: load with cache streaming hint (won't pollute cache)
        // Recalculation is faster than storing intermediate values and reading back
        float ev = expf(inv_temperature * (__ldcs(x + i) - global_maxval));
        // __stcs: store with cache streaming hint
        __stcs(out + idx * T + i, ev * norm);
    }
}


__global__ void attention_value_kernel1(float* out, const float* att, const float* inp,
                                       int B, int T, int C, int NH) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * T * NH;

    if (idx < total_threads) {
        int h = idx % NH;
        int t = (idx / NH) % T;
        int b = idx / (NH * T);

        int C3 = C*3;
        int hs = C / NH; // head size

        float* out_bth = out + b * T * C + t * C + h * hs;
        const float* att_bth = att + b*NH*T*T + h*T*T + t*T;

        for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
        for (int t2 = 0; t2 <= t; t2++) {
           const  float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
            float att_btht2 = att_bth[t2];
            for (int i = 0; i < hs; i++) {
                out_bth[i] += att_btht2 * value_t2[i];
            }
        }
    }
}

__global__
void attention_forward_kernel2(
    const float* Q,
    const float* K,
    const float* V,
    const int N,
    const int d,
    const int Tc,
    const int Tr,
    const int Bc,
    const int Br,
    const float softmax_scale,
    float* l,
    float* m,
    float* O
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {
            // if past the end of the sequence, break
            if (i * Br + tx >= N) {
                break;
            }

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            // S[tx][y] = Sum_{x = 0}^{d-1} {Qi[tx][x] * Kj[y][x]}
            // row_m = Max_{y = 0}^{Bc-1} S[tx][y]
            // with causal masking
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                if (j * Bc + y >= N) {
                    break;
                }
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                if (i * Br + tx < j * Bc + y)
                    sum = -INFINITY;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // implement softmax with causal masking
            // P = exp(S - row_m), row_l = rowsum(P)
            // P[tx][y] = exp(S[tx][y] - row_m)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                if (j * Bc + y >= N) {
                    break;
                }
                if (i * Br + tx < j * Bc + y)
                    S[(Bc * tx) + y] = 0;
                else
                    S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    if (j * Bc + y >= N) {
                        break;
                    }
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]

    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = \
            (b * N * 3 * NH * d)
            +   (n * 3 * NH * d)
            +       (0 * NH * d)
            +          (nh_ * d)
            +                d_;

        q[idx] = inp[inp_idx];
        k[idx] = inp[inp_idx + NH * d];
        v[idx] = inp[inp_idx + 2 * (NH * d)];
    }
}

__global__ void unpermute_kernel(const float* inp, float *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = inp[idx];
    }
}

__global__ void scale_kernel(float* inp, float scale, int B, int NH, int T) {
    // scales the pre-softmax attention scores by scale
    // and sets the autoregressive locations to -INFINITY
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * T * T) {
        int rest = idx % (NH * T * T);
        rest = rest % (T * T);
        int t2 = rest / T;
        int t = rest % T;
        if (t > t2) {
            inp[idx] = -INFINITY;
        } else {
            inp[idx] *= scale;
        }
    }
}

// direct translation of the CPU kernel. Each warp handles ont (b, h, t) combination.
// The important changes compared to the CPU version:
//  - each inner loop is handled by a warp
//  - don't write non-autoregressive parts
//  - reordered the last loops so that we can do all writing in the outer loop.
__global__ void attention_forward_fused1(float* out, float* preatt, float* att,
                                         const float* inp,
                                         int B, int T, int C, int NH) {
    // input is (B, T, 3C) Q,K,V
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int t = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    int h = blockIdx.y;
    int b = blockIdx.z;

    if(t >= T) return;

    const float* query_t = inp + b * T * C3 + t * C3 + h * hs;
    float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
    float* att_bth = att + b*NH*T*T + h*T*T + t*T;

    // pass 1: calculate query dot key and maxval
    float maxval = -INFINITY;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

        // (query_t) dot (key_t2)
        float val = 0.0f;
        for (int i = warp.thread_rank(); i < hs; i += warp.size()) {
            val += query_t[i] * key_t2[i];
        }
        val = cg::reduce(warp, val, cg::plus<float>{});
        val *= scale;
        maxval = max(maxval, val);
        if(warp.thread_rank() == 0) {
            preatt_bth[t2] = val;
        }
    }

    // pass 2: calculate the exp and keep track of sum
    float expsum = 0.0f;
    for (int t2 = warp.thread_rank(); t2 <= t; t2 += warp.size()) {
        float expv = expf(preatt_bth[t2] - maxval);
        expsum += expv;
    }

    expsum = cg::reduce(warp, expsum, cg::plus<float>{});

    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

    // pass 3: normalize to get the softmax is combined with the next loop to reduce memory round-trips
    for (int t2 = warp.thread_rank(); t2 <= t; t2 += warp.size()) {
        att_bth[t2] = expf(preatt_bth[t2] - maxval) * expsum_inv;
    }

    // pass 4: accumulate weighted values into the output of attention
    float* out_bth = out + b * T * C + t * C + h * hs;
    for (int i = warp.thread_rank(); i < hs; i += warp.size()) {
        float o = 0.f;
        for (int t2 = 0; t2 <= t; t2++) {
            const float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
            float att_btht2 = att_bth[t2];
            o += att_btht2 * value_t2[i];
        }
        out_bth[i] = o;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void attention_forward1(float* out, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    // attention calculation
    int total_threads = B * NH * T * T;
    int num_blocks = ceil_div(total_threads, block_size);
    attention_query_key_kernel1<<<num_blocks, block_size>>>(preatt, inp, B, T, C, NH);
    // softmax and value accumulation
    total_threads = B * T * NH;
    num_blocks = ceil_div(total_threads, block_size);
    attention_softmax_kernel1<<<num_blocks, block_size>>>(att, preatt, B, T, NH);
    attention_value_kernel1<<<num_blocks, block_size>>>(out, att, inp, B, T, C, NH);
}


void attention_forward2(float* out,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    // TODO there should be no mallocs inside any of these functions!
    // not fixing this because we don't intend to use attention_forward2,
    // it seems to be way too slow as is

    // these are hardcoded to 32 for now
    const int Bc = 32;
    const int Br = 32;
    // renaming these to be consistent with the kernel
    // const int B = B;
    const int nh = NH;
    const int N = T;
    const int d = C / NH;
    // more
    const int Tc = ceil((float) N / Bc);
    const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);
    // create some temporary memory
    float* l;
    float* m;
    cudaCheck(cudaMalloc(&l, B * nh * N * sizeof(float)));
    cudaCheck(cudaMalloc(&m, B * nh * N * sizeof(float)));
    cudaCheck(cudaMemset(l, 0, B * nh * N * sizeof(float)));
    cudaCheck(cudaMemset(m, -10000.0f, B * nh * N * sizeof(float)));

    // calculate SRAM size needed per block, ensure we have enough shared memory
    int col_tile_size = Bc * d;  // size of Kj, Vj
    int row_tile_size = Br * d;  // size of Qi
    const int sram_size =
        (2 * col_tile_size * sizeof(float))  // SRAM size for Kj, Vj
        + (row_tile_size * sizeof(float))  // SRAM size for Qi
        + (Bc * Br * sizeof(float));  // SRAM size for S
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (sram_size > max_sram_size) {
        printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);
        printf("SRAM size exceeds maximum shared memory per block\n");
        printf("Try decreasing col_tile_size or row_tile_size further\n");
        exit(1);
    }

    // grid and block dims
    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Br);  // Br threads per block

    // okay so now, this kernel wants Q,K,V to all be of shape (B, nh, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, nh, d)
    // so we have to permute the tensor using a kernel with block_size
    float *q, *k, *v;
    cudaCheck(cudaMalloc(&q, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&k, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&v, B * T * C * sizeof(float)));
    int total_threads = B * N * nh * d;
    int num_blocks = ceil_div(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, N, nh, d);

    // now actually call the flash attention kernel
    attention_forward_kernel2<<<grid_dim, block_dim, sram_size>>>(
        q, k, v,
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l, m, out
    );

    // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    unpermute_kernel<<<num_blocks, block_size>>>(out, q, B, N, nh, d);
    cudaCheck(cudaMemcpy(out, q, B * T * C * sizeof(float), cudaMemcpyDeviceToDevice));

    // free memory
    cudaCheck(cudaFree(l));
    cudaCheck(cudaFree(m));
    cudaCheck(cudaFree(q));
    cudaCheck(cudaFree(k));
    cudaCheck(cudaFree(v));
}

void attention_forward3(float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                       const float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            T, T, HS,
                            &alpha,
                            k, HS, T * HS,
                            q, HS, T * HS,
                            &beta,
                            preatt, T, T * T,
                            B * NH));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0f / sqrtf(HS);
    total_threads = B * NH * T * T;
    num_blocks = ceil_div(total_threads, block_size);
    scale_kernel<<<num_blocks, block_size>>>(preatt, scale, B, NH, T);

    // softmax. preatt is (B, NH, T, T) but we view it as (B * NH * T, T) and use the softmax kernel
    int softmax_block_size = 256;
    int grid_size = B * NH * T;
    size_t shared_mem_size = 2 * softmax_block_size / 32 * sizeof(float);
    softmax_forward_kernel4<<<grid_size, softmax_block_size, shared_mem_size>>>(att, preatt, B * NH * T, T);

    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            HS, T, T,
                            &alpha,
                            v, HS, T * HS,
                            att, T, T * T,
                            &beta,
                            vaccum, HS, T * HS,
                            B * NH));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
}

void attention_forward4(float* out, float* vaccum, float* qkvr, float* preatt, float* att,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size) {
    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     T, T, HS,
                                     &alpha,
                                     k, HS, T * HS,
                                     q, HS, T * HS,
                                     &beta,
                                     preatt, T, T * T,
                                     B * NH));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int softmax_block_size = 256;
    int grid_size = ceil_div(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);

    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     HS, T, T,
                                     &alpha,
                                     v, HS, T * HS,
                                     att, T, T * T,
                                     &beta,
                                     vaccum, HS, T * HS,
                                     B * NH));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
}


__global__ void softmax_forward_kernel5_lowp(floatX* out, float inv_temperature,
                                             const floatX* inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(T % 4  == 0);
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const floatX* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    // Same thing but without float4, one at a time
    for (int i = warp.thread_rank(); i < pos_by_4; i += warp.size()) {
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, (float)x[4*i + k]);
        }
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * ((float)x[4*i + k] - maxval));
        }
    }

    if(4*pos_by_4 + warp.thread_rank() <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, (float)x[4*pos_by_4 + warp.thread_rank()]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * ((float)x[4*pos_by_4 + warp.thread_rank()] - maxval));
    }

    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    sumval *= expf(inv_temperature * (maxval - global_maxval));

    float sum = cg::reduce(warp, sumval, cg::plus<float>{});
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = warp.thread_rank(); i <= own_pos; i += warp.size()) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = expf(inv_temperature * ((float)__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, (floatX)(ev * norm));
    }
}

__global__ void permute_kernel_lowp(floatX* q, floatX* k, floatX* v,
                                    const float* inp,
                                    int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = \
            (b * N * 3 * NH * d)
            +   (n * 3 * NH * d)
            +       (0 * NH * d)
            +          (nh_ * d)
            +                d_;

        q[idx] = (floatX)inp[inp_idx];
        k[idx] = (floatX)inp[inp_idx + NH * d];
        v[idx] = (floatX)inp[inp_idx + 2 * (NH * d)];
    }
}

__global__ void unpermute_kernel_lowp(const floatX* inp, float *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = (float)inp[idx];
    }
}

void attention_forward5(float* out, floatX* vaccum, floatX* qkvr, floatX* preatt, floatX* att,
                        const float* inp,
                        int B, int T, int C, int NH,
                        const int block_size, bool skip_permute=false) {
    // FP16 version of kernel 4 (with permute/unpermute doing FP32<->FP16)
    // That permute can be skipped on perf runs to analyse its performance impact
    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    int HS = C / NH; // head size
    floatX *q = qkvr + 0 * B * T * C;
    floatX *k = qkvr + 1 * B * T * C;
    floatX* v = qkvr + 2 * B * T * C;

    int total_threads = B * NH * T * HS;
    int num_blocks = ceil_div(total_threads, block_size);
    if (!skip_permute || first_run_validation) {
        permute_kernel_lowp<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    }

    // IMPORTANT: alpha/beta are FP32 for CUBLAS_COMPUTE_32F even if FP16 inputs/outputs
    // But need FP16 scale for CUBLAS_COMPUTE_16F (no errors otherwise, just garbage results *sigh*)
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const floatX alpha_lowp = (floatX)alpha;
    const floatX beta_lowp = (floatX)beta;
    void* alpha_ptr = CUBLAS_LOWP_COMPUTE == CUBLAS_COMPUTE_16F ? (void*)&alpha_lowp : (void*)&alpha;
    void* beta_ptr = CUBLAS_LOWP_COMPUTE == CUBLAS_COMPUTE_16F ? (void*)&beta_lowp : (void*)&beta;

    // batched matrix multiply with cuBLAS
    cublasCheck(cublasGemmStridedBatchedEx(cublas_handle,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     T, T, HS,
                                     alpha_ptr,
                                     k, CUBLAS_LOWP, HS, T * HS,
                                     q, CUBLAS_LOWP, HS, T * HS,
                                     beta_ptr,
                                     preatt, CUBLAS_LOWP, T, T * T,
                                     B * NH,
                                     CUBLAS_LOWP_COMPUTE,
                                     CUBLAS_GEMM_DEFAULT));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0f / sqrtf(HS);
    int softmax_block_size = 256;
    int grid_size = ceil_div(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5_lowp<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);

    // new approach: first cuBLAS another batched matmul
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasGemmStridedBatchedEx(cublas_handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     HS, T, T,
                                     alpha_ptr,
                                     v, CUBLAS_LOWP, HS, T * HS,
                                     att, CUBLAS_LOWP, T, T * T,
                                     beta_ptr,
                                     vaccum, CUBLAS_LOWP, HS, T * HS,
                                     B * NH,
                                     CUBLAS_LOWP_COMPUTE,
                                     CUBLAS_GEMM_DEFAULT));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    if(!skip_permute || first_run_validation) {
        unpermute_kernel_lowp<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
    }
}

#ifdef ENABLE_CUDNN
using graph_tensors_fwd = std::tuple<std::shared_ptr<fe::graph::Graph>,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // Q,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // K,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // V,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // Attn_scale,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // O
                                     std::shared_ptr<fe::graph::Tensor_attributes>>; // Stats

// Need a cache because graph->build_operation_graph() is slow but everything else seems fast
using cache_type_fwd = std::unordered_map<std::size_t, graph_tensors_fwd>;

// Loosely based on cuDNN frontend samples functions and massively simplified
template <typename... Args>
auto lookup_cache_or_build_graph_fwd(Args... args) {
    static cache_type_fwd user_maintained_cache_fwd;
    auto [B, H, T, HS, is_inference_only] = std::make_tuple(args...);

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_dim({B, H, T, HS})
                               .set_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_dim({B, H, T, HS})
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_dim({B, H, T, HS})
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("attn_scale")
                                .set_dim({1, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_is_pass_by_value(true)
                                .set_data_type(fe::DataType_t::FLOAT));

    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only);
    sdpa_options.set_attn_scale(attn_scale);
    sdpa_options.set_causal_mask(true);

    // Create the graph operation and get the output tensors back
    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->set_output(true).set_dim({B, H, T, HS}).set_stride({H * HS * T, HS, H * HS, 1});

    assert(stats == nullptr || is_inference_only == false);
    if (is_inference_only == false) {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
                               .set_dim({B, H, T, 1})
                               .set_stride({H * T, T, 1, 1});
    }

    assert(graph->validate().is_good());
    auto key = graph->key();
    auto it = user_maintained_cache_fwd.find(key);
    if (it != user_maintained_cache_fwd.end()) {
        return it->second;
    }

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    assert(graph->build_operation_graph(cudnn_handle).is_good());
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    assert(graph->check_support(cudnn_handle).is_good());
    assert(graph->build_plans(cudnn_handle).is_good());

    auto tuple = std::make_tuple(graph, Q, K, V, attn_scale, O, stats);
    user_maintained_cache_fwd.insert({key, tuple});
    return tuple;
}

// Used on first run only so we can validate against the CPU results
__global__ void fp32_to_lowp_kernel(floatX* out, const float* inp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = (floatX)inp[idx];
}

__global__ void lowp_to_fp32_kernel(const floatX* inp, float *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = (float)inp[idx];
}

void attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             float* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             float* in_fp32,  // fp32 input
                             float* out_fp32, // fp32 output for validation
                             int B, int T, int C, int NH) {
    static bool first_run_validation = true;
    int HS = C / NH; // number of features per head
    bool is_inference_only = (stats == nullptr);

    // Convert from FP32 to FP16/BF16 on 1st run to get correct results
    const int block_size = 64; // smallest full occupancy block size on modern GPUs
    if (first_run_validation) {
        int total_threads = B * T * C * 3;
        assert(total_threads % block_size == 0);
        int num_blocks = total_threads / block_size;
        fp32_to_lowp_kernel<<<num_blocks, block_size>>>(inp, in_fp32);
    }

    // Get graph and tensors from cache (or generate it on first use)
    auto [graph, Q, K, V, attn_scale, O, softmax_stats] =
        lookup_cache_or_build_graph_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    void* devPtrQ = inp;
    void* devPtrK = (inp + C);
    void* devPtrV = (inp + 2 * C);
    float attn_scale_cpu = 1.0 / sqrtf(HS);
    void* devPtrO = out;

    // Build variant pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, devPtrQ}, {K, devPtrK}, {V, devPtrV}, {attn_scale, &attn_scale_cpu}, {O, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    if (is_inference_only == false) {
        variant_pack[softmax_stats] = stats;
    }

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    if (graph->get_workspace_size() > cudnn_workspace_size) {
        if (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    // Execute graph
    assert(graph->execute(cudnn_handle, variant_pack, cudnn_workspace).is_good());
    cudaCheck(cudaGetLastError());

    // Optionally convert back from FP16/BF16 to FP32
    if (first_run_validation) {
        int total_threads = B * T * C;
        assert(total_threads % block_size == 0);
        int num_blocks = total_threads / block_size;
        lowp_to_fp32_kernel<<<num_blocks, block_size>>>(out, out_fp32);
    }
    cudaCheck(cudaGetLastError());
    first_run_validation = false;
}

#endif // ENABLE_CUDNN

// kernel version dispatch
void attention_forward(int kernel_num,
                       float* out, float* stats, float* vaccum,
                       float* qkvr, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH,
                       const int block_size) {
    switch (kernel_num) {
        case 1:
            attention_forward1(out, preatt, att, inp, B, T, C, NH, block_size);
            break;
        case 2:
            attention_forward2(out, inp, B, T, C, NH, block_size);
            break;
        case 3:
            attention_forward3(out, vaccum, qkvr, preatt, att, inp, B, T, C, NH, block_size);
            break;
        case 4:
            attention_forward4(out, vaccum, qkvr, preatt, att, inp, B, T, C, NH, block_size);
            break;
        case 5:
            attention_forward5(out, (floatX*)vaccum, (floatX*)qkvr,
                               (floatX*)preatt, (floatX*)att,
                               inp, B, T, C, NH, block_size, false);
            break;
        case 6: // skip permutes for perf passes (to analyse perf as if in/out were truly 16-bit)
            attention_forward5(out, (floatX*)vaccum, (floatX*)qkvr,
                               (floatX*)preatt, (floatX*)att,
                               inp, B, T, C, NH, block_size, true);
            break;
        #ifdef ENABLE_CUDNN
        case 10:
            // note: validation only cares about out, which is out_fp32 of the function
            // inp is hackily converted to FP16 into qkvr only on the first run
            // similarly, vaccum is converted to FP32 into out only on the first run
            attention_forward_cudnn((floatX*)vaccum, stats, (floatX*)qkvr, inp, out, B, T, C, NH);
            break;
        #endif
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
    int NH = 12;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);

    // setup cuBLAS (and cuDNN if needed)
    cublasCreate(&cublas_handle);
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    printf("enable_tf32: %d\n", enable_tf32);
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

    #ifdef ENABLE_CUDNN
    checkCudnnErr(cudnnCreate(&cudnn_handle));
    #endif

    // create host memory of random numbers
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* preatt = (float*)malloc(B * NH * T * T * sizeof(float));
    float* att = (float*)malloc(B * NH * T * T * sizeof(float));
    //float* inp = make_random_float(B * T * 3 * C, 10.0f);
    float* inp = make_random_float(B * T * 3 * C);

    // move to GPU
    float* d_out;
    float* d_stats; // for cuDNN
    float* d_vaccum;
    float* d_qkvr;
    float* d_preatt;
    float* d_att;
    float* d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_stats, B * NH * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_vaccum, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_qkvr, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_preatt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_att, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * 3 * C * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);
    int block_sizes[] = {32, 64, 128, 256, 512};

    // Lower accuracy requirements for FP16 (1e-4f also too much for TF32 on kernels 3 & 4)
    float accuracy_threshold = (kernel_num <= 4) ? 1e-3f : 1e-2f;

    // first check the correctness of the kernel
    attention_forward_cpu(out, preatt, att, inp, B, T, C, NH);
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        attention_forward(kernel_num, d_out, d_stats, d_vaccum, d_qkvr, d_preatt, d_att, d_inp, B, T, C, NH, block_size);
        // all kernels should produce the correct output out
        // todo - make accuracy threshold dynamic and depend on FP16 vs FP32?
        validate_result(d_out, out, "out", B * T * C, accuracy_threshold);
        // but as for preatt and att, things get a bit more complicated:
        if (kernel_num != 2 && kernel_num < 5) {
            // kernel 2 (knowingly) fails att/preatt because it uses a different algorithm
            // that estimates the softmax online and never materializes preatt/att
            validate_result(d_att, att, "att", B * NH * T * T, accuracy_threshold);
        }
        if (kernel_num != 2 && kernel_num < 4) {
            // kernel 4 (knowingly) fails preatt because it fuses the scale normalization
            // into the softmax, so preatt is off by 1.0f / sqrt(HS)
            // but att and out (checked below) should match.
            validate_result(d_preatt, preatt, "preatt", B * NH * T * T, accuracy_threshold);
        }
    }
    printf("All results match. Starting benchmarks.\n\n");
    first_run_validation = false;

    // benchmark speed of the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 100;

        float elapsed_time = benchmark_kernel(repeat_times, attention_forward,
                                              kernel_num, d_out, d_stats, d_vaccum, d_qkvr, d_preatt, d_att,
                                              d_inp, B, T, C, NH, block_size);

        printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
    }

    // free memory
    free(out);
    free(preatt);
    free(att);
    free(inp);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_vaccum));
    cudaCheck(cudaFree(d_qkvr));
    cudaCheck(cudaFree(d_preatt));
    cudaCheck(cudaFree(d_att));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_stats));
    cublasDestroy(cublas_handle);

    #ifdef ENABLE_CUDNN
    cudnnDestroy(cudnn_handle);
    if (cudnn_workspace_size > 0) {
        cudaCheck(cudaFree(cudnn_workspace));
    }
    #endif

    return 0;
}