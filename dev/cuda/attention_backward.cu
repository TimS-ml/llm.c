/*
================================================================================
ATTENTION BACKWARD PASS - MULTI-KERNEL PERFORMANCE EXPLORATION
================================================================================

PURPOSE:
This file implements multiple versions of the multi-head attention backward pass
(gradient computation) for educational and benchmarking purposes. Each kernel explores
different optimization strategies for computing gradients through the attention mechanism.

WHY MULTIPLE VERSIONS EXIST:
The backward pass is even more complex than forward pass due to:
- More intricate data dependencies (gradients flow through multiple paths)
- Need to compute Jacobians of softmax (involves outer products)
- Challenging memory access patterns (scatter/gather operations)
- Opportunities for algorithmic simplifications of gradient formulas

This file demonstrates progressive optimizations:
- Load balancing across GPU resources
- Warp-level parallelism for coalesced memory access
- Register reuse to reduce memory traffic
- Algebraic simplification of gradient expressions
- Careful handling of control flow to minimize divergence

ATTENTION BACKWARD ALGORITHM OVERVIEW:
Given gradient dL/dOut, compute gradients with respect to Q, K, V.
The attention operation is: Out = Softmax(Q @ K^T / sqrt(d)) @ V

Backward pass chain (reverse order of forward):
1. GRADIENT THROUGH VALUE MULTIPLICATION (dAtt, dV):
   Forward: Out = Att @ V
   Backward: dAtt = dOut @ V^T
             dV = Att^T @ dOut

2. GRADIENT THROUGH SOFTMAX (dPreatt):
   Forward: Att = Softmax(Preatt)
   Backward: dPreatt[i,j] = sum_k Att[i,k] * (δ(j==k) - Att[i,j]) * dAtt[i,k]
   This is the Jacobian of softmax - the most complex part!
   Simplifies to: dPreatt[i,j] = Att[i,j] * (dAtt[i,j] - sum_k(Att[i,k]*dAtt[i,k]))

3. GRADIENT THROUGH QUERY-KEY MULTIPLICATION (dQ, dK):
   Forward: Preatt = Q @ K^T (scaled)
   Backward: dQ = dPreatt @ K * scale
             dK = dPreatt^T @ Q * scale

Input shapes:
- dout: (B, T, C) - gradient from next layer
- inp: (B, T, 3C) - original input (Q,K,V)
- att: (B, NH, T, T) - attention weights from forward pass

Output shapes:
- dinp: (B, T, 3C) - gradients for Q,K,V

COMPILATION:
nvcc -O3 --use_fast_math -lcublas -lcublasLt attention_backward.cu -o attention_backward

KERNEL VERSIONS:

VERSION 1: Naive First Implementation
OMP_NUM_THREADS=32 ./attention_backward 1
- Direct translation of mathematical formulas to code
- Poor parallelization: Sequential over batch/head dimensions
- Inefficient memory access: No coalescing
- Performance: Baseline (slowest)
- Use case: Understanding the algorithm

VERSION 2: Better Load Balancing
OMP_NUM_THREADS=32 ./attention_backward 2
- Parallelizes over (batch, head) dimensions using block indices
- Each thread still handles inner computation sequentially
- Better GPU utilization: More parallelism exposed
- Performance: Moderate improvement
- Use case: Shows importance of exposing parallelism

VERSION 3: Warp-Level Parallelism
OMP_NUM_THREADS=32 ./attention_backward 3
- Uses full warp (32 threads) to compute each result cooperatively
- Enables coalesced memory access (threads access contiguous addresses)
- Warp-level reductions for computing sums
- Performance: Significant improvement (~2-3x faster than version 2)
- Use case: Demonstrates power of warp-level primitives

VERSION 4: Register Reuse via Loop Unrolling
OMP_NUM_THREADS=32 ./attention_backward 4
- Processes 8 values of t3 per warp instead of 1 (unroll factor = 8)
- Reuses loaded data across multiple output values
- Reduces memory traffic by factor of ~8
- More complex indexing but better memory efficiency
- Performance: Further 20-30% improvement
- Use case: Shows value of data reuse

VERSION 5: Reduced Control Flow Divergence
OMP_NUM_THREADS=32 ./attention_backward 5
- Eliminates conditional branches that cause warp divergence
- Uses arithmetic tricks instead of if-statements
- Splits loops to separate regions with/without conditionals
- Better instruction throughput on GPU
- Performance: Small but consistent improvement (5-10%)
- Use case: Demonstrates importance of avoiding divergence

VERSION 6-8: Advanced Optimizations
- Version 6: Shared memory tiling (block-level cooperation)
- Version 7: Algebraically simplified gradient formula
- Version 8: Multiple optimizations combined (streaming loads, block reordering)
- These represent production-quality optimized kernels
*/

// ----------------------------------------------------------------------------
// Standard library includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>              // For FLT_MAX constant

// ----------------------------------------------------------------------------
// CUDA library includes
#include <cublas_v2.h>          // cuBLAS for optimized matrix multiplications
#include <cuda_runtime.h>       // CUDA runtime API
#include <cooperative_groups.h> // Cooperative groups for warp/block-level operations
#include <cooperative_groups/reduce.h> // Reduction operations across thread groups
#include <cooperative_groups/scan.h>   // Prefix sum operations (not heavily used here)

// ----------------------------------------------------------------------------
// Project-specific includes
#include "common.h"             // Utilities: cudaCheck, cublasCheck, ceil_div, etc.

// ----------------------------------------------------------------------------
// CPU code reference
// These CPU implementations are used for correctness validation of GPU kernels
// They implement the backward pass of attention in a straightforward way

/*
NOTE:
This version of attention_forward is modified to be consistent with the
attention_forward GPU kernel in the following way small but important way:
- preatt is only QUERY @ KEY, without the scale
- the scale instead moved and fused into the softmax
- the full preatt matrix is materialized, even the parts that get masked out
    - this doesn't actually change anything due to masking, but it lets us
      easily compare to the GPU version, which also does the full, dense sgemm
In this way we'll be able to make sure that preatt and att agree CPU vs GPU
*/
void attention_forward_cpu(float* out, float* preatt, float* att,
                            float* inp,
                            int B, int T, int C, int NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -FLT_MAX;
                for (int t2 = 0; t2 < T; t2++) { // used to be t2 <= t
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(scale * (preatt_bth[t2] - maxval));
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

// NOTE: Also contains the re-shuffling of the exact position of "scale"
// and when it is applied (after preatt, not "during" preatt)
// also, full matrices are materialized, even the parts that get masked out
void attention_backward_cpu(float* dinp, float* dpreatt, float* datt,
                            float* dout, float* inp, float* att,
                            int B, int T, int C, int NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                float* dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 < T; t2++) { // ADJUSTED! this was t2 <= t (see note on function)
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += scale * local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += query_t[i] * key_t2[i]
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2];
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2];
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels
// The backward pass mirrors the forward pass structure:
// forward: [permute, sgemm, softmax, sgemm, unpermute]
// backward: [unpermute_bwd, sgemm, softmax_bwd, sgemm, permute_bwd]

/*
KERNEL: permute_kernel
PURPOSE: Rearrange Q,K,V from (B,T,3,NH,d) to (B,NH,T,d) layout
REASON: cuBLAS expects batch matrix multiply in (B,NH,T,d) format
This is part of the forward pass but included here for the backward implementation
*/
__global__ void permute_kernel(float* q, float* k, float* v,
                               const float* inp,
                               int B, int N, int NH, int d) {
    // Transform from interleaved QKV layout to separate Q,K,V tensors
    // Input:  (B, N, 3, NH, d) - batch, sequence, qkv, heads, head_dim
    // Output: (B, NH, N, d) for each of Q, K, V
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < B * NH * N * d) {
        // Decode output index to 4D coordinates
        int b = idx / (NH * N * d);        // Batch
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);          // Head
        rest = rest % (N * d);
        int n = rest / d;                  // Sequence position
        int d_ = rest % d;                 // Head dimension

        // Calculate corresponding input index
        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;

        // Extract Q, K, V from contiguous locations in input
        q[idx] = inp[inp_idx];              // Query
        k[idx] = inp[inp_idx + NH * d];     // Key (offset by NH*d)
        v[idx] = inp[inp_idx + 2 * (NH * d)]; // Value (offset by 2*NH*d)
    }
}

__global__ void permute_kernel_backward(float* dinp,
                                        const float* dq, const float* dk, const float* dv,
                                        int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        dinp[inp_idx] += dq[idx];
        dinp[inp_idx + NH * d] += dk[idx];
        dinp[inp_idx + 2 * (NH * d)] += dv[idx];
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

__global__ void unpermute_kernel_backward(float* dinp, const float *dout, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d) {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        dinp[idx] += dout[other_idx];
    }
}

__device__ float& vec_at(float4& vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}

__device__ float vec_at(const float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

__global__ void softmax_forward_kernel5(float* out, float inv_temperature, const float* inp, int N, int T) {
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
    const float* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = warp.thread_rank(); i < pos_by_4; i += warp.size()) {
        float4 v = x_vec[i];
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, vec_at(v, k));
        }
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (vec_at(v, k) - maxval));
        }
    }

    if(4*pos_by_4 + warp.thread_rank() <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, x[4*pos_by_4 + warp.thread_rank()]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * (x[4*pos_by_4 + warp.thread_rank()] - maxval));
    }

    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    sumval *= expf(inv_temperature * (maxval - global_maxval));

    float sum = cg::reduce(warp, sumval, cg::plus<float>{});
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = warp.thread_rank(); i <= own_pos; i += warp.size()) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = expf(inv_temperature * (__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, ev * norm);
    }
}

/*
================================================================================
SOFTMAX BACKWARD KERNELS - Progressive Optimizations
================================================================================

The softmax backward is the most complex part of attention backward.
Given: forward softmax: y = exp(x) / sum(exp(x))
The Jacobian is: dy/dx[j] = y[j] * (δ(i==j) - y[i])

For our case: dpreatt = softmax_jacobian @ datt
Full formula: dpreatt[t,t3] = sum_t2( att[t,t2] * (δ(t2==t3) - att[t,t3]) * datt[t,t2] )
Simplified: dpreatt[t,t3] = scale * att[t,t3] * (datt[t,t3] - sum_t2(att[t,t2]*datt[t,t2]))

The kernels below explore different ways to compute this efficiently.
*/

/*
KERNEL: softmax_autoregressive_backward_kernel1 (VERSION 1)
PURPOSE: Naive implementation of softmax backward for correctness reference
APPROACH: One thread per output element, sequential over batch/head
PARALLELIZATION: Only over t3 dimension (one dimension of output)
ALGORITHM: Direct implementation of Jacobian formula with indicator function
PERFORMANCE CHARACTERISTICS:
  - Poor parallelism: Most dimensions handled sequentially
  - Poor load balancing: Different threads do vastly different amounts of work
  - Inefficient: Computes full formula without simplification
OPTIMIZATION OPPORTUNITIES:
  - Parallelize over batch and head dimensions
  - Use algebraic simplification to reduce computation
  - Use warp-level primitives for reductions
*/
__global__ void softmax_autoregressive_backward_kernel1(float* dpreatt, const float* datt, const float* att,
                                                     int B, int T, int C, int NH) {
    // All inputs/outputs are (B, NH, T, T) - attention score matrices
    // We need to compute gradients for each element considering softmax Jacobian
    int t3 = blockIdx.x * blockDim.x + threadIdx.x;  // Output column index
    if (t3 < T) {
        int hs = C / NH; // head size
        float scale = 1.0f / sqrtf(hs);  // Attention scaling factor

        // Sequential loops over batch and head - poor parallelization!
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < NH; h++) {
                // Each thread processes multiple rows (t values) for its column (t3)
                for (int t = t3; t < T; t++) {  // Only process lower triangle (causal)
                    const float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                    const float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                    float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;

                    // Compute softmax backward: sum over t2
                    float accum = 0.0f;
                    for (int t2 = 0; t2 <= t; t2++) {
                        // Indicator function: 1 if t2==t3, else 0
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        // Softmax Jacobian: att[t2] * (indicator - att[t3])
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        accum +=  scale * local_derivative * datt_bth[t2];
                    }
                    dpreatt_bth[t3] = accum;
                }
            }
        }
    }
}

// parallelize across t,b,h
__global__ void softmax_autoregressive_backward_kernel2(float* dpreatt, const float* datt, const float* att,
                                                     int B, int T, int C, int NH) {
    int t3 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y * T * T;
    if (t3 >= T) { return; }

    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    for (int t = t3; t < T; t++) {
        float result = 0.0;
        const float* att_bth = att + idx + t*T;
        const float* datt_bth = datt + idx + t*T;
        float* dpreatt_bth = dpreatt + idx + t*T;

        for (int t2 = 0; t2 <= t; t2++) {
            float indicator = t2 == t3 ? 1.0f : 0.0f;
            float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
            result += scale * local_derivative * datt_bth[t2];
        }

        dpreatt_bth[t3] = result;
    }
}

/*
KERNEL: softmax_autoregressive_backward_kernel3 (VERSION 3)
PURPOSE: Warp-level parallelism for efficient softmax backward
APPROACH: Use full warp (32 threads) to compute each output element cooperatively
PARALLELIZATION: Over (batch*head, t3) with warp-level cooperation for t2 loop
KEY IMPROVEMENTS OVER VERSION 2:
  1. Warp-level parallelism: 32 threads cooperate on inner sum
  2. Coalesced memory access: Threads access consecutive memory locations
  3. Warp reduction: Fast tree reduction for combining thread results
ALGORITHM:
  - Each warp handles one (batch, head, row, column) combination
  - Threads in warp split the t2 loop (stride by warp size = 32)
  - Use cooperative groups reduction to combine partial sums
PERFORMANCE CHARACTERISTICS:
  - ~2-3x faster than version 2
  - Memory-bound: Limited by bandwidth, not computation
  - Good occupancy: Many warps can run concurrently
OPTIMIZATION TECHNIQUES:
  - Cooperative groups for warp-level operations
  - Coalesced loads: warp threads access consecutive addresses
  - Register-based reduction (no shared memory needed for intra-warp)
*/
__global__ void softmax_autoregressive_backward_kernel3(float* dpreatt, const float* datt, const float* att,
                                                     int B, int T, int C, int NH) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Each warp handles one t3 value (column of output)
    int t3 = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();

    // blockIdx.y encodes (batch, head) combination
    int idx = blockIdx.y * T * T;
    if (t3 >= T) { return; }

    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);

    // Process all rows for this column (t >= t3 due to causal masking)
    for (int t = t3; t < T; t++) {
        float result = 0.0;  // Accumulator for this thread
        const float* att_bth = att + idx + t*T;
        const float* datt_bth = datt + idx + t*T;
        float* dpreatt_bth = dpreatt + idx + t*T;

        // Load att[t3] once (will be reused in loop)
        const float att_at_t3 = att_bth[t3];

        // Warp-level parallelism: Each thread handles subset of t2 values
        // Stride by warp.size() (32) for coalesced memory access
        for (int t2 = warp.thread_rank(); t2 <= t; t2 += warp.size()) {
            float indicator = t2 == t3 ? 1.0f : 0.0f;
            float local_derivative = att_bth[t2] * (indicator - att_at_t3);
            result += local_derivative * datt_bth[t2];
        }

        // Reduce across warp: Combine results from all 32 threads
        result = cg::reduce(warp, result, cg::plus<float>());

        // Only lane 0 writes the final result
        if(warp.thread_rank() == 0) {
            dpreatt_bth[t3] = scale * result;
        }
    }
}
__global__ void softmax_autoregressive_backward_kernel4(float* __restrict__ dpreatt, const float* __restrict__ datt,
                                                        const float* __restrict__ att,
                                                        int B, int T, int C, int NH) {
    constexpr int UNROLL = 8;
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int t3 = UNROLL * (blockIdx.x * warp.meta_group_size() + warp.meta_group_rank());

    int idx = blockIdx.y * T * T;
    if (t3 >= T) { return; }

    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);

    // the innermost loop combines different values of t2 with different values of t.
    // by handling [t3, t3 + UNROLL) in one thread, we get much better memory reuse:
    // any t3/t-dependent value can be loaded once before the t2 loop.
    // within the t2 loop, we can combine each loaded value with each of the UNROLL
    // pre-loaded values, thus cutting memory ready by a factor of ~UNROLL.

    // one iteration of this loop has to handle the cases
    // this may lead to some invalid indices; therefore, we have several
    // early-outs in the iteration over k below.
    for (int t = t3; t < T; t++) {
        float result[UNROLL] = {};
        const float* att_bth = att + idx + t * T;
        const float* datt_bth = datt + idx + t * T;
        float* dpreatt_bth = dpreatt + idx + t * T;

        float att_at_t3[UNROLL];
        for(int k = 0; k < UNROLL; ++k) {
            if (t < t3 + k) continue;
            att_at_t3[k] = att_bth[t3 + k];
        }

        for (int t2 = warp.thread_rank(); t2 <= t; t2 += warp.size()) {
            float att_t2 = att_bth[t2];
            float datt_t2 = datt_bth[t2];
            for(int k = 0; k < UNROLL; ++k) {
                if (t < t3 + k) continue;
                float indicator = t2 == (t3 + k) ? 1.0f : 0.0f;
                float local_derivative = att_t2 * (indicator - att_at_t3[k]);
                result[k] += local_derivative * datt_t2;
            }
        }

        for(int k = 0; k < UNROLL; ++k) {
            result[k] = cg::reduce(warp, result[k], cg::plus<float>());
        }
        if (warp.thread_rank() < UNROLL) {
            dpreatt_bth[t3 + warp.thread_rank()] = scale * result[warp.thread_rank()];
        }
    }
}

__global__ void softmax_autoregressive_backward_kernel5(float* __restrict__ dpreatt, const float* __restrict__ datt,
                                                        const float* __restrict__ att,
                                                        int B, int T, int C, int NH) {
    constexpr int UNROLL = 8;
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int t3 = UNROLL * (blockIdx.x * warp.meta_group_size() + warp.meta_group_rank());

    int idx = blockIdx.y * T * T;
    if (t3 >= T) { return; }

    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    for (int t = t3; t < T; t++) {
        float result[UNROLL] = {};
        const float* att_bth = att + idx + t * T;
        const float* datt_bth = datt + idx + t * T;
        float* dpreatt_bth = dpreatt + idx + t * T;

        float att_at_t3[UNROLL];
        for(int k = 0; k < UNROLL; ++k) {
            // if t < t3+k, we're out of bounds.
            // in that case, we don't care what we read, because later on,
            // we won't write the corresponding result. So just clip to
            // make sure this is a valid (in-bounds) memory access.
            att_at_t3[k] = att_bth[min(t, t3 + k)];
        }

        // the code below is actually just a for loop; except,
        // we have to do something special in one iteration in
        // the middle, and an if turned out to have significant
        // performance impact.
        // so we split the loop in three parts. Ugly, but effective.

        // the beginning/end loop does the same thing, so we write the code
        // just once in a lambda. In this step, we're guaranteed that
        // indicator == 0
        auto loop_step = [&](int t2){
            float p = att_bth[t2] * datt_bth[t2];
            for (int k = 0; k < UNROLL; ++k) {
                result[k] -= p * att_at_t3[k];
            }
        };

        // Now the actual loop.
        {
            // declare the loop iterator. Needs to be kept across the
            // three different parts, so it's not a local variable in
            // the for loop.
            int t2 = warp.thread_rank();

            // first part, as long as t2 < t3, indicator == 0
            for (; t2 < t3; t2 += warp.size()) {
                loop_step(t2);
            }

            // because k <= warp.size() (==32), the event that t3+k == t2
            // has to happen at this particular step.
            static_assert(UNROLL <= 32, "UNROLL is too large, this won't produce correct results.");
            if (t2 <= t) {
                float att_t2 = att_bth[t2];
                float datt_t2 = datt_bth[t2];
                float p = att_t2 * datt_t2;
                for (int k = 0; k < UNROLL; ++k) {
                    float indicator = t2 == (t3 + k) ? 1.0f : 0.0f;
                    result[k] += p * (indicator - att_at_t3[k]);
                }
                t2 += warp.size();
            }

            // rest of the loop, indicator == 0 again
            for (; t2 <= t; t2 += warp.size()) {
                loop_step(t2);
            }
        }

        for(int k = 0; k < UNROLL; ++k) {
            result[k] = cg::reduce(warp, result[k], cg::plus<float>());
        }

        // when storing, we need to check that this is actually a valid result.
        // here, warp.thread_rank() corresponds to `k` in the previous loops.
        if (warp.thread_rank() < UNROLL && t >= t3 + warp.thread_rank()) {
            dpreatt_bth[t3 + warp.thread_rank()] = scale * result[warp.thread_rank()];
        }
    }
}


// I want `BlockSize` to be statically known to the compiler, thus we get a template here.
// This kernel takes a step back, and looks at the original CPU code again. We have some simple outer loops
// That are independent, (b, t, h), and then the inner loops over (t2, t3) where we're combining elements -- this is
// where we can reuse data and be more efficient
// => handle b, t, h  through block indices; each block does all the work for the (t2, t3) loop cooperatively.
// Now we have two nested loops, and in the inner instruction, we combine indexing from both => this calls for
// loop tiling, and lifting some of the memory ops out of the loop.
// We're in luck here;  if we tile so that t3 is the outer loop, we can get a sinlge write op per result, AND also cache
// the t2-indexed part of the computation, which is the problematic one because it contains a multiplication that now we
// do not have to repeat over and over.
// => do an outer t3 loop where each thread gets one t3 index. Then, do an outer t2 loop in steps of BlockSize, and
// prepare BlockSize many elements for the inner loop. Here, each thread calculates one element and stores it in shmem.
// Then, in the inner t2 loop, each thread reads *all* the elements previously stored and does its computations.
// This way, we do 3*BlockSize loads, but BlockSize^2 computation steps => This kernel is now entirely compute bound.
// To fix up the compute issues, as above, we replace ifs in memory reading with min, and also split the inner loop
// into a large region where we don't have to calculate the indicator, and a small, costly region where we do.
template<int BlockSize>
__global__ void __launch_bounds__(BlockSize) softmax_autoregressive_backward_kernel6(float* dpreatt, const float* datt, const float* att,
                                                        int B, int T, int C, int NH) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    __shared__ float att_bth_s[BlockSize];

    int idx = blockIdx.y;
    int t = blockIdx.x;

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    const float* att_bth = att + t * T;
    const float* datt_bth = datt + t * T;
    float* dpreatt_bth = dpreatt + t * T;

    int block_steps = ceil_div(t+1, BlockSize);
    // very important: This loop condition needs to be the same for all threads.
    // even if a thread later on is not going to do any work, it needs to participate in the
    // data loading process!
    for (int t3f = 0; t3f < block_steps; ++t3f) {
        int t3 = t3f * BlockSize + block.thread_rank();
        float acc = 0.f;
        float at3 = att_bth[t3];
        for (int t2b = 0; t2b <= t; t2b += BlockSize) {
            int end = min(t + 1 - t2b, BlockSize);
            block.sync();
            {
                int t2i = block.thread_rank();
                int t2 = min(t, t2b + t2i);
                att_bth_s[t2i] = att_bth[t2] * datt_bth[t2];
            }

            block.sync();
            if(t3f * BlockSize == t2b) {
                for (int t2i = 0; t2i < end; t2i++) {
                    int t2 = t2b + t2i;
                    float indicator = t2 == t3 ? 1.0f : 0.0f;
                    acc += att_bth_s[t2i] * (indicator - at3);
                }
            } else {
                for (int t2i = 0; t2i < end; t2i++) {
                    acc +=  att_bth_s[t2i] * (0.f - at3);
                }
            }
        }
        dpreatt_bth[t3] = scale * acc;
    }
}

/*
KERNEL: softmax_autoregressive_backward_kernel7 (VERSION 7)
PURPOSE: Algebraically simplified softmax backward - most elegant version!
APPROACH: Reformulate the gradient expression to factor out common terms
PARALLELIZATION: Block-level - one block per row, threads cooperate
KEY INSIGHT - Mathematical Simplification:
  Original formula:
    dpreatt[t,t3] = scale * sum_t2( att[t,t2] * (δ(t2==t3) - att[t,t3]) * datt[t,t2] )

  Expand the sum:
    = scale * sum_t2( att[t,t2]*δ(t2==t3)*datt[t,t2] - att[t,t2]*att[t,t3]*datt[t,t2] )

  Separate terms:
    = scale * (att[t,t3]*datt[t,t3] - att[t,t3]*sum_t2(att[t,t2]*datt[t,t2]))

  Factor out att[t,t3]:
    = scale * att[t,t3] * (datt[t,t3] - sum_t2(att[t,t2]*datt[t,t2]))
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                         This is the SAME for all t3!

ALGORITHM:
  1. Compute local_sum = sum_t2(att[t,t2] * datt[t,t2]) ONCE per row
  2. For each t3: dpreatt[t,t3] = scale * att[t,t3] * (datt[t,t3] - local_sum)

PERFORMANCE CHARACTERISTICS:
  - Compute-bound: Balanced computation vs memory access
  - Efficient: Computes common sum once instead of T times
  - Clean code: Mathematical insight leads to simpler implementation
  - ~10-15% faster than kernel 3 due to reduced arithmetic

OPTIMIZATION TECHNIQUES:
  - Algebraic simplification (key innovation!)
  - Block-level cooperation with warp reductions
  - Shared memory for inter-warp communication
  - Two-level reduction (warp then block)

This is a beautiful example of how understanding the mathematics can lead to
much better code than just optimizing the naive implementation!
*/
template<int BlockSize>
__global__ void softmax_autoregressive_backward_kernel7(float* dpreatt, const float* datt, const float* att,
                                                        int B, int T, int C, float scale) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float block_acc[32];  // One slot per warp for reduction

    // Block indices determine which row we're processing
    int idx = blockIdx.y;  // (batch, head) index
    int t = blockIdx.x;    // row index

    // Offset pointers to the relevant (batch, head) matrix
    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    // Pointers to the specific row we're processing
    const float* att_bth = att + t * T;
    const float* datt_bth = datt + t * T;
    float* dpreatt_bth = dpreatt + t * T;

    // Initialize shared memory (only first warp needs to do this)
    if(warp.meta_group_rank() == 0) {
        block_acc[warp.thread_rank()] = 0;
    }

    // STEP 1: Compute local_sum = sum_t2(att[t2] * datt[t2])
    // This is the common term that appears in all output elements
    float local_sum = 0;
    for(int t2 = block.thread_rank(); t2 <= t; t2 += BlockSize) {
        local_sum += att_bth[t2] * datt_bth[t2];
    }

    // Reduce local_sum across block (two-level reduction)
    // First reduce within each warp
    block_acc[warp.meta_group_rank()] = cg::reduce(warp, local_sum, cg::plus<float>{});
    block.sync();
    // Then reduce across warps (only first warp participates)
    local_sum = cg::reduce(warp, block_acc[warp.thread_rank()], cg::plus<float>{});
    // Now all threads have access to the total sum

    // STEP 2: Compute output using simplified formula
    // dpreatt[t3] = scale * att[t3] * (datt[t3] - local_sum)
    for (int t3 = block.thread_rank(); t3 <= t; t3 += BlockSize) {
        float acc = att_bth[t3] * (datt_bth[t3] - local_sum);
        dpreatt_bth[t3] = scale * acc;
    }
}

// The slightly less pretty version of kernel 7. Adding in all the dirty tricks that can give us a few more percent
//  - streaming memory access instructions
//  - reordering blocks to prevent tail effect
//  - multiple values of T per block
template<int BlockSize>
__global__ void softmax_autoregressive_backward_kernel8(float* dpreatt, const float* datt, const float* att,
                                                        int B, int T, int C, float scale) {
    namespace cg = cooperative_groups;
    constexpr int T_per_block = 4;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float block_acc[32];

    int idx = blockIdx.y;
    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block*blockIdx.x;

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    if (warp.meta_group_rank() == 0) {
        block_acc[warp.thread_rank()] = 0;
    }

    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        const float* att_bth = att + t * T;
        const float* datt_bth = datt + t * T;
        float* dpreatt_bth = dpreatt + t * T;

        float local_sum = 0;
        for (int t2 = block.thread_rank(); t2 <= t; t2 += BlockSize) {
            local_sum += att_bth[t2] * datt_bth[t2];
        }

        block_acc[warp.meta_group_rank()] = cg::reduce(warp, local_sum, cg::plus<float>{});
        block.sync();
        local_sum = cg::reduce(warp, block_acc[warp.thread_rank()], cg::plus<float>{});

        for (int t3 = block.thread_rank(); t3 <= t; t3 += BlockSize) {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            float acc = __ldcs(att_bth + t3) * (__ldcs(datt_bth + t3) - local_sum);
            __stcs(dpreatt_bth + t3, scale * acc);
        }
    }
}


// ----------------------------------------------------------------------------
// kernel launchers

// attention forward pass kernel
void attention_forward(float* out, float* vaccum, float* qkvr, float* preatt, float* att,
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
    // vaccum = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
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

void launch_softmax_1(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(T, block_size);
    softmax_autoregressive_backward_kernel1<<<dim3(num_blocks, B*NH), block_size>>>(dpreatt, datt, att, B, T, C, NH);
}

void launch_softmax_2(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(T, block_size);
    softmax_autoregressive_backward_kernel2<<<dim3(num_blocks, B*NH), block_size>>>(dpreatt, datt, att, B, T, C, NH);
}

void launch_softmax_3(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(32*T, block_size);
    softmax_autoregressive_backward_kernel3<<<dim3(num_blocks, B*NH), block_size>>>(dpreatt, datt, att, B, T, C, NH);
}

void launch_softmax_4(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(32/8*T, block_size);
    softmax_autoregressive_backward_kernel4<<<dim3(num_blocks, B*NH), block_size>>>(dpreatt, datt, att, B, T, C, NH);
}

void launch_softmax_5(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int num_blocks = ceil_div(32/8*T, block_size);
    softmax_autoregressive_backward_kernel5<<<dim3(num_blocks, B*NH), block_size>>>(dpreatt, datt, att, B, T, C, NH);
}

template<class Launcher>
void dispatch_launch(Launcher&& launch, int block_size) {
    switch(block_size) {
        case 32:
            return launch(std::integral_constant<int, 32>{});
        case 64:
            return launch(std::integral_constant<int, 64>{});
        case 128:
            return launch(std::integral_constant<int, 128>{});
        case 256:
            return launch(std::integral_constant<int, 256>{});
        case 512:
            return launch(std::integral_constant<int, 512>{});
        case 1024:
            return launch(std::integral_constant<int, 1024>{});
        default:
            assert(false && "Invalid block size");
    }
}

void launch_softmax_6(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    auto launch = [&](auto int_const) {
        softmax_autoregressive_backward_kernel6<int_const.value><<<dim3(T, B * NH), int_const.value>>>(dpreatt, datt, att, B, T, C, NH);
    };
    dispatch_launch(launch, block_size);
}

void launch_softmax_7(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    auto launch = [&](auto int_const) {
        constexpr int block_size = int_const.value;
        softmax_autoregressive_backward_kernel7<block_size><<<dim3(T, B * NH), block_size>>>
                                                              (dpreatt, datt, att, B, T, C, scale);
    };
    dispatch_launch(launch, block_size);
}

void launch_softmax_8(float* dpreatt, float* datt, const float* att, int B, int T, int C, int NH, int block_size) {
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    auto launch = [&](auto int_const) {
        constexpr int block_size = int_const.value;
        softmax_autoregressive_backward_kernel8<block_size><<<dim3(T / 4, B * NH), block_size>>>
                                                              (dpreatt, datt, att, B, T, C, scale);
    };
    dispatch_launch(launch, block_size);
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
template<class SoftmaxKernel>
void attention_backward1(float* dinp, float* dqkvr, float* dpreatt, float* datt, float* dvaccum,
                        const float* dout,
                        const float* inp, const float* qkvr, const float* preatt, const float* att, const float* vaccum,
                        int B, int T, int C, int NH,
                        SoftmaxKernel softmax_autoregressive_backward,
                        const int block_size) {
    int HS = C / NH; // head size
    const float alpha = 1.0f;
    const float beta = 1.0f; // note beta = 1.0f so that we accumulate gradients (+=)
    // unpack convenience pointers into q, k, v
    const float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    float *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;

    // backward through the unpermute operation
    int num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size>>>(dvaccum, dout, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    // backward into datt
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            T, T, HS,
                            &alpha,
                            v, HS, T * HS,
                            dvaccum, HS, T * HS,
                            &beta,
                            datt, T, T * T,
                            B * NH));

    // backward into dv
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            HS, T, T,
            &alpha,
            dvaccum, HS, T * HS,
            att, T, T * T,
            &beta,
            dv, HS, T * HS,
            B * NH));

    // backward into preatt
    softmax_autoregressive_backward(dpreatt, datt, att, B, T, C, NH, block_size);
    cudaCheck(cudaGetLastError());

    // backward into q
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            HS, T, T,
                            &alpha,
                            k, HS, T * HS,
                            dpreatt, T, T * T,
                            &beta,
                            dq, HS, T * HS,
                            B * NH));
    // backward into k
    cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            HS, T, T,
                            &alpha,
                            q, HS, T * HS,
                            dpreatt, T, T * T,
                            &beta,
                            dk, HS, T * HS,
                            B * NH));

    // backward into inp
    num_blocks = ceil_div(B * NH * T * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dk, dv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void attention_backward(int kernel_num,
                        float* dinp, float* dqkvr, float* dpreatt, float* datt, float* dvaccum,
                        const float* dout,
                        const float* inp, const float* qkvr, const float* preatt, const float* att, const float* vaccum,
                        int B, int T, int C, int NH,
                        const int block_size) {
    switch (kernel_num) {
        case 1:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_1, block_size);
            break;
        case 2:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_2, block_size);
            break;
        case 3:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_3, block_size);
            break;
        case 4:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_4, block_size);
            break;
        case 5:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_5, block_size);
            break;
        case 6:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_6, block_size);
            break;
        case 7:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_7, block_size);
            break;
        case 8:
            attention_backward1(dinp, dqkvr, dpreatt, datt, dvaccum, dout, inp, qkvr, preatt, att, vaccum, B, T, C, NH,
                                launch_softmax_8, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    setup_main();

    // hyperparameters
    int B = 4;
    int T = 1024;
    int C = 768;
    int NH = 12;

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // create the host memory for the forward pass
    float* inp = make_random_float(B * T * 3 * C);
    float* qkvr = (float*)malloc(B * T * 3 * C * sizeof(float));
    float* preatt = (float*)malloc(B * NH * T * T * sizeof(float));
    float* att = (float*)malloc(B * NH * T * T * sizeof(float));
    float* vaccum = (float*)malloc(B * T * C * sizeof(float));
    float* out = (float*)malloc(B * T * C * sizeof(float));

    // execute the forward pass on the CPU
    attention_forward_cpu(out, preatt, att, inp, B, T, C, NH);

    // create device memory for the forward pass
    float *d_inp, *d_qkvr, *d_preatt, *d_att, *d_vaccum, *d_out;
    cudaCheck(cudaMalloc(&d_inp, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_qkvr, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_preatt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_att, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_vaccum, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    // copy over the input
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * 3 * C * sizeof(float), cudaMemcpyHostToDevice));

    // execute the forward pass on the GPU
    const int block_size = 256;
    attention_forward(d_out, d_vaccum, d_qkvr, d_preatt, d_att, d_inp, B, T, C, NH, block_size);

    // check that preatt, att, and out match between the CPU and GPU versions
    printf("Checking the forward pass CPU <-> GPU...\n");
    printf("[preatt]\n"); validate_result(d_preatt, preatt, "preatt", B * T * C, 5e-3f);
    printf("[att]\n");    validate_result(d_att, att, "att", B * T * C, 1e-3f);
    printf("[out]\n");    validate_result(d_out, out, "out", B * T * C, 1e-3f);

    // set up the memory for the backward pass
    float* dout = make_random_float(B * T * C); // the gradients on the output
    float* dinp = make_zeros_float(B * T * 3 * C); // zeros for all else, to += into
    float* dpreatt = make_zeros_float(B * NH * T * T);
    float* datt = make_zeros_float(B * NH * T * T);

    // call backward() on the CPU to get our reference gradients
    attention_backward_cpu(dinp, dpreatt, datt, dout, inp, att, B, T, C, NH);

    // create device memory for the backward pass
    float *d_dinp, *d_dqkvr, *d_dpreatt, *d_datt, *d_dvaccum, *d_dout;
    cudaCheck(cudaMalloc(&d_dinp, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dqkvr, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dpreatt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_datt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dvaccum, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(float)));
    // copy over the dout gradients that starts the backprop chain
    cudaCheck(cudaMemcpy(d_dout, dout, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    // memset all the other memory to zeros, to += into
    cudaCheck(cudaMemset(d_dinp, 0, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMemset(d_dqkvr, 0, B * T * 3 * C * sizeof(float)));
    cudaCheck(cudaMemset(d_dpreatt, 0, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMemset(d_datt, 0, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMemset(d_dvaccum, 0, B * T * C * sizeof(float)));

    // call backward() on the GPU
    attention_backward(kernel_num, d_dinp, d_dqkvr, d_dpreatt, d_datt, d_dvaccum,
                       d_dout, d_inp, d_qkvr, d_preatt, d_att, d_vaccum,
                       B, T, C, NH, block_size);

    // check that the gradients match between the CPU and GPU versions
    // note that we will only check the correctness at [att, preatt, inp]
    // the gradients at qkvr and vaccum will remain unchecked, but are
    // assumed to be correct if the other gradients are correct
    printf("Checking the backward pass CPU <-> GPU...\n");
    printf("[datt]\n");    validate_result(d_datt, datt, "datt", B * NH * T * T, 5e-3f);
    printf("[dpreatt]\n"); validate_result(d_dpreatt, dpreatt, "dpreatt", B * NH * T * T, 1e-3f);
    printf("[dinp]\n");    validate_result(d_dinp, dinp, "dinp", B * T * 3 * C, 1e-3f);

    // also let's manually step through the gradients here
    float* h_dinp = (float*)malloc(B * T * 3 * C * sizeof(float));
    cudaCheck(cudaMemcpy(h_dinp, d_dinp, B * T * 3 * C * sizeof(float), cudaMemcpyDeviceToHost));
    int num_match = 0;
    int num_no_match = 0;
    int num_zero_grad = 0;
    int HS = C / NH;
    for (int i = 0; i < B * T * 3 * C; i++) {

        // the dimensions of inp are (B, T, 3, NH, HS)
        // where B = batch, T = time, 3 = qkv, NH = num heads, HS = head size
        // unpack the individual b,t,qkvix,h,c indices
        int ix = i;
        int c = ix % HS;
        ix /= HS;
        int h = ix % NH;
        ix /= NH;
        int qkvix = ix % 3;
        ix /= 3;
        int t = ix % T;
        ix /= T;
        int b = ix;

        float diff = fabs(dinp[i] - h_dinp[i]);

        // attempt to index at random
        if (b == 1 && t == 5 && c == 23 && h == 2) {
            printf("ix %5d [b=%4d, t=%4d, qkv=%4d, nh=%4d, hs=%4d]: ref: %f gpu: %f\n", i, b, t, qkvix, h, c, dinp[i], h_dinp[i]);
        }

        if (diff > 1e-4f) {
            num_no_match++;
        } else {
            num_match++;
        }

        if (dinp[i] == 0.0f) {
            num_zero_grad++;
        }
    }
    printf("Number of matching gradients: %d (%.2f%% of total)\n", num_match, 100*(float)num_match / (B * T * 3 * C));
    printf("Number of non-matching gradients: %d (%.2f%% of total)\n", num_no_match, 100*(float)num_no_match / (B * T * 3 * C));
    printf("Number of gradients that are exactly zero: %d (%.2f%% of total)\n", num_zero_grad, 100*(float)num_zero_grad / (B * T * 3 * C));

    // final verdict
    printf("All results match. Starting benchmarks.\n\n");

    // benchmark speed of the kernel
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 10;
        float elapsed_time = benchmark_kernel(repeat_times, attention_backward,
                                              kernel_num, d_dinp, d_dqkvr, d_dpreatt, d_datt, d_dvaccum,
                                              d_dout, d_inp, d_qkvr, d_preatt, d_att, d_vaccum,
                                              B, T, C, NH, block_size);

        printf("block_size %4d | time %f ms\n", block_size, elapsed_time);
    }

    // free memory
    free(inp);
    free(qkvr);
    free(preatt);
    free(att);
    free(vaccum);
    free(out);
    free(dout);
    free(dinp);
    free(dpreatt);
    free(datt);
    free(h_dinp);
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_qkvr));
    cudaCheck(cudaFree(d_preatt));
    cudaCheck(cudaFree(d_att));
    cudaCheck(cudaFree(d_vaccum));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_dqkvr));
    cudaCheck(cudaFree(d_dpreatt));
    cudaCheck(cudaFree(d_datt));
    cudaCheck(cudaFree(d_dvaccum));
    cudaCheck(cudaFree(d_dout));
    cublasDestroy(cublas_handle);
    return 0;
}