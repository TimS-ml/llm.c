/*
==============================================================================
Multi-Head Self-Attention Layer
==============================================================================

PURPOSE:
Implements multi-head scaled dot-product self-attention, the core mechanism
of transformer models. This is a fallback implementation used when cuDNN's
Flash Attention is not available or not used.

MATHEMATICAL OPERATION:
For input x of shape (B, T, C) where B=batch, T=sequence length, C=channels:

1. Linear projections to Query, Key, Value:
   Q, K, V = x @ W_q, x @ W_k, x @ W_v  [each: B, T, C]

2. Reshape to multi-head format:
   Q, K, V -> (B, NH, T, HS) where NH=num_heads, HS=head_size, C=NH*HS

3. Compute attention scores (scaled dot-product):
   scores = (Q @ K^T) / sqrt(HS)  [shape: B, NH, T, T]

4. Apply causal mask and softmax:
   attn = softmax(mask(scores))   [only attend to previous positions]

5. Apply attention to values:
   out = attn @ V                  [shape: B, NH, T, HS]

6. Reshape and concatenate heads:
   out -> (B, T, C)

This allows the model to attend to different positions in the sequence and
learn different representation subspaces with different attention heads.

FORWARD PASS ALGORITHM:
1. permute_kernel: Rearrange QKV from (B, T, 3, NH, HS) to separate Q, K, V
   tensors of shape (B, NH, T, HS)

2. matmul_cublaslt: Compute Q @ K^T = preatt using batched matrix multiplication
   - Batch count: B * NH (one matmul per batch and per head)
   - Output shape: (B, NH, T, T)

3. softmax_forward_kernel5: Apply scaling, causal masking, and softmax
   - Uses online softmax algorithm for numerical stability
   - Each warp processes one row (one target position)
   - Only computes lower triangular (causal) part
   - Iterates backwards to improve cache locality for next matmul

4. matmul_cublaslt: Compute attn @ V = vaccum
   - Output shape: (B, NH, T, HS)

5. unpermute_kernel: Rearrange from (B, NH, T, HS) to (B, T, NH, HS) = (B, T, C)

BACKWARD PASS ALGORITHM:
Gradients flow backwards through each operation:

1. unpermute_kernel_backward: dout (B, T, C) -> dvaccum (B, NH, T, HS)

2. Gradient w.r.t. attention weights:
   datt = dvaccum @ V^T

3. Gradient w.r.t. values:
   dV = attn^T @ dvaccum

4. softmax_autoregressive_backward_inplace_kernel:
   Backprop through softmax with formula:
   dpreatt[i,j] = attn[i,j] * (datt[i,j] - sum_k(attn[i,k] * datt[i,k]))
   - In-place: overwrites datt with dpreatt
   - Handles causal masking by explicitly zeroing non-causal elements

5. Gradient w.r.t. queries and keys:
   dQ = dpreatt @ K
   dK = dpreatt^T @ Q

6. permute_kernel_backward: Merge dQ, dK, dV back to (B, T, 3, NH, HS)

CUDA KERNEL IMPLEMENTATIONS:

1. Permute kernels: Simple index transformations, memory-bound
   - Each thread handles one element
   - Uses __ldcs for streaming cache access

2. Softmax kernel:
   - Online algorithm maintains running max and sum
   - Avoids numerical overflow by computing exp(x - max)
   - Warp-level reductions for max and sum
   - Processes rows in reverse order for better cache locality

3. Softmax backward:
   - Processes multiple rows per block (T_per_block=4)
   - Block-level reductions to compute local sum
   - Reverse order to maximize cache hits

MEMORY LAYOUTS:
- QKV combined: (B, T, 3, NH, HS) - interleaved in memory
- Q, K, V separated: (B, NH, T, HS) - heads are batched dimension
- Attention scores: (B, NH, T, T) - lower triangular used (causal)
- Output: (B, T, C) where C = NH * HS

OPTIMIZATIONS:
1. Batched cuBLASLt for efficient GEMM operations
2. Online softmax algorithm (single pass, numerically stable)
3. Backward iteration order for cache locality
4. In-place softmax gradient (saves memory bandwidth)
5. Streaming cache hints to avoid pollution

PERFORMANCE CONSIDERATIONS:
- Attention has O(T^2) complexity in sequence length T
- Memory usage: O(B * NH * T^2) for attention scores
- Causal masking reduces computation by ~50%
- cuBLASLt provides optimized tensor core usage on modern GPUs
- For very long sequences, consider Flash Attention instead

REFERENCES:
- Attention Is All You Need (Vaswani et al., 2017) - Original transformer
  https://arxiv.org/abs/1706.03762
- Online normalizer calculation for softmax (algorithm)
- Flash Attention (Dao et al., 2022) - More efficient attention for long sequences
  https://arxiv.org/abs/2205.14135
*/
#include <assert.h>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include "cublas_common.h"

// ----------------------------------------------------------------------------
// CUDA kernels

// inputs floatX, outputs FP32 (for current FP32-only activation path for this WIP)
__global__ void permute_kernel(floatX* q, floatX* k, floatX* v,
                               const floatX* inp,
                               int B, int N, int NH, int d) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { return; }

    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    q[idx] = __ldcs(&inp[inp_idx]);
    k[idx] = __ldcs(&inp[inp_idx + NH * d]);
    v[idx] = __ldcs(&inp[inp_idx + 2 * (NH * d)]);
}

__global__ void permute_kernel_backward(floatX* dinp,
                                        const floatX* dq, const floatX* dk, const floatX* dv,
                                        int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;

    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    dinp[inp_idx] = dq[idx];
    dinp[inp_idx + NH * d] = dk[idx];
    dinp[inp_idx + 2 * (NH * d)] = dv[idx];
}

__global__ void unpermute_kernel(floatX* inp, floatX *out, int B, int N, int NH, int d) {
   // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)

    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    out[other_idx] = __ldcs(&inp[idx]);
}

__global__ void unpermute_kernel_backward(floatX* dinp, const floatX *dout, int B, int N, int NH, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    dinp[idx] = (floatX)dout[other_idx];
}

__global__ void softmax_forward_kernel5(floatX* out, float inv_temperature, const floatX* inp, int N, int T) {
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    assert(T % 4  == 0);
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // micro-optimization: we iterate backwards so that
    // after the softmax backward operation completes, the cache retains the
    // part of the matrix close to the upper left corner, which benefits the
    // matmul operation that immediately follows.
    // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); // forward order
    int idx = (gridDim.x - blockIdx.x - 1) * num_warps + warp_id; // backward order
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const floatX* x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    const float flt_max = 340282346638528859811704183484516925440.0f; // to avoid including float.h
    float maxval = -flt_max;
    float sumval = 0.0f;

    const floatX* x_aligned = reinterpret_cast<const floatX*>(__builtin_assume_aligned(x, 16));
    for (int i = lane_id; i < pos_by_4; i += WARP_SIZE) {
        float regarray[4];
        for (int k = 0; k < 4; ++k) {
            regarray[k] = (float)x_aligned[4*i + k];
        }
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, regarray[k]);
        }
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (regarray[k] - maxval));
        }
    }

    if(4*pos_by_4 + lane_id <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, (float)x[4*pos_by_4 + lane_id]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * ((float)x[4*pos_by_4 + lane_id] - maxval));
    }

    float global_maxval = warpReduceMax(maxval);
    sumval *= expf(inv_temperature * (maxval - global_maxval));

    float sum = warpReduceSum(sumval);
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = lane_id; i <= own_pos; i += WARP_SIZE) {
        // recalculation is faster than doing the round-trip through memory.
        float ev = expf(inv_temperature * ((float)__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, (floatX)(ev * norm));
    }
}

__global__ void softmax_autoregressive_backward_inplace_kernel(floatX* datt, const floatX* att,
                                                               int B, int T, int C, float scale) {
    constexpr const int BlockSize = 256;
    constexpr int T_per_block = 4;

    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block*blockIdx.x;
    int idx = blockIdx.y;

    att += idx * T * T;
    datt += idx * T * T;

    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) return;
        const floatX* att_bth = att + t * T;
        const floatX* datt_bth = datt + t * T;
        floatX* dpreatt_bth = datt + t * T;

        float local_sum = 0;
        for (int t2 = threadIdx.x; t2 <= t; t2 += BlockSize) {
            local_sum += (float)att_bth[t2] * (float)datt_bth[t2];
        }

        local_sum = blockReduce<warpReduceSum>(local_sum);

        for (int t3 = threadIdx.x; t3 < T; t3 += BlockSize) {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            if(t3 <= t) {
                float acc = (float) __ldcs(att_bth + t3) * ((float) __ldcs(datt_bth + t3) - local_sum);
                __stcs(dpreatt_bth + t3, (floatX) (scale * acc));
            } else {
                // explicitly set non-causal elements to zero
                __stcs(dpreatt_bth + t3, (floatX)0.f);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void attention_forward(floatX* out, floatX* qkvr, floatX* att,
                       floatX* inp,
                       int B, int T, int C, int NH, cudaStream_t stream) {
    NVTX_RANGE_FN();
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    const int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size, 0, stream>>>(q, k, v, inp, B, T, NH, HS);

    floatX* preatt = inp; // reuse inp as scratch buffer
    matmul_cublaslt(preatt, k, q, nullptr, T, T, HS, stream, true, false, B * NH, T * HS, T * HS, T * T);

    // multiply all elements of preatt elementwise by scale
    float scale = 1.f / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * WARP_SIZE, block_size);
    softmax_forward_kernel5<<<grid_size, block_size, 0, stream>>>(att, scale, preatt, B * NH, T);

    // new approach: first cuBLAS another batched matmul
    floatX* vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    matmul_cublaslt(vaccum, v, att, nullptr, HS, T, T, stream, false, false, B * NH, T * HS, T * T, T * HS);

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size, 0, stream>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
void attention_backward(floatX* dinp, floatX* dqkvr, floatX* datt, floatX* scratch,
                        const floatX* dout,
                        const floatX* qkvr, const floatX* att,
                        int B, int T, int C, int NH, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    const int HS = C / NH; // head size

    // unpack convenience pointers into q, k, v
    const floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    floatX *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;

    // backward through the unpermute operation
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(scratch, dout, B, T, NH, HS);
    // backward into datt
    matmul_cublaslt(datt, v, scratch, nullptr, T, T, HS, stream, true, false, B * NH, T * HS, T * HS, T * T);
    // backward into dv
    matmul_cublaslt(dv, scratch, att, nullptr, HS, T, T, stream, false, true, B * NH, T * HS, T * T, T * HS);
    const float scale = 1.0f / sqrtf((float)HS);
    // backward into preatt. this is an in-place operation; datt turns into dpreatt here
    softmax_autoregressive_backward_inplace_kernel<<<dim3(T / 4, B * NH), 256>>>(datt, att, B, T, C, scale);
    const floatX* dpreatt = datt;
    // backward into q
    matmul_cublaslt(dq, k, dpreatt, nullptr, HS, T, T, stream, false, false, B * NH, T * HS, T * T, T * HS);
    // backward into k
    matmul_cublaslt(dk, q, dpreatt, nullptr, HS, T, T, stream, false, true, B * NH, T * HS, T * T, T * HS);
    // backward into inp
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(dinp, dq, dk, dv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}
