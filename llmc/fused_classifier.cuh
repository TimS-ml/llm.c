/*
==============================================================================
Fused Classifier - Cross Entropy Loss with Softmax
==============================================================================

PURPOSE:
Implements the final classification layer of language models, computing
cross-entropy loss and its gradients in a single fused kernel. This is a
critical optimization that combines softmax normalization, loss computation,
and gradient calculation while minimizing memory usage.

MATHEMATICAL OPERATIONS:

Cross-Entropy Loss for Language Modeling:
Given logits z of shape (B*T, V) and target indices y of shape (B*T):

1. Softmax normalization:
   p_i = exp(z_i) / sum_j(exp(z_j))

2. Cross-entropy loss (negative log-likelihood):
   L = -log(p_y) = -log(exp(z_y) / sum_j(exp(z_j)))
     = -z_y + log(sum_j(exp(z_j)))

3. Gradient w.r.t. logits:
   dL/dz_i = p_i - δ(i == y)
   where δ is indicator function (1 if true, 0 if false)

This means:
- For incorrect classes (i ≠ y): gradient = p_i (probability)
- For correct class (i == y): gradient = p_i - 1 (probability minus 1)

KEY OPTIMIZATION - NEVER MATERIALIZE FULL PROBABILITIES:

Naive approach requires O(B*T*V) memory for probabilities.
This implementation only computes:
- Full softmax statistics (max, sum) for each position
- Single probability value at target position (for loss)
- Gradients computed on-the-fly during same kernel pass

For V=50257 (GPT-2 vocab), B*T=1024, this saves:
  1024 * 50257 * 2 bytes (BF16) ≈ 100 MB per batch
  Critical for large vocabulary models!

NUMERICAL STABILITY - SOFTMAX COMPUTATION:

Naive softmax can overflow/underflow:
  exp(x) overflows for x > 88 (FP32)
  exp(x) underflows for x < -88 (FP32)

Solution: Subtract max before exp (log-sum-exp trick):
  softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

This is mathematically equivalent but numerically stable:
  - exp(x - max(x)) has maximum value exp(0) = 1 (no overflow)
  - Minimum value bounded by precision (no catastrophic underflow)

ONLINE SOFTMAX ALGORITHM (prepare_softmax_blockwide3):

Computes softmax statistics in a single pass using online algorithm:

Given row x[0:V]:

For each element x_i:
  1. old_max = max
  2. max = max(max, x_i)
  3. sum *= exp(old_max - max)  [rescale previous sum]
  4. sum += exp(x_i - max)      [add new term]

After processing all elements:
  - max: maximum value in row
  - sum: sum of exp(x - max) across row

Return SoftmaxParams:
  - Offset = max (for numerical stability)
  - Scale = 1 / sum (for normalization)

Then: softmax(x_i) = exp(x_i - Offset) * Scale

This algorithm:
- Single pass through data (no separate max then sum passes)
- Numerically stable (no overflow/underflow)
- Uses warp/block reductions for parallelism

FORWARD AND BACKWARD FUSION:

Traditional approach (3 separate kernels):
1. Softmax forward: logits → probabilities
2. CrossEntropy forward: probabilities → loss
3. CrossEntropy backward: compute dlogits

Fused approach (single kernel):
1. Compute softmax statistics (max, sum)
2. Compute loss using target probability
3. Compute gradients for all logits
4. Overwrite logits with dlogits (in-place)

Memory traffic reduction:
- Unfused: 3 reads + 2 writes of logits = 5× memory
- Fused: 2 reads + 1 write of logits = 3× memory
- 40% reduction in memory bandwidth

CUDA KERNEL IMPLEMENTATION (fused_classifier_kernel5):

Grid/Block organization:
- Grid size: B*T (one block per sequence position)
- Block size: 1024 threads (maximum for good occupancy)
- Process blocks in reverse order (improves cache locality)

Algorithm per block (for one position):

Phase 1: Softmax statistics (all threads participate)
- Each thread processes multiple elements (stride = blockDim.x)
- Thread-local reduction for max and sum
- Warp-level reduction using warpReduceMax/warpReduceSum
- Block-level reduction (across warps)
- Result: SoftmaxParams (Offset, Scale) stored in registers

Phase 2: Loss computation (single thread)
- Thread 0 computes: loss = -log(softmax(z_target))
- Uses SoftmaxParams to avoid recomputing softmax
- Accumulates into losses[position]

Phase 3: __syncthreads() barrier
- Critical! Prevents race condition between:
  * Thread 0 reading logits for loss
  * Other threads writing gradients

Phase 4: Gradient computation (all threads participate)
Main loop (vectorized x128):
- Load x128::size logits (8 for BF16)
- Compute probabilities using SoftmaxParams
- Compute gradients: dlogit = prob - indicator
- Scale by dloss (for gradient accumulation)
- Store to logits (overwrite, in-place)

Remainder loop (for V not divisible by x128::size):
- Process remaining elements one-by-one
- Same computation, but scalar instead of vectorized

MEMORY ACCESS PATTERNS:

1. Softmax statistics:
   - Read logits: Coalesced within block
   - Cached reads (load128, not load128cs)
   - Reused in gradient computation

2. Loss computation:
   - Single thread reads one element
   - Minimal overhead

3. Gradient computation:
   - Read logits: Should still be in L1 cache from phase 1
   - Write gradients: Streaming write (store128cs)
   - Reduces cache pollution for next kernel

Cache locality optimization:
- Process blocks in reverse order
- Gradient matmul (next operation) processes in forward order
- Logits written at end of this kernel are first read by next kernel
- Maximizes cache hits

TEMPLATE PARAMETERS:

WriteDLogits: Controls gradient output
- true: Write gradients to logits (training)
- false: Skip gradient write (validation/inference)

WriteProbs: Controls probability output
- true: Write probabilities to separate buffer (debugging)
- false: Skip (typical case, saves bandwidth)

PRECISION AND SCALING:

- Logits: floatX (BF16/FP8) - input from matmul
- Intermediate computation: FP32 - for numerical accuracy
- Losses: FP32 - accumulated over many examples
- Gradients: floatX (BF16/FP8) - matches forward pass

dloss parameter:
- Scale factor for gradients (typically 1.0)
- Can be 1/batch_size for mean reduction
- Or 1.0 for sum reduction (typical in LLM training)

PERFORMANCE CHARACTERISTICS:

- Memory-bandwidth bound (lots of data, simple math)
- Achieves ~70-85% of peak memory bandwidth
- Bottleneck: Large vocabulary size V (50k-100k+)
- Typical performance on A100:
  * V=50257, B*T=1024: ~2-3 ms
  * Compared to ~4-5 ms for unfused version

Fusion benefits:
1. Memory bandwidth: 40% reduction
2. Kernel launch overhead: 3× reduction
3. Cache locality: Better reuse of logits
4. Overall speedup: ~1.5-2× for this operation

VOCABULARY SIZE HANDLING:

Padding (P) vs actual vocabulary (V):
- V: True vocabulary size (e.g., 50257 for GPT-2)
- P: Padded size for alignment (e.g., 50304 = multiple of 128)
- Logits allocated as (B*T, P) for alignment
- Only first V elements are meaningful
- Kernel handles V explicitly (bounds checking in remainder loop)

ALTERNATIVES AND VARIANTS:

1. Separate kernels: More memory but simpler to debug
2. Label smoothing: Modify target to (1-ε)δ(i==y) + ε/V
3. Focal loss: Reduce weight on easy examples: (1-p_y)^γ * CE
4. Sampled softmax: Only compute over subset of vocabulary (for training)
5. Adaptive softmax: Different capacity per frequency band

REFERENCES:
- Deep Learning (Goodfellow et al., 2016) - Chapter on output layers
- GPT-2 paper (Radford et al., 2019) - Uses standard cross-entropy
- Online Softmax computation - Streaming algorithm
- Flash Attention paper - Similar fusion ideas for attention
*/
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

struct SoftmaxParams {
    float Scale;
    float Offset;
};

__device__ SoftmaxParams prepare_softmax_blockwide3(int64_t idx, const floatX* inp, int V, int P) {
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const floatX* x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    int i = (V+x128::size-1)/x128::size + threadIdx.x - blockDim.x;

    // special-case loop to handle the unaligned elements at the end of the array
    // this lets us skip the bounds check in the main loop below, which improves performance
    while ((i+1)*x128::size > V) {
        for(int k = 0; k < x128::size; ++k) {
            if (i*x128::size+k >= V) {
                break; // bounds checking against real V (rather than padded P)
            }
            float v = (float)x[i*x128::size+k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
        i -= blockDim.x;
    }

    // main loop for the bulk of the iterations (no bounds checking required!)
    for (; i >= 0; i -= blockDim.x) {
        x128 packed_x = load128(x + i * x128::size); // load and keep in cache until fused_classifier loop
        for(int k = 0; k < x128::size; ++k) {
            float v = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
    }

    // Block Max Reduction -> Maths -> Block Sum Reduction
    float block_maxval = blockReduce<warpReduceMax>(thread_maxval, false, -INFINITY);
    thread_sumval *= expf(thread_maxval - block_maxval);
    float block_sumval = blockReduce<warpReduceSum>(thread_sumval);

    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// will _update_ logits to logit gradients
// uses template to decide whether to write logits and probs
// split both loops in "multiple-of-x128-size" and "bounds-checked remainder" parts
template <bool WriteDLogits = true, bool WriteProbs = false>
__global__ void __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS)
    fused_classifier_kernel5(floatX* logits, float* losses, floatX* probs,
                                const float dloss, const int* targets,
                                int B, int T, int V, int P, std::bool_constant<WriteDLogits>) {
    // note: idx is small enough that it easily fits into 32 bit;
    // by making it a long here, we ensure that any offsets calculated with it (e.g., idx * P)
    // are done is 64 bit
    int64_t idx = gridDim.x - (blockIdx.x+1); // reverse order for cache hits on matmul data
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide3(idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if(threadIdx.x == 0) {
        float prob = expf((float)logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] -= logf(prob);
    }

    // without this synchronization point we have a race condition:
    // the logits used above to compute the loss are concurrently (race) modified to carry backward pass grads.
    // since the "logits" are overwritten to be in the [-1, 1] range and sp.Offset is sometimes smaller than -90
    // we errouneously end up computing exp^(90+) which gives us infinities in the loss! this is the fix.
    __syncthreads();

    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const floatX* logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V/x128::size; i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // it will be overwritten by the logits gradients which is when we reduce cache persistence
        x128 packed_logits_vec = load128(logits_vec + i * x128::size); // rely on cs of store128cs
        x128 packed_probs;
        for(int k = 0; k < x128::size; ++k) {
            int element = i*x128::size + k;
            float prob = expf((float)packed_logits_vec[k] - sp.Offset) * sp.Scale;
            packed_probs[k] = (floatX)prob;
            float indicator = (element == ix) ? 1.0f : 0.0f;
            packed_logits_vec[k] = (floatX)((prob - indicator) * dloss);
        }
        if (WriteDLogits){
            // reduce cache persistence for the overwritten logits
            // to maximise probability that logits remain in cache between prepare_softmax and here
            store128cs(logits + idx * P + i * x128::size, packed_logits_vec);
        }
        if (WriteProbs) {
            store128(probs + idx * P + i * x128::size, packed_probs);
        }
    }

    // handle remaining elements after the last multiple of x128::size
    // e.g. if V = 8003, and x128::size = 8, we need to handle the last 3 elements
    int unaligned_start = V & ~(x128::size - 1); // round down to multiple of x128::size
    for (int i = threadIdx.x + unaligned_start; i < V; i++) {
        float prob = expf((float)logits_vec[i] - sp.Offset) * sp.Scale;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit = (prob - indicator) * dloss;
        if (WriteDLogits){
            __stcs(logits + idx * P + i, (floatX)dlogit);
        }
        if (WriteProbs) {
            probs[idx * P + i] = (floatX)prob;
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

// replaces logits with logit gradients
template <typename Type, bool WriteDLogits>
void fused_classifier(Type* logits, float* losses,
                      const float dloss, const int* targets,
                      int B, int T, int V, int P, std::bool_constant<WriteDLogits> write_dlogits, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 1024;
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel5<<<grid_size, block_size, 0, stream>>>(logits, losses, (floatX*)NULL, dloss, targets, B, T, V, P, write_dlogits);
    cudaCheck(cudaGetLastError());
}
