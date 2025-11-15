/*
================================================================================
Layer Normalization Backward Pass - CUDA Kernel Implementations
================================================================================

PURPOSE:
This file explores progressive optimizations for LayerNorm backward pass on GPU.
The backward pass computes gradients with respect to inputs, weights, and biases
during backpropagation in neural network training.

LAYERNORM BACKWARD PASS ALGORITHM:
The backward pass is significantly more complex than the forward pass due to:
1. Computing gradients for THREE outputs (dinp, dweight, dbias)
2. Chain rule dependencies between gradients
3. Reduction operations across batch dimension for dweight/dbias

MATHEMATICAL FORMULATION:
Given upstream gradient dout and forward pass values (inp, mean, rstd):

For each position (b,t):
  norm[i] = (inp[i] - mean) * rstd
  out[i] = norm[i] * weight[i] + bias[i]

Gradients:
  dbias[i] = sum over (b,t) of dout[b,t,i]
  dweight[i] = sum over (b,t) of dout[b,t,i] * norm[b,t,i]

  For dinp, we need chain rule through normalization:
  dnorm[i] = dout[i] * weight[i]

  dinp requires careful derivative of normalization:
  dinp[i] = rstd * (dnorm[i] - mean(dnorm) - norm[i] * mean(dnorm * norm))

  Where:
    mean(dnorm) = sum(dnorm) / C
    mean(dnorm * norm) = sum(dnorm * norm) / C

COMPLEXITY ANALYSIS:
The backward pass is computationally more expensive than forward:
1. Forward: 2 reductions (mean, variance) + 1 pass (normalization)
2. Backward: 2 reductions per row (dnorm stats) + accumulation to global arrays
3. CRITICAL CHALLENGE: dweight and dbias require ATOMIC operations across batch

ATOMIC OPERATION CHALLENGE:
- dweight[i] and dbias[i] must accumulate contributions from all B*T positions
- Multiple thread blocks write to same locations → need atomic operations
- Atomics are SLOW, especially for fp16/bf16
- Major optimization target: minimize atomic operations

PERFORMANCE CONSIDERATIONS:
- Memory-bound like forward pass, but with additional atomic bottleneck
- Atomics to global memory are expensive (100+ cycles)
- Shared memory atomics are faster (20-30 cycles)
- Key strategies:
  1. Use shared memory to accumulate per-block, then atomic once to global
  2. Use scratch buffers to avoid contention
  3. Use warp-level reductions to minimize atomics

OPTIMIZATION PROGRESSION:
Version 1:  Naive with global atomics
Version 2:  Shared memory accumulation + templates
Version 3:  Grid-striding + reduced atomic contention
Version 4:  atomicCAS for bf16 (compare-and-swap)
Version 5:  FP32 scratchpad per block
Version 6:  Single FP32 scratchpad for all blocks
Version 7:  Remove cooperative groups (simpler)
Version 8:  Vectorization with x128 + bank conflict avoidance
Version 9:  Inter-warp reduction without atomics
Version 10: Vector-friendly shared memory layout (most optimized)

TARGET DIMENSIONS:
- B*T (batch × sequence): typically 8K-32K positions
- C (channels): typically 768-1024 for transformers

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt layernorm_backward.cu -o layernorm_backward

Usage:
./layernorm_backward [kernel_version]
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#define ENABLE_BF16
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void layernorm_forward_cpu(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
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
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

void layernorm_backward_cpu(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * C + t * C;
            const float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            const float mean_bt = mean[b * T + t];
            const float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

// ============================================================================
// GPU KERNELS
// ============================================================================

/*
ATOMIC OPERATION HELPERS FOR MIXED PRECISION
---------------------------------------------
These functions enable atomic additions for fp16/bf16 types, which don't
have native atomicAdd support in CUDA. The implementation uses a clever trick:
- Align to the nearest 32-bit boundary
- Load the paired value (fp16/bf16 is 16-bit, so we work with pairs)
- Use atomicAdd on the 32-bit pair
- This allows atomic operations on 16-bit types at the cost of some overhead
*/
#ifdef ENABLE_BF16
__device__ void atomicAddX(__nv_bfloat16* addr, __nv_bfloat16 val) {
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(addr);
    __nv_bfloat162* ptr_bf16 = reinterpret_cast<__nv_bfloat162*>(ptr_val & ~uintptr_t(0x3));

    // Prepare the value to add, setting the other half to zero
    __nv_bfloat162 add_val = (ptr_val & 0x3) ? __halves2bfloat162(__ushort_as_bfloat16(0), val)
                                             : __halves2bfloat162(val, __ushort_as_bfloat16(0));
    atomicAdd(ptr_bf16, add_val);
}
#endif
#ifdef ENABLE_FP16
__device__ void atomicAddX(half* addr, half val) {
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(addr);
    half2* ptr_fp16 = reinterpret_cast<half2*>(ptr_val & ~uintptr_t(0x3));

    // Prepare the value to add, setting the other half to zero
    half2 add_val = (ptr_val & 0x3) ? __halves2half2(__ushort_as_half(0), val)
                                    : __halves2half2(val, __ushort_as_half(0));
    atomicAdd(ptr_fp16, add_val);
}
#endif
__device__ void atomicAddX(float* addr, float val) {
    atomicAdd(addr, val);
}

/*
KERNEL 1: Naive Backward Pass
------------------------------
APPROACH:
- Direct translation from CPU code
- One thread per sequence position (B*T threads)
- Each thread processes entire C dimension sequentially
- Uses global atomics for dweight and dbias

ALGORITHM:
1. Compute two statistics via reduction over C:
   - dnorm_mean: mean of gradients through normalized values
   - dnorm_norm_mean: mean of gradients weighted by normalized values
2. Apply chain rule to compute all three gradients:
   - dinp: gradient w.r.t. input
   - dweight: gradient w.r.t. weight (needs atomic)
   - dbias: gradient w.r.t. bias (needs atomic)

THE THREE GRADIENT TERMS FOR dinp:
The gradient formula has three terms from the chain rule:
  dinp[i] = rstd * (dnorm[i] - mean(dnorm) - norm[i] * mean(dnorm * norm))

Term 1 (dnorm[i]): Direct gradient through the normalized value
Term 2 (mean(dnorm)): Correction for mean subtraction in normalization
Term 3 (norm[i] * mean(dnorm * norm)): Correction for variance normalization

This comes from differentiating through the normalization operation.

ATOMIC OPERATIONS:
- Every thread atomically adds to dweight[i] and dbias[i]
- For B*T=8192 positions, each of C elements gets 8192 atomic operations!
- This creates MASSIVE contention on global memory
- Atomics serialize execution → major bottleneck

PERFORMANCE CHARACTERISTICS:
- PROS: Simple, correct, easy to understand
- CONS:
  * Sequential loops over C (slow)
  * Two passes over data (compute stats, then gradients)
  * EXTREMELY slow global atomics (100+ cycles each)
  * Severe atomic contention

EXPECTED PERFORMANCE:
- Baseline (1.0x) - this is our reference
- Dominated by atomic operation latency
*/
__global__ void layernorm_backward_kernel1(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B*T) return;
    int b = idx / T;
    int t = idx % T;

    const float* dout_bt = dout + b * T * C + t * C;
    const float* inp_bt = inp + b * T * C + t * C;
    float* dinp_bt = dinp + b * T * C + t * C;
    const float mean_bt = mean[b * T + t];
    const float rstd_bt = rstd[b * T + t];

    // PASS 1: Compute two reduction statistics over C dimension
    // These are needed for the gradient formula
    float dnorm_mean = 0.0f;        // mean(dnorm)
    float dnorm_norm_mean = 0.0f;   // mean(dnorm * norm)
    for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // PASS 2: Compute and accumulate all gradients
    for (int i = 0; i < C; i++) {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];

        // Gradient contribution to bias (accumulated across all B*T positions)
        // ATOMIC: many threads contend for same dbias[i]
        atomicAdd(&dbias[i], dout_bt[i]);

        // Gradient contribution to weight (accumulated across all B*T positions)
        // ATOMIC: many threads contend for same dweight[i]
        atomicAdd(&dweight[i], norm_bti * dout_bt[i]);

        // Gradient contribution to input (no atomic needed, different memory location per thread)
        float dval = 0.0f;
        dval += dnorm_i;                        // term 1: direct gradient
        dval -= dnorm_mean;                     // term 2: mean correction
        dval -= norm_bti * dnorm_norm_mean;     // term 3: variance correction
        dval *= rstd_bt;                        // final scaling by rstd
        dinp_bt[i] += dval;
    }
}

/*
KERNEL 2: Shared Memory Accumulation with Templates
----------------------------------------------------
APPROACH:
- Use shared memory to accumulate dweight/dbias within each block
- Only atomic to global memory once per block (not once per thread!)
- Use warp-level primitives for reductions
- Template support for mixed precision (fp16/bf16/fp32)

KEY OPTIMIZATION: TWO-LEVEL ACCUMULATION
1. Level 1 (Shared): Warp atomically adds to block-level shared memory
2. Level 2 (Global): Block atomically adds to temporary global buffer
3. Final: Separate kernel copies from temp buffer to output

WHY SHARED MEMORY ATOMICS?
- Shared memory atomics are 3-5x faster than global atomics
- 32 warps in a block → only 32 shared memory atomics (instead of 1024 global)
- Dramatically reduces contention

SHARED MEMORY LAYOUT:
- dbias_shared[C]: accumulator for bias gradients
- dweight_shared[C]: accumulator for weight gradients
- Total: 2*C floats

TEMP BUFFER STRATEGY:
- dweight_tmp and dbias_tmp are FP32 temporary buffers
- Allows high-precision accumulation even with fp16/bf16 output
- Separate copy kernel converts FP32 → output precision

WARP-LEVEL REDUCTIONS:
- Each warp computes dnorm_mean and dnorm_norm_mean using cg::reduce
- Much faster than block-level reduction (no shared memory needed)

PERFORMANCE vs KERNEL 1:
- PROS:
  * Shared memory atomics much faster than global
  * Warp-level reductions efficient
  * Mixed precision support
- CONS:
  * Still one block per row (limited parallelism)
  * Extra temp buffer and copy kernel
  * Multiple kernel launches

EXPECTED PERFORMANCE:
- Typically 3-5x faster than kernel 1
*/
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
__global__ void layernorm_backward_kernel2(Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, float* dweight_tmp, float* dbias_tmp) {
    extern __shared__ float shared[]; // size = 2 * C

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    int N = B * T;
    if(idx >= N) { return; }

    int b = idx / T;
    int t = idx % T;

    const Tdout* dout_bt = dout + b * T * C + t * C;
    const Trest* inp_bt = inp + b * T * C + t * C;
    Tdinp* dinp_bt = dinp + b * T * C + t * C;
    const float mean_bt = (float)mean[b * T + t];
    const float rstd_bt = (float)rstd[b * T + t];

    // Shared memory layout: first half for bias, second half for weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // Initialize shared memory accumulators to zero
    #pragma unroll
    for(int i = threadIdx.x; i < C; i+= blockDim.x){
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    __syncthreads();

    // Compute reduction statistics using warp primitives
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = warp.thread_rank(); i < C; i  += warp.size()) {
        float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = (float)weight[i] * (float)dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = cg::reduce(warp, dnorm_mean, cg::plus<float>{});
    dnorm_norm_mean = cg::reduce(warp, dnorm_norm_mean, cg::plus<float>{});
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // Accumulate gradients to SHARED memory (fast atomics!)
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = (float)weight[i] * (float)dout_bt[i];

        // SHARED MEMORY ATOMIC (fast!)
        atomicAdd(&dbias_shared[i], (float)dout_bt[i]);
        atomicAdd(&dweight_shared[i], norm_bti * (float)dout_bt[i]);

        // dinp gradient (no atomic needed)
        float dval = 0.0f;
        dval += dnorm_i;                        // term 1
        dval -= dnorm_mean;                     // term 2
        dval -= norm_bti * dnorm_norm_mean;     // term 3
        dval *= rstd_bt;
        dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
    }
    __syncthreads();

    // Write block-level accumulation to GLOBAL temp buffer
    // Each block writes once (much better than each thread!)
    for(int i = threadIdx.x; i < C; i+= blockDim.x) {
        atomicAdd(&dbias_tmp[i], dbias_shared[i]);
        atomicAdd(&dweight_tmp[i], dweight_shared[i]);
    }
}

/*
Helper kernel to copy from FP32 temp buffer to output type
This allows high-precision accumulation with mixed precision output
*/
template <typename Tparams>
__global__ void copy_to_dweight_dbias(int C, Tparams* dbias, Tparams* dweight, float* dbias_tmp, float* dweight_tmp) {
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < C; i += blockDim.x * gridDim.x) {
        dbias[i] = (Tparams)dbias_tmp[i];
        dweight[i] = (Tparams)dweight_tmp[i];
    }
}

/*
KERNEL 3: Grid-Striding Pattern for Reduced Atomic Contention
---------------------------------------------------------------
APPROACH:
- Use grid-striding loop: each warp processes multiple rows
- Fewer blocks → fewer atomic operations to global memory
- Accumulate many rows per block before writing to global

GRID-STRIDING PATTERN:
Instead of one warp per row, we use:
  for (int idx = base_idx; idx < B*T; idx += warps_in_grid)

This means:
- Warp 0 processes rows 0, warps_in_grid, 2*warps_in_grid, ...
- Each warp handles multiple rows, accumulating in shared memory
- MUCH fewer atomic operations to global memory

ATOMIC REDUCTION STRATEGY:
- Level 1: Warp accumulates in shared memory across multiple rows
- Level 2: Block writes to global memory ONCE (using atomicAddX for mixed precision)

WHY THIS IS BETTER:
- If we have 108 SMs and 32 warps/block:
  * We can use ~108 blocks (one per SM)
  * Each processes B*T/108 rows
  * Only 108 atomic operations per element (vs B*T in kernel 1!)

CACHE STREAMING:
- Uses __ldcs for loads (cache streaming)
- Indicates data won't be reused → better cache utilization

PERFORMANCE vs KERNEL 2:
- PROS:
  * Dramatically fewer atomic operations
  * Better SM utilization
  * No temp buffer needed
  * Single kernel launch
- CONS:
  * More complex indexing
  * Potential load imbalance if B*T not divisible by warps_in_grid

EXPECTED PERFORMANCE:
- Typically 1.5-2x faster than kernel 2 (6-10x faster than kernel 1)
*/
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
__global__ void layernorm_backward_kernel3(Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C) {
    extern __shared__ float shared[]; // size = 2 * C

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int base_idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();

    // Shared memory layout
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // Initialize shared memory
    #pragma unroll 4
    for(int i = threadIdx.x; i < C; i+= blockDim.x){
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    __syncthreads();

    // GRID-STRIDING LOOP: each warp processes multiple rows
    int warps_in_grid = gridDim.x * warp.meta_group_size();
    for (int idx = base_idx; idx < B * T; idx += warps_in_grid) {
        int b = idx / T;
        int t = idx % T;

        const Tdout* dout_bt = dout + b * T * C + t * C;
        const Trest* inp_bt = inp + b * T * C + t * C;
        Tdinp* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // Compute reduction statistics
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warp.thread_rank(); i < C; i  += warp.size()) {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = cg::reduce(warp, dnorm_mean, cg::plus<float>{});
        dnorm_norm_mean = cg::reduce(warp, dnorm_norm_mean, cg::plus<float>{});
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // Accumulate gradients with cache streaming
        for (int i = warp.thread_rank(); i < C; i += warp.size()) {
            float dout_i = (float)__ldcs(&dout_bt[i]);       // streaming load
            float norm_bti = ((float)__ldcs(&inp_bt[i]) - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;

            // Accumulate to shared memory
            atomicAdd(&dbias_shared[i], dout_i);
            atomicAdd(&dweight_shared[i], norm_bti * dout_i);

            // dinp gradient
            float dval = 0.0f;
            dval += dnorm_i;
            dval -= dnorm_mean;
            dval -= norm_bti * dnorm_norm_mean;
            dval *= rstd_bt;
            dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
        }
    }
    __syncthreads();

    // Write accumulated gradients to global memory (ONCE per block, not per warp!)
    for(int i = threadIdx.x; i < C; i+= blockDim.x) {
        atomicAddX(&dbias[i], (Tparams)dbias_shared[i]);
        atomicAddX(&dweight[i], (Tparams)dweight_shared[i]);
    }
}

/*
KERNEL 4: atomicCAS for BF16 (Compare-And-Swap)
------------------------------------------------
APPROACH:
- Same as kernel 3, but uses atomicCAS instead of atomicAdd for bf16
- atomicCAS = atomic Compare-And-Swap (lock-free synchronization)

ATOMICAS TECHNIQUE:
For bf16/fp16 which lack native atomicAdd:
1. Read current value from memory
2. Compute new value = current + delta
3. atomicCAS attempts to write new value if current hasn't changed
4. If changed by another thread, retry with updated current value

This is a lock-free retry loop that ensures correctness without locks.

WHY atomicCAS vs atomicAdd:
- atomicAdd for bf16 requires pairing (operates on 2 values at once)
- atomicCAS can update single bf16 value (but may retry multiple times)
- Trade-off: simpler indexing vs potential retry overhead

PERFORMANCE vs KERNEL 3:
- Similar for low contention
- May be slower with high contention (more retries)
- Benefit: cleaner for non-paired memory layouts
*/
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
__global__ void layernorm_backward_kernel4(Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C) {
    extern __shared__ float shared[]; // size = 2 * C

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int base_idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for(int i = threadIdx.x; i < C; i+= blockDim.x){
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    __syncthreads();

    int warps_in_grid = gridDim.x * warp.meta_group_size();
    for (int idx = base_idx; idx < B * T; idx += warps_in_grid) {
        int b = idx / T;
        int t = idx % T;

        const Tdout* dout_bt = dout + b * T * C + t * C;
        const Trest* inp_bt = inp + b * T * C + t * C;
        Tdinp* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warp.thread_rank(); i < C; i  += warp.size()) {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = cg::reduce(warp, dnorm_mean, cg::plus<float>{});
        dnorm_norm_mean = cg::reduce(warp, dnorm_norm_mean, cg::plus<float>{});
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = warp.thread_rank(); i < C; i += warp.size()) {
            float dout_i = (float)__ldcs(&dout_bt[i]);
            float norm_bti = ((float)__ldcs(&inp_bt[i]) - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;
            // gradient contribution to bias
            atomicAdd(&dbias_shared[i], dout_i);
            // gradient contribution to weight
            atomicAdd(&dweight_shared[i], norm_bti * dout_i);
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
        }
    }
    __syncthreads();

    __nv_bfloat162* dbiasVec2 = reinterpret_cast<__nv_bfloat162*>(dbias);
    __nv_bfloat162* dweightVec2 = reinterpret_cast<__nv_bfloat162*>(dweight);

    // write to global memory
    for(int i = threadIdx.x; i < C/2; i+= blockDim.x) {
        __nv_bfloat162 add_dbias = __halves2bfloat162((__nv_bfloat16)dbias_shared[i*2], (__nv_bfloat16)dbias_shared[i*2+1]);
        __nv_bfloat162 add_dweight = __halves2bfloat162((__nv_bfloat16)dweight_shared[i*2], (__nv_bfloat16)dweight_shared[i*2+1]);

        // Get the current value from L2 cache
        __nv_bfloat162 current_dbias = __ldcg(&dbiasVec2[i]);
        __nv_bfloat162 current_dweight = __ldcg(&dweightVec2[i]);

        // Add the two values
        __nv_bfloat162 new_dbias = add_dbias + current_dbias;
        __nv_bfloat162 new_dweight = add_dweight + current_dweight;

        // Write the result back to L2 cache using 32-bit integer atomic compare and exchange
        unsigned int current_dbias32b = *reinterpret_cast<unsigned int*>(&current_dbias);
        unsigned int current_dweight32b = *reinterpret_cast<unsigned int*>(&current_dweight);

        unsigned int new_dbias32b = *reinterpret_cast<unsigned int*>(&new_dbias);
        unsigned int new_dweight32b = *reinterpret_cast<unsigned int*>(&new_dweight);

        unsigned int old_dbias32b = atomicCAS((unsigned int*)&dbiasVec2[i], current_dbias32b, new_dbias32b);
        unsigned int old_dweight32b = atomicCAS((unsigned int*)&dweightVec2[i], current_dweight32b, new_dweight32b);

        // If the value has changed between read and atomic, we need to try again
        while (old_dbias32b != current_dbias32b) {
            current_dbias32b = old_dbias32b;
            new_dbias = *reinterpret_cast<__nv_bfloat162*>(&current_dbias32b) + add_dbias;
            new_dbias32b = *reinterpret_cast<unsigned int*>(&new_dbias);
            old_dbias32b = atomicCAS((unsigned int*)&dbiasVec2[i], current_dbias32b, new_dbias32b);
        }

        while (old_dweight32b != current_dweight32b) {
            current_dweight32b = old_dweight32b;
            new_dweight = *reinterpret_cast<__nv_bfloat162*>(&current_dweight32b) + add_dweight;
            new_dweight32b = *reinterpret_cast<unsigned int*>(&new_dweight);
            old_dweight32b = atomicCAS((unsigned int*)&dweightVec2[i], current_dweight32b, new_dweight32b);
        }
    }
}

/*
KERNEL 5: FP32 Scratchpad Per Block with Flag-Based Synchronization
--------------------------------------------------------------------
APPROACH:
- Each block writes its accumulation to a private scratchpad in global memory
- Use atomic flag to coordinate: last block to finish does the final reduction
- NO atomics during accumulation (only one atomic increment of flag)

SCRATCHPAD STRATEGY:
- scratch buffer layout: [flag][block0_dbias][block0_dweight][block1_dbias]...
- Each block accumulates in shared memory, then writes to its scratchpad
- Last block (detected via atomic flag) sums all scratchpads

FLAG-BASED SYNCHRONIZATION:
1. Each block increments atomic flag when done
2. Block that sees flag == gridDim.x-1 is the last one
3. Last block performs final reduction across all scratchpads
4. Avoids atomic contention during accumulation!

ADVANTAGES:
- Only ONE atomic operation per block (the flag increment)
- All accumulation is conflict-free writes to separate memory
- Final reduction is serial but only happens once

PERFORMANCE vs KERNEL 3:
- PROS:
  * Eliminates atomic contention completely
  * Clean separation of phases
  * High precision with FP32 scratch
- CONS:
  * Requires scratch buffer (extra memory)
  * Last block does more work (potential imbalance)
  * Memory traffic for scratchpad writes/reads

EXPECTED PERFORMANCE:
- Typically 1.2-1.4x faster than kernel 3 for large batches
*/
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
__global__ void layernorm_backward_kernel5(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C) {
    extern __shared__ float shared[]; // size = 2 * C + 1

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int base_idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for(int i = threadIdx.x; i < C; i+= blockDim.x){
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    unsigned int *tmp_flag = (unsigned int*)(shared + C*2);
    __syncthreads();

    int warps_in_grid = gridDim.x * warp.meta_group_size();
    for (int idx = base_idx; idx < B * T; idx += warps_in_grid) {
        int b = idx / T;
        int t = idx % T;

        const Tdout* dout_bt = dout + b * T * C + t * C;
        const Trest* inp_bt = inp + b * T * C + t * C;
        Tdinp* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warp.thread_rank(); i < C; i  += warp.size()) {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = cg::reduce(warp, dnorm_mean, cg::plus<float>{});
        dnorm_norm_mean = cg::reduce(warp, dnorm_norm_mean, cg::plus<float>{});
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = warp.thread_rank(); i < C; i += warp.size()) {
            float dout_i = (float)__ldcs(&dout_bt[i]);
            float norm_bti = ((float)__ldcs(&inp_bt[i]) - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;
            // gradient contribution to bias
            atomicAdd(&dbias_shared[i], dout_i);
            // gradient contribution to weight
            atomicAdd(&dweight_shared[i], norm_bti * dout_i);
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
        }
    }
    __syncthreads();

    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C * gridDim.x;
    unsigned int* scratchFlag = (unsigned int*)(scratch + (2 * C * gridDim.x));

    for(int i = threadIdx.x; i < C; i+= blockDim.x) {
        scratch_dbias[i + C*blockIdx.x] = dbias_shared[i];
        scratch_dweight[i + C*blockIdx.x] = dweight_shared[i];
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        *tmp_flag = atomicAdd(scratchFlag, 1);
    }
    __syncthreads();
    if (*tmp_flag == gridDim.x-1) {
        // last block to finish, accumulate the scratchpad
        for (int i = threadIdx.x; i < C; i += blockDim.x) {
            float dbias_sum = 0.0f;
            float dweight_sum = 0.0f;
            #pragma unroll 8
            for (int j = 0; j < gridDim.x; j++) {
                dbias_sum += scratch_dbias[i + j*C];
                dweight_sum += scratch_dweight[i + j*C];
            }
            dbias[i] = (Tparams)((float)dbias[i] + dbias_sum);
            dweight[i] = (Tparams)((float)dweight[i] + dweight_sum);
        }
    }
}

/*
KERNEL 6: Single Shared Scratchpad (Memory-Optimized)
------------------------------------------------------
APPROACH:
- Same algorithm as kernel 5, but with SINGLE shared scratchpad for all blocks
- Reduces memory footprint significantly
- Uses atomics to scratchpad, but then flag-based final reduction

MEMORY OPTIMIZATION:
Kernel 5: scratch size = gridDim.x * 2 * C * sizeof(float)
Kernel 6: scratch size = 2 * C * sizeof(float) + flag

For 108 blocks, C=768:
- Kernel 5: 108 * 2 * 768 * 4 = 662KB
- Kernel 6: 2 * 768 * 4 = 6KB
This is 100x less memory!

TRADE-OFF:
- Kernel 5: No atomics during accumulation, but high memory
- Kernel 6: Atomics to shared scratchpad, but low memory

The scratchpad atomics are less contended than direct atomics to dweight/dbias
because all blocks share the work of incrementing the same locations.

PERFORMANCE vs KERNEL 5:
- PROS: Dramatically less memory (important for large models)
- CONS: Re-introduces some atomic contention (to scratchpad)
- Usually similar performance, sometimes slightly slower
*/
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
__global__ void layernorm_backward_kernel6(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C) {
    extern __shared__ float shared[]; // size = 2 * C + 1

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int base_idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for(int i = threadIdx.x; i < C; i+= blockDim.x){
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    unsigned int *tmp_flag = (unsigned int*)(shared + C*2);
    __syncthreads();

    int warps_in_grid = gridDim.x * warp.meta_group_size();
    for (int idx = base_idx; idx < B * T; idx += warps_in_grid) {
        int b = idx / T;
        int t = idx % T;

        const Tdout* dout_bt = dout + b * T * C + t * C;
        const Trest* inp_bt = inp + b * T * C + t * C;
        Tdinp* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warp.thread_rank(); i < C; i  += warp.size()) {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = cg::reduce(warp, dnorm_mean, cg::plus<float>{});
        dnorm_norm_mean = cg::reduce(warp, dnorm_norm_mean, cg::plus<float>{});
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = warp.thread_rank(); i < C; i += warp.size()) {
            float dout_i = (float)__ldcs(&dout_bt[i]);
            float norm_bti = ((float)__ldcs(&inp_bt[i]) - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;
            // gradient contribution to bias
            atomicAdd(&dbias_shared[i], dout_i);
            // gradient contribution to weight
            atomicAdd(&dweight_shared[i], norm_bti * dout_i);
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
        }
    }

    // Accumulate into a FP32 scratchpad
    // BF16 atomics are potentially much slower... and this is more precise!
    __syncthreads();
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    unsigned int* scratchFlag = (unsigned int*)(scratch + (2 * C));
    for(int i = threadIdx.x; i < C; i+= blockDim.x) {
        atomicAdd(&scratch_dbias[i], dbias_shared[i]);
        atomicAdd(&scratch_dweight[i], dweight_shared[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *tmp_flag = atomicAdd(scratchFlag, 1);
    }
    __syncthreads();
    if (*tmp_flag == gridDim.x-1) {
        for(int i = threadIdx.x; i < C; i+= blockDim.x) {
            // todo - potentially do stochastic rounding here as well
            dbias[i] = (Tparams)scratch_dbias[i];
            dweight[i] = (Tparams)scratch_dweight[i];
        }
    }
}


/*
KERNEL 7: Simplified Without Cooperative Groups
------------------------------------------------
APPROACH:
- Same algorithm as kernel 6
- Removes cooperative groups dependency
- Uses manual warp reduction instead of cg::reduce

SIMPLIFICATION:
- Uses warpReduceSum() helper function instead of cooperative groups
- Computes warp ID and lane ID manually
- No templates - uses floatX typedef

WHY REMOVE COOPERATIVE GROUPS:
- Simpler code, easier to understand and modify
- Slightly less overhead from cooperative groups API
- More control over warp-level operations

This is a teaching/debugging version that's more explicit about operations.

PERFORMANCE vs KERNEL 6:
- Essentially identical (same algorithm, different API)
- May be very slightly faster due to less abstraction overhead
*/
__global__ void layernorm_backward_kernel7(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                        const floatX* dout, const floatX* inp, const floatX* weight, const floatX* mean, const floatX* rstd,
                        int B, int T, int C) {
    extern __shared__ float shared[]; // size = 2 * C + 1
    int warpId = threadIdx.x / warpSize; // warp index within a block
    int warpsInBlock = blockDim.x / warpSize;
    int base_idx = blockIdx.x * warpsInBlock + warpId;
    int warpThreadIdx = threadIdx.x % warpSize; // Thread index within the warp
    int warps_in_grid = gridDim.x * warpsInBlock;

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    #pragma unroll 4
    for(int i = threadIdx.x; i < C; i+= blockDim.x){
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    unsigned int *tmp_flag = (unsigned int*)(shared + C*2);
    __syncthreads();

    for (int idx = base_idx; idx < B * T; idx += warps_in_grid) {
        int b = idx / T;
        int t = idx % T;

        const floatX* dout_bt = dout + b * T * C + t * C;
        const floatX* inp_bt = inp + b * T * C + t * C;
        floatX* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx; i < C; i  += warpSize) {
            float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * (float)dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = warpReduceSum(dnorm_mean);
        dnorm_norm_mean = warpReduceSum(dnorm_norm_mean);

        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = warpThreadIdx; i < C; i += warpSize) {
            float dout_i = (float)__ldcs(&dout_bt[i]);
            float norm_bti = ((float)__ldcs(&inp_bt[i]) - mean_bt) * rstd_bt;
            float dnorm_i = (float)weight[i] * dout_i;
            // gradient contribution to bias
            atomicAdd(&dbias_shared[i], dout_i);
            // gradient contribution to weight
            atomicAdd(&dweight_shared[i], norm_bti * dout_i);
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] = (floatX)((float)dinp_bt[i] + dval);
        }
    }

    // Accumulate into a FP32 scratchpad
    // BF16 atomics are potentially much slower... and this is more precise!
    __syncthreads();
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    unsigned int* scratchFlag = (unsigned int*)(scratch + (2 * C));
    for(int i = threadIdx.x; i < C; i+= blockDim.x) {
        atomicAdd(&scratch_dbias[i], dbias_shared[i]);
        atomicAdd(&scratch_dweight[i], dweight_shared[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *tmp_flag = atomicAdd(scratchFlag, 1);
    }
    __syncthreads();
    if (*tmp_flag == gridDim.x-1) {
        for(int i = threadIdx.x; i < C; i+= blockDim.x) {
            // todo - potentially do stochastic rounding here as well
            dbias[i] = (floatX)scratch_dbias[i];
            dweight[i] = (floatX)scratch_dweight[i];
        }
    }
}

/*
KERNEL 8: Vectorization with Bank Conflict Avoidance
-----------------------------------------------------
APPROACH:
- Add vectorized loads/stores using x128 (128-bit = 4 elements)
- Solve shared memory bank conflict problem
- Reorder indexing for optimal shared memory access

VECTORIZATION (x128):
- Load/store 4 floats at once instead of 1
- Reduces memory transactions by 4x
- Improves memory bandwidth utilization

BANK CONFLICT PROBLEM:
When using vectorized loads, naive indexing causes 8-way bank conflicts:
- x128 loads 4 elements, but atomics are 32-bit (1 element)
- Threads in warp access: [0,4,8,12,...] - stride of 4
- This maps to same bank repeatedly → bank conflict!

SOLUTION: Reorder shared memory indexing
Instead of: shared[global_index]
Use: shared[shared_index] where shared_index spreads accesses
Layout: shared[warpThread + iteration*WARP_SIZE] instead of [thread*4 + iteration]

This ensures consecutive threads access consecutive banks (no conflicts).

FINAL REORDERING:
After accumulation, reorder from shared-memory-friendly to global-memory-friendly
indexing before writing final results.

PERFORMANCE vs KERNEL 7:
- PROS:
  * 4x fewer memory transactions (vectorization)
  * No shared memory bank conflicts
  * Better memory bandwidth
- CONS:
  * More complex indexing
  * Reordering overhead
  * Requires C divisible by 4*32=128

EXPECTED PERFORMANCE:
- Typically 1.3-1.5x faster than kernel 7
- Can achieve 60-70% of peak memory bandwidth
*/
__global__ void __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS)
                layernorm_backward_kernel8(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                                            const floatX* dout, const floatX* inp, const floatX* weight,
                                            const floatX* mean, const floatX* rstd,
                                            int B, int T, int C) {
    extern __shared__ float shared[]; // size = 2 * C + 1
    int warpId = threadIdx.x / warpSize; // warp index within a block
    int warpsInBlock = blockDim.x / warpSize; //number of warps in block
    int baseIdx = blockIdx.x * warpsInBlock + warpId;
    int warpThreadIdx = threadIdx.x % warpSize; // Thread index within the warp
    int warpsInGrid = gridDim.x * warpsInBlock;
    int C_per_iteration = warpSize * x128::size;
    int iterations_C = C / C_per_iteration;

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;

    // init shared memory to zero
    for(int i = threadIdx.x; i < C; i+= blockDim.x){
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    unsigned int *tmp_flag = (unsigned int*)(shared + C*2);
    __syncthreads();

    for (int idx = baseIdx; idx < B * T; idx += warpsInGrid) {
        int b = idx / T;
        int t = idx % T;

        const floatX* dout_bt = dout + b * T * C + t * C;
        const floatX* inp_bt = inp + b * T * C + t * C;
        floatX* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += warpSize * x128::size) {
            x128 dout128_i   = load128(dout_bt + i);
            x128 inp128_i    = load128(inp_bt  + i);
            x128 weight128_i = load128(weight  + i);
            for (int k = 0; k < x128::size; k++) {
                float norm_bti = ((float)inp128_i[k] - mean_bt) * rstd_bt;
                float dnorm_i = (float)weight128_i[k] * (float)dout128_i[k];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
        }
        dnorm_mean = warpReduceSum(dnorm_mean) / C;
        dnorm_norm_mean = warpReduceSum(dnorm_norm_mean) / C;

        // now iterate again and accumulate all the gradients
        // unfortunately we cannot use the same index for x128 arrays and shared memory
        // as atomics can only be 32-bit rather than 128-bit (at least pre-SM90/Hopper)
        // so this would result in an 8-way bank conflict, and kill performance
        // so instead, we use a shared memory friendly index, and reorder before the final write
        for (int i = 0; i < iterations_C; i++) {
            int global_index = (warpThreadIdx * x128::size) + (i * C_per_iteration);
            int shared_index = warpThreadIdx + (i * C_per_iteration);
            x128 dout128   = load128cs(dout_bt + global_index);
            x128 inp128    = load128cs(inp_bt  + global_index);
            x128 dinp128   = load128(dinp_bt   + global_index);
            x128 weight128 = load128(weight    + global_index);

            for (int x = 0; x < x128::size; x++) {
                float dout_i = (float)dout128[x];
                float norm_bti = ((float)inp128[x] - mean_bt) * rstd_bt;
                float dnorm_i = (float)weight128[x] * dout_i;
                // gradient contribution to bias (using shared memory friendly index)
                atomicAdd(&dbias_shared[shared_index + x*warpSize], dout_i);
                // gradient contribution to weight (using shared memory friendly index)
                atomicAdd(&dweight_shared[shared_index + x*warpSize], norm_bti * dout_i);
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp128[x] = (floatX)((float)dinp128[x] + dval);
            }
            // cache in L2 as this is read by the next kernel, but bypass L1 to minimise thrashing
            store128cg(dinp_bt + global_index, dinp128);
        }
    }
    // Accumulate into a FP32 scratchpad
    // BF16 atomics are potentially much slower... and this is more precise!
    // todo - could potentially avoid the extra copy if floatX is FP32, fairly negligible though
    __syncthreads();
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    unsigned int* scratchFlag = (unsigned int*)(scratch + (2 * C));
    for(int i = threadIdx.x; i < C; i+= blockDim.x) {
        // global atomics in the same "shared memory banking friendly" order
        atomicAdd(&scratch_dbias[i], dbias_shared[i]);
        atomicAdd(&scratch_dweight[i], dweight_shared[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *tmp_flag = atomicInc(scratchFlag, gridDim.x);
    }
    __syncthreads();
    if (*tmp_flag == gridDim.x-1) {
        for (int i = warpId; i < iterations_C; i += warpsInBlock) {
            // reorder from atomic/shared memory-friendly index to real global memory index
            // and convert from float/FP32 to floatX/BF16 for the final write
            int global_index = (warpThreadIdx * x128::size) + (i * C_per_iteration);
            int shared_index = warpThreadIdx + (i * C_per_iteration);

            x128 dbias128 = load128(dbias + global_index);
            x128 dweight128 = load128(dweight + global_index);
            for (int x = 0; x < x128::size; x++) {
                float s_db = scratch_dbias[shared_index + x*warpSize];
                float s_dw = scratch_dweight[shared_index + x*warpSize];
                dbias128[x] = (floatX)(s_db + (float)dbias128[x]);
                dweight128[x] = (floatX)(s_dw + (float)dweight128[x]);
            }
            store128(dbias + global_index, dbias128);
            store128(dweight + global_index, dweight128);
        }
    }
}

/*
KERNEL 9: Inter-Warp Reduction Without Atomics
-----------------------------------------------
APPROACH:
- Eliminate shared memory atomics by using inter-warp reduction
- Multiple warps in block cooperate using shared memory as buffer
- Only warp 0 actually writes to shared memory (no atomics!)

INTER-WARP REDUCTION TECHNIQUE:
Instead of each warp atomically adding to shared memory:
1. Non-warp-0 warps write their contributions to temp shared memory
2. __syncthreads() to ensure all writes complete
3. Warp 0 reads all contributions and sums them
4. Warp 0 writes the final sum to accumulator
5. __syncthreads() before next iteration

This replaces atomics with barriers + serial reduction in warp 0.

WHY THIS WORKS:
- Atomics serialize execution anyway
- By having warp 0 do serial reduction, we get same effect without atomic overhead
- Barriers are cheaper than atomics for small warp counts

SCRATCHPAD STRATEGY:
- Per-block scratchpad (like kernel 5)
- Final reduction by last block (flag-based synchronization)
- All accumulation is atomic-free!

PERFORMANCE vs KERNEL 8:
- PROS:
  * No shared memory atomics (eliminates contention)
  * Cleaner execution pattern
  * Better for high warp counts
- CONS:
  * More barriers (synchronization overhead)
  * Warp 0 does more work (potential imbalance)
  * More shared memory usage for temp buffers

EXPECTED PERFORMANCE:
- Typically 1.1-1.3x faster than kernel 8 for high warp counts
- Best for blocks with many warps (16-32 warps)
*/
__global__ void layernorm_backward_kernel9(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                                            const floatX* dout, const floatX* inp, const floatX* weight,
                                            const floatX* mean, const floatX* rstd,
                                            int B, int T, int C) {
    if(C % (32 * x128::size) != 0) {
        if(threadIdx.x == 0 && blockIdx.x == 0) {
            printf("Number of channels is not a multiple of 32 * x128::size");
        }
        __trap();       // prefer to crash here than run into a deadlock later on
    }
    int BLOCK_SIZE = blockDim.x;
    int warpsInBlock = BLOCK_SIZE / WARP_SIZE; //number of warps in block
    extern __shared__ float shared[]; // size = 2 * C + 1

    int warpId = threadIdx.x / WARP_SIZE; // warp index within a block
    int baseIdx = blockIdx.x * warpsInBlock + warpId;
    int warpThreadIdx = threadIdx.x % WARP_SIZE; // Thread index within the warp
    int warpsInGrid = gridDim.x * warpsInBlock;
    int C_per_iteration = WARP_SIZE * x128::size;
    int iterations_C = ceil_div(C, C_per_iteration) + 2;

    // the first half of shared memory is bias, second is weight
    float* dbias_shared = shared;
    float* dweight_shared = shared + C;
    float* dbias_tmp_shared = shared + 2 * C;
    float* dweight_tmp_shared = shared + 2 * C + BLOCK_SIZE;

    // init shared memory to zero
    for(int i = threadIdx.x; i < C; i+= BLOCK_SIZE){
       dbias_shared[i] = 0.0f;
       dweight_shared[i] = 0.0f;
    }
    unsigned int *tmp_flag = (unsigned int*)(shared + 2*C + 2*BLOCK_SIZE);
    __syncthreads();

    for (int idx = baseIdx; idx < B * T; idx += warpsInGrid) {
        int b = idx / T;
        int t = idx % T;

        const floatX* dout_bt = dout + b * T * C + t * C;
        const floatX* inp_bt = inp + b * T * C + t * C;
        floatX* dinp_bt = dinp + b * T * C + t * C;
        const float mean_bt = (float)mean[b * T + t];
        const float rstd_bt = (float)rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += WARP_SIZE * x128::size) {
            x128 dout128_i   = load128(dout_bt + i);
            x128 inp128_i    = load128(inp_bt  + i);
            x128 weight128_i = load128(weight  + i);
            for (int k = 0; k < x128::size; k++) {
                float norm_bti = ((float)inp128_i[k] - mean_bt) * rstd_bt;
                float dnorm_i = (float)weight128_i[k] * (float)dout128_i[k];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
        }
        dnorm_mean = warpReduceSum(dnorm_mean) / C;
        dnorm_norm_mean = warpReduceSum(dnorm_norm_mean) / C;

        // now iterate again and accumulate all the gradients
        // unfortunately we cannot use the same index for x128 arrays and shared memory
        // as atomics can only be 32-bit rather than 128-bit (at least pre-SM90/Hopper)
        // so this would result in an 8-way bank conflict, and kill performance
        // so instead, we use a shared memory friendly index, and reorder before the final write
        for (int i = 0; i < iterations_C; i++) {
            int global_index = (warpThreadIdx * x128::size) + (i * C_per_iteration);
            int shared_index = warpThreadIdx + (i * C_per_iteration);
            if (global_index >= C) {
                break;
            }

            x128 dout128   = load128cs(dout_bt + global_index);
            x128 inp128    = load128cs(inp_bt  + global_index);
            x128 dinp128   = load128(dinp_bt   + global_index);
            x128 weight128 = load128(weight    + global_index);

            for (int x = 0; x < x128::size; x++) {
                float dout_i = (float)dout128[x];
                float norm_bti = ((float)inp128[x] - mean_bt) * rstd_bt;
                float dnorm_i = (float)weight128[x] * dout_i;

                // sum up the gradients for bias and weight across the entire block
                // this is basically a reduction (but only inter-warp, not intra-warp)
                // doing it this way allows us to avoid using atomics while using many warps
                if (warpId != 0) {
                    dbias_tmp_shared[threadIdx.x] = dout_i;
                    dweight_tmp_shared[threadIdx.x] = norm_bti * dout_i;
                }
                __syncthreads();
                if (warpId == 0) {
                    float dbias_tmp = dout_i;
                    float dweight_tmp = norm_bti * dout_i;
                    for (int j = 1; j < warpsInBlock; j++) {
                        dbias_tmp += dbias_tmp_shared[threadIdx.x + j * WARP_SIZE];
                        dweight_tmp += dweight_tmp_shared[threadIdx.x + j * WARP_SIZE];
                    }
                    // gradient contribution to bias (using shared memory friendly index)
                    dbias_shared[shared_index + x*WARP_SIZE] += dbias_tmp;
                    // gradient contribution to weight (using shared memory friendly index)
                    dweight_shared[shared_index + x*WARP_SIZE] += dweight_tmp;
                }
                __syncthreads();

                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp128[x] = (floatX)((float)dinp128[x] + dval);
            }
            // cache in L2 as this is read by the next kernel, but bypass L1 to minimise thrashing
            store128cg(dinp_bt + global_index, dinp128);
        }
    }
    __syncthreads();
    // Each block writes its partial sum to global memory
    // The last block to finish becomes responsible for summing up all the partial sums
    // This is done by atomically incrementing a flag (cleared to 0 before launching the kernel)
    unsigned int* scratchFlag = (unsigned int*)(scratch);
    // Increment scratch pointer by a full cacheline so that everything remains cacheline aligned
    scratch += 32;
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    for(int i = threadIdx.x; i < C; i+= BLOCK_SIZE) {
        // Write to global memory in the same "shared memory banking friendly" order
        scratch_dbias[i + 2*C*blockIdx.x] = dbias_shared[i];
        scratch_dweight[i + 2*C*blockIdx.x] = dweight_shared[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *tmp_flag = atomicInc(scratchFlag, gridDim.x);
    }
    __syncthreads();
    if (*tmp_flag == gridDim.x-1) {
        // Reduction of the partial sums by the final block
        // todo - there isn't enough parallelism even inside that single SM...
        // ==> so could maybe split into another kernel with YET ANOTHER level of reduction?!
        for(int i = threadIdx.x * f128::size; i < C; i+= BLOCK_SIZE * f128::size) {
            f128 dbias_accum = f128::zeros();
            f128 dweight_accum = f128::zeros();

            for (int read_block_idx = 0; read_block_idx < gridDim.x; read_block_idx++) {
                int offset = i + 2*C*read_block_idx;
                f128 dbias128 = load128(scratch_dbias + offset);
                f128 dweight128 = load128(scratch_dweight + offset);
                for(int k = 0; k < f128::size; k++) {
                    dbias_accum[k] += dbias128[k];
                    dweight_accum[k] += dweight128[k];
                }
            }
            store128(dbias_shared + i, dbias_accum);
            store128(dweight_shared + i, dweight_accum);
        }
        __syncthreads();

        // reorder from atomic/shared memory-friendly index to real global memory index
        // and convert from float/FP32 to floatX/BF16 for the final write
        // this is separate also because it cannot use as many warps as the above (f128 vs x128)
        // todo - if we split this code into another kernel, we could maybe do it at the same time?
        for (int i = warpId; i < iterations_C; i += warpsInBlock) {
            int global_index = (warpThreadIdx * x128::size) + (i * C_per_iteration);
            int shared_index = warpThreadIdx + (i * C_per_iteration);
            if (global_index >= C) {
                break;
            }

            x128 dbias128 = load128(dbias + global_index);
            x128 dweight128 = load128(dweight + global_index);
            for (int x = 0; x < x128::size; x++) {
                float s_db = dbias_shared[shared_index + x*WARP_SIZE];
                float s_dw = dweight_shared[shared_index + x*WARP_SIZE];
                dbias128[x] = (floatX)(s_db + (float)dbias128[x]);
                dweight128[x] = (floatX)(s_dw + (float)dweight128[x]);
            }
            store128(dbias + global_index, dbias128);
            store128(dweight + global_index, dweight128);
        }
    }
}


/*
KERNEL 10: Vectorized Shared Memory with Optimized Layout (Most Optimized)
---------------------------------------------------------------------------
APPROACH:
- Similar to kernel 9, but uses f128 vectors for shared memory access
- Vectorized shared memory eliminates bank conflicts entirely
- Fewer barriers needed due to vector-level synchronization
- Carefully tuned to avoid register spills

VECTORIZED SHARED MEMORY:
- Use f128 (128-bit float vectors) for shared memory operations
- Shared memory accesses are naturally aligned and conflict-free
- Reduces number of shared memory transactions

KEY OPTIMIZATIONS:
1. Vector loads/stores (f128 and x128) throughout
2. Shared memory sized to rounded_C (aligned to vector boundaries)
3. Temp buffers offset to avoid warp 0 from wasting space
4. Careful register management to avoid spills

REGISTER PRESSURE:
This kernel is at the edge of register limits:
- __launch_bounds__(512, 2): max 512 threads, min 2 blocks per SM
- Any additional complexity can cause register spills (huge perf hit!)
- Many "obvious" optimizations actually hurt performance due to spills
- This is a finely-tuned balance

MEMORY LAYOUT:
- rounded_C: C rounded up to vector alignment
- dbias_shared[rounded_C]: aligned for vector access
- dweight_shared[rounded_C]: aligned for vector access
- Temp buffers for inter-warp reduction
- Pointer arithmetic tricks to save registers

BANK CONFLICT ELIMINATION:
By using f128 vectors for shared memory:
- Each access loads 128 bits (4 floats)
- Consecutive threads access consecutive 128-bit chunks
- Perfect bank distribution (no conflicts at all!)

PERFORMANCE vs KERNEL 9:
- PROS:
  * Vectorized shared memory (no bank conflicts)
  * Fewer barriers needed
  * Optimal memory access patterns
  * Best overall performance
- CONS:
  * Most complex code
  * Fragile (register spills if modified)
  * Requires careful tuning
  * Higher shared memory usage

EXPECTED PERFORMANCE:
- Typically 1.1-1.2x faster than kernel 9
- Often achieves 75-85% of peak memory bandwidth
- Best overall backward pass kernel for typical configurations

IMPORTANT NOTES:
- This kernel is extremely sensitive to changes
- Many "improvements" actually hurt due to register spills
- If modifying, check register usage with --ptxas-options=-v
- Register spills can reduce performance by 2-3x!
*/
__global__ void __launch_bounds__(512, 2)
layernorm_backward_kernel10(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                            const floatX* dout, const floatX* inp, const floatX* weight,
                            const floatX* mean, const floatX* rstd,
                            int B, int T, int C) {
    int BLOCK_SIZE = blockDim.x;
    int warpsInBlock = BLOCK_SIZE / WARP_SIZE; //number of warps in block
    extern __shared__ float shared[]; // size = 2 * C + 1

    int warpId = threadIdx.x / WARP_SIZE; // warp index within a block
    int baseIdx = blockIdx.x * warpsInBlock + warpId;
    int warpThreadIdx = threadIdx.x % WARP_SIZE; // Thread index within the warp
    int warpsInGrid = gridDim.x * warpsInBlock;
    int C_per_iteration = WARP_SIZE * x128::size;
    int iterations_C = ceil_div(C, C_per_iteration); // + 2;

    // the first half of shared memory is bias, second is weight
    size_t rounded_C = ceil_div(C, (32 * x128::size)) * (32 * x128::size);
    float* dbias_shared = shared;
    float* dweight_shared = shared + rounded_C;
    // warp zero doesn't actually write to the _tmp_shared memory locations, so we don't need to reserve memory
    // the obvious solution is to change the addressing below to use (threadId.x-32) as offset, but that causes
    // register spills, so instead we mess with the base pointer here, which doesn't increase register usage.
    float* dbias_tmp_shared = shared + 2 * rounded_C - WARP_SIZE * f128::size;
    float* dweight_tmp_shared = shared + 2 * rounded_C + f128::size * BLOCK_SIZE - 2 * WARP_SIZE * f128::size;

    // init shared memory to zero
    for(int i = threadIdx.x * f128::size; i < rounded_C; i += BLOCK_SIZE * f128::size) {
        store128(dbias_shared + i, f128::zeros());
        store128(dweight_shared + i, f128::zeros());
    }
    __syncthreads();

    for (int bt = baseIdx; bt < B * T; bt += warpsInGrid) {
        const floatX* dout_bt = dout + bt * C;
        const floatX* inp_bt = inp +bt * C;
        floatX* dinp_bt = dinp + bt * C;

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += WARP_SIZE * x128::size) {
            x128 dout128_i   = load128(dout_bt + i);
            x128 inp128_i    = load128(inp_bt  + i);
            x128 weight128_i = load128(weight  + i);
            for (int k = 0; k < x128::size; k++) {
                float dnorm_i = (float)weight128_i[k] * (float)dout128_i[k];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * (float)inp128_i[k];
            }
        }

        const float mean_bt = (float)mean[bt];
        const float rstd_bt = (float)rstd[bt];
        dnorm_mean = warpReduceSum(dnorm_mean) / C;
        dnorm_norm_mean = warpReduceSum(dnorm_norm_mean) / C * rstd_bt - dnorm_mean * mean_bt * rstd_bt;

        for (int c = 0; c < iterations_C; c++) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);

            x128 dout128   = x128::zeros();
            x128 inp128    = x128::zeros();
            x128 dinp128   = x128::zeros();
            x128 weight128 = x128::zeros();

            if(global_index < C) {
                dout128 = load128cs(dout_bt + global_index);
                inp128 = load128cs(inp_bt + global_index);
                dinp128 = load128(dinp_bt + global_index);
                weight128 = load128(weight + global_index);
            }

            for(int o = 0; o < x128::size / f128::size; ++o) {
                f128 dbias_f;
                f128 dweight_f;
                for(int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    float dout_i = (float)dout128[x];
                    float norm_bti = ((float)inp128[x] - mean_bt) * rstd_bt;
                    dbias_f[i] = dout_i;
                    dweight_f[i] = norm_bti * dout_i;

                    float dval = 0.0f;
                    dval += (float) weight128[x] * (float)dout128[x]; // term 1
                    dval -= dnorm_mean; // term 2
                    dval -= norm_bti * dnorm_norm_mean; // term 3
                    dval *= rstd_bt; // final scale
                    dinp128[x] = (floatX) ((float) dinp128[x] + dval);
                }

                if (warpId != 0) {
                    store128(dbias_tmp_shared + threadIdx.x * f128::size, dbias_f);
                    // this seems to generate a 64-bit store, instead of 128-bit.
                    // however, forcing 128-bit (e.g., using inline ptx), results in register
                    // spilling and much worse performance, so we'll keep it like this for now
                    // but ideally, we could reduce the register pressure a little.
                    store128(dweight_tmp_shared + threadIdx.x * f128::size, dweight_f);
                }
                __syncthreads();
                if (warpId == 0) {
                    for (int j = 1; j < warpsInBlock; j++) {
                        f128 dbias_tmp = load128(dbias_tmp_shared + f128::size * (threadIdx.x + j * WARP_SIZE));
                        f128 dweight_tmp = load128(dweight_tmp_shared + f128::size * (threadIdx.x + j * WARP_SIZE));
                        for(int i = 0; i < f128::size; ++i) {
                            dbias_f[i] += dbias_tmp[i];
                            dweight_f[i] += dweight_tmp[i];
                        }
                    }
                }
                __syncthreads();
                if (warpId == 0) {
                    f128 db_old = load128(dbias_shared + global_index + f128::size * o);
                    f128 dw_old = load128(dweight_shared + global_index + f128::size * o);
                    for(int i = 0; i < f128::size; ++i) {
                        dbias_f[i] += db_old[i];
                        dweight_f[i] += dw_old[i];
                    }
                    store128(dbias_shared + global_index + f128::size * o, dbias_f);
                    store128(dweight_shared + global_index + f128::size * o, dweight_f);
                }
            }
            if(global_index < C) {
                // cache in L2 as this is read by the next kernel, but bypass L1 to minimise thrashing
                store128cg(dinp_bt + global_index, dinp128);
            }
        }
    }
    __syncthreads();
    // Each block writes its partial sum to global memory
    // The last block to finish becomes responsible for summing up all the partial sums
    // This is done by atomically incrementing a flag (cleared to 0 before launching the kernel)
    unsigned int* scratchFlag = (unsigned int*)(scratch);
    // Increment scratch pointer by a full cacheline so that everything remains cacheline aligned
    scratch += 32;
    float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    for(int i = threadIdx.x * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
        // Write to global memory in the same "shared memory banking friendly" order
        store128(scratch_dbias + i + 2*C*blockIdx.x, load128(dbias_shared + i));
        store128(scratch_dweight + i + 2*C*blockIdx.x, load128(dweight_shared + i));
    }
    __syncthreads();
    // that portion of shared memory is no longer used, so we can repurpose it for the scratch flag.
    unsigned int *tmp_flag = (unsigned int*)(shared + 2*rounded_C);
    if (threadIdx.x == 0) {
        *tmp_flag = atomicInc(scratchFlag, gridDim.x);
    }
    __syncthreads();
    if (*tmp_flag == gridDim.x-1) {
        // Reduction of the partial sums by the final block
        // todo - there isn't enough parallelism even inside that single SM...
        // ==> so could maybe split into another kernel with YET ANOTHER level of reduction?!
        for(int i = threadIdx.x * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
            f128 dbias_accum = f128::zeros();
            f128 dweight_accum = f128::zeros();

            for (int read_block_idx = 0; read_block_idx < gridDim.x; read_block_idx++) {
                int offset = i + 2*C*read_block_idx;
                f128 dbias128 = load128(scratch_dbias + offset);
                f128 dweight128 = load128(scratch_dweight + offset);
                for(int k = 0; k < f128::size; k++) {
                    dbias_accum[k] += dbias128[k];
                    dweight_accum[k] += dweight128[k];
                }
            }
            store128(dbias_shared + i, dbias_accum);
            store128(dweight_shared + i, dweight_accum);
        }
        __syncthreads();

        // convert from float/FP32 to floatX/BF16 for the final write
        // this is separate because it cannot use as many warps as the above (f128 vs x128)
        // todo - if we split this code into another kernel, we could maybe do it at the same time?
        for (int c = warpId; c < iterations_C; c += warpsInBlock) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);
            if (global_index >= C) {
                break;
            }

            x128 dbias128 = load128(dbias + global_index);
            x128 dweight128 = load128(dweight + global_index);
            for(int o = 0; o < x128::size / f128::size; ++o) {
                f128 s_db = load128(dbias_shared + global_index + o * f128::size);
                f128 s_dw = load128(dweight_shared + global_index + o * f128::size);
                for(int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    dbias128[x] = (floatX)(s_db[i] + (float)dbias128[x]);
                    dweight128[x] = (floatX)(s_dw[i] + (float)dweight128[x]);
                }
            }
            store128(dbias + global_index, dbias128);
            store128(dweight + global_index, dweight128);
        }
    }
}


// ----------------------------------------------------------------------------
// kernel launchers

void layernorm_backward1(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const float* weight, const float* mean, const float* rstd,
                        int B, int T, int C, const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    layernorm_backward_kernel1<<<grid_size, block_size>>>(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward2(Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(32*N, block_size);
    size_t shared_mem_size = 2 * C * sizeof(float);
    float* dweight_tmp;
    float* dbias_tmp;
    cudaCheck(cudaMalloc(&dweight_tmp, C * sizeof(float)));
    cudaCheck(cudaMalloc(&dbias_tmp, C * sizeof(float)));
    cudaMemset(dweight_tmp, 0, C * sizeof(float));
    cudaMemset(dbias_tmp, 0, C * sizeof(float));
    layernorm_backward_kernel2<<<grid_size, block_size, shared_mem_size>>>(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, dweight_tmp, dbias_tmp);
    copy_to_dweight_dbias<<<1, 512>>>(C, dweight, dbias, dweight_tmp, dbias_tmp);
    cudaCheck(cudaFree(dweight_tmp));
    cudaCheck(cudaFree(dbias_tmp));
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward3(Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
    const int grid_size = (1024/block_size) * cuda_num_SMs;
    size_t shared_mem_size = 2 * C * sizeof(float);
    layernorm_backward_kernel3<<<grid_size, block_size, shared_mem_size>>>(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward4(Tdinp* dinp, Tparams* dweight, Tparams* dbias,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
        const int grid_size = (1024/block_size) * cuda_num_SMs;
        size_t shared_mem_size = 2 * C * sizeof(float);
        layernorm_backward_kernel4<<<grid_size, block_size, shared_mem_size>>>(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward5(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
        const int grid_size = 1 * cuda_num_SMs; // only support 1 block per SM for simplicity, 1024 threads is best anyway
        size_t shared_mem_size = (2 * C + 1) * sizeof(float);
        cudaMemset(scratch, 0, (grid_size * 2 * C + 1) * sizeof(float));
        layernorm_backward_kernel5<<<grid_size, block_size, shared_mem_size>>>(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C);
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward6(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
        const int grid_size = (1024/block_size) * cuda_num_SMs;
        size_t shared_mem_size = (2 * C + 1) * sizeof(float);

        // Including this as part of the timing until we can parallelise it
        // It should fully hide the cost and improve kernel perf by >5% if done in parallel using CUDA streams
        cudaMemset(scratch, 0, (1 + 2 * C) * sizeof(float));

        layernorm_backward_kernel6<<<grid_size, block_size, shared_mem_size>>>(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C);
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward7(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
        const int grid_size = (1024/block_size) * cuda_num_SMs;
        size_t shared_mem_size = (2 * C + 1) * sizeof(float);

        // Including this as part of the timing until we can parallelise it
        // It should fully hide the cost and improve kernel perf by >5% if done in parallel using CUDA streams
        cudaMemset(scratch, 0, (1 + 2 * C) * sizeof(float));

        layernorm_backward_kernel7<<<grid_size, block_size, shared_mem_size>>>(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C);
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward8(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {
        const int grid_size = (1024/block_size) * cuda_num_SMs;
        size_t shared_mem_size = (2 * C + 1) * sizeof(float);

        // Including this as part of the timing until we can parallelise it
        // It should fully hide the cost and improve kernel perf by >5% if done in parallel using CUDA streams
        cudaMemset(scratch, 0, (1 + 2 * C) * sizeof(float));

        layernorm_backward_kernel8<<<grid_size, block_size, shared_mem_size>>>(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C);
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward9(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                        const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                        int B, int T, int C, int block_size) {

        assert(C % (32 * x128::size) == 0  && "Channels must be divisible by (32 * x128::size)");
        const int grid_size = (1024/block_size) * cuda_num_SMs; // todo - heuristics for other GPUs?
        size_t shared_mem_size = (2 * C + 2 * block_size + 1) * sizeof(float);

        cudaMemset(scratch, 0, 1 * sizeof(float)); // just need to memset the flag for this version
        layernorm_backward_kernel9<<<grid_size, block_size, shared_mem_size>>>(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C);
}

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_backward10(Tdinp* dinp, Tparams* dweight, Tparams* dbias, float* scratch,
                         const Tdout* dout, const Trest* inp, const Tparams* weight, const Trest* mean, const Trest* rstd,
                         int B, int T, int C, int block_size) {
        if(block_size == 1024) {
            block_size = 512;
        }
        //assert(C % (32 * x128::size) == 0  && "Channels must be divisible by (32 * x128::size)");
        const int grid_size = (1024/block_size) * cuda_num_SMs; // todo - heuristics for other GPUs?
        size_t rounded_C = ceil_div(C, (32 * x128::size)) * (32 * x128::size);
        size_t shared_mem_size = (2 * rounded_C + 2 * (block_size - 32) * f128::size) * sizeof(float);

        cudaCheck(cudaMemset(scratch, 0, 1 * sizeof(float))); // just need to memset the flag for this version
        layernorm_backward_kernel10<<<grid_size, block_size, shared_mem_size>>>(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C);
        cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void layernorm_backward(int kernel_num,
                        floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,
                        const floatX* dout, const floatX* inp, const floatX* weight, const floatX* mean, const floatX* rstd,
                        int B, int T, int C,
                        const int block_size) {
    switch (kernel_num) {
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        case 1:
            layernorm_backward1(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
#endif
        case 2:
            layernorm_backward2(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 3:
            layernorm_backward3(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
#if defined(ENABLE_BF16)
        case 4:
            layernorm_backward4(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
#endif
        case 5:
            layernorm_backward5(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 6:
            layernorm_backward6(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 7:
            layernorm_backward7(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 8:
            layernorm_backward8(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 9:
            layernorm_backward9(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
        case 10:
            layernorm_backward10(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, B, T, C, block_size);
            break;
    default:
            printf("Invalid kernel number\n");
            exit(1);
    }
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 1600;   // this is the problematic size

    // first do the forward pass in CPU
    float* out = (float*)malloc(B * T * C * sizeof(float));
    float* mean = (float*)malloc(B * T * sizeof(float));
    float* rstd = (float*)malloc(B * T * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(C);
    float* bias = make_random_float(C);
    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    // now do the backward pass, again on CPU
    float *dout = make_random_float(B * T * C);
    float *dinp = make_zeros_float(B * T * C);
    float *dweight = make_zeros_float(C);
    float *dbias = make_zeros_float(C);
    layernorm_backward_cpu(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);

    // the above calculations act as the reference
    // now let's do the same on the GPU

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // move all the variables we need for backward pass onto the GPU
    floatX* d_dinp;
    floatX* d_dweight;
    floatX* d_dbias;
    floatX* d_dout;
    floatX* d_inp;
    floatX* d_weight;
    floatX* d_mean;
    floatX* d_rstd;
    float* d_scratch;
    cudaCheck(cudaMalloc(&d_dinp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dweight, C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dbias, C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_mean, B * T * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_rstd, B * T * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_scratch, (1024/32) * cuda_num_SMs * (2 * C + 1) * sizeof(float)));
    // copy over the "inputs" to the backward call
    cudaCheck(memcpy_convert(d_dout, dout, B * T * C));
    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));
    cudaCheck(memcpy_convert(d_weight, weight, C));
    cudaCheck(memcpy_convert(d_mean, mean, B * T));
    cudaCheck(memcpy_convert(d_rstd, rstd, B * T));

    // launch the kernel
    // removed 768 because it doesn't work for kernel9 despite being OK in train_gpt2.cu?!
    int block_sizes[] = {32, 64, 128, 256, 512, /*768,*/ 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        // init the "outputs" of the backward call to zeros
        cudaCheck(cudaMemset(d_dinp, 0, B * T * C * sizeof(floatX)));
        cudaCheck(cudaMemset(d_dweight, 0, C * sizeof(floatX)));
        cudaCheck(cudaMemset(d_dbias, 0, C * sizeof(floatX)));

        layernorm_backward(kernel_num, d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp, d_weight, d_mean, d_rstd,
                           B, T, C, block_size);

        // check the correctness of the kernel
        float error_threshold_dinp = sizeof(floatX) == 4 ? 1e-3f : 1e-1f; // allow larger errors for BF16/FP16
        float error_threshold_dparams = sizeof(floatX) == 4 ? 1e-3f : 5e-1f; // much, much larger...
        printf("Checking correctness...\n");
        printf("dinp:\n");
        validate_result(d_dinp, dinp, "dinp", B * T * C, error_threshold_dinp);
        printf("dweight:\n");
        validate_result(d_dweight, dweight, "dweight", C, error_threshold_dparams);
        printf("dbias:\n");
        validate_result(d_dbias, dbias, "dbias", C, error_threshold_dparams);

        printf("All results match for block_size=%d.\n\n", block_size);
    }

    // now time the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, layernorm_backward, kernel_num,
                                              d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp, d_weight, d_mean, d_rstd,
                                              B, T, C, block_size);
        printf("block_size %4d time %.4f ms\n", block_size, elapsed_time);
    }

    // cleanups
    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);
    free(dout);
    free(dinp);
    free(dweight);
    free(dbias);
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_dweight));
    cudaCheck(cudaFree(d_dbias));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_mean));
    cudaCheck(cudaFree(d_rstd));
    cudaCheck(cudaFree(d_scratch));
    return 0;
}
