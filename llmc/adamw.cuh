/*
==============================================================================
AdamW Optimizer
==============================================================================

PURPOSE:
Implements the AdamW (Adam with Weight Decay) optimization algorithm, the
standard optimizer for training large language models. AdamW combines adaptive
learning rates with proper weight decay (L2 regularization).

MATHEMATICAL ALGORITHM:

AdamW maintains two momentum buffers per parameter and updates as follows:

Given:
- θ: parameters (weights, biases)
- g: gradients
- m: first moment estimate (momentum)
- v: second moment estimate (RMSprop)
- α: learning rate
- β₁, β₂: exponential decay rates (typically 0.9, 0.999)
- ε: numerical stability constant (typically 1e-8)
- λ: weight decay coefficient (typically 0.01-0.1)
- t: timestep

Algorithm:
1. Gradient scaling:
   g_t = g_t * grad_scale  [for mixed precision training]

2. Update biased first moment (momentum):
   m_t = β₁ * m_{t-1} + (1 - β₁) * g_t

3. Update biased second moment (RMSprop):
   v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²

4. Bias correction:
   m̂_t = m_t / (1 - β₁^t)
   v̂_t = v_t / (1 - β₂^t)

5. Parameter update with weight decay:
   θ_t = θ_{t-1} - α * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})

KEY INSIGHT - AdamW vs Adam:
Traditional Adam applies weight decay to gradients (before moment updates).
AdamW applies weight decay directly to parameters (after moment updates).
This decoupling is crucial for:
- Better generalization in large models
- Independent tuning of learning rate and weight decay
- Correct L2 regularization behavior with adaptive learning rates

IMPLEMENTATION DETAILS:

1. LERP (Linear Interpolation):
   Uses fused multiply-add (FMA) for numerical accuracy:
   lerp(start, end, weight) = fma(weight, end, fma(-weight, start, start))

   This computes: start + weight * (end - start)
   Equivalent to: (1 - weight) * start + weight * end

   Used for momentum updates:
   m = lerp(grad, m, β₁) = β₁ * m + (1 - β₁) * grad

2. Master Weights (Mixed Precision):
   For BF16/FP8 training, maintains two copies of parameters:
   - params_memory: Low precision (BF16/FP8) for forward/backward passes
   - master_params_memory: Full precision (FP32) for accurate updates

   Update process:
   a. Read FP32 master weights
   b. Compute update in FP32
   c. Write back to FP32 master weights
   d. Stochastic round to BF16/FP8 for next forward pass

   This prevents gradient underflow in low precision formats.

3. Stochastic Rounding:
   When converting FP32 → BF16/FP8:
   - Deterministic rounding loses small updates (< 2^-8 for BF16)
   - Stochastic rounding adds random noise based on fractional part
   - Preserves small updates statistically over many steps
   - Critical for convergence in low precision training

   Uses deterministic seed per parameter for reproducibility.

4. Gradient Scaling:
   For numerical stability in mixed precision:
   - Loss is scaled up before backward pass (prevents underflow)
   - Gradients are scaled down here before optimizer step
   - Typical scale: 2^8 to 2^16

CUDA KERNEL IMPLEMENTATION (adamw_kernel3):

Grid/Block organization:
- Block dimension: (blockIdx.x, blockIdx.y)
  * blockIdx.x: Parameter index (up to num_parameters)
  * blockIdx.y: Layer/slice index (for multi-layer updates)
- Thread layout: 512 threads per block
- Each thread handles one parameter

Memory strides:
- w_stride: Offset between consecutive weight slices
- g_stride: Offset between consecutive gradient slices
- s_stride: Offset between consecutive state slices (m, v, master)

This allows batching updates for multiple layers in one kernel launch.

Algorithm flow per thread:
1. Load gradient g[idx], first moment m[idx], second moment v[idx]
2. Update m using LERP: m = β₁*m + (1-β₁)*g
3. Update v using LERP: v = β₂*v + (1-β₂)*g²
4. Apply bias correction: m̂ = m/(1-β₁^t), v̂ = v/(1-β₂^t)
5. Load old parameter (from master weights if available)
6. Compute update: param_new = param_old - α*(m̂/√(v̂+ε) + λ*param_old)
7. Stochastic round param_new to BF16/FP8 → params_memory
8. Store FP32 param_new to master_params_memory (if exists)

INITIALIZATION (init_from_master_kernel):

Converts FP32 master weights → BF16/FP8 parameters with stochastic rounding.
Used when:
- Loading pretrained FP32 checkpoint for BF16/FP8 training
- Initializing from master weights after optimizer state load

MEMORY ACCESS PATTERNS:
- Read: gradient, m, v, old_param (4 reads per parameter)
- Write: m, v, param, master_param (2-4 writes per parameter)
- Coalesced access within warps (consecutive threads access consecutive params)
- Good cache locality for small models, memory-bound for large models

OPTIMIZATIONS:

1. Fused operations:
   All updates in single kernel (no intermediate storage)

2. Fast math:
   - FMA for LERP (single instruction)
   - Fast rsqrt for 1/√v

3. Conditional master weights:
   - Check if master_params_memory != NULL
   - Avoids overhead for FP32 training

4. Deterministic stochastic rounding:
   - Unique seed per parameter
   - Reproducible training runs

5. Multi-layer batching:
   - Update multiple layers in one kernel launch
   - Reduces kernel launch overhead

PERFORMANCE CHARACTERISTICS:
- Memory-bandwidth bound (typical GPU utilization: 60-80%)
- Throughput: ~100-500 GB/s on modern GPUs (A100, H100)
- Much faster than parameters update (happens once per gradient accumulation)
- Typically <5% of total training time

CONVERGENCE PROPERTIES:
- AdamW converges faster than SGD on most tasks
- More stable than Adam (better weight decay)
- Robust to hyperparameter choices
- Standard hyperparameters work across model sizes:
  * β₁ = 0.9 (short-term momentum)
  * β₂ = 0.999 (long-term variance)
  * ε = 1e-8 (numerical stability)
  * λ = 0.01-0.1 (weight decay strength)

ALTERNATIVES:
1. SGD with momentum: Simpler but slower convergence
2. Adam: Original without proper weight decay
3. Adafactor: Memory-efficient (factorized second moment)
4. LAMB: Large batch training extension
5. Lion: Recent simpler alternative (sign-based updates)

REFERENCES:
- Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)
  https://arxiv.org/abs/1412.6980
- Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2017)
  https://arxiv.org/abs/1711.05101
- Mixed Precision Training (Micikevicius et al., 2017)
  https://arxiv.org/abs/1710.03740
*/

// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
__device__ float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

template <typename Tp, typename Tg>
__device__ void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                             float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                             float grad_scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) { return; }  // guard

    // get the gradient, m, and v for this parameter
    float grad = grad_scale * (float)grads_memory[idx];
    float m = m_memory[idx];
    float v = v_memory[idx];
    // update the first moment (momentum)
    m = lerp(grad, m, beta1);
    m_memory[idx] = m;
    // update the second moment (RMSprop)
    v = lerp(grad * grad, v, beta2);
    v_memory[idx] = v;
    m /= beta1_correction;  // m_hat
    v /= beta2_correction;  // v_hat
    // fetch the old value of this parameter as a float, from either source
    float old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params_memory[idx];
    // update this parameter
    float param = old_param - (learning_rate * (m / (sqrtf(v) + eps) + weight_decay * old_param));
    // update our low precision version of the parameters using stochastic rounding
    // this will be used in the next forward pass
    stochastic_rounding(param, &params_memory[idx], seed);
    // write the full, float version of the param into our master copy, if we maintain one
    // this will be used in the next update
    if (master_params_memory != NULL) { master_params_memory[idx] = param; }
}

template <typename Tp, typename Tg>
__global__ void adamw_kernel3(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                              ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                              float grad_scale, unsigned int seed) {
    adamw_update(params_memory + blockIdx.y * w_stride,
                 master_params_memory ? master_params_memory + blockIdx.y * s_stride : NULL,
                 grads_memory + blockIdx.y * g_stride,
                 m_memory + blockIdx.y * s_stride,
                 v_memory + blockIdx.y * s_stride,
                 num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, grad_scale,
                 seed
                 );
}

template <typename Tp>
__global__ void init_from_master_kernel(Tp* params_memory, float* master_params_memory, size_t num_parameters,
                                          ptrdiff_t w_stride, ptrdiff_t s_stride, unsigned int seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) { return; }
    params_memory += blockIdx.y * w_stride; // adjust for layer offset
    master_params_memory += blockIdx.y * s_stride;
    stochastic_rounding(master_params_memory[idx], &params_memory[idx], seed);
}

template <typename Tp, typename Tg>
void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                  ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,  int num_slices, float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  float grad_scale, unsigned int seed, cudaStream_t stream) {
    // AdamW update
    int block_size = 512;
    int num_blocks = CEIL_DIV(num_parameters, block_size);
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    adamw_kernel3<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>(params_memory, master_params_memory, grads_memory,
                                                         m_memory, v_memory, num_parameters, w_stride, g_stride, s_stride,
                                                         learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay,
                                                         grad_scale, seed);
    cudaCheck(cudaGetLastError());
}

template <typename Tp>
void init_from_master(Tp* params_memory, float* master_params_memory, size_t num_parameters,
                        ptrdiff_t w_stride, ptrdiff_t s_stride, int num_slices, unsigned int seed, cudaStream_t stream) {
    int block_size = 512; // must match block size of adamw_update so that RNG also matches
    int num_blocks = CEIL_DIV(num_parameters, block_size);
    init_from_master_kernel<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>
                             (params_memory, master_params_memory, num_parameters, w_stride, s_stride, seed);
    cudaCheck(cudaGetLastError());
}
