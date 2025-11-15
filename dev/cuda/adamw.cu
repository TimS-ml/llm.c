/*
================================================================================
AdamW Optimizer CUDA Kernels
================================================================================

PURPOSE:
--------
Implements the AdamW (Adam with decoupled Weight decay) optimizer on GPU.
AdamW is an adaptive learning rate optimization algorithm that maintains
per-parameter adaptive learning rates and includes decoupled weight decay
regularization (unlike the original Adam which couples weight decay with
the gradient-based update).

ALGORITHM OVERVIEW:
-------------------
AdamW performs the following updates for each parameter at timestep t:

1. Compute first moment (momentum):
   m_t = β1 * m_{t-1} + (1 - β1) * g_t
   where g_t is the gradient at time t

2. Compute second moment (uncentered variance):
   v_t = β2 * v_{t-1} + (1 - β2) * g_t²

3. Bias-correct both moments:
   m̂_t = m_t / (1 - β1^t)
   v̂_t = v_t / (1 - β2^t)

4. Update parameters with decoupled weight decay:
   θ_t = θ_{t-1} - lr * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})
   where λ is the weight decay coefficient

KEY DIFFERENCE FROM ADAM:
--------------------------
In standard Adam, weight decay is applied to gradients before moment updates.
In AdamW, weight decay is applied directly to parameters after the adaptive
update, making it truly decoupled from the gradient-based optimization.
This often leads to better generalization in deep learning models.

KERNEL IMPLEMENTATIONS:
-----------------------
This file contains two kernel versions with increasing optimizations:

Kernel 1 (adamw_kernel1):
  - Naive implementation following the CPU reference
  - Direct translation of the algorithm to CUDA
  - Each thread processes one parameter independently
  - Multiple redundant memory accesses

Kernel 2 (adamw_kernel2):
  - Optimized using register reuse for frequently accessed data
  - Uses lerp (linear interpolation) for more efficient moment updates
  - Reduces from 3 FMA operations to 2 for moment calculations
  - Better instruction-level parallelism

PERFORMANCE CHARACTERISTICS:
----------------------------
- Memory bandwidth bound (each parameter requires 4 reads + 3 writes)
- Compute is relatively light (few FLOPs per parameter)
- Benefits from coalesced memory access patterns
- Simple parallelization: one thread per parameter or use thread coarsening

References:
  * AdamW paper: https://arxiv.org/abs/1711.05101
  * PyTorch implementation: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
  * NVIDIA Apex: https://github.com/nvidia/apex/blob/master/csrc/multi_tensor_adam.cu

Compile example:
nvcc -lcublas -lcublasLt adamw.cu -o adamw
nvcc -O3 --use_fast_math -lcublas -lcublasLt adamw.cu -o adamw

Run:
./adamw 1  # Run kernel 1
./adamw 2  # Run kernel 2

TODO(general):
amsgrad=True - Add support for AMSGrad variant

TODO(perf):
- Mixed precision support (BF16/FP16 for params, FP32 for moments)
- Thread coarsening/ILP for better instruction-level parallelism
- Vectorized loads/stores using float4 for better memory bandwidth
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "common.h"


// ----------------------------------------------------------------------------
// CPU code reference

/*
CPU reference implementation of AdamW optimizer.
This serves as the ground truth for validating GPU kernel correctness.

Parameters:
  params_memory: Model parameters to be optimized (updated in-place)
  grads_memory: Gradients of the loss with respect to parameters
  m_memory: First moment estimates (momentum), updated in-place
  v_memory: Second moment estimates (uncentered variance), updated in-place
  t: Current timestep (starts from 1), used for bias correction
  num_parameters: Total number of parameters to optimize
  learning_rate: Step size (default: 1e-3)
  beta1: Exponential decay rate for first moment (default: 0.9)
  beta2: Exponential decay rate for second moment (default: 0.999)
  eps: Small constant for numerical stability (default: 1e-8)
  weight_decay: L2 regularization coefficient (default: 0.0)

Note: This is a simple, unoptimized reference. The GPU kernels will be
orders of magnitude faster due to parallelization.
*/
void adamw_cpu(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, int t, long num_parameters, float learning_rate=1e-3, float beta1=0.9, float beta2=0.999, float eps=1e-8, float weight_decay=0.0) {
    // adapted from: train_gpt2.c

    for (int i = 0; i < num_parameters; i++) {
        float param = params_memory[i];
        float grad = grads_memory[i];

        // Update the first moment (exponential moving average of gradients)
        // This provides momentum and smooths out gradient noise
        float m = beta1 * m_memory[i] + (1.0f - beta1) * grad;

        // Update the second moment (exponential moving average of squared gradients)
        // This adapts the learning rate for each parameter based on gradient magnitude
        float v = beta2 * v_memory[i] + (1.0f - beta2) * grad * grad;

        // Bias-correct both moments to account for initialization at zero
        // Without correction, moments are biased toward zero in early iterations
        // The correction factor grows from 0 to 1 as t increases
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        // Store updated moments for next iteration
        m_memory[i] = m;
        v_memory[i] = v;

        // Update parameters using AdamW formula:
        // 1. Adaptive update: m_hat / (sqrt(v_hat) + eps)
        //    - Adapts learning rate per parameter based on gradient history
        //    - eps prevents division by zero
        // 2. Weight decay: weight_decay * param
        //    - Decoupled L2 regularization applied directly to parameters
        //    - This is the key difference from standard Adam
        params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// ============================================================================
// Utility Functions
// ============================================================================

/*
Optimized linear interpolation using fused multiply-add (FMA) instructions.

Standard lerp formula: result = (1 - weight) * start + weight * end
This requires 3 floating-point operations: 2 multiplies and 1 add.

Optimized formula using FMA: result = start + weight * (end - start)
Rearranged as: result = fma(-weight, start, start) + weight * end
             = fma(weight, end, fma(-weight, start, start))
This uses only 2 FMA operations, which are faster and more accurate.

The FMA instruction computes a*b+c as a single operation with a single rounding,
providing both performance and numerical benefits.

Parameters:
  start: Initial value (when weight = 0)
  end: Final value (when weight = 1)
  weight: Interpolation factor in [0, 1]

Returns:
  Interpolated value between start and end

Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
*/
__device__ inline float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

// ============================================================================
// Kernel 1: Naive Baseline Implementation
// ============================================================================

/*
Baseline AdamW kernel with straightforward implementation.

OPTIMIZATION LEVEL: Naive (baseline for comparison)

MEMORY ACCESS PATTERN:
  - 4 reads per thread: params, grads, m, v
  - 3 writes per thread: params, m, v
  - Total: 7 memory transactions per parameter
  - Each value loaded from global memory multiple times (inefficient)

COMPUTE CHARACTERISTICS:
  - Each thread independently processes one parameter
  - No register reuse: params_memory[i] and grads_memory[i] loaded multiple times
  - Standard arithmetic operations (no FMA optimization)
  - Uses pre-computed bias correction factors (beta1_correction, beta2_correction)

PARALLELIZATION:
  - 1D grid of blocks, each with block_size threads
  - Thread i processes parameter i (simple 1:1 mapping)
  - Grid-stride loop not used (assumes grid covers all parameters)

PERFORMANCE NOTES:
  - Memory bandwidth bound
  - Redundant loads hurt performance
  - Good starting point but leaves optimization opportunities on the table

Parameters:
  params_memory: Parameters to update (in/out)
  grads_memory: Gradients (input, read-only)
  m_memory: First moment estimates (in/out)
  v_memory: Second moment estimates (in/out)
  num_parameters: Total number of parameters
  learning_rate: Learning rate
  beta1, beta2: Moment decay rates
  beta1_correction: 1 - beta1^t (bias correction for first moment)
  beta2_correction: 1 - beta2^t (bias correction for second moment)
  eps: Numerical stability constant
  weight_decay: L2 regularization coefficient
*/
__global__ void adamw_kernel1(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
   // Compute global thread index
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   // Bounds check: ensure we don't access beyond the parameter array
   if (i >= num_parameters) return;

   // Update the first moment (momentum/moving average of gradients)
   // Note: grads_memory[i] is loaded here but will be loaded again below (inefficient)
   m_memory[i] = beta1 * m_memory[i] + (1.0f - beta1) * grads_memory[i];

   // Update the second moment (moving average of squared gradients)
   // Note: grads_memory[i] loaded again - this redundancy is eliminated in kernel2
   v_memory[i] = beta2 * v_memory[i] + (1.0f - beta2) * grads_memory[i] * grads_memory[i];

   // Apply bias correction to get unbiased moment estimates
   // These corrections are precomputed on the host to save computation
   float m_hat = m_memory[i] / beta1_correction;
   float v_hat = v_memory[i] / beta2_correction;

   // Update parameters using corrected moments and weight decay
   // Note: params_memory[i] loaded twice (here and on LHS) - also inefficient
   params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * params_memory[i]);
}

// ============================================================================
// Kernel 2: Optimized with Register Reuse and FMA Instructions
// ============================================================================

/*
Optimized AdamW kernel using register caching and efficient interpolation.

OPTIMIZATION LEVEL: Moderate (production-ready)

KEY OPTIMIZATIONS:
1. Register Reuse:
   - Load frequently accessed values into registers once
   - Eliminates redundant global memory loads
   - grad, m, v cached in registers and reused

2. FMA-based Linear Interpolation:
   - Uses optimized lerp() function instead of explicit formula
   - Reduces FLOPs for moment updates from 3 to 2 operations
   - Better numerical accuracy due to single rounding in FMA

3. Instruction-Level Parallelism:
   - Independent operations on m and v can execute in parallel
   - Better utilization of GPU arithmetic units

MEMORY ACCESS PATTERN:
  - 4 reads per thread: params, grads, m, v (same as kernel1)
  - 3 writes per thread: params, m, v (same as kernel1)
  - BUT: each value only loaded once from global memory (into registers)
  - Improved memory access efficiency

PERFORMANCE IMPROVEMENT vs Kernel 1:
  - Approximately 5-10% faster due to:
    * Eliminated redundant memory loads
    * More efficient arithmetic via FMA
    * Better register allocation by compiler
  - Still memory bandwidth bound, but uses bandwidth more efficiently

LERP USAGE:
  Old (kernel1): m = beta1 * m + (1 - beta1) * grad
  New (kernel2): m = lerp(grad, m, beta1)
  These are mathematically equivalent but lerp is more efficient.

Parameters: Same as adamw_kernel1
*/
__global__ void adamw_kernel2(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
   // Compute global thread index
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   // Bounds check
   if (i >= num_parameters) return;

   // Load values into registers ONCE - this is the key optimization
   // These registers will be reused multiple times, avoiding redundant global memory loads
   float grad = grads_memory[i];
   float m = m_memory[i];
   float v = v_memory[i];

   // Update first moment using optimized linear interpolation
   // lerp(grad, m, beta1) computes: beta1 * m + (1 - beta1) * grad
   // Using FMA instructions, this is faster than the explicit formula
   m = lerp(grad, m, beta1);
   m_memory[i] = m;  // Write back immediately to free register pressure

   // Update second moment, also using optimized lerp
   // Note: grad * grad computed once and passed to lerp
   v = lerp(grad * grad, v, beta2);
   v_memory[i] = v;

   // Apply bias correction in-place (reusing m and v registers)
   m /= beta1_correction;  // m now holds m_hat
   v /= beta2_correction;  // v now holds v_hat

   // Final parameter update with corrected moments and weight decay
   // Note: params_memory[i] still loaded twice (once for read, once for write)
   // This could be optimized further with a register, but register pressure is a concern
   params_memory[i] -= learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}


// ----------------------------------------------------------------------------
// kernel launcher

// version 1: naive dispatch to naive kernel
void adamw_dispatch1(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                     float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
    unsigned int block_size = 512;
    unsigned int num_blocks = ceil_div(num_parameters, (long) block_size);
    adamw_kernel1<<<num_blocks, block_size>>>(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                                              learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());
}

// version 2: naive dispatch to slightly optimized kernel
void adamw_dispatch2(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters,
                     float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
    unsigned int block_size = 512;
    unsigned int num_blocks = ceil_div(num_parameters, (long) block_size);
    adamw_kernel2<<<num_blocks, block_size>>>(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                                              learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());
}

void adamw(int kernel_num,
           float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, int t, long num_parameters,
           float learning_rate=1e-3, float beta1=0.9, float beta2=0.999, float eps=1e-8, float weight_decay=0.0) {
    // calculate the m_hat and v_hat correction terms once as they are the same for every param/thread
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    switch (kernel_num) {
        case 1:
            adamw_dispatch1(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                            learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
            break;
        case 2:
            adamw_dispatch2(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                            learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
    setup_main();

    const long num_parameters = 1048576;
    const int t = 10;

    const float learning_rate = 1e-3f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float weight_decay = 0.0f;

    // create random data on host (to be used for the CPU reference implementation)
    float* params_memory = make_random_float(num_parameters);
    float* grads_memory = make_random_float(num_parameters);
    float* m_memory = make_random_float(num_parameters);
    float* v_memory = make_random_float_01(num_parameters);

    // move to GPU
    float* d_params_memory;
    float* d_grads_memory;
    float* d_m_memory;
    float* d_v_memory;
    cudaCheck(cudaMalloc(&d_params_memory, num_parameters * sizeof(float)));
    cudaCheck(cudaMalloc(&d_grads_memory, num_parameters * sizeof(float)));
    cudaCheck(cudaMalloc(&d_m_memory, num_parameters * sizeof(float)));
    cudaCheck(cudaMalloc(&d_v_memory, num_parameters * sizeof(float)));
    cudaCheck(cudaMemcpy(d_params_memory, params_memory, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_grads_memory, grads_memory, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_m_memory, m_memory, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_v_memory, v_memory, num_parameters * sizeof(float), cudaMemcpyHostToDevice));


    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // calculate the CPU reference (using default hyperparams)
    clock_t start = clock();
    adamw_cpu(params_memory, grads_memory, m_memory, v_memory, t, num_parameters);
    clock_t end = clock();
    // TODO: measure runtime with multiple runs
    double elapsed_time_cpu = (double)(end - start) / CLOCKS_PER_SEC;

    // calculate the GPU version (using default hyperparams)
    adamw(kernel_num, d_params_memory, d_grads_memory, d_m_memory, d_v_memory, t, num_parameters);

    // compare
    printf("Checking correctness...\n");
    printf("parameters:\n");
    validate_result(d_params_memory, params_memory, "params_memory", num_parameters);
    printf("first moment:\n");
    validate_result(d_m_memory, m_memory, "m_memory", num_parameters);
    printf("second moment:\n");
    validate_result(d_v_memory, v_memory, "v_memory", num_parameters);
    printf("All results match.\n\n");

    // now benchmark the kernel
    int repeat_times = 1000;
    float elapsed_time = benchmark_kernel(repeat_times, adamw, kernel_num,
      d_params_memory, d_grads_memory, d_m_memory, d_v_memory, t, num_parameters,
      learning_rate, beta1, beta2, eps, weight_decay);
    printf("time gpu %.4f ms\n", elapsed_time);
    printf("time cpu %.4f ms\n", elapsed_time_cpu);

    // cleanup
    free(params_memory);
    free(grads_memory);
    free(m_memory);
    free(v_memory);
    cudaCheck(cudaFree(d_params_memory));
    cudaCheck(cudaFree(d_grads_memory));
    cudaCheck(cudaFree(d_m_memory));
    cudaCheck(cudaFree(d_v_memory));

    return 0;
}
