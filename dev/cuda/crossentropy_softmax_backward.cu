/*
Kernels for fused cross-entropy + softmax backward pass.

OPERATION OVERVIEW:
This computes the gradient of the combined softmax + cross-entropy loss in a single
fused operation. This is more efficient than backpropagating through them separately.

Forward pass computed:
  probs = softmax(logits)
  loss = -log(probs[target])

Backward pass computes:
  dlogits[i] = probs[i] - indicator(i == target)

Where indicator is 1 if i is the target token, 0 otherwise.

MATHEMATICAL DERIVATION:
The derivative of cross-entropy(softmax(x)) simplifies beautifully:
- ∂loss/∂logits[i] = probs[i] if i ≠ target
- ∂loss/∂logits[target] = probs[target] - 1

This is a remarkably simple result! The gradient is just the predicted probability
minus the true probability (which is 0 everywhere except 1 at the target).

Example:
  If target is token 5, and probs = [0.1, 0.3, 0.05, ..., 0.2, ...]
  Then dlogits = [0.1, 0.3, 0.05, ..., 0.2-1=-0.8, ...]

ROLE IN TRANSFORMER:
This is the first step of backpropagation for language model training:
1. Forward: Model predicts probabilities for next token
2. Loss: Cross-entropy measures prediction error
3. Backward (this kernel): Compute how to adjust logits to reduce error
4. Remaining backward: Propagate gradients back through entire network

The gradients tell the model: "You predicted token X with probability p, but the
correct token was Y. Adjust your weights to predict Y more and X less."

WHY FUSED IMPLEMENTATION:
Computing softmax backward and cross-entropy backward separately would require:
1. Storing full Jacobian of softmax (V×V matrix) - huge memory
2. More kernel launches and memory traffic

Fusing them exploits the mathematical simplification:
- No need to store/compute full Jacobian
- Single simple formula: probs[i] - indicator[i]
- One kernel launch instead of two

WHY SINGLE KERNEL VERSION:
Like forward cross-entropy, this is embarrassingly parallel:
- Each element dlogits[b,t,v] computed independently
- No reductions needed
- Simple arithmetic: one subtraction
- Already near-optimal

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt crossentropy_softmax_backward.cu -o crossentropy_softmax_backward

version 1 is a straight-forward port from CPU code to kernel, parallel over B,T
./crossentropy_softmax_backward 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void crossentropy_softmax_backward_cpu(float* dlogits,
                           const float* dlosses, const float* probs, const int* targets,
                           int B, int T, int V) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * V + t * V;
            const float* probs_bt = probs + b * T * V + t * V;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

// KERNEL 1: Fused softmax + cross-entropy backward pass
//
// ALGORITHM:
// - Each thread handles one element dlogits[b,t,v]
// - Computes: dlogits[v] = (probs[v] - indicator[v]) * dloss
// - Where indicator[v] = 1 if v is the target token, 0 otherwise
//
// MATHEMATICAL INSIGHT:
// The gradient of cross_entropy(softmax(x)) simplifies to probs - one_hot(target).
// This is one of the most elegant results in deep learning:
//
// Without fusion:
//   Need to compute ∂CE/∂probs and ∂softmax/∂logits separately
//   Softmax Jacobian is V×V (huge for large vocabularies!)
//
// With fusion:
//   Direct formula: dlogits = probs - one_hot(target)
//   No need to store or compute Jacobian
//   This is why we always fuse these operations in practice
//
// PARALLELIZATION:
// - Launch B*T*V threads (one per output gradient element)
// - Fully parallel - each element computed independently
// - High parallelism (V is typically 10K-100K tokens)
//
// MEMORY ACCESS PATTERN:
// - Read probs[b,t,v]: Sequential within each row
// - Read targets[b,t]: Broadcast across V (cached)
// - Read dlosses[b,t]: Broadcast across V (cached)
// - Write dlogits[b,t,v]: Sequential within each row
// - Coalesced memory access (consecutive threads access consecutive memory)
//
// PERFORMANCE CHARACTERISTICS:
// - GPU utilization: Excellent (B*T*V threads, typically millions)
// - Memory bandwidth: Good (coalesced access pattern)
// - Compute: Minimal (subtraction and multiplication)
// - Best for: All cases (already near-optimal)
//
// GRADIENT ACCUMULATION:
// Uses += instead of = to support gradient accumulation
// This allows the same logits to contribute to multiple losses if needed
// (though uncommon in standard language modeling)
//
// NUMERICAL EXAMPLE:
// Say target=5, dloss=1.0, probs=[0.1, 0.2, 0.05, 0.15, 0.3, 0.2]
// Then dlogits=[0.1, 0.2, 0.05, 0.15, 0.3, -0.8] * 1.0
//             =[0.1, 0.2, 0.05, 0.15, 0.3, -0.8]
// Note: dlogits[5] = 0.2 - 1.0 = -0.8 (reduce this logit)
//       dlogits[4] = 0.3 - 0.0 =  0.3 (increase this logit - model was too confident)
//
__global__ void crossentropy_softmax_backward_kernel1(float* dlogits,
                           const float* dlosses, const float* probs, const int* targets,
                           int B, int T, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T * V) {
        int b = i / (T * V);
        int t = (i / V) % T;
        int v = i % V;
        float* dlogits_bt = dlogits + b * T * V + t * V;
        const float* probs_bt = probs + b * T * V + t * V;
        float dloss = dlosses[b * T + t];
        int ix = targets[b * T + t];
        float p = probs_bt[v];
        float indicator = v == ix ? 1.0f : 0.0f;
        dlogits_bt[v] += (p - indicator) * dloss;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void crossentropy_softmax_backward1(float* dlogits,
                           const float* dlosses, const float* probs, const int* targets,
                           int B, int T, int V,
                           const int block_size) {
    const int N = B * T * V;
    const int grid_size = ceil_div(N, block_size);
    crossentropy_softmax_backward_kernel1<<<grid_size, block_size>>>(dlogits, dlosses, probs, targets, B, T, V);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void crossentropy_softmax_backward(int kernel_num,
                           float* dlogits,
                           const float* dlosses, const float* probs, const int* targets,
                           int B, int T, int V,
                           const int block_size) {
    switch (kernel_num) {
        case 1:
            crossentropy_softmax_backward1(dlogits, dlosses, probs, targets, B, T, V, block_size);
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
    int V = 50257;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float* probs = make_random_float_01(B * T * V);
    int* targets = make_random_int(B * T, V);
    float* dlosses = make_random_float(B * T);
    float* dlogits = make_zeros_float(B * T * V);

    // move to GPU
    float* d_probs;
    int* d_targets;
    float* d_dlosses;
    float* d_dlogits;
    cudaCheck(cudaMalloc(&d_probs, B * T * V * sizeof(float)));
    cudaCheck(cudaMalloc(&d_targets, B * T * sizeof(int)));
    cudaCheck(cudaMalloc(&d_dlosses, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dlogits, B * T * V * sizeof(float)));
    cudaCheck(cudaMemcpy(d_probs, probs, B * T * V * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dlosses, dlosses, B * T * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    crossentropy_softmax_backward_cpu(dlogits, dlosses, probs, targets, B, T, V);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        cudaCheck(cudaMemset(d_dlogits, 0, B * T * V * sizeof(float)));
        printf("Checking block size %d.\n", block_size);
        crossentropy_softmax_backward(kernel_num, d_dlogits, d_dlosses, d_probs, d_targets, B, T, V, block_size);
        validate_result(d_dlogits, dlogits, "dlogits", B * T * V, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, crossentropy_softmax_backward,
                                              kernel_num, d_dlogits, d_dlosses, d_probs, d_targets,
                                              B, T, V, block_size);

        printf("block_size %4d | time %.4f ms | per token %.2f µs\n", block_size, elapsed_time, elapsed_time * 1'000 / (B*T));
    }

    // free memory
    free(probs);
    free(targets);
    free(dlosses);
    free(dlogits);
    cudaCheck(cudaFree(d_probs));
    cudaCheck(cudaFree(d_targets));
    cudaCheck(cudaFree(d_dlosses));
    cudaCheck(cudaFree(d_dlogits));

    return 0;
}