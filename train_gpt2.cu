/*
==============================================================================
GPT-2 Transformer Neural Network Training - CUDA GPU-Accelerated Version
==============================================================================

This is the production-grade, GPU-accelerated implementation of GPT-2 training
using NVIDIA CUDA. Unlike the CPU reference implementation (train_gpt2.c), this
version is heavily optimized for high-performance training on modern GPUs.

KEY DIFFERENCES FROM CPU VERSION:
- Uses CUDA kernels for parallelized computation on thousands of GPU cores
- Leverages cuBLAS/cuBLASLt for optimized matrix multiplications
- Supports mixed precision training (FP32, BF16, FP16) for faster computation
- Implements multi-GPU training via NCCL for data parallelism
- Uses GPU memory (VRAM) instead of system RAM for activations and parameters
- Supports advanced optimizations: kernel fusion, gradient checkpointing, etc.

CUDA PROGRAMMING CONCEPTS USED:
- Device Memory: GPU VRAM allocated with cudaMalloc (vs. CPU RAM with malloc)
- Host Memory: CPU RAM, with cudaMemcpy for data transfer between CPU/GPU
- Kernels: GPU functions that run in parallel on thousands of threads
- Streams: CUDA command queues for asynchronous GPU operations
- cuBLAS: NVIDIA's optimized linear algebra library for GPUs

ARCHITECTURE OVERVIEW:
1. Model Parameters: Stored in GPU memory, updated via AdamW optimizer
2. Forward Pass: Computes predictions through transformer layers on GPU
3. Backward Pass: Computes gradients via backpropagation on GPU
4. Optimizer: Updates weights using AdamW with optional master weights (FP32)
5. Multi-GPU: Distributes work across GPUs using Data Parallel or ZeRO

PERFORMANCE FEATURES:
- Mixed precision: BF16 computation with FP32 master weights for stability
- Gradient checkpointing: Recomputes activations during backward to save memory
- GELU fusion: Fuses GELU activation into matrix multiply for efficiency
- Kernel fusion: Combines operations to reduce memory bandwidth
- Multi-GPU support: Data parallelism and ZeRO optimizer sharding

See README.md for detailed usage instructions.
*/
// Standard C/C++ library includes
#include <unistd.h>      // POSIX operating system API
#include <stdio.h>       // Standard I/O operations
#include <stdlib.h>      // Memory allocation, process control
#include <stdarg.h>      // Variable argument lists
#include <string>        // C++ string class
#include <string_view>   // Non-owning string references
#include <sys/stat.h>    // File status and permissions
#include <sys/types.h>   // System data types

// ----------- CPU utilities -----------
// Helper functions for file I/O, directory management, and error checking
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
// defines: create_dir_if_not_exists, find_max_step, ends_with_bin
#include "llmc/utils.h"

// Tokenizer for converting text to/from token IDs (GPT-2 BPE tokenizer)
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"

// Data loading utilities for reading training/validation datasets
// Handles sharded data files and multi-GPU data distribution
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
// defines: evalloader_init, evalloader_reset, evalloader_next_batch, evalloader_free
#include "llmc/dataloader.h"

// Random number generation utilities compatible with PyTorch
// defines: manual_seed, normal_ (same as torch.manual_seed and torch.normal)
#include "llmc/rand.h"

// Learning rate schedulers (linear warmup, cosine decay, etc.)
// defines: lr_scheduler_init, get_learning_rate
#include "llmc/schedulers.h"

// Sampling utilities for text generation from model logits
// defines: sample_softmax, random_f32
#include "llmc/sampler.h"

// Logging utilities for tracking training metrics
// defines: logger_init, logger_log_eval, logger_log_val, logger_log_train
#include "llmc/logger.h"

// Model FLOPs Utilization (MFU) calculation for performance monitoring
// defines: get_flops_promised
#include "llmc/mfu.h"

// Outlier detection for identifying anomalous loss/gradient values
// defines: OutlierDetector, init_detector, update_detector
#include "llmc/outlier_detector.h"

// ----------- GPU utilities -----------
// Core CUDA utilities and common definitions
// WARP_SIZE: Number of threads in a CUDA warp (32 on NVIDIA GPUs)
// MAX_1024_THREADS_BLOCKS: Maximum threads per block for kernel launches
// CEIL_DIV: Ceiling division macro for calculating grid dimensions
// cudaCheck: Macro for checking CUDA API return codes
// PRECISION_MODE: Compilation flag for FP32/FP16/BF16 precision
// NVTX_RANGE_FN: NVIDIA Tools Extension for profiling GPU code
#include "llmc/cuda_common.h"

// CUDA utility functions and data structures
// Packed128: Vectorized memory access structure for coalesced reads/writes
// f128, x128: 128-bit vector types for efficient memory bandwidth utilization
// warpReduceSum, warpReduceMax: Efficient intra-warp reduction primitives
// blockReduce: Block-level reduction using shared memory
// copy_and_cast_kernel: Type conversion between FP32/FP16/BF16
// cudaMallocConditionallyManaged: Smart allocation that falls back to managed memory if needed
#include "llmc/cuda_utils.cuh"

// cuBLAS and cuBLASLt configuration and utilities
// cuBLAS: NVIDIA's GPU-accelerated Basic Linear Algebra Subprograms library
// cuBLASLt: Lightweight cuBLAS with fused operations and mixed precision support
// CUBLAS_LOWP: Low-precision compute mode (FP16/BF16)
// cublasCheck: Error checking for cuBLAS API calls
// cublaslt_workspace_size, cublaslt_workspace: Temporary GPU memory for cuBLASLt
// cublas_compute: Compute type configuration (FP32, TF32, etc.)
// cublaslt_handle, cublas_handle: cuBLAS library handles
#include "llmc/cublas_common.h"

// ----------- Layer implementations in CUDA -----------
// Token and position embedding encoder (maps token IDs to dense vectors)
// defines: encoder_forward, encoder_backward
#include "llmc/encoder.cuh"

// Layer normalization and residual connection kernels
// Fused operations that combine multiple operations into a single kernel
// for reduced memory bandwidth and improved performance
// defines: layernorm_forward, residual_forward, fused_residual_forward5, layernorm_backward
#include "llmc/layernorm.cuh"

// Matrix multiplication using cuBLASLt and GELU activation
// matmul_cublaslt: Low-level interface to cuBLASLt for optimized GEMMs
// matmul_forward/backward: Forward and backward passes for linear layers
// gelu_forward: Gaussian Error Linear Unit activation function
// gelu_backward_inplace: In-place GELU gradient computation
#include "llmc/matmul.cuh"

#ifdef ENABLE_CUDNN
// cuDNN-accelerated attention implementation (optional, faster on some GPUs)
// cuDNN: NVIDIA's Deep Neural Network library with highly optimized primitives
// Uses flash attention and other optimizations for better performance
// defines: create_cudnn, destroy_cudnn, attention_forward_cudnn, attention_backward_cudnn
#include "llmc/cudnn_att.h"
#else
// Custom CUDA attention implementation (used when cuDNN is not available)
// Implements multi-head self-attention with causal masking
// defines: attention_forward, attention_backward
#include "llmc/attention.cuh"
#endif

// Fused classifier kernel that combines final linear layer with softmax and loss
// Computes cross-entropy loss and gradients in a single fused kernel
// defines: fused_classifier
#include "llmc/fused_classifier.cuh"

// AdamW optimizer implementation on GPU
// AdamW: Adam with decoupled weight decay (state-of-the-art optimizer for transformers)
// Maintains first and second moment estimates on GPU for efficiency
// defines: adamw_kernel3
#include "llmc/adamw.cuh"

// Global gradient norm calculation for gradient clipping
// Computes L2 norm of all gradients across all parameters
// defines: global_norm_squared
#include "llmc/global_norm.cuh"

// ----------- Multi-GPU support -----------
// NCCL (NVIDIA Collective Communications Library) for multi-GPU training
// Implements data parallelism and ZeRO optimizer state sharding
// ncclFloatX: NCCL datatype corresponding to the precision mode
// ncclCheck: Error checking for NCCL operations
// MultiGpuConfig: Configuration for multi-GPU setup (rank, world size, etc.)
// ShardInfo: Information about parameter sharding across GPUs
// printf0: Print only on rank 0 to avoid duplicate output
// multi_gpu_config: Global multi-GPU configuration
// multi_gpu_config_init/free: Initialize and cleanup multi-GPU setup
// set_zero_configs: Configure ZeRO optimizer sharding
// multi_gpu_cpu_float_sum: Sum a float value across all GPUs (on CPU)
// multi_gpu_barrier: Synchronize all GPUs
// multi_gpu_get_shard_offset: Get the shard offset for a parameter
// multi_gpu_async_reduce_gradient: Asynchronous gradient averaging/reduction
#include "llmc/zero.cuh"

// ============================================================================
// Global Variables
// ============================================================================

// ----------------------------------------------------------------------------
// I/O Buffer
// ----------------------------------------------------------------------------
// Reusable buffer for constructing file paths throughout the program
// Avoids repeated stack allocations for filename strings
char filename_buffer[512];

// ----------------------------------------------------------------------------
// GPU Device Information and Configuration
// ----------------------------------------------------------------------------
// cudaDeviceProp: Structure containing GPU capabilities and properties
//   - name: GPU model name (e.g., "NVIDIA A100-SXM4-40GB")
//   - major/minor: Compute capability version (e.g., 8.0 for A100)
//   - multiProcessorCount: Number of streaming multiprocessors (SMs)
//   - clockRate: GPU core clock speed
//   - totalGlobalMem: Total GPU memory in bytes
// This is populated in common_start() to query the GPU's capabilities
cudaDeviceProp deviceProp;

// CUDA Stream: A sequence of GPU operations that execute in order
// Using streams allows overlapping computation with data transfers for better performance
// main_stream: The primary stream for all GPU operations in this program
// All kernel launches and memory transfers use this stream for synchronization
cudaStream_t main_stream;

// Buffer size for efficient device <-> disk I/O operations
// 32 MB provides a good balance between memory usage and I/O performance
// Used when reading/writing model checkpoints from disk to GPU memory
constexpr const size_t IO_BUF_SIZE = 32 * 1024 * 1024;

// ============================================================================
// GPT-2 Model Data Structures
// ============================================================================

// ----------------------------------------------------------------------------
// Model Configuration
// ----------------------------------------------------------------------------
// GPT2Config: Hyperparameters defining the model architecture
// These values are set based on which GPT-2 variant is being trained
typedef struct {
    int max_seq_len;        // Maximum sequence length (T), e.g., 1024 for GPT-2, 2048 for GPT-3
                            // Determines the size of the position embedding table

    int vocab_size;         // Vocabulary size (V), e.g., 50257 for GPT-2
                            // Number of unique tokens the model can process

    int padded_vocab_size;  // Vocabulary size padded for GPU efficiency (Vp), e.g., 50304
                            // Padded to be divisible by 128 for optimal memory coalescing
                            // CUDA kernels perform best when tensor dimensions are multiples of warp size

    int num_layers;         // Number of transformer layers (L), e.g., 12 for GPT-2 124M
                            // Each layer contains attention + feedforward sublayers

    int num_heads;          // Number of attention heads (NH), e.g., 12 for GPT-2 124M
                            // Multi-head attention splits channels across heads for diverse representations

    int channels;           // Model dimension / embedding size (C), e.g., 768 for GPT-2 124M
                            // Also called d_model, this is the size of all hidden states
                            // Must be divisible by num_heads
} GPT2Config;

// ----------------------------------------------------------------------------
// Model Parameters (Weights and Biases)
// ----------------------------------------------------------------------------
// Total number of distinct parameter tensors in the GPT-2 model
constexpr const int NUM_PARAMETER_TENSORS = 16;

// ParameterTensors: All learnable weights and biases of the GPT-2 model
// All pointers point into GPU device memory (allocated via cudaMalloc)
// floatX is a typedef that can be float, __half, or __nv_bfloat16 depending on PRECISION_MODE
//
// MEMORY LAYOUT:
// All parameter tensors are allocated in a single contiguous block on the GPU
// This improves memory locality and simplifies memory management
//
// NOTATION: Shapes use (rows, cols) for matrices, matching typical ML notation
//   V = vocab_size, Vp = padded_vocab_size, T = max_seq_len
//   L = num_layers, C = channels, NH = num_heads
typedef struct {
    // Token Embeddings: Maps token IDs to dense vectors
    floatX* wte;        // Shape: (Vp, C) - Token embedding table
                        // wte[token_id] gives the C-dimensional embedding for that token
                        // Weight shared with final output projection (tied embeddings)

    // Position Embeddings: Adds positional information to token embeddings
    floatX* wpe;        // Shape: (maxT, C) - Position embedding table
                        // wpe[position] gives the position embedding for that sequence position
                        // Learned absolute positional encodings (not sinusoidal like in original Transformer)

    // First LayerNorm in each transformer block (pre-attention)
    floatX* ln1w;       // Shape: (L, C) - LayerNorm weights (scale/gain)
    floatX* ln1b;       // Shape: (L, C) - LayerNorm biases (shift/offset)

    // Attention QKV projection: Projects input to Query, Key, Value
    floatX* qkvw;       // Shape: (L, 3*C, C) - QKV weight matrix
                        // Single matrix for all three projections (Q, K, V) for efficiency
                        // Output is split into 3 parts during forward pass
    floatX* qkvb;       // Shape: (L, 3*C) - QKV bias vector

    // Attention output projection: Projects attention output back to residual stream
    floatX* attprojw;   // Shape: (L, C, C) - Attention output projection weight
    floatX* attprojb;   // Shape: (L, C) - Attention output projection bias

    // Second LayerNorm in each transformer block (pre-feedforward)
    floatX* ln2w;       // Shape: (L, C) - LayerNorm weights
    floatX* ln2b;       // Shape: (L, C) - LayerNorm biases

    // Feedforward network: Two-layer MLP with GELU activation
    // First expands to 4*C (typical ratio in transformers)
    floatX* fcw;        // Shape: (L, 4*C, C) - First feedforward layer weight
    floatX* fcb;        // Shape: (L, 4*C) - First feedforward layer bias

    // Second feedforward layer projects back to C dimensions
    floatX* fcprojw;    // Shape: (L, C, 4*C) - Second feedforward layer weight
    floatX* fcprojb;    // Shape: (L, C) - Second feedforward layer bias

    // Final LayerNorm (applied after all transformer layers)
    floatX* lnfw;       // Shape: (C,) - Final LayerNorm weights
    floatX* lnfb;       // Shape: (C,) - Final LayerNorm biases
} ParameterTensors;

// Compile-time assertion to ensure struct layout matches expected size
// This catches errors if fields are added/removed without updating NUM_PARAMETER_TENSORS
static_assert(sizeof(ParameterTensors) == NUM_PARAMETER_TENSORS * sizeof(void*), "Inconsistent sizes!");

// ============================================================================
// Parameter Memory Management Functions
// ============================================================================

/*
 * fill_in_parameter_sizes: Calculate the number of elements in each parameter tensor
 *
 * PARAMETERS:
 *   param_sizes  - Output array of size NUM_PARAMETER_TENSORS to store element counts
 *   param_sizeof - Output array of size NUM_PARAMETER_TENSORS to store per-element sizes in bytes
 *   config       - Model configuration specifying architecture dimensions
 *
 * PURPOSE:
 *   Computes how many elements (not bytes) are in each of the 16 parameter tensors.
 *   This is used to allocate the right amount of GPU memory and to properly
 *   partition the contiguous memory block into individual tensors.
 *
 * MEMORY CALCULATION EXAMPLE (GPT-2 124M):
 *   - wte: 50304 * 768 = 38,633,472 elements (largest tensor)
 *   - Total params ≈ 124M elements ≈ 248 MB (in BF16) or 496 MB (in FP32)
 */
void fill_in_parameter_sizes(size_t* param_sizes, size_t* param_sizeof, GPT2Config config) {
    // Extract commonly used dimensions for readability
    size_t Vp = config.padded_vocab_size;  // Padded vocabulary size
    size_t C = config.channels;             // Model dimension
    size_t maxT = config.max_seq_len;       // Maximum sequence length
    size_t L = config.num_layers;           // Number of transformer layers

    // Calculate element count for each parameter tensor
    // Index order matches the ParameterTensors struct definition
    param_sizes[0] = Vp * C;           // wte: Token embeddings
    param_sizes[1] = maxT * C;         // wpe: Position embeddings
    param_sizes[2] = L * C;            // ln1w: First LayerNorm weights
    param_sizes[3] = L * C;            // ln1b: First LayerNorm biases
    param_sizes[4] = L * (3 * C) * C;  // qkvw: QKV projection weights (3x because Q, K, V)
    param_sizes[5] = L * (3 * C);      // qkvb: QKV projection biases
    param_sizes[6] = L * C * C;        // attprojw: Attention output projection weights
    param_sizes[7] = L * C;            // attprojb: Attention output projection biases
    param_sizes[8] = L * C;            // ln2w: Second LayerNorm weights
    param_sizes[9] = L * C;            // ln2b: Second LayerNorm biases
    param_sizes[10] = L * (4 * C) * C; // fcw: First feedforward weights (4x expansion)
    param_sizes[11] = L * (4 * C);     // fcb: First feedforward biases
    param_sizes[12] = L * C * (4 * C); // fcprojw: Second feedforward weights
    param_sizes[13] = L * C;           // fcprojb: Second feedforward biases
    param_sizes[14] = C;               // lnfw: Final LayerNorm weights
    param_sizes[15] = C;               // lnfb: Final LayerNorm biases

    // Populate element sizes in bytes
    // Currently all parameters use the same type (floatX), but this array allows
    // for future flexibility (e.g., quantized weights with different element sizes)
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        param_sizeof[i] = sizeof(floatX);
    }
}

/*
 * malloc_and_point_parameters: Allocate GPU memory for all model parameters
 *
 * PARAMETERS:
 *   params         - Pointer to ParameterTensors struct to populate with pointers
 *   param_elements - Array of element counts for each tensor (from fill_in_parameter_sizes)
 *   param_sizeof   - Array of per-element sizes in bytes for each tensor
 *
 * RETURNS:
 *   Pointer to the base of the allocated GPU memory block
 *
 * GPU MEMORY ALLOCATION STRATEGY:
 *   Instead of allocating each parameter tensor separately (16 cudaMalloc calls),
 *   this function allocates ONE large contiguous block and partitions it.
 *
 *   BENEFITS:
 *   1. Reduces cudaMalloc overhead (allocator calls are expensive)
 *   2. Improves memory locality (parameters are adjacent in memory)
 *   3. Simplifies memory management (single cudaFree instead of 16)
 *   4. May reduce memory fragmentation
 *
 *   LAYOUT IN GPU MEMORY:
 *   |wte|wpe|ln1w|ln1b|qkvw|qkvb|attprojw|attprojb|ln2w|ln2b|fcw|fcb|fcprojw|fcprojb|lnfw|lnfb|
 *
 * EXAMPLE (GPT-2 124M in BF16):
 *   Total allocation: ~248 MB on GPU
 */
void* malloc_and_point_parameters(ParameterTensors* params, size_t* param_elements, size_t *param_sizeof) {
    // Calculate the total number of bytes needed for all parameters
    size_t num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters_bytes += param_elements[i] * param_sizeof[i];
    }

    // Allocate all parameters in one contiguous block on the GPU device
    // cudaMalloc allocates in GPU global memory (VRAM), not CPU RAM
    void* params_memory;
    cudaCheck(cudaMalloc((void**)&params_memory, num_parameters_bytes));

    // Set up array of pointers to the parameter fields in the ParameterTensors struct
    // This allows us to iterate through them programmatically
    floatX** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };

    // Partition the contiguous memory block into individual tensors
    // Each tensor pointer in the struct points to its slice of the memory block
    char* params_memory_iterator = (char*)params_memory;  // Use char* for byte-level arithmetic
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = (floatX*)params_memory_iterator;  // Assign this tensor's starting address
        params_memory_iterator += param_elements[i] * param_sizeof[i];  // Advance to next tensor
    }

    // Return the base pointer so caller can free it later with a single cudaFree()
    return params_memory;
}

// ----------------------------------------------------------------------------
// Activation Tensors (Intermediate Values During Forward/Backward Pass)
// ----------------------------------------------------------------------------
constexpr int NUM_ACTIVATION_TENSORS = 21;

/*
 * ActivationTensors: Stores all intermediate activations during forward pass
 *
 * MEMORY ALLOCATION:
 *   Like parameters, all activations are allocated in one contiguous GPU memory block
 *   These tensors are only needed during training, not for inference-only deployment
 *
 * GRADIENT CHECKPOINTING / RECOMPUTATION:
 *   Some tensors can be freed and recomputed during backward pass to save memory
 *   The 'recompute' flag controls which activations are saved vs recomputed:
 *   - recompute=0: Save all activations (uses most memory, fastest)
 *   - recompute=1: Recompute GELU activations during backward
 *   - recompute=2: Recompute GELU + LayerNorm activations during backward (saves most memory)
 *
 * NOTATION:
 *   B = batch_size, T = seq_len, C = channels, L = num_layers, NH = num_heads, V = vocab_size
 */
typedef struct {
    // Initial embeddings (token + position embeddings summed)
    floatX* encoded;        // Shape: (B, T, C)
                            // Output of encoder_forward, input to first transformer layer

    // First LayerNorm activations (pre-attention)
    floatX* ln1;            // Shape: (L, B, T, C)
                            // Normalized input to attention, saved for backward pass
                            // If recompute >= 2, this is NULL and recomputed during backward
    float* ln1_mean;        // Shape: (L, B, T)
                            // Mean values for each LayerNorm, needed for backward
    float* ln1_rstd;        // Shape: (L, B, T)
                            // Reciprocal standard deviation (1/sqrt(var + eps)) for backward

    // Attention output
    floatX* atty;           // Shape: (L, B, T, C)
                            // Output of attention mechanism, before residual connection

    // Attention weights (different sizes for cuDNN vs custom attention)
#if ENABLE_CUDNN
    float* att;             // Shape: (L, B, NH, T) - cuDNN format
                            // cuDNN uses a compact representation, stores only statistics
#else
    floatX* att;            // Shape: (L, B, NH, T, T) - Full attention matrix
                            // att[l, b, h, i, j] = attention weight from position i to position j
                            // Causal masking ensures att[..., i, j] = 0 for j > i
#endif

    // First residual connection output (after attention)
    floatX* residual2;      // Shape: (L, B, T, C)
                            // residual2 = residual + atty (skip connection)

    // Second LayerNorm activations (pre-feedforward)
    floatX* ln2;            // Shape: (L, B, T, C)
                            // Normalized input to feedforward network
                            // If recompute >= 2, this is NULL and recomputed
    float* ln2_mean;        // Shape: (L, B, T)
    float* ln2_rstd;        // Shape: (L, B, T)

    // Feedforward network activations
    floatX* fch;            // Shape: (L, B, T, 4*C)
                            // First feedforward layer output (before GELU activation)
    floatX* fch_gelu;       // Shape: (L, B, T, 4*C) if recompute < 1, else (B, T, 4*C)
                            // After GELU activation: fch_gelu = GELU(fch)
                            // If recompute >= 1, only one layer's worth is allocated and reused

    // Second residual connection output (after feedforward)
    floatX* residual3;      // Shape: (L, B, T, C)
                            // residual3 = residual2 + fcproj_output (skip connection)
                            // This is the final output of each transformer layer

    // Final LayerNorm (after all transformer layers)
    floatX* lnf;            // Shape: (B, T, C)
                            // Final normalized hidden states before output projection
                            // If recompute >= 2, this buffer is reused for ALL layernorms
    float* lnf_mean;        // Shape: (B, T)
    float* lnf_rstd;        // Shape: (B, T)

    // Loss computation
    float* losses;          // Shape: (B, T)
                            // Per-token cross-entropy losses, accumulated across micro-steps
                            // losses[b, t] = -log P(target[b, t] | input[b, :t])

    // QKV buffer (GPU-specific optimization)
    floatX* qkvr;           // Shape: (L, B, T, 3*C)
                            // Stores Query, Key, Value projections before splitting
                            // Not present in CPU version, needed for efficient GPU kernels

    // Multi-purpose output/scratch buffer
    floatX* output;         // Shape: max(B*T*3*C, B*NH*T*T, B*T*Vp)
                            // FORWARD PASS (inference): Stores final logits (B, T, Vp)
                            // BACKWARD PASS (training): Stores gradient of logits (dlogits)
                            // INTERMEDIATE: Used as scratch space during transformer blocks
                            // Size is the maximum of all uses to allow reuse

    // Additional scratch buffers for intermediate computations
    floatX* scratch_bt4c;   // Shape: (B, T, 4*C)
                            // Temporary storage for feedforward backward pass
    floatX* scratch_btc;    // Shape: (B, T, C)
                            // Temporary storage for gradient accumulation
} ActivationTensors;


struct TensorSpec {
    void** ptr;
    size_t size;
    DType type;
};


#define TENSOR_SPEC(pointer, size) TensorSpec{(void**)(&pointer), (size), dtype_of(pointer)};

void fill_in_activation_sizes(const ActivationTensors* data, TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS], size_t B, size_t T, GPT2Config config, int recompute) {
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;
    tensors[0] = TENSOR_SPEC(data->encoded, B * T * C);
    // if recompute >= 1 then we will recompute the layernorm forward activation during backward pass
    tensors[1] = TENSOR_SPEC(data->ln1,  (recompute < 2) ? L * B * T * C : 0);
    tensors[2] = TENSOR_SPEC(data->ln1_mean, L * B * T);
    tensors[3] = TENSOR_SPEC(data->ln1_rstd, L * B * T);
    tensors[4] = TENSOR_SPEC(data->atty, L * B * T * C);
    #ifdef ENABLE_CUDNN
    // FP32 stats tensor for cuDNN to be passed to backward pass
    tensors[5] = TENSOR_SPEC(data->att, L * B * NH * T);
    #else
    tensors[5] = TENSOR_SPEC(data->att, L * B * NH * T * T);
    #endif
    tensors[6] = TENSOR_SPEC(data->residual2, L * B * T * C);
    // if recompute >= 1 then we will recompute the layernorm forward activation during backward pass
    tensors[7] = TENSOR_SPEC(data->ln2, (recompute < 2) ? L * B * T * C : 0);
    tensors[8] = TENSOR_SPEC(data->ln2_mean, L * B * T);
    tensors[9] = TENSOR_SPEC(data->ln2_rstd, L * B * T);
    tensors[10] = TENSOR_SPEC(data->fch, L * B * T * 4*C);
    // if recompute >= 1 then we will recompute gelu_forward during backward and use this as scratch buffer
    tensors[11] = TENSOR_SPEC(data->fch_gelu, (recompute < 1) ? L * B * T * 4*C : B * T * 4*C);
    tensors[12] = TENSOR_SPEC(data->residual3, L * B * T * C);
    tensors[13] = TENSOR_SPEC(data->lnf, B * T * C);
    tensors[14] = TENSOR_SPEC(data->lnf_mean, B * T);
    tensors[15] = TENSOR_SPEC(data->lnf_rstd, B * T);
    tensors[16] = TENSOR_SPEC(data->losses, B * T);
    tensors[17] = TENSOR_SPEC(data->qkvr, L * B * T * 3*C);
    tensors[18] = TENSOR_SPEC(data->output, B * T * max(3*C, max(NH*T, Vp)));

    tensors[19] = TENSOR_SPEC(data->scratch_bt4c, B * T * 4 * C);
    tensors[20] = TENSOR_SPEC(data->scratch_btc, B * T * C);
}

void* malloc_and_point_activations(TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS]) {
    size_t bytes = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        bytes += tensors[i].size * sizeof_dtype(tensors[i].type);
    }

    printf0("allocating %d MiB for activations\n", (int)round(bytes / (1024 * 1024)));

    void* acts_memory;
    cudaCheck(cudaMalloc((void**)&acts_memory, bytes));

    // cudaMalloc does not guarantee initial memory values so we memset the allocation here
    // this matters because e.g. non-cuDNN attention assumes the attention buffer is zeroed
    // todo - up to ~100ms on slow GPUs, could theoretically be more selective, but this is safer
    cudaCheck(cudaMemset(acts_memory, 0, bytes));

    char* acts_memory_iterator = (char*)acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        // extra protection so we don't accidentally use an empty buffer
        if(tensors[i].size == 0) {
            *(tensors[i].ptr) = NULL;
        }else {
            *(tensors[i].ptr) = acts_memory_iterator;
            acts_memory_iterator += tensors[i].size * sizeof_dtype(tensors[i].type);
        }
    }
    return acts_memory;
}

// ----------------------------------------------------------------------------
// Main GPT-2 Model Structure
// ----------------------------------------------------------------------------
/*
 * GPT2: Complete state of the GPT-2 model including parameters, activations, and optimizer state
 *
 * MEMORY ARCHITECTURE:
 *   The model uses a carefully designed memory layout to maximize GPU efficiency:
 *   1. Parameters (params_memory): Model weights on GPU (BF16/FP16/FP32)
 *   2. Gradients (grads_memory): Parameter gradients on GPU (same type as params)
 *   3. Optimizer state (m_memory, v_memory): AdamW moments on GPU (FP32)
 *   4. Master weights (master_weights): Optional FP32 copy of params for stability
 *   5. Activations (acts_memory): Intermediate values during forward/backward
 *
 * MIXED PRECISION TRAINING:
 *   For BF16/FP16 training with master weights:
 *   - Forward/backward passes use BF16/FP16 for speed
 *   - Optimizer maintains FP32 master weights for numerical stability
 *   - After each update, master weights are rounded down to BF16/FP16
 *
 * MULTI-GPU SUPPORT:
 *   - ZeRO Stage 0: All GPUs have full copy of params, grads averaged
 *   - ZeRO Stage 1: Optimizer states sharded across GPUs
 *   See multi_gpu_config in zero.cuh for details
 */
typedef struct {
    // Model architecture configuration
    GPT2Config config;

    // ========== Parameters (Model Weights) ==========
    ParameterTensors params;                        // Pointers to individual parameter tensors on GPU
    size_t param_elements[NUM_PARAMETER_TENSORS];   // Number of elements in each tensor
    size_t param_sizeof[NUM_PARAMETER_TENSORS];     // Size of each element in bytes (sizeof(floatX))
    void* params_memory;                            // Base pointer to allocated GPU memory for all params
    size_t num_parameters;                          // Total number of parameters (e.g., 124M)
    size_t num_parameters_bytes;                    // Total bytes for all parameters

    // ========== Gradients ==========
    ParameterTensors grads;                         // Gradients of parameters (same structure as params)
    void* grads_memory;                             // Base pointer to GPU memory for all gradients

    // ========== AdamW Optimizer State ==========
    // AdamW maintains exponential moving averages of gradients and squared gradients
    float* m_memory;                                // First moment estimates (momentum) - always FP32
    float* v_memory;                                // Second moment estimates (RMSprop) - always FP32
    float* master_weights;                          // FP32 master copy of params (NULL if disabled)
                                                    // Enables stable mixed-precision training

    // ========== Activations (Forward Pass Intermediate Values) ==========
    ActivationTensors acts;                         // Pointers to activation tensors
    TensorSpec acts_specs[NUM_ACTIVATION_TENSORS];  // Size and type specifications for each activation
    void* acts_memory;                              // Base pointer to GPU memory for all activations

    // ========== Runtime State ==========
    int batch_size;                                 // Current batch size (B), set on first forward()
    int seq_len;                                    // Current sequence length (T), set on first forward()
    int* inputs;                                    // Input token IDs on GPU, shape: (B, T)
    int* targets;                                   // Target token IDs on GPU, shape: (B, T)

    // ========== Loss Tracking ==========
    float mean_loss;                                // Mean loss from last backward pass (averaged across GPUs)
                                                    // -1.0f indicates no loss computed yet
    float* accumulated_mean_loss;                   // GPU buffer to accumulate loss across micro-steps
    float* cpu_losses;                              // CPU buffer for per-token losses, shape: (B, T)
                                                    // Allocated with cudaMallocHost (pinned memory) for fast transfers

    // ========== Random Number Generation ==========
    unsigned long long rng_state;                   // Current RNG state for stochastic rounding
    unsigned long long rng_state_last_update;       // RNG state before last optimizer update
                                                    // Allows replaying rounding from master weights

    // ========== Training Configuration Flags ==========
    int use_master_weights;                         // 1 = keep FP32 master weights, 0 = disabled
    bool init_state;                                // True if optimizer state needs initialization
    int gelu_fusion;                                // GELU fusion mode:
                                                    // 0 = no fusion, 1 = fuse in forward, 2 = fuse in forward+backward
    int recompute;                                  // Activation recomputation mode (gradient checkpointing):
                                                    // 0 = save all, 1 = recompute GELU, 2 = recompute GELU+LayerNorm

    // ========== CPU Scratch Buffers (for encoder_backward workload distribution) ==========
    int* workload_indices;                          // CPU buffer for encoder backward pass
                                                    // Size: B*T*num_c_groups (int)
    int4* bucket_info;                              // CPU buffer for encoder backward pass
                                                    // Size: B*T*num_c_groups (int4)
} GPT2;

void gpt2_init_common(GPT2 *model) {
    // common inits outside of the model weights
    // memory lazily initialized in forward()
    model->acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->accumulated_mean_loss = NULL;
    model->cpu_losses = NULL;
    // the B,T params are determined and set, fixed on first batch in forward()
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f designates no loss, set at end of forward()
    model->params_memory = NULL;
    // memory lazily initialized in backward()
    model->grads_memory = NULL;
    model->workload_indices = NULL; // on cpu, for encoder_backward
    model->bucket_info = NULL; // on cpu, for encoder_backward
    // memory lazily initialized in update()
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->master_weights = NULL;
    // other default settings
    model->rng_state = 13371337 + multi_gpu_config.process_rank; // used in stochastic rounding
    model->use_master_weights = 1; // safe default: do keep master weights in fp32
    model->init_state = true;
    model->recompute = 1; // good default: recompute gelu but not layernorm
    model->gelu_fusion = 0; //deviceProp.major >= 9 ? 2 : 0; // default: off for now (default must match main())
}

void gpt2_allocate_weights(GPT2 *model) {
    // fill in all the parameter tensor dimensions and types
    fill_in_parameter_sizes(model->param_elements, model->param_sizeof, model->config);
    model->num_parameters = 0;
    model->num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        model->num_parameters += model->param_elements[i];
        model->num_parameters_bytes += model->param_elements[i] * model->param_sizeof[i];
    }
    // create memory for model parameters on the device
    assert(model->params_memory == nullptr);
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_elements, model->param_sizeof);
}

void gpt2_allocate_state(GPT2 *model, int B, int T) {
    printf0("allocating %d MiB for parameter gradients\n", (int)round(model->num_parameters * sizeof(floatX) / (1024 * 1024)));
    assert(model->grads_memory == nullptr);
    model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_elements, model->param_sizeof);

    // record the current B,T as well
    model->batch_size = B;
    model->seq_len = T;

    // allocate the space
    fill_in_activation_sizes(&model->acts, model->acts_specs, B, T, model->config, model->recompute);
    model->acts_memory = malloc_and_point_activations(model->acts_specs);
    // also create memory for caching inputs and targets
    cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&model->targets, B * T * sizeof(int)));
    cudaCheck(cudaMalloc(((void**)&model->accumulated_mean_loss), sizeof(float)));
    cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * T * sizeof(float)));

    // initialise cpu scratch buffers for encoder backward
    size_t num_c_groups = CEIL_DIV(model->config.channels, (WARP_SIZE * x128::size));
    assert((size_t)(model->batch_size * model->seq_len) * num_c_groups < (1ULL<<31ULL)); // todo - maybe an issue for llama3-400B(?)
    model->workload_indices = (int*)mallocCheck(sizeof(int) * model->batch_size * model->seq_len * num_c_groups);
    model->bucket_info = (int4*)mallocCheck(sizeof(int4) * model->batch_size * model->seq_len * num_c_groups);

    // cudaMallocConditionallyManaged can fall back to cudaMallocManaged if not enough memory on device
    // and returns a status code of 1 if it had to fall back, in that case we want to print warning.
    int memory_status = 0;

    // we will now init the optimizer states and master weights
    // this is usually a substantial amount of memory allocation right here.
    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters; // num parameters we are responsible for
    printf0("allocating %zu MiB for AdamW optimizer state m\n", (shard_num_parameters * sizeof(float)) >> 20);
    printf0("allocating %zu MiB for AdamW optimizer state v\n", (shard_num_parameters * sizeof(float)) >> 20);
    assert(model->m_memory == nullptr);
    assert(model->v_memory == nullptr);
    memory_status |= cudaMallocConditionallyManaged((void**)&model->m_memory, shard_num_parameters * sizeof(float));
    memory_status |= cudaMallocConditionallyManaged((void**)&model->v_memory, shard_num_parameters * sizeof(float));

    if (model->use_master_weights == 1) {
        assert(model->master_weights == nullptr);
        printf0("allocating %zu MiB for master copy of params\n", (shard_num_parameters * sizeof(float)) >> 20);
        memory_status |= cudaMallocConditionallyManaged((void**) &model->master_weights, shard_num_parameters * sizeof(float));
    }

    // report on mixed memory allocation status (re-using our float reduce function, bit awk ok)
    int reduced_memory_status = (int) multi_gpu_cpu_float_sum((float)memory_status, &multi_gpu_config);
    if (reduced_memory_status >= 1) {
        printf0("WARNING: Fell back to cudaMallocManaged when initializing m,v,master_weights on %d GPUs\n", reduced_memory_status);
        printf0("         Prevents an OOM, but code may run much slower due to device <-> host memory movement\n");
    }
    // report on device memory usage
    size_t free, total;
    cudaCheck(cudaMemGetInfo(&free, &total));
    printf0("device memory usage: %zd MiB / %zd MiB\n", (total-free) / 1024 / 1024, total / 1024 / 1024);
    // give an estimate of the maximum batch size
    size_t bytes_per_sequence = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        bytes_per_sequence += model->acts_specs[i].size * sizeof_dtype(model->acts_specs[i].type) / B;
    }
    printf0("memory per sequence: %zu MiB\n", bytes_per_sequence / 1024 / 1024);
    printf0(" -> estimated maximum batch size: %zu\n", B + free / bytes_per_sequence);
}

void gpt2_write_to_checkpoint(GPT2 *model, const char* checkpoint_path) {
    // write the model to a checkpoint file
    printf0("Writing model to %s\n", checkpoint_path);
    FILE *model_file = fopenCheck(checkpoint_path, "wb");
    // write the header first
    int model_header[256];
    memset(model_header, 0, sizeof(model_header));
    model_header[0] = 20240326; // magic number
    assert(PRECISION_MODE == PRECISION_FP32 || PRECISION_MODE == PRECISION_BF16);
    model_header[1] = PRECISION_MODE == PRECISION_FP32 ? 3 : 5; // version
    model_header[2] = model->config.max_seq_len;
    model_header[3] = model->config.vocab_size;
    model_header[4] = model->config.num_layers;
    model_header[5] = model->config.num_heads;
    model_header[6] = model->config.channels;
    model_header[7] = model->config.padded_vocab_size;
    fwriteCheck(model_header, sizeof(int), 256, model_file);
    // write the parameters
    device_to_file(model_file, model->params_memory, model->num_parameters_bytes,
                   IO_BUF_SIZE, main_stream);
    // close file, we're done
    fcloseCheck(model_file);
}

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path, bool weight_init=true) {
    // If weight_init is true, we will load the weights from this checkpoint .bin file
    // We sometimes want this to be false, if we are going to initialize these weights from
    // the master weights that are instead stored in the state .bin file.
    // In that case, this function mostly loads the model hyperparameters from the header.

    if (PRECISION_MODE == PRECISION_FP16) {
        // TODO for later perhaps, would require us dynamically converting the
        // model weights from fp32 to fp16 online, here in this function, or writing
        // the fp16 weights directly from Python, which we only do for fp32/bf16 atm.
        fprintf(stderr, "build_from_checkpoint() does not support fp16 right now.\n");
        exit(EXIT_FAILURE);
    }

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(EXIT_FAILURE); }
    int version = model_header[1];
    if (!(version == 3 || version == 5)) {
        // 3 = fp32, padded vocab
        // 5 = bf16, padded vocab, layernorms also in bf16
        fprintf(stderr, "Bad version in model file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }

    // check if the precision mode of the checkpoing matches the model precision
    if (weight_init) {
        if (PRECISION_MODE == PRECISION_BF16 && version != 5) {
            fprintf(stderr, "Precision is configured as BF16 but model at %s is not.\n", checkpoint_path);
            fprintf(stderr, "---> HINT: are you sure you're loading a _bf16.bin file?\n");
            exit(EXIT_FAILURE);
        }
        if (PRECISION_MODE == PRECISION_FP32 && version != 3) {
            fprintf(stderr, "Precision is configured as FP32 but model at %s is not.\n", checkpoint_path);
            fprintf(stderr, "---> HINT: to turn on FP32 you have to compile like: `make train_gpt2cu PRECISION=FP32`\n");
            fprintf(stderr, "---> HINT: are you sure you're loading a .bin file without any _bf16 in the name?\n");
            exit(EXIT_FAILURE);
        }
    }

    // read in hyperparameters
    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];

    // allocate memory for the model parameters
    gpt2_allocate_weights(model);

    // read in the parameters if weight_init is true
    if (weight_init) {
        assert(model->params_memory != NULL);
        file_to_device(model->params_memory, model_file, model->num_parameters_bytes, IO_BUF_SIZE, main_stream);
    }
    fcloseCheck(model_file);

    // only return from this function once we are certain the params are ready on the GPU
    cudaCheck(cudaDeviceSynchronize());
}

void gpt2_set_hyperparameters(GPT2Config* config, const char* depth_str) {
    int depth = atoi(depth_str);
    assert(depth > 0); // atoi returns 0 if not a number
    int channels, num_heads;
    if      (depth == 6)  { channels = 384; num_heads = 6; }   // (unofficial) gpt2-tiny (30M)
    else if (depth == 12) { channels = 768; num_heads = 12; }  // gpt2 (124M)
    else if (depth == 24) { channels = 1024; num_heads = 16; } // gpt2-medium (350M)
    else if (depth == 36) { channels = 1280; num_heads = 20; } // gpt2-large (774M)
    else if (depth == 48) { channels = 1600; num_heads = 25; } // gpt2-xl (1558M)
    else if (depth == 60) { channels = 1920; num_heads = 30; } // (unofficial) 2.7B
    else if (depth == 72) { channels = 2880; num_heads = 30; } // (unofficial) 7.3B
    else if (depth == 84) { channels = 3456; num_heads = 36; } // (unofficial) 12.2B
    else { fprintf(stderr, "Unsupported GPT-2 depth: %d\n", depth); exit(EXIT_FAILURE); }
    config->num_layers = depth;
    config->channels = channels;
    config->num_heads = num_heads;
    config->max_seq_len = 1024;
}

void gpt3_set_hyperparameters(GPT2Config* config, const char* channels_str) {
    // we use channels instead of depth for GPT-3 because GPT-3 model depths are not one-to-one
    // note that our models are not necessarily identical to GPT-3 because
    // we use dense attention, not the alternating dense/banded attention of GPT-3
    int channels = atoi(channels_str);
    assert(channels > 0); // atoi returns 0 if not a number
    int depth, head_size;
    if      (channels == 384)   { depth = 6;  head_size = 64; }  // (unofficial) gpt3-tiny (31M)
    else if (channels == 768)   { depth = 12; head_size = 64; }  // gpt3-small (125M)
    else if (channels == 1024)  { depth = 24; head_size = 64; }  // gpt3-medium (350M)
    else if (channels == 1536)  { depth = 24; head_size = 96; }  // gpt3-large (760M)
    else if (channels == 2048)  { depth = 24; head_size = 128; } // gpt3-xl (1.3B) [heads fixed]
    else if (channels == 2560)  { depth = 32; head_size = 80; }  // gpt3-2.7B
    else if (channels == 4096)  { depth = 32; head_size = 128; } // gpt3-6.7B
    else if (channels == 5140)  { depth = 40; head_size = 128; } // gpt3-13B
    else if (channels == 12288) { depth = 96; head_size = 128; } // gpt3 (175B)
    else { fprintf(stderr, "Unsupported GPT-3 channels: %d\n", channels); exit(EXIT_FAILURE); }
    assert(channels % head_size == 0);
    config->num_layers = depth;
    config->channels = channels;
    config->num_heads = channels / head_size;
    config->max_seq_len = 2048; // NOTE: GPT-3 uses context length of 2048 tokens, up from 1024 in GPT-2
}

void gpt_build_from_descriptor(GPT2 *model, const char* descriptor) {
    // The model descriptor can be:
    // - legacy format "dX", where X is number, e.g. "d12". This creates GPT-2 model with 12 layers.
    // - new explicit format "gpt2:dX", same as above, e.g. "gpt2:d48" for GPT-2 with 48 layers.
    // - "gpt3:cX", where X is now the channel count, e.g. "gpt3:c768" is the smallest GPT-3 model.

    // check the valid prexies and dispatch to the right setup function
    assert(descriptor != NULL);
    size_t len = strlen(descriptor);
    if (len > 1 && descriptor[0] == 'd') {
        gpt2_set_hyperparameters(&model->config, descriptor + 1); // pass along the depth str without the 'd'
    } else if (len > 6 && strncmp(descriptor, "gpt2:d", 6) == 0) {
        gpt2_set_hyperparameters(&model->config, descriptor + 6); // pass along the depth str without the 'gpt2:d'
    } else if (len > 6 && strncmp(descriptor, "gpt3:c", 6) == 0) {
        gpt3_set_hyperparameters(&model->config, descriptor + 6); // pass along the channels str without the 'gpt3:c'
    } else {
        fprintf(stderr, "Unsupported model descriptor: %s\n", descriptor); exit(EXIT_FAILURE);
    }

    // both GPT-2 and GPT-3 use the same tokenizer with 50257 tokens
    model->config.vocab_size = 50257;
    model->config.padded_vocab_size = 50304; // padded to 128 for CUDA kernel efficiency

    gpt2_allocate_weights(model);

    // allocate and random init the memory for all the parameters with GPT-2 schema
    // weights ~N(0, 0.02), biases 0, c_proj weights ~N(0, 0.02/(2*L)**0.5)
    // NOTE: assuming all parameters are of the type floatX, could be relaxed later
    mt19937_state init_rng;
    manual_seed(&init_rng, 42);
    floatX* params_memory_cpu = (floatX*)mallocCheck(model->num_parameters_bytes);
    memset(params_memory_cpu, 0, model->num_parameters_bytes);
    // fill in all the weights with random values
    float residual_scale = 1.0f / sqrtf(2.0f * model->config.num_layers);
    // we have to init all these tensors exactly in the order that PyTorch initializes them
    // so that we can match them up and get correctness and exactly the same initial conditions
    size_t L = model->config.num_layers;
    size_t offset = 0;
    for (int l = 0; l < L; l++) {
        offset = 0;
        for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
            // the layernorm parameters are all initialized to 1
            if (l == 0 && (i == 2 || i == 8 || i == 14)) { // only at l = 0 to init these just once
                for (size_t j = 0; j < model->param_elements[i]; j++) {
                    params_memory_cpu[offset + j] = 1.0f;
                }
            }
            // weights tensors are handled here
            if ((l == 0 && (i == 0 || i == 1)) // only at l = 0, init the wte and wpe tensors
              || i == 4 || i == 6 || i == 10 || i == 12) {
                size_t n = model->param_elements[i];
                size_t layer_offset = 0;
                if (i == 0) {
                    // for wte tensor (padded vocab) override to init V instead of Vp rows
                    n = model->config.vocab_size * model->config.channels;
                }
                if (i == 4 || i == 6 || i == 10 || i == 12) {
                    // weight tensors, we are only initializing layer l
                    assert(n % L == 0);
                    n = n / L;
                    layer_offset = l * n;
                }
                // in GPT-2, the projections back into the residual stream are additionally
                // scaled by 1/sqrt(2*L) for training stability
                float scale = (i == 6 || i == 12) ? 0.02f * residual_scale : 0.02f;
                // okay let's draw the random numbers and write them
                float *fp32_buffer = (float*)mallocCheck(n * sizeof(float));
                normal_(fp32_buffer, n, 0.0f, scale, &init_rng);
                for (size_t j = 0; j < n; j++) {
                    params_memory_cpu[offset + layer_offset + j] = (floatX)fp32_buffer[j];
                }
                free(fp32_buffer);
            }
            offset += model->param_elements[i];
        }
    }

    // copy them to GPU
    cudaCheck(cudaMemcpy(model->params_memory, params_memory_cpu, model->num_parameters_bytes, cudaMemcpyHostToDevice));
    free(params_memory_cpu);
}

// ============================================================================
// Forward Pass: Compute Model Predictions
// ============================================================================

/*
 * gpt2_forward: Execute forward pass through the GPT-2 model on GPU
 *
 * PARAMETERS:
 *   model  - Pointer to GPT2 model structure (must be initialized)
 *   inputs - Host (CPU) memory pointer to input token IDs, shape: (B, T)
 *   B      - Batch size (number of sequences to process in parallel)
 *   T      - Sequence length (number of tokens in each sequence)
 *
 * OUTPUTS:
 *   After execution, model->acts.output contains logits, shape: (B, T, Vp)
 *   logits[b, t, v] = unnormalized log probability of token v at position t in sequence b
 *
 * ALGORITHM:
 *   1. Encoder: Convert token IDs to embeddings (token + position embeddings)
 *   2. For each transformer layer:
 *      a. LayerNorm + Multi-head self-attention + Residual connection
 *      b. LayerNorm + Feedforward network + Residual connection
 *   3. Final LayerNorm
 *   4. Project to vocabulary size (logits)
 *
 * GPU SYNCHRONIZATION:
 *   - All operations are launched on main_stream
 *   - Function ends with cudaDeviceSynchronize() to ensure completion
 *   - This makes the function blocking (host waits for GPU)
 *
 * PERFORMANCE NOTES:
 *   - Uses cuBLASLt for matrix multiplications (highly optimized)
 *   - Attention can use cuDNN (if ENABLE_CUDNN) or custom CUDA kernels
 *   - Fusion optimizations: fused residual+layernorm, optional GELU fusion
 *   - Memory reuse: scratch buffers are reused to minimize allocations
 *
 * MEMORY TRANSFERS:
 *   - Input token IDs are copied from host to device (cudaMemcpy)
 *   - All computation happens on GPU
 *   - Logits remain on GPU (copied to host only if needed for inference)
 */
void gpt2_forward(GPT2 *model, const int* inputs, size_t B, size_t T) {
    // NVTX markers for profiling with NVIDIA Nsight Systems
    NVTX_RANGE_FN();

    // Use size_t instead of int to avoid overflow in large models
    // Example: l * B * NH * T * T can overflow 32-bit int at B=16 for large models
    // size_t is 64-bit, providing much larger range

    // Validate model initialization
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // Extract model configuration for convenience
    const size_t V = model->config.vocab_size;          // Actual vocabulary size (50257 for GPT-2)
    const size_t Vp = model->config.padded_vocab_size;  // Padded vocab size (50304 for GPT-2)
    const size_t L = model->config.num_layers;          // Number of transformer layers
    const size_t NH = model->config.num_heads;          // Number of attention heads
    const size_t C = model->config.channels;            // Model dimension

    // Validate batch size and sequence length
    // The model's activation buffers are allocated for specific (B, T) dimensions
    // Smaller (B, T) is OK for inference, but larger would overflow buffers
    if (B > model->batch_size || T > model->seq_len) {
        printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
        exit(EXIT_FAILURE);
    }

    // ========== Transfer Input Data to GPU ==========
    // Copy input token IDs from CPU (host) to GPU (device)
    // This is one of the few host-device data transfers in the forward pass
    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));

    // Validate input tokens on CPU while GPU copy is in progress
    // All token IDs must be in range [0, V), otherwise indexing errors will occur
    tokenCheck(inputs, B*T, V);

    // ========== Forward Pass Computation ==========
    // Extract pointers for convenience and readability
    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;

    // STEP 1: Encoder - Convert token IDs to embeddings
    // encoder_forward adds token embeddings (wte) and position embeddings (wpe)
    // Output shape: (B, T, C) stored in acts.encoded
    // Formula: encoded[b, t, :] = wte[inputs[b, t], :] + wpe[t, :]
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C, main_stream);

    // STEP 2: First LayerNorm (applied before first transformer layer)
    // This LayerNorm is NOT fused with a residual connection (unlike later ones)
    // If recompute >= 2, we reuse the lnf buffer to save memory
    layernorm_forward((model->recompute < 2) ? acts.ln1 : acts.lnf, acts.ln1_mean, acts.ln1_rstd, acts.encoded, params.ln1w, params.ln1b, B, T, C, main_stream);

    // ========== STEP 3: Transformer Layers ==========
    // Each layer applies: Attention block + Feedforward block
    // Both blocks use Pre-LayerNorm architecture (normalize before operation)
    // and include residual connections (skip connections)
    for (int l = 0; l < L; l++) {
        // NVTX profiling marker for this layer
        NvtxRange layer_range("Layer", l);

        // ========== Get Input Residual Stream for This Layer ==========
        // The residual stream carries information through the network
        // Layer 0 uses the initial encoded embeddings
        // Subsequent layers use the output (residual3) from the previous layer
        floatX* residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // ========== Get Pointers to This Layer's Parameters ==========
        // All parameter tensors are stored contiguously per layer
        // We offset into the params tensors to get this layer's slice
        floatX* l_qkvw = params.qkvw + l * 3*C * C;       // QKV projection weights
        floatX* l_qkvb = params.qkvb + l * 3*C;           // QKV projection biases
        floatX* l_attprojw = params.attprojw + l * C * C; // Attention output projection weights
        floatX* l_attprojb = params.attprojb + l * C;     // Attention output projection biases
        floatX* l_ln2w = params.ln2w + l * C;             // Second LayerNorm weights
        floatX* l_ln2b = params.ln2b + l * C;             // Second LayerNorm biases
        floatX* l_fcw = params.fcw + l * 4*C * C;         // Feedforward first layer weights
        floatX* l_fcb = params.fcb + l * 4*C;             // Feedforward first layer biases
        floatX* l_fcprojw = params.fcprojw + l * C * 4*C; // Feedforward second layer weights
        floatX* l_fcprojb = params.fcprojb + l * C;       // Feedforward second layer biases

        // ========== Get Pointers to This Layer's Activation Buffers ==========
        // If recompute >= 2, LayerNorm activations are not saved, we reuse lnf buffer
        floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + l * B * T * C : acts.lnf;
        floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;     // QKV output buffer
        floatX* l_atty = acts.atty + l * B * T * C;       // Attention output
        floatX* l_residual2 = acts.residual2 + l * B * T * C; // After attention residual connection
        floatX* l_ln2 = (model->recompute < 2) ? acts.ln2 + l * B * T * C : acts.lnf;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;    // Second LayerNorm mean
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;    // Second LayerNorm rstd
        floatX* l_fch = acts.fch + l * B * T * 4*C;       // Feedforward hidden layer (before GELU)

        // GRADIENT CHECKPOINTING OPTIMIZATION:
        // If recompute >= 1, we don't save GELU activations for all layers
        // Instead, we reuse a single buffer and recompute GELU during backward
        // This saves significant memory: L * B * T * 4*C elements
        floatX* l_fch_gelu = (model->recompute < 1) ? acts.fch_gelu + l * B * T * 4*C : acts.fch_gelu;

        floatX* l_residual3 = acts.residual3 + l * B * T * C; // Final layer output

        // Scratch buffer for intermediate computations
        // Reused for QKV matmul output, attention projection, feedforward projection
        floatX* scratch = (floatX*)acts.output;

        // ========== ATTENTION BLOCK ==========
        // Implements multi-head self-attention with causal masking
        // Input: l_ln1 (normalized residual stream)
        // Output: l_atty (attention output, before residual connection)

        #ifdef ENABLE_CUDNN
        // ===== cuDNN Attention Path (Optimized) =====
        // Uses NVIDIA's cuDNN library for highly optimized attention
        float* l_att = (float*)acts.att + l * B * NH * T;

        // 1. Compute Q, K, V projections using cuBLASLt (matrix multiply)
        //    l_qkvr = l_ln1 @ l_qkvw + l_qkvb
        //    Output shape: (B, T, 3*C) where 3*C = C for Q + C for K + C for V
        matmul_forward_cublaslt(l_qkvr, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);

        // 2. Apply multi-head attention using cuDNN
        //    Internally splits l_qkvr into Q, K, V and computes attention
        attention_forward_cudnn(l_atty, (float*)l_att, l_qkvr, B, T, NH, C, main_stream);

        #else
        // ===== Custom CUDA Attention Path =====
        // Used when cuDNN is not available or disabled
        floatX* l_att = acts.att + l * B * NH * T * T;

        // Clear attention buffer if sequence length changed
        // Causal masking requires future positions to be zero
        if (T != model->seq_len) {
            cudaCheck(cudaMemset(l_att, 0, B * NH * T * T * sizeof(floatX)));
        }

        // 1. Compute QKV projections (same as cuDNN path)
        matmul_forward_cublaslt(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);

        // 2. Apply custom attention kernel
        //    Computes: softmax(Q @ K^T / sqrt(d_k)) @ V with causal masking
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH, main_stream);
        #endif

        // 3. Project attention output back to residual stream dimension
        //    scratch = l_atty @ l_attprojw + l_attprojb
        matmul_forward_cublaslt(scratch, l_atty, l_attprojw, l_attprojb, B, T, C, C, main_stream);

        // 4. FUSED: Add residual connection + Apply LayerNorm for next block
        //    l_residual2 = residual + scratch (residual connection)
        //    l_ln2 = LayerNorm(l_residual2) (prepare for feedforward)
        //    This fusion reduces memory bandwidth by combining operations
        fused_residual_forward5(l_residual2, l_ln2, l_ln2_mean, l_ln2_rstd, residual, scratch, l_ln2w, l_ln2b, B*T, C, main_stream);

        // ========== FEEDFORWARD BLOCK ==========
        // Two-layer MLP with GELU activation: FFN(x) = W2 @ GELU(W1 @ x + b1) + b2
        // Expands to 4*C then projects back to C (typical transformer ratio)

        // 1. First feedforward layer with optional GELU fusion
        //    l_fch = l_ln2 @ l_fcw + l_fcb (linear transformation)
        //    l_fch_gelu = GELU(l_fch) (activation, may be fused into matmul)
        //    gelu_fusion flag controls whether GELU is fused into the matrix multiply
        matmul_forward_cublaslt(l_fch_gelu, l_ln2, l_fcw, l_fcb, B, T, C, 4*C, main_stream, l_fch, model->gelu_fusion);

        // 2. Second feedforward layer (projection back to C dimensions)
        //    scratch = l_fch_gelu @ l_fcprojw + l_fcprojb
        matmul_forward_cublaslt(scratch, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C, main_stream);

        // ========== CROSS-LAYER FUSION OPTIMIZATION ==========
        // Fuse this layer's residual with next layer's LayerNorm to reduce memory traffic
        if(l+1 != L) {
            // Not the last layer: prepare first LayerNorm for next layer
            floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + (l + 1) * B * T * C : acts.lnf;
            float* l_ln1_mean = acts.ln1_mean + (l + 1) * B * T;
            float* l_ln1_rstd = acts.ln1_rstd + (l + 1) * B * T;
            const floatX* l_ln1w = params.ln1w + (l + 1) * C;
            const floatX* l_ln1b = params.ln1b + (l + 1) * C;

            // l_residual3 = l_residual2 + scratch (residual connection)
            // l_ln1 = LayerNorm(l_residual3) (prepare for next layer's attention)
            fused_residual_forward5(l_residual3, l_ln1, l_ln1_mean, l_ln1_rstd, l_residual2, scratch, l_ln1w, l_ln1b,
                                    B * T, C, main_stream);
        } else {
            // Last layer: apply final LayerNorm instead of next layer's LayerNorm
            // l_residual3 = l_residual2 + scratch (residual connection)
            // acts.lnf = LayerNorm(l_residual3) (prepare for output projection)
            fused_residual_forward5(l_residual3, acts.lnf, acts.lnf_mean, acts.lnf_rstd, l_residual2, scratch,
                                    params.lnfw, params.lnfb,
                                    B * T, C, main_stream);
        }
    }

    // ========== STEP 4: Output Projection to Vocabulary ==========
    // Project final hidden states to vocabulary logits
    // acts.output = acts.lnf @ params.wte^T (weight tying: same as token embeddings)
    // Output shape: (B, T, Vp) - logits for each token position
    // Note: No bias in the final projection (standard in GPT-2)
    matmul_forward_cublaslt(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp, main_stream);

    // ========== Synchronize GPU ==========
    // Wait for all GPU operations to complete before returning
    // This ensures activations are ready for subsequent loss computation or backward pass
    cudaCheck(cudaDeviceSynchronize());
}


// Forwards both the model and the loss and is used for validation splits and evals.
// In particular it populates cpu_losses with loss at each token.
// Some of the evals (e.g. HellaSwag) require the per-token losses, which are produced here.
float gpt2_validate(GPT2 *model, const int* inputs, const int* targets, size_t B, size_t T) {
    assert(targets != NULL);
    // forward the model itself
    gpt2_forward(model, inputs, B, T);
    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;

    NvtxRange classifier_and_loss_range("classifier_and_loss");
    ActivationTensors acts = model->acts;
    float mean_loss = 0.0f;
    // fused classifier: does the forward pass and first part of the backward pass
    const float dloss = 1.0f / (B * T); // results in the uniform average loss over all elements
    // note: we don't need to generate dlogits here
    cudaCheck(cudaMemset(acts.losses, 0, B*T*sizeof(float)));
    cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(targets, B*T, V); // while the memcpy is underway, validate the targets
    fused_classifier(acts.output, acts.losses, dloss, model->targets, B, T, V, Vp, False, main_stream);
    cudaCheck(cudaMemcpy(model->cpu_losses, acts.losses, B * T * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < B*T; i++) {
        mean_loss += model->cpu_losses[i];
    }
    mean_loss /= B*T;
    cudaCheck(cudaDeviceSynchronize());
    return mean_loss;
}

// ============================================================================
// Backward Pass: Compute Gradients via Backpropagation
// ============================================================================

/*
 * gpt2_backward_and_reduce: Execute backward pass and gradient reduction across GPUs
 *
 * PARAMETERS:
 *   model            - GPT2 model with populated activations from forward pass
 *   inputs           - Input token IDs (only used for encoder backward)
 *   targets          - Target token IDs for loss computation, shape: (B, T)
 *   grad_accum_steps - Number of gradient accumulation steps
 *   micro_step       - Current micro-step index (0 to grad_accum_steps-1)
 *
 * GRADIENT ACCUMULATION:
 *   Instead of updating weights after every batch, we can accumulate gradients
 *   across multiple micro-batches to simulate a larger effective batch size.
 *   This is crucial when GPU memory limits the batch size.
 *
 *   EXAMPLE:
 *     grad_accum_steps = 4, micro_batch_size = 8
 *     Effective batch size = 4 * 8 = 32
 *
 *   STEPS:
 *     1. micro_step 0: Zero gradients, forward, backward (gradients += ...)
 *     2. micro_step 1-2: Forward, backward (gradients += ...)
 *     3. micro_step 3: Forward, backward (gradients += ...), reduce gradients, update weights
 *
 * MULTI-GPU GRADIENT REDUCTION:
 *   On the last micro-step, gradients are synchronized across all GPUs:
 *   - ZeRO Stage 0: Average gradients via AllReduce
 *   - ZeRO Stage 1: ReduceScatter to shard gradients across GPUs
 *
 * ALGORITHM:
 *   1. Compute loss and dL/d(logits) via fused classifier
 *   2. Backpropagate through final LayerNorm
 *   3. For each transformer layer (in reverse):
 *      - Backpropagate through feedforward block
 *      - Backpropagate through attention block
 *   4. Backpropagate through encoder (embedding layers)
 *   5. On last step: Reduce gradients across GPUs
 *
 * MEMORY REUSE:
 *   - Uses acts.output as scratch buffer for intermediate gradients
 *   - Recomputes activations if gradient checkpointing is enabled
 *   - Gradient buffers are reused across layers
 */
void gpt2_backward_and_reduce(GPT2 *model, int* inputs, const int* targets, int grad_accum_steps, int micro_step) {
    // Validate gradient memory is allocated
    if(model->grads_memory == nullptr) {
        fprintf(stderr, "Need to allocate gradients before backward");
        exit(EXIT_FAILURE);
    }

    // NVTX profiling marker
    NVTX_RANGE_FN();

    // Check if this is the last micro-step (when we'll do gradient reduction and weight update)
    bool last_step = micro_step == grad_accum_steps - 1;

    // ========== Initialize Gradient Accumulation ==========
    // On the first micro-step, zero out gradient buffers
    // On subsequent micro-steps, we accumulate (+=) into existing gradients
    if (micro_step == 0) {
        // Two accumulation buffers need to be zeroed:
        // 1) acts.losses: Accumulates per-token losses across micro-steps
        // 2) grads_memory: Accumulates parameter gradients across micro-steps
        cudaCheck(cudaMemsetAsync(model->acts.losses, 0, model->batch_size * model->seq_len * sizeof(float), main_stream));
        cudaCheck(cudaMemsetAsync(model->grads_memory, 0, model->num_parameters * sizeof(floatX), main_stream));
    }

    // Extract model dimensions (use size_t to avoid overflow in large models)
    const size_t B = model->batch_size;
    const size_t T = model->seq_len;
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;

    // Extract pointers for convenience
    ParameterTensors params = model->params;
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;

    // ========== STEP 1: Compute Loss and Gradient of Logits ==========
    // The fused classifier combines multiple operations for efficiency:
    // 1. Compute cross-entropy loss for each token
    // 2. Accumulate losses into acts.losses
    // 3. Compute gradient dL/d(logits) and store in acts.output
    NvtxRange classifier_and_loss_range("classifier_and_loss");

    // Loss scaling factor to get mean loss across all tokens and micro-steps
    // We divide by (B * T * grad_accum_steps) so gradients are properly averaged
    const float dloss = 1.0f / (float)(B * T * grad_accum_steps);

    // Copy target token IDs from CPU to GPU
    cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(targets, B*T, V);  // Validate targets while copy is in progress

    // FUSED CLASSIFIER KERNEL:
    // - Forward: Computes softmax and cross-entropy loss
    // - Backward: Computes dL/d(logits) = softmax_probs - one_hot(targets)
    // - Acts.output now contains gradients instead of logits
    fused_classifier(acts.output, acts.losses, dloss, model->targets, B, T, V, Vp, True, main_stream);

    // ========== STEP 2: Initialize Backward Pass ==========
    // Backward pass proceeds in reverse order of forward pass
    // We'll accumulate gradients flowing back through the residual stream

    // Main gradient buffer: holds dL/d(hidden_states) flowing through network
    // Uses scratch_btc buffer to save memory
    floatX* dresidual = (floatX*)model->acts.scratch_btc;
    cudaCheck(cudaMemset(dresidual, 0, B * T * C * sizeof(floatX)));

    // Reuse acts.output as scratch space for backward pass
    // Forward pass no longer needs the logits, backward pass wrote dlogits there
    float*  scratchF = (float*)acts.output;   // For FP32 scratch operations
    floatX* scratchX = (floatX*)acts.output;  // For mixed-precision scratch operations

    // ========== STEP 3: Backpropagate Through Output Projection ==========
    // The gradient chain starts from dL/d(logits) computed by fused_classifier
    // Now we backpropagate through: logits = lnf @ wte^T

    // Compute gradients for final projection (classifier head)
    // - model->acts.scratch_bt4c receives dL/d(lnf)
    // - grads.wte receives dL/d(wte) [accumulated because embeddings are shared]
    // Input gradient: acts.output (dL/d(logits))
    matmul_backward(model->acts.scratch_bt4c, grads.wte, NULL, acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp, main_stream);

    // ========== STEP 4: Backpropagate Through Final LayerNorm ==========
    // Get the input to final LayerNorm (output of last transformer layer)
    floatX* residual = acts.residual3 + (L-1) * B * T * C;

    // Compute gradients for final LayerNorm
    // - dresidual receives dL/d(residual3) - gradient flowing into last layer
    // - grads.lnfw, grads.lnfb receive dL/d(lnfw), dL/d(lnfb) - LayerNorm param gradients
    // Input gradient: model->acts.scratch_bt4c (dL/d(lnf))
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, scratchF, model->acts.scratch_bt4c, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C, main_stream);

    // MEMORY REUSE OPTIMIZATION:
    // The last layer's residual3 buffer is no longer needed, reuse it as scratch space
    floatX* dl_btc = residual;

    // ========== STEP 5: Backpropagate Through Transformer Layers ==========
    // Process layers in reverse order (L-1 down to 0)
    // Each layer: Backprop through feedforward block, then attention block
    for (int l = L-1; l >= 0; l--) {
        NvtxRange layer_range("Layer", l);

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        floatX* l_ln1w = params.ln1w + l * C;
        floatX* l_ln1b = params.ln1b + l * C;
        floatX* l_qkvw = params.qkvw + l * 3*C * C;
        floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_ln2b = params.ln2b + l * C;
        floatX* l_fcw = params.fcw + l * 4*C * C;
        floatX* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        floatX* dl_ln1w = grads.ln1w + l * C;
        floatX* dl_ln1b = grads.ln1b + l * C;
        floatX* dl_qkvw = grads.qkvw + l * 3*C * C;
        floatX* dl_qkvb = grads.qkvb + l * 3*C;
        floatX* dl_attprojw = grads.attprojw + l * C * C;
        floatX* dl_attprojb = grads.attprojb + l * C;
        floatX* dl_ln2w = grads.ln2w + l * C;
        floatX* dl_ln2b = grads.ln2b + l * C;
        floatX* dl_fcw = grads.fcw + l * 4*C * C;
        floatX* dl_fcb = grads.fcb + l * 4*C;
        floatX* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        floatX* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + l * B * T * C : acts.lnf;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;
        floatX* l_atty = acts.atty + l * B * T * C;
        floatX* l_residual2 = acts.residual2 + l * B * T * C;
        floatX* l_ln2 = (model->recompute < 2) ? acts.ln2 + l * B * T * C : acts.lnf;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        floatX* l_fch_pre_gelu = acts.fch + l * B * T * 4*C;
        floatX* l_fch_gelu = (model->recompute < 1) ? acts.fch_gelu + l * B * T * 4*C : acts.fch_gelu;
        // get the pointers of the gradients of the activations for this layer
        // notice that there is no l *, because we just have a single copy, and keep
        // re-using this memory in every Transformer block as we calculate backward pass

        floatX* dl_bt4c = (floatX*)model->acts.scratch_bt4c;

        // start the backward pass for this layer
        if(model->recompute >= 1) {
            // recompute >= 1 means we recompute gelu. in this case,
            // l_fch_gelu is just a buffer, so re-compute the gelu from l_fch here
            gelu_forward(l_fch_gelu, l_fch_pre_gelu, B*T*4*C, main_stream);
        }
        matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, scratchF, B, T, 4*C, C, main_stream, l_fch_pre_gelu, model->gelu_fusion);
        if(model->recompute >= 2) {
            // same as gelu above, l_ln1 and l_ln2 are just buffers if recompute >= 2, recompute them here on demand
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C, main_stream);
        }
        matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, scratchF, B, T, C, 4 * C, main_stream);
        // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
        layernorm_backward(dresidual, dl_ln2w, dl_ln2b, scratchF, dl_btc, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C, main_stream);
        matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, scratchF, B, T, C, C, main_stream);

        #ifdef ENABLE_CUDNN
        float* l_att = (float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
        attention_backward_cudnn(dl_bt4c, dl_btc, l_qkvr, l_atty, (float*)l_att, B, T, NH, C, main_stream);
        #else
        floatX* l_att = acts.att + l * B * NH * T * T;
        // we need B x T x (4)C buffers. l_atty and l_fch aren't needed anymore at this point, so reuse their memory
        floatX* buffer_a = l_atty;
        floatX* buffer_b = l_fch_pre_gelu;        // this is B x T x 4C, so even larger than what we need
        attention_backward(dl_bt4c, buffer_b, scratchX, buffer_a, dl_btc, l_qkvr, l_att, B, T, C, NH, main_stream);
        #endif
        if(model->recompute >= 2) {
            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C, main_stream);
        }
        // QKV parameter gradients
        matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, scratchF, B, T, C, 3 * C, main_stream);
        // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, scratchF, dl_btc, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C, main_stream);

        // Accumulate gradients from this layer in a background stream.
        if(last_step) {
            floatX* const pointers[] = {
                dl_ln1w, dl_ln1b,
                dl_qkvw, dl_qkvb,
                dl_attprojw, dl_attprojb,
                dl_ln2w, dl_ln2b,
                dl_fcw, dl_fcb,
                dl_fcprojw, dl_fcprojb
            };
            const size_t nelem[] = {
                C, C,
                3 * C * C, 3 * C,
                C * C, C,
                C, C,
                4 * C * C, 4 * C,
                C * 4 * C, C
            };
            multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, main_stream);
        }
    }
    encoder_backward(grads.wte, grads.wpe, scratchX, model->workload_indices, model->bucket_info,
                     dresidual, model->inputs, inputs, B, T, C, random_u32(&model->rng_state), main_stream);

    // Aggregate all gradients that are not part of the transformer blocks
    if(last_step) {
        // reduce all the losses within the current GPU (across all microsteps)
        global_sum_deterministic(model->accumulated_mean_loss, acts.losses, B*T, main_stream);
        // reduce loss across GPUs to a single, final float across all microsteps and GPUs
        #if MULTI_GPU
        ncclCheck(ncclAllReduce(model->accumulated_mean_loss, model->accumulated_mean_loss, sizeof(float), ncclFloat, ncclAvg, multi_gpu_config.nccl_comm, main_stream));
        #endif
        cudaCheck(cudaMemcpyAsync(&model->mean_loss, model->accumulated_mean_loss, sizeof(float), cudaMemcpyDeviceToHost, main_stream));
        // reduce the gradients for non-transformer block parameters
        floatX* const pointers[] = {grads.wte, grads.wpe, grads.lnfw, grads.lnfb};
        const size_t nelem[] = {Vp * C, T * C, C, C};
        multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, main_stream);
    }

    cudaCheck(cudaDeviceSynchronize());
    if(last_step) {
        model->mean_loss /= B*T*grad_accum_steps;
    } else {
        model->mean_loss = -1.f; // no loss available yet
    }
}

// Gets the offset of a specific tensor for a specific layer in the GPT2 model
// layer_id is ignored for weights that are not part of a transformer block
ShardInfo gpt2_get_tensor_at_layer(const GPT2 *model, int layer_id, int param_tensor_id) {
    // first offset our way to the parameter tensor start
    ptrdiff_t offset = 0;
    for (int i = 0; i < param_tensor_id; i++) {
        offset += (ptrdiff_t)model->param_elements[i];
    }
    size_t size = model->param_elements[param_tensor_id] ;
    // if we are in the transformer block, we need to additionally offset by the layer id
    if(2 <= param_tensor_id && param_tensor_id <= 13) {
        size /= model->config.num_layers;
        offset += (ptrdiff_t)(layer_id * size);
    }
    return {offset, size};
}

float gpt2_calculate_grad_norm(GPT2 *model, MultiGpuConfig* multi_gpu_config) {
    NVTX_RANGE_FN();
    floatX* grads_memory = (floatX*)model->grads_memory;

    // repurposing this buffer (which isn't needed now) to write grad norm into it
    float* grad_norm_squared = (float*)model->acts.output;
    float grad_norm_squared_cpu = 0.0f;

    int num_slices[2] = {1, model->config.num_layers};
    int max_num_block_sums = get_max_num_block_sums(num_slices, 2);
    if (multi_gpu_config->zero_stage == 1) {
        // because of the ncclReduceScatter() in backward,
        // grads_memory only contains the averaged gradients at the local shards,
        // so we only calculate the grad norm at the grads_memory belonging to the local shards
        for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
            ShardInfo tensor = gpt2_get_tensor_at_layer(model, 0, i);
            ShardInfo shard = multi_gpu_get_shard_offset(tensor.size, multi_gpu_config, 1);
            ptrdiff_t offset = tensor.offset + shard.offset;
            bool is_first_pass = (i == 0);
            if((i < 2 || i > 13)) {
                global_norm_squared(grad_norm_squared, grads_memory + offset, shard.size, 0, 1,
                                    max_num_block_sums, is_first_pass, main_stream);
            } else {
                global_norm_squared(grad_norm_squared, grads_memory + offset, shard.size, tensor.size, model->config.num_layers,
                                    max_num_block_sums, is_first_pass, main_stream);
            }
        }
        global_sum_deterministic(grad_norm_squared, grad_norm_squared, max_num_block_sums, main_stream);
#if MULTI_GPU
        // further sum the (partial) squared norm across all GPUs
        ncclCheck(ncclAllReduce(grad_norm_squared, grad_norm_squared, sizeof(float), ncclFloat, ncclSum, multi_gpu_config->nccl_comm, main_stream));
#endif
    } else {
        // in regular DDP, backward has averaged the gradients across all GPUs
        // so each GPU can compute the squared norm over the whole grad vector, with no added comms needed
        global_norm_squared(grad_norm_squared, grads_memory, model->num_parameters, 0, 1, max_num_block_sums, true, main_stream);
        global_sum_deterministic(grad_norm_squared, grad_norm_squared, max_num_block_sums, main_stream);
    }
    cudaCheck(cudaMemcpy(&grad_norm_squared_cpu, grad_norm_squared, sizeof(float), cudaMemcpyDeviceToHost));
    float grad_norm_cpu = sqrtf(grad_norm_squared_cpu);
    return grad_norm_cpu;
}

// ============================================================================
// Optimizer: AdamW Weight Update
// ============================================================================

/*
 * gpt2_update: Update model parameters using the AdamW optimizer
 *
 * PARAMETERS:
 *   model              - GPT2 model with computed gradients
 *   learning_rate      - Learning rate (alpha), typically 3e-4 to 6e-4 for GPT-2
 *   beta1              - Exponential decay rate for first moment (momentum), typically 0.9
 *   beta2              - Exponential decay rate for second moment (RMSprop), typically 0.95 or 0.999
 *   eps                - Small constant for numerical stability, typically 1e-8
 *   weight_decay       - L2 regularization coefficient, typically 0.1
 *   grad_scale         - Gradient scaling factor (for gradient clipping)
 *   t                  - Current optimization step (1-indexed for bias correction)
 *   multi_gpu_config   - Multi-GPU configuration for ZeRO optimizer sharding
 *   init_from_master_only - If true, only copy master weights to params (skip gradient update)
 *
 * ADAMW ALGORITHM:
 *   AdamW is Adam with decoupled weight decay (better than L2 regularization)
 *   For each parameter theta:
 *     1. m_t = beta1 * m_{t-1} + (1 - beta1) * gradient
 *     2. v_t = beta2 * v_{t-1} + (1 - beta2) * gradient^2
 *     3. m_hat = m_t / (1 - beta1^t)  [bias correction]
 *     4. v_hat = v_t / (1 - beta2^t)  [bias correction]
 *     5. theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta_{t-1})
 *
 * MIXED PRECISION TRAINING:
 *   - Parameters (params): BF16/FP16 for fast computation
 *   - Gradients (grads): BF16/FP16 (computed in same precision as forward/backward)
 *   - Optimizer state (m, v): FP32 for numerical stability
 *   - Master weights (optional): FP32 copy of params for accurate updates
 *
 *   UPDATE FLOW WITH MASTER WEIGHTS:
 *     1. Compute update in FP32: delta = AdamW(grad_fp32, m, v)
 *     2. Apply to master weights: master_weights -= lr * delta
 *     3. Round down to BF16/FP16: params = (floatX) master_weights
 *
 * MULTI-GPU OPTIMIZER SHARDING (ZeRO Stage 1):
 *   - Each GPU only stores optimizer state (m, v, master_weights) for a shard of parameters
 *   - Reduces memory: 4x to 8x less optimizer memory per GPU
 *   - After update, AllGather broadcasts updated params to all GPUs
 *   - Gradients are already reduced/sharded by gpt2_backward_and_reduce
 *
 * SELECTIVE WEIGHT DECAY:
 *   Weight decay is only applied to 2D weight matrices (not biases or LayerNorm params)
 *   Applied to: wte, wpe, qkvw, attprojw, fcw, fcprojw
 *   Not applied to: biases (qkvb, attprojb, fcb, fcprojb) and LayerNorm (ln1w, ln1b, ln2w, ln2b, lnfw, lnfb)
 *
 * STOCHASTIC ROUNDING:
 *   When converting FP32 master weights to BF16/FP16, we use stochastic rounding
 *   instead of round-to-nearest. This prevents systematic rounding bias during training.
 *   Uses model->rng_state for reproducibility.
 */
void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, float grad_scale, int t,
                 MultiGpuConfig* multi_gpu_config, bool init_from_master_only=false) {
    // NVTX profiling marker
    NVTX_RANGE_FN();

    // Validate optimizer state is allocated
    if(model->grads_memory == nullptr || model->m_memory == nullptr || model->v_memory == nullptr) {
        fprintf(stderr, "Need to allocate optimizer state before update");
        exit(EXIT_FAILURE);
    }

    bool init_state = model->init_state;
    if(init_state) {
        model->init_state = false;
        NvtxRange rng("InitOpt");
        cudaCheck(cudaMemset(model->m_memory, 0, multi_gpu_config->shard_num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->v_memory, 0, multi_gpu_config->shard_num_parameters * sizeof(float)));
    }

    // save RNG state at this point so we can round from master weights identically when restoring from a checkpoint
    model->rng_state_last_update = model->rng_state;

    // AdamW update
    // handle adamw for all the transformer blocks
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        // generate a unique seed for each tensor
        unsigned int seed = random_u32(&model->rng_state);

        int num_layers = model->config.num_layers;
        if((i < 2 || i > 13)) {
            num_layers = 1;
        }

        ShardInfo tensor = gpt2_get_tensor_at_layer(model, 0, i);
        ShardInfo shard = multi_gpu_get_shard_offset(tensor.size, multi_gpu_config, 1);
        ptrdiff_t local_offset_full = tensor.offset + shard.offset;
        ptrdiff_t local_offset_partial = tensor.offset / multi_gpu_config->num_processes;

        // we only want to weight decay the 2D tensors and leave all 1D tensors alone
        // in particular this also decays the embedding weights, but this is ok:
        // - the token embeddings are weight shared and participate in the final projection to logits
        // - the position embeddings actively participate at every forward/backward pass
        float wd = (i == 0 || i == 1 || i == 4 || i == 6 || i == 10 || i == 12) ? weight_decay : 0.0f;
        floatX* param_ptr = (floatX*)model->params_memory + local_offset_full;
        floatX* grad_ptr = (floatX*)model->grads_memory + local_offset_full;

        ptrdiff_t opt_state_offset = multi_gpu_config->zero_stage < 1 ?  local_offset_full : local_offset_partial;
        float* m_ptr = model->m_memory + opt_state_offset;
        float* v_ptr = model->v_memory + opt_state_offset;
        float* master_ptr = nullptr;
        if (model->master_weights != nullptr) { master_ptr = model->master_weights + opt_state_offset; }
        if(init_state && model->master_weights != nullptr ) {
            size_t grid_size = CEIL_DIV(shard.size, 512);
            copy_and_cast_kernel<<<dim3(grid_size, num_layers), 512, 0, main_stream>>>(master_ptr, param_ptr, shard.size,
                                                                     shard.size, tensor.size);
            cudaCheck(cudaGetLastError());
        }

        if (init_from_master_only) {
            // when resuming training from a checkpoint with master weights (allows changing precision)
            init_from_master(param_ptr, master_ptr, shard.size, tensor.size, shard.size, num_layers, seed, main_stream);
        } else {
            // ok finally call the kernel to update the weights with AdamW
            adamw_update(param_ptr, master_ptr, grad_ptr,
                        m_ptr, v_ptr,
                        shard.size, tensor.size, tensor.size, shard.size, num_layers,
                        learning_rate,
                        beta1, beta2, t, eps, wd, grad_scale, seed, main_stream);
        }

        if (multi_gpu_config->zero_stage == 1) {
#if MULTI_GPU
            ncclCheck(ncclGroupStart());
            for(int l = 0; l < num_layers; ++l) {
                // gather updated shards of model->params_memory from each process
                ncclCheck(ncclAllGather(param_ptr + l * tensor.size,
                                        (floatX*) model->params_memory + tensor.offset + l * tensor.size,
                                        shard.size, ncclFloatX,
                                        multi_gpu_config->nccl_comm, multi_gpu_config->nccl_stream));
            }
            ncclCheck(ncclGroupEnd());
#endif
        }
    }

    cudaCheck(cudaDeviceSynchronize());
}

float gpt2_estimate_mfu(GPT2 *model, int num_tokens, float dt) {
    /*
    Estimate model flops utilization (MFU)
    ref: Section 2.1 of https://arxiv.org/pdf/2001.08361
    Note: Ideally, the N here would be only the parameters that actually
    participate in matrix multiplications. In this N, we are over-estimating by
    including LayerNorm params, biases, and the position embedding weights,
    but these are very small terms. Also keep in mind that we would want to exclude
    the token embedding weights, but in GPT-2 these are weight shared, so they
    participate in the classifier matmul, so they are correct to be included in N.
    Note 2: The first term (6 * N) in flops_per_token is all weight matmuls, the
    second is the attention matmul, which is also usually a small contribution.
    */
    size_t N = model->num_parameters;
    int L = model->config.num_layers;
    int C = model->config.channels;
    int T = model->seq_len;
    size_t flops_per_token = 6 * N + (size_t)6 * L * C * T;
    size_t flops_per_step = flops_per_token * num_tokens;
    // express our flops throughput as ratio of A100 bfloat16 peak flops
    float flops_achieved = (float)flops_per_step * (1.0f / dt); // per second
    float flops_promised = get_flops_promised(deviceProp.name, PRECISION_MODE) * 1e12f;
    if(flops_promised < 0) {
        return -1.f;   // don't know
    }
    float mfu = flops_achieved / flops_promised;
    return mfu;
}

void gpt2_free(GPT2 *model) {
    cudaFreeCheck(&model->params_memory);
    cudaFreeCheck(&model->grads_memory);
    cudaFreeCheck(&model->m_memory);
    cudaFreeCheck(&model->v_memory);
    cudaFreeCheck(&model->master_weights);
    cudaFreeCheck(&model->acts_memory);
    cudaFreeCheck(&model->inputs);
    cudaFreeCheck(&model->targets);
    cudaFreeCheck(&model->accumulated_mean_loss);
    cudaCheck(cudaFreeHost(model->cpu_losses));
    free(model->workload_indices);
    free(model->bucket_info);
}

// ----------------------------------------------------------------------------
// common init & free code for all of train/test/profile

void common_start(bool override_enable_tf32 = true, bool print_device_info = true) {

    // get CUDA device infos
    cudaCheck(cudaGetDeviceProperties(&deviceProp, multi_gpu_config.local_device_idx));
    if (print_device_info) {
        printf("[System]\n");
        printf("Device %d: %s\n", multi_gpu_config.local_device_idx, deviceProp.name);
    }

    // set up the cuda streams. atm everything is on the single main stream
    cudaCheck(cudaStreamCreate(&main_stream));
    nvtxNameCudaStreamA(main_stream, "main stream");

    // set up cuBLAS and cuBLASLt
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    bool enable_tf32 = PRECISION_MODE == PRECISION_FP32 && deviceProp.major >= 8 && override_enable_tf32;
    cublas_compute = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

    #ifdef ENABLE_CUDNN
    create_cudnn();
    #endif
}

void common_free(GPT2 &model) {
    cudaCheck(cudaStreamDestroy(main_stream));
    cudaCheck(cudaFree(cublaslt_workspace));
    cublasCheck(cublasLtDestroy(cublaslt_handle));
    #ifdef ENABLE_CUDNN
    destroy_cudnn();
    #endif
}


void save_state(const char* filename, int step, GPT2* model, DataLoader* loader) {
    printf("Writing state to %s\n", filename);
    FILE *state_file = fopenCheck(filename, "wb");
    int state_header[256];
    memset(state_header, 0, sizeof(state_header));
    // basic identifying information
    state_header[0] = 20240527; // magic number
    state_header[1] = 1; // version number
    state_header[2] = multi_gpu_config.num_processes; // number of processes
    state_header[3] = multi_gpu_config.process_rank; // rank of this process
    state_header[4] = model->use_master_weights;  // whether we're using fp32 master weights
    state_header[5] = loader->should_shuffle; // shuffle state of the dataloader
    // int main state, start at 10 to leave some padding
    state_header[10] = step; // step of the optimization
    // model rng state, start at 20 to leave some padding
    *((unsigned long long*)&state_header[20]) = model->rng_state; // random number generator state
    *((unsigned long long*)&state_header[22]) = model->rng_state_last_update; // last gpt2_update
    // dataloader state, start at 30 to leave some padding
    *((size_t*)&state_header[30]) = loader->current_shard_idx; // shard of the dataset
    *((size_t*)&state_header[32]) = loader->current_sample_idx; // position in shard
    fwriteCheck(state_header, sizeof(int), 256, state_file);

    // write AdamW m, v, and master_weights here (they are all float)
    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters;
    device_to_file(state_file, model->m_memory, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    device_to_file(state_file, model->v_memory, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    if(model->use_master_weights) {
        device_to_file(state_file, model->master_weights, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    }

    // write dataloader state if we are using the Permuted version of it
    if (loader->should_shuffle) {
        fwriteCheck(&loader->glob_result.gl_pathc, sizeof(size_t), 1, state_file);  // number of shards
        fwriteCheck(loader->shard_indices, sizeof(int), loader->glob_result.gl_pathc, state_file);
        fwriteCheck(&loader->shard_num_samples, sizeof(size_t), 1, state_file);
        fwriteCheck(loader->intra_shard_indices, sizeof(int), loader->shard_num_samples, state_file);
        fwriteCheck(&loader->shuffle_rng, sizeof(mt19937_state), 1, state_file);
    }
    fcloseCheck(state_file);
}

void load_state(int* step, GPT2* model, DataLoader* loader, const char* filename) {
    FILE *state_file = fopenCheck(filename, "rb");
    int state_header[256];
    freadCheck(state_header, sizeof(int), 256, state_file);
    assert(state_header[0] == 20240527); // magic number
    assert(state_header[1] == 1); // version number
    assert(state_header[2] == multi_gpu_config.num_processes); // number of processes
    assert(state_header[3] == multi_gpu_config.process_rank); // rank of this process
    int use_master_weights = state_header[4];  // whether we're using fp32 master weights
    int should_shuffle = state_header[5]; // shuffle state of the dataloader
    *step = state_header[10]; // step of the optimization
    model->rng_state = *((unsigned long long*)&state_header[20]); // random number generator state
    model->rng_state_last_update = *((unsigned long long*)&state_header[22]); // last gpt2_update
    size_t current_shard_idx = *((size_t*)&state_header[30]); // shard index
    size_t current_sample_idx = *((size_t*)&state_header[32]); // position in shard

    // read AdamW m, v, master_weights (they are all float)
    // allocate all the needed memory as necessary
    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters;
    if(use_master_weights == 1 && !model->use_master_weights) {
        printf0("Warning: Master weights are present in state, but not enabled for current run.");
    } else if (use_master_weights == 0 && model->use_master_weights) {
        printf0("Error: Master weights requested, but not present in state file.");
        exit(EXIT_FAILURE);
    }

    model->init_state = false;      // we just got the state from file, no need to do first-touch init
    assert(model->m_memory != nullptr);
    assert(model->v_memory != nullptr);
    file_to_device(model->m_memory, state_file, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    file_to_device(model->v_memory, state_file, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
    if(model->use_master_weights) {
        assert(model->master_weights != nullptr);
        file_to_device(model->master_weights, state_file, shard_num_parameters * sizeof(float), IO_BUF_SIZE, main_stream);
        // restore weights from the master weights using the RNG state before last weight update
        model->rng_state = model->rng_state_last_update;
        gpt2_update(model, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, &multi_gpu_config, /* init_from_master_only*/ true);
        model->rng_state = *((unsigned long long*)&state_header[20]); // use final RNG state from checkpoint after this
    }

    // revive the DataLoader object and its state
    loader->should_shuffle = should_shuffle;
    if (should_shuffle == 1) {
        // ensure the number of shards matches
        size_t glob_result_gl_pathc;
        freadCheck(&glob_result_gl_pathc, sizeof(size_t), 1, state_file);
        assert(glob_result_gl_pathc == loader->glob_result.gl_pathc);
        // read the shard indices
        loader->shard_indices = (int*)mallocCheck(loader->glob_result.gl_pathc * sizeof(int));
        freadCheck(loader->shard_indices, sizeof(int), loader->glob_result.gl_pathc, state_file);
        // ensure the number of samples matches
        size_t shard_num_samples;
        freadCheck(&shard_num_samples, sizeof(size_t), 1, state_file);
        assert(shard_num_samples == loader->shard_num_samples);
        // read the intra-shard indices
        loader->intra_shard_indices = (int*)mallocCheck(loader->shard_num_samples * sizeof(int));
        freadCheck(loader->intra_shard_indices, sizeof(int), loader->shard_num_samples, state_file);
        // read the shuffle rng state
        freadCheck(&loader->shuffle_rng, sizeof(mt19937_state), 1, state_file);
    }
    dataloader_resume(loader, current_shard_idx, current_sample_idx);

    // all done, close state file
    fcloseCheck(state_file);
}

void write_checkpoint(const char* output_log_dir, int step, GPT2* model, DataLoader* train_loader, MultiGpuConfig* multi_gpu_config) {
    // a checkpoint contains: model weights, optimizer/dataloader state, and a DONE file
    printf0("Writing checkpoint at step %d\n", step);
    int rank = multi_gpu_config->process_rank;
    // only rank 0 writes the model file because it is the same across all ranks
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/model_%08d.bin", output_log_dir, step);
        gpt2_write_to_checkpoint(model, filename_buffer);
    }
    // all ranks write their state file
    snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin", output_log_dir, step, rank);
    save_state(filename_buffer, step, model, train_loader);
    // DONE file is a signal that this checkpoint as a whole is complete
    multi_gpu_barrier(multi_gpu_config);
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/DONE_%08d", output_log_dir, step);
        FILE* done_file = fopenCheck(filename_buffer, "w");
        fcloseCheck(done_file);
    }
}

void delete_checkpoint(const char* output_log_dir, int step, MultiGpuConfig* multi_gpu_config) {
    // mirrors write_checkpoint function, cleans up checkpoint from disk
    printf0("Deleting checkpoint at step %d\n", step);
    int rank = multi_gpu_config->process_rank;
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/model_%08d.bin", output_log_dir, step);
        remove(filename_buffer);
    }
    snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin", output_log_dir, step, rank);
    remove(filename_buffer);
    if (rank == 0) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/DONE_%08d", output_log_dir, step);
        remove(filename_buffer);
    }
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.cu), we'll skip everything below this point

// ----------------------------------------------------------------------------
// training resumption logic, very useful when jobs crash once in a while
// the goal is that we can resume optimization from any checkpoint, bit-perfect
// note that "state" refers to things not already saved in the model checkpoint file

// ----------------------------------------------------------------------------
// CLI, poor man's argparse
// (all single letters have been claimed now)

void error_usage() {
    fprintf(stderr, "Usage:   ./train_gpt2cu [options]\n");
    fprintf(stderr, "Options:\n");
    // file system input / output
    fprintf(stderr, "  -i <string> train data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_train.bin)\n");
    fprintf(stderr, "  -j <string> val data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_val.bin)\n");
    fprintf(stderr, "  -e <string> input .bin filename or descriptor, see code comments as docs. (default = gpt2_124M_bf16.bin)\n");
    fprintf(stderr, "  -o <string> output log dir (default = NULL, no logging)\n");
    fprintf(stderr, "  -lg <int>   log gpu info every x steps (default = -1; disabled)\n");
    fprintf(stderr, "  -n <int>    write optimization checkpoints every how many steps? (default 0, don't)\n");
    fprintf(stderr, "  -nk <int>   max number of checkpoints to keep in the directory, removing old ones (0 = disable, default)\n");
    fprintf(stderr, "  -nm <int>   every how many step checkpoints are considered major? major checkpoints never get deleted.\n");
    fprintf(stderr, "  -y <int>    resume optimization found inside output log dir? (0=restart/overwrite, 1=resume/append)\n");
    // token layout for each step of the optimization
    fprintf(stderr, "  -b <int>    (per-GPU, micro) batch size B (default = 4)\n");
    fprintf(stderr, "  -t <int>    sequence length T (default = 1024)\n");
    fprintf(stderr, "  -d <int>    total desired batch size (default = B * T * num_processes, i.e. no grad accumulation\n");
    // workload (number of steps)
    fprintf(stderr, "  -x <int>    max_steps of optimization to run (-1 (default) = disable, run 1 epoch)\n");
    // optimization
    fprintf(stderr, "  -k <string> learning rate scheduler (default = cosine)\n");
    fprintf(stderr, "  -l <float>  learning rate (default = 3e-4f)\n");
    fprintf(stderr, "  -u <int>    learning rate warmup iterations (default = 0, no warmup)\n");
    fprintf(stderr, "  -q <float>  learning rate decay: final fraction, at end of training (default = 1.0 (no decay))\n");
    fprintf(stderr, "  -c <float>  weight decay (default = 0.0f)\n");
    fprintf(stderr, "  -sl <float> outlier stability: skip update if loss goes above this in zscore (0.0f=off)\n");
    fprintf(stderr, "  -sg <float> outlier stability: skip update if grad_norm goes above this in zscore (0.0f=off)\n");
    // evaluation
    fprintf(stderr, "  -v <int>    val_loss_every, how often we evaluate val loss (default = 20)\n");
    fprintf(stderr, "  -m <int>    val_max_steps, up to how many val batches to estimate val loss? (default = 20)\n");
    fprintf(stderr, "  -s <int>    sample_every, how often we inference the model (default = 20)\n");
    fprintf(stderr, "  -g <int>    genT, how many steps of inference we do (default = 64)\n");
    fprintf(stderr, "  -h <int>    hellaswag eval run? (default = 0)\n");
    // debugging
    fprintf(stderr, "  -a <int>    overfit a single batch? 0/1. useful for debugging\n");
    // numerics
    fprintf(stderr, "  -f <int>    enable_tf32 override (default: 1, set to 0 to disable tf32)\n");
    fprintf(stderr, "  -w <int>    keep f32 copy of weights for the optimizer? (default: 1)\n");
    fprintf(stderr, "  -ge <int>   gelu fusion: 0=none, 1=forward, 2=forward+backward (default: 2 for >=SM90, 0 for older GPUs)\n");
    // memory management
    fprintf(stderr, "  -z <int>    zero_stage, Zero Optimization Stage, 0,1,2,3 (default = 0)\n");
    fprintf(stderr, "  -r <int>    recompute: less memory but less speed. (default = 1), 0|1|2 = none,gelu,gelu+ln\n");
    // multi-node settings
    fprintf(stderr, "  -pn <int>    num_processes (default = 1)\n");
    fprintf(stderr, "  -pr <int>    process_rank (default = 0)\n");
    fprintf(stderr, "  -pg <int>    gpus_per_node (default = 8)\n");
    fprintf(stderr, "  -pm <string> nccl_init_method: tcp,fs,mpi (default = mpi)\n");
    fprintf(stderr, "  -ps <string> server_ip - used only when nccl_init_method is tcp (default = -1)\n");
    fprintf(stderr, "  -pp <string> fs_path - used only when nccl_init_method is fs (default = /tmp)\n");
    exit(EXIT_FAILURE);
}

// ----------------------------------------------------------------------------
// main training loop
// ============================================================================
// Main Training Loop
// ============================================================================

/*
 * PROGRAM OVERVIEW:
 * This is the main entry point for GPT-2 training on GPU using CUDA.
 * The program implements a complete training loop with the following features:
 *
 * TRAINING LOOP STRUCTURE:
 *   For each training step:
 *     1. Load batch of data (input tokens and target tokens)
 *     2. Forward pass: Compute predictions and loss
 *     3. Backward pass: Compute gradients via backpropagation
 *     4. Optimizer step: Update weights using AdamW
 *     5. Periodically: Validation, checkpointing, text generation
 *
 * GRADIENT ACCUMULATION:
 *   To simulate larger batch sizes when GPU memory is limited:
 *   - Run multiple forward/backward passes (micro-batches)
 *   - Accumulate gradients across micro-batches
 *   - Update weights once after all micro-batches
 *   - Effective batch size = micro_batch_size * num_gpus * grad_accum_steps
 *
 * MULTI-GPU TRAINING:
 *   Supports data parallelism and ZeRO optimizer sharding:
 *   - Data parallel: Each GPU processes different data, gradients averaged
 *   - ZeRO Stage 1: Optimizer states sharded across GPUs (saves memory)
 *   - Uses NCCL for efficient GPU-to-GPU communication
 *
 * CHECKPOINTING:
 *   Periodically saves training state to disk:
 *   - Model weights (params)
 *   - Optimizer state (m, v, master_weights)
 *   - Training progress (step number, RNG state)
 *   - Can resume training from any checkpoint
 *
 * MIXED PRECISION:
 *   Uses BF16/FP16 for computation, FP32 for optimizer state:
 *   - Faster training (2-3x speedup on modern GPUs)
 *   - Reduced memory usage (2x less for activations and parameters)
 *   - Maintains accuracy with FP32 master weights
 *
 * MEMORY OPTIMIZATIONS:
 *   - Gradient checkpointing: Recompute activations during backward
 *   - Kernel fusion: Combine operations to reduce memory bandwidth
 *   - Activation reuse: Share memory buffers across layers
 *
 * COMMAND LINE ARGUMENTS:
 *   -i: Training data pattern (e.g., "data/train*.bin")
 *   -j: Validation data pattern
 *   -e: Model weights to load (checkpoint or descriptor like "d12" for 12-layer GPT-2)
 *   -o: Output directory for logs and checkpoints
 *   -b: Per-GPU batch size (micro-batch size)
 *   -t: Sequence length (T)
 *   -d: Total batch size across all GPUs and gradient accumulation
 *   -l: Learning rate
 *   -c: Weight decay coefficient
 *   -x: Maximum training steps
 *   -v: Validation frequency (steps)
 *   -r: Recompute mode (0=none, 1=GELU, 2=GELU+LayerNorm)
 *   -w: Use FP32 master weights (1=yes, 0=no)
 *   -z: ZeRO stage (0=disabled, 1=optimizer state sharding)
 *   See error_usage() for complete list of arguments
 */
int main(int argc, char *argv[]) {
    // ========== Command Line Arguments ==========
    // Default values for training configuration
    const char* train_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* val_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* load_filename = "gpt2_124M_bf16.bin"; // BF16 weights of the model
    const char* lr_scheduler_type = "cosine";  // Learning rate schedule: "cosine", "linear", "constant"
    const char* output_log_dir = NULL;
    int checkpoint_every = 0; // write checkpoints every how many steps?
    int checkpoints_keep = 0; // how long checkpoint history do we keep? (in units of checkpoints)
    int major_checkpoint_every = 0; // major checkpoints never get deleted when maintaining history
    int resume = 0; // resume the optimization, if one is found inside output_log_dir?
    int B = 4; // batch size
    int T = 1024; // sequence length max
    int total_batch_size = -1; // will be calculated down below later, if not provided
    float learning_rate = 3e-4f;
    int log_gpu_every = -1;
    int warmup_iterations = 0;
    float final_learning_rate_frac = 1.0f; // final fraction of learning rate, at end of training
    float weight_decay = 0.0f;
    float skip_update_lossz = 0.0f; // skip update if loss goes above this in zscore
    float skip_update_gradz = 0.0f; // skip update if grad_norm goes above this in zscore
    int val_loss_every = 20; // every how many steps do we eval validation loss?
    int val_max_steps = 20; // how many batches max do we eval for validation loss?
    int sample_every = 20; // every how many steps to do inference?
    int genT = 64; // number of steps of inference we will do
    int overfit_single_batch = 0; // useful for debugging, 1 = only load a single data batch once
    int max_steps = -1;
    int override_enable_tf32 = 1;
    int use_master_weights = 1;
    int gelu_fusion = -1; // 0 = none, 1 = forward, 2 = forward+backward (-1 => per-GPU default)
    int recompute = 1; // recompute during backward setting, 0 = none, 1 = recompute gelu
    int zero_stage = 0; // Zero Optimization Stage for Multi-GPU training
    int hellaswag_eval = 0;
    // multi-node settings
    int num_processes = 1;  // this should be set by the slurm environment
    int process_rank = 0;  // this should be set by the slurm environment
    int gpus_per_node = 8;  // this should be set by the slurm environment
    char nccl_init_method[256] = "mpi";  // "tcp" or "fs" or "mpi"
    char server_ip[256] = "";  // used if init_method set to "tcp" -> set to your server ip address
    char fs_path[256] = "";  // used if init_method set to "fs" -> set to a shared filesystem path
    for (int i = 1; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (!(strlen(argv[i]) == 2 || strlen(argv[i]) == 3)) { error_usage(); } // must be -x[y] (one dash, one or two letters)
        // read in the args
        if (argv[i][1] == 'i') { train_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'j') { val_data_pattern = argv[i+1]; }
        else if (argv[i][1] == 'e') { load_filename = argv[i+1]; }
        else if (argv[i][1] == 'o') { output_log_dir = argv[i+1]; }
        else if (argv[i][1] == 'n' && argv[i][2] == '\0') { checkpoint_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'y') { resume = atoi(argv[i+1]); }
        else if (argv[i][1] == 'b') { B = atoi(argv[i+1]); } // Per-GPU (micro) batch size
        else if (argv[i][1] == 't') { T = atoi(argv[i+1]); }
        else if (argv[i][1] == 'd') { total_batch_size = atoi(argv[i+1]); }
        else if (argv[i][1] == 'l' && argv[i][2] == '\0') { learning_rate = atof(argv[i+1]); }
        else if (argv[i][1] == 'l' && argv[i][2] == 'g') { log_gpu_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'u') { warmup_iterations = atoi(argv[i+1]); }
        else if (argv[i][1] == 'q') { final_learning_rate_frac = atof(argv[i+1]); }
        else if (argv[i][1] == 'c') { weight_decay = atof(argv[i+1]); }
        else if (argv[i][1] == 'x') { max_steps = atoi(argv[i+1]); }
        else if (argv[i][1] == 'v') { val_loss_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'm') { val_max_steps = atoi(argv[i+1]); }
        else if (argv[i][1] == 's' && argv[i][2] == '\0') { sample_every = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g' && argv[i][2] == 'e') { gelu_fusion = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g') { genT = atoi(argv[i+1]); }
        else if (argv[i][1] == 'a') { overfit_single_batch = atoi(argv[i+1]); }
        else if (argv[i][1] == 'f') { override_enable_tf32 = atoi(argv[i+1]); }
        else if (argv[i][1] == 'w') { use_master_weights = atoi(argv[i+1]); }
        else if (argv[i][1] == 'z') { zero_stage = atoi(argv[i+1]); }
        else if (argv[i][1] == 'r') { recompute = atoi(argv[i+1]); }
        else if (argv[i][1] == 'h') { hellaswag_eval = atoi(argv[i+1]); }
        else if (argv[i][1] == 'k') { lr_scheduler_type = argv[i+1]; }
        else if (argv[i][1] == 'p' && argv[i][2] == 'i') { strcpy(nccl_init_method, argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'f') { strcpy(fs_path, argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 's') { strcpy(server_ip, argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'n') { num_processes = atoi(argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'r') { process_rank = atoi(argv[i+1]); }
        else if (argv[i][1] == 'p' && argv[i][2] == 'g') { gpus_per_node = atoi(argv[i+1]); }
        else if (argv[i][1] == 's' && argv[i][2] == 'l') { skip_update_lossz = atof(argv[i+1]); }
        else if (argv[i][1] == 's' && argv[i][2] == 'g') { skip_update_gradz = atof(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'k') { checkpoints_keep = atoi(argv[i+1]); }
        else if (argv[i][1] == 'n' && argv[i][2] == 'm') { major_checkpoint_every = atoi(argv[i+1]); }
        else { error_usage(); }
    }

    multi_gpu_config = multi_gpu_config_init(num_processes, process_rank, gpus_per_node, server_ip, fs_path, nccl_init_method);
    common_start(override_enable_tf32, false); // common init code for train/test/profile

    // should do a bit more error checking here
    assert(warmup_iterations >= 0);
    if (output_log_dir != NULL) {
        assert(strlen(output_log_dir) < 400); // careful bunch of hardcoded snprintf around this
    }
    int tokens_per_fwdbwd = B * T * multi_gpu_config.num_processes; // one micro-batch processes this many tokens
    // calculate sensible default for total batch size as assuming no gradient accumulation
    if (total_batch_size == -1) { total_batch_size = tokens_per_fwdbwd; }
    // in the future, we might want to set gelu fusion to 2 for SM90+ and 0 for other GPUs
    if (gelu_fusion == -1) { gelu_fusion = 0; } // (deviceProp.major >= 9) ? 2 : 0; } // in gpt2_init_common for test_gpt2cu...
    // calculate the number of gradient accumulation steps from the desired total batch size
    assert(total_batch_size % tokens_per_fwdbwd == 0);
    int grad_accum_steps = total_batch_size / tokens_per_fwdbwd;
    // if we're only overfitting a single batch for debugging, let's overfit the first batch
    // from val instead of train split, because val is smaller and faster. (train_gpt2.py does the same)
    if (overfit_single_batch == 1) { train_data_pattern = val_data_pattern; }
    printf0("+-----------------------+----------------------------------------------------+\n");
    printf0("| Parameter             | Value                                              |\n");
    printf0("+-----------------------+----------------------------------------------------+\n");
    printf0("| train data pattern    | %-50s |\n", train_data_pattern);
    printf0("| val data pattern      | %-50s |\n", val_data_pattern);
    printf0("| output log dir        | %-50s |\n", output_log_dir == NULL ? "NULL" : output_log_dir);
    printf0("| checkpoint_every      | %-50d |\n", checkpoint_every);
    printf0("| resume                | %-50d |\n", resume);
    printf0("| micro batch size B    | %-50d |\n", B);
    printf0("| sequence length T     | %-50d |\n", T);
    printf0("| total batch size      | %-50d |\n", total_batch_size);
    printf0("| LR scheduler          | %-50s |\n", lr_scheduler_type);
    printf0("| learning rate (LR)    | %-50e |\n", learning_rate);
    printf0("| warmup iterations     | %-50d |\n", warmup_iterations);
    printf0("| final LR fraction     | %-50e |\n", final_learning_rate_frac);
    printf0("| weight decay          | %-50e |\n", weight_decay);
    printf0("| skip update lossz     | %-50f |\n", skip_update_lossz);
    printf0("| skip update gradz     | %-50f |\n", skip_update_gradz);
    printf0("| max_steps             | %-50d |\n", max_steps);
    printf0("| val_loss_every        | %-50d |\n", val_loss_every);
    printf0("| val_max_steps         | %-50d |\n", val_max_steps);
    printf0("| sample_every          | %-50d |\n", sample_every);
    printf0("| genT                  | %-50d |\n", genT);
    printf0("| overfit_single_batch  | %-50d |\n", overfit_single_batch);
    printf0("| use_master_weights    | %-50s |\n", use_master_weights ? "enabled" : "disabled");
    printf0("| gelu_fusion           | %-50d |\n", gelu_fusion);
    printf0("| recompute             | %-50d |\n", recompute);
    printf0("+-----------------------+----------------------------------------------------+\n");
    const char* precision_str = (PRECISION_MODE == PRECISION_FP32)
                              ? (cublas_compute == CUBLAS_COMPUTE_32F_FAST_TF32 ? "TF32" : "FP32")
                              : (PRECISION_MODE == PRECISION_FP16 ? "FP16" : "BF16");
    printf0("| device                | %-50s |\n", deviceProp.name);
    printf0("| peak TFlops           | %-50.1f |\n", get_flops_promised(deviceProp.name, PRECISION_MODE));
    printf0("| precision             | %-50s |\n", precision_str);
    printf0("+-----------------------+----------------------------------------------------+\n");

    // figure out if we are going to be resuming the optimization
    int resuming = 0;
    // find the DONE file with the highest step count
    int resume_max_step = find_max_step(output_log_dir);
    if (resume == 1) { // is -y 1 resume flag set?
        assert(output_log_dir != NULL);
        if (resume_max_step != -1) {
            resuming = 1; // -y 1 is set, and we found a checkpoint we can resume from
            snprintf(filename_buffer, sizeof(filename_buffer), "%s/model_%08d.bin", output_log_dir, resume_max_step);
        }
    }

    // build the GPT-2 model
    GPT2 model;
    gpt2_init_common(&model);
    if (resuming == 1) {
        // if `-y 1` was set, then we are resuming from the latest checkpoint
        // if we are using master weights, we'll init them later inside load_state()
        bool weight_init = !use_master_weights;
        gpt2_build_from_checkpoint(&model, filename_buffer, weight_init);
    } else if (ends_with_bin(load_filename)) {
        // otherwise, if this is a .bin file, we assume it's a model, let's init from it
        gpt2_build_from_checkpoint(&model, load_filename);
    } else {
        // if it's not .bin, it could be a "special descriptor". This descriptor is used to
        // construct GPT-2 / GPT-3 models in a convenient format. See the function for docs.
        gpt_build_from_descriptor(&model, load_filename);
    }

    model.use_master_weights = use_master_weights;
    model.gelu_fusion = gelu_fusion;
    model.recompute = recompute;
    printf0("| weight init method    | %-50s |\n", resuming == 1 ? "intermediate checkpoint" : load_filename);
    printf0("| max_sequence_length T | %-50d |\n", model.config.max_seq_len);
    printf0("| vocab_size V          | %-50d |\n", model.config.vocab_size);
    printf0("| padded_vocab_size Vp  | %-50d |\n", model.config.padded_vocab_size);
    printf0("| num_layers L          | %-50d |\n", model.config.num_layers);
    printf0("| num_heads NH          | %-50d |\n", model.config.num_heads);
    printf0("| channels C            | %-50d |\n", model.config.channels);
    printf0("| num_parameters        | %-50zu |\n", model.num_parameters);
    printf0("+-----------------------+----------------------------------------------------+\n");

    // build DataLoaders for both train and val
    int permute_train_loader = (overfit_single_batch == 1) ? 0 : 1;
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_data_pattern, B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes, permute_train_loader);
    dataloader_init(&val_loader, val_data_pattern, B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes, 0);
    // figure out the number of training steps we will run for
    int train_num_batches = max_steps; // passed in from command line
    if (train_num_batches == -1) {
        // sensible default is to train for exactly one epoch
        size_t ntok = train_loader.num_tokens;
        // the number of (outer loop) steps each process should take for us to reach one epoch
        train_num_batches = ntok / total_batch_size;
    }
    // figure out the number of validation steps to run for
    int val_num_batches = val_max_steps; // passed in from command line
    if (val_num_batches == -1) {
        // sensible default is to evaluate the full validation split
        size_t ntok = val_loader.num_tokens;
        // note that unlike the training loop, there is no gradient accumulation inner loop here
        val_num_batches = ntok / tokens_per_fwdbwd;
    }
    printf0("| train_num_batches     | %-50d |\n", train_num_batches);
    printf0("| val_num_batches       | %-50d |\n", val_num_batches);
    printf0("+-----------------------+----------------------------------------------------+\n");

    // build an EvalLoader for HellaSwag
    EvalLoader eval_loader;
    const char* hellaswag_path = "dev/data/hellaswag/hellaswag_val.bin";
    const bool hellaswag_available = access(hellaswag_path, F_OK) == 0;
    const bool run_hellaswag = hellaswag_eval && hellaswag_available;
    if (run_hellaswag) {
        evalloader_init(&eval_loader, hellaswag_path, B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes);
    }
    printf0("| run hellaswag         | %-50s |\n", run_hellaswag ? "yes" : "no");
    printf0("+-----------------------+----------------------------------------------------+\n");

    // pretty print in a table the multi-gpu configuration as well
    set_zero_configs(&multi_gpu_config, zero_stage, model.num_parameters);
    printf0("| num_processes         | %-50d |\n", multi_gpu_config.num_processes);
    printf0("| zero_stage            | %-50d |\n", multi_gpu_config.zero_stage);
    printf0("+-----------------------+----------------------------------------------------+\n");

    // prints outside of pretty table to here and below
    if (!hellaswag_available) {
        printf0("HellaSwag eval not found at %s, skipping its evaluation\n", hellaswag_path);
        printf0("You can run `python dev/data/hellaswag.py` to export and use it with `-h 1`.\n");
    }
    // more prints related to allocations from gpt2_build_from_checkpoint down here to not mess up our table above
    printf0("num_parameters: %zu => bytes: %zu\n", model.num_parameters, model.num_parameters_bytes);
    printf0("allocated %d MiB for model parameters\n", (int)round(model.num_parameters_bytes / (1024 * 1024)));
    // few more prints for gradient accumulation math up above
    printf0("batch_size B=%d * seq_len T=%d * num_processes=%d and total_batch_size=%d\n",
            B, T, multi_gpu_config.num_processes, total_batch_size);
    printf0("=> setting grad_accum_steps=%d\n", grad_accum_steps);

    // set up logging
    if (multi_gpu_config.process_rank == 0) { create_dir_if_not_exists(output_log_dir); }
    Logger logger;
    logger_init(&logger, output_log_dir, multi_gpu_config.process_rank, resume);

    // set up the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // set up learning rate scheduler
    LearningRateScheduler lr_scheduler;
    lr_scheduler_init(&lr_scheduler, lr_scheduler_type, learning_rate,
                      warmup_iterations, train_num_batches, final_learning_rate_frac);

    // some memory for generating samples from the model
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    floatX* cpu_logits_raw = (floatX*)mallocCheck(model.config.vocab_size * sizeof(floatX));
    float*  cpu_logits = (float*)mallocCheck(model.config.vocab_size * sizeof(float));

    // if we found a checkpoint to resume from, load the optimization state
    int step = 0;
    gpt2_allocate_state(&model, B, T);
    if (resuming == 1) {
        snprintf(filename_buffer, sizeof(filename_buffer), "%s/state_%08d_%05d.bin", output_log_dir, resume_max_step, multi_gpu_config.process_rank);
        load_state(&step, &model, &train_loader, filename_buffer);
    }

    // init an OutlierDetector the training loss
    OutlierDetector loss_outlier_detector, grad_norm_outlier_detector;
    init_detector(&loss_outlier_detector);
    init_detector(&grad_norm_outlier_detector);

    // do some checks here before we kick off training
    // cross-check the desired sequence length T with the model's max sequence length
    if (T < model.config.max_seq_len) {
        printf0("!!!!!!!!\n");
        printf0("WARNING:\n");
        printf0("- The training sequence length is: T=%d (set with -t)\n", T);
        printf0("- The model's max sequence length is: max_seq_len=%d\n", model.config.max_seq_len);
        printf0("You are attempting to train with a sequence length shorter than the model's max.\n");
        printf0("This will lead to unused parameters in the wpe position embedding weights.\n");
        printf0("If you know what you're doing you can ignore this warning.\n");
        printf0("If you're like ???, you are most likely misconfiguring your training run.\n");
        printf0("---> HINT: If you're training GPT-2 use -t 1024. If GPT-3, use -t 2048.\n");
        printf0("!!!!!!!!\n");
    }
    // in any case, this must be true or we'd index beyond the model's wpe (position embedding table)
    assert(T <= model.config.max_seq_len);

    // train
    cudaEvent_t start, end;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&end));
    cudaCheck(cudaProfilerStart());
    double total_sum_iteration_time_s = 0.0;
    float ema_tokens_per_second = 0.0f;
    for (; step <= train_num_batches; step++) {
        NvtxRange step_range("Train step", step);

        int last_step = step == train_num_batches;

        // once in a while estimate the validation loss (all processes collaborate)
        if (step % val_loss_every == 0 || last_step) {
            NvtxRange validation_range("validation");
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                val_loss += gpt2_validate(&model, val_loader.inputs, val_loader.targets, B, T);
            }
            val_loss /= val_num_batches;
            val_loss = multi_gpu_cpu_float_sum(val_loss, &multi_gpu_config) / multi_gpu_config.num_processes;
            printf0("val loss %f\n", val_loss);
            logger_log_val(&logger, step, val_loss);
        }

        // once in a while estimate HellaSwag accuracy (all processes collaborate)
        if (run_hellaswag &&
           ((step > 0 && step % val_loss_every == 0) || last_step)) {
            NvtxRange evaluation_range("evaluation");
            float eval_acc_norm = 0.0f;
            evalloader_reset(&eval_loader);
            for (int i = 0; i < eval_loader.num_batches; i++) {
                if (i % 10 == 0) { printf("evaluating HellaSwag: %d/%d\r", i, eval_loader.num_batches); }
                evalloader_next_batch(&eval_loader);
                gpt2_validate(&model, eval_loader.inputs, eval_loader.targets, B, T);
                int correct = evalloader_stat_losses(&eval_loader, model.cpu_losses);
                eval_acc_norm += (float)correct;
            }
            // careful because not all ranks may have the exact same allocation of number of examples
            eval_acc_norm = multi_gpu_cpu_float_sum(eval_acc_norm, &multi_gpu_config);
            printf0("HellaSwag: %d/%d = %f\n", (int)eval_acc_norm, eval_loader.num_examples, eval_acc_norm / eval_loader.num_examples);
            logger_log_eval(&logger, step, eval_acc_norm / eval_loader.num_examples);
        }

        // once in a while do model inference to print generated text (only rank 0)
        if (multi_gpu_config.process_rank == 0 && sample_every > 0 &&
           (step > 0 && (step % sample_every) == 0 || last_step)) {
            NvtxRange generation_range("generation");
            unsigned long long sample_rng_state = 1337;
            // fill up gen_tokens with the <|endoftext|> token, which kicks off the generation
            int eot_token = tokenizer.eot_token;
            for(int i = 0; i < B * T; ++i) {
                gen_tokens[i] = eot_token;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++) {
                NvtxRange generation_range("Generation step", t);
                // we try not to be too wasteful for inference by not calculating all of B,T
                // Using a smaller B is always bit-for-bit identical, but T is more tricky
                // for non-CUDNN, we need to make sure the attention buffer is memset to 0
                // for cuDNN, it might suddenly decide to use a slightly different algorithm...
                // on cuDNN 9.2.1 with cuDNN FrontEnd 1.5.2, T >= 256 seems bit-for-bit identical
                // (but even if it wasn't fully identical that's probably not the end of the world)
                // note this is still somewhat wasteful because we don't have a KV cache!
                gpt2_forward(&model, gen_tokens, 1, CEIL_DIV(t, min(T,256)) * min(T,256));
                // get the V-dimensional vector probs[0, t-1, :]
                floatX* logits = model.acts.output + (t - 1) * model.config.padded_vocab_size;
                // move probs back to CPU and sample (note we only move the first vocab_size logits, ignoring the padding)
                cudaCheck(cudaMemcpy(cpu_logits_raw, logits, model.config.vocab_size * sizeof(floatX), cudaMemcpyDeviceToHost));
                // convert to FP32 into cpu_logits (this does nothing useful if floatX == float)
                for (int i = 0; i < model.config.vocab_size; i++) {
                    cpu_logits[i] = (float)cpu_logits_raw[i];
                }
                // sample the next token
                float coin = random_f32(&sample_rng_state);
                int next_token = sample_softmax(cpu_logits, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // once in a while checkpoint the optimization state (all ranks)
        if ((checkpoint_every > 0 && output_log_dir != NULL && resuming == 0) &&
            ((step > 0 && step % checkpoint_every == 0) || last_step)) {
            // writes model .bin file, state .bin files, and DONE file for step
            write_checkpoint(output_log_dir, step, &model, &train_loader, &multi_gpu_config);
            // we only keep checkpoints_keep checkpoints on disk to save space
            // so now that we wrote a new checkpoint, delete one old one (unless it is a "major" checkpoint)
            // we only do this is checkpoint keeping is turned on (checkpoints_keep > 0)
            int step_delete = step - checkpoints_keep * checkpoint_every;
            if (checkpoints_keep > 0 && step_delete > 0 &&
               (major_checkpoint_every == 0 || step_delete % major_checkpoint_every != 0)
                ) {
                delete_checkpoint(output_log_dir, step_delete, &multi_gpu_config);
            }
        }
        resuming = 0;

        // bit confusing: we want to make sure to eval and sample on 0th iteration
        // but also after the very last iteration. so we loop for step <= train_num_batches
        // instead of just < train_num_batches (one extra due to <=), only to do
        // the validation/sampling one last time, and then we break right here as we're done.
        if (last_step) { break; }

        // --------------- TRAINING SECTION BEGIN -----------------
        if (overfit_single_batch == 1) {
            // if we are trying to overfit a single batch, we reset the loader here
            dataloader_reset(&train_loader);
        }
        // do one training step, doing forward/backward/update on total_batch_size tokens
        cudaCheck(cudaEventRecord(start));
        // gradient and loss accumulation loop over micro-batches
        for (int micro_step = 0; micro_step < grad_accum_steps; micro_step++) {
            // fetch the next data batch
            dataloader_next_batch(&train_loader);
            // forward pass. note that we pass in grad_accum_steps, which scales down the loss
            gpt2_forward(&model, train_loader.inputs, B, T);
            // backward pass. all model params accumulate gradients with += inside this inner loop
            gpt2_backward_and_reduce(&model, train_loader.inputs, train_loader.targets, grad_accum_steps, micro_step);
        }
        float zloss = (float)(update_detector(&loss_outlier_detector, (double)model.mean_loss)); // loss z-score
        // fetch the next learning rate
        float step_learning_rate = get_learning_rate(&lr_scheduler, step);
        // calculate the gradient norm and how much we wish to scale the gradient
        float grad_norm = gpt2_calculate_grad_norm(&model, &multi_gpu_config);
        float zgrad = (float)(update_detector(&grad_norm_outlier_detector, (double)grad_norm)); // grad z-score
        // update the model parameters
        if (isfinite(zloss) && skip_update_lossz != 0.0f && zloss > skip_update_lossz) {
            printf0("skipping update due to loss z-score of %f\n", zloss);
        } else if (isfinite(zgrad) && skip_update_gradz != 0.0f && zgrad > skip_update_gradz) {
            printf0("skipping update due to grad z-score of %f\n", zgrad);
        } else {
            // clip the gradient norm to a maximum value
            float grad_clip = 1.0f;
            float grad_scale = (grad_norm > grad_clip) ? grad_clip / grad_norm : 1.0f;
            gpt2_update(&model, step_learning_rate, 0.9f, 0.95f, 1e-8f, weight_decay, grad_scale, step+1, &multi_gpu_config);
        }
        cudaCheck(cudaEventRecord(end));
        cudaCheck(cudaEventSynchronize(end)); // wait for the end event to finish to get correct timings
        // --------------- TRAINING SECTION END -------------------
        // everything that follows now is just diagnostics, prints, logging, etc.

        // todo - move or double-buffer all of this timing logic to avoid idling the GPU at this point!
        float time_elapsed_ms;
        cudaCheck(cudaEventElapsedTime(&time_elapsed_ms, start, end));
        size_t tokens_processed = (size_t)multi_gpu_config.num_processes * B * T * grad_accum_steps;
        float tokens_per_second = tokens_processed / time_elapsed_ms * 1000.0f;
        float bias_corrected_ema_tokens_per_second = tokens_per_second; // by default set to non-ema version
        if (step > 0) { // consider the first batch to be a warmup (e.g. cuBLAS/cuDNN initialisation)
            total_sum_iteration_time_s += time_elapsed_ms / 1000.0f;
            // smooth out the tok/s with an exponential moving average, and bias correct just like in AdamW
            ema_tokens_per_second = 0.95f * ema_tokens_per_second + 0.05f * tokens_per_second;
            bias_corrected_ema_tokens_per_second = ema_tokens_per_second / (1.0f - powf(0.95f, step));
        }
        float mfu = gpt2_estimate_mfu(&model, B * T * grad_accum_steps, time_elapsed_ms / 1000.0f);
        printf0("step %4d/%d | loss %7.6f (%+.2fz)| norm %6.4f (%+.2fz)| lr %.2e | %.2f ms | %.1f%% bf16 MFU | %.0f tok/s\n",
                step + 1, train_num_batches, model.mean_loss, zloss, grad_norm, zgrad, step_learning_rate,
                time_elapsed_ms, 100*mfu, bias_corrected_ema_tokens_per_second);
        if(log_gpu_every > 0 && (step + 1) % log_gpu_every == 0) {
            GPUUtilInfo gpu_info = get_gpu_utilization_info();
            printf0("                  compute %2.1f%% | memory: %2.1f%% | fan: %2d%% | %4d MHz / %4d MHz | %3d W / %3d W | %d°C / %d°C | %s\n",
                    gpu_info.gpu_utilization, gpu_info.mem_utilization, gpu_info.fan, gpu_info.clock, gpu_info.max_clock, gpu_info.power / 1000, gpu_info.power_limit / 1000,
                    gpu_info.temperature, gpu_info.temp_slowdown, gpu_info.throttle_reason);
        }
        logger_log_train(&logger, step, model.mean_loss, step_learning_rate, grad_norm);

        // disable the profiler after 3 steps of optimization
        if (step == 3) { cudaProfilerStop(); }
    }
    // add a total average, for optimizations that are only mild improvements (excluding 1st batch as warmup)
    printf0("total average iteration time: %f ms\n", total_sum_iteration_time_s / (train_num_batches-1) * 1000);

    // free and destroy everything
    cudaCheck(cudaEventDestroy(end));
    cudaCheck(cudaEventDestroy(start));
    if (run_hellaswag) { evalloader_free(&eval_loader); }
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    free(cpu_logits_raw);
    free(cpu_logits);
    free(gen_tokens);
    multi_gpu_config_free(&multi_gpu_config);
    gpt2_free(&model);
    common_free(model);
    return 0;
}
#endif
