/*
GPT-2 CUDA/GPU Testing Suite
=============================

PURPOSE:
This file tests the CUDA/GPU implementation of GPT-2 training (train_gpt2.cu) by validating
correctness across different precision modes (FP32, BF16, FP16) and testing determinism.

WHAT IS BEING TESTED:
1. Forward pass correctness - logits must match PyTorch output
2. Backward pass correctness - all parameter gradients must match PyTorch (with precision-aware tolerances)
3. Training loop correctness - loss values over 10 steps must match PyTorch
4. Determinism - saving/loading state must produce identical results

KEY DIFFERENCES FROM CPU TEST (test_gpt2.c):
- Supports mixed precision (BF16/FP16) with relaxed tolerances
- Tests GPU-specific features (CUDA kernels, cuDNN, etc.)
- Includes checkpoint save/load determinism test
- More sophisticated error checking with relative error metrics
- Handles GPU<->CPU memory transfers for validation

HOW IT WORKS:
- Loads GPT-2 124M model (FP32 or BF16 version depending on compilation flags)
- Loads reference data from Python/PyTorch (gpt2_124M_debug_state.bin)
- Runs forward/backward passes on GPU
- Copies results back to CPU for comparison with reference values
- Uses precision-aware tolerances (looser for BF16/FP16, tight for FP32)
- Tests determinism by saving state, running steps, reloading, and verifying identical results

HOW TO INTERPRET RESULTS:
- "OK" status: Values match within tolerance - implementation is correct for that precision
- "NOT OK" status: Values differ beyond tolerance - indicates a potential bug
- "max diff": Maximum absolute difference found
- "rel error": Relative error (helps understand if large differences are proportionally small)
- "X% of maximum error": How close we are to the tolerance threshold (100% = at limit)
- For BF16/FP16: Higher tolerances are expected due to reduced precision
- Final "overall okay: 1" = PASS, "0" = FAIL

RUNNING THE TEST:
1. Generate reference data: python train_gpt2.py
2. Compile for FP32: make test_gpt2cu
   Or for BF16: make test_gpt2cu PRECISION=BF16
   Or for FP16: make test_gpt2cu PRECISION=FP16
3. Run: ./test_gpt2cu [options]
   Options: -w 0/1 (use master weights), -r 0/1 (recompute), -ge 0/1 (gelu fusion)
4. Check final output: "overall okay: 1" = PASS
*/

#define TESTING
#include "train_gpt2.cu"

// ----------------------------------------------------------------------------
// Tensor Validation Helper Function (GPU Version)
// ----------------------------------------------------------------------------

/*
check_tensor: Advanced tensor comparison with precision-aware error checking

This function is more sophisticated than the CPU version because it handles
mixed precision scenarios where exact matches are impossible.

PARAMETERS:
- a: Calculated tensor (from our CUDA implementation)
- b: Reference tensor (from PyTorch, always FP32)
- n: Number of elements in the tensors
- label: Name of the tensor being checked (for debug output)
- threshold: Base absolute error tolerance (default 1.0, but usually set per-tensor)

RETURNS:
- 1 if tensors match within tolerance
- 0 if any element exceeds tolerance

TOLERANCE CALCULATION:
For each element, the effective tolerance is:
  t_eff = threshold + |b[i]| * epsilon
Where epsilon = 0.079 for BF16 (representing its precision limits)

This allows larger absolute errors for larger values, which is appropriate
for reduced precision formats like BF16.
*/
int check_tensor(float *a, float *b, int n, const char* label, float threshold=1e-0) {
    // a is the calculated tensor, b is the reference tensor
    int print_upto = 10;        // Print first 10 elements for visual inspection
    int ok = 1;                  // Overall pass/fail flag
    float max_diff = 0.0f;       // Maximum absolute difference found
    float max_rel_error = 0.0f;  // Relative error at the point of max difference
    float max_to_threshold = 0.f; // Worst-case ratio of error to tolerance
    float max_a = 0.0f;          // Value from 'a' at max diff location
    float max_b = 0.0f;          // Value from 'b' at max diff location
    float epsilon = 0.079;       // BF16 precision limit (~1/13, derived from mantissa bits)

    printf("---\n");
    printf("checking tensor: %s\n", label);

    // Check each element
    for (int i = 0; i < n; i++) {
        // Calculate effective tolerance for this element (scales with magnitude)
        float t_eff = threshold + fabs(b[i]) * epsilon;

        // Calculate absolute difference
        float diff = fabsf(a[i] - b[i]);

        // Track worst-case error relative to threshold
        max_to_threshold = max(max_to_threshold, diff / t_eff);

        // If this is the largest difference, record details for reporting
        if (diff > max_diff) {
            max_diff = diff;
            float denom = fabsf(b[i]);
            max_rel_error = (denom == 0.0f) ? 0.0f : diff / denom;  // Avoid division by zero
            max_a = a[i];
            max_b = b[i];
        }

        // Check if this element exceeds tolerance
        if (diff > t_eff) {
            ok = 0;
        }

        // Print first few elements for manual inspection
        if (i < print_upto) {
            printf(diff <= t_eff ? "OK " :  "NOT OK ");
            printf("%f %f\n", a[i], b[i]);
        }
    }

    // Print summary results
    // The percentage tells us how much of the allowed error budget we used
    // 100% = exactly at tolerance limit, >100% = failure
    if (ok) {
        printf("TENSOR OK, max diff: %.3e, with rel error: %.3e (calculated=%10f, ref=%10f), %.2f%% of maximum error\n",
                max_diff, max_rel_error, max_a, max_b, max_to_threshold*100);
    } else {
        printf("TENSOR NOT OK, max diff: %.3e, with rel error: %.3e (calculated=%10f, ref=%10f), %.2f%% of maximum error\n",
                max_diff, max_rel_error, max_a, max_b, max_to_threshold*100);
    }
    return ok;
}

// ----------------------------------------------------------------------------
// Reference Tensors (FP32 on CPU)
// ----------------------------------------------------------------------------

/*
FloatParameterTensors: CPU-side FP32 reference data structure

This mirrors the ParameterTensors structure used in training, but:
1. Always uses float (FP32) regardless of GPU precision mode
2. Stored on CPU (not GPU) for easy comparison
3. Contains reference values from PyTorch

These are the "ground truth" values we compare against.
The GPU may use BF16 or FP16, but we convert to FP32 for comparison.
*/
typedef struct {
    float*  wte;      // (Vp, C)    - Token embeddings
    float*  wpe;      // (maxT, C)  - Position embeddings
    float*  ln1w;     // (L, C)     - Layer norm 1 weights
    float*  ln1b;     // (L, C)     - Layer norm 1 biases
    float*  qkvw;     // (L, 3*C, C) - Attention QKV weights
    float*  qkvb;     // (L, 3*C)   - Attention QKV biases
    float*  attprojw; // (L, C, C)  - Attention projection weights
    float*  attprojb; // (L, C)     - Attention projection biases
    float*  ln2w;     // (L, C)     - Layer norm 2 weights
    float*  ln2b;     // (L, C)     - Layer norm 2 biases
    float*  fcw;      // (L, 4*C, C) - MLP first layer weights
    float*  fcb;      // (L, 4*C)   - MLP first layer biases
    float*  fcprojw;  // (L, C, 4*C) - MLP projection weights
    float*  fcprojb;  // (L, C)     - MLP projection biases
    float*  lnfw;     // (C)        - Final layer norm weights
    float*  lnfb;     // (C)        - Final layer norm biases
} FloatParameterTensors;
static_assert(sizeof(FloatParameterTensors) == NUM_PARAMETER_TENSORS * sizeof(void*), "Inconsistent sizes!");

/*
float_cpu_malloc_and_point_parameters: Allocate FP32 CPU memory for reference parameters

This is like malloc_and_point_parameters, but:
1. Allocates on CPU (not GPU)
2. Always uses float (not floatX)
3. Used to store reference gradients from PyTorch

PARAMETERS:
- params: Pointer to FloatParameterTensors struct to populate
- param_sizes: Array of element counts for each tensor

RETURNS:
- Pointer to the allocated memory block (for later freeing)
*/
float* float_cpu_malloc_and_point_parameters(FloatParameterTensors* params, size_t* param_sizes) {
    // calculate the total number of parameters
    size_t num_parameters = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // everything is float so number of bytes to allocate is a simple multiplication
    float* params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

// ----------------------------------------------------------------------------
// Main Test Program
// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    // ========================================================================
    // STEP 1: Initialize GPU/Multi-GPU environment
    // ========================================================================
    // Set up NCCL for multi-GPU (though we use single GPU for testing)
    char nccl_init_method[256] = "mpi";  // MPI initialization method
    int num_processes = -1;    // -1 = use MPI defaults
    int process_rank = -1;     // -1 = use MPI defaults
    int gpus_per_node = -1;    // -1 = use MPI defaults
    char server_ip[256] = "";  // Not needed for MPI
    char fs_path[256] = "";    // Not needed for MPI
    multi_gpu_config = multi_gpu_config_init(num_processes, process_rank, gpus_per_node, server_ip, fs_path, nccl_init_method);

    // Initialize CUDA, cuBLAS, etc.
    common_start(false, true);

    // ========================================================================
    // STEP 2: Load the appropriate checkpoint based on precision mode
    // ========================================================================
    // The BF16 checkpoint has different numerical values due to quantization
    #if defined(ENABLE_BF16)
    const char* load_filename = "gpt2_124M_bf16.bin";
    #else
    const char* load_filename = "gpt2_124M.bin";
    #endif

    // Initialize and load the GPT-2 model
    GPT2 model;
    gpt2_init_common(&model);
    gpt2_build_from_checkpoint(&model, load_filename);
    // Extract model configuration
    size_t V = model.config.vocab_size;         // Vocabulary size (e.g., 50257)
    size_t Vp = model.config.padded_vocab_size; // Padded vocab (for GPU efficiency)
    size_t maxT = model.config.max_seq_len;     // Max sequence length (e.g., 1024)

    // ========================================================================
    // STEP 3: Parse command-line arguments to configure model options
    // ========================================================================
    // These flags allow testing different model configurations:
    // -w 0/1: Use master weights (FP32 copy of weights for numerical stability)
    // -r 0/1: Enable activation recomputation (trade compute for memory)
    // -ge 0/1: Enable GELU fusion (fuse GELU activation with other operations)
    for (int i = 1; i < argc; i+=2) {
        if (i + 1 >= argc) { exit(EXIT_FAILURE);  } // Must have value after flag
        if (!(strlen(argv[i]) == 2 || strlen(argv[i]) == 3)) { exit(EXIT_FAILURE); } // Must be -x or -xy
        if (argv[i][0] != '-') { exit(EXIT_FAILURE); } // Must start with dash

        // Parse each flag
        if (argv[i][1] == 'w') { model.use_master_weights = atoi(argv[i+1]); }
        else if (argv[i][1] == 'r') { model.recompute = atoi(argv[i+1]); }
        else if (argv[i][1] == 'g' && argv[i][2] == 'e') { model.gelu_fusion = atoi(argv[i+1]); }
    }

    // ========================================================================
    // STEP 4: Load reference data from PyTorch
    // ========================================================================
    FILE *state_file = fopenCheck("gpt2_124M_debug_state.bin", "rb");

    // Read and validate header
    int state_header[256];
    freadCheck(state_header, sizeof(int), 256, state_file);

    // Validate magic number and version
    if (state_header[0] != 20240327) { fprintf(stderr, "Bad magic state file\n"); exit(EXIT_FAILURE); }
    if (state_header[1] != 2) {
        fprintf(stderr, "Bad version in state file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }

    // Extract batch size and sequence length
    int B = state_header[2]; // Batch size (e.g., 4)
    int T = state_header[3]; // Sequence length (e.g., 64)
    assert(0 <= T && T <= maxT);
    printf("[State]\n");
    printf("batch_size: %d\n", B);
    printf("seq_len: %d\n", T);

    // Configure ZeRO optimizer settings (not used in single-GPU test)
    set_zero_configs(&multi_gpu_config, 0, model.num_parameters);

    // Read reference data from Python/PyTorch:
    // 1) Input and target token sequences
    int* x = (int*)mallocCheck(B * T * sizeof(int));  // Input tokens
    int* y = (int*)mallocCheck(B * T * sizeof(int));  // Target tokens
    freadCheck(x, sizeof(int), B*T, state_file);
    freadCheck(y, sizeof(int), B*T, state_file);

    // 2) Expected forward pass results
    float* expected_logits = (float*) mallocCheck(B * T * V * sizeof(float)); // Expected output logits
    float* expected_loss = (float*) mallocCheck(1 * sizeof(float));           // Expected loss value
    freadCheck(expected_logits, sizeof(float), B*T*V, state_file);
    freadCheck(expected_loss, sizeof(float), 1, state_file);

    // 3) Expected backward pass results (gradients)
    // These are always FP32 regardless of GPU precision mode
    FloatParameterTensors expected_grads;
    float* expected_grads_memory = float_cpu_malloc_and_point_parameters(&expected_grads, model.param_elements);
    freadCheck(expected_grads_memory, sizeof(float), model.num_parameters, state_file);
    fcloseCheck(state_file);

    // Allocate CPU memory for copying GPU gradients (which may be mixed precision)
    // We'll copy from GPU (potentially BF16/FP16) to CPU, then convert to FP32 for comparison
    void* grads_memory_cpu = mallocCheck(model.num_parameters_bytes);        // Mixed precision buffer
    float* grads_memory_cpu_float = (float*)mallocCheck(model.num_parameters * sizeof(float)); // FP32 buffer

    // Global pass/fail flag
    int allok = 1;

    // ========================================================================
    // STEP 5: Allocate GPU memory for activations and run initial forward pass
    // ========================================================================
    gpt2_allocate_state(&model, B, T);

    // Run forward pass WITHOUT targets (loss computation) to get logits
    gpt2_forward(&model, x, B, T);

    // ========================================================================
    // STEP 6: Validate logits from forward pass
    // ========================================================================
    // Copy logits from GPU to CPU for comparison
    // Note: logits on GPU may be in BF16/FP16, but we convert to FP32 for comparison
    floatX* logits_cpu_raw = (floatX*)mallocCheck(B * T * Vp * sizeof(floatX));  // Raw GPU format
    float* logits_cpu = (float*)mallocCheck(B * T * Vp * sizeof(float));         // Converted to FP32
    cudaCheck(cudaMemcpy(logits_cpu_raw, model.acts.output, B * T * Vp * sizeof(floatX), cudaMemcpyDeviceToHost));

    // Convert from floatX (possibly BF16/FP16) to float for comparison
    for (int i = 0; i < B * T * Vp; i++) {
        logits_cpu[i] = (float)logits_cpu_raw[i];
    }

    // Set precision-dependent tolerances
    // BF16/FP16 accumulate more error, especially through attention and softmax operations
    float logit_accuracy_threshold = 1e-3f;  // FP32 tolerance
    float loss_diff_threshold = 1e-5f;       // FP32 tolerance
    #if defined(ENABLE_BF16) || defined(ENABLE_F16)
    logit_accuracy_threshold = 25.0f;  // Much looser for reduced precision
    loss_diff_threshold = 0.05f;       // Correspondingly looser for loss
    #endif

    // Compare logits element-by-element
    // IMPORTANT: Only compare up to V (not Vp) to avoid padded columns
    int logits_ok = 1;
    float max_diff = 0.0f;
    for (int bt = 0; bt < B*T; bt++) {
        for (int v = 0; v < V; v++) {
            int i = bt * Vp + v; // linearized index
            if (i < 10) {
                printf("%f, %f\n", expected_logits[i], logits_cpu[i]);
            }
            float diff = fabsf(expected_logits[bt*V + v] - logits_cpu[i]);
            max_diff = fmaxf(max_diff, diff);
            if (diff >= logit_accuracy_threshold) {
                printf("MISMATCH AT INDEX %d,%d: ", bt, v);
                printf("%f %f\n", expected_logits[bt*V + v], logits_cpu[i]);
                logits_ok = 0;
                bt = B*T; // to break out of both loops
                break;
            }
        }
    }
    allok = allok && logits_ok;
    if(!logits_ok) { printf("NOT "); }
    printf("OK (LOGITS)\n");
    printf("logit max diff: %f\n", max_diff);

    // ========================================================================
    // STEP 7: Run 10 training iterations and validate
    // ========================================================================
    // This tests the complete training pipeline:
    // forward -> backward -> gradient calculation -> parameter update
    float losses[10];
    for (int step = 0; step < 10; step++) {
        // Time this iteration
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        // Forward pass: compute logits and loss
        gpt2_forward(&model, x, B, T);

        // Backward pass: compute gradients and reduce across GPUs (if multi-GPU)
        // Arguments: inputs, targets, grad_accum_steps=1, total_batch_size=0 (auto)
        gpt2_backward_and_reduce(&model, x, y, 1, 0);

        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        // ====================================================================
        // COMPREHENSIVE GRADIENT VALIDATION AT STEP 0
        // ====================================================================
        // We only validate gradients on the first step because:
        // 1. Gradient checking is expensive (requires GPU->CPU copies and conversions)
        // 2. If step 0 gradients are correct and all losses match, the implementation is working
        if (step == 0) {
            // Copy gradients from GPU to CPU
            // Note: Gradients may be in mixed precision (BF16/FP16)
            cudaCheck(cudaMemcpy(grads_memory_cpu, model.grads_memory, model.num_parameters_bytes, cudaMemcpyDeviceToHost));

            // Convert all gradients from mixed precision to FP32 for comparison
            // Iterate through each parameter tensor and convert to FP32
            char* src_iterator = (char*)grads_memory_cpu; // Source: mixed precision gradients from GPU
            float* dst_iterator = (float*)grads_memory_cpu_float; // Destination: FP32 gradients
            float* exp_iterator = expected_grads_memory; // Expected: FP32 gradients from PyTorch
            float* tensors1[NUM_PARAMETER_TENSORS]; // Pointers to calculated gradients (FP32)
            float* tensors2[NUM_PARAMETER_TENSORS]; // Pointers to expected gradients (FP32)

            // Process each parameter tensor
            for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
                if (model.param_sizeof[i] == sizeof(float)) {
                    // This gradient is already FP32 => copy directly
                    memcpy(dst_iterator, src_iterator, model.param_elements[i] * sizeof(float));
                } else {
                    // This gradient is in reduced precision (BF16/FP16) => convert to FP32
                    assert(model.param_sizeof[i] == sizeof(floatX)); // Only floatX is supported
                    for (size_t j = 0; j < model.param_elements[i]; j++) {
                        dst_iterator[j] = ((floatX*)src_iterator)[j]; // Convert each element
                    }
                }

                // Record pointers for comparison
                tensors1[i] = dst_iterator; // Our calculated gradients (now in FP32)
                tensors2[i] = exp_iterator; // Expected gradients from PyTorch

                // Advance all iterators to next tensor
                src_iterator += model.param_elements[i] * model.param_sizeof[i]; // Advance by actual byte size
                dst_iterator += model.param_elements[i]; // Advance by float count
                exp_iterator += model.param_elements[i]; // Advance by float count
            }

            // ----------------------------------------------------------------
            // Set per-tensor gradient tolerances
            // ----------------------------------------------------------------
            // These tolerances were determined empirically by inspecting gradient differences.
            // BF16/FP16 precision is lower than FP32, so we must accept larger errors.
            //
            // NOTE: If code changes trigger failures here, it may be acceptable if:
            // 1. The difference is small (close to threshold)
            // 2. Stochastic rounding is adding non-deterministic noise
            // 3. Different GPU hardware uses different matmul algorithms
            // Always manually review before adjusting tolerances!

            // Tolerance for each of the 16 parameter tensors (empirically determined for BF16)
            float grad_thresholds[NUM_PARAMETER_TENSORS] = {
                    5e-1f,   // wte: Token embeddings (large due to sparse updates)
                    4e-3f,   // wpe: Position embeddings
                    1e-1f,   // ln1w: Layer norm 1 weights
                    4e-2f,   // ln1b: Layer norm 1 biases
                    5e-2f,   // qkvw: Attention QKV weights
                    3.5e-2f, // qkvb: Attention QKV biases
                    2e-2f,   // attprojw: Attention projection weights
                    3e-2f,   // attprojb: Attention projection biases
                    5e-2f,   // ln2w: Layer norm 2 weights
                    3e-2f,   // ln2b: Layer norm 2 biases
                    3e-2f,   // fcw: MLP first layer weights
                    3e-2f,   // fcb: MLP first layer biases
                    2e-2f,   // fcprojw: MLP projection weights
                    1e-2f,   // fcprojb: MLP projection biases
                    1e-1f,   // lnfw: Final layer norm weights
                    2e-2f    // lnfb: Final layer norm biases
            };

            // FP32 mode can use much tighter tolerances
            #if defined(ENABLE_FP32)
            for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
                grad_thresholds[i] = 1e-6f;  // Near machine precision for FP32
            }
            #endif

            // Names for each parameter tensor (for readable output)
            const char* names[NUM_PARAMETER_TENSORS] = {
                    "wte", "wpe", "ln1w", "ln1b", "qkvw", "qkvb", "attrpojw",
                    "attprojb", "ln2w", "ln2b", "fcw", "fcb", "fcprojw", "fcprojb",
                    "lnfw", "lnfb"
            };

            // Validate each gradient tensor
            size_t* count = model.param_elements;
            for(int i = 0; i < NUM_PARAMETER_TENSORS; ++i) {
                allok = allok & check_tensor(tensors1[i], tensors2[i], count[i], names[i], grad_thresholds[i]);
            }
        }

        // ====================================================================
        // PARAMETER UPDATE: Apply gradients with gradient clipping
        // ====================================================================
        // Calculate L2 norm of all gradients (for gradient clipping)
        float grad_norm = gpt2_calculate_grad_norm(&model, &multi_gpu_config);
        // Clip gradients if norm > 1.0 (prevents training instability)
        float grad_scale = (grad_norm > 1.0f) ? 1.0f / grad_norm : 1.0f;
        // Update parameters using AdamW optimizer
        // Args: lr=1e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_scale, step
        gpt2_update(&model, 1e-4f, 0.9f, 0.95f, 1e-8f, 0.0f, grad_scale, step+1, &multi_gpu_config);

        // Report timing and loss for this step
        printf("step %d: loss %f (took %f ms)\n", step+1, model.mean_loss, time_elapsed_s * 1000);

        // Round loss to 6 decimal places (matching PyTorch's print formatting)
        // This ensures we compare apples-to-apples with the reference losses
        float rounded_loss = roundf(model.mean_loss * 1000000) / 1000000;
        losses[step] = rounded_loss;
    }

    // ========================================================================
    // STEP 8: Validate loss values across all 10 training steps
    // ========================================================================
    // These are the expected losses from PyTorch (rounded to 6 decimal places)
    float expected_losses[10] = {
        5.270009f,  // Step 1
        4.060681f,  // Step 2
        3.320085f,  // Step 3
        2.717550f,  // Step 4
        2.181066f,  // Step 5
        1.653923f,  // Step 6
        1.168050f,  // Step 7
        0.736873f,  // Step 8
        0.401021f,  // Step 9
        0.187493f   // Step 10
    };

    // Compare each loss value against PyTorch reference
    for (int i = 0; i < 10; i++) {
        if (fabsf(losses[i] - expected_losses[i]) >= loss_diff_threshold) {
            printf("LOSS MISMATCH AT STEP %d: %f %f\n", i+1, losses[i], expected_losses[i]);
            allok = 0;
        } else {
            printf("loss ok at step %d: %f %f\n", i+1, losses[i], expected_losses[i]);
        }
    }

    // ========================================================================
    // STEP 9: Test determinism by saving/loading state
    // ========================================================================
    // This ensures that saving and loading checkpoints works correctly
    // and that training is deterministic (same inputs -> same outputs)

    // Save current model state
    gpt2_write_to_checkpoint(&model, "test_gpt2cu_model.ckpt");

    // Initialize data loader for determinism test
    DataLoader loader;
    dataloader_init(&loader, "dev/data/tinyshakespeare/tiny_shakespeare_val.bin", B, T, multi_gpu_config.process_rank, multi_gpu_config.num_processes, 1);

    // Save optimizer state (includes momentum terms, etc.)
    save_state("test_gpt2cu_state.ckpt", 10, &model, &loader);

    // Run 10 more training steps and record results
    int tokens[10];  // Record first token of each batch
    for (int step = 0; step < 10; step++) {
        dataloader_next_batch(&loader);
        gpt2_forward(&model, loader.inputs, B, T);
        gpt2_backward_and_reduce(&model, loader.inputs, loader.targets, 1, 0);
        gpt2_update(&model, 1e-4f, 0.9f, 0.95f, 1e-8f, 0.0f, 1.0f, step+11, &multi_gpu_config);
        losses[step] = model.mean_loss;
        tokens[step] = loader.inputs[0];  // Save first token for comparison
    }

    // --- Reload from checkpoint and verify determinism ---
    // Free current model and reload from saved checkpoint
    gpt2_free(&model);
    gpt2_build_from_checkpoint(&model, "test_gpt2cu_model.ckpt");
    int ld_step;
    gpt2_allocate_state(&model, B, T);
    load_state(&ld_step, &model, &loader, "test_gpt2cu_state.ckpt");

    // Run the same 10 steps again - should produce IDENTICAL results
    for (int step = 0; step < 10; step++) {
        dataloader_next_batch(&loader);
        gpt2_forward(&model, loader.inputs, B, T);
        gpt2_backward_and_reduce(&model, loader.inputs, loader.targets, 1, 0);
        gpt2_update(&model, 1e-4f, 0.9f, 0.95f, 1e-8f, 0.0f, 1.0f, step+11, &multi_gpu_config);

        // Check that dataloader produces the same tokens
        if(loader.inputs[0] != tokens[step]) {
            printf("Nondeterminism! Token mismatch at step %d: %d vs %d\n", step, tokens[step], loader.inputs[0]);
            allok = false;
            break;
        }

        // Check that loss is EXACTLY the same (bit-for-bit)
        if(losses[step] != model.mean_loss) {
            printf("Nondeterminism! Loss mismatch at step %d: %.15f vs %.15f\n", step, losses[step], model.mean_loss);
            allok = false;
            break;
        } else {
            printf("loss ok at step %d: %f %f\n", step, losses[step], model.mean_loss);
        }
    }

    // ========================================================================
    // FINAL RESULT
    // ========================================================================
    printf("overall okay: %d\n", allok);

    // ========================================================================
    // CLEANUP
    // ========================================================================
    // Delete temporary checkpoint files
    remove("test_gpt2cu_model.ckpt");
    remove("test_gpt2cu_state.ckpt");

    // Free all allocated memory
    dataloader_free(&loader);
    gpt2_free(&model);
    common_free(model);
    free(x);
    free(y);
    free(logits_cpu_raw);
    free(logits_cpu);
    free(expected_logits);
    free(expected_loss);
    free(expected_grads_memory);
    free(grads_memory_cpu);
    free(grads_memory_cpu_float);

    return allok ? EXIT_SUCCESS : EXIT_FAILURE;
}
