/*
GPT-2 CPU Testing Suite
=======================

PURPOSE:
This file tests the CPU implementation of GPT-2 training (train_gpt2.c) by validating
the forward pass, backward pass, and parameter updates against reference values from PyTorch.

WHAT IS BEING TESTED:
1. Forward pass correctness - logits must match PyTorch output
2. Backward pass correctness - all parameter gradients must match PyTorch
3. Training loop correctness - loss values over 10 steps must match PyTorch
4. Parameter update correctness - optimizer state updates are implicitly validated through loss matching

HOW IT WORKS:
- Loads a pre-trained GPT-2 124M model checkpoint
- Loads reference data from Python/PyTorch (gpt2_124M_debug_state.bin) which contains:
  * Input tokens (x) and target tokens (y)
  * Expected logits from forward pass
  * Expected loss value
  * Expected parameter gradients from backward pass
  * Expected loss values for 10 training iterations
- Runs the same computations in C and compares outputs element-by-element
- Uses tolerances (e.g., 2e-2 for gradients, 1e-2 for logits/loss) to account for floating point differences

HOW TO INTERPRET RESULTS:
- "OK" status: Values match within tolerance - implementation is correct
- "NOT OK" status: Values differ beyond tolerance - indicates a bug in the implementation
- maxdiff: The maximum absolute difference found - smaller is better
- At the end, "overall okay: 1" means all tests passed, "0" means at least one test failed

RUNNING THE TEST:
1. First generate reference data: python train_gpt2.py
2. Compile: make test_gpt2
3. Run: ./test_gpt2
4. Check final output: "overall okay: 1" = PASS, "overall okay: 0" = FAIL
*/

#define TESTING
#include "train_gpt2.c"

// ----------------------------------------------------------------------------
// Tensor Validation Helper Function
// ----------------------------------------------------------------------------

/*
check_tensor: Compares two tensors element-by-element and validates they match within tolerance

This is a "poor man's tensor checker" that compares calculated values against reference values.
It serves as the core validation function for all our gradient and activation checks.

PARAMETERS:
- a: Calculated tensor (from our C implementation)
- b: Reference tensor (from PyTorch)
- n: Number of elements in the tensors
- label: Name of the tensor being checked (for debug output)

RETURNS:
- 1 if tensors match within tolerance (2e-2)
- 0 if any element differs by more than tolerance

WHAT IT DOES:
1. Compares each element: |a[i] - b[i]|
2. Prints the first 5 elements for visual inspection
3. Tracks the maximum difference across all elements
4. Returns OK/NOT OK based on tolerance check
*/
int check_tensor(float *a, float *b, int n, const char* label) {
    int print_upto = 5;
    int ok = 1;
    float maxdiff = 0.0f;
    float tol = 2e-2f;
    printf("%s\n", label);
    for (int i = 0; i < n; i++) {
        // look at the diffence at position i of these two tensors
        float diff = fabsf(a[i] - b[i]);

        // keep track of the overall error
        ok = ok && (diff <= tol);
        if (diff > maxdiff) { maxdiff = diff; }

        // for the first few elements of each tensor, pretty print
        // the actual numbers, so we can do a visual, qualitative proof/assessment
        if (i < print_upto) {
            if (diff <= tol) {
                if (i < print_upto) { printf("OK "); }
            } else {
                if (i < print_upto) { printf("NOT OK "); }
            }
            printf("%f %f\n", a[i], b[i]);
        }
    }
    // print the final result for this tensor
    if (ok) {
        printf("TENSOR OK, maxdiff = %e\n", maxdiff);
    } else {
        printf("TENSOR NOT OK, maxdiff = %e\n", maxdiff);
    }
    return ok;
}

// ----------------------------------------------------------------------------
// Main Test Program
// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {

    // ========================================================================
    // STEP 1: Load the GPT-2 model from checkpoint
    // ========================================================================
    // This loads the pre-trained 124M parameter GPT-2 model
    // The checkpoint contains all the trained weights and biases
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // Extract model configuration parameters for convenience
    int C = model.config.channels;          // Channel dimension (embedding size, e.g., 768)
    int V = model.config.vocab_size;        // Vocabulary size (e.g., 50257)
    int Vp = model.config.padded_vocab_size; // Padded vocab size (for efficient computation)
    int maxT = model.config.max_seq_len;    // Maximum sequence length (e.g., 1024)
    int L = model.config.num_layers;        // Number of transformer layers (e.g., 12)

    // ========================================================================
    // STEP 2: Load reference data from PyTorch for validation
    // ========================================================================
    // This binary file contains reference values generated by train_gpt2.py
    // We'll compare our C implementation's outputs against these references
    FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
    if (state_file == NULL) { printf("Error opening state file\n"); return 1; }

    // Read and validate the file header
    int state_header[256];
    freadCheck(state_header, sizeof(int), 256, state_file);

    // Magic number check: ensures we're reading the correct file format
    if (state_header[0] != 20240327) { printf("Bad magic state file\n"); return 1; }

    // Version check: ensures compatibility between this test and the Python reference
    if (state_header[1] != 2) {
        printf("Bad version in state file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        return 1;
    }

    // Extract batch size and sequence length from the header
    int B = state_header[2]; // Batch size (e.g., 4) - how many sequences we process in parallel
    int T = state_header[3]; // Sequence length (e.g., 64) - how many tokens per sequence
    printf("[State]\n");
    printf("batch_size: %d\n", B);
    printf("seq_len: %d\n", T);

    // Allocate memory for expected gradients (reference values from PyTorch)
    ParameterTensors expected_grads;
    float* expected_grads_memory = malloc_and_point_parameters(&expected_grads, model.param_sizes);

    // Allocate memory for inputs and expected outputs
    // These are used purely for validation, not for actual training
    int* x = (int*) malloc(B * T * sizeof(int));           // Input token IDs
    int* y = (int*) malloc(B * T * sizeof(int));           // Target token IDs (for loss calculation)
    float* expected_logits = (float*) malloc(B * T * V * sizeof(float)); // Expected output probabilities
    float* expected_loss = (float*) malloc(1 * sizeof(float));           // Expected loss value

    // Read all reference data from the Python-generated file
    // This data represents the "ground truth" that our C implementation must match
    freadCheck(x, sizeof(int), B*T, state_file);                          // Input tokens
    freadCheck(y, sizeof(int), B*T, state_file);                          // Target tokens
    freadCheck(expected_logits, sizeof(float), B*T*V, state_file);        // Expected forward pass output
    freadCheck(expected_loss, sizeof(float), 1, state_file);              // Expected loss
    freadCheck(expected_grads_memory, sizeof(float), model.num_parameters, state_file); // Expected gradients
    fcloseCheck(state_file);

    // Global pass/fail flag for the entire test suite
    // Will be set to 0 if any validation check fails
    int allok = 1;

    // ========================================================================
    // STEP 3: Run 10 training iterations and validate at each step
    // ========================================================================
    // These are the expected loss values from PyTorch after each training step
    // Our C implementation must match these closely to be considered correct
    // The loss should decrease as the model learns to predict the next token
    float expected_losses[10] = {
        5.270007133483887f,     // Step 0: Initial loss (model is untrained on this data)
        4.059706687927246f,     // Step 1: Loss after first gradient update
        3.3751230239868164f,    // Step 2: Model is learning...
        2.8007826805114746f,    // Step 3
        2.315382242202759f,     // Step 4
        1.8490285873413086f,    // Step 5
        1.3946564197540283f,    // Step 6
        0.9991465210914612f,    // Step 7
        0.6240804195404053f,    // Step 8
        0.37651097774505615f    // Step 9: Model has memorized the training sequence
    };

    // Main training loop: 10 iterations to validate the entire training pipeline
    for (int step = 0; step < 10; step++) {

        // Time the forward and backward passes for performance monitoring
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        // Forward pass: compute model outputs (logits) and loss
        gpt2_forward(&model, x, y, B, T);

        // Zero out gradients before backward pass (they accumulate otherwise)
        gpt2_zero_grad(&model);

        // Backward pass: compute gradients of loss with respect to all parameters
        gpt2_backward(&model);

        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        // ====================================================================
        // SPECIAL VALIDATION AT STEP 0: Check logits, loss, and ALL gradients
        // ====================================================================
        // We only do comprehensive validation on the first step because:
        // 1. It's expensive to validate all tensors
        // 2. If step 0 is correct and losses match at all steps, the implementation is working
        if (step == 0) {
            // The model has just computed logits - let's validate them against PyTorch
            // --- Validate Logits (Forward Pass Output) ---
            // Logits are the raw output scores for each vocabulary token
            // Shape: [B*T, V] - for each position, we predict next token probabilities
            int logits_ok = 1;
            float* calculated_logits = model.acts.logits;
            float max_diff = 0.0f;

            // Compare each logit value against the PyTorch reference
            for (int bt = 0; bt < B*T; bt++) {           // For each position in the batch
                for (int v = 0; v < V; v++) {             // For each vocabulary token
                    // Note: We only loop to V (not Vp) because padding is ignored
                    int i = bt * Vp + v; // Linearized index using padded vocab size

                    // Print first 10 values for manual inspection
                    if (i < 10) {
                        printf("%f, %f\n", expected_logits[i], calculated_logits[i]);
                    }

                    // Check if difference exceeds tolerance
                    float diff = fabsf(expected_logits[bt*V + v] - calculated_logits[i]);
                    max_diff = fmaxf(max_diff, diff);
                    if (diff >= 1e-2f) {  // Tolerance: 0.01
                        printf("MISMATCH AT INDEX %d,%d: ", bt, v);
                        printf("%f %f\n", expected_logits[bt*V + v], calculated_logits[i]);
                        logits_ok = 0;
                        bt = B*T; // Break out of both loops
                        break;
                    }
                }
            }
            if(!logits_ok) { printf("NOT "); }
            printf("OK (LOGITS), max_diff = %e\n", max_diff);
            allok = allok && logits_ok;

            // --- Validate Loss (Scalar Output) ---
            // The loss is a single number representing how well the model predicts
            // Lower loss = better predictions
            if (fabsf(model.mean_loss - *expected_loss) >= 1e-2) {
                printf("LOSS MISMATCH: %f %f\n", model.mean_loss, *expected_loss);
                allok = 0;
            } else {
                printf("LOSS OK: %f %f\n", model.mean_loss, *expected_loss);
            }

            // --- Validate ALL Parameter Gradients (Backward Pass Output) ---
            // Gradients tell us how to adjust each parameter to reduce the loss
            // We check all 16 parameter tensors in the GPT-2 model
            int gradoks[16];
            ParameterTensors grads = model.grads;

            // Check gradients for each parameter tensor:
            // "d" prefix indicates gradient (e.g., "dwte" = gradient of wte)
            gradoks[0] = check_tensor(grads.wte, expected_grads.wte, V*C, "dwte");           // Token embedding gradients
            gradoks[1] = check_tensor(grads.wpe, expected_grads.wpe, maxT*C, "dwpe");        // Position embedding gradients
            gradoks[2] = check_tensor(grads.ln1w, expected_grads.ln1w, L*C, "dln1w");        // Layer norm 1 weights
            gradoks[3] = check_tensor(grads.ln1b, expected_grads.ln1b, L*C, "dln1b");        // Layer norm 1 biases
            gradoks[4] = check_tensor(grads.qkvw, expected_grads.qkvw, L*3*C*C, "dqkvw");    // Attention QKV weights
            gradoks[5] = check_tensor(grads.qkvb, expected_grads.qkvb, L*3*C, "dqkvb");      // Attention QKV biases
            gradoks[6] = check_tensor(grads.attprojw, expected_grads.attprojw, L*C*C, "dattprojw");  // Attention projection weights
            gradoks[7] = check_tensor(grads.attprojb, expected_grads.attprojb, L*C, "dattprojb");    // Attention projection biases
            gradoks[8] = check_tensor(grads.ln2w, expected_grads.ln2w, L*C, "dln2w");        // Layer norm 2 weights
            gradoks[9] = check_tensor(grads.ln2b, expected_grads.ln2b, L*C, "dln2b");        // Layer norm 2 biases
            gradoks[10] = check_tensor(grads.fcw, expected_grads.fcw, L*4*C*C, "dfcw");      // MLP first layer weights
            gradoks[11] = check_tensor(grads.fcb, expected_grads.fcb, L*4*C, "dfcb");        // MLP first layer biases
            gradoks[12] = check_tensor(grads.fcprojw, expected_grads.fcprojw, L*C*4*C, "dfcprojw");  // MLP projection weights
            gradoks[13] = check_tensor(grads.fcprojb, expected_grads.fcprojb, L*C, "dfcprojb");      // MLP projection biases
            gradoks[14] = check_tensor(grads.lnfw, expected_grads.lnfw, C, "dlnfw");         // Final layer norm weights
            gradoks[15] = check_tensor(grads.lnfb, expected_grads.lnfb, C, "dlnfb");         // Final layer norm biases

            // Aggregate results: all gradient checks must pass
            for (int i = 0; i < 16; i++) {
                allok = allok && gradoks[i];
            }
        }

        // ====================================================================
        // PARAMETER UPDATE: Apply gradients using AdamW optimizer
        // ====================================================================
        // Parameters: learning_rate=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step+1);

        // --- Validate loss at this step ---
        // Even though we only checked gradients at step 0, we check loss at every step
        // This ensures the entire training pipeline (forward + backward + update) is working
        float expected_loss = expected_losses[step];
        float actual_loss = model.mean_loss;
        int step_loss_ok = fabsf(expected_loss - actual_loss) < 1e-2;  // Tolerance: 0.01
        allok = allok && step_loss_ok;

        // Print timing and validation results for this step
        // The time helps us monitor performance, OK flag shows if this step passed
        printf("step %d: loss %f (took %f ms) OK = %d\n", step, model.mean_loss, time_elapsed_s * 1000, step_loss_ok);
    }

    // ========================================================================
    // FINAL RESULT
    // ========================================================================
    // If allok == 1, all tests passed - the C implementation matches PyTorch!
    // If allok == 0, at least one test failed - there's a bug to fix
    printf("overall okay: %d\n", allok);

    // ========================================================================
    // CLEANUP
    // ========================================================================
    // Free all allocated memory to prevent leaks
    free(x);
    free(y);
    free(expected_logits);
    free(expected_loss);
    free(expected_grads_memory);
    gpt2_free(&model);
    return 0;
}
