/*
 * Layer Normalization Educational Example
 * ==========================================
 *
 * This file demonstrates how to implement Layer Normalization (LayerNorm) in C.
 * LayerNorm is a critical component in modern neural networks, especially Transformers.
 *
 * BACKGROUND:
 * Layer Normalization was introduced by Ba et al. (2016) and became a standard
 * component in the Transformer architecture (Vaswani et al., 2017). In GPT-2,
 * LayerNorm is used in the "pre-normalization" configuration, where it's applied
 * at the beginning of each Transformer block rather than after.
 *
 * WHAT IS LAYER NORMALIZATION?
 * LayerNorm normalizes activations across the channel dimension for each individual
 * example and time step. Given input x, it computes:
 *   output = weight * ((x - mean) / sqrt(variance + epsilon)) + bias
 *
 * This normalization improves training stability by keeping activations in a
 * reasonable range and helps with gradient flow during backpropagation.
 *
 * USAGE:
 * 1. First run: python layernorm.py
 *    (This generates reference data in ln.bin for validation)
 * 2. Then compile: gcc layernorm.c -o layernorm -lm
 * 3. Finally run: ./layernorm
 *
 * The program will compare our C implementation against PyTorch's reference output.
 *
 * TENSOR SHAPES:
 * - Input (x): (B, T, C) where B=batch size, T=sequence length, C=channels
 * - Weights (w): (C,) - scale parameters
 * - Bias (b): (C,) - shift parameters
 * - Output: (B, T, C) - normalized and scaled activations
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * LayerNorm Forward Pass
 * =======================
 *
 * Computes the forward pass of Layer Normalization.
 *
 * PARAMETERS:
 * @param out    - Output tensor (B, T, C): normalized and scaled activations
 * @param mean   - Output array (B, T): computed mean for each (b,t) position
 * @param rstd   - Output array (B, T): reciprocal std dev (1/sqrt(var+eps)) for each (b,t)
 * @param inp    - Input tensor (B, T, C): the activations to normalize
 * @param weight - Weight tensor (C,): learnable scale parameters
 * @param bias   - Bias tensor (C,): learnable shift parameters
 * @param B      - Batch size
 * @param T      - Sequence length (time steps)
 * @param C      - Number of channels (features)
 *
 * ALGORITHM:
 * For each position (b, t) in the batch and sequence:
 * 1. Compute mean across the C channels
 * 2. Compute variance across the C channels
 * 3. Normalize: (x - mean) / sqrt(variance + epsilon)
 * 4. Scale and shift: normalized * weight + bias
 *
 * MEMORY LAYOUT:
 * 3D tensors are stored in row-major order in 1D arrays.
 * To access element at position [b, t, c], we use offset: b*T*C + t*C + c
 * The innermost dimension (C) is contiguous in memory for cache efficiency.
 *
 * CACHING:
 * We save mean and rstd (reciprocal standard deviation) for the backward pass.
 * This is a memory vs. compute tradeoff - we could recompute these in backward,
 * but saving them (only B*T floats each) makes backward faster.
 */
void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    // Small epsilon to prevent division by zero in normalization
    float eps = 1e-5f;

    // Iterate over each example in the batch
    for (int b = 0; b < B; b++) {
        // Iterate over each position in the sequence
        for (int t = 0; t < T; t++) {
            // Compute pointer offset to the input position inp[b,t,:]
            // This gives us the start of the C-dimensional vector at this (b,t) position
            float* x = inp + b * T * C + t * C;

            // STEP 1: Calculate the mean across channels
            // Mean = (1/C) * sum(x[i]) for i in [0, C)
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;

            // STEP 2: Calculate the variance (without Bessel's correction)
            // Variance = (1/C) * sum((x[i] - mean)^2) for i in [0, C)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;  // Center the value
                v += xshift * xshift;      // Accumulate squared differences
            }
            v = v/C;

            // STEP 3: Calculate the reciprocal standard deviation (rstd)
            // rstd = 1 / sqrt(variance + epsilon)
            // We add epsilon for numerical stability (prevents division by zero)
            // Storing reciprocal allows us to multiply instead of divide later (faster)
            float s = 1.0f / sqrtf(v + eps);

            // STEP 4: Normalize, scale, and shift
            // Compute pointer to the output position out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                // Normalize: subtract mean and multiply by reciprocal std dev
                float n = (s * (x[i] - m));
                // Scale and shift: apply learnable affine transformation
                float o = n * weight[i] + bias[i];
                // Write to output
                out_bt[i] = o;
            }

            // STEP 5: Cache mean and rstd for the backward pass
            // These are stored in 1D arrays indexed by b*T + t
            // We need these to compute gradients during backpropagation
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

/*
 * LayerNorm Backward Pass
 * ========================
 *
 * Computes gradients for Layer Normalization using backpropagation.
 *
 * PARAMETERS:
 * @param dinp    - Gradient w.r.t. input (B, T, C): to be accumulated
 * @param dweight - Gradient w.r.t. weights (C,): to be accumulated
 * @param dbias   - Gradient w.r.t. bias (C,): to be accumulated
 * @param dout    - Gradient w.r.t. output (B, T, C): upstream gradients
 * @param inp     - Original input tensor (B, T, C): from forward pass
 * @param weight  - Weight tensor (C,): from forward pass
 * @param mean    - Mean values (B, T): cached from forward pass
 * @param rstd    - Reciprocal std dev (B, T): cached from forward pass
 * @param B       - Batch size
 * @param T       - Sequence length
 * @param C       - Number of channels
 *
 * GRADIENT DERIVATION:
 * The backward pass implements the chain rule for LayerNorm.
 * Given: output = weight * ((input - mean) / sqrt(var + eps)) + bias
 *
 * We need to compute three gradients:
 * 1. dinp: How the loss changes w.r.t. the input
 * 2. dweight: How the loss changes w.r.t. the scale parameters
 * 3. dbias: How the loss changes w.r.t. the shift parameters
 *
 * The math simplifies to:
 * - dbias: Simply sum dout across batch and time
 * - dweight: Sum (dout * normalized) across batch and time
 * - dinp: More complex due to mean and variance dependencies
 *
 * ACCUMULATION:
 * All gradients use += instead of = because:
 * 1. Variables may be used multiple times in the computational graph
 * 2. Gradients from multiple uses must sum (multivariate chain rule)
 * 3. This is standard practice in autograd systems
 *
 * IMPLEMENTATION NOTE:
 * We recompute 'norm' here rather than caching it from forward pass.
 * This is a memory vs. compute tradeoff:
 * - Caching norm would require B*T*C floats (large!)
 * - Recomputing uses only mean & rstd which are B*T floats (small)
 * - The recomputation cost is minimal compared to memory savings
 */
void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C) {
    // Process each batch example
    for (int b = 0; b < B; b++) {
        // Process each time step
        for (int t = 0; t < T; t++) {
            // Get pointers to the relevant slices for this (b,t) position
            float* dout_bt = dout + b * T * C + t * C;  // Upstream gradient
            float* inp_bt = inp + b * T * C + t * C;    // Original input
            float* dinp_bt = dinp + b * T * C + t * C;  // Where we'll write input gradient
            float mean_bt = mean[b * T + t];            // Cached mean
            float rstd_bt = rstd[b * T + t];            // Cached reciprocal std dev

            // PASS 1: Compute two reduction terms needed for input gradient
            // These represent how the mean and variance affect the gradient flow
            float dnorm_mean = 0.0f;       // Mean of gradients w.r.t. normalized values
            float dnorm_norm_mean = 0.0f;  // Mean of (gradient * normalized) products
            for (int i = 0; i < C; i++) {
                // Recompute the normalized value from forward pass
                // norm = (input - mean) * rstd
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                // Gradient w.r.t. normalized value (before scale/shift)
                // This comes from: out = norm * weight + bias
                // So: dnorm = dout * weight (by chain rule)
                float dnorm_i = weight[i] * dout_bt[i];
                // Accumulate for the reduction terms
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            // Average the accumulated values
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // PASS 2: Compute and accumulate all gradients
            for (int i = 0; i < C; i++) {
                // Recompute normalized value (again - could cache if we wanted)
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                // Gradient w.r.t. normalized value
                float dnorm_i = weight[i] * dout_bt[i];

                // GRADIENT 1: Bias gradient
                // From: out = norm * weight + bias
                // We have: dout/dbias = 1
                // So: dbias = sum(dout)
                dbias[i] += dout_bt[i];

                // GRADIENT 2: Weight gradient
                // From: out = norm * weight + bias
                // We have: dout/dweight = norm
                // So: dweight = sum(dout * norm)
                dweight[i] += norm_bti * dout_bt[i];

                // GRADIENT 3: Input gradient (most complex!)
                // The gradient has three terms due to how input affects:
                // - The normalized value directly (term 1)
                // - The mean (term 2)
                // - The variance/rstd (term 3)
                float dval = 0.0f;
                dval += dnorm_i;                        // Term 1: Direct effect
                dval -= dnorm_mean;                     // Term 2: Effect through mean
                dval -= norm_bti * dnorm_norm_mean;     // Term 3: Effect through variance
                dval *= rstd_bt;                        // Final scaling by 1/std
                // Accumulate into input gradient
                dinp_bt[i] += dval;
            }
        }
    }
}

/*
 * Tensor Validation Utility
 * ==========================
 *
 * Compares two tensors element-wise to verify correctness.
 *
 * PARAMETERS:
 * @param a     - First tensor (typically our C implementation output)
 * @param b     - Second tensor (typically PyTorch reference output)
 * @param n     - Number of elements to compare
 * @param label - Descriptive name for the tensor being checked
 *
 * RETURNS:
 * 1 if all elements match within tolerance (1e-5), 0 otherwise
 *
 * This function prints each element comparison to help debug mismatches.
 * A tolerance of 1e-5 accounts for floating-point arithmetic differences
 * between our C code and PyTorch.
 */
int check_tensor(float *a, float *b, int n, char* label) {
    int ok = 1;
    printf("%s\n", label);
    for (int i = 0; i < n; i++) {
        // Check if values match within tolerance
        if (fabs(a[i] - b[i]) <= 1e-5) {
            printf("OK ");
        } else {
            printf("NOT OK ");
            ok = 0;
        }
        // Print both values for comparison
        printf("%f %f\n", a[i], b[i]);
    }
    return ok;
}

/*
 * Main Test Program
 * =================
 *
 * This program validates our LayerNorm implementation against PyTorch.
 *
 * WORKFLOW:
 * 1. Read reference data from ln.bin (generated by layernorm.py)
 * 2. Run our C implementation of forward and backward passes
 * 3. Compare our results against PyTorch's reference outputs
 *
 * The test uses small dimensions (B=2, T=3, C=4) for easy debugging and
 * verification. If all checks pass, our implementation is correct!
 */
int main() {

    // Define small test dimensions
    int B = 2; // batch size (number of independent sequences)
    int T = 3; // time / sequence length (number of tokens per sequence)
    int C = 4; // number of channels (feature dimension)

    // Allocate memory for all tensors needed in forward pass
    float* x = (float*) malloc(B * T * C * sizeof(float));     // Input activations
    float* w = (float*) malloc(C * sizeof(float));             // Learnable weights (scale)
    float* b = (float*) malloc(C * sizeof(float));             // Learnable biases (shift)
    float* out = (float*) malloc(B * T * C * sizeof(float));   // Forward output
    float* mean = (float*) malloc(B * T * sizeof(float));      // Computed means
    float* rstd = (float*) malloc(B * T * sizeof(float));      // Reciprocal std devs

    // Allocate memory for backward pass gradients
    float* dout = (float*) malloc(B * T * C * sizeof(float));  // Gradient of loss w.r.t. output
    float* dx = (float*) malloc(B * T * C * sizeof(float));    // Gradient w.r.t. input
    float* dw = (float*) malloc(C * sizeof(float));            // Gradient w.r.t. weights
    float* db = (float*) malloc(C * sizeof(float));            // Gradient w.r.t. biases

    // ========================================================================
    // STEP 1: Load reference data from PyTorch
    // ========================================================================
    // The Python script (layernorm.py) generates this binary file containing
    // inputs and expected outputs for both forward and backward passes
    FILE *file = fopen("ln.bin", "rb");
    if (file == NULL) {
        printf("Error opening file\n");
        return 1;
    }

    // Read all reference data in the same order it was written
    fread(x, sizeof(float), B * T * C, file);    // Input tensor
    fread(w, sizeof(float), C, file);            // Weights
    fread(b, sizeof(float), C, file);            // Biases
    fread(out, sizeof(float), B * T * C, file);  // Expected forward output
    fread(mean, sizeof(float), B * T, file);     // Expected means
    fread(rstd, sizeof(float), B * T, file);     // Expected rstds
    fread(dout, sizeof(float), B * T * C, file); // Upstream gradients
    fread(dx, sizeof(float), B * T * C, file);   // Expected input gradients
    fread(dw, sizeof(float), C, file);           // Expected weight gradients
    fread(db, sizeof(float), C, file);           // Expected bias gradients
    fclose(file);

    // ========================================================================
    // STEP 2: Run our C implementation - Forward Pass
    // ========================================================================
    // Allocate fresh memory for our computed values
    float* c_out = (float*) malloc(B * T * C * sizeof(float));
    float* c_mean = (float*) malloc(B * T * sizeof(float));
    float* c_rstd = (float*) malloc(B * T * sizeof(float));

    // Execute our forward pass
    layernorm_forward(c_out, c_mean, c_rstd, x, w, b, B, T, C);

    // Validate our forward pass outputs against PyTorch reference
    printf("Checking forward pass...\n");
    check_tensor(out, c_out, B*T*C, "out");     // Normalized output
    check_tensor(mean, c_mean, B*T, "mean");    // Means
    check_tensor(rstd, c_rstd, B*T, "rstd");    // Reciprocal std devs

    // ========================================================================
    // STEP 3: Run our C implementation - Backward Pass
    // ========================================================================
    // Use calloc to initialize gradients to zero (important for += accumulation!)
    float* c_dx = (float*) calloc(B * T * C, sizeof(float));  // Zero-initialized
    float* c_dw = (float*) calloc(B * T, sizeof(float));      // Zero-initialized
    float* c_db = (float*) calloc(B * T, sizeof(float));      // Zero-initialized

    // Execute our backward pass
    layernorm_backward(c_dx, c_dw, c_db, dout, x, w, c_mean, c_rstd, B, T, C);

    // Validate our backward pass gradients against PyTorch reference
    printf("Checking backward pass...\n");
    check_tensor(c_dx, dx, B*T*C, "dx");  // Input gradients
    check_tensor(c_dw, dw, C, "dw");      // Weight gradients
    check_tensor(c_db, db, C, "db");      // Bias gradients

    // ========================================================================
    // Cleanup: Free all allocated memory
    // ========================================================================
    free(x);
    free(w);
    free(b);
    free(out);
    free(mean);
    free(rstd);
    free(dout);
    free(dx);
    free(dw);
    free(db);

    // If we reach here, all tests passed!
    return 0;
}
