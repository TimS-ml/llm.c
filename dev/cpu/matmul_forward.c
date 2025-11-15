/*
 * Matrix Multiplication Forward Pass - CPU Kernels
 * =================================================
 *
 * This file contains CPU implementations of matrix multiplication (matmul) for
 * the forward pass in neural network training. Matmul is one of the most
 * computationally intensive operations in deep learning, so optimizing it
 * is critical for performance.
 *
 * WHAT IS MATMUL IN NEURAL NETWORKS?
 * In a Transformer or similar architecture, matmul is used for:
 * - Attention projections (Q, K, V matrices)
 * - Feed-forward network layers
 * - Output projections
 *
 * TENSOR SHAPES:
 * Input (inp):    (B, T, C)   - B batches, T time steps, C input channels
 * Weight:         (OC, C)     - OC output channels, C input channels
 * Bias:           (OC,)       - Optional bias vector
 * Output (out):   (B, T, OC)  - B batches, T time steps, OC output channels
 *
 * OPERATION:
 * For each position (b, t), we compute:
 *   out[b,t,:] = inp[b,t,:] @ weight.T + bias
 * Where @ is matrix multiplication and .T is transpose.
 *
 * IMPLEMENTATIONS:
 * This file contains multiple kernel implementations with different optimizations:
 * 1. Naive CPU implementation (reference, easy to understand)
 * 2. Optimized implementations with loop tiling and unrolling
 *
 * COMPILE EXAMPLES:
 * MSVC:
 *   cl.exe /O2 /fp:fast /Qvec-report:2 /I. /I ..\..\dev matmul_forward.c
 *   cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX /I. /I ..\..\dev matmul_forward.c
 *   cl.exe /O2 /fp:fast /Qvec-report:2 /arch:AVX2 /I. /I ..\..\dev matmul_forward.c
 *
 * GCC/Clang:
 *   gcc -O3 -march=native matmul_forward.c -lm -o matmul_forward
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

// ============================================================================
// Kernel #0: Naive CPU Reference Implementation
// ============================================================================

/*
 * Matrix Multiplication Forward - Naive CPU Implementation
 * =========================================================
 *
 * This is the straightforward, easy-to-understand implementation of matmul.
 * It serves as a reference for correctness but is not optimized for performance.
 *
 * PARAMETERS:
 * @param out    - Output tensor (B, T, OC): will contain the result
 * @param inp    - Input tensor (B, T, C): input activations
 * @param weight - Weight matrix (OC, C): learnable parameters
 * @param bias   - Bias vector (OC,): optional bias (can be NULL)
 * @param B      - Batch size
 * @param T      - Sequence length (time steps)
 * @param C      - Input channels (input feature dimension)
 * @param OC     - Output channels (output feature dimension)
 *
 * ALGORITHM:
 * For each batch b and time step t:
 *   For each output channel o:
 *     1. Initialize with bias[o] (or 0 if no bias)
 *     2. Compute dot product of inp[b,t,:] with weight[o,:]
 *     3. Store result in out[b,t,o]
 *
 * MEMORY LAYOUT:
 * - inp[b,t,c] is at offset: b*T*C + t*C + c
 * - weight[o,c] is at offset: o*C + c (row-major order)
 * - out[b,t,o] is at offset: b*T*OC + t*OC + o
 *
 * PERFORMANCE NOTES:
 * This implementation is simple but not cache-friendly. The weight matrix
 * is traversed row by row, which works well, but there are no blocking
 * optimizations to maximize cache reuse.
 */
void matmul_forward_cpu(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // Iterate over each batch
    for (int b = 0; b < B; b++) {
        // Iterate over each time step
        for (int t = 0; t < T; t++) {
            // Get pointers to the current (b,t) slice
            float* out_bt = out + b * T * OC + t * OC;        // Output slice
            const float* inp_bt = inp + b * T * C + t * C;    // Input slice

            // Iterate over each output channel
            for (int o = 0; o < OC; o++) {
                // Initialize accumulator with bias (or 0 if no bias)
                float val = (bias != NULL) ? bias[o] : 0.0f;

                // Get pointer to the o-th row of the weight matrix
                // This row contains the weights that will be multiplied with the input
                const float* wrow = weight + o*C;

                // Compute dot product: inp[b,t,:] · weight[o,:]
                // This is the core of the matrix multiplication
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }

                // Store the computed value in the output
                out_bt[o] = val;
            }
        }
    }
}

// ============================================================================
// Kernel #1: Optimized CPU Implementation with Loop Tiling
// ============================================================================

/*
 * Matrix Multiplication Forward - Optimized with Loop Unrolling and Tiling
 * =========================================================================
 *
 * This implementation applies several performance optimizations:
 * 1. Loop fusion: Collapse B and T loops into a single loop
 * 2. Loop tiling: Process LOOP_UNROLL positions at once
 * 3. Weight reuse: Load each weight once and use it LOOP_UNROLL times
 * 4. Register blocking: Keep intermediate results in registers/L1 cache
 *
 * WHY IS THIS FASTER?
 * The key insight is that the weight matrix is shared across all (b,t) positions.
 * By processing multiple positions simultaneously, we can load each weight value
 * once from memory and reuse it multiple times, reducing memory bandwidth pressure.
 *
 * OPTIMIZATION DETAILS:
 * - The B*T dimension is processed in blocks of LOOP_UNROLL (8)
 * - For each output channel, we keep LOOP_UNROLL accumulators in registers
 * - Each weight value is loaded once and used for all LOOP_UNROLL positions
 * - Modern compilers can vectorize the inner loop using SIMD instructions (AVX/SSE)
 * - The compiler will generate FMA (Fused Multiply-Add) instructions for efficiency
 *
 * REQUIREMENTS:
 * - B*T must be a multiple of LOOP_UNROLL (currently 8)
 * - This is typically true in practice (batch sizes are often powers of 2)
 *
 * PERFORMANCE:
 * This can be 2-4x faster than the naive implementation due to better cache
 * utilization and enabling SIMD vectorization by the compiler.
 */
void matmul_forward_ngc92(float* out,
    const float* inp, const float* weight, const float* bias,
    int B, int T, int C, int OC) {

    // Define the unroll factor (how many positions to process simultaneously)
    // 8 is chosen because:
    // - It fits well in registers on most CPUs
    // - It's a power of 2 (good for alignment)
    // - It allows good SIMD utilization (AVX can process 8 floats at once)
    #define LOOP_UNROLL 8

    // Verify that B*T is divisible by LOOP_UNROLL
    // This ensures our tiled loop processes all positions exactly
    if (B * T % LOOP_UNROLL != 0) {
        printf("MUST BE A MULTIPLE OF 8"); // FIXME: Could fall back to naive version
        return;
    }

    // Main optimization: Collapse B and T loops, then process in blocks
    // Instead of: for b, for t, for o, for c
    // We do: for bt (in blocks), for o, for c (reusing weights)
    for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        // For each output channel
        for (int o = 0; o < OC; o++) {
            // Allocate LOOP_UNROLL accumulators in registers/L1 cache
            // These will hold the dot product results for LOOP_UNROLL positions
            float result[LOOP_UNROLL];

            // Initialize accumulators with bias (or 0)
            for (int ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
            }

            // Compute dot products for all LOOP_UNROLL positions simultaneously
            // This is where the magic happens: we load each weight once and use it
            // for all LOOP_UNROLL positions before moving to the next weight
            for (int i = 0; i < C; i++) {
                // Load the weight ONCE for this input channel and output channel
                float w = weight[i + o * C];

                // Use this weight for all LOOP_UNROLL positions
                // The compiler will vectorize this loop using SIMD instructions
                // and generate efficient FMA (Fused Multiply-Add) operations
                for (int ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                    int bt = obt + ibt;  // Actual (b,t) index (flattened)
                    // Accumulate: result += input * weight
                    result[ibt] += inp[bt * C + i] * w;
                }
            }

            // Write all LOOP_UNROLL results back to main memory
            for (int ibt = 0; ibt < LOOP_UNROLL; ++ibt) {
                int bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

// Number of kernel implementations available
#define NUM_KERNELS 2

/*
 * Kernel Dispatcher
 * =================
 *
 * This function routes to different matmul implementations based on kernel_num.
 * This allows easy benchmarking and comparison of different implementations.
 *
 * PARAMETERS:
 * @param kernel_num - Which kernel to use (0 = naive, 1 = optimized)
 * @param out        - Output tensor (B, T, OC)
 * @param inp        - Input tensor (B, T, C)
 * @param weight     - Weight matrix (OC, C)
 * @param bias       - Bias vector (OC,) or NULL
 * @param B          - Batch size
 * @param T          - Sequence length
 * @param C          - Input channels
 * @param OC         - Output channels
 *
 * AVAILABLE KERNELS:
 * 0 - matmul_forward_cpu: Naive reference implementation
 * 1 - matmul_forward_ngc92: Optimized with loop tiling and unrolling
 */
void matmul_forward(int kernel_num,
    float* out,
    const float* inp, const float* weight, const float* bias,
    int B, int T, int C, int OC) {

    switch (kernel_num) {
        case 0:
            matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);
            break;
        case 1:
            matmul_forward_ngc92(out, inp, weight, bias, B, T, C, OC);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}


// Forward declarations of utility functions
void validate_results_cpu(const float* device_result, const float* cpu_reference, const char* name, int num_elements, float tolerance);
float* make_random_float(size_t N);

/*
 * Main Test and Benchmark Program
 * ================================
 *
 * This program:
 * 1. Validates all kernel implementations against the reference
 * 2. Benchmarks each kernel's performance
 *
 * TEST CONFIGURATION:
 * The test uses realistic dimensions from GPT-2 style models:
 * - B=8: Batch size (number of sequences processed in parallel)
 * - T=1024: Sequence length (number of tokens per sequence)
 * - C=768: Input channels (embedding dimension)
 * - OC=3072: Output channels (4x expansion in MLP, typical for Transformers)
 *
 * This represents the matrix multiply in the feed-forward layer of a
 * Transformer, which is one of the most computationally intensive parts.
 */
int main(int argc, char **argv) {
    // Seed random number generator for reproducibility
    srand(0);

    // Define problem dimensions (GPT-2 style settings)
    int B = 8;          // Batch size
    int T = 1024;       // Sequence length
    int C = 768;        // Input channels (embedding dimension)
    int OC = 768 * 4;   // Output channels (4x expansion typical in Transformer MLP)
    int RUNS = 4;       // Number of benchmark iterations per kernel

    // Re-seed for consistent test data generation
    srand(137);

    // Allocate test data (all initialized with random values)
    float* out = make_random_float(B * T * OC);      // Output tensor
    float* inp = make_random_float(B * T * C);       // Input activations
    float* weight = make_random_float(OC * C);       // Weight matrix
    float* bias = make_random_float(OC);             // Bias vector

    // These gradients are allocated but not used in this forward-only test
    float* grad_out = make_random_float(B * T * OC);
    float* grad_inp = make_random_float(B * T * C);
    float* grad_weight = make_random_float(OC * C);
    float* grad_bias = make_random_float(OC);

    // ========================================================================
    // STEP 1: Generate reference results using the naive CPU implementation
    // ========================================================================
    printf("> Calculating reference\n");
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    // ========================================================================
    // STEP 2: Validate all kernel implementations
    // ========================================================================
    // Each kernel should produce the same results as the reference
    for (int kernel_num = 0; kernel_num < NUM_KERNELS; kernel_num++) {
        printf("> Verifying kernel #%d\n", kernel_num);

        // Re-seed to generate identical random data for each kernel
        srand(137);

        // Allocate fresh memory for this kernel's test
        float* kernel_out = make_random_float(B * T * OC);
        float* kernel_inp = make_random_float(B * T * C);
        float* kernel_weight = make_random_float(OC * C);
        float* kernel_bias = make_random_float(OC);

        // Run the kernel
        matmul_forward(kernel_num, kernel_out, kernel_inp, kernel_weight, kernel_bias, B, T, C, OC);

        // Validate against reference (tolerance of 1e-5 for floating-point differences)
        validate_results_cpu(kernel_out, out, "out", B * T * OC, 1e-5);

        // Clean up kernel-specific allocations
        free(kernel_out);
        free(kernel_inp);
        free(kernel_weight);
        free(kernel_bias);
    }

    printf("All kernels passed! Starting benchmarks.\n\n");

    // ========================================================================
    // STEP 3: Benchmark all kernels
    // ========================================================================
    // Measure wall-clock time for each kernel implementation
    for (int kernel_num = 0; kernel_num < NUM_KERNELS; kernel_num++) {
        printf("> Running kernel #%d\n", kernel_num);

        // Get starting time
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        // Run the kernel multiple times to get stable measurements
        for (int i = 0; i < RUNS; i++) {
            matmul_forward(kernel_num, out, inp, weight, bias, B, T, C, OC);
        }

        // Get ending time and calculate elapsed time
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        // Print average time per run
        printf("> Kernel #%d, (took %f ms)\n", kernel_num, time_elapsed_s * 1000);
    }

    // ========================================================================
    // Cleanup: Free all allocated memory
    // ========================================================================
    free(out);
    free(inp);
    free(weight);
    free(bias);

    free(grad_out);
    free(grad_inp);
    free(grad_weight);
    free(grad_bias);

    return 0;
}

/*
 * Random Float Array Generator
 * =============================
 *
 * Allocates and fills an array with random floats in the range [-1, 1].
 *
 * PARAMETERS:
 * @param N - Number of elements to allocate
 *
 * RETURNS:
 * Pointer to newly allocated array filled with random values
 *
 * NOTE: Caller is responsible for freeing the returned memory.
 * The range [-1, 1] is typical for neural network initialization.
 */
float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        // Generate random float in range [-1, 1]
        // rand()/RAND_MAX gives [0, 1], multiply by 2 and subtract 1 gives [-1, 1]
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    return arr;
}

/*
 * Result Validator
 * ================
 *
 * Compares two float arrays element-wise to verify correctness.
 * Exits with failure if any mismatch exceeds tolerance.
 *
 * PARAMETERS:
 * @param kernel_result  - Array from the kernel being tested
 * @param cpu_reference  - Array from the reference implementation
 * @param name           - Descriptive name for error messages
 * @param num_elements   - Number of elements to compare
 * @param tolerance      - Maximum acceptable absolute difference
 *
 * VALIDATION STRATEGY:
 * - Prints first 5 element comparisons for visual inspection
 * - Uses adaptive tolerance: tolerance + |reference_value|
 *   (allows for larger absolute errors when values are large)
 * - Stops after 10 mismatches to avoid excessive output
 * - Exits with failure if any mismatches are found
 *
 * The adaptive tolerance accounts for floating-point precision limits:
 * larger numbers naturally have larger absolute errors while maintaining
 * the same relative precision.
 */
void validate_results_cpu(const float* kernel_result, const float* cpu_reference, const char* name, int num_elements, float tolerance) {
    int nfaults = 0;

    for (int i = 0; i < num_elements; i++) {
        // Print the first few comparisons for manual inspection
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], kernel_result[i]);
        }

        // Compute effective tolerance (adaptive based on magnitude)
        // This accounts for floating-point precision limits with large numbers
        float t_eff = tolerance + fabs(cpu_reference[i]);

        // Check if values match within tolerance
        if (fabs(cpu_reference[i] - kernel_result[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs CPU_new: %f\n",
                   name, i, cpu_reference[i], kernel_result[i]);
            nfaults++;

            // Stop after 10 errors to avoid overwhelming output
            if (nfaults >= 10) {
                exit(EXIT_FAILURE);
            }
        }
    }

    // If any mismatches were found, exit with failure
    if (nfaults > 0) {
        exit(EXIT_FAILURE);
    }

    printf("OK\n");
}