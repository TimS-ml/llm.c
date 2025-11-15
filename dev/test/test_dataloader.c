/*
 * DataLoader Testing Suite
 * =========================
 *
 * This test suite validates the DataLoader, which is responsible for efficiently
 * loading training data from disk during neural network training.
 *
 * WHAT IS A DATALOADER?
 * The DataLoader reads tokenized text data from multiple shard files and provides
 * batches of training examples. It's a critical component that feeds data to the
 * training loop.
 *
 * KEY FEATURES BEING TESTED:
 * 1. Multi-shard loading: Reading from multiple data files
 * 2. Multi-process support: Different processes read different portions of data
 * 3. Shuffling: Random permutation of data for better training
 * 4. Batching: Organizing tokens into (B, T) shaped batches
 *
 * DATALOADER FUNCTIONALITY:
 * - Reads tokenized data shards (binary files with uint16_t tokens)
 * - Generates input/target pairs: input[t] -> target[t+1]
 * - Supports distributed training (multiple processes, each with their own slice)
 * - Can shuffle data for better generalization
 * - Handles epoch boundaries (automatically moves to next shard)
 *
 * TEST ORGANIZATION:
 * - test_simple(): Basic single-process, non-shuffled loading
 * - test_multiprocess_simple(): Multi-process without shuffling
 * - test_shuffled(): Single-process with shuffling
 * - test_multiprocess_shuffled(): Multi-process with shuffling
 *
 * COMPILE AND RUN:
 * From dev/test directory:
 *   gcc -O3 -I../../llmc -o test_dataloader test_dataloader.c -lm && ./test_dataloader
 *
 * TODOs:
 * - test load/save state of DataLoader
 */

#include <unistd.h>
#include "../../llmc/dataloader.h"  // DataLoader implementation

// ============================================================================
// Test Configuration Constants
// ============================================================================

#define SHARD_NAME_LEN 64
char shard_name[SHARD_NAME_LEN];  // Buffer for shard filenames
const int num_tokens = 140;        // Number of tokens per shard (small for testing)
int num_shards = 4;                // Number of shard files to create

// ============================================================================
// Validation Helper Functions
// ============================================================================

/*
 * Range Validator
 * ===============
 *
 * Verifies that a token array contains consecutive integers in a given range.
 *
 * PARAMETERS:
 * @param tokens - Array of token IDs to check
 * @param start  - Expected first value
 * @param end    - Expected value after last (exclusive)
 * @param file   - Source file (for error reporting)
 * @param line   - Source line (for error reporting)
 *
 * USAGE:
 * Use the checkRange() macro, which automatically passes __FILE__ and __LINE__:
 *   checkRange(tokens, 0, 10);  // Expects tokens = [0, 1, 2, ..., 9]
 *
 * This is used to verify that non-shuffled data is loaded in the correct order.
 */
void check_range(const int *tokens, const int start, const int end, const char *file, int line) {
    int n = end - start;
    for (int i = 0; i < n; i++) {
        int token = tokens[i];
        if (token != start + i) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, start + i);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", line);
            exit(EXIT_FAILURE);
        }
    }
}
// Macro that automatically captures file/line for error reporting
#define checkRange(tokens, start, end) check_range(tokens, start, end, __FILE__, __LINE__)

/*
 * Equality Validator
 * ==================
 *
 * Verifies that all elements in an array equal a specific value.
 *
 * PARAMETERS:
 * @param tokens   - Array of token IDs to check
 * @param n        - Number of elements to check
 * @param expected - Expected value for all elements
 * @param file     - Source file (for error reporting)
 * @param line     - Source line (for error reporting)
 *
 * USAGE:
 * Use the checkEquals() macro:
 *   checkEquals(tokens, 10, 0);  // Expects all 10 tokens to be 0
 *
 * This is used to verify that certain tokens were never seen (count == 0)
 * or were seen a specific number of times.
 */
void check_equals(const int *tokens, const int n, const int expected, const char *file, int line) {
    for (int i = 0; i < n; i++) {
        int token = tokens[i];
        if (token != expected) {
            fprintf(stderr, "Error: tokens[%d] = %d, expected %d\n", i, token, expected);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", line);
            exit(EXIT_FAILURE);
        }
    }
}
// Macro that automatically captures file/line for error reporting
#define checkEquals(tokens, n, expected) check_equals(tokens, n, expected, __FILE__, __LINE__)

// ============================================================================
// Test Functions
// ============================================================================

/*
 * Test: Simple Sequential Loading
 * ================================
 *
 * Tests the most basic DataLoader functionality:
 * - Multi-shard: Reads from multiple data files
 * - Single-process: Only one process (no distributed training)
 * - Not shuffled: Data returned in sequential order
 *
 * EXPECTED BEHAVIOR:
 * The DataLoader should return all tokens in exact sequential order:
 * - Shard 0: tokens 0, 1, 2, ..., 139
 * - Shard 1: tokens 140, 141, 142, ..., 279
 * - Shard 2: tokens 280, 281, 282, ..., 419
 * - Shard 3: tokens 420, 421, 422, ..., 559
 *
 * For each batch:
 * - inputs should be [start, start+1, ..., start+BT-1]
 * - targets should be [start+1, start+2, ..., start+BT] (shifted by 1)
 *
 * This tests the fundamental data loading mechanism without any
 * complexities from shuffling or multi-processing.
 */
void test_simple(void) {
    printf("test_simple... ");

    // Configure batch dimensions
    int B = 4;  // Batch size (4 sequences processed together)
    int T = 8;  // Sequence length (8 tokens per sequence)

    // Configure as single-process, non-shuffled
    int process_rank = 0;      // This process's rank (0 for single process)
    int num_processes = 1;     // Total number of processes
    int should_shuffle = 0;    // No shuffling - sequential order

    // Initialize DataLoader with wildcard pattern for shard files
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, num_processes, should_shuffle);

    // Calculate how many batches fit in each shard
    int batches_fit = num_tokens / (B * T);  // 140 / 32 = 4 batches per shard
    int BT = B * T;  // Total tokens per batch = 32

    // Test over multiple epochs to verify epoch boundaries work correctly
    int num_epochs = 4;
    for (int e = 0; e < num_epochs; e++) {  // Epoch loop
        for (int s = 0; s < num_shards; s++) {  // Shard loop
            // Calculate starting token ID for this shard
            // Shard 0: start=0, Shard 1: start=140, etc.
            int start = s * num_tokens;

            for (int b = 0; b < batches_fit; b++) {  // Batch loop
                // Get next batch from DataLoader
                dataloader_next_batch(&loader);

                // Verify inputs are sequential: [start, start+1, ..., start+BT-1]
                checkRange(loader.inputs, start, start + BT);

                // Verify targets are inputs shifted by 1: [start+1, ..., start+BT]
                // This implements the language modeling objective: predict next token
                checkRange(loader.targets, start + 1, start + BT + 1);

                // Move to next batch's starting position
                start += BT;
            }
        }
    }

    // Cleanup
    dataloader_free(&loader);
    printf("OK\n");
}

/*
 * Test: Multi-Process Sequential Loading
 * =======================================
 *
 * Tests DataLoader with multiple processes (simulated in a single program).
 * - Multi-shard: Reads from multiple data files
 * - Multi-process: Simulates 2 processes reading different data portions
 * - Not shuffled: Data returned in sequential order
 *
 * DISTRIBUTED TRAINING SIMULATION:
 * In real distributed training, each GPU runs a separate process, and each
 * process needs different data. This test simulates that by creating two
 * DataLoaders with different process ranks.
 *
 * EXPECTED BEHAVIOR:
 * The two loaders should alternate in token space:
 * - Loader 0 (rank=0): tokens [0-31], [64-95], [128-159], ...
 * - Loader 1 (rank=1): tokens [32-63], [96-127], [160-191], ...
 *
 * Each batch is B*T=32 tokens, and they interleave perfectly so that
 * together they cover all tokens without overlap.
 *
 * This ensures the DataLoader correctly partitions data across processes
 * for distributed training.
 */
void test_multiprocess_simple(void) {
    printf("test_multiprocess_simple... ");

    // Batch configuration
    int B = 4;
    int T = 8;
    int num_processes = 2;     // Simulate 2 GPUs/processes
    int should_shuffle = 0;    // Sequential order

    // Create two DataLoaders, one for each simulated process
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loader0, loader1;
    dataloader_init(&loader0, shard_name, B, T, 0, num_processes, should_shuffle);  // Process rank 0
    dataloader_init(&loader1, shard_name, B, T, 1, num_processes, should_shuffle);  // Process rank 1

    // With 2 processes, each gets half the batches
    int batches_fit = num_tokens / (B * T * num_processes);  // 140 / 64 = 2 batches per process per shard
    int BT = B * T;
    int num_epochs = 4;

    for (int e = 0; e < num_epochs; e++) {
        for (int s = 0; s < num_shards; s++) {
            int start = s * num_tokens;
            for (int b = 0; b < batches_fit; b++) {
                // Get batches from both loaders
                dataloader_next_batch(&loader0);
                dataloader_next_batch(&loader1);

                // Loader 0 should get tokens [start, start+BT)
                checkRange(loader0.inputs, start, start + BT);
                checkRange(loader0.targets, start + 1, start + BT + 1);

                // Loader 1 should get tokens [start+BT, start+2*BT)
                checkRange(loader1.inputs, start + BT, start + 2*BT);
                checkRange(loader1.targets, start + BT + 1, start + 2*BT + 1);

                // Both loaders together covered 2*BT tokens
                start += 2*BT;
            }
        }
    }

    dataloader_free(&loader0);
    dataloader_free(&loader1);
    printf("OK\n");
}

/*
 * Test: Shuffled Data Loading
 * ============================
 *
 * Tests DataLoader with shuffling enabled.
 * - Multi-shard: Reads from multiple data files
 * - Single-process: Only one process
 * - Shuffled: Data returned in random permuted order
 *
 * WHY SHUFFLE?
 * Shuffling training data is critical for good neural network training:
 * - Breaks correlations between consecutive examples
 * - Prevents the model from memorizing data order
 * - Improves generalization
 * - Reduces variance in gradient estimates
 *
 * VALIDATION STRATEGY:
 * Since shuffling is random, we can't check exact token order.
 * Instead, we verify that:
 * 1. All tokens that should be seen are seen exactly once per epoch
 * 2. Tokens outside the valid range are never seen
 *
 * We maintain counters for each token ID and verify the counts
 * match expectations after each epoch.
 */
void test_shuffled(void) {
    printf("test_shuffled... ");

    // Batch configuration
    int B = 4;
    int T = 8;
    int process_rank = 0;
    int num_processes = 1;
    int should_shuffle = 1;  // Enable shuffling!

    // Initialize DataLoader with shuffling
    snprintf(shard_name, 64, "shard_????.bin");
    DataLoader loader;
    dataloader_init(&loader, shard_name, B, T, process_rank, num_processes, should_shuffle);

    // ========================================================================
    // Allocate counters to track how many times we see each token
    // ========================================================================
    int total_tokens = num_shards * num_tokens;  // 4 shards * 140 tokens = 560 total
    int *num_seen_inputs = (int *)calloc(total_tokens, sizeof(int));   // All initialized to 0
    int *num_seen_targets = (int *)calloc(total_tokens, sizeof(int));  // All initialized to 0

    int batches_fit = num_tokens / (B * T);  // Batches per shard
    int BT = B * T;
    int num_epochs = 4;

    // ========================================================================
    // Collect statistics over multiple epochs
    // ========================================================================
    for (int e = 0; e < num_epochs; e++) {
        for (int s = 0; s < num_shards; s++) {
            int start = s * num_tokens;
            for (int b = 0; b < batches_fit; b++) {
                dataloader_next_batch(&loader);

                // Count how many times we see each token
                for (int i = 0; i < BT; i++) {
                    int input_token = loader.inputs[i];
                    int target_token = loader.targets[i];

                    // Sanity check: tokens should be in valid range
                    assert(input_token >= 0 && input_token < total_tokens);
                    assert(target_token >= 0 && target_token < total_tokens);

                    // Increment counters
                    num_seen_inputs[input_token]++;
                    num_seen_targets[target_token]++;
                }
                start += BT;
            }
        }
    }

    // ========================================================================
    // Verify token counts are correct
    // ========================================================================
    int tokens_fit = batches_fit * BT;  // Number of tokens actually used per shard

    for (int s = 0; s < num_shards; s++) {
        int start = s * num_tokens;  // Starting token ID for this shard

        // VERIFY INPUTS:
        // - Tokens [start, start+tokens_fit) should be seen num_epochs times
        //   (these are the tokens that fit in complete batches)
        checkEquals(num_seen_inputs + start, tokens_fit, num_epochs);

        // - Remaining tokens in this shard should never be seen
        //   (these are leftover tokens that don't fit in a complete batch)
        checkEquals(num_seen_inputs + start + tokens_fit, num_tokens - tokens_fit, 0);

        // VERIFY TARGETS:
        // Targets are offset by 1 (next token prediction)
        checkEquals(num_seen_targets + start + 1, tokens_fit, num_epochs);

        // For the last shard, account for the final token not being a target
        int remaining = (s == (num_shards - 1)) ? num_tokens - tokens_fit - 1 : num_tokens - tokens_fit;
        checkEquals(num_seen_targets + start + 1 + tokens_fit, remaining, 0);
    }

    // Cleanup
    dataloader_free(&loader);
    free(num_seen_inputs);
    free(num_seen_targets);
    printf("OK\n");
}

/*
 * Test: Multi-Process Shuffled Loading
 * =====================================
 *
 * Tests the most complex DataLoader configuration:
 * - Multi-shard: Multiple data files
 * - Multi-process: Simulates distributed training with 2 processes
 * - Shuffled: Random permutation of data
 *
 * REAL-WORLD SCENARIO:
 * This represents the typical production setup for training large models:
 * - Multiple GPUs (each running a separate process)
 * - Data distributed across processes for parallel training
 * - Shuffling enabled for better generalization
 *
 * VALIDATION:
 * Similar to test_shuffled(), we can't verify exact order.
 * Instead, we verify that across both processes:
 * - All expected tokens are seen exactly num_epochs times total
 * - No overlap between processes (each token seen by only one process)
 * - No tokens are missed
 */
void test_multiprocess_shuffled(void) {
    printf("test_multiprocess_shuffled... ");

    // Configuration
    int B = 4;
    int T = 8;
    const int num_processes = 2;
    int should_shuffle = 0;  // NOTE: Original code has 0, but comment says shuffled!

    // Create one DataLoader per process
    snprintf(shard_name, SHARD_NAME_LEN, "shard_????.bin");
    DataLoader loaders[num_processes];
    for (int i = 0; i < num_processes; i++) {
        dataloader_init(&loaders[i], shard_name, B, T, i, num_processes, should_shuffle);
    }

    // Allocate counters for all tokens across all processes
    int total_tokens = num_shards * num_tokens;
    int *num_seen_inputs = (int *)calloc(total_tokens, sizeof(int));
    int *num_seen_targets = (int *)calloc(total_tokens, sizeof(int));

    // With multiple processes, each gets a fraction of the batches
    int batches_fit = num_tokens / (B * T * num_processes);
    int BT = B * T;
    int num_epochs = 4;

    // Collect statistics from all processes
    for (int e = 0; e < num_epochs; e++) {
        for (int s = 0; s < num_shards; s++) {
            int start = s * num_tokens;
            for (int b = 0; b < batches_fit; b++) {
                // Get batches from all processes
                for (int n = 0; n < num_processes; n++) {
                    DataLoader *loader = &loaders[n];
                    dataloader_next_batch(loader);

                    // Count tokens from this process
                    for (int i = 0; i < BT; i++) {
                        int input_token = loader->inputs[i];
                        int target_token = loader->targets[i];

                        // Validate token range
                        assert(input_token >= 0 && input_token < total_tokens);
                        assert(target_token >= 0 && target_token < total_tokens);

                        // Increment counters
                        num_seen_inputs[input_token]++;
                        num_seen_targets[target_token]++;
                    }
                    start += BT;
                }
            }
        }
    }

    // Verify token counts across all processes
    int tokens_fit = batches_fit * (B * T * num_processes);
    for (int s = 0; s < num_shards; s++) {
        int start = s * num_tokens;

        // Verify inputs: used tokens seen num_epochs times, others never
        checkEquals(num_seen_inputs + start, tokens_fit, num_epochs);
        checkEquals(num_seen_inputs + start + tokens_fit, num_tokens - tokens_fit, 0);

        // Verify targets: same but offset by 1
        checkEquals(num_seen_targets + start + 1, tokens_fit, num_epochs);
        int remaining = (s == (num_shards - 1)) ? num_tokens - tokens_fit - 1 : num_tokens - tokens_fit;
        checkEquals(num_seen_targets + start + 1 + tokens_fit, remaining, 0);
    }

    // Cleanup all loaders
    for (int i = 0; i < num_processes; i++) {
        dataloader_free(&loaders[i]);
    }
    free(num_seen_inputs);
    free(num_seen_targets);
    printf("OK\n");
}

/*
 * Main Test Program
 * =================
 *
 * Sets up test data, runs all DataLoader tests, and cleans up.
 *
 * TEST DATA STRUCTURE:
 * Creates 4 shard files, each containing 140 tokens:
 * - shard_0000.bin: tokens [0, 1, 2, ..., 139]
 * - shard_0001.bin: tokens [140, 141, 142, ..., 279]
 * - shard_0002.bin: tokens [280, 281, 282, ..., 419]
 * - shard_0003.bin: tokens [420, 421, 422, ..., 559]
 *
 * Each token ID is unique across all shards, making it easy to verify
 * that the DataLoader is reading the correct data.
 *
 * SHARD FILE FORMAT:
 * Each shard file contains:
 * - Header (3 ints):
 *   [0] Magic number: 20240520 (for file format identification)
 *   [1] Version: 1 (format version)
 *   [2] Token count: number of tokens in this shard
 * - Tokens (array of uint16_t): the actual token IDs
 */
int main(void) {

    // ========================================================================
    // SETUP: Generate test data shards
    // ========================================================================
    int header[HEADER_SIZE];
    uint16_t tokens[num_tokens];

    for (int shard_id = 0; shard_id < num_shards; shard_id++) {
        // Generate unique token IDs across all shards
        // Shard 0: 0-139, Shard 1: 140-279, Shard 2: 280-419, Shard 3: 420-559
        int token_offset = shard_id * num_tokens;
        for (int i = 0; i < num_tokens; i++) {
            tokens[i] = token_offset + i;
        }

        // Write shard file
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);

        // Prepare header
        header[0] = 20240520;    // Magic number for file format validation
        header[1] = 1;           // Version number
        header[2] = num_tokens;  // Number of tokens in this shard

        // Write header and tokens to binary file
        FILE* shard_file = fopenCheck(shard_name, "wb");
        fwrite(header, sizeof(int), HEADER_SIZE, shard_file);
        fwrite(tokens, sizeof(uint16_t), num_tokens, shard_file);
        fcloseCheck(shard_file);

        printf("Wrote shard %s\n", shard_name);
    }

    // ========================================================================
    // RUN ALL TESTS
    // ========================================================================
    // Each test validates a different aspect of the DataLoader:
    // 1. Basic sequential loading
    // 2. Multi-process data partitioning
    // 3. Shuffled data loading
    // 4. Multi-process + shuffled (full production configuration)

    test_simple();
    test_multiprocess_simple();
    test_shuffled();
    test_multiprocess_shuffled();

    // ========================================================================
    // CLEANUP: Remove test shard files
    // ========================================================================
    for (int shard_id = 0; shard_id < num_shards; shard_id++) {
        snprintf(shard_name, SHARD_NAME_LEN, "shard_%04d.bin", shard_id);
        remove(shard_name);
    }

    printf("\nAll DataLoader tests passed!\n");
    return EXIT_SUCCESS;
}