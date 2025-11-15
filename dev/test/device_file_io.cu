/*
 * Device-File I/O Testing Suite
 * ==============================
 *
 * This test validates the CUDA device-to-file and file-to-device I/O functions
 * that are critical for saving and loading model checkpoints during training.
 *
 * WHAT IS BEING TESTED?
 * The llmc library includes functions to efficiently transfer data between:
 * - GPU device memory (where model weights and activations live during training)
 * - Disk files (for saving checkpoints and loading pretrained weights)
 *
 * WHY IS THIS IMPORTANT?
 * Training large language models requires:
 * 1. Saving checkpoints periodically (to resume if training crashes)
 * 2. Loading pretrained weights (to continue training or for inference)
 * 3. Handling data larger than CPU RAM (by using buffers)
 *
 * The functions being tested handle buffered I/O, which allows processing
 * data in chunks rather than requiring the entire model to fit in CPU memory.
 *
 * TEST STRATEGY:
 * 1. Generate random data on GPU
 * 2. Write it to a file using device_to_file()
 * 3. Read it back using file_to_device()
 * 4. Verify the data matches exactly
 * 5. Test with different buffer sizes and data sizes
 *
 * COMPILE AND RUN:
 * From the dev/test directory:
 *   nvcc -o device_file_io device_file_io.cu && ./device_file_io
 */

#include "../../llmc/cuda_common.h"  // CUDA utility functions and error checking
#include <vector>                     // For std::vector
#include <random>                     // For random number generation
#include <cstdio>                     // For file I/O
#include <algorithm>                  // For std::generate

/*
 * Single Test Case
 * ================
 *
 * Tests the device-file I/O functions with specific parameters.
 *
 * PARAMETERS:
 * @param nelem        - Number of float elements to test
 * @param wt_buf_size  - Buffer size for writing (in bytes)
 * @param rd_buf_size  - Buffer size for reading (in bytes)
 *
 * HOW IT WORKS:
 * 1. Allocate GPU memory
 * 2. Generate random test data on CPU
 * 3. Copy test data to GPU
 * 4. Write GPU data to file using device_to_file() with specified buffer
 * 5. Read file back to GPU using file_to_device() with specified buffer
 * 6. Copy result back to CPU
 * 7. Verify every element matches original data exactly
 *
 * BUFFER SIZES:
 * The buffer size parameters test different scenarios:
 * - Buffers larger than data: Should work efficiently
 * - Buffers smaller than data: Requires multiple I/O operations
 * - Mismatched read/write buffers: Tests flexibility
 *
 * This ensures the I/O functions correctly handle chunked transfers
 * regardless of buffer configuration.
 */
void test(size_t nelem, size_t wt_buf_size, size_t rd_buf_size) {

    // ========================================================================
    // STEP 1: Allocate GPU memory for the test data
    // ========================================================================
    float* data;
    cudaCheck(cudaMalloc(&data, nelem*sizeof(float)));

    // ========================================================================
    // STEP 2: Generate random test data on CPU
    // ========================================================================
    // We use a wide range [-100, 100] to test with realistic values
    std::vector<float> random_data(nelem);
    std::mt19937 rng(42);  // Mersenne Twister RNG with fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-100.f, 100.f);
    std::generate(random_data.begin(), random_data.end(), [&](){ return dist(rng); });

    // ========================================================================
    // STEP 3: Copy test data from CPU to GPU
    // ========================================================================
    cudaCheck(cudaMemcpy(data, random_data.data(), random_data.size()*sizeof(float), cudaMemcpyHostToDevice));

    // ========================================================================
    // STEP 4: Create CUDA stream for async operations
    // ========================================================================
    // The I/O functions support asynchronous operation via CUDA streams
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ========================================================================
    // STEP 5: Write GPU data to file (device_to_file)
    // ========================================================================
    // This tests the device-to-file transfer with the specified write buffer size
    FILE* tmp = fopenCheck("tmp.bin", "w");
    device_to_file(tmp, data, nelem * sizeof(float), wt_buf_size, stream);
    fcloseCheck(tmp);

    // ========================================================================
    // STEP 6: Allocate new GPU memory for reading back
    // ========================================================================
    // We use separate memory to ensure we're actually reading from file,
    // not just using cached data
    float* reload;
    cudaCheck(cudaMalloc(&reload, nelem*sizeof(float)));

    // ========================================================================
    // STEP 7: Read file back to GPU (file_to_device)
    // ========================================================================
    // This tests the file-to-device transfer with the specified read buffer size
    tmp = fopenCheck("tmp.bin", "r");
    file_to_device(reload, tmp, nelem * sizeof(float), rd_buf_size, stream);
    fcloseCheck(tmp);

    // ========================================================================
    // STEP 8: Copy result back to CPU for verification
    // ========================================================================
    std::vector<float> cmp(nelem);
    cudaCheck(cudaMemcpy(cmp.data(), reload, nelem * sizeof(float), cudaMemcpyDeviceToHost));

    // ========================================================================
    // STEP 9: Verify every element matches exactly
    // ========================================================================
    // We require bit-exact matches since we're testing I/O, not computation
    for(int i = 0; i < nelem; ++i) {
        if(random_data[i] != cmp[i])  {
            fprintf(stderr, "FAIL: Mismatch at position %d: %f vs %f\n", i, random_data[i], cmp[i]);
            remove("tmp.bin");
            exit(EXIT_FAILURE);
        }
    }

    // ========================================================================
    // STEP 10: Cleanup
    // ========================================================================
    cudaCheck(cudaFree(reload));
    cudaCheck(cudaFree(data));
    remove("tmp.bin");  // Delete temporary test file
}

/*
 * Main Test Suite
 * ===============
 *
 * Runs multiple test cases with different configurations to thoroughly
 * validate the device-file I/O functions.
 *
 * TEST CASES:
 *
 * Test 1: Buffers larger than data
 *   - 1025 elements, 10000 byte buffers
 *   - Tests that oversized buffers work correctly
 *   - Should transfer in a single operation
 *
 * Test 2: Small, mismatched buffers
 *   - 1025 elements, 1024 byte write buffer, 513 byte read buffer
 *   - Tests chunked I/O with different buffer sizes for read/write
 *   - Ensures the functions handle non-matching buffer sizes correctly
 *
 * Test 3: Exact match buffers
 *   - 500 elements, buffers exactly sized for the data
 *   - Tests edge case where buffer size equals data size
 *   - Should transfer in exactly one operation
 *
 * Test 4: Large array
 *   - 125,000 elements (~500KB), 10000 byte buffers
 *   - Tests with realistic checkpoint sizes
 *   - Requires multiple chunked transfers
 *
 * If all tests pass, the I/O functions are working correctly!
 */
int main() {
    // Test 1: Large buffers (larger than data)
    // Data: 1025 floats = 4100 bytes, Buffers: 10000 bytes
    test(1025, 10000, 10000);

    // Test 2: Small, different buffers (smaller than data)
    // Data: 1025 floats = 4100 bytes
    // Write buffer: 1024 bytes, Read buffer: 513 bytes
    test(1025, 1024, 513);

    // Test 3: Exact match (buffer size exactly equals data size)
    // Data: 500 floats = 2000 bytes, Buffers: 2000 bytes
    test(500, 500*sizeof(float), 500*sizeof(float));

    // Test 4: Large array (realistic checkpoint size)
    // Data: 125,000 floats = 500KB, Buffers: 10000 bytes
    // This requires ~50 chunked transfers
    test(125'000, 10000, 10000);

    // If we reach here, all tests passed!
    printf("All device-file I/O tests passed!\n");
    return 0;
}