/*
================================================================================
Tensor Permutation (Transpose) CUDA Kernels
================================================================================

PURPOSE:
--------
Demonstrates how to permute (transpose) a 4D tensor on GPU, changing the order
of dimensions. This operation is fundamental in deep learning for operations like:
  - Reshaping activations between different layer formats
  - Preparing data for attention mechanisms
  - Converting between NCHW and NHWC formats
  - Implementing transpose operations in matrix multiplications

EDUCATIONAL GOAL:
-----------------
This file serves as a tutorial for understanding:
1. How multidimensional arrays are stored in linear (1D) memory
2. How to compute indices for accessing elements in flattened arrays
3. How to map between different dimension orderings
4. Coalesced vs. non-coalesced memory access patterns

SPECIFIC TASK:
--------------
Permute a 4D tensor from shape (dim1, dim2, dim3, dim4) to (dim4, dim3, dim1, dim2).
Example: If input is (24, 42, 20, 32), output will be (32, 20, 24, 42).

This specific permutation pattern appears in transformer models when rearranging
attention tensors for efficient computation.

Compile example:
nvcc -O3 permute.cu -o permute

Run:
./permute

================================================================================
MATHEMATICAL FOUNDATION: Flattening Multidimensional Arrays
================================================================================

MEMORY LAYOUT:
--------------
GPUs (and most programming languages) store multidimensional arrays in linear
memory using "row-major" order. For a 4D array with dimensions:
  dim1 = size of the 1st dimension
  dim2 = size of the 2nd dimension
  dim3 = size of the 3rd dimension
  dim4 = size of the 4th dimension

The element at position (i1, i2, i3, i4) is stored at linear index:
  linear_idx = i1 × (dim2 × dim3 × dim4) + i2 × (dim3 × dim4) + i3 × dim4 + i4

This is called "row-major" because the last dimension (dim4) varies fastest.

INTUITION:
----------
Think of it as nested loops:
  for i1 in range(dim1):          # Outermost: slowest changing
    for i2 in range(dim2):
      for i3 in range(dim3):
        for i4 in range(dim4):    # Innermost: fastest changing
          # Process element at (i1, i2, i3, i4)

Elements are laid out in memory in the order these nested loops visit them.

REVERSE MAPPING: From Linear Index to Multidimensional Indices
----------------------------------------------------------------
Given a linear index (idx) in range [0, dim1×dim2×dim3×dim4), we can recover
the original 4D indices using division and modulo operations:

  i1 = (idx / (dim2 × dim3 × dim4)) % dim1
  i2 = (idx / (dim3 × dim4)) % dim2
  i3 = (idx / dim4) % dim3
  i4 = idx % dim4

PATTERN RECOGNITION:
--------------------
The formula for each dimension follows a pattern:
  - Divide by the product of all dimensions to the RIGHT
  - Take modulo with the current dimension size

This "peels off" one dimension at a time, from left to right.

EXAMPLE:
--------
For a tensor of shape (2, 3, 4, 5) with 120 total elements:
  Linear index idx = 73

  i1 = 73 / (3×4×5) = 73 / 60 = 1 (with remainder 13)
  i2 = 13 / (4×5) = 13 / 20 = 0 (with remainder 13)
  i3 = 13 / 5 = 2 (with remainder 3)
  i4 = 3 % 5 = 3

  So idx=73 corresponds to position (1, 0, 2, 3)



================================================================================
THE PERMUTATION OPERATION
================================================================================

GOAL:
-----
Transform tensor from shape (dim1, dim2, dim3, dim4) to (dim4, dim3, dim1, dim2)

DIMENSION MAPPING:
------------------
  Original position → New position
  dim1 (1st)  →  new 3rd dimension
  dim2 (2nd)  →  new 4th dimension
  dim3 (3rd)  →  new 2nd dimension
  dim4 (4th)  →  new 1st dimension

ALGORITHM:
----------
For each element in the original tensor:
  1. Compute linear index (idx) of current thread
  2. Extract 4D indices (i1, i2, i3, i4) from idx
  3. Compute NEW linear index for permuted layout
  4. Copy value from old[idx] to new[permuted_idx]

COMPUTING THE PERMUTED INDEX:
------------------------------
The new tensor has shape (dim4, dim3, dim1, dim2).
The element that was at (i1, i2, i3, i4) moves to (i4, i3, i1, i2).

Using row-major ordering for the NEW shape:
  permuted_idx = i4 × (dim3 × dim1 × dim2) + i3 × (dim1 × dim2) + i1 × dim2 + i2

BREAKDOWN:
----------
  i4 × (dim3 × dim1 × dim2):
    - i4 is now the outermost dimension
    - For each unit increase in i4, we skip an entire (dim3 × dim1 × dim2) block

  i3 × (dim1 × dim2):
    - Within the i4 block, i3 is the next dimension
    - For each unit increase in i3, we skip a (dim1 × dim2) block

  i1 × dim2:
    - Within the i3 block, i1 comes next
    - For each unit increase in i1, we skip dim2 elements

  i2:
    - i2 is the innermost (fastest-varying) dimension
    - Direct offset within the i1 block

MEMORY ACCESS PATTERN:
----------------------
IMPORTANT: This operation has non-coalesced memory access!
- Threads with consecutive IDs read consecutive locations (good)
- But they write to scattered locations (bad for performance)
- This is an inherent challenge in transpose/permutation operations
- More sophisticated implementations might use shared memory tiling to improve this

GENERALIZATION:
---------------
This same pattern works for any tensor permutation:
1. Extract indices from source layout
2. Reorder indices according to desired permutation
3. Compute target index using target layout formula
4. Copy element from source[old_idx] to target[new_idx]

*/


#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include "common.h"

/*
CPU reference implementation of 4D tensor permutation.
Serves as ground truth for validating GPU kernel correctness.

This function permutes a 4D tensor from (dim1, dim2, dim3, dim4) to (dim4, dim3, dim1, dim2).

Parameters:
  matrix: Input tensor in original layout (read-only)
  out_matrix: Output tensor in permuted layout (write-only)
  dim1, dim2, dim3, dim4: Dimensions of the input tensor

Algorithm:
  For each element in the tensor:
    1. Extract 4D indices from linear index
    2. Compute new linear index with permuted dimension order
    3. Copy element to new location

Performance: O(N) where N = dim1 × dim2 × dim3 × dim4 (total elements)
*/
void permute_cpu(const float* matrix, float* out_matrix, int dim1, int dim2, int dim3, int dim4) {
    int total_threads = dim1 * dim2 * dim3 * dim4;

    for (int idx = 0; idx < total_threads; idx++) {
        // ===================================================================
        // Step 1: Extract 4D indices from linear index
        // ===================================================================
        // Recover the original 4D position (i1, i2, i3, i4) from idx
        int i1 = (idx / (dim2 * dim3 * dim4)) % dim1;
        int i2 = (idx / (dim3 * dim4)) % dim2;
        int i3 = (idx / dim4) % dim3;
        int i4 = idx % dim4;

        // ===================================================================
        // Step 2: Compute new linear index for permuted layout
        // ===================================================================
        // The element at (i1, i2, i3, i4) moves to position (i4, i3, i1, i2)
        // in the new tensor with shape (dim4, dim3, dim1, dim2)
        int permuted_idx = i4 * (dim3 * dim1 * dim2) + i3 * (dim1 * dim2) + i1 * dim2 + i2;

        // ===================================================================
        // Step 3: Copy element to new location
        // ===================================================================
        out_matrix[permuted_idx] = matrix[idx];
    }
}

/*
================================================================================
GPU Kernel: 4D Tensor Permutation
================================================================================

PARALLELIZATION STRATEGY:
--------------------------
- Each thread handles exactly one element of the tensor
- Threads are assigned consecutive linear indices
- Thread idx processes the element at linear position idx in the input

THREAD/BLOCK ORGANIZATION:
--------------------------
- 1D grid of 1D blocks (simplest organization for this task)
- Block size: 256 threads (configurable)
- Grid size: ceil(total_elements / block_size)

MEMORY ACCESS PATTERN:
----------------------
READ (from matrix):
  - Coalesced: Consecutive threads read consecutive memory locations
  - Optimal for GPU memory bandwidth

WRITE (to out_matrix):
  - Non-coalesced: Consecutive threads write to scattered locations
  - This is unavoidable for general permutations
  - Performance limited by write pattern

OPTIMIZATION OPPORTUNITIES (not implemented here):
---------------------------------------------------
1. Shared memory tiling:
   - Load tiles into shared memory
   - Perform permutation within shared memory
   - Write out in coalesced pattern

2. Vectorized loads/stores:
   - Use float4 for 128-bit transactions when possible
   - Requires alignment and dimension divisibility

3. Bank conflict avoidance:
   - Careful shared memory padding
   - Helps with shared memory tiling approach

This implementation prioritizes simplicity and clarity over maximum performance.

Parameters:
  matrix: Input tensor (dim1, dim2, dim3, dim4)
  out_matrix: Output tensor (dim4, dim3, dim1, dim2)
  dim1, dim2, dim3, dim4: Dimensions of input tensor
*/
__global__ void permute_kernel(const float* matrix, float* out_matrix, int dim1, int dim2, int dim3, int dim4) {
    // Compute this thread's linear index into the tensor
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: only process if idx is valid
    // Total elements = dim1 × dim2 × dim3 × dim4
    if (idx < dim1 * dim2 * dim3 * dim4) {
        // ===================================================================
        // Step 1: Extract 4D indices from linear index
        // ===================================================================
        // Decompose linear index into (i1, i2, i3, i4) coordinates
        int i1 = (idx / (dim2 * dim3 * dim4)) % dim1;
        int i2 = (idx / (dim3 * dim4)) % dim2;
        int i3 = (idx / dim4) % dim3;
        int i4 = idx % dim4;

        // ===================================================================
        // Step 2: Compute target index for permuted layout
        // ===================================================================
        // Map (i1, i2, i3, i4) → (i4, i3, i1, i2) in new shape (dim4, dim3, dim1, dim2)
        int permuted_idx = i4 * (dim3 * dim1 * dim2) + i3 * (dim1 * dim2) + i1 * dim2 + i2;

        // ===================================================================
        // Step 3: Perform the copy
        // ===================================================================
        // Read from coalesced location, write to potentially scattered location
        out_matrix[permuted_idx] = matrix[idx];
    }
}


int main() {
    int dim_1 = 24;
    int dim_2 = 42;
    int dim_3 = 20;
    int dim_4 = 32;

    // Set up the device
    int deviceIdx = 0;
    cudaSetDevice(deviceIdx);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // Allocate host memory
    float* matrix = make_random_float(dim_1 * dim_2 * dim_3 * dim_4);
    float* permuted_matrix = (float*)malloc(dim_1 * dim_2 * dim_3 * dim_4 * sizeof(float));

    // Initialize the matrix with random values

    // Allocate device memory
    float *d_matrix, *d_permuted_matrix;
    cudaMalloc(&d_matrix, dim_1 * dim_2 * dim_3 * dim_4 * sizeof(float));
    cudaMalloc(&d_permuted_matrix, dim_1 * dim_2 * dim_3 * dim_4 * sizeof(float));

    // Copy matrix from host to device
    cudaMemcpy(d_matrix, matrix, dim_1 * dim_2 * dim_3 * dim_4 * sizeof(float), cudaMemcpyHostToDevice);

    // Perform permutation on CPU
    clock_t start = clock();
    permute_cpu(matrix, permuted_matrix, dim_1, dim_2, dim_3, dim_4);
    clock_t end = clock();
    double elapsed_time_cpu = (double)(end - start) / CLOCKS_PER_SEC;

    // Define block and grid sizes
    dim3 blockSize(256);
    int totalThreads = dim_1 * dim_2 * dim_3 * dim_4;
    int gridSize = (totalThreads + blockSize.x - 1) / blockSize.x; // Compute grid size

    // Launch CUDA kernel to perform permutation
    permute_kernel<<<gridSize, blockSize>>>(d_matrix, d_permuted_matrix, dim_1, dim_2, dim_3, dim_4);
    cudaDeviceSynchronize(); // Ensure kernel execution is complete

    // Verify results
    printf("Checking correctness...\n");
    validate_result(d_permuted_matrix, permuted_matrix, "permuted_matrix", dim_1 * dim_2 * dim_3 * dim_4, 1e-5f);

    printf("All results match.\n\n");
    // benchmark kernel
    int repeat_times = 1000;
    float elapsed_time = benchmark_kernel(repeat_times, permute_kernel,
                                          d_matrix, d_permuted_matrix, dim_1, dim_2, dim_3, dim_4
    );
    printf("time gpu %.4f ms\n", elapsed_time);
    printf("time cpu %.4f ms\n", elapsed_time_cpu);

    // Free allocated memory
    free(matrix);
    free(permuted_matrix);
    cudaFree(d_matrix);
    cudaFree(d_permuted_matrix);

    return 0;
}
