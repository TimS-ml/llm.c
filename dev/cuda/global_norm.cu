/*
================================================================================
Global Norm CUDA Kernels
================================================================================

PURPOSE:
--------
Computes the L2 norm (sum of squares) of a large tensor using all available
GPU streaming multiprocessors (SMs) cooperatively. This is essential for
gradient clipping in deep learning training.

WHAT IS GLOBAL NORM?
--------------------
In deep learning, "global norm" typically refers to the L2 norm of ALL
gradients concatenated together:

  global_norm = sqrt(sum(g1² + g2² + ... + gN²))

where g1, g2, ..., gN are all the gradient tensors.

This kernel computes: sum(x²) for all elements x in the input tensor.
The square root is typically taken afterward on the CPU.

WHY GLOBAL NORM?
----------------
1. Gradient Clipping: Prevents exploding gradients in training
   If global_norm > threshold, scale all gradients by (threshold/global_norm)

2. Monitoring: Track gradient magnitude to detect training issues
   Too small → vanishing gradients
   Too large → exploding gradients

THE CHALLENGE:
--------------
Unlike per-row or per-channel norms where each block can work independently,
global norm requires reducing across the ENTIRE tensor. This creates a
coordination challenge:
  - Must minimize atomic operations (slow!)
  - Must utilize ALL SMs efficiently
  - Must avoid thread divergence

KERNEL IMPLEMENTATIONS:
-----------------------
This file contains 4 kernel versions with different reduction strategies:

Kernel 1 (norm_kernel1):
  - Block-level reduction using cooperative groups
  - One atomic add per block
  - Uses shared memory for intra-block reduction

Kernel 2 (norm_kernel2):
  - Warp-level reduction only (no shared memory)
  - One atomic add per warp (more atomics, less shared memory)
  - Simpler but more contention

Kernel 3 (norm_kernel3):
  - Block-level reduction using custom blockReduce function
  - One atomic add per block
  - Similar to kernel 1 but different reduction implementation

Kernel 4 (norm_kernel4):
  - Two-stage reduction for determinism
  - Each block writes partial sum to output array (no atomics in first stage)
  - Second kernel aggregates partial sums
  - Deterministic results (same order of operations every time)

PERFORMANCE CHARACTERISTICS:
----------------------------
- Memory bandwidth bound (each element read once)
- Atomic operations are the bottleneck
- Kernel 4 trades atomic performance for determinism
- All kernels use grid-stride loop to process data

NUMERICAL PRECISION:
--------------------
- Input can be BF16/FP16, but accumulation is in FP32
- Important for numerical stability with large tensors
- Final result is FP32

Compile example:
nvcc -O3 --use_fast_math global_norm.cu -o global_norm

Run:
./global_norm 1  # Run kernel 1
./global_norm 2  # Run kernel 2
./global_norm 3  # Run kernel 3
./global_norm 4  # Run kernel 4 (deterministic)
*/


#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

// turn on bf16 as default, done up here for now
#define ENABLE_BF16
#include "common.h"

// Global device properties, set in main()
cudaDeviceProp deviceProp;

/*
CPU reference implementation for validation.

Computes sum of squares: result = x1² + x2² + ... + xN²

Uses double precision accumulation to minimize numerical error and provide
an accurate reference for validating GPU implementations.

Parameters:
  data: Input array (on CPU)
  count: Number of elements

Returns:
  Sum of squares (NOT the square root - caller takes sqrt if needed)
*/
float global_norm_cpu(const float* data, size_t count) {
    // Accumulate in double precision for accurate reference
    // This avoids accumulation errors that occur with float
    double acc = 0.0;
    for(size_t i = 0; i < count; ++i) {
        acc  += (double)data[i] * (double)data[i];
    }
    return (float)acc;
}


/*
================================================================================
Kernel 1: Block-Level Reduction with Cooperative Groups
================================================================================

STRATEGY:
---------
1. Each thread processes multiple elements using grid-stride loop
2. Reduce within each warp using cooperative groups shuffle operations
3. Reduce across warps using shared memory
4. One thread per block atomically adds block sum to global output

KEY OPTIMIZATIONS:
------------------
- Grid-stride loop: Each block processes maximum amount of data
- Two-level reduction: warp-level (fast) then block-level
- Shared memory: Only 32 floats (one per warp), minimal usage
- One atomic per block: Minimizes atomic contention

GRID-STRIDE LOOP:
-----------------
Instead of each thread processing one element, threads iterate with
stride = grid_width to process many elements. Benefits:
  - Better hardware utilization
  - Allows flexible grid size (doesn't need to match data size)
  - Improves L2 cache reuse

COOPERATIVE GROUPS:
-------------------
Uses cg::reduce for warp-level reduction:
  - Hardware-accelerated shuffle operations
  - No shared memory needed for warp reduce
  - Very fast (single instruction)

MEMORY ACCESS:
--------------
- Coalesced reads (consecutive threads read consecutive addresses)
- Template parameter T allows BF16/FP16 input (converted to FP32 for compute)

Parameters:
  out: Output location for sum (single float, will be atomically updated)
  data: Input tensor
  count: Number of elements in data
*/
template<class T>
__global__ void norm_kernel1(float* out, const T* data, size_t count) {
    // Set up cooperative groups for efficient reduction
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Shared memory for storing warp-level results
    // One float per warp in the block (typically 32 warps max → 128 bytes)
    __shared__ float block_result[32];

    // ===================================================================
    // Phase 1: Thread-level accumulation (grid-stride loop)
    // ===================================================================
    size_t index = threadIdx.x + blockDim.x * blockIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;  // Total threads in grid
    float accumulator = 0.f;

    // Grid-stride loop: process elements at indices index, index+grid_width, index+2*grid_width, ...
    for(size_t i = index; i < count; i += grid_width) {
        // Convert to float (handles BF16/FP16 input) and square
        accumulator += (float)data[i] * (float)data[i];
    }

    // ===================================================================
    // Phase 2: Warp-level reduction
    // ===================================================================
    // Reduce accumulator across all threads in this warp using shuffle operations
    // After this, lane 0 of each warp holds the sum for that warp
    float warp_result = cg::reduce(warp, accumulator, cg::plus<float>{});

    // Warp leader writes warp sum to shared memory
    // meta_group_rank() is the warp ID within the block (0, 1, 2, ...)
    block_result[warp.meta_group_rank()] = warp_result;

    // Synchronize to ensure all warps have written their results
    block.sync();

    // ===================================================================
    // Phase 3: Block-level reduction (warp 0 only)
    // ===================================================================
    // Only the first warp performs the final reduction
    if(warp.meta_group_rank() == 0) {
        // Each thread in warp 0 loads one warp's result (or 0 if out of bounds)
        // meta_group_size() is the number of warps in the block
        float gather = warp.thread_rank() < warp.meta_group_size() ?
                       block_result[warp.thread_rank()] : 0.f;

        // Reduce across warp 0 to get final block sum
        float block_sum = cg::reduce(warp, gather, cg::plus<float>{});

        // ===================================================================
        // Phase 4: Atomic update (one thread per block)
        // ===================================================================
        // Thread 0 of warp 0 atomically adds this block's sum to global output
        if(warp.thread_rank() == 0) {
            atomicAdd(out, block_sum);
        }
    }
}

/*
================================================================================
Kernel 2: Warp-Level Reduction Only (No Shared Memory)
================================================================================

STRATEGY: Simpler approach - warp reduce, then each warp atomically updates output

TRADE-OFFS:
-----------
Pros:
  - No shared memory usage
  - Simpler code
  - No block-level synchronization

Cons:
  - More atomic operations (one per warp instead of one per block)
  - More contention on global atomic
  - Typically slower than kernel1

Example for A100 (108 SMs, 2048 max threads/SM):
  - Total threads: 2048 × 108 = 221,184
  - With block_size=512: 432 blocks
  - Warps: 221,184/32 = 6,912 warps
  - Atomic operations: 6,912 (vs 432 in kernel1)

For ~100M parameters: each thread processes ~500 elements
*/
template<class T>
__global__ void norm_kernel2(float* out, const T* data, size_t count) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Grid-stride loop for thread-level accumulation
    size_t index = threadIdx.x + blockDim.x * blockIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }

    // Warp-level reduction using cooperative groups
    float warp_result = cg::reduce(warp, accumulator, cg::plus<float>{});

    // Each warp leader directly atomically adds to output (more atomics than kernel1)
    if(warp.thread_rank() == 0) {
        atomicAdd(out, warp_result);
    }
}

/*
================================================================================
Kernel 3: Block Reduction Using Custom blockReduce Function
================================================================================

STRATEGY: Similar to kernel1 but uses custom blockReduce helper (from common.h)

This kernel demonstrates using the reusable blockReduce template function
instead of cooperative groups. Functionally equivalent to kernel1.

The blockReduce function:
  - Performs warp-level reduction via warpReduceSum
  - Uses shared memory for cross-warp reduction
  - One atomic add per block (like kernel1)

Performance: Similar to kernel1
*/
template<class T>
__global__ void norm_kernel3(float* out, const T* data, size_t count) {
    // Grid-stride loop for accumulation
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }

    // Use custom block-level reduction (defined in common.h)
    // Template parameter specifies warp reduction function
    float block_sum = blockReduce<warpReduceSum>(accumulator);

    // Thread 0 atomically adds block result
    if(threadIdx.x == 0) {
        atomicAdd(out, block_sum);
    }
}

/*
================================================================================
Kernel 4: Two-Stage Deterministic Reduction
================================================================================

MOTIVATION: Achieve deterministic results

PROBLEM WITH ATOMICS:
----------------------
Atomic operations can execute in ANY order (non-deterministic), leading to:
  - Different rounding errors in different runs
  - Non-reproducible results (even with same input and seed)
  - Due to floating-point non-associativity: (a+b)+c ≠ a+(b+c)

SOLUTION: Two-stage approach
----------------------------
Stage 1 (norm_kernel4):
  - Each block computes partial sum (no atomics!)
  - Writes result to out[blockIdx.x]
  - Deterministic within each block

Stage 2 (global_norm_aggregate_kernel):
  - Single block reads all partial sums
  - Reduces them in a fixed order
  - Writes final result to out[0]

This ensures the SAME ORDER of additions every run → deterministic result

Performance: Similar to kernel3 (slightly slower due to extra kernel launch overhead)
*/
template<class T>
__global__ void norm_kernel4(float* out, const T* data, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }

    // Block-level reduction
    float block_sum = blockReduce<warpReduceSum>(accumulator);

    // Each block writes its partial sum to a unique location (NO ATOMICS!)
    // This is the key difference from kernel3
    if(threadIdx.x == 0) {
        out[blockIdx.x] = block_sum;  // Deterministic write, no race conditions
    }
}

/*
Second stage kernel: Aggregates partial sums from kernel4.

Launched with a single block that reads all partial sums and reduces them.
Since there's only one block, the order of operations is deterministic.

Parameters:
  out: Array containing partial sums from each block (in), final result (out[0])
  count: Number of partial sums (= number of blocks in first kernel)
*/
__global__ void global_norm_aggregate_kernel(float* out, size_t count) {
    size_t index = threadIdx.x;

    // Each thread loads one partial sum (or 0 if out of bounds)
    float block_sum = (index < count) ? out[index] : 0.f;

    // Reduce all partial sums in a deterministic order
    float sum = blockReduce<warpReduceSum>(block_sum);

    // Thread 0 writes the final result
    if(threadIdx.x == 0) {
        out[0] = sum;  // Final norm squared value
    }
}

// ----------------------------------------------------------------------------
// Kernel Launchers
// ----------------------------------------------------------------------------

/*
Launch configuration strategy: Fill the GPU completely

GRID SIZE CALCULATION:
----------------------
We want to utilize all SMs (streaming multiprocessors) on the GPU.
  - cuda_num_SMs: Number of SMs on this GPU (e.g., 108 for A100)
  - cuda_threads_per_SM: Max threads per SM (e.g., 2048 for A100)
  - Total threads possible: cuda_num_SMs × cuda_threads_per_SM

Grid size = Total threads / block_size

WHY NOT USE ceil_div?
---------------------
We deliberately use integer division (not ceiling):
  - One block too few: Tiny performance impact (slightly less parallelism)
  - One block too many: CATASTROPHIC (that block must wait for all others to finish first)

Since cuda_threads_per_SM is typically a multiple of common block sizes
(512, 1024), the division is usually exact anyway.

Note: cuda_num_SMs and cuda_threads_per_SM are global variables set in main()
from cudaDeviceProp.
*/
template<typename T>
void global_norm1(float* out, const T* values, size_t count, int block_size) {
    // Calculate grid size to fill all SMs
    const int grid_size = cuda_threads_per_SM * cuda_num_SMs / block_size;
    assert(grid_size > 0);      // Sanity check to catch configuration errors

    // Launch kernel with calculated grid size
    norm_kernel1<<<grid_size, block_size>>>(out, values, count);
    cudaCheck(cudaGetLastError());
}

template<typename T>
void global_norm2(float* out, const T* values, size_t count, int block_size) {
    // ditto
    const int grid_size = cuda_threads_per_SM * cuda_num_SMs / block_size;
    assert(grid_size > 0);      // gives a better error than letting the call below fail
    norm_kernel2<<<grid_size, block_size>>>(out, values, count);
    cudaCheck(cudaGetLastError());
}

template<typename T>
void global_norm3(float* out, const T* values, size_t count, int block_size) {
    // launch just enough blocks to fill the grid. deliberately no DIV_CEIL.
    // having one block less than possible is a tiny performance hit, having
    // one block too many is catastrophic, since it only can start once all the other
    // blocks finish. anyway, I think cuda_threads_per_SM should be a multiple of 512
    // on all gpus, so the division really is going to be exact.
    const int grid_size = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / block_size;
    assert(grid_size > 0);  // gives a better error than letting the call below fail
    norm_kernel3<<<grid_size, block_size>>>(out, values, count);
    cudaCheck(cudaGetLastError());
}

/*
Launcher for kernel4 (two-stage deterministic reduction)

SPECIAL CONSTRAINTS:
--------------------
1. grid_size must be < 1024:
   - Second kernel uses single block with 1024 threads
   - Each thread loads one partial sum
   - Therefore, can handle at most 1024 partial sums

2. Minimum block_size of 128:
   - Ensures we don't create too many blocks
   - If user requests smaller block_size, we override it

LAUNCH SEQUENCE:
----------------
1. First kernel: Compute partial sums (one per block)
2. Second kernel: Aggregate partial sums into final result
*/
template<typename T>
void global_norm4(float* out, const T* values, size_t count, int block_size) {
    // Enforce minimum block size to limit number of blocks
    if (block_size <= 64) {
        block_size = 128;
    }

    // Calculate grid size to fill GPU
    const int grid_size = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / block_size;
    assert(grid_size > 0);

    // CRITICAL: grid_size must fit in second kernel's single block
    assert(grid_size < 1024);  // Second kernel has 1024 threads max

    // Stage 1: Compute partial sums (one per block, written to out[0..grid_size-1])
    norm_kernel4<<<grid_size, block_size>>>(out, values, count);
    cudaCheck(cudaGetLastError());

    // Stage 2: Aggregate partial sums (reads out[0..grid_size-1], writes final result to out[0])
    global_norm_aggregate_kernel<<<1, 1024>>>(out, grid_size);
    cudaCheck(cudaGetLastError());
}

void global_norm(int kernel_num, float* out, const floatX* values, size_t count, int block_size) {
    switch (kernel_num) {
        case 1:
            return global_norm1(out, values, count, block_size);
        case 2:
            return global_norm2(out, values, count, block_size);
        case 3:
            return global_norm3(out, values, count, block_size);
        case 4:
            return global_norm4(out, values, count, block_size);
    }
}

int main(int argc, const char **argv) {
    setup_main();
    cudaGetDeviceProperties(&deviceProp, 0);

    int C = 768;
    int L = 12;

    size_t num_params = (size_t)(C * 4*C + C*C) * 2 * L;

    // create host memory of random numbers
    float* inp = make_random_float(num_params);
    // scale them down
    for(size_t i = 0; i < num_params; ++i) {
        inp[i] *= 1e-3;
    }

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    float out = global_norm_cpu(inp, num_params);

    // move to GPU
    float* d_out;
    floatX* d_inp;
    cudaCheck(cudaMalloc(&d_out,  1024 * sizeof(float)));  // 1024 needed for kernel 4
    cudaCheck(cudaMalloc(&d_inp, num_params * sizeof(floatX)));
    cudaCheck(memcpy_convert(d_inp, inp, num_params));

    int block_sizes[] = {32, 64, 128, 256, 512, 768, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        cudaCheck(cudaMemset(d_out, 0, sizeof(float)));
        global_norm(kernel_num, d_out, d_inp, num_params, block_size);
        validate_result(d_out, &out, "out", 1, 1e-2f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, global_norm,
                                              kernel_num, d_out, d_inp,
                                              num_params, block_size);
        size_t memory_ops = num_params * sizeof(floatX);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(inp);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
}