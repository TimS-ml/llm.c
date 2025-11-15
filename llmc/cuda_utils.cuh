/*
================================================================================
File: cuda_utils.cuh
Purpose: Device-side CUDA utilities for high-performance kernel implementations
================================================================================

Overview:
---------
This header provides essential utilities for writing efficient CUDA kernels.
It includes vectorized memory access, type conversion, warp/block-level
communication primitives, and memory allocation helpers.

Key Components:
--------------
1. Packed128: 128-bit vectorized memory operations (4x throughput vs scalar)
2. DType: Runtime type identification and size queries
3. Copy/Cast: Type conversion kernels between FP32/FP16/BF16
4. Warp Reductions: Fast intra-warp sum/max using shuffle instructions
5. Block Reductions: Scalable block-wide reductions using shared memory
6. Memory Allocation: Fallback to managed memory when device memory exhausted
7. Stochastic Rounding: High-quality random number generation for BF16 conversion

Design Philosophy:
-----------------
- Maximize memory bandwidth through vectorized access (Packed128)
- Minimize synchronization overhead (warp-level primitives)
- Provide generic implementations that work across precisions
- Deterministic operations where needed (single-block reductions)

CUDA Concepts Explained:
-----------------------
- Warp: 32 threads that execute in lockstep, can communicate without atomics
- Shuffle: Fast register exchange within a warp (1-cycle latency)
- Shared Memory: Per-block on-chip memory, ~100x faster than global memory
- Coalesced Access: Aligned, contiguous memory access pattern (full bandwidth)
- Streaming Stores: Cache hints to bypass L1 for write-only data

Note: This is a .cuh file (CUDA header), intended for inclusion in .cu files.
      All functions marked __device__ can only be called from GPU kernels.
*/

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include "cuda_common.h"

// ============================================================================
// PACKED128: 128-BIT VECTORIZED MEMORY ACCESS
// ============================================================================
// Modern GPUs can load/store 128 bits (16 bytes) in a single instruction
// (LDG.128, STS.128). This provides 4x higher throughput compared to scalar
// loads/stores, which is critical for memory-bound kernels.
//
// Why Vectorized Access Matters:
// ------------------------------
// - Memory bandwidth is the primary bottleneck in many kernels
// - Scalar loads waste bus bandwidth due to alignment requirements
// - 128-bit access fully utilizes the memory transaction size
// - Reduces instruction count and register pressure
//
// Comparison to float4:
// --------------------
// - float4 only works for FP32 (4 floats = 16 bytes)
// - Packed128 works for any type: FP32 (4 elements), BF16 (8 elements), FP16 (8 elements)
// - Provides a uniform interface across precision modes
//
// Usage Requirements:
// ------------------
// - Address must be 16-byte aligned
// - Access size must be multiple of 16 bytes
// - Best for bulk data transfers in kernels
//
// Example:
//     __global__ void copy_kernel(floatX* dst, const floatX* src, int n) {
//         int idx = blockIdx.x * blockDim.x + threadIdx.x;
//         if (idx * x128::size < n) {
//             x128 data = load128(src + idx * x128::size);
//             store128(dst + idx * x128::size, data);
//         }
//     }
// ============================================================================

/**
 * Packed128 - 128-bit aligned data structure for vectorized memory access
 *
 * @tparam ElementType: The underlying data type (float, __nv_bfloat16, half, etc.)
 *
 * This structure wraps an array of ElementType with 16-byte alignment, ensuring
 * the compiler uses 128-bit load/store instructions. The size is computed at
 * compile time based on ElementType:
 * - float: 4 elements (4 bytes each)
 * - __nv_bfloat16: 8 elements (2 bytes each)
 * - half: 8 elements (2 bytes each)
 */
template<class ElementType>
struct alignas(16) Packed128 {
    Packed128() = default;
    __device__ explicit Packed128(int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }

    __device__  static Packed128 constant(ElementType value) {
        Packed128 result;
        for(int k = 0; k < size; ++k) {
            result.payload[k] = value;
        }
        return result;
    }
    __device__ static Packed128 zeros() {
        return constant(0.f);
    }
    __device__ static Packed128 ones() {
        return constant(1.f);
    }

    __device__ ElementType& operator[](int index) {
        return payload[index];
    }
    __device__ const ElementType& operator[](int index) const {
        return payload[index];
    }
    __device__ int4 get_bits() const {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }
    static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);
    ElementType payload[size];  // Array of elements, total size = 16 bytes
};

/**
 * load128 - Loads 128 bits from aligned memory
 *
 * @param address: Pointer to aligned memory (must be 16-byte aligned!)
 * @return: Packed128 containing the loaded data
 *
 * Uses a single LDG.128 instruction for maximum bandwidth.
 * The reinterpret_cast to int4* tells the compiler to use vectorized load.
 */
template<class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}

/**
 * load128cs - Loads 128 bits with streaming cache hint
 *
 * @param address: Pointer to aligned memory (must be 16-byte aligned!)
 * @return: Packed128 containing the loaded data
 *
 * Uses __ldcs (load cache streaming) which bypasses L1 cache. Best for data
 * that will only be accessed once, reducing cache pollution.
 */
template<class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address) {
    return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
}

/**
 * store128 - Stores 128 bits to aligned memory
 *
 * @param target: Pointer to aligned memory (must be 16-byte aligned!)
 * @param value: Packed128 containing data to store
 *
 * Uses a single STS.128 instruction for maximum bandwidth.
 */
template<class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}

/**
 * store128cs - Stores 128 bits with streaming cache hint
 *
 * @param target: Pointer to aligned memory (must be 16-byte aligned!)
 * @param value: Packed128 containing data to store
 *
 * Uses __stcs (store cache streaming) which bypasses L1 cache. Best for
 * write-only data that won't be read back soon.
 */
template<class ElementType>
__device__ void store128cs(ElementType* target, Packed128<ElementType> value) {
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}

/**
 * store128cg - Stores 128 bits with cache-global hint
 *
 * @param target: Pointer to aligned memory (must be 16-byte aligned!)
 * @param value: Packed128 containing data to store
 *
 * Uses __stcg which caches data in L2 but bypasses L1. Good for data that
 * might be reused but not immediately (e.g., by other thread blocks).
 */
template<class ElementType>
__device__ void store128cg(ElementType* target, Packed128<ElementType> value) {
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}

/**
 * Convenient typedefs for common Packed128 instantiations:
 * - f128: Packed128<float> (4 FP32 values)
 * - x128: Packed128<floatX> (4 or 8 values depending on precision mode)
 */
typedef Packed128<float> f128;
typedef Packed128<floatX> x128;

// ============================================================================
// DTYPE SUPPORT: RUNTIME TYPE IDENTIFICATION
// ============================================================================
// Provides runtime type information for tensors, enabling generic code that
// handles multiple precision modes. Useful when precision is determined at
// runtime rather than compile time.
// ============================================================================

/**
 * DType - Enumeration for tensor data types
 *
 * Used to tag tensors with their precision at runtime, enabling:
 * - Generic tensor operations that dispatch based on type
 * - Mixed-precision workflows (different tensors with different types)
 * - Type-checking and validation
 */
enum class DType : uint8_t {
    FP32,  // 32-bit floating point
    FP16,  // 16-bit half precision
    BF16   // 16-bit bfloat16
};

/**
 * sizeof_dtype - Returns the size in bytes of a scalar of the given type
 *
 * @param type: DType enumeration value
 * @return: Size in bytes (4 for FP32, 2 for FP16/BF16)
 *
 * Useful for computing buffer sizes and memory allocation amounts.
 */
size_t sizeof_dtype(DType type) {
    switch (type) {
        case DType::FP32:
            return sizeof(float);
        case DType::FP16:
            return sizeof(half);
        case DType::BF16:
            return sizeof(nv_bfloat16);
        default: // handle or get compiler warning
            fprintf(stderr, "Unknown datatype\n");
            exit(EXIT_FAILURE);
    }
}

/**
 * dtype_of - Overloaded functions to query DType from a pointer
 *
 * These allow compile-time dispatch based on pointer type.
 * Usage: DType t = dtype_of(my_float_ptr);
 */
DType dtype_of(float* f) { return DType::FP32; }
DType dtype_of(nv_bfloat16 * f) { return DType::BF16; }
DType dtype_of(half * f) { return DType::FP16; }


// ============================================================================
// COPY AND CAST FUNCTIONS: TYPE CONVERSION KERNELS
// ============================================================================
// Utilities for converting data between different precision formats.
// Useful for mixed-precision training, model export, and debugging.
// ============================================================================

/**
 * cast_value - Template function for type conversion between precisions
 *
 * @tparam Td: Destination type
 * @tparam Ts: Source type
 * @param val: Value to convert
 * @return: Converted value in destination type
 *
 * Provides specialized implementations for common conversions:
 * - float -> float (no-op)
 * - half -> float (use __half2float intrinsic)
 * - bfloat16 -> float (use __bfloat162float intrinsic)
 */
template<typename Td, typename Ts>
__device__ Td cast_value(Ts val);

template<>
__device__ float cast_value<float, float>(float val) {
    return val;
}

template<>
__device__ float cast_value<float, half>(half val) {
    return __half2float(val);
}

template<>
__device__ float cast_value<float, __nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

/**
 * copy_and_cast_kernel - GPU kernel for copying and converting data between types
 *
 * @tparam Td: Destination type
 * @tparam Ts: Source type
 * @param dst: Destination array
 * @param src: Source array
 * @param n: Number of elements per row
 * @param stride_dst: Stride between rows in destination
 * @param stride_src: Stride between rows in source
 *
 * This kernel supports batched conversion with strided access, useful for
 * converting multiple sequences or batch elements in parallel.
 */
template<typename Td, typename Ts>
__global__ void copy_and_cast_kernel(Td* dst, const Ts* src, size_t n, ptrdiff_t stride_dst, ptrdiff_t stride_src) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // need to try grid stride looping for more perf later
    if (idx < n) {
        dst[idx + stride_dst * blockIdx.y] = cast_value<Td, Ts>(src[idx + stride_src * blockIdx.y]);
    }
}

// ============================================================================
// WARP/BLOCK COMMUNICATION PRIMITIVES
// ============================================================================
// High-performance reduction operations using CUDA's warp shuffle instructions
// and shared memory. These are fundamental building blocks for many kernels.
//
// CUDA Reduction Hierarchy:
// ------------------------
// 1. Warp-level (32 threads): Use shuffle instructions (no shared memory needed)
//    - Extremely fast: ~1 cycle per shuffle operation
//    - No synchronization needed (threads in a warp are implicitly synchronized)
//    - Limited to 32 threads
//
// 2. Block-level (up to 1024 threads): Combine warp + shared memory
//    - Each warp reduces independently using shuffles
//    - Warp results stored in shared memory
//    - Final warp reduces the shared memory values
//    - Requires __syncthreads() for correctness
//
// Why These Are Fast:
// -----------------
// - Shuffle: Direct register exchange, no memory access
// - Shared memory: ~100x faster than global memory
// - Tree reduction: O(log N) steps instead of O(N)
//
// Common Use Cases:
// ----------------
// - Computing sums (gradients, statistics)
// - Finding max/min (softmax, normalization)
// - Computing norms (gradient clipping)
// - Reductions as part of larger kernels (attention, layernorm)
// ============================================================================

/**
 * warpReduceSum - Sum reduction across a warp using shuffle instructions
 *
 * @param val: Input value from this thread
 * @return: Sum of all values across the warp (same on all threads)
 *
 * Algorithm (tree reduction using XOR butterfly pattern):
 * 1. Each thread starts with its value
 * 2. offset=16: Each thread adds value from thread 16 positions away
 * 3. offset=8: Each thread adds value from thread 8 positions away
 * 4. offset=4, 2, 1: Continue halving offset
 * 5. After 5 steps, all threads hold the warp sum
 *
 * Example (8 threads for simplicity):
 *   Initial: [1, 2, 3, 4, 5, 6, 7, 8]
 *   offset=4: [1+5, 2+6, 3+7, 4+8, 5+1, 6+2, 7+3, 8+4] = [6,8,10,12,6,8,10,12]
 *   offset=2: [6+10, 8+12, 10+6, 12+8, ...] = [16,20,16,20,16,20,16,20]
 *   offset=1: [16+20, 20+16, ...] = [36,36,36,36,36,36,36,36]
 *
 * Performance: ~5 cycles total for 32 threads!
 */
__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * warpReduceMax - Max reduction across a warp using shuffle instructions
 *
 * @param val: Input value from this thread
 * @return: Maximum value across the warp (same on all threads)
 *
 * Same algorithm as warpReduceSum, but using fmaxf instead of addition.
 * Useful for finding the maximum value in softmax, normalization, etc.
 */
__device__ inline float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/**
 * blockReduce - Generic reduction across an entire thread block
 *
 * @tparam warp_reduction: Warp-level reduction function (warpReduceSum or warpReduceMax)
 * @param val: Input value from this thread
 * @param final_sync: Whether to add final __syncthreads() (needed if called in loop)
 * @param out_of_bounds: Value to use for threads beyond num_warps (e.g., 0 for sum, -inf for max)
 * @return: Reduced value across all threads in the block (same on all threads)
 *
 * Three-stage reduction:
 * 1. Intra-warp reduction: Each warp reduces its 32 values using shuffles
 * 2. Cross-warp reduction: Lane 0 of each warp writes result to shared memory
 * 3. Final reduction: First warp reads shared values and reduces them
 *
 * Shared Memory Usage:
 * - Uses 128 bytes (32 floats) of static shared memory per call
 * - If called in a loop, set final_sync=true to avoid conflicts
 * - Multiple calls in the same function use separate shared memory instances
 *
 * Performance (1024 threads):
 * - Step 1: 32 warps × 5 cycles = ~5 cycles (parallel)
 * - Step 2: 32 writes to shared memory
 * - Step 3: 1 warp × 5 cycles = ~5 cycles
 * - Total: ~10-15 cycles for 1024-way reduction!
 *
 * Requirements:
 * - All 32 threads in each warp must participate
 * - Block size should be a multiple of 32
 */
using reduction_func_t = float (*) (float);
template<reduction_func_t warp_reduction>
__device__ inline float blockReduce(float val, bool final_sync=false, float out_of_bounds=0.0f) {
    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    __shared__ float shared_val[WARP_SIZE];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    float warp_val = warp_reduction(val);
    if (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    float block_val = warp_reduction(warp_val);

    if (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    return block_val;
}

/**
 * global_sum_single_block_kernel - Deterministic sum reduction using a single block
 *
 * @tparam Float: Input data type (float, half, or bfloat16)
 * @param result: Output pointer for the sum (single float value)
 * @param values: Input array to sum
 * @param count: Number of elements in the input array
 *
 * This kernel achieves deterministic results by using only a single thread block,
 * which ensures that all additions happen in the same order every time. This is
 * important for:
 * - Reproducible training runs (same results with same random seed)
 * - Debugging numerical issues
 * - Verifying correctness against reference implementations
 *
 * Non-determinism in multi-block reductions:
 * - Different blocks may finish in different orders
 * - Floating-point addition is not associative: (a+b)+c != a+(b+c)
 * - Order changes -> slightly different results -> divergent training
 *
 * Performance trade-off:
 * - Single block: Slow for large arrays, but deterministic
 * - Multi-block: Fast but non-deterministic
 * - Use this for critical reductions (e.g., gradient norms, loss computation)
 *
 * Algorithm:
 * 1. Each thread sums a subset of elements using grid-stride loop
 * 2. Block reduction combines all thread sums
 * 3. Thread 0 writes final result
 */
template<class Float>
__global__ void global_sum_single_block_kernel(float* result, const Float* values, size_t count) {
    assert(gridDim.x == 1);     // only a single block!
    float thread_sum = 0;
    for(size_t index = threadIdx.x; index < count; index += blockDim.x) {
        thread_sum += (float)values[index];
    }

    float reduction = blockReduce<warpReduceSum>(thread_sum, true);
    if(threadIdx.x == 0) {
        *result = reduction;
    }
}

template<class Float>
void global_sum_deterministic(float* result, const Float* values, int count, cudaStream_t stream) {
    global_sum_single_block_kernel<<<1, 1024, 0, stream>>>(result, values, count);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================
// Provides allocation functions with automatic fallback to managed memory
// when device memory is exhausted. This improves robustness when running
// large models that might exceed available GPU memory.
// ============================================================================

/**
 * cudaMallocConditionallyManaged - Allocates GPU memory with managed fallback
 *
 * @param out: Output pointer (will be set to allocated memory)
 * @param bytes: Number of bytes to allocate
 * @param file: Source file name (automatically provided by macro)
 * @param line: Line number (automatically provided by macro)
 * @return: 0 if allocated on device, 1 if fell back to managed memory
 *
 * Allocation strategy:
 * 1. First, try cudaMalloc (fastest, device-only memory)
 * 2. If OOM, fallback to cudaMallocManaged (slower, but allows CPU access)
 * 3. Set preference for managed memory to stay on device when possible
 *
 * Managed Memory (Unified Memory):
 * - Automatically migrates between CPU and GPU as needed
 * - Slower than device memory due to migration overhead
 * - Allows running models larger than GPU memory (using CPU memory as backup)
 * - Good fallback when device memory is exhausted
 *
 * Return value usage:
 * - Caller can check if managed memory was used
 * - Can log warnings or adjust batch size if fallback occurs
 * - Helps diagnose memory pressure issues
 *
 * Performance impact:
 * - Device memory: Full bandwidth (~900 GB/s on A100)
 * - Managed memory: Reduced bandwidth due to PCIe transfers
 *
 * Note: Called via the cudaMallocConditionallyManaged macro, not directly.
 */
int cudaMallocConditionallyManaged(void** out, size_t bytes, const char *file, int line) {
    // try to allocate
    cudaError_t err = cudaMalloc(out, bytes);
    if(err == cudaErrorMemoryAllocation) {
        // if we OOM, fallback to a managed allocation. slower but at least won't crash.
        cudaGetLastError(); // reset the error before the next API call
        cudaCheck_(cudaMallocManaged(out, bytes), file, line);
        cudaCheck_(cudaMemAdvise(*out, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId), file, line);
        return 1;
    } else {
        cudaCheck_(err, file, line);
        return 0;
    }
}

/**
 * cudaMallocConditionallyManaged - Macro wrapper
 *
 * Usage: cudaMallocConditionallyManaged(&d_ptr, num_bytes);
 */
#define cudaMallocConditionallyManaged(out, bytes)\
(cudaMallocConditionallyManaged((void**)out, bytes, __FILE__, __LINE__))

// ============================================================================
// RANDOM NUMBER GENERATION FOR STOCHASTIC ROUNDING
// ============================================================================
// Provides high-quality random number generation for stochastic rounding when
// converting FP32 -> BF16. Stochastic rounding helps prevent gradient
// accumulation errors and improves training stability in low precision.
//
// Why Stochastic Rounding?
// -----------------------
// Standard rounding (round-to-nearest):
// - Small gradients (< 2^-8) often round to zero
// - Leads to "dead" weights that never update
// - Accumulates bias over many updates
//
// Stochastic rounding:
// - Small values have probability of rounding up or down
// - Expected value = true value (unbiased)
// - Prevents gradient accumulation errors
// - Critical for training in BF16
//
// SquirrelNoise5:
// --------------
// A fast, high-quality hash function that generates pseudo-random numbers
// from thread coordinates and a global seed. Benefits:
// - Deterministic: Same input -> same output
// - No state: Each thread generates random numbers independently
// - High quality: Passes statistical randomness tests
// - Fast: Pure arithmetic, no memory access
//
// Source: http://eiserloh.net/noise/SquirrelNoise5.hpp
// ============================================================================

/**
 * SquirrelNoise5 - Fast hash function for high-quality pseudo-random numbers
 *
 * @param positionX: Input position (typically thread index)
 * @param seed: Random seed (changed per training step)
 * @return: Pseudo-random 32-bit unsigned integer
 *
 * This hash function uses bit-mixing operations to generate random-looking
 * output from structured input. Each thread can generate independent random
 * numbers by using its thread/block index as positionX.
 *
 * Properties:
 * - Deterministic: f(x, seed) always returns same value
 * - Avalanche effect: Small input change -> completely different output
 * - Uniform distribution: All output values equally likely
 * - No collisions: Different inputs -> different outputs (usually)
 */
__device__ __host__ constexpr unsigned int SquirrelNoise5(unsigned int positionX, unsigned int seed)
{
    constexpr unsigned int SQ5_BIT_NOISE1 = 0xd2a80a3f;	// 11010010101010000000101000111111
    constexpr unsigned int SQ5_BIT_NOISE2 = 0xa884f197;	// 10101000100001001111000110010111
    constexpr unsigned int SQ5_BIT_NOISE3 = 0x6C736F4B; // 01101100011100110110111101001011
    constexpr unsigned int SQ5_BIT_NOISE4 = 0xB79F3ABB;	// 10110111100111110011101010111011
    constexpr unsigned int SQ5_BIT_NOISE5 = 0x1b56c4f5;	// 00011011010101101100010011110101
    unsigned int mangledBits = positionX;
    mangledBits *= SQ5_BIT_NOISE1;
    mangledBits += seed;
    mangledBits ^= (mangledBits >> 9);
    mangledBits += SQ5_BIT_NOISE2;
    mangledBits ^= (mangledBits >> 11);
    mangledBits *= SQ5_BIT_NOISE3;
    mangledBits ^= (mangledBits >> 13);
    mangledBits += SQ5_BIT_NOISE4;
    mangledBits ^= (mangledBits >> 15);
    mangledBits *= SQ5_BIT_NOISE5;
    mangledBits ^= (mangledBits >> 17);
    return mangledBits;
}

/**
 * Get2dNoiseUint - Generates random number from 2D coordinates
 *
 * @param indexX: X coordinate (e.g., threadIdx.x)
 * @param indexY: Y coordinate (e.g., blockIdx.x * blockDim.x + blockIdx.y)
 * @param seed: Random seed
 * @return: Pseudo-random 32-bit unsigned integer
 *
 * Combines 2D thread coordinates into a single random number using SquirrelNoise5.
 * Each unique (x, y, seed) combination produces a different random value.
 */
__device__ __host__ constexpr unsigned int Get2dNoiseUint(int indexX, int indexY, unsigned int seed)
{
    constexpr unsigned int PRIME_NUMBER = 198491317u; // Large prime number with non-boring bits
    unsigned int x = static_cast<unsigned int>(indexX);
    unsigned int y = static_cast<unsigned int>(indexY);

    return SquirrelNoise5(x + (PRIME_NUMBER * y), seed);
}

/**
 * stochastic_rounding - Converts FP32 to BF16 using stochastic rounding
 *
 * @param in: Input FP32 value
 * @param out: Output BF16 value (written by this function)
 * @param seed: Random seed (should be updated each training step)
 *
 * Stochastic Rounding Algorithm:
 * ------------------------------
 * BF16 has 16 bits total: 1 sign + 8 exponent + 7 mantissa
 * FP32 has 32 bits total: 1 sign + 8 exponent + 23 mantissa
 *
 * Conversion truncates the lower 16 bits of FP32. Instead of always rounding
 * down, we:
 * 1. Generate a random threshold in [0, 65535] (16 bits)
 * 2. Extract the 16 bits that would be truncated
 * 3. If truncated bits > threshold, round up; else round down
 *
 * This ensures:
 * - Expected value of rounded number = original value
 * - No systematic bias (unlike round-to-nearest)
 * - Gradients don't "disappear" in low precision
 *
 * Example:
 *   Original: 1.000001 (in FP32)
 *   Truncated bits represent: 0.000001
 *   Random threshold: uniform in [0, max_representable_fraction]
 *   Round up with probability ≈ 0.000001
 *   Round down with probability ≈ 0.999999
 *   -> Expected value preserved!
 *
 * Performance note: Each thread gets a unique random number based on its
 * position, ensuring different threads don't use the same random values.
 */
__device__ __forceinline__ void stochastic_rounding(float in, __nv_bfloat16 *out, unsigned int seed) {
    // makes sure each thread gets a different random number
    unsigned int random = Get2dNoiseUint(threadIdx.x, blockIdx.x * blockDim.x + blockIdx.y, seed);
    unsigned int threshold = random & 0xFFFF;
    unsigned int float_bits = __float_as_uint(in);
    unsigned int rounded_bits = float_bits & 0x0000FFFF;
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits  & ~0xFFFF);
    *out = __float2bfloat16_rn(__uint_as_float(float_bits));
}
__device__ __forceinline__ void stochastic_rounding(float in, half *out, unsigned int random) {
    *out = (float)in; // todo - implement this...
}
__device__ __forceinline__ void stochastic_rounding(float in, float *out, unsigned int random) {
    *out = in; // dummy function for when floatX is float (FP32 mode)
}

#endif