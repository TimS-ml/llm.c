/*
================================================================================
File: cuda_common.h
Purpose: Common utilities, definitions, and error-checking for CUDA code
================================================================================

Overview:
---------
This header file provides the foundational utilities and definitions needed by
all CUDA kernels and host code in the llm.c project. It establishes the runtime
environment for GPU computation including precision settings, error checking,
profiling integration, and data transfer utilities.

Key Components:
--------------
1. Global Settings: Device properties, warp size, thread block configurations
2. Error Checking: CUDA error checking macros and utilities
3. Precision Modes: Compile-time selection of FP32, FP16, or BF16
4. Cache Hints: Load/store operations with streaming cache hints
5. Profiling: NVTX integration for performance analysis
6. File I/O: Efficient CUDA memory <-> file transfer with double buffering

Design Philosophy:
-----------------
- Compile-time precision selection via preprocessor flags
- Zero-overhead abstractions through inlining and macros
- Consistent error checking across all CUDA API calls
- Architecture-specific optimizations (Volta, Ampere, Hopper, Ada)

Usage:
------
This header should be included in all CUDA source files. It's typically
included before any kernel implementations or CUDA API calls.

    #include "cuda_common.h"
    // Now you can use floatX, cudaCheck, PRECISION_MODE, etc.
*/
#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <type_traits>      // std::bool_constant
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#include <cuda_profiler_api.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "utils.h"

// ============================================================================
// GLOBAL DEFINES AND SETTINGS
// ============================================================================

/**
 * deviceProp - CUDA device properties for the current GPU
 *
 * This extern declaration allows all kernels to access device properties
 * (compute capability, SM count, memory size, etc.) without passing them
 * as parameters. The actual variable is defined in the main program file.
 *
 * Common uses:
 * - Query compute capability to enable architecture-specific optimizations
 * - Determine number of SMs for optimal grid sizing
 * - Check available memory before allocating large buffers
 */
extern cudaDeviceProp deviceProp;

/**
 * WARP_SIZE - Number of threads in a warp
 *
 * A warp is the fundamental unit of execution in CUDA. All threads in a warp
 * execute in lockstep (SIMT - Single Instruction, Multiple Thread).
 *
 * Value: 32 threads (constant across all NVIDIA GPUs to date)
 *
 * This is defined as a macro rather than using warpSize from deviceProp because:
 * 1. It's a compile-time constant, enabling better optimization
 * 2. It can be used in constant expressions and template parameters
 */
#define WARP_SIZE 32U

/**
 * MAX_1024_THREADS_BLOCKS - Maximum resident blocks for 1024-thread kernels
 *
 * This controls the __launch_bounds__ for kernels that use 1024 threads per block.
 * The value determines how many such blocks can be resident per SM simultaneously.
 *
 * Values:
 * - 2 for A100 (SM 8.0) and H100 (SM 9.0+): Allows 2 blocks/SM for better latency hiding
 * - 1 for older architectures: Limited resources only allow 1 block/SM
 *
 * Benefits of 2 blocks/SM:
 * - Better latency hiding through more concurrent warps
 * - Improved utilization when some warps are stalled on memory accesses
 *
 * Note: This must be a compile-time constant to use with __launch_bounds__
 */
#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#else
#define MAX_1024_THREADS_BLOCKS 1
#endif

/**
 * CEIL_DIV - Ceiling division macro for computing grid/block dimensions
 *
 * Usage: CEIL_DIV(total_elements, threads_per_block)
 *
 * Computes ceiling(M/N) efficiently using integer arithmetic.
 * Essential for calculating grid dimensions when launching kernels.
 *
 * Example:
 *     int num_blocks = CEIL_DIV(1000, 256);  // = 4 blocks
 *     kernel<<<num_blocks, 256>>>(...);
 */
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

/**
 * True/False - Compile-time boolean constants for template parameters
 *
 * These provide named boolean values that can be passed as template parameters,
 * making kernel instantiation more readable and enabling compile-time branching.
 *
 * Usage:
 *     template<typename T, std::bool_constant<bool> UseCache>
 *     __global__ void kernel(...) {
 *         if constexpr (UseCache) { ... }
 *     }
 *     kernel<float, True><<<...>>>(...);  // Cache-enabled version
 */
constexpr std::bool_constant<true> True;
constexpr std::bool_constant<false> False;

// ============================================================================
// ERROR CHECKING
// ============================================================================
// These utilities provide comprehensive error checking for CUDA API calls,
// similar to the pattern used in utils.h for standard C library functions.
// ============================================================================

/**
 * cudaCheck_ - Checks CUDA API call return codes
 *
 * @param error: cudaError_t returned by a CUDA API call
 * @param file: Source file name (automatically provided by macro)
 * @param line: Line number (automatically provided by macro)
 *
 * Verifies that a CUDA API call succeeded. If error != cudaSuccess, prints:
 * - The file and line where the error occurred
 * - A human-readable error message from cudaGetErrorString()
 * - Then exits the program
 *
 * The underscore suffix allows this to be called directly if needed, though
 * users should typically use the cudaCheck macro instead.
 *
 * Common CUDA errors caught:
 * - cudaErrorMemoryAllocation: Out of GPU memory
 * - cudaErrorInvalidValue: Invalid parameter to API call
 * - cudaErrorLaunchFailure: Kernel launch failed
 * - cudaErrorInvalidDeviceFunction: Kernel not found or incompatible
 */
inline void cudaCheck_(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

/**
 * cudaCheck - Macro wrapper for cudaCheck_
 *
 * Usage: cudaCheck(cudaMalloc(&ptr, size));
 *
 * Wraps any CUDA API call and automatically injects file/line information.
 * This is the primary way to call CUDA APIs in this codebase.
 *
 * Example patterns:
 *     cudaCheck(cudaMalloc(&d_data, bytes));
 *     cudaCheck(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
 *     cudaCheck(cudaDeviceSynchronize());
 */
#define cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

/**
 * cudaFreeCheck - Frees GPU memory with error checking and pointer reset
 *
 * @param ptr: Pointer to pointer to GPU memory (T** type)
 * @param file: Source file name (automatically provided by macro)
 * @param line: Line number (automatically provided by macro)
 *
 * This template function:
 * 1. Calls cudaFree() on the dereferenced pointer
 * 2. Checks for errors (using the same pattern as cudaCheck_)
 * 3. Sets the pointer to nullptr to prevent double-free bugs
 *
 * The double-pointer parameter allows the function to set the original
 * pointer to nullptr, making it safer than raw cudaFree().
 *
 * Benefits over plain cudaFree:
 * - Error checking (catches invalid pointers, double frees)
 * - Automatic pointer nullification (prevents use-after-free)
 * - Consistent error reporting
 *
 * Note: Called via the cudaFreeCheck macro, not directly.
 */
template<class T>
inline void cudaFreeCheck(T** ptr, const char *file, int line) {
    cudaError_t error = cudaFree(*ptr);
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    *ptr = nullptr;
}

/**
 * cudaFreeCheck - Macro wrapper for the templated cudaFreeCheck function
 *
 * Usage: cudaFreeCheck(&d_ptr);
 *
 * Note the & (address-of) operator - we pass the address of the pointer
 * so the function can set it to nullptr after freeing.
 */
#define cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))

// ============================================================================
// CUDA PRECISION SETTINGS
// ============================================================================
// This section enables compile-time selection of the floating-point precision
// used throughout the training code. The precision is controlled by preprocessor
// flags (ENABLE_FP32, ENABLE_FP16, ENABLE_BF16).
//
// Precision Trade-offs:
// --------------------
// FP32 (32-bit float):
//   + Highest precision, no accuracy concerns
//   + Fully supported on all GPUs
//   - Slowest (1x tensor core throughput)
//   - Highest memory usage
//
// FP16 (16-bit half precision):
//   + Fast on modern GPUs (8-16x throughput vs FP32)
//   + Half the memory of FP32
//   - Narrow dynamic range (can overflow/underflow)
//   - May require gradient scaling for stable training
//   ! Not recommended for LLM training due to numerical issues
//
// BF16 (16-bit bfloat):
//   + Fast on modern GPUs (8-16x throughput vs FP32)
//   + Same dynamic range as FP32 (wider than FP16)
//   + No gradient scaling needed
//   + Industry standard for LLM training
//   - Supported only on Ampere+ GPUs (compute capability >= 8.0)
//   ! This is the default and recommended mode
// ============================================================================

/**
 * PrecisionMode - Enumeration of supported precision modes
 *
 * Used for runtime queries and conditional code paths that depend on precision.
 */
enum PrecisionMode {
    PRECISION_FP32,    // 32-bit floating point
    PRECISION_FP16,    // 16-bit half precision
    PRECISION_BF16     // 16-bit bfloat16
};

/**
 * floatX - Typedef for the active precision mode
 *
 * This typedef allows kernels to be written generically, working with whatever
 * precision mode was selected at compile time.
 *
 * Compilation flags:
 * -DENABLE_FP32  -> floatX = float
 * -DENABLE_FP16  -> floatX = half
 * (default)      -> floatX = __nv_bfloat16
 *
 * Usage in kernels:
 *     __global__ void kernel(floatX* data) {
 *         floatX value = data[idx];
 *         // ... operates with whatever precision was selected
 *     }
 */
#if defined(ENABLE_FP32)
typedef float floatX;
#define PRECISION_MODE PRECISION_FP32
// use fp16 (note: this may require gradient scaler, currently not implemented!)
#elif defined(ENABLE_FP16)
typedef half floatX;
#define PRECISION_MODE PRECISION_FP16
#else // Default to bfloat16
typedef __nv_bfloat16 floatX;
#define PRECISION_MODE PRECISION_BF16
#endif

// ============================================================================
// LOAD AND STORE WITH STREAMING CACHE HINTS
// ============================================================================
// These functions use special PTX instructions to control caching behavior
// when accessing memory. This can significantly improve performance for
// certain memory access patterns.
//
// Cache Hints:
// -----------
// __ldcs (load cache streaming):
//   - Loads data with streaming hint, bypassing L1 cache
//   - Good for data accessed only once (e.g., reading input data)
//   - Reduces cache pollution
//
// __stcs (store cache streaming):
//   - Stores data with streaming hint, bypassing L1 cache
//   - Good for write-only data (e.g., writing final outputs)
//
// Compiler Compatibility Issue:
// ----------------------------
// Older nvcc versions (<12) don't provide __ldcs/__stcs for bfloat16, even
// though bf16 is just an unsigned short under the hood. We need to carefully
// define our own versions only when they don't already exist, otherwise we
// get compiler errors:
// - "no viable overload" on older architectures (SM 5.2)
// - "function already exists" on newer architectures (SM 8.0)
//
// The conditional compilation below handles these cases.
// ============================================================================

#if defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
/**
 * __ldcs - Load with cache streaming hint for bfloat16
 *
 * Custom implementation for older CUDA versions that don't provide this intrinsic.
 * Reinterprets the bf16 value as unsigned short, loads it with streaming hint,
 * then reinterprets back to bfloat16.
 */
__device__ floatX __ldcs(const floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<const unsigned short*>(address));
    return __nv_bfloat16_raw{bf};
}

/**
 * __stcs - Store with cache streaming hint for bfloat16
 *
 * Custom implementation for older CUDA versions that don't provide this intrinsic.
 * Reinterprets the bf16 value as unsigned short, stores it with streaming hint.
 */
__device__ void __stcs(floatX* address, floatX value) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)value).x);
}
#endif

// ============================================================================
// PROFILER UTILITIES
// ============================================================================
// Integration with NVIDIA Nsight Systems for performance profiling and analysis.
// NVTX (NVIDIA Tools Extension) allows marking code regions in the profiler
// timeline, making it easier to understand where time is spent.
//
// Usage with Nsight Systems:
// --------------------------
// 1. Wrap code sections with NvtxRange objects or NVTX_RANGE_FN() macro
// 2. Run: nsys profile ./your_program
// 3. Open the generated .nsys-rep file in Nsight Systems
// 4. See labeled regions in the timeline view
//
// Benefits:
// - Visual understanding of kernel execution order
// - Identify synchronization points and gaps
// - Measure time spent in different code sections
// - Easier debugging of performance issues
// ============================================================================

/**
 * NvtxRange - RAII wrapper for NVTX range markers
 *
 * This class uses C++ RAII (Resource Acquisition Is Initialization) to
 * automatically push/pop NVTX ranges, ensuring they're always properly closed.
 *
 * Usage:
 *     {
 *         NvtxRange range("Forward Pass");
 *         // ... code to profile ...
 *     }  // Range automatically ends here
 *
 * When profiling, the "Forward Pass" label will appear in Nsight Systems,
 * marking exactly where this code executes in the timeline.
 */
class NvtxRange {
 public:
    NvtxRange(const char* s) { nvtxRangePush(s); }
    NvtxRange(const std::string& base_str, int number) {
        std::string range_string = base_str + " " + std::to_string(number);
        nvtxRangePush(range_string.c_str());
    }
    ~NvtxRange() { nvtxRangePop(); }
};

/**
 * NVTX_RANGE_FN - Convenience macro to profile an entire function
 *
 * Usage: Place at the beginning of any function:
 *     void my_function() {
 *         NVTX_RANGE_FN();
 *         // ... function body ...
 *     }
 *
 * This automatically creates an NvtxRange with the function name, making
 * it easy to profile functions without manual labeling.
 */
#define NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// ============================================================================
// UTILITIES TO READ & WRITE BETWEEN CUDA MEMORY <-> FILES
// ============================================================================
// These functions provide efficient, asynchronous I/O between GPU memory and
// files using double-buffering technique to overlap computation with I/O.
//
// Double Buffering Strategy:
// -------------------------
// Instead of copying all data at once (which would be slow), these functions
// use two ping-pong buffers:
// 1. While buffer A is being written to/read from disk
// 2. Buffer B is being filled from/copied to GPU asynchronously
// 3. Swap buffers and repeat
//
// This overlaps disk I/O with GPU transfers, significantly reducing total time.
//
// Memory Type:
// -----------
// Uses cudaMallocHost (pinned memory) for staging buffers because:
// - Enables asynchronous cudaMemcpy operations
// - Much faster transfer speeds than pageable memory
// - For writes: uses cudaHostAllocWriteCombined for better write performance
//
// Common Use Cases:
// ----------------
// - Saving model checkpoints to disk
// - Loading large models or optimizer states from disk
// - Checkpointing during training
// ============================================================================

/**
 * device_to_file - Copies data from GPU memory to a file efficiently
 *
 * @param dest: File pointer to write to (must be open for writing)
 * @param src: GPU memory pointer (source data)
 * @param num_bytes: Total number of bytes to transfer
 * @param buffer_size: Size of staging buffers (larger = fewer iterations, more memory)
 * @param stream: CUDA stream for asynchronous operations
 *
 * This function uses double buffering to overlap GPU->CPU transfer with disk writes:
 * 1. Allocates two pinned staging buffers (total 2 * buffer_size bytes)
 * 2. Prime: Copy first chunk from GPU to buffer A, wait
 * 3. Loop:
 *    - Start copying next chunk GPU -> buffer B (async)
 *    - While that's happening, write buffer A to disk
 *    - Wait for GPU copy to complete, swap buffers
 * 4. Write final buffer to disk
 * 5. Free staging buffers
 *
 * Performance tips:
 * - buffer_size should be large enough to amortize overhead (e.g., 32MB)
 * - Use a non-default stream to enable better concurrency
 * - Consider using O_DIRECT flags when opening files for large transfers
 */
inline void device_to_file(FILE* dest, void* src, size_t num_bytes, size_t buffer_size, cudaStream_t stream) {
    // allocate pinned buffer for faster, async transfer
    char* buffer_space;
    cudaCheck(cudaMallocHost(&buffer_space, 2*buffer_size));
    // split allocation in two
    void* read_buffer = buffer_space;
    void* write_buffer = buffer_space + buffer_size;

    // prime the read buffer; first copy means we have to wait
    char* gpu_read_ptr = (char*)src;
    size_t copy_amount = std::min(buffer_size, num_bytes);
    cudaCheck(cudaMemcpyAsync(read_buffer, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    size_t rest_bytes = num_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    gpu_read_ptr += copy_amount;

    std::swap(read_buffer, write_buffer);
    // now the main loop; as long as there are bytes left
    while(rest_bytes > 0) {
        // initiate next read
        copy_amount = std::min(buffer_size, rest_bytes);
        cudaCheck(cudaMemcpyAsync(read_buffer, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
        // while this is going on, transfer the write buffer to disk
        fwriteCheck(write_buffer, 1, write_buffer_size, dest);
        cudaCheck(cudaStreamSynchronize(stream));     // wait for both buffers to be ready.

        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    // make sure to write the last remaining write buffer
    fwriteCheck(write_buffer, 1, write_buffer_size, dest);
    cudaCheck(cudaFreeHost(buffer_space));
}

/**
 * file_to_device - Copies data from a file to GPU memory efficiently
 *
 * @param dest: GPU memory pointer (destination)
 * @param src: File pointer to read from (must be open for reading)
 * @param num_bytes: Total number of bytes to transfer
 * @param buffer_size: Size of staging buffers (larger = fewer iterations, more memory)
 * @param stream: CUDA stream for asynchronous operations
 *
 * This function uses double buffering to overlap disk reads with CPU->GPU transfer:
 * 1. Allocates two pinned staging buffers with cudaHostAllocWriteCombined flag
 *    (WC memory is optimized for CPU-write, GPU-read patterns)
 * 2. Prime: Read first chunk from disk into buffer A
 * 3. Loop:
 *    - Start copying buffer B -> GPU (async)
 *    - While that's happening, read next chunk from disk into buffer A
 *    - Wait for GPU copy to complete, swap buffers
 * 4. Copy final buffer to GPU
 * 5. Free staging buffers
 *
 * Performance tips:
 * - buffer_size should be large (e.g., 32-64MB) for best throughput
 * - Use a non-default stream to enable concurrency with other operations
 * - File should ideally be on fast storage (NVMe SSD, RAM disk)
 * - Consider using direct I/O to bypass filesystem cache for very large files
 *
 * cudaHostAllocWriteCombined:
 * - Optimized for host writes, device reads (our use case)
 * - Bypasses CPU caches, reducing memory traffic
 * - Faster for sequential CPU writes followed by GPU reads
 */
inline void file_to_device(void* dest, FILE* src, size_t num_bytes, size_t buffer_size, cudaStream_t stream) {
     // allocate pinned buffer for faster, async transfer
     // from the docs (https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html)
     // WC memory is a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.
    char* buffer_space;
    cudaCheck(cudaMallocHost(&buffer_space, 2*buffer_size, cudaHostAllocWriteCombined));
    // split allocation in two
    void* read_buffer = buffer_space;
    void* write_buffer = buffer_space + buffer_size;

    // prime the read buffer;
    char* gpu_write_ptr = (char*)dest;
    size_t copy_amount = std::min(buffer_size, num_bytes);
    freadCheck(read_buffer, 1, copy_amount, src);

    size_t rest_bytes = num_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    std::swap(read_buffer, write_buffer);

    // now the main loop; as long as there are bytes left
    while(rest_bytes > 0) {
        // initiate next read
        copy_amount = std::min(buffer_size, rest_bytes);
        cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_buffer, write_buffer_size, cudaMemcpyHostToDevice, stream));
        gpu_write_ptr += write_buffer_size;
        // while this is going on, read from disk
        freadCheck(read_buffer, 1, copy_amount, src);
        cudaCheck(cudaStreamSynchronize(stream));     // wait for both buffers to be ready.

        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
    }

    // copy the last remaining write buffer to gpu
    cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_buffer, write_buffer_size, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    cudaCheck(cudaFreeHost(buffer_space));
}

#endif // CUDA_COMMON_H