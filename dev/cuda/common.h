/*
================================================================================
common.h - Common CUDA Utilities and Helper Functions
================================================================================

PURPOSE:
--------
Provides reusable utilities for CUDA kernel development and testing:
  - Error checking macros (CUDA, cuBLAS)
  - Reduction primitives (warp-level, block-level)
  - Packed data structures for vectorized memory access
  - Random data generation for testing
  - Result validation and benchmarking infrastructure
  - cuBLAS/cuBLASLt initialization and configuration

USAGE:
------
Include this header in any CUDA kernel file that needs these utilities:
  #include "common.h"

This header is included by most kernel files in the dev/cuda/ directory.

KEY COMPONENTS:
---------------
1. Error Checking:
   - cudaCheck(err): Check CUDA API calls
   - cublasCheck(status): Check cuBLAS API calls

2. Reduction Primitives:
   - warpReduceSum/Max: Fast warp-level reductions using shuffle
   - blockReduce<T>: Block-level reduction with shared memory

3. Memory Access:
   - Packed128<T>: 128-bit aligned loads/stores for coalescing
   - load128/store128: Vectorized memory operations
   - load128cs/store128cs: With cache streaming hints

4. Testing Infrastructure:
   - make_random_float/int: Generate test data
   - validate_result: Compare GPU vs CPU results
   - benchmark_kernel: Measure kernel performance

5. cuBLAS Setup:
   - setup_main(): Initialize cuBLAS and get device properties
   - Global handles: cublas_handle, cublaslt_handle

DESIGN PHILOSOPHY:
------------------
- Header-only for easy inclusion
- Template-based for type flexibility
- Minimal dependencies (CUDA runtime, cuBLAS only)
- Educational: Clear code over maximum optimization
*/

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <float.h>

#define WARP_SIZE 32U

// Global device properties (set by setup_main())
extern cudaDeviceProp deviceProp;

/*
================================================================================
Basic Utility Functions
================================================================================
*/

/*
Integer ceiling division.

Computes ceil(dividend / divisor) for integer types.
Avoids floating point conversion and is exact for integers.

Example:
  ceil_div(10, 3) = 4 (since 10/3 = 3.33... rounds up to 4)
  ceil_div(9, 3) = 3 (exact division)

Common use: Calculating number of blocks needed to cover all threads:
  int num_blocks = ceil_div(total_threads, block_size);
*/
template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor-1) / divisor;
}

/*
================================================================================
Warp-Level Reduction Primitives
================================================================================

These functions perform reductions across all 32 threads in a warp using
warp shuffle instructions, which are very fast (single instruction).

WARP SHUFFLE:
-------------
__shfl_xor_sync exchanges values between threads in a warp without shared memory:
  - Very fast (single clock cycle)
  - No memory access
  - No synchronization overhead

ALGORITHM (butterfly reduction):
---------------------------------
Starting with each thread holding a value:
  Step 1: Each thread exchanges with thread 16 away (offset=16)
  Step 2: Each thread exchanges with thread 8 away (offset=8)
  Step 3: offset=4, then 2, then 1
  Result: Thread 0 holds the sum of all 32 values

Visual example (8 threads, for simplicity):
  Initial: [a, b, c, d, e, f, g, h]
  offset=4: [a+e, b+f, c+g, d+h, e+a, f+b, g+c, h+d]
  offset=2: [a+c+e+g, b+d+f+h, c+a+g+e, d+b+h+f, ...]
  offset=1: [a+b+c+d+e+f+g+h, a+b+c+d+e+f+g+h, ...]
*/

/*
Warp-level sum reduction using shuffle instructions.

All 32 threads in the warp must call this function.
After execution, all threads have the sum, but typically only thread 0 uses it.

Parameters:
  val: Value from this thread

Returns:
  Sum of val across all 32 threads in the warp

Requirement: All threads in warp must be active (0xFFFFFFFF mask)
*/
__device__ float warpReduceSum(float val) {
    // Butterfly reduction pattern
    for (int offset = 16; offset > 0; offset /= 2) {
        // Each thread exchanges with thread (threadIdx.x XOR offset)
        // Then adds the received value to its own
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/*
================================================================================
Block-Level Reduction Template
================================================================================

Reduces a value across ALL threads in a block (up to 1024 threads).

ALGORITHM:
----------
Three-stage reduction:
  1. Warp-level: Each warp reduces using shuffle (fast)
  2. Cross-warp: Warp leaders write to shared memory, sync, read back
  3. Final warp: First warp reduces the per-warp results

SHARED MEMORY:
--------------
Uses 32 floats of shared memory (128 bytes).
This is unique per call site (static allocation), which avoids needing
an extra __syncthreads() at the end (unless called in a loop).

If called in a loop, shared memory is reused, so set final_sync=true.

TEMPLATE PARAMETER:
-------------------
warp_reduction: Function pointer to warp reduction (warpReduceSum or warpReduceMax)
  Allows this template to work for different reduction operations.

PARAMETERS:
-----------
val: Value from this thread to reduce
final_sync: Whether to synchronize at end (true if called in loop)
out_of_bounds: Value to use for threads beyond num_warps (usually 0.0f or -FLT_MAX)

EXAMPLE USAGE:
--------------
  float sum = blockReduce<warpReduceSum>(my_value);  // All threads get sum
  if (threadIdx.x == 0) {
      output[blockIdx.x] = sum;  // Only thread 0 writes
  }

REQUIREMENTS:
-------------
- All threads in block must call this
- warp_reduction function must return same value to all threads in warp
*/
using reduction_func_t = float (*) (float);

template<reduction_func_t warp_reduction>
__device__ inline float blockReduce(float val, bool final_sync, float out_of_bounds) {
    // Shared memory for storing per-warp results (one per warp in block)
    __shared__ float shared_val[WARP_SIZE];

    const int lane_id = threadIdx.x % WARP_SIZE;      // Thread index within warp (0-31)
    const int warp_id = threadIdx.x / WARP_SIZE;       // Warp index within block
    const int num_warps = blockDim.x / WARP_SIZE;     // Number of warps in block

    // ===================================================================
    // Stage 1: Reduce within each warp
    // ===================================================================
    float warp_val = warp_reduction(val);

    // Warp leader (lane 0) writes warp result to shared memory
    if (lane_id == 0) { shared_val[warp_id] = warp_val; }

    // Wait for all warps to write their results
    __syncthreads();

    // ===================================================================
    // Stage 2: Reduce across warps (warp 0 only)
    // ===================================================================
    // Each thread in warp 0 loads one warp's result
    // Threads beyond num_warps load out_of_bounds value
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;

    // Warp 0 reduces to get final block result
    float block_val = warp_reduction(warp_val);

    // ===================================================================
    // Optional final sync
    // ===================================================================
    // If called in loop, shared memory will be reused, so we need to sync
    if (final_sync) {
        __syncthreads();
    }

    return block_val;  // All threads return same value (from lane 0 of warp 0)
}

/*
Convenience wrapper with default parameters.
Most common use case: no loop, sum reduction starting from 0.
*/
template<reduction_func_t warp_reduction>
__device__ inline float blockReduce(float val) {
    return blockReduce<warp_reduction>(val, false, 0.0f);
}

// ----------------------------------------------------------------------------
// checking utils

// CUDA error checking
void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

/*
================================================================================
cuBLAS Configuration and Global State
================================================================================

These globals are initialized by setup_main() and used throughout kernels.
*/

// cuBLAS workspace buffer
// Hopper (H100) needs 32MB, older GPUs need only 4MB
// Used by cuBLASLt for temporary storage during matrix multiplications
static size_t cublaslt_workspace_size = 32 * 1024 * 1024;
static void* cublaslt_workspace = NULL;

// Compute type for cuBLAS operations
// Either CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_32F_FAST_TF32
static cublasComputeType_t cublas_compute_type;

// cuBLAS handles for matrix operations
cublasHandle_t cublas_handle;        // Standard cuBLAS API
cublasLtHandle_t cublaslt_handle;    // cuBLASLt (more flexible, better performance)

// GPU architecture information (from cudaDeviceProp)
int cuda_arch_major = 0;              // e.g., 8 for Ampere (A100), 9 for Hopper (H100)
int cuda_arch_minor = 0;              // Minor version number

// GPU capability metrics
int cuda_num_SMs = 0;                 // Number of streaming multiprocessors
                                       // Used for persistent kernels (1 block per SM)

int cuda_threads_per_SM = 0;          // Max threads per SM (e.g., 2048 for A100)
                                       // Used to calculate grid size to fill GPU

// ----------------------------------------------------------------------------
// to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#else
#define MAX_1024_THREADS_BLOCKS 1
#endif

/*
================================================================================
Packed128: Vectorized Memory Access Template
================================================================================

PURPOSE:
--------
Forces the compiler to use 128-bit (16-byte) memory transactions, which are
the most efficient on modern GPUs.

WHY 128-BIT LOADS?
------------------
GPU memory controllers work most efficiently with:
  - 32-bit (4 byte) - baseline
  - 64-bit (8 byte) - 2x faster
  - 128-bit (16 byte) - 4x faster (best!)

A 128-bit load in a single instruction reads:
  - 4 × float (32-bit each)
  - 8 × half/bf16 (16-bit each)
  - 16 × int8 (8-bit each)

MEMORY BANDWIDTH:
-----------------
Example: Reading 1M floats
  - Individual loads: 1M transactions
  - float4 (128-bit): 250K transactions → 4x less overhead

COMPARISON TO float4:
---------------------
float4 is hardcoded for 32-bit floats.
Packed128 works with ANY type (float, half, bfloat16, etc.) while still
forcing 128-bit loads/stores.

REQUIREMENTS:
-------------
1. Data must be 16-byte aligned in memory
2. Access patterns should be coalesced (consecutive threads → consecutive addresses)

TEMPLATE PARAMETER:
-------------------
ElementType: Type of individual elements (float, __nv_bfloat16, half, etc.)

USAGE EXAMPLE:
--------------
  // Define type alias for convenience
  using f128 = Packed128<float>;  // Holds 4 floats

  // Load 4 consecutive floats in one instruction
  f128 data = load128(array_ptr);

  // Access individual elements
  float x = data[0];
  float y = data[1];

  // Modify and store back
  data[0] = x * 2.0f;
  store128(output_ptr, data);
*/

template<class ElementType>
struct alignas(16) Packed128 {
    // Note: = default implicitly generates a __device__ function, but explicitly
    // adding __device__ causes a lot of warnings.
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
        return constant(0);
    }

    __device__ static Packed128 ones() {
        return constant(1);
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
    // e.g. sizeof(int4) is 16 (4 X 4 bytes), sizeof(bfloat16) = 2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in one Packed128
    static constexpr const int size = sizeof(int4) / sizeof(ElementType);
    ElementType payload[size];
};

// short-form typedef
typedef Packed128<float> f128;

// load a Packed128 from an aligned memory address
template<class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address) {
    return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
}
// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class ElementType>
__device__ void store128cs(ElementType* target, Packed128<ElementType> value) {
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ void store128cg(ElementType* target, Packed128<ElementType> value) {
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}

// ----------------------------------------------------------------------------
// reduced/mixed precision utilities

#if defined(ENABLE_BF16)

typedef __nv_bfloat16 floatX;
typedef __nv_bfloat16 floatN;
#define CUBLAS_LOWP CUDA_R_16BF // CUDA_R_16F or CUDA_R_16BF (or CUDA_R_32F)
// CUBLAS_COMPUTE_32F or CUBLAS_COMPUTE_16F (for CUDA_R_16F only, potentially slower?!)
#define CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#elif defined(ENABLE_FP16)

typedef half floatX;
typedef half floatN;

#else

typedef float floatX;
typedef float floatN;
#endif

typedef Packed128<floatX> x128;


// older nvcc does not provide __ldcs and __stcs for bfloat16, despite these actually just being unsigned shorts.
// we need to be careful here to only define our own versions if none already exist, otherwise the compiler will
// complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)
#if defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(const floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<const unsigned short*>(address));
    return __nv_bfloat16_raw{bf};
}

__device__ void __stcs(floatX* address, floatX value) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)value).x);
}
#endif


// ----------------------------------------------------------------------------
// random utils

float* make_random_float_01(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX); // range 0..1
    }
    return arr;
}

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    return arr;
}

int* make_random_int(size_t N, int V) {
    int* arr = (int*)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() % V; // range 0..V-1
    }
    return arr;
}

float* make_zeros_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    return arr;
}

float* make_ones_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = 1.0f;
    }
    return arr;
}

// ----------------------------------------------------------------------------
// testing and benchmarking utils

template<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, float* h_ptr, size_t count) {
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*)malloc(count * sizeof(TargetType));
    for (int i = 0; i < count; i++) {
        converted[i] = (TargetType)h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, count * sizeof(TargetType), cudaMemcpyHostToDevice);
    free(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    return status;
}

void setup_main() {
    srand(0);   // determinism

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    cuda_num_SMs = deviceProp.multiProcessorCount;
    cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    cuda_arch_major = deviceProp.major;
    cuda_arch_minor = deviceProp.minor;

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = cuda_arch_major >= 8 ? 1 : 0;
    // TODO implement common CLI for all tests/benchmarks
    // if (override_enable_tf32 == 0) { enable_tf32 = 0; } // force to zero via arg
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4) {
    D* out_gpu = (D*)malloc(num_elements * sizeof(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(D), cudaMemcpyDeviceToHost));
    int nfaults = 0;
#ifndef ENABLE_BF16
    float epsilon = FLT_EPSILON;
#else
    float epsilon = 0.079;
#endif
    for (int i = 0; i < num_elements; i++) {
        // Skip masked elements
        if(!isfinite(cpu_reference[i]))
            continue;

        // print the first few comparisons
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }
        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified additional tolerance
        float t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        // ensure correctness for all elements.
        if (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
            nfaults ++;
            if (nfaults >= 10) {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    if (nfaults > 0) {
        free(out_gpu);
        exit(EXIT_FAILURE);
    }

    free(out_gpu);
}

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // prepare buffer to scrub L2 cache between benchmarks
    // just memset a large dummy array, recommended by
    // https://stackoverflow.com/questions/31429377/how-can-i-clear-flush-the-l2-cache-and-the-tlb-of-a-gpu
    // and apparently used in nvbench.
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    void* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    float elapsed_time = 0.f;
    for (int i = 0; i < repeats; i++) {
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // now we can start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        float single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }

    cudaCheck(cudaFree(flush_buffer));

    return elapsed_time / repeats;
}