/*
================================================================================
File: cublas_common.h
Purpose: cuBLAS library configuration, globals, and error checking utilities
================================================================================

Overview:
---------
This header provides the necessary configuration and error-checking utilities
for using NVIDIA's cuBLAS library, which provides optimized BLAS (Basic Linear
Algebra Subprograms) operations on CUDA GPUs.

cuBLAS in LLM Training:
-----------------------
cuBLAS is critical for LLM training because matrix multiplications (GEMMs)
dominate the computational cost:
- Attention: Q @ K^T, (attention weights) @ V
- Feed-forward: input @ W1, hidden @ W2
- Embeddings: tokens @ embedding_matrix
- ~90% of training time is spent in cuBLAS GEMMs

cuBLAS vs. Hand-Written Kernels:
--------------------------------
- cuBLAS: Highly optimized by NVIDIA, uses tensor cores, supports all precisions
- Custom kernels: Can be faster for specific sizes/patterns, but hard to beat cuBLAS
- This codebase uses cuBLAS for all matrix multiplications

Key Components:
--------------
1. Precision configuration: Maps compile-time precision to cuBLAS datatypes
2. Global workspace: cuBLAS needs scratch space for some operations
3. Compute type: Controls precision of intermediate computations
4. Error checking: Validates cuBLAS API calls

Usage:
------
Include this header before calling any cuBLAS functions. The workspace and
handle are initialized in the main program.
*/
#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// ============================================================================
// CUBLAS PRECISION SETTINGS
// ============================================================================
// Maps the compile-time precision mode (FP32/FP16/BF16) to the corresponding
// cuBLAS datatype constant. This ensures cuBLAS operations use the same
// precision as the rest of the code.
// ============================================================================

/**
 * CUBLAS_LOWP - cuBLAS datatype for low-precision operations
 *
 * Set based on the active precision mode:
 * - ENABLE_FP32 -> CUDA_R_32F (32-bit float)
 * - ENABLE_FP16 -> CUDA_R_16F (16-bit half)
 * - Default     -> CUDA_R_16BF (16-bit bfloat16)
 *
 * Used when calling cuBLAS functions to specify input/output data types.
 */
#if defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#elif defined(ENABLE_FP16)
#define CUBLAS_LOWP CUDA_R_16F
#else // default to bfloat16
#define CUBLAS_LOWP CUDA_R_16BF
#endif

// ============================================================================
// CUBLAS GLOBAL STATE
// ============================================================================
// cuBLAS requires global state for workspace memory and library handles.
// These are initialized once at program startup and used throughout training.
// ============================================================================

/**
 * cublaslt_workspace_size - Size of workspace buffer for cuBLASLt
 *
 * cuBLASLt (cuBLAS Light) is a flexible API that can use workspace memory
 * for intermediate computations, potentially improving performance.
 *
 * Size: 32 MiB (required for Hopper architecture, but safe for all GPUs)
 * - Hopper (H100): Needs 32 MiB for some optimized kernels
 * - Older GPUs: 4 MiB is sufficient, but using 32 MiB doesn't hurt
 */
const size_t cublaslt_workspace_size = 32 * 1024 * 1024;

/**
 * cublaslt_workspace - Pointer to workspace buffer (allocated at startup)
 */
void* cublaslt_workspace = NULL;

/**
 * cublas_compute - Compute type for cuBLAS operations
 *
 * Controls the precision used for intermediate computations in GEMM:
 * - CUBLAS_COMPUTE_32F: Use FP32 for intermediate sums (accumulate in FP32)
 *
 * Even when using BF16 inputs/outputs, accumulating in FP32 improves accuracy.
 * This is the standard practice for mixed-precision training.
 */
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;

/**
 * cublaslt_handle - cuBLASLt library handle (initialized at startup)
 */
cublasLtHandle_t cublaslt_handle;

// ============================================================================
// ERROR CHECKING
// ============================================================================

/**
 * cublasCheck - Validates cuBLAS API call return codes
 *
 * @param status: Return value from cuBLAS function
 * @param file: Source file name (automatically provided by macro)
 * @param line: Line number (automatically provided by macro)
 *
 * If status != CUBLAS_STATUS_SUCCESS, prints error information and exits.
 * Common cuBLAS errors:
 * - CUBLAS_STATUS_NOT_INITIALIZED: Handle not created
 * - CUBLAS_STATUS_ALLOC_FAILED: Out of memory
 * - CUBLAS_STATUS_INVALID_VALUE: Invalid parameter
 * - CUBLAS_STATUS_ARCH_MISMATCH: Operation not supported on this GPU
 */
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

#endif // CUBLAS_COMMON_H