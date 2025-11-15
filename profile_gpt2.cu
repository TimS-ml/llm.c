/*
GPT-2 CUDA Kernel Profiling Tool
=================================

PURPOSE:
This is a specialized profiling tool designed to analyze the performance of individual
CUDA kernels in the GPT-2 training loop. It runs a minimal training iteration specifically
optimized for profiling with NVIDIA's profiling tools (nsys, ncu).

WHAT THIS DOES:
- Loads a GPT-2 124M BF16 model
- Runs ONE complete training iteration (forward + backward + update)
- Uses only 1 transformer layer (all layers use identical kernels, so profiling one is sufficient)
- Synchronizes GPU at the end to ensure all kernels complete before exit

WHY USE THIS:
- Testing full training is slow and generates huge profile files
- This gives you kernel-level performance data quickly
- Helps identify bottlenecks in individual CUDA kernels
- Useful for optimizing specific operations (attention, matmul, etc.)

COMPILATION:
make profile_gpt2cu NO_MULTI_GPU=1

Note: NO_MULTI_GPU=1 is important - we don't want multi-GPU overhead in profiles

HOW TO USE WITH NVIDIA NSIGHT COMPUTE (ncu):
--------------------------------------------
Basic profiling (quick):
  sudo ncu -o profile -f ./profile_gpt2cu

Comprehensive profiling (detailed, slow):
  sudo ncu --set full --import-source yes -o profile -f ./profile_gpt2cu

Profile specific kernels only:
  sudo ncu --kernel-name-base regex --kernel-regex-base "matmul.*" -o profile -f ./profile_gpt2cu

Command-line flag breakdown:
- `--set full`: Collect ALL performance metrics (memory, compute, occupancy, etc.)
  Warning: This is VERY slow, but gives comprehensive data
- `--set default`: Collect standard metrics (faster, usually sufficient)
- `--import-source yes`: Include source code in the profile for line-by-line analysis
- `-o profile`: Output file name (creates profile.ncu-rep)
- `-f`: Force overwrite of existing profile file
- `--kernel-name-base regex`: Use regex to filter kernels
- `--kernel-regex-base "pattern"`: Only profile kernels matching this pattern

HOW TO USE WITH NVIDIA NSIGHT SYSTEMS (nsys):
---------------------------------------------
System-wide timeline profiling:
  sudo nsys profile -o profile --stats=true ./profile_gpt2cu

This gives you:
- Timeline view of all kernel launches
- CPU-GPU synchronization points
- Memory transfers
- Overall system utilization

VIEWING RESULTS:
---------------
1. For .ncu-rep files: Open with NVIDIA Nsight Compute GUI
   - Download Nsight Compute for your desktop OS
   - File -> Open -> select profile.ncu-rep
   - Analyze kernel performance, memory usage, register pressure, etc.

2. For .nsys-rep files: Open with NVIDIA Nsight Systems GUI
   - Download Nsight Systems for your desktop OS
   - File -> Open -> select profile.nsys-rep
   - View timeline, identify gaps, optimize scheduling

TIP: Run profiling on a cloud GPU, then rsync the .ncu-rep or .nsys-rep file
     to your local machine for viewing in the GUI.

INTERPRETING RESULTS:
--------------------
Key metrics to look for in Nsight Compute:
- Achieved Occupancy: Higher is better (aim for >50%)
- Memory Throughput: Should be close to theoretical maximum
- Compute Throughput: % of peak FLOPS being utilized
- Warp Execution Efficiency: Measures branch divergence (aim for >80%)
- Memory Access Pattern: Coalesced vs. uncoalesced accesses

Red flags:
- Low occupancy + low memory throughput = kernel launch overhead
- High memory throughput but slow kernel = memory-bound (need better caching)
- Low warp efficiency = branch divergence issues

ADJUSTING WORKLOAD SIZE:
-----------------------
If you run out of memory (OOM), decrease B (batch size) or T (sequence length):
- Start with B=24, T=1024 (default)
- Try B=12, T=1024 or B=24, T=512
- Minimum: B=4, T=64 (still gives useful profile data)
- Keep them as powers of 2 for best GPU utilization
*/

#define TESTING
#include "train_gpt2.cu"

// ----------------------------------------------------------------------------
// Main Profiling Entry Point
// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    // ========================================================================
    // STEP 1: Initialize (minimal setup, no multi-GPU)
    // ========================================================================
    char nccl_init_method[256] = "mpi";
    int num_processes = -1;  // Not used in single-GPU profiling
    int process_rank = -1;
    int gpus_per_node = -1;
    char server_ip[256] = "";
    char fs_path[256] = "";
    multi_gpu_config = multi_gpu_config_init(num_processes, process_rank, gpus_per_node, server_ip, fs_path, nccl_init_method);

    // Initialize CUDA, cuBLAS, cuDNN
    // first arg = profiling mode, second arg = timing mode
    common_start(true, true);

    // ========================================================================
    // STEP 2: Load model (BF16 version for realistic profiling)
    // ========================================================================
    GPT2 model;
    gpt2_init_common(&model);
    gpt2_build_from_checkpoint(&model, "gpt2_124M_bf16.bin");

    // ========================================================================
    // STEP 3: Configure workload size
    // ========================================================================
    // These determine the size of the profiling run
    // Larger values = more realistic, but slower profiling
    int B = 24;    // Batch size (number of sequences processed in parallel)
    int T = 1024;  // Sequence length (number of tokens per sequence)

    // NOTE: If you get OOM errors, decrease these values
    // Try: B=12, T=1024 or B=24, T=512 or B=4, T=64
    printf("batch size: %d\n", B);
    printf("sequence length: %d\n", T);

    // ========================================================================
    // STEP 4: Create dummy input data
    // ========================================================================
    // We don't care about actual values, just kernel performance
    int* x = (int*)mallocCheck(B * T * sizeof(int));  // Input tokens
    int* y = (int*)mallocCheck(B * T * sizeof(int));  // Target tokens
    for(int i = 0; i < B * T; ++i) {
        x[i] = i % model.config.vocab_size;  // Simple pattern
        y[i] = i % model.config.vocab_size;
    }

    // ========================================================================
    // STEP 5: Optimize for profiling
    // ========================================================================
    // IMPORTANT: Set layers to 1 because:
    // - All transformer layers use identical kernels
    // - Profiling 1 layer gives same info as profiling 12 layers
    // - Makes profiling 12x faster
    model.config.num_layers = 1;

    // Configure optimizer (not important for profiling, but required)
    set_zero_configs(&multi_gpu_config, 0, model.num_parameters);

    // ========================================================================
    // STEP 6: Run one complete training iteration
    // ========================================================================
    // This is what we're profiling!
    gpt2_allocate_state(&model, B, T);

    // Forward pass: Input -> Logits
    gpt2_forward(&model, x, B, T);

    // Backward pass: Gradients computation
    gpt2_backward_and_reduce(&model, x, y, 1, 0);

    // Gradient clipping
    float grad_norm = gpt2_calculate_grad_norm(&model, &multi_gpu_config);
    float grad_scale = (grad_norm > 1.0f) ? 1.0f / grad_norm : 1.0f;

    // Parameter update (AdamW optimizer)
    gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, grad_scale, 1, &multi_gpu_config);

    // ========================================================================
    // STEP 7: Synchronize before exit
    // ========================================================================
    // CRITICAL: Ensures all GPU work completes before profiler stops recording
    // Without this, profiler may miss kernel launches or report incorrect timings
    cudaCheck(cudaDeviceSynchronize());

    // ========================================================================
    // CLEANUP
    // ========================================================================
    gpt2_free(&model);
    common_free(model);

    return 0;
}
