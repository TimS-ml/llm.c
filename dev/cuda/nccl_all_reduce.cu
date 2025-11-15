/*
================================================================================
NCCL All-Reduce Multi-GPU Communication Test
================================================================================

PURPOSE:
--------
Demonstrates NVIDIA Collective Communications Library (NCCL) for efficient
multi-GPU communication. This file tests the "all-reduce" operation, which is
fundamental to distributed deep learning training.

WHAT IS NCCL?
-------------
NCCL (pronounced "Nickel") is NVIDIA's library for multi-GPU and multi-node
collective communication operations. It provides highly optimized implementations
of operations like:
  - All-Reduce: Combine values from all GPUs, distribute result to all
  - Broadcast: Send data from one GPU to all others
  - Reduce: Combine values from all GPUs to one GPU
  - All-Gather: Gather data from all GPUs, distribute complete set to all
  - Reduce-Scatter: Reduce then scatter results across GPUs

WHAT IS ALL-REDUCE?
-------------------
All-Reduce performs a reduction operation (e.g., SUM, MAX, MIN) across data
from multiple GPUs and distributes the result to all participating GPUs.

Example with 3 GPUs:
  Before All-Reduce:
    GPU 0: [1, 1, 1, 1]
    GPU 1: [2, 2, 2, 2]
    GPU 2: [3, 3, 3, 3]

  After All-Reduce with SUM:
    GPU 0: [6, 6, 6, 6]  (1+2+3 for each element)
    GPU 1: [6, 6, 6, 6]
    GPU 2: [6, 6, 6, 6]

WHY IS ALL-REDUCE IMPORTANT?
-----------------------------
In distributed deep learning (data parallelism):
1. Each GPU processes a different batch of data
2. Each GPU computes gradients on its local batch
3. All-Reduce SUMS gradients from all GPUs
4. Each GPU now has the average gradient across all batches
5. Each GPU updates its parameters (which remain synchronized)

This allows scaling training to multiple GPUs while maintaining correctness.

MPI vs NCCL:
------------
- MPI (Message Passing Interface): General-purpose inter-process communication
  Used here for process initialization and coordination

- NCCL: GPU-optimized collective operations
  Much faster than MPI for GPU-to-GPU communication
  Uses direct GPU-to-GPU transfers (NVLink, PCIe, InfiniBand)

MULTI-GPU vs MULTI-NODE:
-------------------------
- Multi-GPU: Multiple GPUs on same machine (this example)
  Communication via PCIe or NVLink (very fast)

- Multi-Node: GPUs across multiple machines
  Communication via network (InfiniBand, Ethernet)
  NCCL optimizes this with RDMA and topology-aware algorithms

THIS TEST:
----------
1. Each GPU fills a buffer with its rank number (GPU 0→1, GPU 1→2, etc.)
2. Performs all-reduce SUM operation
3. Verifies each GPU has the sum of all rank numbers

Compile example:
nvcc -lmpi -lnccl -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib/ -lcublas -lcublasLt nccl_all_reduce.cu -o nccl_all_reduce

Run on 2 local GPUs (set -np to a different value to change GPU count):
mpirun -np 2 ./nccl_all_reduce

Run on 4 GPUs:
mpirun -np 4 ./nccl_all_reduce

Run on multiple nodes (example for 2 nodes with 4 GPUs each):
mpirun -np 8 -npernode 4 --host node1,node2 ./nccl_all_reduce

*/

#include "common.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/*
================================================================================
Error Checking Macros for NCCL and MPI
================================================================================

These macros provide consistent error checking with file and line information,
making debugging much easier in multi-GPU environments where errors can be
distributed across processes.
*/

/*
NCCL error checking wrapper.
Checks if a NCCL function call succeeded and prints detailed error info if not.

Usage:
  ncclCheck(ncclAllReduce(...));
*/
void nccl_check(ncclResult_t status, const char *file, int line) {
  if (status != ncclSuccess) {
    printf("[NCCL ERROR] at file %s:%d:\n%s\n", file, line,
           ncclGetErrorString(status));
    exit(EXIT_FAILURE);
  }
}
#define ncclCheck(err) (nccl_check(err, __FILE__, __LINE__))

/*
MPI error checking wrapper.
Checks if an MPI function call succeeded and prints detailed error info if not.

Usage:
  mpiCheck(MPI_Init(&argc, &argv));
*/
void mpi_check(int status, const char *file, int line) {
  if (status != MPI_SUCCESS) {
    char mpi_error[4096];
    int mpi_error_len = 0;
    assert(MPI_Error_string(status, &mpi_error[0], &mpi_error_len) ==
           MPI_SUCCESS);
    printf("[MPI ERROR] at file %s:%d:\n%.*s\n", file, line, mpi_error_len,
           mpi_error);
    exit(EXIT_FAILURE);
  }
}
#define mpiCheck(err) (mpi_check(err, __FILE__, __LINE__))

/*
================================================================================
Utility Kernel and Functions
================================================================================
*/

/*
Simple kernel to fill a vector with a constant value.
Used to initialize each GPU's buffer with a unique value (its rank).

Each thread sets one element of the array. This demonstrates basic
GPU parallelism before we get to the more complex multi-GPU operations.

Parameters:
  data: GPU memory buffer to fill
  N: Number of elements in the buffer
  value: Constant value to write to all elements
*/
__global__ void set_vector(float *data, int N, float value) {
  // Compute global thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Bounds check to avoid out-of-bounds writes
  if (i < N) {
    data[i] = value;
  }
}

/*
Integer ceiling division utility.
Returns ceil(a/b) for positive integers.

Example: cdiv(10, 3) = 4 (since we need 4 blocks of size 3 to cover 10 elements)
*/
size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

/*
================================================================================
Multi-GPU Configuration Structure
================================================================================

Encapsulates all information needed for multi-GPU distributed training.
Each MPI process (typically one per GPU) has its own MultiGpuConfig instance.

TERMINOLOGY:
------------
Process: An MPI process, typically one per GPU
Rank: Unique identifier for each process (0, 1, 2, ...)
Node/Host: A physical machine that may contain multiple GPUs
Local device: GPU index on the current machine (different from global rank)

EXAMPLE SETUP:
--------------
2 nodes, each with 4 GPUs (8 GPUs total):

Node 0:
  Process 0: rank=0, local_device_idx=0
  Process 1: rank=1, local_device_idx=1
  Process 2: rank=2, local_device_idx=2
  Process 3: rank=3, local_device_idx=3

Node 1:
  Process 4: rank=4, local_device_idx=0
  Process 5: rank=5, local_device_idx=1
  Process 6: rank=6, local_device_idx=2
  Process 7: rank=7, local_device_idx=3

Note how local_device_idx resets to 0 on each node!
*/
typedef struct {
  int process_rank;      // Global rank of this process (0 to num_processes-1)
                         // Unique across ALL nodes
                         // Used for: identifying process in logs, determining roles

  int num_processes;     // Total number of processes across all nodes
                         // Equal to total number of GPUs being used
                         // Used for: sizing all-reduce operations, computing averages

  int local_device_idx;  // GPU index on THIS machine (0 to GPUs_per_node-1)
                         // May be same across different nodes
                         // Used for: cudaSetDevice() to select which GPU to use

  ncclComm_t nccl_comm;  // NCCL communicator object
                         // Contains topology info and communication channels
                         // Used for: all NCCL collective operations
                         // Must be initialized before use, destroyed after
} MultiGpuConfig;

/*
================================================================================
Multi-Node GPU Assignment Algorithm
================================================================================

THE PROBLEM:
------------
When running across multiple nodes (machines), each process needs to know:
  - Which GPU on its local machine to use (local_device_idx)
  - We can't just use process_rank because that's global, not per-machine

Example problem:
  8 processes on 2 nodes (4 GPUs per node)
  Process 4 has rank=4, but should use GPU 0 on node 1
  (not GPU 4, which doesn't exist!)

THE SOLUTION:
-------------
1. Each process gets its hostname (all processes on same node have same hostname)
2. Hash the hostname to a number
3. Share all hostname hashes via MPI_Allgather (all processes learn all hostnames)
4. Count how many processes with the same hostname have lower rank
5. That count becomes the local_device_idx

WALKTHROUGH EXAMPLE:
--------------------
2 nodes, 4 processes:
  Rank 0, 1 on "node0" → hash to 12345
  Rank 2, 3 on "node1" → hash to 67890

After MPI_Allgather, all processes know:
  [12345, 12345, 67890, 67890]

Rank 0: hash=12345, count predecessors with hash 12345 → 0 → local_device_idx=0
Rank 1: hash=12345, count predecessors with hash 12345 → 1 → local_device_idx=1
Rank 2: hash=67890, count predecessors with hash 67890 → 0 → local_device_idx=0
Rank 3: hash=67890, count predecessors with hash 67890 → 1 → local_device_idx=1

Perfect! Each node gets local indices 0, 1.

This algorithm is from NCCL documentation:
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-2-one-device-per-process-or-thread

Parameters:
  process_rank: Global rank of this process
  num_processes: Total number of processes across all nodes

Returns:
  local_device_idx: Which GPU to use on this machine (0, 1, 2, ...)
*/
int multi_gpu_get_local_device_idx(int process_rank, int num_processes) {
  // ===================================================================
  // Step 1: Get hostname and compute its hash
  // ===================================================================
  char hostname[1024];
  hostname[1023] = '\0';
  // Get the machine's hostname (e.g., "gpu-node-01")
  // All processes on same machine return the same hostname
  gethostname(hostname, 1023);

  // Truncate at first '.' to handle FQDNs (e.g., "node.cluster.com" → "node")
  for (int i=0; i < 1024; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        break;
    }
  }

  // Hash the hostname using djb2 algorithm
  // This converts string to a number for easier comparison
  uint64_t hostname_hash = 5381;
  for (int c = 0; hostname[c] != '\0'; c++){
    hostname_hash = ((hostname_hash << 5) + hostname_hash) ^ hostname[c];
  }

  // ===================================================================
  // Step 2: Exchange hostname hashes across all processes
  // ===================================================================
  // Allocate array to hold hashes from all processes
  uint64_t* all_hostsname_hashes = (uint64_t*)malloc(num_processes * sizeof(uint64_t));

  // Place our hash in the array at our rank position
  all_hostsname_hashes[process_rank] = hostname_hash;

  // MPI_Allgather: Everyone sends their hash, everyone receives all hashes
  // After this, all processes have identical copies of all_hostsname_hashes
  // MPI_IN_PLACE means we use the same buffer for send and receive
  mpiCheck(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                          all_hostsname_hashes, sizeof(uint64_t), MPI_BYTE,
                          MPI_COMM_WORLD));

  // ===================================================================
  // Step 3: Count how many same-host processes have lower rank
  // ===================================================================
  int local_device_idx = 0;
  for (int current_process = 0; current_process < num_processes; ++current_process) {
     if (current_process == process_rank) {
      // Reached our own rank, stop counting
      // local_device_idx now contains the count of same-host predecessors
      break;
     }
     if (all_hostsname_hashes[current_process] == all_hostsname_hashes[process_rank]) {
      // This process is on the same machine and has lower rank
      // So it will use a GPU with lower index
      // We increment to get the next available GPU
      local_device_idx++;
     }
  }

  free(all_hostsname_hashes);
  return local_device_idx;
}

/*
================================================================================
Multi-GPU Initialization Function
================================================================================

Initializes MPI and NCCL for multi-GPU training. This is called once at the
start of training, before any GPU operations.

INITIALIZATION SEQUENCE:
------------------------
1. Initialize MPI (inter-process communication)
2. Get process rank and total number of processes
3. Determine which local GPU to use on this machine
4. Set CUDA device to the determined GPU
5. Initialize NCCL communicator (for GPU-to-GPU communication)

NCCL COMMUNICATOR INITIALIZATION:
---------------------------------
NCCL requires all processes to share a unique ID:
  - Process 0 generates a unique ID using ncclGetUniqueId()
  - Process 0 broadcasts this ID to all other processes via MPI
  - All processes use this ID to initialize their NCCL communicator
  - The communicators from all processes form a "communication group"

This ensures all GPUs can communicate in coordinated collective operations.

IMPORTANT: This must be called by ALL processes simultaneously.
If any process fails or doesn't call this, the initialization will hang.

Parameters:
  argc, argv: Command line arguments (modified by MPI_Init)

Returns:
  MultiGpuConfig structure with all information needed for multi-GPU ops
*/
MultiGpuConfig multi_gpu_config_init(int *argc, char ***argv) {
    MultiGpuConfig result;

    // ===================================================================
    // Step 1: Initialize MPI
    // ===================================================================
    // MPI provides inter-process communication
    // Must be called before any other MPI functions
    mpiCheck(MPI_Init(argc, argv));

    // Get this process's rank (unique ID: 0, 1, 2, ...)
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &result.process_rank));

    // Get total number of processes
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &result.num_processes));

    // ===================================================================
    // Step 2: Determine local GPU index
    // ===================================================================
    // Figure out which GPU on this machine this process should use
    // This handles multi-node setups correctly
    result.local_device_idx = multi_gpu_get_local_device_idx(result.process_rank,
                                                               result.num_processes);

    printf("[Process rank %d] Using GPU %d\n", result.process_rank, result.local_device_idx);

    // ===================================================================
    // Step 3: Set CUDA device
    // ===================================================================
    // All subsequent CUDA operations on this process will use this GPU
    cudaCheck(cudaSetDevice(result.local_device_idx));

    // ===================================================================
    // Step 4: Initialize NCCL communicator
    // ===================================================================
    ncclUniqueId nccl_id;

    // Only rank 0 generates the unique ID
    // This ID will be shared with all other processes
    if (result.process_rank == 0) {
        ncclCheck(ncclGetUniqueId(&nccl_id));
    }

    // Broadcast the unique ID from rank 0 to all other ranks
    // After this, all processes have the same nccl_id
    mpiCheck(MPI_Bcast((void *)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Each process initializes its NCCL communicator with:
    //   - Total number of processes (num_processes)
    //   - Shared unique ID (nccl_id)
    //   - Its own rank (process_rank)
    // This creates a communication group spanning all GPUs
    ncclCheck(ncclCommInitRank(&result.nccl_comm, result.num_processes,
                                nccl_id, result.process_rank));

    return result;
}

/*
Cleanup function for multi-GPU resources.
Must be called at end of program by all processes.

Destroys:
  1. NCCL communicator (frees communication resources)
  2. MPI (shuts down inter-process communication)

IMPORTANT: All processes must call this. If any process exits early without
calling this, other processes may hang waiting for coordination.
*/
void multi_gpu_config_free(const MultiGpuConfig* multi_gpu_config) {
    // Destroy NCCL communicator
    // This closes all communication channels established for collectives
    ncclCommDestroy(multi_gpu_config->nccl_comm);

    // Finalize MPI
    // After this, no MPI functions can be called
    mpiCheck(MPI_Finalize());
}

/*
Simple utility to compute mean of an array.
Uses double precision accumulation to avoid numerical errors.

Used for verification: compute mean of buffer to check all-reduce results.

Parameters:
  arr: Array of floats (on CPU)
  size: Number of elements
  process_rank: Rank of process (for debugging, not actually used)

Returns:
  Mean value of the array
*/
float get_mean(float *arr, size_t size, int process_rank) {
  // Use double precision to avoid accumulation errors
  double sum = 0.0;
  for (size_t i = 0; i < size; ++i) {
    sum += arr[i];
  }
  return sum / size;
}

/*
================================================================================
Main Function: NCCL All-Reduce Test
================================================================================

OVERVIEW:
---------
1. Initialize multi-GPU setup (MPI + NCCL)
2. Each GPU fills buffer with (rank + 1)
3. Perform all-reduce SUM across all GPUs
4. Verify result: each GPU should have sum of all ranks
5. Clean up resources

TEST PROCEDURE:
---------------
With N GPUs (ranks 0 to N-1):

Before all-reduce:
  GPU 0 buffer: [1, 1, 1, ...]
  GPU 1 buffer: [2, 2, 2, ...]
  ...
  GPU N-1 buffer: [N, N, N, ...]

After all-reduce SUM:
  All GPUs: [1+2+...+N, 1+2+...+N, ...]

Expected sum = 1+2+...+N = N*(N+1)/2

For 2 GPUs: 1+2 = 3
For 4 GPUs: 1+2+3+4 = 10
For 8 GPUs: 1+2+...+8 = 36
*/
int main(int argc, char **argv) {
  // Buffer size: 32 million floats = 128 MB
  // Large enough to test performance, small enough for any GPU
  const size_t all_reduce_buffer_size = 32 * 1024 * 1024;

  // Kernel launch configuration
  const size_t threads_per_block = 1024;

  // ===================================================================
  // Step 1: Initialize Multi-GPU Configuration
  // ===================================================================
  // This sets up MPI for inter-process communication and NCCL for
  // fast GPU-to-GPU collective operations
  MultiGpuConfig multi_gpu_config = multi_gpu_config_init(&argc, &argv);

  // ===================================================================
  // Step 2: Allocate GPU memory buffer
  // ===================================================================
  // Each process allocates a buffer on ITS OWN GPU
  // Important: This is local to each GPU, not shared across GPUs
  float *all_reduce_buffer;
  cudaCheck(
      cudaMalloc(&all_reduce_buffer, all_reduce_buffer_size * sizeof(float)));

  // ===================================================================
  // Step 3: Initialize buffer with unique value per GPU
  // ===================================================================
  // Launch kernel to fill buffer with (rank + 1)
  // GPU 0 fills with 1.0, GPU 1 with 2.0, etc.
  int n_blocks = cdiv(all_reduce_buffer_size, threads_per_block);
  set_vector<<<n_blocks, threads_per_block>>>(
      all_reduce_buffer, all_reduce_buffer_size,
      (float)(multi_gpu_config.process_rank + 1));
  cudaCheck(cudaGetLastError());

  // ===================================================================
  // Step 4: Verify buffer initialization (before all-reduce)
  // ===================================================================
  // Allocate host memory to inspect GPU buffer
  float *all_reduce_buffer_host =
      (float *)malloc(all_reduce_buffer_size * sizeof(float));

  // Copy GPU buffer to CPU for verification
  cudaCheck(cudaMemcpy(all_reduce_buffer_host, all_reduce_buffer,
                       sizeof(float) * all_reduce_buffer_size,
                       cudaMemcpyDeviceToHost));

  // Print mean value (should be rank+1)
  // GPU 0: mean=1.0, GPU 1: mean=2.0, etc.
  printf("[Process rank %d] average value before all reduce is %.6f\n",
         multi_gpu_config.process_rank,
         get_mean(all_reduce_buffer_host, all_reduce_buffer_size,
                  multi_gpu_config.process_rank));

  // ===================================================================
  // Step 5: Allocate receive buffer for all-reduce result
  // ===================================================================
  // Note: We could use all_reduce_buffer for both input and output (in-place),
  // but using separate buffers makes the operation clearer for this demo
  float *all_reduce_buffer_recv;
  cudaCheck(cudaMalloc(&all_reduce_buffer_recv,
                       all_reduce_buffer_size * sizeof(float)));

  // ===================================================================
  // Step 6: Perform All-Reduce Operation
  // ===================================================================
  /*
  ncclAllReduce performs a reduction across all GPUs and distributes result to all.

  Parameters:
    sendbuff: Input buffer (on this GPU)
    recvbuff: Output buffer (on this GPU)
    count: Number of elements to reduce
    datatype: Element type (ncclFloat = 32-bit float)
    op: Reduction operation (ncclSum = element-wise sum)
    comm: NCCL communicator (links all participating GPUs)
    stream: CUDA stream (0 = default stream, synchronous)

  What happens:
    1. Each GPU provides its buffer (filled with rank+1)
    2. NCCL computes element-wise sum across all GPUs
    3. Result is written to recvbuff on ALL GPUs (broadcast)
    4. After completion, all GPUs have identical recvbuff contents

  For N GPUs:
    result[i] = buffer_gpu0[i] + buffer_gpu1[i] + ... + buffer_gpuN-1[i]
              = 1 + 2 + ... + N = N*(N+1)/2

  Performance: NCCL uses optimized ring or tree algorithms, achieving
  near-linear scaling with number of GPUs.
  */
  ncclCheck(ncclAllReduce(
      (const void *)all_reduce_buffer,      // Source: each GPU's unique values
      (void *)all_reduce_buffer_recv,       // Destination: will contain sum
      all_reduce_buffer_size,                // Number of elements
      ncclFloat,                             // Data type
      ncclSum,                               // Reduction operation
      multi_gpu_config.nccl_comm,           // NCCL communicator
      0));                                   // CUDA stream (0 = default)


  // ===================================================================
  // Step 7: Verify All-Reduce Result
  // ===================================================================
  // Copy result back to CPU for verification
  cudaCheck(cudaMemcpy(all_reduce_buffer_host, all_reduce_buffer_recv,
                       sizeof(float) * all_reduce_buffer_size,
                       cudaMemcpyDeviceToHost));

  // Compute mean of reduced buffer
  float all_reduce_mean_value = get_mean(all_reduce_buffer_host,
                                          all_reduce_buffer_size,
                                          multi_gpu_config.process_rank);

  printf("[Process rank %d] average value after all reduce is %.6f\n",
         multi_gpu_config.process_rank, all_reduce_mean_value);

  // ===================================================================
  // Step 8: Validate Result
  // ===================================================================
  // Compute expected result: sum of (1 + 2 + ... + num_processes)
  // For N processes: expected = N*(N+1)/2
  float expected_all_reduce_mean_value = 0.0;
  for (int i = 0; i != multi_gpu_config.num_processes; ++i) {
    expected_all_reduce_mean_value += i + 1;
  }

  // Check if result matches expected (with small tolerance for floating point error)
  if (abs(expected_all_reduce_mean_value - all_reduce_mean_value) > 1e-5) {
    printf("[Process rank %d] ERROR: Unexpected all reduce value: %.8f, expected %.8f\n",
           multi_gpu_config.process_rank, all_reduce_mean_value,
           expected_all_reduce_mean_value);
  } else {
    printf("[Process rank %d] Checked against expected mean value. All good!\n",
           multi_gpu_config.process_rank);
  }

  // ===================================================================
  // Step 9: Cleanup
  // ===================================================================
  // Free host memory
  free(all_reduce_buffer_host);

  // Free GPU memory
  cudaCheck(cudaFree(all_reduce_buffer));
  cudaCheck(cudaFree(all_reduce_buffer_recv));

  // Destroy NCCL and MPI resources
  // Important: All processes must reach this point for clean shutdown
  multi_gpu_config_free(&multi_gpu_config);

  /*
  SUCCESSFUL OUTPUT EXAMPLE (2 GPUs):
  ------------------------------------
  [Process rank 0] Using GPU 0
  [Process rank 1] Using GPU 0
  [Process rank 0] average value before all reduce is 1.000000
  [Process rank 1] average value before all reduce is 2.000000
  [Process rank 0] average value after all reduce is 3.000000
  [Process rank 1] average value after all reduce is 3.000000
  [Process rank 0] Checked against expected mean value. All good!
  [Process rank 1] Checked against expected mean value. All good!

  Note: Both GPUs end with value 3.0 = 1+2 (the sum)
  */
}
