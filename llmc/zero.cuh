/*
==============================================================================
ZeRO (Zero Redundancy Optimizer) - Multi-GPU Training Utilities
==============================================================================

PURPOSE:
Implements ZeRO optimization strategies for distributed training across
multiple GPUs. ZeRO reduces memory redundancy by partitioning optimizer
states, gradients, and parameters across GPUs while maintaining computational
efficiency.

ZERO OPTIMIZATION STAGES:

Stage 0 (Disabled - Baseline Data Parallel):
- Each GPU maintains full copy of: parameters, gradients, optimizer states
- All-reduce gradients after backward pass
- Memory: O(Ψ + K*Ψ) per GPU where Ψ = model size, K = optimizer state factor
- Memory redundancy: N×
- Communication: All-reduce gradients (bandwidth-optimal)

Stage 1 (Optimizer State Sharding - OSS):
- Each GPU maintains full copy of: parameters, gradients
- Optimizer states (m, v, master weights) partitioned across GPUs
- Each GPU only updates its partition of parameters
- All-gather parameters after update (if needed)
- Memory: O(Ψ + Ψ/N + K*Ψ/N) ≈ O(Ψ) for large K
- Memory saved: ~4× for AdamW (K=2) + master weights
- Communication: Reduce-scatter gradients + all-gather parameters

Stage 2 (Gradient Sharding - SDP):
- Each GPU maintains full copy of: parameters
- Gradients + optimizer states partitioned across GPUs
- Reduce-scatter gradients to owner GPU during backward
- Memory: O(Ψ + Ψ/N + K*Ψ/N)
- Memory saved: ~8× for AdamW
- Communication: Reduce-scatter gradients (more efficient than stage 1)

Stage 3 (Parameter Sharding - FSDP):
- Parameters + gradients + optimizer states all partitioned
- All-gather parameters just-in-time for forward/backward
- Discard non-owned parameters after use
- Memory: O(Ψ/N + Ψ/N + K*Ψ/N) = O((K+2)*Ψ/N)
- Memory saved: ~N× total
- Communication: All-gather parameters (2× for forward + backward)
- Trade-off: More communication for extreme memory savings

This implementation currently supports Stages 0 and 1.

MULTI-GPU CONFIGURATION (MultiGpuConfig):

Key fields:
- process_rank: Rank of this process (0 to num_processes-1)
- num_processes: Total number of GPUs/processes
- local_device_idx: GPU index on this machine (for multi-node)
- zero_stage: Which ZeRO optimization level (0, 1, 2, 3)
- shard_num_parameters: Number of parameters this GPU owns
- nccl_comm: NCCL communicator for collective operations
- nccl_stream: Dedicated CUDA stream for NCCL operations
- compute_nccl_sync: Event for synchronizing compute and NCCL streams

NCCL (NVIDIA Collective Communications Library):

NCCL provides optimized multi-GPU communication primitives:

1. All-Reduce:
   All GPUs contribute data, all receive the reduced result
   Result[i] = sum_gpu(Input_gpu[i])
   Used in Stage 0 for gradient averaging

2. Reduce-Scatter:
   All GPUs contribute data, each receives a partition of reduced result
   GPU_k receives: sum_gpu(Input_gpu[k*chunk_size : (k+1)*chunk_size])
   Used in Stage 1 for gradient reduction with partitioning

3. All-Gather:
   Each GPU contributes a partition, all receive the full concatenated result
   Inverse of reduce-scatter
   Used to reconstruct full parameters from sharded optimizer state

4. Broadcast:
   One GPU sends data to all others
   Used for distributing random seeds, configuration, etc.

NCCL INITIALIZATION METHODS:

Three methods for initializing NCCL across machines:

1. MPI (Message Passing Interface):
   - Requires MPI installation and PMIx support
   - Rank 0 generates NCCL unique ID
   - MPI_Bcast distributes ID to all ranks
   - Most reliable for HPC clusters
   - Requires --use-mpi flag

2. TCP (Socket-based):
   - Rank 0 creates TCP server on specified IP/port (12345)
   - Other ranks connect as TCP clients
   - Server sends NCCL unique ID to all clients
   - Works across machines with network connectivity
   - Requires --server-ip flag

3. Filesystem (Shared FS):
   - Rank 0 writes NCCL unique ID to shared file
   - Other ranks poll until file appears, then read ID
   - Requires shared filesystem (NFS, Lustre, etc.)
   - Simplest but has race conditions (naive sleep-based sync)
   - Requires --fs-path flag

LOCAL DEVICE SELECTION (multi_gpu_get_local_device_idx):

For multi-node training, determines which GPU on each machine:
- Hash hostname to identify machine
- All-gather hostnames from all processes
- Count processes on same machine before this rank
- This count becomes local_device_idx

Example: 4 processes on 2 machines (2 GPUs each):
- Machine A: Rank 0 (GPU 0), Rank 1 (GPU 1)
- Machine B: Rank 2 (GPU 0), Rank 3 (GPU 1)

GRADIENT REDUCTION (multi_gpu_async_reduce_gradient):

Template function that reduces gradients across GPUs:

Parameters:
- pointers[N]: Array of gradient buffer pointers
- pointers_sizes[N]: Size of each buffer
- config: Multi-GPU configuration
- compute_stream: CUDA stream for computation

Algorithm:
1. Record event on compute stream (marks completion of backward pass)
2. Wait for event on NCCL stream (ensures gradients are ready)
3. Launch NCCL group (batches multiple operations):
   - Stage 0: ncclAllReduce with ncclAvg (average gradients)
   - Stage 1: ncclReduceScatter with ncclAvg (partition + average)
4. NCCL operations run asynchronously on nccl_stream

Advantages:
- Asynchronous: Doesn't block CPU or compute stream
- Overlapping: Communication can overlap with next forward pass
- Batching: Multiple gradients in single NCCL group (more efficient)
- Deterministic: No race conditions or atomics

SHARDING UTILITIES (multi_gpu_get_shard_offset):

Computes which partition of a tensor this GPU owns:

Given tensor with 'elements' and zero_stage:
- If zero_stage >= shard_at_stage: Return this GPU's partition
  * offset = rank * (elements / num_processes)
  * size = elements / num_processes
- Otherwise: Return full tensor (no sharding at this stage)

Example: 1000 parameters, 4 GPUs, Stage 1:
- GPU 0: offset=0, size=250 (elements 0-249)
- GPU 1: offset=250, size=250 (elements 250-499)
- GPU 2: offset=500, size=250 (elements 500-749)
- GPU 3: offset=750, size=250 (elements 750-999)

SYNCHRONIZATION (multi_gpu_barrier):

Ensures all GPUs reach the same point before continuing:
- Uses ncclAllReduce on a dummy buffer
- Synchronizes both NCCL and CUDA streams
- Used for checkpointing, validation, etc.

MEMORY AND COMMUNICATION COSTS:

For model with Ψ parameters on N GPUs:

Stage 0 (Data Parallel):
- Memory per GPU: 4Ψ (FP32) or 2Ψ (BF16) params
                + 12Ψ (FP32) optimizer state
                = ~16Ψ bytes total
- Communication: 2Ψ per step (all-reduce gradients)

Stage 1 (Optimizer Sharding):
- Memory per GPU: 2Ψ params + 12Ψ/N optimizer state
                ≈ 2Ψ + 12Ψ/N bytes (4× savings for N=4)
- Communication: 2Ψ per step (reduce-scatter gradients)

Stage 2 (Gradient Sharding):
- Memory per GPU: 2Ψ params + 2Ψ/N gradients + 12Ψ/N optimizer
                ≈ 2Ψ + 14Ψ/N bytes
- Communication: 2Ψ per step (reduce-scatter)

Stage 3 (Full Sharding):
- Memory per GPU: 14Ψ/N bytes (N× savings!)
- Communication: 4Ψ per step (2× all-gather for forward+backward)

PERFORMANCE CONSIDERATIONS:

1. Network Bandwidth:
   - NVLink: 300-600 GB/s (intra-node)
   - InfiniBand: 100-200 GB/s (inter-node)
   - Ethernet: 10-100 GB/s (slower)
   - Stage 1/2 preferred for fast interconnects
   - Stage 3 only for extreme memory constraints

2. Overlap Opportunities:
   - Gradients reduced as soon as computed (layer-by-layer)
   - Next forward pass can start while reduction ongoing
   - ~30-50% communication hiding possible

3. Scaling Efficiency:
   - Stage 0/1: Near-linear scaling up to bandwidth limit
   - Stage 2: Good scaling with fast interconnect
   - Stage 3: Communication overhead limits scaling

IMPLEMENTATION NOTES:

- Uses CUDA events (not cudaStreamSynchronize) to avoid CPU/GPU sync
- NCCL operations are asynchronous (return immediately)
- Template functions for compile-time size checking (safety)
- printf0 macro: Only rank 0 prints (avoids duplicate output)

REFERENCES:
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
  (Rajbhandari et al., 2019): https://arxiv.org/abs/1910.02054
- NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
- DeepSpeed ZeRO: https://www.deepspeed.ai/tutorials/zero/
*/

#ifndef LLMC_ZERO_CUH
#define LLMC_ZERO_CUH

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>

#ifdef MULTI_GPU
#include <nccl.h>
#ifdef USE_MPI
#include <mpi.h>
#endif
#endif

// defines: fcloseCheck, fwriteCheck, scloseCheck, sclosesocketCheck
#include "utils.h"

// ----------------------------------------------------------------------------
// Multi-GPU related
#ifdef MULTI_GPU

#if defined(ENABLE_FP32)
const ncclDataType_t ncclFloatX = ncclFloat;
#elif defined(ENABLE_FP16)
const ncclDataType_t ncclFloatX = ncclHalf;
#else // Default to bfloat16
const ncclDataType_t ncclFloatX = ncclBfloat16;
#endif

void nccl_check(ncclResult_t status, const char *file, int line) {
    if (status != ncclSuccess) {
        printf("[NCCL ERROR] at file %s:%d:\n%s\n", file, line, ncclGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}
#define ncclCheck(err) (nccl_check(err, __FILE__, __LINE__))

#ifdef USE_MPI
void mpi_check(int status, const char *file, int line) {
    if (status != MPI_SUCCESS) {
        char mpi_error[4096];
        int mpi_error_len = 0;
        assert(MPI_Error_string(status, &mpi_error[0], &mpi_error_len) == MPI_SUCCESS);
        printf("[MPI ERROR] at file %s:%d:\n%.*s\n", file, line, mpi_error_len, mpi_error);
        exit(EXIT_FAILURE);
    }
}
#define mpiCheck(err) (mpi_check(err, __FILE__, __LINE__))
#endif

#endif // MULTI_GPU

// ----------------------------------------------------------------------------
// Parameters specific to training on multiple GPUs.
typedef struct {
    int process_rank;      // Rank of this process among all processes. 0 if no multi-GPU.
    int num_processes;     // Total number of processes. 1 if no multi-GPU.
    int local_device_idx;  // This process GPU index on current machine. 0 if no multi-GPU.

    // Zero Redundancy Optimizer stage - https://fairscale.readthedocs.io/en/stable/deep_dive/oss_sdp_fsdp.html
    // 0-Disabled
    // 1-Optimizer State Sharding (OSS)
    // 2-Optimizer + Gradient State Sharding (SDP)
    // 3-Optimizer + Gradient + Horizontal Model Sharding (FSDP)
    int zero_stage;
    size_t shard_num_parameters;
#ifdef MULTI_GPU
    ncclComm_t nccl_comm;       // NCCL communication primitive, used for collective multi-GPU work.
    cudaStream_t nccl_stream;   // CUDA Stream to perform NCCL operations.
    cudaEvent_t compute_nccl_sync; // Event used to synchronize NCCL with the compute
    float* unified_buffer;
#endif
} MultiGpuConfig;

// one global variable to hold the multi-GPU configuration for this process
// inline, so we can include this header multiple times without getting multiple definitions
inline MultiGpuConfig multi_gpu_config;

#ifdef MULTI_GPU

#ifdef _WIN32
void send_nccl_id_to_clients_windows(ncclUniqueId *nccl_id, SOCKET client_sockets[], int num_clients) {
    for (int i = 0; i < num_clients; ++i) {
        if (send(client_sockets[i], (const char *)nccl_id, sizeof(*nccl_id), 0) == SOCKET_ERROR) {
            printf("Failed to send nccl_id");
            WSACleanup();
            exit(EXIT_FAILURE);
        }
        closesocketCheck(client_sockets[i]);
    }
}
#else
void send_nccl_id_to_clients(ncclUniqueId *nccl_id, int client_sockets[], int num_clients) {
    for (int i = 0; i < num_clients; ++i) {
        if (send(client_sockets[i], nccl_id, sizeof(*nccl_id), 0) == -1) {
            printf("Failed to send nccl_id");
            exit(EXIT_FAILURE);
        }
        scloseCheck(client_sockets[i]);
    }
}
#endif

#ifdef _WIN32
// Same as get_nccl_id_via_tcp but for Windows
ncclUniqueId get_nccl_id_via_tcp_windows(MultiGpuConfig* result, const char* server_ip) {
    ncclUniqueId nccl_id;

    int SERVER_PORT = 12345;  // hardcoded an arbitrary port number between 1024 and 49151 (registered ports)
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        printf("WSAStartup failed");
        exit(EXIT_FAILURE);
    }

    if (result->process_rank == 0) {
        ncclCheck(ncclGetUniqueId(&nccl_id));

        int MAX_CLIENTS = result->num_processes - 1;
        SOCKET client_sockets[MAX_CLIENTS];
        int num_clients = 0;
        SOCKET server_socket, new_socket;
        struct sockaddr_in address;
        int addrlen = sizeof(address);

        // Step 1) create a server TCP socket
        if ((server_socket = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
            printf("Socket failed");
            WSACleanup();
            exit(EXIT_FAILURE);
        }

        // Step 2) set the server address and port
        address.sin_family = AF_INET;  // IPv4
        address.sin_addr.s_addr = inet_addr(server_ip);
        address.sin_port = htons(SERVER_PORT);

        // Step 3) bind the socket to the address and port
        if (bind(server_socket, (struct sockaddr *)&address, sizeof(address)) == SOCKET_ERROR) {
            printf("Bind failed");
            closesocketCheck(server_socket);
            WSACleanup();
            exit(EXIT_FAILURE);
        }

        // Step 4) MAX_CLIENTS specifies the maximum number of clients that can be queued for this server
        if (listen(server_socket, MAX_CLIENTS) == SOCKET_ERROR) {
            printf("Listen failed");
            closesocketCheck(server_socket);
            WSACleanup();
            exit(EXIT_FAILURE);
        }

        // Step 5) accept connections from clients
        printf("Waiting for clients to connect...\n");
        while (num_clients < MAX_CLIENTS) {
            if ((new_socket = accept(server_socket, (struct sockaddr *)&address, &addrlen)) == INVALID_SOCKET) {
                printf("Accept failed");
                closesocketCheck(server_socket);
                WSACleanup();
                exit(EXIT_FAILURE);
            }
            client_sockets[num_clients++] = new_socket;
            printf("Client %d connected\n", num_clients);
        }

        // Step 6) send the NCCL ID to all clients
        send_nccl_id_to_clients_windows(&nccl_id, client_sockets, num_clients);
        printf("NCCL ID sent to all clients\n");

        closesocketCheck(server_socket);
    } else {
        int num_connection_attempts = 5;
        int time_to_sleep = 2;
        SOCKET client_socket;
        struct sockaddr_in serv_addr;

        // Step 1) create a client TCP socket
        if ((client_socket = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
            printf("Socket creation error");
            WSACleanup();
            exit(EXIT_FAILURE);
        }

        // Step 2) set the server address and port
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(SERVER_PORT);
        if (inet_pton(AF_INET, server_ip, &serv_addr.sin_addr) <= 0) {
            printf("Invalid address or address not supported");
            closesocketCheck(client_socket);
            WSACleanup();
            exit(EXIT_FAILURE);
        }

        // Step 3) Try to connect to the server - retry up to `num_connection_attempts` times if the connection fails
        while (connect(client_socket, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) == SOCKET_ERROR) {
            printf("%d Connection failed, retrying in %d seconds\n", result->process_rank, time_to_sleep);
            if (--num_connection_attempts == 0) {
                printf("Failed to connect to the server\n");
                closesocketCheck(client_socket);
                WSACleanup();
                exit(EXIT_FAILURE);
            }
            Sleep(time_to_sleep * 1000);
        }

        // Step 4) receive the NCCL ID from the server
        if (recv(client_socket, (char *)&nccl_id, sizeof(nccl_id), 0) <= 0) {
            printf("Failed to receive nccl_id");
            closesocketCheck(client_socket);
            WSACleanup();
            exit(EXIT_FAILURE);
        }

        printf("Received NCCL ID\n");
        closesocketCheck(client_socket);
    }

    WSACleanup();
    return nccl_id;
}
#else
ncclUniqueId get_nccl_id_via_tcp(MultiGpuConfig* result, const char* server_ip) {
    ncclUniqueId nccl_id;

    int SERVER_PORT = 12345;  // hardcoded an arbitrary port number between 1024 and 49151 (registered ports)
    if (result->process_rank == 0) {
        ncclCheck(ncclGetUniqueId(&nccl_id));

        int MAX_CLIENTS = result->num_processes - 1;
        int client_sockets[MAX_CLIENTS];
        int num_clients = 0;
        int server_socket, new_socket;
        struct sockaddr_in address;
        int addrlen = sizeof(address);
        int opt = 1;

        // Step 1) create a server TCP socket
        if ((server_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
            printf("Socket failed");
            exit(EXIT_FAILURE);
        }

        // Step 2) set socket options
        // SOL_SOCKET - means that option is configured at socket level
        // SO_REUSEADDR - allows to bind to an address which is in a TIME_WAIT state (already used by another socket) - useful when restarting the server
        // SO_REUSEPORT - allows to bind to the same port multiple times
        if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
            printf("Setsockopt failed");
            exit(EXIT_FAILURE);
        }

        // Step 3) set the server address and port
        address.sin_family = AF_INET;  // IPv4
        address.sin_addr.s_addr = inet_addr(server_ip); // alternatively use INADDR_ANY to bind to all interfaces, currently we only allow ethernet
        address.sin_port = htons(SERVER_PORT);

        // Step 4) bind the socket to the address and port
        if (bind(server_socket, (struct sockaddr *)&address, sizeof(address)) < 0) {
            printf("Bind failed");
            exit(EXIT_FAILURE);
        }

        // Step 5) MAX_CLIENTS specifies the maximum number of clients that can be queued for this server
        if (listen(server_socket, MAX_CLIENTS) < 0) {
            printf("Listen failed");
            exit(EXIT_FAILURE);
        }

        // Step 6) accept connections from clients
        printf("Waiting for clients to connect...\n");
        while (num_clients < MAX_CLIENTS) {
            if ((new_socket = accept(server_socket, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
                printf("Accept failed");
                exit(EXIT_FAILURE);
            }
            client_sockets[num_clients++] = new_socket;
            printf("Client %d connected\n", num_clients);
        }

        // Step 7) send the NCCL ID to all clients
        send_nccl_id_to_clients(&nccl_id, client_sockets, num_clients);
        printf("NCCL ID sent to all clients\n");

        scloseCheck(server_socket);
    } else {
        int num_connection_attempts = 5;
        int time_to_sleep = 2;
        int client_socket;
        struct sockaddr_in serv_addr;

        // Step 1) create a client TCP socket
        if ((client_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
            printf("Socket creation error");
            exit(EXIT_FAILURE);
        }

        // Step 2) set the server address and port
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(SERVER_PORT);
        if (inet_pton(AF_INET, server_ip, &serv_addr.sin_addr) <= 0) {
            printf("Invalid address or address not supported");
            exit(EXIT_FAILURE);
        }

        // Step 3) Try to connect to the server - retry up to `num_connection_attempts` times if the connection fails
        while (connect(client_socket, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
            printf("%d Connection failed, retrying in %d seconds\n", result->process_rank, time_to_sleep);
            if (--num_connection_attempts == 0) {
                printf("Failed to connect to the server\n");
                exit(EXIT_FAILURE);
            }
            sleep(time_to_sleep);
        }

        // Step 4) receive the NCCL ID from the server
        if (recv(client_socket, &nccl_id, sizeof(nccl_id), 0) <= 0) {
            printf("Failed to receive nccl_id");
            exit(EXIT_FAILURE);
        }

        printf("Received NCCL ID\n");
        scloseCheck(client_socket);
    }

    return nccl_id;
}
#endif

ncclUniqueId get_nccl_id_via_fs(MultiGpuConfig* result, char* fs_path) {
    // Works assuming that the filesystem is shared among all processes
    ncclUniqueId nccl_id;
    FILE* idFile;
    static char filename[1024];
    snprintf(filename, sizeof(filename), "%s/ncclUniqueId.sync", fs_path);

    if (result->process_rank != 0) {  // client processse should wait for the server to write to the file
        // This is a naive and not 100% robust way to synchronize the processes but it should work almost always
        sleep(2);
    }

    if (result->process_rank == 0) {
        ncclCheck(ncclGetUniqueId(&nccl_id));
        idFile = fopen(filename, "wb");
        assert(idFile != NULL);
        fwriteCheck(&nccl_id, sizeof(nccl_id), 1, idFile);
        fcloseCheck(idFile);
    } else {
        // Other ranks wait until the file is available and read the unique ID
        do {
            sleep(1);  // 1 second
            idFile = fopen(filename, "rb");
            if (idFile != NULL) break;
        } while (idFile == NULL);
        freadCheck(&nccl_id, sizeof(nccl_id), 1, idFile);
        fcloseCheck(idFile);
    }

    return nccl_id;
}

#ifdef USE_MPI
// Determine which GPU this process should use.
// Processes on the same machines use different GPU indicies. Processes on other machines don't.
// Copied from NCCL examples: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-2-one-device-per-process-or-thread
int multi_gpu_get_local_device_idx(int process_rank, int num_processes) {
    char hostname[1024];
    hostname[1023] = '\0';
    // All processes on the same machine will share the same hostname.
    gethostname(hostname, 1023);
    for (int i=0; i < 1024; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            break;
        }
    }
    uint64_t hostname_hash = 5381u;
    for (int c = 0; hostname[c] != '\0'; c++){ hostname_hash = ((hostname_hash << 5u) + hostname_hash) ^ hostname[c]; }

    // Distribute all hostname hashes to all processes.
    uint64_t* all_hostsname_hashes = (uint64_t*)malloc(num_processes * sizeof(uint64_t));
    all_hostsname_hashes[process_rank] = hostname_hash;
    mpiCheck(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_hostsname_hashes, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));

    // Identify which GPU we need to use.
    int local_device_idx = 0;
    for (int current_process = 0; current_process < num_processes; ++current_process) {
        if (current_process == process_rank) {
        // Found my gpu, local_device_idx now has my target GPU index.
        break;
        }
        if (all_hostsname_hashes[current_process] == all_hostsname_hashes[process_rank]) {
        // This process ID runs on the same machine, but it's not me, skip this GPU
        local_device_idx++;
        }
    }

    free(all_hostsname_hashes);
    return local_device_idx;
}
#endif

#endif

MultiGpuConfig multi_gpu_config_init(int num_processes, int process_rank, int gpus_per_node, char* server_ip, char* fs_path, char* init_method) {
#ifdef MULTI_GPU
    MultiGpuConfig result;
    ncclUniqueId nccl_id;
    // Get nccl_id using MPI, TCP, or FS (file system synchronization) methods
    // On newer slurm versions (slurm-wlm package) PMIx is disabled so we can not use MPI for NCCL init in multi node setup
    if (strcmp(init_method, "mpi") == 0) {
        #ifdef USE_MPI
        mpiCheck(MPI_Init(NULL, NULL));
        mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &result.process_rank));
        mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &result.num_processes));
        result.local_device_idx = multi_gpu_get_local_device_idx(result.process_rank, result.num_processes);
        if (result.process_rank == 0) {
            ncclCheck(ncclGetUniqueId(&nccl_id));
        }
        mpiCheck(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
        #else
        printf("MPI support is disabled. Please enable MPI support to use MPI-based NCCL-init method.\n");
        exit(EXIT_FAILURE);
        #endif
    } else {
        result.process_rank = process_rank;
        result.num_processes = num_processes;
        result.local_device_idx = process_rank % gpus_per_node;
        if (strcmp(init_method, "tcp") == 0) {
            #ifdef _WIN32
            nccl_id = get_nccl_id_via_tcp_windows(&result, server_ip);
            #else
            nccl_id = get_nccl_id_via_tcp(&result, server_ip);
            #endif
        } else if (strcmp(init_method, "fs") == 0) {
            nccl_id = get_nccl_id_via_fs(&result, fs_path);
        } else {
            printf("Invalid NCCL-init method\n");
            exit(EXIT_FAILURE);
        }
    }
    cudaCheck(cudaSetDevice(result.local_device_idx));
    ncclCheck(ncclCommInitRank(&result.nccl_comm, result.num_processes, nccl_id, result.process_rank));
    cudaCheck(cudaStreamCreate(&result.nccl_stream));
    // event without timing for maximum performance
    cudaCheck(cudaEventCreate(&result.compute_nccl_sync, cudaEventDisableTiming));
    nvtxNameCudaStreamA(result.nccl_stream, "nccl stream");
    nvtxNameCudaEventA(result.compute_nccl_sync, "nccl compute sync");
    cudaCheck(cudaMallocManaged(&result.unified_buffer, sizeof(float)));
    return result;
#else
    printf("Multi-GPU support is disabled. Using a single GPU.\n");
    cudaCheck(cudaSetDevice(0));
    MultiGpuConfig result;
    result.process_rank = 0;
    result.num_processes = 1;
    result.local_device_idx = 0;
    return result;
#endif
}

void multi_gpu_config_free(MultiGpuConfig* config) {
#ifdef MULTI_GPU
    ncclCheck(ncclCommDestroy(config->nccl_comm));
    cudaCheck(cudaStreamDestroy(config->nccl_stream));
    cudaCheck(cudaEventDestroy(config->compute_nccl_sync));
    cudaCheck(cudaFree(config->unified_buffer));
    #ifdef USE_MPI
    mpiCheck(MPI_Finalize());
    #endif
#endif
}

void multi_gpu_barrier(const MultiGpuConfig* config) {
#ifdef MULTI_GPU
    if (config->num_processes > 1) {
        ncclCheck(ncclAllReduce(config->unified_buffer, config->unified_buffer, sizeof(float), ncclFloat, ncclSum, config->nccl_comm, config->nccl_stream));
    }
    cudaCheck(cudaDeviceSynchronize());
#endif
}

// Offset and size of a tensor shard
typedef struct {
    ptrdiff_t offset;
    size_t size;
} ShardInfo;

// Get info about sharding for a tensor of elements many numbers
ShardInfo multi_gpu_get_shard_offset(size_t elements, const MultiGpuConfig* config, int shard_at_stage) {
    const int nproc = config->num_processes;
    if(config->zero_stage >= shard_at_stage) {
        if (elements % nproc != 0) {
            fprintf(stderr, "Number of elements %zu must be a multiple of the number of processes %d\n", elements, nproc);
            exit(EXIT_FAILURE);
        }
        return {(ptrdiff_t) (config->process_rank * (elements / nproc)), elements / nproc};
    } else {
        return {0, elements};
    }
}

// Block NCCL stream until computations on compute_stream are done, then aggregate multiple pointers in an NCCL group.
// This can work either as an all-reduce (i.e., no ZeRo), or a reduce-scatter (ZeRO 1).
// The awkward `(&pointers)[N]` syntax ensures we are capturing the parameters as sized arrays, so that it becomes impossible
// to call this function if pointers and pointers_sizes do not match.
template<int N>
void multi_gpu_async_reduce_gradient(
        floatX* const (&pointers)[N], const size_t (&pointers_sizes)[N],
        MultiGpuConfig* config, cudaStream_t compute_stream) {
    if (config->num_processes == 1) {
        return; // no multi-GPU, just exit.
    }

#ifdef MULTI_GPU
    NVTX_RANGE_FN();
    // mark an event on the compute stream, and immediately wait on this in the nccl stream
    // this means that the nccl stream won't start executing before all compute kernels that
    // have been submitted before this point have finished.
    // by using an event instead of cudaSyncStream, we avoid having to synchronize the host, and
    // can enqueue new work to the GPU right away.
    cudaCheck(cudaEventRecord(config->compute_nccl_sync, compute_stream));
    cudaCheck(cudaStreamWaitEvent(config->nccl_stream, config->compute_nccl_sync));
    ncclCheck(ncclGroupStart()); // NCCL group: aggregate all pointers in a single NCCL GPU kernel.
    for (int i = 0; i < N; ++i) {
        if(config->zero_stage == 0) {
            ncclCheck(ncclAllReduce(
                    pointers[i], pointers[i],
                    pointers_sizes[i],
                    ncclFloatX, ncclAvg,
                    config->nccl_comm, config->nccl_stream
            ));
        } else if(config->zero_stage == 1) {
            assert(pointers_sizes[i] % config->num_processes == 0);
            size_t shard_size = pointers_sizes[i] / config->num_processes;
            ptrdiff_t shard_offset = (ptrdiff_t)shard_size * config->process_rank;
            ncclCheck(ncclReduceScatter(
                    pointers[i], pointers[i] + shard_offset,
                    shard_size,
                    ncclFloatX, ncclAvg,
                    config->nccl_comm, config->nccl_stream
            ));
        }
    }
    ncclCheck(ncclGroupEnd());
#endif
}

// convenience macro that only prints if the rank of process is zero
#define printf0(...) if (::multi_gpu_config.process_rank == 0) { printf(__VA_ARGS__); }

void set_zero_configs(MultiGpuConfig* config, int zero_stage, size_t total_parameters) {
    config->zero_stage = 0;
    config->shard_num_parameters = total_parameters;
    // Check the Zero Stage and define sharding parameters
    if (zero_stage == 0) {
        printf0("| Zero Optimization is disabled                                              |\n");
    }
    else if (zero_stage == 1) {
        if (total_parameters % config->num_processes != 0) {
            printf0("| Zero Optimization is disabled, Can't equally partition parameters          |\n");
            config->zero_stage = 0;
        }
        else {
            config->zero_stage = 1;
            config->shard_num_parameters = total_parameters / config->num_processes;
        }
    }
    else{
        printf0("| Disabling Zero Optimization, Zero Stage2 and Stage3 are not yet supported  |\n");
        config->zero_stage = 0;
    }
}

// Compute sum of a single CPU value across all GPU processes. No-op when multi-GPU is disabled.
float multi_gpu_cpu_float_sum(float value, MultiGpuConfig* config) {
#ifdef MULTI_GPU
    if (config->num_processes == 1) return value;

    float* unified_buffer = config->unified_buffer;
    *unified_buffer = value;
    ncclCheck(ncclAllReduce(unified_buffer, unified_buffer, sizeof(float), ncclFloat, ncclSum, config->nccl_comm, config->nccl_stream));
    cudaCheck(cudaDeviceSynchronize());
    return *unified_buffer;
#else
    return value;
#endif
}

#endif

