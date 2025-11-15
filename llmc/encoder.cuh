/*
==============================================================================
Encoder Layer - Token and Positional Embeddings
==============================================================================

PURPOSE:
This file implements the embedding layer for transformer models (GPT-2 style),
which combines two types of encodings:
  1. Token embeddings (wte): Map each input token ID to a learnable vector
  2. Positional embeddings (wpe): Add position-dependent information to each token

MATHEMATICAL OPERATION:
For input token IDs inp[b, t] at batch b, position t:
  out[b, t, c] = wte[inp[b, t], c] + wpe[t, c]

This is the first operation in a transformer, converting discrete token IDs
into continuous vector representations that encode both semantic meaning (from
wte) and positional information (from wpe).

FORWARD PASS:
- Input: Token IDs (B, T) + embedding tables wte(V, C) and wpe(T, C)
- Output: Combined embeddings (B, T, C)
- Each thread processes x128::size elements (8 for BF16, enabling vectorization)
- Uses load128cs to stream data through cache without polluting it

BACKWARD PASS:
Gradients from upstream flow back to both embedding tables:

1. WPE (Position Embeddings) Backward:
   - Deterministic: Each position accumulates gradients across the batch
   - Each thread handles x128::size channels for a specific (t, c) position
   - Accumulates across batch dimension B without atomics
   - Uses stochastic rounding when converting FP32 accumulator to BF16/FP8

2. WTE (Token Embeddings) Backward:
   - More complex due to non-uniform token distribution
   - Uses bucket-based approach for determinism and load balancing:
     * Bucket = (vocabulary_index, channel_group)
     * CPU pre-processes inputs to create buckets and sort by size
     * Each warp handles WARP_SIZE * x128::size channels (256 for BF16)
     * Largest buckets processed first to minimize GPU idle time
   - Within each bucket:
     * Multiple batch-time positions may map to same token
     * Each warp accumulates gradients for all occurrences
     * Results combined in shared memory, then written to global memory
   - Fully deterministic (no race conditions or atomics)

MEMORY ACCESS PATTERNS:
- Forward: Coalesced reads from wte and wpe, coalesced writes to out
- Backward WPE: Coalesced reads from dout, single write per position
- Backward WTE: Scattered reads from dout (based on token distribution),
  accumulated in shared memory, single write per vocabulary entry

OPTIMIZATIONS:
1. Vectorized loads/stores using x128 (processes 8 BF16 values at once)
2. Streaming cache hints (__ldcs/__stcs) to avoid cache pollution
3. Bucket-based load balancing for non-uniform token distributions
4. Stochastic rounding for low-precision training (BF16/FP8)
5. CPU/GPU overlap: WPE kernel launches while CPU prepares WTE buckets

PERFORMANCE CONSIDERATIONS:
- Forward pass is memory-bandwidth bound (simple addition)
- WPE backward is efficient due to uniform access pattern
- WTE backward efficiency depends on vocabulary usage distribution
- Bucket sorting on CPU ensures balanced GPU workload
- Determinism achieved without performance loss (no atomics needed)

REFERENCES:
- Attention Is All You Need (Vaswani et al., 2017) - Introduced positional embeddings
- GPT-2 (Radford et al., 2019) - Uses learned positional embeddings
*/
#include <assert.h>
#include <stdint.h>
#include <utility>              // std::pair
#include <vector>
#include <algorithm>
#include <unordered_map>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

__global__ void encoder_forward_kernel3(floatX* out,
                               const int* inp, const floatX* wte, const floatX* wpe,
                               int B, int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int N = B * T * C;
    if (idx >= N) { return; }

    int bt = idx / C;
    int b = bt / T;
    int t = bt % T;
    int c = idx % C;

    int ix = inp[b * T + t];

    floatX* out_btc = out + b * T * C + t * C + c;
    const floatX* wte_ix = wte + ix * C + c;
    const floatX* wpe_tc = wpe + t * C + c;

    x128 packed_out;
    x128 wte128 = load128cs(wte_ix);
    x128 wpe128 = load128cs(wpe_tc);
    for (int k = 0; k < x128::size; k++) {
        packed_out[k] = (floatX)((float)wte128[k] + (float)wpe128[k]);
    }
    store128(out_btc, packed_out);
}

template <int BLOCK_SIZE=256>
__global__ void wte_backward_kernel(floatX* dwte,
                                    const int4* bucket_info, const int* workload_indices, const floatX* dout, const int* inp,
                                    unsigned int seed, int B, int T, int C) {
    // In order to be deterministic, we preprocess the inputs on the cpu into "buckets"
    // Each bucket corresponds to (WARP_SIZE * x128::size) channels for a single vocabulary token
    // Each thread handles x128::size channels, e.g. 256 per warp for BF16
    // Each block handles (BLOCK_SIZE / WARP_SIZE) elements in a single bucket in parallel
    // If a bucket has less than 8 elements, some warps will return immediately
    // If a bucket has more than 8 elements, we will loop over all of them
    // The buckets are sorted on the CPU so the largest buckets start 1st
    int bucket = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int c_per_warp = WARP_SIZE * x128::size;

    int bucket_start_idx = bucket_info[bucket].x;
    int bucket_size = bucket_info[bucket].y;
    int bucket_ix = bucket_info[bucket].z;
    int c = bucket_info[bucket].w * c_per_warp + (lane_id * x128::size);

    // Each thread handles "x128::size" channels, so at fp8, each warp would handle 512 channels
    // If C is not a multiple of this (e.g. 768), some buckets/c_groups cannot use the entire warp
    if (c >= C) { return; }
    // Exit early if this is a small bucket and this warp doesn't have any items to process
    if (warp_id >= bucket_size) { return; }

    float accum[x128::size] = {0.0f};
    __shared__ float accum_shared[x128::size * BLOCK_SIZE];

    for(int item = warp_id; item < bucket_size; item += BLOCK_SIZE/WARP_SIZE) {
        int bt = workload_indices[bucket_start_idx + item];

        const floatX* dout_btc = dout + bt * C + c;
        x128 packed_inp1 = load128cs(dout_btc);
        for (int k = 0; k < packed_inp1.size; k++) {
            accum[k] += (float)packed_inp1[k];
        }
    }

    if (warp_id != 0) {
        // we accumulate into warp 0, so only the other warps need to write to shared memory
        for (int k = 0; k < x128::size; k++) {
            accum_shared[threadIdx.x + k * BLOCK_SIZE] = accum[k];
        }
        return; // only warp 0 is needed after writing to shared memory
    }

    // Read dwte for warp 0 even if other warps are not finished yet to maximise latency tolerance
    floatX* dwte_ix = dwte + bucket_ix * C + c;
    x128 packed_in_out = load128(dwte_ix);

    // note: threads which have returned are considered synchronised by CUDA so no risk of deadlock
    __syncthreads();

    // Accumulate into warp 0's registers by reading the values of the other warps in shared memory
    for (int i = threadIdx.x+WARP_SIZE; i < min(BLOCK_SIZE, bucket_size*WARP_SIZE); i += WARP_SIZE) {
        for (int k = 0; k < x128::size; k++) {
            accum[k] += accum_shared[i + k * BLOCK_SIZE];
        }
    }

    // Add the result to dwte and write back to global memory (read-modify-write)
    for (unsigned int k = 0; k < x128::size; k++) {
        // We use stochastic rounding to go from FP32 to BF16
        // The seed is deterministic and unique for each parameter to guarantee we have determinism AND
        // to avoid **potential** issues with positionX int SquirrelNoise5 argument overflowing which is UB
        // and that somehow messing the quality of random numbers
        stochastic_rounding(accum[k] + (float)packed_in_out[k], &packed_in_out[k], seed + bucket * WARP_SIZE + threadIdx.x + k);
    }
    store128(dwte_ix, packed_in_out);
}

__global__ void wpe_backward_kernel(floatX* dwpe,
                                    const floatX* dout, const int* inp,
                                    int B, int T, int C, unsigned int seed) {
    // Each thread handles x128::size "channel positions", e.g. 256 per warp for BF16
    // For gpt2-124M BF16, C=768 and T=1024, so 3 warps per channel and 3072 warps in total
    // For each "channel position" we sum the gradients for every batch at that C/T element
    // This way each dwte element is only updated once, and the kernel is fully deterministic!
    // The previous kernel was not deterministic, as batches were aggregated with atomicAdd
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx >= T * C) { return; }

    // if C is not a multiple of WARP_SIZE*x128::size, it's OK for some warps to handle multiple t
    int t = idx / C;
    int c = idx % C;
    float accum[x128::size] = {0.0f};

    for (int b = 0; b < B; b++) {
        x128 packed_dout = load128cs(dout + (b * T * C) + (t * C) + c); // will never be read again
        for (int k = 0; k < x128::size; k++) {
            accum[k] += (float)packed_dout[k];
        }
    }

    floatX* dwpe_tc = dwpe + (t * C) + c;
    x128 packed_dwpe = load128(dwpe_tc);
    for (unsigned int k = 0; k < x128::size; k++) {
        // We use stochastic rounding to go from FP32 to BF16
        // The seed is deterministic and unique for each parameter to guarantee we have determinism AND
        // to avoid **potential** issues with positionX int SquirrelNoise5 argument overflowing which is UB
        // and that somehow messing the quality of random numbers
        stochastic_rounding(accum[k] + (float)packed_dwpe[k], &packed_dwpe[k], seed + idx + k);
    }
    store128(dwpe_tc, packed_dwpe);
}

// ----------------------------------------------------------------------------
// kernel launchers

void encoder_forward(floatX* out,
                     const int* inp, const floatX* wte, const floatX* wpe,
                     int B, int T, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    const int N = B * T * C;
    const int grid_size = CEIL_DIV(N, (int)(block_size * x128::size));
    encoder_forward_kernel3<<<grid_size, block_size, 0, stream>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

// Fully deterministic (see comments in wte_backward_kernel and wpe_backward_kernel for more details)
void encoder_backward(floatX* dwte, floatX* dwpe, floatX* scratch, // gpu outputs & scratch
                      int* workload_indices, int4* bucket_info,    // cpu scratch buffers
                      const floatX* dout, const int* inp, const int* inputs_cpu, // cpu/gpu inputs
                      int B, int T, int C, unsigned int seed, cudaStream_t stream) {
    NVTX_RANGE_FN();

    // Launch wpe kernel first (so it runs on the GPU in parallel with the CPU pre-processing for wte)
    const int block_size = 256;
    const int N = T * C / x128::size;
    const int grid_size = CEIL_DIV(N, block_size);
    wpe_backward_kernel<<<grid_size, block_size, 0, stream>>>(dwpe, dout, inp, B, T, C, seed);
    cudaCheck(cudaGetLastError());

    // check the GPU scratch buffer is large enough to hold the bucket info and workload indices
    // todo - this is trivially true given hardcoded scratch buffer size here, is this useful?
    int num_c_groups = CEIL_DIV(C, x128::size * WARP_SIZE);
    assert(B*T*num_c_groups * (sizeof(int4)+sizeof(int)) <= B*T*3*C * sizeof(floatX));

    // Step 1: Sort inputs into buckets
    int total_items = 0;
    std::unordered_map<uint64_t, std::vector<uint64_t>> buckets;
    for (uint64_t bt = 0; bt < B * T; bt++) {
        for (uint64_t c_group = 0; c_group < num_c_groups; c_group++) {
            // todo - passing c_group/inputs_cpu[bt] in data to avoid a second hash lookup is a bit hacky
            uint64_t data = bt + (c_group<<32ULL) + ((uint64_t)inputs_cpu[bt]<<42ULL);
            buckets[c_group + num_c_groups * inputs_cpu[bt]].push_back(data);
            total_items++;
        }
    }

    // Step 2: Sort buckets by size in descending order
    // this is so the largest buckets are processed first by the GPU
    // otherwise, if they started late, they would still be running with the rest of the GPU idle
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> sortedBuckets(buckets.begin(), buckets.end());
    std::sort(sortedBuckets.begin(), sortedBuckets.end(), // ugly because we don't have a typedef for the std::pair
              [](const std::pair<uint64_t, std::vector<uint64_t>>& a, const std::pair<uint64_t, std::vector<uint64_t>>& b) {
                  return a.second.size() > b.second.size();
              });

    int num_buckets = buckets.size();
    int bucket_index = 0;
    int workload_index = 0;
    for (const auto& bucket : sortedBuckets) {
        bucket_info[bucket_index].x = workload_index; // bucket start
        bucket_info[bucket_index].y = bucket.second.size(); // bucket size
        bucket_info[bucket_index].z = (bucket.second[0] >> 42ULL) & ((1ULL<<20ULL)-1); // bucket ix
        bucket_info[bucket_index].w = (bucket.second[0] >> 32ULL) & ((1ULL<<10ULL)-1); // bucket c

        for (uint64_t idx : bucket.second) {
            workload_indices[workload_index++] = (int)(idx & ((1ULL<<31ULL)-1ULL));
        }
        bucket_index++;
    }

    // Step 3: Copy data from host to device (async until the last one to avoid synchronising CPU/GPU twice)
    // todo - could use CUDA events (even without streams) to avoid CPU/GPU synchronisation completely
    int4* d_bucket_info = (int4*)scratch;
    int*  d_workload_indices = (int*)(scratch + B*T*num_c_groups * sizeof(int4));
    cudaCheck(cudaMemcpyAsync(d_bucket_info, bucket_info, num_buckets * sizeof(int4), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_workload_indices, workload_indices, total_items * sizeof(int), cudaMemcpyHostToDevice, stream));

    // Launch wte kernel
    // todo - profile block sizes on more content (depends on number of buckets and on GPU?)
    wte_backward_kernel<256><<<num_buckets, 256, 0, stream>>>(dwte, d_bucket_info, d_workload_indices, dout, inp, seed, B, T, C);
    cudaCheck(cudaGetLastError());
}
