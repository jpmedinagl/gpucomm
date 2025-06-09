#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/scan.h>

constexpr int LOG_PART = 2;
constexpr int NUM_PART = 1 << LOG_PART;
constexpr int TUP_PER_TR = 8;
constexpr int BLK_SIZE = 256;

#define CHECK_CUDA_ERR(call)                                                   \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA check error: %s\n", cudaGetErrorString(err));    \
        exit(EXIT_FAILURE);                                                    \
        }                                                                      \
    }

std::vector<int> generate_tuples(int nrows, uint32_t seed = 1) {
    std::vector<int> rows;
    rows.reserve(nrows);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> range(0, 255);

    for (int64_t i = 0; i < nrows; ++i) {
        // Fill with random data
        rows.emplace_back(range(rng));
    }

    return rows;
}

__device__ uint32_t hash(int val, int npartitions) {
    return abs(val) % npartitions;
}

template<typename Tuple>
__global__ void build_histogram_p1(
    const Tuple * tuples,
    int ntuples,
    uint32_t * global_hist
) {
    // shared histogram for block
    __shared__ uint32_t shared_hist[NUM_PART];

    for (int i = 0; i < NUM_PART; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();

    // hash into partition
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (uint32_t idx = tid * TUP_PER_TR; idx < ntuples; idx += gridDim.x * blockDim.x * TUP_PER_TR) {
        Tuple reg[TUP_PER_TR];

        #pragma unroll
        for (int v = 0; v < TUP_PER_TR; ++v) {
            if (idx + v < ntuples) {
                reg[v] = tuples[idx + v];
            }
        }

        #pragma unroll
        for (int v = 0; v < TUP_PER_TR; ++v) {
            if (idx + v < ntuples) {
                uint32_t part = hash(reg[v], NUM_PART);
                atomicAdd(&shared_hist[part], 1);
            }
        }
    }
    __syncthreads();

    // shared histograms into global
    for (int i = threadIdx.x; i < NUM_PART; i += blockDim.x) {
        global_hist[blockIdx.x * NUM_PART + i] = shared_hist[i];
    }
}

void exclusive_scan_by_key(
    uint32_t * global_hist,
    int blocks,
    int npartitions
) {
    thrust::device_vector<uint32_t> d_hist(global_hist, global_hist + blocks * npartitions);

    thrust::exclusive_scan(d_hist.begin(), d_hist.end(), d_hist.begin());
}

template<typename Tuple>
__global__ void reorder_pass_p1(
    const Tuple * tuples,
    Tuple * partitions,
    uint32_t ntuples,
    uint32_t *g_scan
) {
    // per block starting offsets
    __shared__ uint32_t shared_writept[NUM_PART];

    // define shared pointer from prefix scan
    for (int i = threadIdx.x; i < NUM_PART; i += blockDim.x) {
        shared_writept[i] = g_scan[blockIdx.x * NUM_PART + i];
    }
    __syncthreads();

    // use hash and pt to write to the partitions
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t idx = tid * TUP_PER_TR; idx < ntuples; idx += gridDim.x * blockDim.x * TUP_PER_TR) {
        Tuple reg[TUP_PER_TR];

        #pragma unroll
        for (int v = 0; v < TUP_PER_TR; ++v) {
            if (idx + v < ntuples) reg[v] = tuples[idx + v];    
        }

        #pragma unroll
        for (int v = 0; v < TUP_PER_TR; ++v) {
            if (idx + v < ntuples) {
                uint32_t part = hash(reg[v], NUM_PART);
                uint32_t pos = atomicAdd(&shared_writept[part], 1);
                partitions[pos] = reg[v];
            }
        }
    }
}

int main(int argc, char *argv[]) 
{   
    int blocks = 1024;

    int ntuples = 1048576;

    int npartitions = 4;
    int partition_size = ntuples;

    // generate data
    std::vector<int> tuples = generate_tuples(ntuples);

    int *d_tuples;
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_tuples, sizeof(int) * tuples.size()));
    CHECK_CUDA_ERR(cudaMemcpy(d_tuples, tuples.data(), sizeof(int) * tuples.size(), cudaMemcpyHostToDevice));

    // create separate partitions
    int* d_partitions;
    cudaMalloc(&d_partitions, sizeof(int) * npartitions * partition_size);
    
    // histograms
    uint32_t *d_hist;
    cudaMalloc(&d_hist, blocks * NUM_PART * sizeof(uint32_t));

    // kernels
    auto start = std::chrono::high_resolution_clock::now();   

    build_histogram_p1<<<blocks, BLK_SIZE>>>(
        d_tuples, ntuples, d_hist
    );

    exclusive_scan_by_key(d_hist, blocks, NUM_PART);

    reorder_pass_p1<<<blocks, BLK_SIZE>>>(
        d_tuples, d_partitions, ntuples, d_hist
    );
    
    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto stage_time = 
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // verification

    // loop through various partitions - print the first 10 elements
    // std::cout << "tuples: ";
    // for (int i = 0; i < 10; i++) {
    //     std::cout << tuples[i] << " ";
    // }
    // std::cout << std::endl;

    // for (int i = 0; i < npartitions; i++) {
    //     std::cout << "partition " << i << ": ";
    //     std::vector<int> host_partition(tuples_per_partition);

    //     cudaMemcpy(host_partition.data(), h_partitions[i], 
    //             sizeof(int) * tuples_per_partition, 
    //             cudaMemcpyDeviceToHost);

    //     for (int j = 0; j < 10; ++j) {
    //         std::cout << host_partition[j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    
    std::cout << "============= Total time and Throughput =============\n";

    std::cout << "Num blocks: " << blocks << std::endl;
    std::cout << "Block size: " << BLK_SIZE << std::endl;

    std::cout << "Total time: " << stage_time.count() / 1e6 << " ms" << std::endl;
    std::cout << "Total Throughput: "
        << (ntuples * sizeof(int)) / (stage_time.count() / 1e9) /
                1e9
            << " GB/s" << std::endl;
    
    cudaFree(d_tuples);
    cudaFree(d_partitions);
    cudaFree(d_hist);
    
    return 0;
}