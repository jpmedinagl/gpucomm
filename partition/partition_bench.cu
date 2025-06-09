#include <iostream>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>
#include <random>

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

__device__ int hash(int val, int npartitions) {
    return abs(val) % npartitions;
}

__global__ void partition_kernel_hash (
    int * tuples,
    int ntuples,
    int ** partitions,
    int npartitions,
    int * partition_offsets
) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;

    // loop - each thread has multiple tuples

    for (int i = thread_id; i < ntuples; i += gridSize) {
        int tuple = tuples[i];
        int dest = hash(tuple, npartitions);
    
        int offset = atomicAdd(&partition_offsets[dest], 1);
    
        partitions[dest][offset] = tuple;
    }
}

int main(int argc, char *argv[]) 
{
    int nrows = 1048576;

    int npartitions = 4;
    int tuples_per_partition = 1024 * 1024;

    // generate data
    std::vector<int> tuples = generate_tuples(nrows);
    
    int *d_tuples;
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_tuples, 
                                sizeof(int) * tuples.size()));
    CHECK_CUDA_ERR(cudaMemcpy(d_tuples, tuples.data(), 
                                sizeof(int) * tuples.size(),
                                cudaMemcpyHostToDevice));

    // create separate partitions
    std::vector<int*> h_partitions(npartitions);

    for (int i = 0; i < npartitions; ++i) {
        cudaMalloc(&h_partitions[i], sizeof(int) * tuples_per_partition);
    }

    int** d_partitions;
    cudaMalloc(&d_partitions, sizeof(int*) * npartitions);

    cudaMemcpy(d_partitions, h_partitions.data(), 
        sizeof(int*) * npartitions, cudaMemcpyHostToDevice);

    // partition offsets
    std::vector<int> partition_offsets(npartitions, 0);
    int* d_partition_offsets;
    cudaMalloc(&d_partition_offsets, sizeof(int) * npartitions);
    cudaMemcpy(d_partition_offsets, partition_offsets.data(), 
        sizeof(int) * npartitions, cudaMemcpyHostToDevice);

    // kernels
    auto start = std::chrono::high_resolution_clock::now();

    int numBlocks;
    int blockSize;

    cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, partition_kernel_hash);

    partition_kernel_hash<<<numBlocks, blockSize>>>(
        d_tuples,
        nrows,
        d_partitions,
        npartitions,
        d_partition_offsets
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

    std::cout << "Num blocks: " << numBlocks << std::endl;
    std::cout << "block SIZE: " << blockSize << std::endl;

    std::cout << "Total time: " << stage_time.count() / 1e6 << " ms" << std::endl;
    std::cout << "Total Throughput: "
        << (nrows * sizeof(int)) / (stage_time.count() / 1e9) /
                1e9
            << " GB/s" << std::endl;
    
    cudaFree(d_tuples);
    cudaFree(d_partitions);
    cudaFree(d_partition_offsets);
    
    for (int i = 0; i < npartitions; ++i) {
        cudaFree(h_partitions[i]);
    }
    
    return 0;
}