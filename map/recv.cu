#include <cuco/static_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <iostream>

template <typename Map>
__global__ void find_in_map_kernel(Map map, int* queries, int* results, int num_queries) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_queries) {
        auto found = map.find(queries[idx]);

         if (found != map.end()) {
            // If the key is found, store the value
            results[idx] = found->second; // Assuming the iterator points to a pair {key, value}
        } else {
            // Key not found, handle as needed
            results[idx] = -1;
        }
    }
}

int main() {
    cudaIpcMemHandle_t handle;
    int fd = open("ipc_handle.bin", O_RDONLY);
    read(fd, &handle, sizeof(handle));
    close(fd);

    auto find_ref;
    cudaIpcOpenMemHandle((void**)&find_ref, handle, cudaIpcMemLazyEnablePeerAccess);

    find_in_map_kernel<<<(queries.size() + 255) / 256, 256>>>(find_ref, d_queries, d_results, queries.size());

    cudaMemcpy(thrust::raw_pointer_cast(results.data()), d_results, queries.size() * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Query results:\n";
    for (int i = 0; i < queries.size(); ++i) {
        std::cout << "Key " << queries[i] << " -> Value ";
        if (results[i] == -1) {
            std::cout << "NOT FOUND";
        } else {
            std::cout << results[i];
        }
        std::cout << "\n";
    }

    cudaFree(d_queries);
    cudaFree(d_results);

    return 0;
}