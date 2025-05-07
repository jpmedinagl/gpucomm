#include <cuco/static_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <iostream>

int main() {
    using Key = int;
    using Value = int;

    constexpr std::size_t num_keys = 100'000;
    constexpr std::size_t capacity = num_keys * 1.3;

    auto map = cuco::static_map{capacity,
                              cuco::empty_key{-1},
                              cuco::empty_value{-1},
                              cuda::std::equal_to<Key>{},
                              cuco::linear_probing<1, cuco::default_hash_function<Key>>{}};

    // Create a sequence of keys and values
    thrust::device_vector<Key> keys(num_keys);
    thrust::device_vector<Value> values(num_keys);
    thrust::sequence(keys.begin(), keys.end(), 0);    // Keys: 0, 1, 2, ..., num_keys-1
    thrust::sequence(values.begin(), values.end(), 10); // Values: 10, 11, 12, ..., num_keys+9

    // Insert key-value pairs into the map
    auto pairs_begin = thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), values.begin()));
    map.insert(pairs_begin, pairs_begin + num_keys);

    // Define some queries
    thrust::device_vector<Key> queries = {0, 1, num_keys-1, num_keys, -1};
    thrust::device_vector<Value> results(queries.size(), -1);

    // Launch a kernel to perform the lookups
    int* d_queries;
    int* d_results;
    cudaMalloc(&d_queries, queries.size() * sizeof(int));
    cudaMalloc(&d_results, queries.size() * sizeof(int));
    cudaMemcpy(d_queries, thrust::raw_pointer_cast(queries.data()), queries.size() * sizeof(int), cudaMemcpyDeviceToDevice);

    auto find_ref = map.ref(cuco::find);

    cudaIpcMemHandle_t handle;
    cudaIpcGetMemHandle(&handle, find_ref);

    int fd = open("ipc_handle.bin", O_CREAT | O_WRONLY, 0666);
    write(fd, &handle, sizeof(handle));
    close(fd);

    return 0;
}