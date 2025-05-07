#include <cuco/static_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>
#include <iostream>

using key_type = int;
using value_type = int;
using pair_type = thrust::pair<key_type, value_type>;

__global__ void lookup_kernel(cuco::static_map<key_type, value_type>* map, key_type query) {
    auto result = map->find(query);
    if (result != map->end()) {
        printf("Found key %d -> value %d\n", query, result->second);
    } else {
        printf("Key %d not found\n", query);
    }
}

int main() {
    thrust::host_vector<pair_type> h_pairs{
        {1, 100},
        {2, 200},
        {3, 300}
    };

    thrust::device_vector<pair_type> d_pairs = h_pairs;

    std::size_t num_items = d_pairs.size();
    std::size_t capacity = static_cast<std::size_t>(num_items * 2); // More than num_items

    cuco::static_map<key_type, value_type> map{
        capacity,
        cuco::empty_key<key_type>{-1},
        cuco::empty_value<value_type>{-1},
        cuco::default_hash_function<key_type>{},
        cuco::default_equal_to<key_type>{}
    };

    map.insert(d_pairs.begin(), d_pairs.end());

    lookup_kernel<<<1, 1>>>(&map, 2);
    lookup_kernel<<<1, 1>>>(&map, 1);
    cudaDeviceSynchronize();

    return 0;
}
