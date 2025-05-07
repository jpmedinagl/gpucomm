#include <cuco/static_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>  // Required for host_vector
#include <thrust/iterator/zip_iterator.h>
#include <iostream>

int main() {
    using Key = int;
    using Value = int;
    
    constexpr std::size_t num_keys = 100'000;
    constexpr std::size_t capacity = num_keys * 1.3;
    
    using hash_map = cuco::static_map<Key, Value>;
    
    hash_map map{capacity, cuco::empty_key<Key>{-1}, cuco::empty_value<Value>{-1}};
    
    thrust::device_vector<Key> keys(num_keys);
    thrust::device_vector<Value> values(num_keys);
    thrust::sequence(keys.begin(), keys.end(), 0);    // Keys: 0, 1, 2, ..., num_keys-1
    thrust::sequence(values.begin(), values.end(), 10); // Values: 10, 11, 12, ..., num_keys+9
    
    auto pairs_begin = thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), values.begin()));
    
    map.insert(pairs_begin, pairs_begin + num_keys);
    
    constexpr std::size_t num_queries = 5;
    thrust::device_vector<Key> queries = {0, 1, num_keys-1, num_keys, -1};
    thrust::device_vector<Value> results(num_queries, -1);
    
    map.find(queries.begin(), queries.end(), results.begin());
    
    thrust::host_vector<Value> host_results = results;
    std::cout << "Query results:\n";
    for (int i = 0; i < num_queries; ++i) {
        std::cout << "Key " << queries[i] << " -> Value ";
        if (host_results[i] == -1) {
            std::cout << "NOT FOUND";
        } else {
            std::cout << host_results[i];
        }
        std::cout << "\n";
    }
    
    return 0;
}