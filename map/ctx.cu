#include <iostream>
#include <random>

#include "types.cuh"

using map_t = cuco::static_map<endpoint_id_t, forwarding_table_entry_t>;

template <typename Map>
__global__ void find_in_map_kernel(Map map, int* queries, int* results, int num_queries) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_queries) {
        auto found = map.find(queries[idx]);

         if (found != map.end()) {
            results[idx] = found->second;
        } else {
            results[idx] = -1;
        }
    }
}

template <typename MultiMap>
__global__ void search_in_kernel(MultiMap map, 
    thrust::device_vector<channel_id_t> keys_to_find,
    thrust::device_vector<cuco::pair<channel_id_t, column_type_t>> d_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread is within bounds
    if (idx < keys_to_find.size()) {
        channel_id_t key = keys_to_find[idx];

        // Find how many values are associated with the key
        auto output_size = map.count_outer(&key, &key + 1);

        // Allocate result space for this key
        thrust::device_vector<cuco::pair<channel_id_t, column_type_t>> local_results(output_size);

        // Retrieve key-value pairs
        auto output_end = map.retrieve_outer(&key, &key + 1, local_results.begin());
        int retrieve_size = output_end - local_results.begin();

        // Copy result to global results array (ensure no race condition)
        for (int i = 0; i < retrieve_size; ++i) {
            d_results[idx * retrieve_size + i] = local_results[i];
        }
    }
}

void inspect_ctx(EndpointGPUCtx* gpu_ctx) {
    printf("Forwarding Table (Count: %d):\n", gpu_ctx->forwarding_table_count);
    for (int i = 0; i < gpu_ctx->forwarding_table_count; ++i) {
        auto entry = gpu_ctx->forwarding_table[i];
        printf("  Dest EP: %u, Next hops count: %d\n", entry.dest_ep_id, entry.next_hops_count);
        for (int j = 0; j < entry.next_hops_count; ++j) {
            printf("    Next hop[%d]: %u\n", j, entry.next_hops[j]);
        }
    }

    printf("\nSchemas (Count: %d):\n", gpu_ctx->schemas_count);
    for (int i = 0; i < gpu_ctx->schemas_count; ++i) {
        auto schema = gpu_ctx->schemas[i];
        printf("  Channel ID: %u, Columns count: %d\n", schema.channel_id, schema.columns_count);
        for (int j = 0; j < schema.columns_count; ++j) {
            printf("    Column[%d] type: %d\n", j, schema.column_types[j]);
        }
    }

    printf("\nPartition (Count: %d):\n", gpu_ctx->partition_keys_count);
    for (int i = 0; i < gpu_ctx->partition_keys_count; ++i) {
        auto partition_key = gpu_ctx->partition_keys[i];
        printf("  Channel ID: %u, Partition Key: %d\n", partition_key.channel_id, partition_key.partition_key);
    }
}

int main() {
    EndpointGPUCtxLoader loader;

    for (endpoint_id_t i = 0; i < 5; ++i) {
        forwarding_table_entry_t entry{};
        entry.dest_ep_id = i;
        entry.next_hops_count = 2;
        entry.next_hops[0] = (i + 1) % MAX_ENDPOINTS;
        entry.next_hops[1] = (i + 2) % MAX_ENDPOINTS;
        loader.add_forwarding_table_entry(entry);
    }

    for (endpoint_id_t i = 0; i < 5; ++i) {
        address_t send_buf = reinterpret_cast<address_t>(malloc(1024));
        address_t recv_buf = reinterpret_cast<address_t>(malloc(1024));
        loader.add_address_translation(i, send_buf, recv_buf);
    }

    for (channel_id_t i = 0; i < 3; ++i) {
        column_type_t types[3] = {
            COLUMN_TYPE_INT, COLUMN_TYPE_FLOAT, COLUMN_TYPE_DOUBLE};
        loader.add_schema(i, types, 3);
    }

    for (channel_id_t i = 0; i < 3; ++i) {
        loader.add_partition_key(i, i * 10);
    }

    for (channel_id_t i = 0; i < 3; ++i) {
        endpoint_id_t dests[] = {1, 2, 3};
        loader.add_destinations(i, dests, 3);
    }

    std::cout << "GPU context successfully initialized and transferred." << std::endl;

    inspect_ctx(&loader.ctx);

    // Endpoint context is loaded

    // PARTITION MAP

    // using Key_ft = channel_id_t;
    // using Value_ft = partition_key_t;

    // std::size_t num_keys = loader.ctx.partition_keys_count;
    // std::size_t capacity = static_cast<std::size_t>(num_keys * 1.3);

    // // Build host vectors from forwarding table
    // thrust::host_vector<Key_ft> h_keys(num_keys);
    // thrust::host_vector<Value_ft> h_values(num_keys);

    // for (std::size_t i = 0; i < num_keys; ++i) {
    //     h_keys[i] = loader.ctx.partition_keys[i].channel_id;
    //     h_values[i] = loader.ctx.partition_keys[i].partition_key;
    // }

    // // Move to device
    // thrust::device_vector<Key_ft> keys = h_keys;
    // thrust::device_vector<Value_ft> values = h_values;

    // // Build map
    // auto partition_map = cuco::static_map{
    //     capacity,
    //     cuco::empty_key{-1},
    //     cuco::empty_value{-1},
    //     cuda::std::equal_to<Key_ft>{},
    //     cuco::linear_probing<1, cuco::default_hash_function<Key_ft>>{}
    // };

    // // Insert key-value pairs
    // auto zipped = thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), values.begin()));
    // partition_map.insert(zipped, zipped + num_keys);

    // // Map fully complete

    // thrust::device_vector<Key_ft> queries = {0, 1, 2};
    // thrust::device_vector<Value_ft> results(queries.size(), -1);

    // // Launch a kernel to perform the lookups
    // int* d_queries;
    // int* d_results;
    // cudaMalloc(&d_queries, queries.size() * sizeof(int));
    // cudaMalloc(&d_results, queries.size() * sizeof(int));
    // cudaMemcpy(d_queries, thrust::raw_pointer_cast(queries.data()), queries.size() * sizeof(int), cudaMemcpyDeviceToDevice);

    // auto find_ref = partition_map.ref(cuco::find);

    // find_in_map_kernel<<<(queries.size() + 255) / 256, 256>>>(find_ref, d_queries, d_results, queries.size());

    // // Copy results back to the host
    // cudaMemcpy(thrust::raw_pointer_cast(results.data()), d_results, queries.size() * sizeof(int), cudaMemcpyDeviceToHost);

    // // Output the results
    // std::cout << "\nQuery results:\n";
    // for (int i = 0; i < queries.size(); ++i) {
    //     std::cout << "Key " << queries[i] << " -> Value ";
    //     if (results[i] == -1) {
    //         std::cout << "NOT FOUND";
    //     } else {
    //         std::cout << results[i];
    //     }
    //     std::cout << "\n";
    // }

    // // Clean up
    // cudaFree(d_queries);
    // cudaFree(d_results);

    // SCHEMA MULTIMAP
    using Key_schema = channel_id_t;     // typically an int
    using Value_schema = column_type_t;  // typically an enum or small int

    Key_schema empty_key_sentinel     = 100;
    Value_schema empty_value_sentinel = COLUMN_TYPE_NOT_SUPPORTED;

    std::size_t total_pairs = 0;

    // Count how many (channel_id, column_type) pairs we’ll insert
    for (int i = 0; i < loader.ctx.schemas_count; ++i) {
        total_pairs += loader.ctx.schemas[i].columns_count;
    }

    std::size_t capacity_schema = static_cast<std::size_t>(total_pairs * 2);

    auto schema_map = cuco::experimental::static_multimap<Key_schema, Value_schema>{
        capacity_schema, 
        cuco::empty_key{empty_key_sentinel}, 
        cuco::empty_value{empty_value_sentinel}
    };

    thrust::device_vector<cuco::pair<Key_schema, Value_schema>> pairs(total_pairs);

    std::size_t index = 0;
    for (int i = 0; i < loader.ctx.schemas_count; ++i) {
        const auto& schema = loader.ctx.schemas[i];
        for (int j = 0; j < schema.columns_count; ++j) {
            pairs[index] = cuco::pair<Key_schema, Value_schema>{schema.channel_id, schema.column_types[j]};
            ++index;
        }
    }

    schema_map.insert(pairs.begin(), pairs.end());

    
    thrust::device_vector<channel_id_t> keys_to_find(1, 0);  // Only searching for key 0

    // Prepare the result space
    thrust::device_vector<cuco::pair<channel_id_t, column_type_t>> d_results(10);  // Adjust size accordingly

    // Launch the kernel (use appropriate block and grid sizes)
    int block_size = 256; // Example block size
    int grid_size = (keys_to_find.size() + block_size - 1) / block_size;

    // auto find_ref = schema_map.ref(cuco::find);

    // search_in_kernel<<<grid_size, block_size>>>(find_ref, keys_to_find, d_results);

    // Synchronize the kernel and check for errors
    //cudaDeviceSynchronize();

    // Print the results (for key = 0)
    //for (int i = 0; i < d_results.size(); ++i) {
    //    if (d_results[i].first != empty_key_sentinel) {  // Check if this is a valid result
    //        std::cout << "Key: " << d_results[i].first << ", Value: " << d_results[i].second << std::endl;
    //    }
    //}

    return 0;
}

