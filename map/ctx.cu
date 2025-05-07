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

__global__ void inspect_gpu_ctx(EndpointGPUCtx* gpu_ctx) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {
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

    cudaIpcMemHandle_t handle = loader.load_to_gpu();

    std::cout << "GPU context successfully initialized and transferred." << std::endl;

    inspect_gpu_ctx<<<1, 1>>>(loader.data_on_gpu);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());  

    // Endpoint context is loaded

    using Key_ft = endpoint_id_t;
    using Value_ft = forwarding_table_entry_t;

    constexpr std::size_t num_keys = loader.ctx.forwarding_table_count;
    constexpr std::size_t capacity = static_cast<std::size_t>(num_keys * 1.3);

    // Build host vectors from forwarding table
    thrust::host_vector<Key_ft> h_keys(num_keys);
    thrust::host_vector<Value_ft> h_values(num_keys);

    for (std::size_t i = 0; i < num_keys; ++i) {
        h_keys[i] = loader.ctx.forwarding_table[i].dest_ep_id;
        h_values[i] = loader.ctx.forwarding_table[i];
    }

    // Move to device
    thrust::device_vector<Key_ft> keys = h_keys;
    thrust::device_vector<Value_ft> values = h_values;

    // Build map
    cuco::static_map<Key_t, Value_ft> forwarding_map{
        capacity,
        cuco::empty_key{std::numeric_limits<Key_t>::max()},
        cuco::empty_value{forwarding_table_entry_t{}}, // Default invalid entry
        cuda::std::equal_to<Key_t>{},
        cuco::linear_probing<1, cuco::default_hash_function<Key_t>>{}
    };

    // Insert key-value pairs
    auto zipped = thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), values.begin()));
    forwarding_map.insert(zipped, zipped + num_keys);

    // Map fully complete

    thrust::device_vector<Key> queries = {0, 1, num_keys-1, num_keys, -1};
    thrust::device_vector<Value> results(queries.size(), -1);

    // Launch a kernel to perform the lookups
    int* d_queries;
    int* d_results;
    cudaMalloc(&d_queries, queries.size() * sizeof(int));
    cudaMalloc(&d_results, queries.size() * sizeof(int));
    cudaMemcpy(d_queries, thrust::raw_pointer_cast(queries.data()), queries.size() * sizeof(int), cudaMemcpyDeviceToDevice);

    auto find_ref = forwarding_map.ref(cuco::find);

    find_in_map_kernel<<<(queries.size() + 255) / 256, 256>>>(find_ref, d_queries, d_results, queries.size());

    // Copy results back to the host
    cudaMemcpy(thrust::raw_pointer_cast(results.data()), d_results, queries.size() * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the results
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

    // Clean up
    cudaFree(d_queries);
    cudaFree(d_results);


    return 0;
}
