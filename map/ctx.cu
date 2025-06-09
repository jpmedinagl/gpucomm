#include <iostream>
#include <random>

#include "types.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr uint32_t BUFFER_SIZE = 5;

template <typename Map>
__global__ void find_in_map_kernel(Map map, int* queries, int* results, int num_queries) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_queries) {
        auto found = map.find(queries[idx]);

        if (found != map.end()) {
			printf("key = %d, value = %d\n", queries[idx], found->second);
            results[idx] = found->second;	
        } else {
            results[idx] = -1;
        }
    }
}

using Key_schema = channel_id_t;
using Value_schema = column_type_t;
using Map_t        = cuco::static_multimap<Key_schema, Value_schema>;
using View_t       = typename Map_t::device_view;

template <uint32_t BUFFER_SIZE>
__global__ void retrieve_from_multimap_kernel(
    View_t map,
    Key_schema const*          queries,
    Key_schema*                out_keys,     
    Value_schema*              out_values,   
    int*                       out_counts,
    int                        num_queries)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_queries) return;

    // 1) Query key
    Key_schema key = queries[tid];
    printf("thread %d process key: %d\n", tid, key);

    // 2) Cooperative groups for flush vs. probe
    auto cta = cg::this_thread_block();
    constexpr uint32_t tile_size = Map_t::probe_sequence_type::cg_size;
    auto flush_tile = cg::tiled_partition<tile_size>(cta);  // for flushing
    auto probe_tile = cg::tiled_partition<tile_size>(cta);  // for probing

    // 3) Shared-memory staging
    __shared__ uint32_t flush_counter;
    __shared__ cuda::std::atomic<uint32_t> match_count;
    __shared__ View_t::value_type          staging_buf[BUFFER_SIZE];

    if (threadIdx.x == 0) {
        flush_counter = 0;
        match_count.store(0, cuda::std::memory_order_relaxed);
    }
    __syncthreads();

    // 4) Call the templated retrieve<B>() API
    map.template retrieve<BUFFER_SIZE>(
        flush_tile,         // flushing CG: thread_block_tile<tile_size>
        probe_tile,         // probing CG: thread_block_tile<tile_size>
        key,                // lookup key
        &flush_counter,     // shared atomic flush counter
        staging_buf,        // shared staging buffer (pair<key,value>[BUFFER_SIZE])
        &match_count,       // shared atomic match counter
        staging_buf         // output iterator → writes into staging_buf
    );

    __syncthreads();

    // 5) One thread writes results globally
    if (threadIdx.x == 0) {
        uint32_t n = match_count.load(cuda::std::memory_order_relaxed);
        out_counts[tid] = static_cast<int>(n);

        printf("[tid %d] found %u matches for key %d\n", tid, n, key);

        for (int i = 0; i < n; ++i) {
            int   global_idx = tid * BUFFER_SIZE + i;
            auto  val        = staging_buf[i].second;

            out_keys  [global_idx] = key;
            out_values[global_idx] = val;

            printf("    [%d] value = %d\n", i, val);
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
	printf("\n");
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

    // PARTITION MAP EXAMPLE: cuco::static_map
    /*
    using Key_p = channel_id_t;
    using Value_p = partition_key_t;

    std::size_t num_keys = loader.ctx.partition_keys_count;
    std::size_t capacity = static_cast<std::size_t>(num_keys * 1.3);

    // Build host vectors from forwarding table
    thrust::host_vector<Key_p> h_keys(num_keys);
    thrust::host_vector<Value_p> h_values(num_keys);

    for (std::size_t i = 0; i < num_keys; ++i) {
		h_keys[i] = loader.ctx.partition_keys[i].channel_id;
		h_values[i] = loader.ctx.partition_keys[i].partition_key;
    }

    // Move to device
    thrust::device_vector<Key_p> keys = h_keys;
    thrust::device_vector<Value_p> values = h_values;

    // Build map
    auto partition_map = cuco::static_map{
		capacity,
		cuco::empty_key{-1},
		cuco::empty_value{-1},
        cuda::std::equal_to<Key_p>{},
        cuco::linear_probing<1, cuco::default_hash_function<Key_p>>{}
    };

    // Insert key-value pairs
    auto zipped = thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), values.begin()));
    partition_map.insert(zipped, zipped + num_keys);

    // Map fully complete

    thrust::device_vector<Key_p> queries = {0, 1, 2};
    thrust::device_vector<Value_p> results(queries.size(), -1);

    // Launch a kernel to perform the lookups
    int* d_queries;
    int* d_results;
    cudaMalloc(&d_queries, queries.size() * sizeof(int));
    cudaMalloc(&d_results, queries.size() * sizeof(int));
    cudaMemcpy(d_queries, thrust::raw_pointer_cast(queries.data()), queries.size() * sizeof(int), cudaMemcpyDeviceToDevice);

    auto find_ref = partition_map.ref(cuco::find);

    find_in_map_kernel<<<(queries.size() + 255) / 256, 256>>>(find_ref, d_queries, d_results, queries.size());

    Copy results back to the host
    cudaMemcpy(thrust::raw_pointer_cast(results.data()), d_results, queries.size() * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the results
    std::cout << "\nQuery results:\n";
    for (int i = 0; i < queries.size(); ++i) {
        std::cout << "Key " << queries[i] << " -> Value ";
        if (results[i] == -1) {
            std::cout << "NOT FOUND";
        } else {
            std::cout << results[i];
        }
        std::cout << "\n";
    }

    Clean up
    cudaFree(d_queries);
    cudaFree(d_results);
    */




    // Multimap test insertion
    // cuco::static_multimap<int,int> m{10, cuco::empty_key{-1}, cuco::empty_value{-1}};

    // // host‐vector of {key,value} pairs:
    // std::vector<cuco::pair<int,int>> h{{0,100},{0,101},{1,200}};
    // thrust::device_vector<decltype(h)::value_type> dv = h;

    // // device‐bulk insert:
    // m.insert(dv.begin(), dv.end());
    // cudaDeviceSynchronize();
    
    // thrust::device_vector<Key_schema> d_keys_to_check = {0};
    // size_t total_count = m.count(d_keys_to_check.begin(), d_keys_to_check.end());
    // std::cout << "Total occurrences of key 0: " << total_count << std::endl;

    // d_keys_to_check = {1};
    // total_count = m.count(d_keys_to_check.begin(), d_keys_to_check.end());
    // std::cout << "Total occurrences of key 1: " << total_count << std::endl;

    // d_keys_to_check = {2};
    // total_count = m.count(d_keys_to_check.begin(), d_keys_to_check.end());
    // std::cout << "Total occurrences of key 2: " << total_count << std::endl;

    // exit(1);

    



    // SCHEMA MULTIMAP EXAMPLE: cuco::static_multimap
    Key_schema empty_key_sentinel = 100;
    Value_schema empty_value_sentinel = COLUMN_TYPE_NOT_SUPPORTED;

    std::size_t total_pairs = 0;

    // Count how many (channel_id, column_type) pairs we’ll insert
    for (int i = 0; i < loader.ctx.schemas_count; ++i) {
        total_pairs += loader.ctx.schemas[i].columns_count;
    }

    std::size_t capacity_schema = static_cast<std::size_t>(total_pairs * 2);

    auto schema_map = cuco::static_multimap<Key_schema, Value_schema>{
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
    printf("\n");

    schema_map.insert(pairs.begin(), pairs.end());


    // Test the insertion:
    thrust::device_vector<Key_schema> d_keys_to_check = {0};
    size_t total_count = schema_map.count(d_keys_to_check.begin(), d_keys_to_check.end());
    std::cout << "Total occurrences of key 0: " << total_count << std::endl;

    d_keys_to_check = {1};
    total_count = schema_map.count(d_keys_to_check.begin(), d_keys_to_check.end());
    std::cout << "Total occurrences of key 1: " << total_count << std::endl;

    d_keys_to_check = {2};
    total_count = schema_map.count(d_keys_to_check.begin(), d_keys_to_check.end());
    std::cout << "Total occurrences of key 2: " << total_count << std::endl;
    
    printf("\n");


    // Test kernel retrieval
    thrust::device_vector<Key_schema> queries = {2, 1};

    int num_queries = queries.size();
    constexpr uint32_t max_results_per_key = 10;

    thrust::device_vector<Key_schema> result_keys(queries.size() * max_results_per_key, 100);
    thrust::device_vector<Value_schema> result_values(queries.size() * max_results_per_key, COLUMN_TYPE_NOT_SUPPORTED);
    thrust::device_vector<int> value_counts(queries.size(), 0);

    Key_schema* d_queries = thrust::raw_pointer_cast(queries.data());
    Key_schema* d_result_keys = thrust::raw_pointer_cast(result_keys.data());
    Value_schema* d_result_values = thrust::raw_pointer_cast(result_values.data());
    int* d_value_counts = thrust::raw_pointer_cast(value_counts.data());

    // auto retrieve_ref = schema_map.ref(cuco::retrieve);

    // multimap does not have ref, it uses device view
    View_t schema_map_device_view = schema_map.get_device_view();

    constexpr int THREADS_PER_BLOCK = 256;
    int blocks = (num_queries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    retrieve_from_multimap_kernel<max_results_per_key>
    <<<blocks, THREADS_PER_BLOCK>>>(
        schema_map_device_view,
        d_queries,
        d_result_keys,
        d_result_values,
        d_value_counts,
        num_queries
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    return 0;
}

