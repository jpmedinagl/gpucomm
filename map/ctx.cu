#include <iostream>
#include <random>

#include "types.cuh"

__global__ void inspect_gpu_ctx(EndpointGPUCtx* gpu_ctx) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) {  // Just use one thread to print for simplicity
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

    // Fill up with some dummy forwarding table entries
    for (endpoint_id_t i = 0; i < 5; ++i) {
        forwarding_table_entry_t entry{};
        entry.dest_ep_id = i;
        entry.next_hops_count = 2;
        entry.next_hops[0] = (i + 1) % MAX_ENDPOINTS;
        entry.next_hops[1] = (i + 2) % MAX_ENDPOINTS;
        loader.add_forwarding_table_entry(entry);
    }

    // Fill up with dummy address translations
    for (endpoint_id_t i = 0; i < 5; ++i) {
        address_t send_buf = reinterpret_cast<address_t>(malloc(1024));
        address_t recv_buf = reinterpret_cast<address_t>(malloc(1024));
        loader.add_address_translation(i, send_buf, recv_buf);
    }

    // Fill up with dummy schema entries
    for (channel_id_t i = 0; i < 3; ++i) {
        column_type_t types[3] = {
            COLUMN_TYPE_INT, COLUMN_TYPE_FLOAT, COLUMN_TYPE_DOUBLE};
        loader.add_schema(i, types, 3);
    }

    // Fill up with dummy partition keys
    for (channel_id_t i = 0; i < 3; ++i) {
        loader.add_partition_key(i, i * 10);
    }

    // Fill up with dummy destination entries
    for (channel_id_t i = 0; i < 3; ++i) {
        endpoint_id_t dests[] = {1, 2, 3};
        loader.add_destinations(i, dests, 3);
    }

    // Transfer to GPU
    cudaIpcMemHandle_t handle = loader.load_to_gpu();

    std::cout << "GPU context successfully initialized and transferred." << std::endl;

    // cudaSetDevice(0);
    inspect_gpu_ctx<<<1, 1>>>(loader.data_on_gpu);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());  

    return 0;
}
