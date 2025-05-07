#pragma once

#include <cstdint>

const uint16_t MAX_NEIGHBOURS{16};

const uint16_t MAX_ENDPOINTS{1024};

const uint16_t MAX_CHANNELS{1024};

const uint16_t MAX_SCHEMA_COLUMNS{64};

const uint8_t TOP_K_PATHS{3};

const uint16_t UCP_RKEY_MAX_SIZE{256};

const uint32_t NON_GPU_EP_ID_STARTING_OFFSET{512};

#include "thrust/device_vector.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuco/static_map.cuh>
#include <cuco/static_multimap.cuh>
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <vector>


#define CHECK_CUDA_ERR(call)                                                   \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA check error: %s\n", cudaGetErrorString(err));      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

using agent_id_t = uint32_t;
using channel_id_t = uint32_t;
using endpoint_id_t = uint32_t;
using device_uuid_t = uint64_t;
using device_local_id_t = uint32_t;
using partition_key_t = int;
using byte_t = unsigned char;

typedef enum {
  CPU,
  GPU,
  NIC,
  NVSwitch,
  PCIeSwitch,
  NetworkSwitch,
  DEVICE_NOT_SUPPORTED
} device_type_t;

typedef enum {
  Infiniband,
  Ethernet,
  NVLink,
  PCIe,
  QPI,
  LINK_NOT_SUPPORTED
} link_type_t;

typedef enum {
  COLUMN_TYPE_BYTE = 0,
  COLUMN_TYPE_INT = 1,
  COLUMN_TYPE_LONG = 2,
  COLUMN_TYPE_FLOAT = 3,
  COLUMN_TYPE_DOUBLE = 4,

  COLUMN_TYPE_NOT_SUPPORTED = 99,
} column_type_t;

struct forwarding_table_entry_t {
  endpoint_id_t dest_ep_id{0};
  int next_hops_count{0};
  endpoint_id_t next_hops[MAX_NEIGHBOURS]{
      0}; // TODO: make this dynamic allocation
};

struct schema_entry_t {
  channel_id_t channel_id{1000000};
  column_type_t column_types[MAX_SCHEMA_COLUMNS]{COLUMN_TYPE_NOT_SUPPORTED};
  int columns_count{0};
};

struct partition_key_entry_t {
  channel_id_t channel_id{1000000};
  partition_key_t partition_key{0};
};

struct address_translation_entry_t {
  endpoint_id_t endpoint_id{0};
  byte_t *send_buffer{nullptr};
  byte_t *receive_buffer{nullptr};
};

struct destination_entry_t {
  channel_id_t channel_id{1000000};
  endpoint_id_t dests[MAX_ENDPOINTS]{0};
  int dests_count{0};
};

using address_t = uint64_t;

struct EndpointGPUCtx {
  forwarding_table_entry_t forwarding_table[MAX_ENDPOINTS]{0};
  int forwarding_table_count{0};

  // WARN: Used MAX_ENDPOINTS to use the array as a map, in real-world usage, we
  // should use a real map on GPU, like cudo::static_map
  address_translation_entry_t address_translation[MAX_ENDPOINTS]{0};
  int address_translation_count{0};

  schema_entry_t schemas[MAX_CHANNELS]{0};
  int schemas_count{0};

  partition_key_entry_t partition_keys[MAX_CHANNELS]{0};
  int partition_keys_count{0};

  destination_entry_t destinations[MAX_CHANNELS]{0};
  int destinations_count{0};

  size_t column_types_sizes[5]{0};
};

struct EndpointGPUCtxLoader {
  EndpointGPUCtx ctx{};

  EndpointGPUCtx *data_on_gpu{nullptr};

  std::mutex fwd_tbl_mutex{};
  std::mutex addr_trans_mutex{};
  std::mutex schema_mutex{};
  std::mutex part_key_mutex{};
  std::mutex dest_mutex{};

  EndpointGPUCtxLoader() {
    // Initialize the column types sizes map
    ctx.column_types_sizes[COLUMN_TYPE_BYTE] = sizeof(uint8_t);
    ctx.column_types_sizes[COLUMN_TYPE_INT] = sizeof(int32_t);
    ctx.column_types_sizes[COLUMN_TYPE_LONG] = sizeof(int64_t);
    ctx.column_types_sizes[COLUMN_TYPE_FLOAT] = sizeof(float);
    ctx.column_types_sizes[COLUMN_TYPE_DOUBLE] = sizeof(double);
  }

  cudaIpcMemHandle_t load_to_gpu() {
    CHECK_CUDA_ERR(cudaMalloc((void **)&data_on_gpu, sizeof(EndpointGPUCtx)));
    CHECK_CUDA_ERR(cudaMemcpy(data_on_gpu, &ctx, sizeof(EndpointGPUCtx),
                              cudaMemcpyHostToDevice));

    cudaIpcMemHandle_t handle{};
    CHECK_CUDA_ERR(cudaIpcGetMemHandle(&handle, data_on_gpu));

    return handle;
  }

  void add_forwarding_table_entry(forwarding_table_entry_t entry) {
    std::unique_lock lock(fwd_tbl_mutex);

    ctx.forwarding_table[entry.dest_ep_id] = entry;
    // std::memcpy(&ctx.forwarding_table[ctx.forwarding_table_count], &entry,
    //             sizeof(forwarding_table_entry_t));
    ctx.forwarding_table_count++;

    lock.unlock();
  }

  void add_address_translation(endpoint_id_t endpoint_id, address_t send_addr,
                               address_t receive_addr) {
    std::unique_lock lock(addr_trans_mutex);

    address_translation_entry_t entry{};
    entry.endpoint_id = endpoint_id;
    entry.send_buffer = (byte_t *)send_addr;
    entry.receive_buffer = (byte_t *)receive_addr;

    ctx.address_translation[entry.endpoint_id] = entry;
    // std::memcpy(&ctx.address_translation[ctx.address_translation_count],
    //             &entry, sizeof(address_translation_entry_t));
    ctx.address_translation_count++;

    lock.unlock();
  }

  void add_schema(channel_id_t channel_id, column_type_t *schema,
                  int columns_count) {
    std::unique_lock lock(schema_mutex);

    schema_entry_t entry{};
    entry.channel_id = channel_id;
    entry.columns_count = columns_count;
    for (int i = 0; i < columns_count; i++) {
      entry.column_types[i] = schema[i];
    }

    ctx.schemas[entry.channel_id] = entry;
    // std::memcpy(&ctx.schemas[ctx.schemas_count], &entry,
    //             sizeof(schema_entry_t));

    ctx.schemas_count++;

    lock.unlock();
  }

  void add_partition_key(channel_id_t channel_id, partition_key_t part_key) {
    std::unique_lock lock(part_key_mutex);

    partition_key_entry_t entry{};
    entry.channel_id = channel_id;
    entry.partition_key = part_key;

    ctx.partition_keys[entry.channel_id] = entry;
    // std::memcpy(&ctx.partition_keys[ctx.partition_keys_count], &entry,
    //             sizeof(partition_key_entry_t));
    ctx.partition_keys_count++;

    lock.unlock();
  }

  void add_destinations(channel_id_t channel_id, endpoint_id_t *dests,
                        int dests_count) {

    std::unique_lock lock(dest_mutex);

    destination_entry_t entry{};
    entry.channel_id = channel_id;
    entry.dests_count = dests_count;
    for (int i = 0; i < dests_count; i++) {
      entry.dests[i] = dests[i];
    }

    ctx.destinations[entry.channel_id] = entry;
    // std::memcpy(&ctx.destinations[ctx.destinations_count], &entry,
    //             sizeof(destination_entry_t));

    ctx.destinations_count++;

    lock.unlock();
  }
};