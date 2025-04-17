#include <ucp/api/ucp.h>
#include <cuda_runtime.h>
#include <ucs/memory/memory_type.h>
#include <ucp/api/ucp_def.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>     // For close()
#include <stdlib.h>     // For exit()
#include <string.h>     // For memset()
#include <cstdio>       // For perror()
#include <ucp/api/ucp.h>
#include <ucs/type/status.h>

#include <inttypes.h>

#include <stdio.h>

#define BUFFER_SIZE 512
#define PORT 12345

#define CHECK_ERROR(_cond, _msg)                                               \
    do {                                                                       \
        if (_cond) {                                                           \
            printf("Failed to %s\n", _msg);                                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(_func)                                                      \
    do {                                                                       \
        cudaError_t _result = (_func);                                         \
        if (cudaSuccess != _result) {                                          \
            printf("%s failed: %s\n", #_func, cudaGetErrorString(_result));    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define UCS_CHECK(_expr)                                                       \
    do {                                                                       \
        ucs_status_t _status = (_expr);                                        \
        if (UCS_OK != _status) {                                               \
            printf("%s failed: %s\n", #_expr, ucs_status_string(_status));     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

typedef struct {
    int gpu_id;
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_ep_h ep;
    ucp_mem_h memh;
    void* gpu_buffer;
    size_t buffer_size;
    ucp_address_t* remote_worker_addr;
    size_t remote_worker_addr_len;
    uintptr_t* remote_buffer_addr;
    ucp_rkey_h remote_rkey;
} gpu_worker_t;

// Basic socket send, loop to ensure full data transfer
void socket_send(int sockfd, const void* data, size_t size) {
    size_t sent = 0;
    while (sent < size) {
        ssize_t res = send(sockfd, (char*)data + sent, size - sent, 0);
        if (res <= 0) {
            perror("send failed");
            exit(1);
        }
        sent += res;
    }
}

// Basic socket recv, loop to ensure full data transfer
void socket_recv(int sockfd, void* buffer, size_t size) {
    size_t received = 0;
    while (received < size) {
        ssize_t res = recv(sockfd, (char*)buffer + received, size - received, 0);
        if (res <= 0) {
            perror("recv failed");
            exit(1);
        }
        received += res;
    }
}

void init_worker(gpu_worker_t* worker, int gpu_id)
{
    // Initialize worker and gpu buffer + register memory
    CUDA_CHECK(cudaSetDevice(gpu_id));
    worker->gpu_id = gpu_id;

    ucp_params_t params;
    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_PARAM_FIELD_FEATURES; 
    params.features = UCP_FEATURE_RMA;
    
    UCS_CHECK(ucp_init(&params, NULL, &worker->context));

    ucp_worker_params_t worker_params;
    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    UCS_CHECK(ucp_worker_create(worker->context, &worker_params, &worker->worker));
}

void register_memory(gpu_worker_t* worker)
{
    ucp_mem_map_params_t mem_params;
    memset(&mem_params, 0, sizeof(mem_params));
    mem_params.field_mask = // UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                          UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                          UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                          UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    // mem_params.address = buffer;
    mem_params.length = BUFFER_SIZE;
    mem_params.flags = UCP_MEM_MAP_ALLOCATE;
    mem_params.memory_type = UCS_MEMORY_TYPE_CUDA;

    UCS_CHECK(ucp_mem_map(worker->context, &mem_params, &worker->memh));

    ucp_mem_attr_t attr = {
    .field_mask = UCP_MEM_ATTR_FIELD_ADDRESS |
                 UCP_MEM_ATTR_FIELD_LENGTH |
                 UCP_MEM_ATTR_FIELD_MEM_TYPE
    };
    UCS_CHECK(ucp_mem_query(worker->memh, &attr));

    printf("Registered memory: %p, size: %zu, type: %d\n",
        attr.address, attr.length, attr.mem_type);

    worker->gpu_buffer = attr.address;
    worker->buffer_size = attr.length;

    printf("GPU %d buffer address: %p (%zu)\n", worker->gpu_id, worker->gpu_buffer, worker->buffer_size);

    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, worker->gpu_buffer));

    const char* mem_type_str =
    attributes.type == cudaMemoryTypeDevice  ? "Device" :
    attributes.type == cudaMemoryTypeHost    ? "Host (Pinned)" :
    attributes.type == cudaMemoryTypeManaged ? "Managed" :
                                               "Unknown";

    printf("Pointer type: %s (device: %d)\n", mem_type_str, attributes.device);
}

int init_gpu_worker(gpu_worker_t* worker, int gpu_id) 
{   
    init_worker(worker, gpu_id);
    register_memory(worker);
    return 0;
}

// Exchange addresses of ucp workers and buffers and create an endpoint
void exchange_addresses(gpu_worker_t* local, int sockfd) 
{
    // Exchange the worker addresses
    ucp_address_t* local_worker_addr;
    size_t local_worker_len;
    UCS_CHECK(ucp_worker_get_address(local->worker, &local_worker_addr, &local_worker_len));
    
    uint64_t addr_header = *((uint64_t*)local_worker_addr);
    printf("Worker address: %p (%zu)\n", addr_header, local_worker_len);
    
    // Send to remote worker
    socket_send(sockfd, &local_worker_len, sizeof(local_worker_len));
    socket_send(sockfd, local_worker_addr, local_worker_len);

    ucp_worker_release_address(local->worker, local_worker_addr);
    
    socket_recv(sockfd, &local->remote_worker_addr_len, sizeof(local->remote_worker_addr_len));
    local->remote_worker_addr = (ucp_address_t*)malloc(local->remote_worker_addr_len);
    socket_recv(sockfd, local->remote_worker_addr, local->remote_worker_addr_len);

    uint64_t remote_addr_header = *((uint64_t*)local->remote_worker_addr);
    printf("Worker address: %p (%zu)\n", remote_addr_header, local->remote_worker_addr_len);
    
    // Exchange the gpu buffers
    uintptr_t local_buf_addr = (uintptr_t)local->gpu_buffer;
    printf("Sending  local buffer address: %p\n", local_buf_addr);
    socket_send(sockfd, &local_buf_addr, sizeof(uintptr_t));
    socket_recv(sockfd, &local->remote_buffer_addr, sizeof(uintptr_t));
    printf("Received remote buffer address: %p\n", local->remote_buffer_addr);

    ucp_ep_params_t ep_params;
    memset(&ep_params, 0, sizeof(ep_params));
    
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = local->remote_worker_addr;

    UCS_CHECK(ucp_ep_create(local->worker, &ep_params, &local->ep));

    // Send rkey
    size_t rkey_size;
    void *rkey_buffer;
    UCS_CHECK(ucp_rkey_pack(local->context, local->memh, &rkey_buffer, &rkey_size));
    printf("Rkey send: %p %zu\n", rkey_buffer, rkey_size);
    socket_send(sockfd, &rkey_size, sizeof(rkey_size));
    socket_send(sockfd, rkey_buffer, rkey_size);
    ucp_rkey_buffer_release(rkey_buffer);

    // Receive rkey
    size_t remote_rkey_size;
    socket_recv(sockfd, &remote_rkey_size, sizeof(remote_rkey_size));
    void *remote_rkey_buffer = malloc(remote_rkey_size);
    socket_recv(sockfd, remote_rkey_buffer, remote_rkey_size);
    printf("Rkey recv: %p %zu\n", remote_rkey_buffer, remote_rkey_size);
    UCS_CHECK(ucp_ep_rkey_unpack(local->ep, remote_rkey_buffer, &local->remote_rkey));

    printf("Local rkey size: %zu, Remote rkey size: %zu\n", 
       rkey_size, remote_rkey_size);

    // free(rkey_buffer);
    free(remote_rkey_buffer);
    close(sockfd);
}

void gpu_worker_teardown(gpu_worker_t* worker) {
    if (worker->remote_rkey) ucp_rkey_destroy(worker->remote_rkey);
    if (worker->ep) ucp_ep_destroy(worker->ep);
    if (worker->memh) ucp_mem_unmap(worker->context, worker->memh);
    if (worker->worker) ucp_worker_destroy(worker->worker);
    if (worker->context) ucp_cleanup(worker->context);
    if (worker->remote_worker_addr) free(worker->remote_worker_addr);
    if (worker->gpu_buffer) cudaFree(worker->gpu_buffer);
}
