#include "sender.cuh"

#include <new>

// __global__ void init_ringbuffer_kernel(RingBuffer* rb, void* buffer, size_t num_chunks) 
// {
//     // new (rb) RingBuffer(buffer, num_chunks);
//     rb->init(buffer, num_chunks);
// }

// __global__ void push_kernel(RingBuffer* rb, void* data, size_t size, bool* success) {
//     *success = rb->enqueue(data); //, CHUNK_SIZE);
// }

// __global__ void get_head_kernel(RingBuffer* rb, void** head) {
//     *head = rb->head;
// }

void Sender::recv_addr(int sockfd)
{
    // 1. receive remote key
    size_t remote_rkey_size;
    socket_recv(sockfd, &remote_rkey_size, sizeof(remote_rkey_size));

    void *remote_rkey_buffer = malloc(remote_rkey_size);
    socket_recv(sockfd, remote_rkey_buffer, remote_rkey_size);

    printf("rkey recv: %p %zu\n\n", remote_rkey_buffer, remote_rkey_size);

    UCS_CHECK(ucp_ep_rkey_unpack(ep, remote_rkey_buffer, &remote_rkey));

    free(remote_rkey_buffer);

    // 2. receive ring buffer information
    socket_recv(sockfd, &remote, sizeof(void *));

    printf("remote info:\n");
    printf("    rand_ptr: %p\n", remote);
}

Sender::Sender(ucp_context_h ctx, ucp_worker_h wrk, ucp_ep_h endpoint, int sockfd)
    : context(ctx), worker(wrk), ep(endpoint)
{
    recv_addr(sockfd);
}

void Sender::process_req(void* request) 
{
    if (UCS_PTR_IS_ERR(request)) {
        printf("UCX operation failed: %s\n", 
              ucs_status_string(UCS_PTR_STATUS(request)));
        exit(1);
    }
    
    ucs_status_t status;
    do {
        ucp_worker_progress(worker);
        status = ucp_request_check_status(request);
    } while (status == UCS_INPROGRESS);
    
    ucp_request_free(request);
}

void Sender::remote_push(int gpu_id) 
{   
    (void) gpu_id;
    
    void* new_value = reinterpret_cast<void*>(0x20);

    // 2. Critical: Specify memory type for GPU operation
    ucp_request_param_t put_params = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_MEMORY_TYPE,
        .memory_type = UCS_MEMORY_TYPE_CUDA
    };

    // 3. Execute the remote put
    void* request = ucp_put_nbx(
        ep,                    // Endpoint to receiver
        &new_value,            // Source address (host memory)
        sizeof(void*),         // Size of pointer
        (uintptr_t)remote,  // Remote GPU address
        remote_rkey,           // Remote key
        &put_params
    );

    // 4. Block until completion
    if (UCS_PTR_IS_PTR(request)) {
        ucs_status_t status;
        do {
            ucp_worker_progress(worker);
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        ucp_request_free(request);
        
        if (status != UCS_OK) {
            fprintf(stderr, "Remote push failed: %s\n", 
                    ucs_status_string(status));
            return;
        }
    } else if (UCS_PTR_STATUS(request) != UCS_OK) {
        fprintf(stderr, "Failed to start remote push: %s\n",
                ucs_status_string(UCS_PTR_STATUS(request)));
        return;
    }

    printf("Successfully updated remote rand at GPU address %p to 0x20\n", 
           remote);
}

void Sender::verify_remote_update() {
    // 1. Allocate and register pinned host memory
    void* verification_value;
    cudaMallocHost(&verification_value, sizeof(void*));
    
    // Register the host memory with UCX
    ucp_mem_h memh;
    ucp_mem_map_params_t params = {
        .field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                     UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                     UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE,
        .address = verification_value,
        .length = sizeof(void*),
        .memory_type = UCS_MEMORY_TYPE_HOST
    };
    UCS_CHECK(ucp_mem_map(context, &params, &memh));

    // 2. Perform remote get with explicit memory handle
    ucp_request_param_t get_params = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_MEMORY_TYPE |
                       UCP_OP_ATTR_FIELD_MEMH,
        .memory_type = UCS_MEMORY_TYPE_HOST,
        .memh = memh
    };
    
    void* request = ucp_get_nbx(
        ep,
        verification_value,          // Local destination
        sizeof(void*),               // Size
        (uintptr_t)remote,  // Remote GPU address
        remote_rkey,
        &get_params
    );

    // 3. Force completion
    if (UCS_PTR_IS_PTR(request)) {
        ucs_status_t status;
        do {
            ucp_worker_progress(worker);
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        ucp_request_free(request);
    }

    // 4. Print results
    printf("[VERIFY] Remote rand @ %p contains: %p\n", 
          remote, *reinterpret_cast<void**>(verification_value));

    // 5. Cleanup
    ucp_mem_unmap(context, memh);
    cudaFreeHost(verification_value);
}