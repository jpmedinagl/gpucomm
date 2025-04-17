#include "receiver.cuh"

void Receiver::send_addr(int sockfd)
{
    // 1. send key
    size_t rkey_size;
    void *rkey_buffer;
    UCS_CHECK(ucp_rkey_pack(context, memh, &rkey_buffer, &rkey_size));
    
    printf("rkey send: %p %zu\n\n", rkey_buffer, rkey_size);

    socket_send(sockfd, &rkey_size, sizeof(rkey_size));
    socket_send(sockfd, rkey_buffer, rkey_size);
    ucp_rkey_buffer_release(rkey_buffer);

    // 2. send ring buffer information
    socket_send(sockfd, &buffer, sizeof(buffer));

    printf("local info:\n");
    printf("    buf_ptr %p\n", (void*)&buffer);
    printf("    buf: %p\n", (void*)buffer);
}

Receiver::Receiver(ucp_context_h ctx, ucp_worker_h wrk, ucp_ep_h endpoint,
                   int sockfd)
    : context(ctx), worker(wrk), ep(endpoint)
{    
    uint64_t* remote_buffer;
    cudaMalloc(&remote_buffer, sizeof(uint64_t));

    uint64_t init_value = 0xa;
    cudaMemcpy(remote_buffer, &init_value, sizeof(uint64_t), cudaMemcpyHostToDevice);

    ucp_mem_map_params_t mem_map_params = {
        .field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                      UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                      UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE,
        .address = remote_buffer,
        .length = sizeof(uint64_t),
        .memory_type = UCS_MEMORY_TYPE_CUDA
    };
    
    ucp_mem_map(ucp_context, &mem_map_params, &memh);

    buffer = remote_buffer;

    send_addr(sockfd);
}

void Receiver::print_rb()
{
    // Print the values of rand_ptr and rand
    printf("rand: %p\n", rand);
}
