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
    socket_send(sockfd, &rand, sizeof(void*))

    printf("local info:\n");
    printf("    rand_ptr %p\n", &rand);
    printf("    rand: %p\n", rand);
}

Receiver::Receiver(ucp_context_h ctx, ucp_worker_h wrk, ucp_ep_h endpoint,
                   int sockfd)
    : context(ctx), worker(wrk), ep(endpoint)
{    
    // 1. allocate buffer
    void* gpu_memory;
    // const size_t total_size = sizeof(RingBuffer) + (NUM_CHUNKS + CHUNK_SIZE);
    cudaMalloc(&gpu_memory, sizeof(void *));

    void* initial_value = reinterpret_cast<void*>(0x10);
    cudaMemcpy(gpu_memory, &initial_value, sizeof(void*), cudaMemcpyHostToDevice);

    // 2. map memory
    ucp_mem_map_params_t params = {
        .field_mask = 
                      UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                      UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                      UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE,
        .address = gpu_memory,
        .length = sizeof(void *),
        .memory_type = UCS_MEMORY_TYPE_CUDA
    };
    UCS_CHECK(ucp_mem_map(context, &params, &memh));

    rand = gpu_memory;

    send_addr(sockfd);
}

void Receiver::print_rb()
{
    // Print the values of rand_ptr and rand
    printf("rand: %p\n", rand);

    void* current_value;
    cudaMemcpy(&current_value, rand, sizeof(void*), cudaMemcpyDeviceToHost);
    printf("*rand: %p\n", current_value);
}
