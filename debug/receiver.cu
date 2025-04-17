#include "receiver.cuh"

#include <iostream>
#include <new>

__global__ void export_rb_metadata(RingBuffer* rb, RingBufferRemoteInfo* metadata) 
{
    metadata->buffer_addr = reinterpret_cast<uintptr_t>(rb->buffer);
    metadata->tail_addr_ptr = reinterpret_cast<uintptr_t>(&(rb->tail));
    metadata->head_addr = reinterpret_cast<uintptr_t>(rb->head);
    metadata->tail_addr = reinterpret_cast<uintptr_t>(rb->tail);
    metadata->size = rb->size;
}

__global__ void init_ringbuffer_kernel(RingBuffer* rb, void* buffer, size_t num_chunks) 
{
    // new (rb) RingBuffer(buffer, num_chunks);
    rb->init(buffer, num_chunks);
}

__global__ void dequeue_kernel(RingBuffer* rb, void* out_chunk, bool* success) 
{
    *success = rb->dequeue(out_chunk);
}

__global__ void get_head_kernel(RingBuffer* rb, void** head) {
    *head = rb->head;
}

__global__ void get_tail_kernel(RingBuffer* rb, void** tail) {
    *tail = rb->tail;
}

__global__ void get_tail_ptr_kernel(RingBuffer* rb, void*** tail_ptr) {
    *tail_ptr = &(rb->tail);
}

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

    socket_send(sockfd, &rand_ptr, sizeof(void *));
    socket_send(sockfd, &rand, sizeof(void *));

    printf("local ring buffer info:\n");
    printf("    rand_ptr %p\n", rand_ptr);
    printf("    rand: %p\n", rand);
}

Receiver::Receiver(ucp_context_h ctx, ucp_worker_h wrk, ucp_ep_h endpoint,
                   int sockfd)
    : context(ctx), worker(wrk), ep(endpoint)
{    
    // 1. allocate buffer
    // void* gpu_memory;
    // const size_t total_size = sizeof(RingBuffer) + (NUM_CHUNKS + CHUNK_SIZE);
    // cudaMalloc(&gpu_memory, total_size);

    // 2. map memory
    ucp_mem_map_params_t params = {
        .field_mask = 
                    //   UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                      UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                      UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                      UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE,
        // .address = gpu_memory,
        .length = 2 * sizeof(void *),
        .flags = UCP_MEM_MAP_ALLOCATE,
        .memory_type = UCS_MEMORY_TYPE_CUDA
    };
    UCS_CHECK(ucp_mem_map(context, &params, &memh));

    ucp_mem_attr_t attr = {
    .field_mask = UCP_MEM_ATTR_FIELD_ADDRESS |
                 UCP_MEM_ATTR_FIELD_LENGTH |
                 UCP_MEM_ATTR_FIELD_MEM_TYPE
    };
    UCS_CHECK(ucp_mem_query(memh, &attr));

    printf("Registered memory: %p, size: %zu, type: %d\n",
        attr.address, attr.length, attr.mem_type);

    rand_ptr = attr.address;

    // Assign rand to the next memory location after rand_ptr
    rand = static_cast<void*>(static_cast<char*>(attr.address) + sizeof(void *));

    // Ensure rand_ptr points to rand (within the mapped region)
    memcpy(rand_ptr, &rand, sizeof(void*));

    send_addr(sockfd);
}

void Receiver::print_rb()
{
    // Print the values of rand_ptr and rand
    printf("rand_ptr: %p, rand: %p\n", rand_ptr, rand);

    // Access the memory location pointed to by rand_ptr (i.e., the address of rand)
    printf("Value pointed to by rand_ptr: %p\n", *(void**)rand_ptr);

    // Print the data in rand (you can modify this as per the actual type you expect in rand)
    int value_in_rand;
    memcpy(&value_in_rand, rand, sizeof(value_in_rand));
    printf("Data in rand: %d\n", value_in_rand);
}
