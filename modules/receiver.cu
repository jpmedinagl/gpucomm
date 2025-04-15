#include "receiver.cuh"

#include <iostream>
#include <new>

RingBufferRemoteInfo export_metadata(const RingBuffer& rb) {
    return {
        reinterpret_cast<uintptr_t>(rb.buffer),
        // &head,
        reinterpret_cast<uintptr_t>(&rb.tail),
        reinterpret_cast<uintptr_t>(rb.head),
        reinterpret_cast<uintptr_t>(rb.tail),
        rb.size
    };
};

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
    
    printf("Rkey send: %p %zu\n", rkey_buffer, rkey_size);

    socket_send(sockfd, &rkey_size, sizeof(rkey_size));
    socket_send(sockfd, rkey_buffer, rkey_size);
    ucp_rkey_buffer_release(rkey_buffer);

    // 2. send ring buffer information
    RingBuffer host_rb;
    cudaMemcpy(&host_rb, d_ringbuf, sizeof(RingBuffer), cudaMemcpyDeviceToHost);

    RingBufferRemoteInfo meta = export_metadata(host_rb);
    socket_send(sockfd, &meta, sizeof(meta));

    printf("Local ring buffer info:\n");
    printf("buf: %p\n", meta.buffer_addr);
    printf("tail_ptr: %p\n", meta.tail_addr_ptr);
    printf("head: %p\n", meta.head_addr);
    printf("tail: %p\n", meta.tail_addr);
    printf("size: %p\n\n", meta.size);
}

Receiver::Receiver(ucp_context_h ctx, ucp_worker_h wrk, ucp_ep_h endpoint,
                   int sockfd)
    : context(ctx), worker(wrk), ep(endpoint)
{    
    // 1. allocate buffer
    void* gpu_memory;
    const size_t total_size = sizeof(RingBuffer) + (NUM_CHUNKS + CHUNK_SIZE);
    cudaMalloc(&gpu_memory, total_size);

    // 2. map memory
    ucp_mem_map_params_t params = {
        .field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                      UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                      UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE,
        .address = gpu_memory,
        .length = total_size,
        .memory_type = UCS_MEMORY_TYPE_CUDA
    };
    UCS_CHECK(ucp_mem_map(context, &params, &memh));

    // 3. intialize ring buffer
    d_ringbuf = reinterpret_cast<RingBuffer*>(gpu_memory);
    void* data_buffer = reinterpret_cast<char*>(gpu_memory) + sizeof(RingBuffer);

    init_ringbuffer_kernel<<<1, 1>>>(d_ringbuf, data_buffer, NUM_CHUNKS);
    cudaDeviceSynchronize();

    send_addr(sockfd);
}

void Receiver::dequeue(void* out_chunk) 
{
    void* local_head;
    void* local_tail;
    void** local_tail_ptr;

    // Launch the kernel to get the head and tail pointers
    get_head_kernel<<<1, 1>>>(d_ringbuf, &local_head);
    get_tail_kernel<<<1, 1>>>(d_ringbuf, &local_tail);
    get_tail_ptr_kernel<<<1, 1>>>(d_ringbuf, &local_tail_ptr);

    // Synchronize to ensure the kernel execution completes
    cudaDeviceSynchronize();

    // Now copy the head and tail pointers from device to host
    void* local_head_host;
    void* local_tail_host;
    void** local_tail_ptr_host;
    cudaMemcpy(&local_head_host, &local_head, sizeof(void*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&local_tail_host, &local_tail, sizeof(void*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&local_tail_ptr_host, &local_tail_ptr, sizeof(void**), cudaMemcpyDeviceToHost);

    // Print the head and tail pointers
    printf("Receiver head pointer (host-side): %p\n", local_head_host);
    printf("Receiver tail pointer (host-side): %p\n", local_tail_host);
    printf("Receiver tail pointer ptr (host-side): %p\n", local_tail_ptr_host);

    // dequeue from ring buffer
    bool success = false;
    dequeue_kernel<<<1, 1>>>(d_ringbuf, out_chunk, &success);

    if (!success) {
        std::cout << "Buffer empty" << std::endl;
    }

    get_head_kernel<<<1, 1>>>(d_ringbuf, &local_head);
    get_tail_kernel<<<1, 1>>>(d_ringbuf, &local_tail);

    // Synchronize to ensure the kernel execution completes
    cudaDeviceSynchronize();

    // Now copy the head and tail pointers from device to host
    cudaMemcpy(&local_head_host, &local_head, sizeof(void*), cudaMemcpyDeviceToHost);
    cudaMemcpy(&local_tail_host, &local_tail, sizeof(void*), cudaMemcpyDeviceToHost);

    // Print the head and tail pointers
    printf("Receiver head pointer (host-side): %p\n", local_head_host);
    printf("Receiver tail pointer (host-side): %p\n", local_tail_host);
}
