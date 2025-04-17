#include "utils.h"
#include "sender.cuh"

int main() 
{
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_ep_h ep;

    CUDA_CHECK(cudaSetDevice(0));

    init(&context, &worker);
    
    // Socket setup
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr))) {
        perror("connect failed");
        exit(1);
    }

    printf("socked connected\n");

    // exchange addresses + keys !
    // this is done for testing only
    // can I assume that the endpoints are created + keys are exchanged ?

    // create ep

    create_ep(sockfd, worker, &ep);

    // define different chunks to send
    void* gpu_chunks[NUM_CHUNKS];
    for (int i = 0; i < NUM_CHUNKS; i++) {
        cudaMalloc(&gpu_chunks[i], CHUNK_SIZE);
        
        // Simple host buffer with pattern
        char host_data[CHUNK_SIZE];
        memset(host_data, 'A' + i, CHUNK_SIZE); // 'A', 'B', etc.
        
        cudaMemcpy(gpu_chunks[i], host_data, CHUNK_SIZE, cudaMemcpyHostToDevice);
    }

    // now everything is setup, can create sender module

    // Sender sender(context, worker, ep, memh, remote_worker, remote_rkey);
    Sender sender(context, worker, ep, sockfd);

    printf("Sender connected\n\n");   

    sender.remote_push(1);
    // for (int i = 0; i < NUM_CHUNKS; i++) {
    //     sender.remote_push(1);
    //     printf("Sent chunk %d\n", i);
    // }
    sender.verify_remote_update();

    return 0;
}