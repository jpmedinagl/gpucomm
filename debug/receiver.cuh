#ifndef RECEIVER_H
#define RECEIVER_H

#include "utils.h"
#include "ring.cuh"

class Receiver {
private:
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_ep_h ep;
    ucp_mem_h memh;

    uint64_t * buffer;

    void send_addr(int sockfd);

public:
    Receiver(ucp_context_h ctx, ucp_worker_h wrk, ucp_ep_h endpoint, int sockfd);

    void print_rb();
};

#endif // RECEIVER_H