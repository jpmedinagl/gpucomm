#ifndef SENDER_H
#define SENDER_H

#include <ucp/api/ucp.h>
#include <cstddef>
#include <cstdint>

#include "utils.h"
#include "ring.cuh"

class Sender {
private:
    ucp_context_h context;
    ucp_worker_h worker;

    // ring buffer is registered for rdma
    ucp_mem_h memh;
    uintptr_t tmp_debug;

    // needs to keep track of the next things for every single gpu...?
    ucp_ep_h ep;

    // ucp_address_t* remote_worker;
    ucp_rkey_h remote_rkey;

    uintptr_t remote_rand_ptr;
    uintptr_t remote_rand;

    void process_req(void * request);

    void recv_addr(int sockfd);

public:
    Sender(ucp_context_h ctx, ucp_worker_h wrk, ucp_ep_h endpoint, int sockfd);

    void remote_push();
};

#endif // SENDER_H