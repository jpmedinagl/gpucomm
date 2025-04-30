#include <cuda_runtime.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

__global__ void fill(int *buf, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        buf[i] = i;
}

__global__ void send(int *sendBuf, int *recBuf, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        recBuf[i] = sendBuf[i];
}

int main() {
    cudaSetDevice(0);

    const int n = 256;

    int * sendBuf;
    cudaMalloc(&sendBuf, n * sizeof(int));
    fill<<256, 1>>(sendBuf, n);
    cudaDeviceSynchronize();

    cudaIpcMemHandle_t handle;
    int fd = open("ipc_handle.bin", O_RDONLY);
    read(fd, &handle, sizeof(handle));
    close(fd);

    int *recBuf;
    cudaIpcOpenMemHandle((void**)&recBuf, handle, cudaIpcMemLazyEnablePeerAccess);


    send<<256, 1>>(sendBuf, recBuf, n);
    cudaDeviceSynchronize();

    printf("GPU 0 sent to receiver\n");

    cudaFree(sendBuf);
    cudaIpcCloseMemHandle(recBuf);
    return 0;
}