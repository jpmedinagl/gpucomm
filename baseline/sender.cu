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

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    fill<<<blocks, threadsPerBlock>>>(sendBuf, n);
    cudaDeviceSynchronize();

    cudaIpcMemHandle_t handle;
    int fd = open("ipc_handle.bin", O_RDONLY);
    read(fd, &handle, sizeof(handle));
    close(fd);

    int *recBuf;
    cudaIpcOpenMemHandle((void**)&recBuf, handle, cudaIpcMemLazyEnablePeerAccess);

    float elapsedTime = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    send<<<blocks, threadsPerBlock>>>(sendBuf, recBuf, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    printf("GPU 0 sent to receiver\n");

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel send time: %.3f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(sendBuf);
    cudaIpcCloseMemHandle(recBuf);
    return 0;
}