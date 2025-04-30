#include <cuda_runtime.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    cudaSetDevice(1);

    const int n = 256;

    int * recBuf;
    cudaMalloc(&recBuf, n * sizeof(int));

    cudaIpcMemHandle_t handle;
    cudaIpcGetMemHandle(&handle, recBuf);

    int fd = open("ipc_handle.bin", O_CREAT | O_WRONLY, 0666);
    write(fd, &handle, sizeof(handle));
    close(fd);

    printf("GPU 1 ready\n");

    int * hostBuf = (int *)malloc(n * sizeof(int));

    while (1) {
        sleep(3);

        cudaMemcpy(hostBuf, recBuf, n * sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("Received\n");
        for (int i = 0; i < n; i++) {
            printf("%d ", hostBuf[i]);
        }
        printf("\n");
        
        free(hostBuf);
        cudaFree(recBuf);
        exit(1);
    }

    return 0;
}