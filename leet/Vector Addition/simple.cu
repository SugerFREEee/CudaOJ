#include <cuda_runtime.h>

#define THREAD_NUM 256
#define MAX_BLOCK_NUM 1024


// 简单版
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = gtid; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = THREAD_NUM;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGrid = min(blocksPerGrid, MAX_BLOCK_NUM);
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
