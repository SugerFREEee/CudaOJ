#include <cuda_runtime.h>
#define MAX_BLOCK_NUM 1024
#define WARP_NUM 32
#define THREAD_NUM (WARP_NUM * 32)


__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i = gtid ;i<N*N;i+=stride)
    {
        int r = i / N;
        int c = i % N;

        if(r < N && c < N)
            C[r*N+c] = A[r*N+c] + B[r*N+c];
        
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = THREAD_NUM;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGrid = min(blocksPerGrid, MAX_BLOCK_NUM);

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
