#include <cuda_runtime.h>

// 简单版：shared memory tiling，每个线程算 C 的一个元素。面试手撕首选。
// C = A×B, A:M×N, B:N×K, C:M×K, row-major。沿归约维 N 分块。
#define BM 16
#define BN 16
#define BK 32

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float shared_A[BM][BK+1];
    __shared__ float shared_B[BK][BN];
    float acc = 0.0f;

    int m = blockIdx.y * BM;
    int k_out = blockIdx.x * BN;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int stride = blockDim.x * blockDim.y;

    for (int n0 = 0; n0 < N; n0 += BK) {
        for (int i = tid; i < BM * BK; i += stride) {
            int tile_row = i / BK;
            int tile_col = i % BK;
            int global_row = m + tile_row;
            int global_col = n0 + tile_col;
            shared_A[tile_row][tile_col] = (global_row < M && global_col < N)
                ? A[global_row * N + global_col] : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += stride) {
            int tile_row = i / BN;
            int tile_col = i % BN;
            int global_row = n0 + tile_row;
            int global_col = k_out + tile_col;
            shared_B[tile_row][tile_col] = (global_row < N && global_col < K)
                ? B[global_row * K + global_col] : 0.0f;
        }

        __syncthreads();
        for (int h = 0; h < BK; ++h) {
            acc += shared_A[threadIdx.y][h] * shared_B[h][threadIdx.x];
        }
        __syncthreads();
    }

    int row = m + threadIdx.y;
    int col = k_out + threadIdx.x;
    if (row < M && col < K) {
        C[row * K + col] = acc;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
// A: M x N, B: N x K, C: M x K, all row-major
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BM, BN);
    dim3 blocksPerGrid((K + BN - 1) / BN,
                       (M + BM - 1) / BM);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
