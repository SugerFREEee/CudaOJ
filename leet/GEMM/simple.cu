#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 简单版：一线程一元素，FP32 累加，最后写回 half。面试手撕首选。
// C = alpha*A*B + beta*C，A:M×K，B:K×N，C:M×N，row-major，half 存储。
__global__ void gemm_naive(const half* A, const half* B, half* C,
                           int M, int N, int K, float alpha, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += __half2float(A[(size_t)row * K + k]) * __half2float(B[(size_t)k * N + col]);
        }
        float c0 = __half2float(C[(size_t)row * N + col]);
        C[(size_t)row * N + col] = __float2half(alpha * acc + beta * c0);
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const half* A, const half* B, half* C,
                      int M, int N, int K, float alpha, float beta) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_naive<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
