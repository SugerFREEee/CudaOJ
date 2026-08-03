#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// 最优版：WMMA / Tensor Core，16x16x16 tile，FP32 累加，最后 alpha/beta 融合写回 half。
// C = alpha*A*B + beta*C，A:M×K，B:K×N，C:M×N，row-major，half 存储。
// 需要 -arch=sm_70 及以上编译。M/N/K 非 16 倍数时退回标量 kernel 保证正确。
#define WM 16
#define WN 16
#define WK 16

// 一个 block = 一个 warp（32 线程），负责一个 16x16 的 C tile。
__global__ void gemm_wmma(const half* A, const half* B, half* C,
                          int M, int N, int K, float alpha, float beta) {
    int tileRow = blockIdx.x * WM;
    int tileCol = blockIdx.y * WN;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < K; k += WK) {
        wmma::load_matrix_sync(a_frag, A + (size_t)tileRow * K + k, K);
        wmma::load_matrix_sync(b_frag, B + (size_t)k * N + tileCol, N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // 把 FP32 累加结果落到 shared，再融合 alpha/beta 写回 half。
    __shared__ float ctile[WM * WN];
    wmma::store_matrix_sync(ctile, acc_frag, WN, wmma::mem_row_major);

    int lane = threadIdx.x;  // 0..31
    for (int idx = lane; idx < WM * WN; idx += 32) {
        int r = idx / WN;
        int c = idx % WN;
        int gr = tileRow + r;
        int gc = tileCol + c;
        float acc = ctile[idx];
        float c0 = __half2float(C[(size_t)gr * N + gc]);
        C[(size_t)gr * N + gc] = __float2half(alpha * acc + beta * c0);
    }
}

// 标量 fallback：M/N/K 非 16 倍数时用，保证任意尺寸正确。
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

    if (M % 16 == 0 && N % 16 == 0 && K % 16 == 0) {
        dim3 grid((M + WM - 1) / WM, (N + WN - 1) / WN);
        gemm_wmma<<<grid, 32>>>(A, B, C, M, N, K, alpha, beta);
    } else {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        gemm_naive<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    }
    cudaDeviceSynchronize();
}
