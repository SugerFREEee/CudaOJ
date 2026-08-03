#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// 默认测试 optimal.cu，可用 -DSOLVE_FILE='"simple.cu"' 测试简单版。
// GEMM optimal 用 WMMA，需 nvcc -arch=sm_70 及以上编译。
#ifndef SOLVE_FILE
#define SOLVE_FILE "optimal.cu"
#endif
#include SOLVE_FILE

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                            \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// CPU reference GEMM in FP32 then round to FP16 for comparison.
void cpuGemm(const std::vector<half> &A, const std::vector<half> &B,
             const std::vector<half> &C_in, std::vector<half> &C_out,
             int M, int N, int K, float alpha, float beta) {
    C_out.resize(static_cast<size_t>(M) * N);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = __half2float(A[static_cast<size_t>(m) * K + k]);
                float b = __half2float(B[static_cast<size_t>(k) * N + n]);
                acc += a * b;
            }
            float c = __half2float(C_in[static_cast<size_t>(m) * N + n]);
            C_out[static_cast<size_t>(m) * N + n] = __float2half(alpha * acc + beta * c);
        }
    }
}

bool nearlyEqualHalf(half actual, half expected) {
    float a = __half2float(actual);
    float e = __half2float(expected);
    float diff = std::fabs(a - e);
    float scale = std::fmax(1.0f, std::fabs(e));
    // FP16 has ~3 decimal digits of precision; allow generous relative tolerance.
    return diff <= 5e-2f * scale + 1e-3f;
}

std::vector<half> makeRandomHalfMatrix(int rows, int cols,
                                       unsigned int seed, float lo, float hi) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<half> v(static_cast<size_t>(rows) * cols);
    for (half &h : v) {
        h = __float2half(dist(gen));
    }
    return v;
}

bool runCase(const char *name, int M, int N, int K, float alpha, float beta, unsigned int seed) {
    // A: M x K, B: K x N, C: M x N — sizes derived from M,N,K to avoid mismatch.
    std::vector<half> A = makeRandomHalfMatrix(M, K, seed,     -1.0f, 1.0f);
    std::vector<half> B = makeRandomHalfMatrix(K, N, seed + 1, -1.0f, 1.0f);
    std::vector<half> C_in = makeRandomHalfMatrix(M, N, seed + 2, -1.0f, 1.0f);

    std::vector<half> C_ref;
    cpuGemm(A, B, C_in, C_ref, M, N, K, alpha, beta);

    size_t bytesA = static_cast<size_t>(M) * K * sizeof(half);
    size_t bytesB = static_cast<size_t>(K) * N * sizeof(half);
    size_t bytesC = static_cast<size_t>(M) * N * sizeof(half);

    half *dA = nullptr;
    half *dB = nullptr;
    half *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));
    CHECK_CUDA(cudaMemcpy(dA, A.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B.data(), bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, C_in.data(), bytesC, cudaMemcpyHostToDevice));

    solve(dA, dB, dC, M, N, K, alpha, beta);
    CHECK_CUDA(cudaGetLastError());

    std::vector<half> C_gpu(static_cast<size_t>(M) * N);
    CHECK_CUDA(cudaMemcpy(C_gpu.data(), dC, bytesC, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            size_t idx = static_cast<size_t>(m) * N + n;
            if (!nearlyEqualHalf(C_gpu[idx], C_ref[idx])) {
                std::printf("[FAIL] %s: C[%d][%d] got %.4f, expected %.4f\n",
                            name, m, n,
                            __half2float(C_gpu[idx]), __half2float(C_ref[idx]));
                return false;
            }
        }
    }
    std::printf("[PASS] %s, M=%d N=%d K=%d alpha=%.1f beta=%.1f\n",
                name, M, N, K, static_cast<double>(alpha), static_cast<double>(beta));
    return true;
}

int main() {
    int passed = 0;
    int total = 0;

    ++total; passed += runCase("tiny square alpha=1 beta=0",       4,   4,   4, 1.0f, 0.0f, 1)  ? 1 : 0;
    ++total; passed += runCase("non-square 16x8x32 a=.5 b=.5",    16,   8,  32, 0.5f, 0.5f, 10) ? 1 : 0;
    ++total; passed += runCase("alpha=0 beta=2",                   8,  16,  16, 0.0f, 2.0f, 20) ? 1 : 0;
    ++total; passed += runCase("medium 64x128x64 a=1 b=1",        64, 128,  64, 1.0f, 1.0f, 30) ? 1 : 0;
    ++total; passed += runCase("non-pow2 33x50x47 a=1.5 b=.25",   33,  50,  47, 1.5f, 0.25f, 40) ? 1 : 0;
    ++total; passed += runCase("aligned 128x256x128 a=1 b=0",    128, 256, 128, 1.0f, 0.0f, 50) ? 1 : 0;
    ++total; passed += runCase("perf-shape 256x256x256 a=1 b=0", 256, 256, 256, 1.0f, 0.0f, 60) ? 1 : 0;

    std::printf("\n%d/%d cases passed\n", passed, total);
    return passed == total ? 0 : 1;
}
