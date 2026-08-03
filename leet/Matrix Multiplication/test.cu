#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

// 默认测试 optimal.cu，可用 -DSOLVE_FILE='"simple.cu"' 测试简单版。
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

// CPU reference: C = A * B, A is MxN, B is NxK, C is MxK, all row-major.
void cpuMatMul(const std::vector<float> &A, const std::vector<float> &B,
               std::vector<float> &C, int M, int N, int K) {
    C.assign(static_cast<size_t>(M) * K, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float a = A[static_cast<size_t>(m) * N + n];
            for (int k = 0; k < K; ++k) {
                C[static_cast<size_t>(m) * K + k] += a * B[static_cast<size_t>(n) * K + k];
            }
        }
    }
}

bool nearlyEqual(float actual, float expected) {
    float diff = std::fabs(actual - expected);
    float scale = std::fmax(1.0f, std::fabs(expected));
    return diff <= 1e-3f * scale;
}

std::vector<float> makeRandomMatrix(int rows, int cols, unsigned int seed, float lo, float hi) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(static_cast<size_t>(rows) * cols);
    for (float &x : v) x = dist(gen);
    return v;
}

bool runCase(const char *name, int M, int N, int K, unsigned int seed) {
    auto hA = makeRandomMatrix(M, N, seed,     -1.0f, 1.0f);
    auto hB = makeRandomMatrix(N, K, seed + 1, -1.0f, 1.0f);
    std::vector<float> hC_ref;
    cpuMatMul(hA, hB, hC_ref, M, N, K);

    size_t bytesA = static_cast<size_t>(M) * N * sizeof(float);
    size_t bytesB = static_cast<size_t>(N) * K * sizeof(float);
    size_t bytesC = static_cast<size_t>(M) * K * sizeof(float);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));

    solve(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());

    std::vector<float> hC_gpu(static_cast<size_t>(M) * K);
    CHECK_CUDA(cudaMemcpy(hC_gpu.data(), dC, bytesC, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            size_t idx = static_cast<size_t>(m) * K + k;
            if (!nearlyEqual(hC_gpu[idx], hC_ref[idx])) {
                std::printf("[FAIL] %s: C[%d][%d] got %.6f, expected %.6f\n",
                            name, m, k, hC_gpu[idx], hC_ref[idx]);
                return false;
            }
        }
    }
    std::printf("[PASS] %s, M=%d N=%d K=%d\n", name, M, N, K);
    return true;
}

int main() {
    int passed = 0;
    int total  = 0;

    ++total; passed += runCase("single element 1x1x1",         1,   1,   1, 1) ? 1 : 0;
    ++total; passed += runCase("small square 4x4x4",            4,   4,   4, 2) ? 1 : 0;
    ++total; passed += runCase("small square 16x16x16",        16,  16,  16, 3) ? 1 : 0;
    ++total; passed += runCase("one block boundary 32x32x32",  32,  32,  32, 4) ? 1 : 0;
    ++total; passed += runCase("non power-of-two 33x47x27",    33,  47,  27, 5) ? 1 : 0;
    ++total; passed += runCase("tall non-square 64x16x32",     64,  16,  32, 6) ? 1 : 0;
    ++total; passed += runCase("wide non-square 16x64x32",     16,  64,  32, 7) ? 1 : 0;
    ++total; passed += runCase("medium 128x128x128",          128, 128, 128, 8) ? 1 : 0;
    ++total; passed += runCase("medium non-pow2 97x113x79",    97, 113,  79, 9) ? 1 : 0;
    ++total; passed += runCase("large smoke 256x256x256",     256, 256, 256, 10) ? 1 : 0;

    std::printf("\n%d/%d cases passed\n", passed, total);
    return passed == total ? 0 : 1;
}
