#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

// 默认包含 solve.cu，可用 -DSOLVE_FILE='"optimal.cu"' 等切换。
#ifndef SOLVE_FILE
#define SOLVE_FILE "solve.cu"
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

bool nearlyEqual(float actual, float expected) {
    float diff = std::fabs(actual - expected);
    float scale = std::fmax(1.0f, std::fmax(std::fabs(actual), std::fabs(expected)));
    return diff <= 1e-5f * scale;
}

std::vector<float> makeMatrix(int N, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    std::vector<float> v(static_cast<size_t>(N) * N);
    for (float &x : v) x = dist(gen);
    return v;
}

bool runCase(const char *name, int N, unsigned int seed) {
    auto hA = makeMatrix(N, seed);
    auto hB = makeMatrix(N, seed + 1);
    std::vector<float> ref(static_cast<size_t>(N) * N);
    for (size_t i = 0; i < ref.size(); ++i) ref[i] = hA[i] + hB[i];

    std::vector<float> hC(static_cast<size_t>(N) * N, 0.0f);
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dC, bytes));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytes));

    solve(dA, dB, dC, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    for (size_t i = 0; i < ref.size(); ++i) {
        if (!nearlyEqual(hC[i], ref[i])) {
            std::printf("[FAIL] %s: index %zu, got %.6f, expected %.6f\n", name, i, hC[i], ref[i]);
            return false;
        }
    }
    std::printf("[PASS] %s, N=%d\n", name, N);
    return true;
}

int main() {
    int passed = 0, total = 0;

    ++total; passed += runCase("2x2", 2, 1) ? 1 : 0;
    ++total; passed += runCase("1x1", 1, 2) ? 1 : 0;
    ++total; passed += runCase("small 33x33", 33, 3) ? 1 : 0;
    ++total; passed += runCase("256x256", 256, 4) ? 1 : 0;
    ++total; passed += runCase("non-aligned 1023x1023", 1023, 5) ? 1 : 0;
    ++total; passed += runCase("perf-shape 4096x4096", 4096, 6) ? 1 : 0;

    std::printf("\n%d/%d cases passed\n", passed, total);
    return passed == total ? 0 : 1;
}
