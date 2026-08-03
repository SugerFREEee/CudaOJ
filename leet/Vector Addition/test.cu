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

bool nearlyEqual(float actual, float expected) {
    float diff = std::fabs(actual - expected);
    float scale = std::fmax(1.0f, std::fmax(std::fabs(actual), std::fabs(expected)));
    return diff <= 1e-5f * scale;
}

bool runCase(const char *name, const std::vector<float> &hostA,
             const std::vector<float> &hostB) {
    int n = static_cast<int>(hostA.size());
    std::vector<float> hostC(n, 0.0f);
    std::vector<float> expected(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        expected[i] = hostA[i] + hostB[i];
    }

    float *deviceA = nullptr;
    float *deviceB = nullptr;
    float *deviceC = nullptr;
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    CHECK_CUDA(cudaMalloc(&deviceA, bytes));
    CHECK_CUDA(cudaMalloc(&deviceB, bytes));
    CHECK_CUDA(cudaMalloc(&deviceC, bytes));
    CHECK_CUDA(cudaMemcpy(deviceA, hostA.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceB, hostB.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(deviceC, 0, bytes));

    solve(deviceA, deviceB, deviceC, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(hostC.data(), deviceC, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(deviceA));
    CHECK_CUDA(cudaFree(deviceB));
    CHECK_CUDA(cudaFree(deviceC));

    for (int i = 0; i < n; ++i) {
        if (!nearlyEqual(hostC[i], expected[i])) {
            std::printf("[FAIL] %s: index %d, got %.8f, expected %.8f\n", name, i,
                        hostC[i], expected[i]);
            return false;
        }
    }

    std::printf("[PASS] %s, N=%d\n", name, n);
    return true;
}

std::vector<float> makeRandomVector(int n, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    std::vector<float> values(n);
    for (float &value : values) {
        value = dist(gen);
    }
    return values;
}

int main() {
    int passed = 0;
    int total = 0;

    {
        std::vector<float> a{1.0f, 2.5f, -3.0f, 4.25f};
        std::vector<float> b{2.0f, -0.5f, 7.0f, -4.25f};
        ++total;
        passed += runCase("small fixed sample", a, b) ? 1 : 0;
    }

    {
        int n = 257;
        std::vector<float> a = makeRandomVector(n, 123);
        std::vector<float> b = makeRandomVector(n, 456);
        ++total;
        passed += runCase("non block multiple", a, b) ? 1 : 0;
    }

    {
        int n = 4096;
        std::vector<float> a = makeRandomVector(n, 789);
        std::vector<float> b = makeRandomVector(n, 101112);
        ++total;
        passed += runCase("larger random sample", a, b) ? 1 : 0;
    }

    std::printf("\n%d/%d cases passed\n", passed, total);
    return passed == total ? 0 : 1;
}
