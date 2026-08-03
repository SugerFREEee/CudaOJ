#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
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

float cpuSum(const std::vector<float> &values) {
    double sum = 0.0;
    for (float value : values) {
        sum += static_cast<double>(value);
    }
    return static_cast<float>(sum);
}

bool nearlyEqual(float actual, float expected) {
    float diff = std::fabs(actual - expected);
    float scale = std::fmax(1.0f, std::fabs(expected));
    return diff <= 1e-4f * scale;
}

bool runCase(const char *name, const std::vector<float> &input) {
    int n = static_cast<int>(input.size());
    float expected = cpuSum(input);
    float actual = 0.0f;

    float *deviceInput = nullptr;
    float *deviceOutput = nullptr;
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    CHECK_CUDA(cudaMalloc(&deviceInput, bytes));
    CHECK_CUDA(cudaMalloc(&deviceOutput, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(deviceInput, input.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(deviceOutput, 0, sizeof(float)));

    solve(deviceInput, deviceOutput, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(&actual, deviceOutput, sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(deviceInput));
    CHECK_CUDA(cudaFree(deviceOutput));

    if (!nearlyEqual(actual, expected)) {
        std::printf("[FAIL] %s: got %.8f, expected %.8f, abs diff %.8f\n", name,
                    actual, expected, std::fabs(actual - expected));
        return false;
    }

    std::printf("[PASS] %s, N=%d, sum=%.8f\n", name, n, actual);
    return true;
}

std::vector<float> makeRandomVector(int n, unsigned int seed, float low, float high) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(low, high);

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
        std::vector<float> input{1.0f, 2.0f, 3.5f, -4.0f, 8.25f};
        ++total;
        passed += runCase("small fixed sample", input) ? 1 : 0;
    }

    {
        std::vector<float> input(256);
        for (int i = 0; i < static_cast<int>(input.size()); ++i) {
            input[i] = static_cast<float>(i % 17) - 8.0f;
        }
        ++total;
        passed += runCase("one block sized sample", input) ? 1 : 0;
    }

    {
        std::vector<float> input = makeRandomVector(1000, 1234, -10.0f, 10.0f);
        ++total;
        passed += runCase("non power of two random sample", input) ? 1 : 0;
    }

    {
        std::vector<float> input = makeRandomVector(1 << 20, 5678, -1.0f, 1.0f);
        ++total;
        passed += runCase("large random sample", input) ? 1 : 0;
    }

    std::printf("\n%d/%d cases passed\n", passed, total);
    return passed == total ? 0 : 1;
}
