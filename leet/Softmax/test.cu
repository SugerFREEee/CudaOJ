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

// CPU reference softmax with the max trick, computed in double for accuracy.
void cpuSoftmax(const std::vector<float> &input, std::vector<float> &output) {
    int n = static_cast<int>(input.size());
    output.resize(n);

    float maxVal = input[0];
    for (float v : input) maxVal = std::fmax(maxVal, v);

    double sum = 0.0;
    for (float v : input) sum += std::exp(static_cast<double>(v) - maxVal);

    for (int i = 0; i < n; ++i) {
        output[i] = static_cast<float>(std::exp(static_cast<double>(input[i]) - maxVal) / sum);
    }
}

bool nearlyEqual(float actual, float expected) {
    float diff = std::fabs(actual - expected);
    float scale = std::fmax(1e-6f, std::fabs(expected));
    return diff <= 1e-4f * scale + 1e-7f;
}

std::vector<float> makeInput(int n, unsigned int seed, float lo, float hi) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (float &x : v) x = dist(gen);
    return v;
}

bool runCase(const char *name, const std::vector<float> &input) {
    int n = static_cast<int>(input.size());
    std::vector<float> ref;
    cpuSoftmax(input, ref);

    std::vector<float> gpu(n, 0.0f);
    float *dIn = nullptr;
    float *dOut = nullptr;
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, input.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dOut, 0, bytes));

    solve(dIn, dOut, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(gpu.data(), dOut, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));

    double gpuSum = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!nearlyEqual(gpu[i], ref[i])) {
            std::printf("[FAIL] %s: index %d, got %.8f, expected %.8f\n",
                        name, i, gpu[i], ref[i]);
            return false;
        }
        gpuSum += gpu[i];
    }
    // softmax output must sum to 1.
    if (std::fabs(gpuSum - 1.0) > 1e-3) {
        std::printf("[FAIL] %s: output sum %.6f, expected ~1.0\n", name, gpuSum);
        return false;
    }

    std::printf("[PASS] %s, N=%d, sum=%.6f\n", name, n, gpuSum);
    return true;
}

int main() {
    int passed = 0;
    int total = 0;

    {
        std::vector<float> in{1.0f, 2.0f, 3.0f};
        ++total; passed += runCase("tiny fixed sample", in) ? 1 : 0;
    }
    {
        std::vector<float> in{5.0f};
        ++total; passed += runCase("single element", in) ? 1 : 0;
    }
    {
        // Large magnitude values to exercise the overflow-safe max trick.
        std::vector<float> in{1000.0f, 1001.0f, 1002.0f, 999.0f};
        ++total; passed += runCase("large magnitude overflow trick", in) ? 1 : 0;
    }
    {
        std::vector<float> in(257);
        for (int i = 0; i < 257; ++i) in[i] = static_cast<float>((i % 13) - 6);
        ++total; passed += runCase("non block multiple", in) ? 1 : 0;
    }
    {
        auto in = makeInput(1000, 123, -10.0f, 10.0f);
        ++total; passed += runCase("random 1000", in) ? 1 : 0;
    }
    {
        auto in = makeInput(500000, 456, -50.0f, 50.0f);
        ++total; passed += runCase("large random 500000", in) ? 1 : 0;
    }

    std::printf("\n%d/%d cases passed\n", passed, total);
    return passed == total ? 0 : 1;
}
