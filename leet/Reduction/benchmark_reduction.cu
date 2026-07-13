#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#ifndef SOLVE_FILE
#error "Compile with -DSOLVE_FILE=\"path/to/solve.cu\""
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

constexpr int kWarmupIterations = 20;
constexpr int kMeasureIterations = 100;
constexpr int kDefaultElements = 1 << 24;  // 16,777,216 floats = 64 MiB.

float cpuSum(const std::vector<float> &input) {
    double sum = 0.0;
    for (float value : input) {
        sum += static_cast<double>(value);
    }
    return static_cast<float>(sum);
}

bool nearlyEqual(float actual, float expected) {
    float diff = std::fabs(actual - expected);
    float scale = std::fmax(1.0f, std::fabs(expected));
    return diff <= 1e-4f * scale;
}

int main(int argc, char **argv) {
    int n = kDefaultElements;
    if (argc == 2) {
        n = std::atoi(argv[1]);
    }
    if (n <= 0) {
        std::fprintf(stderr, "N must be positive.\n");
        return 1;
    }

    std::mt19937 gen(20260702);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> hostInput(n);
    for (float &value : hostInput) {
        value = dist(gen);
    }
    float expected = cpuSum(hostInput);

    float *deviceInput = nullptr;
    float *deviceOutput = nullptr;
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    CHECK_CUDA(cudaMalloc(&deviceInput, bytes));
    CHECK_CUDA(cudaMalloc(&deviceOutput, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(deviceInput, hostInput.data(), bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < kWarmupIterations; ++i) {
        CHECK_CUDA(cudaMemset(deviceOutput, 0, sizeof(float)));
        solve(deviceInput, deviceOutput, n);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < kMeasureIterations; ++i) {
        CHECK_CUDA(cudaMemset(deviceOutput, 0, sizeof(float)));
        solve(deviceInput, deviceOutput, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float totalMs = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, start, stop));

    float actual = 0.0f;
    CHECK_CUDA(cudaMemcpy(&actual, deviceOutput, sizeof(float), cudaMemcpyDeviceToHost));

    double averageMs = totalMs / kMeasureIterations;
    double gibPerSecond = static_cast<double>(bytes) / (averageMs * 1.0e6);

    std::printf("N: %d floats (%.2f MiB)\n", n, bytes / (1024.0 * 1024.0));
    std::printf("iterations: warmup=%d, measure=%d\n", kWarmupIterations, kMeasureIterations);
    std::printf("average solve time: %.4f ms\n", averageMs);
    std::printf("effective input-read bandwidth: %.2f GB/s\n", gibPerSecond);
    std::printf("result: %.8f, expected: %.8f, %s\n", actual, expected,
                nearlyEqual(actual, expected) ? "PASS" : "FAIL");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(deviceInput));
    CHECK_CUDA(cudaFree(deviceOutput));
    return nearlyEqual(actual, expected) ? 0 : 1;
}
