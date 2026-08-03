#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

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

constexpr int kWarmup  = 10;
constexpr int kMeasure = 50;
constexpr int kDefaultInput  = 1500000;  // Performance case per problem statement.
constexpr int kDefaultKernel = 2047;

int main(int argc, char **argv) {
    int input_size = kDefaultInput;
    int kernel_size = kDefaultKernel;
    if (argc == 3) {
        input_size = std::atoi(argv[1]);
        kernel_size = std::atoi(argv[2]);
    }
    if (input_size <= 0 || kernel_size <= 0 || kernel_size > input_size) {
        std::fprintf(stderr, "invalid sizes.\n");
        return 1;
    }
    int output_size = input_size - kernel_size + 1;

    std::mt19937 gen(20260702);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> hIn(input_size), hKer(kernel_size);
    for (auto &x : hIn) x = dist(gen);
    for (auto &x : hKer) x = dist(gen);

    float *dIn = nullptr, *dKer = nullptr, *dOut = nullptr;
    CHECK_CUDA(cudaMalloc(&dIn, (size_t)input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dKer, (size_t)kernel_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, (size_t)output_size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dIn, hIn.data(), (size_t)input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dKer, hKer.data(), (size_t)kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < kWarmup; ++i) solve(dIn, dKer, dOut, input_size, kernel_size);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));

    CHECK_CUDA(cudaEventRecord(ev0));
    for (int i = 0; i < kMeasure; ++i) solve(dIn, dKer, dOut, input_size, kernel_size);
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));

    float totalMs = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, ev0, ev1));
    double avgMs = totalMs / kMeasure;
    // 2 * output_size * kernel_size FLOPs (mul + add)
    double gflops = 2.0 * (double)output_size * kernel_size / (avgMs * 1e6);

    std::printf("input=%d kernel=%d output=%d\n", input_size, kernel_size, output_size);
    std::printf("iterations: warmup=%d, measure=%d\n", kWarmup, kMeasure);
    std::printf("average: %.4f ms\n", avgMs);
    std::printf("GFLOPS: %.2f\n", gflops);

    CHECK_CUDA(cudaEventDestroy(ev0));
    CHECK_CUDA(cudaEventDestroy(ev1));
    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dKer));
    CHECK_CUDA(cudaFree(dOut));
    return 0;
}
