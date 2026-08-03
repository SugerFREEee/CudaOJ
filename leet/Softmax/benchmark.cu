#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

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

constexpr int kWarmup    = 20;
constexpr int kMeasure   = 100;
constexpr int kDefaultN  = 500000;  // Performance case per problem statement.

int main(int argc, char **argv) {
    int n = kDefaultN;
    if (argc == 2) n = std::atoi(argv[1]);
    if (n <= 0) { std::fprintf(stderr, "N must be positive.\n"); return 1; }

    std::mt19937 gen(20260702);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    std::vector<float> hIn(n);
    for (float &x : hIn) x = dist(gen);

    float *dIn = nullptr;
    float *dOut = nullptr;
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < kWarmup; ++i) solve(dIn, dOut, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));

    CHECK_CUDA(cudaEventRecord(ev0));
    for (int i = 0; i < kMeasure; ++i) solve(dIn, dOut, n);
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));

    float totalMs = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, ev0, ev1));
    double avgMs = totalMs / kMeasure;

    // Verify the output sums to ~1 as a sanity check.
    std::vector<float> hOut(n);
    CHECK_CUDA(cudaMemcpy(hOut.data(), dOut, bytes, cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (float v : hOut) sum += v;

    std::printf("N=%d (%.2f MiB)\n", n, bytes / (1024.0 * 1024.0));
    std::printf("iterations: warmup=%d, measure=%d\n", kWarmup, kMeasure);
    std::printf("average: %.4f ms\n", avgMs);
    std::printf("output sum: %.6f (expected ~1.0)\n", sum);

    CHECK_CUDA(cudaEventDestroy(ev0));
    CHECK_CUDA(cudaEventDestroy(ev1));
    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    return 0;
}
