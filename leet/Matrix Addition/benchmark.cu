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

constexpr int kWarmup  = 20;
constexpr int kMeasure = 100;
constexpr int kDefaultN = 4096;  // Performance case per problem statement.

int main(int argc, char **argv) {
    int N = kDefaultN;
    if (argc == 2) N = std::atoi(argv[1]);
    if (N <= 0) { std::fprintf(stderr, "N must be positive.\n"); return 1; }

    std::mt19937 gen(20260702);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    size_t elems = static_cast<size_t>(N) * N;
    std::vector<float> hA(elems), hB(elems);
    for (auto &x : hA) x = dist(gen);
    for (auto &x : hB) x = dist(gen);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    size_t bytes = elems * sizeof(float);
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dC, bytes));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < kWarmup; ++i) solve(dA, dB, dC, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));

    CHECK_CUDA(cudaEventRecord(ev0));
    for (int i = 0; i < kMeasure; ++i) solve(dA, dB, dC, N);
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));

    float totalMs = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, ev0, ev1));
    double avgMs = totalMs / kMeasure;
    // 2 reads + 1 write per element
    double gbps = 3.0 * bytes / (avgMs * 1e6);

    std::printf("N=%d (%.2f MiB per matrix)\n", N, bytes / (1024.0 * 1024.0));
    std::printf("iterations: warmup=%d, measure=%d\n", kWarmup, kMeasure);
    std::printf("average: %.4f ms\n", avgMs);
    std::printf("effective bandwidth (3x IO): %.2f GB/s\n", gbps);

    CHECK_CUDA(cudaEventDestroy(ev0));
    CHECK_CUDA(cudaEventDestroy(ev1));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    return 0;
}
