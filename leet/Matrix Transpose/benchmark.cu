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

constexpr int kWarmup  = 10;
constexpr int kMeasure = 50;
// Performance case per the Matrix Transpose problem.
constexpr int kDefaultRows = 7000;
constexpr int kDefaultCols = 6000;

int main(int argc, char **argv) {
    int rows = kDefaultRows, cols = kDefaultCols;
    if (argc == 3) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
    }
    if (rows <= 0 || cols <= 0) { std::fprintf(stderr, "rows,cols must be positive.\n"); return 1; }

    std::mt19937 gen(20260702);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> hIn(static_cast<size_t>(rows) * cols);
    for (float &x : hIn) x = dist(gen);

    float *dIn = nullptr;
    float *dOut = nullptr;
    size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < kWarmup; ++i) solve(dIn, dOut, rows, cols);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));

    CHECK_CUDA(cudaEventRecord(ev0));
    for (int i = 0; i < kMeasure; ++i) solve(dIn, dOut, rows, cols);
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));

    float totalMs = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, ev0, ev1));
    double avgMs = totalMs / kMeasure;
    // transpose reads once + writes once -> 2x bytes moved
    double gbps = 2.0 * bytes / (avgMs * 1e6);

    std::printf("rows=%d cols=%d (%.2f MiB)\n", rows, cols, bytes / (1024.0 * 1024.0));
    std::printf("iterations: warmup=%d, measure=%d\n", kWarmup, kMeasure);
    std::printf("average: %.4f ms\n", avgMs);
    std::printf("effective bandwidth (2x IO): %.2f GB/s\n", gbps);

    CHECK_CUDA(cudaEventDestroy(ev0));
    CHECK_CUDA(cudaEventDestroy(ev1));
    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    return 0;
}
