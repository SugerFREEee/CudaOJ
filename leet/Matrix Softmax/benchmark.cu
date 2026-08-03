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

constexpr int kWarmup   = 10;
constexpr int kMeasure  = 50;
// Performance case matches the doc's benchmark: (M, N) = (4096, 4096).
constexpr int kDefaultM = 4096;
constexpr int kDefaultN = 4096;

int main(int argc, char **argv) {
    int M = kDefaultM, N = kDefaultN;
    if (argc == 3) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
    }
    if (M <= 0 || N <= 0) { std::fprintf(stderr, "M,N must be positive.\n"); return 1; }

    std::mt19937 gen(20260702);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> hIn(static_cast<size_t>(M) * N);
    for (float &x : hIn) x = dist(gen);

    float *dIn = nullptr;
    float *dOut = nullptr;
    size_t bytes = static_cast<size_t>(M) * N * sizeof(float);
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn.data(), bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < kWarmup; ++i) solve(dIn, dOut, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));

    CHECK_CUDA(cudaEventRecord(ev0));
    for (int i = 0; i < kMeasure; ++i) solve(dIn, dOut, M, N);
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));

    float totalMs = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, ev0, ev1));
    double avgMs = totalMs / kMeasure;
    // Softmax reads input once and writes output once at minimum -> ~2x bytes moved.
    double gbps = 2.0 * bytes / (avgMs * 1e6);

    // Sanity: verify a few row sums ~ 1.
    std::vector<float> hOut(static_cast<size_t>(M) * N);
    CHECK_CUDA(cudaMemcpy(hOut.data(), dOut, bytes, cudaMemcpyDeviceToHost));
    double maxRowErr = 0.0;
    for (int r = 0; r < M; r += (M / 8 + 1)) {
        double s = 0.0;
        for (int j = 0; j < N; ++j) s += hOut[static_cast<size_t>(r) * N + j];
        maxRowErr = std::fmax(maxRowErr, std::fabs(s - 1.0));
    }

    std::printf("M=%d N=%d (%.2f MiB)\n", M, N, bytes / (1024.0 * 1024.0));
    std::printf("iterations: warmup=%d, measure=%d\n", kWarmup, kMeasure);
    std::printf("average: %.4f ms\n", avgMs);
    std::printf("effective bandwidth (2x IO): %.2f GB/s\n", gbps);
    std::printf("max sampled row-sum error: %.6f (expected ~0)\n", maxRowErr);

    CHECK_CUDA(cudaEventDestroy(ev0));
    CHECK_CUDA(cudaEventDestroy(ev1));
    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    return 0;
}
