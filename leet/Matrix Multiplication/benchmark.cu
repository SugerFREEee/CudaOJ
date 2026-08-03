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

constexpr int kWarmup  = 5;
constexpr int kMeasure = 20;
// Default performance case per problem statement.
constexpr int kDefaultM = 8192;
constexpr int kDefaultN = 6144;
constexpr int kDefaultK = 4096;

static std::vector<float> randMat(int r, int c, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    std::vector<float> v(static_cast<size_t>(r) * c);
    for (float &x : v) x = d(gen);
    return v;
}

int main(int argc, char **argv) {
    int M = kDefaultM, N = kDefaultN, K = kDefaultK;
    if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }

    auto hA = randMat(M, N, 1);
    auto hB = randMat(N, K, 2);

    size_t bytesA = static_cast<size_t>(M) * N * sizeof(float);
    size_t bytesB = static_cast<size_t>(N) * K * sizeof(float);
    size_t bytesC = static_cast<size_t>(M) * K * sizeof(float);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));

    for (int i = 0; i < kWarmup; ++i)
        solve(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));

    CHECK_CUDA(cudaEventRecord(ev0));
    for (int i = 0; i < kMeasure; ++i)
        solve(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));

    float totalMs = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, ev0, ev1));
    double avgMs   = totalMs / kMeasure;
    // FP32 GEMM: 2*M*N*K FLOPs
    double tflops  = 2.0 * M * N * K / (avgMs * 1e9);

    std::printf("M=%d N=%d K=%d\n", M, N, K);
    std::printf("iterations: warmup=%d, measure=%d\n", kWarmup, kMeasure);
    std::printf("average: %.4f ms\n", avgMs);
    std::printf("TFLOPS: %.3f\n", tflops);

    CHECK_CUDA(cudaEventDestroy(ev0));
    CHECK_CUDA(cudaEventDestroy(ev1));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    return 0;
}
