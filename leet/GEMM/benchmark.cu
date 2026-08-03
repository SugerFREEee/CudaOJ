#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_fp16.h>
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
constexpr int kDefaultM = 1024;
constexpr int kDefaultN = 1024;
constexpr int kDefaultK = 1024;

std::vector<half> randHalf(int r, int c, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    std::vector<half> v(static_cast<size_t>(r) * c);
    for (half &h : v) h = __float2half(d(gen));
    return v;
}

bool nearlyEqualHalf(half a, half e) {
    float diff = std::fabs(__half2float(a) - __half2float(e));
    float scale = std::fmax(1.0f, std::fabs(__half2float(e)));
    return diff <= 5e-2f * scale + 1e-3f;
}

void cpuGemm(const std::vector<half> &A, const std::vector<half> &B,
             const std::vector<half> &C0, std::vector<half> &Cout,
             int M, int N, int K, float alpha, float beta) {
    Cout.resize(static_cast<size_t>(M) * N);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += __half2float(A[(size_t)m*K+k]) * __half2float(B[(size_t)k*N+n]);
            float c = __half2float(C0[(size_t)m*N+n]);
            Cout[(size_t)m*N+n] = __float2half(alpha*acc + beta*c);
        }
}

int main(int argc, char **argv) {
    int M = kDefaultM, N = kDefaultN, K = kDefaultK;
    if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }

    auto hA = randHalf(M, K, 1);
    auto hB = randHalf(K, N, 2);
    auto hC = randHalf(M, N, 3);
    float alpha = 1.0f, beta = 0.0f;

    size_t bytesA = (size_t)M*K*sizeof(half);
    size_t bytesB = (size_t)K*N*sizeof(half);
    size_t bytesC = (size_t)M*N*sizeof(half);

    half *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC.data(), bytesC, cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < kWarmup; ++i)
        solve(dA, dB, dC, M, N, K, alpha, beta);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));

    CHECK_CUDA(cudaEventRecord(ev0));
    for (int i = 0; i < kMeasure; ++i)
        solve(dA, dB, dC, M, N, K, alpha, beta);
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));

    float totalMs = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&totalMs, ev0, ev1));
    double avgMs = totalMs / kMeasure;
    // FLOP count: 2*M*N*K multiply-adds
    double tflops = 2.0 * M * N * K / (avgMs * 1e9);

    // Correctness check against small CPU reference (only when small enough)
    bool correct = true;
    if (M <= 128 && N <= 128 && K <= 128) {
        std::vector<half> Cref, Cgpu((size_t)M*N);
        cpuGemm(hA, hB, hC, Cref, M, N, K, alpha, beta);
        CHECK_CUDA(cudaMemcpy(Cgpu.data(), dC, bytesC, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < (size_t)M*N; ++i) {
            if (!nearlyEqualHalf(Cgpu[i], Cref[i])) { correct = false; break; }
        }
    }

    std::printf("M=%d N=%d K=%d\n", M, N, K);
    std::printf("iterations: warmup=%d, measure=%d\n", kWarmup, kMeasure);
    std::printf("average: %.4f ms\n", avgMs);
    std::printf("TFLOPS: %.3f\n", tflops);
    if (M <= 128 && N <= 128 && K <= 128)
        std::printf("correctness: %s\n", correct ? "PASS" : "FAIL");
    else
        std::printf("correctness: skipped (matrix too large for CPU ref)\n");

    CHECK_CUDA(cudaEventDestroy(ev0));
    CHECK_CUDA(cudaEventDestroy(ev1));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    return correct ? 0 : 1;
}
