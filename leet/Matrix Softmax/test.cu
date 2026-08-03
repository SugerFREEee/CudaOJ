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

// CPU reference: row-wise softmax with the max trick, computed in double.
void cpuRowSoftmax(const std::vector<float> &in, std::vector<float> &out, int M, int N) {
    out.resize(static_cast<size_t>(M) * N);
    for (int r = 0; r < M; ++r) {
        const float *row = in.data() + static_cast<size_t>(r) * N;
        float *orow = out.data() + static_cast<size_t>(r) * N;

        float maxVal = row[0];
        for (int j = 0; j < N; ++j) maxVal = std::fmax(maxVal, row[j]);

        double sum = 0.0;
        for (int j = 0; j < N; ++j) sum += std::exp(static_cast<double>(row[j]) - maxVal);

        for (int j = 0; j < N; ++j) {
            orow[j] = static_cast<float>(std::exp(static_cast<double>(row[j]) - maxVal) / sum);
        }
    }
}

bool nearlyEqual(float actual, float expected) {
    float diff = std::fabs(actual - expected);
    float scale = std::fmax(1e-6f, std::fabs(expected));
    return diff <= 1e-4f * scale + 1e-7f;
}

std::vector<float> makeInput(int M, int N, unsigned int seed, float lo, float hi) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(static_cast<size_t>(M) * N);
    for (float &x : v) x = dist(gen);
    return v;
}

bool runCase(const char *name, const std::vector<float> &input, int M, int N) {
    std::vector<float> ref;
    cpuRowSoftmax(input, ref, M, N);

    std::vector<float> gpu(static_cast<size_t>(M) * N, 0.0f);
    float *dIn = nullptr;
    float *dOut = nullptr;
    size_t bytes = static_cast<size_t>(M) * N * sizeof(float);

    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, input.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dOut, 0, bytes));

    solve(dIn, dOut, M, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(gpu.data(), dOut, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));

    for (int r = 0; r < M; ++r) {
        double rowSum = 0.0;
        for (int j = 0; j < N; ++j) {
            size_t idx = static_cast<size_t>(r) * N + j;
            if (!nearlyEqual(gpu[idx], ref[idx])) {
                std::printf("[FAIL] %s: row %d col %d, got %.8f, expected %.8f\n",
                            name, r, j, gpu[idx], ref[idx]);
                return false;
            }
            rowSum += gpu[idx];
        }
        if (std::fabs(rowSum - 1.0) > 1e-3) {
            std::printf("[FAIL] %s: row %d sum %.6f, expected ~1.0\n", name, r, rowSum);
            return false;
        }
    }

    std::printf("[PASS] %s, M=%d N=%d\n", name, M, N);
    return true;
}

int main() {
    int passed = 0;
    int total = 0;

    {
        // small fixed matrix, each row independent
        std::vector<float> in{1.0f, 2.0f, 3.0f,
                              -1.0f, 0.0f, 1.0f};
        ++total; passed += runCase("2x3 fixed", in, 2, 3) ? 1 : 0;
    }
    {
        // single row = plain softmax
        auto in = makeInput(1, 100, 1, -5.0f, 5.0f);
        ++total; passed += runCase("single row 1x100", in, 1, 100) ? 1 : 0;
    }
    {
        // single column: each row has 1 element -> output all 1.0
        auto in = makeInput(50, 1, 2, -10.0f, 10.0f);
        ++total; passed += runCase("single column 50x1", in, 50, 1) ? 1 : 0;
    }
    {
        // large magnitude values to exercise the max trick per row
        std::vector<float> in{1000.0f, 1001.0f, 1002.0f, 999.0f,
                              -1000.0f, -1001.0f, -999.0f, -1002.0f};
        ++total; passed += runCase("2x4 overflow trick", in, 2, 4) ? 1 : 0;
    }
    {
        // N not a multiple of block/warp size
        auto in = makeInput(64, 257, 3, -8.0f, 8.0f);
        ++total; passed += runCase("64x257 non aligned", in, 64, 257) ? 1 : 0;
    }
    {
        // wide rows
        auto in = makeInput(16, 4096, 4, -20.0f, 20.0f);
        ++total; passed += runCase("16x4096 wide rows", in, 16, 4096) ? 1 : 0;
    }
    {
        // many rows
        auto in = makeInput(4096, 128, 5, -10.0f, 10.0f);
        ++total; passed += runCase("4096x128 many rows", in, 4096, 128) ? 1 : 0;
    }

    std::printf("\n%d/%d cases passed\n", passed, total);
    return passed == total ? 0 : 1;
}
