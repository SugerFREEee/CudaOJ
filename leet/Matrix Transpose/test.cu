#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "solve.cu"

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                            \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

bool nearlyEqual(float actual, float expected) {
    float diff = std::fabs(actual - expected);
    float scale = std::fmax(1.0f, std::fabs(expected));
    return diff <= 1e-6f * scale;
}

std::vector<float> makeInput(int rows, int cols, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    std::vector<float> input(static_cast<size_t>(rows) * cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            // Mix deterministic coordinates with random noise. This makes index bugs easier to spot.
            input[static_cast<size_t>(row) * cols + col] =
                row * 0.125f + col * 0.5f + dist(gen) * 0.001f;
        }
    }
    return input;
}

std::vector<float> cpuTranspose(const std::vector<float> &input, int rows, int cols) {
    std::vector<float> expected(static_cast<size_t>(rows) * cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            expected[static_cast<size_t>(col) * rows + row] =
                input[static_cast<size_t>(row) * cols + col];
        }
    }
    return expected;
}

bool runCase(const char *name, int rows, int cols, unsigned int seed) {
    std::vector<float> input = makeInput(rows, cols, seed);
    std::vector<float> expected = cpuTranspose(input, rows, cols);
    std::vector<float> actual(static_cast<size_t>(rows) * cols, 0.0f);

    float *deviceInput = nullptr;
    float *deviceOutput = nullptr;
    size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);

    CHECK_CUDA(cudaMalloc(&deviceInput, bytes));
    CHECK_CUDA(cudaMalloc(&deviceOutput, bytes));
    CHECK_CUDA(cudaMemcpy(deviceInput, input.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(deviceOutput, 0, bytes));

    solve(deviceInput, deviceOutput, rows, cols);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(actual.data(), deviceOutput, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(deviceInput));
    CHECK_CUDA(cudaFree(deviceOutput));

    for (int outRow = 0; outRow < cols; ++outRow) {
        for (int outCol = 0; outCol < rows; ++outCol) {
            size_t idx = static_cast<size_t>(outRow) * rows + outCol;
            if (!nearlyEqual(actual[idx], expected[idx])) {
                std::printf("[FAIL] %s: rows=%d cols=%d, output(%d,%d), got %.8f, expected %.8f\n",
                            name, rows, cols, outRow, outCol, actual[idx], expected[idx]);
                return false;
            }
        }
    }

    std::printf("[PASS] %s, input=%dx%d, output=%dx%d\n", name, rows, cols, cols, rows);
    return true;
}

int main() {
    int passed = 0;
    int total = 0;

    ++total;
    passed += runCase("single element", 1, 1, 1) ? 1 : 0;

    ++total;
    passed += runCase("small rectangular", 3, 5, 2) ? 1 : 0;

    ++total;
    passed += runCase("small square", 32, 32, 3) ? 1 : 0;

    ++total;
    passed += runCase("non tile multiple", 65, 97, 4) ? 1 : 0;

    ++total;
    passed += runCase("wide matrix", 17, 129, 5) ? 1 : 0;

    ++total;
    passed += runCase("tall matrix", 131, 19, 6) ? 1 : 0;

    // A larger correctness smoke test. The official performance case is rows=7000, cols=6000.
    // Keep this smaller so local functional tests stay quick and memory-light.
    ++total;
    passed += runCase("large smoke", 1024, 1537, 7) ? 1 : 0;

    std::printf("\n%d/%d cases passed\n", passed, total);
    return passed == total ? 0 : 1;
}
