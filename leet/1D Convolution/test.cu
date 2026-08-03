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

// CPU reference: valid convolution (no kernel flip, i.e. sliding dot product).
// output[i] = sum_{j} input[i+j] * kernel[j], computed in double for accuracy.
void cpuConv(const std::vector<float> &in, const std::vector<float> &ker,
             std::vector<float> &out, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    out.resize(output_size);
    for (int i = 0; i < output_size; ++i) {
        double acc = 0.0;
        for (int j = 0; j < kernel_size; ++j) {
            acc += static_cast<double>(in[i + j]) * static_cast<double>(ker[j]);
        }
        out[i] = static_cast<float>(acc);
    }
}

bool nearlyEqual(float a, float e) {
    float diff = std::fabs(a - e);
    float scale = std::fmax(1.0f, std::fabs(e));
    return diff <= 1e-3f * scale;
}

std::vector<float> makeArray(int n, unsigned int seed, float lo, float hi) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (float &x : v) x = dist(gen);
    return v;
}

bool runCase(const char *name, int input_size, int kernel_size, unsigned int seed) {
    auto hIn = makeArray(input_size, seed, -1.0f, 1.0f);
    auto hKer = makeArray(kernel_size, seed + 1, -1.0f, 1.0f);
    std::vector<float> ref;
    cpuConv(hIn, hKer, ref, input_size, kernel_size);
    int output_size = input_size - kernel_size + 1;

    std::vector<float> hOut(output_size, 0.0f);
    float *dIn = nullptr, *dKer = nullptr, *dOut = nullptr;
    CHECK_CUDA(cudaMalloc(&dIn, (size_t)input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dKer, (size_t)kernel_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, (size_t)output_size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dIn, hIn.data(), (size_t)input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dKer, hKer.data(), (size_t)kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dOut, 0, (size_t)output_size * sizeof(float)));

    solve(dIn, dKer, dOut, input_size, kernel_size);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(hOut.data(), dOut, (size_t)output_size * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dKer));
    CHECK_CUDA(cudaFree(dOut));

    for (int i = 0; i < output_size; ++i) {
        if (!nearlyEqual(hOut[i], ref[i])) {
            std::printf("[FAIL] %s: index %d, got %.6f, expected %.6f\n", name, i, hOut[i], ref[i]);
            return false;
        }
    }
    std::printf("[PASS] %s, input=%d kernel=%d output=%d\n", name, input_size, kernel_size, output_size);
    return true;
}

int main() {
    int passed = 0, total = 0;

    ++total; passed += runCase("kernel_size=1 (copy)",       10,    1, 1)  ? 1 : 0;
    ++total; passed += runCase("kernel==input (single out)", 16,   16, 2)  ? 1 : 0;
    ++total; passed += runCase("small 32/5",                 32,    5, 3)  ? 1 : 0;
    ++total; passed += runCase("non block multiple 1000/17", 1000, 17, 4)  ? 1 : 0;
    ++total; passed += runCase("medium 100000/128",       100000,  128, 5) ? 1 : 0;
    ++total; passed += runCase("perf-shape 1.5M/2047",   1500000, 2047, 6) ? 1 : 0;

    std::printf("\n%d/%d cases passed\n", passed, total);
    return passed == total ? 0 : 1;
}
