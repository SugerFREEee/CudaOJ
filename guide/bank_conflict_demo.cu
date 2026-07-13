#include <cstdio>
#include <cuda_runtime.h>

// 它对比了几种访问方式：

// shared[threadIdx.x]：每个线程访问不同 bank，基本无冲突
// shared[threadIdx.x * 2]：典型 2-way bank conflict
// shared[0]：所有线程读同一个地址，走广播，不算冲突
// tile[32][32] 按列读：32 个线程都落到同一个 bank，冲突很明显
// tile[32][33] 按列读：加一列 padding 后错开 bank，冲突消失
// tile[32][32] 大约 6.9 ms，而加 padding 的 tile[32][33] 大约 0.35 ms

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                            \
            return 1;                                                          \
        }                                                                      \
    } while (0)

constexpr int kThreadsPerBlock = 32;
constexpr int kBlocks = 4096;
constexpr int kIters = 4096;

// Bank id = (address / 4 bytes) % 32 for float.
// STRIDE=1: thread i reads shared[i], every lane hits a different bank.
// STRIDE=2: thread i reads shared[i * 2], lane 0 and 16 hit the same bank, etc.
template <int STRIDE>
__global__ void strideAccessKernel(float *out) {
    __shared__ volatile float shared[kThreadsPerBlock * STRIDE];

    int lane = threadIdx.x;
    shared[lane * STRIDE] = static_cast<float>(lane);
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < kIters; ++i) {
        sum += shared[lane * STRIDE];
    }

    out[blockIdx.x * blockDim.x + lane] = sum;
}

// All lanes read the same address. NVIDIA shared memory supports broadcast,
// so this is not treated as a 32-way bank conflict.
__global__ void broadcastKernel(float *out) {
    __shared__ volatile float shared[kThreadsPerBlock];

    int lane = threadIdx.x;
    shared[lane] = static_cast<float>(lane);
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < kIters; ++i) {
        sum += shared[0];
    }

    out[blockIdx.x * blockDim.x + lane] = sum;
}

// A warp reads one column from a 32-column row-major tile:
// tile[lane][0] addresses differ by 32 floats, so every lane maps to bank 0.
__global__ void columnReadConflictKernel(float *out) {
    __shared__ volatile float tile[32][32];

    int lane = threadIdx.x;
    tile[lane][0] = static_cast<float>(lane);
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < kIters; ++i) {
        sum += tile[lane][0];
    }

    out[blockIdx.x * blockDim.x + lane] = sum;
}

// Add one padding column. Now tile[lane][0] addresses differ by 33 floats,
// so bank ids become 0, 1, 2, ..., 31 instead of all 0.
__global__ void columnReadPaddedKernel(float *out) {
    __shared__ volatile float tile[32][33];

    int lane = threadIdx.x;
    tile[lane][0] = static_cast<float>(lane);
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < kIters; ++i) {
        sum += tile[lane][0];
    }

    out[blockIdx.x * blockDim.x + lane] = sum;
}

float timeKernel(void (*kernel)(float *), float *out) {
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<kBlocks, kThreadsPerBlock>>>(out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

void printBankMapping() {
    std::printf("Bank mapping examples for one warp:\n");
    std::printf("  shared[threadIdx.x]     -> bank = lane %% 32, no conflict\n");
    std::printf("  shared[threadIdx.x * 2] -> bank = (lane * 2) %% 32, 2-way conflict\n");
    std::printf("  tile[lane][0] with [32][32] -> bank = (lane * 32) %% 32 = 0, 32-way conflict\n");
    std::printf("  tile[lane][0] with [32][33] -> bank = (lane * 33) %% 32 = lane, no conflict\n\n");
}

int main() {
    float *out = nullptr;
    CHECK_CUDA(cudaMalloc(&out, kBlocks * kThreadsPerBlock * sizeof(float)));

    printBankMapping();

    // Warm up kernels before timing.
    strideAccessKernel<1><<<kBlocks, kThreadsPerBlock>>>(out);
    strideAccessKernel<2><<<kBlocks, kThreadsPerBlock>>>(out);
    broadcastKernel<<<kBlocks, kThreadsPerBlock>>>(out);
    columnReadConflictKernel<<<kBlocks, kThreadsPerBlock>>>(out);
    columnReadPaddedKernel<<<kBlocks, kThreadsPerBlock>>>(out);
    CHECK_CUDA(cudaDeviceSynchronize());

    float noConflictMs = timeKernel(strideAccessKernel<1>, out);
    float twoWayConflictMs = timeKernel(strideAccessKernel<2>, out);
    float broadcastMs = timeKernel(broadcastKernel, out);
    float columnConflictMs = timeKernel(columnReadConflictKernel, out);
    float paddedMs = timeKernel(columnReadPaddedKernel, out);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(out));

    std::printf("Timing, lower is better:\n");
    std::printf("  1D no conflict, stride=1:       %.3f ms\n", noConflictMs);
    std::printf("  1D 2-way conflict, stride=2:    %.3f ms\n", twoWayConflictMs);
    std::printf("  broadcast, all read shared[0]:  %.3f ms\n", broadcastMs);
    std::printf("  2D column read, tile[32][32]:   %.3f ms\n", columnConflictMs);
    std::printf("  2D padded read, tile[32][33]:   %.3f ms\n", paddedMs);

    return 0;
}
