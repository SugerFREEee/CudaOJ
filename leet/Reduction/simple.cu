#include <cuda_runtime.h>
#define MAX_BLOCK_NUM 32
// 简单版：单 kernel。grid-stride 求局部和 -> warp 归约 -> block 归约 -> atomicAdd 汇总。
// 面试手撕首选：一个 kernel 讲清楚 warp shuffle + block 两级归约 + 跨 block 合并。
__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduction(const float* input, float* output, int N) {
    __shared__ float warp_sums[32];

    int tid = threadIdx.x;
    int laneIdx = tid % 32;
    int warpIdx = tid / 32;
    int numWarps = blockDim.x / 32;

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float val = 0.0f;
    for (int i = globalIdx; i < N; i += stride) {
        val += input[i];
    }

    val = warpReduce(val);
    if (laneIdx == 0) {
        warp_sums[warpIdx] = val;
    }
    __syncthreads();

    val = (tid < numWarps) ? warp_sums[tid] : 0.0f;
    if (warpIdx == 0) {
        val = warpReduce(val);
    }

    if (tid == 0) {
        atomicAdd(output, val);
    }
}

extern "C" void solve(const float *input, float *output, int N) {
    int threadsPerBlock = 256;
    int blockNum = (N + threadsPerBlock - 1) / threadsPerBlock;
    blockNum = blockNum < MAX_BLOCK_NUM ? blockNum : MAX_BLOCK_NUM;

    cudaMemset(output, 0, sizeof(float));
    reduction<<<blockNum, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
