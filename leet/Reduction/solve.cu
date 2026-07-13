
#include <cuda_runtime.h>

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

    // 多 block 正确复用：每个线程负责 grid-stride 上的一串元素
    for (int i = globalIdx; i < N; i += stride) {
        val += input[i];
    }

    // 第一级：每个 warp 内部归约
    val = warpReduce(val);

    // 每个 warp 的 lane 0 写出该 warp 的 partial sum
    if (laneIdx == 0) {
        warp_sums[warpIdx] = val;
    }

    __syncthreads();

    // 第二级：让 warp 0 归约所有 warp 的 partial sum
    val = (tid < numWarps) ? warp_sums[tid] : 0.0f;

    if (warpIdx == 0) {
        val = warpReduce(val);
    }

    // 每个 block 只产生一个 block sum，用 atomicAdd 合并到 output[0]
    if (tid == 0) {
        atomicAdd(output, val);
    }
}

extern "C" void solve(const float *input, float *output, int N) {
    int threadsPerBlock = 256;
    int blockNum = (N + threadsPerBlock - 1) / threadsPerBlock;
    blockNum = blockNum < 32 ? blockNum : 32;

    cudaMemset(output, 0, sizeof(float));
    reduction<<<blockNum, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}