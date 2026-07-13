
#include <cuda_runtime.h>
#define MAX_WARP_NUM 32
#define MAX_BLOCK_NUM 128


__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduction(float* input, float* output, int N) {
    __shared__ float warp_sums[MAX_WARP_NUM];

    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    int laneIdx = tid % 32;
    int warpIdx = tid / 32;
    int numWarps = blockDim.x / 32;

    
    int stride = blockDim.x * gridDim.x;

    int vec_N = N / 4;
    float4* vec_input = reinterpret_cast<float4*>(input);

    float sum = 0.0f;

    // 多 block 正确复用：每个线程负责 grid-stride 上的一串元素
    for (int i = gtid; i < vec_N; i += stride) {
        float4 val = vec_input[i];
        sum += val.x + val.y + val.z + val.w;
    }

    int tail_start = vec_N * 4;
    for (int i = tail_start + gtid; i < N; i += stride) {
        sum += input[i];
    }

    // 第一级：每个 warp 内部归约
    sum = warpReduce(sum);

    // 每个 warp 的 lane 0 写出该 warp 的 partial sum
    if (laneIdx == 0) {
        warp_sums[warpIdx] = sum;
    }

    __syncthreads();

    // 第二级：让 warp 0 归约所有 warp 的 partial sum
    sum = (tid < numWarps) ? warp_sums[tid] : 0.0f;

    if (warpIdx == 0) {
        sum = warpReduce(sum);
    }

    if(tid == 0)
    {
        atomicAdd(output, sum);
    }


}

extern "C" void solve(float *input, float *output, int N) {
    int threadsPerBlock = 32*MAX_WARP_NUM;

    int workItems = (N+3)/4;
    int blockNum = (workItems + threadsPerBlock - 1) / threadsPerBlock;

    blockNum = blockNum < MAX_BLOCK_NUM ? blockNum : MAX_BLOCK_NUM;

    cudaMemset(output, 0, sizeof(float));
    reduction<<<blockNum, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}