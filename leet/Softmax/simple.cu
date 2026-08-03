#include <cuda_runtime.h>
#include <cfloat>  // FLT_MAX

// 简单版：单 block（1024 线程）处理整个数组，三遍 grid-stride 扫描。
// pass1 求 max，pass2 求 sum(exp(x-max))，pass3 归一化。
// 单 block 不需要跨 block 同步，逻辑最清晰，面试手撕首选。N 任意大都能跑（靠 grid-stride）。
#define BLOCK 1024

__global__ void softmax_simple(const float* input, float* output, int N) {
    __shared__ float sdata[BLOCK];
    __shared__ float s_max;
    __shared__ float s_sum;

    int tid = threadIdx.x;

    // pass1: 求全局最大值
    float m = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) m = fmaxf(m, input[i]);
    sdata[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) s_max = sdata[0];
    __syncthreads();
    float mx = s_max;

    // pass2: 求 sum(exp(x - max))
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) sum += expf(input[i] - mx);
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) s_sum = sdata[0];
    __syncthreads();
    float inv = 1.0f / s_sum;

    // pass3: 归一化输出
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - mx) * inv;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;
    softmax_simple<<<1, BLOCK>>>(input, output, N);
    cudaDeviceSynchronize();
}
