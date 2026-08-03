#include <cuda_runtime.h>
#include <cfloat>  // FLT_MAX

#define THREADS_PER_BLOCK 256
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / 32)
#define MAX_BLOCK_NUM 1024

// 最优版：多 block 协作 + online softmax（一遍同时求 max 和 sum）+ warp shuffle 两级归约。
// stage1 各 block 求局部 (m,d) -> stage2 合并成全局 (m,d) -> stage3 归一化。
// online 归约的哨兵用 -FLT_MAX（不能用 -INFINITY，否则空 lane 相减出 NaN）。

__device__ __forceinline__ void warpReduceOnline(float& m, float& d) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(0xffffffff, m, offset);
        float d2 = __shfl_down_sync(0xffffffff, d, offset);
        float m_new = fmaxf(m, m2);
        d = d * expf(m - m_new) + d2 * expf(m2 - m_new);
        m = m_new;
    }
}

__device__ __forceinline__ void blockReduceOnline(float& max_val, float& sum) {
    __shared__ float shared_max_val[WARPS_PER_BLOCK];
    __shared__ float shared_sum[WARPS_PER_BLOCK];

    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    warpReduceOnline(max_val, sum);
    if (lane == 0) {
        shared_max_val[wid] = max_val;
        shared_sum[wid] = sum;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        float m = (threadIdx.x < WARPS_PER_BLOCK) ? shared_max_val[threadIdx.x] : -FLT_MAX;
        float d = (threadIdx.x < WARPS_PER_BLOCK) ? shared_sum[threadIdx.x] : 0.0f;
        warpReduceOnline(m, d);
        max_val = m;
        sum = d;
    }
}

__global__ void softmax_kernel_stage3(const float* device_max_val, const float* device_sum,
                                      const float* input, float* output, int N) {
    __shared__ float s_max;
    __shared__ float s_sum;
    if (threadIdx.x == 0) {
        s_max = device_max_val[0];
        s_sum = device_sum[0];
    }
    __syncthreads();

    float inv = 1.0f / s_sum;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = gtid; i < N; i += stride) {
        output[i] = expf(input[i] - s_max) * inv;
    }
}

__global__ void softmax_kernel_stage2(float* device_max_val, float* device_sum, int blocksPerGrid) {
    int tid = threadIdx.x;
    int stride = blockDim.x;
    float max_val = -FLT_MAX;
    float sum = 0.0f;

    for (int i = tid; i < blocksPerGrid; i += stride) {
        float val = device_max_val[i];
        float d = device_sum[i];
        float max_mew = fmaxf(max_val, val);
        sum = sum * expf(max_val - max_mew) + d * expf(val - max_mew);
        max_val = max_mew;
    }

    blockReduceOnline(max_val, sum);

    if (threadIdx.x == 0) {
        device_max_val[0] = max_val;
        device_sum[0] = sum;
    }
}

__global__ void softmax_kernel_stage1(const float* input, float* device_max_val, float* device_sum, int N) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float max_val = -FLT_MAX;
    float sum = 0.0f;

    for (int i = gtid; i < N; i += stride) {
        float val = input[i];
        float max_mew = fmaxf(max_val, val);
        sum = sum * expf(max_val - max_mew) + expf(val - max_mew);
        max_val = max_mew;
    }

    blockReduceOnline(max_val, sum);

    if (threadIdx.x == 0) {
        device_max_val[blockIdx.x] = max_val;
        device_sum[blockIdx.x] = sum;
    }
}

static float* g_max = nullptr;
static float* g_sum = nullptr;

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    if (g_max == nullptr) {
        cudaMalloc(&g_max, sizeof(float) * MAX_BLOCK_NUM);
        cudaMalloc(&g_sum, sizeof(float) * MAX_BLOCK_NUM);
    }

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGrid = (blocksPerGrid > MAX_BLOCK_NUM) ? MAX_BLOCK_NUM : blocksPerGrid;
    if (blocksPerGrid < 1) blocksPerGrid = 1;

    softmax_kernel_stage1<<<blocksPerGrid, threadsPerBlock>>>(input, g_max, g_sum, N);
    softmax_kernel_stage2<<<1, threadsPerBlock>>>(g_max, g_sum, blocksPerGrid);
    softmax_kernel_stage3<<<blocksPerGrid, threadsPerBlock>>>(g_max, g_sum, input, output, N);
    cudaDeviceSynchronize();
}
