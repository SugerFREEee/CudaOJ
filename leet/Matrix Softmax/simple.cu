#include <cuda_runtime.h>
#include <cfloat>  // FLT_MAX

// 简单版
// 面试手撕首选：不涉及跨 block 同步，每行独立。
#define BLOCK 256
#define MAX_BLOCK_NUM 1024
#define WARP_NUM (BLOCK/32)

__device__ void warp_shuffle(float &m, float &d)
{
    for(int i=16;i>0;i>>=1)
    {
        float m2 = __shfl_down_sync(0xFFFFFFFF, m, i);
        float d2 = __shfl_down_sync(0xFFFFFFFF, d, i);

        float m_new = fmaxf(m, m2);

        d = d * expf(m - m_new) + d2*expf(m2 - m_new);
        m = m_new;
    }
}

__global__ void row_softmax_simple(const float* input, float* output, int M, int N) {
    int bid = blockIdx.x;

    __shared__ float shared_m[WARP_NUM];
    __shared__ float shared_d[WARP_NUM];
    for(int row = bid; row < M; row += gridDim.x)
    {
        float m = -FLT_MAX;
        float d = 0.0f;

        for(int col = threadIdx.x; col < N; col += blockDim.x)
        {
            float val = input[row * N + col];
            float m_new = fmaxf(val, m);
            d = d * expf(m - m_new) + expf(val - m_new);
            m = m_new;
        }
        warp_shuffle(m, d);

        int tid = threadIdx.x % 32;
        int wid = threadIdx.x / 32;
        int nwarps = blockDim.x / 32;

        if(tid == 0)
        {
            shared_m[wid] = m;
            shared_d[wid] = d;
        }

        __syncthreads();


        if(wid == 0)
        {
            m = (tid < nwarps) ? shared_m[tid] : -FLT_MAX;
            d = (tid < nwarps) ? shared_d[tid] : 0.0f;

            warp_shuffle(m, d);

            if(tid == 0)
            {
                shared_m[0] = m;
                shared_d[0] = d;
            }
        }
        __syncthreads();

        for(int col = threadIdx.x; col < N; col += blockDim.x)
        {
            float val = input[row * N + col];
            output[row * N + col] = expf(val - shared_m[0]) / shared_d[0];
        }
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int M, int N) {
    if (M <= 0 || N <= 0) return;
    int block_num = (M + BLOCK - 1) / BLOCK;
    block_num = min(block_num, MAX_BLOCK_NUM);
    row_softmax_simple<<<M, BLOCK>>>(input, output, M, N);
    cudaDeviceSynchronize();
}
