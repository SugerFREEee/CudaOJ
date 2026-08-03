#include <cuda_runtime.h>
#include <cfloat>  // FLT_MAX

// 最优版：一个 block 处理一行 + online softmax（一遍求 max/sum）+ warp shuffle 两级归约
// + float4 向量化 + grid-stride 遍历行（适配任意 M）。
// 说明：float4 要求每行起始 16B 对齐；仅当 N%4==0 时行起始才对齐，否则退回标量路径。
#define THREADS 256
#define WARPS (THREADS / 32)

__device__ __forceinline__ void warpReduceOnline(float& m, float& d) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(0xffffffff, m, offset);
        float d2 = __shfl_down_sync(0xffffffff, d, offset);
        float mn = fmaxf(m, m2);
        d = d * expf(m - mn) + d2 * expf(m2 - mn);
        m = mn;
    }
}

__global__ void row_softmax_opt(const float* input, float* output, int M, int N) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    __shared__ float warp_m[WARPS];
    __shared__ float warp_d[WARPS];
    __shared__ float s_max;
    __shared__ float s_inv;

    // 仅当 N%4==0 时每行起始才 16B 对齐，才能安全用 float4；否则走标量。
    int N4 = (N % 4 == 0) ? (N / 4) : 0;

    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        const float* x = input + (size_t)row * N;
        float* y = output + (size_t)row * N;
        const float4* x4 = reinterpret_cast<const float4*>(x);

        // 第 1 遍：online 递推求局部 (m, d)
        float m = -FLT_MAX;
        float d = 0.0f;
        for (int i = tid; i < N4; i += blockDim.x) {
            float4 v = x4[i];
            float vals[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float mn = fmaxf(m, vals[k]);
                d = d * expf(m - mn) + expf(vals[k] - mn);
                m = mn;
            }
        }
        for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
            float xi = x[i];
            float mn = fmaxf(m, xi);
            d = d * expf(m - mn) + expf(xi - mn);
            m = mn;
        }

        // 两级归约
        warpReduceOnline(m, d);
        if (lane == 0) {
            warp_m[wid] = m;
            warp_d[wid] = d;
        }
        __syncthreads();
        if (tid < 32) {
            float mm = (tid < WARPS) ? warp_m[tid] : -FLT_MAX;
            float dd = (tid < WARPS) ? warp_d[tid] : 0.0f;
            warpReduceOnline(mm, dd);
            if (tid == 0) {
                s_max = mm;
                s_inv = 1.0f / dd;
            }
        }
        __syncthreads();
        float mx = s_max;
        float inv = s_inv;

        // 第 2 遍：float4 向量化归一化
        float4* y4 = reinterpret_cast<float4*>(y);
        for (int i = tid; i < N4; i += blockDim.x) {
            float4 v = x4[i];
            float4 r;
            r.x = expf(v.x - mx) * inv;
            r.y = expf(v.y - mx) * inv;
            r.z = expf(v.z - mx) * inv;
            r.w = expf(v.w - mx) * inv;
            y4[i] = r;
        }
        for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
            y[i] = expf(x[i] - mx) * inv;
        }

        __syncthreads();  // 下一行复用 shared 之前先同步
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int M, int N) {
    if (M <= 0 || N <= 0) return;

    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);
    int grid = smCount > 0 ? smCount * 4 : M;
    if (grid > M) grid = M;   // 行数少时退化为一行一 block
    if (grid < 1) grid = 1;

    row_softmax_opt<<<grid, THREADS>>>(input, output, M, N);
    cudaDeviceSynchronize();
}
