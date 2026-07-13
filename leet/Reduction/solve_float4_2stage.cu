
#include <cuda_runtime.h>
#define MAX_WARP_NUM 32
#define MAX_BLOCK_NUM 128

// The partial buffer is reused across solve calls in this CUDA module.
// This keeps allocation overhead out of the reduction hot path.
static float *g_partial = nullptr;

__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void stage1(float* input, float* partial, int N) {
    __shared__ float shared[MAX_WARP_NUM];

    int tid = threadIdx.x;
    int gtid = tid + blockIdx.x * blockDim.x;
    
    int laneid = tid & 31;
    int warpid = tid / 32;
    int numWarps = blockDim.x / 32;

    int stride = blockDim.x * gridDim.x;
    
    float4 *vec_input = reinterpret_cast<float4*>(input);
    int vec_num = N/4;

    float sum = 0.0;

    for(int t = gtid;t<vec_num;t+=stride)
    {
        float4 val = vec_input[t];
        sum += val.x + val.y + val.z + val.w;
    }

    for(int t = gtid + vec_num*4;t<N;t+=stride)
    {
        sum += input[t];
    }


    sum = warpReduce(sum);

    if(laneid == 0)
    {
        shared[warpid] = sum;
    }

    __syncthreads();

    if(warpid == 0)
    {
        sum = (tid < numWarps) ? shared[tid] : 0.0f;
        sum = warpReduce(sum);
    }

    if(tid == 0)
    {
        partial[blockIdx.x] = sum;
    }

}


__global__ void stage2(const float* partial, float* output, int blockNum) {
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < blockNum; i += 32) {
        sum += partial[i];
    }
    sum = warpReduce(sum);
    if(tid == 0)
    {
        *output = sum;
    }

}

extern "C" void solve(float *input, float *output, int N) {
    if (N <= 0) {
        cudaMemset(output, 0, sizeof(float));
        return;
    }

    constexpr int threadsPerBlock = 32 * MAX_WARP_NUM;
    int workItems = (N + 3) / 4;
    int blockNum = (workItems + threadsPerBlock - 1) / threadsPerBlock;
    blockNum = blockNum < MAX_BLOCK_NUM ? blockNum : MAX_BLOCK_NUM;

    if (g_partial == nullptr) {
        cudaMalloc(&g_partial, MAX_BLOCK_NUM * sizeof(float));
    }

    stage1<<<blockNum, threadsPerBlock>>>(input, g_partial, N);
    stage2<<<1, 32>>>(g_partial, output, blockNum);
}
