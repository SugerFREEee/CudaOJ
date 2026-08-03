#include <cuda_runtime.h>

// 最优版：float4 向量化 + grid-stride。
// vector add 是纯 memory-bound，优化核心是「用更少指令搬运相同数据 + 打满带宽」。
// - float4：一条指令读/写 4 个 float，减少指令数和访存事务数。
// - grid-stride：用固定规模的 grid 覆盖任意 N，block 数按 SM 数设定，保证占用率。
__global__ void vector_add_vec4(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C, int N) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int n4 = N / 4;
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    // 主循环：向量化处理 4 对元素
    for (int i = gtid; i < n4; i += stride) {
        float4 a = A4[i];
        float4 b = B4[i];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        C4[i] = c;
    }

    // 尾部：N 不是 4 的倍数时，处理剩余 1~3 个元素
    for (int i = n4 * 4 + gtid; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    if (N <= 0) return;

    int threadsPerBlock = 256;
    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);
    // 按 SM 数设定 grid，配合 grid-stride 覆盖任意 N。
    int blocksPerGrid = smCount > 0 ? smCount * 32 : (N + threadsPerBlock - 1) / threadsPerBlock;

    int n4 = N / 4;
    int need = (n4 > 0 ? n4 : N + threadsPerBlock - 1) / threadsPerBlock + 1;
    if (blocksPerGrid > need && need > 0) blocksPerGrid = need;
    if (blocksPerGrid < 1) blocksPerGrid = 1;

    vector_add_vec4<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
