#include <cuda_runtime.h>

// 性能版 2：thread coarsening —— 每个线程算 COARSE 个输出，把加载进 shared 的 tile
// 摊到更多输出上复用，摊薄 halo 的加载开销（kernel 越大、halo 越大，收益越明显）。
// 一个 block 负责 BLOCK*COARSE 个连续输出，tile = BLOCK*COARSE + kernel_size - 1。
// kernel 放 constant memory；input tile(含 halo) 放 dynamic shared。
#define BLOCK 256
#define COARSE 4
#define MAX_KERNEL 2048  // kernel_size <= 2047

__constant__ float c_kernel[MAX_KERNEL];

extern __shared__ float s_in[];  // 大小 = BLOCK*COARSE + kernel_size - 1

__global__ void conv1d_coarsen(const float* input, float* output,
                               int input_size, int kernel_size) {
    int outs_per_block = BLOCK * COARSE;
    int out_base = blockIdx.x * outs_per_block;   // 本 block 第一个输出下标
    int tid = threadIdx.x;
    int tile = outs_per_block + kernel_size - 1;   // 核心区 + halo

    // 协作加载 input tile(含 halo) 到 shared
    for (int i = tid; i < tile; i += BLOCK) {
        int gidx = out_base + i;
        s_in[i] = (gidx < input_size) ? input[gidx] : 0.0f;
    }
    __syncthreads();

    int output_size = input_size - kernel_size + 1;

    // 每个线程算 COARSE 个输出；相邻线程在同一 c 上写相邻 output，保持合并访存
    #pragma unroll
    for (int c = 0; c < COARSE; ++c) {
        int local = tid + c * BLOCK;      // 在 block tile 内的输出偏移
        int o = out_base + local;
        if (o < output_size) {
            float sum = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < kernel_size; ++j) {
                sum += s_in[local + j] * c_kernel[j];
            }
            output[o] = sum;
        }
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_size, int kernel_size) {
    if (input_size <= 0 || kernel_size <= 0 || kernel_size > input_size) return;

    cudaMemcpyToSymbol(c_kernel, kernel, (size_t)kernel_size * sizeof(float),
                       0, cudaMemcpyDeviceToDevice);

    int output_size = input_size - kernel_size + 1;
    int outs_per_block = BLOCK * COARSE;
    int blocksPerGrid = (output_size + outs_per_block - 1) / outs_per_block;
    size_t sharedBytes = (size_t)(outs_per_block + kernel_size - 1) * sizeof(float);

    conv1d_coarsen<<<blocksPerGrid, BLOCK, sharedBytes>>>(input, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}
