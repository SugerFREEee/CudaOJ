#include <cuda_runtime.h>

// 性能版：input tile + halo 进 shared memory，kernel 放 constant memory。
// 思路：一个 block 负责连续 BLOCK 个输出，需要 [out_base, out_base+BLOCK+kernel_size-2] 这段
// input（含 halo = kernel_size-1）。先协作把这段 input 一次性搬进 shared，
// 之后每个线程的点积全部从 shared 读，避免每个输出重复读全局 input。
// kernel 被所有输出复用，放 __constant__，warp 内同 j 读同地址是广播，效率高。
#define BLOCK 256
#define MAX_KERNEL 2048  // kernel_size <= 2047

__constant__ float c_kernel[MAX_KERNEL];

extern __shared__ float s_in[];  // 大小 = BLOCK + kernel_size - 1（动态 shared）

__global__ void conv1d_tiled(const float* input, float* output,
                             int input_size, int kernel_size) {
    int out_base = blockIdx.x * BLOCK;      // 本 block 第一个输出下标
    int tid = threadIdx.x;
    int tile = BLOCK + kernel_size - 1;     // 需要加载的 input 元素数（含 halo）

    // 协作加载 input tile 到 shared，越界补 0（valid 卷积不会真正用到越界，补 0 只为安全）
    for (int i = tid; i < tile; i += BLOCK) {
        int gidx = out_base + i;
        s_in[i] = (gidx < input_size) ? input[gidx] : 0.0f;
    }
    __syncthreads();

    int output_size = input_size - kernel_size + 1;
    int o = out_base + tid;
    if (o < output_size) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int j = 0; j < kernel_size; ++j) {
            sum += s_in[tid + j] * c_kernel[j];
        }
        output[o] = sum;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_size, int kernel_size) {
    if (input_size <= 0 || kernel_size <= 0 || kernel_size > input_size) return;

    // kernel 拷进 constant memory（device->symbol）
    cudaMemcpyToSymbol(c_kernel, kernel, (size_t)kernel_size * sizeof(float),
                       0, cudaMemcpyDeviceToDevice);

    int output_size = input_size - kernel_size + 1;
    int blocksPerGrid = (output_size + BLOCK - 1) / BLOCK;
    size_t sharedBytes = (size_t)(BLOCK + kernel_size - 1) * sizeof(float);

    conv1d_tiled<<<blocksPerGrid, BLOCK, sharedBytes>>>(input, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}
