#include <cuda_runtime.h>

// 简单版：一线程一元素，直接读写全局内存。
// 读 in[y*cols+x] 连续（合并），写 out[x*rows+y] stride=rows（非合并）——所以慢，但最好写。
__global__ void transpose_naive(const float* input, float* output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 列
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 行
    if (x < cols && y < rows) {
        output[(size_t)x * rows + y] = input[(size_t)y * cols + x];
    }
}

// input: rows x cols, output: cols x rows, all row-major
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);
    transpose_naive<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
