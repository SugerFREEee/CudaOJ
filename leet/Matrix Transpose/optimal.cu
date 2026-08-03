#include <cuda_runtime.h>

// 最优版：shared memory tile 中转 + padding 消除 bank conflict。
// 核心：把全局内存的「非合并写」变成「shared 中转 + 合并读写」。
// - 合并读：一个 warp 连续读 input 一行的连续列。
// - shared tile [TILE_x][TILE_x+1]：+1 padding 让按列读 shared 时错开 bank，避免 32-way conflict。
// - 合并写：转置后连续线程写 output 连续地址。
#define TILE_x 32
#define TILE_y 8

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float shared_input[TILE_x][TILE_x + 1];
    int x = blockIdx.x * TILE_x + threadIdx.x;
    int y = blockIdx.y * TILE_x + threadIdx.y;

    // 合并读入 tile
    for (int temp_y = 0; temp_y < TILE_x; temp_y += TILE_y) {
        if (x < cols && temp_y + y < rows) {
            shared_input[threadIdx.y + temp_y][threadIdx.x] = input[(size_t)(temp_y + y) * cols + x];
        }
    }
    __syncthreads();

    // 找到 block 转置后的位置
    x = blockIdx.y * TILE_x + threadIdx.x;
    y = blockIdx.x * TILE_x + threadIdx.y;

    // 从 shared 转置读 + 合并写出
    for (int temp_y = 0; temp_y < TILE_x; temp_y += TILE_y) {
        if (x < rows && temp_y + y < cols) {
            output[(size_t)(temp_y + y) * rows + x] = shared_input[threadIdx.x][threadIdx.y + temp_y];
        }
    }
}

// input: rows x cols, output: cols x rows, all row-major
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TILE_x, TILE_y);
    dim3 blocksPerGrid((cols + TILE_x - 1) / TILE_x,
                       (rows + TILE_x - 1) / TILE_x);
    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
