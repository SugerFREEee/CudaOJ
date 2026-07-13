#include <cuda_runtime.h>
#define TILE_x 32
#define TILE_y 8

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float shared_input[TILE_x][TILE_x+1];
    int x = blockIdx.x *  TILE_x+ threadIdx.x;
    int y = blockIdx.y * TILE_x + threadIdx.y;

    for(int temp_y = 0; temp_y < TILE_x; temp_y += TILE_y) {
        if (x < cols && temp_y+y < rows) {
            shared_input[threadIdx.y+temp_y][threadIdx.x] = input[(temp_y+y)*cols + x];
        }
    }
    __syncthreads();


    // 找到block转置后的位置
    x = blockIdx.y * TILE_x + threadIdx.x;
    y = blockIdx.x * TILE_x + threadIdx.y;


    for(int temp_y = 0; temp_y < TILE_x; temp_y += TILE_y) {
        if (x < rows && temp_y+y < cols) {
            output[(temp_y+y)*rows + x] = shared_input[threadIdx.x][threadIdx.y+temp_y];
        }
    }
    
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TILE_x, TILE_y);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}