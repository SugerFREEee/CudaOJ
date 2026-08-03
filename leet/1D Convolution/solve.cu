#include <cuda_runtime.h>
#define TILE 2047
#define MAX_BLOCK_NUM 1024
#define WARP_NUM 8
#define THREAD_NUM (32 * WARP_NUM)

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
        __shared__ float K[TILE];
        
        int gtid = blockDim.x * blockIdx.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for(int i = threadIdx.x; i<kernel_size;i += blockDim.x)
            K[i] = kernel[i];

        __syncthreads();

        for(int i = gtid; i<input_size - kernel_size + 1;i += stride)
        {
            float sum = 0.0f;
            for(int j = 0;j<kernel_size;j++)
            {
                sum += K[j] * input[i+j];
            }
            output[i] = sum;
        }
        
            
    }

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = THREAD_NUM;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    blocksPerGrid = min(blocksPerGrid, MAX_BLOCK_NUM);

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}
