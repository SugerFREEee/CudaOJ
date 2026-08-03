#include <cuda_runtime.h>

// 最优版：128x128 block tile + 8x8 register tile + float4 向量化 + Global->Shared 双缓冲(ping-pong)。
// C = A×B, A:M×N, B:N×K, C:M×K, row-major。
// 核心：外积式寄存器分块把 shared 读复用最大化；双缓冲让下一 tile 的加载与本 tile 计算重叠。
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

#define FLOAT4(p)  (reinterpret_cast<float4*>(&(p))[0])
#define CFLOAT4(p) (reinterpret_cast<const float4*>(&(p))[0])

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    __shared__ float As[2][BK][BM];   // 双缓冲，A 转置存放
    __shared__ float Bs[2][BK][BN];

    int tid = threadIdx.x;
    int tx  = tid % 16;
    int ty  = tid / 16;

    float c_frag[TM][TN] = {0.0f};
    float a_frag[TM], b_frag[TN];

    int a_row = tid / 2;         // M
    int a_col = (tid % 2) * 4;   // N
    int b_row = tid / 32;        // N
    int b_col = (tid % 32) * 4;  // K

    bool aAligned = (N % 4 == 0);
    bool bAligned = (K % 4 == 0);

    // ---- Prologue: 加载 tile k0=0 到 buffer 0 ----
    {
        int gr = blockRow + a_row, gc = a_col;
        float4 v = {0.f, 0.f, 0.f, 0.f};
        if (gr < M) {
            if (aAligned && gc + 3 < N) v = CFLOAT4(A[gr * N + gc]);
            else { if (gc<N) v.x=A[gr*N+gc]; if (gc+1<N) v.y=A[gr*N+gc+1];
                   if (gc+2<N) v.z=A[gr*N+gc+2]; if (gc+3<N) v.w=A[gr*N+gc+3]; }
        }
        As[0][a_col+0][a_row]=v.x; As[0][a_col+1][a_row]=v.y;
        As[0][a_col+2][a_row]=v.z; As[0][a_col+3][a_row]=v.w;

        int gr2 = b_row, gc2 = blockCol + b_col;
        float4 w = {0.f, 0.f, 0.f, 0.f};
        if (gr2 < N) {
            if (bAligned && gc2 + 3 < K) w = CFLOAT4(B[gr2 * K + gc2]);
            else { if (gc2<K) w.x=B[gr2*K+gc2]; if (gc2+1<K) w.y=B[gr2*K+gc2+1];
                   if (gc2+2<K) w.z=B[gr2*K+gc2+2]; if (gc2+3<K) w.w=B[gr2*K+gc2+3]; }
        }
        FLOAT4(Bs[0][b_row][b_col]) = w;
    }
    __syncthreads();

    int buf = 0;
    for (int k0 = BK; k0 < N + BK; k0 += BK) {   // 本轮计算 tile (k0-BK)
        int next = buf ^ 1;

        // 1) 预取下一 tile (列偏移 k0) 到寄存器
        float4 va = {0.f, 0.f, 0.f, 0.f};
        float4 vb = {0.f, 0.f, 0.f, 0.f};
        if (k0 < N) {
            int gr = blockRow + a_row, gc = k0 + a_col;
            if (gr < M) {
                if (aAligned && gc + 3 < N) va = CFLOAT4(A[gr * N + gc]);
                else { if (gc<N) va.x=A[gr*N+gc]; if (gc+1<N) va.y=A[gr*N+gc+1];
                       if (gc+2<N) va.z=A[gr*N+gc+2]; if (gc+3<N) va.w=A[gr*N+gc+3]; }
            }
            int gr2 = k0 + b_row, gc2 = blockCol + b_col;
            if (gr2 < N) {
                if (bAligned && gc2 + 3 < K) vb = CFLOAT4(B[gr2 * K + gc2]);
                else { if (gc2<K) vb.x=B[gr2*K+gc2]; if (gc2+1<K) vb.y=B[gr2*K+gc2+1];
                       if (gc2+2<K) vb.z=B[gr2*K+gc2+2]; if (gc2+3<K) vb.w=B[gr2*K+gc2+3]; }
            }
        }

        // 2) 计算当前 buffer
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            FLOAT4(a_frag[0]) = FLOAT4(As[buf][kk][ty * 4]);
            FLOAT4(a_frag[4]) = FLOAT4(As[buf][kk][ty * 4 + 64]);
            FLOAT4(b_frag[0]) = FLOAT4(Bs[buf][kk][tx * 4]);
            FLOAT4(b_frag[4]) = FLOAT4(Bs[buf][kk][tx * 4 + 64]);
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    c_frag[i][j] += a_frag[i] * b_frag[j];
        }

        // 3) 暂存写入 next buffer 并切换
        if (k0 < N) {
            As[next][a_col+0][a_row]=va.x; As[next][a_col+1][a_row]=va.y;
            As[next][a_col+2][a_row]=va.z; As[next][a_col+3][a_row]=va.w;
            FLOAT4(Bs[next][b_row][b_col]) = vb;
            __syncthreads();
            buf = next;
        }
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int mrow = blockRow + (i < 4 ? ty * 4 + i : ty * 4 + 64 + (i - 4));
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int kcol = blockCol + (j < 4 ? tx * 4 + j : tx * 4 + 64 + (j - 4));
            if (mrow < M && kcol < K)
                C[mrow * K + kcol] = c_frag[i][j];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
// A: M x N, B: N x K, C: M x K, all row-major
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((K + BN - 1) / BN,
                       (M + BM - 1) / BM);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
