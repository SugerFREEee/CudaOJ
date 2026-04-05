import torch
import triton
import triton.language as tl

@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bk, stride_bn, stride_cm, stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # 获取当前程序实例的 ID（2D 网格）
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # 计算当前块的起始位置
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 遍历 K 维度
    for k in range(0, K, BLOCK_SIZE_K):
        # 计算当前 K 块的起始位置
        block_start_k = k
        
        # 加载 A 矩阵的当前块
        a_ptr = a + block_start_m * stride_am + block_start_k * stride_an
        a_block = tl.load(a_ptr + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_an)
        
        # 加载 B 矩阵的当前块
        b_ptr = b + block_start_k * stride_bk + block_start_n * stride_bn
        b_block = tl.load(b_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_bn)
        
        # 计算点积并累加到 acc
        acc += tl.dot(a_block, b_block)
    
    # 计算 C 矩阵的当前块的起始位置
    c_ptr = c + block_start_m * stride_cm + block_start_n * stride_ck
    
    # 存储结果到 C 矩阵
    tl.store(c_ptr + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_cm + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_ck, acc)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    # 计算张量的步长
    stride_am, stride_an = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_ck = c.stride()

    # 定义块大小
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32

    # 计算网格大小（2D 网格）
    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N)
    )

    # 调用 kernel
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bk, stride_bn, stride_cm, stride_ck,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
