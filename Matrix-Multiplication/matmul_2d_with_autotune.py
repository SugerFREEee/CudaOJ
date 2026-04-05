import torch
import triton
import triton.language as tl
from utils import get_2d_autotune_config

# Roughly speaking, the kernel that we will write will implement the following blocked
# algorithm to multiply a (M, K) by a (K, N) matrix:
#
#  .. code-block:: python
#
#    # Do in parallel
#    for m in range(0, M, BLOCK_SIZE_M):
#      # Do in parallel
#      for n in range(0, N, BLOCK_SIZE_N):
#        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
#        for k in range(0, K, BLOCK_SIZE_K):
#          a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
#          b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
#          acc += dot(a, b)
#        C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
#
# where each iteration of the doubly-nested for-loop is performed by a dedicated Triton program instance.
@triton.autotune(
    configs=get_2d_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # 获取当前程序实例的 ID
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

    # 计算网格大小（使用 lambda 函数动态计算，以便与自动调优的块大小匹配）
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N'])
    )

    # 调用 kernel（不再显式传递块大小参数，由自动调优机制处理）
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck
    )



