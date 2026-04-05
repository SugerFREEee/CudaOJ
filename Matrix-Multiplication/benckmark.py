import torch
import triton
import triton.testing
# 导入带autotune的实现
from matmul_2d_with_autotune import solve as solve_2d_with_autotune
from matmul_1d_with_autotune import solve as solve_1d_with_autotune
from matmul_l2_with_autotune import solve as solve_l2_with_autotune
# 导入不带autotune的实现
from matmul_1d_no_autotune import solve as solve_1d_no_autotune
from matmul_2d_no_autotune import solve as solve_2d_no_autotune
from matmul_l2_no_autotune import solve as solve_l2_no_autotune
# 设置设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# 矩阵乘法函数包装
def matmul_2d_with_autotune(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=DEVICE, dtype=a.dtype)
    solve_2d_with_autotune(a, b, c, M, N, K)
    return c

def matmul_1d_with_autotune(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=DEVICE, dtype=a.dtype)
    solve_1d_with_autotune(a, b, c, M, N, K)
    return c

def matmul_l2_with_autotune(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=DEVICE, dtype=a.dtype)
    solve_l2_with_autotune(a, b, c, M, N, K)
    return c

def matmul_1d_no_autotune(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=DEVICE, dtype=a.dtype)
    solve_1d_no_autotune(a, b, c, M, N, K)
    return c

def matmul_2d_no_autotune(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=DEVICE, dtype=a.dtype)
    solve_2d_no_autotune(a, b, c, M, N, K)
    return c

def matmul_l2_no_autotune(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=DEVICE, dtype=a.dtype)
    solve_l2_no_autotune(a, b, c, M, N, K)
    return c

# 基准测试
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # 作为 x 轴的参数名
        x_vals=[2**i for i in range(8, 15, 1)],  # 不同的矩阵大小
        x_log=True,  # x 轴是对数刻度
        line_arg='provider',  # 不同线条的参数名
        line_vals=['torch', '1d_with_autotune', '2d_with_autotune', 'l2_with_autotune', '1d_no_autotune', '2d_no_autotune', 'l2_no_autotune'],  # 不同的实现
        line_names=['Torch', '1D Triton (with autotune)', '2D Triton (with autotune)', 'L2 Optimized Triton (with autotune)', '1D Triton (no autotune)', '2D Triton (no autotune)', 'L2 Optimized Triton (no autotune)'],  # 线条标签
        styles=[('green', '-'), ('cyan', '-'), ('blue', '-'), ('red', '-'), ('purple', '-'), ('orange', '-'), ('brown', '-')],  # 线条样式
        ylabel='GFLOPS',  # y 轴标签
        plot_name='matrix-multiplication-performance',  # 图表名称
        args={},  # 其他固定参数
    )
)
def benchmark(size, provider):
    # 创建随机矩阵
    M = size
    N = size
    K = size
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float32)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    
    # 选择不同的实现
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    elif provider == '1d_with_autotune':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_1d_with_autotune(a, b), quantiles=quantiles)
    elif provider == '2d_with_autotune':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_2d_with_autotune(a, b), quantiles=quantiles)
    elif provider == 'l2_with_autotune':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_l2_with_autotune(a, b), quantiles=quantiles)
    elif provider == '1d_no_autotune':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_1d_no_autotune(a, b), quantiles=quantiles)
    elif provider == '2d_no_autotune':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_2d_no_autotune(a, b), quantiles=quantiles)
    elif provider == 'l2_no_autotune':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_l2_no_autotune(a, b), quantiles=quantiles)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    # 计算 GFLOPS
    flops = 2 * M * N * K
    gflops = lambda ms: flops / ms * 1e-6
    return gflops(ms), gflops(max_ms), gflops(min_ms)

if __name__ == '__main__':
    # 运行基准测试
    benchmark.run(print_data=True, show_plots=False, save_path='./benchmark_results')
