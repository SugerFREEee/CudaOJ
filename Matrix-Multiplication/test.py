import torch
import triton
# from regular_kernel import solve
from L2_optimatized_kernel import solve


# 设置设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# 测试函数
def test_matrix_multiplication():
    # 测试 float32 输入
    print("\nTesting with float32 inputs:")
    torch.manual_seed(0)
    M, N, K = 512, 512, 512
    a = torch.rand((M, K), device=DEVICE, dtype=torch.float32) - 0.5
    b = torch.rand((K, N), device=DEVICE, dtype=torch.float32) - 0.5
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float32)
    
    # 调用 Triton 实现
    solve(a, b, c, M, N, K)
    triton_output = c
    
    # 调用 PyTorch 实现
    torch_output = torch.matmul(a, b)
    
    print(f"triton_output shape: {triton_output.shape}")
    print(f"torch_output shape: {torch_output.shape}")
    
    # 比较结果
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match for float32")
    else:
        print("❌ Triton and Torch differ for float32")
        max_diff = torch.max(torch.abs(triton_output - torch_output))
        print(f"Maximum difference: {max_diff}")
    
    # 测试 float16 输入
    print("\nTesting with float16 inputs:")
    torch.manual_seed(0)
    a = torch.rand((M, K), device=DEVICE, dtype=torch.float16) - 0.5
    b = torch.rand((K, N), device=DEVICE, dtype=torch.float16) - 0.5
    c = torch.empty((M, N), device=DEVICE, dtype=torch.float16)
    
    # 调用 Triton 实现
    solve(a, b, c, M, N, K)
    triton_output = c
    
    # 调用 PyTorch 实现
    torch_output = torch.matmul(a, b)
    
    print(f"triton_output shape: {triton_output.shape}")
    print(f"torch_output shape: {torch_output.shape}")
    
    # 比较结果（float16 精度较低，使用较大的容差）
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match for float16")
    else:
        print("❌ Triton and Torch differ for float16")
        max_diff = torch.max(torch.abs(triton_output - torch_output))
        print(f"Maximum difference: {max_diff}")
    
    # 测试不同大小的矩阵
    print("\nTesting with different matrix sizes:")
    test_sizes = [(128, 128, 128), (256, 256, 256), (1024, 1024, 1024)]
    
    for size in test_sizes:
        M, N, K = size
        print(f"\nTesting with matrix size: {M}x{K} * {K}x{N}")
        
        torch.manual_seed(0)
        a = torch.rand((M, K), device=DEVICE, dtype=torch.float32)
        b = torch.rand((K, N), device=DEVICE, dtype=torch.float32)
        c = torch.empty((M, N), device=DEVICE, dtype=torch.float32)
        
        # 调用 Triton 实现
        solve(a, b, c, M, N, K)
        triton_output = c
        
        # 调用 PyTorch 实现
        torch_output = torch.matmul(a, b)
        
        # 比较结果
        if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
            print(f"✅ Triton and Torch match for {M}x{K} * {K}x{N}")
        else:
            print(f"❌ Triton and Torch differ for {M}x{K} * {K}x{N}")
            max_diff = torch.max(torch.abs(triton_output - torch_output))
            print(f"Maximum difference: {max_diff}")

if __name__ == "__main__":
    test_matrix_multiplication()
