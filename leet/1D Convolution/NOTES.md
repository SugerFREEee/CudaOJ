# 1D Convolution — 优化笔记

## 题目
1D "valid" 卷积：`output[i] = sum_{j=0}^{kernel_size-1} input[i+j] * kernel[j]`，
`output_size = input_size - kernel_size + 1`（kernel 只在完全覆盖 input 处输出）。
约束 input_size ≤ 1.5M，kernel_size ≤ 2047，性能规模 1.5M / 2047。
注意：这里按「滑动点积」定义（不翻转 kernel），CPU 参考实现与之一致。

## 签名
```cpp
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_size, int kernel_size);
```

## 优化点
1. **kernel 放 constant memory 或 shared memory**：kernel 被所有输出位置复用，且 kernel_size ≤ 2047（≤ 8KB，可放 `__constant__` 上限 64KB 内）。constant memory 广播读对「同一 warp 读同一 kernel[j]」非常友好。
2. **input tile 进 shared memory（halo）**：一个 block 计算 BLOCK 个输出，需要 `BLOCK + kernel_size - 1` 个连续 input（含 halo）。先协作加载到 shared，再每线程从 shared 做点积，避免每个输出重复读全局 input。
3. **合并访存**：相邻线程算相邻 output，读 input 连续、写 output 连续。
4. **寄存器累加 + 循环展开**：内层对 kernel 的点积用 `#pragma unroll`（kernel_size 大时分段展开）。
5. **每线程算多个输出（thread coarsening）**：进一步复用 shared 里的 input tile。

## Corner cases 与针对性改法
- **kernel_size = 1**：退化成逐元素乘（output = input * kernel[0]），shared/constant 依然适用。
- **kernel_size = input_size**：output_size = 1，只有一个输出；可用一个 block 做归约点积。
- **kernel_size 很大（接近 2047）**：`BLOCK + kernel_size - 1` 可能超过 shared 容量（48KB/block ≈ 12K floats）。BLOCK=256、kernel=2047 时 tile≈2302 floats≈9KB，可行；若 kernel 更大需减小 BLOCK 或分段。
- **input_size 极大（1.5M）**：output ~1.5M，注意用 `size_t`/`long` 计算总量；grid 用 grid-stride 或足够 block 覆盖。
- **kernel_size > input_size**：output_size ≤ 0，非法输入，应提前判断返回。
- **精度**：kernel_size 大时点积累加误差累积，CPU 参考用 double；GPU 端 float 累加一般够，必要时用 double 或 Kahan。
- **constant memory 限制**：`__constant__` 是编译期固定大小数组；kernel_size 运行时可变，需按上限 2047 声明 `__constant__ float c_kernel[2048];` 再 `cudaMemcpyToSymbol`。

> 注：本目录仅提供题目模板 `solve.cu` 与 `test.cu`/`benchmark.cu`，解题代码留空。
