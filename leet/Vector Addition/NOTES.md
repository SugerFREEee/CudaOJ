# Vector Addition — 优化笔记

## 题目
`C[i] = A[i] + B[i]`，A/B/C 均为 device pointer，长度 N。纯 memory-bound。

## 两个版本
- `simple.cu`：一线程一元素，`idx < N` 边界判断。面试手撕版。
- `optimal.cu`：float4 向量化 + grid-stride + 按 SM 数配 grid。

## 优化点
1. **float4 向量化**：一条指令搬 4 个 float，指令数和访存事务数降到 1/4。要求指针 16B 对齐（`cudaMalloc` 天然满足）。
2. **grid-stride loop**：固定 grid 规模覆盖任意 N，避免超大 N 时 grid 过大；block 数按 `SM数 × 倍数` 设定，保证占用率。
3. **`__restrict__`**：告诉编译器指针不重叠，利于访存重排和寄存器复用。
4. **尾部处理**：N 不是 4 的倍数时，用标量循环处理剩余 1~3 个元素。
5. 本质是带宽瓶颈，达到接近 HBM 峰值带宽即到顶，没有计算可优化。

## Corner cases 与针对性改法
- **N 极大（> INT_MAX / 4）**：`int` 索引会溢出。改用 `size_t` 计算索引和 n4，grid-stride 步长也用 `size_t`。
- **N 不是 4 的倍数**：必须有尾部标量循环，否则漏算最后几个元素。
- **指针未对齐**（如传入 `A+1`）：`reinterpret_cast<float4*>` 会触发未对齐访问。需先用标量处理前缀到 16B 对齐，再进 float4 主循环；或直接退回标量版。
- **N 很小（如 < 1000）**：float4 和多 block 收益不明显，简单版反而启动开销更低。
- **多次调用**：`cudaDeviceGetAttribute` 每次都查有开销，可缓存 SM 数到 static 变量。
