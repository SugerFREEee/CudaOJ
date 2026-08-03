# Matrix Multiplication (FP32 SGEMM) — 优化笔记

## 题目
`C = A × B`，A:M×N，B:N×K，C:M×K，全部 row-major，FP32。性能规模 M=8192, N=6144, K=4096。compute-bound。

## 两个版本
- `simple.cu`：shared memory tiling，每线程算 1 个 C 元素（BM=BN=16, BK=32）。面试手撕首选。
- `optimal.cu`：128×128 block tile + 8×8 register tile + float4 向量化 + Global→Shared 双缓冲(ping-pong)。

## 优化点（从 simple 到 optimal 的演进）
1. **shared tiling**：把 global 的重复访问换成 shared，降低全局访存量（simple 已做）。
2. **register tiling（每线程 8×8）**：每线程算 64 个输出，外积复用寄存器里的 a_frag/b_frag，把算术强度从 ~4 提到 ~32 FLOP/byte，摆脱 MIO throttle。
3. **float4 向量化**：Global→Shared 的加载和 shared→register 的读取都用 float4，减少指令数。
4. **A 转置存 shared（As[BK][BM]）**：让计算阶段按列取 A 也能连续/float4 读。
5. **双缓冲 ping-pong**：`As[2]/Bs[2]`，预取下一个 K-tile 到寄存器再写另一 buffer，加载与计算重叠，隐藏全局访存延迟。
6. **归约维分块（BK=8）**：平衡 shared 容量与复用。

## Corner cases 与针对性改法
- **N 或 K 不是 4 的倍数**：optimal 用 `aAligned/bAligned` 运行时判断，未对齐/边界走标量加载，保证正确。
- **M/N/K 不是 tile(128) 的倍数**：加载和写回都有边界判断（`gr<M` 等），非整除也对；边缘 block 部分线程空转。
- **超大 M（如 8192+）**：grid.y 可能很大但仍 < 2^31，没问题；若 M×K 索引超 int，`C[mrow*K+kcol]` 需改 `size_t`（当前是 int，超大时要注意）。
- **超大 K（宽输出）**：grid.x 增大；同上注意索引类型。
- **N 很大（归约维长）**：K-loop 变长，双缓冲收益更明显；可增大 BK 或每线程再多算。
- **想进一步逼近 cuBLAS**：warp-level tiling（4×8 warp 布局最大化 shared broadcast）、bank-conflict-free 的 swizzle 布局、`cp.async` 异步拷贝（Ampere+）、Tensor Core（若可用低精度）。
- **小矩阵**：128×128 tile 会让大量 block 空转，simple 版或更小 tile 更合适。
