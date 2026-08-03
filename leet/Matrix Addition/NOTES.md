# Matrix Addition — 优化笔记

## 题目
两个 N×N 矩阵逐元素相加 `C = A + B`，row-major。约束 1 ≤ N ≤ 4096，性能规模 N=4096。
本质是 N*N 个元素的逐元素加法，等价于一个长度 N² 的 vector add，纯 memory-bound。

## 签名
```cpp
extern "C" void solve(const float* A, const float* B, float* C, int N);  // A,B,C 均为 N x N row-major
```

## 优化点（与 Vector Addition 一致）
1. **按 1D 展平处理**：N×N 连续存储，直接当长度 `N*N` 的一维数组处理，避免 2D 索引开销。
2. **float4 向量化**：一条指令读/写 4 个 float，指令数和访存事务降到 1/4（要求元素数是 4 的倍数或做尾部处理；N×N 中 N 为偶数时 N² 是 4 的倍数）。
3. **grid-stride loop**：固定 grid（按 SM 数）覆盖任意 N²，保证占用率。
4. **`__restrict__`**：指针不重叠，利于访存重排。
5. 纯带宽瓶颈，达到接近 HBM 峰值带宽即到顶。

## Corner cases 与针对性改法
- **N 极大导致 N*N 溢出 int**：N=4096 时 N²=16.7M 仍在 int 范围；但更大规模要用 `size_t` 计算总元素数和索引。
- **N² 不是 4 的倍数**（N 为奇数，如 1023）：float4 主循环后必须有标量尾部循环。
- **指针未 16B 对齐**：`cudaMalloc` 天然对齐；若传入偏移指针需先处理对齐前缀。
- **N 很小（如 2、1）**：float4 和多 block 收益不明显，naive 一线程一元素启动开销更低。
- **非方阵 / 只给一个维度的变体**：若题目实际是 rows×cols，把总数换成 `rows*cols` 即可，逻辑不变。

> 注：本目录仅提供题目模板 `solve.cu` 与 `test.cu`/`benchmark.cu`，解题代码留空。
