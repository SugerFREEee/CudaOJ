# Reduction — 优化笔记

## 题目
对长度 N 的 float 数组求和，结果写入 `output[0]`。纯 memory-bound。

## 两个版本
- `simple.cu`：单 kernel，grid-stride 局部和 → warp shuffle 归约 → block 归约 → `atomicAdd` 汇总。
- `optimal.cu`：float4 向量化 + 两阶段归约（stage1 出 block partial，stage2 合并）+ 复用 partial buffer。

## 优化点
1. **warp shuffle 归约**：`__shfl_down_sync` 在寄存器层面归约，无 shared bank conflict、warp 内天然同步，比纯 shared memory 树形归约快。
2. **两级归约**：先 warp 内，再 warp 间（block 内），最后跨 block。
3. **float4 向量化**：一次读 4 个 float，减少指令数和访存事务。
4. **两阶段 vs atomicAdd**：block 多时全局 `atomicAdd` 会成为热点且浮点顺序不定；两阶段用 stage2 单 block 合并 partial，去掉热点、结果更稳定。
5. **复用 partial buffer**：`g_partial` 跨调用复用，把 `cudaMalloc/cudaFree` 移出热路径（反复调用时收益明显）。
6. **block 数按 SM 数设定**：配合 grid-stride，既覆盖任意 N 又保证占用率。

## Corner cases 与针对性改法
- **N 极大（> ~5亿，index 溢出 int）**：`vec_num`、`stride`、索引改用 `size_t`；`blockNum` 计算避免 `(N+3)` 溢出。
- **N 不是 4 的倍数**：float4 主循环后必须有标量尾部循环。
- **超大 N 且要极致带宽**：可让每线程一次处理多个 float4（thread coarsening），进一步减少循环开销。
- **求和精度要求高**：float 累加有舍入误差，改用 Kahan 求和或 double 累加器；两阶段本身比单 atomic 顺序更可控。
- **N 特别小**：简单版单 kernel + atomic 启动开销更低，两阶段的额外 kernel launch 不划算。
- **partial 数超过 MAX_BLOCK_NUM**：当前 cap 到 128；若要更多 block，stage2 需改成能归约任意长度（grid-stride）或再加一层。
