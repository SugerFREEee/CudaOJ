# Matrix Transpose — 优化笔记

## 题目
把 rows×cols 矩阵转置成 cols×rows（row-major）。纯 memory-bound，无计算，性能完全取决于访存效率。性能规模 rows=7000, cols=6000。

## 两个版本
- `simple.cu`：一线程一元素，直接 `out[x*rows+y]=in[y*cols+x]`。读合并、写非合并（stride=rows），慢但最好写。
- `optimal.cu`：shared memory tile 中转 + padding 消除 bank conflict，读写都合并。

## 优化点
1. **shared 中转**：把「非合并写全局」变成「合并读入 shared → shared 内换行列 → 合并写出全局」。全局访存全部合并。
2. **tile padding `[32][32+1]`**：按列读 shared 时，一行 32 元素正好等于 32 个 bank，会 32-way conflict；加 1 列 padding 让每行错开 1 个 bank，冲突消除。
3. **矩形 block（32×8）+ 每线程处理多元素**：block 只需 256 线程即可覆盖 32×32 tile，循环 `TILE_x/TILE_y=4` 次；提升占用率、减少 block 数。
4. **合并读 / 合并写**：warp 内连续线程访问连续全局地址。

## Corner cases 与针对性改法
- **rows/cols 不是 tile(32) 的倍数**：kernel 里有 `x<cols`、`temp_y+y<rows` 边界判断，非整除也正确；只是边缘 tile 有线程空转。
- **超大矩阵（rows×cols 超过 int）**：全局索引 `(size_t)row*cols` 已用 size_t 防溢出；grid 维度也要确认不超过上限（x/y 维 < 2^31）。
- **方阵 in-place 转置**：本实现是 out-of-place；若要 in-place 需按对角线分块交换，逻辑不同。
- **想再压带宽**：可用 float4 向量化读写（要求 cols 是 4 的倍数且行对齐），或用 swizzle 布局进一步优化 bank。
- **极端长条矩阵（如 1×N 或 N×1）**：本质是内存拷贝，tile 方案收益低，直接 memcpy/naive 即可。
- **bank conflict 验证**：把 padding 去掉（`[32][32]`）用 Nsight Compute 看 shared load conflict，能直观对比 padding 效果。
