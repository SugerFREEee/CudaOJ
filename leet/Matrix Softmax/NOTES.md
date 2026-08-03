# Matrix Softmax (行 softmax) — 优化笔记

## 题目
对 M×N 矩阵（row-major）**按行**做 softmax：每行独立 `y[r][j]=exp(x[r][j]-max_r)/sum_r`。性能规模 M=N=4096。

## 两个版本
- `simple.cu`：一个 block 处理一行，行内三遍扫描 + shared 树形归约。每行独立、无跨 block 同步，最好写。
- `optimal.cu`：一个 block 处理一行 + online softmax（一遍求 max/sum）+ warp shuffle 两级归约 + float4 向量化 + grid-stride 遍历行。

## 优化点
1. **一 block 一行**：行是天然的并行/归约单元，规约都在 block 内完成，不需要跨 block 同步。
2. **online softmax**：一遍同时求 max 和 sum，读输入从 3 遍降到 2 遍。
3. **warp shuffle 两级归约**：warp 内 + warp 间，寄存器级，快且无 bank conflict。
4. **float4 向量化**：读输入、写输出各减少 4 倍指令/事务。
5. **grid-stride 遍历行**：grid 固定为 `SM数×倍数`，M 很大时避免过多 block 的调度开销，且 block/shared 状态可跨行复用；M 小时退化为一行一 block。
6. **倒数乘法代替除法**。
7. **哨兵用 -FLT_MAX**（避免 online 合并里 -INF 相减出 NaN）。

## Corner cases 与针对性改法
- **N 不是 4 的倍数**：非首行的行起始不是 16B 对齐，`float4` 会触发未对齐访问。optimal 里用 `N4=(N%4==0)?N/4:0` 门控，不对齐就整体走标量。若要对不对齐也向量化，需按行做对齐前缀处理。
- **超大 N（一行非常长，如 N≥100K）**：一个 block 的线程数有限，行内循环变长；可考虑「多 block 协作处理一行」（行内分段 + 两段式/atomic 合并），类似 1D softmax 的做法。
- **超大 M（几百万行）**：grid-stride 已能覆盖；注意行偏移 `row*N` 用 `size_t` 防 int 溢出。
- **N=1（单列）**：每行仅一个元素，输出恒为 1；simple/optimal 都正确（sum=exp(0)=1）。
- **想做 one-pass**：每行元素少时可把行数据缓存进寄存器，省第二遍读取（doc V3 思路）。
- **接 Attention 场景**：不要单独跑 softmax，应融合进 FlashAttention（online softmax 增量更新 O），避免落地 N×N 矩阵。
