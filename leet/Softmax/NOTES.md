# Softmax (1D) — 优化笔记

## 题目
对长度 N 的一维数组做 softmax，`output[i] = exp(x[i]-max)/sum`，用 max trick 防溢出。约束 N ≤ 500000，性能规模 N=500000。属于「M=1、N 很大」的多 block 协作场景。

## 两个版本
- `simple.cu`：单 block（1024 线程）三遍扫描——pass1 求 max，pass2 求 sum，pass3 归一化。无跨 block 同步，最好写。
- `optimal.cu`：多 block + online softmax（一遍同时求 max/sum）+ warp shuffle 两级归约 + 三 kernel（stage1 局部 / stage2 合并 / stage3 归一化）+ 复用 buffer。

## 优化点
1. **online softmax**：`d = d*exp(m_old-m_new) + exp(x-m_new)`，一遍同时维护 max 和 sum，把读输入从 3 遍降到 2 遍。
2. **warp shuffle 归约合并 (m,d)**：`__shfl_down_sync` 同时传 m 和 d，寄存器级归约。
3. **多 block 协作**：单数组用多 block 并行，两段式 kernel 完成跨 block 合并（stage2）。
4. **倒数乘法代替除法**：归一化预算 `inv = 1/sum` 再乘，省掉每元素一次除法。
5. **复用 g_max/g_sum buffer**：跨调用复用，避免热路径反复 cudaMalloc/cudaFree。
6. **哨兵用 -FLT_MAX 而非 -INFINITY**：online 合并里 `exp(m-m_new)`，两个 -INF 相减 = NaN；用有限极小值 `exp(0)=1` 不出 NaN。

## Corner cases 与针对性改法
- **超大 N（index 溢出 int）**：索引、stride 改 `size_t`。
- **N 极大且要极致带宽**：加 float4 向量化读输入/写输出（stage1/stage3），并按 SM 数配 grid。
- **想做到真正 one-pass（只读一遍输入）**：需 cooperative groups `grid.sync()` 在单 kernel 内跨 block 同步，把输入缓存在寄存器，归一化直接用寄存器值（仅当每线程元素数少时可行）。
- **全是相同值 / 全 -inf 输入**：max trick 保证 `x-max ≤ 0`，sum ≥ 1，不会除 0；simple 版单元素行 sum=exp(0)=1 正确。
- **N 很小**：simple 单 block 版启动开销更低，optimal 的三 kernel 不划算。
- **精度要求高**：sum 用 double 累加；float 版对 softmax 一般足够。
