# GEMM (half + FP32 累加) — 优化笔记

## 题目
`C = alpha*A*B + beta*C`，A/B/C 为 half（FP16），A:M×K，B:K×N，C:M×N，row-major。累加用 FP32，结果转回 half。alpha/beta 为 float。只允许 WMMA，不许其他外部库。性能规模 M=N=K=1024。

## 两个版本
- `simple.cu`：一线程一元素，FP32 累加，写回 half。面试手撕首选。
- `optimal.cu`：WMMA / Tensor Core，16×16×16 tile，FP32 累加，alpha/beta 融合写回；非 16 倍数尺寸退回标量 kernel 保证正确。

## 编译
optimal 用 WMMA，必须 `-arch=sm_70` 及以上：
```
nvcc -arch=sm_80 test.cu -o gemm_test && ./gemm_test
nvcc -arch=sm_80 -O3 -DSOLVE_FILE='"optimal.cu"' benchmark.cu -o gemm_bench && ./gemm_bench
```

## 优化点
1. **Tensor Core（WMMA）**：`load_matrix_sync / mma_sync / store_matrix_sync`，一条 mma 指令完成 16×16×16 的矩阵乘累加，吞吐远高于 CUDA core。
2. **FP32 累加**：accumulator fragment 是 float，避免 half 累加的精度损失（题目硬性要求）。
3. **alpha/beta 融合**：累加结果先落 shared float tile，再逐元素 `alpha*acc+beta*C` 写回 half，一次 kernel 完成。
4. **一 warp 一 tile**：每个 block（32 线程）算一个 16×16 输出 tile，K 维步进 16。

## Corner cases 与针对性改法
- **M/N/K 非 16 倍数**（如 33×50×47、4×4×4）：WMMA 只能整 tile 操作，optimal 里用 `M%16==0 && N%16==0 && K%16==0` 判断，不满足就退回标量 kernel。若想 WMMA 也覆盖，需把矩阵 pad 到 16 的倍数（额外 buffer + 拷贝）。
- **超大 M/N/K（如 4096）**：当前一 warp 一 tile，tile 太多、每 warp 只算一个 16×16 会受启动/占用限制。应让一个 block 多个 warp 各算多个 tile（warp tiling），并把 A/B tile 先搬进 shared 再喂 WMMA，复用数据、减少全局访存。
- **超大 K（归约维很长）**：K-loop 变长；可用多 warp 沿 K 分段 + shared 累加，或 split-K 后再规约。
- **beta=0**：可跳过读 C，直接写 alpha*acc，省一次读。
- **精度**：half 输入本身精度有限，测试容差要放宽（相对 5e-2 级别）；累加务必 FP32。
- **对齐**：WMMA 的 `load_matrix_sync` 对 leading dimension 有对齐/倍数要求，pad 时注意 ldm 仍是 16 的倍数。
- **想接近 cuBLAS/CUTLASS**：多级 tiling（block/warp/wmma 三级）、shared 双缓冲、`cp.async` 异步加载、swizzle 消除 bank conflict。
