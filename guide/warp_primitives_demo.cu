// 编译方式：nvcc -arch=sm_80 warp_primitives_demo.cu -o warp_primitives_demo
// 说明：__match_any_sync 和 __match_all_sync 需要 Volta+ 架构目标；本机 A800 对应 sm_80。
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                            \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;  // 32 个 bit 全为 1，表示一个 warp 的 32 个 lane 都参与。

struct LaneResult {
    int lane;
    int original;
    int shfl_from_lane0;
    int shfl_up_1;
    int shfl_down_1;
    int shfl_xor_1;
    int shfl_width_8_down_1;
    int pred_even;
    int all_even;
    int any_even;
    unsigned ballot_even;
    unsigned active_mask_before_branch;
    unsigned active_mask_inside_branch;
    unsigned match_any_key_mask;
    unsigned match_all_same_mask;
    int match_all_same_pred;
    unsigned match_all_mixed_mask;
    int match_all_mixed_pred;
    int warp_reduce_sum_lane0;
    int subwarp_width8_sum;
    int syncwarp_shared_value;
};

// 用 __shfl_down_sync 做 warp 内求和。
// 思路：lane 0 先拿 lane 16 的值，再拿 lane 8/4/2/1 的值，最终 lane 0 拥有 32 个 lane 的总和。
// 注意：这个函数返回时，只有 lane 0 的结果是完整总和；其他 lane 拿到的是部分和。
__device__ int warpReduceSum(int value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullMask, value, offset);
    }
    return value;
}

// width=8 表示把一个 32-lane warp 逻辑切成 4 个 8-lane 子组：0~7、8~15、16~23、24~31。
// 每个子组内部独立做 shfl_down，不会跨到相邻子组取值。
__device__ int subWarpWidth8ReduceSum(int value) {
    for (int offset = 4; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullMask, value, offset, 8);
    }
    return value;
}

__global__ void warpPrimitiveDemoKernel(LaneResult *out) {
    __shared__ int shared_value;

    int lane = threadIdx.x & 31;   // lane 是线程在当前 warp 内的编号，范围 0~31。
    int value = lane + 100;        // 给每个 lane 一个容易观察的寄存器值：100, 101, 102, ...。
    int pred_even = (lane % 2 == 0);
    int key = lane / 8;            // 每 8 个 lane 分到同一个 key，用来演示 match_any 分组。

    LaneResult result{};
    result.lane = lane;
    result.original = value;

    // __shfl_sync(mask, value, srcLane)：所有参与线程都读取 srcLane 线程的 value。
    // 这里 srcLane=0，所以 32 个 lane 读到的都是 lane 0 的 value，也就是 100。
    result.shfl_from_lane0 = __shfl_sync(kFullMask, value, 0);

    // __shfl_up_sync(mask, value, delta)：从更低编号 lane 读取，源 lane = lane - delta。
    // lane 3 会读 lane 2 的 value；lane 0 没有 lane -1，所以保留自己的 value。
    result.shfl_up_1 = __shfl_up_sync(kFullMask, value, 1);

    // __shfl_down_sync(mask, value, delta)：从更高编号 lane 读取，源 lane = lane + delta。
    // lane 3 会读 lane 4 的 value；lane 31 没有 lane 32，所以保留自己的 value。
    result.shfl_down_1 = __shfl_down_sync(kFullMask, value, 1);

    // __shfl_xor_sync(mask, value, laneMask)：源 lane = lane ^ laneMask。
    // laneMask=1 时，0<->1、2<->3、4<->5 ... 相邻两两交换。
    result.shfl_xor_1 = __shfl_xor_sync(kFullMask, value, 1);

    // width 参数把 warp 切成多个逻辑子组。width=8 时，lane 7 的 down(1) 不会读取 lane 8，
    // 因为 lane 8 属于下一个子组，所以 lane 7 保留自己的 value。
    result.shfl_width_8_down_1 = __shfl_down_sync(kFullMask, value, 1, 8);

    // __all_sync：参与线程的 predicate 全部为 true 才返回 1。
    // 这里奇数 lane 的 pred_even 是 false，所以 all_even = 0。
    result.pred_even = pred_even;
    result.all_even = __all_sync(kFullMask, pred_even);

    // __any_sync：参与线程里只要有任意一个 predicate 为 true 就返回 1。
    // 这里偶数 lane 为 true，所以 any_even = 1。
    result.any_even = __any_sync(kFullMask, pred_even);

    // __ballot_sync：把每个 lane 的 predicate 收集成 32-bit 位图。
    // bit i 对应 lane i；偶数 lane 为 true，所以结果是 0x55555555。
    result.ballot_even = __ballot_sync(kFullMask, pred_even);

    // __activemask：返回当前正在执行这条指令的活跃 lane 位图。
    // 分支前一个 warp 的 32 个 lane 都活跃，所以通常是 0xffffffff。
    result.active_mask_before_branch = __activemask();

    // 在 lane < 16 的分支内部，只有 0~15 号 lane 执行这条 __activemask，
    // 因此从 lane 0 看到的活跃掩码通常是 0x0000ffff。
    if (lane < 16) {
        result.active_mask_inside_branch = __activemask();
    } else {
        result.active_mask_inside_branch = 0;
    }

    // __match_any_sync：找出和当前 lane 的 key 相同的所有 lane，返回它们的 bitmask。
    // key=lane/8，所以 lane 0~7 一组，8~15 一组，16~23 一组，24~31 一组。
    result.match_any_key_mask = __match_any_sync(kFullMask, key);

    // __match_all_sync：判断所有参与线程的值是否完全相同。
    // 这里所有 lane 都传入常量 7，所以 pred=1，mask=0xffffffff。
    int all_same_pred = 0;
    result.match_all_same_mask = __match_all_sync(kFullMask, 7, &all_same_pred);
    result.match_all_same_pred = all_same_pred;

    // 这里传入 key=lane/8，不同 lane 的 key 不全相同，所以 pred=0，mask=0。
    int all_mixed_pred = 0;
    result.match_all_mixed_mask = __match_all_sync(kFullMask, key, &all_mixed_pred);
    result.match_all_mixed_pred = all_mixed_pred;

    // warp 级归约：输入是 lane+1，也就是 1..32，总和应该是 528。
    int reduced = warpReduceSum(lane + 1);
    result.warp_reduce_sum_lane0 = (lane == 0) ? reduced : -1;

    // width=8 的子 warp 归约：
    // lane 0 得到 1..8 的和 36；lane 8 得到 9..16 的和 100；以此类推。
    int sub_sum = subWarpWidth8ReduceSum(lane + 1);
    result.subwarp_width8_sum = (lane % 8 == 0) ? sub_sum : -1;

    // __syncwarp：warp 级同步屏障。
    // lane 0 写 shared_value 后，其他 lane 需要等它写完再读。
    // 在 Volta+ 独立线程调度下，warp 内线程不应盲目假设天然同步。
    if (lane == 0) {
        shared_value = 1234;
    }
    __syncwarp(kFullMask);
    result.syncwarp_shared_value = shared_value;

    out[lane] = result;
}

void printMask(unsigned mask) {
    std::printf("0x%08x", mask);
}

void printShuffleDemo(const LaneResult *r) {
    std::printf("\n=== Shuffle 指令：warp 内寄存器直接交换 ===\n");
    std::printf("lane | 原值 | shfl读lane0 | up读低1位 | down读高1位 | xor相邻交换 | down(width=8)\n");
    for (int lane = 0; lane < 10; ++lane) {
        std::printf("%4d | %4d | %11d | %10d | %12d | %11d | %13d\n",
                    r[lane].lane, r[lane].original, r[lane].shfl_from_lane0,
                    r[lane].shfl_up_1, r[lane].shfl_down_1,
                    r[lane].shfl_xor_1, r[lane].shfl_width_8_down_1);
    }
    std::printf("说明：shuffle 读的是其他 lane 的寄存器值，不经过 shared memory。\n");
    std::printf("说明：up/down 在边界位置没有合法源 lane 时，会保留调用线程自己的值。\n");
}

void printVoteDemo(const LaneResult *r) {
    std::printf("\n=== Vote 指令：把每个 lane 的布尔条件汇总成 warp 级结果 ===\n");
    std::printf("本例 predicate: lane %% 2 == 0，即偶数 lane 为 true。\n");
    std::printf("__all_sync：是否所有 lane 都满足条件 -> %d\n", r[0].all_even);
    std::printf("__any_sync：是否至少一个 lane 满足条件 -> %d\n", r[0].any_even);
    std::printf("__ballot_sync：每个 lane 的条件结果组成 bitmask -> ");
    printMask(r[0].ballot_even);
    std::printf("，偶数 bit 被置 1。\n");
}

void printMaskDemo(const LaneResult *r) {
    std::printf("\n=== __activemask 与 __syncwarp：观察活跃 lane 和做 warp 内同步 ===\n");
    std::printf("分支前 __activemask：");
    printMask(r[0].active_mask_before_branch);
    std::printf("，表示 32 个 lane 都在执行。\n");
    std::printf("lane < 16 分支内部 __activemask：");
    printMask(r[0].active_mask_inside_branch);
    std::printf("，表示只有 lane 0~15 正在执行该分支。\n");
    std::printf("__syncwarp 后，lane 17 读到 lane 0 写入的 shared_value = %d。\n",
                r[17].syncwarp_shared_value);
}

void printMatchDemo(const LaneResult *r) {
    std::printf("\n=== Match 指令：按值在 warp 内自动分组，Volta+ 支持 ===\n");
    for (int lane : {0, 8, 16, 24}) {
        std::printf("lane %2d 的 key=lane/8，__match_any_sync 返回同 key 组 mask：", lane);
        printMask(r[lane].match_any_key_mask);
        std::printf("\n");
    }
    std::printf("__match_all_sync 传入常量 7：mask=");
    printMask(r[0].match_all_same_mask);
    std::printf("，pred=%d，说明所有 lane 的值都相同。\n", r[0].match_all_same_pred);
    std::printf("__match_all_sync 传入 mixed key：mask=");
    printMask(r[0].match_all_mixed_mask);
    std::printf("，pred=%d，说明不是所有 lane 的值都相同。\n", r[0].match_all_mixed_pred);
}

void printReductionDemo(const LaneResult *r) {
    std::printf("\n=== 归约示例：用 __shfl_down_sync 替代 shared memory 交换 ===\n");
    std::printf("lane 0 保存 1..32 的 warp 总和 -> %d\n", r[0].warp_reduce_sum_lane0);
    std::printf("width=8 时，lane 0/8/16/24 分别保存各 8-lane 子组的和 -> %d, %d, %d, %d\n",
                r[0].subwarp_width8_sum, r[8].subwarp_width8_sum,
                r[16].subwarp_width8_sum, r[24].subwarp_width8_sum);
}

int main() {
    static_assert(kWarpSize == 32, "这个示例假设 CUDA warp 固定为 32 个 lane。");

    LaneResult host[kWarpSize]{};
    LaneResult *device = nullptr;
    CHECK_CUDA(cudaMalloc(&device, sizeof(host)));

    warpPrimitiveDemoKernel<<<1, kWarpSize>>>(device);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device));

    std::printf("CUDA Warp 原语演示：启动 1 个 block，1 个 warp，共 32 个 lane。\n");
    std::printf("建议先看每个表格的 lane 0~9，再结合源码注释理解源 lane 如何计算。\n");
    printShuffleDemo(host);
    printVoteDemo(host);
    printMaskDemo(host);
    printMatchDemo(host);
    printReductionDemo(host);

    return 0;
}
