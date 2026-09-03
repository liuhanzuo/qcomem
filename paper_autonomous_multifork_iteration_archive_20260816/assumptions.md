# 假设与约束（Assumptions）

1. 使用的结果与证据均来自 `Trial 1840837` 的完整完成记录（Run 237580）。
2. kernel/几何设置保持固定（Q16、batch=1、single stream、vLLM unified_attention、4095 token 文档）。
3. PG-19 仅用于 report 所述实验边界，不包含 source6-9、source68-99、test_v2 任务。
4. 所有内存结论均为 PyTorch allocator 指标（allocated/reserved），不等价 NVML 进程峰值。
5. 未执行多文档、scheduler 并发、ragged batch、Q8/Q4 的对照。
