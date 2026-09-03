# Q-CoMem Apple MLX Hybrid Replay：2026-08-11 实验记录

## 1. 结论状态

今天完成了 Qwen3.5-9B-4bit 在 Apple M4 Pro（24 GB unified memory）上的第一轮
Hybrid Replay 实机验证。当前可以保留两个结论：

1. **功能与存储结论成立。** 冻结策略 `depth=7, residual Q4, attention Q4,
   linear Q8` 在 512/2048/4096 tokens 的 9 次运行中均与 dense 生成相同的 8 个 token，
   完整持久化状态相对 exact prefix 压缩 `10.40–12.50×`。
2. **正式性能结论尚未成立。** 第一轮完整矩阵发生热状态变化和约 11.84 GiB 新增 swap；
   随后又发现 replay decode 未使用 suffix cache。该缺陷已修复，512-token 诊断显示
   packed replay 中位数比 dense 快 `1.24×`，但诊断以 `record-only` 运行且机器仍保留
   历史 swap，不能进入论文主表。

原始文件：

- 完整矩阵（环境无效、修复前）：`results/qcomem_mlx_hybrid_replay.json`
- 512 修复验证（record-only）：`results/qcomem_mlx_hybrid_replay_fixed_diagnostic_512.json`

## 2. 完整矩阵：可用的功能与存储数据

模型固定为 `mlx-community/Qwen3.5-9B-4bit`，revision
`8b2b98c00a6b4d291155e4890773ca8f769aee53`。每个长度运行 3 次，四条路径循环轮换，
greedy 生成 8 tokens。

| context | exact prefix | packed replay | 压缩倍数 | residual RRMSE | attention RRMSE | linear RRMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 65.125 MiB | 5.212 MiB | 12.50× | 0.08480 | 0.10434 | 0.01199 |
| 2048 | 113.125 MiB | 10.274 MiB | 11.01× | 0.08414 | 0.10387 | 0.01192 |
| 4096 | 177.125 MiB | 17.024 MiB | 10.40× | 0.08449 | 0.10397 | 0.01111 |

三种长度、三次重复中的 exact Q16 replay 和 Q4/Q4/Q8 replay 均与 dense 逐 token
一致。这个检查只覆盖冻结 prompt 的 8-token greedy 轨迹，不能代替下游 QA/F1 或长生成
质量评测。

该完整矩阵不能用于性能比较：

- thermal state：`nominal -> fair`；
- swap used：约 `0.55 GiB -> 12.39 GiB`，新增约 `11.84 GiB`；
- `formal_result_eligible=false`；
- invalid reasons：`post_run_thermal_state_fair`、`swap_growth_above_limit`。

## 3. 已定位并修复的 replay 性能缺陷

修复前，replay 首次执行 suffix prefill 后没有保留上层 KV/SSM cache。每生成一个新 token，
代码都会把整个 document residual、query residual 和已生成 residual 再跑一遍上层 25 层，
使 8-token decode 近似重复执行 8 次 suffix prefill。

修复后：

- 第一次 Read 只对完整 residual 做一次 suffix prefill，并建立 suffix-only cache；
- 每个后续 token 先通过 lower cache 得到一个 residual，再只用 suffix cache 处理这一个 token；
- tiny Qwen3.5 hybrid 单元测试仍验证 dense、exact prefix、Q16 replay 的生成完全一致；
- 全部 17 项测试通过。

## 4. 512-token 修复后诊断

本次只运行 512 tokens、3 次重复，`record-only`，不覆盖正式结果。运行前后 thermal state
均为 nominal，本次 swap 反而减少 8 MiB；但起始仍有约 7.86 GiB 历史 swap，因此保守地
保留诊断标签。

| 路径/阶段 | 中位时间 | 相对 dense |
|---|---:|---:|
| dense generation | 1.671 s | 1.00× |
| exact prefix generation | 0.246 s | 6.80× faster |
| Q16 replay generation | 1.356 s | 1.23× faster |
| Q4/Q4/Q8 replay generation | 1.351 s | 1.24× faster |
| full-prefix Write | 1.384 s | — |
| lower replay Write | 0.317 s | 4.37× faster than full-prefix Write |
| replay quantize | 0.0026 s | — |

packed replay 相对修复前从 9.612 s 降到 1.351 s，快 `7.11×`。3 次重复仍全部与 dense
生成一致，完整持久化压缩仍为 `12.50×`。这说明 suffix cache 修复方向正确，但当前只能
写成“512-token diagnostic observed 1.24×”，不能写成正式端侧 speedup。

## 5. 下一次正式采集

正式重跑前需要重启，使历史 swap 接近 0，并按
[实验前提协议](EXPERIMENT_PROTOCOL_ZH.md) 分成两组：

1. **冷态/交互式主表：** 每个 context length 使用独立进程，从 nominal 开始；至少
   5 个 session，通过 `--order-offset` 在 session 间轮换路径顺序，报告配对 speedup、
   median、IQR、CV 和范围。
2. **持续负载表：** 另起 session，从 nominal 开始，丢弃 warm-up 后连续 20 轮；逐轮
   报告 thermal state、TTFT/throughput、内存和 swap，单独分析热稳态平台段。

2048/4096 tokens 的修复后延迟仍需重测；在此之前不能用修复前的慢 replay 数字，也不能
从 512-token 诊断外推长上下文 speedup。
