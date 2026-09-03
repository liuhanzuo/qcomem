# Quant-static LoRA 部署语义门禁结果（2026-08-12）

## 结论

固定使用 200-step quant-static LoRA checkpoint 的全 query-position semantic gate 已完成，
结果为 **failed**。训练采用的 `uncached_full_document_plus_query_sequence` 与部署采用的
`cached_document_prefill_then_full_query_continuation` 在真实 Qwen3.5-35B-A3B 上不等价。

Launcher 按预注册协议 fail closed，因此没有继续读取或运行 60 条 LongBench validation，
也没有产生 quant-static LoRA 下游 F1。这个结果不能解释为 LoRA 下游能力退化；它首先证明
当前 checkpoint 的训练执行语义不能直接代表部署执行语义。

## 任务与输入

- Job / Trial：`235193 / 1832331`；终态 `Failed`；
- 页面：<https://qs2.devops.xiaohongshu.com/model/production/job/trial/235193/1832331>；
- 硬件：单节点 8×H20-141G，节点健康，非 OOM、非调度或平台故障；
- checkpoint：step 200，SHA-256
  `7cd2573227017431e5560c34ae676d7d9473689ff8d6a40bfc0734232d51366c`；
- PG-19 semantic 数据 SHA-256：
  `ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c`；
- validation 数据 SHA-256：
  `1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe`；
- `test_v2_used=false`；LongBench test-v2 source index 68--99 未读取；
- 提交 YAML：`qs/qcomem-quant-lora-validation-20260812e.yaml`；
- 本地产物：`results/gpu-quant-lora-validation-20260812e/`。

提交前完成了 11 项 focused tests、validation source index 6--35 的 60 条唯一性检查、
三个输入 SHA 检查和 QS dry-run。代码使用独立远端快照
`qcomem_gpu_quant_lora_validation_20260812e`，避免共享目录在运行中发生漂移。

## 前置 exactness

`cached-smoke` 的五项基础检查全部通过：

- `cached_dense=true`；
- `full_prefix=true`；
- `cached_replay=true`；
- `fixed_order_multidoc=true`；
- `per_layer_q16=true`。

因此 semantic gate 失败不是基础 Q16 replay/cached implementation 已失效。

## 全 query-position 结果

16 个 PG-19 train-only 窗口，每个窗口检查 256 个 query positions，共 4096 个位置。

| 指标 | 观测值 | 硬阈值 | 结果 |
|---|---:|---:|---|
| top-1 agreement | 0.97998046875 | 1.0 | fail |
| top-1 匹配位置 | 4014 / 4096 | 4096 / 4096 | fail |
| mean training→deployment KL | 0.0021628225 | ≤ 0.001 | fail |
| max position KL | 0.5228817463 | 仅报告 | — |
| max absolute logit error | 6.4375 | 仅报告 | — |

共有 82 个位置 top-1 不一致，2016 个位置 KL 高于 `1e-3`，108 个位置 KL 高于
`1e-2`。8/8 shard 的 local threshold 均失败；全局决策使用 position-weighted 聚合，
没有放宽任何阈值。

## 为什么没有 60 条 validation F1

执行顺序被固定为：

1. 输入与 checkpoint hard checks；
2. cached exactness；
3. 16-window 全 query-position semantic gate；
4. 只有第 3 步通过，才运行 60 条 LongBench validation。

本次在第 3 步产生 `deployment-semantic-gate.json` 后以非零状态退出。结果目录中 validation
shard 数量为 0，说明下游数据没有在失败后被偷偷继续评测。

## 对 LoRA 路线的含义

历史 quant-static 200-step 训练本身已经出现负趋势：首 20 step 到末 20 step 的 KL loss
从 `0.010212` 上升到 `0.010898`（`+6.72%`）。本次又确认其 merged-uncached 训练边界与部署
边界不等价，因此不应继续把该 checkpoint 当作部署形态的精度恢复候选。

下一步应先让真实部署边界具备可训练性：优先做 6.19M 参数 LoRA 的
`detached-document-cache` 真实 1-step capability gate；它只能称 query-continuation-only
近似。若要训练 document prefill 与 query continuation 的完整梯度，则必须实现真实 Qwen3.5
functional cache。任一新路径都必须重新训练 checkpoint，再重新执行本门禁，不能复用本次失败的
merged-uncached checkpoint。

