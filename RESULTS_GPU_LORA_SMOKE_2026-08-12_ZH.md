# Q-CoMem LoRA H20 真实链路 Smoke（2026-08-12）

## 结论边界

本次实验只验证两条训练链路能够在真实 Qwen3.5-35B-A3B、8×H20-141GB 上完成前向、反向、DDP 同步和 checkpoint 保存：

1. Interface LoRA：用 dense teacher 蒸馏 split-depth 后的 residual-only student；
2. Quantization-conditioned LoRA：用 Q16 replay teacher 蒸馏真实 pack→dequant 的混合位 student。

它不是精度恢复结论。每条链路只训练 1 step，后续必须使用独立下游 validation/test 比较训练前后差异。

## 可复现信息

- QS trial：`1830043`，状态 `Complete`，总执行时间 14 分 37 秒；
- 模型：Qwen3.5-35B-A3B；
- 硬件：8×H20-141GB；
- 数据：PG-19 train-only 64 本 smoke 子集；
- 数据 SHA-256：`ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c`；
- LongBench test-v2：未读取；两个 metadata 均记录 `test_v2_used=false`；
- 输出目录：`runs/qcomem/lora-dual-real-smoke-20260812a/{interface,quant}`。

## Adapter 范围

默认只在 split depth 7 之后的 full-attention 层安装 LoRA，目标为 `q_proj/k_proj/v_proj/o_proj`：

- full-attention 层：7、11、15、19、23、27、31、35、39；
- 安装模块数：36；
- 可训练参数：6,193,152；
- checkpoint：每条约 74.4 MB，包含 LoRA、optimizer、scheduler 与 RNG 状态；
- lower layers 与 write path 均冻结。

这条默认线没有把 LoRA 加到全部 MoE/MLP 参数；MLP/MoE LoRA 应作为单独消融，不能混入主结果。

## Smoke 数值

| 训练线 | teacher | student store | step-1 KL loss | persistent bytes/样本（rank 均值） |
|---|---|---|---:|---:|
| Interface | dense | Q16 residual-only，chunk=128，overlap=0 | 0.125792 | 2,097,152 |
| Quant | Q16 replay | residual Q4；cache mixed `[8,8,4,4,8,8,8]` | 0.012453 | 4,157,440 |

损失使用 query positions 上 top-64 bidirectional KL，forward/reverse 权重为 0.6/0.4。Quant 线不是 QLoRA：模型权重没有量化，量化对象是持久 residual/KV/recurrent state，梯度只进入 suffix LoRA。

## 权重更新核验

两份 checkpoint 均含 36 个 `lora_a` 和 36 个 `lora_b` 张量。初始为零的 `lora_b` 在一步后已非零：

| 训练线 | `lora_b` L1 总和 | 最大绝对值 |
|---|---:|---:|
| Interface | 258.9403 | 7.99999e-05 |
| Quant | 258.1430 | 7.99998e-05 |

因此 smoke 确实执行了参数更新，而不只是成功走完 forward。日志中的 step 后 learning rate 为 0，是 1-step cosine schedule 已到终点的记录；本步 optimizer update 使用了更新前的学习率。

## 下一步判定标准

正式实验需要至少报告：

1. 未训练 Q16/Q8/Q4/mixed 与训练后对应配置的 paired 下游 F1；
2. prediction/token agreement、灾难性退化率与 paired bootstrap 95% CI；
3. 同一 adapter 是否跨 bit 策略泛化，或是否必须 bit-specific adapter；
4. Interface LoRA 与 quantization-conditioned LoRA 的独立贡献及联合训练消融；
5. adapter 常驻显存、加载成本和吞吐开销。

只有独立下游集上训练后相对未训练有稳定提升，且 Q4/mixed 更接近 Q16，才能声称 LoRA 恢复了量化精度。

## Interface 200-step 训练链路（新增）

在 1-step 链路通过后，Interface 主配置又完成了一次预注册的 200-step pilot：

- QS trial：`1830465`，状态 `Complete`，执行时间 14 分 40 秒；
- 输入窗口：1536 document + 512 query，chunk size 512，overlap 0；
- 优化器：AdamW，learning rate `8e-5`，20-step warmup 后 cosine decay；
- adapter：仍为 depth 7 之后 36 个 attention projection，6,193,152 个可训练参数；
- 每个窗口的持久 residual：6,291,456 bytes；
- checkpoint：50、100、150、200 均成功保存；最终 checkpoint SHA-256 为
  `c93269e31d4e7a3ed990ceb9ab602e56234fdc21011685b842a90263e36fb2c3`；
- 数据仍只来自 PG-19 train smoke64，metadata 明确记录 `test_v2_used=false`。

| 区间均值 | loss | forward KL | reverse KL |
|---|---:|---:|---:|
| step 1--20 | 0.065276 | 0.061696 | 0.070646 |
| step 181--200 | 0.029349 | 0.028875 | 0.030060 |
| 相对变化 | -55.0% | -53.2% | -57.5% |

200 个 step 的记录全部有限，最低单步 loss 为 0.018780（step 184）。这满足“训练链路
能够稳定降低训练分布上的接口蒸馏误差”的阶段性 gate；它仍不是下游精度结论。下一项
必须固定使用 checkpoint-200，在 LongBench validation 的同一 60 条样本上配对比较
`chunk-d7` 与 `chunk-lora-d7`，不能从四个 checkpoint 中按 validation 挑最好结果。

## Frozen-static quant 200-step 训练链路（新增）

第二条预注册 pilot 也已跑满，但结果是一个重要的**负结果**：

- QS trial：`1830598`，状态 `Complete`，执行时间 12 分 42 秒；
- student store：residual Q4、attention Q4、linear Q8、逐层策略
  `[8,8,8,4,8,8,8]`；teacher 为 Q16 replay；
- adapter 范围仍是 depth 7 之后 36 个 attention projection，共 6,193,152 参数；
- 200 个 step 和 checkpoint-50/100/150/200 均成功，数值全部有限；
- 最终 checkpoint SHA-256：
  `7cd2573227017431e5560c34ae676d7d9473689ff8d6a40bfc0734232d51366c`；
- 数据仍只来自 PG-19 train smoke64，`test_v2_used=false`。

| 区间均值 | loss | forward KL | reverse KL |
|---|---:|---:|---:|
| step 1--20 | 0.010212 | 0.010174 | 0.010270 |
| step 181--200 | 0.010898 | 0.010959 | 0.010806 |
| 相对变化 | +6.72% | +7.72% | +5.22% |

因此它只通过了训练链路、参数更新和 checkpoint gate，**没有通过**预注册的“末 20 step
KL 低于首 20 step”趋势 gate。不能据此声称 frozen-static 的量化误差被 LoRA 恢复。
下一轮优先测试更低 learning rate，以及从已收敛的 Interface adapter warm-start；当前
checkpoint-200 只允许作为固定的负对照做一次 validation，不能在四个 checkpoint 中挑点。

此外，当前 differentiable quant student 把 dequantized document 与 query 作为一个未缓存
suffix 序列执行；真实部署则先 document prefill，再以 query 续接 suffix cache。metadata 已
明确两者不声称等价，正式下游结论前还必须通过全 query-position semantic gate。

## Interface LoRA 60-sample validation（新增）

固定使用 Interface checkpoint-200 后，完成了未读取 test-v2 的下游配对验证：

- QS trial：`1830699`，状态 `Complete`；
- 数据：LongBench validation Qasper/2WikiMQA 各 30 条，source index 6--35；
- validation SHA-256：
  `1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe`；
- checkpoint SHA-256：
  `c93269e31d4e7a3ed990ceb9ab602e56234fdc21011685b842a90263e36fb2c3`；
- 对照只运行 dense、冻结 `chunk-d7`、`chunk-lora-d7`。逐 token 重算完整前缀且与
  dense 冗余的 oracle 被预先删除，以避免无必要的 H20 消耗；
- 三条路径保持 LongBench 官方上限：Qasper 128、2WikiMQA 32 个生成 token；
- launcher/每个 shard/聚合器均 hard-check 数据 SHA、checkpoint SHA、60 个唯一 key、
  source index 6--35；结果记录 `test_v2_used=false`。

| 配置 | mean F1 |
|---|---:|
| dense | 0.54312 |
| frozen chunk-d7 | 0.44731 |
| chunk-lora-d7 | 0.49115 |

Interface LoRA 相对 frozen chunk-d7 的 mean F1 delta 为 **+0.04384**，paired bootstrap
95% CI 为 `[-0.00760, +0.10239]`。其中 Qasper 为 +0.06324，CI
`[+0.00202, +0.14804]`；2WikiMQA 为 +0.02444，CI `[-0.04444, +0.10667]`。
prediction exact agreement 为 73.3%，灾难性回退率为 1/60。

这构成“Interface LoRA 能恢复一部分 chunk-local 接口损失”的正向证据，且 Qasper 子集
CI 已高于零；但 overall CI 仍跨零，不能称为统计上稳定的全数据集提升。相对 dense 仍有
-0.05197 mean F1 差距，CI `[-0.12986, +0.02046]`，且因 2WikiMQA 的 -0.11889
delta 没有通过预注册 mean margins。因此当前准确结论是**部分恢复，而非完全恢复或近无损**。

Adapter 由 6,193,152 个 FP32 参数组成，推理常驻量为 24,772,608 B（约 23.63 MiB）。
本轮记录的 generation-time 中位数为 frozen chunk 3.092 s、LoRA 2.811 s；逐样本 ratio
中位数 0.959。但配置顺序固定，且 LoRA 会改变输出长度，这些数字不能当成“LoRA 加速”或
正式 TTFT/TPOT 结论。部署开销仍需在相同生成轨迹、随机配置顺序和 warmup/repeat 协议下单测。

## Quant-static checkpoint 部署语义门禁（新增）

固定使用 quant-static checkpoint-200 后，QS Trial `1832331` 在真实 Qwen3.5-35B-A3B、
单节点 8×H20-141G 上完成了 16 个 PG-19 train-only 窗口的全 query-position 比较。前置
`cached-smoke` 五项 exactness 全部通过，但训练使用的 merged-uncached suffix 与部署使用的
document-prefill→query-continuation 两段 cache 没有通过硬门禁：

| 指标 | 观测值 | 硬阈值 |
|---|---:|---:|
| top-1 agreement | 0.97998046875（4014/4096） | 1.0 |
| mean training→deployment KL | 0.0021628225 | ≤ 0.001 |
| max position KL | 0.5228817463 | 仅报告 |
| max absolute logit error | 6.4375 | 仅报告 |

8/8 shard 均失败，共有 82 个 query positions 的 top-1 不一致。Launcher 按预注册顺序在
semantic gate 后 fail closed，因而 **没有运行** 60 条 LongBench validation，也没有产生
quant-static LoRA 的下游 F1。Trial 的 `Failed` 是预期的门禁非零退出；节点健康，非 OOM、
非排队或平台故障。`test_v2_used=false`。

完整证据见
[RESULTS_GPU_QUANT_LORA_SEMANTIC_GATE_2026-08-12_ZH.md](RESULTS_GPU_QUANT_LORA_SEMANTIC_GATE_2026-08-12_ZH.md)
与 `results/gpu-quant-lora-validation-20260812e/`。原始 aggregate SHA-256 为
`87b70cde56d2a3a49a1073e6d4c33cda964c7cb2e1e20075745ab95fbb4c182e`，完整文件清单由
`artifact.sha256` 固定。

这使当前 quant-static 主线形成两个相互独立的负证据：200-step KL 趋势没有改善，且训练/
部署 suffix 边界不等价。该 checkpoint 不能再作为部署形态的恢复候选。下一项必须先让新的
训练边界通过真实 capability：`detached-document-cache` 只能称 query-continuation-only 近似；
完整 document+query 梯度需要真实 functional cache。

## Detached-document-cache capability 负结果（新增）

QS Trial `1832364` 随后在单节点 8×H20 上验证了上述 query-continuation-only 近似。Document
suffix prefill 在 `no_grad` 下完成，cache tensor 经过 `detach().clone()`；student forward
完成后，8/8 rank 均在第一次 `loss.backward()` 报同一错误：`CopyBackwards` tensor
`[1,32,128,128]` 的 version 为 1、expected 0。节点健康且非 OOM。

因此该 Trial 没有 optimizer step、LoRA update、gradient coverage、真实 cache immutability、
checkpoint 或 semantic shard；后置 all-query deployment semantic gate 也没有运行。历史 mutable
Trial `1830867` 是 version 2、expected 1，本轮变为 1、expected 0，支持“跨 document/query
边界 mutation 已隔离，但 query continuation 内部仍有 mutable state 覆盖反向图”的工程定位。

完整证据见
[RESULTS_GPU_DETACHED_CACHE_AUTOGRAD_2026-08-12_ZH.md](RESULTS_GPU_DETACHED_CACHE_AUTOGRAD_2026-08-12_ZH.md)
和 `results/gpu-lora-quant-detached-capability-20260812a/`。因为 capability gate 未通过，不准备
Interface checkpoint warm-start、learning rate `2e-5` 的 200-step quant LoRA。下一条同语义路线
必须是 Qwen3.5 query-side functional cache，而不是继续增加 wrapper-level clone 变体。
