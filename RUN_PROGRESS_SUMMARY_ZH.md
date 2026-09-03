# MacLLM-Bench 运行与进展汇总（截至 2026-08-14）

本文汇总当前仓库中可审计的实验入口、已完成 run、原始产物、阶段性结论和未闭环事项。
状态只依据仓库内的报告、JSON/JSONL、metadata 和 QS 配置判断；**QS YAML 存在只表示作业已配置，
不等于已经提交或完成**。仓库本身也无法证明某个远端作业此刻仍在运行。

## 1. 一页结论

当前项目已经完成从“residual-only 概念验证”到“完整 lower KV/recurrent state 可复用、混合位宽、
下游验证和部署代价拆解”的主要机制链路：

- Apple MLX 多文档旧版协议完成 5/5 个正式有效重复，证明 Write-once/read-many、真实 bit packing、
  depth-aware calibration 和环境审计可以工作；但该旧版 residual-only 路径存在很大的 split interface gap。
- H20 上保存并重放完整 lower KV/GatedDeltaNet state 后，8 条机制诊断达到 Q16 replay 与 dense
  逐 token 一致，说明早期 interface gap 主要来自“不完整状态接口”，不是必须依赖 LoRA 才能修复的问题。
- 当前最可信的容量/质量工作点是 depth 7、residual Q4、attention cache Q4、linear state Q8：
  在 64 条 legacy frozen test 上约 `14.12x` 持久状态压缩，mean F1 相对 dense 为 `-0.0087`，
  通过预注册 mean-margin gate；在新的 60 条 mixed-bit validation 上 frozen-static 为 `14.10x`，
  mean F1 delta `-0.00052`，没有灾难性退化。
- 部署基准再次确认约 `14.41x` 的**持久文档状态**容量收益，但当前 reference implementation
  的单活跃请求 peak 并未下降，TTFT 也明显慢于 full-prefix。现在能主张的是 resident capacity，
  不能主张 serving latency 或 active-memory 全面领先。
- Interface LoRA 200-step 和固定 checkpoint 的 60 条下游 validation 均已完成。训练 loss 前 20/后 20
  step 均值从 `0.06528` 降至 `0.02935`；下游 mean F1 相对未训练 chunk-d7 提高 `+0.04384`，
  但 overall CI 跨 0，且相对 dense 仍低 `-0.05197`，所以结论只是**部分恢复**。
- Frozen-static quant LoRA 也完成 200 step，但 loss 前 20/后 20 step 均值从 `0.01021` 升至
  `0.01090`（`+6.72%`），没有通过预注册趋势 gate，不能声称量化精度已恢复。
- 部署形态的 cached-two-stage quant LoRA 1-step smoke（QS `1830867`）是另一个负结果：
  8/8 rank 的 forward 成功，但 backward 均因 mutable cache inplace version mismatch 失败；
  无 step、gradient coverage 或 checkpoint。不能声称两段 cache 语义可直接训练。
- Detached-document-cache capability（QS `1832364`）也没有绕过反向图问题：document cache
  已 detach+clone，forward 完成，但 8/8 rank 仍在 query continuation backward 发生同形状
  inplace version mismatch；无 step/update/cache gate/checkpoint/semantic shard。这两个历史失败促成了
  query-side functional cache；该能力后续已由 Trial `1834056` 打通，并在 Trial `1840023`
  扩展到 answer-supervised 156-module 正式训练。
- Dense full-model supervised CE 已完成两级门禁：QS `1831074` 证明 execution；修正后的
  QS `1831289` 使用 FSDP 本地 FP32 persistent shards、BF16 forward 与 FP32 reduce/Adam，
  证明 34,660,610,688 参数全覆盖、40 层 FP32 delta 非零，并有 768,886,188 个参数值对
  下一次 BF16 forward 可见地改变。随后正式 QS `1831595` 完成 384 steps / 3 epochs 和
  三个 FP32 model-only DCP；train-split heldout token CE 在 step 128 从 `2.2415` 降到
  `1.6664`（`-25.66%`），step 256/384 又回升，故冻结 step 128 为最佳 checkpoint。
  统一下游 Trial `1832184` 随后在同一 8×H20 job 内完成 base/SFT ×
  dense/full-state-Q16/Q8/frozen 配对：SFT dense 相对 base dense 为 `-0.02308`，CI 跨 0，
  没有下游恢复证据；SFT Q8 相对 SFT dense 为 `-0.00697`，frozen 相对 SFT Q16 为
  `-0.00139` 且 state 再缩小 `3.59×`。这说明压缩增量仍稳，但普通 dense SFT 没有解决
  split/replay interface，也不是 Q-CoMem cached suffix SFT。
- Q16 cache fork 的 same-dtype storage alias 已修复。H20 Trial `1830738` 的总状态为 Failed：
  外层 dense-vs-incremental gate 是 5/8 通过、3/8 失败；但预注册的同一 source
  eager-vs-COW direct sub-gate 为 8/8 token 与完整 logits bitwise exact，source immutable、
  actual COW 且无 fallback。随后 Trial `1832356` 用同 caller boundary 的 incremental
  full-prefix/Q16 eager/Q16 COW 三方门禁完成 8/8，并跑完 4k short：frozen-static durable
  payload 为 `14.024×` 压缩，但 staging-inclusive total resident 只有 `3.056×`；CUDA/NVML
  peak 反而多约 `548/615 MiB`，TTFT 为 `4.122×`。这验证了 reference COW correctness，也明确
  排除了当前 staging 原型的 active-memory/TTFT 优势。
  HYPIC-lite 仍只有算法原型和静态 TTFT–bytes ledger audit。

因此，当前成熟度可以概括为：**机制与持久容量结论较完整；新版 dense control 与
answer-supervised native-cache LoRA 的下游均是正向点估计，但 CI 都跨 0，不能声称
显著恢复。single-request same-kernel page-reuse v2 已完成并得到 cache 增量显存正结果，
但没有 TTFT 加速；同文档 `N<=32` 的 multi-fork 同时驻留容量曲线也已完成并得到线性
显存正结果，但它采用单流 round-major 顺序执行，不是并发 serving。ragged、continuous
batching、多文档与最终 untouched test 仍未完成。**

### 2026-08-13 更新：长指令 SFT、native-cache LoRA 与真实 paged gate

- 新版 dense control 固定从 post-trained `Qwen3.5-35B-A3B` 初始化，使用 1024 条
  4K 长指令/一般能力 replay/teacher-preservation 数据训练 128 step。Trial `1833962`
  完成，内部隔离 heldout example-equal CE 从 `1.50377` 降至 `0.75625`，选择 step 128；
  两份 FP32 model-only DCP 均已原子发布并校验。它是 dense `use_cache=False` control，
  不是 cache-aware QAT。
- 统一 full-state Trial `1834066` 在固定 60 条 validation 上完成 base/SFT ×
  dense/Q16/Q8/frozen 配对。SFT 相对 base 的 F1 点估计分别为 `+0.01352/-0.00632/
  +0.00188/+0.01384`，CI 均跨 0；frozen-static 的 SFT F1 为 `0.55744`，相对 SFT dense
  为 `+0.00103`，持久状态仍为 `9.66 MiB`、相对 Q16 压缩 `3.59x`。这是正向信号，
  不能写成统计显著提升。
- Qwen3.5 native functional-cache LoRA Trial `1834056` 首次真正跑通了部署边界一致的反向图：
  30 个 GDN 层的 conv/recurrent cache 以 tensor rebind 代替 `copy_`，10 个 full-attention
  层沿用 out-of-place KV 更新。8/8 rank 的 36 模块/72 A-B tensor 梯度与更新门禁通过，
  训练完成 128 step；991 个 query positions 上 native/mutable 推理 top-1 为 `100%`、
  mean KL 为 `8.50e-9`。但 LoRA 相对未启用 adapter 的 frozen-static 下游 F1 为
  `-0.01543`，CI `[-0.06411,+0.02211]`，所以只证明训练/语义链路成立，未证明质量恢复。
- checkpoint 归因 Trial `1834193` 固定同一个 full-state frozen-static caller，依次评测
  adapter-disabled、step 0、64、128。Overall F1 为 `0.54360/0.53392/0.52138/0.52830`；
  三个启用 adapter 的 checkpoint 相对 disabled 的 CI 均跨 0。step 0 已出现负偏移，说明
  下降不能全部归因于 native 128-step；当前 answer-free query-token KL 与最终回答 F1
  明显错配。下一版已改为 answer+EOS supervision、任务均衡和 answer-position teacher
  preservation，并禁止用已消费 validation 反向选 checkpoint。
- 真分页 Python reference 的正式 correctness gate 没有被放宽。Trial `1833998` 中 token
  保持一致，但最终 logits max-abs 为 `0.25--1.375`；按 HF eager 的 BF16 softmax/value
  路径改为 two-pass 后，Trial `1834110` 的浅层误差明显减小，但误差仍从第 23/27 层累积，
  最终 max-abs `1.4375`、relative L2 `0.07662`。因此 benchmark 被 fail-closed 阻断，
  当前仍不能主张真实 PagedAttention 的显存/TTFT 收益；下一实现应使用融合 Triton/vLLM/
  FlashInfer kernel，而不是继续调宽 logits 阈值。

这组新结果把当前判断更新为：**frozen-static 的持久状态容量/质量 knee 仍稳健；新版数据配方
让 dense/frozen SFT 点估计回到正向，但尚未显著；native-cache LoRA 已解决 autograd 能力问题，
其纯 query-token KL 目标尚未转化为下游收益；真正分页 serving kernel 仍是 infra 主缺口。**

### 2026-08-14 更新：answer-supervised LoRA B 与 vLLM compatibility gate

- Answer-supervised native-cache LoRA B 在 Job `237290` / Trial `1840023` 正式完成。
  训练严格使用 domain-only 410 条和独立 official-train heldout 26 条，每个 global
  step 恰好 4 QASPER + 4 2WikiMQA；answer+EOS CE、answer-position teacher KL/hidden
  preservation 均按 example mean，不让长答案主导。Adapter 表面是 36 个 full-attention +
  120 个 GDN 关键 projection，共 156 modules / 26,689,536 个 FP32 参数；MLP
  未覆盖，不声称“全 suffix”。独立 heldout loss 从 `0.740831` 降到 `0.369624`，
  于读取 validation 前冻结 step 128。
- 同 job 固定 60 条 validation 上，frozen-static step 128 F1 为 `0.548575`，相对
  adapter-disabled 为 `+0.005559`，95% CI `[-0.019821,+0.030127]`；旧 step-0 在 2 个
  reference-yes 样本中的 `1/2` yes→no 在 step 64/128 恢复为 `2/2` yes。这是定向
  修复和小幅正点估计，不是总体统计显著提升。旧 Trial `1834056/1834193`
  的 36-module、answer-free query-KL 负结果继续作为历史基线，不被覆盖。
- 这不是纯 cold start：36 个 full-attention modules 来自已知负向的旧 step-0 warm
  start，120 个 GDN modules 是 cold start。训练使用 whole answer block；26 条 heldout
  的 token-by-token teacher-forcing 诊断为 206 positions top-1 无分叉、mean KL
  `0.000507959`，但非零差异意味着不宣称 chunk-boundary 数值等价。
- 成本分账为：每文档 frozen-static persistent state `9.6609 MiB`，Q16 为
  `34.6831 MiB`，小 `3.5901×`；每个模型进程还有共享 `101.8125 MiB` FP32
  adapter。只计 persistent state + adapter 时，break-even 为 `4.0689`，即从第 5
  个常驻文档起 Q4+adapter 低于 Q16；不包含模型权重、active workspace 和 allocator
  reserve。step 128/disabled TTFT 中位比 `1.06451×` 是单次固定顺序诊断，
  不是 ABBA/严格性能 claim。详见
  [answer-supervised LoRA B 报告](RESULTS_GPU_ANSWER_SUPERVISED_NATIVE_LORA_B_2026-08-14_ZH.md)。
- vLLM Q16 paged Trial `1840009` 在 PG-19 train-only 门禁 fail-closed：5/8 rank 产生
  semantic row，3/8 在 full-attention layer 31/31/19 未通过 Transformers eager 与
  vLLM Triton `unified_attention` 的数值兼容阈值。这仅是 **TF-eager compatibility
  negative**，不能归因 page-reuse layout 对错；LongBench validation/test-v2 和
  性能 benchmark 均未运行。详见
  [vLLM paged Q16 负结果](RESULTS_GPU_QWEN35_VLLM_PAGED_Q16_FORMAL_NEGATIVE_2026-08-14_ZH.md)。
- 同 kernel fair v2 Trial `1840486` 已正式完成：fresh full-copy 与
  shared-document/private-tail reuse 在 8/8 workload 的全部生成 token 和逐步
  full-vocab logits 上 bitwise exact。reuse/fresh cached-document TTFT paired median
  ratio 为 `1.009029`，没有加速；缓存增量 CUDA peak allocated ratio 为 `0.496010`
  （约低 `50.4%`），避免中位 `80 MiB` 物理文档 copy，combined unique storage 减少
  `87,819,520 B`。absolute 模型 peak 只差约 `83.64 MiB`；结果严格限于 Q16、batch 1、
  单请求、10 个 full-attention 层，不含 F1、多 query、ragged、NVML 或 kernel speedup。
  source 6--9 在 PG-19 授权后读取，68--99/test-v2 未读。详见
  [same-kernel fair v2 报告](RESULTS_GPU_QWEN35_VLLM_PAGED_FAIR_V2_2026-08-14_ZH.md)。

当前判断因而是：**answer supervision 已修复一个已知 answer-type 失败，但总体
F1 增益仍未显著；多文档容量 claim 必须同时记录共享 adapter 成本；同 kernel
真分页已证明单请求 reuse 精确并降低 cache 增量峰值，但没有 TTFT 加速。下一步是
multi-query/ragged serving 和独立下游质量实验，而不是把 TF eager 当作 layout oracle。**

## 2. Run 状态总表

| 实验线 | 状态 | 主要证据 | 当前可用结论 |
|---|---|---|---|
| MLX 3B smoke | 已完成，工程 smoke | `results/runs.jsonl`，warm-up + 3 次计时 | 基础 MLX 运行链路可用；不是正式研究结果 |
| Apple residual-only 多文档 | **正式完成** | [RESULTS_MULTIDOC_2026-08-11_ZH.md](RESULTS_MULTIDOC_2026-08-11_ZH.md)，5/5 formal eligible | 多文档复用、真实打包和校准链路成立；Q8 相对 BF16 CoMem 误差很小 |
| Apple 9B hybrid replay | 功能完成，正式性能无效 | [RESULTS_MLX_EDGE_2026-08-11_ZH.md](RESULTS_MLX_EDGE_2026-08-11_ZH.md) | `10.40x-12.50x` 持久压缩与 token 一致性可保留；主矩阵 latency 因 thermal/swap 不可入正式表 |
| GPU residual/interface diagnostics | 已完成 | [RESULTS_GPU_DOWNSTREAM_2026-08-11_ZH.md](RESULTS_GPU_DOWNSTREAM_2026-08-11_ZH.md) | 定位旧 residual-only split interface gap，并完成误差拆解 |
| GPU 完整 lower-state replay | 已完成机制诊断 | 同上，第 9 节 | Q16 replay 在 8/8 样本上与 dense token-exact，旧接口缺失可 training-free 消除 |
| GPU packed-state validation | 已完成 | 同上，64 条 validation | d7 Q4-attention/Q8-linear 是通过 gate 的最小状态 |
| GPU frozen legacy test | 已完成，已消费 | 同上，index 36-67，共 64 条 | frozen d7 Q4/Q8 约 `14.12x`，mean F1 delta `-0.0087` |
| GPU layer-wise bit calibration | 已完成 | Job/Trial `233909 / 1827870` | 生成 frozen-static、same-memory、minus-25% 三档 policy；linear state 更敏感 |
| GPU mixed-bit validation | **已完成** | [RESULTS_GPU_MIXED_VALIDATION_2026-08-12_ZH.md](RESULTS_GPU_MIXED_VALIDATION_2026-08-12_ZH.md)，QS `1830116` | 60 条 validation 上 frozen-static 是当前 near-lossless knee |
| GPU deployment Q16 exactness gate | **已完成** | [RESULTS_GPU_DEPLOYMENT_EXACTNESS_2026-08-12_ZH.md](RESULTS_GPU_DEPLOYMENT_EXACTNESS_2026-08-12_ZH.md)，QS `1830101` | 8/8 rank token-exact，可运行部署计时 |
| GPU deployment benchmark | **已完成** | [RESULTS_GPU_DEPLOYMENT_2026-08-12_ZH.md](RESULTS_GPU_DEPLOYMENT_2026-08-12_ZH.md)，QS `1830226` | 持久容量显著下降；当前 active peak/TTFT 没有同步改善 |
| LoRA dual-real 1-step smoke | 已完成 | [RESULTS_GPU_LORA_SMOKE_2026-08-12_ZH.md](RESULTS_GPU_LORA_SMOKE_2026-08-12_ZH.md)，QS `1830043` | Interface 与 quant-conditioned 前反向、DDP、更新和 checkpoint 链路可用 |
| Interface LoRA 200-step | **训练与下游 validation 已完成** | [LoRA 报告](RESULTS_GPU_LORA_SMOKE_2026-08-12_ZH.md)，QS `1830465` / `1830699`，`results/gpu-lora-interface-200-20260812b/`、`results/gpu-interface-lora-validation-20260812a/` | 训练趋势通过；60 条 validation 相对 chunk-d7 `+0.04384`，但 CI 跨 0 且未追平 dense，只能称部分恢复 |
| Quant-static LoRA 200-step | **已完成，趋势 gate 失败** | [LoRA 报告](RESULTS_GPU_LORA_SMOKE_2026-08-12_ZH.md)，QS `1830598`，`results/gpu-lora-quant-static-200-20260812e/` | 末 20 step loss 比首 20 step 高 `6.72%`；checkpoint 只作为固定负对照，不能声称恢复 |
| Quant-static LoRA semantic gate | **失败，validation 被硬阻断** | [semantic gate 报告](RESULTS_GPU_QUANT_LORA_SEMANTIC_GATE_2026-08-12_ZH.md)，QS `1832331`，`results/gpu-quant-lora-validation-20260812e/` | 4096 query positions 上 top-1 `97.998%`、mean KL `0.002163`；merged-uncached 训练与两段 cache 部署不等价，未运行 60 条 F1 |
| Quant cached-two-stage 1-step | **失败，能力 gate 未通过** | [cached autograd 负结果](RESULTS_GPU_CACHED_AUTOGRAD_2026-08-12_ZH.md)，QS `1830867`，`results/gpu-lora-quant-cached-two-stage-smoke-20260812c/` | 8/8 rank backward 为同一 inplace version mismatch；非 OOM/NCCL 根因，无 step/gradient coverage/checkpoint |
| Quant detached-document-cache 1-step | **失败，能力 gate 未通过** | [detached autograd 负结果](RESULTS_GPU_DETACHED_CACHE_AUTOGRAD_2026-08-12_ZH.md)，QS `1832364`，`results/gpu-lora-quant-detached-capability-20260812a/` | Document cache detach+clone 后 forward 通过；8/8 query backward 仍为 version 1、expected 0；无 step/update/cache gate/checkpoint/semantic shard |
| Native functional-cache LoRA 128-step | **正式完成；训练/语义通过，下游未恢复** | [native LoRA 报告](RESULTS_GPU_NATIVE_FUNCTIONAL_LORA_2026-08-13_ZH.md)，QS `1834056`，`results/gpu-native-functional-lora-domain-128-20260813c/` | 8/8 cache/gradient/update gate 通过；heldout KL 降 `23.58%`；991 positions top-1 100%；相对未训练 frozen F1 `-0.01543`，CI 跨 0 |
| Native LoRA checkpoint 归因 | **正式完成；step 0/64/128 均无恢复证据** | [归因报告](RESULTS_GPU_NATIVE_LORA_CHECKPOINT_ATTRIBUTION_2026-08-13_ZH.md)，QS `1834193`，`results/gpu-native-lora-checkpoint-attribution-20260813a/` | disabled/step0/64/128 F1=`0.54360/0.53392/0.52138/0.52830`；三个相对 disabled 的 CI 均跨 0，确认 adapter 初始化偏移与 loss/任务错配 |
| Answer-supervised native-cache LoRA B | **正式完成；定向修复，总体增益未显著** | [正式报告](RESULTS_GPU_ANSWER_SUPERVISED_NATIVE_LORA_B_2026-08-14_ZH.md)，QS `1840023`，`results/gpu-answer-supervised-native-lora-b-20260814a/` | 156 modules；heldout 选 step 128；vs frozen disabled F1 `+0.005559`、CI 跨0；2 个 reference-yes 上旧 yes→no 失败消失；每文档 `9.6609 MiB` 外另有共享 adapter `101.8125 MiB`，只按 state+adapter 从第5个文档起低于 Q16 |
| Suffix-full FSDP smoke | 提交前止损 | QS `1830869` | 使用相同已知失败的 mutable cache 反向图；在 Uncommit、无 Pod/产物时终止，未浪费 8×H20 运行资源 |
| Dense full-model SFT 1-step | **真实 update gate 通过；质量未测试** | [full SFT 报告](RESULTS_GPU_DENSE_SFT_SMOKE_2026-08-12_ZH.md)，QS `1831074` / `1831289` | 首轮仅 execution；FP32-shard 修正版完成全参数/全层 gradient+delta gate，无 checkpoint/quality 结论 |
| Dense full-model SFT 384-step | **正式完成；step 128 最优** | [正式 full SFT 报告](RESULTS_GPU_DENSE_SFT_FORMAL_2026-08-12_ZH.md)，QS `1831595` | heldout token CE `2.2415→1.6664`；后两 epoch 回升，三个 DCP 完整 |
| SFT × full-state Q-CoMem downstream | **正式完成；未观察到 SFT 恢复** | [统一下游报告](RESULTS_GPU_SFT_FULL_STATE_DOWNSTREAM_2026-08-12_ZH.md)，QS `1832184`，`results/gpu-sft-full-state-downstream-20260812b/` | SFT dense vs base `-0.02308`（CI 跨0）；SFT Q8 vs dense `-0.00697`；frozen vs Q16 `-0.00139`，state `3.59×` smaller |
| Dense 长指令/能力保持 SFT 128-step | **正式完成；step 128 最优** | [训练报告](RESULTS_GPU_DENSE_LONG_PRESERVATION_SFT_CONTROL_2026-08-13_ZH.md)，QS `1833962` | post-trained 初始化；内部 heldout CE `1.50377→0.75625`；FP32 DCP 完整；是 dense control，不是 cache-aware QAT |
| 新版 SFT × full-state validation | **正式完成；正向点估计，未显著** | [full-state 报告](RESULTS_GPU_DENSE_LONG_PRESERVATION_SFT_FULL_STATE_2026-08-13_ZH.md)，QS `1834066`，`results/gpu-dense-long-preservation-sft-full-state-validation-20260813a/` | SFT dense/frozen vs base 为 `+0.01352/+0.01384`，CI 均跨0；SFT frozen F1 `0.55744`、state `9.66 MiB` |
| Adapter 下游 validation | Interface 已完成；旧 quant checkpoint 被 semantic gate 拒绝 | `results/gpu-interface-lora-validation-20260812a/`、`results/gpu-quant-lora-validation-20260812e/` | Interface 为部分恢复；旧 quant 训练边界不等价，不能继续跑部署下游 F1 |
| COW staging deployment | **4k short 完成；correctness 通过，active peak/TTFT 为负** | [4k COW short 报告](RESULTS_GPU_COW_4K_SHORT_2026-08-12_ZH.md)，QS `1832356`，`results/gpu-deployment-cow-4k-short-incgate-20260812f/` | 三方 incremental gate 8/8；frozen payload `14.024×`、total resident `3.056×`；CUDA/NVML peak 多 `548/615 MiB`，不是 PagedAttention |
| Qwen3.5 真分页 Python reference | **correctness gate 失败，benchmark 被阻断** | QS `1833998` / `1834110` | 10 full-attention 层均被拦截、token 不变；two-pass BF16 路径仍在深层累积误差，最终 max-abs `1.4375`、relative L2 `0.07662`；不可报告 serving 性能 |
| Qwen3.5 vLLM Q16 paged TF-eager gate | **fail-closed compatibility negative** | [vLLM paged Q16 负结果](RESULTS_GPU_QWEN35_VLLM_PAGED_Q16_FORMAL_NEGATIVE_2026-08-14_ZH.md)，QS `1840009`，`results/gpu-qwen35-vllm-paged-q16-formal-negative-20260814b/` | 5/8 semantic rows，3/8 在 layer 31/31/19 失败；不能归因 layout，未跑 validation/性能 |
| Qwen3.5 vLLM Q16 paged same-kernel fair v2 | **正式完成；精确且省 cache 增量，无 TTFT 加速** | [fair v2 正式报告](RESULTS_GPU_QWEN35_VLLM_PAGED_FAIR_V2_2026-08-14_ZH.md)，QS `1840486`，`results/gpu-qwen35-vllm-paged-fair-v2-20260814c/` | 8/8 全步 token/logit bitwise exact；TTFT ratio `1.009029`；incremental peak allocated 约低 `50.4%`，物理 copy 省 `80 MiB`；Q16/batch1/single-request，非 F1/多 query/总模型减半 |
| Qwen3.5 vLLM Q16 paged multi-fork resident | **正式完成；N<=32 exact，shared-pool 容量线性正结果** | [multi-fork 正式报告](RESULTS_GPU_QWEN35_VLLM_PAGED_MULTIFORK_RESIDENT_2026-08-14_ZH.md)，QS `1840837`，`results/gpu-qwen35-vllm-paged-multifork-resident-20260814a/` | 8/8 rank 全 N token/logit/KV/GDN 与 cross-N exact；fresh/reuse=`80+90N`/`80+5N MiB`，N32 省 `2720 MiB` pool、absolute peak allocated 差约 `2.661 GiB`；单流顺序执行，非并发/速度/F1 |
| HYPIC-lite/suffix composition | 原型与静态账本完成，H20 未提交 | `gpu/hypic_lite.py`、[HYPIC-lite 说明](gpu/HYPIC_LITE_ZH.md)、[下一阶段预注册](NEXT_STAGE_EXPERIMENTS_ZH.md) | 1/4-segment bytes 边界已审计；full-suffix 状态会消耗 14× 容量卖点，尚无真实质量/TTFT 结果 |
| LongBench test-v2 | **冻结、未读取** | `gpu/LONGBENCH_SPLITS_20260812_ZH.md` | 必须等 adapter、bit policy 和 runtime 选择全部冻结后只运行一次 |

## 3. Apple MLX 进展

### 3.1 基础运行与本地 run ledger

`results/runs.jsonl` 当前只有 `mlx-smoke-3b-q4` 的 4 条记录：1 次 warm-up 和 3 次普通 run。
这说明基础采集器能记录 prompt/decode throughput、peak memory 和 raw output，但这个文件**不是全仓库
统一的实验台账**；后续 GPU 和正式 MLX 实验主要通过独立目录与报告追踪。

### 3.2 多文档 residual-only 正式实验

结果见 [RESULTS_MULTIDOC_2026-08-11_ZH.md](RESULTS_MULTIDOC_2026-08-11_ZH.md)：

- 模型为 `mlx-community/Llama-3.2-3B-Instruct-4bit`，6 documents、6 queries；
- 5 个全新 Python 进程全部 `completed` 且 `formal_result_eligible=true`，每轮 swap growth 为 0；
- 严格阈值下 depth `5/7/9` 都选择 Q8；Q8 相对 BF16 CoMem 为 `1.882x` 容量压缩，
  frozen query/答案位置 top-1 agreement 为 100%；
- Q4 为 `3.556x`，但没有稳定通过逐 query 100% agreement 与 max-position KL 门槛；
- BM25 sensitivity 虽保持 100% evidence recall，却让 depth-7 策略回退 BF16，说明 6-query
  calibration 太小，策略对文档组合敏感；
- 旧实现的 BF16 CoMem 相对 dense mean KL 为 `0.43/0.59/0.90`，这是 residual-only、
  document-local Write 的接口缺失，不能把“Q8 接近 BF16 CoMem”误写成“Q8 接近原模型”。

这条线的价值主要是协议、真实 bit packing 和多文档复用验证；它不是当前最终系统路径。

### 3.3 9B hybrid replay

结果见 [RESULTS_MLX_EDGE_2026-08-11_ZH.md](RESULTS_MLX_EDGE_2026-08-11_ZH.md)：

- 模型为 `mlx-community/Qwen3.5-9B-4bit`，冻结配置为 depth 7、residual Q4、attention Q4、linear Q8；
- 512/2048/4096 tokens、每个长度 3 次、生成 8 tokens 的功能矩阵中，packed replay 与 dense
  逐 token 一致，持久状态压缩约 `10.40x-12.50x`；
- 完整矩阵发生热升温及约 11.84 GiB 新增 swap，故 `formal_result_eligible=false`；
- 修复 replay decode suffix-cache 缺陷后的 512-token record-only 诊断约为 `1.24x` 相对 dense，
  只能作为定位结果，不能替代重新采集的正式 latency。

Apple 侧仍缺每个 context 独立冷态 session、多进程重复、median/IQR/CV 和跨平台统一的
`persistent/active/TTFT/TPOT` 指标。

## 4. H20 GPU 主线进展

### 4.1 从 residual-only gap 到完整 lower-state replay

早期 residual-only validation 的大幅掉点曾说明 split 语义是主瓶颈。但后续
`write_lower_replay()` 同时保存并恢复：

- split boundary document residual；
- lower full-attention K/V；
- lower GatedDeltaNet convolution 尾状态与 recurrent state。

在 8 条目标模型机制诊断上，depth 7/10/13 的 Q16 replay 均为 `8/8` dense token-exact，
mean F1 也一致。这一结果修正了早期判断：**旧 residual-only 接口仍有 gap，但当前完整状态
replay 已经在机制样本上消除了该 gap。** 更大的自然任务验证仍需要 exactness/downstream gate，
不能把 8 条诊断推广成所有输入上的 bitwise exact。

### 4.2 完整状态量化与冻结工作点

量化 residual、attention cache 与 linear recurrent state 后得到以下演进：

- d7 residual Q4 + state Q8：8-sample pilot 为 `11.82x`，mean F1 delta `-0.0042`；
- 将 attention 从 Q8 降到 Q4、不降低 linear state，得到 d7 `r4-a4-l8`，pilot 为 `14.10x`；
- 将 linear state 降到 Q4 会在多个 depth 明显掉点，说明 linear state 是当前更敏感组件；
- 64 条 validation 上 d7 Q4/Q8 为最小通过状态，随后冻结；
- 64 条 legacy test（index 36-67）上 frozen d7 Q4/Q8 为 `14.12x`，mean F1 delta `-0.0087`，
  没有 `F1 delta <= -0.5` 的灾难性退化，并通过预注册 mean margins；CI 下界略低于 `-0.02`，
  所以不能写成“统计上严格无损”。

这里的 “Q4/Q8” 指 attention cache Q4、linear state Q8；residual 固定 Q4。

### 4.3 4k/8k/16k 容量曲线

近无损主配置 d7 Q4-residual/Q8-state 的持久状态结果为：

| context | exact prefix | d7 Q4/Q8 | 压缩 |
|---:|---:|---:|---:|
| 4,096 | 141.875 MiB | 12.137 MiB | `11.69x` |
| 8,192 | 221.875 MiB | 20.887 MiB | `10.62x` |
| 16,384 | 381.875 MiB | 38.387 MiB | `9.95x` |

在 4 GiB persistent-state 预算下，16k exact prefix 约容纳 10 个文档，d7 Q4/Q8 约容纳
106 个文档。这是当前“扩大 model-plus-context working set”最直接的证据。

边界也很清楚：16k warm one-token latency 中 prefix 约 `0.262 s`，d7 replay 约 `2.384 s`；
incremental CUDA peak 也没有按持久状态压缩比下降。该 scaling run 的 latency 未按正式 serving
协议随机化/充分预热，因此只保留确定性的 persistent bytes，不把 latency 当正式主表。

### 4.4 Mixed-bit validation

[RESULTS_GPU_MIXED_VALIDATION_2026-08-12_ZH.md](RESULTS_GPU_MIXED_VALIDATION_2026-08-12_ZH.md)
使用 LongBench Qasper/2WikiMQA index 6-35，各 30 条，共 60 条：

| policy | 持久压缩 | 质量结论 |
|---|---:|---|
| frozen-static | `14.10x` | mean F1 vs dense `-0.00052`，CI 跨 0，无灾难性退化；当前首选 |
| same-memory mixed | 比 frozen-static 少 2.75% bytes | 出现约 1.67% 灾难性退化，没有胜过 static |
| minus-25% | `18.08x` | mean delta `-0.05103`，约 8.33% 灾难性退化；只适合作为恢复目标/容量优先档 |

当前严谨表述是“60 条 validation 上观察到约 14.1x 的 near-lossless Pareto knee”，不是
“已经证明无损”。

### 4.5 Deployment exactness、容量、显存与时延

Q16 exactness gate（QS `1830101`）在 8/8 rank 上 replay 与 dense 生成 token 完全一致，
因此 deployment benchmark 可以继续执行。随后 QS `1830226` 完成 8 workloads × 7 configs ×
3 repeats，共 168 条计时记录：

- mixed persistent state 约 9.7 MiB，相对 full-prefix 约 `14.41x`；
- 当前 eager deep-clone/fork、dequantization 和 suffix rebuild 使 mixed 单活跃请求 CUDA peak
  比 full-prefix 约多 543 MiB；
- mixed TTFT median 约 `0.6734 s`，full-prefix 约 `0.1634 s`；
- 全 Q4 可到约 `16.68x`，但质量明显下降；部署 near-lossless knee 仍指向 frozen-static 的约 14x。

这意味着系统卖点应定位在“许多文档状态常驻/更大的 context working set”，而不是
“单 query 更快或峰值显存按 14x 下降”。

Q16 fork 审计随后发现旧 same-dtype `reshape().to()` 可能让可变 lower cache 与 persistent
source 共用 storage，已改为 cache leaf 显式 clone；只读 boundary residual 保持零拷贝。
[COW direct 报告](RESULTS_GPU_COW_DIRECT_GATE_2026-08-12_ZH.md)记录了修复后的 Trial
`1830738`：作业总状态 Failed，原因是 256+32 caller boundary 下 dense single-chunk oracle
与 incremental cache 在 rank 1--3 后续 token 分叉，外层只有 5/8 通过；独立的同源
eager-vs-COW direct sub-gate 则 8/8 完整 logits `torch.equal`，max-abs/relative-L2 均为 0，
persistent source 在两次 request 后均未改变。因为总 gate 未通过，没有运行 short benchmark，
也没有新的 active-peak/TTFT 结论。

后续 [4k COW short](RESULTS_GPU_COW_4K_SHORT_2026-08-12_ZH.md)（QS `1832356`）把 hard gate
修正为相同 caller boundary 的 incremental full-prefix、Q16 eager 与 Q16 COW 三方比较；dense
single-chunk 只作诊断。8/8 rank 的三方 token trace、eager/COW bitwise logits、source immutability
与 COW immutable audit 全部通过，无 fallback；dense diagnostic 为 5/8。随后 8 workloads、
3 configs、24 rows 的 strict aggregation 完成：

- full-prefix total resident 中位 `140.342 MiB`；
- Q16 COW durable payload/staging/total resident 为 `35.915/35.915/56.137 MiB`，total resident
  收益 `2.500×`；
- frozen-static COW 为 `10.007/35.915/45.922 MiB`，必须分开写成 payload `14.024×`、
  total resident `3.056×`；
- frozen-static 相对 full-prefix 的 paired CUDA/NVML peak 中位反而多 `548/615 MiB`，TTFT/TPOT
  为 `4.122×/1.024×`。

这个实现会保留 packed source 与 dense execution template，并让 attention `torch.cat` 在
query/decode 中逐步 materialize；它没有 page allocator、block table、paged kernel 或 serving
scheduler，因此只能叫 audited `paged-cow-staging` reference，不能叫 PagedAttention。

#### 4.5.1 vLLM Q16 paged TF-eager compatibility negative

[vLLM paged Q16 正式负结果](RESULTS_GPU_QWEN35_VLLM_PAGED_Q16_FORMAL_NEGATIVE_2026-08-14_ZH.md)
（Job `237281` / Trial `1840009`）在 8 个 PG-19 train-only windows 上执行 fail-closed gate。
5 个 rank 完成 semantic row，3 个分别在 full-attention layer 31、31、19 没有通过
Transformers eager 与 vLLM Triton `unified_attention` 的逐元素数值阈值。5 个完成
window 虽然都保持 greedy top-1，它们的 KL 也都高于预注册 global mean `0.001`
阈值；另外 3 个缺 semantic row，不能报成完整 8-window 质量估计。

门禁同时更换了 attention 数值后端和 cache ownership/layout，因此结果只能叫
**TF-eager compatibility negative**：既不能证明 page-reuse layout 错，也不能证明它对。
LongBench validation/test-v2、ABBA timing 和 memory benchmark 都没有运行，所以没有
TTFT、TPOT、active-memory 或 capacity 正结果。该历史负结果继续保留，不能被后续
same-kernel run 改写成“HF eager 也精确”。

#### 4.5.2 vLLM Q16 paged same-kernel fair v2

[same-kernel fair v2 正式报告](RESULTS_GPU_QWEN35_VLLM_PAGED_FAIR_V2_2026-08-14_ZH.md)
（Job `237468` / Trial `1840486`）固定同一 vLLM 0.26 Triton
`unified_attention` callable，比较 fresh request-owned full-copy 与
shared-document/private-tail reuse。8/8 validation workload 的全部生成 token 与每步
full-vocab logits bitwise exact；HF eager 诊断保持 8/8 token 相同但 0/8 logit SHA 相同，
不进入授权或性能分母。

fresh-state ABBA 的 reuse/fresh cached-document TTFT paired median ratio 为
`1.009029`，TPOT 为 `1.000620`，没有延迟加速。per-request incremental CUDA peak
allocated paired median ratio 为 `0.496010`，约低 `50.4%`；物理文档 block copy
中位省 `80 MiB`，combined unique storage 省 `87,819,520 B`。但 absolute 模型 peak
只差约 `83.64 MiB`，allocator reserved 不降，不能称总模型显存减半。本轮是
Q16、batch 1、单请求、10 个 full-attention 层的 infra/correctness 实验，不计算 F1，
也没有 multi-query、ragged、NVML peak 或 isolated kernel speedup。

Trial `1840344` 仅在 preflight 因 `en_US` / C locale 排序不同导致 code-ledger SHA
漂移，没有进入 static、PG-19 或 validation。c 版显式固定 `LC_ALL=C`，真实 Pod ledger
与冻结值一致，保留该失败作为发布治理边界而不是算法负结果。

#### 4.5.3 vLLM Q16 paged multi-fork resident

[multi-fork resident 正式报告](RESULTS_GPU_QWEN35_VLLM_PAGED_MULTIFORK_RESIDENT_2026-08-14_ZH.md)
（Job `237580` / Trial `1840837`）把同一 Q16 kernel/ownership 对照扩展到同一份
4095-token PG-19 train-only 文档同时驻留 `N={1,2,4,8,16,32}` 个不同请求。8 个 rank
都完整运行全部 N；各 N 使用同一本 train book 冻结 query bank 的前 N 条 32-token 原文
片段。fresh/reuse 的全部 token、逐步 full-vocab logits、最终 logical K/V、GDN state 和
同一请求跨 N 的隔离均 exact。

解析 full-attention pool 为 fresh `80+90N MiB`、reuse `80+5N MiB`，即每请求节省
`85 MiB`，其中避免 `80 MiB` 文档 block copy。N=32 时 fresh/reuse 为 `2960/240 MiB`；
PyTorch production absolute peak allocated 中位数为 `74,623,183,360 / 71,765,915,136 B`，
相差约 `2.661 GiB`。这次选择 `4095 mod 128=127` 的 near-full-tail stress，不能外推
aligned 4096。N 个 cache 对象同时存活，但模型步在单 CUDA stream 上 round-major 顺序
执行；timing 只保留 raw diagnostic 且不聚合，因此没有 concurrent serving、throughput、
TTFT、ragged、NVML、多文档或 F1 结论。

### 4.6 HYPIC-lite 静态 Pareto 审计

[HYPIC-lite 说明](gpu/HYPIC_LITE_ZH.md) 已分开核算 transition-only、seam KV 和
approximate full-suffix local cache。depth-7/4k 一段 suffix store 为
152,567,808 B（当前 FP32 recurrent runtime）或 127,401,984 B（all-BF16 payload）；
四段 `w=8` 为 383,336,448 / 282,673,152 B，且尚未加入约 9.71 MiB mixed lower
state。transition 因 per-value-head `g/beta` 必须按 32 个 value heads 计数。该结果说明
HYPIC-style full state 不能无条件与 Q-CoMem 的 14× capacity claim 叠加；Q4/Q8
compressed-HYPIC 仍只是 future-work payload 下界。原型未提交 H20，不存在真实 speedup
或质量结论。

## 5. LoRA 当前进展

### 5.1 1-step 双链路 smoke

[RESULTS_GPU_LORA_SMOKE_2026-08-12_ZH.md](RESULTS_GPU_LORA_SMOKE_2026-08-12_ZH.md) 记录了
Qwen3.5-35B-A3B、8×H20 上的 Interface 与 quant-conditioned 两条真实训练链路：

- QS trial `1830043` 为 `Complete`；
- 36 个 LoRA 模块、6,193,152 个可训练参数；
- 前向、反向、DDP 同步、非零梯度更新和 checkpoint 保存均工作；
- `test_v2_used=false`；
- 每条线只训练 1 step，因此没有下游恢复结论。

### 5.2 Interface 200-step 产物

仓库新增 `results/gpu-lora-interface-200-20260812b/`，其中 metadata/metrics 显示：

- `world_size=8`、`last_step=200`、depth 7、rank 32/alpha 64；
- 训练对象仍是 `residual_only_chunk_local` Q16 student 对 dense teacher；
- 前 20 step mean loss 为 `0.06528`，后 20 step 为 `0.02935`，下降约 55%；
- `test_v2_used=false`；
- metadata 明确声明训练时 uncached suffix 与 deployment recurrent-cache 执行**不等价**。

QS trial `1830465` 已完成训练，最终 checkpoint 随后在 QS trial `1830699` 上固定用于
LongBench validation（Qasper/2WikiMQA index 6--35，各 30 条）：

- dense mean F1：`0.54312`；
- 未训练 chunk-d7：`0.44731`；
- chunk-lora-d7：`0.49115`；
- LoRA 相对 chunk-d7：`+0.04384`，paired 95% CI `[-0.00760, +0.10239]`；
- LoRA 相对 dense：`-0.05197`，且 2WikiMQA 的 mean margin 未通过。

因此，200-step 训练和下游 validation 都已闭环，但证据只支持“恢复了一部分 residual-only/chunk-local
接口损失”：overall CI 仍跨 0，也没有恢复到 dense。训练产物在
`results/gpu-lora-interface-200-20260812b/`，validation 产物在
`results/gpu-interface-lora-validation-20260812a/`；两者都明确记录 `test_v2_used=false`。

另外，完整 lower-state replay 已经 training-free 消除了早期机制 interface gap；如果主系统继续采用
完整状态 replay，Interface LoRA 更适合作为 residual-only 表示的消融。主资源优先级应转向 quant-static
或 minus-25% 的质量恢复，而不是把 Interface LoRA 默认当作主链路必需组件。

### 5.3 Quant-static 200-step 负结果与尚未闭环项

QS trial `1830598` 使用与 60-sample frozen-static validation 一致的 depth 7 policy：residual Q4、
attention Q4、linear Q8、`cache_layer_bits=[8,8,8,4,8,8,8]`。200 step 和四个 checkpoint
均成功保存，但前 20/后 20 step mean loss 为 `0.010212 -> 0.010898`（`+6.72%`），forward/reverse
KL 也分别上升 `7.72%`/`5.22%`。因此该 run 通过工程 gate，却**没有通过训练趋势 gate**。
产物位于 `results/gpu-lora-quant-static-200-20260812e/`；checkpoint-200 只能作为固定负对照，
不能从 50/100/150/200 中按 validation 挑选。

后续 QS Trial `1832331` 已补齐全 query-position deployment-semantic gate。16 个 PG-19 窗口、
4096 个 query positions 上 top-1 agreement 为 `0.97998046875`，mean KL 为 `0.0021628225`，
分别没有通过 `1.0` 与 `<=0.001` 的硬阈值；8/8 shard 均失败。前置 cached exactness 五项全过，
节点健康且非 OOM。Launcher 因此没有运行 60 条 LongBench validation，不能为旧 checkpoint
报告下游 F1。完整负结果见
[RESULTS_GPU_QUANT_LORA_SEMANTIC_GATE_2026-08-12_ZH.md](RESULTS_GPU_QUANT_LORA_SEMANTIC_GATE_2026-08-12_ZH.md)。

以下是 2026-08-12–13 时尚未闭环的历史问题；保留它们是为了说明新路径为何
必须换掉 mutable cache 和 answer-free objective，不代表当前 functional-cache 能力仍未实现：

- 直接把两段 mutable cache 放进反向图已由 Trial `1830867` 证明不可行；query-continuation-only
  `detached-document-cache` 又由 Trial `1832364` 证明 wrapper-level detach+clone 仍不足：8/8 rank
  在 query backward 发生 version 1、expected 0 的 inplace mismatch。两轮均非 OOM，且都没有
  optimizer step 或 checkpoint；
- 当时的下一条部署边界路线必须实现 Qwen3.5 query-side functional cache，并依次通过
  backward、全部 module finite/nonzero query gradient、cache immutability、update 与 semantic gate；
- 只有新的部署边界 checkpoint 通过 semantic gate 后，才允许跑固定的 60-sample paired downstream；
- 更低 learning rate、从 Interface adapter warm-start，以及 `minus-25%` 恢复仍只是后续假设，
  不能复用本次被拒绝的 merged-uncached checkpoint；
- 严格 ABBA/多轮相同生成轨迹的 adapter 加载、TTFT/TPOT 开销尚未报告；Trial
  `1840023` 的 `1.06451×` TTFT 只是固定顺序单次诊断；
- test-v2 尚未读取，这是正确的数据治理状态。

### 5.4 Answer-supervised 156-module native-cache LoRA B

[2026-08-14 正式报告](RESULTS_GPU_ANSWER_SUPERVISED_NATIVE_LORA_B_2026-08-14_ZH.md)
记录了 Job `237290` / Trial `1840023`。这次不是在旧 36-module query-KL run 上换一个
统计口径，而是重新冻结训练协议：

- train/official-train heldout 仅使用有真实 deployment boundary 的 QASPER/2WikiMQA
  `410/26` 条；128 个 global steps 的每步全局 batch 都是 4+4，任务 mass 恰好
  `0.5/0.5`；
- document prefill 后用 query + `answer[:-1]` multi-token continuation，只在 answer+EOS
  causal-shift positions 上算 `0.45 CE + 0.35 teacher KL + 0.20 hidden cosine`；三项均先
  position mean、再 example mean；
- surface 为 36 个 suffix full-attention 和 120 个 GDN 关键 projection，共 156
  modules / 312 A-B tensors / 26,689,536 FP32 parameters。MLP 明确排除，因此不称
  “覆盖所有 suffix 线性层”；
- 36 个 full-attention modules 从旧负向 step 0 warm-start，120 个 GDN modules 从
  LoRA cold start，因此结果归因为“answer supervision + 扩 surface 修复 warm-started
  system”，不是纯 cold-start B；
- step 1 全 312 tensors 的 finite/update、functional-cache rebind、FP32 optimizer 和最小
  reserved headroom `6,516,703,232 B` 通过门禁。step 2 的最小 reserved headroom 只有
  `1,426,915,328 B`，协议本就没有对 step 2 重复声称 4 GiB gate；训练最终无 OOM
  完成，但不得倒写成“step 2 也通过 4 GiB”。

只按 independent official-train heldout 选 checkpoint：loss 为
`0.740831 / 0.374557 / 0.369624`（step 0/64/128），因此在读取 validation 前选
step 128。同 job 的 60 条 full-state validation 始终完整跑 disabled、0、64、128；
frozen-static F1 为 `0.543016 / 0.533925 / 0.544805 / 0.548575`。selected step 128
相对 disabled 为 `+0.005559`，95% CI `[-0.019821,+0.030127]`；QASPER/2WikiMQA
分别为 `+0.014452/-0.003333`。旧 step-0 在仅 2 个 reference-yes 定向样本中的
`1/2` yes→no 在 step 64/128 恢复为 `2/2` yes。这支持“已知失败模式被修复，
总体点估计转正”，不支持“总体显著提升”。

部署成本分开记账：

| 对象 | 字节 | MiB | 作用域 |
|---|---:|---:|---|
| frozen-static persistent state | 10,130,160 /文档 | 9.6609 /文档 | 随常驻文档线性增长 |
| Q16 persistent state | 36,367,872 /文档 | 34.6831 /文档 | 随常驻文档线性增长 |
| FP32 answer-LoRA | 106,758,144 /模型进程 | 101.8125 /模型进程 | 同一进程的文档共享 |

每文档 state 相对 Q16 小 `3.5901×`；只将 persistent state 与 shared adapter 相加时，
`106,758,144 / (36,367,872 - 10,130,160) = 4.0689`，所以从第 **5** 个常驻
文档起 Q4+adapter 总增量驻留低于 Q16。这个 break-even 不含模型权重、active
request/workspace、allocator reserve 或临时 activation，不能代替峰值显存 claim。step 128
相对 disabled 的 median TTFT 比 `1.06451×` 仅来自 60 条固定顺序的单次运行，没有
ABBA/多轮重复，因此仅是诊断。validation 6–35 本轮已消费，绝不可反向重选
checkpoint；test-v2 68–99 仍未读取。

## 6. 数据切分与使用边界

固定数据源为 `zai-org/LongBench` revision
`5e628be450b7e67fb7ae6e201bd6d8f7056f7672`，只使用 Qasper 与 2WikiMQA：

| 用途 | 每数据集 source index | 总样本 | 当前状态 |
|---|---:|---:|---|
| pilot | 0-3 | 8 | 已使用 |
| layer-bit calibration | 4-5 | 4 | 已使用 |
| mixed-bit / LoRA full-state validation | 6-35 | 60 | 已使用；Trial `1840023` 只在 heldout 选定 step 128 后读取，不得再用于调参/重选 checkpoint |
| legacy test | 36-67 | 64 | 已使用，不能再作为新 policy/adapter 的 untouched test |
| mixed-bit test-v2 | 68-99 | 64 | 冻结且未读取 |

test-v2 的 SHA256 为
`fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f`。
详细边界见 [gpu/LONGBENCH_SPLITS_20260812_ZH.md](gpu/LONGBENCH_SPLITS_20260812_ZH.md)。

旧版报告里的 index 36-67 当时可以称为 frozen test，但对 2026-08-12 之后的新 mixed-bit policy
和 LoRA 已经是 consumed legacy test。index 6-35 也已用于 mixed-bit、SFT 和 Trial
`1840023` 的 post-selection full-state 评测；后者的 checkpoint 只由独立 official-train
heldout 26 条选择，但这不会让 6-35 重新变成可调参数据。新版最终结论只能使用
68-99 的 test-v2，并且必须等 adapter、bit policy、checkpoint 和 runtime 选择全部冻结。

## 7. 仓库运行入口

### 7.1 Apple / 本地 Make 入口

| 命令 | 作用 | 默认产物/说明 |
|---|---|---|
| `make bootstrap` | 创建本地环境并安装依赖 | `scripts/bootstrap.sh` |
| `make run` | 首次基准入口 | `scripts/run_first_benchmark.sh` |
| `make system-info` | 采集机器信息 | system-info JSON |
| `make smoke` | MLX 3B smoke | `results/runs.jsonl`、`results/raw/` |
| `make context` | 512/2048/8192 context benchmark | `results/summary.json` 等 |
| `make manual-context` | 手写 `mlx.core` 基线 | `results/manual_mlx_context_benchmark.json` |
| `make comem-smoke` | Q-CoMem residual smoke | `results/q_comem_smoke.json` |
| `make comem-multidoc` | 旧版多文档正式入口，带 AC/thermal preflight | `results/q_comem_multidoc_benchmark.json` |
| `make comem-multidoc-diagnostic` | record-only 调试入口 | 永远按相同 formal criteria 评估 |
| `make mlx-replay` | Apple 9B frozen hybrid replay | `results/qcomem_mlx_hybrid_replay.json` |
| `make mlx-replay-diagnostic` | 短上下文 record-only replay 调试 | 不作为正式性能 |
| `make test` | compileall + unittest | 只做代码检查，不产生实验结论 |

Apple 正式运行的环境要求、冷态 session 模板和结果判定见
[TEST_RUNBOOK_ZH.md](TEST_RUNBOOK_ZH.md) 与 [EXPERIMENT_PROTOCOL_ZH.md](EXPERIMENT_PROTOCOL_ZH.md)。

### 7.2 GPU / QS 入口

GPU 运行由 `qs/*.yaml` 固定远端资源、数据和环境，再调用 `gpu/launch_*_8gpu.sh` 或对应
Python runner。主要映射如下：

| 任务 | QS 配置 | Launcher/runner |
|---|---|---|
| downstream | `qs/qcomem-downstream-*.yaml` | `gpu/launch_8gpu.sh`、`gpu/run_downstream.py` |
| replay validation/diagnostic | `qs/qcomem-replay-*.yaml` | `gpu/launch_replay_8gpu.sh`、`gpu/run_replay_diagnostic.py` |
| layer sensitivity | `qs/qcomem-layer-sensitivity*.yaml` | `gpu/launch_layer_sensitivity_8gpu.sh` |
| mixed validation | `qs/qcomem-mixed-validation-reasoning.yaml` | `gpu/launch_mixed_validation_8gpu.sh` |
| deployment/exactness | `qs/qcomem-deployment-*.yaml` | `gpu/launch_deployment_8gpu.sh`、`gpu/run_deployment_bench.py` |
| LoRA smoke/200-step | `qs/qcomem-lora-*.yaml` | `gpu/launch_lora_8gpu.sh`、`gpu/train_qcomem_lora.py` |
| COW/paged | `qs/qcomem-deployment-cow-*.yaml` | deployment launcher + `gpu/qcomem_paged.py` |

部署指标定义与聚合口径见 [DEPLOYMENT_BENCHMARK_ZH.md](DEPLOYMENT_BENCHMARK_ZH.md)，
LoRA 运行和验收见 [LORA_TRAINING_RUNBOOK_ZH.md](LORA_TRAINING_RUNBOOK_ZH.md)。

## 8. 结果与产物索引

| 内容 | 人类可读报告 | 主要机器产物 |
|---|---|---|
| Apple 多文档 | [RESULTS_MULTIDOC_2026-08-11_ZH.md](RESULTS_MULTIDOC_2026-08-11_ZH.md) | `results/q_comem_multidoc_*.json`、`results/formal-20260811-ac.LIgSzd/` |
| Apple 9B replay | [RESULTS_MLX_EDGE_2026-08-11_ZH.md](RESULTS_MLX_EDGE_2026-08-11_ZH.md) | `results/qcomem_mlx_hybrid_replay*.json`、`results/qcomem_mlx_hybrid_store/` |
| GPU downstream/replay/scaling | [RESULTS_GPU_DOWNSTREAM_2026-08-11_ZH.md](RESULTS_GPU_DOWNSTREAM_2026-08-11_ZH.md) | `results/gpu-downstream-*`、`results/gpu-replay-*`、`results/gpu-capacity-*` |
| Mixed-bit validation | [RESULTS_GPU_MIXED_VALIDATION_2026-08-12_ZH.md](RESULTS_GPU_MIXED_VALIDATION_2026-08-12_ZH.md) | `results/gpu-mixed-validation-20260812d/` |
| Deployment exactness | [RESULTS_GPU_DEPLOYMENT_EXACTNESS_2026-08-12_ZH.md](RESULTS_GPU_DEPLOYMENT_EXACTNESS_2026-08-12_ZH.md) | `results/gpu-deployment-exactness-20260812d/` |
| Deployment benchmark | [RESULTS_GPU_DEPLOYMENT_2026-08-12_ZH.md](RESULTS_GPU_DEPLOYMENT_2026-08-12_ZH.md) | `results/gpu-deployment-validation-20260812i/` |
| COW direct paired gate | [RESULTS_GPU_COW_DIRECT_GATE_2026-08-12_ZH.md](RESULTS_GPU_COW_DIRECT_GATE_2026-08-12_ZH.md)，QS `1830738` | `results/gpu-deployment-cow-direct-gate-20260812c/`（总 Trial Failed；direct sub-gate 8/8） |
| COW 4k deployment short | [RESULTS_GPU_COW_4K_SHORT_2026-08-12_ZH.md](RESULTS_GPU_COW_4K_SHORT_2026-08-12_ZH.md)，QS `1832356` | `results/gpu-deployment-cow-4k-short-incgate-20260812f/`（8 shards、24 rows、strict aggregate passed） |
| vLLM Q16 paged TF-eager compatibility negative | [RESULTS_GPU_QWEN35_VLLM_PAGED_Q16_FORMAL_NEGATIVE_2026-08-14_ZH.md](RESULTS_GPU_QWEN35_VLLM_PAGED_Q16_FORMAL_NEGATIVE_2026-08-14_ZH.md)，QS `1840009` | `results/gpu-qwen35-vllm-paged-q16-formal-negative-20260814b/`；fail-closed，未跑 validation/性能，不能归因 layout |
| vLLM Q16 paged same-kernel fair v2 | [RESULTS_GPU_QWEN35_VLLM_PAGED_FAIR_V2_2026-08-14_ZH.md](RESULTS_GPU_QWEN35_VLLM_PAGED_FAIR_V2_2026-08-14_ZH.md)，QS `1840486` | `results/gpu-qwen35-vllm-paged-fair-v2-20260814c/`；8/8 token/logit exact，TTFT ratio `1.009029`，incremental peak allocated 约低 `50.4%`，58 项 ledger |
| vLLM Q16 paged multi-fork resident | [RESULTS_GPU_QWEN35_VLLM_PAGED_MULTIFORK_RESIDENT_2026-08-14_ZH.md](RESULTS_GPU_QWEN35_VLLM_PAGED_MULTIFORK_RESIDENT_2026-08-14_ZH.md)，QS `1840837` | `results/gpu-qwen35-vllm-paged-multifork-resident-20260814a/`；N=1..32 全步/cross-N exact，N32 pool 省 `2720 MiB`，35 项 ledger |
| LoRA smoke | [RESULTS_GPU_LORA_SMOKE_2026-08-12_ZH.md](RESULTS_GPU_LORA_SMOKE_2026-08-12_ZH.md) | `results/gpu-lora-dual-real-smoke-20260812a/` |
| Interface LoRA 200-step + validation | [RESULTS_GPU_LORA_SMOKE_2026-08-12_ZH.md](RESULTS_GPU_LORA_SMOKE_2026-08-12_ZH.md)，QS `1830465` / `1830699` | `results/gpu-lora-interface-200-20260812b/`、`results/gpu-interface-lora-validation-20260812a/` |
| Quant-static LoRA 200-step | [RESULTS_GPU_LORA_SMOKE_2026-08-12_ZH.md](RESULTS_GPU_LORA_SMOKE_2026-08-12_ZH.md)，QS `1830598` | `results/gpu-lora-quant-static-200-20260812e/` |
| Quant-static LoRA semantic gate | [RESULTS_GPU_QUANT_LORA_SEMANTIC_GATE_2026-08-12_ZH.md](RESULTS_GPU_QUANT_LORA_SEMANTIC_GATE_2026-08-12_ZH.md)，QS `1832331` | `results/gpu-quant-lora-validation-20260812e/`；gate failed，validation 未运行 |
| Quant cached autograd negative smoke | [RESULTS_GPU_CACHED_AUTOGRAD_2026-08-12_ZH.md](RESULTS_GPU_CACHED_AUTOGRAD_2026-08-12_ZH.md)，QS `1830867` | `results/gpu-lora-quant-cached-two-stage-smoke-20260812c/` |
| Quant detached-cache autograd negative capability | [RESULTS_GPU_DETACHED_CACHE_AUTOGRAD_2026-08-12_ZH.md](RESULTS_GPU_DETACHED_CACHE_AUTOGRAD_2026-08-12_ZH.md)，QS `1832364` | `results/gpu-lora-quant-detached-capability-20260812a/`；forward 后 8/8 backward 失败，artifact SHA 已固定 |
| Native functional-cache LoRA 128-step 历史负基线 | [RESULTS_GPU_NATIVE_FUNCTIONAL_LORA_2026-08-13_ZH.md](RESULTS_GPU_NATIVE_FUNCTIONAL_LORA_2026-08-13_ZH.md) / [checkpoint 归因](RESULTS_GPU_NATIVE_LORA_CHECKPOINT_ATTRIBUTION_2026-08-13_ZH.md)，QS `1834056/1834193` | `results/gpu-native-functional-lora-domain-128-20260813c/`、`results/gpu-native-lora-checkpoint-attribution-20260813a/` |
| Answer-supervised native-cache LoRA B | [RESULTS_GPU_ANSWER_SUPERVISED_NATIVE_LORA_B_2026-08-14_ZH.md](RESULTS_GPU_ANSWER_SUPERVISED_NATIVE_LORA_B_2026-08-14_ZH.md)，QS `1840023` | `results/gpu-answer-supervised-native-lora-b-20260814a/`；105 个非 pycache 原始 run 文件，scientific ledger 覆盖 126 项 |
| Supervised SFT 数据准备 | [RESULTS_SUPERVISED_SFT_DATA_2026-08-12_ZH.md](RESULTS_SUPERVISED_SFT_DATA_2026-08-12_ZH.md) | `results/supervised-sft-data-smoke-20260812/`、`results/supervised-sft-formal-20260812b/`（test-v2 未读） |
| Dense full-model SFT capability/update smoke | [RESULTS_GPU_DENSE_SFT_SMOKE_2026-08-12_ZH.md](RESULTS_GPU_DENSE_SFT_SMOKE_2026-08-12_ZH.md)，QS `1831074` / `1831289` | `results/gpu-dense-supervised-sft-smoke-20260812a/`、`results/gpu-dense-supervised-sft-fp32-delta-smoke-20260812a/` |
| Dense full-model SFT 384-step formal | [RESULTS_GPU_DENSE_SFT_FORMAL_2026-08-12_ZH.md](RESULTS_GPU_DENSE_SFT_FORMAL_2026-08-12_ZH.md)，QS `1831595` | `results/gpu-dense-supervised-sft-formal-384-20260812b/`；远端保留 step 128/256/384 DCP |
| Dense SFT × full-state Q-CoMem downstream | [RESULTS_GPU_SFT_FULL_STATE_DOWNSTREAM_2026-08-12_ZH.md](RESULTS_GPU_SFT_FULL_STATE_DOWNSTREAM_2026-08-12_ZH.md)，QS `1832184` | `results/gpu-sft-full-state-downstream-20260812b/`；base/SFT 同 job 共 64 shards |

## 9. 当前能说与不能说的结论

### 可以说

- 完整 lower-state replay 已在目标模型机制 gate 上做到 Q16 token-exact；
- 对完整可复用状态进行分类型量化后，depth-7 Q4 residual/Q4 attention/Q8 linear 是当前最稳妥工作点；
- 当前 validation、legacy test、scaling 和 deployment 证据共同支持“约 10x-14x 的近无损持久状态压缩”；
- 该压缩可显著提高固定 persistent-state 预算下可常驻的文档数；
- linear recurrent state 比 attention cache 更不适合直接降到 Q4；
- Interface LoRA 在固定 60 条 validation 上相对未训练 chunk-d7 恢复了部分 mean F1；
- Answer-supervised 156-module native-cache LoRA B 在独立 official-train heldout 上选到
  step 128，使 frozen-static 相对 disabled 的 F1 点估计为 `+0.005559`，并在仅 2 个
  reference-yes 定向样本上消除了旧 step-0 yes→no 失败；这是定向修复证据，
  overall CI 仍跨 0；
- 该 LoRA 的 frozen-static persistent state 是 `9.6609 MiB`/文档，共享 FP32
  adapter 是 `101.8125 MiB`/模型进程；仅按这两项分账时，从第 5 个常驻文档
  起 Q4+adapter 总增量驻留低于 Q16；
- 修复后的 COW 在 8 条 256+32 小门禁上与同源 eager replay 达到完整 logits bitwise exact，
  且 persistent source 未被修改。
- 同边界 incremental full-prefix/Q16 eager/Q16 COW 三方 hard gate 已 8/8 通过；4k short 中
  frozen-static durable payload 为 `14.024×` 压缩，计入 staging 的 total resident 仍有 `3.056×`。
- 35B dense full-model supervised CE 的 8-rank FSDP 路径可行；在 BF16 forward、FP32
  persistent shards/reduce/Adam 下，真实的一步全参数 update 已通过 gradient/delta 硬门禁。
- 384-step dense full-SFT 已完整运行；step 128 在独立 train-split heldout 上把 token CE
  从 `2.2415` 降到 `1.6664`，三份 model-only DCP 完整可追溯；step 256/384 的 heldout 回升，
  因而最佳 checkpoint 是 step 128，后续训练出现过拟合。
- step-128 统一下游已完成；没有观察到 dense SFT F1 恢复。对 SFT 模型，Q8 相对 dense
  仍通过 mean margin，frozen 相对 Q16 的量化增量近似无损且 state 再缩小 `3.59×`；
  完整 frozen-vs-dense 未通过 margin，损失主要来自 Q16 split/replay interface。

### 还不能说

- 不能说所有输入都严格无损或 bitwise logit exact；
- 不能把 validation 或 legacy test 当作新 adapter/policy 的最终 untouched test；
- 不能说 14x persistent compression 会带来 14x active-memory 降低；
- 不能说 Q-CoMem 当前 TTFT/TPOT 优于 full-prefix；
- 不能把 Interface LoRA 的部分恢复写成 overall 统计稳定、完全恢复或追平 dense；
- 不能把 answer-supervised LoRA B 的 `+0.005559` 点估计写成统计显著的总体
  质量恢复；也不能从 2 个 reference-yes 样本推广出广义 yes/no 能力改善；
- 不能隐藏 `101.8125 MiB` 共享 adapter，或把“第 5 个文档的 persistent
  state+adapter break-even”代换成模型总驻留、active-memory 或峰值显存收益；
- 不能把 step 128/disabled `1.06451×` TTFT 诊断当作严格性能结论；它是单次
  固定顺序，没有 ABBA/多轮重复；
- 不能说旧 merged-uncached quant LoRA 已恢复 frozen-static、mixed 或 minus-25%
  的下游质量；
- 不能说 mutable cached-two-stage quant/suffix-full 已可训练；
- 不能说 detached-document-cache 已可训练、已验证真实 cache immutability，或已有 query-side
  gradient/update；Trial `1832364` 在这些后置 gate 之前就已 backward 失败；
- 不能把 full-SFT 的 train-split heldout CE 改善写成 LongBench 生成 F1 改善；正式下游的
  SFT-vs-base 点估计反而为负且 CI 跨 0。不能说它优于 LoRA、恢复完整 split/replay，
  或成为 Q-CoMem cached suffix SFT；step 384 也不是最佳权重；
- 不能用旧 Trial `1830738` 的 direct sub-gate 冒充总 gate；新 Trial `1832356` 虽已通过三方
  总 gate，但不能说 COW 已降低 4k active peak 或改善 TTFT：实测 CUDA/NVML peak 与 TTFT 均为负；
  也不能把 `paged-cow-staging` 称为真正 PagedAttention/zero-copy serving；
- 不能把 Trial `1840009` 的 TF-eager compatibility negative 归因为 page-reuse layout
  错误或正确；该 run 未跑 validation/性能，没有 vLLM paged TTFT、TPOT 或显存 claim；
- 不能把 Trial `1840486` 的 cache 增量 peak 约减半写成总模型显存减半：absolute peak
  只差约 `83.64 MiB`；也不能外推 F1、多 query、ragged batch、Q8/Q4、NVML peak 或
  isolated kernel speedup；Trial `1840344` 只是 locale preflight 失败，不是 GPU 算法结果；
- 不能把 Trial `1840837` 的 simultaneous resident N 曲线写成并发 serving 或吞吐提升：
  N 个对象同时存活，但模型步为单流 round-major 顺序执行；`2960/240 MiB` 是 Q16
  full-attention pool，不是整个模型显存比值，也不能外推 aligned 4096、ragged、多文档、
  NVML、F1 或速度；
- 不能说 HYPIC-lite 已通过真实 H20 gate；
- 不能把 Apple thermal/swap invalid 的主矩阵 latency 放入正式性能主表。

## 10. 下一步建议顺序

1. Answer-supervised 156-module native-cache LoRA B 的训练和同 job downstream 已经完成。
   不用已消费的 validation 6–35 重选 checkpoint 或调参；若继续区分 answer objective、
   36-module warm start、120-module GDN surface 和 MLP 消融，必须使用新的 train-heldout/
   未消费评测数据，并把纯 cold-start 与 hybrid-init 分开。
2. 现有证据只支持定向 yes→no 修复与总体正点估计。下一质量阶段应在不触碰
   test-v2 的前提下扩大 answer-type/长答案定向 audit，并预注册足以区分小幅
   F1 改变的样本量；不把当前 2 个 yes 样本当作广义结论。
3. vLLM paged same-kernel fair v2 已完成单请求 correctness/ABBA；multi-fork resident 又完成
   同文档 N<=32 的 shared-pool 容量曲线，且全步/cross-N exact。下一 infra 实验不再重复
   解析驻留曲线，而应进入真实 concurrent/continuous batching、ragged、多文档回收复用与
   独立下游质量；继续把 HF compatibility、NVML、isolated kernel latency 与主 ownership
   对照分开。
4. COW 三方 correctness 与 4k active-memory/TTFT short 已完成；结果显示 dense staging 会把
   payload `14.024×` 稀释为 total resident `3.056×`，且进程 peak/TTFT 为负。下一轮只推进
   去掉 persistent dense template 的 append-only block/page cache 或真实 paged kernel，并继续
   以 total resident、paired CUDA/NVML peak 与 TTFT 联合验收。
5. 冻结 adapter、bit policy、checkpoint 和 runtime 后，只运行一次 LongBench test-v2，并报告 paired bootstrap CI、
   catastrophic regression、token agreement 与每数据集结果。
6. 按冷态 session 协议重采 Apple 512/2048/4096 正式性能，补齐统一的 persistent/active/TTFT/TPOT 口径。
7. 静态账本已完成；只有在接受其负面容量边界后，才运行 suffix composition/HYPIC-lite
   H20 ablation。必须继续分开报告 transition-only、seam KV 和 full suffix cache，并把
   Q4/Q8 compressed variant 标成新近似组合而非完整 HYPIC。

## 11. 运行与报告硬边界

- Apple 正式结果必须同时满足 `status=completed`、`formal_result_eligible=true`、
  `environment_assessment.reasons=[]`；
- 模型、数据 revision 与数据 SHA256 必须冻结；
- 同一性能配置要跨进程重复并报告 median/IQR/CV，不能挑最快值；
- persistent state、active request state、model weights 和 temporary workspace 必须分栏；
- process-shared adapter 必须与 per-document persistent state 分栏，并明确 break-even 是否排除
  model weights、active workspace 和 allocator reserve；
- TTFT、TPOT、Write、restore/dequantize、fork/materialization 必须分段计时；
- Q16 token gate 是 deployment pipeline 前置条件，但不能替代更大下游 validation/test；
- test-v2 在所有选择冻结前不得读取，运行后也不得据此回调 policy 或 checkpoint。

## 12. 当前仓库可追溯性缺口

- `results/runs.jsonl` 只覆盖早期 MLX smoke，不是统一 run registry；
- 早期 LoRA smoke、两条 200-step 与 Interface validation 仍合并在同一 Markdown 报告；
  Trial `1840023` 的 answer-supervised run 已有独立报告和 scientific-artifact ledger；
- 一些 QS YAML 是模板、retry 或待提交配置，文件名本身无法表达 scheduler 最终状态；
- 建议后续为每个远端 run 固定记录 `job_id/trial_id/status/commit/config/data_sha256/result_dir/report/formal_eligible`，
  避免再从多个报告与目录反推当前进展。
