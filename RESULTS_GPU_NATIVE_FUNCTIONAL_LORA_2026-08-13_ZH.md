# Native functional-cache LoRA 正式实验结果（2026-08-13）

## 结论

真实 Qwen3.5-35B-A3B 的 native functional-cache LoRA 训练路径已经完成端到端闭环：
8×H20 上的真实 backward、36 个 q/k/v/o LoRA 模块更新、cache version/rebind、
显存、heldout checkpoint 选择、functional→mutable 部署语义门禁和固定 60 条下游评估均已完成。
正式任务 `235749 / 1834056` 终态为 `Complete`。

这个实验给出两个不同层面的结论：

1. **训练基础设施已经成立。** step 1 的 8 个 rank 均有 72/72 个有限且非零的 LoRA
   梯度和更新；native functional cache 与标准 mutable 部署 caller 在 991 个 query positions
   上 top-1 一致率为 100%，平均 KL 仅 `8.50e-9`。
2. **本轮 LoRA 不能宣称恢复了下游精度。** 26 条 heldout 的 teacher-KL loss 从
   `0.22005` 降到 `0.16816`，但固定 60 条下游上，trained LoRA Q4 的平均 F1
   `0.52694` 低于 untrained Q4 的 `0.54237`，配对差值 `-0.01543`，95% CI
   `[-0.06411, +0.02211]`。它通过了预注册的 dense non-inferiority margin，但没有优于
   untrained Q4，且 2WikiMQA 与 Qasper 的方向相反。

因此，本轮最稳健的正结果仍是：**不加 LoRA 的 frozen Q4 状态将每请求 persistent state
从 34.68 MiB 降至 9.66 MiB（3.59×），60 条平均 F1 与 dense 仅差 `-0.00052`。**
LoRA 路线的可训练性已解决，但目标函数/数据选择仍需调整。

## 正式任务与输入

- Job / Trial：`235749 / 1834056`；终态 `Complete`；
- 页面：<https://qs2.devops.xiaohongshu.com/model/production/job/trial/235749/1834056>；
- 硬件：单节点 8×NVIDIA H20-3e，每卡 `143771 MiB`；
- 运行时间：`2026-08-13T05:16:57Z` 至 `05:31:49Z`；
- 模型：Qwen3.5-35B-A3B；
- 训练数据：410 条 domain rows，Qasper 256 + 2WikiMQA 154；
- heldout：26 条 domain rows，Qasper 12 + 2WikiMQA 14；
- 训练 view：document 最多 1536 tokens，query 不截断，本批 query 最长 115 tokens；
- 下游：Qasper、2WikiMQA 各 30 条，固定 source index `6--35`；
- validation SHA-256：`1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe`；
- `test_v2_used=false`，未读取 source index `68--99`；
- code ledger SHA-256：`1895ce50ca602777f0cc57b35902a223c91f6e957ecabfa8dba8cd1244213c3b`。

训练配置为 residual Q4、attention Q4、linear Q8、7 层 cache bits
`[8,8,8,4,8,8,8]`；LoRA rank 32、alpha 64，目标为 suffix 中 36 个
q/k/v/o projection。训练 128 步，learning rate `2e-5`，warmup 8，cosine decay。
初始化只复用 Interface step-200 LoRA 权重，SHA-256 为
`c93269e31d4e7a3ed990ceb9ab602e56234fdc21011685b842a90263e36fb2c3`；
optimizer/scheduler/step 均重置，历史 merged checkpoint 没有用于主结果。

## Step-0 与真实 Step-1 hard gate

step-0 checkpoint 仅由 rank 0 原子写入，随后 8 个 rank 分别加载并核对同一 SHA 和 payload。
其 SHA-256 为 `5e53f4a90facbf83ba41e4b7fdbe46a0460dc7a87ec9211491680c8c68a9ac9a`，
包含 72 个 LoRA tensors，不含 optimizer state，`optimizer_steps=0`。

| 门禁 | 结果 |
|---|---:|
| world size | 8 / 8 |
| 每 rank LoRA modules | 36 / 36 |
| 每 rank finite + nonzero gradient tensors | 72 / 72 |
| 每 rank finite + nonzero updated tensors | 72 / 72 |
| query positions expected = observed | 8 / 8 ranks |
| 原 cache tensor versions 不变 | 8 / 8 ranks |
| 新 cache paths 全部 rebind | 8 / 8 ranks |
| 最低显存余量 | 53,004,271,616 bytes（49.36 GiB） |
| step-1 hard gate | passed |

step-1 query positions 依次为 `[115,113,111,110,110,109,109,108]`，每个 rank
均逐项相等。门禁通过后在同一个任务内继续到 128 步，没有另建 smoke job。
本路径只声明 **multi-token document prefill + full query continuation** 的 autograd；
不声明 Qwen GDN single-token update 的 autograd 支持。

## Heldout checkpoint 选择

选择指标是 26 条样本的 example-equal mean top-k bidirectional KL；Q16 mutable teacher 与
Q4 native-functional student 使用完全相同的 document/query boundary。

| checkpoint | mean heldout KL loss | 相对 step 0 |
|---:|---:|---:|
| 0 | 0.2200457 | — |
| 64 | 0.1745157 | -0.0455300 |
| 128 | **0.1681635** | **-0.0518822（-23.58%）** |

最终选择 step 128，SHA-256 为
`a47cfba5e5a8c8e62674fbc0a0e3bd716c366f4d1380100eb42da1c2db903843`。

## Native training caller → mutable deployment caller 语义门禁

16 条 heldout 样本、991 个 query positions 的 all-position 结果：

| 指标 | 观测值 | 硬阈值 | 结果 |
|---|---:|---:|---|
| top-1 agreement | 1.0 | 1.0 | pass |
| mean functional→mutable KL | `8.5021e-9` | `≤1e-6` | pass |
| max position KL | `1.5344e-7` | 仅报告 | — |
| max absolute logit error | 0.0 | 仅报告 | — |
| cache gate | 16 / 16 samples | 全部通过 | pass |

这证明本轮 native functional-cache 训练 caller 与实际下游所用 standard mutable caller
在固定相同边界时等价；它修复了旧 merged-uncached LoRA 的训练/部署语义错位。

## 60 条 full-state 下游结果

每个配置均覆盖 Qasper 和 2WikiMQA source index `6--35`，共 60 条；所有对比按同一个
example 配对。F1 的置信区间为配对 bootstrap 95% CI。

| 配置 | mean F1 | Δ vs dense | 95% CI vs dense | mean persistent | 压缩 vs Q16 | median TTFT |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.542888 | 0 | `[0,0]` | — | — | 0.6499 s |
| Q16 replay | 0.546238 | +0.003350 | `[-0.02665,+0.04335]` | 34.683 MiB | 1.00× | 0.7029 s |
| frozen Q4，无 LoRA | 0.542368 | -0.000520 | `[-0.03371,+0.04263]` | 9.661 MiB | **3.590×** | 0.6657 s |
| frozen Q4，native LoRA | 0.526936 | -0.015952 | `[-0.06942,+0.03557]` | 9.661 MiB | **3.590×** | 0.6759 s |

预注册 non-inferiority margin 是整体 `-0.02`、每数据集 `-0.03`。trained LoRA 对 dense
整体为 `-0.01595`；2WikiMQA 为 `-0.02937`，Qasper 为 `-0.00254`，因此形式上通过门禁。
但该门禁只说明没有越过预设下限，不代表 LoRA 带来了提升。

更关键的配对比较是 trained LoRA 与完全相同 Q4 store 的 untrained control：

| 范围 | F1 Δ（LoRA - untrained Q4） | paired bootstrap 95% CI |
|---|---:|---:|
| 全部 60 条 | **-0.015432** | `[-0.064107,+0.022111]` |
| 2WikiMQA 30 条 | **-0.043810** | `[-0.138095,+0.026667]` |
| Qasper 30 条 | **+0.012946** | `[-0.003630,+0.038909]` |

三个区间都包含 0；当前样本量下不能声称显著提升或显著下降。但方向高度异质：Qasper
小幅改善，2WikiMQA 退化。LoRA 与 untrained Q4 的 prediction exact agreement 为
`0.8333`，共改变 10 / 60 个预测；其中 2WikiMQA source index 27、28 相对 dense 的
F1 分别下降 `-1.0`、`-0.7143`，构成 2 / 60 个预定义 catastrophic regressions。

## 显存与延迟解释

- Q16 mean persistent state：`36,367,872 bytes = 34.683 MiB`；
- Q4 mean persistent state：`10,130,160 bytes = 9.661 MiB`；
- Q16 / Q4：`3.5901×`；
- LoRA 参数：`24,772,608 bytes = 23.625 MiB`；
- step-128 checkpoint 文件：`74,496,491 bytes = 71.05 MiB`；
- adapter 加载中位数：`0.0909 s`；
- LoRA / untrained Q4 median TTFT：`1.0153×`，约增加 `10.2 ms`。

23.625 MiB adapter 是模型级、可跨请求共享的常驻权重，不是每个 document/session 都复制的
persistent state；因此不能把它逐请求加到 9.661 MiB 上。为了公平测 TTFT，dense/Q16/Q4
control 也加载并常驻同一个 adapter，只在非候选配置上禁用它。

## 应当怎样解释本轮 LoRA

teacher-KL heldout 与下游 F1 给出了不同信号：step 128 在同分布 KL 上持续改善，却没有改善
60 条生成式 F1。这说明当前 top-k token-distribution distillation objective 过度优化 teacher
对齐，尚未充分约束最终答案决策；domain mixture 也可能让 Qasper 收益覆盖不了 2WikiMQA
的退化。

下一轮建议保持已经通过的 native functional-cache 基础设施不变，只改训练研究变量：

1. 以数据集均衡 batch 或 loss reweight，避免 Qasper 256 / 2WikiMQA 154 的比例直接主导；
2. 将 heldout 选择从单一 KL 扩展为 KL + 任务答案 proxy，并按数据集分别报告；
3. 尝试更弱更新：更低 LR、更低 LoRA rank、adapter norm/输出漂移正则，及 step 32/64；
4. 加入具备真实 document/query boundary 的 general replay；没有真实边界的数据不伪造；
5. 下一轮仍先过 step-1、all-position semantic gate，再读固定 validation，继续不碰 test-v2。

在完成这些实验之前，论文中的准确表述应是：**native functional-cache LoRA 可训练且训练/部署
语义一致；本轮降低了 heldout KL，但没有在 60 条下游上优于 untrained Q4。Q4 状态本身实现
3.59× persistent-memory 压缩并保持近 dense F1。**

## 产物与审计说明

- 远端完整 run：
  `/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/native-lora-domain-128-20260813c/`；
- 本地精简审计包：`results/gpu-native-functional-lora-domain-128-20260813c/`；
- 机器可读摘要：`formal-run-audit.json`；
- 关键原始产物：`native-step1-hard-gate.json`、`heldout-selection.json`、
  `native-semantic-gate.json`、`replay_analysis.json`；
- checkpoint 只保存在远端；本地包保存其 SHA ledger，避免复制约 170 MiB 的三个 checkpoint。

checkpoint 内嵌的 `semantics.functional_cache_capability.capability_gate_passed=false` 是共享
metadata 中继承的历史、运行前声明，不是本 trial 的最终门禁结果。评估完成后若直接改 checkpoint
metadata 会使已冻结的 checkpoint SHA 和所有下游 provenance 失效，因此没有篡改 checkpoint。
本 trial 的权威 capability 证据是 `native-step1-hard-gate.json` 与
`native-semantic-gate.json`；后续实现应让新 checkpoint metadata 显式引用这两个外部门禁。

审计过程中还保留了两个非科学结果的 fail-closed trial：`1833988` 在训练前因 step-0 多 rank
写竞争风险被主动终止；`1834047` 因 code ledger 的相对/绝对路径摘要不一致在 4 秒内退出。
两者均没有 checkpoint，也没有进入训练，未混入本报告结果。
