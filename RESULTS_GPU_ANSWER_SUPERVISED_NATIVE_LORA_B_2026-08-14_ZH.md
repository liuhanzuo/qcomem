# Answer-supervised Native-cache LoRA B 正式实验（2026-08-14）

## 一句话结论

这次实验回答了一个比“训练 loss 是否下降”更严格的问题：在冻结的 Qwen3.5-35B-A3B、
frozen-static Q-CoMem 路径和同一批 60 条 validation 上，基于真实 answer boundary 的
task-balanced LoRA 训练，能否修复旧 step-0 warm start 的已知失败，同时保持压缩后的部署能力。

结果是一个**有针对性修复、但总体增益尚未显著**的正向信号：

- official-train heldout loss 从 step 0 的 `0.740831` 降到 step 128 的 `0.369624`，因此按
  预注册规则选择 step 128；
- frozen-static step 128 的 validation F1 为 `0.548575`，相对 adapter-disabled frozen-static
  的 paired delta 为 `+0.005559`，95% CI `[-0.019821, +0.030127]`，跨 0；
- 旧 warm-start step 0 在 2WikiMQA 的两个 reference-yes 样本中有 `1/2` 预测成 no；
  step 64 和 step 128 都恢复为 `2/2` yes，已知 yes→no 失败在这两个审计样本上消失；
- frozen-static 每文档 persistent state 为 `9.6609 MiB`，Q16 为 `34.6831 MiB`，小
  `3.59×`；但 answer-LoRA 另有每模型进程共享的 `101.8125 MiB` FP32 adapter，不能把它
  隐藏进“每文档 9.66 MiB”的口径；
- 只计算 persistent state + shared adapter 时，break-even 为 `4.0689` 个常驻文档，
  即从第 **5** 个文档起，Q4 frozen-static + adapter 的总增量驻留低于 Q16；
- step 128 相对 adapter-disabled 的 median TTFT 为 `1.06451×`，但这是一次固定顺序运行，
  没有 ABBA / 重复轮次，只能作为诊断，不能作为严格性能 claim。

因此，当前最准确的表述是：**answer supervision 修复了已知 answer-type 失败，并把总体下游
点估计从 step 0 的负值推到 step 128 的小幅正值；60 条 validation 尚不足以证明总体能力提升。**

## 正式任务与冻结身份

- QS：Job `237290` / Trial `1840023`；
- 页面：<https://qs2.devops.xiaohongshu.com/model/production/job/trial/237290/1840023>；
- 终态：`Complete`，执行时间 `0d00h29m50s`；
- 资源：queue `385` / cloud `6` / cluster `53` / package `183`，单节点 `8 × H20-141G`；
- 环境：PyTorch `2.11.0+cu129`、Transformers `5.14.1`；
- 模型：冻结的 post-trained `Qwen3.5-35B-A3B-59d61f3`；
- 正式 launcher：`12:32:24` 写入 `00_start`，`12:49:25` 写入 `99_done`；
- 训练结束：`02_training_complete`，时间 `12:46:20`；
- 下游结束：`05_downstream_complete`，时间 `12:49:25`；
- 没有 `FAILED` / `FAILED_PHASE`，没有迁移、重提或 restart；
- 提交前远端 focused tests `17/17` 通过；同 job preflight 又运行一次并 `17/17` 通过。

关键冻结 SHA256：

| 对象 | SHA256 |
|---|---|
| 16-file code ledger | `e3df8eec6e1eead01157fe15aeaa3a3cd7b8fd216f4a2b88243102012e947185` |
| YAML | `98836ea2e7ac102d2caefd5952ba1dd62c8e8a322f8b01d626a0bc93977fcf5c` |
| static audit | `d85ed21ef68c715af075b18f8520b1b71aff53a4810fb10f612d6185f663c906` |
| train data | `3c67f6afe30eec191fd23446de5a8bf7282abea709f1b3ad42dd26f25107203c` |
| official-train heldout | `af01ef2153e2143af1848ba5bd78781030d1c81000ef734cc847f0fd37baa9a5` |
| data manifest | `c67a382517ccda2a858b6a9646efb1fd89427c1b4b76f8ee6cc4e3e091e2354b` |
| independent data audit | `5cee7230c29db385ac06cf228106410f939e4960ec4277d68c63e245a4be5747` |
| validation parent | `1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe` |
| warm-start step 0 | `5e53f4a90facbf83ba41e4b7fdbe46a0460dc7a87ec9211491680c8c68a9ac9a` |
| model artifact ledger | `fa050ef64c76caaa353223541c6ad8b80be8a5f6f5c11430db2d7d4f2c4dfb5c` |
| model weight ledger | `a0352fd3fd47b4edcebf3269b5f8745490d3defb9eaedf2a4c4dc8ccae32ddf2` |

LongBench test-v2 的冻结 digest 被 launcher 与 runner 显式拒绝；本轮没有读取 test-v2
source `68–99`。

## 训练协议

### 数据与 schedule

主实验只使用具有真实 `deployment_boundary` 的 domain 数据，不为 Tulu/general rows 伪造
document/query 边界：

| split | QASPER | 2WikiMQA | 合计 |
|---|---:|---:|---:|
| train | 256 | 154 | 410 |
| independent official-train heldout | 12 | 14 | 26 |

训练共 128 个 global steps、8 ranks。每个 step 恰好 4 个 QASPER + 4 个 2WikiMQA，每个
rank 一个样本，因此两个任务的优化 mass 都是 `0.5`，没有再乘 inverse-occurrence weight，
也没有按 answer token 数给长答案更高权重。完整 schedule 中：

- 256 个 QASPER 每条恰好出现 2 次；
- 154 个 2WikiMQA 中，104 条出现 3 次、50 条出现 4 次；
- schedule SHA256 为
  `10905c91782d6c8e67fbdac79fb2fb51225732c5be3b9040ca2422a360c79345`。

### Answer objective

每条样本按以下边界执行：

1. document prefill；
2. query + `answer[:-1]` 作为 multi-token continuation；
3. causal shift 后，只在 answer + EOS 位置计算 loss：最后一个 query hidden 预测
   `answer[0]`，其余 answer hidden 预测后续 token 和 EOS。

loss 为：

`0.45 × hard CE + 0.35 × frozen dense teacher top-k/tail KL + 0.20 × hidden cosine`

三项都先做 answer-position mean，再做 example mean。student full-vocab projection 按 32 个位置
分块，并使用 non-reentrant activation checkpoint 在 backward 重算，避免同时保留全部 answer
logits。teacher cache 在 adapter 安装和 optimizer 创建前生成，共 436 条、5992 个 answer/EOS
位置，8 个 shard 的 ledger 全部通过。

### Adapter surface 与初始化归因

suffix depth 为 7，实际覆盖：

- 36 个 full-attention 关键 projection；
- 120 个 GDN 关键 projection；
- 合计 156 modules、312 个 A/B tensors、26,689,536 个 FP32 trainable parameters；
- MLP 没有覆盖，是明确排除项，不能把本实验称为“全 suffix 所有线性层”。

FP32 adapter 参数为 `106,758,144 B`；参数 + gradient + Adam m/v 静态估计为
`427,032,576 B`。

初始化不是纯 cold start：36 个 full-attention modules 来自旧 step-0 warm start，且该旧点在
已知 downstream 上是负向点估计；新增的 120 个 GDN modules 为标准 LoRA cold start（A
Kaiming、B 为 0）。所以本实验检验的是：**answer supervision + surface 从 36 扩到 156，
能否修复 warm-started system**，不是纯 cold-start LoRA B。

## 同 job 硬门禁与显存边界

teacher、adapter、optimizer 和正式训练都在同一个 8-GPU job 内完成，没有另提 GPU smoke。

Step 1：

- 8 ranks loss finite；
- 312 个 adapter gradient/update 都 finite；
- warm-start full-attention 的 72 tensors 全部 nonzero grad/update；
- cold GDN-A 第一轮允许合法的 0 grad/update；GDN-B 的 120 tensors 全部 nonzero；
- native functional cache、multi-token continuation、cache rebind、原 cache path/version 不变通过；
- 312 个 adapter 参数及 936 个 Adam states 都为 FP32；
- 8 ranks 最小 reserved headroom 为 `6,516,703,232 B`，高于预注册 `4 GiB` 门槛。

Step 2：

- full-attention、GDN-A、GDN-B 共 312 tensors 全部 finite 且 nonzero grad/update；
- native functional cache 与 FP32 optimizer 继续通过；
- **Step 2 协议没有再次声明 4 GiB reserved-headroom gate。** 实测 8 ranks 最小 reserved
  headroom 仅 `1,426,915,328 B`；最小 allocated-headroom 仍为 `7,318,294,528 B`。训练随后
  完整跑到 step 128，没有 OOM，但不能把 step 2 误写成“也通过 4 GiB headroom 门禁”。

## Heldout 选择结果

checkpoint 始终保存并评估 `0 / 64 / 128`，只用独立 official-train heldout 选 best：

| step | overall loss | CE | KL | hidden | 相对 step 0 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.740831 | 1.629324 | 0.016250 | 0.009738 | — |
| 64 | 0.374557 | 0.610883 | 0.261151 | 0.041282 | -49.44% |
| 128 | **0.369624** | 0.611924 | 0.247572 | 0.038040 | **-50.11%** |

因此选择 step 128；checkpoint SHA256 为
`2d5eea2dce244cfe4e4f9431946dd41790a3aa87bbeaf3c24b89f8b7a005f960`。
validation `6–35` 没有参与 checkpoint 选择，且三个 step 无论 validation 表现如何都必须完整
评估，不能在事后改选 step 64。

## Teacher-forced chunk 与 token-by-token 诊断

训练使用 `query + answer[:-1]` 的一个 multi-token block，而真实部署是 query multi-token 后再
逐 token decode。Qwen GDN 对 chunk boundary 可能数值敏感，因此本轮只做诊断，不宣称两者
等价。

在 step 128、26 条 heldout、206 个 answer positions 上：

- top-1 divergence：`0/206`；
- example-mean whole→token KL：`0.000507959`；
- 最大 KL：`0.0158421`；
- 最大 absolute logit difference：`5.53906`。

这说明当前 heldout 上 greedy top-1 稳定，但非零 KL 和 logit difference 仍存在；结果不能升级
成 chunk-boundary 数值等价证明。

## 60 条 full-state validation 结果

checkpoint 在读取 validation 前已经冻结。随后在同一 8-GPU job、同一 caller、相同 prompt、
greedy decoding 下，完整运行 6 个条件；每个条件都是 QASPER 30 条 + 2WikiMQA 30 条：

| 条件 | overall F1 | QASPER | 2WikiMQA |
|---|---:|---:|---:|
| dense adapter-disabled control | 0.541601 | 0.433520 | 0.649683 |
| Q16 adapter-disabled control | 0.546238 | 0.422793 | 0.669683 |
| frozen-static adapter-disabled | 0.543016 | 0.421905 | 0.664127 |
| frozen-static LoRA step 0 | 0.533925 | 0.427532 | 0.640317 |
| frozen-static LoRA step 64 | 0.544805 | 0.435958 | 0.653651 |
| frozen-static LoRA step 128 | **0.548575** | **0.436357** | 0.660794 |

关键 paired comparisons（10,000 次、固定 seed 的 paired bootstrap）：

| comparison | F1 delta | 95% CI | 解释 |
|---|---:|---:|---|
| step 0 − frozen disabled | -0.009091 | [-0.062497, +0.039884] | 旧 warm start 为负点估计 |
| step 64 − frozen disabled | +0.001789 | [-0.030617, +0.029368] | 点估计转正，CI 跨 0 |
| step 128 − frozen disabled | **+0.005559** | **[-0.019821, +0.030127]** | selected 结果；CI 跨 0 |
| step 64 − step 0 | +0.010880 | [-0.038837, +0.059981] | 修复方向为正，CI 跨 0 |
| step 128 − step 64 | +0.003771 | [-0.003247, +0.012912] | 128 略高，CI 跨 0 |
| step 128 − dense disabled | +0.006974 | [-0.028917, +0.051977] | CI 跨 0 |
| step 128 − Q16 disabled | +0.002337 | [-0.027538, +0.030324] | CI 跨 0 |

selected step 128 相对 frozen disabled 的分数据集结果：

- QASPER：`+0.014452`，95% CI `[-0.008175, +0.044125]`；
- 2WikiMQA：`-0.003333`，95% CI `[-0.050000, +0.040000]`。

所以“总体恢复”仍只是点估计；目前更明确的收益出现在 QASPER，2WikiMQA 平均 F1 基本持平。

## 已知 yes→no 失败是否修复

2WikiMQA 的 30 条 validation 中只有 2 条 reference answer type 为 yes，2 条为 no，另外 26 条
为 entity。这是一个小但预先关注的定向审计：

| 条件 | reference-yes→yes | reference-yes→no |
|---|---:|---:|
| frozen disabled | 2 | 0 |
| LoRA step 0 | 1 | **1** |
| LoRA step 64 | 2 | 0 |
| LoRA step 128 | 2 | 0 |

step 128 相对 disabled 有 6/30 prediction text 改变，但 prediction type 改变为 0。也就是说，
answer supervision 确实消除了旧 step-0 的已知 yes→no 错误；但 reference-yes 只有 2 条，
不能从这里推出广义 yes/no 能力已经显著提升。

## Persistent state、shared adapter 与多文档 break-even

所有 Q4/Q16 标签只描述 persistent document state，不描述模型权重量化：

| 条件 | 每文档 persistent bytes | MiB | 相对 Q16 |
|---|---:|---:|---:|
| Q16 adapter-disabled | 36,367,872 | 34.6831 | 1.00× |
| frozen-static Q4/mixed-bit | 10,130,160 | 9.6609 | **3.5901× smaller** |

per-document state 减少 `72.15%`。但 step 128 还需要一个共享 FP32 adapter：

- parameters：26,689,536；
- resident bytes：106,758,144；
- resident MiB：101.8125；
- scope：每个模型进程一份，可被同一进程内的多个文档摊销。

只比较 persistent state + adapter，不计模型权重、active working set、allocator reserve 或临时
activation，break-even 为：

`106,758,144 / (36,367,872 - 10,130,160) = 4.0689`

因此整数容量点为第 5 个常驻文档：

| 常驻文档数 | frozen + shared adapter | Q16 | frozen − Q16 |
|---:|---:|---:|---:|
| 1 | 116,888,304 B | 36,367,872 B | +80,520,432 B |
| 4 | 147,278,784 B | 145,471,488 B | +1,807,296 B |
| 5 | **157,408,944 B** | **181,839,360 B** | **-24,430,416 B** |
| 10 | 208,059,744 B | 363,678,720 B | -155,618,976 B |

这个 break-even 只支持“多文档 persistent capacity”口径；不能拿它替代端到端峰值显存或单文档
部署可行性结论。

## TTFT 诊断，不作严格性能 claim

从 48 个原始 shards 的 60 条 paired rows 独立重算 median TTFT：

| 条件 | median TTFT |
|---|---:|
| dense disabled | 0.648663 s |
| Q16 disabled | 0.701073 s |
| frozen disabled | 0.663103 s |
| frozen LoRA step 0 | 0.706184 s |
| frozen LoRA step 64 | 0.706394 s |
| frozen LoRA step 128 | 0.705880 s |

step 128 / frozen disabled 为 `1.064510×`，即点估计慢约 `6.45%`。但六个 condition 是同一
进程中的固定顺序单次运行，没有 ABBA、随机化顺序、重复轮次、功率/温度稳定区间和置信区间；
因此只能报告为诊断，不能写成 adapter 带来确定的 6.45% latency overhead。

## 能说与不能说

### 当前证据支持

1. answer-only + teacher preservation 的 native-cache LoRA 能在 official-train heldout 上把优化
   objective 降低约 50%，并在预注册规则下选择 step 128。
2. step 0 的已知 2Wiki yes→no 失败在 step 64/128 上消失。
3. validation 总体 F1 点估计从 step 0 相对 disabled 的 `-0.00909`，变为 step 128 的
   `+0.00556`。
4. frozen-static per-document state 为 Q16 的 `1/3.59`；计入共享 adapter 后，第 5 个常驻文档
   开始体现总 persistent-capacity 优势。
5. native functional-cache 多 token 训练、156-module surface、FP32 optimizer、step1/2 更新门禁
   和同 job 0/64/128 full-state attribution 均按协议完成。

### 当前证据不支持

1. 不能说下游总体能力已经显著提升；selected delta 的 95% CI 跨 0。
2. 不能用 validation 重新选择 step、bit policy、loss weight 或 adapter surface。该 validation
   在此前实验已经被消费，本轮只允许 post-selection attribution；继续调参会造成泄漏。
3. 不能说训练的 whole answer block 与逐 token decode 数值等价；只能说 206 个 heldout
   positions 的 top-1 没有分叉。
4. 不能把 `9.66 MiB` 当成模型总增量；共享 adapter 是 `101.81 MiB`，active memory 另计。
5. 不能把单次固定顺序 TTFT 当成严格性能结果。
6. 不能把本实验解释为纯 cold-start LoRA，也不能把未覆盖 MLP 的 surface 称为全 suffix。

## 本地归档与完整性

本地保留两部分：

- 原始 run 的完整字节镜像：
  `results/gpu-answer-supervised-native-lora-b-20260814a/`；
- 冻结 code snapshot、YAML、static audit、dry-run 和唯一 submit receipt：
  `results/gpu-answer-supervised-native-lora-b-20260814a-release-evidence/`。

原始 run 镜像与远端逐字节树复核结果：

- 文件数：2792；
- 总 bytes：418,349,317；
- 全树 canonical SHA256：
  `27c6fd1318fa5088d24d95a97e06b34cedf932288373c9cd6189b370a9d43370`；
- 排除运行环境 `pycache/` 后：105 个原始科学证据文件、350,033,153 B；
- 非-pycache canonical SHA256：
  `09ec7408285d08863b568900486b48e1f625b1f6f0dc7386a1167de9a6c44b09`。

完整镜像保留 `pycache/` 和 release code snapshot 中的 `__pycache__/`，只是为了
byte-for-byte archive；它们不属于 scientific artifact。独立 scientific ledger 位于：

`results/gpu-answer-supervised-native-lora-b-20260814a.scientific-artifacts.sha256`

它覆盖 105 个非-pycache 原始 run 文件、20 个非-pycache release evidence 文件，以及本报告，
共 126 项；ledger 自身不递归包含自己。

关键 ledger 复核计数：

- frozen code：16/16；运行前、训练前、downstream 前共记录 48 个 `OK`；
- model artifacts：6 个文件 × 3 次，共 18 个 `OK`；
- model weights：14 个 shards × 2 次，共 28 个 `OK`；
- teacher：8 个 rank shards + manifest，共 9/9；
- checkpoints：step 0/64/128，共 3/3；
- downstream：8 ranks × 6 conditions = 48 shards；每个 condition 60/60 rows，aggregate
  与原始 row mean F1 一致。

Trial 完成并拉取归档后，又从远端冻结路径逐项重算了 code 16 项、model artifacts 6 项、
input/init/validation 6 项、checkpoints 3 项、teacher 9 项和 14 个完整 model-weight shards；
全部为 `OK`，不是只依赖运行时留下的日志文本。

主分析文件：

`results/gpu-answer-supervised-native-lora-b-20260814a/downstream/answer-full-state-downstream-analysis.json`

其 SHA256 为：

`eb4392ea0561e3ee2449b1df3a2448cef427ef59fb0a20eeea7af539ba524bb6`

## 下一步

validation 已经消费，不能再围绕这 60 条继续调 loss 或 checkpoint。下一步应先冻结一个新的、
未接触 evaluation plan：

1. 在 official-train 内增加独立 answer-type/evidence-retention diagnostics，而不是回看 validation
   调参；
2. 若要判断总体提升，使用尚未消费、与 test-v2 隔离的新 development/evaluation split，或等所有
   算法选择完全冻结后再做唯一 final test；
3. 增加多文档常驻容量实验，直接验证 1/4/5/10 文档的 measured resident memory，并把模型、
   adapter、persistent state、active working set 分账；
4. 对 TTFT 使用 ABBA / randomized order、多轮重复、功率与温度控制，再给置信区间；
5. 另做纯 cold-start 156-module 和 MLP-surface ablation，拆开 warm-start repair、GDN surface
   扩展与 answer objective 的贡献。
