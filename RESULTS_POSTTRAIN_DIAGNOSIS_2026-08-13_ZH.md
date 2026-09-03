# Q-CoMem 后训练结果诊断与下一步预注册实验（2026-08-13）

## 1. 结论摘要

本次审计覆盖两条最新结果：

1. native functional LoRA 的内部 heldout KL 从 `0.220046` 降到 `0.168163`，相对改善 `23.58%`，但固定 60 条 full-state 下游评测相对 adapter-disabled Q4 的平均 F1 为 `-0.015432`；
2. dense Full SFT 的 dense overall F1 点估计为 `+0.013524`，但 Qasper 为 `-0.020730`、2WikiMQA 为 `+0.047778`；SFT 后 Q16 相对 base Q16 为 `-0.006323`，相对同一 SFT dense 为 `-0.016497`。

审计结论如下。

- LoRA 的首要问题是训练目标与生成任务失配：当前训练数据明确不含 answer/EOS，loss 只约束 query/prompt token 上的 Q16 top-k KL。因此 `23.58%` 的 KL 改善不能推出回答 F1 改善。
- LoRA 结果还存在基线混淆：step0 是 Interface LoRA warm-start，最终下游 reference 却是 adapter-disabled Q4。由于 step0/64 都没有下游 shard，当前不能判断负增益来自 warm-start、native 训练还是 checkpoint 选择。
- LoRA 的 adapter 覆盖是合理的次级瓶颈：当前 36 个模块只覆盖 suffix 中 9 个 full-attention 层的 q/k/v/o，没有覆盖 24 个 GDN 层和 33 个 suffix MLP。
- Dense SFT 的主要问题是 task 配额、单参考 CE 和 checkpoint 选择造成的输出风格偏移；其训练 forward 完全不含 Q16/Q8/CoMem 状态，因此不能期待 Q16 稳定性随 dense CE 自动改善。
- 所有最新下游差值的 bootstrap 置信区间均跨过 0；当前结果应描述为机制线索和点估计，而不能描述为统计显著的提升或回退。

本文件只使用官方 train 数据、内部 heldout 和已冻结的 LongBench validation source 6--35 artifact。没有读取或使用 test-v2/source 68--99。

## 2. Native functional LoRA 证据

### 2.1 训练目标与生成任务不一致

训练 view 的构造函数明确标注为 `answer-free document/query training view`，并强制：

- `document + query == input_ids[:first_target]`；
- query 中不得出现 answer 或 EOS；
- 输出记录只保存 `document_ids` 和 `query_ids`。

对应代码：`gpu/qcomem_native_lora_protocol.py:64-165`。

实际训练中，teacher 和 student 只产生 query 长度的 logits，随后在 teacher top-k support 上计算双向 KL。没有 assistant answer token，也没有 ground-truth answer CE：

- Q16 teacher：`gpu/qcomem_lora.py:1148-1169`；
- query-only top-k target 和 student loss：`gpu/qcomem_lora.py:1187-1265`。

因此 heldout loss 的含义是“Q4+LoRA 在问题 prompt token 上更接近 Q16”，不是“回答更准确”。当前 checkpoint 选择指标 `example_equal_mean_topk_bidirectional_kl` 也延续了同一 surrogate 失配。

### 2.2 step0 基线混淆与缺失归因实验

训练 metadata 表明 step0 从以下 Interface checkpoint warm-start：

```text
lora-interface-200-agent-20260812b/checkpoint-000200.pt
SHA256 c93269e31d4e7a3ed990ceb9ab602e56234fdc21011685b842a90263e36fb2c3
```

native run 中保存了 step0、64、128：

```text
checkpoint-000000.pt
checkpoint-000064.pt
checkpoint-000128.pt
```

但 full-state downstream 只运行了最终 step128。现有 downstream shard 全部绑定：

```text
SHA256 a47cfba5e5a8c8e62674fbc0a0e3bd716c366f4d1380100eb42da1c2db903843
```

step0 和 step64 没有 adapter-enabled downstream shard。当前 `-0.015432` 是“最终 LoRA”相对“adapter-disabled Q4”的差值，不是 step128 相对 step0 的训练增益。这个缺口必须先补齐，才能做因果归因。

### 2.3 2WikiMQA 逐样本与类别变化

固定 30 条 2WikiMQA 的 reference 类别为：

| 类别 | 数量 |
|---|---:|
| 实体/自由文本 | 26 |
| yes | 2 |
| no | 2 |

预测类别变化：

| 配置 | 实体/自由文本 | yes | no | 弃答 |
|---|---:|---:|---:|---:|
| adapter-disabled Q4 | 25 | 2 | 2 | 1 |
| Q4 + LoRA step128 | 26 | 1 | 3 | 0 |

30 条中只有 5 条输出改变，F1 净损主要由三条决定：

| source index | reference | disabled Q4 | LoRA | F1 变化 |
|---:|---|---|---|---:|
| 27 | yes | Yes | No | `-1.000` |
| 28 | Leustach Rátót | Leustach Rátót | Rathold (I) from the kindred Rátót | `-0.714` |
| 12 | Guy II, Count of Soissons | John I, Count of Soissons | Guy II, Count of Soissons | `+0.400` |

另外两条只是从一个错误答案变为另一个错误答案，F1 均保持 0。按 reference 类别汇总：

- yes：`1.000 -> 0.500`；
- no：`1.000 -> 1.000`；
- 实体/自由文本：`0.612454 -> 0.600366`。

这不是广泛的能力崩溃，而是少数 greedy decision flip。训练 2WikiMQA 的 154 条样本中 yes 只有 15 条、no 有 34 条，方向上与 `yes -> no` 的变化一致，但评测样本过少，不能单独据此断言类别偏差已经被统计确认。

### 2.4 数据与 adapter 覆盖

native LoRA train 共 410 条：

- Qasper 256，占 `62.44%`；
- 2WikiMQA 154，占 `37.56%`；
- 266/410 文档被截断到 1536 token；
- 下游评测最大输入为 4096 token。

当前 LoRA 只安装于 layers 7、11、15、19、23、27、31、35、39 的 q/k/v/o，共 36 个模块和 6,193,152 个参数。suffix layers 7--39 中：

- 24 个 GDN 层的 `in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj` 未覆盖；
- 33 个 MLP 的 `gate_proj/up_proj/down_proj` 未覆盖。

因此 adapter 可以在 9 个 full-attention block 中做补偿，但不能直接调整大多数 token-mixer 和所有 suffix MLP。这是次于目标失配的覆盖限制。

### 2.5 统计边界

最终 LoRA 相对 disabled Q4：

| 范围 | F1 delta | paired bootstrap 95% CI |
|---|---:|---|
| overall 60 | `-0.015432` | `[-0.064107, 0.022111]` |
| Qasper 30 | `+0.012946` | `[-0.003630, 0.038909]` |
| 2WikiMQA 30 | `-0.043810` | `[-0.138095, 0.026667]` |

全部区间跨 0。报告时必须保留这一不确定性。

## 3. Dense Full SFT 与 Q16 诊断

### 3.1 数据与目标配额

训练 1024 条的 example 配额为：

| 数据/stratum | 样本数 |
|---|---:|
| Qasper domain | 256 |
| 2WikiMQA domain | 154 |
| Tulu general replay | 307 |
| Tulu teacher preservation | 307 |

对应常量见 `gpu/build_deployment_aware_sft.py:45-55`。全局按 example 等权，不按 target token 等权。

冻结 teacher KL/hidden preservation 只用于 307 条 `teacher_preservation` Tulu 样本。Qasper 和 2WikiMQA domain 样本只用 hard CE。因此 preservation 约束没有直接保护两个最终评测任务。

### 3.2 输出风格偏移

Qasper dense 输出长度发生明显收缩：

| 模型 | 平均生成 token | 中位数生成 token | `unanswerable` 数量 |
|---|---:|---:|---:|
| base dense | 12.83 | 6.5 | 9 |
| SFT dense | 7.37 | 3.0 | 10 |

这会提高部分短抽取回答的 token F1，但会损害列表、多项指标或需要完整描述的答案。逐样本中同时存在：

- 更简洁后 F1 提升的回答；
- 正确解释被压成 `unanswerable` 的 `-0.552` 回退；
- 多项答案因为删减内容而回退。

因此同源 heldout CE 从 `1.503768` 降至 `0.756249` 并不足以预测 LongBench 生成 F1。checkpoint 只按 heldout CE 选择 step128，也是目标失配的一部分。

### 3.3 Dense 训练没有约束 Q16

Dense Full SFT forward 使用：

```python
use_cache=False
```

训练过程中没有 Q16、Q8 或 frozen-static CoMem state。对应代码为 `gpu/deployment_aware_sft.py:379-440`。所以训练只优化 dense teacher-forced token likelihood，不能保证量化 persistent state 后的 greedy generation 与 dense 单调接近。

SFT 模型的 Q16 相对同一 SFT dense 只有 6/60 条预测变化，但其中包含少数大幅 exact-answer flip，例如：

- `Cunimund -> Alboin`，F1 `-1.0`；
- `Leustach Rátót -> Rathold Rátót`，F1 `-0.5`；
- 另有一条错误实体变为正确实体，F1 `+0.4`。

Q8 只改变 3/60 条。量化位宽与最终离散 F1 不要求单调；这里应描述为“greedy 生成边界对 state perturbation 敏感”，而不是 Q16 数值实现不稳定。

### 3.4 点估计与置信区间

| 对比 | overall delta | Qasper delta | 2WikiMQA delta | overall 95% CI |
|---|---:|---:|---:|---|
| SFT dense vs base dense | `+0.013524` | `-0.020730` | `+0.047778` | `[-0.037817, 0.073927]` |
| SFT Q16 vs base Q16 | `-0.006323` | `-0.003757` | `-0.008889` | `[-0.040617, 0.026690]` |
| SFT Q16 vs SFT dense | `-0.016497` | `+0.003672` | `-0.036667` | `[-0.060327, 0.015496]` |

所有 overall CI 均跨 0。

### 3.5 metadata 错误提醒

`artifacts/metadata.json` 中的 `parameter_plan.distillation_used` 被记录为 `false`，但同一 artifact 同时证明：

- 生成了 307 条 frozen teacher targets；
- teacher target 共覆盖 68,814 个 assistant target positions；
- teacher-preservation loss 使用 `0.45 hard CE + 0.35 KL + 0.20 hidden cosine`；
- 对应 teacher shard 已在 optimizer 创建前冻结。

实际训练代码也会对 `teacher_required` 样本加载 teacher targets。因此 `distillation_used=false` 是 metadata 语义错误，不代表训练没有 distillation。正式报告或发布 artifact 前应更正为至少以下之一：

```text
distillation_used: true
distillation_scope: teacher_preservation_stratum_only
distillation_records: 307
```

不要修改已冻结结果本身；应在新 manifest 或勘误记录中保留原值、指出错误并给出修正解释。

## 4. 预注册实验 A：现有 checkpoint 下游归因

### 4.1 目的

不重新训练，只补齐 step0/64/128 的相同 adapter-enabled downstream，区分 warm-start、native 更新和 checkpoint surrogate 三种可能根因。

### 4.2 固定配置

- 数据：现有冻结 LongBench validation source 6--35，共 60 条；不读取其他 source；
- 配置：`replay-d7-frozen-static`；
- 解码：现有 LongBench official prompt、greedy decoding 和 dataset max-new-tokens；
- 比较组：
  1. adapter-disabled Q4；
  2. Q4 + `checkpoint-000000.pt`；
  3. Q4 + `checkpoint-000064.pt`；
  4. Q4 + `checkpoint-000128.pt`；
- 每个 checkpoint 必须记录 SHA256，并验证 36 个 LoRA 模块均实际启用；
- 不允许按该 60 条重新选择或修改 checkpoint。

### 4.3 预注册指标

- overall、Qasper、2WikiMQA mean F1 delta vs adapter-disabled Q4；
- paired bootstrap 95% CI；
- prediction exact agreement 和 token-sequence agreement；
- 2WikiMQA yes/no/entity/弃答 confusion；
- catastrophic regression rate，阈值 `sample F1 delta <= -0.5`。

### 4.4 预注册解释规则

- step0 已显著为负，而 step64/128 没有继续下降：Interface warm-start 是主要来源；
- step0 接近 disabled，但 step64、128 单调下降：answer-free query KL 是主要来源；
- step64 优于 step128：heldout query KL checkpoint selection 或后半程训练过度；
- 三个 checkpoint 都接近 disabled 且 CI 大范围跨 0：现有 60 条不足以支持有害或有效结论。

该实验不产生新的训练消耗，是进入下一次训练前的强制归因 gate。

## 5. 预注册实验 B：answer-supervised task-balanced expanded LoRA

### 5.1 目标

直接优化部署时的 assistant answer generation，并同时修复：

- answer-free surrogate；
- Qasper/2WikiMQA 配额不平衡；
- 2WikiMQA yes/no 偏斜；
- GDN/MLP adapter 覆盖不足；
- Interface warm-start 混淆。

### 5.2 数据配额

使用官方 train-only 数据，建立 hash 和 document-family 不相交的 train/heldout：

#### Train：512 条

- Qasper 256；
- 2WikiMQA 256。

2WikiMQA 固定配额：

- yes 32；
- no 32；
- 实体/自由文本 192。

Qasper 按完整 assistant target token 数分层：

- `<=4` token：80；
- `5--16` token：88；
- `>16` token：88；
- yes/no/unanswerable 合计不得超过 20%。

#### Heldout：128 条

- Qasper 64；
- 2WikiMQA 64；
- 与 train 的 source ID、document family、normalized prompt/context hash 均为零交集；
- 2WikiMQA heldout 的 yes/no 各至少 16 条，以获得可解释的类别准确率。

文档仍先使用当前可证明可训练的 1536-token 上限，并保留 evidence-first + head/middle/tail 截断；query 和 answer 不允许截断。这样先隔离目标与 adapter 覆盖，不同时引入 3K/4K 激活内存变量。

### 5.3 训练执行语义

- document 在同一 native functional cache boundary 上 prefill；
- continuation 改为 `query_ids + shifted_answer_prefix_ids`；
- ground-truth loss 只施加在 assistant answer positions；
- teacher 为 adapter-disabled Q16 replay；
- student 为当前 Q4/Q8 mixed frozen-static state + LoRA；
- teacher 和 student 使用相同 document/query/answer caller boundary；
- LoRA 从严格零增量初始化，不使用 Interface checkpoint warm-start。

### 5.4 Loss

每个 assistant answer position 使用：

```text
L = 0.50 * hard_answer_CE
  + 0.40 * Q16_top64_plus_tail_KL
  + 0.10 * normalized_hidden_cosine_distance
```

KL 必须包含 teacher top64 的真实 log-probability 和一个 tail probability bucket；不能沿用当前“只在 top-k 内重新归一化”的 query KL。

### 5.5 Adapter 覆盖

只训练 suffix layers 7--39：

- 9 个 full-attention 层：`q_proj/k_proj/v_proj/o_proj`，36 个模块；
- 24 个 GDN 层：`in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj`，120 个模块；
- 33 个 MLP：`gate_proj/up_proj/down_proj`，99 个模块；
- 合计 255 个 LoRA 模块。

固定配置：

- rank 16；
- alpha 32；
- dropout 0；
- 预计 trainable 参数 23,406,336；
- hard cap 25,000,000，超出即 fail closed。

### 5.6 优化器与步数

- global batch：8；
- steps：128，相当于 512 条 train 两遍；
- learning rate：`1e-5`；
- warmup：8 steps；
- schedule：cosine-to-zero；
- weight decay：0；
- max grad norm：1.0；
- seed：17；
- checkpoint：0、32、64、96、128。

### 5.7 checkpoint 选择与成功标准

checkpoint 只用 128 条内部 heldout 选择：

1. 最大化 Q4+LoRA 的 Qasper/2WikiMQA greedy macro-F1；
2. tie-break 使用较低的 Q4+LoRA vs Q16 answer-position KL；
3. 不得使用固定 LongBench 60 条选择 checkpoint。

内部 heldout 成功标准：

- macro-F1 delta vs adapter-disabled Q4 `>= 0`；
- Qasper 和 2WikiMQA 各自 delta 均 `>= -0.01`；
- Q4+LoRA 相对 Q16 mean F1 delta `>= -0.01`；
- 平衡 yes/no 子集准确率相对 disabled Q4 不下降超过 2 个百分点；
- catastrophic regression rate 不高于 disabled Q4。

选择完成后，固定 60 条 source 6--35 只用于一次探索性 full-state 复核，必须同时报告点估计和 paired bootstrap CI，不能把已经多次观察过的 60 条当作新的盲测集。

## 6. 实验顺序

1. 先执行实验 A；它不训练，能直接解除当前最关键的基线混淆。
2. 无论实验 A 指向 warm-start 还是 query-KL，实验 B 均从零 LoRA 开始，并使用 answer-position CE+KL。
3. 只有实验 B 在内部 heldout 通过预注册标准后，才运行固定 60 条 full-state 复核。
4. 若实验 B 仍表现为 heldout answer KL 改善但 F1 不改善，再单独做 current-36 vs expanded-255 的 adapter coverage ablation；本轮不额外预注册第三个训练实验。

## 7. Artifact 位置

远端结果目录：

```text
/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/native-lora-domain-128-20260813c
/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/dense-long-preservation-full-sft-control-20260813a
/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/dense-long-preservation-sft-full-state-validation-20260813a
```

本次工作只新增本诊断文档；没有修改训练、评测或基础设施代码，也没有提交 GPU 任务。
