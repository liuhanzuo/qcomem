# Q-CoMem 下一阶段实验预注册（2026-08-12）

这个文档在查看新 LoRA/downstream 结果前固定下一阶段的比较对象、数据边界和
通过条件。它不改变已经消费过的 pilot/calibration/validation 分工，也不提前读取
LongBench test-v2。

## 1. 数据边界

- LoRA 训练：只使用 PG-19 `train/` 对象；当前 200-step 为 64-book 功能 smoke。
- 开发验证：LongBench Qasper/2WikiMQA source index 6--35，每类 30 条。
- 校准：index 4--5，不混入开发验证。
- 冻结测试：index 68--99；adapter、bit policy 和 runtime 选择前不读取。
- 36--67 是已消费的 legacy test，不再冒充 untouched test。

## 2. LoRA 阶段 A：200-step 链路与趋势

首先并行运行两条 8×H20 作业：

| 训练线 | student | teacher | 目的 |
|---|---|---|---|
| Interface | depth-7 residual-only/chunk-local Q16 | dense | 验证 CoMem 接口误差能否被 suffix adapter 学习 |
| Frozen-static quant | residual Q4 + cache `[8,8,8,4,8,8,8]` | Q16 replay | 验证量化条件蒸馏能否稳定降低 state noise |

两条线都只训练 depth 7 以后 full-attention 层的 `q/k/v/o_proj` LoRA，不训练
backbone、lower layers 或 Write 路径。默认 rank 32/alpha 64，上限 1 亿可训练参数。

200-step 是 smoke，不是正式训练结论。验收条件：

1. 8 个 rank 均完成，loss/gradient 全部有限；
2. checkpoint-50/100/150/200 可恢复，LoRA 模块数、参数量和 bit semantics 与训练前一致；
3. 后 20 step 平均 KL 低于前 20 step；
4. metadata 保留 `test_v2_used=false` 和训练数据 SHA-256；
5. 明确标记当前 quant student 为未缓存的 document+query suffix 训练图，与部署时的
   document/query 两段 recurrent cache 不完全等价。

阶段 A 只使用 checkpoint-200 做一次开发验证，不在 LongBench validation 上从四个
checkpoint 中挑最好的再声称为测试结果。

截至 2026-08-12 的执行状态：Interface 线通过趋势 gate（首 20→末 20 loss
`0.065276→0.029349`）；frozen-static quant 线虽然完成 200 step，但没有通过趋势 gate
（`0.010212→0.010898`）。因此 quant checkpoint 只作为固定负对照，后续需降低 learning
rate 或从 Interface checkpoint warm-start 后重新训练，不能把本轮写成精度恢复成功。

## 3. LoRA 阶段 B：下游恢复

Frozen-static checkpoint 只对 `replay-d7-frozen-static` 开启；dense、prefix、Q16 replay
始终关闭 adapter。和未训练的同一配置做 paired 比较，报告：

- mean F1 与每数据集 F1；
- 相对未训练学生、Q16 replay 和 dense 的 paired delta/95% bootstrap CI；
- prediction exact、token-position agreement 和灾难性退化率；
- adapter 常驻字节、加载时间、TTFT 和 TPOT 差值。

当前 frozen-static 已经非常接近 Q16，因此“没有显著提升”不等于训练失败；但不能
恶化到超出预注册 margin。只有阶段 A/B 正常，才为 minus-25（18.08×）单独投入
第三个 8 卡作业；该配置才是验证“LoRA 能否恢复极限压缩”的主对象。

## 4. COW/paged lower-state 验收

当前实测的 mixed active lower-state peak 约 35.9 MiB，suffix decode cache 约 121.7 MiB，
Q-CoMem 相对 full-prefix 的成对 CUDA peak 差为约 +543 MiB。三个数字不是同一口径。

COW/paged 原型首先只对第一项负责：

1. document persistent pages 在多 query 间 storage-identical，且不被 recurrent kernel 原地修改；
2. query/decode 只分配 delta pages，不 deep-clone 整个 document lower state；
3. Q16 逐 token 与 eager replay 一致；
4. 分开报告 shared document bytes、private delta bytes、materialization bytes 和 fallback 次数；
5. 如 Transformers kernel 必须连续/可变 cache，fallback 必须明示记录，不冒充 zero-copy。

在真实 H20 复测前，不预设 COW 能消除全部 543 MiB peak 差。

## 5. Suffix composition/TTFT 验收

当前 mixed TTFT 中位 0.6734 s，full-prefix 为 0.1634 s。HYPIC-inspired 对照分三级：

1. naive end-state reuse：只复用 linear recurrent end-state，报告质量错误，不预设 exact；
2. transition composition：若能提取每段 transition operator，与 full recompute 比 logits/token；
3. seam repair：`w=0/8`，报告边界重算 token 数、TTFT 和质量。

不能直接把完整 HYPIC 的 suffix full-attention local KV 加入后仍声称 14×。当前约
4k-token 实测中，mixed persistent 约 9.71 MiB，suffix decode cache 约 121.85 MiB；若后者
也变成每文档持久状态，总量约 131.6 MiB，相对 full-prefix 139.94 MiB 只剩约
1.06× 压缩。因此实验必须分开报告 linear transition-only、seam-only KV 和 full suffix
cache 的新增字节。

静态 shape audit 已完成，详见 [HYPIC-lite 说明](gpu/HYPIC_LITE_ZH.md)。depth-7 suffix
实际包含 24 个 linear-attention 和 9 个 full-attention 层；dense transition 按 32 个
value heads 计数，而不是 16 个 key heads，因为 keys 会 repeat，且 `g/beta` 是
per-value-head。4096 token 的当前账本为：

| 分段 | state 口径 | full suffix KV | end-state | transition | conv tail | suffix 持久合计 |
|---|---|---:|---:|---:|---:|---:|
| 1 | Transformers runtime FP32 | 75,497,472 B | 50,331,648 B | 25,165,824 B | 1,572,864 B | 152,567,808 B |
| 1 | all-BF16 payload | 75,497,472 B | 25,165,824 B | 25,165,824 B | 1,572,864 B | 127,401,984 B |
| 4，`w=8` | Transformers runtime FP32 | 75,055,104 B | 201,326,592 B | 100,663,296 B | 6,291,456 B | 383,336,448 B |
| 4，`w=8` | all-BF16 payload | 75,055,104 B | 100,663,296 B | 100,663,296 B | 6,291,456 B | 282,673,152 B |

四段另有 442,368 B seam KV 在线预算。上述 suffix 合计尚未加入约 9.71 MiB 的 mixed
lower state；一段 BF16 已只剩约 1.07× 的 full-prefix 容量差，四段则明显更差。因此这项
审计的当前结果是负面 Pareto 边界，不是完成的 HYPIC 实验。Q4/Q8 compressed-HYPIC 目前
只有 payload-only 下界，缺少 metadata、量化 transition/KV kernel 和质量验证，只作为未来
方向。原型、8×H20 launcher 和 QS YAML 已准备，但尚未提交 H20；same-packed reference
必须固定为 frozen-static `[8,8,8,4,8,8,8]`，legacy policy 只能显式作为 ablation。

真正通过的配置必须同时满足：不读 test-v2；8-rank Q16 token gate 通过；TTFT 低于
当前 Q-CoMem；持久字节不能回到 full-prefix 级别。若只完成 approximate HYPIC-lite，则作为
quality/latency ablation，不写成 HYPIC 完整复现。
