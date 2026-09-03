# 35B 全参数 SFT 正式训练结果（2026-08-12）

## 结论

Qwen3.5-35B-A3B 的 384-step、3-epoch dense full-model SFT 已在 8×H20-141GB
上完整结束。最佳 checkpoint 是 **step 128（第 1 个 epoch）**：128 条独立 train-split
CE-heldout 上，answer+EOS token-weighted CE 从 `2.241542` 降到 `1.666413`，绝对变化
`-0.575129`、相对改善 `25.66%`；perplexity 从 `9.4078` 降到 `5.2931`，128 条中
112 条的逐样本 CE 改善。

继续训练没有带来更好的泛化：step 256/384 的 heldout CE 回升到 `1.733414/1.799828`。
与此同时训练 CE 仍继续下降，构成清楚的过拟合信号。因此后续下游评测应固定使用 step 128，
不能使用最后的 step 384，也不应声称“3 epochs 优于 1 epoch”。

这次结果证明 full-SFT 对独立 train-split heldout 的监督似然有真实改善；它还没有证明
LongBench 生成 F1 改善、优于 Interface LoRA，或恢复 Q-CoMem Q4/Q8 的压缩误差。上述问题必须
在同一个 step-128 checkpoint 上做冻结的 paired downstream evaluation 后回答。

## 运行与数据

- Job / Trial：`234883 / 1831595`，终态 `Complete`；
- QS 页面：<https://qs2.devops.xiaohongshu.com/model/production/job/trial/234883/1831595>；
- QS 执行时间：31 min 30 s；launcher `00_start` 到 `99_done` 为 29 min 50 s；
- 模型：Qwen3.5-35B-A3B，文本参数 `34,660,610,688`，全部参与训练；
- 分布式：8-rank FSDP1 `FULL_SHARD`，persistent parameter/gradient reduce/Adam 为 FP32，
  forward parameter 为 BF16；
- 训练集：Qasper 512 + 2WikiMQA 512，共 1,024 条；
- CE-heldout：Qasper 64 + 2WikiMQA 64，共 128 条；
- 每个 global batch 固定 Qasper 4 + 2WikiMQA 4；384 steps 恰好让每条训练样本出现 3 次；
- 目标：只监督完整 `selected_answer + EOS`，prompt label 为 `-100`；
- optimizer：AdamW，peak LR `1e-6`、20-step linear warmup、cosine decay、weight decay 0、
  global grad clip 1.0；384 个 optimizer update 的 LR 均大于 0，最后一次 update 后降为 0；
- train JSONL SHA-256：
  `b6b1a88226b3060b6ba6b600793d90470820511ae38096b4db99af8b65f05257`；
- heldout JSONL SHA-256：
  `069c6649e73a0bdbe7b300a1a32f6b89fa5ad23d43fcfa03c85f101d5c7ac10e`；
- split manifest SHA-256：
  `e527eeac4f110005057bcc3936093c6b6ce60252591cd57373785c4995f2ff15`；
- train/heldout 的 source ID、连通组和四类文本指纹交集均为 0；全部来自官方 train；
  validation/test 未用于训练或 checkpoint 选择，LongBench test-v2 未读取。

## Heldout 结果

| Step | Epoch | Overall token CE | PPL | Qasper token CE | 2Wiki token CE | Mean-example CE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | base | 2.241542 | 9.4078 | 2.591904 | 0.803575 | 2.133957 |
| **128** | **1** | **1.666413** | **5.2931** | **2.017826** | **0.224132** | **1.067689** |
| 256 | 2 | 1.733414 | 5.6599 | 2.092520 | 0.259555 | 1.125795 |
| 384 | 3 | 1.799828 | 6.0486 | 2.174848 | 0.260659 | 1.182357 |

相对 step 0，step 128 的 token-weighted CE：

- overall：`-0.575129`（`-25.66%`）；
- Qasper：`-0.574077`（`-22.15%`）；
- 2WikiMQA：`-0.579443`（`-72.11%`）；
- 逐样本改善数：overall `112/128`，Qasper `58/64`，2WikiMQA `54/64`。

作为结果后审计而非预注册显著性检验，对 128 个 paired examples 做固定 seed 的 20,000 次
bootstrap，step 128 相对 step 0 的 token-weighted CE delta 95% interval 为：overall
`[-0.6882, -0.4913]`，Qasper `[-0.7024, -0.4820]`，2WikiMQA
`[-0.8149, -0.3989]`。这支持 heldout 改善不是少数样本造成，但 heldout 已参与 checkpoint
选择，不能再把它称为 untouched final test。

### 过拟合证据

step 128 以后，训练窗口 CE 继续下降，而 heldout 反向上升：

| 区间 | Train token CE 均值 |
|---|---:|
| step 109--128 | 1.7732 |
| step 129--148 | 1.3686 |
| step 237--256 | 1.1066 |
| step 257--276 | 0.7941 |
| step 365--384 | 0.8454 |

step 256 相对最佳 step 128 的 heldout CE 恶化 `+0.067000`；step 384 恶化
`+0.133415`。后审计 paired bootstrap 的 overall 95% intervals 分别为
`[+0.0460,+0.0900]` 和 `[+0.1006,+0.1706]`。恶化主要来自 Qasper；2WikiMQA
在 step 128 后的差异较小且 interval 跨 0。

所有 384 个 train loss 和 pre-clip grad norm 均 finite。step 229 出现一次较大的 pre-clip
grad norm `1112.28`，随后仍按阈值 1.0 裁剪，未出现 NaN、OOM 或训练中断。它应保留为稳定性
审计项，不能从本次 run 推断未裁剪训练同样稳定。

## 参数更新与显存

step 1 使用 warmup LR `5e-8`，仍通过完整数值门禁：

- gradient coverage：`34,660,610,688 / 34,660,610,688`；missing/nonfinite 为 0；
- finite nonzero gradient elements：`32,973,948,530`；裁剪后全局 L2 约 1；
- FP32 shard changed elements：`32,042,878,427`（92.45%）；
- 下一次 BF16 forward 可见的 changed elements：`31,547,370`；
- 40 层以及 embedding/final norm/lm_head 的 FP32 delta 门禁均通过；
- 每 rank 的 FP32 Adam moment coverage 完整，step 值为 1、nonfinite 为 0。

rank-0 runtime ledger 记录：

| 指标 | 结果 |
|---|---:|
| pre-FSDP 完整 BF16 load | 42.34 s |
| runtime max allocated | 83.14 GiB |
| runtime max reserved | 115.87 GiB |
| OOM | 无 |

`max_reserved` 是 allocator 保留量，不是同时存活 tensor；这里也只有 rank-0 峰值，不能写成
8 卡逐卡完全相同的测量。

## Checkpoint

step 128/256/384 均保存为 FP32 model-only、FSDP/DCP sharded checkpoint；没有 rank-0 full
model gather。三份都有 `checkpoint-manifest.json` 和绑定 manifest/payload digest 的 `_SUCCESS`。

| Step | 实际 payload | Manifest SHA-256 | Payload directory SHA-256 |
|---:|---:|---|---|
| **128** | **129.13 GiB** | `cd22fbca...16c647` | `b9ebac8a...885e` |
| 256 | 129.13 GiB | `0ad0d8c7...08c36` | `9711b9c4...f7bf6` |
| 384 | 129.13 GiB | `397b2a1c...63d77` | `3202bed7...172f` |

三份合计约 `387.39 GiB`。每个 epoch 的训练加 heldout 约 4.5 分钟，而每次 DCP 写入与
rank-0 payload hashing 约 4.2 分钟；这说明完整 FP32 checkpoint I/O 已与计算同量级。

最佳 checkpoint：

```text
/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/
runs/qcomem/dense-supervised-sft-formal-384-20260812b/artifacts/checkpoints/
step-000128-eval-model-only-fp32
```

它是 model-only checkpoint：足以做评测/导出，但没有 Adam、scheduler 或 RNG，不能从 step 128
精确恢复训练。若未来需要长训 resume，必须另存 full-resume DCP，不能把本目录冒充可恢复断点。

## 与 LoRA / Q-CoMem 的关系

- Interface LoRA 只有约 6.19M trainable parameters 和约 23.6 MiB adapter，成本远低于
  34.66B-parameter full-SFT；其已有指标是 LongBench validation F1，而本节是 train-split
  heldout CE，两者不能直接比较谁更好。
- full-SFT 的 step-128 结果说明任务监督能够改善模型的答案似然，但训练路径是 dense、
  `use_cache=False`；它没有训练 Q4/Q8 cache tolerance，也没有绕过 mutable-cache autograd blocker。
- 后续统一下游已完成，见
  [SFT × 完整 lower-state Q-CoMem 报告](RESULTS_GPU_SFT_FULL_STATE_DOWNSTREAM_2026-08-12_ZH.md)：
  SFT dense 相对 base dense 的 F1 点估计为 `-0.02308` 且 CI 跨 0，没有下游恢复证据；
  Q8/frozen 的量化增量仍小，但 SFT 后的 Q16 split/replay interface gap 变大。因此不能把本报告的
  heldout CE 改善外推成生成 F1 改善。

## 本地产物

小型结果文件已复制到：

`results/gpu-dense-supervised-sft-formal-384-20260812b/`

其中包括 metadata、384-step train metrics、四个 heldout phase、candidate comparisons、
best-checkpoint pointer 和 train log；约 387 GiB 的 DCP payload 仅保留在远端，不复制进仓库。
