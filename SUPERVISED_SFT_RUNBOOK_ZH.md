# Supervised CE SFT：能力边界与 1-step 运行手册

这条线与现有 LoRA / top-k KL 蒸馏是两种不同实验，不能混名：

- `dense_full_model_sft_smoke`：真实的 supervised token cross-entropy。Qwen3.5 文本模型的全部参数可训练，只监督 `selected_answer + EOS`。
- `qcomem_suffix_supervised_sft`：当前仅有 fail-closed capability gate，没有训练实现。缓存 query、answer teacher forcing 与逐 token decode 的 chunk 语义尚未通过真实模型验证，因此不能宣称已经完成 Q-CoMem suffix SFT。

这里已经有直接的负面实证：H20 Trial `1830867` 的 cached-two-stage mutable-cache autograd smoke 失败，说明当前 Qwen3.5 mutable suffix cache 训练 backward 路径不可用。这进一步要求 suffix supervised 路径保持 fail-closed，不能用 uncached dense teacher forcing 冒充实现。`dense_full_model_sft_smoke` 明确设置 `use_cache=False`，因此不经过该 mutable-cache autograd 路径，不受这次失败影响。

## 训练样本与 prompt

正式输入是 converter 生成的 JSONL。每条至少包含：

`dataset, source_split, source_id, context, input, answers, selected_answer, provenance`

训练端直接调用 `run_downstream.prompt_parts`，使用和下游评测完全相同的 chat template、Qasper/2Wiki prompt 与 context head+tail 截断。最终序列是：

`document prompt + query prompt + selected_answer + EOS`

prompt 的 label 全部为 `-100`；只有 answer 和 trainer 追加的一个 EOS 计算 causal CE。首个 answer token 由最后一个 query-prompt hidden state 预测，避免 off-by-one。为减少显存，LM head 只投影真正预测 answer/EOS 的 predecessor positions，不物化整段 prompt 的 vocabulary logits。

JSONL 即使带 converter 预计算的 `training_target` 或顶层 tokenized 字段，训练也会忽略它们，并用当前模型 tokenizer 与 `max_sequence_tokens` 重新生成 `input_ids/labels`。这样可以防止 tokenizer/model revision 漂移与旧截断配置污染训练口径。

manifest 中的 tokenizer metadata 不是说明性字段。CPU preflight 与 8 个训练 rank 都会加载 `$MODEL_DIR` tokenizer，并逐项核对 tokenizer class、`vocab_size`、`eos_token_id` 与 chat-template SHA256；`requested_revision` 必须非空，且路径或 revision 必须锁到模型目录标识 `59d61f3`。正式 converter 命令应显式使用 `--tokenizer "$MODEL_DIR" --tokenizer-revision 59d61f3 --max-sequence-tokens 1024`。只要 tokenizer/chat template 漂移，运行会在加载 35B 权重前失败。

## 数据硬门禁

正式运行必须同时固定：

- prepared train JSONL 的路径和 SHA256；
- converter manifest 的路径和 SHA256。

还必须固定两份 `sha256sum` ledger：

- `code.sha256` 精确包含 `supervised_sft.py`、`train_supervised_sft.py`、`preflight_supervised_sft.py`、launcher、config 与 `run_downstream.py`；
- `model-artifacts.sha256` 精确包含 `$MODEL_DIR/config.json`、`model.safetensors.index.json`、`tokenizer_config.json`、`vocab.json`、`merges.txt` 与 `chat_template.jinja`。

launcher 会先拒绝任何缺失 artifact，再核验两份 ledger 自身的预期 SHA 和其中每个文件的 SHA；preflight 与 trainer 会独立解析、复核，并把完整 ledger 写进 `metadata.json`。这不是所有 weight shard 的逐文件 hash，但至少把权重 index、模型 config 与 tokenizer artifacts 锁死，能区分同参数量但模型/tokenizer revision 已漂移的目录。

训练端会重新计算两个 SHA，并要求 manifest 满足：

- `schema_version == qcomem-supervised-qa-v1`；
- `output_jsonl_sha256` 与实际 JSONL SHA 完全一致；
- 净化后的 `output_overlap_count == 0`，但保留真实的 `detected_overlap_count` 与 hash-only `overlap_report`，不能用清零检测数掩盖命中；
- `overlap_policy=fail` 时，passed manifest 必须满足 detected/dropped 都为 0；`overlap_policy=drop` 时允许 detected > 0，但必须逐数据集证明 `dropped_examples == overlap_examples`、全局 `sum(dropped_examples) == detected_overlap_count`；
- converter 仍需全量扫描并记录 `full_eligible_examples`，但正式 capacity smoke 使用 `max_output_per_dataset=4`，所以每个数据集应满足 `selected_for_output_examples == written_examples == 4`，不能误要求 `written_examples == full_eligible_examples`；
- 4+4 smoke 的 deterministic strategy 是 `first_n_target_valid_eligible_in_official_source_order-v1`：若完整 `selected_answer + EOS` 超过该数据集官方生成上限，只在选样阶段记录 `output_selection_skipped_answer_over_cap` 与 train source-id SHA 后继续 official order；答案绝不截断，leakage-audited `full_eligible_examples` 不因此减少，任何其他 target 构建错误仍硬失败；
- Qasper 与 2Wiki 的 `source_split == train`；
- 每个源都有非空 revision/license 与合法 archive/extracted-file SHA；
- `raw_test_v2_read_by_converter == false`；
- test-v2 状态只能是 `deferred_not_read` 或 `blind_hash_manifest`；
- manifest 的 per-dataset written count 与 JSONL 全量扫描一致。

manifest 的 `prompt_protocol` 也必须固定 `run_downstream.prompt_parts`、`supervised_sft.build_supervised_example`、1024-token 上限及 answer+EOS reservation；其中两个源码 SHA 必须与运行时文件一致。因为 trainer 会重建 labels，这个门禁防止 converter 和训练端在截断或 label mask 上静默分叉。

训练 loader 会校验 JSONL 的每一行，而不只是 smoke 取到的前 8 条。因此把 validation/test 行藏在 limit 之后也会失败。顶层和 provenance 的 `source_split` 都必须严格等于 `train`。

## 计算和存储边界

当前只允许：

- 8 ranks；
- FSDP1 `FULL_SHARD`；
- BF16 load/forward/buffer；FSDP wrap 后 persistent local optimizer shard、gradient reduce 和
  Adam moments 必须为 FP32；
- 一张卡一个样本，全局 8 个样本；
- 正式 smoke JSONL/manifest 必须精确为 Qasper 4 条、2Wiki 4 条，不接受完整训练集或偏向单一数据集的前 8 条；
- `steps=1`、gradient accumulation 1；
- metadata-only，不写模型或 optimizer checkpoint；
- 40 个 transformer 层的 MLP/MoE 使用 non-reentrant activation checkpoint；attention/recurrent cache 不 checkpoint。

单 GPU、DDP、超过一步、真实 checkpoint 写出都会被拒绝。模型 exact gate 固定为移除视觉塔后的 34,660,610,688 个文本参数，并再次核验 FSDP 各 shard 的参数总和。`optimizer.step()` 前要求 global gradient norm 有限且严格大于 0，否则 smoke 失败。

需要注意：当前 Transformers loader 在 FSDP wrap 前每 rank 会短暂持有一个完整 BF16 文本模型。这是 capacity smoke 的已知峰值，不是 DDP；结果 metadata 会明确记录该限制。

还要区分“执行到 optimizer step”和“参数发生了有效数值变化”。历史 Trial `1831074` 的
底层参数与 Adam moments 都是 BF16，learning rate 为 `1e-6`，因此只能判定 execution
capability。当前正式 smoke 必须同时通过 FP32 before/after delta、全层 gradient coverage、
optimizer step/moment finite gate；不能再复用历史 BF16 state 口径。

## 准备与运行

正式 4+4 prepared JSONL/manifest 已生成并通过 CPU preflight；可审计副本位于
`results/supervised-sft-formal-20260812b/`。通用模板仍保留 `REQUIRED_*` 占位符，不能直接提交；
本次冻结后的实际配置为 `qs/qcomem-dense-supervised-sft-smoke-20260812a.yaml`。后续新 run 仍需：

1. 将 JSONL、manifest、两个数据 SHA 与两份 integrity-ledger SHA 填入 `qs/qcomem-dense-supervised-sft-smoke.template.yaml` 的副本，并设置唯一 run ID/run directory。
2. 同步 `supervised_sft.py`、`train_supervised_sft.py`、launcher、config、tests 与 converter artifacts 到远端。
3. 先运行 `py_compile`、`test_supervised_sft.py`、`bash -n`，再执行 QS dry-run。
4. 只提交 1 个 8×H20 task；检查 `metadata.json` 的 exact parameter gate、FSDP FULL_SHARD、positive finite grad norm、one step 与 metadata-only 标志。

当前不可做的 claim：1-step smoke 只证明 supervised CE、梯度、FSDP/显存路径可执行；不证明模型质量改善，也不证明 Q-CoMem cached suffix supervised SFT 已可用。

## 已完成的 H20 capability smoke

Job/Trial `234718/1831074` 已 `Complete`，详见
[全参数 SFT 结果](RESULTS_GPU_DENSE_SFT_SMOKE_2026-08-12_ZH.md) 与机器产物
`results/gpu-dense-supervised-sft-smoke-20260812a/`。关键判定：

- 系统/反传 capability：**PASS**。34,660,610,688 个文本参数通过 exact gate，8-rank
  FSDP backward、裁剪与 AdamW step 调用完成；
- 数值上有效的 full-SFT update：**INCONCLUSIVE**。没有 parameter delta/checkpoint，
  BF16 权重与 moments 在 `lr=1e-6` 下存在静默舍入风险；
- 质量/压缩恢复：**未测试**。8 条 train-only smoke 数据、一步、无 validation；
- cached suffix supervised SFT：**能力 gate 仍关闭**，不能用 dense `use_cache=False`
  结果替代。

随后 Job/Trial `234809/1831289` 用修正后的 PyTorch 原生方案完成第二次 smoke：模型仍以
BF16 load，FSDP wrap 后调用 `fsdp.float()`，因此只把每 rank 的 persistent local shard 提升
为 FP32；forward parameter 保持 BF16，gradient reduce、optimizer parameters 和 Adam moments
均为 FP32。结果：

- 34,660,610,688 参数 gradient coverage 完整，missing/nonfinite 为 0；
- 40 层、embedding、final norm、lm_head 的 FP32 parameter delta 均非零；
- 32,377,707,746 个 FP32 elements 改变，768,886,188 个值对下一次 BF16 forward 可见改变；
- max allocated 83.14 GiB/rank，无 OOM；
- 14 个真实 weight shards、代码/运行依赖/model artifacts ledger 均通过；
- 仍为 metadata-only，一步，无 checkpoint/quality claim。

完整结果见 [全参数 SFT 报告](RESULTS_GPU_DENSE_SFT_SMOKE_2026-08-12_ZH.md)，机器产物为
`results/gpu-dense-supervised-sft-fp32-delta-smoke-20260812a/`。这关闭了 ULP/有效更新门禁；
长训前剩余工作是规模化 train-only 数据、FSDP/DCP 分片 checkpoint+resume、独立 heldout CE
和冻结 checkpoint 后的下游 validation。
