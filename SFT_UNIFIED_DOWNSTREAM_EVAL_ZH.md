# Full-SFT 最佳 checkpoint 统一下游评测准备（历史设计稿）

> 状态更新：本文件记录的是提交前设计，其中 residual-only A 线最终没有提交。
> 正式实验已改为同一作业内 base/SFT × dense/完整 lower-state Q16/Q8/frozen 配对，
> 并于 Trial `1832184` 完成。请以
> [正式统一下游结果](RESULTS_GPU_SFT_FULL_STATE_DOWNSTREAM_2026-08-12_ZH.md) 为准；
> 下文不能作为当前结果或完整 Q-CoMem 实现说明。

## 当前结论

正式 384-step full-SFT 已结束；训练内的 train-split held-out CE 选择了 step 128：

- checkpoint：`step-000128-eval-model-only-fp32`
- checkpoint manifest SHA256：`cd22fbca3adfd668d24032726ff60425d432aa49f6105fe1cc9a5a0ef616c647`
- step 0 / 128 / 256 / 384 overall token-weighted CE：`2.24154 / 1.66641 / 1.73341 / 1.79983`
- step 128 比 step 0 低约 25.66%，但这只是从官方 train 切出的 checkpoint-selection diagnostic，不是最终下游结果。

本轮没有读取 LongBench test-v2，也没有提交 GPU。需要运行的新代码已经准备好。

## 为什么现有 `run_downstream.py` 不能直接接 checkpoint

正式 checkpoint 是从 8-rank FSDP FULL_SHARD 保存的、逻辑大小约 138.64 GB 的 FP32 DCP；`run_downstream.py` 则假设每个进程已经拥有一份完整 BF16 Hugging Face 模型。二者的布局和入口都不同：

1. DCP 的 key 对应 `DenseSupervisedCausalLM(language_model, lm_head)`，不是外层多模态 conditional-generation wrapper；
2. DCP loader 需要初始化 distributed process group；
3. CoMem 的手工分层执行依赖可直接访问完整 `language_model.layers`，不能简单绕过 root FSDP wrapper；
4. 因此应把 DCP 重新分片加载到每个 rank 的完整 BF16 text core，再复用外层模型和 `TorchSplitCausalLM` 做推理。

`dcp_replicated_load_preflight.py` 已在远端 PyTorch 2.11 环境完成 2-rank CPU 小模型门禁：FP32 FSDP DCP 能正确重分片到每个 rank 的完整 BF16 replica，两个 replica 完全一致，且确实覆盖了独立初始化的 base 权重。这验证了 API/布局路径；35B 模型的 H20 峰值显存仍须在正式评测作业中实测。

## 最低成本执行矩阵

冻结验证集为 LongBench revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672` 的 Qasper 与 2WikiMQA source index 6--35，共 60 条。输入 SHA256 是 `1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe`。

最低成本 A 线只需要新跑一个 8×H20 job，而且只跑 SFT dense：

| 模型/路径 | 精度 | 新运行？ | 用途 |
|---|---:|---:|---|
| base dense | BF16 | 否 | 复用现有 60 条 paired 结果 |
| base chunk-d7 | residual Q16 | 否 | 复用 Interface-LoRA validation 的 frozen interface baseline |
| Interface LoRA chunk-d7 | residual Q16 | 否 | 复用 checkpoint `c932...b2c3` 的现有结果 |
| SFT step128 dense | BF16 | 是 | 测 full-SFT 的真实下游增益 |
每个 rank 只负责 7--8 条 SFT dense generation。base dense、base chunk-d7 和 Interface LoRA 全部从冻结的既有 60 条 paired artifacts 复用；聚合器会硬校验数据 SHA、source 6--35、生成协议及 LoRA checkpoint SHA。A 线不运行任何 SFT+CoMem/Q4/Q8，因此不会把 residual-only 消融冒充完整 Q-CoMem。

统一协议：4096 max input、按数据集采用 128/32 max new tokens、depth 7、chunk 512、overlap 0、group size 64、同一 prompt 和逐样本 paired F1。聚合器会输出总体和分数据集差值、paired bootstrap 95% CI、prediction agreement、灾难性回归率，以及 residual store 大小。

## 已准备的入口

- `gpu/run_sft_dcp_downstream.py`：严格校验 DCP/data SHA，一次完整 payload integrity pass，把 step128 加载到 8 份 BF16 replica，只做 dense 推理。
- `gpu/aggregate_sft_dcp_downstream.py`：合并新 SFT shards，并与已有 base/Interface-LoRA 的同 60 条结果做 paired 比较。
- `gpu/dcp_replicated_load_preflight.py`：先做便宜的 DCP layout round-trip 门禁。
- `gpu/launch_sft_dcp_downstream_8gpu.sh`：完整 preflight、推理、聚合入口。
- `qs/qcomem-sft-dcp-downstream-prepared-20260812a.yaml`：已填好当前 step128、数据和旧 LoRA artifact 的路径与 SHA；仅准备，尚未提交。

## 结果边界与阻断点

1. A 线完全不运行 Q4/Q8。完整 lower-state Q-CoMem 必须另走 `run_replay_diagnostic.run_config`，不能复用 residual-only `run_downstream.run_config` 来冒充。
2. Interface LoRA 是 PG-19 teacher/student distillation，full-SFT 是 1024 条 QA answer-only CE。它们可以比较最终部署路径的质量，但不是训练数据、目标和训练算力严格等预算的算法对照。
3. full-SFT 训练最大序列长度是 1024，评测最大输入是 4096；最终结果应明确这是长上下文外推验证。
4. DCP payload 约 138.65 GB。loader 会由 rank 0 做一次完整 hash；8 个 BF16 replica 各约 64.56 GiB。小模型已证明 API 可行，但真实模型加载时的临时峰值和并行存储流量仍是唯一未消除的运行风险。
5. 若真实 DCP→BF16 replica 加载超出 H20 峰值，不能退回到手工访问 FSDP 内层做 CoMem（会绕开 root FSDP 的参数 all-gather）。合理 fallback 是单独物化一次 BF16 Hugging Face/safetensors checkpoint，再复用现有独立推理脚本；这会多占约 65 GB 持久存储，但之后所有评测更便宜。
6. 冻结 reader 要求文件恰好包含两数据集的 source 6--35，并绑定 revision/SHA；脚本拒绝 test-v2 路径和已知 test-v2 SHA。原始 test-v2 继续不读。

## B 线：完整 lower-state Q-CoMem 的最小接入

DCP 载入后的对象仍是完整 Hugging Face wrapper，因此可以在同一 8-GPU job、同一份每-rank SFT BF16 replica 上直接调用 `run_replay_diagnostic.run_config`。最小完整状态矩阵应是：

- `dense`
- `replay-d7-layer-q16`：residual/cache 全 Q16 的完整 replay 基线
- `replay-d7-layer-q8`：residual/cache 全 Q8
- `replay-d7-frozen-static`：residual Q4、attention Q4、linear Q8、逐层 `(8,8,8,4,8,8,8)`

这条 B 线会通过 `write_lower_replay` 同时保存 depth-7 document residual 与 lower-layer KV/recurrent cache，再由 `LowerReplayState.quantize` 压缩二者，才有资格称为完整 lower-state Q-CoMem。当前尚需独立 runner/聚合器把 DCP loader、严格 6--35 reader 和这些 replay configs 接起来，并实测 35B DCP→BF16 replica 峰值；所以它没有被放入可提交的 A 线 YAML。
