# Q-CoMem LoRA 训练说明

这条训练线把两类误差分开处理，并且都只训练 split depth 之后的 LoRA：

1. **Interface LoRA**：student 使用 Q16 chunk-local、residual-only CoMem；teacher 使用未启用 adapter 的 dense/full-replay。它恢复的是丢弃 lower KV/recurrent state 和 chunk-local Write 引入的接口误差。
2. **quantization-conditioned LoRA**：student 对持久化 residual、attention KV、linear recurrent state 做真实 pack→unpack fake-quant；teacher 使用 Q16 exact replay。它恢复的是状态量化误差。

第二种方法不是 QLoRA：模型权重没有量化，量化对象是 CoMem 持久状态，梯度只进入 suffix LoRA。

## 不可破坏的实验边界

- backbone、Write 路径和 split 之前的 lower layers 全部冻结。35B MoE 的默认配置只替换 suffix attention `q/k/v/o_proj`；给全部专家的 `gate/up/down_proj` 安装 LoRA 可能导致参数量和 DDP 通信爆炸，只能作为单独预算、单独 hard gate 的消融。
- 启动时打印 matched module 数和 trainable parameter 数；默认 `max_trainable_params=100000000`，超过立即退出，避免安静地启动超大 adapter。
- teacher 与 student 共用一份 35B 权重。每步先在 `no_grad + adapter disabled` 下产生 teacher top-64，再做 student forward，因此不会同时常驻两份模型。
- loss 只覆盖 query 对应的位置，为 `0.6 KL(teacher || student) + 0.4 KL(student || teacher)`；两边都在 teacher top-k support 上重新归一化。
- 训练和调参只能使用 PG-19 train/calibration。**禁止读取 LongBench test-v2**：Qasper/2Wiki source index 68–99、远端 `qcomem-longbench-test-v2/longbench_test_v2.jsonl`、SHA256 前后缀 `fe046477…eaaa5f`。test-v2 仅供 adapter 与 bit policy 冻结后的最终一次评测。
- Quant student 现在有三个显式执行选项：历史 `merged-uncached` 将 document/query residual
  拼成一个未缓存 suffix 序列；`cached-two-stage` 先 prefill document suffix cache，再把完整 query
  residual 一次性续接；`detached-document-cache` 在 `no_grad` 下 prefill document，随后对 cache
  tensor `detach().clone()`，只对 query continuation 反传。真实 8×H20 Trial `1830867` 已证明
  mutable `cached-two-stage` 在 8/8 rank 的 backward 都发生 inplace version mismatch，因此该能力
  gate 已失败。detached 路径只是 query-continuation-only 近似，也尚未通过真实 smoke；functional
  immutable cache 只有 fail-closed 设计 gate。三者不能互相冒充。

## 数据格式

PG-19 的正式来源固定为 Google DeepMind 官方仓库及其公开 GCS bucket。仓库中的
`gpu/prepare_pg19_train_subset.py` 可以准备只含 `train/` 对象、逐文件校验 GCS MD5、并
记录 generation 与最终 JSONL SHA256 的功能 smoke 子集：

```bash
python gpu/prepare_pg19_train_subset.py \
  --books 64 \
  --output /path/to/pg19/qcomem_pg19_train_smoke64.jsonl \
  --manifest /path/to/pg19/qcomem_pg19_train_smoke64.manifest.json
```

这个 64-book 子集只用于真实模型 forward/backward、显存和 checkpoint smoke；正式
4000-step 结果必须改用完整 PG-19 train，不能把 smoke 子集的 loss 下降写成正式质量结论。

推荐将 PG-19 train 整理成 JSONL，每行一本书：

```json
{"id":"pg19-000001","text":"...book text..."}
```

loader 会按 `context_tokens + query_tokens` 切连续窗口。Interface 默认按 revision recipe 使用 1536 token context 和后续 512 token query，总长 2048；chunk 512、overlap 0。也可直接提供 tokenized JSONL：

功能 smoke 还设置 `max_windows_per_record`，防止 `dataset_limit` 被第一本长书全部占满；
64-book/1-step smoke 每本取 1 个窗口，200-step smoke 每本最多取 32 个窗口。正式完整语料
训练可以取消这个上限并改用 streaming/randomized dataloader。

```json
{"id":"w0","document_ids":[1,2],"query_ids":[3,4]}
```

若设置 `teacher_source=offline`，每条 tokenized record 还必须包含形状为 `[query_tokens, top_k]` 的 `teacher_topk_indices` 和 `teacher_topk_logits`。当前 smoke 默认使用更简单的同进程 online teacher。

## 本地静态检查

```bash
cd /Users/liuhanzuo/MacLLM-Bench
PYTHONPATH=gpu python3 -m unittest gpu/test_qcomem_lora.py
PYTHONPATH=gpu python3 -m unittest gpu/test_qcomem_suffix_full.py
python3 -m py_compile gpu/qcomem_lora.py gpu/train_qcomem_lora.py \
  gpu/qcomem_suffix_full.py gpu/train_qcomem_suffix_full.py
bash -n gpu/launch_lora_8gpu.sh
bash -n gpu/launch_suffix_full_8gpu.sh
```

这些检查只使用 tiny fake model、tiny fake data 和真实 pack/dequant 算子，不下载或载入真实大模型。

## 8×H20 的 200-step smoke

Interface 配置：

```bash
CODE_DIR=/path/to/qcomem_gpu \
MODEL_DIR=/path/to/Qwen3.5-35B-A3B \
DATA_FILE=/path/to/pg19_train.jsonl \
RUN_DIR=/path/to/runs/lora-interface-smoke \
ENV_DIR=/path/to/vllm-cu129-v1 \
CONFIG_FILE=/path/to/qcomem_gpu/configs/lora_interface_smoke_200.json \
bash gpu/launch_lora_8gpu.sh
```

量化恢复只需将 `CONFIG_FILE` 换成 `lora_quant_smoke_200.json`。该配置使用统一字段：

```json
{
  "residual_bits": 4,
  "attention_bits": 4,
  "linear_bits": 8,
  "cache_layer_bits": [8, 8, 8, 4, 8, 8, 8]
}
```

`cache_layer_bits` 必须与 depth 内实际保存的 cache layers 数目一致。移除它即可使用 attention/linear 的两档统一位宽。
主配置现与 60 条 validation 上最好的 frozen-static 14.10× operating point 对齐。仓库另有
`lora_quant_q8_smoke_200.json` 和 `lora_quant_minus25_smoke_200.json`，分别用于低风险 Q8
恢复对照与 18.08× 极限压缩恢复；三者必须生成独立 checkpoint，不能在 validation 上
挑最好 checkpoint 后再把同一数据声称为 test。

仓库提供两个 200-step QS 模板：

- `qs/qcomem-lora-interface-smoke.yaml`
- `qs/qcomem-lora-quant-smoke.yaml`

提交前先确认 PG-19 文件确实存在，并对 YAML 做 `qs --dry-run`。此外，
`qs/qcomem-lora-dual-real-smoke.yaml` 已在 8×H20 上完成 Interface/Quant 各 1 step 的
真实链路验证（trial `1830043`）；loss、checkpoint 和非零 LoRA 更新核验见
[RESULTS_GPU_LORA_SMOKE_2026-08-12_ZH.md](RESULTS_GPU_LORA_SMOKE_2026-08-12_ZH.md)。
这不等价于已经运行 200-step 或正式训练。

## Checkpoint、恢复和输出

每 50 step 保存 `checkpoint-XXXXXX.pt`，同时更新 `latest`。恢复时设置：

```bash
RESUME_FILE=/path/to/checkpoint-000050.pt bash gpu/launch_lora_8gpu.sh
```

`RESUME_FILE` 只允许完全相同的 mode、bit policy、数据、world size 和 schedule，恢复 optimizer/scheduler/step。Interface adapter 继续训练 Quant 时必须使用：

```bash
INIT_ADAPTER_FILE=/path/to/interface/checkpoint-004000.pt bash gpu/launch_lora_8gpu.sh
```

该路径只加载 LoRA 权重并将 optimizer/scheduler/step 从零重建，不能用 `RESUME_FILE` 冒充跨模式续训。

checkpoint 只保存 LoRA 权重、optimizer/scheduler、随机状态和 metadata，不复制 35B backbone。`metadata.json` 固化：

- adapter 目标模块、rank/alpha、参数量和 adapter 字节数；
- Interface 或量化状态的存储语义与 bit policy；
- Write/lower 不可训练、suffix-only、`is_qlora=false`；
- 每步观察到的平均持久状态字节数；
- `test_v2_used=false`。

## 接回下游评测

`run_replay_diagnostic.py` 可以直接安装 `qcomem_suffix_lora_v1` checkpoint。为了避免把
“adapter 改变了 dense/Q16 teacher”误当成“恢复了量化误差”，必须显式指定 adapter 只对
哪个量化配置启用。8 卡 launcher 示例：

```bash
LORA_CHECKPOINT=/path/to/checkpoint-000200.pt \
LORA_APPLY_TO_CONFIGS=replay-d7-frozen-static \
CODE_DIR=/path/to/qcomem_gpu \
MODEL_DIR=/path/to/Qwen3.5-35B-A3B \
DATA_FILE=/path/to/longbench_validation.jsonl \
RUN_DIR=/path/to/runs/lora-frozen-static-validation \
ENV_DIR=/path/to/vllm-cu129-v1 \
bash gpu/launch_mixed_validation_8gpu.sh
```

runner 会默认关闭刚载入的 adapter，只在精确命名的目标 config 上开启，并在每个 shard
记录 checkpoint 路径、SHA-256、训练 metadata、目标配置和本配置是否实际启用。Q8 或
minus-25% checkpoint 需要把目标配置换成与其 bit policy 完全一致的 suite config；若新增
策略名称，应先固定 `CONFIG_SUITES`，不能临时复用一个名称却改变位宽。

Interface checkpoint 使用独立的 residual-only validation launcher，不能拿 replay
layer-validation 代评。launcher 会同时 hard-check validation SHA、checkpoint SHA、Qasper/
2Wiki 各 30 条、两边 source index 都严格为 6--35，并按路径与 SHA 拒绝 test-v2：
评测只运行 dense、冻结 chunk-d7 和 chunk-lora-d7；不运行逐生成 token 重算完整前缀且
与 dense 对照冗余的 oracle-d7。

```bash
EXPECTED_VALIDATION_SHA256=1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe \
EXPECTED_LORA_CHECKPOINT_SHA256=c93269e31d4e7a3ed990ceb9ab602e56234fdc21011685b842a90263e36fb2c3 \
LORA_CHECKPOINT=/path/to/interface/checkpoint-000200.pt \
CODE_DIR=/path/to/qcomem_gpu MODEL_DIR=/path/to/Qwen3.5-35B-A3B \
DATA_FILE=/path/to/longbench_validation.jsonl RUN_DIR=/path/to/interface-validation \
ENV_DIR=/path/to/vllm-cu129-v1 \
bash gpu/launch_interface_lora_validation_8gpu.sh
```

Quant checkpoint 在部署结论前还需跑训练/部署 suffix 语义 gate。两边共用一次 full-query
lower continuation；历史训练侧是未缓存的 document+query suffix，部署侧是 suffix document
cache prefill 后一次 full-query continuation。工具默认以 16 个 query position 为一块做 vocab
projection，既覆盖完整 query 轨迹，也不会保留 `[query, vocab]` 全量 logits：

```bash
python gpu/run_lora_deployment_semantic_gate.py \
  --model /path/to/Qwen3.5-35B-A3B --data /path/to/pg19_train.jsonl \
  --checkpoint /path/to/quant/checkpoint-000200.pt \
  --output /path/to/deployment-semantic-gate.json \
  --depth 7 --residual-bits 4 --attention-bits 4 --linear-bits 8 \
  --cache-layer-bits 8,8,8,4,8,8,8 --projection-block-size 16
```

报告逐 query position 给出 top-1 agreement、training→deployment KL 和最大 logit
误差，并汇总 mean/max KL；不能只用最后一个 query 位置代表训练 loss 的全部位置。

真实 trainability 负结果见
[cached-two-stage 反向图报告](RESULTS_GPU_CACHED_AUTOGRAD_2026-08-12_ZH.md)：Trial `1830867`
的 8/8 rank 都在 backward 报 `[1,32,128,128]` CopyBackwards inplace version mismatch；
不是 OOM，也不是 NCCL 根因。没有 step、gradient coverage 或 checkpoint。因而 semantic eval
相似也不能替代 autograd capability gate。

## Suffix-full 参数蒸馏容量上界（不是 full-model SFT）

除 LoRA 外，仓库提供第一层容量上界 `suffix_full_distillation`：lower/write、embedding、最终
norm 和 lm_head 冻结，depth 7--39 的 transformer layer 参数全部训练。它仍使用与 LoRA 公平
一致的 Q16 teacher top-64 query-position 双向 KL，student 使用真实 frozen-static
pack→dequant；当前 review config 已改为 `detached-document-cache`，只用于设计/本地检查，
尚未授权真实 smoke。它**不是**真正端到端 full-model SFT，也不是 QLoRA。

冻结模型 checkout 的 hard preflight 必须精确得到：

- suffix trainable：`27,751,037,952`；
- attention：`1,054,614,528`；
- MLP/MoE：`26,696,288,256`；
- norm/other：`135,168`。

任一数值变化都会在 FSDP wrap 前退出。训练只允许 8-rank `FSDP FULL_SHARD`、BF16 参数与
BF16 reduce、`use_orig_params=true`；禁止用 DDP 在 8 张卡复制 27.75B 个 trainable 参数。
activation checkpoint 只包 suffix MLP/MoE：这些模块占绝大多数参数且没有 cache 副作用。
不能 checkpoint 整个 attention layer，因为 backward recompute 会重复修改 DynamicCache/
GatedDeltaNet cache，存在语义错误风险。

精确逻辑字节账如下（不含 frozen model shard、activation、all-gather 与 allocator fragmentation）：

| 项 | 全局 | 理想 8-way shard/rank |
|---|---:|---:|
| BF16 trainable parameters | 51.69 GiB | 6.46 GiB |
| BF16 gradients | 51.69 GiB | 6.46 GiB |
| Adam 两个 BF16 moments | 103.38 GiB | 12.92 GiB |
| Adam 两个 FP32 moments | 206.76 GiB | 25.85 GiB |
| model-only suffix checkpoint | 51.69 GiB | 6.46 GiB |
| suffix + BF16 moments checkpoint | 155.07 GiB | 19.38 GiB |
| suffix + FP32 moments checkpoint | 258.45 GiB | 32.31 GiB |

optimizer state dtype 不预设为 FP32；1-step 后逐 rank 记录真实 dtype/bytes。text model BF16
FULL_SHARD 的理想常驻约 8.07 GiB/rank，所以 parameters + gradients + optimizer 的静态下界约
27.45 GiB/rank（BF16 moments）或 40.38 GiB/rank（FP32 moments），再加 layer all-gather、
cache、activation 和临时张量。当前 HF loader 在 FSDP wrap 前每卡会短暂加载一份完整 BF16
模型，这是启动峰值限制，但 steady-state 训练参数并非 DDP 复制。

为避免仅做 1 step 就写出 155--258 GiB 的无用 optimizer checkpoint，review smoke 固定
`checkpoint_mode=metadata-only`，只保存参数/optimizer ledger、loss、gradient norm 和 CUDA peak。
正式多步实现 sharded resume 前，不允许 rank 0 full-gather checkpoint。

原建议的首个 review smoke 参数为：8×H20、512 document + 128 query、每 rank batch 1、
gradient accumulation 1、1 step、learning rate `1e-6`、frozen-static
`r4/a4/l8/[8,8,8,4,8,8,8]`。入口为：

```bash
CODE_DIR=/path/to/qcomem_gpu MODEL_DIR=/path/to/Qwen3.5-35B-A3B \
DATA_FILE=/path/to/qcomem_pg19_train_smoke64.jsonl \
EXPECTED_DATA_SHA256=ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c \
RUN_DIR=/path/to/runs/suffix-full-smoke ENV_DIR=/path/to/vllm-cu129-v1 \
CONFIG_FILE=/path/to/qcomem_gpu/configs/suffix_full_distillation_smoke_1.json \
bash gpu/launch_suffix_full_8gpu.sh
```

mutable cached-two-stage LoRA smoke 已失败；使用相同执行语义的 suffix-full Trial `1830869`
在 Uncommit、无 Pod 时止损终止。`qs/qcomem-suffix-full-distillation-smoke.yaml` 现在只保留为
detached 设计模板，**不得提交**，直到 LoRA 规模先通过真实 capability smoke。不要为新的近似
路径直接消耗 27.75B 参数 FSDP 资源。

真正 `end_to_end_full_model_sft_qat` 是第二层设计，当前 capability gate 明确失败：lower/write
仍在 inference/no-grad 路径，residual、attention cache 和 linear recurrent state 都没有 STE，
也没有验证 DynamicCache/GDN state 的端到端 autograd 与 sharded resume。CLI 只会写
`capability-gate.json` 后退出。token CE SFT 也只标为独立未来消融；当前未实现，不能把
suffix-full KL 蒸馏改名为“全量 SFT”。

## Smoke 通过条件与正式训练

200-step 只用于检查：8 个 rank 都有有限 loss、forward/reverse KL 总体下降、LoRA checkpoint 可恢复、显存无持续爬升、记录的 persistent bytes 与独立 infra benchmark 一致。不能用它声称下游精度恢复。

通过后再把 `steps` 提到 revision recipe 的 4000，保持 rank 32、alpha 64、2048-token 窗口、学习率 8e-5、100-step warmup、cosine decay 和 8×H20。200-step smoke 为了仍有充分衰减阶段只用 20-step warmup。建议依次训练：

1. Q16 residual-only Interface LoRA；
2. 从该 adapter 继续训练固定 Q4/Q8 quantization-conditioned LoRA；
3. 再训练冻结后的 mixed-bit policy；
4. 只在 validation 冻结 adapter/checkpoint 后，运行一次 LongBench test-v2。

当前限制：LoRA 仅覆盖实际暴露为 `nn.Linear` 的目标投影；某些 fused MoE expert 参数不会自动包装。top-k KL 不显式建模 teacher top-k 之外的概率质量。LoRA 训练器固定每 rank batch size 1，通过 DDP 和 gradient accumulation 扩大有效 batch；suffix-full 容量上界则只允许 FSDP FULL_SHARD。两者都没有自动下载 PG-19。历史 `merged-uncached` quant run 与部署执行形态不同；真实 mutable `cached-two-stage` backward 已失败。`detached-document-cache` 会克隆冻结的 document cache，只让 query suffix 计算对 suffix 参数反传，必须始终用这个名称与 claim limit；query 内部仍有 mutable recurrent update，真实 gate 前不能称其可训练。真正 functional-cache 需要 out-of-place DynamicCache/GDN state，目前尚未实现。
