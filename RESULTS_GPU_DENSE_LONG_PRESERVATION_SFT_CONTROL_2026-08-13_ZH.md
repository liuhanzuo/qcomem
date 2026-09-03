# 长文档指令保持 Dense Full-SFT Control 结果（2026-08-13）

## 结论与边界

新版 4K dense Full-SFT control 已完整训练并通过硬门禁。内部 official-train
heldout 的 example-equal CE 从 `1.50377` 降到 `0.75625`，step 128 优于
step 64，因此被自动选为最佳 checkpoint。

这证明的是：post-trained Qwen3.5-35B-A3B 可以在 4K、完整 40 层 FP32-FSDP
训练下，同时学习长文档 QA、一般指令 replay 和冻结 teacher preservation。
它**不是** Q-CoMem cache-aware/QAT 训练，也不能仅凭 train-source heldout CE
声称下游能力已提升。独立的 full-state Q-CoMem validation 已作为单独正式任务提交。

## 正式任务

- Job `235741` / Trial `1833962`，状态 `Complete`。
- 资源：queue 385，单节点 `8 × H20-141G`。
- QS 生命周期：`2026-08-13 12:48:31–13:14:06`，约 25 分 35 秒。
- 正式 launcher 生命周期：`12:49:59–13:13:58`。
- 初始化：post-trained `Qwen3.5-35B-A3B`，没有改用 Base。
- 训练：1024 examples，8 rank 每 rank 每步 1 条，1 epoch / 128 global steps，
  max length 4096，LR `5e-7`，8-step warmup 后 cosine。
- FSDP1 `FULL_SHARD`：persistent parameter/Adam/reduction 为 FP32，forward 为 BF16；
  `use_cache=False`，40 个完整 decoder layer 使用 non-reentrant activation checkpoint。

## 冻结数据

训练集严格按 example 均衡，而不是按 target-token 加权：

| Stratum | 数量 | 来源 |
|---|---:|---|
| domain | 410 | QASPER 256（128 文档 × 2 query）+ 2WikiMQA 154 |
| general replay | 307 | Tulu3 persona-IF |
| teacher preservation | 307 | 与 general replay 不相交的 Tulu3 persona-IF |

内部 heldout 为 64 条：domain/general/teacher 分别为 `26/19/19`。训练与
heldout 的 source、document、context、prompt 和 example hash 交集全部为 0。
410 条 domain row 全部带有可审计的 `document_input_ids` / `query_input_ids`
边界，二者非空且严格重建 answer labels 之前的 prompt prefix；answer/EOS 没有混入 query。

冻结 SHA256：

- train：`3c67f6afe30eec191fd23446de5a8bf7282abea709f1b3ad42dd26f25107203c`
- heldout：`af01ef2153e2143af1848ba5bd78781030d1c81000ef734cc847f0fd37baa9a5`
- data manifest：`c67a382517ccda2a858b6a9646efb1fd89427c1b4b76f8ee6cc4e3e091e2354b`
- independent audit：`5cee7230c29db385ac06cf228106410f939e4960ec4277d68c63e245a4be5747`

数据 builder 和独立 auditor 均未读取 LongBench validation、legacy test 或 test-v2 raw rows；
这里只使用了既有 hash-only heldout ledger 做排重。

## Teacher preservation

唯一正式作业在 optimizer 创建前，使用同一个冻结 post-trained checkpoint 生成：

- 8 个 rank shard；
- 307 个 teacher rows / 68,814 个 assistant target positions；
- top-32 log-prob + tail probability bucket；
- BF16 normalized hidden state；
- 总 shard bytes：`300,495,607`。

每个 shard 的 SHA256、精确 bytes 和 schedule indices 均在 manifest 中冻结。
teacher stratum 的 loss 为 `0.45 hard CE + 0.35 KL(T=1) + 0.20 hidden cosine`；
另外两个 stratum 使用 hard CE。

## Step-1 4K 硬门禁

step 1 的八条长度为
`[4096, 602, 597, 4096, 591, 589, 4096, 589]`，首条训练 metric 记录
`step1_gate_event_recorded=true`。后续行该字段为 false 只是单次事件标记，不表示失败。

门禁结果：

- finite objective 和 finite/nonzero global grad norm 通过；
- 34,660,610,688 个参数的完整梯度覆盖通过，40/40 decoder layers 均被覆盖；
- FP32 logical delta 与 BF16-forward-visible delta 均非零；
- 8 rank Adam FP32 moments/step/coverage 均通过；
- 最大 step-1 reserved 下仍有至少 `30,961,106,944` bytes（28.83 GiB）headroom，
  高于预注册的 512 MiB 门槛；
- 全程 runtime peak allocated/reserved 分别为 83.20/116.11 GiB（rank 0 telemetry）。

## 内部 heldout 结果

| Stratum | step 0 CE | step 64 CE | step 128 CE | step 0→128 |
|---|---:|---:|---:|---:|
| Overall | 1.50377 | 0.80299 | **0.75625** | -49.71% |
| Domain | 1.80797 | 0.57589 | **0.52842** | -70.77% |
| General replay | 1.23384 | 0.94258 | **0.90201** | -26.89% |
| Teacher preservation | 1.35742 | 0.97418 | **0.92225** | -32.06% |

该 heldout 来自 official train sources，只用于 checkpoint selection，不是最终下游测试。

## Checkpoint

两个 checkpoint 都是 reshardable、model-only、FP32 DCP，不含 optimizer/scheduler/RNG，
也没有 rank-0 full gather。

| Step | heldout CE | manifest SHA256 | payload-directory SHA256 |
|---:|---:|---|---|
| 64 | 0.802993 | `479f6d20afb54bef243157c4fbf6f6cb55bbc1b3ab3aa536a1bfe64b9534a217` | `4c85978bc833d26c28aea2f220211a3fd21fb9a08de8731a25b7ff254937ff1b` |
| **128** | **0.756249** | `12f837bdde1dadf5c820d4527685bfe3bf321f5494e25db3a4bb662aab168c57` | `89c788e4112c7c9dc7e801201979cdcb69600e067cd9497db456cb63f50ba65b` |

step 128 logical model bytes 为 `138,642,442,752`，实际 DCP payload bytes 为
`138,651,858,875`。`best-checkpoint.json` 选择 step 128，并显式标注 selection metric
只是 diagnostic train-source heldout。

## 独立 full-state validation

Job `235777` / Trial `1834066` 已完成：

- validation parent SHA256：
  `1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe`；
- 严格只选 QASPER/2WikiMQA 各 source index `6–35`，共 60 条；排除 calibration `4–5`；
- test-v2 path/hash/raw-row 门禁开启，`68–99` 不读取；
- 同一个 8×H20 job、同一个 caller 中先评 Base，再 collective load step 128 DCP；
- 两个模型都运行 dense、full-state Q16、full-state Q8、frozen-static 四条路径；
- frozen-static 固定为 residual Q4 / attention Q4 / linear Q8 / lower layer
  `[8,8,8,4,8,8,8]`；这些 bit 只作用于 persistent document state，模型权重仍为 BF16。

结果中，SFT dense 相对同 job Base dense 为 `+0.01352` F1，frozen-static
为 `+0.01384`；二者的 paired 95% CI 都跨 0，所以是正向信号而非显著提升。
SFT frozen-static F1 为 `0.55744`，相对 SFT dense 为 `+0.00103`，persistent
state 相对 Q16 压缩 `3.59×`。完整分析见
`RESULTS_GPU_DENSE_LONG_PRESERVATION_SFT_FULL_STATE_2026-08-13_ZH.md`。

正式 YAML 为
`qs/qcomem-dense-long-preservation-sft-full-state-validation-20260813a.yaml`。

## 本地审计包

本地小型审计包位于
`results/gpu-dense-long-preservation-sft-control-20260813a/`。它保存 metadata、完整
train/heldout metrics、teacher manifest、两个 DCP manifest/_SUCCESS、数据审计和完整性日志；
约 130 GiB 的 DCP payload 仍保存在远端，不在本地复制。

审计包文件 SHA256 见
`results/gpu-dense-long-preservation-sft-control-20260813a/artifact-ledger.sha256`。
