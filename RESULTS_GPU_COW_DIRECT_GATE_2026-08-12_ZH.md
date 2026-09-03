# Q-CoMem COW Direct Paired Gate（2026-08-12）

## 结论

QS Trial `1830738` 的**作业终态是 Failed**，不能写成“总 exactness gate 通过”。应把两个
命题严格分开：

- 外层 dense single-chunk oracle 对 incremental cache 的 token gate 为 **5/8 rank 通过、
  3/8 rank 失败**；失败的是 rank 1--3；
- 本次预注册的核心命题——从**同一份 Q16 persistent source** 分别运行 eager deep-clone 与
  `paged-cow-staging`——为 **8/8 rank 通过**。所有生成位置的完整 logits tensor 均
  `torch.equal`，每 rank 的最大绝对误差和最大 relative-L2 都是 0；persistent source 在
  eager 后与 COW 后都未改变，COW 没有 fallback。

因此，本结果证明修复后的 COW fork 在这 8 条 256+32-token 小门禁上与 eager replay
bitwise 等价；它**没有**证明 dense/full-prefix/Q-CoMem 三条路径对所有 chunk 划分都 token
一致，也没有测 4k workload 的 TTFT、峰值显存或吞吐。按门禁规则没有启动 short benchmark。

## 实验边界

- Job / Trial：`234600 / 1830738`；状态 `Failed`；
- 资源：queue `385`、cluster `53`、package `183`，8×H20-141G；
- 模型：Qwen3.5-35B-A3B，PyTorch `2.11.0+cu129`，Transformers `5.14.1`；
- 数据：公开 LongBench validation，Qasper 与 2WikiMQA 的 source index 6--9；
- 数据 revision：`5e628be450b7e67fb7ae6e201bd6d8f7056f7672`；
- 数据 SHA-256：`1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe`；
- 冻结 test-v2 index 68--99 未读取；
- 每 rank：document 256 token、query 32 token、最多检查 4 个 emitted logits；
- fork strategy：`paged-cow-staging`；只运行 `GATE_ONLY=1`。

## 1. 同源 eager-vs-COW direct sub-gate

下表的 token 是每个 logits 位置的 argmax；rank 5、6 的最后一个 `248046` 是 EOS，因此常规
`generated_token_ids` 不包含它，但 direct trace 仍检查了该位置的完整 logits。

| rank | workload | eager / COW emitted argmax | token exact | logits bitwise | max-abs | max relative-L2 | source immutable | effective strategy |
|---:|---|---|---|---|---:|---:|---|---|
| 0 | qasper-6 | `[3018,383,264,7059]` | 是 | 是 | 0 | 0 | eager 后、COW 后均是 | COW，无 fallback |
| 1 | qasper-7 | `[3018,383,279,328]` | 是 | 是 | 0 | 0 | eager 后、COW 后均是 | COW，无 fallback |
| 2 | qasper-8 | `[3018,383,279,328]` | 是 | 是 | 0 | 0 | eager 后、COW 后均是 | COW，无 fallback |
| 3 | qasper-9 | `[3018,383,279,27416]` | 是 | 是 | 0 | 0 | eager 后、COW 后均是 | COW，无 fallback |
| 4 | 2wikimqa-6 | `[314,17075,401,49623]` | 是 | 是 | 0 | 0 | eager 后、COW 后均是 | COW，无 fallback |
| 5 | 2wikimqa-7 | `[6764,30,248046]` | 是 | 是 | 0 | 0 | eager 后、COW 后均是 | COW，无 fallback |
| 6 | 2wikimqa-8 | `[3756,30,248046]` | 是 | 是 | 0 | 0 | eager 后、COW 后均是 | COW，无 fallback |
| 7 | 2wikimqa-9 | `[69593,19643,506,279]` | 是 | 是 | 0 | 0 | eager 后、COW 后均是 | COW，无 fallback |

每个 rank 都对 persistent source 的 15 个 tensor、14,548,992 B（13.875 MiB）做完整
`torch.equal` snapshot；eager 后和 COW 后的 changed-index 都为空。COW request 另外验证：

- `same_persistent_source=true`；
- `strategy_effective=paged-cow-staging`，`fallback_reason=null`；
- shared immutable audit 为 verified；guarded tensor 为 document residual 与 attention K/V；
- H20 inference tensor 不提供 version counter，所以 `version_guarded_tensors=0`。审计仍包含
  storage pointer、受控 read/`torch.cat`/rebind update 和 16 点内容采样；persistent source
  本身则由上述全 tensor snapshot 兜底。

## 2. 外层 dense gate 为什么失败

| rank | dense generated tokens | full-prefix / Q16 COW generated tokens | 最早 token 分叉 | dense→incremental trace-wide max-abs | 外层 gate |
|---:|---|---|---|---:|---|
| 0 | `[3018,383,264,7059]` | `[3018,383,264,7059]` | 无 | 1.25 | 通过 |
| 1 | `[3018,383,6758,446]` | `[3018,383,279,328]` | step 2：`6758→279` | 20.451171875 | 失败 |
| 2 | `[3018,383,3299,37794]` | `[3018,383,279,328]` | step 2：`3299→279` | 18.21875 | 失败 |
| 3 | `[3018,383,279,198]` | `[3018,383,279,27416]` | step 3：`198→27416` | 11.390625 | 失败 |
| 4 | `[314,17075,401,49623]` | `[314,17075,401,49623]` | 无 | 0.564453125 | 通过 |
| 5 | `[6764,30]` | `[6764,30]` | 无 | 0.75 | 通过 |
| 6 | `[3756,30]` | `[3756,30]` | 无 | 0.8125 | 通过 |
| 7 | `[69593,19643,506,279]` | `[69593,19643,506,279]` | 无 | 0.8406982421875 | 通过 |

这里的 max-abs 是旧 schema 保存的**整条 trace 最大值**，不能冒充“最早分叉 step 的误差”。
本次 artifact 没有保存 full-prefix-vs-Q16 的逐步 logits pairwise tensor：能严格确认的是两者
8/8 token 序列相同，并且它们相对 dense 的 trace-wide max-abs 在每 rank 数值相同；这些证据
仍不足以声称 full-prefix 与 Q16 COW 8/8 logits bitwise 相同。runner 已增加三方逐步 pairwise、
首个 logit/token 分叉位置和 per-step max-abs/relative-L2，留待以后的小型诊断复核；本轮不为此
重复申请 8 卡。

### Caller-visible 执行边界

这里记录传给模型/cache API 的 token chunk，不代表内部 CUDA/Triton tile：

| 路径 | 4 个 emitted logits（rank 0--4、7） | 3 个 emitted logits（rank 5、6） |
|---|---|---|
| dense recompute | full-history `[288,289,290,291]` | `[288,289,290]` |
| full-prefix | document `[256]` → query `[32]` → decode `[1,1,1]` | document `[256]` → query `[32]` → decode `[1,1]` |
| Q16 eager/COW lower 7 层 | document `[256]` → query `[32]` → decode `[1,1,1]` | document `[256]` → query `[32]` → decode `[1,1]` |
| Q16 eager/COW suffix 33 层 | document seed `[256]` → query `[32]` → decode `[1,1,1]` | document seed `[256]` → query `[32]` → decode `[1,1]` |

dense 第一次把 document+query 作为一个 288-token chunk；标准 full-prefix 和 Q-CoMem 都保留
`256 + 32` 边界。Qwen3.5 的 GatedDeltaNet/conv 路径对 chunk 划分存在数值敏感性：本轮
full-prefix 与 Q16 COW 走相同 caller boundary 并产生相同 token，而 dense single-chunk 在
3 个 Qasper 样本的后续 decode 分叉。由于三方逐步 logits 尚未保存，这里只把它作为由执行边界
与现有结果支持的解释，不写成已经完成的 kernel-level 因果证明。

## 3. COW 状态生命周期

| 时点 | shared | private | 解释 |
|---|---:|---:|---|
| initial fork（所有 rank） | 1,572,864 B（1.5 MiB） | 12,976,128 B（12.375 MiB） | residual + attention K/V 只读共享；linear/conv 私有 |
| first query 后（所有 rank） | 1,048,576 B（1.0 MiB） | 13,565,952 B（12.9375 MiB） | attention 首次 `torch.cat` 后物化为私有；residual 尚存 |
| final（rank 0--4、7） | 0 | 13,572,096 B（12.9434 MiB） | residual 已释放，3 个 decode 增量已写入 |
| final（rank 5、6） | 0 | 13,570,048 B（12.9414 MiB） | EOS 提前结束，只有 2 个 decode 增量 |

Q16 persistent source 为 13.875 MiB；dense staging 的逻辑大小也是 13.875 MiB。两者共享
1 MiB 的只读 boundary residual，因此按唯一 storage 计算的准备阶段总驻留量为约
26.75 MiB，而不是把两个逻辑大小直接相加为 27.75 MiB。这是 256-token correctness gate 的
账本，不是 4k active-peak benchmark；标准 Transformers attention 在首次 query 后仍物化
完整 document+query K/V，所以不能据此宣称已经实现真正 PagedAttention。

## 4. Q16 alias 修复

旧 `PackedTensor.dequantize()` 的 Q16 分支使用 `reshape().to(same_dtype)`，可能返回 persistent
storage 的 view。Qwen3.5 `conv_states/recurrent_states` 使用原地更新，这会让所谓 deep-clone
污染 source。修复后 Q16 cache leaf 显式 `.clone()`；回归测试要求 source、fork A、fork B
的 data pointer 三者不同，并在修改 fork A 后验证 source/fork B 未改变。

`PackedResidual` Q16 没有无谓 clone：LowerReplayState contract 明确把 boundary residual
视为只读；生成只读取它以建立 suffix cache，随后释放 request-local 引用。该零拷贝 contract
与可变 cache leaf 的 clone contract 已在代码注释和测试中分开。

旧 deployment trial `1830226` 的每一行重新 build store 后只服务一个 request，不能解释为
Q16 multi-query correctness。任何旧代码下“同一 Q16 store 连续服务多个 query”的结果都应
作废重跑；Q4/Q8 dequantize 本来就产生新 tensor，不受这个 Q16 same-dtype alias bug 影响。

## 原始产物

- 本地目录：`results/gpu-deployment-cow-direct-gate-20260812c/`；
- 8 个 shard：`deployment-shard-{0..7}.json`；
- 日志与 preflight：`logs/rank-*.log`、`deployment-tests.log`、`stages/FAILED`；
- 本轮精确代码快照：`code_snapshot/`；
- QS 配置：`qs/qcomem-deployment-cow-direct-gate-rlab.yaml`；
- 实现与测试：`gpu/qcomem_torch.py`、`gpu/qcomem_paged.py`、
  `gpu/qcomem_deployment.py`、`gpu/test_qcomem_torch.py`、
  `gpu/test_qcomem_paged.py`、`gpu/test_qcomem_deployment.py`。
