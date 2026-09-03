# Q-CoMem 4k COW 部署短测（2026-08-12）

## 结论

QS Job `235197` / Trial `1832356` 已在 8×H20 上 `Complete`，严格聚合得到
8 shards、8 workloads、3 configs、24 条 measurement。结果分成两个不同口径：

- frozen-static 的**可持久化 packed payload**中位数为 `10.007 MiB`，相对
  full-prefix Q16 的 `140.342 MiB` 是 `14.024×` 压缩；
- 当前 `paged-cow-staging` 原型还同时保留 dense materialization template，按唯一
  tensor storage 核算的**持久总 resident**为 `45.922 MiB`，实际 resident 收益只有
  `3.056×`；
- 当前原型没有降低进程峰值：frozen-static 相对 full-prefix 的 paired CUDA allocated
  peak 中位**多** `548.2 MiB`，NVML sampled process peak 中位**多** `615 MiB`；
- frozen-static 的 paired TTFT 为 full-prefix 的 `4.122×`，TPOT 为 `1.024×`。

因此，这个 short 证明了 audited COW 执行的正确性，也量出了 staging 的真实代价；它没有
证明 active-memory 或 TTFT 优化。`14.024× payload` 与 `3.056× total resident` 必须分开报告。

## 实验协议与来源硬门禁

- 模型：Qwen3.5-35B-A3B，BF16 权重；
- 硬件：8×NVIDIA H20 141GB，queue `RL_main`（408），资源包 183；
- validation：公开 LongBench Qasper 与 2WikiMQA，各 source index 6--9，共 8 workloads；
- 数据 SHA-256：`1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe`；
- source revision：`5e628be450b7e67fb7ae6e201bd6d8f7056f7672`；
- `test_v2_consumed=false`，source 68--99 未读取；
- max input 4096 tokens，最多生成 8 tokens；7/8 workload 被 4096 上限截断；
- 每配置 1 次 warmup + 1 次正式 measurement，配置顺序按 workload 固定随机化；
- 配置为 `full-prefix-q16`、`qcomem-d7-r16-a16-l16` 和
  `qcomem-d7-frozen-static`；后者严格等于 residual Q4、attention Q4、linear Q8、
  `cache_layer_bits=[8,8,8,4,8,8,8]`。

专用 launcher 会在加载模型前拒绝错误数据 SHA、test-v2 路径、非 6--9 source、非 8 workloads、
错误配置、非 4k/8-token/1+1 协议以及非空 RUN_DIR。聚合器另外 fail-closed 检查 rank 0--7、
每个 workload/repeat 的完整配置排列、全部必需 memory/TTFT/NVML 字段、COW 无 fallback，且
`capacity_document_denominator_nbytes == persistent_total_resident_nbytes`。

## Correctness hard gate

旧门禁把 dense single-chunk recompute 当 oracle；Qwen3.5 recurrent state 对 chunk boundary
敏感，因此它会把“调用边界差异”混成“COW 错误”。本次冻结的新门禁使用相同 caller-visible
边界比较三个 incremental 路径：

1. incremental full-prefix Q16；
2. 同一 exact-Q16 persistent source 的 eager fork；
3. 同一 source 的 COW fork。

gate 输入边界为 document `[256]`、query `[32]`、随后单 token decode `[1,1,...]`，最多
生成 4 tokens。结果为：

- 8/8 rank 的 full-prefix vs eager Q16 token trace exact；
- 8/8 rank 的 full-prefix vs COW Q16 token trace exact；
- 8/8 rank 的 eager Q16 vs COW Q16 token trace exact 且完整 logits bitwise equal；
- 8/8 rank 在 eager 后、COW 后的 source full-tensor snapshot 均未改变；
- 8/8 rank 的 COW immutable audit 通过，`strategy_effective=paged-cow-staging`，无 fallback；
- dense single-chunk diagnostic 只有 5/8 通过，rank 1--3 分叉；该项按预注册语义只记录，
  不阻断同边界 incremental hard gate。

正式 4k workload 中，三配置的 greedy token 序列也在 8/8 workload 上相同；但只有 8 个
validation workload、最多 8 个生成 token，因此它只是 short 一致性观察，不替代 60-sample
下游 validation，更不替代冻结的最终 test。

## 持久状态与 COW 生命周期

以下均为 8 workload 的中位数。`payload` 是可落盘/可驻留的 durable Q-CoMem source；
`materialized staging` 是 COW 原型为执行准备的 dense template 逻辑字节；`total resident`
按 source 与 template 的唯一 tensor storage 去重，不能再把前两列机械相加。Q16 boundary
residual 是只读零拷贝，所以 Q16 行尤其不满足简单相加；frozen-static 行两者没有这部分重叠。

| 配置 | payload MiB | staging MiB | total resident MiB | payload vs full | total resident vs full |
|---|---:|---:|---:|---:|---:|
| full-prefix Q16 | 140.342 | 0 | 140.342 | 1.000× | 1.000× |
| Q-CoMem Q16 COW | 35.915 | 35.915 | 56.137 | 3.908× | 2.500× |
| frozen-static COW | 10.007 | 35.915 | 45.922 | 14.024× | 3.056× |

Q16 与 frozen-static 使用同一个 dense execution template，因此 request-local COW 生命周期相同：

| 时点 | shared MiB | private MiB |
|---|---:|---:|
| fork 初始 | 23.540 | 12.375 |
| query 后 | 15.693 | 20.375 |
| decode 结束 | 0 | 20.388 |

初始时只读 document residual 与 attention K/V 可共享，linear convolution/recurrent state
必须私有复制。query/decode 中现有 attention update 通过 `torch.cat` 生成连续 K/V，逐步把共享
storage materialize 为私有 storage；document residual 在 suffix seed 后释放。因此最终 shared
降到 0。这是“COW staging 能正确运行”的证据，不是长期维持共享页的证据。

request-local selected lower fork 的 peak/steady 中位为 `36.068/20.388 MiB`；suffix decode KV
peak/steady 为 `121.731/121.678 MiB`。后者仍覆盖长文档的 suffix layers，是当前 active peak
没有随 packed payload 同比例下降的重要原因。

## CUDA/NVML 与时延负结果

| 配置 | CUDA allocated peak MiB | NVML process peak MiB | TTFT s | TPOT s | F1 median |
|---|---:|---:|---:|---:|---:|
| full-prefix Q16 | 66,624.955 | 67,150 | 0.1685 | 0.05371 | 0.22876 |
| Q-CoMem Q16 COW | 67,182.169 | 67,785 | 0.7069 | 0.08509 | 0.22876 |
| frozen-static COW | 67,193.475 | 67,772 | 0.6761 | 0.05749 | 0.22876 |

以 workload/repeat 成对计算：

- Q16 COW 相对 full-prefix 的 CUDA/NVML peak 中位多 `576.7/626 MiB`，TTFT/TPOT 为
  `4.267×/1.535×`；
- frozen-static COW 相对 full-prefix的 CUDA/NVML peak 中位多 `548.2/615 MiB`，TTFT/TPOT 为
  `4.122×/1.024×`；
- 两个 COW 配置的 paired F1 delta 都为 0，但这个 8-token short 不是质量主实验；
- 只有 1 次正式 repeat，Q16 TPOT 存在明显 workload outlier，因此 short 的 latency 只用于
  筛查方向，不能当稳定 production 吞吐结论。

## 实现边界：不是 PagedAttention

`paged-cow-staging` 是 Python/PyTorch reference prototype：它审计 cache tensor 类型，初始共享
不可变 residual/attention storage，私有复制 mutable linear state，并在现有模型 cache API 需要时
materialize。它没有 block table、page allocator、paged KV kernel、跨请求页复用 scheduler，也没有
vLLM/TensorRT-LLM 式 PagedAttention。因此：

- 可以说 same-source COW 的三方 incremental correctness hard gate 通过；
- 可以说 packed payload 的 durable-store 压缩为 14.024×，当前 staging-inclusive resident 为 3.056×；
- 不能把它称为 PagedAttention，不能声称 active peak 已下降，也不能用 payload 压缩比代替
  model+workspace+active request 的进程峰值压缩比。

下一步最值得做的是把 dense template 的全量 materialization 移出 persistent path：实现真正的
packed-page/block-table lower cache，或至少让 attention K/V 按 append-only blocks 延迟 materialize，
并复用 suffix seed/cache；验收必须同时看 `persistent_total_resident`、CUDA/NVML paired peak 和 TTFT，
不能只优化 payload ledger。

## 调度记录与原始产物

queue436 上的首个提交 Trial `1832343` 因平台事务停在 `Uncommit`，没有 Pod、没有 RUN_DIR、没有
GPU 执行，确认 `Terminated` 后才在 queue408 提交本次唯一有效替代 Trial `1832356`；两者没有重叠。
有效 Trial 于 22:39:31 Running，22:44:02 Complete。

- 本地完整产物：`results/gpu-deployment-cow-4k-short-incgate-20260812f/`；
- 聚合结果：`deployment-summary.json`，SHA-256
  `df08e2dfba48981935f071a42ae5b7417e2b5ae46d9a7ad5512531a49e08c879`；
- 8 个原始 shard、rank logs、preflight tests、GPU before/after 与 stages 均已拉回；
- QS 配置：`qs/qcomem-cow-4k-short-incgate-20260812f.yaml`；
- runner SHA-256：`qcomem_deployment.py` =
  `2f2c7b3e1371ea2d552d300c59fbe3786b3db1295d860ae8ed79915c977bafa6`，
  `qcomem_paged.py` =
  `0b7135d55185a46f945022bcba4aec27e8d1a8bdcabbd1ea5e39d5cb24f607d0`，
  `run_deployment_bench.py` =
  `e9554f0328c39d30dc65e56962265c6f50ac4574cbfb6a0062f988d91964c40b`，
  `aggregate_deployment.py` =
  `d12d54c131455dbdeb387345662821a819de81f3f46e9f6a97cf531061cda4d2`。
