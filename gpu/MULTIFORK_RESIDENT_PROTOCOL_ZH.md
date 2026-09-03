# Q16 same-kernel multi-fork resident 正式协议

## 1. 实验问题

本实验只回答一个容量问题：在同一份约 4K 的文档 KV cache 已经构建后，若同时保留
`N={1,2,4,8,16,32}` 个请求，完整复制文档的 fresh control 与共享只读文档、仅保留
私有 tail/append 页的 reuse 路径，显存驻留量如何随 N 增长。

它不是吞吐实验，也不测试 vLLM engine scheduler。所有请求对象会在首个 model step
前一次性创建并保持强引用，但 8 个生成轮次在一条 CUDA stream 上按
`(round_index, request_index)` 串行执行。raw 时间只用于诊断，不汇总、不做 speedup、
TTFT 或并发吞吐结论。

## 2. 冻结配置

- 模型：Qwen3.5-35B-A3B，BF16 权重，Q16 paged KV。
- full-attention 层：`3,7,11,15,19,23,27,31,35,39`；其余 30 层均走相同的
  functional GDN 状态路径。
- 每个 rank：一个不同的 PG19 `train/` 原文对象，一个 4095-token 文档窗口。
- page size：128；document tail：127，故这是接近整页 tail 的压力配置。
- 每请求 query：32 token；生成 8 token；实际写入 cache 为 `32+7=39` token。
- N 曲线：`1,2,4,8,16,32`；每个 rank 必须完整运行全部 N。
- 运行顺序：`1,32,2,16,4,8`；arm 首后顺序随该序列交替。此平衡不构成 ABBA
  性能设计，因此仍禁止速度结论。
- 8 个 rank 各绑定一张 H20；不同 N 不拆到不同 rank。

4095 是有意选择的 worst-case partial-tail stress，不能泛化到 page-aligned 4096。
对齐 4096 的理论对照是 tail copy 为 0，private capacity 约 2.5 MiB/request；本实验
不把这个理论值写成实测结果。

## 3. PG19-only 输入与 query bank

本路径没有 LongBench 参数、validation stage 或 source index。它拒绝路径名中出现
LongBench、test-v2 或 `68-99`。source 6–9、source 68–99 和 test-v2 均不得读取、
stat 或 hash。

每个 rank 的 32 个 query 都来自与该 rank 文档相同的 PG19 train book：

1. query bank 从文档窗口末端和原相邻 query 之后开始；
2. 每段 32 token，offset stride 固定为 64，因此两两不重叠；
3. protocol manifest 冻结 source object、文档 offset、32 个 query offset、逐 query
   token SHA256、整 bank SHA256；
4. N 只取同一 bank 的前 N 项，保证 N 曲线嵌套可比；
5. formal runner 禁止 synthetic slot marker。marker 生成器只允许出现在 CPU unit test。

PG19 JSONL、独立 manifest、重建 windows SHA、query-bank manifest、代码 ledger、模型
manifest、模型 artifact/weight ledger 和 runtime protocol manifest 都必须在 GPU 前
完成外部 SHA pin 与内容语义校验。

## 4. 两个 arm 的唯一差异

`fresh`：公共 source document 仍保留，同时为每个请求创建独立的
`document + one-request private reservation` 物理 pool，并复制完整物理 document
blocks（包含最后一页 padding）。N 个 fresh arena 的 K/V storage 必须两两不相交。

`reuse`：只有一份只读 source document；source arena 按 N 预留 N 份 private capacity，
N 个 request sequence 共享 source arena，但每份 private physical reservation 在同一
arena 内两两不相交。

两边必须使用同一个已解析的 `vllm unified_attention` callable、同一 Q16/page/GQA/
position/mask 语义。30 个 linear GDN 层的 persistent base 相同，query 后每请求状态也
必须一致。fresh 与 reuse 之外没有不同的 attention kernel。

## 5. 4095-token 解析显存公式

Q16、2 个 KV heads、head dim 256、page 128、10 个 full-attention 层下：

- 每层一个 physical block：262,144 B；10 层合计 2,621,440 B。
- 文档物理 allocation：83,886,080 B（80 MiB）。
- 文档有效 payload：83,865,600 B；padding：20,480 B。
- 每请求 private reservation：2 pages/layer，即 5,242,880 B（5 MiB）。
- 每个 fresh request pool：89,128,960 B（85 MiB）。
- 每请求 partial-tail staging copy：2,600,960 B（127 tokens）。

以公共 source 为受控共同基线：

- reuse：`80 + 5N MiB`；
- fresh：`80 + 90N MiB`；
- controlled fresh−reuse：`85N MiB`；
- fresh physical document copy：`80N MiB`；reuse 为 0；
- N=32：fresh 2,960 MiB，reuse 240 MiB。

active payload/page 已包含在 private reservation 中，不得再次相加。source 的 N 份
private reservation 必须与 document allocation 分列；fresh 的 N 份 duplicate document、
padding 和 private reservation也必须分列。combined unique storage 仅保留在 raw artifact
作为去重诊断，不能进入主容量拟合或 claim。

## 6. 正确性门禁

对每个 rank、每个 N、每个 arm，aggregate 必须从 raw 重放而非信任布尔汇总：

- N 个对象在 setup 和 generation snapshot 时同时存活；request/sequence identity 不同；
- 每请求 8 轮、每轮 10 层，call 顺序和 append delta 为 `[32,1,1,1,1,1,1,1]`；
- fused call 命中完整，dense fallback=0，full K/V concat=0，full-document staging copy=0；
- production mask 不物化、不 host sync；position_ids 在 Qwen3.5 post-RoPE 接口中已消费；
- fresh/reuse 每请求每步以 `torch.equal` 比较完整词表 logits，并比较 token 与 logit SHA；
- 每请求最终 logical K/V 和 60 个 functional GDN state tensors 的 digest exact；
- source document 和 persistent GDN base 前后 immutable；
- 同一 query index 在所有包含它的 N 上，token、全量 logits、最终 K/V/GDN 保持 exact，
  即 cross-N prefix isolation gate。

任何一个 raw row、call、trajectory、allocator snapshot、storage layer 或 query provenance
缺失/不自洽都必须使 aggregate 失败。

## 7. allocator 与 timing 边界

每个 arm 记录 PyTorch allocator 的 current/peak allocated 与 reserved：common document
prefill、Q16 pack、resident setup、generation-only 和 setup+generation。每个 phase-before
必须先 reset peak；相邻 phase current 值必须连续；每个 arm GC/empty-cache 后必须恢复
冻结的 current allocated/reserved baseline。

主报告可使用：

- 解析 Q16 pool/current resident 曲线；
- setup 后与生成后 current allocator；
- 每个 production model step 完成并 sync 后、且在 `isfinite`、SHA 和 CPU logits
  clone 诊断之前采集的 peak。

这些是 PyTorch allocator 数值，不是 NVML、整卡峰值或完整模型容量。raw seconds 是单次、
validation-instrumented diagnostic；没有 ABBA 重复，因此不得产生 speedup ratio。

## 8. 发布与终态治理

launcher 固定 `LC_ALL=C`，所有 ledger 按 C locale 排序；冻结 code snapshot 不得可写。
长时间 Python 阶段有 hard timeout，8 个 GPU rank 子进程有受控终止与清理；trap 安装
之后发生的失败或信号必须写 `FAILED`、`FAILED_PHASE` 和阶段 marker。trap 安装前的参数、
路径或 RUN_DIR 拒绝只保证非零退出，不宣称已有 marker。scientific ledger 本身没有额外
timeout；`99_done` 只能在它生成并 `sha256sum -c` 成功之后写入。

允许的最终 claim 仅限：Q16、batch1、单文档、10 个 full-attention 层、N≤32、4095-token
partial-tail stress、PG19 train-only、同 kernel fresh-vs-reuse 的逐请求 exact 与容量曲线。
明确不包含：F1/下游质量、多文档、ragged/continuous batching、scheduler、真实并发吞吐、
TTFT/kernel speedup、NVML、aligned4096 实测或 N>32。
