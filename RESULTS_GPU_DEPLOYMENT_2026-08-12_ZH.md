# Q-CoMem H20 部署/KV 基准（2026-08-12）

## 结论

这次实验确认了 Q-CoMem 的主要部署价值在“高密度驻留可复用文档状态”，
不是当前实现下的单请求峰值显存或 TTFT。在约 4k-token 文档上：

- `qcomem-d7-mixed` 的中位持久状态是 9.742 MiB，相对 full-prefix 的
  140.342 MiB 压缩 14.41×，每文档减少约 130.6 MiB（93.1%）；
- 在 4 GiB safety headroom、单活跃请求的估算下，可驻留文档数从中位
  518 增加到 7,390（约 14.3×）；
- 该 mixed 配置的平均 F1 相对 full-prefix Q16 为 -0.00022，但本实验只有
  8 个 validation workload，不是最终质量结论；
- 当前 eager fork/dequant + suffix 重建使 TTFT 是 full-prefix 的 4.14×，活跃
  CUDA peak 还多约 543 MiB。因此不能宣称“单 query 峰值显存降低 14×”。

后续 [4k COW 部署短测](RESULTS_GPU_COW_4K_SHORT_2026-08-12_ZH.md)（QS `1832356`）
已经把同一 caller boundary 的 incremental full-prefix、Q16 eager 与 Q16 COW 三方 hard gate
跑通，并完成 8-workload short。frozen-static 的 durable packed payload 为 `10.007 MiB`，
相对 full-prefix 是 `14.024×`；但当前 COW staging-inclusive total resident 为 `45.922 MiB`，
实际只有 `3.056×`。CUDA/NVML 进程 peak 反而分别多约 `548/615 MiB`，TTFT 为 `4.122×`。
因此新结果验证了 COW correctness，却进一步确认 reference staging 还不是 active-memory 优化。

## 实验边界

> **Q16 reusable-state 勘误：** 后续审计发现本 trial 使用的旧 Q16 cache fork 在
> `reshape().to(same_dtype)` 时可能与 packed persistent source 共用 storage。该 trial 每行
> 都重新 build store 且只执行一个 request，因此表中首次 query 的时延/质量不等价于
> multi-query 复用验证；任何基于旧代码、从同一 Q16 store 连续发多个 query 的结果均作废并
> 必须重跑。Q4/Q8 与 mixed cache 的 dequantize 会分配新 tensor，不受这个 alias bug 影响。
> 修复版要求 fork A、fork B、source storage 独立，并新增同一 source 下 eager-vs-COW 的
> 逐步 token + bitwise logits direct paired gate。

- QS trial：`1830226`，状态 `Complete`，执行时间 9 分 37 秒；
- 模型：Qwen3.5-35B-A3B，BF16 参数逻辑字节 64.56 GiB；
- 硬件：8×NVIDIA H20-3e 141GB；
- 软件：PyTorch 2.11.0+cu129，Transformers 5.14.1；
- 数据：公开 LongBench Qasper/2WikiMQA validation，各 4 条，共 8 条；
- source index：6--9；校准样本 4--5 排除；冻结的 test-v2 68--99 未读取；
- 输入上限 4096 token，最多生成 32 token；7/8 样本因为上限被截断；
- 每张 GPU 处理一个 workload，每配置 1 次预热 + 3 次随机顺序重复；
- 共 8 workload × 7 配置 × 3 repeat = 168 条计时记录；
- 8/8 shard 的 dense/full-prefix/Q16 replay 逐 token exactness hard gate 全部通过。

数据 SHA-256 为
`1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe`，层策略 SHA-256 为
`f34d4b89e9936c8d58d27df69268250f7985c9f4db1d9cef4d3041a06df36e87`。

## 主要结果

| 配置 | 持久状态 MiB（median） | vs full-prefix | 单活跃请求容量估算（median） | TTFT s（median） | TPOT s（median） | tokens/s（median） | 平均 F1 | Δ full-prefix F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense recompute | 0 | — | — | 0.6492 | 0.6487 | 1.358 | 0.39423 | +0.00286 |
| full-prefix Q16 | 140.342 | 1.00× | 518 | 0.1634 | 0.05411 | 13.380 | 0.39137 | 0 |
| split replay Q16 | 35.915 | 3.91× | 2,010 | 0.6703 | 0.05450 | 7.204 | 0.39137 | 0 |
| Q8 residual/state | 15.892 | 8.83× | 4,536 | 0.6729 | 0.05544 | 7.150 | 0.39137 | 0 |
| Q4 residual/attention + Q8 linear | 10.007 | 14.02× | 7,194 | 0.6741 | 0.05497 | 7.190 | 0.39013 | -0.00125 |
| per-layer mixed `[8,8,4,4,8,8,8]` | 9.742 | 14.41× | 7,390 | 0.6734 | 0.05507 | 7.193 | 0.39115 | -0.00022 |
| all-Q4 residual/state | 8.414 | 16.68× | 8,585 | 0.6736 | 0.05502 | 7.186 | 0.36090 | -0.03047 |

F1 先在每个 workload 内对 3 次 repeat 平均，再对 8 个 workload 取均值。repeat 是
时延重复，不是 24 个独立质量样本。Q4-attention/Q8-linear 在 `qasper-8` 的
3 次中有 1 次生成了另一个贪心序列，表中保留真实平均，不选择性删除。

## 显存口径

### 1. corpus 持久显存确实显著下降

full-prefix 每篇约 140.3 MiB，mixed 约 9.74 MiB，all-Q4 约 8.41 MiB。容量估算
用实测模型 allocation、每文档 persistent bytes、单请求峰值与 4 GiB headroom
计算。它是容量模型，还不是实际循环加载到 OOM 的测量。

### 2. 当前单活跃 query 峰值没有下降

full-prefix 的中位 CUDA allocated peak 约 65.08 GiB，mixed 约 65.61 GiB。成对差值的
中位数是 mixed 多占 568,963,072 bytes（约 543 MiB）。原因是当前 fork 会真实
dequantize/deep-clone lower state，同时 suffix decode cache 仍需要覆盖整篇文档。因此
“持久 store 压缩 14×”与“单活跃请求显存压缩 14×”是两个不同命题。

## 时延口径

- full-prefix 已经有全层 document cache，因此 TTFT 最快；
- Q-CoMem 需要 fork/dequantize lower state，并从 boundary residual 重建 33 层 suffix
  document cache，当前 TTFT 约为 full-prefix 的 4.14×，也没有快于 dense prefill；
- suffix cache 建立后，Q-CoMem TPOT 只比 full-prefix 中位高约 1.2%，并比
  dense 每 token 全历史重算快约 11.8×；
- Q-CoMem 中位端到端吞吐约 7.19 token/s，介于 dense 的 1.36 与
  full-prefix 的 13.38 token/s 之间。

这个结果把系统卖点限定得更准确：当 corpus 很大、每篇文档被重复查询，且
full-prefix cache 的驻留容量是瓶颈时，Q-CoMem 有价值；对只有一个热 prefix 的会话，
标准 full-prefix cache 当前更合适。

后续 [HYPIC-lite 静态审计](gpu/HYPIC_LITE_ZH.md) 已把第 2 项 TTFT 路线的内存边界算清：
depth-7/4k 一段的 full-suffix store 为 152,567,808 B（当前 FP32 recurrent runtime）或
127,401,984 B（理想 BF16 payload）；四段 `w=8` 则为 383,336,448 / 282,673,152 B，
尚未加入约 9.71 MiB 的 Q-CoMem lower state。transition 必须按 32 个 value heads 计数。
这说明无压缩 full-suffix composition 会消耗掉甚至反转本结果的容量优势；该原型尚未进行
H20 计时/质量实验，也不是完整 HYPIC 复现。对应后续门槛已冻结在
[下一阶段实验预注册](NEXT_STAGE_EXPERIMENTS_ZH.md)。

### 后续：4k COW staging short

QS `1832356` 在 validation source 6--9 上完成 8 workloads × 3 configs × 1 repeat。新 hard gate
不再用 dense single-chunk 阻断，而要求相同 document `[256]`、query `[32]` 与单 token decode
边界下，incremental full-prefix、Q16 eager 和 Q16 COW 三方 token trace 一致，同时 eager/COW
完整 logits bitwise equal。8/8 rank 均通过，source immutability 与 COW immutable audit 也均通过；
dense diagnostic 仅 5/8 通过，按冻结语义只记录。

| 配置 | durable payload MiB | materialized staging MiB | total resident MiB | total resident vs full | CUDA peak MiB | NVML peak MiB | TTFT s | TPOT s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full-prefix Q16 | 140.342 | 0 | 140.342 | 1.000× | 66,624.955 | 67,150 | 0.1685 | 0.05371 |
| Q16 COW | 35.915 | 35.915 | 56.137 | 2.500× | 67,182.169 | 67,785 | 0.7069 | 0.08509 |
| frozen-static COW | 10.007 | 35.915 | 45.922 | 3.056× | 67,193.475 | 67,772 | 0.6761 | 0.05749 |

`payload` 表示可持久化的 packed source；`total resident` 按 source/template 的唯一 tensor storage
去重，是容量分母。两者不可混用：frozen-static 的 payload 是 `14.024×` 压缩，但当前真实 resident
只有 `3.056×`。这个 `paged-cow-staging` 只是在 Python/PyTorch cache API 上审计共享与按需
materialization，没有 block table、page allocator、paged kernel 或 serving scheduler，不能称为
PagedAttention。完整解释、COW shared/private 生命周期和负结果见独立报告。

## 下一步

1. COW staging correctness 与 4k short 已完成，但 active peak 为负；下一步去掉 persistent dense
   template，使用 append-only block/page lower cache 或真实 paged kernel，再以 total resident、
   paired CUDA/NVML peak 和 TTFT 联合验收；
2. 仅把 suffix state composition/HYPIC-inspired 作为 TTFT–bytes 对照推进：分别测
   transition-only、seam-only KV 和 approximate full-suffix cache；若新增持久字节回到
   full-prefix，不把更低 token-layer 计算写成部署收益；
3. 对 mixed 与更激进位宽运行 bit-specific LoRA，再做独立 downstream validation；
4. 加入真实 multi-query scheduler、多并发和 corpus 逐篇加载到 OOM 的 capacity sweep；
5. 在 Apple unified memory 上重测同一组 persistent/active/decode 口径。这个方法压缩的
   是文档记忆，不是模型权重；只有 cache/state 是内存瓶颈时，才能支撑更大模型或
   更大工作集。

## 原始产物

- 聚合结果：`results/gpu-deployment-validation-20260812i/deployment-summary.json`；
- 分片结果：`results/gpu-deployment-validation-20260812i/deployment-shard-{0..7}.json`；
- GPU 环境：`results/gpu-deployment-validation-20260812i/gpus-before.csv` 与 `gpus-after.csv`；
- 实验定义：`DEPLOYMENT_BENCHMARK_ZH.md`；
- runner/aggregator：`gpu/run_deployment_bench.py`、`gpu/qcomem_deployment.py`、
  `gpu/aggregate_deployment.py`。
- COW 4k short 报告：`RESULTS_GPU_COW_4K_SHORT_2026-08-12_ZH.md`；
- COW short 完整产物：`results/gpu-deployment-cow-4k-short-incgate-20260812f/`，其中
  `deployment-summary.json` SHA-256 为
  `df08e2dfba48981935f071a42ae5b7417e2b5ae46d9a7ad5512531a49e08c879`。
