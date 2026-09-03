# Q-CoMem 多文档正式实验结果（2026-08-11）

## 结论摘要

本轮完成了 5 个全新进程的 frozen-selector 正式重复，以及 1 个 BM25 selector
sensitivity run。所有正式 run 均满足 AC、电源模式、温度、后台 CPU 和 swap 增长
门槛。

主要观察：

1. 在严格预注册阈值下，frozen selector 的 depth-aware policy 为
   `depth 5/7/9 -> Q8/Q8/Q8`，5 次重复完全一致。
2. Q8 相对 BF16 residual 实现 1.882x 容量压缩，并在所有 frozen query/答案位置上
   保持 100% top-1 agreement；mean KL 约为 `6.7e-5` 至 `1.31e-4`。
3. Q4 实现 3.556x 容量压缩，平均 KL 很小，但严格的逐 query top-1 或最大 position
   KL 门槛没有全部通过。若把每个 query 的 agreement 门槛预注册为 90%，策略会变成
   `Q4/Q8/Q4`，说明 layer-wise bit policy 确实可能存在。
4. BM25 保持 100% evidence recall，但只替换两篇非 relevant 的第二文档，就使严格策略
   变成 `Q8/BF16/Q8`。当前 6-query 校准集太小，不能据此冻结部署策略。
5. 最大问题不是 residual quantization，而是当前 chunk/document-local Write 的
   **split interface gap**：BF16 CoMem 相对 dense prompt 的 mean KL 为
   `0.43/0.59/0.90`，答案位置 top-1 agreement 只有约 `69%/72%/63%`。在修复这个
   interface 之前，不能把 Q8 的“相对 BF16 CoMem 等价”解释为“相对原模型等价”。

因此，本轮支持的结论是：**MLX 上的真实 residual bit packing、一次 Write 多次 Read、
逐深度 calibration 和环境审计已经跑通；Q8 quantization gap 很小，但当前 CoMem
interface quality 还不足以进入任务级加速结论。**

## 1. 数据与运行状态

正式原始文件：

- 首次正式运行和持久化 store：
  [`q_comem_multidoc_benchmark.json`](results/q_comem_multidoc_benchmark.json)
- 5 次独立 frozen 重复：
  [`results/formal-20260811-ac.LIgSzd/`](results/formal-20260811-ac.LIgSzd/)
- BM25 sensitivity：
  [`q_comem_multidoc_bm25.json`](results/q_comem_multidoc_bm25.json)

实验设置：

| 项目 | 设置 |
|---|---|
| 机器 | MacBook Pro，M4 Pro，24 GB unified memory |
| 模型 | `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| revision | `7f0dc925e0d0afb0322d96f9255cfddf2ba5636e` |
| 数据 | 6 documents，341 corpus tokens，6 queries |
| frozen selection | 每个 query 固定 2 documents，evidence recall 100% |
| depth | 5、7、9 / 28 layers |
| residual bits | 2、4、8、16 |
| group/chunk/overlap | 64 / 64 / 16 |
| 质量评估 | expected answer 全 token teacher forcing |
| 重复方式 | 5 个全新 Python 进程，轮间冷却 60 秒 |

5 次 frozen run 的环境审计：

| 条件 | 结果 |
|---|---|
| completed / formal eligible | 5/5，5/5 |
| AC before/after | 5/5，5/5 |
| active `powermode` | 全部为 2 |
| thermal before/after | 全部 nominal |
| preflight CPU | 10.9% 至 23.9%，低于 35% 门槛 |
| 初始 swap | 568.31 MiB |
| 每轮 swap growth | 全部 0 bytes |
| 模型加载时间 | median 0.7025 s，p5/p95 0.6391/0.8285 s |

初始 swap 不接近 0，是本轮仍应披露的限制；但 5 次运行均没有新增 swap，符合当前
预注册有效性规则。

## 2. 容量结果

每个 depth 的 residual shape 相同，所以容量结果一致：

| bits | corpus bytes | corpus MiB | 压缩率 | 平均 selected bytes/query |
|---:|---:|---:|---:|---:|
| BF16 | 2,095,104 | 1.998 | 1.000x | 700,416 |
| Q8 | 1,113,024 | 1.061 | 1.882x | 372,096 |
| Q4 | 589,248 | 0.562 | 3.556x | 196,992 |
| Q2 | 327,360 | 0.312 | 6.400x | 109,440 |

这些是 packed integers、scale 和 bias 的实际 `nbytes`，不是名义 bit 推算。Q4 的实际
压缩率不是 4x，因为 affine scale/bias 也需要存储。

## 3. Frozen selector：量化质量

### Q4 与 Q8

| depth | bits | relative RMSE | mean KL | max position KL | mean top-1 | 最差 query top-1 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | Q4 | 0.01832 | 0.001708 | 0.01147 | 97.98% | 93.75% |
| 5 | Q8 | 0.00485 | 0.000067 | 0.000267 | 100% | 100% |
| 7 | Q4 | 0.02187 | 0.001402 | 0.02166 | 100% | 100% |
| 7 | Q8 | 0.00619 | 0.000131 | 0.001239 | 100% | 100% |
| 9 | Q4 | 0.02652 | 0.000953 | 0.00589 | 99.07% | 94.44% |
| 9 | Q8 | 0.00693 | 0.000099 | 0.000430 | 100% | 100% |

严格 calibration 要求：relative RMSE `<=0.05`、所有答案位置 max KL `<=0.02`、每个
query top-1 agreement `==100%`。因此：

- depth 5：Q4 因最差 query agreement 93.75% 失败，选择 Q8；
- depth 7：Q4 的 max-position KL 0.02166 超过 0.02，选择 Q8；
- depth 9：Q4 因最差 query agreement 94.44% 失败，选择 Q8。

Q2 在三个 depth 的 relative RMSE 为 `5.64%/6.18%/7.07%`，均超过 5%，且最大
position KL 为 `0.396/0.360/0.145`，本轮可以直接排除。

### Threshold sensitivity

固定 KL 与 RMSE 门槛，只改变“每个 query 的最低 top-1 agreement”：

| 最低 agreement | depth 5 | depth 7 | depth 9 |
|---:|---:|---:|---:|
| 100% | Q8 | Q8 | Q8 |
| 99% | Q8 | Q8 | Q8 |
| 95% | Q8 | Q8 | Q8 |
| 90% | Q4 | Q8 | Q4 |

`Q4/Q8/Q4` 是敏感性结果，不是当前主策略。不能看到结果后再把 90% 当作正式阈值；
需要在更大的 calibration split 上预注册门槛，再在独立 test split 上验证。

## 4. 时间与摊销

下面使用 5 次进程、共 30 个 query 测量。online Q8 包含 query lower Write、选中文档
dequantize 和 suffix Read；dense 是相同选中文档与 teacher-forced query 的完整 forward。

| depth | dense median | Q8 online median | Q8 p5/p95 | 相对 dense | corpus Write median | 粗略 break-even |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 194.70 ms | 176.16 ms | 171.95/177.59 ms | 1.105x，-9.52% | 87.60 ms | 4.72 queries |
| 7 | 194.70 ms | 169.17 ms | 164.82/171.45 ms | 1.151x，-13.11% | 120.29 ms | 4.71 queries |
| 9 | 194.70 ms | 162.22 ms | 157.92/163.85 ms | 1.200x，-16.68% | 153.61 ms | 4.73 queries |

粗略 break-even 使用：

```text
G* = corpus_write / (dense_query - q8_online_query)
```

这组时间只能作为机制验证，不能当成真实 RAG TTFT：

- corpus 只有 341 tokens；
- dense baseline 也只读取同样的两个 selected documents；
- query 中含 teacher-forced answer；
- 当前没有增量 decode KV cache；
- fetch/SSD page-in 尚未实现；
- selector latency 未计入。

Q2/Q4/Q8/BF16 的 online 时间非常接近，说明在这个小 selected pack 上 suffix compute
占主导，减少 residual bytes 还没有转化成明显 latency 收益。更长 selected context 和
SSD/RAM tier 实验才可能显示 Q4 的带宽优势。

## 5. 当前主要失败点：split interface gap

这里比较的是未量化 BF16 CoMem 与 dense prompt，因此与 residual bit width 无关：

| depth | mean KL vs dense | max position KL | mean top-1 agreement |
|---:|---:|---:|---:|
| 5 | 0.4335 | 2.6990 | 69.25% |
| 7 | 0.5894 | 4.0336 | 71.89% |
| 9 | 0.9031 | 5.5747 | 62.64% |

量化 Q8 的 mean KL 约 `1e-4`，而 interface KL 是 `0.4-0.9`，相差数千倍。因此当前
优先级不能是继续优化 Q4 kernel，而应先恢复 BF16 CoMem 的任务质量。

可能来源包括：

1. 每篇 document 的 lower layers 独立运行，丢失跨文档和 query 的下层上下文；
2. 当前只做普通 causal suffix，没有复现 revision 的 sink、segment/block mask 和 pack
   语义；
3. left overlap 只能修复单文档 chunk 边界，不能自动修复跨文档接口；
4. 当前模型没有针对 split residual 注入训练 suffix LoRA。

下一阶段必须增加 continuous-prefix oracle、matched raw replay、不同 overlap、revision
mask/sink 和可选 suffix LoRA，先定位每项 interface error。

## 6. BM25 sensitivity

BM25 run 也满足正式环境条件，evidence recall 为 100%。它与 frozen selector 的差异是：

- `q-write-read` 的第二文档从 `mixed-bits` 换成 `mlx-streams`；
- `q-valid-run` 的第二文档从 `unified-memory` 换成 `comem-split`。

严格策略变为：

```text
depth 5 -> Q8
depth 7 -> BF16
depth 9 -> Q8
```

depth 7 的 Q8 在 BM25 文档组合下 mean top-1 agreement 为 99.07%，最差 query 只有
94.44%，因此严格 100% 门槛回退 BF16。这表明 6-query calibration 很容易被单个文档
组合支配。正式研究必须使用独立的 calibration/test split，并报告策略跨 selector 和
corpus domain 的稳定性。

## 7. 结果使用边界与下一步

当前可以保留的工程结果：

- 多文档 residual 确实只 Write 一次并被多 query 复用；
- 2/4/8 bit 使用 MLX affine quantize 的真实打包；
- actual store bytes 和每 query selected bytes 可审计；
- 自动 depth-aware policy 在 query 集合上聚合；
- 电源、温度、后台 CPU 和 swap 前后检查已经工作；
- 重复运行的时间波动很小。

下一步按优先级：

1. 修复并拆解 BF16 split interface gap；
2. 扩展到 RULER/LongBench/LoCoMo 的真实 multi-query calibration/test split；
3. 每个 `(depth,bits)` 用独立进程测 peak memory。当前 sweep 会同时保留多个候选 store，
   所以现有 MLX peak memory 不能用于声称“Q4 降低了运行峰值内存”；
4. 加入带 KV cache 的自由生成，分开报告 TTFT 和 decode；
5. 加入 RAM-resident 与 SSD/mmap fetch，之后再判断低 bit 是否降低在线延迟和能耗。
