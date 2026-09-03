# Q-CoMem-Edge：面向端侧 Hybrid LLM 重复查询的低比特 Split Replay

> Research proposal v0.6，2026-08-14

## 摘要

重复查询同一批稳定文档时，标准 RAG 或长上下文推理会反复执行完整 prefill。CoMem 将文档块预先执行到 Transformer 的中间深度 `j`，持久化每个 token 的中间 residual `h_j`，查询时只选择少量相关块并运行剩余的上层网络。Revision 已经证明这种“预付部分深度”的设计存在可测量的速度-质量曲线，但 `h_j` 仍以 BF16 保存，存储量随语料线性增长，而且尚未在 Apple Silicon 的统一内存和本地 SSD 上完成端到端验证。

本项目提出 **Q-CoMem-Edge**：持久化一个低比特 split residual `h_j`，并补充 query 在 lower layers 精确读取文档所必需的少量 full-attention KV 与 linear-attention recurrent state。它用 split 以上的重计算换取比全模型 prefix cache 更小的持久状态，并根据 split depth 的敏感度选择 `b(j)`。在 Apple Silicon 上，系统通过 MLX/Metal 实现 RAM-SSD 分层读取和融合反量化，并根据语料规模、预期复用次数和生成长度在 raw replay、exact prefix cache、BF16 CoMem 与 Q-CoMem 之间做运行时选择。研究目标不是声称“第一次缓存 hidden state”或“第一次复用 hybrid state”，而是回答一个更窄、更可证伪的问题：

> 在 Apple Silicon 上，压缩的 split-depth residual 加 lower replay state 能否在保持
> dense 质量的同时，以显著更少的每文档持久字节扩大端侧可热驻留文档/上下文工作集；
> 在这个 capacity-first 目标成立后，warm TTFT 与能耗代价能否被准确量化并进一步降低？

当前 H20 算法主证据应以修正 suffix chunk boundary 后的 60 条 source index
6–35 validation 为准：Qwen3.5-35B-A3B depth-7 `residual Q4 / full-attention Q4 /
linear-state Q8` 的完整 persistent state 约压缩 `14.10×`，mean F1 相对 dense 为
`-0.00052`，且无灾难性退化。更早的 64 条 validation 和 source 36–67 的 64 条当时
frozen 评测仍是历史证据；后者对新 policy/adapter 已是 consumed legacy test，不能再称
untouched。真正 blind 的 test-v2 是 source 68–99，截至本版仍未读取。更激进的全 Q2
虽达到 30.39x，却让 pilot F1 从 0.410 降到 0.177。因此论文主线是
state-type-aware 的一个数量级观察性近无损压缩；极限压缩只作为 rate-distortion
失效端点。Apple 端的 MLX/Metal、SSD、能耗与热稳定性仍是下一阶段必须实测的系统贡献。

修正 Qwen3.5 suffix document/query chunk boundary 后，新的 60 条 validation（source index
6--35）再次得到 frozen static 14.10x、mean F1 delta vs dense -0.00052、vs Q16 replay
-0.00387，且无灾难性退化。逐层 same-memory mixed 只有 14.50x，却将 vs Q16 delta 扩大到
-0.01259；18.08x 的 minus-25% 策略为 -0.05438，paired CI [-0.11561, -0.00171]。
因此当前证据支持 14x knee，但不支持“由 4-prompt calibration 得到的 layer-wise policy
优于 state-type static”。后续 mixed-bit claim 必须以扩大校准集后的结果为准。

8×H20 部署基准进一步说明了系统价值和代价各在哪里。在 8 个约 4k-token
LongBench workload、7 种配置、3 次随机顺序重复上，per-layer mixed 的持久状态
为 9.742 MiB，full-prefix 为 140.342 MiB（14.41×）；带一个活跃请求的容量估算
从中位 518 篇增加到 7,390 篇。其平均 F1 相对 full-prefix 为 -0.00022。但当前
eager fork/dequant + suffix 重建的 TTFT 是 full-prefix 的 4.14×，单请求 CUDA peak
还多约 543 MiB。因此已验证的卖点是 **corpus persistent capacity**，不是当前
实现的单 query 峰值显存或 warm TTFT。

2026-08-14 的 answer-supervised native-cache LoRA B（Job `237290` / Trial `1840023`）
补上了质量恢复的正式证据，但没有改变上述谨慎结论。该 run 在 36 个 suffix
full-attention 和 120 个 GDN 关键 projection 上共训练 156 modules，只用独立
official-train heldout 选 step 128。固定 60 条 full-state validation 上，step 128
相对 adapter-disabled frozen-static 的 F1 为 `+0.005559`，95% CI
`[-0.019821,+0.030127]`；旧 step-0 在仅 2 个 reference-yes 样本上的 `1/2`
yes→no 在 step 64/128 恢复为 `2/2` yes。所以它证明了定向失败修复和总体
正点估计，没有证明总体显著提升。旧 Trial `1834056/1834193` 的 36-module、
answer-free query-KL 负结果继续作为历史基线，不被新 run 覆盖。新 run 也不是
纯 cold start：36 个 full-attention modules 沿用已知负向的旧 step-0 warm start，
120 个 GDN modules 为 cold start，MLP 没有覆盖。详见
[answer-supervised LoRA B 报告](RESULTS_GPU_ANSWER_SUPERVISED_NATIVE_LORA_B_2026-08-14_ZH.md)。

这条质量线还引入了不能隐藏的共享成本：frozen-static persistent state 为
`9.6609 MiB`/文档，Q16 为 `34.6831 MiB`/文档，而 answer-LoRA 另有每个模型
进程 `101.8125 MiB` 的 FP32 adapter。只统计 persistent state + shared adapter 时，
break-even 为 `4.0689` 个常驻文档，即从第 5 个文档起 Q4+adapter 低于 Q16。
该口径不含模型权重、active workspace、allocator reserve 或临时 activation，不能替代
峰值显存结论。step 128/disabled 的 median TTFT 比 `1.06451×` 只是单次固定
顺序诊断，没有 ABBA 或重复轮次，不作严格性能 claim。

HYPIC-inspired suffix audit 进一步说明了为什么 TTFT 不能反过来成为当前主卖点。真实
Qwen3.5 depth-7 suffix 包含 24 个 linear-attention 层和 9 个 interleaved
full-attention 层。若为一段 4k 文档额外持久化 full suffix KV、linear end-state、dense
transition 和 conv tail，Transformers 当前 FP32 recurrent-state 口径需要
152,567,808 B，理想 all-BF16 payload 仍需 127,401,984 B；加上约 9.71 MiB 的 mixed
lower state 后，几乎回到或超过约 139.94 MiB 的 full prefix。四段 `w=8` 时，因为每段
都要保存 end-state/transition，两个口径进一步升到 383,336,448 / 282,673,152 B。
这条 [HYPIC-lite 审计](gpu/HYPIC_LITE_ZH.md) 是能力边界和未来 TTFT 消融，不是完整
HYPIC 复现，也不能支撑“同时保持 14× 并消除 suffix 重建”的结论。因此 proposal 的
核心收敛为 **compressed split-depth document memory on-device / capacity-first**；TTFT
优化保留为重要但次级的 systems objective。

真分页 infra 最初得到的是可归因的跨 backend 负结果边界。Job `237281` / Trial
`1840009` 在 8 个 PG-19 train-only windows 中只有 5 个完成 semantic row，3 个在
full-attention layer 31/31/19 未通过 Transformers eager 与 vLLM Triton
`unified_attention` 的数值兼容门。这是 **TF-eager compatibility negative**，不能
证明 page-reuse layout 错误或正确；LongBench validation/test-v2 和性能 benchmark
都未运行。后续 same-kernel fair v2 Trial `1840486` 已在同一 `unified_attention`
callable 中完成 fresh full-copy 与 shared-document/private-tail reuse 对照：8/8 workload
的全部 token/logit bitwise exact；cached-document TTFT ratio `1.009029`，没有加速；
cache 增量 peak allocated 约低 `50.4%`，物理 copy 中位省 `80 MiB`。absolute 模型 peak
只差约 `83.64 MiB`，范围限于 Q16/batch1/single-request/10 full layers，不能外推 F1、
多 query 或总显存减半。
详见 [vLLM paged Q16 负结果](RESULTS_GPU_QWEN35_VLLM_PAGED_Q16_FORMAL_NEGATIVE_2026-08-14_ZH.md)。
正式正结果见
[same-kernel fair v2 报告](RESULTS_GPU_QWEN35_VLLM_PAGED_FAIR_V2_2026-08-14_ZH.md)。

同文档 multi-fork resident Trial `1840837` 随后把该正交 ownership 对照扩展到
`N={1,2,4,8,16,32}` 个同时存活的请求对象。8 个 rank 的所有 N 上，fresh/reuse 的
token、逐步 full-vocab logits、最终 K/V/GDN 和同一请求跨 N 的结果均 exact。4095-token
partial-tail stress 下，可重放 full-attention pool 从 fresh `80+90N MiB` 降为 reuse
`80+5N MiB`；N=32 节省 `2720 MiB`，PyTorch absolute peak allocated 中位数相差约
`2.661 GiB`。这闭合了单文档常驻 request-capacity 的线性机制证据，但模型步仍为单 CUDA
stream round-major 顺序执行，不能冒充 concurrent serving、throughput/TTFT、ragged、
多文档、NVML 或下游质量。详见
[multi-fork resident 报告](RESULTS_GPU_QWEN35_VLLM_PAGED_MULTIFORK_RESIDENT_2026-08-14_ZH.md)。

## 1. 研究背景与 revision 给出的出发点

本 proposal 以本地 [CoMem revision PDF](/Users/liuhanzuo/Downloads/112_CoMem_Reusing_Transformer_.pdf) 为直接起点。Revision 已经把系统边界澄清得比较完整：

- 在 Qwen3-8B、相同 selected chunks、相同 pack 和相同 LoRA 的 `j=0` 与 `j=12` 对照中，Read latency 从 931.9 ms 降到 664.4 ms，即 1.403x；RULER Cohort B 从 99.19 降到 96.07。
- `j=6/9/12/18` 形成逐渐加速、逐渐掉点的部署曲线；`j=18` 出现明显质量坍塌。
- 连续前缀 `h_12` oracle 恢复到 99.19，说明主要误差来自 chunk-local Write 丢失下层文档上下文，而不是上层 suffix 本身没有能力。
- Overlap-Write 的 `w=32` 将 synthetic multikey 从 92.5 提升到 98.5，只增加约 5.7% 的理论 Write FLOPs，不增加持久化字节或在线 Read。
- BF16 `h_12` 为 8 KiB/token；128K token 约 1 GiB，16M token 约 128 GiB。固定 Read 可以保持有界，但 store、索引和选择器仍随语料增长。
- 一次性查询不一定获益。相同 adapter、包含 Write 的流水线在 8K/16K 仍慢于 dense，从 32K 才开始交叉；是否值得写入 store 取决于后续查询次数。
- 在等延迟边界上，raw replay 的结果可能优于 CoMem，且选择器差异会成为一阶因素。因此后续研究必须冻结证据和 selector，再单独研究表示与系统开销。

这些结果意味着端侧研究的重点不应只是“能不能跑”，而应是：低比特表示是否改变质量边界、统一内存是否改变分层代价、真实复用次数是否足以摊销 Write。

## 2. 文献空缺判断

### 2.1 结论

截至 2026-08-11，这个大方向不是空白，但下列交叉点仍有明确空缺：

> **hybrid LLM 上的压缩 split-depth residual + 必要 lower replay state + Apple 统一内存/SSD 运行时 + 相同证据的持久字节/TTFT/质量/能耗 Pareto 评估。**

没有找到一篇工作同时覆盖以上五项。这个结论是基于公开文献检索，不应表述为对未公开工作的绝对排除。

### 2.2 与最接近工作的边界

| 工作 | 保存对象 | 主要目的 | 低比特持久化 | Apple 实验 | 与本项目的关键差异 |
|---|---|---|---|---|---|
| CoMem revision | 一个 split 的逐 token `h_j` | 跨查询复用文档的下层深度 | 未评估 | 否 | 本项目的算法起点 |
| [HYPIC](https://arxiv.org/abs/2607.01299) | 每段、每层的 full-attention KV 与 linear transition/end-state | hybrid LLM 任意位置 segment cache 组合 | 否 | 否 | 最高相关竞品；它“多存全层状态、少重算”，本项目“少存 split 状态、多重算 suffix” |
| [MLX-LM prompt cache](https://github.com/ml-explore/mlx-lm) / [SGLang RadixAttention](https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf) | 全模型 prefix cache | 相同前缀的精确跨请求复用 | 否 | MLX 是 | 必须作为 exact 质量与 warm TTFT 基线；Q-CoMem 只有在持久字节—TTFT Pareto 上更好才成立 |
| [CacheBlend](https://arxiv.org/abs/2405.16444) | 多个位置无关 chunk 的 KV | RAG 多文档 KV 融合与选择性重算 | 否 | 否 | 对多文档/随机顺序 claim 构成直接威胁，但目前主要面向纯 attention 模型 |
| [KVTuner](https://arxiv.org/abs/2502.04420) / [KIVI](https://arxiv.org/abs/2402.02750) | 逐层 mixed-bit / uniform low-bit KV | 压缩 decode/prefix KV | 是 | 否 | 已覆盖 layer-wise mixed precision；本项目不能把“不同层不同 bit”本身作为新颖性 |
| [HCache](https://arxiv.org/abs/2410.05004) | 各层 hidden states | 从存储恢复完整 KV，重叠 I/O 与 GEMM | 否 | 否 | 保存和恢复的是层级 state，目标是 serving state restoration，不是 query-selected suffix Read |
| [KV-Direct](https://arxiv.org/abs/2603.19664) | residual checkpoint | 按需重建 KV、限制 decode cache | 未量化 residual | M3 Max + MLX | 目标是自回归 KV 冗余和有界 cache，不是稳定文档的跨查询深度复用 |
| [Agent Memory Below the Prompt](https://arxiv.org/abs/2603.04428) | 全层 Q4 KV | 多 agent cache 落盘与恢复 | 是 | M4 Pro 24 GB | 已覆盖“Apple + SSD + Q4”，但保存对象仍是完整 KV，而不是单一 `h_j` |
| [LLMCache](https://arxiv.org/abs/2512.16843) | 任意层中间 activation | 对语义相似输入复用计算 | 否 | 否 | 主要在 BERT/GPT-2 上验证，没有 CoMem 的固定证据和 suffix read protocol |
| [RSCE](https://aclanthology.org/2026.knowfm-1.11/) | 每篇文档一个 mean-pooled residual vector + fact block | 极端上下文压缩与 O(1) 注入 | 不是本问题 | 否 | 丢弃 token 级结构，回答的是极端压缩和 latent injection，不是可读的 token-level suffix interface |
| [CacheNotes](https://aclanthology.org/2026.eacl-long.309/) | 任务感知的紧凑可复用 KV | 一次压缩语料，多次回答复杂问题 | 未聚焦 Apple residual | 否 | 需要 task-focused compression，不研究“预付多少 Transformer 深度” |
| Q-CoMem-Edge | 量化的单一 `h_j` | 端侧稳定语料的重复查询 | 是 | M4 Pro | 联合研究 `j x w x bits x tier x reuse`，并做公平端到端对照 |

需要特别避免四个过强 claim：

1. 不能声称首次用 hidden state/residual 恢复或复用 LLM state，HCache 和 KV-Direct 已经覆盖相关思想。
2. 不能声称首次在 Apple 上持久化低比特模型状态，Q4 KV persistence 已经在同为 24 GB 的 M4 Pro 上做过。
3. 不能声称首次在 hybrid LLM 上复用/组合 recurrent state；HYPIC 已经正面解决这一问题。
4. 不能声称首次做 layer-wise mixed-bit cache；KVTuner 已经系统研究 KV 的逐层敏感度与位宽搜索。

可以守住的 claim 是：**系统研究 hybrid LLM 上“只持久化压缩 split residual 与必要 lower state、重算 suffix”的端侧内存—计算折中，并把它与 Apple 统一内存/SSD 的真实端到端摊销结合起来。**

## 3. 核心研究问题与假设

### RQ1：中间 residual 能压到多少 bit？

在固定 `j`、固定 selected chunks 和固定 suffix 的条件下，比较 BF16、INT8、INT4、INT2，以及混合 Q4/Q8。研究 per-token、per-group scale、裁剪、异常值旁路和旋转对下游质量的影响。

- H1：INT8 应接近 BF16，可作为正确性基线。
- H2：INT4 可能在 `j <= L/3` 时保持可接受质量，但越深的 `h_j` 对扰动越敏感。
- H3：少量异常值通道使用 Q8/BF16、其余使用 Q4，会优于统一 Q4。

这些都是待验证假设，不作为预设结论。已有 [QuaRot](https://arxiv.org/abs/2404.00456) 表明旋转可以缓解 residual/activation outlier，但它研究的是端到端 W4A4 推理；这里需要重新验证“持久化后再注入 suffix”的误差传播。

### RQ2：压缩节省的是容量，还是也能降低在线延迟和能耗？

Apple CPU 和 GPU 共享统一内存，不存在传统离散 GPU 的 H2D 复制。RAM-resident Q4 的收益主要来自较少的 DRAM 读取，代价是反量化；SSD-resident Q4 还会减少 page-in 和文件读取量。因此必须分别测量：

- RAM-resident：`fetch + dequant + suffix Read` 是否快于 BF16 Read；
- SSD-resident：压缩后的 I/O 节省是否覆盖 mmap/page fault 和反量化；
- 长时间运行：单位回答能耗和热降频是否改善。

### RQ3：什么负载才值得写 CoMem store？

令一次文档 Write 时间为 `T_write`，单次 raw replay 为 `T_replay`，Q-CoMem 单次在线成本为 `T_fetch + T_dequant + T_read`。在相同 decode 下，理论 break-even 为：

```text
G* = T_write / (T_replay - T_fetch - T_dequant - T_read)
```

本项目不只报告单次最快值，而是测量不同语料长度、生成长度、存储层级和查询次数下的 `G*`。如果分母小于等于零，则该配置永远不应被选中。

### RQ4：不同 split depth 是否应该使用不同 bit width？

CoMem 的一个部署配置只保存某个 split 的 `h_j`，并不保存所有层的 activation。因此这里的 layer-wise mixed precision 不是给每层都保留一份量化 residual，而是学习深度到精度的映射：

```text
b*(j) = minimum b such that quality_drop(j, b) <= epsilon
```

深层 residual 可能携带更集中的语义信息，对量化扰动更敏感；也可能因为距离输出更近、误差传播层数更少而更稳定，不能预设方向。使用 BF16 `h_j` 的 suffix logits 作为同深度 teacher，定义：

```text
S(j, b) = KL(p(y | h_j) || p(y | dequant(quant_b(h_j))))
```

再根据 KL、top-1 agreement 和自然任务质量选择最低可用位宽。第一阶段得到全局 `b(j)`；第二阶段再研究 chunk-wise 和 channel-wise Q4/Q8 混合。每个 chunk 默认只保存一个 `(j, b)`，避免因为保存所有层而抵消 CoMem 的容量优势。

除 split residual 的 `b(j)` 外，完整 replay state 还需要第二级策略：split 以下每个 cache
layer 可以选择不同位宽。令组件 `c=0` 表示 residual，`c=1..j` 表示 lower cache layer，
校准得到每个候选位宽的实际字节 `M(c,b)` 和扰动 `D(c,b)`，求解：

```text
minimize    sum_c D(c, b_c)
subject to  sum_c M(c, b_c) <= memory_budget
            b_c in {2, 4, 8, 16}
```

这里的 `D` 可以先用重建误差，正式 H20 校准则使用 held-out prompt 的 first-token KL；
选择后必须在未参与校准的自然任务上自由生成验证。代码采用 multiple-choice knapsack 的
精确 Pareto 动态规划，不把字节预算粗糙分桶。这个设计的潜在贡献是 **split residual 与
hybrid lower state 的联合预算分配**，不是“首次逐层 mixed-bit KV”（KVTuner 已覆盖后者）。

### RQ5：量化、深度和上下文修复之间如何联合选择？

联合变量为：

```text
split depth j
depth-aware bit width b(j)
overlap width w
optional chunk/channel precision
placement tier m
selection budget k
expected reuse count G
```

目标不是寻找一个全局最优配置，而是得到可复现的 Pareto frontier，并训练或拟合一个轻量 cost model，在部署时选择 raw replay、BF16 CoMem、Q-CoMem 或 Q4 KV。

### RQ6：split replay 相比 exact prefix cache 到底少存了什么？

比较时必须计算完整持久状态，而不只计算 residual 文件：

```text
S_qcomem(j) = S_quantized_residual(j)
              + S_full_attention_KV_below_j
              + S_linear_recurrent_state_below_j
S_prefix = S_attention_KV_all_layers + S_linear_state_all_layers
```

对于 Qwen3.5 这类 hybrid 模型，linear recurrent state 基本不随 token 数增长，而 full-attention KV 随文档长度线性增长。浅 split 可能显著少于 full prefix cache，但 suffix 重计算更多；深 split 相反。主结果必须画出 `persistent bytes/document × warm TTFT × downstream quality` Pareto frontier，不能只用 residual 的 3.56x 压缩率代表整个系统。

### 3.7 卖点层级：一个数量级近无损，而不是 30x 无损

当前 35B pilot 已经把三种可能的表述分开：

1. **主论文卖点**：state-type-aware mixed precision。`Q4 residual + Q4 full-attention
   cache + Q8 linear state` 在 depth 7 的 64-sample validation 上，完整持久状态相对 exact
   prefix 小 14.10x，mean F1 相对 dense 为 +0.0012，Qasper/2Wiki delta 为
   -0.0112/+0.0135；当时冻结、现已消费的 legacy test 36–67 为 14.12x、mean F1 delta -0.0087，
   Qasper/2Wiki delta -0.0175/0，两轮都通过预注册 mean margins。相反，将 linear state
   降到 Q4 会明显掉点。这只作为历史容量/质量证据；对新 adapter 不是 untouched test。
2. **系统卖点**：内存—计算 Pareto。exact prefix latency 更低，Q-CoMem persistent bytes
   更少、Write 更快；二者都不是全面支配另一方。
3. **极限压缩边界**：全 Q2 可达 30.39x，但 mean F1 从 0.410 降到 0.177。它应作为
   rate-distortion 图的失败端点，不能宣传成“30x 仍无损”。

新的逐层联合验证进一步收窄了第 1 点：当前主配置应称为 **state-type-aware static
precision**，而不是已经证明优越的 layer-wise mixed precision。same-memory layer-wise
策略只多压缩约 2.8%，质量却更差；逐层 mixed bit 目前是待改进方法与消融，不是主卖点。

### 3.8 HYPIC-lite 的负面 Pareto 边界：少算 suffix 不等于少存状态

[HYPIC-lite 参考原型](gpu/HYPIC_LITE_ZH.md) 将 suffix TTFT 对照拆成 naive end-state
reuse、可组合 affine transition 和 seam `w=0/8`。对于固定的一层输入，GatedDeltaNet
segment 满足：

```text
S_out = T_C @ S_in + S_C|0
```

当前 Transformers 公共 cache 只暴露 recurrent end-state 和 conv tail，不暴露 `T_C`。
原型通过内部 FLA `chunk_gated_delta_rule` 做第二次 `S0=I`、zero-value 调用提取 dense
transition；这是 reference path，生产实现需要一次 prefill 同时发射 `(T_C, S_C|0)` 的
fused kernel。更重要的是，Qwen3.5 suffix 的 linear/full-attention 层交错：linear
transition-only 或 transition + seam-only KV 都不能通过公开 cache 接口跳过 full-attention
所需的全部文档 token/KV。当前能跑到 query logits 的 full-suffix local-KV splice 省略了
跨段 hidden-state 影响，只能标为 approximate，不能写成端到端 exact composition 或完整
HYPIC 复现。

depth 7、4096-token 的分项账如下。四段使用 `w=8`，其 442,368 B seam KV 是在线边界
预算，单独列出而不并入持久 suffix 合计：

| 场景 | full suffix KV | `S_C|0` | `T_C` | conv tail | seam KV budget | suffix 持久合计 |
|---|---:|---:|---:|---:|---:|---:|
| 1 segment，runtime state FP32 | 75,497,472 B | 50,331,648 B | 25,165,824 B | 1,572,864 B | 0 B | 152,567,808 B |
| 1 segment，all-BF16 payload | 75,497,472 B | 25,165,824 B | 25,165,824 B | 1,572,864 B | 0 B | 127,401,984 B |
| 4 segments，`w=8`，runtime state FP32 | 75,055,104 B | 201,326,592 B | 100,663,296 B | 6,291,456 B | 442,368 B | 383,336,448 B |
| 4 segments，`w=8`，all-BF16 payload | 75,055,104 B | 100,663,296 B | 100,663,296 B | 6,291,456 B | 442,368 B | 282,673,152 B |

`T_C` 必须按 32 个 value heads 而不是 16 个 key heads 计数：Qwen3.5 会把 keys repeat
到 value heads，而 `g/beta` 又是 per-value-head，所以每个 value head 有不同的 dense
transition。四段 BF16 transition 因而是
`24 layers × 4 segments × 32 heads × 128² × 2 B = 100,663,296 B`。

在计算侧，当前 Q-CoMem 的 4k suffix rebuild 是 `4096 × 33 = 135,168` 个
token-layer forwards；四段 `w=8` 的原型在线 seam 账面只剩 `3 × 8 × 33 = 792`，减少
99.41%。但这并不补偿上述 per-document persistent state。Q8/Q4 compressed-HYPIC 的
payload-only 下界是一段约 60.75/30.38 MiB、四段约 134.79/67.39 MiB；它们未计量化
metadata，缺少 quantized transition compose/KV kernels，也没有质量结果，只能列为未来
研究方向，不能当成已实现 baseline。

因此主线选择是有意的：Q-CoMem 以 suffix 重计算换取一个数量级的文档持久容量；HYPIC
类方法以更多 per-segment/all-layer state 换 TTFT。论文首先验证前者能否扩大端侧
model-plus-context working set，再把 suffix composition、compressed-HYPIC 和 fused
dequant 作为次级 TTFT–bytes Pareto 探索，而不是预设二者可以无代价合并。

“让端侧部署更大模型”的含义也必须准确。Q-CoMem 不压缩模型权重，不能让本来放不下的
BF16/量化权重凭空放入设备。它降低的是权重之外的状态预算：

```text
M_device >= M_weights + M_adapter_shared + M_active_workspace
            + N_hot_documents * S_state + M_runtime
```

因此它可以让一个已经量化、权重勉强能放入统一内存的大模型，同时常驻更长上下文或更多
hot documents，减少 swap/eviction；也可以把节省出的状态预算让给稍大的量化模型。但论文
应写成 **enabling larger model-plus-context working sets**，而不是 model compression。
当启用 Trial `1840023` 的 answer-LoRA 时，`M_adapter_shared=106,758,144 B`；这是
每个模型进程一份的 FP32 共享成本，不是每文档成本，也不能省略。在当前
Q16/Q4 state 分别为 `36,367,872/10,130,160 B` 时，只就公式中的
`M_adapter_shared + N*S_state` 两项，Q4+adapter 从 `N=5` 起低于 Q16；模型权重和
active/runtime 项仍必须另行报告。

为防止 test contamination，Qasper/2WikiMQA 的 source index 已按 `pilot 0–3 /
calibration 4–5 / mixed validation 6–35 / legacy test 36–67 / frozen test-v2 68–99`
冻结。最终 adapter 与 bit policy 必须先在 train/calibration/validation 上选定，再在
test-v2 上只运行一次；不能根据 test-v2 结果继续调 depth、bit 或 LoRA。
Trial `1840023` 只用独立 official-train heldout 26 条选 step 128，然后才读取
validation 6–35 做 full-state post-selection 评测；这 60 条现在已被消费，不得用来
反向重选 checkpoint 或调整新 surface/objective。test-v2 68–99 仍未读取。

用已测 16k tensor bytes 可以把这个系统收益写得更具体。下表只给 persistent-state 容量，
不扣模型权重和 active workspace；H20 与 MLX 的 tensor shape 相同，但最终 Apple 数字必须
在 unified memory 上复测：

| persistent-state budget | exact prefix（381.875 MiB/文档） | d7 Q4-residual/Q8-state（38.387 MiB/文档） |
|---:|---:|---:|
| 2 GiB | 5 个 16k 文档 | 53 个 16k 文档 |
| 4 GiB | 10 个 16k 文档 | 106 个 16k 文档 |
| 8 GiB | 21 个 16k 文档 | 213 个 16k 文档 |

因此最稳的端侧表述不是“凭空部署大十倍的模型”，而是：**在量化权重已经可驻留的前提
下，把可热驻留的稳定长文档或等价上下文 token 数扩大约一个数量级，并降低 eviction、
swap 与重复 Write。** 这也是 Apple unified memory 场景最值得单独验证的 selling point。

## 4. 方法设计

### 4.1 Write

1. 默认把一篇稳定文档连续执行层 `0..j-1`，得到 token-level `h_j`。
2. 同时保存这些 lower layers 的 full-attention KV，以及 linear-attention 的 convolution/recurrent end-state。
3. `h_j` 量化后写入 store；第一版保持 replay state 为 BF16，第二版再单独校准 KV/state 位宽，避免把两种误差混在一起。
4. 每个 query 从不可变 document state fork 出独立可变 cache；query prompt 首次以 chunk continuation 方式通过 lower layers，之后生成 token 只增量更新 state。
5. chunk-local/overlap Write 保留为低 Write 延迟或 PIC 扩展消融，不再作为自然任务的默认接口。
6. store header 记录模型、tokenizer、adapter、`j`、dtype、量化版本、原始 token 范围、cache schema 和文档版本。

### 4.2 Residual quantizer

第一版使用 MLX 原生 affine per-token/per-group 量化，group size `g=64/128`。每组保存 packed integers、FP16 scale 和 FP16 bias：

```text
q, scale, bias = affine_quantize(x_group, bits=b)
x_hat = q * scale + bias
```

若 scale 和 bias 都用 FP16，近似每 token 存储为：

```text
S_b = b*d/8 + 4*d/g bytes
```

以 `d=4096, g=64` 为例：

- BF16：8192 bytes/token；
- Q8：约 4352 bytes/token，约 1.88x 压缩；
- Q4：约 2304 bytes/token，约 3.56x 压缩；
- 128K store 从约 1 GiB 降到约 288 MiB；
- revision 中一次 50.3 MB 的 BF16 selected pack，Q4 后约为 14.1 MB。

Affine 版本优先用于建立可靠基线，因为 MLX 已提供 GPU quantize/dequantize 和真实 bit packing。后续再比较无 bias 的 symmetric 格式；后者理论容量更小，但必须把自定义 kernel 的误差和性能单独列为 ablation。

#### 4.2.1 Depth-aware mixed-precision policy

对候选深度 `J={j_1,...,j_n}` 和候选位宽 `B={2,4,8,16}`，校准器依次执行：

1. 捕获 BF16 `h_j` 并运行 suffix，得到同深度 reference logits；
2. 对每个 `b` 量化 `h_j`，测量 residual error、logit KL 和 top-1 agreement；
3. 在满足预注册质量阈值的配置中选择实际持久化字节最少的 `b*(j)`；
4. 若没有低比特配置满足阈值，则该深度回退到 BF16 或被部署策略拒绝。

输出不是一个统一的 Q4 checkpoint，而是一张可审计的策略表，例如：

```text
depth 6  -> Q4
depth 9  -> Q4
depth 12 -> Q8
depth 18 -> BF16 / reject
```

主论文先研究“每个部署深度一个 bit width”。把同一次 Read 中的不同 chunk 放在不同深度，会引入异构 suffix 调度和新的模型接口，应作为扩展实验而不是第一阶段的必要条件。

第二版只在第一版失败时增加：百分位裁剪、异常值 channel side buffer、Hadamard rotation 或按 chunk 敏感度分配 Q4/Q8。避免一开始把所有量化技巧混在一起，导致无法定位收益来源。

### 4.3 Select 与公平性控制

主实验预先冻结每个 query 的 chunk IDs 和顺序，所有方法读取完全相同的证据。选择器实验单列：

- 主隔离实验：frozen BM25 ranking；
- 鲁棒性实验：iterative BM25 与 frozen BGE；
- 所有表格同时报告 evidence recall 和 answer quality。

这样可以把“量化误差”“depth interface 误差”和“retrieval 误差”分开。

### 4.4 Read

1. 从 RAM 或 mmap store 读取 packed document residual、scale 与 lower replay state。
2. 在 Apple GPU 上反量化 `h_j`。
3. query 在 lower layers 读取并更新文档 KV/recurrent state，生成与 continuous-prefix 一致的 query residual。
4. 把 document/query residual 送入层 `j..L-1` 得到首 token；生产版本为 suffix 建立增量 decode cache。
5. 与 exact full-prefix cache 比较 warm TTFT 和总持久字节；只与 dense full recompute 比较会高估贡献。

性能版本使用 `mlx.core.fast.metal_kernel()` 实现 fused unpack + dequant，必要时进一步融合第一层 suffix 的 RMSNorm。MLX 官方支持通过 Python/C++ JIT 编译自定义 Metal kernel：[Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)。

#### 4.4.1 当前 MLX reference implementation

项目现在已经在 `mlx.core`/MLX-LM 层实现冻结的 `d7-r4-a4-l8` 路径，而不是调用高层
generate helper：

1. lower Write 直接推进 Qwen3.5 的 `KVCache` 与 GatedDeltaNet `ArraysCache`；
2. 任意 shape tensor 先 flatten/pad，再用 MLX affine Q2/Q4/Q8 真正 bit-pack；
3. packed store 分别记录 residual、full-attention KV、conv/recurrent state、scale/bias；
4. query 为每次请求 fork 独立 cache，避免 recurrent kernel 修改共享文档边界；
5. suffix 只投影最后一个 token 的 vocabulary logits，避免生成时物化整段 vocab tensor；
6. safetensors 可保存/恢复完整 hybrid replay state，并保留 class/type/bit metadata。
7. residual 与每个 lower cache layer 可独立选择 Q2/Q4/Q8/Q16，并由实测
   storage–distortion profile 在指定字节预算下自动分配。
8. 多篇文档可以按冻结顺序连续写入同一个 lower cache；该 exact baseline 等价于 token
   级拼接，但不声称支持任意重排或删除后的零重算。

4 层 tiny Qwen3.5 hybrid regression 中，dense、exact prefix、Q16 replay 连续 3 个 greedy
token 完全一致，Q4/Q4/Q8 store 的保存/恢复 logits 也逐元素一致。新增 regression 还验证
了三文档连续写入的 exact 等价性和自动逐层 bit 的预算约束。正式 Apple 性能入口为
`make mlx-replay`；它固定模型 revision、配置轮换、逐长度 warm-up，并复用统一的 AC、thermal、
swap 有效性门槛。MLX 与 H20 reference path 都已加入 suffix 增量 cache；MLX 512-token
修复诊断中 packed replay generation 为 dense 的 1.24x。尚未做 fused Metal dequant，且
完整正式 Mac 运行受 swap/thermal gate 判为无效，因此该数字仍是诊断值而不是最终上界。

H20 线已完成 layer sensitivity、60-sample mixed validation 和部署基准。depth 7 的
residual 加 7 个 lower cache layer 由 8 张 H20 分工校准 Q2/Q4/Q8/Q16；
`aggregate_layer_sensitivity.py` 输出“与冻结策略同内存、再减 25%、极限 Q2”策略。
修正 document/query suffix chunk boundary 后，60 条联合 validation 和 8-rank 部署 hard gate
均通过。完整数字分别见 `RESULTS_GPU_MIXED_VALIDATION_2026-08-12_ZH.md` 与
`RESULTS_GPU_DEPLOYMENT_2026-08-12_ZH.md`。

质量恢复线已超过早期 1-step smoke。Trial `1834056/1834193` 的 36-module、answer-free
query-KL 路线完成了 native functional-cache 能力验证，但 step 0/64/128 相对 disabled
的下游点估计均为负，是保留的历史负基线。正式 Trial `1840023` 改为
answer+EOS supervision、任务均衡、answer-position teacher preservation 和 156-module
full-attention+GDN surface；只按 independent official-train heldout 选 step 128。同 job
60 条 validation 的 step 128 vs frozen-disabled 为 `+0.005559`，CI 跨0；该结果是小幅
正点估计与定向修复，不是显著质量恢复。

suffix TTFT 线目前只完成独立的 HYPIC-lite 算法原型、CPU affine correctness tests 和上述
静态 tensor ledger，尚未提交 H20。原型默认 same-packed reference 使用预注册的
frozen-static `[8,8,8,4,8,8,8]`，只读 validation index 6--35，并拒绝冻结的 test-v2。
它不能取代部署结果，也不能写成 HYPIC 实测复现；后续状态和验收条件见
[下一阶段预注册](NEXT_STAGE_EXPERIMENTS_ZH.md)，当前能力边界见
[HYPIC-lite 说明](gpu/HYPIC_LITE_ZH.md)。

真分页 kernel 线已运行 vLLM Q16 的 Trial `1840009`，但在 PG-19 train-only 阶段
因 TF-eager vs `unified_attention` compatibility 失败而 fail-closed，未进入 LongBench 或
性能 benchmark。这不是 page-reuse layout 结论。可归因的 same-kernel fair v2 已由
Trial `1840486` 完成：同一 `unified_attention` kernel 下 8/8 全步 token/logit exact，
reuse 避免中位 `80 MiB` 文档 block copy，combined unique storage 减少
`87,819,520 B`，cache 增量 peak allocated paired ratio 为 `0.496010`。TTFT ratio
`1.009029`，因此当前是 memory 正结果而非速度正结果；batch 1/单请求边界也意味着尚未
完成 serving capacity 曲线。

### 4.5 Tiered store 与运行时决策

三级存储：

- Hot：RAM-resident packed residual；
- Warm：本地 NVMe mmap；
- Cold：保留 raw text/token IDs，按需重写 residual。

Apple 统一内存意味着把数组从“GPU tensor”移到“CPU tensor”不会减少机器总内存；真正降低容量的是量化、淘汰和 SSD。MLX 的数组位于统一内存，CPU/GPU 计算流可直接访问同一内存池：[MLX Unified Memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)。

运行时 cost model 根据 `store size、selected bytes、j、b、tier、预计 G、预计输出长度` 选择执行路径。该选择器必须使用离线 profile 拟合，不能用测试答案质量作为 oracle。

### 4.6 编辑与失效

每个 chunk 保存 source token hash 和相邻依赖。文档局部修改时：

- 重写被修改 chunk；
- 若 `w>0`，同时重写其后一个受左 overlap 影响的 chunk；
- 不允许旧模型/旧 adapter/旧 tokenizer 的 residual 被静默读取；
- 分别报告 full rebuild 与 1%/5% 局部 edit 的写放大和完成时间。

## 5. 实验设计

### 5.1 硬件与模型

本地目标机：MacBook Pro，Apple M4 Pro，16-core GPU，24 GB 统一内存。

模型分三档：

1. 开发正确性：Qwen3-0.6B/1.7B；
2. 端侧主结果：当前项目已有的 Llama-3.2-3B-Instruct-4bit，加一组 Qwen 3B/4B 级模型；
3. 与 revision 对齐：Qwen3-8B，只做少量关键点，最终训练与完整 sweep 可转到单张数据中心 GPU。

跨模型的 `j` 使用归一化深度，例如 `j/L in {1/6, 1/4, 1/3, 1/2}`，再补充 revision 的 `j={6,9,12,18}` 对齐实验。

### 5.2 主实验矩阵

| 轴 | 主设置 | 扩展设置 |
|---|---|---|
| split `j` | `L/6, L/4, L/3` | `L/2` 作为失效边界 |
| overlap `w` | `0, 32` | `64, 128` |
| bits `b(j)` | 每个 `j` 独立测试 `16, 8, 4` | `2, mixed 4/8` 与自动最低位宽策略 |
| tier | RAM, mmap SSD | cold rewrite |
| top-k | 4, 8, 12 | 根据等延迟预算变化 |
| corpus | 32K, 128K, 1M token | 4M 仅做系统 scaling |
| generation | 1, 32, 128 token | 512 token 观察 decode 稀释 |
| reuse `G` | 1, 2, 4, 8, 16, 32, 64 | 直到 break-even 稳定 |

完整笛卡尔积过大。先在 1.7B 上筛选 Pareto 配置，再在 3B/4B 上只复现 6-10 个关键点，8B 只复现 3-4 个结论性配置。

### 5.3 Baselines

必须实现或严格对齐：

1. Dense full-context；
2. exact full-model prefix cache（GPU Transformers 与 Apple MLX-LM），输出逐 token 对齐；
3. matched raw replay `j=0`，相同 selected chunks；
4. BF16 replay-CoMem；
5. Q8/Q4 replay-CoMem，并统计 residual + lower state 的完整持久字节；
6. KIVI/KVTuner 风格的 uniform/mixed-bit KV baseline；
7. HYPIC-lite：naive end-state、transition composition、seam `w=0/8`；按
   [审计口径](gpu/HYPIC_LITE_ZH.md) 分开报告 transition-only、seam KV 与 full suffix
   local cache，明确 approximate/reference 状态，不冒充完整 HYPIC；
8. CacheBlend（在其原生支持的纯 attention 模型上复现）；
9. 等 TTFT raw replay：允许 replay 使用更多或更少 chunks，使时延落在 Q-CoMem 的 ±5%。

HCache、KV-Direct、RSCE 与 CacheNotes 可作为机制边界和 reported reference；若无法在同一 MLX runtime 中公平重现，不应把跨论文数字直接排成速度排行榜。

### 5.4 任务

- 机制诊断：RULER Cohort B、NIAH、多 key、variable tracking；
- 自然长文档 QA：LongBench 子集、Qasper、2WikiMultiHopQA；
- 多轮稳定记忆：LoCoMo；
- 代码库重复查询：RepoBench-C 或固定仓库上的多 query protocol；
- 编辑负载：稳定手册/代码仓，注入 1% 与 5% 文档修改。

自然任务必须以“同一 corpus 对应多次 query”的方式组织，不能把互不相关的单次样本伪装成重复查询收益。

### 5.5 指标

质量：

- task score、macro average、paired bootstrap 95% CI；
- evidence recall；
- BF16 与量化 logits 的 KL divergence、top-1 agreement；
- 按 split、token position、chunk type 的 residual reconstruction error。

系统：

- Write latency 与吞吐；
- Select、fetch、dequant、suffix Read、decode 分段时间；
- TTFT、端到端 latency、decode tok/s；
- store bytes/token、selected bytes/query、SSD read bytes/query；
- MLX active/peak memory、进程 RSS、系统 memory pressure 与 swap；
- p50/p95/p99，在单 query 和并发 `1/2/4` 下分别测量；
- 15 分钟持续负载下的能耗/answer、平均功率与热降频。

所有 latency 在 warm-up 后至少运行 20 次，报告中位数和分位数；冷 SSD、warm page cache、RAM-resident 分开报告。

### 5.6 笔记本运行前提

端侧性能数值只有在电源和系统状态受控时才可比较。正式 run 必须连接 AC、关闭低功耗
模式、开始与结束时没有热/性能 warning，并记录后台 CPU、memory pressure 和 swap
变化。模型与 tokenizer 使用 immutable revision，corpus/query split 和文档选择顺序
冻结。配置至少以多个全新进程重复，不能把同一进程里的连续计时当作独立样本。

本项目的自动门槛、失效条件和最小报告字段以
[实验协议](EXPERIMENT_PROTOCOL_ZH.md) 为准。电池或低功耗条件下允许做 correctness
diagnostic，但必须写入 `formal_result_eligible=false`，不得进入论文性能表。

## 6. 是否需要训练：GPU 还是 Apple NPU？

### 6.1 最小可行原型不需要训练

以下工作全部可以 training-free 完成：

- BF16 split Write/Read 的 MLX 复现；
- INT8/INT4 post-training quantization；
- scale/calibration 统计；
- RAM/SSD store、mmap、Metal dequant kernel；
- RULER/LoCoMo 质量测试和系统 benchmark。

所以第一阶段不应先申请大 GPU。先用本机 M4 Pro GPU 验证 Q4 是否存在可用的质量-容量点。如果 Q4 已经接近 BF16，论文可以以 PTQ + systems 为主。

### 6.2 论文级质量恢复大概率需要 GPU 训练

Revision 显示 `j=12` 的高质量结果依赖 suffix LoRA/self-distillation，而且不同 `j` 使用了不同 adapter。若要回答“量化后能否恢复质量”，至少需要以下一种训练：

- 每个关键 `j` 的 suffix LoRA distillation；
- quantization-aware LoRA，使 suffix 适应 `h_j` 的量化噪声；
- 学习 rotation/scale 或 mixed-precision router。

推荐损失：

```text
L = lambda_logits * KL(student_logits, teacher_logits)
  + lambda_hidden * MSE(student_upper_hidden, teacher_upper_hidden)
  + lambda_task * CE(answer)
```

Teacher 使用 matched raw replay 或 BF16 continuous-prefix oracle。为了适配 24 GB 内存，可先离线保存 teacher top-k logits/selected hidden targets，避免训练时同时常驻两份 8B 模型。

Trial `1840023` 已完成一个 35B 正式实例：在真实 document-prefill→query+answer
continuation 边界上，对 answer+EOS 使用 CE，对同一 answer positions 使用冻结 dense
teacher KL/hidden preservation，并对 QASPER/2WikiMQA 每个 global step 严格 4+4 均衡。
156 个 modules 覆盖 suffix full-attention/GDN 关键 projection，但不包 MLP。该设计
将 independent heldout loss 降低 `50.11%`，并使 60 条 validation 的 frozen-static
F1 相对 disabled 为 `+0.005559`；因 CI `[-0.019821,+0.030127]` 跨 0，这还只是
正点估计。训练的 whole-answer multi-token block 与真实 token-by-token decode 也不宣称
数值等价；26 条 heldout 诊断虽然 206 positions top-1 无分叉，mean KL 仍为
`0.000507959`。后续如要区分 objective、warm-start 和 surface 的贡献，必须在新的
train-heldout/未消费评测上做纯 cold-start、MLP 与 surface 消融，不得用已消费的
validation 6–35 反向重选。

### 6.3 Apple Neural Engine 不能替代这部分 GPU 训练

这里要区分三个概念：

- M4 Pro 的 Apple Neural Engine，即通常说的 NPU；
- M4/M5 GPU 上通过 Metal 执行的矩阵和自定义 kernel；
- [M5 GPU 内部新增的 Neural Accelerators](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)，它也不是 M4 Pro 的独立 Neural Engine。

当前 MLX 对外提供的计算设备是 CPU 和 GPU，不提供 ANE device：[MLX devices](https://ml-explore.github.io/mlx/build/html/python/devices_and_streams.html)。Apple 的官方训练工作流也明确把 PyTorch、JAX、TensorFlow 和 MLX 训练放在 Metal GPU 上：[Train your machine learning and AI models on Apple GPUs](https://developer.apple.com/videos/play/wwdc2024/10160/)。

Core ML/Core AI 可以在部署推理时把受支持的图调度到 CPU、GPU 和 Neural Engine；Core ML 也支持受限的 updateable model personalization，但这不等于一个可供任意 Transformer LoRA、动态 mask、split residual 和自定义量化 kernel 使用的通用 ANE 训练后端。参考：[Core ML on-device inference](https://developer.apple.com/videos/play/wwdc2024/10161/) 与 [on-device model updates](https://developer.apple.com/documentation/CoreML/personalizing-a-model-with-on-device-updates)。

因此结论是：

- **研究原型与训练：使用 MLX/Metal GPU。**
- **NPU：不够，也不是当前代码能直接选择的 MLX backend。**
- **后期产品化：可以把固定、受支持的子图转换到 Core ML/Core AI，验证 ANE 是否参与推理；retrieval、mmap、动态 packing 和自定义 dequant 仍可能留在 CPU/GPU。**

### 6.4 这台 M4 Pro 24 GB 能承担到什么程度？

| 工作 | 本机可行性 | 建议 |
|---|---|---|
| 0.6B/1.7B BF16 或 4-bit 推理、PTQ sweep | 很适合 | 全部本地完成 |
| 3B/4B 4-bit + LoRA，短 chunk、batch 1 | 可行 | gradient checkpointing，逐配置训练 |
| 8B 4-bit inference 与少量 LoRA 试验 | 勉强可做 | 控制序列长度并预留系统内存，避免 swap |
| 8B 多个 `j x bits x w` 的完整训练网格 | 不经济 | 使用单张 A100/H100/L40S 级 GPU |
| 8B BF16 全量训练 | 不适合 | 本项目也没有必要做 |

推荐算力策略：**M4 Pro 完成所有 systems 和小模型结论；只把最终 8B adapter/distillation 的少量关键配置放到数据中心 GPU。** 这既保留端侧论文的真实性，也不会让本地训练时间成为项目瓶颈。

## 7. 在 MacLLM-Bench 中的实现路线

当前 [manual_mlx.py](/Users/liuhanzuo/MacLLM-Bench/src/macllm_bench/manual_mlx.py) 已经具备显式 GPU stream、手写 KV cache、chunked prefill、同步点和内存测量，可作为系统测量底座。但 CoMem 需要绕过模型顶层 `__call__`，直接控制 embedding、`model.layers[:j]`、`model.layers[j:]`、norm 和 lm_head。

建议新增：

```text
src/macllm_bench/comem_model.py      # Qwen/Llama 的 split Write/Read adapter
src/macllm_bench/comem_quant.py      # Q8/Q4 pack、scale、Metal dequant
src/macllm_bench/comem_store.py      # manifest、mmap、版本和失效
src/macllm_bench/comem_select.py     # frozen ranking/BM25 接口
src/macllm_bench/comem_generate.py   # Read 与完整 decode cache 组织
src/macllm_bench/comem_bench.py      # 分段计时、质量、能耗和结果 schema
tests/test_comem_equivalence.py       # j=0/BF16/连续前缀正确性
```

实现顺序：

1. `j=0` 必须逐 token/logit 对齐当前 raw replay；
2. continuous-prefix `h_j` 必须对齐未切块 full forward；
3. 加 lower full-attention KV 与 linear recurrent-state replay，要求逐 token 对齐 continuous-prefix；
4. 实现 exact full-prefix cache，作为持久字节和 warm TTFT 基线；
5. 加 chunk-local Write，复现并分解 revision 的 interface gap；
6. 加 Q8/Q4 residual，再校准 lower KV/state mixed precision；
7. 再写 Metal fused dequant；
8. 最后加入 SSD mmap、selector 和必要的训练。

## 8. 预期贡献

如果假设成立，论文可以主张三项贡献：

1. **Representation contribution**：给出 split depth、residual bit width 与 lower-state precision 的联合率失真/任务质量曲线，并得到可审计的 sensitivity-aware policy；当前主配置实现约 14.1x 完整状态压缩，新意不建立在“首次 mixed bit”上，逐层 policy 只有在后续验证确实胜过 static 时才升级为主贡献。
2. **Systems contribution**：面向 Apple unified memory 的 capacity-first Q-CoMem runtime，
   支持 RAM/SSD 分层、版本化 store 和融合 Metal dequant；首先验证固定统一内存下能热驻留
   多少稳定文档/上下文，再完整报告 TTFT、active memory、I/O、tail latency、能耗和热行为，
   不把次级延迟目标包装成已实现优势。
3. **Deployment contribution**：一个经实测校准的 reuse-aware cost model，决定何时 raw
   replay、何时写 Q-CoMem、何时使用 exact prefix，使固定统一内存容纳更大的
   model-plus-context working set；当单一热 prefix 更适合 exact prefix 时也必须选择后者。

如果只有“BF16 CoMem 在 Mac 上跑通”，贡献不足；如果只有“把 `h_j` 转成 INT4 文件”，也很难与现有 activation/KV quantization 拉开差异。

## 9. 风险、反例与停止条件

| 风险 | 诊断 | 应对/停止条件 |
|---|---|---|
| Q4 residual 质量严重下降 | Q8 正常、Q4 在多个 `j` 掉点 >3 | 尝试 mixed Q4/Q8 或 rotation；仍失败则把主结论改为 Q8，不能宣称 4-bit 成功 |
| RAM-resident Q4 比 BF16 更慢 | dequant 大于带宽节省 | 融合 dequant；若仍慢，则只主张容量/SSD 收益，不主张 RAM latency |
| suffix composition 只能靠全层状态换 TTFT | full suffix store 回到或超过 exact-prefix bytes | 保留为负面 Pareto/quality ablation；研究 compressed transition/KV，但在 kernel 与质量验证前不升级为主方法 |
| equal-latency raw replay 在自然任务持续领先 | 相同 selector 下质量差 >5 | 缩小到高复用/大语料场景；若系统和质量都无优势，停止该路线 |
| break-even 查询次数过高 | 128K 下 `G*>50` 或常见负载达不到 | 只面向离线导入的个人知识库/代码库，不泛化到一次性文档 |
| selector 掩盖表示效果 | evidence recall 波动 | 主表冻结 chunk IDs，selector 单独做敏感性分析 |
| 模型/adapter 更新导致 store 失效 | hash 不匹配 | 强制版本隔离并报告 rebuild 成本 |
| 24 GB 出现 swap，污染结果 | `vm_stat`/memory pressure 异常 | 降模型或配置；带 swap 的运行不得进入主结果 |

## 10. 时间表与最小发表单元

### 8-10 周计划

- 第 1 周：Qwen/Llama split adapter，完成 `j=0` 与 continuous-prefix correctness。
- 第 2 周：lower KV/recurrent replay 与 exact prefix cache，完成逐 token correctness。
- 第 3 周：Q8/Q4 reference quantizer、store format、单元测试。
- 第 4 周：Metal fused dequant，RAM/SSD microbenchmark。
- 第 5-6 周：1.7B/3B 主实验，筛选 `j x w x bits` Pareto 点。
- 第 7 周：suffix LoRA 或 quantization-aware LoRA，只训练关键配置。
- 第 8 周：自然任务、equal-latency replay、break-even 与编辑实验。
- 第 9 周：持续负载、能耗、p95/p99、统计置信区间。
- 第 10 周：8B 关键点复现、论文和 artifact 整理。

### 最小发表单元

至少需要同时满足：

1. Q4 residual 相对 BF16 接近 3.5x 缩小，并报告 residual + lower state 的总持久容量；
2. exact replay 在至少一个自然重复查询任务上逐 token 对齐 dense，量化版满足预注册质量阈值；
3. 在固定端侧内存预算和受控质量下，相对 exact full-prefix cache 实测显著更多的 hot
   documents/context working set；容量必须包含 residual、lower state 和所有 per-document
   metadata，并单独记录 model/process-shared adapter，不能隐藏 suffix/full-attention cache
   或把共享 adapter 摊成不可见的每文档成本；
4. 完整报告 persistent bytes—active memory—warm TTFT Pareto。TTFT 是次级目标：允许
   capacity-first 配置慢于 exact prefix，但必须给出适用负载、reuse/break-even 和明确的
   负面边界；若声称 latency 改进，则必须由 Apple/H20 实测而非 token-layer 推算支持；
5. 在明确的 `G` 区间内胜过 equal-latency raw replay；
6. 两个模型家族、至少一个 3B/4B 端侧主模型；
7. 所有结论包含 write、fetch、dequant、read、decode 和 store 容量边界。

## 11. 最终判断

这个想法有研究空间，但最佳表述不是“CoMem 能不能部署在 Apple 上”，而是：

> **中间 residual 是否是一种比 token replay 和 full KV 更适合端侧重复查询的持久化状态；如果是，它在什么 bit width、什么 split depth、什么存储层级和多少次复用后成立？**

当前 35B GPU 证据已经从 mechanism diagnostic 扩展到自然任务和部署测量：exact
lower replay 通过逐 token gate；60 条 validation 上的 frozen static 达到 14.10×
观察性近无损 knee；8 个部署 workload 上的 mixed 持久状态为 full-prefix 的
1/14.41。全 Q4 和更激进位宽会掉点，证明不能把最低 bit 当成默认卖点。
同时，部署基准表明当前实现的 TTFT 与 active peak 尚未优于 exact prefix，所以算法
和 runtime 并未“全部结束”。HYPIC-lite audit 又表明，直接持久化 suffix full-attention
KV 与逐段 linear transition/end-state 会把每文档状态拉回甚至推过 full-prefix；四段
`w=8` 的 runtime/BF16 suffix payload 已达 383,336,448/282,673,152 B。它验证了一个重要
边界：不能为了 TTFT 无条件牺牲已经成立的 14× capacity 卖点。copy-on-write/paged state
仍是 active-memory 方向；suffix composition 只在新增字节受控时才是候选优化。

因此最稳、也最有端侧意义的论文定位是 **compressed split-depth document memory
on-device**：在模型权重已经可驻留的前提下，以 state-type-aware static precision 扩大
稳定文档和上下文 working set。TTFT 是必须测量和继续优化的次级系统指标，而不是当前
已验证贡献。未来的 Q4/Q8 compressed-HYPIC 可以探索新的 TTFT–bytes 点，但在量化
compose/KV kernel、真实持久字节和下游质量完成前，只能作为 future work。

第一阶段 PTQ 与系统可行性验证不需要训练，M4 Pro 的 Metal GPU 足够完成；数据中心
GPU 用于 35B correctness/下游对照与 adapter 训练。为了攻击 all-Q4/18× 档的质量损失，
仓库已分开保留 Interface LoRA、旧 36-module answer-free native-cache LoRA 和新 156-module
answer-supervised native-cache LoRA。旧 native run/attribution 是负点估计的历史基线；新 Trial
`1840023` 用独立 heldout 选 step 128，在同 job 60 条 validation 上相对 disabled 得到
`+0.005559`，但 CI 跨 0。因此当前可说“已知 answer-type 失败被定向修复、总体
点估计转正”，不能说“精度已显著恢复”。容量 claim 也必须把 `9.6609 MiB`/
文档 persistent state 与 `101.8125 MiB`/模型进程 adapter 分开；只按这两项从
第 5 个常驻文档起才低于 Q16。

infra 的 same-kernel 单请求门禁已经收官：Trial `1840486` 在同一
`unified_attention` kernel 上比较 request-owned full-copy 与
shared-document/private-tail，8/8 全步 token/logit bitwise exact；cache 增量 peak
allocated 约低 `50.4%`，但 TTFT ratio `1.009029`，没有速度收益。该结论不覆盖
F1、多 query、ragged batch、Q8/Q4、NVML peak 或 isolated kernel speedup。Trial
`1840009` 的 TF-eager compatibility negative 继续作为跨 backend 边界；Trial
`1840344` 仅是 locale code-ledger preflight 失败，不是算法结果。后续 Trial `1840837`
已完成同一 4095-token PG-19 train-only 文档、N<=32 请求同时驻留的 shared-pool 容量曲线：
fresh/reuse=`80+90N`/`80+5N MiB`，N32 省 `2720 MiB` pool，absolute peak allocated
中位数差约 `2.661 GiB`，且所有 N 的 token/logit/KV/GDN 与 cross-N isolation exact。
它仍是单流 round-major 顺序服务，不是 concurrent scheduler 或吞吐结果。下一 infra 阶段
应转向 continuous batching/ragged、多文档回收复用、NVML 与独立质量，而不是重复解析
resident-capacity 曲线。
Apple Neural Engine 适合后期固定图部署验证，不足以承担当前研究所需的通用
Transformer 训练和动态 split-residual runtime。
