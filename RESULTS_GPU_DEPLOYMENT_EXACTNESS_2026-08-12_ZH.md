# Q-CoMem 部署 Q16 Exactness Gate（2026-08-12）

## 结果

- QS trial：`1830101`；
- 状态：`Complete`；
- 模型：Qwen3.5-35B-A3B；
- 硬件：8×H20-141GB；
- 数据：LongBench validation source index 6--9，Qasper/2WikiMQA 各 4 条；
- 结果：8/8 rank 的 Q16 replay 生成 token 与 dense oracle 完全一致；
- test-v2：未读取。

本 gate 会分别比较 dense recompute、标准 full-prefix Q16 和 split-depth=7 的 Q-CoMem Q16 replay。它要求完整生成 token 序列相同；logit 差异另外记录，不用放宽的浮点容差替代 token hard gate。

| rank | workload | Q16 replay token exact | full-prefix 最大 logit 差 | replay 最大 logit 差 |
|---:|---|---|---:|---:|
| 0 | qasper-6 | 是 | 3.0085 | 3.0085 |
| 1 | qasper-7 | 是 | 4.2920 | 4.5801 |
| 2 | qasper-8 | 是 | 18.7812 | 18.7812 |
| 3 | qasper-9 | 是 | 11.7578 | 11.7578 |
| 4 | 2wikimqa-6 | 是 | 2.1875 | 2.1875 |
| 5 | 2wikimqa-7 | 是 | 1.7734 | 1.7734 |
| 6 | 2wikimqa-8 | 是 | 1.3242 | 1.3242 |
| 7 | 2wikimqa-9 | 是 | 1.1719 | 1.1719 |

## 修复内容

旧实现把 `document_residual + query_residual` 合并成一个 suffix chunk。标准 full-prefix 则先建立 document cache，再用 query chunk 延伸。Qwen3.5 的 GatedDeltaNet/卷积 recurrent state 在 BF16 下对 chunk boundary 数值敏感，旧实现曾在 `2wikimqa-8` 的第二个 decode token 分叉。

修复后，lower 与 suffix 两侧都严格按以下顺序运行：

1. document chunk；
2. query chunk；
3. 单 token decode。

修复没有降低 gate。上表显示 replay 与 full-prefix 的 logit 差异几乎逐样本相同；剩余 cached-vs-full 数值差主要来自标准缓存执行路径，不是 split replay 额外引入的系统偏差。

## 结论边界

这证明修复后的 Q16 部署路径在本 gate 的 8 个样本上 token-exact，可继续运行显存、TTFT 和 TPOT 基准。它不等价于对所有输入严格 bitwise logit exact，也不能替代完整 60 样本下游 validation。
