# Qwen3.5 页式 attention 精确性门禁：真实负结果

## 结论

真实 Qwen3.5-35B-A3B 的当前 Python 页式 attention 参考实现没有通过原始
`rtol=0.02, atol=0.05` 的 final-logit 同 caller 门禁，因此没有进入正式 8-workload
benchmark，也不能宣称端到端部署收益。门禁按设计 fail-closed，没有放宽阈值。

- 有效诊断 Trial：Job `235797` / Trial `1834110`，终态 `Failed`；
- QS 页面：<https://qs2.devops.xiaohongshu.com/model/production/job/trial/235797/1834110>；
- 冻结 validation SHA256：
  `1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe`；
- 只读取 source index 6--9；未读取 source >=68 / test-v2；
- full-attention 层由配置动态得到：`3,7,11,15,19,23,27,31,35,39`；
- legacy 与 fixed 均完整拦截 10/10 层，dense fallback 均为 0。

## 诊断结果

对同一冻结 document/query，依次执行 Transformers eager、Trial `1833998` 的 legacy
FP32 页式算术、以及 two-pass BF16-weight 修正版，并保存每个 full-attention 模块的输出。

| 层 | legacy max abs | legacy rel-L2 | fixed max abs | fixed rel-L2 | fixed close |
|---:|---:|---:|---:|---:|:---:|
| 3  | 0.002441 | 0.003219 | 0.000122 | 0.000185 | 是 |
| 7  | 0.008789 | 0.026608 | 0.006836 | 0.030777 | 是 |
| 11 | 0.024414 | 0.032184 | 0.024902 | 0.032787 | 是 |
| 15 | 0.023438 | 0.049987 | 0.046875 | 0.043277 | 是 |
| 19 | 0.036987 | 0.080116 | 0.039062 | 0.067631 | 是 |
| 23 | 0.039062 | 0.079144 | 0.109375 | 0.069186 | **否** |
| 27 | 0.160156 | 0.124449 | 0.253906 | 0.101731 | 否 |
| 31 | 0.281250 | 0.113231 | 0.234375 | 0.093827 | 否 |
| 35 | 0.380859 | 0.174384 | 0.320312 | 0.150842 | 否 |
| 39 | 0.549805 | 0.110180 | 0.492188 | 0.086085 | 否 |

final logits：

| 路径 | max abs | mean abs | rel-L2 | logits close | greedy token exact |
|---|---:|---:|---:|:---:|:---:|
| legacy FP32 | 0.8125 | 0.118824 | 0.045573 | 否 | 是 |
| two-pass BF16 weight | 1.4375 | 0.200369 | 0.076619 | 否 | 是 |

修正版在第 3 层显著缩小局部误差，并从第 15 层起大多降低 relative-L2；但它在第 23 层
首次不满足逐层 close，误差随后经深层网络累积，final logits 反而比 legacy 更差。仅凭 greedy
token 相同不能把精确性门禁判为通过。

## 原因审计

修复前实现与 Hugging Face eager 的 dtype 路径确实不同：eager 使用 FP32 softmax 后把权重
cast 回 BF16，再做 BF16 weight/value matmul；legacy 则用未归一 FP32 page weights 与 FP32
value 累计。two-pass 修复了这一点，也避免了重复 causal mask。

但独立 page matmul 的归约形状和舍入顺序仍不同于 eager 的单次 dense concatenated GEMM。
即使每页都使用 BF16 operands / FP32 accumulator，把多个 page partial outputs 相加也不能保证
逐位复现一个完整 K 维 GEMM；这些微小差异经过 10 个 full-attention 层以及中间的 GDN/MoE
层传播后，会形成上表中的深层误差。当前数据支持这是首要剩余原因，但要证明到具体 kernel
指令级别仍需实现级 profiler 或 fused-kernel A/B。

## 下一步（未执行）

1. 实现 Triton/CUDA fused paged attention：一个 kernel 内完成跨 page 的 online softmax 与
   value reduction，明确固定 BF16/FP32 accumulator 语义，再跑同一 10 层门禁。
2. 或直接接入已验证的 production PagedAttention / FlashInfer 类 kernel；Python reference 仅保留
   cache ownership、page table、量化与 memory oracle，不再承担性能/数值声明。
3. 如果研究目标改为“任务质量几乎无损”而不是 final-logit 等价，应另立预注册门槛：LongBench
   F1/EM、perplexity、greedy token agreement 与多种 seed/workload 的置信区间。不能把这次失败的
   logit 门禁事后改成较松阈值。

## Artifact

- `results/gpu-qwen35-paged-real-negative-20260813d/paged-dtype-diagnostic.json`
- `results/gpu-qwen35-paged-real-negative-20260813d/rank0-dtype-diagnostic.log`
- `results/gpu-qwen35-paged-real-negative-20260813d/code.sha256`

diagnostic JSON SHA256：
`0f6c3033487de2d3edfe218d0357ba08bfeff22e5f6fc8b71636125ba3e761eb`。

## 保留的失败运行

- Trial `1833998`：旧 FP32 页式路径 8/8 rank 失败，final max-abs 为 0.25--1.375；
- Trial `1834067`：1-GPU 混部资源不支持挂载 TidalFS，无 Pod/无代码执行；
- Trial `1834082`：诊断输入错误地把 1D LongBench tokens 按 2D 索引，未产生算法诊断；
- Trial `1834110`：修复输入后产生本报告的真实算法负结果，并在 benchmark 前按门禁退出。
