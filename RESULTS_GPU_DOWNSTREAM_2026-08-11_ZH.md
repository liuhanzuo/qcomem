# Q-CoMem GPU 下游任务实验（2026-08-11）

## 当前状态

- 8-sample mechanism pilot A：完成；发现 Write 路径保留 autograd graph，内存数据无效。
- 8-sample corrected pilot B：完成；预测与 A 逐样本完全一致，内存与时间数据有效。
- 64-sample disjoint validation：完成；按结果揭晓前冻结的规则完成 paired 分析。
- 64-sample Q16 interface diagnostic：完成；exact oracle、document-local、chunk-local 已拆解。
- 8-sample lower replay diagnostic：完成；depth 7/10/13 均与 dense 逐 token 完全一致。
- exact prefix + replay Q8/Q4 diagnostic：完成；已得到完整持久字节—速度折中。
- lower KV/recurrent-state Q2/Q4/Q8 diagnostic：完成；Q8 近无损，Q4/Q2 给出失效边界。
- 4k/8k/16k extreme capacity scaling：完成；验证持久容量随长度近线性增长。
- 4k/8k/16k near-lossless Q8-state scaling：完成；depth 7 在 16k 仍接近 10x。
- attention/linear mixed-bit diagnostic：完成；attention Q4 / linear Q8 是 pilot Pareto 点。
- 64-sample packed-state replay validation：完成；d7 Q4/Q8 通过预注册 mean margins。
- 64-sample untouched packed-state test：完成；冻结后的 d7 Q4/Q8 通过 mean margins。
- depth-7 逐层位宽敏感度校准：完成；8×H20 生成 residual + 7 个 cache layer 的
  Q2/Q4/Q8/Q16 profile 和三档 mixed-bit policy。

远端任务：

| run | Job / Trial | 状态 | 用途 |
|---|---|---|---|
| pilot A | `232761 / 1825100` | Complete | 首次端到端 smoke，定位评测实现问题 |
| pilot B | `232787 / 1825176` | Complete | 修复 autograd 后的可用 pilot |
| validation | `232818 / 1825247` | Complete | 与 pilot 不重叠的 32 Qasper + 32 2WikiMQA |
| interface diagnostic | `232878 / 1825354` | Complete | 拆分 query-local 与 chunk-local interface loss |
| lower replay diagnostic | `232964 / 1825504` | Complete | 验证 lower KV/GatedDeltaNet state replay 的逐 token 正确性 |
| prefix/quant replay diagnostic | `232972 / 1825527` | Complete | exact prefix cache 与 replay Q8/Q4 的完整持久字节对照 |
| state quantization diagnostic | `232989 / 1825582` | Complete | lower KV、conv/recurrent state 的 Q2/Q4/Q8 率失真 |
| extreme capacity scaling | `233003 / 1825614` | Complete | 4k/8k/16k 的 Q4-state 持久容量与 one-token latency |
| Q8-state capacity scaling | `233034 / 1825665` | Complete | 近无损主配置在 4k/8k/16k 的容量曲线 |
| mixed state precision | `233042 / 1825699` | Complete | 拆分 attention Q4 与 linear Q4 的敏感性 |
| packed-state replay validation | `233077 / 1825759` | Complete | 在 64 条 disjoint validation 上验证 Q4/Q8 Pareto 点 |
| frozen packed-state test | `233107 / 1825835` | Complete | 在 untouched index 36–67 上只验证冻结后的 d7 Q4/Q8 |
| layer-wise bit calibration | `233909 / 1827870` | Complete | 逐层 first-token KL 校准与 mixed-bit policy 搜索 |

本地原始结果保存在
[`results/gpu-downstream-pilot-20260811b/`](results/gpu-downstream-pilot-20260811b/)
和
[`results/gpu-downstream-validation-20260811a/`](results/gpu-downstream-validation-20260811a/)，
接口诊断保存在
[`results/gpu-interface-diagnostic-20260811a/`](results/gpu-interface-diagnostic-20260811a/)。
lower replay 原始结果保存在
[`results/gpu-replay-diagnostic-20260811a/`](results/gpu-replay-diagnostic-20260811a/)。
prefix/quant replay 原始结果保存在
[`results/gpu-replay-quant-diagnostic-20260811a/`](results/gpu-replay-quant-diagnostic-20260811a/)。
state quantization 原始结果保存在
[`results/gpu-replay-state-diagnostic-20260811a/`](results/gpu-replay-state-diagnostic-20260811a/)。
extreme capacity scaling 保存在
[`results/gpu-capacity-scaling-20260811a/`](results/gpu-capacity-scaling-20260811a/)。
Q8-state capacity scaling 保存在
[`results/gpu-capacity-quality-scaling-20260811a/`](results/gpu-capacity-quality-scaling-20260811a/)。
mixed-bit 原始结果保存在
[`results/gpu-replay-mixed-diagnostic-20260811a/`](results/gpu-replay-mixed-diagnostic-20260811a/)。
packed-state validation 原始结果保存在
[`results/gpu-replay-validation-20260811a/`](results/gpu-replay-validation-20260811a/)。
冻结配置的独立 test 原始结果保存在
[`results/gpu-replay-test-20260811a/`](results/gpu-replay-test-20260811a/)。

## 1. 实验设置

| 项目 | 设置 |
|---|---|
| GPU | 8 × NVIDIA H20-3e，单卡 143,771 MiB |
| 模型 | Qwen3.5-35B-A3B，BF16，40 层，MoE 256 experts / 8 active |
| 软件 | PyTorch 2.11.0+cu129，Transformers 5.14.1 |
| 执行方式 | 8 个独立单卡进程；本轮 10 个配置 round-robin 分配 |
| split depth | 7、10、13 |
| residual bits | Q16、Q8、Q4 |
| chunk / overlap / group | 512 / 64 / 64 |
| pilot 数据 | Qasper 4 条 + 2WikiMQA 4 条，source index 0–3 |
| pilot 输入/输出上限 | 4096 / 32 tokens |
| decode | greedy；当前实现每个输出 token 重新执行所需层，没有 KV cache |

Q8/Q4 是 residual 的真实 affine group quantization 和 bit packing。实际容量包括 packed
integer、BF16 scale 和 BF16 bias，不使用名义 bit 数估算。

## 2. corrected pilot B 结果

该 pilot 只有 8 条样本，不能用于论文质量结论；它回答的是实现是否端到端工作，以及哪些
配置值得进入更大的 validation。

| 配置 | 总 F1 | Qasper F1 | 2Wiki F1 | store MiB | 压缩率 | relative RMSE | s/output-token | peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.4179 | 0.3358 | 0.5000 | — | — | — | 0.6409 | 65.410 |
| d7-Q16 | 0.3736 | 0.2472 | 0.5000 | 14.852 | 1.000x | 0 | 0.5606 | 65.423 |
| d7-Q8 | 0.3764 | 0.2527 | 0.5000 | 7.890 | 1.882x | 0.0099 | 0.5666 | 65.423 |
| d7-Q4 | 0.3749 | 0.2498 | 0.5000 | 4.177 | 3.556x | 0.1450 | 0.5615 | 65.423 |
| d10-Q16 | 0.3999 | 0.2999 | 0.5000 | 14.852 | 1.000x | 0 | 0.5456 | 65.427 |
| d10-Q8 | 0.3989 | 0.2978 | 0.5000 | 7.890 | 1.882x | 0.0098 | 0.5434 | 65.426 |
| d10-Q4 | 0.3799 | 0.2599 | 0.5000 | 4.177 | 3.556x | 0.1381 | 0.5406 | 65.426 |
| d13-Q16 | 0.3931 | 0.2861 | 0.5000 | 14.852 | 1.000x | 0 | 0.4904 | 65.427 |
| d13-Q8 | 0.3677 | 0.2353 | 0.5000 | 7.890 | 1.882x | 0.0098 | 0.4637 | 65.424 |
| d13-Q4 | 0.3650 | 0.2301 | 0.5000 | 4.177 | 3.556x | 0.1373 | 0.4491 | 65.426 |

初步观察：

1. Q8 的 residual relative RMSE 稳定在约 0.98%，Q4 为 13.7%–14.5%。
2. depth 10 的 Q8 相对同 depth Q16 仅下降 0.0011 F1；Q4 下降 0.0200。
3. depth 13 的 Q8/Q4 相对 Q16 分别下降 0.0254/0.0280，说明 bit 与 split depth 存在
   交互，支持学习 depth-aware bit policy，而不是全层固定一个 bit。
4. Q16 相对 dense 的 interface gap 在 depth 7/10/13 分别为
   `-0.0443/-0.0180/-0.0248` F1；不能把这部分损失归因于量化。
5. 2Wiki 只有 4 条且所有配置都恰好 0.5 F1，完全没有区分力；这正是需要独立 validation
   的原因。

## 3. autograd 评测 bug 与修复

pilot A 的文档 lower-layer Write 没有包在 `torch.inference_mode()` 中。虽然模型处于
`eval()`，PyTorch 仍会记录反向图；因此 A 的预测是正确的，但显存与 Write 时间无效。

| depth（Q16） | A peak | B peak | 减少 | Write 加速 |
|---:|---:|---:|---:|---:|
| 7 | 102.21 GiB | 65.42 GiB | 36.79 GiB | 3.02x |
| 10 | 116.05 GiB | 65.43 GiB | 50.62 GiB | 3.07x |
| 13 | 129.83 GiB | 65.43 GiB | 64.40 GiB | 3.75x |

A/B 的 10 个配置、80 条 prediction、输出长度和 F1 全部逐项一致。修复后模型权重常驻
64.56 GiB，单样本额外峰值约 0.85–0.87 GiB。

Q4/Q8 没有显著降低这次的 CUDA peak 是预期结果：当前只有一条 selected context，Read
前必须把选中的 packed residual 恢复成 BF16 才能进入 suffix layers。低 bit 的主要收益
是大 corpus 的持久化 RAM/SSD 容量与传输量，不是消除当前选中上下文的计算工作区。

## 4. validation 协议与预注册判定

validation 数据来自 `zai-org/LongBench` revision
`5e628be450b7e67fb7ae6e201bd6d8f7056f7672`，取 Qasper 和 2WikiMQA 各 source index
4–35，共 64 条，与 pilot index 0–3 不重叠。合并文件 SHA256：

```text
1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe
```

协议对齐 LongBench v1 官方配置：

- 使用官方 Qasper / 2WikiMQA prompt；
- Qasper `max_new_tokens=128`，2WikiMQA `max_new_tokens=32`；
- greedy decoding；
- 完整 prompt 超过 4096 token 时保留 instruction/question，并从 context 中间截断；
- 对每个样本报告原始/保留 context tokens 和是否截断。

验证结果揭晓前冻结以下规则：

1. interface gap：只比较各 depth 的 Q16 与 dense，单独报告，不参与 bit 量化归因。
2. quantization gap：Q8/Q4 只与相同 depth 的 Q16 做 paired 比较。
3. 非劣界限：总体 mean F1 delta 不低于 `-0.02`，且任一数据集的 mean delta 不低于
   `-0.03`；同时报告 paired bootstrap 95% CI，不在看到结果后修改界限。
4. 若多个 bit 满足约束，选择真实 `stored_residual_nbytes` 最小者。
5. prediction exact agreement、EOS 长度变化、relative RMSE 和逐样本 catastrophic regression
   必须报告，但不替代任务 F1 主判据。
6. 64 条 validation 用于选策略；最终仍需另取不重叠 test split 验证，不能把 validation
   同时当作最终 test。

独立 test 已在 validation 结果揭晓前冻结：同一 revision 的 Qasper 与 2WikiMQA 各取
source index 36–67，共 64 条，不与 pilot 0–3 或 validation 4–35 重叠。它不会再用于选择
depth/bit；文件位于远端 `data/qcomem-longbench-test/longbench_test.jsonl`，SHA256：

```text
f985ea4c841a450fc0967f706b5a0d2c4a9fe279f7c0f2b580cf123be8b6860b
```

官方配置来源：

- [LongBench dataset2prompt.json](https://github.com/THUDM/LongBench/blob/main/LongBench/config/dataset2prompt.json)
- [LongBench dataset2maxlen.json](https://github.com/THUDM/LongBench/blob/main/LongBench/config/dataset2maxlen.json)
- [LongBench prediction protocol](https://github.com/THUDM/LongBench/blob/main/LongBench/pred.py)

## 5. validation 结果

### 5.1 任务分数、容量和速度

| 配置 | 总 F1 | Qasper | 2Wiki | store MiB | 压缩率 | s/output-token |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.5420 | 0.4362 | 0.6478 | — | — | 0.7174 |
| d7-Q16 | 0.4522 | 0.4139 | 0.4905 | 14.923 | 1.000x | 0.6322 |
| d7-Q8 | 0.4496 | 0.4188 | 0.4804 | 7.928 | 1.882x | 0.6284 |
| d7-Q4 | 0.4502 | 0.4098 | 0.4905 | 4.197 | 3.556x | 0.6280 |
| d10-Q16 | 0.4167 | 0.3999 | 0.4336 | 14.923 | 1.000x | 0.6058 |
| d10-Q8 | 0.3937 | 0.3746 | 0.4127 | 7.928 | 1.882x | 0.6055 |
| d10-Q4 | 0.4093 | 0.4058 | 0.4127 | 4.197 | 3.556x | 0.6037 |
| d13-Q16 | 0.4010 | 0.3903 | 0.4116 | 14.923 | 1.000x | 0.5725 |
| d13-Q8 | 0.4012 | 0.3908 | 0.4116 | 7.928 | 1.882x | 0.5703 |
| d13-Q4 | 0.4022 | 0.3927 | 0.4116 | 4.197 | 3.556x | 0.5607 |

Q8 的 mean relative RMSE 为 0.98%–0.99%，Q4 为 13.70%–14.48%。所有配置的 CUDA
peak 仍约为 65.43 GiB，其中模型权重常驻 64.56 GiB，单样本额外峰值约 0.85–0.87
GiB。Q4/Q8 降低的是 persistent residual store，不是已经选中并解量化后的计算工作区。

### 5.2 interface gap：当前首要问题

| depth | Q16 − dense mean F1 | paired bootstrap 95% CI | Qasper delta | 2Wiki delta | prediction agreement |
|---:|---:|---:|---:|---:|---:|
| 7 | -0.0898 | [-0.1729, -0.0117] | -0.0223 | -0.1573 | 59.38% |
| 10 | -0.1253 | [-0.2081, -0.0513] | -0.0363 | -0.2142 | 46.88% |
| 13 | -0.1410 | [-0.2266, -0.0626] | -0.0459 | -0.2362 | 39.06% |

这是未量化 Q16 split 与 dense 的差距，不能归因于 Q4/Q8。2Wiki 的多文档问题尤其严重：
当前 lower layers 对每个 chunk 和 query 独立执行，丢失 query 对文档、chunk 对前序 chunk
的下层注意力；split 越深，丢失的交互层越多。

Qasper 25/32、2Wiki 28/32 的原始 context 超过 4096-token budget。不过 interface
对比的 dense 与 Q16 使用完全相同的截断输入，因此截断不解释两者差距。事实上，11 条
未截断样本上的 interface delta 也很大；这更支持 split 语义本身是主因。

### 5.3 quantization gap 与预注册 policy

| depth | Q8 − Q16 mean F1（95% CI） | Q8 | Q4 − Q16 mean F1（95% CI） | Q4 | 预注册选择 |
|---:|---:|---:|---:|---:|---:|
| 7 | -0.0026 [-0.0151, 0.0073] | pass | -0.0020 [-0.0051, 0] | pass | Q4 |
| 10 | -0.0231 [-0.0670, 0.0060] | fail | -0.0074 [-0.0313, 0.0089] | pass | Q4 |
| 13 | +0.0002 [0, 0.0007] | pass | +0.0012 [0, 0.0036] | pass | Q4 |

按预注册均值门槛，validation policy 是 `depth 7/10/13 -> Q4/Q4/Q4`。这轮没有选择出
mixed-bit policy，不能为了迎合“不同层不同 bit”的假设而改门槛。

depth 10 出现 Q8 fail、Q4 pass 的非单调现象。逐样本检查显示：

- 2Wiki source index 11：Q16 正确输出 `Marie of Hohenstaufen`，Q8/Q4 都输出
  `Adelaide of Burgundy`，F1 delta 为 -0.667；
- Qasper source index 25：Q16/Q4 输出 `Unanswerable`，Q8 输出另一句答案，F1 delta
  为 -1.0。

因此 Q8 的均值被两次离散生成翻转拉低，而 Q4 只遇到一次；这不是“4 bit 数值误差比
8 bit 更小”。它说明 64 条 validation 仍不足以稳定判定 depth 10，且自由生成的 argmax
对局部 logit 扰动非单调。Q4 policy 只能视为下一阶段候选，不能直接进入最终 test 或部署。

## 6. Q16 interface 误差分解

诊断使用同一批 64 条 validation 样本，并运行四类 control：

1. dense；
2. continuous-prefix oracle：document + query 一起执行 lower layers；
3. continuous-document、query-local split：文档不分 chunk，但 query lower 独立；
4. chunk-local split：当前 512-token chunk、64-token overlap 实现。

oracle-d10 与 dense 的 64/64 prediction、输出长度和 F1 全部相同，mean F1 delta 和
bootstrap CI 都严格为 0。这证明手写 layer boundary、position、rotary 和 hybrid
attention mask 是正确的；损失来自为了复用而改变 lower-layer 可见性。

| depth | document − dense | 95% CI | chunk − document | 95% CI | chunk − dense |
|---:|---:|---:|---:|---:|---:|
| 7 | -0.0319 | [-0.0907, 0.0227] | -0.0579 | [-0.1234, -0.0067] | -0.0898 |
| 10 | -0.0979 | [-0.1771, -0.0253] | -0.0273 | [-0.0952, 0.0360] | -0.1253 |
| 13 | -0.1449 | [-0.2305, -0.0642] | +0.0039 | [-0.0455, 0.0538] | -0.1410 |

解释：

- depth 7 的主要额外损失来自 chunk boundary；去掉 chunk 后，总 gap 从 -0.0898 缩小到
  -0.0319。Qasper 几乎恢复到 dense（-0.0003），2Wiki 仍下降 -0.0635。
- depth 10/13 的主要问题不是 chunk，而是 query lower layers 看不到文档。depth 越深，
  被切断的 query-document 交互层越多，document-local gap 从 -0.098 增至 -0.145。
- depth 13 的 chunk 与 continuous-document 均值差为 +0.0039、CI 跨 0；增大 chunk
  不能修复这个深度的主误差。
- continuous-document Write 还比 chunk Write 更快：depth 7/10/13 分别约
  `0.150/0.163/0.209 s`，而 chunk 为 `0.341/0.386/0.478 s`。当前 4k context 下，
  分块的重复 overlap 与 kernel launch 成本没有换来速度收益。

## 7. 工程与研究决策

扩大 bit sweep 不是当前最高优先级。lower-layer KV/state replay 已实现：文档 lower
layers 只 Write 一次，query 在对应层读取文档的 full-attention KV，以及 Qwen3.5
linear-attention layers 的 convolution/recurrent state。第 9 节给出目标 35B 模型的
逐 token 验证；下一步是 exact full-prefix cache 和量化 replay 的总持久字节对照。

depth-7 continuous-document 仍保留为不存 lower state 的 training-free ablation；对 4k
context 不再默认切 512-token chunk。exact replay 已消除未量化接口损失，因此当前证据
不支持先做 suffix LoRA/adapter 训练；训练只在低比特 replay 的独立 test 仍掉点时考虑。

资源方面，validation 的旧 round-robin 曾造成 rank 0/1 尾部。接口诊断改为每张卡运行
同一组 control、只处理 1/8 样本，绝大多数运行时间八张卡都在工作；最后仅因个别样本输出
更长产生约一分钟自然尾部。普通 10-config launcher 也已加入 cost-aware assignment 测试。

## 8. 仍不能声称的结果

- 当前时间不是带 KV cache 的生产 decode 吞吐；只能比较这份手写机制实现。
- validation 是策略选择集，不是独立 test；不能把选择与最终报告用在同一批样本上。
- 4096-token 截断不能代表模型的 262k 上下文能力；应额外按 0–4k、4–8k、8k+ 分桶。
- 当前 store 只覆盖每条样本的一份 context，不等于大 corpus 多 query 的 resident store。
- 当前 replay suffix 在每个输出 token 上仍重算完整 boundary sequence，尚未加入生产级
  suffix decode cache；当前时间不能当作 vLLM/MLX serving 吞吐。

## 9. Lower KV/recurrent-state replay：35B 目标模型结果

### 9.1 实现

`TorchSplitCausalLM.write_lower_replay()` 连续处理文档到 split depth，同时保存：

- split boundary 的 document residual；
- split 以下 full-attention layers 的 K/V；
- split 以下 GatedDeltaNet layers 的 causal-convolution 尾状态与 recurrent state。

每个 query 从不可变 document cache fork 一份可变 state。query prompt 第一次以多 token
continuation 方式通过 lower layers；后续生成 token 只增量更新 lower state。suffix 当前仍
读取完整 boundary residual，因此这是 correctness/mechanism 实现，不是最终生产 decode。

tiny Qwen3.5 混合层单测先要求 dense、continuous oracle 和 replay 连续 3 个 token 完全
一致；通过后才提交目标模型任务。目标任务使用 pilot 的 4 Qasper + 4 2WikiMQA、官方
prompt、4096 input token 和最多 32 output token；8 张 H20 各处理 1 个样本和全部配置。

### 9.2 结果

| 配置 | dense token exact | mean F1 | Write s | generation s | residual MiB | lower state MiB | 总持久 MiB | max peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 8/8 | 0.4097 | — | 11.399 | — | — | — | 65.410 |
| replay-d7 | 8/8 | 0.4097 | 0.158 | 7.646 | 14.886 | 19.818 | 34.704 | 65.471 |
| replay-d10 | 8/8 | 0.4097 | 0.162 | 6.087 | 14.886 | 31.386 | 46.272 | 65.503 |
| replay-d13 | 8/8 | 0.4097 | 0.220 | 5.592 | 14.886 | 42.954 | 57.841 | 65.612 |

三个 depth 的 prediction、生成长度、token IDs 和 F1 均与 dense 完全一致。相对这份不带
KV decode cache 的 dense 机制实现，generation wall time 分别缩短约 `1.49x/1.87x/2.04x`。
这证明此前 `-0.09~-0.14 F1` 的 interface gap 可以 training-free 消除，不需要先做 LoRA。

更重要的容量结论是：必须报告 `residual + lower state`，不能只报告 14.886 MiB residual。
随着 split 加深，residual 大小不变，但 lower state 从 19.818 MiB 增到 42.954 MiB。
Q4/Q8 只压 residual 时，总容量压缩会明显小于 residual 自身的 3.56x/1.88x；因此下一轮
必须同时比较 exact full-model prefix cache，并把 lower KV/recurrent state 纳入 mixed-bit
校准对象。

该结果仍只是 8-sample mechanism diagnostic。逐 token exactness 是强接口证据，但不能
代替更大下游 test；量化 replay 的自然任务结论需等待独立 validation/test。

### 9.3 Exact prefix 与 Q8/Q4 完整持久状态

第二个 8-sample diagnostic 在相同数据和协议上加入 exact full-model prefix cache，并只
量化 split residual；lower KV/recurrent state 暂时保持 BF16。所有数字都包含 scale/bias
元数据，`总持久 = packed residual + lower state`。

| 配置 | dense token exact | mean F1 | Write s | generation s | residual MiB | state MiB | 总持久 MiB | prefix / 总持久 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 8/8 | 0.4097 | — | 9.451 | — | — | — | — |
| exact prefix | 8/8 | 0.4097 | 0.649 | 1.743 | — | 136.306 | 136.306 | 1.00x |
| replay-d7-Q8 | 8/8 | 0.4097 | 0.221 | 6.605 | 7.908 | 19.818 | 27.726 | 4.92x |
| replay-d7-Q4 | 7/8 | 0.4097 | 0.146 | 6.604 | 4.187 | 19.818 | 24.005 | 5.68x |
| replay-d10-Q8 | 8/8 | 0.4097 | 0.167 | 6.111 | 7.908 | 31.386 | 39.295 | 3.47x |
| replay-d10-Q4 | 8/8 | 0.4097 | 0.164 | 6.103 | 4.187 | 31.386 | 35.573 | 3.83x |
| replay-d13-Q8 | 8/8 | 0.4097 | 0.205 | 5.605 | 7.908 | 42.954 | 50.863 | 2.68x |
| replay-d13-Q4 | 8/8 | 0.4097 | 0.222 | 5.590 | 4.187 | 42.954 | 47.141 | 2.89x |

Q8 residual relative RMSE 为 `0.977%–0.981%`，三个 depth 的 token IDs 都与 dense 完全
一致。Q4 为 `13.92%–14.62%`；depth 10/13 仍逐 token 一致，depth 7 的 Qasper source
index 3 出现一次等义改写：dense 输出 “Text sequences of context tweets ...”，Q4 输出
“Context tweets (text sequences ...) ...”，两者任务 F1 都是 0.5。因此不能把 7/8 exact
写成量化无损，但这轮 8 条样本上 Q4 没有任务分数损失。

exact prefix 与 replay 构成明确的 Pareto trade-off：prefix 的 generation 最快，但 depth-7
Q8/Q4 的持久状态比 prefix 小 `4.92x/5.68x`；replay 写入也约快 3–4x。随着 depth 变深，
replay generation 变快，但 lower state 增长使容量优势缩小。这正是 split depth 用计算换内存的
核心曲线。

只量化 residual 的总容量收益并不大：相对 Q16 replay，Q4 的总状态只缩小约
`1.45x/1.30x/1.23x`（depth 7/10/13），因为 lower cache 已成为主体。下一阶段不应继续
盲目增加 residual bit sweep，而应校准 lower full-attention KV 与 recurrent state 的精度。
当前 packed store 在 query fork 时会反量化 residual、复制可变 cache，所以 CUDA active
peak 仍约 65.8–66.2 GiB；这是 reference 实现开销，不能用 persistent bytes 推断 active peak。

### 9.4 Lower state 量化：从“几倍”到“一个数量级”

第三个 8-sample diagnostic 将 residual、full-attention KV 和 linear-attention
convolution/recurrent state 的位宽独立控制。命名 `r4-a8-l8` 分别表示 residual Q4、
attention cache Q8、linear state Q8。三类数据都是真实 affine group packing，group size
64，容量包含 FP16 scale/bias。

| 配置 | mean F1 | dense token exact | 总持久 MiB | prefix / 总持久 | residual RMSE | attn RMSE | linear RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact prefix | 0.4097 | 8/8 | 136.306 | 1.00x | — | — | — |
| d7-r4-a16-l16 | 0.4097 | 7/8 | 24.005 | 5.68x | 14.62% | 0 | 0 |
| d7-r4-a8-l8 | 0.4056 | 6/8 | 11.528 | **11.82x** | 14.62% | 0.59% | 0.62% |
| d10-r4-a8-l8 | 0.4097 | 7/8 | 16.611 | **8.21x** | 13.95% | 0.60% | 0.61% |
| d13-r4-a8-l8 | 0.4097 | 8/8 | 21.694 | **6.28x** | 13.92% | 0.61% | 0.60% |
| d7-r4-a4-l4 | 0.3110 | 2/8 | 8.073 | 16.88x | 14.62% | 9.98% | 9.39% |
| d10-r4-a4-l4 | 0.3135 | 3/8 | 10.764 | 12.66x | 13.95% | 10.15% | 9.41% |
| d13-r4-a4-l4 | 0.3151 | 2/8 | 13.455 | 10.13x | 13.92% | 10.32% | 9.45% |
| d7-r2-a2-l2 | 0.1773 | 1/8 | 4.485 | **30.39x** | 64.35% | 50.24% | 48.74% |

最强且目前可信的候选是 `d7-r4-a8-l8`：相对 exact prefix 缩小 11.82x，mean F1 只下降
0.0042。depth 10/13 的 Q8 state 在这 8 条上 mean F1 与 dense 相同，但容量优势依次缩小。
这把主卖点从“residual 几倍压缩”推进到“完整可复用状态约一个数量级压缩、近乎无损”。

Q4 state 是清楚的失效边界：虽然可达 10–17x，mean F1 下降约 0.095。Q2 全压缩达到
30.39x，但 F1 下降 0.232，不能作为“极限压缩仍几乎无损”的证据。`d7-r2-a4-l4`
与 `d7-r4-a4-l4` 分数相同，说明在这个失效区间，cache Q4 误差已经主导，继续把 residual
降到 Q2 只省容量、不再解释质量变化。

因此下一步不是宣传 30x，而是：

1. 以 `Q4 residual + Q8 state` 作为主系统点；
2. 分别测试 `attention Q4 / linear Q8` 与 `attention Q8 / linear Q4`，定位敏感对象；
3. 在独立 validation/test 上预注册近无损阈值；
4. 把 30x Q2 作为 rate-distortion 曲线的失败端点。

### 9.4.1 Mixed precision：linear state 是当前敏感对象

第四个 8-sample diagnostic 固定 residual Q4，只把 full-attention cache 与 linear
recurrent state 中的一类从 Q8 降到 Q4：

| depth | attention / linear | mean F1 | dense delta | 总持久 MiB | prefix / 总持久 |
|---:|---:|---:|---:|---:|---:|
| 7 | Q8 / Q8 | 0.4056 | -0.0042 | 11.528 | 11.82x |
| 7 | **Q4 / Q8** | **0.4056** | **-0.0042** | **9.667** | **14.10x** |
| 7 | Q8 / Q4 | 0.3666 | -0.0431 | 9.934 | 13.72x |
| 10 | Q8 / Q8 | 0.4097 | 0 | 16.611 | 8.21x |
| 10 | **Q4 / Q8** | **0.4097** | **0** | **12.889** | **10.58x** |
| 10 | Q8 / Q4 | 0.2611 | -0.1486 | 14.486 | 9.41x |
| 13 | Q8 / Q8 | 0.4097 | 0 | 21.694 | 6.28x |
| 13 | **Q4 / Q8** | **0.4097** | **0** | **16.111** | **8.46x** |
| 13 | Q8 / Q4 | 0.2798 | -0.1299 | 19.037 | 7.16x |

attention Q4 的相对 RMSE 约 10%，但三种 depth 的 mean F1 都与各自 Q8/Q8 相同；linear
Q4 的相对 RMSE 约 9.4%，却在三种 depth 上都明显掉点。误差范数相近而任务敏感度不同，
说明位宽不能只按 tensor reconstruction error 分配，必须按 state 类型和下游扰动校准。

pilot 最强候选因此更新为 `d7-r4-a4-l8`：14.10x 状态压缩、mean F1 delta -0.0042。

64-sample validation 已进一步完成：

| 配置 | mean F1 | dense delta | paired bootstrap 95% CI | Qasper delta | 2Wiki delta | MiB | prefix / state | mean-margin gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.5420 | 0 | [0, 0] | 0 | 0 | — | — | — |
| exact prefix | 0.5473 | +0.0053 | [-0.0239, +0.0452] | -0.0081 | +0.0188 | 136.493 | 1.00x | pass |
| d7 Q8/Q8 | 0.5394 | -0.0025 | [-0.0405, +0.0419] | -0.0088 | +0.0037 | 11.548 | 11.82x | pass |
| **d7 Q4/Q8** | **0.5431** | **+0.0012** | **[-0.0297, +0.0430]** | **-0.0112** | **+0.0135** | **9.683** | **14.10x** | **pass** |
| d10 Q4/Q8 | 0.5414 | -0.0006 | [-0.0307, +0.0422] | -0.0199 | +0.0188 | 12.910 | 10.57x | pass |
| d13 Q4/Q8 | 0.5207 | -0.0213 | [-0.0792, +0.0324] | +0.0012 | -0.0438 | 16.138 | 8.46x | **fail** |

这里的 `pass` 严格指预注册的 overall/per-dataset **mean** margins，不是说置信区间下界也
通过了非劣界限；64 条的 paired CI 仍较宽。d13 在 8 条 pilot 上与 dense 相同，却在
validation 失败，正好说明不能用小 pilot 直接形成结论。

d7 Q4/Q8 是通过 gate 的最小状态，随后冻结。独立 test 任务 `233107 / 1825835` 只运行
dense、exact prefix 和这一种候选，没有再根据 test 调 depth 或 bit。

独立 test 结果如下：

| 配置 | mean F1 | dense delta | paired bootstrap 95% CI | Qasper delta | 2Wiki delta | MiB | prefix / state | mean-margin gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.5424 | 0 | [0, 0] | 0 | 0 | — | — | — |
| exact prefix | 0.5389 | -0.0035 | [-0.0097, +0.0008] | -0.0070 | 0 | 135.260 | 1.00x | pass |
| **frozen d7 Q4/Q8** | **0.5337** | **-0.0087** | **[-0.0237, +0.0012]** | **-0.0175** | **0** | **9.579** | **14.12x** | **pass** |

冻结配置在 untouched test 上保持约 14x 压缩，overall 与每数据集 mean delta 均通过门槛，
且没有 `F1 delta <= -0.5` 的 catastrophic regression。CI 下界略低于 `-0.02`，所以正确
表述是“独立 test 上观察到近无损并通过预注册 mean margins”，而不是“统计上证明严格
无损”。2Wiki 本轮三种路径的 mean F1 恰好一致，主要区分信号来自 Qasper。

### 9.4.2 逐层位宽敏感度校准

本轮在 `Reasoning_Rollout` 队列使用 8×H20-141G；Job / Trial 为
`233909 / 1827870`，总执行时间 6 分 48 秒。depth 固定为 7，一张卡负责 residual，另外
七张卡分别负责一个 lower cache layer。每个组件单独切换 Q2/Q4/Q8/Q16，其余组件保持
Q16；目标是 4 条 calibration prompt（Qasper、2WikiMQA 各 2 条，最大 2048 tokens）上的
mean first-token KL，group size 为 64。

正式校准前的 exactness gate 已通过：cached dense、full prefix、lower replay、固定顺序
多文档 replay 和逐层 Q16 replay 的三个生成 token 均与 oracle 完全一致。

| policy | residual bits | cache layer bits | persistent MiB | 相对全 Q16 | 预测 KL |
|---|---:|---|---:|---:|---:|
| 全 Q16 | 16 | 16/16/16/16/16/16/16 | 23.921 | 1.00x | 约 0 |
| frozen static | 4 | 8/8/8/4/8/8/8 | 6.634 | 3.61x | 0.06449 |
| same-memory mixed | 4 | 8/8/4/4/8/8/8 | 6.368 | 3.76x | 0.05682 |
| minus 25% mixed | 4 | 8/8/2/2/2/8/2 | 4.958 | 4.83x | 0.12735 |
| extreme Q2 floor | 2 | 2/2/2/2/2/2/2 | 2.800 | 8.54x | 0.70698 |

在相同 frozen budget 下，搜索策略少用约 4.0% bytes，组件 KL 加和低约 11.9%。但这不能
直接写成下游质量提升：profile 只有 4 条 prompt，而且组件是逐个扰动后再假设 KL 可加。
例如 cache layer 2 的 Q4 测得 KL 反而低于 Q8，明显可能受小样本和 argmax 非单调影响。
因此这轮回答的是“逐层分配器已经能工作并给出候选”，不是“mixed-bit 已在下游任务上被
证明优于固定 bit”。下一步必须冻结上述三档 policy，在更大 validation 上进行联合量化的
端到端生成，再把胜出策略送入 untouched test。

当前 Transformers 环境缺少 FLA / causal-conv1d fast path，运行回退到 PyTorch 实现；
这不影响本轮数值校准，但本次 wall time 不能作为正式 serving 性能数据。

### 9.5 4k/8k/16k 容量曲线：持久容量与 active peak 必须分开

capacity scaling 使用固定自然语言段落构造精确的 4096/8192/16384-token 文档；4k、8k
各有 3 个独立单卡重复，16k 有 2 个。每个配置只生成 1 token，因此用于测量状态容量、
Write 和 warm one-token latency，不用于质量判断。第一轮使用全 Q4 state，目的是测极限
容量；其质量失效已由第 9.4 节单独证明。

| tokens | exact prefix MiB | d7-Q16 MiB | d7-all-Q4 MiB | prefix / d7-Q4 | d10-Q4 MiB | d13-Q4 MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 141.875 | 36.375 | 8.543 | 16.61x | 11.391 | 14.238 |
| 8,192 | 221.875 | 60.375 | 15.293 | 14.51x | 20.391 | 25.488 |
| 16,384 | 381.875 | 108.375 | 28.793 | 13.26x | 38.391 | 47.988 |

state 随 token 数近线性增长，但存在固定 recurrent/metadata 开销，所以短文档压缩比更高。
16k 时，全 Q4 depth-7 仍把 persistent state 从 381.9 MiB 降到 28.8 MiB。不过这个数字
只能作为 capacity upper bound，因为全 Q4 的下游 F1 已失败；近无损 Q8-state 曲线另行运行。

系统层面有两个重要反例：

- exact prefix 在 8k/16k 的 warm one-token generation 约 `0.214/0.262 s`，depth-7 replay
  为 `1.073/2.384 s`；持久容量优势不是免费加速。
- 16k 的 incremental CUDA peak，prefix 为 3.575 GiB，depth-7 Q4 replay 仍为 3.410 GiB。
  当前 fork/dequant reference implementation 主要降低 persistent state，并未按 13x 比例
  降低 active compute workspace。

replay 的 Write 明显更快：depth-7 Q4 在 4k/8k/16k 为 `0.184/0.272/0.586 s`，prefix 为
`0.939/1.301/2.777 s`，约快 4.7–5.1x。因而其系统价值更接近“低成本建立大量可复用文档
状态，并接受每次 query 较多 suffix 重计算”，而不是替代追求最低 latency 的 exact prefix。

近无损主配置 `Q4 residual + Q8 state` 的独立 scaling 结果为：

| tokens | exact prefix MiB | d7-Q4/Q8 MiB | 压缩 | d10-Q4/Q8 MiB | 压缩 | d13-Q4/Q8 MiB | 压缩 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 141.875 | 12.137 | **11.69x** | 17.516 | 8.10x | 22.895 | 6.20x |
| 8,192 | 221.875 | 20.887 | **10.62x** | 30.516 | 7.27x | 40.145 | 5.53x |
| 16,384 | 381.875 | 38.387 | **9.95x** | 56.516 | 6.76x | 74.645 | 5.12x |

因此“约一个数量级近无损状态压缩”不只在 4k pilot 成立；depth 7 到 16k 仍保持 9.95x。
压缩率随长度略降，是因为 exact prefix 与 replay 的固定 state/metadata 比例不同。

换算成 hot-document 容量，在不计权重和 active workspace 的 4 GiB persistent-state 预算
内，16k exact prefix 最多容纳 10 个文档，d7 Q4-residual/Q8-state 可容纳 106 个。这是
“更大 model-plus-context working set”的直接证据；它不是模型权重量化，也不能单独让
原本装不下的权重装入设备。

本轮 latency 不作为正式 scaling 结论：每个独立进程按固定配置顺序只计时一次，没有为
full-prefix cached decode、packed-cache restore 等路径分别 warm-up，结果中 prefix one-token
latency 随长度出现非物理波动。persistent tensor bytes 是确定性结果；正式 TTFT/TPOT 需要
随机化配置顺序、逐配置预热并用多次重复重新采集。

## 10. Related work 更新与论文定位

2026 年 7 月的 [HYPIC](https://arxiv.org/abs/2607.01299) 是当前最直接的竞品。它针对
Qwen3.5 hybrid stack，为每个 segment 保存 linear-attention 的 zero-start end-state 与
segment-cumulative transition operator，并为 full-attention 保存 segment-local KV、通过
seam window 修复边界。它已经占据“hybrid 全层 position-independent state composition”
这条路线，因此本项目不能再声称首次 hybrid state reuse。

本项目可守住的定位是：**只持久化一个压缩 split residual 与必要 lower state，用 suffix
重计算换取比全模型 prefix/PIC 更低的端侧持久内存。** 最重要的实验图应是
`persistent bytes/document—warm TTFT—downstream quality` Pareto frontier。

必须正面对照：

1. exact full-model prefix cache（GPU Transformers 与 Apple MLX-LM）；
2. HYPIC/HYPIC-lite；
3. [CacheBlend](https://arxiv.org/abs/2405.16444) 的多文档 KV reuse；
4. [KVTuner](https://arxiv.org/abs/2502.04420) 与 KIVI 的 mixed/uniform low-bit KV；
5. Apple 上 MLX-LM prompt cache。

HYPIC 当前论文没有公开完整实现，第一阶段可做 transition/seam 的 HYPIC-lite 消融，完整
系统作为 reported reference。KVTuner 已经覆盖 layer-wise mixed precision，因此我们的
新意不能建立在“不同层用不同 bit”本身，而应落在量化对象、split-depth 计算换内存、
hybrid exactness 和 Apple 端侧 Pareto 上。
