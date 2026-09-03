# Dense SFT × 完整 lower-state Q-CoMem 下游结果

## 一句话结论

这次 dense full-model SFT **没有证明下游能力恢复**：step 128 虽然把内部
train-split heldout token CE 降低了 25.66%，但在冻结的 LongBench validation
60 条上，dense F1 的点估计反而从 `0.54160` 降到 `0.51852`，差值
`-0.02308`，paired bootstrap 95% CI 为 `[-0.11052, +0.06712]`。区间跨 0，
所以不能声称显著退化，但更不能声称 SFT 带来了下游提升。

另一方面，压缩本身仍然有效：在同一个 SFT 模型内，完整 Q8 lower-state
相对 dense 只差 `-0.00697` F1，且通过预注册 mean margin；frozen mixed-bit
相对 Q16 只差 `-0.00139`，同时把 persistent document state 再缩小
`3.59×`。因此当前瓶颈主要是 **dense→split/replay interface**，而不是
Q16→mixed-bit 的量化步骤。

## 正式任务与协议

- 正式成功任务：Job `235134` / Trial `1832184`，8×H20-141G，queue 408。
- QS 总执行时间：约 12 分 33 秒（包含容器前后开销）；实验 stage 生命周期为
  11 分 19 秒，即 `2026-08-12T13:34:16Z–13:45:35Z`。
- checkpoint：正式 dense SFT 的 step 128，manifest SHA256
  `cd22fbca3adfd668d24032726ff60425d432aa49f6105fe1cc9a5a0ef616c647`。
- 数据父文件 SHA256：
  `1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe`。
- 父文件只含 Qasper/2WikiMQA source `4–35` 共 64 条；本实验严格排除校准
  source `4–5`，评估 source `6–35` 共 60 条。
- 同一 8 卡作业内，先对 base 模型跑四个配置，再 collective load SFT DCP，
  随后在同样 60 条上跑相同四个配置，避免跨机器状态成为主要混杂因素。
- 四配置为 dense、完整 lower-state Q16、完整 lower-state Q8，以及 frozen
  mixed-bit：residual Q4 / attention state Q4 / linear state Q8 / lower-layer
  policy `[8,8,8,4,8,8,8]`。
- Q4/Q8 指 persistent document residual 与 lower-layer KV/recurrent/conv state，
  不指模型权重量化；模型推理权重仍为 BF16。
- LongBench raw test-v2 未读取。

最初 Trial `1832171` 在模型加载前被数据门禁安全拦截：门禁误把 64 条父文件
当成应物理只含 60 条的文件。修复后的协议先绑定父文件 SHA，再排除 4–5；
旧失败目录保留，不与正式结果混用。

## F1 主结果

| 模型阶段 | 路径 | Mean F1 | 相对同阶段 dense | 相对同路径 base | Persistent state |
|---|---|---:|---:|---:|---:|
| Base | Dense | 0.54160 | 0 | — | — |
| Base | Full-state Q16 | 0.54624 | +0.00464 | — | 34.68 MiB |
| Base | Full-state Q8 | 0.54620 | +0.00459 | — | 15.24 MiB |
| Base | Frozen mixed-bit | 0.54237 | +0.00077 | — | 9.66 MiB |
| SFT step128 | Dense | 0.51852 | 0 | -0.02308 | — |
| SFT step128 | Full-state Q16 | 0.49552 | -0.02300 | -0.05072 | 34.68 MiB |
| SFT step128 | Full-state Q8 | 0.51155 | -0.00697 | -0.03465 | 15.24 MiB |
| SFT step128 | Frozen mixed-bit | 0.49413 | -0.02438 | -0.04823 | 9.66 MiB |

### SFT 是否改善下游

四条 SFT-vs-base paired comparison 的点估计全部为负，且均未通过预注册
mean margin；但 60 条样本下的 95% CI 都跨 0：

| 同路径 SFT - Base | F1 delta | Paired bootstrap 95% CI |
|---|---:|---:|
| Dense | -0.02308 | [-0.11052, +0.06712] |
| Full-state Q16 | -0.05072 | [-0.12743, +0.02615] |
| Full-state Q8 | -0.03465 | [-0.11564, +0.04617] |
| Frozen mixed-bit | -0.04823 | [-0.12650, +0.02808] |

Dense 路径按数据集看，2WikiMQA 为 `+0.00238`，Qasper 为 `-0.04855`；
因此负向点估计主要来自 Qasper。60 条上 dense 有 12 条改善、13 条退化、
35 条 F1 不变。结论应写成“没有观察到 SFT 下游增益”，而不是“已证明 SFT
必然降低能力”。

内部 heldout CE 与下游 F1 的方向不一致，说明当前 checkpoint-selection
指标不充分。可能原因包括：1024 条/3 epochs 的小规模训练、answer+EOS CE
与自由生成 F1 的目标差异、训练最长 1024 token 而验证最长 4096 token，
以及 dense 训练从未见过 split/replay/quantized-state 扰动。这些是待验证解释，
不是本实验已经证明的因果机制。

## Q8 与 frozen mixed-bit 的真实含义

### Q8 是当前 SFT 模型上的最佳 Pareto 点

- SFT Q8 vs SFT dense：`-0.00697`，95% CI
  `[-0.01886, +0.00276]`，通过预注册 overall/per-dataset mean margin。
- persistent state 为 Q16 的 `1/2.276`。
- SFT Q8 vs SFT Q16 的点估计为 `+0.01603`，但 CI
  `[-0.00290, +0.05094]`；提升主要受一个 2WikiMQA 样本驱动，因此不能声称
  “Q8 系统性优于 Q16”，只能说没有观察到 Q8 的额外质量损失。
- 逐样本为 3 条改善、55 条 F1 相同、2 条退化；这一稀疏翻转分布进一步说明
  `+0.01603` 不应被解释成稳定增益。
- Base Q8 vs Base Q16 更接近零：`-0.00004`，CI
  `[-0.01579, +0.01352]`。

“最佳 Pareto 点”只指本次冻结评测的点估计：旧/新 base dense 重复之间仍有 1/60 条
greedy 输出发生分支变化，因此 Q8、Q16、dense 的小差值不是 bitwise 稳定排序。投稿若要把它们
排成严格次序，需要同协议重复运行或更强的 deterministic gate。

### Frozen 的量化增量近似无损，但完整路径不再无损

- SFT frozen vs SFT Q16：`-0.00139`，CI
  `[-0.00653, +0.00320]`，prediction exact agreement `90%`，1 条改善、56 条 F1
  相同、3 条退化且无灾难性退化；
  persistent state 比 Q16 小 `3.59×`。
- 但 SFT frozen vs SFT dense：`-0.02438`，CI
  `[-0.06673, -0.00077]`，没有通过预注册 mean margin。
- 原因是 SFT Q16 interface 本身相对 SFT dense 已有 `-0.02300` 的 gap；从
  Q16 再压到 frozen 只额外损失 `-0.00139`。

因此可以声称：**Q16→frozen mixed-bit 的压缩步骤近似无损**。不能声称：
**SFT dense→frozen 的完整部署路径仍然近似无损**。

对未 SFT 的 base 模型，frozen vs dense 仍为 `+0.00077`，CI
`[-0.03151, +0.04460]`，并通过 mean margin；这与此前的结论一致。

## Persistent memory 与 DCP 加载

同一批平均约 4k-token 文档的 persistent state：

| Store | Mean bytes | Mean MiB | 相对 Q16 | 相对历史 full-prefix cache |
|---|---:|---:|---:|---:|
| Full-state Q16 | 36,367,872 | 34.68 | 1.00× | 3.93× smaller |
| Full-state Q8 | 15,978,096 | 15.24 | 2.28× smaller | 8.94× smaller |
| Frozen mixed-bit | 10,130,160 | 9.66 | 3.59× smaller | 14.10× smaller |

最后一列使用此前同一 validation slice 的 full-prefix mean
`142,853,120` bytes；它是跨作业的存储形状对照，不是本次同作业 latency 对照。

DCP 实际加载也通过：

- 每 rank BF16 模型常驻：`69,354,811,904` bytes = `64.59 GiB`。
- 8 rank 最大加载峰值：`69,816,927,232` bytes = `65.02 GiB`。
- collective load：约 `218.18 s`。
- parameter sample gate：base 在 8 rank 均为 `0/2079` 改变，SFT 在 8 rank 均为
  `420/2079` 改变，且各自跨 rank digest 一致；这证明加载确实改变了可见 BF16 权重样本，
  不是误评 base 模型。
- 三个模型/数据/code ledger 均被 launcher 强校验并写入 `analysis.json`。
- 结束后 8 张 GPU 的 used memory 均为 0 MiB。

## 下一步决策

1. 不继续扩大当前 dense SFT 训练。它优化了内部 CE，但没有恢复冻结下游 F1，
   且训练到 256/384 step 已出现 heldout 回升。
2. 把训练目标改为 replay-aware：teacher 用 dense/full-prefix logits，student 用
   完整 Q16/Q8/frozen lower-state 路径；同时对 query 全位置做 KL/CE，而不是
   只做普通 dense answer CE。
3. 真正的 cached-two-stage 反向仍受 mutable GDN cache autograd 阻断；短期可先
   用 detached document state 训练 query/suffix，长期实现 functional cache。
4. 预注册下一轮主门禁：SFT/adapter 后 Q16-vs-dense interface gap，Q8-vs-Q16
   quantization gap，以及 frozen-vs-Q16 compression gap；不要只看训练 KL 或
   train-split heldout CE。

## 产物

- 本地完整审计包：`results/gpu-sft-full-state-downstream-20260812b/`
- 主分析：`results/gpu-sft-full-state-downstream-20260812b/analysis.json`
- 主分析 SHA256：
  `814b081439dd4a466f62a59059ba9b708cc8c5ac7747c443d394043965e6b484`
- 正式 YAML：`qs/qcomem-sft-full-state-downstream-20260812b.yaml`
- 成功 Trial：`1832184`
