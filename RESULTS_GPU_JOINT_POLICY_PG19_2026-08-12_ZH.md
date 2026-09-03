# Q-CoMem PG-19 Joint Mixed-Bit 校准结果（2026-08-12）

## 结论先行

扩大校准集并用真实联合量化复核后，得到一个值得保留的同内存档候选：

```text
residual Q4；lower-cache layers = [Q8, Q8, Q8, Q2, Q8, Q8, Q8]
```

它在当前 PG-19 train calibration 上，比 frozen static 再少 `4.93%` 持久字节，并取得更低
的 joint objective；但配对 bootstrap 95% CI 跨过 0，因此目前只能称为 **PG-19 校准候选**，
不能声称下游能力优于 frozen static，也不能继承以前 LongBench validation 的标签。

更激进的 minus-25% 候选虽然比 frozen static 少 `28.48%` 持久字节，但校准质量显著变差，
本轮不应晋级。

## 任务与数据边界

- QS Job/Trial：`235196 / 1832355`，状态 `Complete`；
- 页面：<https://qs2.devops.xiaohongshu.com/model/production/job/trial/235196/1832355>；
- 队列与资源：queue 400，8×H20-141G；与 LoRA 的 queue 408 分离；
- 执行时间：8 分 24 秒；UTC `14:38:43` 开始、`14:45:48` 完成；
- 模型：Qwen3.5-35B-A3B；全程 inference，无 backward、optimizer 或 checkpoint；
- 每卡加载一份模型，实测最大 CUDA allocated 为 `70,178,397,696` bytes，约
  `65.36 GiB`，低于 141 GiB；
- 数据只来自官方 `deepmind-gutenberg/train/*.txt`；64-book provenance-locked PG-19
  train 子集中按固定 hash 选 32 本，每本一个窗口；
- 每个窗口为 document 1024 token + continuation query 128 token，在 8 个 query
  position 测量 logits，共 `32 × 8 = 256` 个位置；
- **没有读取 LongBench source 6--35，也没有读取 frozen test-v2 source 68--99，自动
  policy 没有使用任何正式 validation 的答案、F1 或标签。**

固定校验值：

| 对象 | SHA-256 |
|---|---|
| PG-19 train JSONL | `ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c` |
| 独立 provenance manifest | `5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c` |
| tokenizer 后 32 个冻结窗口 | `5d295ac27424ed71b3036bcdb0c3ee2bc30e6bf92460a53770ab92d3d1a0a3a6` |
| 阶段 2 前冻结的 candidate 文件 | `dc3b72c7c50bf016e2499140cda615b074f1d76ff69456718b51dd539a271bd7` |

## 方法

阶段 1 把 residual 和 7 个 lower-cache layer 分给 8 张卡，每个组件分别测
Q2/Q4/Q8/Q16。组件的可加 score 只负责从 `4^8 = 65,536` 个组合中提名 6 个
same-memory 和 6 个 minus-25% 候选。

阶段 2 先冻结候选文件，再对 12 个自动候选和 Q16、frozen static、uniform Q8 三个
control 运行真实 `residual + 7 layers` 联合 pack/dequant。最终排序完全依据真实联合结果，
不把阶段 1 的可加近似当作结论。

预先固定的校准目标为：

```text
mean forward KL
+ mean positive PG-19 next-token NLL delta
+ 0.1 × (1 - top-1 agreement)
```

其中 positive NLL delta 只累计 candidate 相对 Q16 teacher 变差的 token；原始有正有负的
mean NLL delta 也同时保留。

## 主要结果

下表的“持久状态”只对应本轮固定的 `1024 + 128` token 校准窗口和当前 packing metadata；
它可以在本表内做相对比较，不能直接冒充此前 4096-token deployment benchmark 的 MiB。

| policy | bits：residual；7 layers | 持久状态 MiB | 相对 Q16 压缩 | forward KL | mean NLL Δ | positive NLL Δ | top-1 agreement | joint objective |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q16 control | `16; 16,16,16,16,16,16,16` | 18.375 | 1.00× | 0 | 0 | 0 | 100.00% | 0 |
| uniform Q8 | `8; 8,8,8,8,8,8,8` | 6.574 | 2.80× | 0.012639 | +0.009158 | 0.037001 | 94.53% | 0.055109 |
| frozen static | `4; 8,8,8,4,8,8,8` | 5.074 | 3.62× | 0.013286 | +0.004101 | 0.036456 | 94.92% | 0.054820 |
| auto same-memory | `4; 8,8,8,2,8,8,8` | 4.824 | 3.81× | 0.010055 | +0.002898 | 0.034486 | 95.70% | 0.048838 |
| auto minus-25% | `4; 8,8,4,2,4,4,2` | 3.629 | 5.06× | 0.052083 | +0.032330 | 0.085373 | 89.45% | 0.148003 |

Q16 control 在 32 个窗口、256 个测量位置上得到 KL、NLL delta、relative logit L2 和
max-abs error 全为 0，top-1 agreement 为 100%，说明 teacher/control exactness 门禁通过。

## 相对 frozen static 的判断

| 预算档 | 自动候选 | 字节变化 | joint objective Δ | paired bootstrap 95% CI | 本轮判断 |
|---|---|---:|---:|---|---|
| same-memory | `pg19-joint-auto-00` | -4.93% | -0.005982 | [-0.015477, +0.002687] | 点估计更好，但 CI 跨 0；保留为校准候选 |
| minus-25% | `pg19-joint-auto-07` | -28.48% | +0.093183 | [+0.066416, +0.122756] | 质量显著变差；不晋级 |

same-memory 候选把原 frozen static 的第 4 个 lower-cache layer 从 Q4 降到 Q2，反而在
当前 32-book 校准上的 KL、positive NLL delta 和 top-1 agreement 三项都略好。这不意味着
“Q2 天生更准”：有限样本、量化误差交互和非单调扰动都可能造成这个结果；CI 跨 0 也说明
不能把点估计写成稳定增益。

minus-25% 的 CI 整体大于 0，说明在这组 PG-19 calibration windows 上，新增的 28.48%
压缩伴随明确质量代价。它最多可作为将来量化感知 LoRA/SFT 的恢复目标，不能作为当前
near-lossless operating point。

## 可以 claim 与不能 claim 的内容

现在可以说：

- joint pipeline 已经能在真实模型前向中显式控制 residual 和每个 lower-cache layer 的
  Q2/Q4/Q8/Q16，并按实际 pack/dequant 结果选策略；
- 32-book PG-19 train calibration 上出现一个比 frozen static 少 4.93% 字节、点估计更优
  的候选；
- 28.48% 更激进档在当前校准上显著变差；
- Q16 exactness、数据 provenance、candidate-before-eval freeze 和 8-rank 完整性门禁均
  通过。

现在不能说：

- 自动候选提高了 LongBench F1 或任何下游任务精度；本轮根本没有使用下游 QA 标签；
- 自动候选统计显著优于 frozen static；same-memory 的 95% CI 跨 0；
- 本轮 4.824 MiB 就是 4096-token 实际部署显存，或等价于整模型 GPU 显存；它只是固定
  校准窗口下的持久文档状态；
- 阶段 1 的 predicted component objective 是真实性能；它只是候选召回的可加近似；
- 可以用已经揭晓的 source 6--35 validation 结果继续选 policy。这样会把正式 validation
  标签泄漏回自动策略搜索。

## 下一步冻结协议

1. 保留 frozen static 和 Q16 为不可删除的 controls；把
   `4; 8,8,8,2,8,8,8` 冻结为一个预注册研究候选，不因旧 validation 结果再调 bits；
2. 先等 mixed-bit、LoRA/SFT、KV/runtime 三条线的算法和 checkpoint 全部冻结；
3. 再制定一次新的独立下游协议，把 frozen static、冻结自动候选和必要的 Q16 control
   同时放入；若最终使用 test-v2 68--99，只允许一次性报告，不能看完再回头选择 policy；
4. 若希望在 final test-v2 前继续开发，应使用独立、预先划分且不与 LongBench 正式
   validation/test 重叠的数据，而不是复用 source 6--35；
5. 只有独立下游结果与实际 4096-token deployment benchmark 都完成后，才能判断新增的
   4.93% state reduction 是否值得替换论文主线的 frozen static。

完整机器可读产物位于
`results/gpu-joint-policy-pg19-calibration-20260812a/`，核心汇总为
`joint_policy.json`，冻结候选为 `joint-policy-candidates.json`；8 个 profile、8 个 joint
eval、数据审计、GPU ledger、测试日志和阶段时间戳均已保留。
