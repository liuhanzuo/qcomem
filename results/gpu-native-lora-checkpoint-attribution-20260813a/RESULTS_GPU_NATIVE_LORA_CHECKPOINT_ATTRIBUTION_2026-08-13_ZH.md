# Native LoRA checkpoint 归因实验 A（2026-08-13）

## 结论

固定同一 full-state frozen-static Q4/Q8 policy 后，adapter disabled、native LoRA
step 0、64、128 四个条件均在 LongBench validation source index `6--35` 的同一 60 条
样本上完成配对推理。三个 LoRA checkpoint 的总体 F1 都没有高于 adapter-disabled
control；所有相对 control 的 95% 配对 bootstrap 区间均包含 0。因此，这轮证据不支持
“训练步数增加恢复了下游精度”的表述。

step 128 相比 step 64 的总体 F1 回升 `+0.00692`，但 95% CI 为
`[-0.04308, +0.05556]`。这只能视为采样内波动，不能解释为 late-training recovery。
Qasper 与 2WikiMQA 的方向仍然不同，也进一步说明单一 teacher-KL checkpoint 指标不能代替
生成式任务评价。

## 正式任务与冻结协议

- Job / Trial：`235811 / 1834193`，终态 `Complete`；
- 页面：<https://qs2.devops.xiaohongshu.com/model/production/job/trial/235811/1834193>；
- 硬件：单节点 8×NVIDIA H20-3e；总运行时间 8 分 10 秒；
- 来源训练任务：`235749 / 1834056`，本任务只做推理归因，没有新训练；
- 数据：Qasper、2WikiMQA 各 30 条，固定 source index `6--35`；
- validation SHA-256：
  `1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe`；
- prompt：LongBench v1 official；decode：greedy argmax；最大输入 4096 tokens；
  Qasper/2Wiki 最大生成分别为 128/32 tokens；
- policy：depth 7，residual Q4、attention Q4、linear Q8、lower-cache bits
  `[8,8,8,4,8,8,8]`；document residual 与完整 lower-layer
  KV/recurrent/conv state 都包含在内；
- 四个条件在每个 rank 的同一模型进程中顺序执行；disabled control 也安装并常驻 step-0
  adapter 结构，只关闭 adapter，避免改变 caller/model layout；
- 这批 validation 在原正式 LoRA 实验中已经消费。本实验只做预注册归因，结果不得用于重新
  选择 checkpoint 或 policy；test-v2 source `68--99` 没有读取。

checkpoint SHA-256：

| checkpoint | SHA-256 |
|---|---|
| step 0 | `5e53f4a90facbf83ba41e4b7fdbe46a0460dc7a87ec9211491680c8c68a9ac9a` |
| step 64 | `29d30bc121494f4e868b20cdfa605a69466798115c8e5cc1746d252ad457b9fb` |
| step 128 | `a47cfba5e5a8c8e62674fbc0a0e3bd716c366f4d1380100eb42da1c2db903843` |

## F1 与配对置信区间

| 条件 | Overall F1 | Qasper F1 | 2WikiMQA F1 |
|---|---:|---:|---:|
| adapter disabled | **0.54360** | 0.42307 | **0.66413** |
| native LoRA step 0 | 0.53392 | 0.42753 | 0.64032 |
| native LoRA step 64 | 0.52138 | 0.38911 | 0.65365 |
| native LoRA step 128 | 0.52830 | **0.43628** | 0.62032 |

相对 adapter-disabled control：

| 条件 | Overall Δ | 95% paired bootstrap CI | Qasper Δ | 2WikiMQA Δ |
|---|---:|---:|---:|---:|
| step 0 | -0.00968 | [-0.06329, +0.04004] | +0.00446 | -0.02381 |
| step 64 | -0.02222 | [-0.09074, +0.03818] | -0.03396 | -0.01048 |
| step 128 | -0.01530 | [-0.06410, +0.02267] | +0.01320 | -0.04381 |

checkpoint 之间：

| 比较 | Overall Δ | 95% paired bootstrap CI |
|---|---:|---:|
| step 64 - step 0 | -0.01254 | [-0.05383, +0.01629] |
| step 128 - step 0 | -0.00563 | [-0.04874, +0.02584] |
| step 128 - step 64 | +0.00692 | [-0.04308, +0.05556] |

## Prediction 与 token agreement

相对 adapter-disabled：

| 条件 | prediction exact | token-sequence exact | mean aligned-token agreement |
|---|---:|---:|---:|
| step 0 | 0.8333 | 0.8333 | 0.8842 |
| step 64 | 0.8000 | 0.8000 | 0.8653 |
| step 128 | 0.8167 | 0.8167 | 0.8678 |

step 128 与 step 64 的 prediction/token-sequence exact agreement 均为 `0.9167`，mean
aligned-token agreement 为 `0.9492`。训练后期的变化数量已经较少，但变化并没有稳定转化为
更高的总体 F1。

## 2WikiMQA yes/no/entity/abstain confusion

分类规则先应用 LongBench answer normalization；标准化结果精确为 yes/no 时分别归类，空串、
unanswerable/unknown/cannot answer 等归为 abstain，其余归为 entity。矩阵行是真值类型、列是
预测类型。

本切片的 2WikiMQA 真值构成为 `yes=2, no=2, entity=26, abstain=0`：

| 条件 | yes→yes | yes→no | no→no | entity→entity | abstain predictions |
|---|---:|---:|---:|---:|---:|
| adapter disabled | 2 | 0 | 2 | 26 | 0 |
| step 0 | 1 | 1 | 2 | 26 | 0 |
| step 64 | 1 | 1 | 2 | 26 | 0 |
| step 128 | 1 | 1 | 2 | 26 | 0 |

因此类型级别上，所有启用 LoRA 的 checkpoint 都引入了同一个 `yes→no` flip；没有出现
no→yes、entity/abstain 互换。不过 entity 类内部仍可能换成错误实体，所以类型矩阵不能代替 F1。

## 解释边界

这项实验把“LoRA 初始化本身”和“继续训练带来的变化”拆开了：step 0 相对 disabled 已让
10/60 个 prediction 改变；step 128 相对 step 0 的总体 Δ 仍为 `-0.00563`，区间跨 0。
所以目前既不能把下降全归因于 128 步训练，也不能说训练成功恢复了 adapter 初始化造成的偏移。

更可靠的判断是：当前 Interface warm-start adapter 已改变生成分布，而本轮 domain KL 训练没有
在这 60 条生成式 validation 上产生一致恢复。下一轮若继续优化，应在新的、未消费的开发集上
预先定义任务答案 proxy/数据集均衡目标，再冻结 checkpoint；不能用本报告的 source `6--35`
反向挑 step 或超参数。

## 产物

- 远端 run：
  `/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/native-lora-checkpoint-attribution-20260813a/`；
- 本地精简审计包：`results/gpu-native-lora-checkpoint-attribution-20260813a/`；
- 机器可读汇总：`checkpoint-attribution-analysis.json`，SHA-256
  `ea3e7c77f06a8e4e8a63f40b11376c5763010c57ca23f4985b8443b95f7f220d`；
- code ledger SHA-256：
  `85b2cb775f1b968934d32c9f7cf775577990c4e79ee8d38fd70a3012a3752af0`。
