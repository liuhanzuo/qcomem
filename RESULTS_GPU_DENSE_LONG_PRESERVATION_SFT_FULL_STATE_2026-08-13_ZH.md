# 新版 Dense SFT × Full-state Q-CoMem Validation（2026-08-13）

## 一句话结论

新版 long-document/general-replay/teacher-preservation Dense Full-SFT 得到了一个**正向但尚未显著**的结果：
同一正式作业内，dense F1 从 Base 的 `0.54289` 升到 SFT 的 `0.55641`，paired delta
为 `+0.01352`，但 95% CI `[-0.03782, +0.07393]` 跨 0。因此可以说新版训练
消除了上一版的负向点估计，并出现了下游增益信号；还不能说已经证明能力稳定提升。

系统端结果更有意思：SFT frozen-static 的 F1 为 `0.55744`，相对同一个 SFT dense
为 `+0.00103`，同时 persistent document state 从 Q16 的 `34.68 MiB` 降到
`9.66 MiB`（`3.59×`）。它通过预注册的 mean non-inferiority margin，但 CI 仍跨 0；
因此目前应表述为“在这 60 条 validation 上没有观察到 mixed-bit endpoint 的额外平均损失”，
不能表述为“量化提高了模型能力”。

## 正式任务与协议

- Job `235777` / Trial `1834066`，状态 `Complete`。
- 资源：queue 385，单节点 `8 × H20-141G`。
- QS 生命周期：`2026-08-13 13:17:30–13:27:55`，约 10 分 25 秒。
- launcher stage：`13:19:12–13:27:33`。
- validation parent SHA256：
  `1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe`。
- 父文件只含 QASPER/2WikiMQA source `4–35`；正式 evaluator 严格选两类各
  source `6–35`，共 60 条，并排除 calibration `4–5`。
- LongBench test-v2 path/hash/raw rows 均未读取，analysis 记录
  `raw_test_v2_read=false`。
- 同一个 8 卡作业内先用 Base 运行四条路径，再完整验证并 collective load SFT
  step-128 DCP，随后对同一批样本运行相同四条路径，避免跨 job 状态成为主要混杂因素。
- SFT checkpoint manifest SHA256：
  `12f837bdde1dadf5c820d4527685bfe3bf321f5494e25db3a4bb662aab168c57`；
  payload-directory SHA256：
  `89c788e4112c7c9dc7e801201979cdcb69600e067cd9497db456cb63f50ba65b`。
- DCP rank-0 对约 139 GB payload 做了一次完整 hash/index 验证；8-rank collective
  load 为约 `222.8 s`，加载后的最大 GPU allocated peak 为约 `65.01 GiB`。

四条路径均使用 BF16 模型权重；Q16/Q8/Q4 只描述 persistent document residual 与
lower-layer KV/recurrent/conv state：

1. `dense`；
2. depth-7 full-state Q16；
3. depth-7 full-state Q8；
4. frozen-static：residual Q4、attention state Q4、linear state Q8、lower-layer
   bits `[8,8,8,4,8,8,8]`。

## 主 F1 结果

| 模型 | Dense | Full-state Q16 | Full-state Q8 | Frozen-static |
|---|---:|---:|---:|---:|
| Base | 0.54289 | 0.54624 | 0.54620 | 0.54360 |
| 新版 SFT | **0.55641** | 0.53991 | 0.54808 | **0.55744** |
| SFT − Base | +0.01352 | -0.00632 | +0.00188 | +0.01384 |

四条 SFT−Base paired CI 均跨 0：

| 同一路径 SFT − Base | F1 delta | Paired bootstrap 95% CI | Mean margin |
|---|---:|---:|---|
| Dense | +0.01352 | [-0.03782, +0.07393] | pass |
| Full-state Q16 | -0.00632 | [-0.04062, +0.02669] | pass |
| Full-state Q8 | +0.00188 | [-0.03998, +0.05167] | pass |
| Frozen-static | +0.01384 | [-0.04036, +0.07613] | pass |

这里的预注册 mean gate 是点估计门禁：overall delta 至少 `-0.02`，且每个数据集
delta 至少 `-0.03`；95% CI 独立报告，不作为该 gate 的判定条件。

## 改善来自哪里

新版 SFT 的正向点估计主要来自 2WikiMQA，而不是两个数据集都变好：

| 路径 | 2WikiMQA SFT−Base | QASPER SFT−Base |
|---|---:|---:|
| Dense | +0.04778 | -0.02073 |
| Q16 | -0.00889 | -0.00376 |
| Q8 | +0.01111 | -0.00734 |
| Frozen-static | +0.05000 | -0.02232 |

这与新版数据加入多文档 2Wiki domain 的设计方向一致，但只是相关性信号，不是因果证明。
当前结论更准确地说是：**多文档能力有正向信号，长文档 QASPER 仍是主要短板。**

## SFT 模型内部的部署路径

| SFT 路径 | F1 | 相对 SFT dense | 95% CI | Mean margin | Persistent state | 相对 Q16 |
|---|---:|---:|---:|---|---:|---:|
| Dense | 0.55641 | 0 | — | — | — | — |
| Q16 | 0.53991 | -0.01650 | [-0.06033, +0.01550] | **fail** | 34.68 MiB | 1.00× |
| Q8 | 0.54808 | -0.00833 | [-0.02500, 0] | pass | 15.24 MiB | 2.28× |
| Frozen-static | 0.55744 | +0.00103 | [-0.03230, +0.04372] | pass | 9.66 MiB | 3.59× |

Q16 的 overall delta 尚高于 `-0.02`，但 2WikiMQA delta 为 `-0.03667`，低于
per-dataset `-0.03` 门槛，所以失败。Q8/frozen 反而比 Q16 点估计更高，这是 greedy
generation 分支和小样本非单调性的典型表现；不能解释成“bit 越低越准”。直接可支持的结论只有：

- Q8 和 frozen-static endpoint 在本次 mean gate 上通过；
- frozen-static 用 Q16 的 `1/3.59` persistent bytes；
- Q16 split/replay interface 仍不稳定，压缩路径排序需要重复运行或更大样本确认。

Base frozen-static 相对 Base dense 也只有 `+0.00071`，95% CI
`[-0.03149, +0.04443]`，说明 frozen endpoint 的总体近等在 SFT 前后都能观察到。

## 对当前研究判断的影响

1. 新版数据设计值得保留。与同 job Base 配对后，dense 和 frozen-static 都出现约
   `+0.014` 的正向点估计，明显好于只看内部 CE 的证据等级。
2. 仍不能把 Dense Full-SFT 作为最终算法贡献。它的 forward 没有 Q-CoMem replay、
   quantization 或 cache-aware objective；真正的算法训练证据应来自 native functional-cache LoRA。
3. 当前最有吸引力的系统点是 frozen-static：`9.66 MiB` persistent state、相对
   Q16 `3.59×` 压缩、相对同 SFT dense 的 mean F1 近等。
4. QASPER 没有随 2Wiki 一起改善。下一轮应单独审计 QASPER 文档截断、evidence retention、
   answer style 和每文档多 query 的贡献，不能只增加总训练步数。
5. validation 已用于本轮分析和下一步决策；继续保持 test-v2 `68–99` 封存，等
   checkpoint、native LoRA 和 bit policy 全部冻结后再进行唯一一次 final test。

## 产物

- 正式 YAML：
  `qs/qcomem-dense-long-preservation-sft-full-state-validation-20260813a.yaml`
- 本地完整审计包：
  `results/gpu-dense-long-preservation-sft-full-state-validation-20260813a/`
- 主分析：
  `results/gpu-dense-long-preservation-sft-full-state-validation-20260813a/analysis.json`
- 主分析 SHA256：
  `b47d5cbba79d5dbd2292fe5295547a79a8a01228cd98cb663789712a73be15f4`
- 90 个远端结果/日志/ledger/stage 文件的本地 SHA 清单：
  `results/gpu-dense-long-preservation-sft-full-state-validation-20260813a/artifact-ledger.sha256`
