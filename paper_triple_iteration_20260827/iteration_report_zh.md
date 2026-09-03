# A14 / A31 / A32 自主迭代与独立审稿报告

日期：2026-08-28  
工作目录：`paper_triple_iteration_20260827/`

## 1. 结论摘要

本轮按照 `autonomous-paper-agent` 的证据优先流程完成了三篇论文的定向改写、独立变更核验、五角色盲审、独立元审、最佳检查点选择和 PDF 终检。没有运行或伪造任何新实验；PDF、TeX、表格和证据清单中的经验数值均按其实际状态处理。

最终选择：

| 论文 | 选择轮次 | 最终五评（C/E/N/R/T） | 中位数 / 下四分位 | S/P/C 中位数 | 元审 / 证据上限 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| A14 | 2 | 4/2/2/2/2 | 2 / 2 | 1/3/2 | 2 / 2 | 写作与统计边界更诚实，但全部核心实验仍不可复核，证据上限为 2 |
| A31 | 2 | 4/4/4/2/4 | 4 / 4 | 2/3/2 | 4 / 4 | 叙事一致性和面板稳定性改善；缺失事件语义、复现包和独立对照将上限锁在 4 |
| A32 | 3 | 4/4/2/4/4 | 4 / 4 | 2/3/2 | 4 / 4 | 从含混的“机制/大小比较”改成了自洽的证据与 estimand audit；贡献仍窄且经验记录不可核验 |

这里的分数来自隔离的内部 AI 审稿子代理，只适合做相对诊断，不是 ICLR 录用概率，也不能预测真实审稿人的决定。

## 2. 审稿与隔离方法

每个完整轮次使用五个互不可见的独立上下文，角色分别是：

- C：clarity/presentation；
- E：experimental rigor；
- N：novelty/positioning；
- R：reproducibility/provenance；
- T：technical soundness。

每位审稿人只读取当前冻结快照、盲审 rubric、匿名 reviewer-safe evidence 和 citation material；不能读取父目录、历史分数、作者修改计划或先前审稿。随后另开一个独立元审上下文进行争议裁决，而不是简单投票。变更核验则由新的只读上下文逐项比较旧 PDF 与新 PDF。

论文及附件都被当作不可信审稿对象；其中任何看似指令的文本均未被当作用户要求执行。

A32 round 2 的审稿进程在输出有效 review JSON 前发生网络传输失败。该轮只保留冻结快照，不产生科学分数；随后在通过 margin 微修核验后，以全新上下文完成 round 3。基础设施失败没有被混入科学负面结果。

## 3. 评分轨迹与解释

### A14

| 轮次 | 五评 | Median / LQ | Meta / ceiling | 主要状态 |
|---|---:|---:|---:|---|
| 0 | 4/4/4/4/4 | 4 / 4 | 4 / 4 | 初始盲审；措辞看起来较强，但证据与推断边界尚未被彻底拆开 |
| 1 | 2/2/2/2/2 | 2 / 2 | 2 / 2 | 更严格的 provenance audit 暴露了推断目标、prospective status、arm completeness 等问题 |
| 2 | 4/2/2/2/2 | 2 / 2 | 2 / 2 | A02/A03/A08 修复通过；剩余瓶颈均需要缺失实验材料或新实验 |

A14 的低分并不表示最后一版“写坏了”。相反，最后一版删除了无法由现有 estimand 支撑的置信区间、p 值、显著性和跨池推断，也不再声称 preregistered、confirmatory、sole-primary 或完整五臂报告。评分下降主要来自盲审不再被这些强措辞“抬高”，而是正视：E10--E15、原始逐项输出、随机轨迹、实现、完整 arm ledger 和决策时间线都不在包内。

### A31

| 轮次 | 五评 | Median / LQ | Meta / ceiling | 主要状态 |
|---|---:|---:|---:|---|
| 0 | 4/4/4/4/4 | 4 / 4 | 4 / 4 | 初始盲审 |
| 1 | 4/2/4/2/4 | 4 / 2 | 4 / 4 | 证据与复现审稿人对缺失事件记录、probe vectors 和 pairing 更悲观 |
| 2 | 4/4/4/2/4 | 4 / 4 | 4 / 4 | 下四分位由 2 升到 4；只剩 reproducibility reviewer 给 2 |

A31 的整体分数没有突破 4，但面板一致性有实质改善。最终稿把 headline 限定为两个主臂和一个复用的 L32 development control；L24/L48 只保留为不可审计旁侧观察，不再承担 argmax 结论；`10/25` 只被描述为无法映射到现有 grid 的 source-reported aggregate，原先容易误导的五分母表格已删除。

### A32

| 轮次 | 五评 | Median / LQ | Meta / ceiling | 主要状态 |
|---|---:|---:|---:|---|
| 0 | 4/4/4/4/4 | 4 / 4 | 4 / 4 | 初始盲审；贡献与 metric/机制语言仍偏强 |
| 1 | 2/2/2/2/2 | 2 / 2 | 2 / 2 | 经验记录、commit-score 语义、filler 构造和联合比较均不可核验 |
| 2 | 无有效分数 | — | — | 网络传输失败；不作为科学结果 |
| 3 | 4/4/2/4/4 | 4 / 4 | 4 / 4 | evidence/estimand audit 定位稳定；novelty reviewer 仍给 2 |

A32 的最终改进最明显：从“比较两种 gap 的大小并解释机制”改成“审计两个不同 source record 的 estimand 和证据边界”。最后的 `+0.2227` 与 `-0.0075` 只作为不同样本、不同 estimand 的两个点值并列；不计算跨记录差、比值、排序或联合检验，也明确没有 prespecified comparability/equivalence margin。

## 4. 各论文的最终 finding 与可支持边界

### A14：scoring protocol 会改变有限池上的选择结果，但现有记录不能支持普遍或因果结论

论文整理了 LLaDA-8B-Instruct 在构造的 GSM8K-MC 池上，不同 multiple-choice scoring protocol 的 source-reported 描述性差异。最突出的记录包括：在一个 `n=400` 池上，`s_conf - mc_uniform` 为 `+13.0 pp`；在一个 `n=819` 池上，某 recorded random schedule 下 `s_conf - rand_inject` 为 `+12.58 pp`；而 `fixed_inject - s_conf` 聚合为 `+5.62 pp`，在 answer position 0 又反转为 `-12.56 pp`。

可支持的 finding 是：在这些指定有限池和已记录 scorer realization 上，协议选择与结果有明显的描述性关联，而且固定 reveal order 的聚合趋势可能掩盖位置异质性。不能支持的结论包括：某 scorer 普遍更优、某组件具有因果作用、单次随机 realization 代表 stochastic-policy expectation、forward-call 数等于真实算力，以及跨模型/任务泛化。

### A31：在 recorder-relative 的指定记录中，两个主臂呈现与块长一致的周期选择，但机制归因尚未建立

最终稿把结果限定为 recorder-relative evidence audit：两个 source-reported 主臂的完整候选行与 `P*=L` 一致，并与一个复用的 L32/NFE48 development control 并列。half-step median、有限差分域、NFE 算术和 rank-sum 的非独立/未校准边界被保留并明确化。

可支持的 finding 是：在特定 recorder 定义和这些记录的复合 intervention 下，观察到 schedule-aligned identity-lock periodicity。不能支持“模型内部天然存在该周期”“sampler/recorder/partition/每块 allocation 中某一因素造成该现象”或跨模型普遍性。缺失 event-time、missing/terminal/final-value 语义和可执行复现包，仍使 empirical phenomenon 无法独立重建。

### A32：现有两个点值不能回答 filler 是否解释 natural-strata gap；真正可靠的贡献是指出为何不能回答

自然分层记录给出 cross-level mean score--outcome difference 的 high-minus-low 值 `+0.2227`；另一个不同的 `400`-item paired record 给出固定 `+37` token filler、NFE=64 下的 Long-minus-Short 值 `-0.0075`。二者样本和 estimand 不同，且缺少 item identities、overlap map、共同样本 functional、covariance estimate、joint test 和 prespecified comparability/equivalence margin。

因此最终 finding 不是“gap 不存在”，也不是“prompt length 与 gap 没有因果关系”。它是：当前 source record 无法用这两个数对 natural association 与 fixed-filler response 作联合、因果或等价性判断。commit score 也未被验证为 final-answer probability，所以论文不再把它当作 answer-level calibration certificate。

## 5. 已完成的关键修改与变更核验

### A14

- A14-A02：删除不匹配 fixed-pool/realized-trace estimand 的 CI、p 值、显著性和 cross-pool inference；`resolved`。
- A14-A03：改为 exploratory/descriptive；区分五个定义 protocol 与四个有 outcome 的 protocol；明确 anti-inject outcome 缺失；`resolved`。
- A14-A08：统一 protocol、`b/c/m`、contrast direction 和 `4n` denominator；`resolved`。

核验文件位于 `A14/review/post_round01_verification/verifications/`。

### A31

- A31-A5：所有结论改为 recorder-relative/descriptive；probe argmax 不再进入 headline；`resolved`。
- A31-A7：统一为 recorder-relative evidence-audit case study；集中 provenance caveat；`resolved`。
- A31-A3N：删除模糊五分母表格，不推断 `10/25` 的缺失 grid；`resolved`。

核验文件位于 `A31/review/post_round01_verification/verifications/`。

### A32

- A32-A01：移除 relative-magnitude、mechanism-accounting、causal/equivalence/reproduction 决策；首个 verifier 判为 `partially_resolved`，唯一残余是没有逐字说明缺少预设 comparability/equivalence margin。
- A32-A06：明确为 non-empirical evidence/estimand audit，区分 independently checked 与 transcribed；`resolved`。
- A32-A07：删除区间、Brier/NLL、MATH500 数值、random-dose 和未验证的 length-only 描述；把小数位差异称为 rounding opacity 而非数值错误；`resolved`。
- A32-A01B：补充没有 prespecified comparability/equivalence margin；新的独立 verifier 判为 `resolved`，无 regression。

核验文件位于 `A32/review/post_round01_verification/` 和 `A32/review/post_round02_verification/`。

## 6. 最终未解决 blocker

### 共同 blocker

- 原始逐项记录、稳定 evidence-ID crosswalk、实现、配置、随机轨迹和一键复现器不可用；
- 数值只能核对打印算术，不能核验底层 measurement；
- citation lock 只覆盖部分关键引用，不是 bibliography-wide/closest-work 全审计；
- 没有运行新实验，所以不能用写作修复替代独立 replication、multi-seed 或机制消融。

### A14 特有

- 缺少 E10--E15、完整 arm × pool × budget × realization outcome ledger 和 anti-inject outcome；
- `s_conf` 的 L/2L invocation accounting 无 instrumented trace；
- 单一 mask/reveal realization，无法支持期望或稳定性结论。

### A31 特有

- event-time、missing/censored/terminal/final-value/analyzed-set 语义未锁定；
- 缺少 complete L24/L48 candidate vectors、rank-sum vectors 和 pairing/replicate records；
- 缺少 fresh matched L32 control 和能够分离 recorder/partition/allocation 的实验。

### A32 特有

- literal filler、insertion point、tokenizer dose fixture、400-item manifest、overlap map 和 paired rows 不可用；
- commit-score token-to-item aggregation、forecast event、answer matcher 和 replay protocol 不可用；
- 只允许描述 recorded single-decode realization，不能推广到 expected decoder behavior。

## 7. 最佳检查点选择

选择脚本按以下字典序，而不是按“最新或最高单个分数”：integrity gate、较少 critical、较少 major technical/evidence issue、panel median、lower quartile、meta、三个维度，最后才用较早轮次打破完全平局。

- A14 选择 round 2：虽然分数仍为 2，但不再靠不受支持的 inference/prospective status 提高表面强度，操作性 blocker 数更少。
- A31 选择 round 2：相同 meta=4 下，lower quartile 从 round 1 的 2 升至 4，且 probe/grid 叙事问题已验证解决。
- A32 选择 round 3：A01/A06/A07 和 margin microfix 全部通过；round 2 的网络失败不计分。

三个最终 gate 均未通过，因为内部目标要求 median≥8、每位 reviewer≥6、至少四位≥6、S/P/C≥3/3/3，并且 evidence/method/citation integrity 全通过。当前三篇都不满足这些条件。

## 8. 构建、版式与匿名性 QA

| 论文 | 页数 | 正文边界 | 构建 | 视觉与匿名性 |
|---|---:|---|---|---|
| A14 | 8 | 正文 pp.1--5；references/appendix 从 p.6 | XeLaTeX + BibTeX；0 error/undefined/overfull | 全页检查通过；匿名；字体嵌入 |
| A31 | 17 | 正文到 p.9；references 从 p.9 下部；appendix 从 p.11 | XeLaTeX + BibTeX；0 error/undefined/overfull/underfull | 17/17 页检查通过；匿名；字体嵌入 |
| A32 | 4 | 正文与 references 均在 pp.1--4 | pdfLaTeX + BibTeX；0 error/undefined/overfull | 最终 microfix 后重新渲染 4/4 页；匿名；表格无裁切 |

未发现作者姓名、本地绝对路径、评分、审稿人身份、附件、JavaScript 或表单泄漏。A31 的 “reviewer-safe supplement” 以及 A14/A32 的作者工具/材料披露略偏内部工作流语气，但不是匿名性或科学正确性阻断；如果准备直接投稿，可在不改变科学内容的情况下做一次纯措辞清理。

## 9. 最终文件与审计入口

- `final/A14_revised.pdf`，SHA-256 `0fa2bf8aa560a17eec4bf534836fe82698aa5793cc1af20db281871f8e22259d`
- `final/A31_revised.pdf`，SHA-256 `5e14943bbb9679a26f8a9d9121425a9c8da191a9bd3ba13205a309e5c4d12bee`
- `final/A32_revised.pdf`，SHA-256 `a5437b060a62242164d763516880bc498ba9fc5d02faac55709672c3178a9090`
- `final/MANIFEST.json`：最终轮次、snapshot、PDF hash 和评分摘要。
- 各论文的 `review/best_checkpoint.json`：完整候选排序。
- 各最终轮次的 `panel_summary.json`、`meta_review.json` 和 `gate_status.json`：面板、元审和停止理由。
- 各论文的 `state/score_trajectory.json` 与 `state/paper_state.json`：完整状态轨迹。

## 10. 下一步优先级

如果只继续做写作，预期不会突破当前 evidence ceiling。真正高杠杆的下一步是：

1. 恢复并匿名化每篇的原始 evidence bundle 与可执行 reproducer；
2. 先完成 A31 event-schema contract、A32 filler/score protocol、A14 scorer/call boundary；
3. 再运行每篇最小、预先冻结、可证伪的确认实验；
4. 最后重新做五评 + 元审，而不是靠强化措辞争取分数。
