# 14 篇论文证据保真润色与三路隔离盲审详报

生成日期：2026-08-26

## 一、结论先行

本轮对 14 个唯一论文编号采用最新可用版本进行证据保真润色，并对每份润色后的冻结 PDF 进行三次互相隔离的 ICLR 风格盲审。三条审稿通道分别聚焦新颖性与定位、技术正确性、实验严谨性；审稿人看不到旧分数、编辑日志和彼此意见。评分使用 ICLR 离散档 2/4/6/8/10。

润色的目标是改善论证结构、段落推进、术语一致性、证据边界、表图叙事和 LaTeX 版式；没有新增实验、虚构结果、改变统计量或把事后分析改写成预注册结果。

> 重要限制：A23 与 P3 缺少可确认的精确 LaTeX 源，本轮是由 PDF 保守重建；必须在正式投稿前与作者持有的真实源逐句核对。A1 的主文内容仍延至第 13 页，不符合 9 页主文上限，因此不应直接提交。

## 二、总评分表

| 论文 | R1 新颖性 | R2 正确性 | R3 实验 | 中位数 | 分歧跨度 | 接收侧票数 | 旧中位数 | 变化 | 当前证据上限中位数 | 完成必需修改后的预测中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 4 | 4 | 2 | 4 | 2 | 0/3 | NA | NA | 4 | 6 |
| A10 | 2 | 2 | 2 | 2 | 0 | 0/3 | NA | NA | 2 | 4 |
| A14 | 4 | 6 | 6 | 6 | 2 | 2/3 | 6 | 0 | 6 | 8 |
| A23 | 2 | 2 | 2 | 2 | 0 | 0/3 | NA | NA | 2 | 6 |
| A31 | 4 | 4 | 4 | 4 | 0 | 0/3 | 4 | 0 | 4 | 6 |
| A32 | 4 | 6 | 6 | 6 | 2 | 2/3 | NA | NA | 6 | 8 |
| C1 | 4 | 6 | 4 | 4 | 2 | 1/3 | 6 | -2 | 4 | 6 |
| C2 | 4 | 4 | 4 | 4 | 0 | 0/3 | 6 | -2 | 4 | 6 |
| C3 | 2 | 2 | 4 | 2 | 2 | 0/3 | 2 | 0 | 4 | 4 |
| P1 | 2 | 2 | 4 | 2 | 2 | 0/3 | 2 | 0 | 4 | 4 |
| P3 | 4 | 4 | 4 | 4 | 0 | 0/3 | 4 | 0 | 4 | 6 |
| P5 | 4 | 4 | 4 | 4 | 0 | 0/3 | 4 | 0 | 4 | 6 |
| P7 | 4 | 4 | 4 | 4 | 0 | 0/3 | 4 | 0 | 4 | 6 |
| S1 | 2 | 2 | 2 | 2 | 0 | 0/3 | 4 | -2 | 2 | 6 |

按中位数分组：

- 接收侧（≥6）：A14, A32
- 边缘档（4）：A1, A31, C1, C2, P3, P5, P7
- 拒绝档（2）：A10, A23, C3, P1, S1
- 高分歧（跨度≥4）：无

与可用旧盲审中位数比较：

- 上升：无
- 不变：A14, A31, C3, P1, P3, P5, P7
- 下降：C1, C2, S1
- 标为 NA 的论文没有可比的旧三评中位数，不能据此断言升降。
- 该对比是描述性的，不是润色效果的因果估计：新旧轮次的审稿上下文、角色和随机性并不完全相同；分数不升并不等于写作没有改善，分数下降也不能单独归因于本轮编辑。

## 三、审稿协议与可审计性

- 每篇论文的评分对象是同一份 SHA-256 冻结快照；冻结 PDF 与编辑工作目录分离。
- 三位审稿人分别写入 R1/R2/R3 独占目录，不读取其他审稿人输出。
- 审稿人只读取 PDF 与 reviewer protocol/rubric/schema；C2、C3 额外按技能要求读取 KV-cache 校准锚点。
- 所有 JSON 均通过严格字段、枚举、分值范围、reviewer/role、round 和快照哈希校验。
- 审稿未联网，因此 closest-work 与引文完整性意见属于稿内定位判断，不等于外部文献核验。

问题索引总计：

- 致命问题条目：12；主要问题条目：164；次要问题条目：35。
- 实验严谨性：144 条 reviewer issue 标注。
- 技术正确性：100 条 reviewer issue 标注。
- 限制与负责任表述：90 条 reviewer issue 标注。
- 重要性：66 条 reviewer issue 标注。
- 可复现性：66 条 reviewer issue 标注。
- 清晰度：53 条 reviewer issue 标注。
- 新颖性：24 条 reviewer issue 标注。
- 引文完整性：10 条 reviewer issue 标注。

## 四、论文级详报

### A1

#### 交付与来源状态

- 选定输入：`/Users/liuhanzuo/Downloads/A1-paper.pdf`
- 源状态：`exact_latex`
- 润色 PDF：`/Users/liuhanzuo/MacLLM-Bench/output/pdf/paper_polish_20260826/A1-polished.pdf`
- 冻结 SHA-256：`5bd024b4e45e0b174fcb88835505ca50a995ce03d2d71f68a3ca0ec19706cb40`
- 总页数：23；主文状态：主文内容延至第13页；不符合9页主文上限。
- 版面核验：pass；构建：pass_with_nonfatal_float_warnings。
- 旧评分基线：NA；旧中位数：NA。

#### 三评量化结果

| 审稿人 | 角色 | Overall | Confidence | 建议 | Soundness | Presentation | Contribution | 当前上限 | 必需修改后预测 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| R1 | 新颖性与定位 | 4 | 4 | 略低于接收线 | 3 | 3 | 2 | 4 | 6 |
| R2 | 技术正确性 | 4 | 4 | 略低于接收线 | 3 | 2 | 2 | 4 | 6 |
| R3 | 实验严谨性 | 2 | 4 | 拒绝 | 2 | 2 | 1 | 2 | 6 |

三评中位数为 **4**，均值 3.33，跨度 2，接收侧票数 0/3。

#### 编辑记录

- [结构审计](work/A1/structure_audit.md)
- [语义锁](work/A1/semantic_lock.md)
- [修订日志](work/A1/revision_log.md)
- [待核验事项](work/A1/needs_verification.md)

**修订日志原文：**

> # A1 Revision Log
>
> ## Structure and claims
>
> - Rebuilt the abstract around the three audited questions (level, order, pattern) and moved technical denominator qualifications after the primary results.
> - Preserved the asymmetric conclusion: cross-bed level is descriptive, order is rule-dependent, and the registered pattern is not identified.
> - Removed the main-text subsection that disclosed an earlier external review score and argued against that review. The scientific limitations and parser-lineage boundaries remain in the paper and appendix.
>
> ## Clarity and terminology
>
> - Replaced repeated defensive wording with direct statements of the applicable population, rule, and estimand.
> - Tightened the conclusion without changing the decomposition, direction, or scope of any result.
>
> ## Integrity checks
>
> - Citation keys and labels are unchanged.
> - The set of numeric tokens in `main.tex` is unchanged; repeated occurrences decreased only where the old-review/rebuttal text was removed.
> - Final build has no undefined citation/reference, duplicate-label, fatal LaTeX, or overfull-box diagnostic.

**待核验事项原文：**

> # A1 Needs Verification
>
> - References begin on page 13, so the main text occupies roughly 12 pages---about three pages above the nominal ICLR nine-page allowance. Reaching nine pages safely would require moving several load-bearing evidentiary tables and audit details to the appendix; this high-impact pass does not claim venue-length compliance.
> - Generation seeds and temperatures absent from the frozen records remain unrecoverable, as already disclosed in the manuscript.
> - The parser-lineage analysis remains a deterministic proxy rather than a human-adjudicated parser-precision study; no prose edit closes that evidence gap.

#### R1（新颖性与定位）完整评议

**论文概述：** 本文审计 LLaDA-8B-Instruct 在两个自由生成 GSM8K 评测床上的答案解析敏感性。canonical 判尺允许末位数字 fallback，strict 判尺只接受显式答案标记。论文考察三类问题：不同床上的 canonical 水平能否比较（J1）、跨预算排序在换判尺后是否翻转（J2）、Bed 2 的两个预算点能否视为近似持平（J3）。主要发现是 Bed 1 的 canonical 正确中 88.58% 来自 fallback，而 Bed 2 仅为 1.32%–2.52%；因此 level 比较被判为不可作能力排序。J2 的原始符号规则有 3/28 翻转，但有利的配对门为 0/23；J3 在主要条件 estimand 上差约 1.02 个百分点且区间跨零，结论为未识别。

**最强的已核实贡献：** 最强且由 PDF 内证据直接支持的贡献，是定量揭示同一 canonical 指标可由截然不同的解析通道构成：B1-32 的 fallback-rescue share 为 535/604=88.58%，而三个主要 Bed 2 臂仅为 1.32%–2.52%（PDF p.2，Table 1），并据此明确限制跨床 accuracy level 的解释。这是一个具体、可复核且对评测报告有实际警示作用的测量审计。

**维度理由：**

- Soundness：论文对格式条件化是生成后选择、跨床 level 不可作能力排序、J3 未识别等边界表述相当克制，配对区间和多重性处理也大体透明；但 J2 的有利判定门在看过数据后才被操作化，跨床任务与格式率极不相似，且若干关键生成元数据缺失，因此主要结论只能按探索性测量审计理解。
- Presentation：正文提供了清楚的符号、床/臂/判尺区分及逐项解释清单，全部 23 页视觉检查未见图表裁切；然而篇幅和附录非常密集，并在 PDF 第 23 页直接印出既往内外部评审分数，这与双盲独立评审语境不相容且可能污染读者判断。
- Contribution：最可信的价值是把同一批输出在 canonical 与 strict 解析下的差异做成可审计的案例研究，并明确撤回若干过度主张。相关工作部分也承认恒等式、配对检验和解析审计组件本身均属标准工具；证据只覆盖一个模型、两个不可直接比较的 GSM8K 床和少数规则解析器，因而尚不足以形成广泛的新方法或一般性经验规律。

**优点：**

- 对选择偏差的表述负责：PDF pp.4–7（§3.2、§5.1）明确指出 P(correct|formatted) 条件在生成后事件上，不能作为无条件能力或因果排序。
- 证据链较完整：PDF pp.8–11 的 Tables 5–8 同时报告原始符号、有利门、较宽松门、配对区间和多重性敏感度，没有只保留有利读法。
- J3 没有把非显著差异写成等价：PDF pp.9–12 与 p.23（Table 20 的 J3 行）明确区分点估计、跨零区间、事后 0.5 pp 容差和未识别的主要 MDA80。
- 论文主动更正了 Bed 1 并非多项选择任务、无 1/K chance baseline 的旧表述（PDF p.2，§2），并在限制部分列出跨模型、跨任务和元数据缺口。

**问题与可验证修复：**

##### A1-I1 · 主要 · 新颖性、重要性

- 位置：PDF p.3，§2.1 Related work and closest-work map；PDF pp.12–13，§6–§7
- 观察证据：论文将最接近工作定位为标准答案抽取、选择性预测和 paired audit，并承认 q、a、s=qa 恒等式、区间和检验组件本身并非新方法；实际新证据是一套单模型、双床、特定 fallback 规则的事后审计，同床外部解析仅覆盖规则重解析。
- 重要性：ICLR 论文需要超出单一工件修复记录的可迁移认识。当前结果能说明这一评测资产有问题，却尚未证明所提审计结论或报告规范在更广泛的 dLLM、任务和 judge 上改变科学结论。
- 必需修复：把贡献收窄为案例研究并加入真正的最近方法基线；更理想的是在至少另一模型、另一自由生成任务和多个独立解析器（含 learned judge/Answer Regeneration）上预注册复现同一审计。
- 验证标准：在新增床上预先固定 canonical/strict/learned 三类 judge，报告 rescue share、判尺间 item-level disagreement、人工金标误差和关键结论是否翻转；若效应方向与解释边界跨床保持，才支持可迁移贡献。
- 仍需证据：跨模型、跨任务、跨解析器的独立复现及人工标注对照。
- 预期影响：high；判断置信度：high。

##### A1-I2 · 主要 · 实验严谨性、技术正确性、限制与负责任表述

- 位置：PDF pp.8–11，§5.3–§5.5，Tables 6–9；PDF pp.21–23，Appendices D–F
- 观察证据：原始点估计规则得到 3/28 次翻转；在观察数据后操作化的、有方向偏好的 paired gate 得到 0/23；两个较宽松的 definite gate 各得到 2 次。论文自己说明 estimand wording 虽预注册，但门的具体实现是 post hoc，且 raw reversals 都接近小于 1 pp 的平局。
- 重要性：0/23 是标题级叙事中最容易被读成“排序稳健”的数字，但它来自事后选择且通过排除近似平局使翻转难以入选。并列的合理规则给出不同计数时，不能将最有利门当作确认性证据。
- 必需修复：把 3/28 设为当前数据的主要描述，0/23 及两个宽松门全部标为探索性敏感度；在新数据前唯一锁定门、方向、最小实际差异和多重性方案后复现。
- 验证标准：在不可见的新输出上按冻结协议计算所有 planned pairs；主门及预先声明的敏感门都应给出计数、配对效应和区间，且不得再按观察结果调整阈值。
- 仍需证据：独立确认床的预注册协议、时间戳和 item-level paired outputs。
- 预期影响：high；判断置信度：high。

##### A1-I3 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：PDF p.2，§2 与 Table 1；PDF pp.5–7，§3.2、§5.1 与 Table 4
- 观察证据：Bed 1 使用 1319 个 GSM8K test 项，Bed 2 使用 712 个 train-short-B 项，操纵轴和生成记录也不同；格式化率约为 9.9% 对 98.3%，fallback rescue share 相差 35.2–67.1 倍。条件准确率差异因此比较的是两个完全不同的生成后选中群体。
- 重要性：这组数据无法区分模型能力、题目难度、提示/解码差异和格式服从度。尽管作者在文字上撤回能力排序，J1 仍占据大量篇幅，却不能提供可解释的跨床效应。
- 必需修复：在相同 item、prompt、解码设置和可比格式干预下构造共同支持，或将 J1 完全降格为不可比较性的诊断示例，不再报告暗示模型表现差异的 level gap。
- 验证标准：对同一问题逐项生成两种格式条件，报告无条件正确率、格式率、联合正确且格式化率及预先定义的因果/描述 estimand；检查条件化结论是否在共同支持或逆概率加权后仍存在。
- 仍需证据：相同题目上的配对生成记录和完整 prompt/decoder 元数据。
- 预期影响：high；判断置信度：high。

##### A1-I4 · 主要 · 实验严谨性、重要性

- 位置：PDF pp.9–12，§5.4–§6，Tables 8–9；PDF p.23，Table 20 的 J3 行
- 观察证据：共同支持上的主要条件 estimand 为 310/351 对 310/347，差 +1.018 pp，95% CI 为 [−2.23, 4.17] pp；0.5 pp 等价容差是在看过数据后加入，主要判尺的 MDA80 又无法由已交付摘要识别。只有辅助 strict estimand 出现精确共同分子/联合平局。
- 重要性：当前数据既不能证明差异，也不能证明预先定义范围内的等价；因此 J3 不构成正向科学发现，只能说明现有摘要不足。
- 必需修复：将 J3 保留为明确的未识别负结果，或在预先锁定主要 estimand、等价界值和功效后收集足量的 item-paired 数据。
- 验证标准：新研究的双单侧等价检验在预注册界值内通过，且主要判尺的成对 CI 完全落入该界值；同时报告功效与缺失/不可解析项处理。
- 仍需证据：有预注册等价界值的足量成对样本及主要判尺完整输出。
- 预期影响：medium；判断置信度：high。

##### A1-I5 · 主要 · 可复现性、实验严谨性

- 位置：PDF p.2，§2；PDF pp.12–17，§6 与 Appendix A；PDF p.22，Appendix E
- 观察证据：PDF p.2 明确说明冻结记录没有采样种子和温度；论文还披露原计划臂并非全部可复现，若干分析依赖不可再分发或不同 lineage 的历史工件。解析层能重算，但原始生成层无法从论文信息完整重建。
- 重要性：J1/J2 的差异可能受生成随机性和配置影响，而这正是当前记录无法量化的变化来源。缺失的生成 provenance 限制了独立复核和新床验证。
- 必需修复：公开或在匿名工件中提供每臂完整 prompt、温度、随机种子、checkpoint、sampler、输出及哈希；若确实无法恢复，则删除依赖不可重建臂的确认性措辞并把它们标为历史观察。
- 验证标准：从发布清单一键重建所有纳入主分析的臂，逐项复算 Table 1、J2 inventory 与 J3 paired counts，并验证哈希或预先声明的随机误差带。
- 仍需证据：完整生成 manifest、原始输出、运行命令及可访问匿名工件。
- 预期影响：high；判断置信度：high。

##### A1-I6 · 次要 · 清晰度

- 位置：PDF p.23，Appendix F，Table 20 的 Final synthesis 行
- 观察证据：匿名投稿正文直接列出一个内部六席 panel 和一个外部模型评审的分数，并说明稿件如何针对外部分数定位。
- 重要性：既往评分不是论文的科学证据，放入盲审 PDF 会锚定或污染独立评审，也让附录看起来像修订日志而非学术内容。
- 必需修复：从投稿 PDF 删除全部既往评审分数、代理名称和面向评分的元叙事，仅保留与科学结论有关的解释限制。
- 验证标准：检查最终匿名 PDF，不再出现 panel、review score、overall grade 或针对某个评审等级的措辞。
- 仍需证据：清理后的匿名 PDF。
- 预期影响：medium；判断置信度：high。

**给作者的问题：**

- 作者能否在一批此前未查看的新输出上预先锁定 J2 的唯一翻转门，并把 3/28、0/23 与两个 2/· 结果中的哪一个定义为主结论？
- 能否在相同题目、相同提示、相同解码配置和相同格式率支持下复现 J1，而不是比较 test split 与 train-short-B 两个不同床？
- 为什么只做规则解析器同床重解析，而未运行文中列出的 learned judge/xFinder 或 Answer Regeneration 类基线，并报告其错误类型和人工金标一致性？
- 能否补齐每个生成臂的采样种子、温度及缺失的第九个计划臂，或明确说明这些信息永远不可恢复？
- 为何在匿名投稿 PDF p.23 保留既往评审分数？作者是否会在正式投稿版删除这部分以免影响盲审？

**能提高评分的证据：**

- 在全新数据上预注册并确认 J2 的唯一翻转门，而不是继续依赖事后有利门。
- 在相同题目和解码设置下建立共同支持的格式干预，并跨至少两个模型与任务复现 rescue-share/结论翻转。
- 加入 learned judge、xFinder 或 Answer Regeneration 类解析器，并用人工金标报告准确性和错误类型。
- 补齐可重建的生成 provenance、缺失计划臂及公开匿名工件。

**会降低评分的证据：**

- 独立金标显示 canonical fallback 的大量 rescue 实为误判，或不同解析器 lineage 使主要计数失稳。
- 新数据上预先锁定的 J2 门产生明显排序翻转，否定当前稳健叙事。
- 共同支持实验显示跨床差异主要由题目/提示或生成配置驱动，而非格式通道。
- 主分析所依赖的冻结臂无法验证完整性或与表中哈希、计数不一致。

**伦理标记：** 否。未发现需要升级的伦理问题；研究使用公开数学题评测，不涉及人类受试者或部署决策。

**LLM 使用披露：** 本审稿由隔离运行的 AI 子代理 R1 独立完成，仅用于内部投稿前质量控制；未与任何其他评审者或评审代理通信，也未查看其输出。

**评审限制：**

- 仅审阅指定 SHA256 的 23 页冻结 PDF；未读取源文件、实验资产、修订日志或任何其他评审输出。
- 按隔离任务约束未联网，未独立核验外部引文、相关工作覆盖完整性或匿名工件可访问性；新颖性判断仅相对于 PDF 自述的相关工作。
- PDF p.23 所列既往评审分数被视为不可信审稿对象内容并明确排除在本评分依据之外。
- 逐页视觉核查未发现裁切或无法解析页面。

#### R2（技术正确性）完整评议

**论文概述：** 论文审计一个答案解析器的最后数值 fallback 对 LLaDA-8B-Instruct 在两个 GSM8K 派生床上的影响。Bed 1 的显式格式覆盖约 10%–27%，Bed 2 约 97%–98%；B1-32 的 604 个 canonical correct 中 535 个由 fallback rescue，而 Bed 2 主 arms 的 rescue share 仅约 1%–3%。作者分别讨论跨床 level、同床 arm order 与注册 tie pattern：跨床条件正确率仍不可作为能力尺度；Bed 1 原始点估计有 3/28 个顺序反转，而后验、有利的 paired-definite gate 为 0/23；J3 的主条件 estimand 为 +1.018 pp、区间跨零，故记为 not_identified。

**最强的已核实贡献：** 第 2、7–12 页表 1、5–9 最可靠地支持一个范围明确的测量结论：在冻结的 B1-32 与三个 Bed 2 主 arms 上，canonical correct 中 fallback rescue 的占比相差约 35–67 倍，而 Bed 1 八个可复现 arms 中格式内参考匹配率只比非格式 fallback 通道高 5.8–9.5 pp。作者进一步用 raw 3/28 与 post-hoc gate 0/23 并列，正确显示排序结论依赖判定规则，而非宣称零反转定律。

**维度理由：**

- Soundness：作者对 common-support、配对单位、后验门、选择后条件化和不可识别结论处理得较谨慎，J3 的不确定性表述也基本正确。但严格判尺存在两个相互不一致的冻结来源，J2 的零反转门在看过结果后按有利方向设定，且生成配置与第九个计划 arm 无法恢复。核心数值可作为床内描述，不能承担更广的顺序或能力结论。
- Presentation：主要 estimand、分母和 raw/gated 读法有明确表格，视觉版面无缺页。然而 23 页中大量 provenance、修订史和重复限制显著遮蔽核心结果；最后一页还放入既往评审分数，既与双盲稿件无关，也会造成不当锚定。
- Contribution：论文最有价值的是把 fallback-inclusive 准确率拆为格式覆盖、格式内正确率及联合量，并以 item-paired 数据揭示规则依赖。该框架主要复用标准选择偏差、McNemar 与解析器敏感性思想，实证仅覆盖同一模型的两个非同质床，且作者自己承认 level 不可比较、pattern 不可识别，知识增量有限。

**优点：**

- 明确区分 q=P(fmt)、a=P(correct|fmt) 与 s=qa，并承认 a 条件于 post-generation 行为、跨床不是共同能力尺度。
- 发现 B2-128 full pool 与 B2-160 half pool 不同后，将 J3 重算到 356 个共同问题并使用 item-paired bootstrap。
- 撤回把共享问题对当独立 Bernoulli trials 所形成的零事件上界，改报 raw gap strata，统计边界披露诚实。
- 把有利的 J2 criterion、0.5 pp J3 tolerance、最终 hierarchy 和其他偏离逐项标为 post hoc，而不是伪装成完整预注册。

**问题与可验证修复：**

##### A1-R2-01 · 主要 · 技术正确性、实验严谨性、可复现性

- 位置：第 3 页 §2 Rulers；第 9 页表 5；第 18 页表 12；第 20 页表 17
- 观察证据：同一所谓严格判尺有两个冻结来源：B1-128、B1-256、B0-256、H-PRED-95 的 formatted-correct 分别相差 4、6、4、1 项，导致 Bonferroni 计数从 S1 的 3/8 变为 S2 的 4/8。正文指定 S1 为所有计算来源，但未用独立答案真值说明它为何比直接 whitelist 更符合定义。
- 重要性：论文审计的对象正是 parser measurement；若严格 ruler 本身不能由声明规则唯一生成，通道正确率、显著性以及依赖 strict cell 的 J2/J3 都带有未量化的 measurement uncertainty。方向目前稳定并不等于构念已验证。
- 必需修复：冻结一个公开、唯一的 strict parser，对所有原始输出重跑；对 S1/S2 不一致项做盲人工或独立 extractor adjudication，并把标签不确定性传播到所有主表和 pair verdict。
- 验证标准：从原始输出只按发布代码重建 strict labels，应逐项得到唯一结果；由不知道 source 的标注者复核全部分歧，并报告在最不利 adjudication 下 headline 与 gate 是否保持。
- 仍需证据：逐项分歧清单、独立真值、唯一 parser 版本与重算结果。
- 预期影响：high；判断置信度：high。

##### A1-R2-02 · 主要 · 实验严谨性、技术正确性、限制与负责任表述

- 位置：第 6–8 页 §4、§5.3；第 10 页表 6–7；第 21–22 页 Appendix E
- 观察证据：‘definite’ criterion 在看过数据后按保持原 claim 的方向具体化为 McNemar p<.05 且 paired CI 排零。所有三个 raw reversals 都位于 <1 pp 的四对近似平局中，而该门将该 stratum 全部排除，遂由 raw 3/28 得到 gated 0/23；28 对又共享同一 1,319 个问题。
- 重要性：零值是按结果选择且与反转机制近共线的条件统计，不是预先控制错误率的稳健性证据，也不能估计总体反转概率。论文虽披露这一点，但标题式‘survives’读法仍必须以 raw 结果为主，且不能靠同一床上的更多后验门升级。
- 必需修复：把 3/28 作为唯一主 order 结果，0/23 仅留作探索性选择敏感性；在独立 arms/bed 上预先冻结 criterion、family 与分母进行确认。
- 验证标准：新数据运行前提交完整 gate；既报告所有 raw signs，也报告 gated set，并用问题级 resampling 对整个 arm-ranking functional 给不确定性。
- 仍需证据：独立预注册复现和 arm-level/item-clustered inference。
- 预期影响：high；判断置信度：high。

##### A1-R2-03 · 主要 · 重要性、实验严谨性、限制与负责任表述

- 位置：第 2–5 页 §2–3.2；第 12 页 §6
- 观察证据：Bed 1 使用 GSM8K test、free-form diffusion generation；Bed 2 使用 training split short band、不同 prompt/task format，并称 autoregressive evaluation。其格式覆盖分别约 10%–27% 与 97%–98%，条件化后选中的 population 几乎不重叠。
- 重要性：巨大 rescue-share 差异无法分离 parser、prompt、split、decode mode 与输出长度等因素。作者正确拒绝 capability ranking，但这也意味着跨床部分主要证明了预先明显的不可比性，无法给一般 parser effect 或校准规则提供外部效度。
- 必需修复：在同一问题、prompt、生成输出和 decode 配置上只替换 parser；再跨至少另一模型/任务重复。若不新增实验，将标题、摘要与贡献限定为这两个历史档案的 forensic audit。
- 验证标准：固定生成文本的交叉 parser 矩阵应识别纯 measurement effect；固定 parser 的 matched-bed 对照应量化 bed effect，并报告交互项。
- 仍需证据：matched-output parser intervention 与跨床复制。
- 预期影响：high；判断置信度：high。

##### A1-R2-04 · 主要 · 可复现性、实验严谨性

- 位置：第 2 页 §2；第 8 页 §5.3；第 14–17 页 Appendix A；第 22 页 Appendix E
- 观察证据：sampling seeds 与 temperatures 未记录；原 frozen generation trials 不随匿名包再分发，只提供汇总表的 role locators。计划中的第九 Bed 1 arm 也无法从 shipped records 识别，使 36 对只剩 28 对可复现。
- 重要性：审计可从表格重算不等于实验可重现；无法重新生成、逐项复判或确定计划全集，使 parser 变化与历史 artifact selection 纠缠。对 missing arm 的 adversarial bound 也不能替代真实记录。
- 必需修复：发布去标识的逐项输出、完整 config、trial manifest 与所有计划 arms；若配置不可恢复，明确把工作降为不可再执行的二次档案分析。
- 验证标准：第三方从匿名包重建每个 arm 的 output→canonical/strict label→全部表格；manifest 必须证明计划 arm 集无遗漏。
- 仍需证据：逐项输出、完整配置、原计划全集与第三方重放。
- 预期影响：high；判断置信度：high。

##### A1-R2-05 · 主要 · 清晰度、限制与负责任表述

- 位置：第 23 页表 20 的 Final synthesis 行
- 观察证据：稿件在解释清单中披露并对比既往内部与外部评审分数，还声称当前定位依据其中一项 grade。该信息不是科学证据，也与本次盲审对象无关。
- 重要性：在投稿正文中呈现既往分数会直接锚定后续 reviewer，并可能泄露审稿/编辑过程上下文。它违背独立评价原则，也削弱论文对 measurement hygiene 的可信度。
- 必需修复：删除所有既往 reviewer、panel、score、grade 与‘positioned against’表述；只保留由论文数据支持的科学结论。
- 验证标准：对最终 PDF 做文本扫描，确认不存在先前评分、评审身份、目标分数或要求 reviewer 采纳的元评审内容。
- 仍需证据：清理后的盲审稿。
- 预期影响：high；判断置信度：high。

**给作者的问题：**

- 严格判尺既定义为 hash/boxed whitelist，为何 S1 tight-correct 与 S2 raw-whitelist 在四个 arms 上仍相差 1–6 个 correct items？哪一个才是可由规则唯一重建的 measurement？
- 能否在完全独立的新输出集上预先冻结 paired-definite criterion，再检验 0/23 是否复现，而不是继续使用看过 reversal 后设定的门？
- Bed 1 与 Bed 2 在 split、prompt/task format、decode mode 和格式覆盖上均不同，除宣布不可比外，当前跨床数值对一般 parser audit 提供了什么可迁移预测？
- 缺失的第九 arm、temperature 与 generation seeds 是否可从原始执行环境恢复；若不能，哪些 headline 可以由匿名 artifact 从原始文本逐项重建？
- 为什么双盲投稿的 Table 20 要披露既往内部与外部评审分数？请确认这些元评审信息会从审稿对象中完全删除。

**能提高评分的证据：**

- 用唯一、独立 adjudicated 的 strict parser 重建所有 item labels，并证明主要方向不依赖 S1/S2。
- 在新输出/新床上预注册 J2 gate，完成 matched-output parser intervention 与跨模型复制。
- 发布足以逐项重建的匿名 artifact，补齐 missing arm/config，并删除所有既往评审信息。

**会降低评分的证据：**

- 独立 adjudication 显示通道差异主要来自 strict parser 标注错误。
- 预注册复现中 raw 或 gated order 方向/反转结构不稳定。
- 完整配置或逐项输出无法提供，使主表只能依赖不可审计汇总。

**伦理标记：** 是。不涉及受试者或敏感数据，但第 23 页把既往评审分数写入当前双盲稿件，构成审稿过程完整性与不当锚定风险；本评审未使用这些分数。

**LLM 使用披露：** 本审稿由隔离的 AI 子代理 R2 生成，仅用于内部投稿前质量控制；未与其他评审通信，未读取其他评审目录、作者计划、编辑上下文、旧稿或历史评分。

**评审限制：**

- 遵循隔离要求未联网，未核验外部引文、模型说明或邻近工作。
- 仅审阅冻结 PDF，未读取或执行代码、preregistration、per-item outputs、manifests 或 trial artifacts，故文中 hash 与重算声明仅按稿件评估。
- PDF 本身在第 23 页包含既往评审分数；该内容属于不可信审稿对象，我未将其用于评分，仅将其存在作为稿件过程问题记录。
- PDF 共 23 页，已全文阅读并逐页视觉核查；无缺页或不可辨认页面，末页留白较多。

#### R3（实验严谨性）完整评议

**论文概述：** 本文审计两个冻结的 free-form GSM8K 生成床，比较 canonical parser 的 fallback-rescue 与 strict explicit-format judge。Bed 1 的 B1-32 中 604 个 canonical-correct 答案有 535 个依赖 fallback（88.58%），而三个主要 Bed 2 arms 的 rescue share 仅 1.32%--2.52%。论文据此拆分格式覆盖 q、格式条件准确率 a 和 strict score s=qa，并讨论三类问题：跨床 level、同床 arm order 与规范化后的 tie pattern。作者最终认为 level 不可作能力排序、order 对规则敏感、J3 pattern 在主 conditional estimand 上证据不足。总体范围较谨慎，但大部分关键判定属于事后审计，且生成/解析 provenance 不完整。

**最强的已核实贡献：** 第 2 页表 1 的可复核计数最有价值：同一 canonical v1.1 judge 下，B1-32 的 fallback-rescue share 为 535/604=88.58%，而主要 Bed 2 arms 仅约 1%--3%，形成约 35--67 倍差异。这直接证明只报告 canonical accuracy 会隐藏两个床截然不同的格式/救援通道组成。

**维度理由：**

- Soundness：从冻结计数出发，q=P(format)、a=P(correct|format)、s=qa 的恒等分解和 common-support paired J3 分析是合理的；巨大的两床 fallback-rescue share 差也确实存在。但 J2 的有利 paired gate 仅在估计措辞上预注册、判定准则在看数据后才 operationalize，并恰好排除所有近零 raw reversals；计划的九个 Bed 1 arms 只有八个可识别。生成 seeds/temperatures 缺失，parser 重实现和 strict-cell 来源仍有逐项漂移，故主要顺序与通道结论只能视为审计性敏感度，而非确认性规律。
- Presentation：主文主动区分 level/order/pattern，并披露大量偏离与负结果；页面渲染本身清晰。然而全文 23 页、20 张表和多层 J/K/S/P/MD80 标签使一个较简单的 parser artifact 被过度复杂化，主结果依赖多处附录追踪。更严重的是第 23 页把既往内部/外部审稿评分直接放进双盲稿件，构成明显的审稿过程污染；这些内容在本审稿中已完全忽略。
- Contribution：最清楚的发现是不同 free-form parser 路径会显著改变冻结 GSM8K 分数，且应同时报告格式覆盖与条件正确率。这一实践提醒有价值，但答案抽取敏感性和 selected-population conditioning 均属已知测量问题；本稿仅重审两个异构旧床，既没有新生成、跨任务验证，也没有一个可泛化的新校准方法。

**优点：**

- 正确指出 free-form GSM8K 没有四选 option set，撤回 chance-rate/above-chance 的不当表述。
- 用 q、a、s=qa 的精确恒等式把格式覆盖、条件正确率和 strict joint score分开，避免把 parser rescue 当作模型能力本身。
- J3 在 common 356-item support 上使用 item-paired 统计，并明确不把跨零区间解释为等价或无效应。
- 附录 E 逐条披露估计对象、population、gate、arm inventory 和零事件界等注册偏离，并撤回把共享题目 pair 当作独立 Bernoulli trials 的错误界。
- 第 20--21 页比较四种 parser lineage，直接显示评分和 definite relations 对 parser 选择敏感。
- 逐页视觉核查显示 23 页排版完整，图表与公式无裁切或乱码。

**问题与可验证修复：**

##### A1-I1 · 致命 · 实验严谨性、技术正确性、限制与负责任表述

- 位置：第 6--8 页第 4 节/J2；第 10 页表 6--7；第 22 页附录 E
- 观察证据：28 个 raw Bed 1 pair 中有 3 个 sign reversals；所谓 definite favorable gate 选出 23 对且 reversal 为 0。论文承认“definite”只有估计措辞预注册，具体要求 exact two-sided McNemar p<0.05、paired CI 排除零且方向有利，是看过数据后才 operationalize。所有 raw reversals 都位于小于 1 pp 的 near-tie stratum，恰被该 gate 排除；两个更宽松 gate 各保留 2 个 reversals。
- 重要性：事后选择且按有利方向条件化的 gate 不能提供确认性的零反转证据。它几乎与“远离 near-tie”同义，因此 0/23 是选择规则的结果，而不是 parser normalization 普遍保持顺序的证据。
- 必需修复：把 3/28 raw result 作为唯一主要结果，将所有 gate 降为明确的探索性敏感度；在独立数据上预注册双向、与观察方向无关的 effect-size/uncertainty 门限，并控制完整 pair family。
- 验证标准：独立 arm family 在锁定 gate 下运行，报告所有 eligible/ineligible pairs 与 raw matrix；零反转结论需在未按方向筛选的预定 family 中通过校准的概率或区间门限。
- 仍需证据：带时间戳的 gate 规范、未见数据的独立 arm records、完整 pair-level effect/CI/p 值和 multiplicity 决策。
- 预期影响：high；判断置信度：high。

##### A1-I2 · 致命 · 实验严谨性、可复现性

- 位置：第 8、10、16--17、22 页 J2 arm inventory、表 6/11 与附录 E
- 观察证据：预注册与 frozen plan 声称九个 Bed 1 arms、36 pairs，但 shipped per-arm records 只能识别八个 arms、28 pairs；第九 arm 未识别。论文给出将缺失 arm 作最坏分配的 36-pair bounds，但没有实际观测。
- 重要性：缺失 arm 属于计划 family 的 20% 关系（8/36 pairs）。对其作对抗计数不能恢复效应、pair dependence 或 gate eligibility，也无法验证完整预注册结论。
- 必需修复：找回第九 arm 的明确配置与逐题输出，按相同 parser lineage 重算全部 36 pairs；若无法恢复，必须把目标 family 正式改为八 arm、明确承认预注册主要分析不可完成。
- 验证标准：arm manifest、题目 IDs 与输出哈希应唯一映射九个 arms；第三方能重建 36-pair matrix并复现 multiplicity/gate 结果。
- 仍需证据：缺失 arm 的配置、逐题生成、parser 输出、完整 manifest 与运行哈希。
- 预期影响：high；判断置信度：high。

##### A1-I3 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第 5 页第 3.2 节；第 7 页 J1；第 9、18 页表 5/13
- 观察证据：B1-32 的 q≈0.099，而 B2-96 的 q≈0.983；条件准确率 a 相差 -32.76 pp，但它们对应由生成后格式行为选择出的完全不同子群，且两个床的任务 split、prompt 与配置也不同。additive share 还随任意 anchor 在 [59.11%,95.87%] 变化。
- 重要性：J1 不能识别统一能力或可加的 parser 贡献；它只描述两个不同 selected populations 的条件率。论文虽承认该点，但这也使“two-bed leveling”的科学产出主要是不可比性诊断。
- 必需修复：若要作跨床 level 结论，需在同一模型、prompt、任务与题目上随机化/操纵格式要求或构造共同支持，并预先定义可识别 estimand；否则标题与摘要应进一步限定为描述性 parser audit。
- 验证标准：同题同配置的格式干预中，格式/救援路径由随机化决定且两组有充分 overlap；用 IPW/标准化或随机化差异估计 common-population effect。
- 仍需证据：共同题目上的受控格式干预、propensity/overlap 诊断和预注册 estimand。
- 预期影响：medium；判断置信度：high。

##### A1-I4 · 主要 · 可复现性、实验严谨性

- 位置：第 2 页 Beds, arms, and rulers；第 12--15 页 Reproducibility/A.1
- 观察证据：冻结 artifacts 未记录 sampling seeds 与 temperatures，trials 也未重新分发；作者明确只能从 shipped tables 复算分析，不能重建生成。两床还包含不同 split、prompt、budget 与配置。
- 重要性：rescue share 与格式排放可能高度依赖温度、seed、prompt 和长度预算。无法重生成意味着无法区分稳定 parser artifact 与某次未记录采样的偶然组成。
- 必需修复：在完整记录 prompt、temperature、每题 seed、model/checkpoint、budget 与软件版本的新床上前瞻性重复，并发布逐题原始回答与 judge traces。
- 验证标准：第三方可端到端重建新床；跨 seeds/temperatures 的 rescue share、channel gap 与 order matrix 方向稳定，并报告异质性。
- 仍需证据：可执行生成清单、逐题 seeds/config、原始文本、parser traces 与哈希。
- 预期影响：high；判断置信度：high。

##### A1-I5 · 主要 · 技术正确性、可复现性

- 位置：第 9 页表 5；第 15、18、20--21 页附录 A.1/D、表 12/17/18
- 观察证据：canonical reimplementation 与冻结结果有 6/2136=0.28% 漂移；S1 tight 与 S2 raw-whitelist strict sources 在多个 arms 相差 1--6 个 formatted-correct items，使 Bonferroni 显著 arm 数从 3/8 变为 4/8。不同 parser lineage 还会显著改变绝对 accuracy 与 definite reversal counts。
- 重要性：论文的主题正是 parser 敏感性，因此 load-bearing label 自身没有独立金标准时，通道 gap 与顺序结论可能只是所选实现的边界效应。
- 必需修复：建立覆盖全部分歧与随机抽样响应的双人盲标金标准，发布 parser unit tests/优先级规范；主分析同时报告金标准及预注册 parser，其他 lineage 作为敏感度。
- 验证标准：两个独立实现逐 response 输出一致；对不一致项人工仲裁后，主要计数、gate 与多重性结论在金标准上保持。
- 仍需证据：分歧 response、各 parser trace、盲标与仲裁记录、单元测试和重算结果。
- 预期影响：high；判断置信度：high。

##### A1-I6 · 主要 · 清晰度、限制与负责任表述

- 位置：第 23 页表 20，Final synthesis 行
- 观察证据：双盲投稿 PDF 明文包含既往内部 panel 与外部 reviewer 的评分、维度判断和定位文字。本审稿因必须全文阅读而被动遇到该信息，但已完全忽略其评分与结论。
- 重要性：向后续审稿人暴露既往评分会锚定判断、破坏评审独立性，并把与科学证据无关的内部 QC 元数据带入匿名稿。
- 必需修复：删除所有既往 reviewer/panel 分数、目标评分、接受倾向与审稿比较；在提交前对 PDF 做自动检索和人工 blind-integrity audit。
- 验证标准：新的冻结 PDF 中检索 reviewer、panel、score、overall、accept/reject 等元数据不再返回任何审稿历史，独立检查者确认无隐含评分线索。
- 仍需证据：清理后的 PDF 与 blind-integrity checklist。
- 预期影响：high；判断置信度：high。

**给作者的问题：**

- J2 的“definite”判据为何选择 exact McNemar p<0.05 且 paired-bootstrap CI 排除零，并要求方向有利？在看到三处 raw reversals 的 gap 分布后，还尝试过哪些阈值或 gate？
- 缺失的第九个 Bed 1 arm 是什么配置，为什么 shipped records 无法识别？它是否可能改变 36-pair 计划 family 的 reversal count 或 multiplicity correction？
- 能否由原始生成日志恢复每个 arm 的 temperature、sampling seed、prompt 和 decode 配置，并对至少一部分 bed 重新生成以验证 rescue-share 稳定性？
- strict-cell S1/S2 的 4--6 条差异中，哪些具体响应会改变 Bed 1 通道准确率和 Bonferroni 结论；是否有与两套 parser 独立的人工金标准？
- 为什么将既往审稿评分及 panel 描述写入双盲投稿 PDF？能否在任何后续版本中彻底移除所有审稿历史、目标分数和外部评价？

**能提高评分的证据：**

- 恢复完整九-arm inventory，并在未参与规则选择的独立床上用预注册双向 gate 复验 order reversals。
- 新生成床完整记录 seeds/temperatures/prompts，显示 rescue share 与 channel gap 跨随机性稳定。
- 人工金标准与独立 parser 实现验证 strict/canonical labels，主要结论不再对 S1/S2 或 lineage 边界敏感。
- 在同题同配置的受控格式干预与额外模型/任务上复现 parser-induced score shifts，并给出可识别 common-population estimand。
- 彻底移除所有既往评分/审稿元信息，显著压缩附录审计为清晰的主结果链。

**会降低评分的证据：**

- 补回第九 arm 后增加 definite reversals或改变 family-level order结论。
- 预注册独立重复使 favorable gate 不再为零 reversal，或结果对阈值选择高度敏感。
- 人工金标准使 5.8--9.5 pp channel gap、显著 arm 数或 J2/J3 判定实质改变。
- 跨 seeds/temperatures 重生成后 rescue-share 35--67× 差异显著缩小或反向。
- 发现更多审稿历史、作者身份或内部目标信息残留在匿名稿/附件中。

**伦理标记：** 是。未涉及人类受试者或私密数据；但第 23 页嵌入既往审稿评分，构成双盲评审完整性问题并可能锚定后续审稿人。该元信息必须在投稿前删除；本评审没有采用其中任何分数或结论。

**LLM 使用披露：** 本审稿由隔离运行的 AI 子代理 R3 完成，仅用于内部投稿前质量控制；未与其他评审通信，未读取作者计划、旧评分或其他评审输出。

**评审限制：**

- 仅审阅指定 SHA256 的冻结 PDF；未访问代码、匿名 supplement、原始生成、解析日志或注册文件，因此不能复算表格或核验资产哈希。
- 按隔离要求未联网，未核验外部引文、引用页面、模型/数据来源或论文声称的资产可访问性。
- 已全文阅读并逐页视觉核查全部 23 页；页面和图表完整可读，无裁切或乱码。
- 全文阅读过程中在第 23 页被动遇到既往审稿评分元信息；为保持独立性，本审稿完全忽略其具体评分与判断，所有结论仅基于论文中的实验与方法证据。

### A10

#### 交付与来源状态

- 选定输入：`/Users/liuhanzuo/Downloads/A10-paper.pdf`
- 源状态：`exact_latex`
- 润色 PDF：`/Users/liuhanzuo/MacLLM-Bench/output/pdf/paper_polish_20260826/A10-polished.pdf`
- 冻结 SHA-256：`3ec51ae6ffcf87653a3cf81af085bb834be3c6c87b953736487d0e0c9fe089b1`
- 总页数：10；主文状态：主文在参考文献前不超过9页。
- 版面核验：pass；构建：pass_with_underfull_warnings。
- 旧评分基线：NA；旧中位数：NA。

#### 三评量化结果

| 审稿人 | 角色 | Overall | Confidence | 建议 | Soundness | Presentation | Contribution | 当前上限 | 必需修改后预测 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| R1 | 新颖性与定位 | 2 | 4 | 拒绝 | 2 | 3 | 1 | 2 | 4 |
| R2 | 技术正确性 | 2 | 4 | 拒绝 | 2 | 3 | 1 | 2 | 4 |
| R3 | 实验严谨性 | 2 | 4 | 拒绝 | 2 | 3 | 1 | 2 | 6 |

三评中位数为 **2**，均值 2.00，跨度 0，接收侧票数 0/3。

#### 编辑记录

- [结构审计](work/A10/structure_audit.md)
- [语义锁](work/A10/semantic_lock.md)
- [修订日志](work/A10/revision_log.md)
- [待核验事项](work/A10/needs_verification.md)

**修订日志原文：**

> # A10 Revision Log
>
> ## Structure and claims
>
> - Simplified the title while retaining the 135M, diffusion-language-model, negative-control, and pilot scope.
> - Reordered the abstract to present the matched loss result, emission audit, missing MAUVE result, and inferential ceiling in sequence.
> - Compressed the introduction's repeated reproduction/diagnosis/new-method taxonomy into one scoped contribution paragraph.
>
> ## Clarity and terminology
>
> - Consolidated repeated “not a new metric/method” language without deleting the boundary.
> - Preserved the distinction between paired data-order repetition and random-weight-draw variance.
>
> ## Integrity checks
>
> - Citation keys, labels, and the complete set and multiplicity of numeric tokens are unchanged.
> - Final build has no undefined citation/reference, duplicate-label, fatal LaTeX, or overfull-box diagnostic.

**待核验事项原文：**

> # A10 Needs Verification
>
> - MAUVE remains not recorded; no ratio or MAUVE capability/quality inference can be added.
> - The generation comparison remains a single-prompt, seed-42 pilot without an independent quality anchor.
> - The conclusion ends on main-text page 9; the paper fits the nominal nine-page main-text boundary in this build, with reproducibility material continuing afterward.

#### R1（新颖性与定位）完整评议

**论文概述：** 本文比较由 SmolLM2-135M 转换而来的 masked-diffusion 模型在 AR 预训练初始化与随机初始化下的短程训练。两臂各训练约 30M tokens/458 steps；四个匹配数据顺序重复的末次记录损失相差 2.262 nats，三次新随机权重初始化也保留损失差的符号。生成实验仅用提示“To be or not to be”、seed-42 checkpoint 对、两个温度和每格 32 个样本；AR-init 输出多为短标点/换行残片，random-init 多为词循环。论文尝试用 teacher PPL、长度、alphabetic yield、top-word share、MAUVE、drift 和 entropy 记录退化，但明确不作质量排序或能力结论。

**最强的已核实贡献：** 最稳健的结果是固定初始化 checkpoint 后，四个数据顺序/训练随机性重复均显示 random-init 的最后记录 masked-denoising loss 高于 AR-init，平均差 2.262 nats、配对 t 区间 [2.220, 2.305]；三次独立随机权重 draw 的差值 2.300、2.281、2.272 nats 也同号（PDF pp.4–5，Tables 3–4、Figure 2）。这验证了该短预算配置下的损失符号，但不支持收敛速度或生成质量结论。

**维度理由：**

- Soundness：匹配数据顺序的四个重复、三次独立随机初始化负对照和大量边界说明是认真之处，但训练只到 458 步且仍未收敛，生成评价只覆盖一个提示与一个 checkpoint seed，两个臂均明显退化；MAUVE 未形成结果、drift 重算门失败，因而可支持的只是非常局部的描述性观察。
- Presentation：论文结构清楚，表格把训练预算、生成网格、退化输出坐标和推断上限放在一起；10 页逐页视觉检查未见裁切。不过正文反复强调什么不构成结论，反映核心科学问题与可用证据之间仍有较大落差。
- Contribution：论文明确声明 companion fields 不是新指标、signal layer 不是新 decoder、随机初始化不是竞争基线，最终贡献只是一份 135M 单提示负对照的报告模板。相对于文中相关工作，这不足以构成 ICLR 级的新方法、实证规律或有广泛影响的评测基准。

**优点：**

- PDF pp.3–5 将固定初始化、数据顺序重复和独立随机权重 draw 清楚分开，避免把 n=4 的配对方差误称为随机初始化方差。
- PDF pp.6–8 同时报告 emitted length、alphabetic-word count、top-word share、长度截断 PPL 与短 stub，而没有把低 PPL 直接等同于更好文本。
- 失败结果被保留：PDF p.8 Table 9 明示 drift 重算的 12.6%–18.1% 聚合误差及 56.8% 最坏误差，MAUVE 也因配置/数值问题不进入结论。
- 限制写得诚实：PDF pp.8–9 明确说明单提示、单 checkpoint、仍在变化的损失、退化输出及无独立质量锚。

**问题与可验证修复：**

##### A10-I1 · 致命 · 新颖性、重要性

- 位置：PDF pp.1–3，Abstract、§1–§2 与 Table 1；PDF p.8，§5；PDF p.9，Conclusion
- 观察证据：论文反复声明这些字段不是新 metric、signal schema 不是新 decoder、random-init 不是竞争 baseline，且没有质量排序、capability 或训练发现。相对于 Table 1 所列最近工作，新增内容主要是把长度、字母词产出和词集中度并列报告在一个极窄 pilot 中。
- 重要性：负结果可以重要，但需要揭示反直觉、可泛化的现象或提供可复用且经验证的方法。当前最强正结果只是预训练初始化在极短预算下比随机初始化损失低，这一方向本身预期性很强；报告字段也未被证明能解决实际选择问题。
- 必需修复：明确一个可证伪的核心研究问题，并用多模型/多提示/多训练阶段证据回答；若贡献是测量方法，则需定义并验证其相对现有退化指标的增量效度。
- 验证标准：在预先固定的数据集与提示套件上比较 companion bundle 与标准重复度、长度、distinct-n、人工质量标签的预测/诊断性能，并在独立模型上复现。
- 仍需证据：跨设置效度研究、强基线和清晰的新增科学命题。
- 预期影响：high；判断置信度：high。

##### A10-I2 · 主要 · 实验严谨性、技术正确性

- 位置：PDF p.4，§3.3 与 Table 2；PDF pp.6–7，§4.2、Tables 5–8；PDF pp.8–9，§5–§7
- 观察证据：生成评价仅有一个提示、一个 seed-42 checkpoint pair、两个温度和每格 32 个样本；两臂输出都明显退化。AR-init 在 τ=0.7 只有 1/32 样本达到 128 tokens，PPL 排序随 teacher、温度和长度 cap 改变，且没有独立人工质量锚。
- 重要性：在长度和退化模式严重不同的条件下，PPL 与简单 emission 统计看似“互补”可能只是测量了样本长度/循环形态，不能说明它们对一般生成质量或初始化比较有判别力。
- 必需修复：扩展到多提示、多生成 seed、多 checkpoint 和至少一个独立人工/任务质量锚；按 continuation 级而不是只按 cell 汇总集中度，并预先规定长度处理。
- 验证标准：在独立保留集上评估各指标与盲人工偏好/任务成功的相关、校准及错误案例；用层级 bootstrap 同时覆盖 prompt、checkpoint 和 sample。
- 仍需证据：多提示重复、逐样本统计、人工质量标签与独立验证集。
- 预期影响：high；判断置信度：high。

##### A10-I3 · 主要 · 实验严谨性、重要性、限制与负责任表述

- 位置：PDF pp.3–5，§3.2、§4.1，Tables 2–4、Figure 2；PDF p.9，Table 11
- 观察证据：两臂仅训练 458 步/约 30M tokens，曲线仍在变化；比较的是保留大量 AR 预训练信息的初始化与从头随机权重。四个重复只改变数据顺序/训练随机性，三次独立 weight draws 也仅给出同一短预算下的三个差值，符号检验 n=4 的最小单侧 p 为 0.0625。
- 重要性：这一设计无法区分初始化带来的预期起点优势、学习速度和最终可达性能，也没有给出等 loss、等计算到收敛或更有竞争力的初始化对照。
- 必需修复：加入 step-0 和密集学习曲线，训练到接近收敛或比较达到相同 loss/质量所需计算；增加多个权重 seed 和合理的中间初始化基线。
- 验证标准：预先定义曲线 AUC、time-to-threshold 和最终窗口效应，在足够多的独立权重 draw 上给出层级区间，并检查差异是否在匹配训练阶段后仍存在。
- 仍需证据：长程学习曲线、step-0、更多独立初始化和阶段匹配对照。
- 预期影响：high；判断置信度：high。

##### A10-I4 · 主要 · 可复现性、技术正确性

- 位置：PDF p.4，Table 2；PDF p.8，Tables 9–10；PDF p.9，Reproducibility Statement
- 观察证据：drift scalar/top-k 重算相差 12.6%–18.1%，最坏记录误差 56.8%，因此没有 drift effect；MAUVE 未记录；principal random-init 使用 ambient RNG 且没有 seed 参数，step 0 也未记录；冻结时无经验证的外部匿名 URL。
- 重要性：失败的量化检查不能支持轨迹结论，缺失权重种子又使主要随机初始化无法从头重建。只提供 checkpoint 哈希可验证现有文件，却不能复现产生过程或检验独立 draw 变异。
- 必需修复：修正并重跑 drift/MAUVE，记录完整配置和误差门；为所有权重、训练与生成随机源提供明确种子，加入 step-0，发布可访问的匿名工件。
- 验证标准：独立环境从记录种子生成相同初始权重，重跑诊断后 scalar/top-k 差异低于预设 1% 门，并从公开工件复算所有主表。
- 仍需证据：可执行匿名工件、完整 RNG provenance、修正后的诊断输出。
- 预期影响：high；判断置信度：high。

**给作者的问题：**

- 这篇论文希望回答的 ICLR 级科学问题究竟是什么：初始化影响短程优化、退化生成的测量，还是一个报告规范？当前三者都被主动限定为非方法、非质量结论。
- 能否在多个预先指定提示、checkpoint、采样种子和训练阶段上，用人工或独立质量标注验证 emission companion fields 的判别效度？
- 为何将固定 30M-token 预算下的 AR 预训练初始化与从头随机初始化作为主要对照；是否可加入参数量/训练阶段匹配、随机预训练初始化或训练到可比 loss 的对照？
- 能否重跑 drift 与 MAUVE，并提供 principal random-init 的真实权重种子和 step-0 记录？

**能提高评分的证据：**

- 跨多个提示、checkpoint、随机种子和模型验证 emission companion bundle 对独立质量标签的增量效度。
- 提供长程或等阶段训练比较，证明初始化结果不只是预期的短程起点差异。
- 修复 drift/MAUVE 管线并通过预先规定的重算误差门。
- 发布可访问、可从种子重建 principal random-init 的匿名工件。

**会降低评分的证据：**

- 多提示重复显示 emission 坐标或退化模式随提示/seed 任意改变，无法稳定诊断。
- 训练延长或阶段匹配后 2.262-nat 差异消失或反转。
- 独立重算不能复现主要损失端点或暴露更多序列化/轨迹错位。
- 人工质量评估显示所提 companion fields 相对标准重复度指标没有增量信息。

**伦理标记：** 否。未见需要升级的伦理风险；论文已提示训练语料可能带有隐私、许可和代表性偏差，但本研究不涉及人类受试者或上线流量。

**LLM 使用披露：** 本审稿由隔离运行的 AI 子代理 R1 独立完成，仅用于内部投稿前质量控制；未与任何其他评审者或评审代理通信，也未查看其输出。

**评审限制：**

- 仅审阅指定 SHA256 的 10 页冻结 PDF；未读取源码、匿名工件、训练日志或任何其他评审输出。
- 按隔离任务约束未联网，未独立核验外部引文、相关工作覆盖完整性或匿名工件可访问性；新颖性判断仅相对于 PDF 自述的相关工作。
- 逐页视觉核查未发现裁切或无法解析页面。

#### R2（技术正确性）完整评议

**论文概述：** 论文比较同一 SmolLM2-135M masked-denoising 架构的 AR-pretrained initialization 与一个 fixed random initialization。四个 data-order/training seeds、30.0M tokens 后，random−AR 的 last-logged loss gap 均约 2.24–2.29 nats；另有三个 fresh random draws 的 loss gap 仍为正。生成只使用 seed-42 checkpoint pair、一个提示、两个温度和每 cell 32 个样本，两臂均退化：AR-init 多为短标点残片，random-init 多为长单词循环。不同 teacher、length cap 和温度可改变甚至反转 PPL 排序；MAUVE 未记录，drift recomputation gate 失败。

**最强的已核实贡献：** 第 5–6 页表 4–7 最可靠地提供了一个透明的负控档案：在固定两组 initialization checkpoints 下，四个 data-order runs 的 random−AR step-450 loss 差均为正；同时，同一 prompt 的冻结输出显示 teacher PPL 与 emitted length/repetition 可给出不同读法，且 PPL 排序对 teacher、cap 和 temperature 敏感。该证据只支持配置级工程诊断，不支持质量或总体初始化效应。

**维度理由：**

- Soundness：主要 arms 的同预算实现和每个 endpoint 均写得清楚，作者也没有把退化输出解释为质量。但 n=4 paired-t 区间建立在无法检验的正态假设上，稳健的同号检验仍未达 0.05；三次随机初始化的 percentile bootstrap 下界也不能支撑总体随机权重结论。生成、PPL 与轨迹诊断只有一个 prompt/checkpoint pair，且 drift 重算失败。
- Presentation：10 页稿件结构紧凑，表 2、4–11 将配置、数值与限制配对，图表清晰无版面缺陷。对 MAUVE 未记录、PPL teacher/cap 反号、无质量锚点和 ambient initialization seed 丢失均有显著披露。
- Contribution：将 emission length、alphabetic yield 和 concentration 与 PPL 并列是合理工程记录习惯，但论文明确既不提出新 metric、训练方法或 decoder，也没有质量结论。预训练权重在极短训练预算下胜过从零初始化以及两个退化输出表面不同，科学新颖性和影响均很低。

**优点：**

- 清楚区分 data-order repeats 与 random-weight draws，没有把四个重复误称为权重初始化方差。
- 报告 paired-t 区间同时给出最小可达 one-sided sign-test p=0.0625，使小样本边界可见。
- 没有从未记录的 MAUVE 值构造比率，也明确说明两个退化 arms 均不能作为质量优胜者。
- 对 teacher dependence、length construction、单 prompt、短训练 horizon、failed drift gate 和缺失 ambient seed 均有直接限制。

**问题与可验证修复：**

##### A10-R2-01 · 主要 · 技术正确性、实验严谨性

- 位置：第 1 页摘要与贡献 1；第 5 页图 2、表 4与 §4.1
- 观察证据：四个差值为 2.2814、2.2893、2.2394、2.2391，paired-t 区间极窄为 [2.2196,2.3050]；但 n=4 无法可靠检验差值正态性，精确 one-sided sign test 的最小 p 仍为 0.0625。四次还共享同一对固定 initialization checkpoints。
- 重要性：t 区间的精度主要反映这一个固定权重对下 data-order 差值很稳定，不能支持更广的重复性或初始化效应。以它作为 headline CI 容易掩盖有效独立单位极少、稳健检验未过阈值。
- 必需修复：将四次结果表述为描述性 fixed-pair consistency，并以 sign/randomization interval 为主；增加足够多、预先设定的 paired runs，使 robust paired inference 有可用分辨率。
- 验证标准：预注册 seeds 和主统计量；报告 sign/permutation、t 与 bootstrap 敏感性，并证明结论不依赖正态假设或单个 seed。
- 仍需证据：更多独立 paired runs 与稳健小样本推断。
- 预期影响：high；判断置信度：high。

##### A10-R2-02 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第 3–4 页 §3.2与表 3；第 8–9 页 §5–6、表 11
- 观察证据：仅三个 fresh random draws，gap 为 2.300、2.281、2.272；论文用 percentile bootstrap 给出下界 2.272>0。三点 bootstrap 的尾部基本退化为观测极值，无法评估权重分布尾部；三次同号的 one-sided sign p 为 0.125。
- 重要性：该结果不能以 95% 置信度推广到随机初始化总体，也不能说明 emission mode 稳定；事实上两次 loop、一次 gray 已显示 draw dependence。
- 必需修复：撤下 population-CI 语言，把三点列为 pilot；用足够多独立 weight seeds、独立 data orders 和层级模型/随机化检验分离 draw 与 run 方差。
- 验证标准：交叉 weight_seed×data_seed 设计，在 draw 级留一法和符号检验下方向仍稳定，并报告 draw-level emission mode 分布。
- 仍需证据：更大多初始化、交叉随机效应实验。
- 预期影响：high；判断置信度：high。

##### A10-R2-03 · 主要 · 重要性、实验严谨性、技术正确性

- 位置：第 3–8 页 §3.3–5；表 5–10
- 观察证据：生成证据来自一个 prompt、seed-42 checkpoint pair、每 cell 32 continuations；无人工或独立质量 anchor。两臂都退化，PPL 在 teacher/cap/temperature 下改变或反号。drift scalar/top-k totals 又相差 12.6%–18.1%，没有误差界。
- 重要性：输出长度与词频当然能区分短 stub 和重复 loop，但这不验证它们能诊断质量、训练状态或一般 degeneration，也不建立相对 PPL 的增量效度。轨迹机制因 failed gate 不能解释。
- 必需修复：在多 prompts、checkpoints、weight draws 与至少一个人工/任务质量锚点上预先定义退化标签，评价各字段的 discriminative/增量效度；修复 drift 记录后独立复跑。
- 验证标准：按 prompt 和 checkpoint 留出测试，比较 PPL-only 与 PPL+E 对 blind degeneration labels 的性能；drift 重算必须通过预定容差。
- 仍需证据：多样生成样本、独立标签与修复后的 trajectory records。
- 预期影响：high；判断置信度：high。

##### A10-R2-04 · 主要 · 可复现性、实验严谨性

- 位置：第 4 页表 2；第 9 页 Reproducibility Statement
- 观察证据：principal random weights 由 host ambient RNG 产生且 seed 未记录；step 0 也未记录。文稿没有 verified external anonymous URL，只列 checkpoint SHA 和本地匿名 bundle 信息，未明确保证权重 bytes 可取得。
- 重要性：若 checkpoint 不随包发布，第三方无法重建这个唯一 fixed draw，标题中的 reproducible 只适用于派生表格而非核心训练条件。SHA 能验证已有 bytes，不能从零恢复 bytes。
- 必需修复：发布匿名 checkpoint 或精确 RNG state/初始化脚本，使权重逐 tensor 重建；提供外部可访问 artifact 与端到端验证命令。
- 验证标准：干净环境重建 principal draw，所有 272 tensors 的 digest 与给定 checkpoint 相同，并复现 step-10/450 endpoints。
- 仍需证据：checkpoint bytes 或完整 RNG state、公开 artifact 和第三方重放。
- 预期影响：high；判断置信度：high。

##### A10-R2-05 · 主要 · 新颖性、重要性、限制与负责任表述

- 位置：第 1–3 页 §1–2；第 8 页 §5–7
- 观察证据：AR-init 继承已预训练 SmolLM2 权重，random-init 从零开始，两者只训练 30M tokens/458 steps，loss 仍在变化；作者明确不提出新训练发现、metric、decoder、质量排序或跨床结论。
- 重要性：预训练初始化在极短预算下低于随机初始化是预期负控，而不是足以改变领域认识的发现。剩余 deliverable 是把常规表面统计量并列报告，尚无采用或性能证据。
- 必需修复：把工作定位为 artifact/engineering note，或提出并验证一个可迁移、可证伪的问题，例如不同预训练适配方案在匹配收敛预算和多质量锚点下的系统比较。
- 验证标准：在 matched compute-to-convergence、多模型/规模和独立任务上验证新的主假设，并相对最近 AR-to-diffusion baselines 量化增量。
- 仍需证据：具有外部效度的训练与评估实验。
- 预期影响：high；判断置信度：high。

**给作者的问题：**

- 在 n=4 且差值分布无法检验时，为何把 paired-t 95% CI 作为主要重复性证据，而不是把同号但 p=0.0625 作为主结论？
- 三个 fresh random draws 的 bootstrap lower bound 等于观测最小值；是否计算过精确同号检验或展示该区间对 population mean 的实际覆盖？
- 若两臂本来都被判定退化且无独立质量标签，单 prompt 的 emission fields 除描述输出外如何验证任何 metric complementarity 或诊断效用？
- principal random-init checkpoint 的 ambient RNG seed 丢失；匿名 bundle 是否实际包含该 checkpoint bytes，还是只有 SHA 与派生输出？
- AR-init 已接受大规模预训练、random-init 从零开始，却只训练 30M tokens；这一几乎必然的 short-horizon loss gap 对 dLLM 研究给出了什么可证伪的新结论？

**能提高评分的证据：**

- 扩大并交叉 weight-seed/data-seed 设计，用稳健推断确认 loss 结论。
- 在多 prompt/checkpoint/model 上引入独立 degeneration/quality labels，验证 emission fields 的增量效度。
- 发布可重建的 initialization checkpoint/RNG state，并修复 drift recomputation。

**会降低评分的证据：**

- 更多 weight draws 使 loss gap 或 emission pattern 不稳定。
- 独立质量标签显示新增 emission fields 不提供可用诊断信息。
- principal checkpoint 无法发布或重建，导致核心负控不可复现。

**伦理标记：** 否。没有人类受试者或私有用户数据；FineWeb-Edu 的许可、隐私与代表性风险在文中简要披露。

**LLM 使用披露：** 本审稿由隔离的 AI 子代理 R2 生成，仅用于内部投稿前质量控制；未与其他评审通信，未读取其他评审目录、作者计划、编辑上下文、旧稿或历史评分。

**评审限制：**

- 遵循隔离要求未联网，未核验外部引文、软件包行为、模型 checkpoint 或数据许可。
- 仅审阅冻结 PDF；未读取或执行匿名 bundle、checkpoints、scripts、trajectory dumps 或 preregistration。
- 未用外部资料判断 MAUVE 的实现细节；评审只依据稿件明确报告的未记录状态。
- PDF 共 10 页，已全文阅读并逐页视觉核查；无缺页、裁切或不可辨认图表。

#### R3（实验严谨性）完整评议

**论文概述：** 本文将 SmolLM2-135M 转换为 masked denoiser，对比保留 AR 预训练权重与固定随机初始化，在每臂 30M token/458 步的匹配预算下训练。四个配对数据顺序/训练随机种子上，随机初始化末次记录 loss 比 AR-init 平均高 2.262 nats（配对 t 区间 [2.220, 2.305]）；三个额外随机权重抽样也保持正 gap。生成侧只在“To be or not to be”单一提示和 seed-42 checkpoint pair 上每格采样 32 次，两个 arm 都退化，MAUVE 未记录，PPL 排序与 loss 排序相反。论文将这些结果定位为负对照报告试点，而非质量或机制结论。

**最强的已核实贡献：** 第 4--5 页表 3--4 的匹配预算训练结果验证了一个非常局部但清楚的事实：在固定训练配方下，AR 初始化相对随机初始化的末次记录 denoising loss 优势在四个数据顺序重复中方向一致，并在三个额外随机权重抽样中保持正方向。

**维度理由：**

- Soundness：在“固定随机权重抽样、相同训练预算”的狭窄问题上，四个数据顺序重复和三个额外初始化均给出一致的 loss-gap 方向，且作者较诚实地区分训练损失、PPL、MAUVE 与文本质量。但 n=4 的配对区间不覆盖权重抽样，n=3 的多初始化控制共用 seed-42 数据顺序；生成端只有一个 prompt、一个 checkpoint pair，两个 arm 均明显退化且无独立质量标签。漂移重算门还以 12.6%--18.1%（局部最高 56.8%）差异失败，故多数信号诊断不能支撑稳定机制结论。
- Presentation：论文的负结果边界、配置表、逐项证据图和失败门披露清晰，10 页逐页视觉检查无排版缺陷。不过主线被 PPL、argmax concentration、entropy、drift、MAUVE 缺失和多种校准诊断分散，读者容易高估这些描述量之间的因果联系。
- Contribution：把负对照作为报告单元测试，并完整展示一个 135M dLLM 的退化输出，是有用的工程记录；但它不是新方法或新指标，核心生成证据只有一个提示且没有质量锚，作者也明确不作能力/质量推断。以 ICLR 标准看，当前增量与普适性明显不足。

**优点：**

- 两主臂训练 token 数、步数与大部分配置匹配，并逐个报告四个 paired endpoint，而非只给均值。
- 第 3--4、9 页明确说明四个 paired seeds 不改变随机权重抽样，避免将其区间误报为初始化总体推断。
- 主动报告两个 arm 均为退化文本、PPL 排序反转、MAUVE 未记录，并拒绝据此给出质量排名。
- 第 7--8 页保留失败的 drift recomputation gate，而没有把失败字段继续包装成漂移效应。
- 提供 checkpoint/脚本/结果哈希与 claim-to-artifact map；逐页视觉核查显示全部 10 页完整可读。

**问题与可验证修复：**

##### A10-I1 · 致命 · 实验严谨性、重要性、技术正确性

- 位置：第 1--2 页摘要/引言与图 1；第 3--4 页表 2；第 6 页表 5--6；第 9 页表 11
- 观察证据：生成评估仅使用一个提示、一个 seed-42 checkpoint pair、每 arm/temperature 32 个 continuation。AR-init 主要产生标点/换行残片，random-init 主要产生单词循环；两者都退化。没有人评、任务正确率或其他独立质量锚，MAUVE 也未成功记录。
- 重要性：当所有比较对象均失败且没有外部目标时，PPL、长度、argmax concentration 或 entropy 的差异无法解释为更好的生成、可用诊断或普适负对照属性。单提示还完全无法估计 prompt population 变异。
- 必需修复：在预先定义的多提示集合、至少两个任务/模型规模和多个 checkpoint/生成种子上评估；加入与研究问题匹配的独立质量锚（盲人评或任务标签），并预先说明每个内部信号预测的外部结果。
- 验证标准：对提示与生成种子分层重采样，报告每个信号对独立质量标签的校准/区分性能及跨床异质性；不能只在退化样本中以 arm identity 作为替代标签。
- 仍需证据：完整提示清单、逐样本输出、盲标协议或任务标签、独立种子和分层统计结果。
- 预期影响：high；判断置信度：high。

##### A10-I2 · 主要 · 实验严谨性、技术正确性

- 位置：第 1 页结果预览；第 3--5 页表 3--4；第 9 页表 11
- 观察证据：四个 paired seeds 只改变数据顺序和训练随机性，随机权重 draw 固定；配对 t 区间基于 n=4，方向 sign test 的最小双侧 p 为 0.0625。额外三个随机初始化虽方向一致，但都用同一 seed-42 数据顺序，并未形成权重×数据顺序的独立重复设计。
- 重要性：窄区间 [2.220, 2.305] 只反映固定权重 draw 下的数据顺序变异，容易被误读为初始化总体的精度。n=3 方向检查不足以估计初始化方差或交互。
- 必需修复：采用预先功效分析的交叉/嵌套重复：多个独立随机权重 draw、多个数据顺序与训练种子，并为 AR-init 提供相同层级的重复；用分层模型或按独立 draw 聚类的区间报告方差分量。
- 验证标准：主要 loss gap 的区间应在权重 draw 和数据顺序两个层级都重采样后仍远离零，并报告 draw×order 交互及异常 draw。
- 仍需证据：每个 weight seed × data-order seed 的 endpoint、预定样本量与分层分析代码。
- 预期影响：high；判断置信度：high。

##### A10-I3 · 主要 · 可复现性、实验严谨性

- 位置：第 4 页表 2；第 9 页 Reproducibility and release
- 观察证据：主 random-init 权重由 host ambient RNG 产生且未记录 seed；step 0 也未记录。论文给出 checkpoint 哈希，但冻结时没有已验证的外部匿名 URL。
- 重要性：该权重 draw 是主要处理变量。若 checkpoint 不可获得，其他研究者无法构造同一处理；即使可获得，也不能验证从初始化代码到权重的确定性生成链。
- 必需修复：发布主 checkpoint，并记录生成权重的 RNG 算法、所有库版本、seed、初始化例程与初始化后哈希；补充从空环境重建的端到端清单。
- 验证标准：第三方按所给 seed/环境生成的 step-0 state dict 应与发布哈希逐字节一致，并能复现至少一个训练 endpoint。
- 仍需证据：可下载 checkpoint、weight seed、初始化代码、环境锁文件和逐阶段哈希。
- 预期影响：high；判断置信度：high。

##### A10-I4 · 主要 · 技术正确性、可复现性、限制与负责任表述

- 位置：第 7--8 页第 4.3 节与表 9
- 观察证据：drift 的 scalar/top-k 重算门要求 1% 容差，但实际总量相差 12.6%--18.1%，random-init 个别单元达到 56.8%。作者因此不报告漂移效应，只保留两个序列化一致性检查。
- 重要性：失败幅度显示 drift 字段定义、索引或聚合链可能有系统性错误；任何把 entropy/argmax 轨迹解释成低置信层导致漂移的叙述均没有可审计数值基础。
- 必需修复：定位差异来源，以逐 token/step 的手工小例验证定义，修复后从原始 logits/ids 重新计算并在独立实现间达到预定容差；在此之前删除所有机制暗示。
- 验证标准：预注册的随机样本与边界案例上，两个独立实现的 drift count/率均在 1% 以内且逐事件差异可解释。
- 仍需证据：原始事件字段、重算代码、最小 worked example、逐事件 diff 和修复后 gate 报告。
- 预期影响：medium；判断置信度：high。

##### A10-I5 · 主要 · 实验严谨性、技术正确性、清晰度

- 位置：第 5、7--8 页第 3.3/4.4 节、表 10
- 观察证据：AUROC 以当前 argmax 是否不同于最终 token 为标签，32 条 trajectory 产生 260,064 个高度相关的 masked 行；区间按 trajectory bootstrap。所谓 calibration multiple 是 real/null95 的置换比值，不是 AUROC 或标准显著性量，random-init 原始 AUROC 约 0.458。
- 重要性：单 prompt、32 trajectory 无法支持 prompt-level 外推，且逐行大样本会给出看似很窄的区间。低于 0.5 的 orientation 也没有由独立任务效用验证，非标准比值很难解释。
- 必需修复：把 trajectory/prompt 设为明确抽样单位，在多提示多种子上重复；预先定义方向与假设检验，报告标准置换 p 值/效应，并用 held-out 外部结果验证 entropy 的用途。
- 验证标准：按 prompt 与 trajectory 两层 bootstrap 后效应方向稳定，且 held-out 质量/错误标签上的预测性能超过预定基线；置换程序的 I 类错误用模拟校准。
- 仍需证据：多提示逐 trajectory 数据、预注册统计量、置换零分布和 held-out 验证结果。
- 预期影响：medium；判断置信度：high。

**给作者的问题：**

- 为什么多初始化控制的 seeds 11/22/33 均使用同一个 seed-42 数据顺序，并与同一 AR-init 参照比较？若同时独立抽样数据顺序、训练随机性和 AR 参照，gap 的方差是多少？
- 在两个 arm 都明显退化的情况下，单提示 PPL、argmax concentration 和 masked entropy 的科学目标是什么；它们能预测哪一个由人类或任务定义的外部结果？
- 表 10 的 AUROC 使用 260,064 个 masked 行但只有 32 条 trajectory。不同 position/step 的强依赖是否会使 trajectory bootstrap 在单提示下仍低估 prompt-level 不确定性？
- 能否提供随机初始化 checkpoint 的可访问资产或确定性 weight seed，使主实验不必依赖当前匿名工件中一个不可重建的 ambient-RNG draw？
- 漂移字段的标量/top-k 重算为何会出现最高 56.8% 的偏差，是否存在 dtype、mask indexing 或 aggregation 定义不一致？

**能提高评分的证据：**

- 多随机权重 draw × 多数据顺序的分层重复仍给出稳定 loss gap，并正确反映初始化总体不确定性。
- 在预先定义的多提示、多任务或多模型床上，内部信号对独立质量/任务标签有可复现的预测价值。
- 确定性重建主随机初始化 checkpoint，并发布可复算的完整训练与生成资产。
- 修复 drift gate 后，两个独立实现逐事件一致且新的机制性分析通过预定验证。

**会降低评分的证据：**

- 增加随机权重 draw 后 loss gap 高度异质、出现频繁反向或分层区间跨零。
- 在多提示上 PPL、entropy 或 argmax concentration 与独立质量标签无关或方向不稳定。
- 主 checkpoint/训练 endpoint 不能由记录配置复现。
- 漂移重算差异源于更广泛的事件流索引或序列化错误，并影响其他表格。

**伦理标记：** 否。未涉及人类受试者或私人数据。潜在责任风险在于读者可能把内部置信信号、PPL 或单提示退化差异误作生成质量指标；论文已明确警告不应作该推断。

**LLM 使用披露：** 本审稿由隔离运行的 AI 子代理 R3 完成，仅用于内部投稿前质量控制；未与其他评审通信，未读取作者计划、旧评分或其他评审输出。

**评审限制：**

- 仅审阅指定 SHA256 的冻结 PDF；未访问代码、匿名工件、checkpoint、训练日志或原始生成，无法复算数值或验证哈希资产。
- 按隔离要求未联网，未核验外部引文、模型/数据许可证以及论文声称的资产是否实际可访问。
- 已全文阅读并逐页视觉核查全部 10 页；页面完整可读，未见裁切、乱码或图表渲染异常。

### A14

#### 交付与来源状态

- 选定输入：`/Users/liuhanzuo/Downloads/A14-paper (1).pdf`
- 源状态：`exact_latex_snapshot`
- 润色 PDF：`/Users/liuhanzuo/MacLLM-Bench/output/pdf/paper_polish_20260826/A14-polished.pdf`
- 冻结 SHA-256：`d97f36aebfe3377236c0acc84ee88257d4b704e0d298ad74e88ccb435793821f`
- 总页数：19；主文状态：主文在参考文献前不超过9页。
- 版面核验：pass；构建：pass_with_underfull_warnings。
- 旧评分基线：6,6,4；旧中位数：6。

#### 三评量化结果

| 审稿人 | 角色 | Overall | Confidence | 建议 | Soundness | Presentation | Contribution | 当前上限 | 必需修改后预测 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| R1 | 新颖性与定位 | 4 | 4 | 略低于接收线 | 3 | 3 | 2 | 4 | 6 |
| R2 | 技术正确性 | 6 | 4 | 略高于接收线 | 3 | 3 | 2 | 6 | 8 |
| R3 | 实验严谨性 | 6 | 4 | 略高于接收线 | 3 | 3 | 2 | 6 | 8 |

三评中位数为 **6**，均值 5.33，跨度 2，接收侧票数 2/3。

#### 编辑记录

- [结构审计](work/A14/structure_audit.md)
- [语义锁](work/A14/semantic_lock.md)
- [修订日志](work/A14/revision_log.md)
- [待核验事项](work/A14/needs_verification.md)

**修订日志原文：**

> # A14 Revision Log
>
> ## Structure and claims
>
> - Split the dense introduction claim paragraph into three pool- and endpoint-specific contributions.
> - Rewrote the conclusion to close the same three claims: tested-budget persistence, fixed-order boundary, and bed-local protocol non-interchangeability.
> - Retained every compute-only, causal, multi-seed, and generalization refusal.
> - Moved the duplicated three-pool figure and endpoint decision tree, plus the detailed fresh-strata table, to the appendix; main-text prose retains the load-bearing results and points to the full evidence.
>
> ## Internal consistency
>
> - Repaired the main-text flip-rate cross-reference to point to the reconstructed appendix table.
> - Renamed three duplicated appendix labels (`fig:forest-app`, `fig:tau-app`, `tab:endpoint-tree-app`) and synchronized their internal references.
>
> ## Integrity checks
>
> - Scientific values were preserved across the main text and appendix; repeated values were removed from the main text when their full tables moved to the appendix.
> - Final build has no undefined citation/reference, duplicate-label, fatal LaTeX, or overfull-box diagnostic, and the conclusion ends on main-text page 9.

**待核验事项原文：**

> # A14 Needs Verification
>
> - The exact PDF-matched `main.tex` and `refs.bib` were preserved, but the appendix file used to build that PDF was not stored in the snapshot. The final build uses the nearest conservative pre-snapshot appendix (`backups/r479_refine6/appendix_refine5.tex`) plus verified referenced assets. This is a reconstruction, not an exact-source claim.
> - The reconstructed final PDF is 19 pages, matching the supplied PDF's total page count, but appendix pagination and layout are not claimed to be source-identical.
> - The conclusion ends on main-text page 9; the final build satisfies the nominal ICLR nine-page main-text allowance.
> - Off-curve budgets, multi-seed behavior, causal factorization, and cross-bed transfer remain unmeasured.

#### R1（新颖性与定位）完整评议

**论文概述：** 本文在一个 LLaDA-8B-Instruct、四选一 GSM8K-MC 评测床上比较五种 scoring/reveal 协议：置信度顺序注入、随机顺序注入、反置信度、固定顺序以及无反馈的随机 mask reference。n=400 的参考预算曲线显示 s_conf 相对 mc_uniform 从单 draw 到高预算均有正的配对准确率差；最关键的总 NFE 对齐 2L/2L 点仍为 +13.0 pp，95% CI [6.75,19.25]。另一个不重叠 n=819 fresh pool 上，s_conf 胜一个固定 seeded-random 日程 +12.58 pp，但 fixed order 又平均胜 s_conf 5.62 pp，且效果随选项构造和答案位置反转。论文据此主张这些 scoring protocols 在该床上不可互换，但不主张置信度排序唯一、必要或因果最优。

**最强的已核实贡献：** 最强证据是 n=400 同题配对的 reference-budget 曲线：在 s_conf 与 mc_uniform 都消耗每候选 2L 次总 NFE 的精确对齐点，s_conf 仍高 13.0 pp，配对区间 [6.75,19.25]、exact p=6.6×10^-5（PDF pp.4–6，Figure 2、Table 3）。这有力排除了“差异完全来自 reference 计算次数更少”的单一解释，但仍是完整协议替换的 bed-local 效应。

**维度理由：**

- Soundness：同题配对、精确 McNemar 检验、Holm 家族、问题 bootstrap 以及精确 2L/2L NFE 对照使“在该床上协议不可互换”得到较好支持。主要缺口是对照替换同时改变反馈、路径与预算结构，fresh reveal-order 结果又依赖单个随机日程并在观察异质性后升格，因此不能识别置信度反馈或揭示顺序的独立因果作用。
- Presentation：19 页 PDF 的主表、分析地图、决策历史和限制总体清晰，逐页视觉核查未见裁切；但注册层级、多个池、多个区间约定和 post-selection 状态十分复杂，读者需要在主文与大量附录间往返才能识别哪个结果真正 load-bearing。
- Contribution：论文给出了一个认真且有用的 bed-local scoring-protocol 测量审计，最强结果排除了“reference 仅因较少 nominal draws 而更差”的狭窄解释。但协议依赖本身并非新现象，本文没有新算法、通用估计理论或跨模型规律；相对于 PDF 所列工作，新增点主要是该 LLaDA/GSM8K-MC 实例的配对量化。

**优点：**

- 对 nominal draw match 与 total-NFE match 作了明确区分，并用同一 n=400 题集给出数值不同的 paired record（PDF pp.4–6，§4、Table 3）。
- 确认性五臂比较采用十个 pairwise tests 的 Holm 家族，主表同时报告 b/c、区间和 exact McNemar p，而非只报边际 accuracy。
- 论文诚实承认 s_conf→mc_uniform 是反馈、路径和 passes 的联合替换，不能单独归因；也明确说明 fixed order 能胜 s_conf，因此置信度既非必要也非唯一最优（PDF pp.8–9，§6–§8）。
- fresh pool 的构造层、答案位置反转、池间异质性与 post-selection 升格均有披露，而不是合并成一个跨床常数。

**问题与可验证修复：**

##### A14-I1 · 主要 · 新颖性、重要性

- 位置：PDF pp.2–3，§2 Related Work and Local Evidence Context；PDF pp.9–10，§7–§8
- 观察证据：论文比较的评分、随机 mask、固定/随机/置信度 reveal 和 paired accuracy 工具均来自或紧邻既有实践；新增证据集中在单一 LLaDA-8B/GSM8K-MC 床。结论也只声称 bed-local non-interchangeability，没有提出新 decoder、评分器或一般理论。
- 重要性：协议实现细节会改变结果是重要警示，但仅证明一个自定义多选床上的差异，尚不足以说明该问题在主流评测、其他 dLLM 或实际生成任务中具有广泛影响。
- 必需修复：与最接近的公开 dLLM scoring/reveal 方法做直接同床比较，并在至少另一个模型、任务和选项构造上预注册复现；清楚界定相对于既有协议敏感性研究新增的可迁移命题。
- 验证标准：冻结统一 item/prompt/judge 后，在多个模型与选择题/自由生成任务上重复 factorial protocol matrix，检验效应方向、量级和 rank disagreement 是否保持。
- 仍需证据：强最近基线、跨模型/任务复现和明确的新颖性差异表。
- 预期影响：high；判断置信度：high。

##### A14-I2 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：PDF pp.3–5，§3–§4、Table 1、Table 3；PDF p.9，§7
- 观察证据：即便 2L/2L 对齐总 NFE，s_conf 仍进行 sequential candidate scoring、使用 ground-truth token 注入并以模型置信度选择 schedule；mc_uniform 使用不同 mask-out path、无 feedback。精确 2L 只对齐 forward 次数，没有保持状态路径、反馈或每次 forward 的 token 工作相同。
- 重要性：+13 pp 能证明完整协议不可互换，却不能定位差异来自置信度、答案注入、顺序、路径还是计算结构。若将其解释成 structured reveal 或 confidence 的优势，会超出识别范围。
- 必需修复：构造正交的 factorial ablation：固定 feedback 后只换 order，固定 order 后只换 feedback，固定 mask path 后只换 selection rule；并测量 token-level FLOPs、wall-clock 与内存。
- 验证标准：对每个单因素对照做同题配对区间和预注册多重性控制；只有某因素在其他因素固定时仍产生稳定差异，才可作对应归因。
- 仍需证据：正交消融、实际计算账本和相同路径对照。
- 预期影响：high；判断置信度：high。

##### A14-I3 · 主要 · 实验严谨性、可复现性

- 位置：PDF p.3，§3（seeded-random provenance）；PDF pp.7–9，Tables 5–6、§6；PDF p.12，Appendix A
- 观察证据：s_conf 对 rand_inject 的 fresh 结果来自一个 persisted random schedule，而非 schedule-seed 平均；原 runner 使用未固定的 Python string hash，只能靠保存的逐候选顺序回放。fresh endpoint 是在发现 pilot/fresh 异质性后升格的 post-selection secondary，且固定顺序效果按构造层和答案位置变化并在 position 0 反转。
- 重要性：单个随机日程和观察后选池不能支持“structured reveal 胜随机策略族”的稳定结论；位置反转还提示多选构造可能驱动平均效应。
- 必需修复：预先固定多个跨进程可复现的 schedule seeds，在同一新题池上平均；按预先规定的构造层和答案位置分层/平衡，并把 fresh pool 当作确认性新研究而非事后升格。
- 验证标准：层级模型或 cluster bootstrap 同时覆盖 item 与 schedule seed；预注册主效应在各构造层保持方向，且 position interaction 被明确估计。
- 仍需证据：多 schedule-seed 重放、平衡新池和预注册交互分析。
- 预期影响：high；判断置信度：high。

##### A14-I4 · 主要 · 实验严谨性、重要性

- 位置：PDF pp.5–9，Tables 2–6、§5–§7；PDF pp.13–15，Appendices B–D
- 观察证据：所有主要结果来自一个 checkpoint 和一个四选一 GSM8K-MC 构造；candidate length 不一但未测 length-normalized likelihood。五选项 probe 依注册门判为 UNDETERMINED；reference 曲线也只覆盖几个离散 operating points，不能给出 dose response 或离床预测。
- 重要性：协议排序可能特定于候选长度、近错 distractor 生成与答案位置。若四选一之外不稳定，则“scoring protocols non-interchangeable”的实际适用范围很窄。
- 必需修复：加入长度归一化和校准后的 scoring 基线，扩展到五选项/不同 distractor 机制/不同任务及 checkpoint，并预先规定推广目标。
- 验证标准：在各构造中报告配对效应与 interaction；若主方向跨 option count、length normalization 和 task 保持且异质性可解释，才支持推广。
- 仍需证据：多构造外部验证与长度控制分析。
- 预期影响：high；判断置信度：high。

##### A14-I5 · 次要 · 清晰度

- 位置：PDF pp.4–8，§3–§6 与 Table 5
- 观察证据：文中并列 pre-registered primary、post-selected decisive、pre-registered conservative、post-selection secondary、单对 k=1 和五检验 sensitivity 等多层标签，且使用不同区间约定。虽有 analysis map，主叙事仍容易让读者把确认性和探索性结果视为同等。
- 重要性：复杂的决策历史会增加选择性报告风险，也掩盖唯一可独立解释的核心结果。
- 必需修复：主文只保留一个确认性主 estimand 和少数预先声明的 secondary；将历史 pilot、事后升格和替代区间集中到附录，并用统一术语标注。
- 验证标准：让独立读者仅凭摘要和主结果表即可准确指出主假设、唯一 multiplicity family、主要区间和探索性结果。
- 仍需证据：简化后的分析层级与统一主表。
- 预期影响：medium；判断置信度：high。

**给作者的问题：**

- 能否用 factorial 设计分别操纵 ground-truth feedback、reveal order、schedule-selection pass 和 reference sampling，使协议差异可被因果分解？
- seeded-random 只固定了一个持久化日程；跨多个预先固定 schedule seeds 后，s_conf 对随机顺序的 +12.58 pp 是否仍成立？
- fresh pool 中 fixed order 的优势在答案位置 0 反转，作者如何排除固定位置、选项生成策略或 scorer 的位置偏置？
- 为何没有报告 length-normalized candidate likelihood、实际 FLOPs/latency/memory，以及更接近现有 dLLM multiple-choice scoring 的外部基线？
- 五选项 probe 的未确定结果是否意味着结论依赖四选一构造；需要多大样本才能进行预注册确认？

**能提高评分的证据：**

- 用正交 factorial 对照识别 feedback、order、path 和计算预算各自的作用。
- 在新题池上跨多个固定 schedule seeds 确认 reveal-order 结果，并控制答案位置/选项构造。
- 跨模型、任务、选项数和 length-normalized scoring 复现主要效应。
- 与最接近的公开 dLLM scoring 方法直接比较，并报告实际 FLOPs/延迟。

**会降低评分的证据：**

- 多 seed 随机日程使 +12.58 pp fresh 效应消失或反转。
- 长度归一化或位置平衡后主要协议差异显著缩小。
- 其他模型/任务上协议排序不稳定且无可解释规律。
- factorial 对照显示差异完全由答案注入或构造泄漏驱动，而非所讨论的 scoring 机制。

**伦理标记：** 否。未发现需要升级的伦理问题；研究为公开数学题上的离线模型评测。

**LLM 使用披露：** 本审稿由隔离运行的 AI 子代理 R1 独立完成，仅用于内部投稿前质量控制；未与任何其他评审者或评审代理通信，也未查看其输出。

**评审限制：**

- 仅审阅指定 SHA256 的 19 页冻结 PDF；未读取源文件、逐题记录、匿名工件或任何其他评审输出。
- 按隔离任务约束未联网，未独立核验外部引文、相关工作覆盖完整性或工件可访问性；新颖性判断仅相对于 PDF 自述的相关工作。
- 逐页视觉核查未发现裁切或无法解析页面。

#### R2（技术正确性）完整评议

**论文概述：** 论文在 LLaDA-8B 与四选一 GSM8K-MC 上比较五种 scoring/selection protocol。预注册的 n=400 matched-L 比较中，s_conf 为 171/400，mc_uniform 为 119/400，逐题差值为 +13.0 pp（95% CI [7.0, 19.0]，McNemar b/c=106/54，Holm 校正 p=4.8e-5）。作者又报告 forward-call 数匹配的 2L 对比、高预算 Monte Carlo、不同答案顺序，以及 fresh n=819 池：其中冻结固定顺序比 s_conf 高 5.62 pp，但效果随正确答案位置明显异质。论文据此主张这些协议不可互换、confidence 并非必要，并将 matched-L 结果视为主要测量发现。

**最强的已核实贡献：** 第 4–7 页的预注册 n=400 matched-L 配对结果最为可靠：在相同题目上，s_conf 相对 mc_uniform 提高 13.0 pp，106 个仅 s_conf 正确对 54 个仅 mc_uniform 正确，区间与校正检验均支持非零差异；第 7–10 页的高预算点与 fresh fixed-order 结果进一步证明协议选择确实会改变测得准确率，而不能将这些 scoring readout 当作可互换实现。

**维度理由：**

- Soundness：预注册的 n=400 matched-L 主比较使用逐题配对统计，效应量、discordant counts、置信区间与 Holm 校正均报告充分；在 nmc=128 的高预算点方向也一致。主要不足是把 forward-call 数直接等同于计算成本、在同一数据上突出事后选定的 2L 比较，以及每个随机解码协议仅观察一个随机实现，因此尚不能识别随机协议的期望性能或成本—准确率曲线。
- Presentation：正文清楚区分 matched-L、total-NFE-fair、固定顺序和随机顺序，并对预注册、补充与事后分析多有标注；19 页 PDF 无缺页且图表可读。但结果族、校正族和决策阈值过多，摘要中的‘更少计算’比实际测量的 forward-call proxy 更强，fresh-pool 分层反转也应更靠前解释。
- Contribution：最可信的新证据是同一模型与构造任务上，confidence-guided scoring 相对 draw-count-matched uniform Monte Carlo 的配对优势，以及一个固定顺序在新池上反而优于 confidence schedule。它是有用的窄测量研究，但没有分离反馈、顺序和估计器三项因素，也未跨模型、任务构造或随机种子验证，因而目前不足以形成一般解码原则。

**优点：**

- 主比较保留了逐题配对结构，完整给出分子分母、discordant counts、绝对百分点差、置信区间和多重校正结果。
- 对 matched draws 与 matched forward calls 给出两个不同资源视角，没有把所有比较混成单一准确率排名。
- fresh n=819 题池用于检验固定顺序，且作者披露按正确答案位置分层时方向反转，避免隐藏重要异质性。
- 多数事后分析、随机顺序实现和五选一 probe 均明确标注其探索性或未决定状态，结论总体较克制。

**问题与可验证修复：**

##### A14-R2-01 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第 1 页摘要；第 3–5 页 §2–3；第 11–12 页关于 NFE matching 的讨论
- 观察证据：资源公平性仅按 forward-call 数计数：s_conf 的 L 次打分加 L 次自适应选择被记为 2L，mc_uniform 的 nmc=2L 被记为相同 NFE。稿件没有测量不同调用的 token/canvas 形状、批处理、FLOPs、wall time、显存或硬件利用率，却在摘要中使用‘less compute’读法。
- 重要性：相同调用次数并不保证相同实际成本，尤其当打分、采样和自适应状态更新具有不同张量形状与并行效率时。准确率优势可以成立，但计算效率这一核心解释尚未被观测量支持。
- 必需修复：将现有结论明确限定为 draw-count/forward-call proxy，或在冻结实现和硬件上测量每题 FLOPs、延迟、吞吐、峰值显存与能耗，并给出预算—准确率曲线。
- 验证标准：在相同硬件、batch size、精度和缓存设置下交错运行各协议，重复多个随机次序；比较实测成本相同时的配对准确率，并报告置信区间。
- 仍需证据：端到端成本计量、运行配置与成本匹配后的准确率结果。
- 预期影响：high；判断置信度：high。

##### A14-R2-02 · 主要 · 实验严谨性、技术正确性、清晰度

- 位置：第 5–7 页 §3.2–3.4；第 13–16 页补充检验与 multiplicity 说明
- 观察证据：matched-L 是明确的预注册主比较；2L/2L 对比及其‘decisive’地位在观察同一 n=400 数据后被提升。稿件给出多个预算点、顺序对比、fresh-pool 分析和不同 correction families，但没有证明 2L endpoint、所有报告族及其决策角色在数据揭示前共同冻结。
- 重要性：在同一题集上选择最有说服力的预算与校正族会使名义 p 值过于乐观。它不会抹掉 +13 pp 的预注册主结果，却削弱‘成本匹配后也已确认’这一更强主张。
- 必需修复：把 matched-L 保持为唯一确认性结果，将 2L 与其余预算列为探索性；在独立题池预先冻结预算网格、主 endpoint、方向、检验族和 stopping rule 后复制。
- 验证标准：公开时间戳 preregistration，并在完全未使用的新题池上一次性执行；校正需覆盖所有会影响主结论的预算与 protocol contrasts。
- 仍需证据：独立的预注册 total-cost 复制及完整 multiplicity 账本。
- 预期影响：high；判断置信度：high。

##### A14-R2-03 · 主要 · 技术正确性、实验严谨性、可复现性

- 位置：第 3–4 页协议定义；第 6–9 页 Monte Carlo 与 random-order 结果；第 17–18 页随机性说明
- 观察证据：mc_uniform 及随机顺序协议在每题/条件下仅有一个随机 realization，未对算法随机性重复 seeds。随机 schedule 的原始 seed 又依赖未固定的 Python string hash，当前只能靠已记录的顺序重建；fresh random-order 分析还在看到 pool heterogeneity 后被提升。
- 重要性：一次 realization 估计的是某个随机轨迹，不是随机协议的期望性能；protocol 间的百分点差可能混入 draw/schedule variance。记录顺序保证当前输出可重算，但不能给出换 seed 后的稳定性。
- 必需修复：对相同题目运行多个独立、显式整数种子，分别估计题目随机性与算法随机性；报告 protocol 平均效应、seed 方差及最坏/最好 seed 敏感性。
- 验证标准：以题目和 seed 为交叉层级做配对/层级 bootstrap；主要方向应在预定比例的 seeds 中保持，且区间排除实质性零效应。
- 仍需证据：多 seed 原始轨迹、确定性 seed 生成规则和层级不确定性分析。
- 预期影响：high；判断置信度：high。

##### A14-R2-04 · 主要 · 重要性、实验严谨性、限制与负责任表述

- 位置：第 7–10 页 §4 fresh n=819；第 10–12 页讨论与五选一 probe；第 18–19 页限制
- 观察证据：fresh n=819 上 fixed order 总体比 s_conf 高 5.62 pp，但按正确答案位置分层时位置 0 上 s_conf 反而高 12.56 pp，位置 1–3 则 fixed 更高。研究只用一个模型、一个构造的 GSM8K-MC recipe，且没有以多组 distractor、选项置换或模型复制；五选一 probe 也被登记为 UNDETERMINED。
- 重要性：总体效果可能主要由选项构造与位置先验加权得到，而不是 confidence schedule 的一般缺陷。‘confidence 非必要’在该构造上成立的描述性证据，不能直接外推到其他选择题格式或 masked-diffusion 模型。
- 必需修复：对每题生成多组平衡置换和独立 distractor sets，跨至少另一模型/数据集复制；预先把位置与构造交互设为主分析。
- 验证标准：在题目内平衡所有正确答案位置后，估计 schedule 主效应与 position×schedule 交互；主结论应跨构造 seed、任务和模型保持。
- 仍需证据：平衡置换、多构造 seed 与跨模型/任务复制。
- 预期影响：high；判断置信度：high。

##### A14-R2-05 · 次要 · 清晰度、重要性

- 位置：全文，尤其第 5–12 页结果与第 13–19 页附录
- 观察证据：稿件在 19 页中并列大量预算、顺序、校正族、fresh-pool、五选一及事后敏感性结果；虽然多数有标签，但主证据层级仍需读者跨多节重建。协议同时改变反馈、顺序和 estimator，正文没有一个最小 factorial 图明确哪些因果因素被识别。
- 重要性：证据层级过密会让一个可靠但窄的 paired finding 被更弱的探索性结论稀释，也容易把‘协议不同’误读为已经解释了差异来源。
- 必需修复：正文压缩为预注册主结果、一个预先定义的资源敏感性和一个独立复制；其余移入附录，并用一张 intervention table 标出每个 arm 同时变化的因素与可识别 estimand。
- 验证标准：不依赖附录即可从主文唯一识别 primary endpoint、confirmatory family、探索性结果及每项结论的适用范围。
- 仍需证据：重构后的结果层级与 intervention-factor 表。
- 预期影响：medium；判断置信度：high。

**给作者的问题：**

- 能否提供每种协议实测的 FLOPs、wall-clock、峰值显存与批处理吞吐，而不是只以 forward-call 数宣称计算更少？
- 2L total-NFE-fair 点是在查看 n=400 结果后为何被称为‘decisive’；其预算与检验族是否在观察结果之前冻结？
- mc_uniform 与随机顺序各只有一次随机实现时，观察到的协议差异中有多少来自算法随机性？能否对同一题运行多个独立 schedule/draw seeds？
- fresh pool 中答案位置 0 与位置 1–3 的效应方向为何相反；这是否由 distractor 构造、选项长度或位置先验驱动？
- 如果把反馈、访问顺序与 likelihood estimator 做完整 factorial intervention，哪一项是 matched-L +13 pp 的真正来源？

**能提高评分的证据：**

- 在独立题池预注册并复制 total-cost-matched 主比较，同时提供实测 FLOPs、延迟、显存和吞吐。
- 对随机 protocol 运行多个显式 seeds，以层级分析证明优势不是单一 draw/schedule realization。
- 用平衡选项置换、多个 distractor constructions 及至少另一模型/任务验证 fixed/confidence 结论。

**会降低评分的证据：**

- 实测计算成本匹配后，s_conf 相对 Monte Carlo 的优势消失或反转。
- 多 seed 结果显示协议排序高度不稳定，当前效应由单一随机轨迹驱动。
- 平衡正确答案位置后 fresh fixed-order 总体优势消失，证明结论主要来自选项构造偏差。

**伦理标记：** 否。未发现涉及人类受试者、敏感数据、隐私、安全或其他需要伦理升级的问题。

**LLM 使用披露：** 本审稿由隔离的 AI 子代理 R2 生成，仅用于内部投稿前质量控制；未与其他评审通信，未读取其他评审目录、作者计划、编辑上下文、旧稿或历史评分。

**评审限制：**

- 遵循隔离要求未联网，未核验外部引文、模型说明或相关工作定位。
- 仅审阅冻结 PDF，未读取或执行代码、preregistration、逐题输出或匿名 artifact，因此文中的复现与 hash 声明仅按稿件内容评价。
- PDF 共 19 页，已全文阅读并逐页视觉核查；未发现缺页或不可辨认页面。

#### R3（实验严谨性）完整评议

**论文概述：** 本文在 LLaDA-8B-Instruct 与 balanced four-choice GSM8K-MC 上比较五种评分协议：confidence、seeded-random、anti-confidence、fixed-order 的 ground-truth injection，以及无 feedback 的 random-mask reference。n=400 参考预算曲线中，s_conf 相对 mc_uniform 在单 draw、每候选 draw-matched、total-NFE-fair 2L/2L 和高预算 128-draw 条件下分别领先约 14、13、13、9 个百分点；2L/2L 的 exact McNemar p=6.6×10^-5。独立 n=819 床上，固定左到右顺序反而比 s_conf 高 5.62 个百分点，并表现出构造 strata 与 answer position 交互。论文据此主张 scoring protocols 在该床不可互换，但不作 feedback 机制或跨床推广。

**最强的已核实贡献：** 第 5--6 页图 2、表 3 的 n=400 配对结果最有说服力：在 reference 使用 2L forwards、与 s_conf 的总 NFE 数相等时，s_conf 仍领先 13.0 pp（109/57 discordants，95% CI [6.75, 19.25]，exact p=6.6×10^-5）。因此在这个冻结床上，差异不能仅由 reference 获得更少 forward 次数解释。

**维度理由：**

- Soundness：n=400 参考预算曲线和 disjoint n=819 reveal-order 床均保留逐题配对记录，主要差异使用 exact McNemar 与明确区间，效应较大且作者公开 post-selection、异质性与位置反转。关键 2L/2L 行确实反驳了“差异只因 reference forward 次数更少”这一测试床内解释。但随机 mask/随机 reveal schedule 每题每 cell 只有一次实现，不能量化 protocol RNG；确认性家族跨多轮预算与事后提升的 2L 端点较碎片化；NFE 只计 forward 次数而未测实际成本。
- Presentation：协议组成、预算账本、成对 discordance 与结论边界通过图表表达得较清楚，19 页逐页视觉检查无排版故障。附录充分披露决策树与历史小样本，但正文/附录共有 23 张表，多种 interval convention、promotion gate 和 status 标签增加了认知负担，核心确认性与探索性结果可进一步压缩。
- Contribution：论文提供了一个扎实的测量案例：同一模型和题目在不同打分协议下可得到不同准确率，且简单的 draw-count 匹配并不足以消除差异。该结论对 benchmark 实践有意义，但只覆盖 LLaDA-8B × 人工构造的四选 GSM8K-MC，协议同时改变 schedule、ground-truth feedback 与 estimator，尚不足以支持一般机制或跨床规律。

**优点：**

- 所有主要准确率比较都以同题 paired records 进行，并报告 b/c discordants、exact McNemar、区间和多重性家族。
- 预算曲线包含单 draw、nominal draw-matched、total-NFE-fair 和高预算 conservative endpoint，而不是只挑一个有利 reference。
- n=819 reveal-order 床与 n=400 参考床不重叠，并加入 fixed-order negative control，直接否定 confidence ordering 必然最优。
- 第 8、11--12 页保留池异质性、构造 strata 与 answer-position reversal，避免将总体均值误作统一机制。
- 明确承认完整 protocol swap 同时改变反馈、schedule 与 estimator，不声称因果 feedback、dose-response 或跨模型规律。
- 逐页视觉核查显示 19 页均完整可读，所有图表、公式和密集附表均无裁切或乱码。

**问题与可验证修复：**

##### A14-I1 · 主要 · 实验严谨性、技术正确性、可复现性

- 位置：第 3 页 Setup（seeded-random 描述）；第 8 页 Persisted-schedule sensitivity；第 9 页 Limitations
- 观察证据：每个 item/cell 只保留一次 random mask 或 seeded reveal order。fresh n=819 的 random arm 是一个 persisted schedule realization，作者明确称其不是 seeded-random family 的平均；名义 seed 还曾受未固定 Python string hash 影响，虽然现在可按已存 order replay。
- 重要性：题目配对区间量化 item 变异，但不能分离 protocol RNG 对 accuracy、discordance 和 arm ranking 的贡献。一个有利或不利的 schedule/mask realization 可能放大床内差异。
- 必需修复：对所有随机协议使用多组预先声明的独立 seeds，并为每题保留多次 mask/order；用 item 与 protocol-seed 的交叉层级分析报告均值、方差和 seed 敏感性。
- 验证标准：主要 gap 在 leave-one-seed-out、随机 seed meta-analysis 和 seed×item 分层 bootstrap 下方向稳定，且预定最差 seed 界仍保留实际意义。
- 仍需证据：逐 item×seed 的 mask/order、score vector、prediction 与分层方差结果。
- 预期影响：high；判断置信度：high。

##### A14-I2 · 主要 · 实验严谨性、限制与负责任表述、清晰度

- 位置：第 4 页 Registration and principal-contrast selection；第 6--7 页表 3/5；第 14--15 页表 12--14
- 观察证据：原 primary 是 nominal matched-L；2L/2L total-NFE-fair 行在分析图中标为 post-selection secondary，却被提升为“decisive”。四个 n=400 budget rows 来自分阶段/独立注册的运行并复用同一 s_conf 记录，正文主要说明单配置的 ten-test Holm family，没有给整条追加预算曲线的统一序贯 family。
- 重要性：结果的原始 p 值很小，方向可能稳健，但事后选择最能回答质疑的端点和分阶段扩展曲线会使“confirmatory curve”地位高于预先承诺的证据。
- 必需修复：将 matched-L 明确作为唯一原 primary、2L 等标为假设驱动的后续确认；或者在独立新题池上一次性预注册全部预算点、主要端点、停止规则和跨端点多重性。
- 验证标准：独立冻结池的统一预注册 family 中，total-NFE-fair 和 conservative high-budget 两个预定端点均通过校正阈值且效应超过预定实用门限。
- 仍需证据：各注册时间戳、端点添加顺序、完整 family 决策与独立确认性运行。
- 预期影响：medium；判断置信度：high。

##### A14-I3 · 主要 · 实验严谨性、技术正确性

- 位置：第 2--4 页图 1、表 1 与 Load-bearing estimand；第 17 页表 19
- 观察证据：2L/2L 只匹配 nominal total forward-call count。s_conf 的 L 次 schedule-selection forwards 与 L 次 scoring forwards，和 mc_uniform 的 2L random-mask scoring draws 在 mask pattern、控制流、重用与潜在 kernel/cache 路径上不同；论文未报告 FLOPs、延时或显存。
- 重要性：NFE 是合理但不完整的计算代理。该行可反驳“forward 次数较少”这一窄解释，却不能称为一般 compute-fair，亦不能支持计算效率结论。
- 必需修复：逐 protocol 测量总 FLOPs/有效 token operations、wall-clock、GPU-hours 与峰值显存，并在等实际预算下重跑；将现有措辞限定为 draw/forward-count matching。
- 验证标准：在预定实际成本容差（例如总 FLOPs 或 GPU 时间±5%）下，配对 accuracy gap 仍保持方向与实用大小。
- 仍需证据：profile 日志、硬件/软件版本、每阶段成本分解与等成本结果。
- 预期影响：medium；判断置信度：medium。

##### A14-I4 · 主要 · 实验严谨性、技术正确性、限制与负责任表述

- 位置：第 8--9 页表 6、fresh-pool strata audit 与 Boundary；第 12 页表 8
- 观察证据：fresh pool 上 fixed 左到右比 s_conf 高 5.62 pp，但差异依赖题目构造 strata，并在 answer position 0 变为 s_conf 领先 12.56 pp、在 positions 1--3 反向。position audit 是事后分析，选项构造并非随机化干预。
- 重要性：总体 fixed-order 优势可能由 option position 与 distractor construction 的耦合产生，而非 reveal schedule 的一般属性；当前只能证明这套题面构造下协议结果不同。
- 必需修复：对每题生成多种随机 option permutations，跨 construction strata 平衡，预注册 schedule×position 交互；最好在第二种 MC 构造/任务上重复。
- 验证标准：在对 option permutation 取平均后，fixed/s_conf 差异与交互区间可重复；若主效应消失，应将结论改为 format-specific interaction。
- 仍需证据：逐题多 permutation score、随机化记录、分层交互模型与独立构造床。
- 预期影响：high；判断置信度：high。

##### A14-I5 · 主要 · 技术正确性、重要性、限制与负责任表述

- 位置：第 2--3 页图 1、表 1；第 9 页 Boundary；第 9 页 Limitations
- 观察证据：s_conf、fixed/rand/anti inject 与 mc_uniform 同时改变 reveal schedule、是否注入 ground truth、schedule-selection forwards 和 scoring estimator/aggregation。单一 composite swap 无法识别 feedback、order 或 estimator 的独立效应；所有结果又来自一个 checkpoint 和一个四选 GSM8K-MC 构造。
- 重要性：“协议不可互换”的床内描述成立，但其解释和可迁移性有限，不能指导哪个设计组件在何种条件下导致差异。
- 必需修复：实施 factorial ablation，正交操纵 feedback、reveal order、selection rule 与 estimator；在额外模型、任务和 option 数上重复并报告交互。
- 验证标准：每个组件的主效应/交互在预注册设计中可识别，且至少一个独立床复现 protocol-level 排名变化。
- 仍需证据：factorial arm 结果、功效分析、跨床数据与组件级效应。
- 预期影响：medium；判断置信度：high。

**给作者的问题：**

- mc_uniform 与 rand_inject 的 mask/order 是每题独立随机、由同一全局 RNG stream 顺序产生，还是由 item-id 派生？若重新选择 10--20 组 protocol seeds，主要 accuracy gap 的 seed 间方差是多少？
- 第 7 页表 5 将 total-NFE 2L/2L 标为 post-selection secondary，却在摘要和结论中称其 decisive。它是在看到 matched-L 与其他 budget 结果后才加入的吗；跨四个 n=400 reference endpoints 的整体 family 如何定义？
- 不同 scorer 的一次 forward 是否在 sequence length、cache、mask density、schedule-selection overhead 与 kernel 路径上等成本？能否报告实际 FLOPs、GPU 时间和峰值显存，而不只用 NFE？
- 固定顺序在 fresh pool 总体胜出但在 answer position 0 反向。若对每题随机置换选项位置并跨多种 distractor construction，fixed 与 confidence 的差异是否仍存在？
- 能否通过 factorial protocol 分离 ground-truth feedback、reveal order、选择规则和最终 aggregation，而不是只比较成套 scorer？

**能提高评分的证据：**

- 多 protocol seeds 的交叉重复证明主要 gap 远大于 RNG 方差，并可复现 fixed-order/position 交互。
- 在独立新题池上统一预注册完整预算曲线与 family，total-NFE-fair、高预算端点均通过校正与实用门限。
- 实际 FLOPs/时间匹配后 s_conf-reference gap 仍保持。
- factorial ablation 分离 feedback、order 与 estimator，并在额外模型/任务上复现。

**会降低评分的证据：**

- 更换 mask/order seeds 后 reference gap 或 reveal-order ranking 大幅波动甚至反向。
- 随机化 option positions 后 fixed-order 优势消失，表明结果主要是构造伪影。
- 按实际计算成本匹配后 2L/2L 差异显著缩小或消失。
- 独立预注册曲线未复现，或跨端点校正后只剩事后选择的单点。

**伦理标记：** 否。不涉及人类受试者或私密数据。负责任使用方面，论文正确提醒基准准确率依赖 scorer/protocol；若忽略该依赖，可能对模型能力作错误比较。

**LLM 使用披露：** 本审稿由隔离运行的 AI 子代理 R3 完成，仅用于内部投稿前质量控制；未与其他评审通信，未读取作者计划、旧评分或其他评审输出。

**评审限制：**

- 仅审阅指定 SHA256 的冻结 PDF；未访问匿名 artifact、逐题记录、注册文件或实现，无法复算 exact tests、验证 seeds/哈希或审计预算账本。
- 按隔离要求未联网，未核验外部引文、模型/数据来源及文中 reproduction bundle 的可访问性。
- 已全文阅读并逐页视觉核查全部 19 页；页面、图表、公式和密集附表均完整可读，无裁切或乱码。

### A23

#### 交付与来源状态

- 选定输入：`/Users/liuhanzuo/Downloads/A23-paper.pdf`
- 源状态：`pdf_only_orphan_source_not_preserved`
- 润色 PDF：`/Users/liuhanzuo/MacLLM-Bench/output/pdf/paper_polish_20260826/A23-polished.pdf`
- 冻结 SHA-256：`b564bcaf9926acc12ba3c30bbf4ff09ab48b02dd88130b866bf3ef5eb432e625`
- 总页数：14；主文状态：主文在参考文献前不超过9页。
- 版面核验：pass；构建：pass_with_underfull_warnings。
- 旧评分基线：NA；旧中位数：NA。

#### 三评量化结果

| 审稿人 | 角色 | Overall | Confidence | 建议 | Soundness | Presentation | Contribution | 当前上限 | 必需修改后预测 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| R1 | 新颖性与定位 | 2 | 4 | 拒绝 | 2 | 3 | 2 | 2 | 6 |
| R2 | 技术正确性 | 2 | 4 | 拒绝 | 1 | 2 | 1 | 2 | 4 |
| R3 | 实验严谨性 | 2 | 4 | 拒绝 | 1 | 3 | 2 | 2 | 6 |

三评中位数为 **2**，均值 2.00，跨度 0，接收侧票数 0/3。

#### 编辑记录

- [结构审计](work/A23/structure_audit.md)
- [语义锁](work/A23/semantic_lock.md)
- [修订日志](work/A23/revision_log.md)
- [待核验事项](work/A23/needs_verification.md)

**修订日志原文：**

> # A23 Revision Log
>
> ## Structure and claims
>
> - Rewrote the abstract to separate aligned rank evidence from risk-at-coverage evidence and removed revision-history wording (“review-driven”).
> - Tightened the three-part contribution paragraph while retaining the exploratory status of the aligned AUROC contrast.
> - Reframed the conclusion around the bounded supported result and the specified decisive replication, rather than rebuttal-style wording.
> - Replaced the remaining appendix phrase “review-driven” with the scientifically relevant status: introduced during revision, post-hoc, and not prospectively registered.
>
> ## Integrity checks
>
> - Citation keys, labels, and the complete set and multiplicity of numeric tokens in the reconstruction source are unchanged.
> - Final build has no undefined citation/reference, duplicate-label, fatal LaTeX, or overfull-box diagnostic.

**待核验事项原文：**

> # A23 Needs Verification
>
> - The exact source corresponding to the supplied 15-page PDF was not preserved. The final 14-page PDF is rebuilt from the closest active same-lineage source and must not be treated as a source-faithful reconstruction of every sentence or layout element in the supplied PDF.
> - The supplied PDF remains the authority for any claim-level comparison that differs from the reconstruction. No unverified missing passage was guessed or silently recreated.
> - The decoder's remasking policy, decode seed, and prompt configuration remain unrecorded, so the decode cannot be regenerated; only the frozen-stream analysis is auditable.
> - The conclusion ends on main-text page 9, within the nominal ICLR nine-page main-text boundary in this build.

#### R1（新颖性与定位）完整评议

**论文概述：** 本文在一条冻结的 LLaDA-8B-Instruct/GSM8K 解码事件流上比较三种置信度 readout：位置首次暴露 first、最后修订 commit、以及修订期均值 all。n=974 的探索性事后对齐比较显示 commit 的 AUROC 0.7377、all 为 0.7100，配对差 +0.0278、95% CI [0.0123,0.0438]；但在 coverage 0.2/0.5/0.8 的 risk 差区间均跨零，仅 c=0.8 在事后 ±1 pp 界值下通过等价。早/晚分层 family 无 Holm survivor。论文还披露生成配置不可恢复、原始 extraction-stage labels 缺失，以及当前 canonical rerun 与一个独立 extractor 在人工样本上的高 disagreement。

**最强的已核实贡献：** 最强的可核结论，是在同一冻结事件流、同一 block-mean decode score 和 decode-cluster bootstrap 下，commit 相对 all 的排序能力增加 +0.0278 AUROC，区间 [0.0123,0.0438]，同时三个实际 risk 增量区间均跨零（PDF pp.4–5，Table 5）。这种把 rank improvement 与 selection utility 分开报告的做法是严谨且有启发性的，但由于标签与生成 provenance 问题只能视为探索性。

**维度理由：**

- Soundness：配对 decode-cluster bootstrap、AUROC 与 risk 分离、TOST 和空 Holm 家族均体现了良好统计意识。但 correctness 标签缺少原始 extraction provenance，独立 extractor 在 n=200 上与当前标签有 30% disagreement；同时 remasking policy、generation seed 和 prompt 未记录，无法重新生成中央 event stream。这些缺口使仅 0.0278 的 AUROC 差难以被视为可靠确认结果。
- Presentation：14 页 PDF 的 estimand、gate、决策矩阵和限制写得细致，逐页视觉检查未见图表裁切。缺点是 R1–R5、H1–H6、多个 rerun 编号和事后边界相当繁复，主结论其实比呈现结构简单得多。
- Contribution：比较 first/commit/all 三种 in-decoding confidence 聚合并把 rank 与实际 selective risk 分开，是一个有意义的测量问题；但所有方法组件来自标准 selective prediction 与聚合分析，证据只是一条不可再生成的单模型单任务事件流，而且 practical risk 增量未建立，当前贡献仍过窄。

**优点：**

- 没有把 AUROC 提升自动转换为 risk 改善：PDF p.5 Table 5 并列报告 rank 和三个 coverage anchor，明确 risk CIs 均跨零。
- 推断单位与成对结构交代清楚，使用 decode-cluster bootstrap，并说明共享 lattice observation 不能重复计为独立证据（PDF p.4，§3.2–§3.3）。
- 对 TOST 的解读负责：PDF p.6 Table 8 明确 ±1 pp 界值是在原始效应后选定，较低 coverage 仍为 inconclusive，并给出下一床约 n=1960–2820 的设计目标。
- 关键可恢复性缺口没有隐藏：PDF pp.5–6 §4 明确表示作者自己也无法 re-decode，结论仅条件于现有 event stream。

**问题与可验证修复：**

##### A23-I1 · 致命 · 技术正确性、实验严谨性、可复现性

- 位置：PDF pp.5–6，§4 Experimental Setup and Reproducibility；PDF p.8，Table 10
- 观察证据：冻结工件没有保留原始 extraction-stage labels；当前数字来自在 frozen prediction text 上事后重跑 canonical extractor。last_num 分支占 257/974，另有 8/974 不可解析。n=200 与一个 canonical-independent extractor 的审计出现 30% label disagreement，论文只能把它称为 noise upper bound，不能给出校准金标误差。
- 重要性：AUROC、risk、TOST 和所有 permutation/bootstraps 都以 correctness label 为结果变量。未解决的标签构念差异可能远大于 +0.0278 AUROC 的 readout 差，使排序结论及实际风险结论都可能随 judge 改变。
- 必需修复：对全部或充分分层样本建立盲人工金标，明确解析协议与分歧裁决；用多个合理 judge 重算所有主结果，并将 judge 作为敏感度轴。
- 验证标准：报告每个 judge 相对人工金标的混淆矩阵及置信区间；commit−all AUROC 与 risk 结论需在预先接受的高准确 judge 下保持，且对 last_num 分支稳健。
- 仍需证据：逐项人工金标、解析分支 provenance 和多 judge 重分析。
- 预期影响：high；判断置信度：high。

##### A23-I2 · 主要 · 可复现性、实验严谨性

- 位置：PDF pp.5–6，§4；PDF p.14，Appendix E
- 观察证据：remasking policy 决定 commit/first 构念，却未记录；generation seed 与 prompt template 也缺失，因此无法 re-decode。分析层虽有 hash 和脚本，但 raw event-stream bundle 是否发布仍取决于 data steward。
- 重要性：中央自变量正是由解码事件轨迹构造的，无法重建生成过程意味着外部人员既不能复现实验，也不能检查 readout 是否对 sampler 细节敏感。若事件流也不可得，只能相信汇总表。
- 必需修复：重新运行一个完整记录 decoder 配置、prompt、seed、checkpoint 和 extraction path 的预注册床，并发布匿名事件流及一键重算脚本。
- 验证标准：独立环境从 manifest re-decode，事件 schema 与哈希通过，随后逐项复算 Table 5、Tables 7–10 和所有决策标签。
- 仍需证据：可访问事件流、完整 decoder manifest 和独立重建记录。
- 预期影响：high；判断置信度：high。

##### A23-I3 · 主要 · 实验严谨性、限制与负责任表述

- 位置：PDF pp.4–6，§3.2–§4、Tables 5–8；PDF p.8，§6
- 观察证据：headline commit−all AUROC 是 post-alignment exploratory estimate；三个 risk CIs 均跨零。±1 pp 等价界是在初始效应后、TOST 重算前设定，只有 coverage 0.8 通过，0.2/0.5 不确定。早/晚 family 的 Holm(3) survivor set 为空。
- 重要性：当前数据支持排序差异的探索性信号，却未证明部署相关 coverage 下风险有改善或等价。事后阈值和同一流上的多轮分析不能替代独立确认。
- 必需修复：在新床前固定主 readout 对、coverage、risk 差界值、AUROC/risk 层级和多重性，按 Table 8 的功效目标收集足量样本。
- 验证标准：新床上主 risk endpoint 的 CI 排除零或完全进入预注册等价界，并且 AUROC 方向一致；所有 planned families 按固定校正报告。
- 仍需证据：n≈1960–2820 的预注册独立确认床。
- 预期影响：high；判断置信度：high。

##### A23-I4 · 主要 · 新颖性、重要性

- 位置：PDF pp.6–8，§5 Related Work、Table 11、§6 Limitations
- 观察证据：first/last/mean aggregation、AUROC 和 risk–coverage 均为标准组件；论文的新颖性定位是把 in-decoding revision 聚合连接到 decode-level selective score。实证只有一个模型、GSM8K、一个 temperature、每题一次生成，且没有对 readout 的干预或第二数据集。
- 重要性：单一事件流上的聚合比较难以说明这是 masked-diffusion decoder 的一般构念问题，而非该 sampler、block 配置或 judge 的偶发现象。实际 risk 增量未建立也限制应用意义。
- 必需修复：扩展到多个 remasking 策略、NFE/block、模型和任务，并与常见后验置信度、verbalized confidence、self-consistency/energy 等强基线比较；最好加入会改变决策的风险门控实验。
- 验证标准：在独立床上按预注册层级模型估计 readout×setting 交互，报告 selective risk、calibration 和 AUROC；证明 commit 的优势不只存在于单一配置。
- 仍需证据：跨配置复现、强置信度基线及实际门控效用。
- 预期影响：high；判断置信度：high。

**给作者的问题：**

- 在 n=200 审计中所谓 canonical-independent extractor 的具体规则是什么？30% label disagreement 中有多少可由人工金标判定谁对谁错？
- 能否找回或重新生成记录了 remasking policy、prompt、generation seed 和原始 extraction path 的确认床，而不是继续扩展同一不可恢复流的事后分析？
- 为什么主打 +0.0278 AUROC，而三个 coverage 的 practical risk 改善均未建立；预期使用场景的最小有意义 risk 差究竟是多少？
- commit/all 的差异是否会在不同 remasking policy、NFE、block length、temperature、模型和数据集上保持？
- event-stream bundle 若不能匿名公开，外部审稿人如何复算三种 readout 和所有 paired endpoints？

**能提高评分的证据：**

- 用人工金标解决当前 30% extractor disagreement，并证明主要 AUROC/risk 结论对 judge 稳健。
- 提供完整可恢复的新事件流与 decoder provenance，并由独立环境重建主表。
- 在按功效设计的新床上预注册确认 practical risk endpoint。
- 跨 remasking、NFE/block、模型和任务比较 commit/first/all 及强置信度基线。

**会降低评分的证据：**

- 人工金标显示 last_num 分支含大量错误，重算后 +0.0278 AUROC 差消失或反转。
- 新可恢复床无法复现 commit 优势，或其方向强依赖 remasking policy。
- 事件流完整性、哈希或 paired decode 对齐无法通过独立检查。
- 更强的标准置信度基线在 rank、calibration 和 selective risk 上全面优于三个 readout。

**伦理标记：** 否。未发现需要升级的伦理问题；研究是公开基准上的离线分析，论文也明确部署前需在目标分布上重新校准。

**LLM 使用披露：** 本审稿由隔离运行的 AI 子代理 R1 独立完成，仅用于内部投稿前质量控制；未与任何其他评审者或评审代理通信，也未查看其输出。

**评审限制：**

- 仅审阅指定 SHA256 的 14 页冻结 PDF；未读取源代码、事件流、匿名工件或任何其他评审输出。
- 按隔离任务约束未联网，未独立核验外部引文、相关工作覆盖完整性或工件可访问性；新颖性判断仅相对于 PDF 自述的相关工作。
- 30% 数字按论文所述仅解释为两个 extractor 的标签分歧/噪声上界，而非已确认的真实错误率。
- 逐页视觉核查未发现裁切或无法解析页面。

#### R2（技术正确性）完整评议

**论文概述：** 论文使用一个冻结的 LLaDA-8B GSM8K 解码事件流（n=974，每题一次 decode），比较 commit、first 与 all 三种块级置信读出。探索性的 paired AUROC 分析称 commit 相对 all 提高 0.0278（95% CI [0.0123, 0.0438]）；在 coverage 0.2、0.5、0.8 处，risk 差分别约为 -0.0058、-0.0067、-0.0008，仅 c=0.8 通过事后设定的 ±1 pp equivalence margin。原始 correctness labels 没有保留，作者改用 canonical parser 重建，并在 n=200 manual equivalence audit 中观察到 canonical 与 last-number 标签有 30% 翻转。稿件据此主张不同 in-decoding readout 具有构念不一致，并讨论离线 risk gating。

**最强的已核实贡献：** 第 3–6 页在同一 frozen stream、同一后验标签下做的配对 readout 比较，至少可靠地显示 readout 定义不是无关实现细节：commit-all 的样本 AUROC 差为 +0.0278，且三个 coverage 点的样本 risk 差接近零。这个结论只能解释为该档案与该重建标签下的描述性敏感性，不能证明真实正确性预测或可部署的风险控制。

**维度理由：**

- Soundness：主结果的 correctness labels 无法由原 extraction stage 恢复，只能事后用 canonical parser 重建；n=200 人工 equivalence audit 中 canonical 与 last-number 规则有 30% 标签翻转，远大于主 AUROC 差 0.0278。与此同时，AUROC 对每个 decode 先跨 8 blocks 聚合，risk@coverage 却先逐 block 计算再平均，两者并非同一 aligned score/estimand。标签与 estimand 两项都动摇核心技术结论。
- Presentation：稿件提供 readout 定义、配对 bootstrap、coverage 阈值和大量敏感性分析，14 页视觉版面完整可读，并较诚实地承认 post-hoc、单流和数据受限。但‘aligned’一词掩盖两个主 metric 的聚合层级差异，manual audit 的 30% label flip 没有在摘要与主结论中得到相称权重。
- Contribution：将 commit/first/all 三种置信读出在同一 frozen event stream 上比较，属于有用的诊断性分析；但当前标签不可靠、解码流仅来自单模型单任务一次运行、没有 held-out calibration 或真实 risk-gated intervention，也没有展示可迁移的选择性预测增益。贡献因此停留在不可验证的事后档案分析。

**优点：**

- 使用同一 decode 的配对比较，避免把不同题目或不同生成样本误当成 readout 效果。
- 同时报告 AUROC 与固定 coverage 风险，而非只挑一个最有利的区分指标。
- 披露原始标签未保留、decoder 配置缺失、主 aligned 分析为事后引入，以及匿名数据发布仍受 provenance 审批。
- 对多个 coverage 点给出置信区间，并承认仅 c=0.8 满足事后 equivalence rule，未把其余点写成确定等价。

**问题与可验证修复：**

##### A23-R2-01 · 致命 · 技术正确性、实验严谨性、可复现性

- 位置：第 2–3 页 §2 标签定义；第 5–7 页主结果；第 8–10 页 manual equivalence audit
- 观察证据：原 extraction stage 的 labels 没有保留，全部主分析改用事后 canonical parser 重新标记。n=200 的所谓 canonical-independent audit 实际比较语义等价判断与 last-number heuristic，观察到约 30% 标签翻转；作者又说明该比例只是 upper bound，不能校准全体 n=974。主 AUROC 差仅 0.0278，risk 差约 0–0.7 pp。
- 重要性：标签不确定性远大于待解释效应，且可能与输出形态和置信 readout 系统相关，因此同一噪声标签下的精确配对区间不能证明对真实 correctness 的改进。当前核心结论可能由 parser choice 决定。
- 必需修复：由不知道 readout 值的独立标注者对全部 974 个最终答案建立 adjudicated semantic gold labels，或恢复原冻结标签及其 parser；随后在 gold labels 上重算所有 AUROC、risk、coverage 和等价性结果。
- 验证标准：至少双人盲标加冲突仲裁，报告一致性；比较 canonical、last-number 与 gold 的混淆矩阵，并证明 commit-all 的主效应在 gold label 下方向、大小和区间均保持。
- 仍需证据：全量盲审 gold labels、标注协议、分歧仲裁和完整重算。
- 预期影响：high；判断置信度：high。

##### A23-R2-02 · 主要 · 技术正确性、清晰度、限制与负责任表述

- 位置：第 2 页 Eq. (1)；第 3–6 页 AUROC/risk 结果；第 11 页 Appendix A Algorithm 1 与 §A.2
- 观察证据：Algorithm 1 的 AUROC 先将每个 decode 的 8 个 block readouts 平均成一个 S_d，再在 974 个 decodes 上排序；Eq. (1)/Appendix A.2 的 risk@coverage 则在每个 block 内按 block readout 排序、计算 risk 后再跨 8 blocks 平均。两者改变了分析单位与排序，不能称为同一个 aligned estimator 的两个指标。
- 重要性：不同聚合顺序可产生不同排名和选择集，因此 AUROC 提升与 risk 近等价并不构成同一 gate 的互补证据；其差异可能只是 estimand 不一致。
- 必需修复：先定义实际部署单位：若每题一个 gate，则两类指标都使用相同 decode-level score；若每 block gating，则 AUROC 也在预先定义的 block estimand 上计算，并处理同题相关性。
- 验证标准：对统一 score/analysis unit 同时重算 AUROC、risk-coverage 曲线与 selected sets；所有 bootstrap 必须在 decode 层聚类，且正文逐项核对相同 coverage 下的成员。
- 仍需证据：统一 estimand 的分析代码、选中样本清单与重算区间。
- 预期影响：high；判断置信度：high。

##### A23-R2-03 · 主要 · 可复现性、实验严谨性

- 位置：第 7–9 页 provenance/限制；第 12–14 页 artifact 与 decoder 说明
- 观察证据：稿件无法给出生成该流的 remasking policy、generation seed 与完整 prompt，不能重新 decode；原始 event stream 是否匿名发布仍取决于 data-steward/provenance 批准，目前仅承诺 analysis layer。
- 重要性：没有 event stream 就无法核查 block readout、标签或聚合；没有生成配置也无法判断此流是否典型或复制结论。对单一、不可再生档案的分析难以作为可审计方法证据。
- 必需修复：发布去标识的全部事件、targets、输出、标签和 analysis manifest，并恢复完整 checkpoint/prompt/decoder/seeds；若不能，明确把工作降格为不可复现实例报告。
- 验证标准：独立方从发布包逐项重建三种 readout、974 个 labels 与全部表图；使用冻结配置重新生成一个独立 stream 并复现方向。
- 仍需证据：完整匿名 event-level artifact、生成配置和独立重放。
- 预期影响：high；判断置信度：high。

##### A23-R2-04 · 主要 · 实验严谨性、重要性、限制与负责任表述

- 位置：第 1 页摘要；第 4–8 页主分析与敏感性；第 9–10 页讨论
- 观察证据：aligned paired AUROC、多个诊断和等价性框架均在已有 stream 上事后形成；没有 held-out stream、重复 generation seeds、另一模型或另一任务。全部证据来自同一模型、GSM8K 与一次 decode/题。
- 重要性：在同一档案上选择 readout、聚合和诊断会低估分析选择的不确定性。即使修复标签，也只能说明该流内的描述性差异，不能支持一般构念或稳定 risk ranking。
- 必需修复：冻结标签、score aggregation、coverage grid、equivalence margin 与检验族，在独立 seeds 和未使用的数据上确认，并至少跨一个模型或任务重复。
- 验证标准：确认集在任何分析前密封；主差值的方向与预设实质阈值需跨 seeds 保持，并报告异质性而非只汇总 pooled result。
- 仍需证据：独立预注册复制、多 seed 与跨场景结果。
- 预期影响：high；判断置信度：high。

##### A23-R2-05 · 主要 · 技术正确性、重要性、限制与负责任表述

- 位置：第 5–7 页 risk@coverage/TOST；第 9–10 页 risk-gating 讨论
- 观察证据：研究只在同一数据上离线排序，没有 held-out calibration、阈值迁移、选择性覆盖保证或实际拒答/早停 intervention。±1 pp 等价界是在查看效应后确定，且仅 c=0.8 通过；c=0.2 与 0.5 的区间不满足该界。
- 重要性：AUROC 或样本内 risk 差不能保证部署时 coverage/risk，事后 margin 也不能把未校准 selector 变成风险控制器。标题中的 risk-gated decoding 因而超过已实现证据。
- 必需修复：若保留 risk-gating 主张，应在独立 calibration/test split 上预设可接受 risk 与 coverage，冻结阈值并执行实际 gate；否则将论文限定为 offline readout comparison。
- 验证标准：只用 calibration set 选择阈值，在 untouched test set 报告带有限样本区间的 achieved coverage、risk 与 baseline；重复 seeds 验证稳定性。
- 仍需证据：预注册等价界、held-out calibration/test 与实际 gating 结果。
- 预期影响：high；判断置信度：high。

**给作者的问题：**

- 原 extraction stage 的正确/错误标签为何未保留；能否由原始答案、target 与当时 parser 版本逐项恢复，而不是使用新的 canonical rule？
- n=200 audit 中 30% label flips 的逐项来源是什么，且在人工 gold label 下 commit-all AUROC 与 risk 差分别是多少？
- 为什么 AUROC 使用每个 decode 跨 8 blocks 平均后的 S_d，而 risk@coverage 是每 block 排序后再平均；哪一个才是拟部署的 risk gate？
- decoder remasking policy、prompt、generation seed 与 checkpoint 能否恢复；若不能，如何复现这 974 条 event stream？
- 能否在 held-out stream 上先校准阈值，再实际拒答/早停并检验 coverage 与 risk 保证，而不只做同一数据上的离线排序？

**能提高评分的证据：**

- 对全部样本建立盲审 semantic gold labels，并在统一 decode-level 或 block-level estimand 上重算，主差异仍保持。
- 发布可逐项重建的 event stream 与完整 decoder 配置，并用多个独立 seeds 复制。
- 在 held-out calibration/test 上预注册并实际执行 risk gate，展示可靠的 coverage-risk 控制。

**会降低评分的证据：**

- gold labels 下 commit-all AUROC 差消失、反转或低于预设实质阈值。
- 统一聚合层级后 AUROC 与 risk 结论不再一致。
- 原始事件或生成配置最终不可提供，导致主结果无法独立审计。

**伦理标记：** 否。未发现涉及人类受试者、敏感个人数据或直接安全风险的问题；主要风险是科学标签与可复现性，不构成伦理升级。

**LLM 使用披露：** 本审稿由隔离的 AI 子代理 R2 生成，仅用于内部投稿前质量控制；未与其他评审通信，未读取其他评审目录、作者计划、编辑上下文、旧稿或历史评分。

**评审限制：**

- 遵循隔离要求未联网，未核验外部引文、模型说明或相关工作。
- 仅审阅冻结 PDF，未读取或执行 event stream、代码、manual labels 或 artifact，因此文中重算与 provenance 声明只能按稿件评估。
- PDF 共 14 页，已全文阅读并逐页视觉核查；未发现缺页或不可辨认页面。

#### R3（实验严谨性）完整评议

**论文概述：** 本文在一个冻结的 LLaDA-8B-Instruct × GSM8K 事件流（974 题、每题一次解码）上比较 commit、first 与 all 三种 in-decoding 置信度汇总。探索性配对分析显示 commit 相对 all 的 AUROC 增量为 0.0278（95% CI [0.0123, 0.0438]），但在覆盖率 0.2、0.5、0.8 上的配对风险差均有跨零区间；仅覆盖率 0.8 的 TOST 支持 ±1 个百分点范围内的等价。论文据此主张排序分离不等于风险门控效用提升，并提出需要前瞻性重复验证。该谨慎结论在统计表述上基本诚实，但标签有效性和解码可复现性缺陷直接限制了其可信度。

**最强的已核实贡献：** 第 5--6 页表 5--8 将同一批 decode 上的排序量（AUROC）和实际选择效用（risk@coverage）分开，并用相同 decode 聚类重采样展示：可检测的 AUROC 差异并不自动转化为在预定覆盖率上的已解析风险差异。这个负向测量提醒在本冻结数据上得到直接支持。

**维度理由：**

- Soundness：配对、按 decode 聚类的自助法以及风险/排序量纲的区分总体合理，但载荷最大的正确性标签无法追溯到原始提取流程。第 8 页表 10 报告在 200 条人工等价审计中有 30% 标签翻转，且原始提取阶段标签未保存；这使 AUROC 与 risk@coverage 的共同结果变量缺乏可校准的有效性。解码的 remasking 策略、生成种子和提示模板也未记录，无法重新生成事件流。
- Presentation：论文对探索性与注册性分析、检测与等价性、共享观测与独立证据做了较清楚的区分，表格和图在逐页视觉检查中均可读；但 R1--R5、H2、PL、RES 等多层标签和大量修订后诊断使主结论较难提取，且“aligned”措辞掩盖了 AUROC 与风险分析仍采用不同聚合层级。
- Contribution：冻结事件流上比较三种置信度读出并并列报告排序与选择效用，是有价值的测量案例；然而证据只来自一个模型、一个任务、每题一次解码，主要正结果又是事后对齐的探索性差异。当前结果不足以形成可推广的顶会级方法或经验结论。

**优点：**

- 对同一 decode 的读出采用配对比较，并以 decode 为聚类单位重采样，避免把 token/block 行错误当成独立样本。
- 明确区分显著性检测、80% 功效敏感度标签和 TOST 实用等价结论，没有把跨零区间表述为等价。
- 第 3--5 页对 ties、retained-set、残差协方差和共享观测进行了边界审计，并明确指出若干诊断不是独立证据。
- 第 6、8--9 页主动披露单模型单任务、无重复生成、原始标签与解码配置不可恢复等关键限制。
- 逐页视觉核查显示 14 页均完整、无裁切或乱码，图表与公式可读。

**问题与可验证修复：**

##### A23-I1 · 致命 · 技术正确性、实验严谨性、可复现性

- 位置：第 5 页第 4 节首段；第 8 页表 10；第 9 页第 6 节
- 观察证据：原始提取阶段的正确性标签没有保留，当前数字来自对冻结预测文本的事后 canonical extractor 重跑。作者抽查 200 条并与另一个提取器比较，标签翻转率达 30%，主要由 last_num 分支分歧驱动；作者明确称其只是噪声上界而非校准估计。
- 重要性：所有 AUROC、风险覆盖曲线和 TOST 都以该二元正确性标签为结果变量。30% 的未校准分歧远大于主效应（风险差多在 1 个百分点内、AUROC 差 0.0278），因此方向和决策均可能由标签规则决定。
- 必需修复：在预先定义、与读出分数无关的规则下建立人工金标准（优先覆盖全部 974 条，至少采用概率抽样并双人盲标、分歧仲裁），报告各提取器的混淆矩阵，并以金标准标签重算全部主分析。
- 验证标准：锁定标签协议后，独立复算者应从冻结回答生成同一标签；在金标准标签上重新估计 commit-all 的 AUROC 差、三个覆盖率风险差及 TOST，并报告标签不确定性的敏感度区间。
- 仍需证据：逐题预测文本、各提取器输出、盲标记录与仲裁结果，以及从标签到表 5/8/9 的可执行映射。
- 预期影响：high；判断置信度：high。

##### A23-I2 · 致命 · 可复现性、实验严谨性

- 位置：第 5 页第 4 节；第 9 页 Data and analysis release；第 9 页结论
- 观察证据：论文明确指出 remasking policy、generation seed 和 prompt template 均未记录，因此包括作者在内无人能重建该 decode bed；原始 event-stream bundle 的释放还取决于项目数据管理员决定。
- 重要性：commit/first 的定义直接依赖 remasking 与修订轨迹。只可重复分析现有流、不可重复生成该流，无法排除结果是某个未记录解码配置或随机实现的偶然产物。
- 必需修复：从明确版本的模型与数据重新生成前瞻性床，完整记录并发布 prompt、remasking 算法及参数、随机种子、软件环境、模型/数据/输出哈希和逐步事件流。
- 验证标准：第三方从配置和种子重新执行后，应逐项匹配事件流哈希或在预先声明的随机重复协议下复现方向与效应区间。
- 仍需证据：可执行解码配置、版本锁、模型与数据标识、每题种子、原始事件流及完整性哈希。
- 预期影响：high；判断置信度：high。

##### A23-I3 · 主要 · 实验严谨性、重要性、限制与负责任表述

- 位置：第 1 页摘要；第 8 页第 6 节；第 9 页结论
- 观察证据：证据仅包含一个模型、GSM8K 一个任务、一个温度和每题一次生成；没有跨生成种子、候选样本、任务或模型重复。论文自己估算，1 pp 界限的决定性重复约需 1960--2820 个 decode。
- 重要性：题目抽样自助法只能量化对这组冻结回答的题目不确定性，不能量化解码随机性或跨床外推。当前区间不足以支持一般性的 readout 一致性判断。
- 必需修复：按预先锁定的读出和标签协议，在至少一个新模型/任务及每题多个独立生成种子上重复；样本量应达到文中功效目标，并分解题目与生成随机效应。
- 验证标准：前瞻性重复中同时报告每个床和分层汇总的 AUROC 差、相同聚合层级的 risk 差与 TOST；检查方向、效应异质性和覆盖率依赖。
- 仍需证据：新床的功效分析、独立种子、逐题逐生成记录和预注册分析代码。
- 预期影响：high；判断置信度：high。

##### A23-I4 · 主要 · 技术正确性、清晰度

- 位置：第 2 页式 (1) 与 2.1 节；第 5 页表 5；第 11--12 页附录 A.1--A.2
- 观察证据：AUROC 使用每个 decode 跨八个 block 平均后的单一分数进行排序；risk@coverage 则按 block 分别形成风险后再跨 block 平均。两者虽然共享 decode 与 readout，却没有共享完全相同的排序单位和聚合算子。
- 重要性：聚合层级本身可改变排序、保留集合和风险，因而 AUROC 与 risk 的差异不能只归因于评价量纲不同。“aligned”会让读者误以为除了指标外其余估计目标完全相同。
- 必需修复：增加真正同单位的主对比：用同一 decode 级分数与同一 retained set 同时计算 AUROC 和风险；或明确将现有分析称为部分对齐并把 block 聚合差异列为潜在解释。
- 验证标准：在共享 decode 级排序下重算三个覆盖率的风险差；若结论不变，报告与原表 5 的差值及 retained-set 重叠。
- 仍需证据：每个 decode 的 block 级和汇总 readout、各覆盖率 retained-set 与两种聚合规则的并排结果。
- 预期影响：medium；判断置信度：high。

##### A23-I5 · 主要 · 实验严谨性、限制与负责任表述

- 位置：第 6 页表 8；第 13--14 页附录 C/E
- 观察证据：配对 AUROC 是看到既有分析后的 post-alignment 探索性对比；±1 pp 的 TOST 界限也在原始效应估计之后、重算之前确定。三个覆盖率中只有 0.8 接受等价，0.2 与 0.5 均不确定。
- 重要性：事后确定对比与等价界限会引入选择自由度，且单一高覆盖率等价不能概括整个风险曲线。压力测试和置换稳定性不能代替独立的确认性重复。
- 必需修复：把现有结果明确限定为假设生成；在独立数据上预注册读出、聚合、覆盖率、1 pp 界限、多重性控制和标签协议后复验。
- 验证标准：独立重复需在预定 family 上同时给出校正后的检测与 TOST 决策；只有预定覆盖率全部达到相应门限时才作曲线级结论。
- 仍需证据：带时间戳的预注册、独立冻结床和完整 family-level 结果。
- 预期影响：high；判断置信度：high。

**给作者的问题：**

- 第 8 页所述 200 条人工等价审计中的 30% 标签翻转，分别有多少条会改变 commit/all 的 AUROC 差和三个覆盖率下的风险差？能否提供以双人盲标共识为金标准的完整敏感度分析？
- 为什么将“aligned”用于表 5 的整组对比，而 AUROC 是 decode 级汇总分数排序、risk 却先按 block 计算再跨八个 block 平均？若两者使用完全相同的 decode 级分数与 retained set，结论是否保持？
- 第 14 页表 16 显示配对 AUROC 为 post-alignment；对齐规则是在看到哪些效果量之后确定的，是否存在其他尝试过但未报告的 readout/coverage/aggregation 组合？
- 能否恢复或重新运行至少一个完整解码批次，记录 prompt、remasking policy、generation seed 和软件/模型哈希，并在前瞻性锁定的 1 pp 等价界限下复验？

**能提高评分的证据：**

- 以盲标金标准重算后，commit-all 的 AUROC 与风险结论在预注册分析中保持，且标签敏感度不足以改变决策。
- 可完整重生成的新床记录全部解码元数据，并在每题多种子及至少一个额外模型/任务上复现。
- 使用完全相同 decode 级排序与 retained set 后，排序分离而低覆盖风险未解析的现象仍存在。
- 达到文中功效目标的前瞻性样本在预定覆盖率上给出明确的检测或实用等价结论。

**会降低评分的证据：**

- 金标准标签使 AUROC 增量消失、反向，或使任一风险/TOST 决策改变。
- 恢复或新建解码床后结果对 remasking、prompt 或生成种子高度不稳定。
- 共享 decode 级聚合后所谓排序与效用分离主要由原先的 block 聚合不一致解释。
- 原始事件流或结果表无法由声明的分析代码和哈希资产复算。

**伦理标记：** 否。论文分析公开基准上的模型输出，未涉及人类受试者、私密用户数据或实时部署；当前主要风险是若将单床置信度读出直接部署为拒答门控，会在未校准分布上造成错误安全感，论文已部分承认需目标分布重校准。

**LLM 使用披露：** 本审稿由隔离运行的 AI 子代理 R3 完成，仅用于内部投稿前质量控制；未与其他评审通信，未读取作者计划、旧评分或其他评审输出。

**评审限制：**

- 仅审阅指定 SHA256 的冻结 PDF；未访问代码、事件流、模型、数据或任何作者侧工件，因此不能实际复算表中数值。
- 按隔离要求未联网，未核验外部引文、数据许可、模型卡或文中所称发布资产是否存在及可访问。
- 已全文阅读并逐页视觉核查全部 14 页；页面渲染完整，未见裁切、乱码或不可辨识图表。

### A31

#### 交付与来源状态

- 选定输入：`/Users/liuhanzuo/Downloads/A31-paper (1).pdf`
- 源状态：`exact_latex_snapshot`
- 润色 PDF：`/Users/liuhanzuo/MacLLM-Bench/output/pdf/paper_polish_20260826/A31-polished.pdf`
- 冻结 SHA-256：`090599d59e42d3dbe36d9cf724ddc469f317f9ea9065ab90a58b321bda30aa6d`
- 总页数：18；主文状态：主文在第9页结束。
- 版面核验：pass；构建：pass。
- 旧评分基线：4,4,4；旧中位数：4。

#### 三评量化结果

| 审稿人 | 角色 | Overall | Confidence | 建议 | Soundness | Presentation | Contribution | 当前上限 | 必需修改后预测 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| R1 | 新颖性与定位 | 4 | 4 | 略低于接收线 | 3 | 3 | 2 | 4 | 6 |
| R2 | 技术正确性 | 4 | 4 | 略低于接收线 | 2 | 3 | 2 | 4 | 6 |
| R3 | 实验严谨性 | 4 | 4 | 略低于接收线 | 2 | 3 | 2 | 4 | 6 |

三评中位数为 **4**，均值 4.00，跨度 0，接收侧票数 0/3。

#### 编辑记录

- [结构审计](work/A31/structure_audit.md)
- [语义锁](work/A31/semantic_lock.md)
- [修订日志](work/A31/revision_log.md)
- [待核验事项](work/A31/needs_verification.md)

**修订日志原文：**

> # A31 Revision Log
>
> ## Structure and claims
>
> - Shortened the title to the empirical recorder-scoped result.
> - Reordered and tightened the abstract while preserving the composite-intervention, non-nested-probe, amended-branch, and mechanism-refusal statements.
> - Consolidated the introduction's reporting-rule contribution and removed repeated recorder/mechanism wording.
> - Replaced a redundant main-text closest-work table with the fuller appendix comparison and compressed related work without dropping its cited comparison axes.
>
> ## Integrity checks
>
> - Citation keys, labels, and the complete set and multiplicity of numeric tokens are unchanged.
> - Final build has no undefined citation/reference, duplicate-label, fatal LaTeX, or overfull-box diagnostic; the conclusion ends on main-text page 9.

**待核验事项原文：**

> # A31 Needs Verification
>
> - The conclusion ends on main-text page 9; references begin on page 10, satisfying the nominal ICLR nine-page main-text allowance.
> - A dual-recorder comparison and a fresh in-session L32/NFE48 control remain GPU-gated future work; the present evidence cannot support cross-recorder or model-mechanism attribution.
> - The composite intervention still co-varies block partition and required per-block step allocation; prose cannot isolate those components.

#### R1（新颖性与定位）完整评议

**论文概述：** 本文研究 masked-diffusion 解码中一个 position-wise identity-lock 诊断是否继承 active-block recorder 的几何结构。作者定义每个位置最后一次变化后的 lock time、跨 decode 中位数剖面及相邻差分，再用边界最小值减内部最大值的 min–max separation 选择候选周期。固定模型与总 NFE=48 的 L16、L32、L64 三臂，以及 canvas-96 的 L24/L48 probes，均得到选中周期 P*=配置 block length；L16/L64 上没有检测到固定 32 recorder-visible component。另一个 L32/NFE32 床上，初始置信度与较早 lock 有中等粒度的残差秩关联。论文将结论严格限制为 recorder-output association，而非模型或 sampler 机制。

**最强的已核实贡献：** 最强的实证结果是五个被测臂都由两个不同统计读数选出自身配置的 block length：min–max 的 modal P*=L 在所有臂为 1.0，非极值 rank-sum 的最大候选也均为 L；主三臂的共享问题联合 bootstrap 在 2000/2000 次中同时满足 P16*=16、P32*=32、P64*=64（PDF pp.3–6，§2、§3.1，Tables 2–3、Figure 2）。非嵌套 L24/L48 probes 进一步排除了仅由 canvas-256 边界嵌套造成的固定 32 读法。

**维度理由：**

- Soundness：五个 block-length 臂的观察结果、共享问题 cluster bootstrap、非极值 rank-sum 和非嵌套 canvas-96 probes 共同支持“active-block recorder 的选中周期随 L 变化”这一窄描述。但 partition 与每块 step allocation 联动、没有同轨迹双 recorder，对原始选择性门的审计又发现 10/25 错误候选触发；因此机制和确认性推断仍不足。
- Presentation：论文清楚区分 a-priori reachability、from-events magnitude 与 mechanism，并把 post-treatment amendment、边界基数和零功效 null 写入限制。18 页逐页视觉检查未见裁切。缺点是大量诊断、修订层级和术语使一个直观的仪器工件结论显得过度复杂。
- Contribution：证明 recorder readout 可追踪记录 schedule 而非模型固有机制，是有用的测量警示；L16/L32/L64 加 L24/L48 probes 也比单床观察更扎实。然而 active-block observation window 产生 schedule-aligned 边界本身高度可预期，论文没有分离 recorder、sampler 与 model state，也没有提出新解码方法或跨模型规律，贡献仍属于窄审计。

**优点：**

- 结论范围克制：PDF pp.3–4 和 pp.8–9 明确将干预称为 partition-plus-required-allocation，并反复否认模型内在或 sampler-state 归因。
- 不是只依赖一个极值统计：PDF pp.3–6 同时报告 min–max selection、共享 cluster bootstrap、rank-sum、边界/内部基数和 L32 弱控制臂的 0.8805 firing probability。
- 非嵌套 canvas-96 的 L24/L48 probes 检查 candidate 32，在几何上补充了主 canvas-256 嵌套设计（PDF pp.5–6，Tables 3–4）。
- Appendix A.4（PDF p.13）公开原门 10/25 的错误候选命中和后续修正，而不是隐藏不利 calibration；Appendix A.1 也列出 post-treatment scope amendment。
- 内容关联分析明确来自独立的 L32/NFE32 床，且只声称在 BIN16 以前的中等 null granularity 成立，没有把 BIN32 零功效点解释成下界（PDF p.8 与 p.14）。

**问题与可验证修复：**

##### A31-I1 · 主要 · 新颖性、重要性

- 位置：PDF pp.1–2，Abstract、§1；PDF pp.9–10，§5 Related Work 与 §6 Conclusion
- 观察证据：论文的核心结论是 active-block recorder 的 identity-lock period 随记录 block length 变化，并将可迁移产出表述为一套报告规则。相关工作表显示最接近研究测量不同维度；本文没有新模型、解码算法或机制识别，且 schedule-aligned 读数对一个只观察 active block 的 recorder 有较强先验可预期性。
- 重要性：当前研究很好地告诫不要把仪器几何误读为模型机制，但其科学增量主要是一个实现特定的反例和报告规范，尚未显示该问题会系统性改变更广泛的 dLLM 结论。
- 必需修复：增加主动/被动双 recorder、多个 sampler/模型/任务上的对照，并与已有 position-resolved commit/periodicity 方法直接在同一维度比较；把可迁移的新命题明确写成可证伪形式。
- 验证标准：在不同 recorder 窗口下记录同一轨迹，并跨至少两个模型和 sampler 预注册检验 P* 是否只跟 recorder L、也跟 sampler latent schedule，或两者交互。
- 仍需证据：跨 recorder、sampler、模型的直接复现和最近同维度基线。
- 预期影响：high；判断置信度：high。

##### A31-I2 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：PDF pp.3–4，§2 Matched intervention and scope；PDF pp.8–9，§4 Limitations
- 观察证据：在固定总 NFE 下改变 L 会必然改变 partition 与每块 step allocation；没有 uniform-allocation arm，也没有同一轨迹的 dual-recorder 对照。active-block recorder 同时受 model state、sampler state 和 observation window 影响。
- 重要性：观察到 P*=L 不能判断边界由记录窗口直接制造，还是由每块解码动态/状态传递产生。没有分解就无法把结果转化为对模型诊断设计的具体机制性建议。
- 必需修复：做 partition×allocation 的可分离实验，并让 active-block 与 full-canvas/passive recorder 同时观察同一底层轨迹；必要时直接记录 sampler state transfer。
- 验证标准：若只换 recorder 而固定轨迹时周期随观察窗口变化，可定位为 recorder artifact；若两 recorder 都随 sampler L 变化，则需要进一步 sampler-state 干预。报告完整交互。
- 仍需证据：factorial schedule 实验和同轨迹双 recorder 日志。
- 预期影响：high；判断置信度：high。

##### A31-I3 · 主要 · 技术正确性、实验严谨性

- 位置：PDF pp.3–7，§2–§3.3，Tables 2–7；PDF p.13，Appendix A.4，Table 11
- 观察证据：min–max separation 是离散极值统计，L48 probe 只有一个 boundary locus，L32 control 的 firing probability 仅 0.8805 且 stability band 触零。更关键的是 Appendix A.4 显示原 within-block threshold 对 25 个错误候选触发 10 次（40%）；修正 margin 的 red-team 是跨条件比较，不是每臂数据打乱下的 false-positive calibration，L24 的 pmargin=0.055/0.071 仍边缘。
- 重要性：bootstrap modal selection=1 主要反映观察样本下的离散稳定性，不等于错误发现受控。缺少有效 arm-specific null 时，五臂都选 L 的显著性强度可能被夸大，尤其是低边界基数 probes。
- 必需修复：在研究前定义有正确 type-I error 的选择性检验，用合成 null、已知干净的真实 arm 或可交换生成机制校准；报告候选搜索的 family-wise error 和低边界基数下的功效。
- 验证标准：大量 null simulations/negative-control runs 下整体误选率不超过预设 α，并在独立 positive controls 上达到预设功效；所有候选与阈值在数据前冻结。
- 仍需证据：有效 null 生成器、选择性误差校准及独立重复。
- 预期影响：high；判断置信度：high。

##### A31-I4 · 主要 · 实验严谨性、限制与负责任表述

- 位置：PDF pp.3–4，Registration and two-layer reachability；PDF p.11，Appendix A.1，Table 9
- 观察证据：min–max estimator 是在 development control 失败后选择、虽在 treatment 前登记；更重要的是 amendment v3 在观察 treatment 数据后撤回 model-intrinsic decision branch。canvas-96 probes 随后单独预注册，但不能使已经看过的主结果恢复确认性。
- 重要性：作者当前的 recorder-level 限定是正确的，但结论层级随数据改变说明原科学假设没有在干净确认流程中通过。透明记录不能消除 outcome-contingent scope change。
- 必需修复：把当前全套分析视为开发性工作，在全新模型/题目/运行上冻结 estimator、候选网格、null、scope 和结论门后一次性复现。
- 验证标准：带时间戳的注册早于任何新运行，所有 planned arms 完成，主结论不依赖事后删改分支；公开 deviations 表应为空或仅含非结论性执行细节。
- 仍需证据：独立、完全预注册的确认研究。
- 预期影响：high；判断置信度：high。

##### A31-I5 · 主要 · 可复现性、重要性

- 位置：PDF pp.8–9，§4 Limitations；PDF pp.16–17，Appendix B、Table 15
- 观察证据：研究只覆盖一个 LLaDA-8B-Instruct-scale 私有本地 checkpoint 和 GSM8K；该 checkpoint 与公开 LLaDA-8B-Instruct 并非 byte-identical，不能授权发布。L32 control 复用既有连续子池；cross-seed variance 不在 estimand 内，匿名 supplement 是否公开也待 provenance approval。
- 重要性：外部研究者无法使用相同 checkpoint 重建生成结果，单一 deterministic seed/题池也无法量化模型、问题采样或运行变异。结果可能只属于本地模型和 recorder 实现。
- 必需修复：在公开 checkpoint 上完整复现并发布 events/scripts；用多个独立 question subsets 或 seeds，重新运行同 session 的 L32 control，并跨任务验证。
- 验证标准：公开工件可在独立环境重建 Table 2–8；公开模型上的预注册 P*=L 关系和无 fixed-32 结果与当前方向一致。
- 仍需证据：公开模型复现、fresh controls、多 seed/subset 与可访问匿名事件流。
- 预期影响：high；判断置信度：high。

**给作者的问题：**

- 如果同一解码轨迹同时用 active-block 与 full-canvas/passive recorder 记录，P*=L 是否只在前者出现？这是区分 recorder 工件与 sampler/model 状态的关键实验。
- 能否在固定 partition 下独立改变 per-block step allocation，或固定 allocation 结构只改 partition，从而拆分当前 composite intervention？
- 原始 within-block threshold 对错误候选有 40% 命中率；为什么修正后的 cross-condition margin 可作为 arm-specific 选择性证据，而不是另一个缺少真实 null 的诊断？
- post-treatment 撤回 model-intrinsic branch 后，作者是否会在全新、完全预注册的数据上重复核心判定而不再修改 estimand/门？
- 私有且与公开 LLaDA-8B-Instruct 非字节一致的 checkpoint 会如何影响可复现性与外部有效性？

**能提高评分的证据：**

- 同一轨迹的 active-block 与 full-canvas/passive 双 recorder 实验明确定位工件来源。
- partition×step-allocation factorial 设计分离当前 composite intervention。
- 在新数据上使用经 null 校准且完全预注册的选择性检验复现五臂关系。
- 在公开 checkpoint、多个 sampler/任务/seeds 上复现并公开完整事件流。

**会降低评分的证据：**

- 有效 null 校准显示修正 margin 仍有高误选率。
- 双 recorder 对照表明 P*=L 只是特定实现的日志重置现象，且对科学诊断没有下游影响。
- 公开模型或 fresh L32 control 无法复现 period-following 关系。
- 更换合理的非极值 estimator 后周期选择不稳定或主要结果消失。

**伦理标记：** 否。未发现需要升级的伦理问题；研究使用公开数学题与离线模型轨迹，不涉及人类受试者或部署。

**LLM 使用披露：** 本审稿由隔离运行的 AI 子代理 R1 独立完成，仅用于内部投稿前质量控制；未与任何其他评审者或评审代理通信，也未查看其输出。

**评审限制：**

- 仅审阅指定 SHA256 的 18 页冻结 PDF；未读取源代码、事件流、注册文件、匿名工件或任何其他评审输出。
- 按隔离任务约束未联网，未独立核验外部引文、相关工作覆盖完整性或工件可访问性；新颖性判断仅相对于 PDF 自述的相关工作。
- 逐页视觉核查未发现裁切或无法解析页面。

#### R2（技术正确性）完整评议

**论文概述：** 论文研究 masked-diffusion 解码事件日志中的 identity-lock 周期是否跟随 schedule block length。主 arms 在固定 NFE=48 下比较 L=16、32、64，另在 96-token canvas 上测试 L=24、48；active-block recorder 只记录当前 sampling block。作者定义边界/内部距离 separator sep(P)，五个 arms 的最大分离均选择 P=L，bootstrap 与 rank-sum 多数支持这一点。原 within-block threshold detector 在错误候选上有 10/25（40%）被判为阳性，作者事后改用 top-minus-second margin，并把结论收窄为 recorder-level。非嵌套 L=24/48 probes、一个独立 L=32/NFE=32 content analysis 及 provenance 审计被用作补充证据。

**最强的已核实贡献：** 第 4–8 页最可靠地支持一个窄描述性结论：在 L=16、32、64 以及 96-token canvas 的 L=24、48 五个冻结事件流中，separator/rank criteria 选出的 period 都是调度块长 L。作者同时明确 active-block recorder 的观测支持集随 L 改变，并撤回模型内在周期主张，因此该结果可作为事件记录器与 schedule geometry 不可分离的具体警示。

**维度理由：**

- Soundness：五个 arms 中估计的 period 都跟随 block length，描述性结果清楚，且作者最终把解释限定在 active-block recorder 而非模型内在动力学。但 detector 在原阈值下对错误候选有 40% 假阳性，修正 margin 是看过该失败后提出；主 intervention 又同时改变分块与每块 allocation，缺少 full-canvas recorder/factorial 对照。因而当前证据更像记录器几何的自证，而非经校准的 schedule-tracking phenomenon。
- Presentation：18 页 PDF 的定义、separator、bootstrap、非嵌套 probes 和撤回的 model-intrinsic 分支均可追踪，版面无缺页且图表可读。作者对 post-hoc detector 与限制披露较好，但正文保留过多历史判尺、manifest 与修订路径，使主结论的确认性边界不够醒目。
- Contribution：把 active-block recorder 的 identity-lock period 与 schedule block length 对齐并做 L=16/32/64、96-token nonnested probes，是有用的 instrumentation warning。不过 active-block recorder 按定义只观察当前块，中心规律很大程度由测量支持集诱导；缺乏真正独立的记录器与机制对照，科学增量有限。

**优点：**

- 加入 L=16 与 L=64 新 arms，并用 L=24/48 非嵌套 canvas probes 减少只在 32 的整数倍上看到别名的单一解释。
- 在发现原 threshold detector 对错误候选产生 40% 阳性后没有继续把它包装为确认，而是披露失败并收窄主张。
- 报告 separator、bootstrap stability、rank-sum 与 candidate grid，允许读者看到选择 P=L 的具体依据而非只看结论。
- 明确区分 recorder-level schedule tracking 与 model-intrinsic periodicity，并将后者标为撤回/未识别。

**问题与可验证修复：**

##### A31-R2-01 · 主要 · 技术正确性、实验严谨性

- 位置：第 3–6 页 detector 定义与主结果；第 8–10 页 red-team/null audit；第 14–16 页阈值敏感性
- 观察证据：原冻结 within-block threshold 在 25 个错误候选中有 10 个（40%）通过，说明不能解释为有校准的 period detector。当前主依据改为看过失败后定义的 top-minus-second margin，并用同一批 treatment arms/candidate structure 评估；没有独立 clean null arm 给出 arm-level 或 family-level FPR。
- 重要性：候选 period 的距离统计天然受块结构影响，事后 margin 可选择性放大 P=L 与次优候选的差。bootstrap 只刻画当前样本稳定性，不能替代在无真实 period 时的错误发现率。
- 必需修复：在不含 schedule-locked signal、但保留相同长度与边际分布的独立 null streams 上冻结 detector 与阈值；同时在未使用的 treatment streams 上评估 power，并校正候选搜索。
- 验证标准：先锁定候选网格、score、margin 与决策阈值；大量独立 null arms 的 family-wise FPR 应低于预设水平，且新 treatment arms 的 P=L power/coverage 达到预设标准。
- 仍需证据：独立 null/treatment calibration、候选搜索校正与完整 FPR/power 曲线。
- 预期影响：high；判断置信度：high。

##### A31-R2-02 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第 2–4 页 recorder 与 intervention 定义；第 6–8 页主结果；第 10–12 页解释边界
- 观察证据：active-block recorder 只记录当前 sampling block，其可见 token/support 在每个 L 边界机械重置。改变 L 还同时改变 partition 与 required per-block allocation；没有在同一 decode 上保持恒定支持集的 full-canvas recorder，也没有分离这两个因素的 factorial intervention。
- 重要性：P=L 很可能是测量窗口定义的几何结果。当前研究证明了 recorder 追随自己被调度的边界，却不能判断日志中是否还有独立的模型行为或 allocation effect；即使限定为 recorder-level，识别内容仍接近定义性。
- 必需修复：对同一随机解码轨迹并行记录 active-block 与 full-canvas/constant-support events，并正交改变 partition boundary 和 per-block allocation；预先定义 recorder×schedule 交互。
- 验证标准：若现象来自 recorder，P=L 应只在 active-block channel 随 L 移动；若来自模型或 allocation，应在恒定支持 recorder 上仍出现可预测变化。factorial interaction 需给区间。
- 仍需证据：双记录器同轨迹数据与正交 intervention。
- 预期影响：high；判断置信度：high。

##### A31-R2-03 · 主要 · 实验严谨性、技术正确性、限制与负责任表述

- 位置：第 7–9 页 96-token nonnested probes；第 13–15 页 probe 细节
- 观察证据：nonnested probes 使用与主 arms 不同的 96-token canvas。L=24 的 corrected margin 证据边缘（报告的 p_margin 约 0.071），L=48 虽约 p=0.021，但 96-token canvas 只有一个内部 48 边界（|B48|=1），对单个边界异常高度敏感。
- 重要性：这两项不足以稳健排除固定 32-token 周期、canvas-specific artifact 或偶然边界。它们是支持性 probes，而不是对 aliasing 的强确认。
- 必需修复：为每个 nonnested L 使用包含多个内部边界的更长 canvas、多个 independent decodes/seeds，并加入固定 P=32 的预注册竞争假设；保持 recorder 和其他配置与主 arms 一致。
- 验证标准：在每 arm 至少多个边界与多个 seeds 下，P=L 相对 P=32 的 margin 区间应排除零，leave-one-boundary-out 后仍保持。
- 仍需证据：多边界、多 seed nonnested arms 与边界影响分析。
- 预期影响：high；判断置信度：high。

##### A31-R2-04 · 主要 · 可复现性、实验严谨性

- 位置：第 9–11 页 provenance；第 16–18 页 artifact/manifest 说明
- 观察证据：实验使用的私有 checkpoint 与公开版本不是 byte-identical，且不能重新发布；event streams/scripts 的匿名发布仍依赖 provenance 批准。稿件还披露 estimator source 在 manifest 后被触碰，虽在第二环境重推结果，但审稿对象中没有可执行 artifact 供独立验证。
- 重要性：中心结果完全依赖事件记录语义与细微 detector 实现；checkpoint、events 或确切代码任一不可得都会阻止第三方重建，也无法排除版本漂移。
- 必需修复：发布所有去标识 event streams、不可变 estimator 版本、环境锁文件与 hashes；若 checkpoint 不能发布，至少在可公开 checkpoint 上运行完整复制并量化差异。
- 验证标准：独立环境从发布 artifact 重建每个 candidate score、separator、bootstrap 与表格；公开 checkpoint 复制应预先定义容许偏差并保持 period-follow-L。
- 仍需证据：可执行匿名 artifact、版本 hashes 与公开 checkpoint 复制。
- 预期影响：high；判断置信度：high。

##### A31-R2-05 · 次要 · 重要性、清晰度、限制与负责任表述

- 位置：第 8–10 页 content analysis；第 10–13 页讨论与结论
- 观察证据：content association 来自另一个 L=32/NFE=32 bed，而非跨主 arms 的共同分析；报告 tau=0.272，并称到 BIN16 仍可见，但 BIN32 已没有足够分辨率/检验力。它不能验证五个 schedule arms 中的 period 机制。
- 重要性：将单独床的弱内容关联与 recorder geometry 并列，容易给中心结论增加并不存在的机制含义。论文当前最清楚的贡献是 instrumentation warning，而不是 content-linked dynamics。
- 必需修复：把 content analysis 明确降为独立探索性附录；若要保留机制贡献，应在所有 arms 用同一预注册 content feature 与分辨率检验 interaction。
- 验证标准：跨 arms 的 content×boundary 或 content×L 交互在预注册粒度下复制，且不依赖选择 bin width；否则结论只保留 recorder-level 描述。
- 仍需证据：跨 arm 内容分析或更窄的最终主张。
- 预期影响：medium；判断置信度：high。

**给作者的问题：**

- active-block recorder 的支持集天然在每个 block 边界重置时，什么可证伪结果会让 corrected detector 不选 P=L？
- 能否在同一 decode 上并行运行 full-canvas/constant-support recorder，以直接分离模型事件与记录器可见性？
- 主 L=16/32/64 arms 同时改变 partition 与每块 allocation；可否做 factorial 设计分别改变二者？
- 修正后的 top-minus-second margin 在独立 clean null arms 上的 false-positive rate 是多少，阈值是否在处理 treatment 前冻结？
- 私有 checkpoint 与 event streams 是否会随匿名 artifact 发布；公开近似 checkpoint 与私有版本的差异能否量化？

**能提高评分的证据：**

- 在同一解码轨迹上用 active-block 与 constant-support/full-canvas 双记录器验证 recorder×schedule 交互。
- 用独立 clean null arms 预注册并校准 corrected detector 的 family-wise false-positive rate 与 power。
- 完成多边界、多 seed 的 nonnested probes，并发布可逐项重建的事件与代码 artifact。

**会降低评分的证据：**

- 恒定支持记录器显示相同 P=L，从而暴露当前 recorder-level 解释不完整或 detector 对 schedule 的普遍混淆。
- 独立 null calibration 显示 corrected margin 仍有高假阳性率。
- 多 seed、多边界 probes 不能稳定区分 P=L 与固定 P=32/其他候选。

**伦理标记：** 否。未发现涉及人类受试者、敏感数据、隐私或直接安全风险的问题。

**LLM 使用披露：** 本审稿由隔离的 AI 子代理 R2 生成，仅用于内部投稿前质量控制；未与其他评审通信，未读取其他评审目录、作者计划、编辑上下文、旧稿或历史评分。

**评审限制：**

- 遵循隔离要求未联网，未核验外部引文、模型说明或相关工作。
- 仅审阅冻结 PDF，未读取或执行 checkpoint、事件日志、代码、manifest 或匿名 supplement，因此 provenance 与重推声明仅按稿件内容评价。
- PDF 共 18 页，已全文阅读并逐页视觉核查；未发现缺页或不可辨认页面。

#### R3（实验严谨性）完整评议

**论文概述：** 本文在一个 LLaDA-8B-Instruct 规模模型的 GSM8K 床上，用 active-block recorder 记录每个位置的 identity-lock time。主设计固定 canvas=256、总 NFE=48，对比 L=16、32、64；另在 canvas=96 上测试 L=24、48。min--max separation estimator 在五个 arm 上均返回 P*=L，作者据此认为 recorder 可见周期跟随 block length，并谨慎否认模型机制归因。论文还给出 spectral residual、content association、registration chronology 和 selectivity gate 审计。核心观察在冻结床上存在，但检测校准、边界抽样与复合干预限制其确认性和推广性。

**最强的已核实贡献：** 第 5--6 页图 2、表 3 与第 6 页表 5 共同显示：在固定 active-block recorder 的五个已测配置上，位置相邻差分的最大分离候选与配置 block length 一致；非极值的 rank-sum 检查也在五个 arm 上独立选出 L。该结果直接支持“当前记录窗口会把 schedule 周期写入测量”的有限结论。

**维度理由：**

- Soundness：五个测试 arm 的 min--max period estimator 均选出配置的 block length，且作者使用共享题目 bootstrap、非嵌套 probe、rank-sum 检查和负证据进行了多角度审计。但注册的“检测”规则主要由 sep>0 与 bootstrap 众数稳定性组成，并不是零假设校准的假阳性检验；附录 A.4 还显示原 within-block gate 对错误候选的触发率为 10/25=40%。修正的 margin permutation 在五个 arm 中只有四个低于 5%，L24 为 0.071，且没有形成全 family 校正。固定边界位置不在 bootstrap 中重采样，L48 仅一个边界位置，因此稳定性被高估。
- Presentation：论文对“recorder-observable association”而非模型机制的范围约束、注册时间线、失败分支和复现实验做了罕见的详细披露，图表均清晰。另一方面，18 页包含多套 gate、legacy verdict、reachability、spectral/content 支线与大量审计表，使主假设与真正的确认性证据不够简洁；标题的强断言也没有反映 L24 校准未过关。
- Contribution：证明 active-block recorder 的位置周期会随 block schedule 改变，是有价值的测量警示，但在该 recorder 定义下相当可预期。干预同时改变 block partition 和每块步数，且仅一个私有 checkpoint、一个任务，无法说明模型内部机制或分离具体原因。当前更像严谨的单床仪器审计，而不是充分普适的方法学贡献。

**优点：**

- 主处理 arm 固定模型、题集、canvas、总 NFE、温度与 remasking 规则，并用共享题目索引进行成对 bootstrap。
- 第 2--4 页清楚区分 a-priori 可达性与 from-events 符号，不把边界集合嵌套误当作效应方向。
- 第 7--9 页明确称 spectral 检查为探索性，并把 active-block recorder 观察与模型/采样器内部机制分开。
- 第 11、13 页完整披露注册后撤回的模型机制分支和原 gate 的 40% 错误候选触发率。
- 报告边界/内部位置基数、L48 单边界、L32 控制复用、私有 checkpoint 与释放限制等会削弱结论的信息。
- 逐页视觉核查显示 18 页均完整可读，图 1--2、公式与密集表格无裁切或乱码。

**问题与可验证修复：**

##### A31-I1 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第 3--4 页 min--max period separation 与表 2；第 6 页表 5；第 13 页附录 A.4 表 11
- 观察证据：注册检测规则要求 sep(P*)>0 且 bootstrap 中 P* 众数比例≥0.95；bootstrap 只说明对题目重采样的选择稳定性，并非 null-calibrated false-positive test。作者后来发现原 within-block permutation gate 在错误候选上触发 10/25（40%）。修正 top-minus-second margin 的离散 p 值在 L16/L64/L48/L32 低于 0.05，但 L24 为 0.071；文中未给五 arm/候选的整体多重性决策。
- 重要性：稳定地选择一个候选不等于排除结构化零假设。已知旧 gate 严重反保守，而新 gate 又未覆盖全部 headline arms，因而“all five detected”比校准证据更强。
- 必需修复：预先定义一个保持位置自相关、边界异方差和候选选择过程的有效零假设；对完整 P* 选择算法校准 family-wise I 类错误，并将 L24 明确列为未通过，直至独立重复达到门限。
- 验证标准：在大量保留现实空间结构但无 schedule-locked 信号的模拟/置换床上，整套选择规则的 family-wise 假阳性率应≤0.05；在新数据上所有预定 arm 的校正 p 值均通过才可作 all-arm claim。
- 仍需证据：零假设生成器、完整选择流程的误报模拟、family 定义、校正决策和独立 L24 重复。
- 预期影响：high；判断置信度：high。

##### A31-I2 · 主要 · 实验严谨性、技术正确性

- 位置：第 5 页表 3；第 7 页第 3.2 节；第 9 页 Boundary cardinality 限制
- 观察证据：bootstrap 仅对 300 个 decode 重采样，canvas boundary loci 固定。表 3 显示 L48 的 boundary set 只有 1 个位置、L24 只有 3 个；L32 稳定带还触及零。
- 重要性：对题目稳定不能代表对边界位置、canvas 对齐、padding 或 offset 的稳定。尤其 L48 的正分离是单一位置读数，无法估计边界层面的变异。
- 必需修复：增加多个 canvas 长度、起点偏移与边界位置重复；在层级重采样/模型中同时把题目和 boundary locus 作为抽样维度，并预定最小边界数。
- 验证标准：leave-one-boundary-out、随机 offset 和 boundary-level bootstrap 后仍选择同一 P*，且区间不依赖任何单一 locus。
- 仍需证据：逐题逐位置 d(r)、多个 offsets/canvases、边界层级方差与影响诊断。
- 预期影响：high；判断置信度：high。

##### A31-I3 · 主要 · 技术正确性、实验严谨性、重要性

- 位置：第 1--4 页摘要、表 1 与 Matched intervention and scope；第 9 页 Composite intervention 限制
- 观察证据：在固定总 NFE 下改变 L 必然同时改变 block partition 和 required per-block step allocation；active-block recorder 又只观察当前 block。论文承认这是 partition-plus-allocation 的复合处理，且没有 full-canvas/passive recorder 对照。
- 重要性：结果不能区分是边界几何、每块更新次数、active window 截断还是其交互造成周期。即使不作模型机制声称，测量层的因果解释仍不唯一，贡献接近 recorder 定义的直接后果。
- 必需修复：采用 factorial intervention 分别操纵 partition 与 step allocation，并在相同 trajectory 上同时运行 active-block 和 full-canvas/passive recorder；比较同一模型状态下的读出。
- 验证标准：在 allocation 固定时只改变 partition、以及 partition 固定时只改变 allocation，分别估计 period/幅度；dual-recorder 差分应定位 active window 的贡献。
- 仍需证据：factorial arm 配置、共享题目/种子输出、dual-recorder 事件与交互效应。
- 预期影响：high；判断置信度：high。

##### A31-I4 · 主要 · 实验严谨性、限制与负责任表述

- 位置：第 2、7、9 页 L32 development control；第 11 页表 9
- 观察证据：L32 使用预存 contiguous decode_id [345,645) 子集；初始 signed-mean estimator 在该 development control 失败后被 min--max 规则替换，随后才运行 treatment arms。L32 的当前 firing probability 仅 0.8805，是五 arm 最弱且稳定带触零。
- 重要性：治疗 arm 对锁定规则可视为前瞻性，但 L32 本身参与了 estimator 选择，不能同时作为无偏确认性 control。弱稳定性提示选择可能贴合该预存样本。
- 必需修复：在 estimator 与 gate 完全锁定后，用新题目、新 session/seed 对 L32 进行独立复验；将原 L32 明确标为 development-only。
- 验证标准：盲运行的新 L32 数据应在预定 detection/null-calibration gate 上通过，并复现效应大小与 rank-sum 结果。
- 仍需证据：锁定时间戳、新样本清单、运行日志和未调参的完整结果。
- 预期影响：medium；判断置信度：high。

##### A31-I5 · 主要 · 可复现性、重要性、限制与负责任表述

- 位置：第 2 页 Model bed；第 9 页 Bed and grid；第 16--17 页附录 B
- 观察证据：所有实证来自一个模型家族/私有 checkpoint、GSM8K、temperature 0 和一组固定 grids。checkpoint 与公开 LLaDA-8B-Instruct 非逐字节相同，且释放需要授权；事件/脚本 supplement 也受 provenance approval 约束。
- 重要性：schedule artifact 可能依赖具体 decoder、recorder、任务长度与实现。没有跨模型/任务/采样规则重复且关键 checkpoint 不可自由获得，普适 reporting rule 的经验基础和复现路径均有限。
- 必需修复：在可公开的至少两个模型实现、不同任务/长度分布和 recorder/decoding variants 上前瞻性重复，并发布足以端到端重建的事件与配置。
- 验证标准：第三方在公开 checkpoint 上能重建 arm 并复现 period-selection pattern；跨床报告预定效应与异质性，而非只合并成功案例。
- 仍需证据：公开 checkpoint/替代床、完整 decode 配置与种子、原始事件、跨床结果。
- 预期影响：medium；判断置信度：high。

**给作者的问题：**

- “detected”为什么由 sep(P*)>0 加 bootstrap mode fraction≥0.95 定义，而不是由一个对无 schedule-locked 周期有有效 I 类错误控制的零假设检验定义？在无结构但保留位置自相关/边界异方差的模拟下，该规则的假阳性率是多少？
- 第 6 页表 5 的 p_margin 是否属于五 arm × 多候选的同一确认性 family？若是，为什么 L24=0.071 仍被标题中的 all-five period claim 覆盖？
- L48 只有一个 boundary position。若平移 canvas 起点、改变 padding/截断或使用多个 boundary offsets，P*=48 的选择频率是否仍稳定？
- 能否做 2×2 或更多水平的 factorial 设计，分别改变 block partition 与 per-block step allocation，并同时记录 active-block 和 full-canvas trajectory，从而分离 schedule 与 recorder window？
- L32 控制用于开发 estimator 且来自预存池；是否计划在未参与开发的新 300 题/新 session 上盲复验其最低的 firing probability 0.8805？

**能提高评分的证据：**

- 完整选择算法在保留空间结构的零假设下通过 family-wise I 类错误校准，且独立 L24/L32 重复通过锁定 gate。
- 多个 canvas offsets 与 boundary loci 的层级重采样证明结果不由单一边界位置驱动。
- factorial partition × allocation 与 dual-recorder 实验分离出具体测量来源。
- 在公开 checkpoint、额外任务和不同解码/recorder 实现上前瞻性复现 schedule-following pattern。

**会降低评分的证据：**

- 结构保留的 null 模拟显示当前 detection rule 的假阳性率显著高于 5%。
- 改变 canvas offset 或删除 L48 单一边界后 period 选择失效。
- 新 L32/L24 确认性运行未复现或需要再次调整 estimator/gate。
- dual-recorder/factorial 实验表明现象完全是实现特定窗口截断，不能支持更广的测量建议。

**伦理标记：** 否。不涉及人类受试者或私密数据。主要责任风险是把 recorder-induced periodicity 误当作模型内部状态或机制；论文对此有明确警示。

**LLM 使用披露：** 本审稿由隔离运行的 AI 子代理 R3 完成，仅用于内部投稿前质量控制；未与其他评审通信，未读取作者计划、旧评分或其他评审输出。

**评审限制：**

- 仅审阅指定 SHA256 的冻结 PDF；未访问代码、事件流、checkpoint、注册文件或匿名 supplement，不能复算表格或核验时间戳/哈希。
- 按隔离要求未联网，未核验外部引文、公开模型差异或文中资产可访问性。
- 已全文阅读并逐页视觉核查全部 18 页；所有页面、图表和公式完整可读，未见裁切或乱码。

### A32

#### 交付与来源状态

- 选定输入：`/Users/liuhanzuo/Downloads/A32-paper.pdf`
- 源状态：`exact_latex_snapshot_assets_repin_required`
- 润色 PDF：`/Users/liuhanzuo/MacLLM-Bench/output/pdf/paper_polish_20260826/A32-polished.pdf`
- 冻结 SHA-256：`c557217eba583f4b47559e4fcbc1c75c71f688febb1f9c633cbdab4d45906277`
- 总页数：12；主文状态：主文在参考文献前不超过9页。
- 版面核验：pass；构建：pass。
- 旧评分基线：NA；旧中位数：NA。

#### 三评量化结果

| 审稿人 | 角色 | Overall | Confidence | 建议 | Soundness | Presentation | Contribution | 当前上限 | 必需修改后预测 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| R1 | 新颖性与定位 | 4 | 3 | 略低于接收线 | 3 | 3 | 2 | 4 | 6 |
| R2 | 技术正确性 | 6 | 4 | 略高于接收线 | 3 | 2 | 3 | 6 | 8 |
| R3 | 实验严谨性 | 6 | 4 | 略高于接收线 | 3 | 3 | 2 | 6 | 8 |

三评中位数为 **6**，均值 5.33，跨度 2，接收侧票数 2/3。

#### 编辑记录

- [结构审计](work/A32/structure_audit.md)
- [语义锁](work/A32/semantic_lock.md)
- [修订日志](work/A32/revision_log.md)
- [待核验事项](work/A32/needs_verification.md)

**修订日志原文：**

> # A32 revision log
>
> ## Scope
>
> Evidence-preserving manuscript polish. No experiments, independent citation verification, or reviewer scoring were performed.
>
> ## Source changes
>
> - Replaced the status-heavy title with a descriptive title centered on the observational and within-item evidence.
> - Rebuilt the abstract around the paper's inferential hierarchy: observational association, randomized length intervention, cross-bed transport limit, and judge-audit contribution.
> - Tightened the introduction and contribution list so that observational, causal, and transport claims are not conflated.
> - Clarified the second-bed `POWER_LIMITED` result and the role of the outcome-defined accuracy-regime analysis.
> - Rewrote the conclusion to preserve the reported directions and statuses while explicitly rejecting a length-to-gap causal inference from the observational contrast.
> - Moved the bibliography before the appendix. Main text now ends on PDF page 8; references and the start of Appendix A occupy page 9. No evidence was deleted to meet the page boundary.
>
> ## Semantic safeguards
>
> - Preserved all reported numbers, intervals, sample sizes, equations, table/figure contents, citation keys, named artifacts, and registered-versus-post-hoc distinctions.
> - Added no experiment, literature claim, or mechanism claim.
> - Narrowed only unsupported causal or transport wording; no negative or inconclusive status was upgraded.
>
> ## Verification
>
> - Built from the isolated source with shell escape disabled, using BibTeX and two final LaTeX passes.
> - Checked the extracted PDF text, cross-reference/citation warnings, page boundary, and rendered pages.
> - Remaining source-dependent items are listed in `needs_verification.md`.

**待核验事项原文：**

> # A32 needs verification
>
> - The novelty phrase “To our knowledge” in Related Work remains source-dependent and was not independently verified in this editing pass.
> - The FULL-193 human-label fill is future work; no prose edit treats it as completed.
> - The author should confirm that calling the +0.1606 interval the “citable” second-bed read remains the intended convention, because the manuscript also designates +0.2382 as the primary v2 value.
> - Artifact paths and one-command reproduction claims were preserved but not executed because the evidence bundle was not supplied with this isolated source copy.
> - The main text ends on PDF page 8; references begin on page 9, and Appendix A begins later on page 9 after the references. The 9-page main-text limit is therefore not a submission blocker in this build, although the venue's current formatting policy should still be checked before submission.

#### R1（新颖性与定位）完整评议

**论文概述：** 论文研究 masked-diffusion 解码中提示长度与 signed confidence-accuracy gap（SCE）的关系。GSM8K/LLaDA-8B-Instruct 的自然长度三分位观察性对比为 +0.2227，主要来自准确率下降；但在 400 个问题上加入 37 个无内容冗余 token 的配对随机干预得到 -0.0075，95% 区间为 [-0.052, 0.039]，从而不支持无内容长度导致观察性幅度的因果解释。MATH500 上原始 v2 判分给出正差异，但 30 个目的性人类锚点校正后的敏感性区间跨零，因此作者将跨床迁移标为 POWER_LIMITED。

**最强的已核实贡献：** 第 2 节与表 1 的同题随机干预：在固定解码预算下，+37-token 无内容填充的配对区间排除了 +0.2227 的观察性幅度，同时保留对小正效应的诚实不确定性；这是论文中最清楚、最可验证且最有决策价值的结果。

**维度理由：**

- Soundness：在论文明确限定的范围内，GSM8K 观察性分解、按问题聚类的 bootstrap 以及 N=400 的配对随机干预基本自洽；尤其是作者没有把包含零的干预区间解释为等效性，也没有把第二床的原始判分读数当作已校正结论。不过，跨床结论依赖 30 个目的性人类锚点，且干预只改变无内容填充长度，因此证据不足以支撑更一般的长度机制或迁移表述。
- Presentation：主文的观察性关联、因果检验和跨床敏感性三层结构清楚，图表也没有发现裁切或重叠。负面结论及区间解释较负责；但 SUPPORT、NOT_IDENTIFIED、POWER_LIMITED、sealed/fragility 等多套审计标签与多个判分口径并列，增加了理解一个本来相对简单结果的负担。
- Contribution：最有价值的是用同题随机长度干预否定一个表面上很大的观察性长度关联，而不是代数恒等式本身。当前证据只覆盖一个模型、一个解码设置以及一种无内容填充干预，第二数据床又未完成人类锚定，因此知识增量较窄，新颖性定位也尚未充分区别于既有校准分解、长度分层分析和提示扰动研究。

**优点：**

- 明确区分观察性长度分层与随机干预，并在第 2 节直言三分位划分是事后分层而非同题操纵。
- 对 SCE、confidence 和 accuracy 同时给出配对/聚类区间，且没有把 Brier/NLL 的恶化错误解释为 signed-gap 机制。
- 第 3、6 节和附录 A/B 对判分器修复、目的性锚点、隐藏假阳性风险及 FULL-193 未完成项披露充分。
- 逐页视觉检查显示 12 页均完整、图表可读、无明显裁切或覆盖。

**问题与可验证修复：**

##### A32-I1 · 主要 · 新颖性、重要性、引文完整性

- 位置：第 1 页第 1 节贡献 1；第 7 页第 7 节 Related Work；第 8 页第 9 节结论
- 观察证据：论文把“exact two-component decomposition”列为首项贡献，但该式由 SCE=E[conf]−E[acc] 的定义直接相减得到；相关工作主要分别介绍 masked diffusion、校准和答案匹配，没有用逐项对照说明其经验分析相对既有长度-校准或提示扰动研究的新边界。
- 重要性：代数恒等式不能单独承担 ICLR 级概念新颖性。若真正贡献是审慎的负面因果检验，论文需要把新颖性放在设计、可排除的效应范围及适用边界上，否则容易把实现/报告新颖性误写成理论新颖性。
- 必需修复：重写贡献与相关工作定位：将恒等式降为记账工具；建立与最接近的长度-校准分析、序列长度扰动和模型置信度测量工作的逐轴比较，明确本文新增的是哪一种随机化、估计量、效应排除或可复现证据。删除或严格限定任何“首次/精确分解”式暗示。
- 验证标准：检查摘要、引言、相关工作和结论是否一致把主贡献表述为经预注册的同题负面检验；并确认至少一张比较表或一段逐项论证覆盖问题、干预、估计量、模型与结论范围，而非只列引用。
- 仍需证据：系统的邻近文献比较；若仍主张方法新颖性，需要指出现有方法无法给出的具体输出，并由本文实验证明。
- 预期影响：high；判断置信度：high。

##### A32-I2 · 主要 · 实验严谨性、重要性、限制与负责任表述

- 位置：第 2 页第 2 节 Within-item causal check 与表 1；第 8 页第 8 节 Limitations
- 观察证据：唯一随机干预是在同一 GSM8K/LLaDA-8B-Instruct 床上加入 37 个统一、无内容的冗余 token；它同时显著降低置信度与准确率，却只排除了这一特定剂量下达到 +0.2227 的 signed-gap 扩大。论文也承认没有覆盖有语义长度、其他剂量、其他模型或解码器。
- 重要性：自然提示长度变化通常伴随信息量、结构、位置和难度变化。无内容 filler 的负结果只否定一个很窄的纯长度通道，不能解释观察性差异的来源，也不足以成为一般“长度不致失校准”的结果。
- 必需修复：将标题、摘要和所有结论严格限定为“+37-token 无内容冗余填充、单模型单解码设置”；若要提高贡献层级，增加预先规定的多剂量、有语义但答案保持不变的干预，并在至少一个额外 masked-diffusion 模型或解码设置上复现。
- 验证标准：逐处核对是否没有把 content-free filler 外推为一般 prompt length；新实验应同题配对、固定预算、报告剂量曲线与聚类区间，并明确检验交互作用而非只并排显著性。
- 仍需证据：内容保持/语义长度干预、多个剂量、第二模型或解码方案的配对结果；最好包含机制可区分的干预设计。
- 预期影响：high；判断置信度：high。

##### A32-I3 · 主要 · 实验严谨性、技术正确性、可复现性、限制与负责任表述

- 位置：第 3–4 页第 3 节；第 6–7 页第 6 节 judge validation；第 10–11 页附录 A/B
- 观察证据：MATH500 的 v2 judge 在 30 个已选锚点上为 30/30，但锚点是目的性而非概率样本；论文披露 accepted stratum 仍存在隐藏 false accept，并明确 FULL-193 两层人工补标尚未完成。相应的校正区间跨零，且作者正确称其为 design sensitivity 而非经典置信区间。
- 重要性：第二床是论文外部有效性的唯一支撑，但当前判分误差的方向和全床比例不可识别。因此原始 +0.2382 不能验证迁移，现有证据只支持“尚未决定”。
- 必需修复：完成预先声明的两层概率抽样人工标注，冻结判分规则后重新估计 judge error、校正后的 SCE 差及不确定性；在完成前，将原始 v2 正区间从摘要的显著位置降为未校正诊断，并避免把它作为跨床复制。
- 验证标准：核查人工样本的抽样概率、双标注/仲裁、盲法、分层权重和缺失处理均预先固定；主结论必须由校正区间的预设判据决定，且所有 judge 口径只保留一个主结果。
- 仍需证据：FULL-193 或统计功效等价的概率抽样人工金标准、标注者一致性、分层误差矩阵以及基于该设计的校正区间。
- 预期影响：high；判断置信度：high。

##### A32-I4 · 次要 · 清晰度、可复现性

- 位置：第 1–4 页的 SUPPORT/NOT_IDENTIFIED/POWER_LIMITED 标签；第 10–11 页多种 sealed、v2、lenient 与 anchor-corrected 口径
- 观察证据：同一第二床同时出现 v2 composite、isin sensitivity、lenient two-component、human-anchor corrected 以及 sealed/fragility lineage 数值；虽然文字逐项限定，读者仍需跨多页才能确定哪一个是主估计、哪一个不可引用。
- 重要性：过多状态词和谱系读数使核心负面结论更难核验，也提高误引未校正正结果的风险。
- 必需修复：在主文加入单一决策表，只列每个研究问题的估计量、主区间、状态和允许的结论；把废弃/谱系数字移到附录并清楚标为非结果。
- 验证标准：一名只读摘要、主结果表和结论的审稿人应能唯一回答三个问题：观察性幅度是多少、随机干预排除了什么、跨床是否建立。
- 仍需证据：不需要新实验；需要统一的结果层级与引用规则。
- 预期影响：medium；判断置信度：high。

**给作者的问题：**

- 除“把差值写成置信度项减准确率项”之外，作者认为本文相对既有 calibration-in-the-large 分解、长度分层校准研究和配对提示扰动实验的最小不可替代新贡献是什么？
- commit-step probability 在 masked-diffusion 的不同提交步或不同生成长度下是否是同一可比量？有没有用标准置信度提取方式或替代聚合方式验证结论不依赖这一选择？
- 如果把无内容 filler 换成保持答案不变但具有语义、位置或推理结构的长度干预，作者预期哪个可证伪机制会产生不同结果？
- MATH500 的 FULL-193 两层概率抽样人工标注若完成，什么预先规定的判据会把 POWER_LIMITED 升级为可迁移或不可迁移结论？

**能提高评分的证据：**

- 完成概率抽样、双人或等价质量控制的 MATH500 人类金标准，使校正后的跨床区间达到预设决策标准。
- 在另一个 masked-diffusion 模型/解码设置上复现配对长度干预，并加入有语义但答案保持不变的多剂量干预。
- 把新颖性明确定位为可排除效应的随机化负结果，并与最接近工作作逐轴比较。

**会降低评分的证据：**

- 概率抽样人工标注显示 MATH500 正差主要由 judge 偏差造成，或符号不稳定。
- 独立复现发现 GSM8K 配对身份、随机化或置信度提取与论文描述不一致。
- 替代合理 commit-confidence 定义使观察性分解或干预结论发生实质反转。

**伦理标记：** 否。未发现人类受试者、个人敏感数据或部署风险需要触发伦理审查；未来人工标注应说明标注者招募、报酬与数据处理，但这不改变当前伦理标记。

**LLM 使用披露：** 本审稿由隔离的 AI 子代理 R1 生成，仅用于内部投稿前质量控制；该子代理只读取冻结 PDF 与指定审稿规则，未与其他评审通信，也未读取其他评审输出。

**评审限制：**

- 本评审严格只依据指定冻结 PDF；未读取代码、数据、附带 artifact、源文件或任何旧稿/修订记录。
- 按任务要求未联网，也未核验外部引文的存在性、准确性或遗漏；新颖性判断仅相对于 PDF 自身的相关工作，因此更广泛文献可能改变定位评价。
- 逐页视觉核查覆盖全部 12 页；未发现明显裁切、重叠或缺页。

#### R2（技术正确性）完整评议

**论文概述：** 论文研究提示长度与 signed confidence–accuracy gap（SCE）之间的关系。GSM8K 观察性极端三分位对比给出 +0.2227 的 gap，并分解为置信度小幅下降与准确率大幅下降；随后在 400 个同题样本上比较 SHORT、固定增加 37 个冗余 token 的 LONG 与随机 0–55 token 的 RANDOM，在固定 NFE 下未观察到 gap 扩大。MATH500 第二床使用自动 judge、30 个目的性人类锚点与偏差修正，但修正区间跨零，因此被标记为 POWER_LIMITED。

**最强的已核实贡献：** 第 2–4 页、图 1 与表 1 的同题结果最可信：在相同问题与固定推理预算下，指定的无内容冗余插入使置信度和准确率同时下降，而 LONG−SHORT 的 SCE 变化为 −0.0075，95% 区间 [−0.052, +0.039]；这直接反驳了“观察到的 +0.2227 必然由纯 token 数增加造成”的简单解释。

**维度理由：**

- Soundness：第一床的分解恒等式、按问题配对的干预、聚类自助区间以及对观察性与因果性边界的区分总体正确。决定性保留意见在第二床：30 个目的性人类锚点并不支持重复抽样推断，附录人为指定的偏差标准误只是设计敏感性参数；此外，所谓长度干预识别的是固定解码预算下插入特定冗余文本的效应，而不是一般的自然长度效应。
- Presentation：论文把观察性对比、干预、两套 judge、锚点修正和三态结论均公开，但主文与附录在“designated v2 estimate”和真正可引用的 lenient-plus-anchor 读数之间来回切换，表格与术语层级过密，读者不易迅速确定唯一主结论。
- Contribution：将 signed calibration-in-the-large gap 精确分成置信度与准确率分量，并用同题干预显示两者可同步下降而 gap 不扩大，是有用且可复用的诊断结果。跨床可迁移性尚未建立，因此贡献主要来自第一床的反机制证据而非广泛定律。

**优点：**

- 明确写出 ∆SCE=∆conf−∆acc，并将 signed gap 与 ECE、adaptive ECE、Brier、NLL 分开，避免把一个均值差冒充完整校准刻画。
- 400 个问题的同题对照、固定解码预算、按问题聚类自助法与两个生成上限/读出敏感性，使第一床方向性结论具有较好的内部可核查性。
- 作者主动报告观察性长度分层受内容与难度混杂、结果分层是 post-outcome、第二床目的性锚点不能产生经典抽样区间等边界。
- 第二床同时展示 lenient、v2、isin-cluster 与人类锚点修正读数，没有掩盖会使结论从正向变为未定的 judge 依赖性。

**问题与可验证修复：**

##### A32-R2-01 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第 2–4 页，§3，图 1，表 1（SHORT/LONG/RANDOM 干预）
- 观察证据：LONG 在每题加入同一类 37-token 内容空洞文本，RANDOM 加入 0–55 token；所有条件固定 NFE=64。论文用该对比否定观察性 +0.2227 gap 的长度机制，同时又承认自然、内容承载长度未被干预。
- 重要性：该设计识别的是“在固定 NFE 下插入这类冗余串”的总效应，token 数与位置、格式、注意力分配及单位 token 计算预算同时改变。因而不能把零结果推广为一般长度效应的界，也不能说明自然长题为何产生观察性关联。
- 必需修复：将因果 estimand 明确命名为 filler-insertion-under-fixed-budget；弱化所有未带该限定的 length→gap 语言。若要主张一般长度机制，加入多种语义等价 filler、位置/格式对照、不同 NFE 以及内容承载长度的随机化对照。
- 验证标准：对至少两种结构不同但等 token 的 filler、位置保持对照和多个 NFE 重做预注册配对分析；只有这些对比方向一致且语义盲评通过，才允许扩大因果范围。
- 仍需证据：逐题干预材料、随机化方案、语义保持盲评与分条件配对效应/区间。
- 预期影响：high；判断置信度：high。

##### A32-R2-02 · 主要 · 技术正确性、实验严谨性

- 位置：第 4–5 页 §4；第 9–12 页附录 A–B，表 5–6，尤其附录 B 的 uncertainty propagation
- 观察证据：30 个锚点为目的性样本而非概率样本；可引用读数 +0.1606 [−0.059,+0.380] 把组件标准误与 se_bias(k)=sqrt(k)/10 组合。论文明确称该量为 conservative convenience bound、非已识别的抽样标准误；accepted stratum 仍有已知隐藏 false accept，FULL-193 未完成。
- 重要性：在没有抽样机制和覆盖证明时，括号区间不能承担频率学意义上的 95% 不确定性；任意的 k 映射会直接控制区间是否跨零。未完整抽取 accepted/rejected 两层还使偏差修正不可识别。
- 必需修复：把该区间严格标为参数化敏感性带且不给予 95% 置信解释；最好按预声明的双层概率抽样完成 FULL-193，并用分层混淆率传播或联合 bootstrap/贝叶斯模型重新估计。
- 验证标准：从完整概率样本重算 FA/FR 与 ∆SCE2,corr，报告抽样权重、联合不确定性和覆盖模拟；检查主结论在合理 judge/标注误差模型下是否稳定。
- 仍需证据：完整双层人类标签、抽样概率、原始 judge×human 混淆表及可重算的联合不确定性分析。
- 预期影响：high；判断置信度：high。

##### A32-R2-03 · 次要 · 清晰度、限制与负责任表述

- 位置：第 3–5 页 §4、图 2；第 10–12 页表 5–6
- 观察证据：正文先称 v2 +0.2382 [0.1639,0.3125] 为 designated second-bed estimate，附录又规定唯一可引用值是 lenient+anchor 的 +0.1606 [−0.059,+0.380]，而 v2 只是 convention sensitivity。
- 重要性：两层主读数具有相反的推断状态；读者可能错误引用排除零的 v2 结果，掩盖论文自己判定的 POWER_LIMITED。
- 必需修复：在摘要、正文、图 2 与结论只保留一个明确的 reporting-layer 主读数；把其他 judge 行统一置于 sensitivity，并在第一次出现时解释不能升级结论。
- 验证标准：全文搜索 +0.2382、+0.1606、designated、citable 与 SUPPORT，确认每次出现的证据等级一致且无互相冲突的主结论。
- 仍需证据：修订后的统一 claim-to-estimand 表和全文一致性检查。
- 预期影响：medium；判断置信度：high。

##### A32-R2-04 · 次要 · 实验严谨性、清晰度

- 位置：第 5 页表 2与 §4 的 outcome-stratified sensitivity
- 观察证据：low-accuracy 与 accuracy=0 子集由观察到的结果定义且彼此嵌套；在 accuracy=0 子集 ∆acc≡0，所以 composite 等于 ∆conf 是构造恒等式。
- 重要性：这是结果条件化诊断，不能被读取为难度控制或机制证据；多重且嵌套的选择也使区间不具确认性。
- 必需修复：保留为明确的 post-outcome 描述图，删除任何“控制难度”措辞；如需难度机制，使用外生、预处理的难度标签并预注册分层。
- 验证标准：检查正文不再把表 2 当作因果支持；新的外生分层应在看模型结果前固定并报告各层样本量与交互检验。
- 仍需证据：外生难度定义或仅需文字收缩。
- 预期影响：low；判断置信度：high。

**给作者的问题：**

- LONG 条件的 +37 token 是对所有问题确定性施加，还是在任何顺序/模板因素上随机化？若只有 RANDOM 剂量随机，标题中的“within-item randomized experiment”具体指哪个随机化单元？
- 能否给出插入文本的逐题语义保持、人类盲评或多种等长 filler 复现，以区分 token 数、位置重排、格式提示与固定 NFE 下计算稀释？
- 为什么附录 B 选择 se_bias(k)=sqrt(k)/10 作为偏差不确定性，而不是对预先定义的概率锚点样本做分层二项/贝叶斯传播？该函数的覆盖性质是什么？
- 在 FULL-193 或至少完整 accepted/rejected 双层概率样本完成前，为什么正文还将 +0.2382 称为 designated second-bed estimate，而不是仅把 +0.1606 [−0.059,+0.380] 作为唯一报告层读数？

**能提高评分的证据：**

- 完成预声明的双层概率人类标注并用可识别的联合不确定性传播确认第二床结论。
- 在多种 filler、位置控制、NFE 和至少第二个模型上复现同题干预方向，明确区分纯长度与计算预算效应。
- 统一第二床唯一主读数并把所有 judge 变体降为清晰的敏感性分析。

**会降低评分的证据：**

- 概率锚点显示 accepted stratum 的假阳性率足以消除或反转第二床 gap。
- 改变 filler、位置或 NFE 后 LONG−SHORT 结论反转，说明当前零结果由单一模板或预算耦合造成。
- 发现所谓同题配对或随机剂量并非按预注册方案执行。

**伦理标记：** 否。未发现需要伦理升级的问题；主要风险是把有限床、目的性人类锚点和特定 filler 干预过度推广，论文已部分披露但仍需更统一的范围约束。

**LLM 使用披露：** 本审稿由隔离的 AI 子代理 R2 生成，仅用于内部投稿前质量控制；未与其他评审通信，未读取其他评审结果、作者计划、编辑上下文或历史评分。

**评审限制：**

- 遵循隔离要求未联网，因此未核验外部引文、论文优先权或数据集/模型文档。
- 仅审阅给定冻结 PDF；未读取源文件、代码、数据、修订日志、旧评审或其他 reviewer 目录，故 artifact 中的配对、hash、bootstrap 与 judge 实现未独立执行。
- PDF 共 12 页，已全文阅读并逐页视觉核查；未发现影响审读的解析或版面缺失。

#### R3（实验严谨性）完整评议

**论文概述：** 本文研究 masked-diffusion 模型中提示长度与有符号置信度—准确率差 SCE=confidence−accuracy 的关系。GSM8K 的事后极端长度三分位比较得到 ΔSCE=+0.2227，95% 题目簇 bootstrap 区间为[0.169,0.277]，分解为置信度下降−0.0314与准确率下降−0.2541。随后在400道题上实施 Short、Long（增加37个无语义冗余 token）和随机剂量臂；Long−Short 的 ΔSCE 为−0.0075，区间[−0.052,+0.039]，因此论文将“无语义长度本身导致差距扩大”判为 NOT_IDENTIFIED，而不是把观察关联解释为因果。MATH500 上自动 judge 给出正向差异，但30个人工锚点的校正区间跨零，故跨床结论被标为 POWER_LIMITED。

**最强的已核实贡献：** 第2节表1所示的400题同题干预最可信：在固定 NFE=64、同题配对及题目簇 bootstrap 下，增加37个无语义 token 同时降低置信度和准确率，但未检测到 SCE 扩大（−0.0075，[−0.052,+0.039]）。这有力地否定了“观察到的约+0.223差距可由该无语义长度操作直接复现”这一具体机制解释；它不证明自然长度没有因果作用。

**维度理由：**

- Soundness：论文把观察性长度关联、同题干预和跨数据床敏感性明确分层，并使用以题目为簇的 bootstrap、成对对照和多重性校正。主要方向性结论由数据支持，但解码随机性未被重复采样，观察效应与填充干预效应之间也未形成同一估计量上的正式检验；MATH500 的人工锚定样本仍不足。
- Presentation：核心估计量、证据等级和撤回边界在正文及表1、表3、表4、表6中较清楚，图表可读且没有视觉截断。材料略显审计化，多个 judge 版本、锚定路线和四态标签增加了阅读负担，但关键数字与限制均能定位。
- Contribution：最有价值的是把自然长度关联拆成置信度与准确率两部分，并用预先声明的同题、无语义填充干预检验一个具体因果机制。当前证据只覆盖一个模型、一个主要任务与一种人工填充操作，第二数据床又处于 POWER_LIMITED，因此尚不足以形成可迁移的一般结论。

**优点：**

- 第2节将 ΔSCE 精确分解为 Δconfidence−Δaccuracy，避免把准确率下降误写成纯校准恶化。
- 表1的同题三臂设计固定解码预算并保存题目级配对；正文还指出逐 decode 的朴素斜率会因共享题目异质性产生伪显著，而题目簇斜率为+0.0009/token、区间跨零。
- 观察性主分析按题目而不是1948条预测进行聚类重采样，并明确说明中间三分位不进入极端对比。
- MATH500 judge 修复、30个锚点、已知隐藏 false accept 以及 FULL-193 尚未完成都被主动披露，没有把不充分的敏感性区间包装成经典置信区间。
- 局限性与证据等级用 SUPPORT、POWER_LIMITED、NOT_IDENTIFIED、NOT_MEASURED 区分，最终结论总体遵守这些边界。

**问题与可验证修复：**

##### A32-R3-01 · 主要 · 实验严谨性、技术正确性、可复现性

- 位置：第2节“Within-item causal check”及表1（PDF第3–4页）；第5节解码计划（PDF第7页）
- 观察证据：每道题在 Short/Long/Random 各臂保存一个结果，并以题目为重采样簇；文中给出一个固定 seed，但未展示跨独立解码 seed 的重复生成或方差分解。
- 重要性：题目簇 bootstrap 正确处理同题多臂相关性，却不能自动估计随机生成器在固定题目上的方差。若解码并非确定性的，一次生成可能使零效应区间过窄，也无法判断结果是否依赖某个随机轨迹。
- 必需修复：明确证明当前解码在固定输入下是确定性的；否则对一个预先指定的题目子集或全体题目运行多个独立解码 seed，并用题目为主簇、seed 为题内重复的层级分析重新估计主对比。
- 验证标准：至少用3–5个独立解码 seed 复现 Long−Short ΔSCE，报告题目间与题内 seed 方差，并检查合并层级95%区间是否仍包含0且排除具有实践意义的正效应界。
- 仍需证据：确定性执行证明，或多 seed 的逐题逐臂输出及层级 bootstrap/混合模型结果。
- 预期影响：high；判断置信度：high。

##### A32-R3-02 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第2节图1与表1附近（PDF第3–4页）及第7节结论（PDF第9页）
- 观察证据：论文以观察性 ΔSCE=+0.2227 与同题填充干预 ΔSCE=−0.0075 对照，并强调后者区间不含前者的量级；但二者来自不同样本构造、不同长度内容和不同估计量，未报告二者差值的联合不确定性。
- 重要性：一个点估计未落入另一个估计量的区间并不等价于正式的效应差异检验。更重要的是，无语义填充只检验一种长度处理，不能单独排除自然题面长度携带的语义、格式或难度机制。
- 必需修复：把结论严格限定为“该无语义填充操作未复现观察量级”，并在共享题目上定义可比较的自然变体/填充变体差值，或给出联合 bootstrap 的 contrast-of-contrasts。
- 验证标准：对共享题目同时重采样所有相关臂，输出(自然长−自然短)−(填充长−短)的95%区间；若新增内容承载臂，还应预先声明等价性/非劣界限。
- 仍需证据：统一估计目标的联合重采样结果，以及更贴近自然长度变化的受控处理。
- 预期影响：high；判断置信度：high。

##### A32-R3-03 · 主要 · 实验严谨性、技术正确性、限制与负责任表述

- 位置：第3节 MATH500 分析（PDF第4–5页）、第6节表4，以及附录表5–6（PDF第10–12页）
- 观察证据：主 v2 judge 的未校正读数为+0.2382 [0.1639,0.3125]，但30个 purposive 人工锚点下的 lenient 校正为+0.1606 [−0.059,+0.380]。论文还发现一个未包含在锚点中的已知 false accept，并明确指出 FULL-193 未完成。
- 重要性：30个非概率锚点无法为整个 accepted/rejected 集合提供可解释的抽样覆盖；已知漏掉的 false accept 直接表明“接受集洁净”路线失效。跨床方向因此不能作为外部有效性证据。
- 必需修复：完成预先声明的 accepted 与 rejected 两层概率样本人工标注，使用盲法双人判定和冲突仲裁，并按抽样概率估计校正量与不确定性。
- 验证标准：报告 FULL-193 或经功效论证的等价设计；在预先冻结的阈值下检查校正后 ΔSCE 区间是否排除0，并公开两层 false-accept/false-reject 计数。
- 仍需证据：完整分层人工标签、抽样权重、盲法/一致性记录与校正后区间。
- 预期影响：high；判断置信度：high。

##### A32-R3-04 · 次要 · 清晰度、限制与负责任表述

- 位置：摘要、贡献列表与第2节对三臂设计的表述（PDF第1、3–4页）
- 观察证据：Random 臂的0–55 token 剂量按预先声明 seed 抽取，但 Short 与 Long 是每题固定的0与37 token 操作。全文多处把整个设计简写为“randomized within-item experiment”。
- 重要性：这种表述可能让读者误以为 Short/Long 处理分配本身随机化并识别平均处理效应；实际最强识别来自同题固定对比，Random 臂才有随机剂量。
- 必需修复：将主设计称为预先声明的同题配对干预，并单独说明 Random 臂的剂量随机化；给出臂顺序、seed 与是否存在顺序效应。
- 验证标准：逐处核查摘要、贡献和结论，确保“randomized”仅修饰确有随机分配的部分，且方法表可重建分配过程。
- 仍需证据：修订后的设计措辞和完整分配说明。
- 预期影响：medium；判断置信度：medium。

##### A32-R3-05 · 次要 · 实验严谨性、可复现性

- 位置：表1注释和第5节设置（PDF第4、7页）
- 观察证据：无语义填充的有效性门只在100题、200个比较上报告0次语义漂移；正文没有充分说明这100题相对最终400题的抽样方式、审阅者盲法和判定一致性。
- 重要性：若有效性子集经过便利选择或审阅者知道处理臂，0/200不足以排除填充改变解析、格式或答案条件的系统风险。
- 必需修复：补充预先冻结的抽样框、盲法、判定规则和审阅者一致性；最好对400题随机概率子样本做独立复核。
- 验证标准：由不了解模型结果与处理标签的第二审阅者复核随机子样本，报告漂移率上界和一致性；若发现漂移，按预先规则重算主结果。
- 仍需证据：有效性门的逐项标签、抽样记录和独立复核统计。
- 预期影响：medium；判断置信度：medium。

**给作者的问题：**

- 各题各臂的生成是否在给定 seed 下完全确定？若不是，为什么每题每臂仅一次生成足以让题目簇 bootstrap 覆盖解码随机性？
- 能否对“观察性 ΔSCE 与 Long−Short ΔSCE 的差异”构造一个共同重采样的直接对比区间，而不只比较一个点估计是否落在另一个区间内？
- 37-token 冗余填充与自然长度差异在语义、格式和难度上完全不同；作者愿否加入内容承载但任务等价的长度操作，以检验更接近自然长度变化的机制？
- MATH500 的30个锚点如何抽取，为什么称为 purposive；已知隐藏 false accept 后，何时会完成预先声明的 accepted/rejected 双层概率抽样？
- 100题、200个比较的语义有效性门是否与最终400题随机独立，且评审者是否对处理臂和模型输出盲法？

**能提高评分的证据：**

- 多独立解码 seed 的层级复现显示同题零效应区间稳定，并预先给出有实践意义的等价界限。
- 在至少一个额外模型家族和额外自然任务上预先注册并复现观察分解与机制对照。
- 完成 MATH500 accepted/rejected 双层概率样本的人审填充，使跨床校正区间获得可解释的覆盖。
- 加入内容承载但任务等价的长度操作，并以联合 contrast-of-contrasts 正式检验观察关联与干预效应差异。

**会降低评分的证据：**

- 跨解码 seed 后 Long−Short ΔSCE 的符号或区间显著不稳定。
- 扩大人工锚点后 MATH500 judge 出现足以改变主结论的系统性 false accept/false reject。
- 发现填充文本改变题意、解析路径或答案格式，且这种漂移与主效应相关。
- 复算发现题目级配对、聚类 bootstrap 或多重性校正没有按文中描述执行。

**伦理标记：** 否。论文使用公开数学基准与模型输出，未见人类受试者、个人数据或部署伤害问题。人工锚点属于结果判定而非受试者研究；仍应说明标注者来源与质量控制，但这不构成本审稿中的实质伦理警报。

**LLM 使用披露：** 本审稿由隔离运行的 AI 子代理 R3 完成，仅用于内部投稿前质量控制；未与其他评审通信，未读取作者计划、旧评分或其他评审输出。

**评审限制：**

- 仅审阅所给冻结 PDF；按隔离要求未访问随附代码、CSV、运行工件或源文件，因此 PDF 中的哈希、预注册时序、逐项配对与可复现性声明未做外部执行核验。
- 未联网，未核验外部引文、相关工作优先权、模型/基准版本或公开讨论；相关不确定性不应被解释为已验证。
- 已逐页视觉核查12页 PDF，未发现影响阅读的图表截断或公式渲染缺陷。

### C1

#### 交付与来源状态

- 选定输入：`/Users/liuhanzuo/Downloads/C1-paper (1).pdf`
- 源状态：`exact_latex_snapshot_assets_repin_required`
- 润色 PDF：`/Users/liuhanzuo/MacLLM-Bench/output/pdf/paper_polish_20260826/C1-polished.pdf`
- 冻结 SHA-256：`319da71b563edd52d9f4d3e6303bcedcc188048744461a33c9d87c7bebb866dc`
- 总页数：28；主文状态：主文在第9页结束。
- 版面核验：pass；构建：pass。
- 旧评分基线：4,6,6；旧中位数：6。

#### 三评量化结果

| 审稿人 | 角色 | Overall | Confidence | 建议 | Soundness | Presentation | Contribution | 当前上限 | 必需修改后预测 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| R1 | 新颖性与定位 | 4 | 4 | 略低于接收线 | 3 | 2 | 2 | 4 | 6 |
| R2 | 技术正确性 | 6 | 4 | 略高于接收线 | 2 | 2 | 3 | 6 | 8 |
| R3 | 实验严谨性 | 4 | 4 | 略低于接收线 | 2 | 2 | 2 | 4 | 6 |

三评中位数为 **4**，均值 4.67，跨度 2，接收侧票数 1/3。

#### 编辑记录

- [结构审计](work/C1/structure_audit.md)
- [语义锁](work/C1/semantic_lock.md)
- [修订日志](work/C1/revision_log.md)
- [待核验事项](work/C1/needs_verification.md)

**修订日志原文：**

> # C1 revision log
>
> ## Scope
>
> Evidence-preserving manuscript polish. No experiments, independent citation verification, or reviewer scoring were performed.
>
> ## Source changes
>
> - Rewrote the abstract to distinguish the arm-independent best-constant reference (v1) from the arm-conditional permutation null (v2), while retaining the headline cell counts and precision qualification.
> - Tightened the introduction's motivation and positioned the floor as a one-directional reporting check rather than a validity certificate.
> - Reworked the discussion into a concise reporting rule and bounded its empirical claim to the measured damage ladder.
> - Clarified that regime, family, depth, corpus, and exposure are confounded, preventing a universal damage-law reading.
> - Added a safe line-break opportunity in the off-MMLU roll-up table to remove an overfull box without changing its content.
>
> ## Semantic safeguards
>
> - Preserved all counts, uncertainty standards, table and figure content, citation keys, experimental states, and retraction-history references.
> - Did not convert the unrun selection-aware joint bootstrap into completed evidence.
> - Added no claim that passing either null establishes construct validity.
>
> ## Verification
>
> - Built from the isolated source with shell escape disabled, using BibTeX and two final LaTeX passes.
> - Checked extracted text, current citation/reference warnings, main-text boundary, and rendered pages.
> - Remaining source-dependent items are listed in `needs_verification.md`.

**待核验事项原文：**

> # C1 needs verification
>
> - Priority/novelty language for the item-varying option-count permutation construction was not independently checked against the full literature.
> - The selection-aware joint bootstrap is explicitly not run; the manuscript's plug-in/Wilson qualification remains binding.
> - Artifact-emitter and numeral-checker claims were preserved but not re-executed because their evidence/code bundle was not part of the isolated source.
> - The main text uses “gate” in legacy identifiers and figure labels; prose now distinguishes those names from scientific validity.

#### R1（新颖性与定位）完整评议

**论文概述：** 论文主张多选评测中的 nominal chance 可能低于一个输入盲预测器可达到的基线，因此提出两个零参考：v1 是由经验金标签边际得到的、与模型臂无关的最佳常数 floor，用于接口可比性筛查；v2 是在可变选项数分层内保持模型预测边际的置换零假设，用于测量逐题信息。实验覆盖 MMLU-Pro、MMLU/BoolQ 及若干小型多选基准，在 OLMo-2 剪枝后愈合与 Llama/Qwen 前 N 层截断模型上比较 chance 与 floor。点估计层面 advertised above-chance 单元大幅减少，但匹配区间后差距很小，且多重校正后两侧均为 0/12；v2 还撤回两项 below-floor competence 标签。

**最强的已核实贡献：** 第 5.1 节表 1 把 chance 与最佳常数 floor 放在同一配对区间和多重校正标准下，显示 MMLU-Pro 非 OLMo 单元从点估计 10/12 对 3/12，收缩到区间 3/12 对 1/12，并在 BH/Bonferroni 后均为 0/12；这种对称证据标准纠正了更强但不成立的早期叙述。

**维度理由：**

- Soundness：v1 的最佳常数基线和 v2 的分层置换零假设定义清楚，作者也修复了 variable-option 非法标签抽样、对称化了 chance/floor 的区间标准，并明确报告多重校正后 0/12 个单元仍显著。核心描述性事实可信，但 v1 区间条件化于观察到的获胜字母且未做选择感知重采样，v2 对现有 27 个单元是看到塌缩形态后的事后重分析，家族、损伤方式与修复暴露又相互混杂。
- Presentation：主张边界总体诚实，关键表格包含点估计、区间和多重校正三层读法。然而 28 页中充斥 repaired/withdrawn/sealed、多个 judge tier、旧/新统计族和大量完整单元表，主线需要跨主文与多个附录才能重建；这已实质阻碍读者判断哪个结果具有确认性。逐页视觉上无裁切，但表格字号与信息密度很高。
- Contribution：“输入盲基线应匹配输出接口”是有用的报告原则，variable-k 分层和三段式 gate 也有实务价值。不过最佳常数分类器、Cohen's-kappa 等价的观测减置换期望以及 option-count 校正均被论文自己承认为已有成分；当前创新主要是把它们组合成报告协议并在一个结构损伤语境中展示，增量较窄。

**优点：**

- 明确区分 arm-independent 的接口 floor 与 arm-conditional 的 item-information null，并说明二者都不证明 construct validity。
- 主动披露 variable-option 校准错误、事后 v2 状态、赢家选择条件化和多重校正后的零结果。
- 实验协议列出模型、损伤方式、候选评分、精度、截断检查、分片完整性和统计种子，PDF 内可追踪性较强。
- 负面对照、常数发射器和 fp32 tie 检查帮助区分接口失败、数值精度与残余任务信息。

**问题与可验证修复：**

##### C1-I1 · 主要 · 新颖性、重要性、引文完整性

- 位置：第 2–4 页第 2 节 Related Work、第 3.1–3.2 节；第 9 页第 6 节 Discussion
- 观察证据：论文明确承认多数类基线、constant-rater property、kappa 的分子形式及 option-count correction 均非新概念；当前差异主要是 variable-k 分层、materiality bar 和在比较前设置 gate。相关工作以段落列举为主，没有展示这些成分与最接近协议在输入、零分布、决策输出和新增能力上的不可替代差异。
- 重要性：如果核心是已知统计量的重新命名与组合，方法新颖性和 ICLR 影响力不足。实际价值可能来自审计案例，但这需要被定位为经验/报告贡献而非新的 null-calibration 原理。
- 必需修复：增加逐轴最近工作比较，明确每个组件是复用、修改还是新增；把 v1/v2 的数学新颖性主张收窄，并用一个现有方法会做出错误决策、而完整协议能预先避免的外部案例证明组合价值。
- 验证标准：相关工作应可让审稿人对每个公式找到最接近先例，并能用一张表唯一识别本文新增的输入、假设、输出与验证证据；摘要和结论不再把组合包装成全新统计原理。
- 仍需证据：完整的邻近方法对照；若主张 gate 本身显著更好，需要在预先定义的外部评测上比较错误决策率或覆盖率。
- 预期影响：high；判断置信度：high。

##### C1-I2 · 主要 · 实验严谨性、技术正确性、限制与负责任表述

- 位置：第 6 页第 4 节 Designated damaged cells and multiplicity；第 7 页第 5.1 节表 1 及其后两段；第 12 页 Reproducibility Statement
- 观察证据：标题性单元计数采用每单元 α=0.05 且不做族内校正；MMLU-Pro 的匹配区间只剩 3/12 对 1/12，多重校正后两侧都是 0/12。最佳常数 floor 的 bootstrap 又条件化于观察到的获胜字母，论文说明因缺少 per-item shards 未运行 selection-aware joint resample。共享题目与嵌套损伤臂使单元计数并非独立重复。
- 重要性：“校准后大幅减少可宣称单元”的强度主要由点估计计数驱动，而正式单元证据在校正后消失；未处理赢家选择与单元依赖会进一步低估边界不确定性。描述性方向可以保留，但不能承担强确认性主张。
- 必需修复：把预先定义的家族/数据集层总体 estimand 设为主结果，采用保持题目与嵌套臂相关结构的联合重采样或层级模型，并纳入 floor winner 的重新选择；在做不到时，将点计数明确降为探索性并从标题性论据中移除。
- 验证标准：主结果必须有单一预设总体检验/区间，重采样中每次重新选择最佳常数，并以模型家族或独立训练实体而非单元行作为适当的推断单位；报告调整前后结论是否改变。
- 仍需证据：逐题预测与金标签、完整损伤臂层级、选择感知联合 bootstrap/permutation 输出及预设总体检验。
- 预期影响：high；判断置信度：high。

##### C1-I3 · 主要 · 实验严谨性、新颖性、限制与负责任表述

- 位置：第 4–5 页第 3.2 节 Pre-registration status；第 7–9 页第 5 节；附录 A.9 第 23–24 页
- 观察证据：v2 的规则、分层和 0.10 materiality constant 虽在重新判定前冻结，但设计者已看过现有 27 个单元的塌缩形态；论文据此也称现有 v2 分析为 post-hoc。家族 materiality bar 依赖同一家族 intact anchor，Llama-2 因 anchor 过弱而整体被阻断。
- 重要性：v2 是论文最有区分力的统计层，但当前数据既参与了问题发现又用于展示修复效果；materiality 决策还会随一个噪声较大的完整模型锚点变化。它目前是有根据的诊断，而非外部验证。
- 必需修复：在未用于设计 v2 的新模型×数据集单元上冻结并执行协议；预先规定 materiality anchor 失败时的处理，并比较绝对效应阈值、同家族相对阈值等合理选择的稳健性。
- 验证标准：新评测单元在任何结果可见前完成时间戳冻结；论文分别报告预注册确认集和旧 27 单元探索集，且所有阈值敏感性遵循事前规则。
- 仍需证据：真正 held-out 的多家族多数据集 v2 结果，以及阈值/锚点协议的冻结记录。
- 预期影响：high；判断置信度：high。

##### C1-I4 · 主要 · 实验严谨性、重要性、限制与负责任表述

- 位置：第 6 页第 4 节 Models and structural damage；第 7 页第 5.1 节末；第 9 页 Discussion
- 观察证据：OLMo-2 使用 prune-then-heal checkpoint，而 Llama-2/Llama-3/Qwen3 只是运行时前 N 层截断；同时家族、深度、训练语料、愈合暴露和保留比例都不同。所有 interval-level off-MMLU clearance 都来自唯一经过愈合的 OLMo-2 家族。
- 重要性：这个模式无法区分“愈合使模型高于 floor”“OLMo 家族特性”“损伤强度不同”或数据适配。它可以描述当前 ladder，但不能验证协议在多种实际压缩/修复系统上的价值。
- 必需修复：至少在两个家族上建立相同损伤与相同愈合预算的交叉设计，或将结果彻底改写为 OLMo ladder 的案例研究，不再用跨家族模式暗示一般性。
- 验证标准：交叉设计应报告家族×损伤制度×深度交互及同预算配对；若不新增实验，摘要、结果和结论均必须只陈述 measured ladder。
- 仍需证据：匹配的跨家族 prune/heal 或截断控制、多个独立 checkpoint/seed，以及相同评测协议。
- 预期影响：high；判断置信度：high。

##### C1-I5 · 次要 · 清晰度、可复现性

- 位置：第 1–9 页主文与第 13–28 页附录，尤其表 15–23
- 观察证据：同一单元的 v1/v2、点估计/区间/多重校正、旧/修复 judge 和多个 tie convention 分散在主文及长附录中；大表虽未视觉溢出，但字号和密度使结果层级难以快速审计。
- 重要性：协议论文的核心价值是让错误参考更难被误用；如果论文自身需要跨十余张表才能确定可引用结果，就削弱了这一目标。
- 必需修复：增加一张不含历史谱系的最终结果矩阵，按研究问题列出唯一主 null、推断单位、校正方法、结论与限制；将修复史和完整单元表留在附录。
- 验证标准：独立读者只凭摘要、方法定义、最终矩阵和结论即可复现所有标题性计数并知道哪些数值是探索性。
- 仍需证据：不需要新实验；需要结果层级和交叉引用重构。
- 预期影响：medium；判断置信度：high。

**给作者的问题：**

- 当一个非恒定预测器低于经验最佳常数 floor、但相对 arm-conditional 置换零假设显著为正时，v1 gate 为什么应阻断臂间比较而不只是阻断绝对能力表述？
- 能否给出与 Balepur majority baseline、kappa/options correction 和 null-model gaming 的逐项算法比较，明确 variable-k 分层与前置 gate 中哪一项是新的？
- 表 1 的 aggregate direction 在共享题目、嵌套深度和同家族单元相关的情况下，预先规定的总体 estimand 和有效独立单位是什么？
- 若在至少两个模型家族都使用相同的 prune-then-heal 或相同训练预算，当前“只有 OLMo healed arms 过 floor”的模式是否仍成立？

**能提高评分的证据：**

- 在真正 held-out 的模型×数据集单元上预注册验证 v2，并展示相对 chance/constant baselines 的稳定决策增益。
- 用选择感知、保持题目与嵌套臂相关结构的总体推断确认 headline direction。
- 在至少两个模型家族上运行匹配的损伤与愈合制度，解除 OLMo-only 解释。
- 明确证明完整 gate 相对最接近既有 null/baseline 协议新增了可验证能力。

**会降低评分的证据：**

- 选择感知或依赖感知分析使 chance 与 floor 的总体差异消失或反向。
- held-out v2 评测显示高假阳性/假阴性，或 materiality 结论对合理阈值高度不稳定。
- 匹配损伤制度后 OLMo 与非 OLMo 的模式不再成立，且论文仍保留一般性叙述。

**伦理标记：** 否。未见人类受试者、敏感数据或直接部署决策；主要风险是错误的能力标签被下游误用，论文已部分通过限定语言缓解。

**LLM 使用披露：** 本审稿由隔离的 AI 子代理 R1 生成，仅用于内部投稿前质量控制；该子代理只读取冻结 PDF 与指定审稿规则，未与其他评审通信，也未读取其他评审输出。

**评审限制：**

- 本评审严格只读取指定冻结 PDF；未读取代码、逐题 shard、artifact、源文件或任何旧稿/修订日志。
- 按任务要求未联网，未核验外部引文或进行补充文献检索；新颖性判断仅相对于 PDF 自身列出的相关工作。
- 逐页视觉核查覆盖全部 28 页；未发现裁切、重叠或缺页，但若干附录表字号和信息密度较高。

#### R2（技术正确性）完整评议

**论文概述：** 论文指出多选评测中“高于 nominal chance”可能仍不如输入盲常量预测器。v1 用观测 gold-label 边际的最佳常量作为跨 arm 的接口 floor；v2 在 option-count strata 内置换某一 arm 的预测向量，用于检测超过其边际匹配的信号。作者在 MMLU-Pro 及五个较小基准、四个模型家族/损伤梯级上展示：按 point estimate，高于 chance 的 damaged cell 远多于高于 floor 的 cell；v2 又重排部分 v1 判断。论文同时修复了 MMLU-Pro 非法标签 null、bootstrap p 值、截断和 OOM 等缺陷。

**最强的已核实贡献：** 第 3–5 页式 (2)–(5) 对两个问题的分离最有价值：v1 的 best constant 是所有 arm 共用、确实可在观测题集上实现的比较基线；v2 的 option-count-stratified permutation 在其定义域内使合法的纯常量预测向量得到零增量。这解释了为什么 nominal chance 能给空预测器虚假正分，而 floor 不会。

**维度理由：**

- Soundness：把 nominal chance、观测 best-constant floor 与 arm-conditional permutation null 分开的核心逻辑成立，且合法选项数修复是重要的自我纠错。但 v1 的主区间固定了在同一标签样本上选择出的获胜字母，没有在重采样中重做 max；v2 仅按 option count 分层，因而排除的是该条件下的全局边际匹配，而不是所有非题目级信息。这些缺口实质限制了 headline inferential labels。
- Presentation：定义、作用域、修复史和逐单元表非常完整，但 28 页中多套 null、校准面、materiality gate 与大量撤回台账反复交叉引用；若干表字号很小，主线被审计细节淹没。
- Contribution：要求 MC 构造在比较前报告可实现的 input-blind reference 是实用贡献；可变选项数内的预测边际置换也是有用的操作化。best-constant 和 chance-corrected agreement 本身并非新统计量，增量主要是将它们组合成报告协议并量化对损伤模型结论的影响。

**优点：**

- 对 MMLU-Pro 的初始 legality-blind balanced null 明确撤回，并用逐题合法选项集合重算 E[f̂] 与 p=0.083；修复方向和下游不变项均说明清楚。
- 区分 arm-independent 接口可比性与 arm-conditional 信息检测，避免把通过某一个 reference 误称为构造有效性证明。
- 公开 multiplicity、power、regime/family confounding、post-hoc v2 规则、精度与 tokenizer/truncation 修复，负结果处理较诚实。
- 完整报告 17 个 MMLU-Pro damaged cells、27 个 v2 cells 与 60 个 off-MMLU cells，使异常值和边界项可见。

**问题与可验证修复：**

##### C1-R2-01 · 主要 · 技术正确性、实验严谨性

- 位置：第 5 页 §3.2 Reference-choice reporting rule；第 7 页表 1；第 13 页 Reproducibility statement；第 23 页表 15
- 观察证据：v1 floor 是 f_const=max_L m_L，但 paired bootstrap 固定观测样本选出的 winning letter；论文明确说 selection-aware joint resample 未运行，因为逐题 letter shards 缺失。主文仍用这些区间形成 3/12、1/12、9/85 等 interval-level counts。
- 重要性：若推断对象是从题目总体抽样后的 best-constant floor，max 的选择必须在每次重采样内重做；固定赢家忽略选择不确定性并可能低估或错估 ∆floor 的方差。若对象只是这个固定题集，模型 accuracy 与 floor 都是确定量，bootstrap 的统计解释又需重新界定。
- 必需修复：恢复逐题 gold/prediction shards，在每次联合重采样中重选常量标签并重算差；在完成前将区间级 headline counts 降为条件于获胜字母的敏感性，不作正式 verdict。
- 验证标准：比较 fixed-winner 与 reselect-winner bootstrap 的全部边界 cell；报告覆盖模拟或至少 verdict-change table，并以选择感知结果更新表 1/15。
- 仍需证据：逐题标签与预测记录、联合重采样代码、重算区间和结论差异。
- 预期影响：high；判断置信度：high。

##### C1-R2-02 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第 4–5 页 §3.2 式 (3)–(5)；第 23–25 页表 17–18
- 观察证据：v2 只在 n_opt strata 内置换预测，保持每个 strata 的整体预测边际。它没有条件化 subject、题型、选项模板或其他可能同时影响模型 letter bias 与 gold marginal 的组变量。
- 重要性：正的 ∆perm 只能证明预测与 gold 的关联超过 n_opt 条件下的整体边际匹配；它仍可能来自 subject-level 或格式级信息，而非单题内容信息。因此“does this arm carry item-level information”比该 null 实际识别的量更强。
- 必需修复：把结论改为“超出 n_opt-stratified marginal alignment”；或预先定义并加入关键 nuisance strata/条件随机化模型，验证 v2 信号不是组级标签先验。
- 验证标准：在 subject×n_opt（以及可行的模板/合法标签集）内重新置换，检查 27-cell verdict 与 materiality 是否保持；稀疏 strata 需报告合并规则。
- 仍需证据：逐题元数据、分层置换结果与对 strata 选择的预先说明。
- 预期影响：high；判断置信度：medium。

##### C1-R2-03 · 主要 · 实验严谨性、技术正确性

- 位置：第 5 页 materiality 与 preregistration status；第 7–9 页 §5；第 19 页附录 A.4–A.5；第 24–25 页表 17–18
- 观察证据：v2 在 27 个已评分 cell 之后设计；作者已看过 collapse 形状。0.10 相对恢复阈值及同家族 intact anchor gate 决定只有一个 damaged cell 为 material；c=0.05 时有四个。v1 的 85-cell headline 又由相关、嵌套 cell 的未校正逐单元决定组成，正式 BH/Bonferroni 后两参考均无 matched-standard cell。
- 重要性：阈值选择和 family definition 可实质改变叙事；把相关 per-cell crossing 的集中性作为 load-bearing aggregate evidence，没有一个预先固定的联合零分布或 family-level uncertainty 支持。
- 必需修复：把 v2 定位为探索性重分析，并报告全阈值曲线而非二值 material 标签；为未来/新数据预注册阈值、family 与聚合统计，并用 cluster-aware 或层级方法控制共享题目/嵌套 arm。
- 验证标准：在冻结规则的新模型/新基准上运行；主聚合检验的 family、阈值和方向须在观察前固定，并报告 simultaneous uncertainty。
- 仍需证据：预注册文件、新的独立 cell 集或严格的选择敏感性分析。
- 预期影响：high；判断置信度：high。

##### C1-R2-04 · 次要 · 实验严谨性、清晰度

- 位置：第 6–9 页 §4–6；第 13–14 页 Limitations 与表 4；第 21–22 页表 13
- 观察证据：只有 OLMo-2 采用 prune-then-heal，其他三家族是 evaluation-time truncation；语料、训练暴露、相对深度和模型家族均不匹配。所有 off-MMLU interval clears 都集中于 OLMo-2 的高层数 healed arms。
- 重要性：该集中性不能区分 healing、family、relative depth 或训练数据；读者容易把一个特定 ladder 的描述读成损伤恢复规律。
- 必需修复：将结果严格命名为 OLMo-2 ladder 描述，避免跨 regime 比较；若要机制或 family 结论，需在至少两个家族上做匹配的 prune-then-heal 对照。
- 验证标准：同 retained fraction、语料/steps 与评分协议的第二家族 healing curve 复现后，再检验 family×regime 交互。
- 仍需证据：匹配的跨家族 healing 实验或更窄的文字。
- 预期影响：medium；判断置信度：high。

##### C1-R2-05 · 次要 · 清晰度

- 位置：全文，尤其第 20–27 页表 12、15、17、18、21、22
- 观察证据：主结论散落于 28 页、多套编号和极密集小字号表中；正文同时保留当前结论、敏感性面、修复史和 16 条撤回台账。
- 重要性：即使数字可追踪，审稿人也很难区分正式 estimand、条件敏感性、历史修复和描述性计数，增加误读和无法快速核验的风险。
- 必需修复：主文只保留两个 estimand、一个主统计表和一个限制表；把完整修复史与逐 cell 网格移入补充材料，并增加单页 claim→estimand→valid inference 映射。
- 验证标准：让未参与项目的读者仅用主文准确复述 v1/v2 的对象、正式推断和不可推断项；同时检查表字号。
- 仍需证据：重组后的 PDF 和独立可读性检查。
- 预期影响：low；判断置信度：high。

**给作者的问题：**

- 为什么 v1 的 paired bootstrap 不在每个重采样内重新选择 argmax gold letter？若 floor 是超总体函数，这一步不可省；若目标只是固定题集，为什么还把 bootstrap 区间解释为推断？
- v2 所谓 item-level information 如何排除 subject/domain、选项模板或其他可见元数据造成的组级预测—答案关联？是否做过 subject×n_opt 或其他预定义 nuisance strata 的置换？
- materiality cutoff 0.10×同家族 intact-anchor recovery fraction 的决策理论依据是什么？表 18 在 c=0.05 时 damaged material cells 从 1 增至 4，为何把 0.10–0.25 称为 intended range？
- 能否提供一个预先固定、选择感知且控制家族的聚合检验，替代 85 个相关 per-cell 决策的描述性计数？

**能提高评分的证据：**

- 以每次重选 best constant 的联合 bootstrap 更新 v1 headline，且结论在边界 cell 上稳定。
- 在 subject/format 等关键 nuisance strata 下 v2 信号仍成立，并在新 cell 集上按预注册 materiality 规则复现。
- 增加第二家族匹配的 prune-then-heal 实验，或彻底移除跨 regime 的机制暗示。

**会降低评分的证据：**

- selection-aware bootstrap 使主要 interval-level floor counts 大幅变化。
- 加入 subject×n_opt 条件后 v2 的 surviving damaged signal 消失，表明原信号仅是组级边际。
- 发现 legality-aware null 或逐题合法标签集合与实际 scorer 不一致。

**伦理标记：** 否。未见人类受试或部署伦理问题。潜在风险是把校准 gate 当成构造有效性认证，论文已明确“necessary, not sufficient”，修订时应保持。

**LLM 使用披露：** 本审稿由隔离的 AI 子代理 R2 生成，仅用于内部投稿前质量控制；未与其他评审通信，未读取其他评审结果、作者计划、编辑上下文或历史评分。

**评审限制：**

- 遵循隔离要求未联网，未核验外部引文、优先权、模型/基准官方实现或 2026 年文献。
- 仅审阅冻结 PDF；未读取或执行论文所述 evidence pack、代码、逐题 shards、修订记录或其他 reviewer 输出，因此作者报告的 hash、重算和修复影响未外部验证。
- PDF 共 28 页，已全文阅读并逐页视觉核查；表 12、15、17、18 等字号很小但可放大阅读，未见页面截断。

#### R3（实验严谨性）完整评议

**论文概述：** 本文主张多选评测不能只以名义随机机会为基线。v1 使用对整个构造固定、输入盲的最佳常量字母 floor；v2 在每个选项数层内置换 gold，使零期望保留具体模型臂的预测边际，并指出 Δperm 等于 κ 的分子。MMLU-Pro 中，非 OLMo 损伤臂从10/12个点估计高于 chance 降至3/12个高于 floor；按95%区间为3/12对1/12，做 BH/Bonferroni 后两边均为0/12。五个较小基准的85个指定损伤 cell 中，只有9个在区间标准上高于 floor，且都来自 healed OLMo-2。论文还公开撤回了对 MMLU-Pro 使用非法十字母均匀零假设的早期读法，并将27-cell v2 结果标为在观察形状后预注册的规则。

**最强的已核实贡献：** 最扎实的结果是第3节的构造分解与表1的 matched-standard 重算：同一批 MMLU-Pro cell 在完全相同的点估计、区间和多重性标准下，从 nominal chance 换成最佳常量 floor 后，许多“高于机会”标签消失；经 BH/Bonferroni 后非 OLMo 的12个 cell 在两种基线下都没有显著清除。这证明参考基线选择可以实质改变未经校正的叙述，同时也限制了作者能宣称的显著结果。

**维度理由：**

- Soundness：v1 最佳常量 floor 与 v2 分选项数置换零假设在代数上定义清楚，MMLU-Pro 非法 A–J 均匀零假设也被主动撤回并修复。可是主计数依赖大量相关的逐 cell 未校正判断，v1 bootstrap 固定了数据选择出的 winning letter，v2 阈值又是在看到塌缩形状后冻结；当前证据仍属探索性。
- Presentation：论文极其透明地保留修复账本、敏感性与失败前提，但28页中大量版本、阈值、零假设和表格来回交叉引用，使最核心的研究问题与主分析被审计细节淹没。图表本身清晰可读。
- Contribution：把 nominal chance 与构造特定、可实现的 input-blind floor 区分开有实践价值，arm-conditional permutation 也给出一个有用诊断。但其统计量与 κ/置换零假设关系已知，当前新意主要在应用与报告规则；对“healing”或广泛损伤规律的实证支持不足。

**优点：**

- 第3节明确区分 arm-independent v1 与 arm-conditional v2，且证明常量预测器在 v2 下恰为零，避免混用两个不同问题。
- 发现 MMLU-Pro 选项数可变后，作者撤回 A–J 均匀零假设，并在表7、表8报告合法性修复及方向变化。
- 表1把 chance 与 floor 放在同一证据标准下，另列95%区间和 BH/Bonferroni，避免只比较不对称的点估计计数。
- 论文主动说明 v1 paired bootstrap 固定 winning letter、当前缺少 selection-aware 联合重采样，并将此列为后续工作。
- 局限性明确承认 OLMo 是唯一 heal 家族，family、regime、depth、corpus 与 exposure 相互混杂。

**问题与可验证修复：**

##### C1-R3-01 · 主要 · 实验严谨性、技术正确性、限制与负责任表述

- 位置：第4节“Designated damaged cells, and multiplicity”与第5节表1（PDF第6–8页）
- 观察证据：论文突出17个 MMLU-Pro 与85个 off-MMLU 指定损伤 cell 的逐 cell 清除计数；这些分析共享同一批题目、模型家族和嵌套深度。表1显示在12个非 OLMo MMLU-Pro cell 上实施 BH/Bonferroni 后，chance 与 floor 两边均为0/12。
- 重要性：未经整体误差控制的“多少 cell 清除”会随阈值、依赖结构和样本精度显著变化，不能支撑整体方向或跨基准普遍性。逐 cell 计数的集中现象也不是一个预先定义的全局检验。
- 必需修复：预先指定一个全局或层级估计目标，以 item 为共享重采样单位并对模型/深度嵌套建模；报告 family-wise 或 FDR 控制后的结论，并把未经校正计数降为描述性。
- 验证标准：对全部指定 cell 进行共享 item 的联合 bootstrap/置换或层级模型，检验 floor 替换导致的总体标签/效应变化，并在预先声明的错误率下复核结论。
- 仍需证据：逐 item、逐 cell 预测及统一的依赖感知多重性分析。
- 预期影响：high；判断置信度：high。

##### C1-R3-02 · 主要 · 技术正确性、实验严谨性

- 位置：第3.3节 reference-choice rule 与第5节表1后说明（PDF第5、8页）；复现性说明（PDF第11页）
- 观察证据：v1 floor 取经验 gold marginal 中占比最高的字母，但 paired bootstrap 条件于观测到的 winning letter；论文承认 selection-aware joint resample 未用于主结果。
- 重要性：最佳常量是由同一数据选择出的最大值。固定胜者会忽略字母胜者在重采样中的切换和 max 操作的不确定性，尤其会影响紧邻 floor 的 cell，并可能让区间偏窄。
- 必需修复：在每次以 item 为单位的联合重采样中重新计算所有 gold letter 比例、重新选择最大常量，再与同一重采样中的模型准确率做差。
- 验证标准：对表15所有 MMLU-Pro cell 输出 selection-aware 区间，并列出相对当前 verdict 发生变化的 cell；主结论应以新分析为准。
- 仍需证据：逐 item gold/prediction 记录和每次重选 winning letter 的联合 bootstrap 结果。
- 预期影响：high；判断置信度：high。

##### C1-R3-03 · 主要 · 实验严谨性、技术正确性、限制与负责任表述

- 位置：第3.3节 v2 materiality rule（PDF第5–6页）与第5节表2、图2（PDF第8–9页）
- 观察证据：0.10 materiality bar 及相关前提是在作者已经看到 collapse defect 的形状后冻结；27个现有 cell 也已存在。论文坦率地将其称为对规则的预注册，而非对当前数据的前瞻验证。
- 重要性：在已见数据上选择阈值和判定结构会把当前 re-analysis 保留为探索性证据；即使代码与规则随后冻结，也不能恢复对同一 cell 的前瞻错误率解释。
- 必需修复：将当前27-cell结果明确标为 hypothesis-generating，并在完全未查看的模型、深度或基准上前瞻冻结阈值、family 与主结局。
- 验证标准：对独立留出集合只运行一次预注册流程，报告所有 cell 与统一多重性校正；在揭盲前验证样本量和锚点前提。
- 仍需证据：带时间戳的前瞻协议及独立验证数据结果。
- 预期影响：high；判断置信度：high。

##### C1-R3-04 · 主要 · 实验严谨性、技术正确性、限制与负责任表述

- 位置：第4节模型与结构损伤（PDF第6页）、第5节 off-MMLU 结果（PDF第8–9页）及附录A局限表4（PDF第13页）
- 观察证据：OLMo-2 使用 prune-then-heal checkpoint，而其他家族只做 truncate-only；同时 family、绝对/相对深度、训练语料与 exposure 均不同。区间上清除 floor 的9/85 cell 全来自 OLMo，但不存在同家族、同深度的 healed/unhealed 因果对照。
- 重要性：该集中现象不能区分 healing、家族特性、层数或额外训练暴露。把它解释为恢复机制证据会超出设计可识别范围。
- 必需修复：将现有结果限制为描述性，或在同一基础模型、相同保留深度和相同评测条件下随机/匹配比较 prune-only 与 prune-then-heal，并控制训练 token 暴露。
- 验证标准：预先注册同家族 factorial 对照，报告逐 item 配对差异与交互项；若做不到，删除所有暗示 healing 因果效应的表述。
- 仍需证据：同家族匹配 checkpoint、训练暴露记录和配对评测。
- 预期影响：high；判断置信度：high。

##### C1-R3-05 · 次要 · 技术正确性、清晰度

- 位置：第3.1–3.3节与附录表7–9、表14–15（PDF第4–5、14–20页）
- 观察证据：对七个构造中的多数，经验最佳常量 floor 与合法平衡零假设之间的差异没有被精确识别；论文仍把观测 max floor 作为硬判定线，同时把 E[f-hat] 仅作敏感性。
- 重要性：经验 max 可以作为该固定题集上的决策规则，但若读者把它理解为总体输入盲能力参数，其抽样噪声会导致边界 cell 翻转。当前决策论与统计推断语义交织。
- 必需修复：明确区分固定题集的运营阈值与总体参考参数；同时报告 floor 的选择感知不确定性，并避免把未能区分两种参考的结果称为“校准已建立”。
- 验证标准：在主表并列固定题集判定和总体参数区间，验证所有叙述均与相应语义一致。
- 仍需证据：选择感知 floor 区间和修订后的判定语义。
- 预期影响：medium；判断置信度：medium。

**给作者的问题：**

- 为什么不在每个 item bootstrap 样本中重新计算 gold marginal 并重新选择最佳常量字母，从而直接给出 selection-aware 的 Δfloor 区间？
- 表1的17个和85个指定损伤 cell 是否在任何结果可见之前完整冻结？这些 cell 共享题目且同一模型深度嵌套，主结论如何控制这种依赖？
- v2 的0.10 materiality bar 和相对恢复前提具体在看到哪些量之后冻结；是否存在完全未查看的独立留出 cell？
- 既然 OLMo 与非 OLMo 的损伤/恢复制度不同，论文为何仍把“9/85均为 healed OLMo”作为方向性证据，而不是纯描述性观察？
- 对 MMLU-Pro 而言，最佳常量 floor 本身未显著偏离合法平衡零假设时，作者希望其扮演决策阈值还是总体参数估计；两者的误差语义如何区分？

**能提高评分的证据：**

- selection-aware 联合 bootstrap 后，核心 chance→floor 结论在预先规定的多重性控制下仍稳定。
- 在完全未查看的独立模型/基准 cell 上前瞻验证 v2 materiality rule。
- 同家族同深度的 prune-only 与 prune-then-heal 对照能够识别 healing 效应。
- 以共享 item 为单位的全局/层级分析支持跨 cell 的总体结论，而不依赖未校正计数。

**会降低评分的证据：**

- 重新选择 winning letter 的 bootstrap 使关键 above-floor cell 消失或方向不稳定。
- 独立前瞻验证中 v2 gate 的 materiality 或锚点前提不能复现。
- 审计发现 designated cell family、阈值或报告规则在结果可见后有未披露改动。
- 逐 item 重算无法复现表1、表15或多重性校正数字。

**伦理标记：** 否。研究使用公开基准和基础模型输出，未见人类受试者或敏感个人数据。主要风险是把输入盲 floor 误读为模型能力或把混杂的 healing 观察过度因果化；这属于报告边界而非独立伦理警报。

**LLM 使用披露：** 本审稿由隔离运行的 AI 子代理 R3 完成，仅用于内部投稿前质量控制；未与其他评审通信，未读取作者计划、旧评分或其他评审输出。

**评审限制：**

- 仅审阅指定冻结 PDF；按隔离要求未访问论文所述逐 item prediction、代码、checker、表格源或预注册文件，因此无法执行联合 bootstrap 或验证时间顺序。
- 未联网，未核验外部引文、相关工作优先权、模型/基准版本或公开讨论；相关不确定性不应被解释为已验证。
- 已逐页视觉核查28页 PDF，未发现图表截断；对版面负担的评价来自正文与附录的整体可读性。

### C2

#### 交付与来源状态

- 选定输入：`/Users/liuhanzuo/Downloads/C2-paper.pdf`
- 源状态：`exact_self_contained_latex_snapshot`
- 润色 PDF：`/Users/liuhanzuo/MacLLM-Bench/output/pdf/paper_polish_20260826/C2-polished.pdf`
- 冻结 SHA-256：`33380752971af33cdffd66c902f0b47bc87490a3e29c35c63920b2751cc5f13e`
- 总页数：28；主文状态：主文在第9页结束。
- 版面核验：pass；构建：pass。
- 旧评分基线：4,6,6；旧中位数：6。

#### 三评量化结果

| 审稿人 | 角色 | Overall | Confidence | 建议 | Soundness | Presentation | Contribution | 当前上限 | 必需修改后预测 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| R1 | 新颖性与定位 | 4 | 4 | 略低于接收线 | 3 | 3 | 2 | 4 | 6 |
| R2 | 技术正确性 | 4 | 4 | 略低于接收线 | 2 | 3 | 2 | 4 | 6 |
| R3 | 实验严谨性 | 4 | 4 | 略低于接收线 | 2 | 3 | 2 | 4 | 6 |

三评中位数为 **4**，均值 4.00，跨度 0，接收侧票数 0/3。

#### 编辑记录

- [结构审计](work/C2/structure_audit.md)
- [语义锁](work/C2/semantic_lock.md)
- [修订日志](work/C2/revision_log.md)
- [待核验事项](work/C2/needs_verification.md)

**修订日志原文：**

> # C2 revision log
>
> ## Scope
>
> Evidence-preserving manuscript polish. No experiments, independent citation verification, or reviewer scoring were performed.
>
> ## Source changes
>
> - Reorganized the abstract around the retrieval operating point, the measured depth-cost trade-off, and the limits of generalization beyond a frozen training-free backbone.
> - Tightened the introduction and closest-work paragraph while retaining the explicit absence of a same-bed prior-system comparison.
> - Replaced “pure-depth” and “depth only” language with “depth-varying” where the lower-band context set remains different.
> - Propagated that attribution boundary through the claim map, protocol, experiment narrative, table header/caption, and conclusion.
> - Distinguished the scored-endpoint relocation null from non-identical raw generations.
>
> ## Semantic safeguards
>
> - Preserved all reported numbers, equations, intervals, multiplicity status, sample sizes, table/figure contents, citation keys, and preregistration qualifiers.
> - Did not upgrade the E--A comparison to a fully deconfounded causal effect.
> - Added no claim of constant system memory, cache-aware-training performance, or superiority over unimplemented prior systems.
>
> ## Verification
>
> - Built from the isolated source with shell escape disabled, using BibTeX and two final LaTeX passes.
> - Checked extracted text, current citation/reference warnings, main-text boundary, and rendered pages.
> - Remaining source-dependent items are listed in `needs_verification.md`.

**待核验事项原文：**

> # C2 needs verification
>
> - The closest-work claim that prior systems do not isolate cache-depth fidelity under fixed retrieval remains literature-dependent and was not independently verified.
> - The mechanism-attribution support of closest-prior citations remains NOT_MEASURED, as the manuscript already states.
> - Evidence bundles and registered run artifacts were not supplied in the isolated source, so only manuscript-internal consistency and compilation were checked.
> - The lower-band context-set difference prevents a fully deconfounded causal reading of E--A; the revised prose keeps this limitation visible.

#### R1（新颖性与定位）完整评议

**论文概述：** 论文研究 CoMem：把长文档按块运行到 Qwen3-8B 的第 j 层并持久化 hidden states，查询时检索固定 k 个块后只重算上层。核心匹配实验在 qa1、oracle selector、同一检索包上比较 j=12 与 j=0；4k 条件下准确率为 48% 对 75%，两种输出 cap 均显示显著负差，深度 sweep 也随 j 增大而下降。五臂实验试图区分 relocation、chunk-local write 与 depth caching。系统测量显示检索式读相对完整 128k prefill 大幅降内存和时间，但在相同 6,657-token read 长度下 j=12 相对 j=0 只少约 0.3 GB；论文没有端到端 decode 收益。

**最强的已核实贡献：** 第 5.1 节表 2/5 的同样本、同检索块、同解码设置负结果：在两个 cap 下 j=0−j=12 的准确率差方向一致，4k 主单元的配对区间排除零，且深度 sweep 给出随缓存深度加深而质量下降的具体操作点；这为“缓存层数不能只按成本选”提供了可信的单床反例。

**维度理由：**

- Soundness：固定 qa1 样本、oracle selector、读长和解码设置下，j=12 相对 j=0 的质量下降方向有配对计数、精确检验、两种 cap 与多种 reader 敏感性支撑，作者也把 4k 幅度称为选择后的 magnitude 而非预设效应量。主要不足是 E2 的所谓 depth-only 切换仍改变低层表示生成时所见上下文，且没有等效性证据排除该通道；系统收益又缺少端到端 decode 与真实同类基线。
- Presentation：论文对 claim、estimand、scope、evidence 的对应关系异常细致，主图与表格视觉完整，负结果边界也较明确。另一方面，28 页包含大量 protocol matrix、reader lineage、五臂字母缩写和历史修复，重要的三件事——质量损失、深度独立成本、检索带来的系统收益——仍需读者跨多节重新拆分。
- Contribution：把 cached-depth read-out 在一个严格匹配点上测成负结果具有实用提醒价值，但 CoMem 机制是既有 retrieval、hidden-state caching 与 upper-layer recomputation 的组合，论文未实现同床最近系统基线。依据指定的 KV-cache 校准锚点，已接收工作通常用真实基线、多模型/工作负载/生命周期、明确内存核算与端到端性能证明贡献；本文只有单模型、单主要任务、无优化 kernel 和无端到端 decode，因而贡献仍属窄的测量报告。

**优点：**

- 明确把检索收益与 cached-depth 的增量收益分开，并在表 7 报告相同读长下仅 0.3 GB 的深度归因内存差。
- 使用配对样本、固定 selector 与多个 cap/reader 检查，报告精确 McNemar、区间、Bonferroni 读法及选择后幅度。
- 对单模型、oracle selector、无端到端 decode、未测索引成本、历史配对摘要和硬件差异披露充分。
- 逐页视觉检查覆盖 28 页，图表未见裁切或覆盖。

**问题与可验证修复：**

##### C2-I1 · 主要 · 新颖性、重要性、引文完整性

- 位置：第 2–3 页第 2 节 Related Work；第 12–13 页附录 A/B；第 9 页结论
- 观察证据：论文将 Layer-Condensed KV、HCache 和 KV-Direct 列为最近工作，也承认没有任何一个在同床实现；CoMem 本身由分块写入、BM25/固定选择、hidden-state store 和上层重算组成，方法独特性主要是组合与测量。最近工作比较停留在定性对象/选择/缓存类型，没有同工作负载下的质量、持久存储、峰值内存或端到端性能。
- 重要性：没有同床最近基线，就无法判断观察到的质量代价是 cached-depth 家族的普遍问题、CoMem 的具体实现问题，还是已被适配/训练方法解决的问题；也无法建立相对已接收 KV-cache 工作的实际贡献。
- 必需修复：在同一 Qwen3、数据、检索包、生成长度和硬件上实现至少一个最接近 hidden-state/KV 缓存基线，或将贡献明确限定为 CoMem 个案的负面测量；比较必须覆盖精度、持久与运行时内存、prefill、完整 decode、吞吐和 exactness/approximation。
- 验证标准：检查所有方法在相同生命周期和质量目标下计费；报告同质量或质量-成本 Pareto，而非分别引用原论文数字。若不新增基线，摘要、标题和结论不得外推到 hidden-state cache 类别。
- 仍需证据：同床最近系统实现与统一端到端测量；若无法实现，需要更强的机制证据说明为什么 CoMem 反例可代表一个方法族。
- 预期影响：high；判断置信度：high。

##### C2-I2 · 主要 · 重要性、实验严谨性、限制与负责任表述

- 位置：第 5 页表 1 的 C6/C7；第 8 页第 5.3 节；第 17 页图 3；第 26–27 页附录 H/J
- 观察证据：128k 的 71.4→18.29 GB 和 72.7→0.68 s 对比把完整上下文换成固定 k 检索读，主要收益并非深度缓存。相同 6,657-token read 下，j=0 与 j=12 的峰值仅 18.2 对 17.9 GB；深度 sweep 没有逐层内存曲线，保留的 wall-clock 不支持时间节省。decode 每 token 重建上层 pack，且 break-even 只覆盖 read/prefill、不含完整生成。
- 重要性：论文标题询问 cached-depth read-out 的成本，但最大的系统数字归因于检索/缩短上下文，而深度处理的独立实益几乎未展示。没有端到端指标，无法判断 24–36 pp 的质量损失换来了什么可部署价值。
- 必需修复：把 full-context 对比改标为 retrieval-system 上界/背景，不作为深度缓存收益；补充 matched j=0 vs j>0 的完整生命周期测量，包括写入、索引、持久存储、prefill、逐 token decode、吞吐、并发和峰值内存，并提供质量-成本曲线。
- 验证标准：所有标题性成本数字都能分解成 selection/retrieval 与 depth 两项；端到端重复查询实验在预设查询数与生成长度上直接测得总延迟和峰值，并与解析的 break-even 一致。
- 仍需证据：优化或至少代表性的完整 decode 实现、多个查询数/生成长度/并发点，以及 matched j=0 基线的同进程测量。
- 预期影响：high；判断置信度：high。

##### C2-I3 · 主要 · 实验严谨性、重要性、限制与负责任表述

- 位置：第 4–6 页第 4–5.1 节；第 20–22 页附录 F/G；第 25 页 Frozen stack and preregistration
- 观察证据：主决策只用一个 Qwen3-8B、qa1、oracle selector、n=100 和固定 k=12；4k 是有效候选中最大的显著差且 anchor 从 1k 改到 4k，论文无 selection-adjusted statistic。pinned/stripped reader 保持方向，但 answer-anchored mnt512 仅 +14 pp、p=0.0649、区间跨零；更宽 reader strip 还会改变幅度。
- 重要性：方向性反例在该床上可信，但效应大小、解析器不变性和任务/模型普遍性都未建立。选择最大差异再强调 +27 pp 容易高估可复现幅度。
- 必需修复：把 +27 pp 降为选择后示例，以跨 cap 的方向/包络为主；在新任务、真实检索与至少第二模型上预注册主要 reader、深度和上下文长度，报告层级或多重性调整后的总体效应。
- 验证标准：确认新主单元在结果可见前冻结，reader 不依赖模型输出事后调整；至少一项非 qa1、非 oracle 评测以同样 paired protocol 复现方向，并报告模型/任务交互。
- 仍需证据：多模型、多任务、真实 selector 的 held-out 配对研究和选择调整；更大 n 的 answer-anchored 验证。
- 预期影响：high；判断置信度：high。

##### C2-I4 · 主要 · 技术正确性、实验严谨性、清晰度

- 位置：第 7 页第 5.2 节五臂 A/D/R/C/E 设计；第 16–18 页附录 D/E；第 21 页表 9
- 观察证据：A 是在选中原始块上从第 0 层重算，E 则读取写入阶段在连续长文档中形成的第 12 层状态。两臂虽有相同选中原始文本和 fresh positions，但低层注意力所见 token 集与表示形成时机不同；因此 E−A 同时包含缓存深度、写时上下文化和重新拼接后的分布移位。论文用另一个 16k dense/reforward 诊断的 ≤6 pp 来约束这一差异，但该诊断不是等效性检验且并非同一五臂 estimand。
- 重要性：这使“depth-only cost”不能被唯一解释为跳过前 j 层造成的表示损失。负结果仍适用于部署操作，但机制归因过强，影响方法设计建议。
- 必需修复：把 A−E 改称为完整 operational cached-depth treatment；若要做机制归因，增加低层上下文匹配的写入/读取控制，例如让 j=0 与 j=12 在相同 token 序列与注意力掩码上生成低层表示，并单独操纵缓存边界。
- 验证标准：新的因子设计应使任意一对归因比较除单个因素外输入 token、position、mask、低层可见上下文和 scorer 完全相同，并给出等效性界限而非“不显著”来排除剩余通道。
- 仍需证据：低层上下文与位置严格匹配的因子实验、足够功效的等效性检验和中间表示诊断。
- 预期影响：high；判断置信度：medium。

**给作者的问题：**

- A 与 E 之间低层网络所见上下文不同：A 在选中包上重算，E 的 h12 在写入时可见更长的连续文档。作者如何把这一区别与“深度本身”的表示损失分开？
- 如果把 j=0 retrieval read 作为真正系统基线，j=12 在端到端多轮生成中何时能以质量损失换来可测的吞吐、延迟或容量增益？
- 为什么没有在同一模型、任务、硬件和服务生命周期下实现 Layer-Condensed KV、HCache 或 KV-Direct 中至少一个最接近基线？
- 主结论对非 oracle 检索、不同 k、不同 Qwen/非 Qwen 模型和非 qa1 任务的预先规定验证计划是什么？

**能提高评分的证据：**

- 同床实现最近 hidden-state/KV-cache 基线，并展示统一质量目标下的端到端 Pareto 优势或有价值的负结论。
- 在多模型、多任务、真实检索器上预注册复现 cached-depth 质量下降方向。
- 完成 matched j=0/j>0 的写入、存储、索引、prefill、decode、并发全生命周期测量。
- 用严格因子设计把低层上下文化、位置重定位和跳层分别识别。

**会降低评分的证据：**

- 在低层上下文严格匹配后 A−E 差异消失，表明当前负结果主要是写时上下文化混杂。
- 非 oracle 检索或不同 reader 下主方向不复现。
- 端到端实现显示 j=12 相对 matched j=0 没有可测成本优势，却仍保留较大质量损失。

**伦理标记：** 否。未见人类受试者。持久 hidden-state/text store 可能继承敏感文本的保留、删除和访问控制义务，论文第 6 节已指出但未来部署应提供实证治理方案。

**LLM 使用披露：** 本审稿由隔离的 AI 子代理 R1 生成，仅用于内部投稿前质量控制；该子代理只读取冻结 PDF、指定审稿规则及针对 C2/C3 的 ICLR KV-cache 校准材料，未与其他评审通信，也未读取其他评审输出。

**评审限制：**

- 本评审严格只读取指定冻结 PDF、官方审稿规则与指定的 ICLR KV-cache 校准材料；未读取代码、数据、artifact 或旧稿。
- 按任务要求未联网，也未核验外部引文；对最近工作的判断仅基于 PDF 的相关工作和指定校准锚点，未机械沿用任何锚点评分。
- 未实测 GPU、内存或运行时间，因此只能审查论文内报告的测量设计与核算边界。
- 逐页视觉核查覆盖全部 28 页；未发现裁切、重叠或缺页，第 28 页存在较大留白但不影响内容完整性。

#### R2（技术正确性）完整评议

**论文概述：** 论文提出 CoMem：每个 512-token chunk 仅存某一深度 j 的 residual activation，BM25 取固定 top-k 后只重算上层。论文强调外部 activation/text/index 仍随上下文线性增长。Qwen3-8B 上，同一 oracle-retrieved chunks 的 j=12 读出显著差于 j=0 全层重前向；深度扫描总体向下。服务测量显示固定短读相对 full context 有 working-set 优势，但在相同约 6.7k read length 下，j=12 相对 j=0 仅省约 0.3 GB。五臂实验尝试分解 chunk-independent encoding 与位置迁移，但保留 lower-band context-set 差异。

**最强的已核实贡献：** 第 4–6 页表 1–2与第 16 页表 5支持一个窄但可靠的负结果：在同一 100 个 qa1 样本、同一 oracle-selected chunks、相同 query/scorer/decoder 下，j=0 准确率 75% 而 j=12 为 48%（mnt160），第二生成上限为 76% 对 52%；即使放弃配对假设，两比例区间仍排除零。因此被部署的中层缓存读法不是对相同检索内容的无损替代。

**维度理由：**

- Soundness：同一检索集上 j=12 低于 j=0 的系统级方向有配对证据且在两个生成上限下稳定；但五臂实验仍未控制 lower-band 所见 context set，不能支持“depth itself”归因。主风险差的所谓 exact conditional CI 也把观测 discordance rate 当成固定量，不能作为总体 paired risk difference 的精确区间。
- Presentation：作者非常清楚地区分外部 O(context) 存储、固定 k read working set、read/prefill 与 end-to-end，并公开 selected magnitude、reader 依赖和未运行 arm。正文结构清晰，但 28 页附录与多个 run/reader/tier 仍造成较高核验负担。
- Contribution：最可信的是一个训练免费、单模型、固定 retrieval 下的负结果与测量协议。没有实现同床 prior baseline，没有 end-to-end decode/latency、索引开销或跨模型证据；深度只节省 0.3 GB 且伴随显著质量损失，因此当前知识增量较窄。

**优点：**

- 按照 KV-cache 专项标尺，论文正确区分 persistent hidden/text/index store、read-time KV working set、权重/临时峰值与 decode；没有把 fixed-k read 称为常数内存系统。
- 同一检索集 j=0 对照是真实 baseline，而不是把 prior 名称贴到作者控制上；作者明确说明未实现 Layer-Condensed KV、HCache 或 KV-Direct。
- 配对键、两个生成 cap、reader taxonomy、unpaired floor、exact McNemar b/c 和逐深度点均给出，且 selected 4k magnitude 与 pointwise multiplicity 被披露。
- 明确承认无 end-to-end latency、无每深度内存曲线、无索引延迟、无训练恢复、无多跳结论和无跨模型推广。

**问题与可验证修复：**

##### C2-R2-01 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第 4 页 §4 五臂设计；第 7 页 §5.2 与表 3；第 8 页 Limitations
- 观察证据：A 的 bottom layers 只看到 selected pack，而 E 的 h12 来自 contiguous document region，因此 E−A 除 cache depth 外还改变 lower-band context set。论文承认该 residual factor，却又写“depth itself costs”并以 16k dense/re-forward 相对 j=0 的非显著 ≤6 pp 作为 bound。
- 重要性：这不是纯 depth intervention；lower-band 接触到不同未检索上下文可显著改变 h12。不同对照上的非拒绝不能给该 confound 的大小作上界，尤其 n=100 且没有等价性设计。C8-D 的机制归因因而未被识别。
- 必需修复：把 C8-D 收缩为被部署计算路径的联合差异，不称 depth itself；或新增同 context set、position、selector、decoder 下仅改变 materialization depth 的 arm，并预先设定等价/归因分析。
- 验证标准：新 arm 与 A 只在 j=0 raw replay versus j=12 cached residual 上不同；逐样本输入 hash、各层 attention context 与 positions 必须一致。对 E−A 之外同时报告新纯对比及不确定性。
- 仍需证据：confound-free forward arm、逐层输入/位置审计和配对结果。
- 预期影响：high；判断置信度：high。

##### C2-R2-02 · 主要 · 技术正确性、实验严谨性

- 位置：第 6 页表 2；第 27–28 页附录 J.1、表 16
- 观察证据：论文对 c/(b+c) 做 Clopper–Pearson，再以 (2p−1)s/n 缩放，称为 risk difference 的 exact conditional interval。该构造条件于观测 discordance count s，并把 s/n 当作无误差尺度。
- 重要性：exact McNemar 条件检验可对 H0:p01=p10 成立，但总体 paired risk difference δ=p01−p10 同时依赖 discordance probability p01+p10。忽略这一部分的不确定性不能给 δ 提供所声称的精确覆盖。论文还把该区间作为 primary magnitude。
- 必需修复：将该区间改称 conditional discordance-imbalance sensitivity，不能叫 exact risk-difference CI；主区间使用适当的 paired risk-difference 方法（例如文中已有的 Newcombe Method 11）并清楚说明覆盖对象。
- 验证标准：通过多项分布模拟覆盖率，比较 CP-rescaled 与正确 paired-difference interval；更新所有主文区间和 selected magnitude 语言。
- 仍需证据：覆盖模拟、方法引用/推导及重算表。
- 预期影响：high；判断置信度：high。

##### C2-R2-03 · 主要 · 实验严谨性、清晰度、限制与负责任表述

- 位置：第 6 页 §5.1；第 19–22 页图 4、表 8、表 10；第 25–26 页 preregistration timeline
- 观察证据：anchor 在看过部分数据后从 1k 移到 4k；4k 是有效 2k/4k/8k 中最大显著 gap，未做选择调整。reader 规则显著改变点值，作者甚至承认更宽 strip rule 可反转符号；answer-anchored mnt512 为 p=0.0649。
- 重要性：固定 pinned reader 下方向明确，但 +27 pp 不是预先指定、reader-invariant 的效应量。标题与摘要的“pre-registered”容易让人误解为当前主 magnitude 在观察前冻结。
- 必需修复：把 +27 pp 降为 selected cell 描述，主结论仅保留预先可辩护的方向；对 2k/4k/8k 和 reader family 做选择/多重性敏感性，或在独立样本上冻结 cell 与 reader 重复。
- 验证标准：新的独立 qa1 样本在运行前固定 length、cap、reader；报告一个主配对区间以及所有预声明 secondary，不再从同一网格挑最大值。
- 仍需证据：独立确认运行或完整选择敏感性分析。
- 预期影响：high；判断置信度：high。

##### C2-R2-04 · 主要 · 重要性、实验严谨性、新颖性

- 位置：第 2–3 页 §1–3；第 8 页 §5.3；第 12–13 页表 4与方法细节；第 18 页表 7
- 观察证据：fixed-k working-set 优势同样由 j=0 short read 获得；depth 在 6,657-token read 仅省 0.3 GB，run-level timing 不支持 depth speedup。decode 每 token 重建 upper pack，未测 end-to-end；BM25/index latency 和 persistent-store baseline 未测，也没有同床 prior implementation。
- 重要性：按 KV-cache 论文的最低证据要求，当前结果没有展示可部署的 end-to-end 速度/内存/质量 Pareto 改善，且无法判断相对既有层轴或恢复方法的实际位置。实证贡献因此主要是内部负结果。
- 必需修复：实现 upper-layer KV reuse 并报告含 retrieval、prefill、decode、store/index 的 end-to-end latency、吞吐和峰值；在相同硬件/质量点至少加入一个真正相关 baseline，或将论文明确重定位为纯测量/负结果短文。
- 验证标准：同一请求分布、相同输出长度和质量约束下测 G=1 及并发；分别报告权重、active KV、persistent store 与 index，并画真实 Pareto front。
- 仍需证据：端到端系统测量、基线实现与质量匹配结果。
- 预期影响：high；判断置信度：high。

##### C2-R2-05 · 次要 · 实验严谨性、可复现性

- 位置：第 15、24–25 页 five-arm cross-trial audit 与 Appendix H
- 观察证据：A/D/R/C 与 E 来自两个 trial 和不同 driver snapshot；原 572907d7 blob 无法从历史恢复，只保存 diff。作者用静态 CLI whitelist 论证无副作用，并以 same-arm cross-trial endpoint 3/500 flips 作噪声界。
- 重要性：静态 diff 证据较强但不可从原 blob 独立重建；两 trial 的非确定性和 raw output 大量变化仍使小效应对比脆弱。当前主差大，不太可能反转，但 provenance 不完整。
- 必需修复：在单一冻结代码版本一次性重跑全部五臂，或把不可恢复 blob 与跨 trial 合并明确列为历史证据限制。
- 验证标准：同一 manifest、driver hash、seed/hardware 下生成 A/D/R/C/E，并比较当前 b/c 与点值。
- 仍需证据：统一 trial 输出与完整可恢复代码 snapshot。
- 预期影响：medium；判断置信度：medium。

**给作者的问题：**

- 能否运行作者自己指出的 confound-free arm：令 A 的 lower band 与 E 在完全相同 contiguous region、相同 positions/context set 上前向，只改变切分/缓存接口？
- 附录 J.1 为什么把对 discordant-sign 比例的 Clopper–Pearson 区间乘以观测 s/n 就称为 paired risk difference 的 exact CI？对总体 δ=p01−p10，s/n 本身也有抽样不确定性。
- 所谓 16k dense/re-forward 的 ≤6 pp ‘bound’如何约束 E–A 的 lower-band context-set confound？这两个对照既非同一变化对象，p=0.43/0.50 也只是非拒绝。
- 在当前实现每个 decode token 都重建 upper-band pack 的情况下，完整 end-to-end latency、吞吐和峰值是多少？0.26 s 的 prefill margin 是否会在 decode 后反号？
- 若没有同床 prior system，CoMem 相对 Layer-Condensed KV/HCache/KV-Direct 的实证价值应如何判断，而不只依赖结构表？

**能提高评分的证据：**

- 运行真正只改变 cache depth、lower-band context set 完全匹配的 arm，并确认方向。
- 用有效 paired risk-difference 区间替换主 CP-rescaled 区间，在独立冻结 cell/reader 上复现。
- 报告含 decode/retrieval/index 的端到端结果及至少一个同床 prior baseline，展示明确 Pareto 价值。

**会降低评分的证据：**

- confound-free arm 显示 E−A 差异主要来自 lower-band context set 而非 cached depth。
- 独立 reader 或新样本使 j=0>j=12 方向消失/反转。
- 完整 decode 后 j=12 的系统成本高于 j=0 且无存储优势，令机制没有可用 operating point。

**伦理标记：** 否。论文提醒 raw text 与 activations 的隐私、加密和 source-bound deletion，当前无用户数据或部署声明需要伦理升级。

**LLM 使用披露：** 本审稿由隔离的 AI 子代理 R2 生成，仅用于内部投稿前质量控制；未与其他评审通信，未读取其他评审结果、作者计划、编辑上下文或历史评分。

**评审限制：**

- 遵循隔离要求未联网，未核验外部引文、2026 年邻近工作、模型配置或硬件说明。
- 按要求完整读取了 KV-cache 专项校准文件，并仅将其用于 C2 的贡献类型、baseline、内存/速度/精确性与复现性判断。
- 仅审阅冻结 PDF；未读取或执行代码、数据、preregistration、trial tree、diff 或 artifact，因此 hash/pairing/运行日志仅按文稿评估。
- PDF 共 28 页，已全文阅读并逐页视觉核查；未见解析缺页，部分附录页留白较多但不影响内容。

#### R3（实验严谨性）完整评议

**论文概述：** 本文研究 CoMem：长文档按块在第j层保存 residual activation，查询时用 BM25 取 top-k 块并只重算上层。核心 BABILong qa1 同检索比较在100个样本上将 j=12 cache 与 j=0 原 token 重前向配对；4k、cap160 下准确率48%对75%，差+27个百分点，cap512 下方向相同。预先声明的深度曲线也总体指向 cache 越深读出越差。五臂 A/D/R/C/E 试图把深度与块独立编码、位置重定位分开，但 E 与 A 的低层注意范围仍不同。系统侧在128k、约6.7k token 固定读长下，j=12 相对 j=0 的峰值显存仅低约0.3GB，read/prefill break-even 约26次查询；decode 每 token 重建上层 pack，故未测端到端延迟或吞吐。

**最强的已核实贡献：** 表2的同检索、同查询、同 decoder 成对比较最可信：在冻结 reader 下，j=0 重前向在两个注册 cap 上均优于 j=12 cache；逐题 discordant counts 的 exact McNemar 检验强烈拒绝零差。即便主要区间构造需修正，论文列出的更宽 Newcombe 交叉检查仍排除0，因此该单模型、qa1、oracle-selector 范围内的方向性负结果较稳健。

**维度理由：**

- Soundness：同检索块的 j=12 与 j=0 成对比较、逐题 McNemar 计数和 reader 敏感性为负方向提供了可信证据。然而主要风险差区间的“exact conditional”构造没有包含随机 discordance 总量，五臂 E−A 也同时改变了低层可见上下文；标题中的 matched/pre-registered 强度高于当前识别。
- Presentation：论文把方向估计、深度归因与系统计量分开，并在表1、表2、表3中列出控制与不可复用边界。28页较长，但主数字、读出约定、选择过程和系统范围都能追踪，图表渲染清晰。
- Contribution：单层 hidden-state cache 加 BM25 top-k 的组合及其负结果有一定经验价值，尤其是显示同块重前向在 qa1 上优于 j=12 cache。可是只覆盖一个8B模型、一个单事实合成任务、oracle selector 和固定深度；没有端到端 serving 优势或现有系统基线，贡献范围有限。

**优点：**

- 表1明确区分 direction、attribution、serving 三种 estimand，避免把质量差、内存差和端到端性能混成一个 Pareto 结论。
- 表2使用100/100 content-hash 配对并报告 b/c discordant counts，使读者可以复核 exact McNemar 方向检验。
- 论文主动披露4k效应量是从有效长度中选择的最大显著量级，并把可复用结论降为方向而非通用+27个百分点。
- reader convention、cap、深度 sweep 与非配对 Newcombe 敏感性均被保留，未掩盖答案解析依赖。
- 系统章节明确说明0.68秒只到 read/prefill、persistent hidden/text/index 随上下文增长、0.3GB不是一般内存收益。

**问题与可验证修复：**

##### C2-R3-01 · 主要 · 技术正确性、实验严谨性

- 位置：第5.1节表2（PDF第6页）及附录J.1、表16（PDF第27–28页）
- 观察证据：主要区间先对 c/(b+c) 做 Clopper–Pearson，再乘以观测 discordance 比例(b+c)/n，得到 mnt160 的[+17.7,+30.5]和 mnt512 的[+14.1,+28.7]个百分点；该构造把随机的(b+c)/n固定为观测值。
- 重要性：条件检验可用于 McNemar 零假设，但这种重缩放没有传播 discordance 总量的抽样不确定性，因此不是总体 paired risk difference 的无条件95%区间。它会让效应量区间显得过窄。
- 必需修复：用对全部四个配对单元计数有效的 matched-pair risk-difference 区间或以题目为单位的配对 bootstrap 作为主区间，并将当前构造准确标为条件区间。
- 验证标准：对两个注册 cap 重算至少一种有覆盖保证的配对风险差区间；报告覆盖模拟或标准方法引用，并检查方向结论与效应量表述是否变化。
- 仍需证据：四格配对计数、修正区间和重算脚本输出。
- 预期影响：high；判断置信度：high。

##### C2-R3-02 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第4节五臂定义与第5.2节表3（PDF第4、7页）
- 观察证据：E−A 在4k和16k分别约−28、−36个百分点，但A的底层在选中 pack 上重前向，E的 h12 则在连续全文上下文中写入；论文第5.2节也承认低层 attended context set 仍不同。另一些分开的 dense/re-forward control 与j=0相差不显著，被描述为残余因素最多约6个百分点。
- 重要性：E−A不只改变 cache depth，因而不能识别“depth itself”的因果效应。其他 cell 未显著不等于对该混杂量给出6个百分点上界，尤其这些 cell并非同一反事实对照。
- 必需修复：把结论改为特定部署管线的联合差异，或新增低层注意范围完全相同、只在读取起始深度上变化的对照。删除由非显著性推导的残余上界。
- 验证标准：在相同选中 pack、相同位置和相同写入上下文上比较 j=0 与 j=12；以预先声明的等价/非劣界限直接估计每个非深度因素。
- 仍需证据：真正单因素的逐题配对臂和预先声明的归因分析。
- 预期影响：high；判断置信度：high。

##### C2-R3-03 · 主要 · 实验严谨性、限制与负责任表述

- 位置：摘要、第4节注册说明及第5.1节选择 caveat（PDF第1、4、6页）
- 观察证据：原计划的1k锚点在查看决定性 cell 前移至4k，但论文同时说明4k的+27个百分点是2k/4k/8k有效长度和 reader 结果中选择出的量级；没有对该选择做推断校正或独立复现。
- 重要性：方向在多个 cell 上一致可以作为探索性稳健性，但把最大显著量级放入标题/摘要会产生选择后夸大。事前冻结分析规则不等于该效应量未被选择。
- 必需修复：将4k数值明确降为选择后的描述，主张只保留跨注册 cap 的方向；或在未查看的固定任务/长度上做一次前瞻复现并以该 cell 为主。
- 验证标准：在独立数据或模型上预先冻结单一长度、reader、cap与区间方法；一次性揭盲并报告结果，或对现有搜索空间做同时区间。
- 仍需证据：独立前瞻复现或选择感知的同时推断。
- 预期影响：high；判断置信度：high。

##### C2-R3-04 · 主要 · 实验严谨性、重要性、限制与负责任表述

- 位置：第4节实验设置、第5.1节表2及第6节局限（PDF第4、6、9页）
- 观察证据：所有 load-bearing 结果来自Qwen3-8B、BABILong qa1、oracle selector、每 cell 100题和单一训练自由管线；没有多模型、多任务、多随机 seed 或部署 BM25 的同样 matched 复现。
- 重要性：单事实合成任务和 oracle selection 有利于隔离方向，却不能判断该损失是否出现在真实检索误差、多跳任务、不同架构或 cache-aware 模型中。标题的“cached-depth read-out”容易被读成更一般的系统边界。
- 必需修复：缩窄标题和结论到当前 model/task/reader，或至少在另一个模型家族、一个非合成任务和实际 BM25 selector 上预先复制方向。
- 验证标准：预先定义跨模型一致性准则，并报告每个新 setting 的配对 discordant counts、选择感知区间及 reader 审计。
- 仍需证据：多模型、多任务、实际 selector 的独立配对结果。
- 预期影响：high；判断置信度：high。

##### C2-R3-05 · 主要 · 实验严谨性、重要性、可复现性

- 位置：第5.3节 serving account（PDF第7–8页）与附录B/H（PDF第13、21–23页）
- 观察证据：相同约6657-token读长下j=12与j=0峰值仅差约0.3GB，来自median-of-three且打印粒度0.1GB；0.68秒在prefill/read结束，decode每token重建upper pack。BM25/index延迟、持久存储、端到端吞吐及可部署先验基线均未完整测量。
- 重要性：约26次查询的 break-even 只适用于截断的prefill账本，不能支持系统实用性；未测成本可能大于0.3GB/0.26秒量级，且没有端到端比较说明该方案值得部署。
- 必需修复：实现可复用上层KV的端到端 decode，纳入选择、索引、存储和并发成本，并与至少一个公开的KV压缩/检索基线及 full-context 比较延迟、吞吐、峰值和质量。
- 验证标准：在固定硬件、预热与并发下运行足够重复，报告分布而非三次中位数；用端到端总成本重算 break-even 并做敏感性分析。
- 仍需证据：端到端系统测量、基线实现、重复试验原始分布和完整成本账本。
- 预期影响：high；判断置信度：high。

**给作者的问题：**

- 为何把条件于观测 discordance 数的 Clopper–Pearson 重缩放区间称为总体 paired risk difference 的主要95%区间？
- 在 A 与 E 之间，A 的底层只看选中 pack，而 E 的 h12 在连续全文上写入；作者如何据此声称只改变 cache depth？
- 4k为何在看到2k/4k/8k结果及 reader 变体后成为标题量级的主 cell；是否存在完全独立的固定 cell 复现？
- 每个 cell 是否仅有一个确定性生成 seed？若模型/解析链含随机性，题目配对如何覆盖 run-to-run 变异？
- 在 decode 每 token 重建 upper-band pack 且 BM25/index 成本未端到端测量时，CoMem相对可部署基线的实际系统假设是什么？

**能提高评分的证据：**

- 用有效的无条件 matched-pair 风险差区间重算后，两个注册 cap 的方向和量级仍稳健。
- 真正只改变读取深度的单因素臂复现 E−A 方向，从而识别 depth attribution。
- 在独立模型与真实检索任务上前瞻复制该负方向，且不依赖选中的4k cell。
- 完整端到端 serving benchmark 显示明确质量—资源权衡并优于有意义基线。

**会降低评分的证据：**

- 有效配对区间或多 seed 复现后，关键方向不再稳定。
- 控制低层注意范围后，所谓深度损失大幅缩小或消失。
- 实际 BM25 和端到端 decode 成本使 break-even 或资源优势消失。
- 独立复核发现4k/reader选择时序与文中注册叙述不一致。

**伦理标记：** 否。研究涉及模型推理效率和公开基准，不涉及个人数据或人类受试者。主要责任性风险是把特定负结果或截断成本账本推广成一般部署结论；论文已有较多范围限定。

**LLM 使用披露：** 本审稿由隔离运行的 AI 子代理 R3 完成，仅用于内部投稿前质量控制；未与其他评审通信，未读取作者计划、旧评分或其他评审输出。

**评审限制：**

- 仅审阅指定冻结 PDF；按隔离要求未访问代码、运行工件、逐题输出、预注册文件或硬件日志，故 content-hash 配对、时间顺序和系统数字未外部执行复核。
- 未联网，未核验外部引文、相关工作优先权、模型/基准版本或公开讨论；相关不确定性不应被解释为已验证。
- 已完整读取指定 KV-cache 校准说明，并逐页视觉核查28页 PDF；未见影响解读的渲染缺陷。

### C3

#### 交付与来源状态

- 选定输入：`/Users/liuhanzuo/Downloads/C3-paper.pdf`
- 源状态：`exact_self_contained_latex_snapshot`
- 润色 PDF：`/Users/liuhanzuo/MacLLM-Bench/output/pdf/paper_polish_20260826/C3-polished.pdf`
- 冻结 SHA-256：`b1eed7ac44c179e68b6d31e3b7235d19ddb1094fc99de461246f36a1de2d5edb`
- 总页数：14；主文状态：主文在第9页结束。
- 版面核验：pass；构建：pass。
- 旧评分基线：2,4,2；旧中位数：2。

#### 三评量化结果

| 审稿人 | 角色 | Overall | Confidence | 建议 | Soundness | Presentation | Contribution | 当前上限 | 必需修改后预测 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| R1 | 新颖性与定位 | 2 | 4 | 拒绝 | 2 | 2 | 1 | 2 | 4 |
| R2 | 技术正确性 | 2 | 4 | 拒绝 | 2 | 2 | 1 | 4 | 4 |
| R3 | 实验严谨性 | 4 | 4 | 略低于接收线 | 2 | 3 | 1 | 4 | 6 |

三评中位数为 **2**，均值 2.67，跨度 2，接收侧票数 0/3。

#### 编辑记录

- [结构审计](work/C3/structure_audit.md)
- [语义锁](work/C3/semantic_lock.md)
- [修订日志](work/C3/revision_log.md)
- [待核验事项](work/C3/needs_verification.md)

**修订日志原文：**

> # C3 revision log
>
> ## Scope
>
> Evidence-preserving manuscript polish. No experiments, independent citation verification, or reviewer scoring were performed.
>
> ## Source changes
>
> - Replaced the promotional title framing with a descriptive title tied to the paper's evidence object.
> - Rewrote the abstract and introduction to foreground the executable evidence bundle, its measured checks, and its explicit limits.
> - Softened unnecessarily adversarial related-work phrasing while retaining the paper's technical distinction.
> - Compressed the conclusion and kept the D1 and 16k attribution boundaries visible.
> - Moved the reproducibility and ethics statement, unchanged in substance, to Appendix A so the conclusion ends on PDF page 9; references begin on page 10.
>
> ## Semantic safeguards
>
> - Preserved every reported count, digest, table/figure value, equation, citation key, case label, and experiment status.
> - Did not imply that scanner/red-team evidence was rerun, that SparseForge transfer was independently reproduced, or that the 16k causal attribution was upgraded beyond inspection grade.
> - Added no new system, experiment, or literature claim.
>
> ## Verification
>
> - Built from the isolated source with shell escape disabled, using BibTeX and two final LaTeX passes.
> - Checked extracted text, current citation/reference warnings, main-text boundary, and rendered pages.
> - Remaining source-dependent items are listed in `needs_verification.md`.

**待核验事项原文：**

> # C3 needs verification
>
> - Scanner and red-team transcript claims were preserved but not re-executed because the bundle/tools are absent from this isolated source copy.
> - The manuscript claims deterministic compilation and a bundle digest; this pass verifies the new PDF build but not the external archive named in the paper.
> - Cross-case scanner transfer to SparseForge is preserved as reported, not independently reproduced.
> - The 16k causal attribution remains inspection-grade and must not be upgraded without a guarded concurrent rerun.
> - The conclusion ends on PDF page 9 and references begin on page 10 after moving the reproducibility and ethics statement to Appendix A; the venue's current formatting policy should still be checked before submission.

#### R1（新颖性与定位）完整评议

**论文概述：** 这是一篇对已撤回 KV-selective serving 研究的审计说明。作者把历史主张分成 MEASURED、DERIVED、ESTIMATE、NOT_MEASURED 等状态，并提供字符串/文件级扫描器来检查每行 scope 是否被越界复用。静态审计发现 full-context worker 缺少 inference-mode guard 和 vendored selector 参数名不匹配；只修复并重跑了前者。32k、G=4 时峰值 70.73 GB、三轮共 12 请求且无 OOM，但 KV-selector、adapter-matched arm、16k guarded rerun、吞吐/延迟和核心成本-质量结论仍未测量，原研究解释保持撤回。

**最强的已核实贡献：** 第 6 节表 4 把一个此前被误读为硬件容量边界的失败定位为应用层 grad-mode leak，并在同一 32k full-context 路径上修复后得到有限但清楚的可运行记录；作者没有把这一修复外推成 selector、吞吐或完整系统结果。

**维度理由：**

- Soundness：论文对撤回范围相当诚实，32k 全上下文 arm 的两行 inference-mode 修复后确实有一个 G=4、三轮的可行性记录；但 NOT_MEASURED register 的完整性只相对于作者自己枚举的 ledger，pattern scanner 只能检查列举短语，跨案例验证只是命中计数。它无法验证语义完整性，也没有独立自然样本上的检测或误报性能。
- Presentation：状态词、表格和 scope labels 组织严谨，14 页视觉完整；然而论文大部分篇幅在解释内部 C/D/S/R 编号、撤回谱系、哪些数字不可引用以及字符串扫描器的覆盖边界。读者很难从中提炼一个可普遍采用且经过验证的方法，内部档案说明压过了科学问题。
- Contribution：论文明确不提供 KV-selector、adapter-matched、16k guarded rerun、并发吞吐或质量结论，真正贡献是作者声明账本上的 NOT_MEASURED 表与短语检查器。论文自己承认 GSN 的 undeveloped goals、checklists 和结构化文档早已存在；当前窄实现没有独立验证或明显超越这些先例，未形成足够的 ICLR 新知识或能力。

**优点：**

- 对撤回的核心 serving/quality 主张没有粉饰，明确列出 KV-selector、adapter、16k rerun 和 LongEval 配对输出仍未测量。
- 把实测、推导、估计、外推和 inspection-grade 证据分开，避免 139.2 GB 的 pre-guard 16k 记录被继续当作硬件极限。
- 披露 G=4 只有 3 个有效独立轮次、G=1 只是路径一致性检查，而非第二个并发数据点。
- 逐页视觉检查覆盖全部 14 页，图表和表格未见裁切或覆盖。

**问题与可验证修复：**

##### C3-I1 · 致命 · 新颖性、重要性、限制与负责任表述

- 位置：第 1 页摘要与第 1 节 Contributions；第 2 页 Related Work；第 9 页结论
- 观察证据：论文主动说明 archived cost-quality conclusion 不恢复，并把贡献限定为 scope statements、作者枚举 ledger 上的 NOT_MEASURED register、per-row labels 和 pattern-scoped gate。第 2 页又承认 GSN 已经显式表示未建立的目标，register 只是该思想的窄 machine-checked instance。
- 重要性：当前稿件既没有新的 KV-serving 方法/结论，也没有被验证的新审计理论。撤回与诚实报告很重要，但本身不是足够的 ICLR 科学贡献；核心产物与既有 assurance/checklist 机制的差异主要是具体格式。
- 必需修复：将稿件重定位为 artifact/reproducibility/negative-results note，或增加一个真正通用的审计方法贡献：形式化语义、可判定保证、第三方数据集和相对现有 assurance/checklist 工具的定量基线。标题与摘要应停止让读者期待 KV-selective serving 研究结果。
- 验证标准：修改后应能用一句不依赖本次撤回内部 ID 的话说明新能力，并在独立项目上以预设指标证明该能力超过现有方法；否则该稿不满足主会贡献门槛。
- 仍需证据：跨组织/跨项目的独立验证、现有工具基线、语义级错误标签与效果指标；或完整恢复原 serving 研究的可验证结果。
- 预期影响：high；判断置信度：high。

##### C3-I2 · 主要 · 技术正确性、实验严谨性、可复现性

- 位置：第 4–6 页第 4–5 节；第 9 页第 8 节；第 11–12 页附录 B/C
- 观察证据：register 的完整性只相对于作者构造的 C/D/S ledger；generator 与 checker 共享同一源，因而只能检测行缺失/状态漂移。所谓跨案例转移是在另一稿上计数 91 个六类 signal hits，对本稿为 68 个；没有金标准、精确率、召回率或下游决策效用。盲读者恢复 14 个已存在 ID 也不能发现未被作者枚举的主张。
- 重要性：同源生成与检查容易证明内部一致性，却不能证明完整性或审计有效性。当前验证接近自我一致性测试，不能支持可推广的报告工具。
- 必需修复：构建第三方定义、独立标注的多项目 claim/violation 数据集；冻结 ledger 生成规则和 scanner 后，报告遗漏、误报、跨域迁移和人工审计节省，并与 GSN/checklist/简单关键词基线比较。
- 验证标准：测试集的主张全集与违规标签不得由 scanner 作者定义；在揭盲前冻结规则，按项目分层留出，并报告逐类混淆矩阵和失败案例。
- 仍需证据：独立标注者、自然项目样本、盲评协议、基线与置信区间。
- 预期影响：high；判断置信度：high。

##### C3-I3 · 主要 · 实验严谨性、重要性、限制与负责任表述

- 位置：第 5 页表 2 的 D1–D6/S1 行；第 8–9 页第 6 节与表 4；第 9 页结论
- 观察证据：修复后的唯一并发结果是 full-context、无 adapter、无 selector 的 32k arm；G=4 只有 3 个有效独立 round。16k guarded rerun、KV-selector finite fetch、adapter recovery、单臂机制隔离与 LongEval 逐题配对均为 NOT_MEASURED，且论文不提供吞吐、可扩展性或延迟结论。
- 重要性：这只证明一个两行 guard 修复使一个操作点不再 OOM，不能回答题目中的 KV-selective serving 成本、质量或可行性。按指定 KV-cache 校准，实用 serving 主张应有真实 baselines、完整生命周期、内存核算和端到端性能；这些核心要素均缺失。
- 必需修复：若保留 serving 主题，先恢复并冻结 selector 与 adapter 身份，运行 matched full-context/KV-selective arms 的多长度、多并发端到端实验，并报告质量、吞吐、p95 延迟、峰值和持久存储；否则删去 serving 研究框架，只作为方法学说明投稿到合适轨道。
- 验证标准：所有标题性 serving 结论必须由可重跑的 matched arms 支持，至少有两个模型/工作负载或充分说明单点范围，并将 3-round 可行性记录与性能估计严格分开。
- 仍需证据：修复后的 selector/adapter、独立重复、完整 decode 与质量测量、真实最近系统基线。
- 预期影响：high；判断置信度：high。

##### C3-I4 · 主要 · 技术正确性、新颖性、清晰度

- 位置：第 3 页第 3.1–3.2 节 G1–G4；第 4 页第 4 节；第 12 页 Coverage self-critique
- 观察证据：scope gate 依赖明确枚举的 token/phrase patterns；论文披露旧版曾让吞吐式表述从 enumeration 外通过，代码块被豁免，语义替换或文本窗口外 clue 也不会被 exact-set checker 发现。六类 scanner 对本稿 27 个 honesty structures 中有 9 个完全无覆盖。
- 重要性：字符串存在/缺失不是语义保证；作者可以用同义改写绕过禁止项，也可以在合法短语附近提出越界推断。机器返回零并不等于论证安全，容易造成虚假的 assurance。
- 必需修复：把 scanner 明确降为 lint，不称其为 scope predicate 的充分保证；若要声称可执行 assurance，需要定义机器可读 claim schema 与语义绑定，加入系统性的同义改写/对抗测试，并由独立标注评估覆盖。
- 验证标准：在未参与开发的人工对抗集上，测试同义改写、否定、跨句引用、代码块和表格复用；报告错误率并确认论文不把通过 scanner 等同于语义正确。
- 仍需证据：语义级标注集、对抗改写、独立评估与简单关键词/规则基线。
- 预期影响：high；判断置信度：high。

##### C3-I5 · 次要 · 清晰度

- 位置：第 1–9 页主文及第 12–14 页附录 D–F
- 观察证据：C4/C7/C9/C13、D1–D6、S1/S3/S6、R142 与七种状态构成高度项目特定的词汇，主文多次重复哪些推断被禁止；第 14 页仅放置一张小表并留有大量空白。
- 重要性：这使稿件更像内部事故档案而非可迁移方法论文，也掩盖了唯一实测修复结果。
- 必需修复：以问题—方法—独立验证—限制重构主文，把内部 ledger glossary、撤回历史和复用纪律压缩为补充材料；增加一个项目无关的最小示例。
- 验证标准：不接触原项目的读者应能在两页内理解工具输入、算法、输出、保证和失败模式，并能将其应用到一个新 archive。
- 仍需证据：不需要新实验来修复版式，但项目无关示例应来自独立数据。
- 预期影响：medium；判断置信度：high。

**给作者的问题：**

- 如果 register 的全集由作者自己定义，第三方如何发现从未进入 ledger 的关键主张，而不是只验证作者列出的行都存在？
- pattern-scoped scanner 相比 GSN、机器可读 checklist、artifact evaluation 和 assertion/CI 工具提供了什么可测的新能力？
- 为什么第 8 节跨案例只比较 68 与 91 个 signal hits，而不报告对人工语义标签的精确率、召回率或遗漏类型？
- 在 KV-selector、adapter 与核心 cost-quality 路径均未重跑的情况下，这篇稿件为何应作为 ICLR 研究论文而不是撤稿说明、artifact report 或 reproducibility note？

**能提高评分的证据：**

- 在第三方定义的自然 archive 数据集上证明 register/scanner 相对现有 assurance/checklist 工具的检测、误报或审计效率优势。
- 提供语义级而非短语级的形式化与对抗覆盖证据。
- 若保留 KV-serving 主题，完成 selector、adapter、质量与端到端性能的 matched rerun。

**会降低评分的证据：**

- 独立审计发现 ledger 中遗漏 load-bearing claim，说明 exact-set completeness 只是在错误全集上的闭合。
- 同义改写或跨句复用可系统绕过 G1–G4，而论文仍把通过结果当作 assurance。
- 复跑显示 32k 结果不可重现，或 guard 之外还有未披露的关键运行差异。

**伦理标记：** 否。未涉及人类受试者或个人数据；主要风险是机器检查通过被误读为科学正确或完整性证明，已在问题 C3-I4 中作为方法责任边界处理。

**LLM 使用披露：** 本审稿由隔离的 AI 子代理 R1 生成，仅用于内部投稿前质量控制；该子代理只读取冻结 PDF、指定审稿规则及针对 C2/C3 的 ICLR KV-cache 校准材料，未与其他评审通信，也未读取其他评审输出。

**评审限制：**

- 本评审严格只读取指定冻结 PDF、官方审稿规则与指定的 ICLR KV-cache 校准材料；未读取论文声称随附的 bundle、scanner、日志、代码或旧稿。
- 因此无法亲自执行论文列出的零 GPU 命令，也无法验证 24/24 manifest、hash、第三方 replay 或 32k GPU 记录；相关判断仅基于 PDF 陈述。
- 按任务要求未联网，也未核验外部引文；新颖性仅相对于 PDF 相关工作与指定校准锚点判断。
- 逐页视觉核查覆盖全部 14 页；未发现裁切、重叠或缺页，第 14 页留白较多。

#### R2（技术正确性）完整评议

**论文概述：** 本文是一份撤回说明与报告工具论文，而非原 KV-selective serving 研究。它将旧档案中的 C/D/S 类主张整理为作者枚举的 NOT_MEASURED register，以模式匹配 scanner 检查部分措辞和状态，并给冻结数值附逐行 scope。一次修复约两行 inference-mode guard 的并发复跑表明：Qwen3-8B、单 H20-3e、L=32k 下，全上下文无 adapter/selector arm 在 G=1 与 G=4 可运行，峰值分别为 29.31 GB 与 70.73 GB；KV selector、matched adapter、16k guarded rerun 仍未测。

**最强的已核实贡献：** 第 7–8 页 §6 与表 4 最可靠地支持一个很窄的诊断结论：原并发 worker 存在线程局部 grad-mode 泄漏；加入 inference-mode guard 后，L=32k 的全上下文、无 adapter、无 selector 路径在 G=1 和 G=4 均得到有限 decode 且无 OOM。作者同时正确限定有效独立重复只有每个并发级别 3 轮，并未把 G=4/G=1 比值解释为吞吐或可扩展性。

**维度理由：**

- Soundness：作者对撤回边界、单次 guarded rerun 与未测项目大体诚实，32k 全上下文 arm 在修复 grad-mode leak 后可运行这一窄事实有日志式证据。但核心 register 的完备性仅相对于作者自建 ledger，scanner 仅覆盖列举字符串；跨案例证据没有独立真值，且正文的 canonical-status 表述与表中状态直接矛盾。因而它只能验证作者定义的内部一致性，不能支撑更广的审计有效性。
- Presentation：逐行 scope、MEASURED/NOT_MEASURED 区分和失败披露较清楚，读者能追踪哪些数字不可复用。然而 14 页中大量内部 C/D/S/R 编号、历史版本与 scanner 细节使论证高度档案化；Table 2、Table 3 caption 的状态词错误尤其损害了其所主张的精确机器检查。
- Contribution：论文没有提供 KV selector、adapter-matched arm、16k guarded rerun、吞吐/延迟或 KV-selective serving 结果；唯一新实测是修复常见 inference guard 后的单机 32k 可行性。NOT_MEASURED register 是既有 assurance/checklist 思路的窄实现，其外部有效性尚未建立，难达到 ICLR 研究贡献门槛。

**优点：**

- 明确撤回旧成本—质量解释，没有用一次修复复跑重新包装成正面的 KV serving 结论。
- 表 2–4 将 arm、并发、estimand、状态和可引用范围逐行列出，并区分 32k 已测、16k inspection-grade 与 selector/adapter 未测。
- 第 8 页坦率说明 G=4 的 12 个请求只有 3 个有效独立轮次，且未保留逐请求时延，避免制造不可恢复的离散度。
- 公开 scanner 的模式范围、verbatim exemption、作者构造 ledger 和非穷尽 taxonomy 等限制，降低了误读风险。

**问题与可验证修复：**

##### C3-R2-01 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第 3 页 §3.1；第 4 页 §4；第 12 页 Appendix C
- 观察证据：register 只要求每个作者枚举的 LEDGER ID 有一行；语义检查只看每个 ID 后有限窗口中的 canonical clues。scope gate 也只匹配明确列出的字面模式，并豁免 verbatim code。作者在第 4 页和第 12 页承认完全遗漏的 claim、未列同义表达及 27 个 honesty structures 中 9 个都可不被发现。
- 重要性：这套机制可以证明给定作者输入下的结构一致性，却不能证明论文中的重要主张均已登记或被正确限定。标题与贡献表述若让读者把它理解为语义完备审计，会把 author-supplied specification 当成独立 ground truth。
- 必需修复：将贡献始终表述为 author-declared ledger 的一致性 lint；若要主张审计完备性，需由独立标注者在不知道 ledger 的情况下抽取完整主张集，并以预定义匹配规则测漏报、误报和语义错配。
- 验证标准：在冻结 ledger 前由多名独立标注者建立 adjudicated claim truth set；对 scanner/register 报告 claim-level precision、recall、漏项类型与一致性，并用留出同义改写做盲测。
- 仍需证据：独立真值集、标注协议、混淆矩阵及留出改写结果。
- 预期影响：high；判断置信度：high。

##### C3-R2-02 · 主要 · 技术正确性、清晰度、可复现性

- 位置：第 5 页表 2 caption 与 D4/D5 行；第 6 页表 3 caption 与 C13 @1M 行
- 观察证据：表 2 caption 写道 CLOSED_WRITING 与 NOT_ADJUDICATED ‘no row here carries them’，但紧接着的 D4、D5 两行状态均为 CLOSED_WRITING。表 3 caption 把 canonical vocabulary 列为 MEASURED、DERIVED、ESTIMATE、NOT_ADJUDICATED，却在 C13 @1M 行使用未列出的 EXTRAPOLATED。
- 重要性：论文的主要卖点正是状态词和表格由单源生成并受到机器检查；可直接目视确认的内部矛盾说明检查没有覆盖最终 caption-to-row consistency，削弱最核心的可信度主张。
- 必需修复：修正两个 caption，并让构建 gate 从最终生成表中抽取实际状态集合，验证 caption 声明、共享 vocabulary 与每行 token 完全一致；加入当前矛盾作为必须失败的 mutation fixture。
- 验证标准：自动解析最终 PDF/LaTeX 表格；若 caption 声称某状态无行而行中出现，或 caption 枚举遗漏已用状态，构建必须非零退出。
- 仍需证据：修订后的表格、final-render scanner transcript 与对应 red-team fixture。
- 预期影响：high；判断置信度：high。

##### C3-R2-03 · 主要 · 实验严谨性、重要性、限制与负责任表述

- 位置：第 9 页 §8；第 11–12 页表 5与 Appendix C
- 观察证据：所谓 transfer 只是在本稿与一篇作者选择的‘different group’ SparseForge 稿件上分别得到 68 与 91 个模式命中。没有独立的诚实/越界真值、precision/recall、未使用文稿或 reviewer outcome；命中数量不同只说明文本中目标词形频率不同。
- 重要性：没有标签的两案例 hit count 不能识别 scanner 是否捕获真正风险、是否大量漏报或误报，更不能支持‘generalizes’。这使方法有效性停留在可运行演示，而非可评价的跨案例工具。
- 必需修复：把 generalization 改为 feasibility demonstration；或在预先冻结的多领域、作者外部语料上，由独立标注者给 load-bearing claims 与 scope violations 建立真值并评价检测性能。
- 验证标准：留一项目外测，预注册 taxonomy/pattern，不按测试稿扩充；报告宏/微 precision、recall、每类错例及 reviewer 时间节约。
- 仍需证据：多案例独立标注 benchmark 与盲测结果。
- 预期影响：high；判断置信度：high。

##### C3-R2-04 · 主要 · 重要性、新颖性、实验严谨性

- 位置：第 1 页摘要与 §1；第 7–9 页 §6–9；第 14 页表 8
- 观察证据：唯一新运行是无 adapter、无 selector 的 full-context arm 在 L=32k、单 GPU、3 轮上的 guard 修复检查。KV selector 因参数名错误未复跑，adapter 不可恢复，16k 未复跑；无端到端吞吐/延迟结论，也没有 single-arm isolation。
- 重要性：修复 inference/no-grad 使用错误并确认一个 operating point 可行是有用的档案纠错，但不构成 KV-cache/serving 方法评价。论文自己承认 register 是既有 GSN/模板思想的窄实例，因此当前实证与方法新颖性都不足以支撑完整 ICLR 论文。
- 必需修复：若作为 serving 论文，完成 selector、matched adapter、16k 与质量/成本端到端对照；若作为元科学论文，则去除 KV 系统贡献暗示并提供独立、多案例的工具效度研究和与现有 assurance/documentation 方法的实证比较。
- 验证标准：serving 路线需在冻结协议下复跑所有关键 arm 并画质量—内存—延迟 Pareto；方法路线需在未参与构建的语料与评审者上比较漏报率和工作量。
- 仍需证据：完整 serving 实验或外部方法有效性实验。
- 预期影响：high；判断置信度：high。

##### C3-R2-05 · 次要 · 清晰度、可复现性

- 位置：第 2 页表 1；第 6–7 页表 3、式 (1)；第 13–14 页表 6与表 8
- 观察证据：符号 G 在不同位置既表示并发，也表示 archived decode model 的生成长度；Q* @128k 又给出 G=1 的 25.8 与 G=512 的 164.2，而后者表中 scope 写作 replicate crossover。作者依靠逐行标签避免直接合并，但同一符号承载不同物理量。
- 重要性：在一篇核心目标是防止跨 scope 误读的文章中，符号重载本身增加误读和错误复用风险，尤其当两个 Q* 数值并列出现时。
- 必需修复：为 concurrency 与 generated length 使用不同符号，并在所有 Q* 行显式打印生成长度、并发、聚合单位和 numerator/denominator 来源。
- 验证标准：全文符号 lint 应拒绝同一符号对应两个定义；逐行 schema 要求 concurrency 与 generation_length 为独立字段。
- 仍需证据：无歧义的重排表格与符号检查。
- 预期影响：medium；判断置信度：high。

**给作者的问题：**

- 若 register 的 claim universe 和 semantic clues 都由作者定义，什么独立程序能发现作者从 ledger 中完全遗漏的一类重要主张，而不是只证明内部 exact-set 一致？
- 第 5 页表 2 caption 为何声称 CLOSED_WRITING 与 NOT_ADJUDICATED 均无 row，尽管 D4、D5 明列 CLOSED_WRITING？第 6 页表 3 caption 又为何未列实际出现的 EXTRAPOLATED？这些文本是否来自未经扫描的生成路径？
- 第 9 页所称 cross-case generalization 除两个作者选择的文稿上的 hit count 外，有何带独立标签的 precision、recall、漏报率或 reviewer-utility 证据？
- 为什么一个不同组的单篇 SparseForge 稿件和本稿足以称 cross-case transfer，而不是两个便利样本上的词汇命中演示？
- 若目标仍是 KV-selective serving 方向，何时会运行修复后的 selector、matched adapter、16k guarded arm，并给出端到端质量、峰值、吞吐与延迟？

**能提高评分的证据：**

- 修复最终表格中 canonical-status 的直接矛盾，并让机器检查覆盖最终渲染文本。
- 在作者外、多领域冻结语料和独立真值上验证 register/scanner 的漏报、误报与 reviewer utility。
- 若保留 KV serving 定位，完成 selector、adapter、16k 与端到端质量—系统测量。

**会降低评分的证据：**

- 独立 claim recovery 发现作者 ledger 系统性遗漏 load-bearing claims。
- 留出同义改写显示模式 gate 很容易在不改变语义的情况下被绕过。
- 修复 selector/adapter 后仍无法产生任何可解释的 serving operating point。

**伦理标记：** 否。论文不涉及人类受试者、个人或敏感数据，也没有模型发布；其主要风险是冻结数字被脱离 scope 复用，作者已显式讨论。

**LLM 使用披露：** 本审稿由隔离的 AI 子代理 R2 生成，仅用于内部投稿前质量控制；未与其他评审通信，未读取其他评审结果、作者计划、编辑上下文或历史评分。

**评审限制：**

- 遵循隔离要求未联网，未核验外部引文、硬件规格、框架 API 或邻近工作。
- 按要求完整读取了 KV-cache 专项校准文件，并仅将其用于 C3 的贡献类型、baseline、内存/速度/精确性与复现性判断。
- 仅审阅冻结 PDF；未读取或执行 bundle、scanner、manifest、diff、日志或数据，故其 exit code、hash 与 artifact 可恢复性均仅按文稿评估。
- PDF 共 14 页，已全文阅读并逐页视觉核查；未见解析缺页，表 2–4 信息密度高但可辨认。

#### R3（实验严谨性）完整评议

**论文概述：** 本文撤回一个原先关于 KV-selective 多查询 serving 的成本—质量解释，并提出三个报告工具：逐主张 scope 标签、相对作者枚举 ledger 的 NOT_MEASURED register、以及对有限措辞模式运行的机器 scope gate。静态审计发现 full-context 并发 worker 缺少 inference-mode guard，以及 vendored selector 的参数名不匹配；前者经约两行修复后在单张 H20、Qwen3-8B、L=32k 上以G=1和G=4各3轮复跑，峰值分别29.31GB与70.73GB且无OOM。后者仅静态诊断。论文明确把16k guarded、KV selector、adapter-matched、LongEval逐题配对和单臂机制隔离标为 NOT_MEASURED，并限制冻结的串行成本、RULER质量与 Q* 派生值不得跨行聚合。

**最强的已核实贡献：** 第6节表4提供了一个范围明确的修复后事实：在给定单模型、单GPU、L=32k、8个查询的实现上，加 inference-mode guard 后 full-context worker 在G=1和G=4均可有限执行且未OOM；G=4只有3个有效独立轮次，故这仅推翻该点上的“环境必然不可行”读法，而不是吞吐、容量边界或 KV selector 性能结论。

**维度理由：**

- Soundness：论文对已测、派生、估计、外推和未测的边界异常坦诚，32k守护复跑也只作可行性记录。但 NOT_MEASURED register 的完备性仅相对于作者枚举的 ledger，措辞 scanner 是易漏同义改写的模式匹配；单个外部稿和单个盲读者不足以验证工具。
- Presentation：表2、表3、表4的逐行 scope 很清楚，撤回内容没有被重新包装成正面 serving 结论。全文可读且无截断，但大量内部 C/D/S/R 编号、构建转录和撤回细节使主线较自指；第13–14页存在较多空白但不妨碍阅读。
- Contribution：稿件本质上是一次撤回后的证据登记与报告工具，而非完成的 KV-selective serving 研究。核心 selector、adapter-matched、16k guarded rerun 与端到端服务均未测；报告工具又只有作者内和单案例验证，因此当前贡献不足以达到主会标准。

**优点：**

- 摘要和结论都明确声明原成本—质量研究已撤回，没有用守护复跑偷偷恢复 KV-selective serving 优势。
- 表2统一列出 MEASURED、DERIVED、ESTIMATE、EXTRAPOLATED、NOT_MEASURED 等状态，并逐行给出关闭条件。
- 第6节区分已修复并复跑的 grad-mode leak 与仅静态诊断的 selector 参数错误，未把二者混成一个已解决 blocker。
- 表4正确说明G=4的12个请求只有3轮有效独立重复，且不把约4倍中位数比率解释为吞吐或扩展性。
- 附录C主动承认六类 scanner 漏掉9/27种最重要的诚实结构，并说明 gate 不是完整诚实证书。

**问题与可验证修复：**

##### C3-R3-01 · 主要 · 重要性、实验严谨性、限制与负责任表述

- 位置：摘要、第1节及第9节结论（PDF第1、9页）；表2（PDF第5页）
- 观察证据：论文明确列出 KV-selector finite fetch、adapter-source recovery、adapter-matched arm、16k guarded rerun、LongEval逐题检验及单臂机制隔离均未测。唯一新增GPU结果是full-context 32k的guarded可行性复跑。
- 重要性：诚实撤回值得肯定，但它不产生原研究问题的答案。当前稿既没有 KV-selective serving 的有效成本—质量证据，也没有经广泛验证的独立报告方法，因此主贡献主要是个案记录。
- 必需修复：选择一个可检验的主贡献：若为系统论文，修复selector并恢复/重建adapter，完成匹配质量和端到端并发测量；若为报告方法论文，则在独立、多项目语料上系统验证 register/gate。
- 验证标准：系统路线需预先定义主 cell、对照、重复和端到端指标并一次性执行；方法路线需在隐藏标注集上报告遗漏率、误报率及与基线比较。
- 仍需证据：完整决定性实验，或多案例独立验证数据。
- 预期影响：high；判断置信度：high。

##### C3-R3-02 · 主要 · 技术正确性、实验严谨性、可复现性

- 位置：第3.1节、第4节及附录C（PDF第3–5、12页）
- 观察证据：register 的 exact-set 检查只保证作者 ledger 与生成表一致；scope gate 枚举有限短语，早期版本曾漏过 throughput 改写。附录C称六类 taxonomy 对27种结构漏掉9种。
- 重要性：同一来源生成与检查可检测转录漂移，却不能检测 ledger 和正文共同遗漏的主张，也不能对未枚举同义改写给出语义覆盖。称其为“complete”或机器检查工具容易被读成超过声明 universe 的保证。
- 必需修复：将“complete”始终限定为 declared ledger，并以独立标注的主张语料评估 semantic recall/precision；加入未见过的释义、否定、跨句和数值组合攻击。
- 验证标准：冻结多篇文稿和盲法 gold claim/scope 标签，在开发后一次性运行，报告每类遗漏、误报及置信区间；至少与关键词搜索和人工核查基线比较。
- 仍需证据：第三方 gold corpus、盲法标注协议、held-out adversarial fixtures 和误差分析。
- 预期影响：high；判断置信度：high。

##### C3-R3-03 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第8节“Method validity: cross-case transfer”与附录C（PDF第9、12页）
- 观察证据：scanner只额外运行在一篇SparseForge稿件上，得到91个signal hits，对本文得到68个；论文据此写“transfer shows the scan generalizes”。一个盲读者在当前PDF中回收14个register ID，但没有独立漏报/误报总体。
- 重要性：不同 hit 数只表明工具在第二篇文本上能匹配字符串，不表明其发现真实边界问题的准确性或跨领域泛化。单个读者、单个案例无法估计可靠性。
- 必需修复：把该句改为可运行性演示，或扩展到多团队、多领域、含已知缺陷与无缺陷文稿的盲法评估，预先定义检出目标。
- 验证标准：至少两名独立标注者建立 gold，报告一致性、precision、recall、每类性能与项目级留一验证；在冻结测试集上只评一次。
- 仍需证据：多案例独立标注及定量泛化结果。
- 预期影响：high；判断置信度：high。

##### C3-R3-04 · 主要 · 实验严谨性、技术正确性、可复现性

- 位置：第6节表4（PDF第8页）
- 观察证据：pre-guard 32k只有一次abort且没有峰值记录；post-guard在G=1与G=4各3轮，其中G=4是唯一并发数据。历史pre-guard 16k峰值139.2GB没有对应guarded 16k复跑。
- 重要性：post-guard可行性本身可信，但把早期边界归因于grad leak仍主要依赖静态代码检查与非对称历史记录。没有同配置 guard on/off 重复，无法量化缺陷的因果贡献或排除其他版本差异。
- 必需修复：在隔离环境中只切换 inference-mode guard，按相同模型、版本、查询、L、G和监控多轮运行；至少补齐16k与32k。
- 验证标准：保存每轮峰值、状态与请求级时延，对 guard on/off 做配对比较；证明除两行guard外环境和代码哈希一致。
- 仍需证据：受控ablation、完整原始记录和精确环境清单。
- 预期影响：high；判断置信度：medium。

##### C3-R3-05 · 次要 · 实验严谨性、清晰度、限制与负责任表述

- 位置：第5节表3、图3及附录D–E（PDF第6–7、13页）
- 观察证据：G=512、128k的citable per-query margin仅0.0404秒，15单位配对bootstrap为[0.026,0.088]秒，但表3只给 Q*=164.2 的点值；1M的 Q*=180.2 又明确超出≤128k测量范围。
- 重要性：当分母很小且不确定时，break-even 比值高度不稳定；只给点值可能让读者低估范围。虽然论文标注了派生/外推，传播不确定性会使scope更可操作。
- 必需修复：对每个Q*传播写入成本和per-query margin的不确定性，给出区间或分布；把1M严格留在外推敏感性表而非可引用主值。
- 验证标准：用配对重采样联合重算 numerator/denominator，报告正分母比例和Q*分位数；检查结论在合理成本定义下是否稳定。
- 仍需证据：逐单位成本数据和Q*不确定性传播。
- 预期影响：medium；判断置信度：medium。

**给作者的问题：**

- register 的 universe 完全由作者 ledger 定义时，什么证据能让第三方相信一个关键主张没有从 ledger 和正文同时遗漏？
- 所谓独立 annotator 是几名、如何招募、是否在分析冻结前完成、是否真正对 register ID 与作者期望盲法？
- 第8节为何说在一个外部稿上的91个 signal hits 证明 scanner generalizes，而没有 ground-truth precision/recall 或失败检测目标？
- 能否在相同32k配置中显式切换 guard on/off 多轮运行，保存完整峰值与每请求时延，以直接归因而非依赖一次历史 abort？
- 既然标题主题是 withdrawn KV-selective study，作者认为主会读者可复用的科学知识究竟是 serving 结果还是报告流程；后者相对已有 assurance case/模型卡的经验增量如何量化？

**能提高评分的证据：**

- 完成 selector、adapter-matched、16k guarded 和端到端并发的预先定义复跑，恢复一个可评价的系统问题。
- 在多项目隐藏 gold corpus 上证明 register/gate 有可接受的 precision/recall，并优于简单基线。
- 受控 guard on/off ablation 复现内存缺陷归因，保存请求级原始记录。
- 多名真正独立标注者验证 ledger 的语义覆盖并报告一致性与遗漏。

**会降低评分的证据：**

- 受控ablation显示guard不是历史内存异常的主要原因。
- held-out释义测试发现pattern gate大量漏过未授权的serving/throughput主张。
- 独立审计发现关键主张同时从作者ledger和register遗漏。
- 冻结数值、Q*输入或表2状态无法从所称记录重建。

**伦理标记：** 否。论文不涉及人类受试者、个人数据或模型发布。主要风险是冻结数字被脱离逐行scope复用；作者已在第7节明确禁止这些读法。

**LLM 使用披露：** 本审稿由隔离运行的 AI 子代理 R3 完成，仅用于内部投稿前质量控制；未与其他评审通信，未读取作者计划、旧评分或其他评审输出。

**评审限制：**

- 仅审阅指定冻结 PDF；按隔离要求未访问其匿名bundle、scanner、代码、日志或SHA清单，因此所有machine-check、复跑和冻结记录声明均未外部执行核验。
- 未联网，未核验外部引文、相关工作优先权、模型/框架版本或公开讨论；相关不确定性不应被解释为已验证。
- 已完整读取指定 KV-cache 校准说明并逐页视觉核查14页 PDF；未见图表截断，只有附录末页较多留白。

### P1

#### 交付与来源状态

- 选定输入：`/Users/liuhanzuo/Downloads/P1-paper (1).pdf`
- 源状态：`exact_main_tex_reconstructed_dependency_tree`
- 润色 PDF：`/Users/liuhanzuo/MacLLM-Bench/output/pdf/paper_polish_20260826/P1-polished.pdf`
- 冻结 SHA-256：`074b0ec75463f223d717e02f07b0e792e9d8865bc2a08328ea8163f2ef081a88`
- 总页数：25；主文状态：主文在第9页结束。
- 版面核验：pass；构建：pass。
- 旧评分基线：2,2,2；旧中位数：2。

#### 三评量化结果

| 审稿人 | 角色 | Overall | Confidence | 建议 | Soundness | Presentation | Contribution | 当前上限 | 必需修改后预测 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| R1 | 新颖性与定位 | 2 | 4 | 拒绝 | 2 | 2 | 2 | 2 | 4 |
| R2 | 技术正确性 | 2 | 4 | 拒绝 | 2 | 2 | 1 | 4 | 4 |
| R3 | 实验严谨性 | 4 | 4 | 略低于接收线 | 2 | 2 | 2 | 4 | 6 |

三评中位数为 **2**，均值 2.67，跨度 2，接收侧票数 0/3。

#### 编辑记录

- [结构审计](work/P1/structure_audit.md)
- [语义锁](work/P1/semantic_lock.md)
- [修订日志](work/P1/revision_log.md)
- [待核验事项](work/P1/needs_verification.md)

**修订日志原文：**

> # P1 revision log
>
> ## Scope
>
> Evidence-preserving manuscript polish. No experiments, independent citation verification, or reviewer scoring were performed.
>
> ## Source changes
>
> - Replaced the generic title with a title that states the auditable portfolio-evaluation contribution.
> - Rewrote the abstract to present the tier counts, fixture-versus-calibration distinction, and independence boundary in a single coherent evidence hierarchy.
> - Tightened the introduction and reduced repeated process framing.
> - Condensed the open-items conclusion and added a final sentence that separates reusable governance infrastructure from unclosed scientific claims.
>
> ## Semantic safeguards
>
> - Preserved all tier counts, status labels, artifact names, manifest/pin claims, equations, tables, figures, citation keys, and open-item states.
> - Did not recast process independence as party independence or third-party execution.
> - Did not close the open external-novelty MERGE decision or treat fixture evidence as scientific calibration evidence.
>
> ## Verification
>
> - Built from the isolated source with shell escape disabled, using BibTeX and two final LaTeX passes.
> - Checked extracted text, current citation/reference warnings, main-text boundary, and rendered pages.
> - Remaining source-dependent items are listed in `needs_verification.md`.

**待核验事项原文：**

> # P1 needs verification
>
> - All replay, checker, manifest, and pin claims were preserved but not executed because the audit workspace named by the source was not supplied in this isolated build copy.
> - The framework's external novelty boundary relative to MLOps governance remains an author decision; MERGE is correctly left open.
> - No prose edit upgrades the internal process-independence records to party independence or third-party execution.
> - The term “fire archive” remains manuscript-specific; the revised abstract defines it before relying on it.

#### R1（新颖性与定位）完整评议

**论文概述：** 论文提出 AUDIT-GATE，用静态、零 GPU 检查判断多代理研究 archive 的预注册决策是否机械可达和可重放，输出 PASS、FAIL 或 POSTPONE，并附冻结输入、checker 和单命令证书。框架包括五字段 endpoint、Clopper–Pearson 可决定性、正控、manifest 五状态、健康臂非退化和 machine-string 投影。六案例 ledger 与七个其他 seat archive 的 decisive cell 在冻结转录层得到 9 PASS/5 FAIL/0 POSTPONE；当前机械提取层的七单元变为 3 PASS/2 FAIL/2 POSTPONE。八个人工 defect fixtures 覆盖未触发分支，但论文承认没有独立标签总体，v2 校准所需的至少 118 个 defect-free archives 仍 blocked。

**最强的已核实贡献：** 论文把“可重放”具体化为记录时 anchors、冻结 checker 和单命令接口，并对自己的首版未固定、自身 manifest 缺失、规则事后修复和 frozen/current verdict 漂移作了可审计披露；作为内部工程纪律，这比只写一段 reproducibility 声明更具体。

**维度理由：**

- Soundness：AUDIT-GATE 的三态输出、endpoint quintuple、manifest 状态和证书规则在 PDF 内定义得较完整，作者也如实披露第一版实现与预注册相冲突、未被固定且不可重放。问题在于 field corpus 只有同一 run 的 14 个单位，自然样本从未触发 POSTPONE，八个补充 fixture 又由 checker 作者构造；缺少独立标签总体，所以检测率、漏报率、误杀率和科学有效性均未建立。
- Presentation：主文能说明 PASS 只代表机械可操作而非科学正确，图表无视觉错误。但 25 页包含 R27/R31/R41/R42/R52/R83 多个版次、冻结与当前两套 tally、26 个 machine strings、18 张表和大量内部路径/sha16；核心算法、验证状态与历史考古交织，显著妨碍读者评估。
- Contribution：把 endpoint registration、positive controls、manifest hashes、缺失标记和 verdict projection 组合成可重放证书有一定工程价值，但这些元素与 provenance、reproducible builds、artifact evaluation、CI/assertion 和 assurance cases 高度邻近。当前没有自然独立数据上的性能或第三方采用，故贡献窄且新颖性未充分建立。

**优点：**

- 持续强调机械 operability 不等于科学正确，PASS 不被用作研究结论背书。
- 保留首版不可重放的 self-FAIL、规则双向不包含关系以及当前/冻结 tally 的不一致，没有事后抹平。
- 为未覆盖的 POSTPONE、PRIM3 状态和不可达主分支提供显式 fixture，并承认它们只证明 branch execution。
- 逐页视觉检查覆盖全部 25 页，图表未见裁切、重叠或缺页。

**问题与可验证修复：**

##### P1-I1 · 主要 · 新颖性、重要性、引文完整性

- 位置：第 1–2 页第 1 节 Contributions；第 7–8 页第 4 节 Related Work；第 22–23 页附录 F
- 观察证据：五字段注册、置信区间可达性、must-fail control、hash/absence manifest、字符串到三态投影和冻结重放均是已有统计或软件治理构件的组合。相关工作承认 provenance、versioning、reproducible builds、reporting standards 与 production assertions，但只用阶段/对象的文字差异定位，没有实现基线或展示现有工具无法检测而 AUDIT-GATE 能可靠检测的自然案例。
- 重要性：当前容易被理解为项目特定 MLOps common sense 的重包装；论文自己甚至设定 MERGE 条款来承认这一风险。没有清楚的概念或实证增量，难以达到 ICLR 主会贡献门槛。
- 必需修复：定义一个项目无关的 threat model 和形式化保证，逐轴比较现有 assurance/artifact/attestation 工具；在独立自然 archives 上运行这些基线和 AUDIT-GATE，展示新增检测类别或显著减少人工成本。
- 验证标准：每个 primitive 都应有最接近先例、相同输入上的基线输出和本文不可替代的增量；若增量只剩组合接口，应诚实定位为工程/工具论文并提供实际采用证据。
- 仍需证据：系统文献对照、可运行基线、独立项目任务和预设效用指标。
- 预期影响：high；判断置信度：high。

##### P1-I2 · 致命 · 实验严谨性、技术正确性、重要性

- 位置：第 5–7 页第 3 节 Field Audit；第 8–9 页第 5–6 节；第 19–20 页表 11–12
- 观察证据：14 个 field units 全来自同一 live run，冻结层只出现 PASS/FAIL 而没有 POSTPONE；八个缺陷 fixture 由 checker 作者构造。论文明确没有 independently labelled archive population，也不报告 detection 或 false-alarm performance；能够判定 5% false-kill 目标的 v2 设计要求至少 118 个独立 defect-free archives，目前 human_pending/blocked。
- 重要性：核心产物是一个审计 gate，但现有证据只能说明作者写的分支会在作者写的输入上执行，不能说明它对自然 archive 有效。没有独立校准，PASS/FAIL 的实际可靠性和工具价值未知，这是当前接收的决定性缺口。
- 必需修复：完成独立、盲标的自然 archive 校准：由非作者定义 operability 金标准，按项目留出，冻结 checker 后报告逐类检测、误杀、POSTPONE、覆盖和置信区间；同时运行最接近简单规则/人工 checklist 基线。
- 验证标准：标签制定者不得接触 checker 规则，开发/测试项目必须隔离；预注册的 false-kill≤5% 和 detection 下界必须在独立测试集上可判定，而非用 authored fixtures 充数。
- 仍需证据：至少达到论文 v2 功效设计的独立 defect-free 样本、多个缺陷家族、第三方标签与执行日志。
- 预期影响：high；判断置信度：high。

##### P1-I3 · 主要 · 技术正确性、可复现性、限制与负责任表述

- 位置：第 2 页引言末；第 4 页第 2.6–2.7 节；第 5–6 页第 3.2 节；第 20–21 页表 12–13
- 观察证据：第一版 group implementation 与预注册 predicate 冲突，看到结果后才改且旧代码未固定、无法重放；PRIM3 从三臂扩为五状态、verdict map 扩展、G2 从常量赋值改为读取谓词。注册 predicate 与 shipped rule 在枚举域上双向不包含（75 与 21 个分歧 multiset）。冻结七单元 tally 6/1/0，而当前机械层为 3/2/2。
- 重要性：这些不是局部 bug，而是决定 verdict 的语义与证据提取发生了观察后变化。论文诚实披露值得肯定，但标题中的“pre-registered”只能描述部分对象，不能给当前框架确认性地位。
- 必需修复：将当前版本明确称为 observation-informed v2，把冻结层降为构建史；重新预注册完整规则、parser、map、group semantics 和校准指标，然后只在未见的新 archives 上评测。删除任何把旧 decisive cell 当作当前规则确认性验证的表述。
- 验证标准：从新预注册 hash 到测试结束，所有决定 verdict 的代码和语义必须字节冻结；测试集不可参与规则设计；当前与历史输出只保留一个主层级。
- 仍需证据：全新 held-out decisive cell、完整冻结链和第三方复跑。
- 预期影响：high；判断置信度：high。

##### P1-I4 · 主要 · 技术正确性、实验严谨性、清晰度

- 位置：第 2–4 页第 2 节 Framework；第 17–20 页 Appendix C/D；第 23–24 页 Appendix G/H
- 观察证据：AUDIT-GATE 依赖 filesystem/string facts、每 archive 一个定制 parser 和关键词映射。字段存在、hash 稳定或 computor 字符串可达，并不证明实现真的计算注册 estimand；解释过的 drift 只依赖 operator-supplied cause registry。当前 seat C/H 还因 G2/threshold 无法机械解析而 POSTPONE。
- 重要性：形式完整性可被语义错误或无关计算轻易满足，而 schema-specific parser 限制跨项目迁移。工具可能稳定重放一个错误 verdict，却无法识别最关键的科学或软件语义偏差。
- 必需修复：明确可保证与不可保证的形式边界，并引入机器可读 typed schema、数据流/调用图绑定或可执行 assertion，把 endpoint 字段连接到实际计算；对 fabricated-but-well-formed archives 做独立对抗评测。
- 验证标准：构造字段与 hash 均合法但计算错 estimand、阈值或数据集的自然istic defects；工具应按预设目标检出，且不依赖为每个样本手写专用 parser。
- 仍需证据：语义级缺陷集、通用 schema/validator、跨项目迁移和对抗结果。
- 预期影响：high；判断置信度：high。

##### P1-I5 · 主要 · 清晰度、可复现性

- 位置：第 1–9 页主文；第 10–25 页附录，尤其 Tables 6–18
- 观察证据：稿件维护冻结/当前两层 verdict、多个 round 与 judge 版本、26 个 unit strings、内部 seat 名称、路径和数十个 sha16；部分主文主张需到 Appendix H 才能得到 canonical current reading。第 25 页仅顶部少量文字，其余大幅留白。
- 重要性：审计工具论文应降低认知和核验成本；当前呈现本身需要读者进行复杂谱系审计，并使方法与一次特定多代理运行难以分离。
- 必需修复：把当前规范、算法伪代码、一个项目无关示例和唯一验证表放在主文；将所有 round archaeology、完整字符串映射与 hash 清单移入 artifact。压缩或删除不再承担当前论证的历史表。
- 验证标准：不了解该 run 的读者应能在主文中唯一确定当前算法、输入 schema、输出语义、验证数据和限制，并在不查内部 seat/round 的情况下复现一个示例。
- 仍需证据：不需要新实验来改善结构；第三方可用性研究可进一步验证重构。
- 预期影响：high；判断置信度：high。

**给作者的问题：**

- AUDIT-GATE 相对现有 artifact-evaluation checklists、reproducible-build attestations、data/model assertions 和 assurance cases 的最小新能力是什么，如何被实验而非术语证明？
- endpoint 的 readout/side/n/threshold/path_id 由谁语义验证？若 archive 提供形式完整但错误或指向无关计算的 quintuple，当前静态 gate 会怎样处理？
- 为什么标题仍称 pre-registered，而 group rule、PRIM3 enumeration、G2 predicate 与 verdict map 都在观察后改变，且第一版实现未固定、不可重放？
- 能否由工具作者之外的团队，在未查看预期标签的情况下，从零安装 bundle 并审计一个新项目，报告耗时、失败点和与人工金标准的一致性？

**能提高评分的证据：**

- 在独立自然 archive 总体上完成预注册、盲标的检测/误杀校准，并达到论文自己设定的决策界限。
- 由非作者团队完成 clean-room 安装、运行和新项目采用，展示实际节省与错误发现。
- 证明相对现有 artifact/attestation/assurance 工具的新增语义级检测能力。
- 用全新 held-out cell 验证当前冻结规则，而不是复用参与规则形成的同一 run。

**会降低评分的证据：**

- 第三方发现字段齐全、hash 稳定但 estimand 错误的 archive 仍被 PASS，且此类错误常见。
- 独立校准显示误杀或漏报远高于注册界限。
- clean-room 第三方无法凭 bundle 重放，或 frozen/current 层仍会被运行脚本意外改写。

**伦理标记：** 否。论文对其他 seat archives 使用匿名键，PDF 未显示个人敏感信息。若未来跨团队审计真实研究档案，应明确访问授权、秘密信息清理与责任归属；当前不足主要是方法有效性而非需触发的伦理违规。

**LLM 使用披露：** 本审稿由隔离的 AI 子代理 R1 生成，仅用于内部投稿前质量控制；该子代理只读取冻结 PDF 与指定审稿规则，未与其他评审通信，也未读取其他评审输出。

**评审限制：**

- 本评审严格只读取指定冻结 PDF；未读取或执行论文列出的 checker、bundle、其他 seat archives、路径、hash 对应文件或任何旧稿。
- 因此无法验证证书命令、sha16、第三方 replay 记录或文件系统状态，相关可重放性只按 PDF 证据评价。
- 按任务要求未联网，也未核验外部引文或补充文献；新颖性仅相对于 PDF 的相关工作判断。
- 逐页视觉核查覆盖全部 25 页；未发现裁切、重叠或缺页，第 25 页有大量空白。

#### R2（技术正确性）完整评议

**论文概述：** 论文提出 AUDIT-GATE，对多代理研究运行的 fire archive 做零 GPU 静态审计，并输出 PASS、FAIL 或 POSTPONE。检查涵盖 endpoint 五元组注册、所谓 computor reachability、boxed controls、manifest/anchor 状态、unreachable primary leg 与 machine-string 映射，且为 verdict 提供冻结输入、checker 和复跑命令。实证包括同一 live run 的六案例 ledger、来自其他 seats 的七 archive decisive cell，以及八个作者构造的 defect fixtures；frozen tier 为 9 PASS/5 FAIL/0 POSTPONE（两组共 14 单元），decisive cell 当前机械 tier 为 3 PASS/2 FAIL/2 POSTPONE。独立自然语料校准和第三方执行均未完成。

**最强的已核实贡献：** 第 2–6 页 §2–3 和第 18–19 页表 11 最可靠地表明：作者实现了一套可重放的三态静态规则与证书接口，并用明确的缺陷注入覆盖 live corpus 未触发的 POSTPONE、CP non-crossing、PRIM4 和五种 anchor 状态。第 8–9 页表 3 的设计数也内部一致，例如 U(0/40)=0.0881 导出 n*=122、补 82 行，36/36 的下界为 0.9026，1/118 的上界为 0.0463。它验证的是规则分支与设计计算，不是自然 archive 上的分类效度。

**维度理由：**

- Soundness：Clopper–Pearson 设计数、三态投影和若干 branch fixture 的内部计算看起来一致，且作者诚实披露 post-observation amendments。不过核心 PRIM3 只检查是否存在 operator-supplied cause；论文亲自证明伪造 cause 即可把 FAIL 变 PASS。最终 verdict map、PRIM3、G2 与 tier reading 又在观察后改变且首版被覆盖，因而不能把当前整体框架视为预注册、独立验证的审计。
- Presentation：框架对象、五元组、PASS/FAIL/POSTPONE 与 frozen/current tier 均有定义，限制也披露充分。但 25 页充满轮次、seat、内部 sha16、历史 machine strings 与多套版本；核心当前结论需要在主文和附录 H/Table 17–18 之间来回核对，图表字号与信息密度偏高。
- Contribution：当前证据证明的是作者编写的静态规则能在同一运行档案和作者制作的 fixtures 上触发预定分支。没有独立标注的自然 archive population、第三方执行、校准后的 detection/false-alarm 性能或实际采用；这更接近一个透明的内部 QA 工具与设计提案，而非已验证的通用研究方法。

**优点：**

- 对第一版 checker 与预注册 diversity predicate 冲突、版本被覆盖且不可复算，明确按自身规则记为 self-FAIL。
- 严格区分 frozen auditor-transcription tier 与 current mechanical-extraction tier，没有把 6/1/0 与 3/2/2 混成一个结果。
- fixture branch coverage 与 detection/false-alarm performance 被明确区分；作者没有把自制缺陷的对角混淆矩阵称作独立校准。
- 把未满足的 PASS/KILL/MERGE/STOP-PASS 条件、第三方未执行和 calibration gap 放在主文结论中，而非只藏在附录。

**问题与可验证修复：**

##### P1-R2-01 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第 1–3 页标题、§1、§2.6；第 5–6 页 §3.2；第 19–20 页表 12
- 观察证据：论文说明 verdict-state map、PRIM3 enumeration、G2 predicate 与 tier reading 均在观察后改变；第一版实现还与注册 diversity predicate 冲突，并在固定前被替换，无法重放。只有特定 decisive cell/部分规则在运行前冻结，而不是当前整套框架。
- 重要性：当前规则是在看到失败模式后扩充和修正的，branch coverage 与 field tally 因而同时承担开发和评价角色。透明披露值得肯定，但不能消除自适应设计造成的乐观偏差；标题会把局部预注册误读为完整方法的确认性验证。
- 必需修复：把标题与摘要改为‘retrospectively developed, transparently versioned’并逐条标明冻结时点；或在全新 archive corpus 上先冻结当前完整规则、state map、tier 与分析，再做一次真正确认性评价。
- 验证标准：公开不可变的 current-rule digest 与时间戳，在未被规则作者看过的新 corpus 上一次性运行；任何规则变化都进入下一版本而不改本次主结果。
- 仍需证据：规则冻结记录、独立确认 corpus 和原样运行 transcript。
- 预期影响：high；判断置信度：high。

##### P1-R2-02 · 主要 · 技术正确性、可复现性、限制与负责任表述

- 位置：第 3 页 §2.3 PRIM3；第 11–12 页 §A.3 与证书讨论；第 18–19 页表 11
- 观察证据：PRIM3 把‘文件 hash 漂移但有 recorded cause’判 PASS、无 cause 判 FAIL。第 12 页明确写出：给 copied-anchor mutation 添加 fabricated cause 会从 FAIL_ANCHOR_DRIFT_UNEXPLAINED 变成 PASS_ANCHOR_DRIFT_EXPLAINED，故 explained drift 仅与 cause registry 一样可信。
- 重要性：最需要 provenance 审计的操作者同时能供应让自己过关的 cause，且规则不验证 cause 的时间、签名、内容或与 drift 的因果对应。这样 PASS 可被事后文字白洗，破坏 manifest gate 的安全意义；允许 reviewer 自行忽略 registry 也不恢复单值 verdict。
- 必需修复：将未经验证的 explained drift 至少判 POSTPONE，而非 PASS；要求 cause 在变更时由不可改日志/外部签名绑定到旧新 digest，并由非作者 verifier 检查具体 diff 与所述原因一致。
- 验证标准：用无关、事后、伪造和正确 cause 的盲测 mutation；只有时间锁定且证据匹配的 cause 才可通过，并报告 verifier 间一致性。
- 仍需证据：防篡改 cause provenance、独立验证协议和对抗测试结果。
- 预期影响：high；判断置信度：high。

##### P1-R2-03 · 主要 · 实验严谨性、重要性、限制与负责任表述

- 位置：第 1 页摘要；第 5–6 页 §3；第 8–9 页 §5–6与表 3
- 观察证据：14 个 frozen field units 全来自同一 live run，只覆盖 PASS/FAIL；八个 POSTPONE/branch cases 是作者构造。七个 other-seat archive 没有独立 operability labels，R42/R52 仍使用相同 workspace、filesystem 与 seat credentials。作者明确说明无自然语料 detection/false-alarm rate、无第三方执行，v2 至少 118 个独立 defect-free archives 仍 human_pending。
- 重要性：当前数据只能证明代码能在选择过的内部案例上运行，不能估计对自然档案的漏报、误报、跨项目泛化或实际 reviewer utility。‘true detection’也缺少独立 adjudication，因此方法有效性尚未建立。
- 必需修复：在作者未参与生成或选择的多项目 archive 样本上，由独立标注者建立 operability truth；冻结规则后报告每类性能、置信区间、项目级划分和第三方从 bundle 复跑成功率。
- 验证标准：执行预先写明的 v2 校准：至少满足设计样本量，按 project/seat 去重与切分，报告 false-open、false-kill、POSTPONE、标注一致性及失败原因。
- 仍需证据：独立自然 archive population、标签、第三方运行与完整混淆矩阵。
- 预期影响：high；判断置信度：high。

##### P1-R2-04 · 主要 · 技术正确性、重要性、清晰度

- 位置：第 1–3 页 §1、§2.1、§2.6；第 22 页 Appendix F
- 观察证据：PRIM1 的 endpoint reachability 基于 filesystem/string-level facts，即在 consumer script 中找到计算符号；证书复算的是 audit verdict，而非执行被审计的科学 endpoint。Appendix F 又明确承认 static inspection 看不到 runtime crash。
- 重要性：存在文件或代码字符串并不保证 import 成功、分支可达、依赖可用或 endpoint 真能对注册输入运行。普通读者会把 mechanically operable/replayable 理解为可执行，而当前 PASS 只证明静态引用与审计自身可复算，构念明显更窄。
- 必需修复：将术语改为 statically registered/referenced；若保留 operable，则加入最小沙箱执行：加载依赖、解析真实注册输入、到达 endpoint 并产生类型/shape 合法的输出，同时记录环境锁定。
- 验证标准：构造包含 dead branch、broken import、版本不兼容、运行时异常但字符串存在的 archives；静态 PASS 不得成为最终 PASS，应由动态层 FAIL/POSTPONE。
- 仍需证据：动态 reachability 层、环境 manifest 和自然故障测试。
- 预期影响：high；判断置信度：high。

##### P1-R2-05 · 次要 · 清晰度、可复现性

- 位置：第 5–6 页表 1；第 10–20 页 Appendix A–E；第 24–25 页 Appendix H–I
- 观察证据：同一 decisive cell 同时出现 frozen auditor-transcription、R42 repaired judge、R52 mechanical extractor 和 current canonical pin；关键 unit 的 PASS/FAIL/POSTPONE 随 tier 改变。尽管作者最终在 Appendix H 单值化，主表仍以前一 tier 为主体且需跨多个密集附录追溯。
- 重要性：论文研究的正是可复算与单值 verdict；高密度版本考古增加读者误引 frozen 9/5/0 或 decisive 6/1/0 为当前结果的概率，也不利于独立核验。
- 必需修复：主文只保留当前机械 verdict 作为主表，把 frozen tier 明确放历史列/附录；提供一张从 archive→checker hash→current verdict→historical divergence 的规范化索引。
- 验证标准：让未参与作者仅凭主文回答每个 unit 当前 verdict 与可复跑命令；若出现 tier 分歧即修订。
- 仍需证据：简化的 canonical table 与读者核验记录。
- 预期影响：medium；判断置信度：high。

**给作者的问题：**

- 为何 PRIM3 将‘存在一条 cause 字符串’直接映射为 PASS，而不要求 cause 在 drift 前时间锁定、由独立方签名或由内容证据验证？
- 既然标题称 Pre-Registered，能否清楚列出哪些最终规则在任何 field outcome 之前冻结，哪些只是观察后的框架开发，并把标题限于真正预注册的 decisive cell？
- 七个 other-seat archives 是否由本研究作者挑选、转录或参与构建？在没有独立 operability truth label 的情况下，‘one true detection’的 true 由谁 adjudicate？
- PRIM1 的 computor-present 字符串/文件检查如何排除 dead branch、错误 import、版本不兼容或运行时崩溃？若不能，为什么称 endpoint mechanically reachable 而非 statically referenced？
- v2 calibration cell 所需的至少 118 个 defect-free archives 和外部 labeller 是否有可执行采集协议、去重单位与 adjudication plan？

**能提高评分的证据：**

- 冻结完整当前规则后，在独立标注的自然 archives 上完成预先设计的校准并报告 false-open/false-kill。
- 把 PRIM3 的 operator-supplied cause 改为可时间验证、独立核验的 provenance 证据。
- 由真正第三方在隔离环境执行 bundle，并加入能发现 dead branch/runtime failure 的动态 reachability 层。

**会降低评分的证据：**

- 独立自然语料显示规则高漏报、高误杀或多数案例只能 POSTPONE。
- 对抗测试表明 fabricated cause 或同义代码结构可系统性绕过核心 gate。
- 第三方无法从冻结证书重放当前 canonical verdict，或重放结果依赖未声明环境状态。

**伦理标记：** 否。论文不涉及人类受试者或敏感数据；主要治理风险是审计 PASS 被误读为科学正确或被操作者通过自报 cause 规避，属于方法可信度而非需升级的研究伦理事件。

**LLM 使用披露：** 本审稿由隔离的 AI 子代理 R2 生成，仅用于内部投稿前质量控制；未与其他评审通信，未读取其他评审结果、作者计划、编辑上下文或历史评分。

**评审限制：**

- 遵循隔离要求未联网，未核验外部引文、相邻审计系统或论文所述软件生态。
- 仅审阅冻结 PDF；未读取或运行 checker、bundle、manifest、fixtures、sha16 对象或 calibration files，因此所有 artifact hash、命令输出和跨 tier 对齐仅按文稿判断。
- 没有其他评审、作者计划、旧稿、修订日志或目标评分上下文；评分是对该快照的独立绝对判断。
- PDF 共 25 页，已全文阅读并逐页视觉核查；未见解析缺页，但多张附录表字号小、信息密集，视觉阅读负担较高。

#### R3（实验严谨性）完整评议

**论文概述：** 本文提出 AUDIT-GATE，用 PASS/FAIL/POSTPONE 审计多代理研究 archive 中一个预注册判定点是否机械可操作和可重算。框架检查端点五元组、置信包络可达性、must-FAIL正对照、锚点的五种状态、不可达主腿和总 verdict 映射。现场证据包含六案例 ledger 与同一次运行中七个其他 seat 的 archive：冻结版为6 PASS/1 FAIL/0 POSTPONE，当前提取器因漂移变为3/2/2。八个作者构造 fixture 覆盖未在自然 archive 中触发的分支。论文披露第一版实现与注册的多样性谓词矛盾且在固定前被覆盖，并说明自然 archive 独立校准仍被阻塞，需要至少118个独立标注的无缺陷 archive。

**最强的已核实贡献：** 最可信的是框架的总 verdict 映射与合成分支演示：表11/图2所述八个 defect-injected fixture 让 POSTPONE、PRIM4、各锚点状态等代码路径按预期触发，且作者没有把这解释成真实 archive 上的检测率。它证明当前 runner 的若干机械分支可执行，但不证明 gate 的分类有效性。

**维度理由：**

- Soundness：三态 verdict、端点可达性、正对照、锚点状态和总映射在形式上较完整，合成 fixtures 也覆盖了多个分支。但第一版实现违背注册规则且不可重算，现版本是在观察后修复；没有独立标注的自然 archive 总体或第三方执行，无法估计 false-kill 和 detection 性能。
- Presentation：作者异常坦诚地披露修订、漂移和未闭合校准，核心表1可读。可是25页中内部轮次、sha16、PRIM/G/R编号和大量证书表占据主导，读者很难快速分辨方法定义、当前证据和未来设计；附录重放命令还有一处缺空格。
- Contribution：把“统计读数是否正确”与“预注册判定是否可被第二台机器重算”分开，是有用的工程问题。当前实现和证据仍局限于同一多代理运行的内部 archive，自制 fixture 只能证明分支可执行，尚未显示该 gate 对真实研究档案的准确性或普适价值。

**优点：**

- 第2节将 PASS 明确限定为判定可重算而非科学结论正确，避免概念外推。
- 第一版实现违背注册谓词且被覆盖的事实被作为 self-FAIL 公开，而不是静默替换。
- 表1同时保留冻结 tier 与当前漂移 tier，说明差异来自 archive/provenance drift 而非简单重算不确定性。
- 图2逐分支区分 live evaluated、live fired 与 fixture fired，明确自制 fixture 只证明机械路径。
- 第4节给出至少118个独立无缺陷 archive 的校准设计，并承认当前不存在满足条件的独立总体和第三方执行。

**问题与可验证修复：**

##### P1-R3-01 · 主要 · 实验严谨性、技术正确性、重要性

- 位置：摘要、第3.3节与第4节表3（PDF第1、7–9页）
- 观察证据：现场 archive 均来自同一live run；八个缺陷 fixture 由作者按 checker 逻辑构造。论文明确没有独立标注的自然 archive 总体、没有第三方运行，且false-kill校准至少需要118个独立无缺陷 archive。
- 重要性：当前数据只能证明代码在已知构造输入上能走到相应分支，不能估计对真实 archive 的误杀、漏检或外部可操作性。没有这些指标，就无法评价 gate 是否比人工审计或简单 schema 检查更有用。
- 必需修复：在作者之外收集并冻结独立、多项目自然 archive，由盲法标注者建立 operability gold；同时包含真实缺陷与无缺陷样本并预先冻结判定阈值。
- 验证标准：按表3注册设计至少完成false-kill样本量，另对每类真实缺陷做有功效的检测评估；报告项目级cluster置信区间、precision/recall及与基线比较。
- 仍需证据：独立自然 archive、第三方 gold 标签、第三方重放日志和误差矩阵。
- 预期影响：high；判断置信度：high。

##### P1-R3-02 · 主要 · 实验严谨性、可复现性、限制与负责任表述

- 位置：摘要、第2.6节“Freeze-discipline disclosure”及附录C表12（PDF第1、4、20页）
- 观察证据：第一版实现与注册的diversity predicate矛盾，而且在固定哈希前被覆盖，故原 verdict 不可重算；随后 verdict map、PRIM3、G2和组规则均在观察结果后修复。
- 重要性：当前证据评估的是后验修订系统，不能凭原始注册获得前瞻误差控制。标题和摘要若不精确区分，会让读者误以为当前 gate 的设计与主结果在观察前已冻结。
- 必需修复：把当前证据明确称为开发/修复集；对最终代码、谓词、verdict map与分析完全重新预注册，并在新 archive 上前瞻运行一次。
- 验证标准：提供新协议时间戳与代码摘要，在无任何规则变更下完成独立验证；任何 amendment 必须使该验证轮失效并另开新轮。
- 仍需证据：干净的前瞻注册、冻结实现和独立验证结果。
- 预期影响：high；判断置信度：high。

##### P1-R3-03 · 主要 · 实验严谨性、限制与负责任表述、重要性

- 位置：第3节 field audit 与表1 Panel B（PDF第5–7页）
- 观察证据：七个所谓 external fire archives 属于同一运行中的其他 seat，并共享workspace、流程和潜在凭据/模板；论文也说明 replay process 不是party-independent。
- 重要性：这些单位并非独立项目样本，相关缺陷和共同基础设施会使6/1或3/2/2计数无法外推。把它们称为external容易高估跨环境验证程度。
- 必需修复：改称cross-seat internal archives并限制解释；真正的外部验证应来自不同实验室、不同工作流和独立操作者。
- 验证标准：由至少两个作者团队仅凭clean-room bundle执行重放，记录环境、失败原因和与gold verdict的一致性。
- 仍需证据：跨项目第三方 archive 与独立执行记录。
- 预期影响：high；判断置信度：high。

##### P1-R3-04 · 主要 · 技术正确性、实验严谨性、清晰度

- 位置：第2.7节 group verdict、表1及附录C表12（PDF第4–7、20页）
- 观察证据：EXECUTION_VALID 的组规则要求投影后出现至少两种状态；旧版与修订版规则在枚举中存在双向 disagreement。论文没有从 archive operability 的目标语义推导“状态多样性”为必要条件。
- 重要性：任意多样性条件可能拒绝全部一致且正确的 archive，或接受包含多种状态但整体不可用的组。后验改规则进一步说明该条件不是稳定构造。
- 必需修复：给出 group verdict 的外部目标定义与损失函数，从该目标推导聚合规则；将状态多样性降为诊断而非通过条件，除非有独立校准支持。
- 验证标准：在所有状态多重集及独立标注案例上构造混淆矩阵，证明规则对预定义operability目标的单调性和错误界限。
- 仍需证据：目标语义、形式性质及独立标注案例验证。
- 预期影响：high；判断置信度：medium。

##### P1-R3-05 · 主要 · 技术正确性、可复现性、限制与负责任表述

- 位置：第2.4节PRIM3、附录A/表4与附录C状态映射（PDF第3、11–12、20–22页）
- 观察证据：锚点哈希漂移只要在cause registry中有记录即可投影为PASS_ANCHOR_DRIFT_EXPLAINED；合成fixture也正是通过加入已登记原因触发PASS。文中未给出cause记录的独立授权、签名或事前不可变机制。
- 重要性：若同一作者可以在看到漂移后补写原因，任意篡改都可能从FAIL变成PASS，削弱哈希锚点的完整性保证。可解释性记录不等于可信授权。
- 必需修复：要求cause在变更前由独立权限签名并进入append-only日志；事后原因只能产生POSTPONE/人工仲裁，不能自动PASS。
- 验证标准：用未授权事后cause、签名错误、时间倒置和registry删除fixture做攻击测试，确认均不能PASS；第三方验证日志链。
- 仍需证据：不可变cause registry规范、签名/时间证明和攻击测试。
- 预期影响：high；判断置信度：high。

##### P1-R3-06 · 次要 · 清晰度、可复现性

- 位置：附录H重放命令，PDF第24页表18附近
- 观察证据：一处命令被排版为“python3analysis/audit_gate.py”，python3与脚本路径之间缺少空格；前文存在正确形式，但该处直接复制会失败。
- 重要性：论文把单命令重放作为核心接口，附录中的可复制错误会给独立执行制造不必要障碍。
- 必需修复：修正空格并对PDF中所有16条命令从干净环境复制执行。
- 验证标准：从最终PDF逐字提取命令，在clean-room bundle中逐条运行并保存exit code。
- 仍需证据：修正后的PDF与命令级重放记录。
- 预期影响：low；判断置信度：high。

**给作者的问题：**

- 标题为何仍称“pre-registered”，而当前可重算实现、PRIM3和组规则均在观察后修订；预注册究竟约束了哪个最终可评价对象？
- 七个“external archives”是否来自不同项目、不同作者和独立凭据，还是仅为同一运行中的其他 seat；它们之间的依赖如何处理？
- EXECUTION_VALID 要求投影状态多样性时，为什么“至少两种状态”是 operability 的必要语义，而不是为当前案例选择的经验规则？
- DRIFT_EXPLAINED 的 cause registry 由谁授权和不可变保存；攻击者或作者事后添加原因能否把任意 drift 变成 PASS？
- 未来118个无缺陷 archive 只校准 false-kill；检测侧36/36每 family 的缺陷类型、抽样框和独立标签将如何建立？

**能提高评分的证据：**

- 在完全冻结的最终gate上完成真正前瞻、跨项目、第三方执行的验证。
- 至少118个独立无缺陷archive和有功效的真实缺陷样本给出可接受false-kill/漏检界限。
- group verdict从明确外部目标推导，并在独立gold上优于简单schema/hash基线。
- cause registry采用事前签名的append-only授权，攻击fixture不能把未授权漂移升级为PASS。

**会降低评分的证据：**

- 第三方无法只凭clean-room bundle重放冻结verdict。
- 独立自然archive校准显示高false-kill或系统漏检。
- 最终规则继续在观察验证结果后修改而仍被称为同一预注册评估。
- 未授权cause或状态映射可以把明显损坏的archive投影为PASS。

**伦理标记：** 否。工作针对研究档案与可重算性，不涉及个人或敏感数据。若未来收集其他团队archive，应处理凭据、未公开研究内容与作者同意；当前PDF未给出需要升级为伦理警报的证据。

**LLM 使用披露：** 本审稿由隔离运行的 AI 子代理 R3 完成，仅用于内部投稿前质量控制；未与其他评审通信，未读取作者计划、旧评分或其他评审输出。

**评审限制：**

- 仅审阅指定冻结 PDF；按隔离要求未访问论文所列代码、JSON、fixture、clean-room bundle、sha16记录或预注册文件，因此未执行任何证书或重放命令。
- 未联网，未核验外部引文、相关工作优先权、外部archive身份或公开讨论；相关不确定性不应被解释为已验证。
- 已逐页视觉核查25页 PDF；除附录H一处命令缺空格外，未发现影响阅读的图表截断。

### P3

#### 交付与来源状态

- 选定输入：`/Users/liuhanzuo/Downloads/P3-paper (1).pdf`
- 源状态：`pdf_only_exact_tex_overwritten`
- 润色 PDF：`/Users/liuhanzuo/MacLLM-Bench/output/pdf/paper_polish_20260826/P3-polished.pdf`
- 冻结 SHA-256：`c047b9974119a16a69aaf7981a3621e3dfcb0cf47672cabb075f6fad8cdf4b83`
- 总页数：8；主文状态：主文不超过9页。
- 版面核验：pass；构建：pass。
- 旧评分基线：2,4,4；旧中位数：4。

#### 三评量化结果

| 审稿人 | 角色 | Overall | Confidence | 建议 | Soundness | Presentation | Contribution | 当前上限 | 必需修改后预测 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| R1 | 新颖性与定位 | 4 | 3 | 略低于接收线 | 2 | 3 | 2 | 4 | 6 |
| R2 | 技术正确性 | 4 | 4 | 略低于接收线 | 2 | 3 | 2 | 6 | 6 |
| R3 | 实验严谨性 | 4 | 4 | 略低于接收线 | 2 | 3 | 2 | 4 | 6 |

三评中位数为 **4**，均值 4.00，跨度 0，接收侧票数 0/3。

#### 编辑记录

- [结构审计](work/P3/structure_audit.md)
- [语义锁](work/P3/semantic_lock.md)
- [修订日志](work/P3/revision_log.md)
- [待核验事项](work/P3/needs_verification.md)

**修订日志原文：**

> # P3 Revision Log
>
> ## Editorial scope
>
> This is a conservative LaTeX reconstruction and evidence-preserving polish of the supplied PDF. The historical source was unavailable; no later manuscript was treated as a substitute. The rendered PDF is therefore the authority for scientific content.
>
> ## Structural changes
>
> - Rebuilt a compilable ICLR manuscript with the same central progression: estimand, three entry constructions, paired same-target test, set-ordered ridge counter-arm, and carrier-limited conclusion.
> - Consolidated repeated explanations of the cross-sectional/paired sign distinction, the sealed-contract erratum, and the B2 routing confound.
> - Moved provenance and implementation detail to the end matter so that the main argument remains readable.
> - Recreated all ten tables and four displayed equations, and retained both supplied figures as crops from the original PDF.
>
> ## Language and claim changes
>
> - Shortened the abstract, introduction, contribution framing, limitations, and conclusion.
> - Replaced broad or causal phrasing with carrier-specific, design-specific language.
> - Made explicit that negative Kendall tau and negative paired slope `g` encode different comparisons.
> - Kept the ridge result as a sign-only counter-arm; the revision does not transport ridge magnitude across carriers.
> - Described B2 as a reference-conditioned routing stress test rather than decisive path evidence.
>
> ## Preserved scientific content
>
> - All sample sizes, schedules, seeds, thresholds, effect estimates, confidence intervals, p-values, detector/agreement results, and table cells were transcribed from the supplied PDF.
> - The undecided general identification verdict and the synthetic-carrier boundary were preserved.
> - No new experiment, citation, scientific result, or mechanistic interpretation was added.
>
> ## Build and QA
>
> - Built twice with `pdflatex` through `latexmk` and shell escape disabled.
> - Checked the final log for LaTeX errors, unresolved citations/references, fatal errors, and overfull boxes; none were found.
> - Extracted final PDF text and checked all locked key values.
> - Rendered and visually inspected all eight pages for clipping, overlap, missing figures, and unreadable tables.
>
> ## Reconstruction caveat
>
> The revision is semantically faithful to the rendered source, but exact source-level fidelity, original float placement, and original citation keys cannot be guaranteed without the missing historical TeX project.

**待核验事项原文：**

> # P3 Needs Verification
>
> - The exact original LaTeX source and citation keys were unavailable. The manuscript was reconstructed from the supplied rendered PDF; bibliography entries reproduce the metadata printed in that PDF.
> - Figure 1 and Figure 2 are faithful crops/extractions from the supplied PDF, not regenerated from the unavailable plotting source.
> - Equation notation and every table cell were visually checked against the PDF, but byte-level comparison to the missing source is impossible.
> - Artifact paths and SHA prefixes are transcribed from the PDF and were not independently opened or rerun in this editorial pass.
> - No claim is made that the reconstructed source reproduces the original line numbering or float placement.

#### R1（新颖性与定位）完整评议

**论文概述：** 论文考察数据价值究竟只依赖所选集合，还是会随样本进入 WSD 优化日程的位置变化。作者在两个 24 维合成 logistic 载体上比较三种进入构造，并以 16 个 seed 为推断单位。横截面 B2 在两载体上均给出较大负 Kendall τ，但 set-ordered routing stress test 表明该信号可由按参考价值路由产生；因此主要证据来自将同一目标在早、晚位置间移动的配对效应。该效应在注册的 sine 载体上为 g=-0.186、95% CI [-0.258,-0.115]，在线性载体上不确定。一个 set-ordered ridge counter-arm 又显示配对估计器在若干分配上有非负偏差。论文据此仅主张载体依赖的受控测量，而不对一般数据价值作二元结论。

**最强的已核实贡献：** 最强的已验证贡献是：在固定样本身份和 held-out readout 的设计中，横截面 B2 相关可以被 reference-conditioned routing 在 set-ordered substrate 上复现，而同一样本早/晚移动的配对统计给出与横截面符号含义不同的读数；这清楚展示了“样本选择/路由信号”与“样本内位置效应”不能混读。

**维度理由：**

- Soundness：论文严谨地区分了横截面 Kendall 符号与同一样本配对效应，并以 seed 为推断单位，但核心的“路径支持”解释仍受两个未消除的识别问题限制：进入更早同时意味着经历更多梯度更新；而在不同 ridge 载体上观察到的正向估计器偏差不能排除 sine 载体上的负向伪差。因而负向 g 是可信的载体内观测，却尚不足以唯一归因为路径顺序。
- Presentation：问题、估计量、符号方向和主张边界总体清楚，图表无裁切且表格可读；不过 B2、配对臂、ridge counter-arm 与多种四态标签之间的叙事较密，且 τ 的幅度门槛与 g 的判定语义没有被明确分开。
- Contribution：同一样本跨进入位置配对、并揭示 value-conditioned routing 可在 set-ordered 世界制造横截面信号，是有用的测量提醒。但实验仅含两个合成 logistic 载体，且相关工作没有建立与数据顺序、课程学习和序列化数据价值研究的精确边界，当前贡献显得窄且可能是对既有训练顺序效应的重新表述。

**优点：**

- 第 2 页公式 (1)-(3) 明确定义 path value、seed-level Kendall 统计量和 set-valued 比较器，并在第 4 页 §4.2 明确解释负 g 与负横截面 τ 的相反语义。
- 第 3-6 页完整报告 16-seed 区间、同 seed 构造对比、Wilcoxon 结果、Bonferroni 家族控制、正负控制和 allocation sweep，没有把统计显著性直接等同于最小有意义效应。
- 第 4 页 §4.3 和第 6-7 页限制/结论主动披露 sealed binary rule 被 canonical ridge allocation 触发、线性载体未识别、跨载体幅度校正不成立，避免了普遍性过度主张。
- 所有页面版面完整，Figure 1 对 same-sample lane 与 B2 stress-test lane 的区分直观。

**问题与可验证修复：**

##### P3-R1-01 · 主要 · 新颖性、重要性、引文完整性

- 位置：PDF 第 1 页 §1（第 33-48 行）与第 6 页 §6 Related Work（第 303-312 行）
- 观察证据：相关工作用 Datamodels/TRAK 代表 set-indexed attribution，用 Pythia 代表训练轨迹，并引用通用的方差与预注册文献；PDF 内没有讨论课程学习、数据/样本排序、重放或采样日程、序列化/在线数据价值、基于排列的价值定义等与“进入顺序影响边际价值”直接相邻的研究。
- 重要性：论文的算法增量并非一个新模型，而是问题设定与识别设计；因此最接近工作的缺失会直接决定这究竟是新的数据价值概念，还是已知训练顺序效应换用 LOO/Kendall 表达。当前 PDF 无法支撑清晰的新颖性边界。
- 必需修复：扩展相关工作并给出 closest-work matrix，逐项比较对象（集合、排列、进入时间）、是否同一样本配对、是否匹配曝光预算、价值定义和推断单位；相应收窄任何 first/new 暗示。
- 验证标准：修订稿应能为至少三个最接近研究类别各给出一条可证伪的差异陈述，且引言贡献句与相关工作中的差异完全一致。
- 仍需证据：经核验的邻近文献及其对局部差异陈述的支持；本审稿不联网，无法替作者确认具体优先权。
- 预期影响：high；判断置信度：medium。

##### P3-R1-02 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：PDF 第 2-3 页 §3 Entry constructions（第 99-148 行）与第 4 页 §4.2（第 190-204 行）
- 观察证据：目标从 0.15T、0.50T 或 0.85T 开始进入，而所有拟合共享同一个终止预算 T=300。把同一目标从 0.85T 移至 0.15T 不仅改变顺序，也显著增加其参与优化的更新次数；文中没有等曝光/等出现次数的顺序交换对照。
- 重要性：观测到的 g 可以由“更多训练曝光”解释，而不要求更一般的非交换路径效应。若论文身份是 path-ordered value，这一区分会改变概念贡献和可迁移解释。
- 必需修复：增加等曝光但顺序不同的配对对照，或把主张严格改为“entry-time/exposure-dependent marginal value”，并明确不识别纯顺序效应。
- 验证标准：在每个目标的更新次数和总权重严格相等时复算 seed-level g；报告预先固定的区间与对照差异，或全文检索确认所有 path/order 表述均已收窄为进入时间与曝光的联合效应。
- 仍需证据：新的 matched-exposure 正式实验，或不依赖该实验的系统性主张收窄。
- 预期影响：high；判断置信度：high。

##### P3-R1-03 · 主要 · 技术正确性、实验严谨性、可复现性

- 位置：PDF 第 4 页 §4.3（第 206-215 行）、第 6 页 Table 6（第 271-277 行）及第 6 页 §7（第 315-323 行）
- 观察证据：已知 set-ordered ridge 的 identical paired estimator 在 canonical allocation 上给出 g=+0.102、CI [+0.028,+0.176]，并在四个 allocation 上均为正点估计。论文据此说该正偏差不能产生 sine 的负 g，但同时承认不能跨载体传输幅度，且只有一个 set-ordered substrate。
- 重要性：不同 substrate 上偏差的符号不必相同。ridge 上的正偏差不能逻辑排除 sine 数据、优化器或标准化分母诱发的负偏差，因此“PATH-SUPPORTED-neg”仍缺少同床校准。
- 必需修复：构造与 sine carrier、目标集、readout、标准化和 allocation 尽可能相同、但由解析或交换性保证 order-invariant 的负对照；同时把现有 ridge 结果表述为发现潜在估计器偏差，而不是排除负向伪差的证书。
- 验证标准：新的同床 set-ordered 对照在冻结规则下覆盖零，或其最坏负偏差界仍与 sine CI 分离；否则 Table 5 的 PATH-SUPPORTED 标签应降为 carrier-specific observed departure。
- 仍需证据：同床负对照的 seed-level 输出、预先固定的判定规则和最坏偏差界。
- 预期影响：high；判断置信度：high。

##### P3-R1-04 · 主要 · 重要性、实验严谨性、限制与负责任表述

- 位置：PDF 第 2 页 §3（第 101-105 行）、第 5 页 Table 5、以及第 6-7 页 §7-8
- 观察证据：证据仅来自两个 24 维合成 logistic 载体；所谓 sine key 仍使用固定线性标签边界，论文也明确表示它不代表一般非线性模型或语言模型。线性载体未识别，唯一负向读数来自一个人工冲突构造。
- 重要性：在缺少自然数据、非凸模型或至少更多独立载体时，结果更像一个测量床案例，而非能改变 ICLR 读者如何选择数据价值范式的结论。主动披露限制提升可信度，但不自动提升贡献规模。
- 必需修复：至少加入一个自然数据/非凸训练载体和一个预先指定的机制变化，或将稿件定位为短篇方法学测量说明，并减少对一般 data value 范式的标题承诺。
- 验证标准：新载体使用同一样本配对和同床 set-ordered 校准，并在多个 seed 上复现可解释的效应；若不新增实验，标题、摘要和结论均须明确为 synthetic logistic case study。
- 仍需证据：跨载体复现、自然任务结果及其完整 seed-level 不确定性。
- 预期影响：high；判断置信度：high。

##### P3-R1-05 · 次要 · 清晰度、技术正确性

- 位置：PDF 第 2 页 §2（第 83-90 行）与第 4-5 页 §4.2/Table 5
- 观察证据：|τ|=0.10 被定义为 Kendall τ 的最小有意义幅度；之后标准化均值差 g 也以四态/路径支持语言呈现，但 PDF 未给出 g 的独立幅度门槛或说明为何可共享 0.10。
- 重要性：τ 与 g 是不同尺度的统计量。模糊的门槛语义会让读者误把 sine CI 低于 -0.10 当作已预注册的实际显著性结论。
- 必需修复：为 g 单独给出预注册判定规则与科学依据，或只报告其区间和方向而不使用跨统计量的幅度标签。
- 验证标准：所有表格标签可由文中明确、唯一、统计量特定的规则机械复算。
- 仍需证据：g 门槛的冻结记录；若不存在，则需删除幅度性标签。
- 预期影响：medium；判断置信度：medium。

**给作者的问题：**

- 能否给出一个保持每个目标总曝光次数/梯度更新次数相同、只交换相对顺序的配对构造？若该构造下 sine 的负 g 仍在，路径顺序解释会明显更强。
- 为何不同 ridge substrate 上的正向估计器偏差足以排除 sine substrate 上的负向估计器伪差？是否存在同一 sine 数据与同一 readout 上、由解析式保证 set-ordered 的校准臂？
- 与课程学习、样本排序/重放日程、在线或序列数据价值、随机排列 Shapley/LOO 估计等邻近工作相比，本文配对估计量的精确新颖性是什么？
- 第 2 页为 Kendall τ 冻结的 |τ|=0.10 幅度门槛，是否也用于第 4-5 页的标准化配对统计 g？若是，二者采用同一数值阈值的科学依据是什么；若否，PATH-SUPPORTED-neg 的操作定义是什么？
- 匿名补充材料是否包含可从完整 SHA-256 解析的原始 seed-level 值与执行命令，而不仅是第 7-8 页列出的文件名和哈希前缀？

**能提高评分的证据：**

- 等曝光、仅改变相对顺序的正式配对实验仍复现 sine 的负向效应。
- 同一 carrier/readout 上的 set-ordered oracle 对照把负向估计器伪差界定在 sine CI 之外。
- 至少一个自然数据或非凸模型载体复现，并保留 seed-level 配对不确定性。
- 完整的 closest-work 比较证明该识别设计区别于既有课程学习、样本排序和序列数据价值方法。

**会降低评分的证据：**

- 负向 g 在等曝光对照下消失，表明现象主要是训练次数而非顺序。
- 同床 set-ordered 对照产生与 sine 同方向、相近幅度的负向偏差。
- 原始 seed-level 记录无法复算 Table 4-6 或同一统计量在工件中出现不一致版本。
- 更完整的 PDF 内文献核验表明同一样本进入位置配对已是既有方法而论文未说明增量。

**伦理标记：** 否。研究使用合成数据与 CPU logistic 模型，PDF 明确说明无人体参与者或个人信息；未发现需要会议伦理升级处理的问题。

**LLM 使用披露：** 本审稿由一个全新、隔离的 AI 子代理生成，仅用于内部投稿前质量控制；该子代理未与任何其他评审通信，也未接触作者计划、旧评审或目标分数。

**评审限制：**

- 按任务约束仅相对于 PDF 自身的相关工作评价新颖性；未联网检索或核验更广文献，因此优先权判断具有不确定性。
- 仅审阅冻结 PDF，未读取源文件、代码、原始 JSON、补充工件或执行实验，无法独立核验数值与哈希前缀所指内容。

#### R2（技术正确性）完整评议

**论文概述：** 论文研究数据样本的经验边际价值是否随其进入 WSD 优化日程的位置变化。作者在两个 24 维合成逻辑回归载体上比较三种进入构造，以种子级 Kendall τ 描述横截面关系，并用同一目标在 0.15T 与 0.85T 间移动的标准化配对差异 g 作为主要识别量。sine 载体得到 g=-0.186、95% CI [-0.258,-0.115]，linear 载体得到接近零且区间跨零的结果；B2 横截面信号还能在集合有序 ridge 中通过按参考价值路由重现。论文据此主张只有 sine 载体支持负向路径偏离，同时把结论限制为合成、载体依赖的测量。

**最强的已核实贡献：** 在论文所给的 16 个种子上，同一 sine 目标的早/晚进入比较给出方向一致于负向的种子级标准化效应及跨种子区间，而 linear 结果明确保留为不确定；同时，B2 路由实验具体展示了强横截面 τ 可以由参考值条件化的排序构造产生，因而不能单独识别路径效应。

**维度理由：**

- Soundness：论文把种子作为推断单位，并用同一样本的早/晚进入对比避免把 B2 的横截面路由相关性直接误读为路径效应，这一核心设计是合理的。然而，集合有序 ridge 对照在四种分配上仅观察到非负偏差，并不能推出该偏差机制在 sine 载体上也不可能改变符号；摘要和结论中的“不能产生负向 sine 偏离”强于现有对照所支持的结论。另有若干关键实现与稳健性细节只以工件名/哈希指代，无法从论文独立核查。
- Presentation：论文清楚区分了负的横截面 Kendall τ 与负的同样本 g 的相反语义，表格也把阈值判定、显著性和构造差异分开呈现。主要不足是中心统计量被称为“paired slope”但实际上是两个端点差异的种子内标准化均值，原始效应尺度和若干对照协议没有完整展示。
- Contribution：同一样本、同一读出下改变进入位置，并把横截面路由混杂与路径敏感性分开的测量框架有一定方法学价值；但证据只来自两个小型合成逻辑回归载体，其中一个结果不确定，尚不足以支持更广的数据价值结论。

**优点：**

- 把种子而非 40 个目标样本作为推断单位，避免明显的目标级伪重复。
- 明确区分横截面 τ 与同样本 g 的符号含义，撤回了一个由代码错误导致的旧解释，并对不确定的 linear 载体保持克制。
- 同时报告集合有序对照、路由压力测试、已知路径正对照和构造间同种子对比，形成比单一相关系数更完整的测量链。
- 局限部分没有把合成 GLM 结果外推到自然数据、TRAK/datamodels 全流程或语言模型。

**问题与可验证修复：**

##### P3-R2-001 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第4页§4.3（行206-215）、第6页表6、第7页结论（行326-331），以及摘要第21-25行
- 观察证据：ridge 对照的四个分配点估计为 +0.102、+0.192、+0.049、+0.145，其中一个区间跨零；论文同时承认不允许跨载体搬运偏差幅度，却据此断言正向集合有序偏差不能产生 sine 的负向偏离。
- 重要性：只在一个不同数据生成机制的 ridge 基底上观察到正向偏差，不能排除估计器偏差随载体几何或训练动态翻转符号。该推断被用来加强摘要和结论中的中心识别主张，因此不是单纯措辞问题。
- 必需修复：把结论严格收窄为所测 ridge 对照的经验结果；若要保留“不能产生”这一排除性表述，需要给出适用于相关估计器/载体类的符号定理，或增加与 sine 数据几何和优化日程匹配、但真值集合有序的负对照族。
- 验证标准：在预先规定的一组与两个载体匹配的集合有序生成机制上重复完全相同的 g 流程，并检验偏差符号的跨种子区间；或者形式证明在所声明条件下 g 的估计偏差恒非负。只要存在一个合理匹配对照出现稳定负偏差，就必须撤销排除性主张。
- 仍需证据：匹配负对照的逐种子结果、生成机制与冻结分析代码，或完整符号证明。
- 预期影响：high；判断置信度：high。

##### P3-R2-002 · 主要 · 可复现性、技术正确性、清晰度

- 位置：第2-3页§3与表1、第4页§4.2、第5-6页§4.4-5及附录B
- 观察证据：论文给出了维度、样本量、步数和正则系数，但没有完整给出两种合成分布的参数、学习率及 WSD 曲线数值、B2 参考值的计算细节、ridge 基底与四种分配的生成规则、已知路径正对照的构造和 detector 的具体统计规则。附录只列文件名与哈希前缀。
- 重要性：这些缺项决定同一样本效应、ridge 偏差和所谓“DISCRIMINATING”证书，审稿人无法仅凭论文重建中心实验或判断对照是否与主实验匹配。
- 必需修复：在论文或匿名补充材料中给出可执行的完整数据生成伪代码、所有优化超参数、B2/reference/positive-control/negative-control 定义、随机数映射和统计流程，并把每个表格单元映射到可复现命令。
- 验证标准：独立环境从公开说明重新生成 16 个主实验种子和全部对照，核对表2、表5-10的逐种子摘要及哈希；数值容差和软件版本需预先规定。
- 仍需证据：匿名代码、冻结配置、逐种子输出和端到端重现记录。
- 预期影响：high；判断置信度：high。

##### P3-R2-003 · 次要 · 技术正确性、清晰度、实验严谨性

- 位置：第4页公式(4)与§4.2、第5页表5
- 观察证据：g_k 是目标级 Δ 的均值除以同一种子内标准差，只比较 0.15T 和 0.85T 两个端点；论文未报告未标准化均值或分母分布，也没有为 g 预设类似 τ_a 的实际意义阈值。
- 重要性：标准化会让不同种子按各自离散度隐式重权，g 的数值不能直接解释为效用斜率或实际效应量；区间排除零只支持方向，不说明影响大小。
- 必需修复：将其重命名为标准化端点对比，同时报告逐种子原始均值、标准差、稳健标准化敏感性和未标准化总体效应；若主张实际重要性，应预先给出 g 的最小效应界。
- 验证标准：比较原始差、当前 g、稳健尺度标准化以及按种子等权汇总的符号与区间，确认结论不由异常小分母驱动。
- 仍需证据：逐种子 Δ 分布和预先规定的效应量分析。
- 预期影响：medium；判断置信度：high。

##### P3-R2-004 · 次要 · 实验严谨性、清晰度

- 位置：第5页表5及§4.4、第5-6页表7和表9
- 观察证据：sine 的 routing placebo 仅给出 g=-0.176 点值；routing-only τ 和 set-valued agreement 也只给点值，论文明确说明后者缺少冻结区间。
- 重要性：这些结果被用于说明主效应对路由置换稳健、B2 可被集合有序世界重现及比较器“alive”，但没有种子级不确定性时无法判断其稳定程度。
- 必需修复：补充预先规定的置换方案、所有种子结果和与主结果同口径的区间；若无法补充，应把这些证据降为单次诊断而非稳健性结论。
- 验证标准：在固定置换种子集合上重跑并报告 g/τ/τ_AB 的种子级区间和 leave-one-seed-out 范围。
- 仍需证据：逐种子 placebo、routing-only 与 agreement 输出。
- 预期影响：low；判断置信度：high。

**给作者的问题：**

- 在同样本对比中，V(early;z)-V(late;z) 的 leave-one-out 基线是否逐目标完全相同并代数消去？若是，为什么仍将 g 称为边际价值“slope”，而不是两个端点训练效用差的标准化均值？
- 有什么理论或同载体匹配对照能够保证集合有序估计偏差的符号不会随数据几何、目标分布或优化器状态改变？若没有，是否愿意把“不能产生负向 sine 偏离”收窄为“所测 ridge 对照未复现该符号”？
- 能否给出每个种子的未标准化 mean_z Δ(z)、sd_z Δ(z) 和 g，说明结论不是由少数小分母种子放大，并预先定义 g 的最小有意义幅度？
- 路由置换 placebo 的精确定义、置换次数、种子级区间以及 B2 参考分数的计算规则是什么？

**能提高评分的证据：**

- 同几何、同优化设置的集合有序匹配对照或一般性符号证明，能够真正排除负向估计偏差。
- 逐种子原始 Δ、g 分母与全部 placebo/路由诊断的冻结不确定性分析。
- 至少一个自然数据或更真实模型载体上的预注册同样本复现，且保留当前克制的范围声明。
- 匿名工件能够从完整配置端到端重现表2、表5-10。

**会降低评分的证据：**

- 匹配集合有序对照出现稳定负向 g，说明 sine 方向可能由估计偏差产生。
- 未标准化 Δ 或 leave-one-seed-out 分析显示结论由少数小分母/异常种子驱动。
- 独立重现无法得到表5的 sine 效应或发现早/晚运行的非目标因素未保持一致。

**伦理标记：** 否。论文使用合成数据、无人体参与者或个人信息；在当前材料中未发现需要单独伦理升级的问题。

**LLM 使用披露：** 本审稿由一个全新、隔离的 AI 子代理生成，仅用于内部投稿前质量控制；子代理只接收冻结 PDF、指定的 ICLR 2026 量表与输出 schema，未与任何其他评审通信，也未查看作者计划、旧评审或目标分数。

**评审限制：**

- 仅审阅了指定冻结 PDF；按隔离要求未访问代码、原始日志、逐种子工件或论文中列出的哈希对象，因此实现与数值只能依据 PDF 判断。
- 按任务要求未联网，未核验参考文献元数据、引用对局部陈述的支持关系或更广泛的最近工作。

#### R3（实验严谨性）完整评议

**论文概述：** 论文研究训练样本的边际价值是否随进入 WSD 优化日程的位置而变化。作者在两个24维合成 logistic 载体上比较三种进入构造，以每个 seed 的 Kendall τ 描述跨样本关系，并用把同一目标从0.15T移动到0.85T的标准化配对差 g 作为主要路径读数。sine 载体得到 g=-0.186、95% CI [-0.258,-0.115]，linear 载体不确定；同时，B2 的跨截面负相关可由集合有序世界中的按参考值路由复现。集合有序 ridge 反事实臂对同一配对估计器给出非负偏差且规范分配显著触发，作者据此仅作符号层面的保守解释，并将结论限制为载体依赖的合成测量。

**最强的已核实贡献：** 最可信的贡献是证明 B2 跨截面信号本身不能识别路径效应：第4–6页的同一样本移动与集合有序路由压力测试共同显示，参考值条件化的路由可以在集合有序世界中复现强负 Kendall τ，因此必须把样本身份固定后再检验进入位置。

**维度理由：**

- Soundness：论文正确地区分了跨样本 Kendall 符号与同一样本早晚移动的配对符号，并以 seed 为推断单位；但承载路径敏感性结论的同一估计器在已知集合有序的 ridge 反事实臂上于四种分配中的三种显著触发。当前仅凭另一载体上的偏差为正，就排除 sine 载体上负向伪影，缺少同载体控制或可运输的符号界，因此核心识别仍有实质缺口。
- Presentation：问题、估计量、符号方向和结论边界总体清楚，图表可读，局限也较坦率。主要可读性问题是第5–6页表8、表9先于表6、表7出现，且配对主结果没有展示16个 seed 的完整分布与原始差值尺度，使关键判断不如跨截面结果易核查。
- Contribution：同一样本配对、显式区分路由选择信号与路径效应、并主动运行集合有序反事实臂，是有用的实验诊断框架；然而证据仅来自两个合成 logistic 载体，线性载体未识别，反事实臂又暴露估计偏差，当前知识增量较窄且尚不足以支持稳定的一般方法结论。

**优点：**

- 第2–4页明确给出 τ 与 g 的方向解释，并反复避免把负的跨截面 τ 误读为负的同一样本路径效应。
- 第3–5页以16个 seed 为推断单位，报告 Student-t 区间、Wilcoxon 检验、六个相关对比的 Bonferroni 家族控制以及预设的 τa=0.10 实质效应门槛。
- 第4–6页没有隐藏负控制失败：规范 ridge 分配触发封存的双侧规则，作者撤回了符号不敏感的后果条款，并拒绝做未经验证的跨载体幅度校正。
- 第6–8页清楚限制到两个合成 logistic 载体，并声明该研究不能普遍决定数据价值是路径有序还是集合有序。

**问题与可验证修复：**

##### P3-R3-01 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第4–6页，§4.2–4.3，表5–7；摘要第21–25行与结论第326–331行
- 观察证据：sine 主配对读数为 g=-0.186、CI [-0.258,-0.115]；但在按构造应当次序不变的 ridge 上，规范分配为 g=+0.102、CI [+0.028,+0.176]，另两种分配也显著为正，只有 Xs[80:120] 覆盖零。论文承认规范分配按封存双侧规则属于 fire，却据跨载体的正偏差方向断言其不能产生 sine 的负偏差。
- 重要性：负控制证明估计流程在真零路径效应下并不居中。偏差符号可能依赖载体几何、目标选择、优化收敛和分配构造；在没有同载体集合有序控制或解析符号界时，ridge 上的正号不能排除 sine 上的负向伪影。这直接决定主路径识别是否成立。
- 必需修复：在尽可能保持 sine 数据几何、目标分布、样本数、日程和优化器不变的前提下构造可证明集合有序的匹配负控制，或给出并验证 g 偏差的解析表达与跨载体可运输符号界。若不能做到，应把 PATH-SUPPORTED-neg 降为与估计偏差未分离的探索性负向读数。
- 验证标准：预先冻结匹配负控制和主载体的 seed、分配及判定规则；确认负控制 g 的区间覆盖零，或由经数值验证的上/下界证明任何集合有序伪影都不可能达到负的 sine 区间；随后在独立 seeds 上复现 sine 的负向效应并报告偏差界后的结论。
- 仍需证据：匹配载体负控制的逐 seed 原始 Δ 与 g、优化收敛诊断、解析或仿真偏差分解，以及独立冻结复验。
- 预期影响：high；判断置信度：high。

##### P3-R3-02 · 主要 · 实验严谨性、清晰度、可复现性

- 位置：第4页，§4.2式(4)及表5；第2页§2关于 τa 的定义
- 观察证据：主量 gk 是40个目标早晚差的均值除以同一组差值的样本标准差，最终仅给出16个 seed 上的 g 均值、t 区间和 Wilcoxon p。论文未给出原始 Δ 的量纲、每个 seed 的分母或 g 分布，也未为 g 预设类似 τa=0.10 的实际意义门槛。
- 重要性：标准化比值会把原始影响幅度与分母稳定性混合；区间排除零只能说明该定义下的方向，而不能说明实际影响大小。若少数 seed 的分母小或目标内分布重尾，n=16 的均值区间可能不稳，且 PATH-SUPPORTED 标签可能把统计非零误当作决策相关效应。
- 必需修复：报告16个 seed 的原始 mean Δ、std Δ、g、目标级差值分布与稳健敏感性；预先定义 g 或原始 Δ 的最小有意义幅度，并说明多重分析家族。
- 验证标准：在冻结代码上重算未标准化、稳健尺度和删除单 seed 的区间；要求方向、实际幅度门槛和结论在预先指定的敏感性分析中一致，并公开逐 seed 表。
- 仍需证据：逐 seed/逐目标配对输出、分母诊断、稳健区间以及事前的实际意义阈值。
- 预期影响：high；判断置信度：medium。

##### P3-R3-03 · 主要 · 可复现性、实验严谨性、引文完整性

- 位置：第7–8页，附录A–B
- 观察证据：附录A称精确历史 LaTeX 与绘图源不可用，定量图由冻结 PDF 忠实抽取；附录B只列工件名及8位 SHA 前缀，并明确本次编辑未独立执行。PDF 本身没有逐 seed 数据、完整配置或可执行重放说明。
- 重要性：主结论依赖多个历史轮次工件、标签修订和负控制分配。短哈希与转录表不足以排除版本混用、数值抄录错误或分析脚本与表格不一致，也不允许独立审稿人重建核心结果。
- 必需修复：在匿名补充材料中冻结原始逐 seed/逐目标输出、数据生成器、配置、分析脚本、环境、完整 SHA-256 和一条端到端重放命令；由干净环境重新生成表2–10和图2。
- 验证标准：独立环境从公开的冻结工件重放后，所有表格单元与图中点/区间逐值一致，完整哈希匹配，并生成包含 seed 数、日程位置、优化终止状态和输出路径的审计日志。
- 仍需证据：匿名可下载的完整工件包及独立重放记录。
- 预期影响：high；判断置信度：high。

##### P3-R3-04 · 次要 · 清晰度、实验严谨性

- 位置：第5–6页，表8、表9、表6、表7、表10的出现顺序
- 观察证据：正文先展示表8和表9，下一页才出现表6、表7和表10；主配对结果只以汇总表出现，没有与图2同等级的逐 seed 可视化。
- 重要性：错序增加核查成本，并使最关键的配对异质性、离群 seed 和控制偏差不如次要跨截面结果透明。
- 必需修复：按首次引用重排表格并增加16个 seed 的配对点图，明确连接同 seed 的 sine、linear 与匹配负控制读数。
- 验证标准：逐页检查所有表格按编号和首次引用顺序出现，且图中点可由公开逐 seed 表一一对应。
- 仍需证据：逐 seed 汇总表和生成图脚本。
- 预期影响：low；判断置信度：high。

**给作者的问题：**

- 同一配对估计器为何会在严格集合有序的 ridge 上产生系统性正 g？能否给出该偏差关于载体、目标选择、优化精度和分配位置的解析分解，并证明其符号在 sine 构造上不能翻转？
- 第4页式(4)把每个 seed 的平均早晚差除以40个目标内的标准差。16个 seed 各自的原始 mean Δ、std Δ 和 g 是多少？结论对接近零的分母、稳健标准化或不标准化效应是否稳定？
- 为何 τ 有预设的最小有意义幅度 τa=0.10，而主配对量 g 没有相应的实际意义门槛？PATH-SUPPORTED-neg 的判定在冻结方案中是否只要求区间排除零？
- 547978c5、6b437ba5、397dd466 等工件是否会随匿名补充材料提供完整文件与全长 SHA-256，以便从数据生成到表5–7端到端重放？
- sine、linear、路由置换及四个 ridge 分配在冻结前分别被指定为主检验、校准或探索分析吗？请给出覆盖这些选择的完整错误率家族。

**能提高评分的证据：**

- 在保持 sine 载体关键几何和训练流程的匹配集合有序负控制上，配对估计量居中于零，或存在经验证的偏差界严格排除负向伪影。
- 在事前冻结的新 seeds 上复现 sine 的原始及标准化早晚差，并跨过预设的实际意义门槛，且对稳健尺度、删除单 seed 和多重性处理稳定。
- 公开完整匿名工件并由干净环境端到端重放所有主表与控制表，逐值匹配。
- 加入至少一个自然数据或更接近目标应用的载体，并在其中运行同样的配对与匹配负控制协议。

**会降低评分的证据：**

- 匹配集合有序控制出现负 g，或其负向区间与 sine 主结果重叠。
- 逐 seed 原始差显示结论由小分母、单一 seed、未记录的选择或后验分析驱动。
- 重放工件不能复现表5–7，或完整哈希对应的配置与论文所述日程/目标不一致。

**伦理标记：** 否。研究使用合成数据、无人体参与者或个人信息；冻结 PDF 中未见需要单独伦理升级的内容。

**LLM 使用披露：** 本审稿由全新、隔离的 AI 子代理 R3 生成，仅用于内部投稿前质量控制；该子代理只读取指定的冻结 PDF、审稿协议、量表与 JSON schema，未与任何其他评审通信，也未接触作者计划、历史评审、目标分数或版本历史。

**评审限制：**

- 按隔离要求仅审阅了冻结 PDF，未读取论文源文件、附录中点名的 JSON/代码/预注册工件，因此无法独立重放或验证其内部完整哈希。
- 按任务要求未联网，参考文献的存在性、元数据与对局部陈述的支持未做外部核验；相关意见仅基于 PDF 中呈现的定位。
- PDF 共8页，文本抽取完整且逐页视觉核查未发现影响内容读取的解析故障。

### P5

#### 交付与来源状态

- 选定输入：`/Users/liuhanzuo/Downloads/P5-paper.pdf`
- 源状态：`exact_latex`
- 润色 PDF：`/Users/liuhanzuo/MacLLM-Bench/output/pdf/paper_polish_20260826/P5-polished.pdf`
- 冻结 SHA-256：`61cb32cfc2143449d9d418c67ae71814cbca1a9fda188facac641abe0285f565`
- 总页数：10；主文状态：主文不超过9页。
- 版面核验：pass；构建：pass。
- 旧评分基线：4,4,4；旧中位数：4。

#### 三评量化结果

| 审稿人 | 角色 | Overall | Confidence | 建议 | Soundness | Presentation | Contribution | 当前上限 | 必需修改后预测 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| R1 | 新颖性与定位 | 4 | 4 | 略低于接收线 | 3 | 4 | 2 | 4 | 6 |
| R2 | 技术正确性 | 4 | 4 | 略低于接收线 | 3 | 3 | 2 | 4 | 6 |
| R3 | 实验严谨性 | 4 | 4 | 略低于接收线 | 3 | 3 | 2 | 4 | 6 |

三评中位数为 **4**，均值 4.00，跨度 0，接收侧票数 0/3。

#### 编辑记录

- [结构审计](work/P5/structure_audit.md)
- [语义锁](work/P5/semantic_lock.md)
- [修订日志](work/P5/revision_log.md)
- [待核验事项](work/P5/needs_verification.md)

**修订日志原文：**

> # P5 Revision Log
>
> ## Editorial scope
>
> Evidence-preserving polish of the exact manuscript source corresponding to the supplied PDF. The intervention, evaluation protocol, frozen gate, numerical results, figures, tables, equations, citations, labels, and artifact identifiers were locked before editing.
>
> ## Structural changes
>
> - Tightened the abstract and introduction around the single causal sequence: endpoint shock, alarm, frozen settlement, and checkpoint rollback.
> - Removed forced page breaks that fragmented the narrative.
> - Moved the separate Route-3 diagnostic from the main argument to the appendix.
> - Reduced repeated treatments of the same result across Results, Discussion, Limitations, and Conclusion.
>
> ## Language and claim changes
>
> - Defined the E1 synthetic endpoint-shock setting in plain language before relying on the internal label.
> - Kept the core positive result narrow: the gate detects the registered shock and improves over deployment of the last checkpoint.
> - Repeatedly but compactly preserved the negative boundary: the gate is outcome-identical to best-validation selection and early stopping on all 64 shocked hold-out records.
> - Distinguished the observed 0/64 healthy alarms in a structurally monotone synthetic carrier from realistic deployment false-alarm calibration.
>
> ## Preserved scientific content
>
> - The 12-epoch schedule, 150/600 intervention size, epochs 9--11, deterministic label map, threshold 0.1354038956, FIT/hold-out split, 64/64 and 0/64 counts, confidence bounds, paired effects, and epoch-8 selector identity are unchanged.
> - No new experiment, citation, result, or deployment claim was added.
>
> ## Build and QA
>
> - Built twice with `pdflatex` through `latexmk` and shell escape disabled.
> - Citation-key, label, and displayed-equation sets match the untouched source exactly.
> - The final log contains no LaTeX errors, unresolved citations/references, fatal errors, or overfull boxes.
> - Extracted the final text and rendered all ten pages; no clipping, overlap, or missing figure/table was observed.

**待核验事项原文：**

> # P5 Needs Verification
>
> - The experiment remains a synthetic carrier. False-alarm performance under naturally non-monotone healthy training trajectories was not measured and must not be inferred from the reported 0/64 result.
> - The gate has no demonstrated outcome advantage over best-validation selection or early stopping in the registered shocked hold-out: all three select epoch 8 on all 64 records.
> - The Route-3 appendix is a distinct diagnostic and should not be presented as confirmatory evidence for the E1 claim.
> - Artifact paths and recorded hashes were preserved but the underlying runs were not rerun during this editorial pass.
> - Bibliographic metadata and external artifact availability were not independently verified; all citation keys are unchanged from the exact source.

#### R1（新颖性与定位）完整评议

**论文概述：** 论文在一个合成五分类 transformer 任务上构造最后四分之一训练期才激活的标签翻转冲击。last-margin m 定义为最终 calibration loss 与历史最小值之差；阈值由 4 个独立 FIT seed 的 healthy/shocked 轨迹冻结，随后在每臂 64 个 hold-out seed 上结算。所有 shocked run 被检测、所有 healthy run 未报警，gate 相对 last 将 outer loss 平均降低 0.3395。然而所有 shocked 轨迹都在冲击前一刻达到最小值，gate、best-val 和 early-stop 在 64/64 run 上返回完全相同的 checkpoint；所有 healthy 轨迹又严格单调下降，使 m 恒为零。因此论文支持的是一个受控 shock 下的 endpoint 污染报警与相对 last 回滚，不支持优于既有 selector 或现实 false-alarm 性能。

**最强的已核实贡献：** 最强的已验证贡献是一个时间上真正分离 clean prefix 与 late label shock 的冻结结算：第 2 页显示 64/64 shocked 轨迹的 argmin 均在 epoch 8、shock 于 epoch 9 开始，第 4-5 页用成对 outer-loss 结果量化 damaged last 与 pre-shock checkpoint 的差异，并明确证明 gate 与 best-val/early-stop 完全同值。

**维度理由：**

- Soundness：对本文实际限定的合成 E1 population，时间干预、FIT/hold-out 分离、64 个配对 seed、精确二项界和记录级 checkpoint 相等性均支持所报告事实。主要缺口不是内部算术错误，而是载体把 healthy margin 结构性固定为零、shock 又使 clean argmin 恰好位于冲击前，因此完美检测和回滚是高度设计化的，不能支持实际部署检测。作者已诚实收窄主张。
- Presentation：这是四篇中最清楚的一篇：Figure 1 直接显示时序干预，Tables 2-6 将观测、边界和下一步证据分开，摘要和结论一致，图表没有视觉缺陷。
- Contribution：时间控制的 endpoint-shock protocol 和对 detection-versus-selection 的诚实分解有方法学价值；但 gate 只在 {last,best-val} 中选择，且 64/64 shocked 情况下与 best-val/early-stop 完全相同，健康假警报又被设计排除。当前没有新 checkpoint 能力，也没有现实 detector operating point，故贡献窄且偏增量。

**优点：**

- 第 2-3 页将干预对象比例（25% examples）与干预时间比例（最后 25% epochs）明确区分，避免把静态噪声误称为时间冲击。
- FIT seed 与 64 个 hold-out seed 分离；第 5 页 Table 4 给出 64/64 与 0/64 对应的保守单侧 Clopper-Pearson 界，而不是只报 1.0/0.0 点估计。
- 第 4 页 Table 3 和 §5.1 明确报告 gate-best-val、gate-early-stop 在 64/64 记录上的精确相等，没有把相等包装成显著性优势。
- 第 5-7 页主动把 m 恒为零识别为设计排除，并把 E2（使 best-val 失败）与 E6（自然非单调 healthy carrier）写成决策性下一步证据。
- 全稿叙述紧凑、图表可读、主张边界在摘要、结果、限制和结论中一致。

**问题与可验证修复：**

##### P5-R1-01 · 主要 · 新颖性、重要性

- 位置：PDF 第 1 页贡献列表（第 42-47 行）、第 3 页 §3.2（第 121-126 行）、第 4 页 Tables 2-3 与第 5 页 §5.5.1
- 观察证据：gate(m,τ) 只能返回 last 或 best-validation；在全部 64 个 shocked hold-out run 上，gate、best-val 与 early-stop 都返回 epoch 8，outer loss 和 accuracy 逐记录完全相同。
- 重要性：当前方法没有发现既有 selector 找不到的 checkpoint，也没有提高其质量。相对一个有意选择的弱部署默认 last 的 0.3395 改善，不能单独构成新的 checkpoint policy；剩余新意只是一个事后报警位。
- 必需修复：执行能使 gate 与 best-val/early-stop 分离的冻结 E2，或把论文重定位为检测协议/负结果短文，并量化报警本身在已有 best-val 管线中的独立决策价值。
- 验证标准：预先固定的 E2 上至少存在足够多的可分 run，且 gate 相对所有已注册 selector 的配对结果和错误代价被完整报告；若仍完全同值，删除 checkpoint-policy 新颖性暗示。
- 仍需证据：新的可分离 shock 实验、逐 run selector identity 和配对 outer 指标。
- 预期影响：high；判断置信度：high。

##### P5-R1-02 · 主要 · 实验严谨性、重要性、限制与负责任表述

- 位置：PDF 第 5 页 §5.3-5.4、Figure 3/Table 5，以及第 6-7 页 §6-7
- 观察证据：704/704 healthy 相邻转移均严格下降，64/64 healthy 的 argmin 在最后 epoch，因此 m 在 healthy population 中恒为 0；shock margins 与 frozen τ 完全分离。
- 重要性：0/64 FAR 不是对检测器抗正常训练波动的检验，而是载体结构的直接结果。没有非零 healthy margin 分布，就不能评估实际部署最关键的阈值权衡。
- 必需修复：在具有自然随机非单调性的真实或至少更具挑战性的 healthy carrier 上重新冻结阈值，并报告事先规定的误报约束、召回界及阈值敏感性。
- 验证标准：healthy hold-out 中存在预先可接受的非退化 m 分布，阈值仍在未用于拟合的 run 上满足误报上界，同时对多个 shock 条件保留召回。
- 仍需证据：E6 或等价的现实 healthy 轨迹、独立 FIT/settlement split、完整 margin 分布。
- 预期影响：high；判断置信度：high。

##### P5-R1-03 · 主要 · 实验严谨性、可复现性、重要性

- 位置：PDF 第 3 页 §3.3/Table 1、第 6-7 页 §7 Limitations 1 与 4
- 观察证据：τ=0.1354 由每臂仅 4 个 FIT seed 的 max-healthy 与 min-shocked 中点得到，且 FIT 与 settlement 使用同一个合成机制；只测试 25% examples 在最后三 epoch 被确定性循环翻标的一个强 shock。
- 重要性：完美分离可能高度依赖这一单一效应大小和时间位置。阈值对正常噪声、较弱/较早/不同类型 shock 的运输性完全未知，当前结果更接近机制自检而非 detector validation。
- 必需修复：冻结一个覆盖 shock 强度、比例、开始时间和至少一种非标签翻转事件的测试网格，并增加 FIT seed 或采用不依赖极值中点的预定义校准准则。
- 验证标准：所有网格条件在盲 hold-out 上报告置信界与失败条件；阈值不因删除任一 FIT seed 或轻微机制变化而跨越主要性能结论。
- 仍需证据：多条件外推实验、FIT leave-one-out 阈值稳定性和失败案例。
- 预期影响：high；判断置信度：high。

##### P5-R1-04 · 主要 · 新颖性、引文完整性、重要性

- 位置：PDF 第 1-2 页 §2 Related Work（第 50-82 行）
- 观察证据：相关工作主要覆盖 SWA、EMA、model soups 和一篇 robustness 引文；PDF 没有定位训练曲线异常/变化点检测、概念漂移、早停与 checkpoint selection、label-noise/poisoning detection 或训练监控文献。
- 重要性：论文的主要产出是一个标量报警而非新 checkpoint，因此其新颖性必须相对于检测与监控文献建立，仅与模型平均方法区分不足以证明贡献独特。
- 必需修复：扩充 closest-work 定位，明确 last-margin 相对既有 loss-rise、change-point、patience/early-stop 和数据污染报警的输入、时机、监督信息和新增证据。
- 验证标准：引言中的每条新颖性主张都能在相关工作中找到具体最近邻与逐项差异；若 last-margin 是已有启发式，应将贡献改为时间控制的审计协议而非新 detector。
- 仍需证据：经核验的最近邻文献及局部支持；本审稿未联网确认具体优先权。
- 预期影响：high；判断置信度：medium。

##### P5-R1-05 · 次要 · 清晰度、限制与负责任表述

- 位置：PDF 标题、摘要与第 6 页 Limitation 1
- 观察证据：标题称“true temporal endpoint data-quality shock”，但干预实际发生在训练最后三个 epoch 内，不是训练结束后的外部 endpoint 事件；正文限制段对此作了说明。
- 重要性：尽管“temporal”相对于静态污染是准确的，“endpoint shock”容易让读者误解为部署后或训练完成后事件。
- 必需修复：在标题或摘要首次出现处改为 late-training endpoint corruption/shock，并用一句话明确它发生在训练内。
- 验证标准：不阅读方法细节的读者也不会把事件理解为训练后事故。
- 仍需证据：无需新实验，仅需术语修订。
- 预期影响：low；判断置信度：medium。

**给作者的问题：**

- 如果部署管线已经保存并部署 best-val，last-margin alarm 除了给出一个事后 flag 之外带来什么新的决策能力？该 flag 的独立成本/价值如何衡量？
- 能否在冻结规则下运行 E2，使 clean argmin 被 shock 覆盖或位移，从而让 gate 与 best-val 的行为真正可分？
- 能否用自然非单调的 healthy 训练轨迹校准阈值并报告 ROC/PR 或预先指定 operating point，而不是 m 恒为零的设计 population？
- 阈值仅由每种模式 4 个 FIT seed 的 max/min 中点确定；对 shock 比例、强度、开始时间或普通优化噪声轻微变化时，阈值稳定性如何？
- 与训练轨迹异常检测、change-point/drift detection、early stopping、checkpoint monitoring 和 label-noise/poisoning detection 文献相比，本文“last margin”报警的精确增量是什么？

**能提高评分的证据：**

- E2 在冻结规则下使 gate 与 best-val/early-stop 可分，并显示独立的 checkpoint 或决策价值。
- E6 在自然非单调 healthy 轨迹上给出非退化误报分布，且独立 hold-out 仍满足预先规定的 FAR/recall 界。
- 多种 shock 时机、强度和类型下的阈值运输结果，而非单一完全可分条件。
- 完整 closest-work 分析证明时间控制审计协议或报警信号相对既有监控方法的明确增量。

**会降低评分的证据：**

- 在非单调 healthy carrier 上出现高误报，或阈值对删除一个 FIT seed 极不稳定。
- 在更弱或稍早 shock 下 gate 与 last 均失败且没有可解释的报警边界。
- 相关工作核验显示 last-minus-best-loss 报警与回滚规则已有直接等价先例。
- 原始记录无法复现 64/64 selector identity 或 Table 3 的配对差异。

**伦理标记：** 否。研究使用程序生成的合成序列和确定性标签翻转，不涉及个人数据或人体参与者；未发现需升级的伦理问题。

**LLM 使用披露：** 本审稿由一个全新、隔离的 AI 子代理生成，仅用于内部投稿前质量控制；该子代理未与任何其他评审通信，也未接触作者计划、旧评审或目标分数。

**评审限制：**

- 按任务约束仅使用 PDF 内相关工作，不联网检索；因此具体优先权与遗漏文献只能作为待核验的新颖性风险。
- 未读取匿名 workspace、代码或 128 条原始 hold-out 记录，无法独立验证执行与逐记录相等性，只能评价 PDF 的内部证据。

#### R2（技术正确性）完整评议

**论文概述：** 论文在一个小型五分类 Transformer 合成任务中构造最终四分之一训练轮次才发生的标签翻转。告警量 m 是最终校准损失减去历次最小校准损失；阈值由健康/冲击各四个 FIT 种子的间隔中点确定。64 个冲击 hold-out 运行全部越过阈值，64 个健康运行均未触发；相对部署 last，门控把平均外层损失降低 0.3395。但所有冲击运行的最优验证点都恰在冲击前一轮，门控、best-val 和 early-stop 因而在 64/64 运行上完全相同。论文将结果限制为检测加相对 last 的回滚，而非优于已有检查点选择器。

**最强的已核实贡献：** 在所给合成协议内，论文清楚验证了时间顺序：冲击前存在干净候选、冲击只在最后三轮生效、最终检查点被损坏；FIT 冻结阈值在独立的 128 个 hold-out 运行上得到 64/64 检出和 0/64 健康误报，并用逐运行配对外层损失量化了相对 last 的回滚收益。

**维度理由：**

- Soundness：对论文实际声明的窄范围而言，时间冲击在第9-11轮才激活、FIT/hold-out 种子分离、逐运行配对外层损失和精确二项界等做法基本正确，且论文没有把 64/64 检出夸大为一般部署结论。主要问题不是结果算错，而是阈值用同一合成冲击机制的极少 FIT 种子校准，健康轨迹又按设计令 m 恒为零，因此中心 operating point 几乎不包含真实告警难度。
- Presentation：因果时间顺序、门控定义、正对照和结论边界都较清楚，三幅图展示了完整运行分布。基线选择器和 rollback 的定义不完整，且多个页面可见红/绿超链接边框，降低了成稿质量。
- Contribution：论文诚实证明了一个末端标签翻转会抬高最终校准损失，last-margin 能发出告警并回滚到已存在的最佳验证检查点；但门控在全部观测运行上与 best-val/early-stop 完全等价，健康臂排除了非单调性，因而新增科学知识和部署价值都较窄。

**优点：**

- 冲击不是从训练开始就存在，而是有明确干净前缀，能够验证“先有可用检查点、后发生末端损坏”的时间顺序。
- 阈值校准种子与结算种子分离，并为召回率和 FAR 给出保守方向的一侧 Clopper-Pearson 界。
- 完整披露 gate、best-val 和 early-stop 在全部冲击运行上完全相同，没有把告警包装为新的优化器或更优选择器。
- 明确把 m恒为0 归因于健康轨迹严格单调，并把真实非单调健康数据列为未测量边界。

**问题与可验证修复：**

##### P5-R2-001 · 主要 · 重要性、实验严谨性、限制与负责任表述

- 位置：第1页摘要与§1、第4页表2-3及§5.1-5.2、第5页§5.5.1、第7页表6与结论
- 观察证据：所有 64 个冲击运行的校准最小点都在 epoch 8，gate、best-val 和 early-stop 的外层损失与准确率逐记录完全相同；健康运行中 last 也是最优点。门控的动作集合本来就只有 {last,best-val}。
- 重要性：相对 last 的 0.3395 改善等价于在一个已知最佳验证点仍可用的构造中启用 best-val，不能证明新检查点选择能力或告警对已有选择器的增量部署价值。作为 ICLR 贡献，当前实验主要确认了一个预期的合成失败模式。
- 必需修复：增加预注册的非共延构造：使冲击覆盖、提前或扰动验证 argmin，或者让告警在尚未观察完整轨迹时做真正前瞻决策；以 best-val、early-stop 和成本匹配的回滚策略为主要对照，而非只对 last。
- 验证标准：在冻结阈值下，报告 gate 与 best-val/early-stop 选择不同检查点的比例，并在独立种子上用配对外层指标检验增量收益；若始终共延，应把贡献严格限定为冲击标签而非检查点政策。
- 仍需证据：不同冲击时刻/强度/类型的独立 hold-out 运行和逐运行选择器身份。
- 预期影响：high；判断置信度：high。

##### P5-R2-002 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第3页§3.3与表1、第4-6页§5.3-5.4、图3和表4-5、第7页局限2与4
- 观察证据：阈值取自同一标签翻转机制下仅 4 个健康和 4 个冲击 FIT 种子的 max/min 中点；所有健康 hold-out 轨迹 704/704 相邻转移严格下降，因此 m 恒为0，而所有冲击 margin 与其完全分离。
- 重要性：这种设计把最困难的误报来源排除掉，并用与测试冲击同分布的正例校准阈值。64/64 与 0/64 主要反映构造分离度，不能支持阈值稳定性、未知冲击检测或实际部署 FAR。
- 必需修复：加入自然非单调健康轨迹、不同于 FIT 的冲击族及阈值敏感性分析；采用嵌套或更充分的校准，预先规定阈值选择和跨机制评估。
- 验证标准：在完全未用于阈值选择的健康/冲击家族上报告 ROC/PR、固定 FAR 下召回、校准集 leave-one-out 阈值范围以及按冲击类型分层的置信区间。
- 仍需证据：现实或至少多机制、非单调的校准与独立测试轨迹。
- 预期影响：high；判断置信度：high。

##### P5-R2-003 · 主要 · 可复现性、实验严谨性、清晰度

- 位置：第3页表1与§3.3-4.1、第4页表2、第6页讨论、第9页附录A
- 观察证据：论文只列出 last、best-val、early-stop、SWA、EMA、soup、rollback、margin-gate 的名称，没有定义 early-stop 规则、SWA/EMA 窗口或衰减、soup 成员/权重、rollback 触发条件。表2 中 rollback 与 gate 的结果不同，但正文未解释算法差异。
- 重要性：这些基线被用于论证替代政策也能吸收末端冲击；缺乏定义使公平性和数值无法独立重建，也无法判断 gate 的比较对象是否成本匹配。
- 必需修复：为每个选择器给出公式、可观察信息集、超参数、检查点存储成本及确定性实现，并解释 rollback 与 gate 的差异。
- 验证标准：从逐 epoch 检查点和冻结配置独立重放所有选择器，逐记录复现表2，并核对 gate/best-val/early-stop 的 64/64 身份。
- 仍需证据：选择器配置、逐运行检查点索引和重放脚本。
- 预期影响：medium；判断置信度：high。

##### P5-R2-004 · 次要 · 实验严谨性、可复现性

- 位置：第3页§4.2（行157-161）
- 观察证据：预注册是在第一个 hold-out 种子已经完成时冻结，而论文只说明没有用 hold-out 轨迹调整阈值，没有说明该结果是否已被查看或是否影响通过条件和分析。
- 重要性：若首个 hold-out 结果已被观察，预注册对完全前瞻性的保护会弱于通常理解，尤其本研究的主要通过条件与单一机制高度相关。
- 必需修复：提供带时间戳的预注册时间线，明确首个 hold-out 结果是否可见、此前已冻结哪些项目，并把任何事后决定标注为探索性。
- 验证标准：核对预注册哈希/时间戳、seed 1 完成时间和访问日志；确认阈值、终点、CI 方向及通过条件均早于结果查看。
- 仍需证据：不可变预注册记录和运行时间线。
- 预期影响：low；判断置信度：medium。

##### P5-R2-005 · 次要 · 清晰度

- 位置：PDF第1、3-7、9-10页的内部引用与文献链接
- 观察证据：渲染后的 PDF 中，多处表/图引用带红色矩形边框，文献引用带绿色矩形边框。
- 重要性：可见链接边框持续打断阅读，并使提交稿显得未完成最终排版检查。
- 必需修复：按 ICLR 模板设置隐藏或统一的链接样式，并重新渲染全稿检查。
- 验证标准：逐页视觉检查最终 PDF，确认所有链接仍可点击但不再显示彩色边框，也没有交叉引用错误。
- 仍需证据：修复后的最终 PDF。
- 预期影响：low；判断置信度：high。

**给作者的问题：**

- 四个 FIT 冲击种子和四个健康种子是否足以稳定确定 τ？若逐个删除 FIT 种子或改变冲击强度/时间，阈值和 hold-out 决策如何变化？
- 第一个 hold-out 种子完成后才冻结预注册时，作者是否看过该种子的轨迹或结果？哪些设计、阈值和通过条件在它完成之前已经不可变地登记？
- Table 2 中 rollback 与 margin-gate 的算法差异是什么，为什么冲击外层损失分别为 0.00106 和 0.00092？SWA、EMA、soup、early-stop 的精确定义与超参数是什么？
- 在校准损失自然非单调、冲击类型与 FIT 不同且 best-val 不再等于理想回滚点时，告警能否仍保持低 FAR 并提供超越 best-val 的决策价值？

**能提高评分的证据：**

- 在自然非单调健康轨迹和未用于 FIT 的多种末端冲击上仍能保持预先规定的 FAR/召回。
- 一个让 gate 与 best-val/early-stop 非共延的预注册实验，显示成本匹配的增量外层收益。
- 阈值对 FIT 种子、冲击强度和时间的稳定性分析，以及透明的预注册时间线。
- 完整定义并可逐记录重放所有基线选择器。

**会降低评分的证据：**

- 加入健康非单调性后 FAR 明显上升，或跨冲击机制阈值失效。
- 揭示首个 hold-out 结果曾影响阈值、终点或通过条件。
- 成本匹配比较显示 gate 从不优于 best-val/early-stop，且告警标签没有独立操作价值。

**伦理标记：** 否。实验为程序生成的合成序列分类任务，不含人体参与者或个人数据；当前材料未显示额外伦理风险。

**LLM 使用披露：** 本审稿由一个全新、隔离的 AI 子代理生成，仅用于内部投稿前质量控制；子代理只接收冻结 PDF、指定的 ICLR 2026 量表与输出 schema，未与任何其他评审通信，也未查看作者计划、旧评审或目标分数。

**评审限制：**

- 仅审阅了指定冻结 PDF；未访问文中所称匿名 workspace、原始 128 条记录、预注册文件或代码，因此不能独立核验运行时间线和逐记录相等性。
- 按任务要求未联网，未核验引用元数据或与现有末端漂移/检查点告警工作的完整新颖性边界。

#### R3（实验严谨性）完整评议

**论文概述：** 论文在一个小型五分类合成序列任务中，把150/600个训练样本的标签仅在最后3个 epoch 翻转，构造真正晚发的端点质量冲击。last-margin m 是最终校准损失减去轨迹最小值；阈值由健康/冲击各4个 FIT seed 的间隙中点冻结，超过阈值则报警并部署 best-validation 而非 last。在64个健康和64个冲击 hold-out seed 上，冲击召回64/64、健康误报0/64；gate 相对 last 的平均外层损失差为-0.3395。所有冲击轨迹都在冲击前的 epoch 8 达到最小值，因此 gate、best-val 和 early-stop 在64/64记录上完全相同。作者据此只主张在该受控载体内检测晚发污染并相对端点回滚，不主张发现新检查点或获得现实部署假警报率。

**最强的已核实贡献：** 最可信的贡献是第2–6页对一个时序干预的完整内部验证：冲击明确只在 epoch 9–11 生效，所有64个冲击 run 的最小校准损失都在 epoch 8，且逐 run 的 last-minus-best-val 外层损失均为正，从而表明端点损坏发生在一个先前可用检查点之后。

**维度理由：**

- Soundness：在所定义的合成 E1 人群内，标签冲击时序、64+64 个 hold-out seed、配对外层损失、Clopper–Pearson 界和逐记录相等性支持了狭义结论。主要可信度折损来自预注册在第一个 hold-out 已完成后才冻结，以及阈值只用每模式4个、与结算完全同构的 FIT seed 标定；前者可能污染盲结算，后者没有检验自然健康波动或机制外运输。
- Presentation：论文结构清楚，图1–3直接展示轨迹与完整分布，且明确区分检测、回滚和检查点选择。超链接彩框与两页大面积留白略显未定稿，bootstrap 细节和基线策略参数也过于依赖外部工件，但不妨碍理解主结果。
- Contribution：真时间端点冲击的控制协议及对检测与选择边界的诚实拆分有一定价值；然而健康 m 恒为0、冲击 m 全部远离0，且 gate 必然只在 last 与 best-val 间选择，使观察接近由构造保证。没有现实假警报分布，也没有相对 best-val/early-stop 的增量效用，当前贡献较窄。

**优点：**

- FIT 与 hold-out seed 分开，训练、校准和 outer split 在每个 run 内分离，并用64个 seed/模式报告完整结算。
- 第4–6页同时报告点估计、保守方向的一侧精确二项界和逐 run 效应图，没有把64/64或0/64写成无不确定性的完美性能。
- 论文主动指出健康轨迹严格单调导致 m 恒为0，因此0误报是设计排除而非部署证据。
- gate、best-val、early-stop 的64/64完全相等被保留为负结果，论文没有把相对 last 的收益包装成检查点选择创新。
- 限制部分明确列出阈值运输、冲击机制、现实健康波动和 E2/E6 后续构造。

**问题与可验证修复：**

##### P5-R3-01 · 主要 · 实验严谨性、可复现性、限制与负责任表述

- 位置：第3页，§4.2，第157–161行；表1与附录表7
- 观察证据：论文明确写道 E1 预注册是在“only the first hold-out seed had completed”时冻结，但结算仍把 hold-out seeds 1–64 全部作为64个独立确认单位；PDF 没有说明 seed 1 的结果在冻结前是否不可见。
- 重要性：如果 seed 1 的任何结算信息可见，阈值以外的通过条件、指标选择、分析代码或叙事均可能受其影响，预注册和完全盲 hold-out 的表述就不成立。即使影响只占1/64，也涉及确认性设计的时间顺序与可审计性。
- 必需修复：提供不可变时间戳、运行日志和访问记录证明冻结前未读取 seed 1 结果；否则把 seed 1 排除于确认结算并按原规则重算63个 seed，或用一批全新未查看的 seeds 进行独立结算，同时将当前结果标为部分前瞻。
- 验证标准：审计预注册提交时间、seed 1 完成/访问时间和版本哈希；在排除 seed 1 或全新 seeds 后，四个预注册通过条件及所有表2–5结果仍按冻结脚本成立。
- 仍需证据：带时间戳的预注册、调度与访问日志，以及排除 seed 1/全新 seed 的冻结重算。
- 预期影响：high；判断置信度：high。

##### P5-R3-02 · 主要 · 实验严谨性、重要性、限制与负责任表述

- 位置：第3页§3.3与表1；第5–7页§5.3–5.5、表4–6及限制2、4
- 观察证据：阈值由每模式仅4个 FIT seed 的 max healthy 与 min shocked 中点给出；全部64个健康 hold-out 轨迹严格单调，故 m=0，而全部64个冲击 margin 都超过阈值。FIT 与 hold-out 使用同一合成机制、同一时点和同一冲击比例，没有自然健康非单调性或机制变化。
- 重要性：当前64/64与0/64主要验证了一个人为完全分离的载体，而不是阈值在有重叠、随机训练波动或冲击异质性下的检测能力。四个 FIT seed 也不足以估计阈值运输不确定性，因此证据无法支持超出 E1 机制的报警价值。
- 必需修复：在更多 FIT seeds 上只用训练侧信息冻结阈值，并在独立 settlement 中加入自然非单调健康 run、多个冲击强度/时点/比例/类型及至少一个更真实载体；报告 ROC/PR、校准与按条件分层的精确界。
- 验证标准：阈值在完全未查看的新结算集上保持预设的 FAR 上界与召回下界，且结果不依赖健康 m 恒为0或单一冲击配置；阈值对 FIT seed 重抽样有稳定区间。
- 仍需证据：扩展 FIT/hold-out 原始轨迹、预先冻结的阈值选择规则、异质性分层结果与阈值稳定性分析。
- 预期影响：high；判断置信度：high。

##### P5-R3-03 · 主要 · 重要性、实验严谨性、新颖性

- 位置：第2–5页，式(2)–(5)、表2–3、§5.5.1
- 观察证据：gate 的动作空间定义为 {last,best-val}，且所有冲击 run 的校准最小值均在 epoch 8；因而 gate、best-val 与 early-stop 在64/64记录上相等，gate-minus-last 与 last-minus-best-val 只是符号相反的同一0.3395差值。
- 重要性：这个实验不能判断报警是否提供了任何超出现有验证选择器的信息或检查点价值；相对一个已知受损的 last 的收益几乎由构造和 gate 定义保证。论文诚实承认该边界，但它显著压低当前贡献和部署相关性。
- 必需修复：实现作者提出的 E2 或等价的预注册构造：冲击应能覆盖、移动或暂时掩盖 clean argmin，使报警分数与 best-val/early-stop 不再结构性共线；同时设置不报警、best-val、early-stop、变化点检测与独立回滚策略。
- 验证标准：在独立 seeds 上，gate 的选择至少在预定义比例的 runs 中与 best-val/early-stop 不同，并以配对 outer 指标和错误率控制显示增量价值；若仍完全同现，应明确否定检查点增量贡献。
- 仍需证据：E2 的冻结协议、逐 run 选择身份、配对 outer 结果和与标准变化检测/选择基线的公平比较。
- 预期影响：high；判断置信度：high。

##### P5-R3-04 · 次要 · 可复现性、清晰度

- 位置：第3–4页§3.4、表3；第9–10页附录C
- 观察证据：PDF 只称两个95%区间来自冻结 bootstrap 分布，未给出重采样次数、seed、区间类型或是否按 run 成对重采样；20,000次单侧 sign-flip 出现零个反向 draw 被写为 p≈0。
- 重要性：主效应很大，细节不太可能改变方向，但缺少算法细节妨碍精确重放；有限 Monte Carlo 下的零计数也应报告分辨率或加一估计，而非近似数学零。
- 必需修复：在正文或附录列出 bootstrap/置换的完整算法、随机种子和 Monte Carlo 分辨率，并报告 p<1/20001 或加一估计。
- 验证标准：独立实现按相同逐 run 成对重采样后复现表3端点至报告精度，且 p 值表达与有限抽样分辨率一致。
- 仍需证据：统计脚本、配置与独立重算日志。
- 预期影响：low；判断置信度：high。

##### P5-R3-05 · 次要 · 可复现性、清晰度、新颖性

- 位置：第1–4页§2、表1–2
- 观察证据：论文列出 SWA、EMA、soup、rollback 和 early-stop，但未在 PDF 中给出各自窗口、衰减率、组成或停止耐心，也没有系统讨论变化点检测、概念漂移监测或序贯报警基线。
- 重要性：这些策略不是主结论的必要条件，但配置缺失使表2的公平性难核查，邻近检测文献的缺位也让贡献边界不够精确。
- 必需修复：补充所有选择/平均策略参数及冻结时间，并将 last-margin 与至少一个标准变化点或漂移检测规则在同一 E1/E2 数据上比较；若不做实验，至少在相关工作中明确差异。
- 验证标准：读者可从 PDF/匿名补充材料逐策略重建表2，且新增基线按同样 FIT/hold-out 分割标定。
- 仍需证据：完整策略配置、冻结记录和邻近基线结果或定位分析。
- 预期影响：low；判断置信度：medium。

**给作者的问题：**

- 第3页称预注册冻结时第一个 hold-out seed 已完成：作者在冻结前是否看过 seed 1 的轨迹、margin、outer 指标或日志？若看过，为何仍把它纳入 confirmatory 的64个 seed？
- 阈值 τ 只由健康/冲击各4个 FIT seed 的 max/min 中点确定。四个 shocked FIT margins 的范围、对单个 FIT seed 的敏感性以及换一批 FIT seeds 后的 τ 分布是什么？
- 在健康轨迹 m 恒为0时，任何 0<τ<min shocked m 都会得到同一 operating point。为什么应把结果归因于 τ=0.1354 的标定，而不是归因于构造产生的完全分离？
- 配对 bootstrap 的重采样单位、重复次数、区间类型和随机种子是什么？表3两个区间为何分别计算且端点略不对称？
- SWA、EMA、soup、rollback 与 early-stop 的确切窗口、衰减、组成和停止规则是什么，是否在查看 hold-out 前冻结？
- 能否在自然非单调健康轨迹与多个冲击时点、比例、映射和强度上冻结一个 E2/E6 阈值，然后完全独立结算，而不让 best-val 与 gate 结构性共线？

**能提高评分的证据：**

- 时间戳与访问审计证明 seed 1 在预注册冻结前完全未被查看，或在排除 seed 1/全新 seeds 后所有预设结果无变化。
- 在自然非单调健康轨迹和多种未见冲击条件上，以更充分 FIT seeds 冻结的阈值仍满足预设 FAR/召回界。
- E2 类构造使 gate 与 best-val/early-stop 可分，并在独立结算上显示配对的增量 outer 价值。
- 与标准变化点/漂移检测方法按相同标定预算的比较显示 last-margin 在简单性、时延或错误率上有明确优势。

**会降低评分的证据：**

- 冻结前确实查看并据此调整了 seed 1 的 hold-out 结果，且排除它或新结算后预设条件不再成立。
- 加入自然健康波动后 FAR 明显超过预设0.10，或阈值对4个 FIT seed 的选择高度不稳定。
- 在不与 best-val 共线的冲击下报警无法提供增量价值，或标准变化点基线在相同信息下明显更稳健。

**伦理标记：** 否。实验是合成序列分类，无人体参与者或个人信息；冻结 PDF 中未见直接伦理风险。部署安全措辞已被作者限制为受控 E1 观察。

**LLM 使用披露：** 本审稿由全新、隔离的 AI 子代理 R3 生成，仅用于内部投稿前质量控制；该子代理只读取指定的冻结 PDF、审稿协议、量表与 JSON schema，未与任何其他评审通信，也未接触作者计划、历史评审、目标分数或版本历史。

**评审限制：**

- 按隔离要求仅审阅冻结 PDF，未读取论文所称匿名 workspace、原始128条记录、预注册或统计脚本，因此无法核验时间戳、访问历史和端到端重放。
- 按任务要求未联网，引用的存在性、元数据及相关工作覆盖未做外部核验；对变化点/漂移文献缺口的判断仅基于 PDF 所列参考文献。
- PDF 共10页，文本抽取完整且逐页视觉核查未发现影响内容读取的解析故障。

### P7

#### 交付与来源状态

- 选定输入：`/Users/liuhanzuo/Downloads/P7-paper.pdf`
- 源状态：`exact_latex`
- 润色 PDF：`/Users/liuhanzuo/MacLLM-Bench/output/pdf/paper_polish_20260826/P7-polished.pdf`
- 冻结 SHA-256：`9a9cd2f2f253ea8e671235b98dec83387d7c07db00fb6491580825254b7602a2`
- 总页数：12；主文状态：主文不超过9页。
- 版面核验：pass；构建：pass。
- 旧评分基线：4,4,4；旧中位数：4。

#### 三评量化结果

| 审稿人 | 角色 | Overall | Confidence | 建议 | Soundness | Presentation | Contribution | 当前上限 | 必需修改后预测 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| R1 | 新颖性与定位 | 4 | 4 | 略低于接收线 | 2 | 2 | 2 | 4 | 6 |
| R2 | 技术正确性 | 4 | 4 | 略低于接收线 | 2 | 2 | 2 | 4 | 6 |
| R3 | 实验严谨性 | 4 | 4 | 略低于接收线 | 2 | 2 | 2 | 4 | 6 |

三评中位数为 **4**，均值 4.00，跨度 0，接收侧票数 0/3。

#### 编辑记录

- [结构审计](work/P7/structure_audit.md)
- [语义锁](work/P7/semantic_lock.md)
- [修订日志](work/P7/revision_log.md)
- [待核验事项](work/P7/needs_verification.md)

**修订日志原文：**

> # P7 Revision Log
>
> ## Editorial scope
>
> Evidence-preserving polish of the exact manuscript source corresponding to the supplied PDF. All reported block measurements, uncertainty records, equations, tables, figures, citations, labels, and provenance strings were locked before editing.
>
> ## Structural changes
>
> - Narrowed the title to identify the work as an exploratory two-checkpoint study.
> - Shortened the abstract, introduction, and conclusion while retaining both checkpoint outcomes and the full evidence boundary.
> - Reduced duplication between the main Results narrative and the appendix uncertainty ledger.
> - Added the missing local table inputs required for a self-contained build; their contents are unchanged.
>
> ## Language and claim changes
>
> - Reconciled an internal inconsistency in the depth discussion. The revised manuscript states that activation-tail turnover and block depth are strongly entangled and that independent predictive value is unresolved.
> - Described the 4B result as an exploratory positive association and the 1.7B estimate as weak/uninformative, not as a proven cross-model difference.
> - Replaced causal or promotional readings of activation-outlier work with observational, compatibility-based language.
> - Removed an uncertain provenance qualifier attached to the unchanged 0.401 value; the remaining label ambiguity is recorded for author verification.
>
> ## Preserved scientific content
>
> - The two checkpoints, probed-block counts, HQQ configuration, turnover definition, primary correlations, depth correlation, partial estimates, intervals, robustness records, AUROC values, hashes, and two-of-eight-cell boundary are unchanged.
> - No new experiment, citation, universal trigger claim, or mechanism claim was added.
>
> ## Build and QA
>
> - Built twice with `pdflatex` through `latexmk` and shell escape disabled.
> - Citation-key, label, and displayed-equation sets match the untouched source exactly.
> - The final log contains no LaTeX errors, unresolved citations/references, fatal errors, or overfull boxes.
> - Extracted the final text and rendered all twelve pages; no clipping, overlap, or missing figure/table was observed.

**待核验事项原文：**

> # P7 Needs Verification
>
> - Only two of eight preregistered cells were executed. Generalization across checkpoints, model families, scales, or deployment distributions remains untested.
> - Turnover and block depth are nearly collinear on the 4B carrier; the reported partial analyses do not establish independent predictive value.
> - The manuscript reports a 0.401 depth-reweighted diagnostic, but the exact historical qualifier attached to that number is internally ambiguous. The value is unchanged; the author should verify its preferred label against the originating analysis artifact.
> - Three-host equality reflects deterministic pipeline replication with an effective independent run sample of one, not three scientific replicates.
> - The registered trigger/damage-gain target was not passed or evaluable at the achieved resolution; the manuscript does not establish a deployable trigger.
> - Bibliographic metadata, run paths, and external artifacts were preserved but not independently verified or rerun during this editorial pass.

#### R1（新颖性与定位）完整评议

**论文概述：** 论文提出 activation-tail turnover：在全精度 Qwen3 模型上比较 calibration 与 deployment prompt family 时，各 block 输入激活 max/mean-absolute 比率的标准化变化，并将其与逐 block 单独做 HQQ 4-bit weight-only 量化所造成的 deployment perplexity 相对增量做 Spearman 相关。在 corrected pad-masked 结果中，Qwen3-4B 的 12 个 probed block 给出 ρ=+0.559，Qwen3-1.7B 的 11 个 block 给出 ρ=+0.118。4B turnover 与 block depth 高度共线（ρ=+0.909），partial correlations 的区间宽且跨零；两模型结果来自单一确定性 draw，三台主机重放 bit-identical 但有效独立 run 数仍为 1。只完成预注册 8 cells 中 2 个，AUROC/damage-gain gate 未通过或未完成，因此论文自定位为 two-checkpoint exploratory boundary study。

**最强的已核实贡献：** 最强的已验证贡献是一个具体、可审计的候选信号压力测试框架：predictor 完全在未量化模型上计算，outcome 由逐 block 单独量化直接测量；该框架在 4B 上发现正向但深度混杂的关系、在 1.7B 上发现弱而不确定的关系，并明确展示任何 layer-wise runtime signal 都必须与 block index 这一零成本结构基线比较。

**维度理由：**

- Soundness：论文对 corrected pad-masked 标签、层依赖、深度共线、两种区间、噪声包络和 deterministic replay 的边界披露较充分。但 4B 只有 12 个刻意深层过采样的非独立层点，1.7B 只有 11 个，partial CI 跨零，预注册 8 cells 仅运行 2 个；更严重的是主分析与附录混入 masked/unmasked 的不一致 Δz 与区间版本。当前可相信的是两个 checkpoint 上的描述性散点，不是独立预测价值或 population inference。
- Presentation：Figure 1-3 视觉上清晰且无裁切，但正文被 C4-C12/F/E 等审计编号和密集表格主导。第 3、7、11 页关于 masked/unmasked 区间与跨 carrier Δz 的文字和数值不一致，使读者难以确定唯一的 canonical result。
- Contribution：用全精度激活域移统计预测单 block PTQ damage 的问题设定有潜在实用性，且要求候选信号击败 block depth 是有价值的设计教训。然而 turnover 在 4B 上与 depth ρ≈0.91、独立 partial 未解析，1.7B 读数弱，未测 fallback policy，AUROC gate 未通过且标签脆弱。现有贡献仍是探索性边界而非新可靠能力。

**优点：**

- 第 2-3 页公式 (1)-(3) 清楚分离未量化 predictor 与真正的 single-block quantization outcome，并报告具体 probe rule、HQQ 配置和 prompt 数。
- 第 4-8 页主动披露 layer 非独立、深度过采样、dominant block、两种 CI、leave-one-block-out、outcome-noise envelope、prompt bootstrap 以及仅运行 2/8 cells。
- 第 5 页 §3.3 正确区分跨三台主机 bit-identical determinism 与 independent replication，明确 neff=1。
- 第 9 页 Related Work 覆盖 runtime fallback、calibration-free quantization、per-layer sensitivity、activation outliers，并承认若要主张部署信号必须超过 depth baseline。
- Figure 2 与 Figure 3 直接显示 dominant layer 和 depth 结构，避免只用相关系数隐藏数据形态。

**问题与可验证修复：**

##### P7-R1-01 · 主要 · 新颖性、重要性、实验严谨性

- 位置：PDF 第 4 页 Table 1/§3.2、第 6 页 Figure 3、第 9 页 §5 Related Work
- 观察证据：在 4B 上 turnover-depth 相关为 +0.909，turnover-damage raw 相关为 +0.559；masked partial ρ(turnover,damage|depth)=+0.266 且 bootstrap CI [-0.49,+0.66]。论文只与 depth 做数值比较，没有在实验中比较其他简单 activation-shift 指标或文中列出的 Hessian/weight-based per-layer sensitivity score。
- 重要性：在不能证明独立于 depth 的预测价值、也不能优于合理候选统计量时，activation-tail turnover 作为新信号的增量未建立；观察可能只是深层 block 与一个 dominant damage point 的共同排序。
- 必需修复：在预先固定的完整或 depth-balanced layer set 上加入一组低成本 predictor baselines，并用 nested/held-out ranking 或跨 checkpoint 验证 turnover 的增量，而不只在同一 12 点样本上做 partial correlation。
- 验证标准：turnover 在未用于选择统计量或阈值的 checkpoint/domain 上，相对 depth 与至少两类合理 activation/offline sensitivity baseline 改善预先指定的 rank/AUROC 指标，并给出 paired uncertainty。
- 仍需证据：新 baseline 计算、独立 checkpoint/domain 或严格 held-out layer evaluation。
- 预期影响：high；判断置信度：high。

##### P7-R1-02 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：PDF 第 3 页 §2.3-2.4、第 5-8 页 Limitations C4/C10/C12/E，以及第 12 页 Table 4
- 观察证据：每个 carrier 仅有一个 checkpoint，分析单位是 11-12 个相互依赖且非交换的层；probe rule 刻意使最深四层过采样，4B 超过一半层的 |damage| 在一个 noise width 内。预注册 2 carriers×2 quantizers×2 shifts 共 8 cells，只执行 HQQ4 单一 shift 的 2 cells。
- 重要性：对非随机、依赖层做普通 layer bootstrap/Fisher inference 不能支持模型总体或部署总体的结论；选择性 probe 又使深度关系部分由设计制造。未完成的注册网格意味着候选信号最基本的稳健性尚未测试。
- 必需修复：至少完成 depth-uniform/full-layer outcome，并增加独立 checkpoint、第二域移和另一量化设置；将当前 layer-resampling 区间降为描述性敏感性，主要推断改用 checkpoint/domain 级复现。
- 验证标准：结论在完整层集与多个独立 checkpoint/domain 上方向一致，且预先规定的 gate 在足够标签分辨率下可判定；若不新增证据，删除任何 inferential/power 语言。
- 仍需证据：未探测层的 single-block PTQ 结果、剩余注册 cells 或新的独立复现。
- 预期影响：high；判断置信度：high。

##### P7-R1-03 · 主要 · 技术正确性、清晰度、可复现性

- 位置：PDF 第 1 页 Abstract、第 3 页 §2.4、第 4 页 §3.1、第 7 页 Table 2/C8 与第 11 页 Appendix Table 3
- 观察证据：masked Table 2 报 4B ρ=0.559、Fisher CI [-0.060,0.868]、bootstrap CI [-0.118,0.928]，两者都含零；但 §2.4/摘要称 interval conventions 对零排除结论不同。跨 carrier masked Fisher-z 差由 Table 1 可得约 0.513，§3.1 与 Table 2 也报 0.513；C8 却写 Δz=+0.79，Appendix Table 3 报 0.794 及另一组区间，且未清楚标注为 superseded unmasked。
- 重要性：这些不是舍入差，而是不同标签/分析版本混入同一稿件。读者无法确定 canonical effect、CI 或 cross-carrier 结论，审计性主张因此受损。
- 必需修复：从一个 canonical masked JSON 自动生成所有主文、表格、图注和附录数值；把历史 unmasked 结果放入明确标为 superseded 的单独表，逐项标注版本，删除旧 Δz/CI 文句。
- 验证标准：对 PDF 全文抽取 ρ、Δz、CI 和 p 后，每个 estimand 只有一个 canonical masked 值；由 Table 1 的 ρ 可机械复算 Table 2/正文 Δz，且摘要关于零排除的陈述与两类 CI 一致。
- 仍需证据：canonical masked analysis artifact、生成脚本和全文数值一致性检查。
- 预期影响：high；判断置信度：high。

##### P7-R1-04 · 主要 · 重要性、实验严谨性、限制与负责任表述

- 位置：PDF 第 2-3 页 §2.3、第 8 页 C11/Scope 与第 9 页结论
- 观察证据：所谓 wikitext/gsm8k 实为每类 64 条单模板 narrative/arithmetic prompts，不是对应真实语料；只测试一个 prompt-family shift、一个 HQQ 4-bit weight-only quantizer、一个 perplexity outcome 和同一家族两个 checkpoint。
- 重要性：低多样模板 shift 可能产生特定的 activation-tail 结构，无法代表真实 traffic drift；这也使 runtime relevance 和跨任务普适性未建立。
- 必需修复：改用真实、去重、多样的 calibration/deployment corpora，至少增加第二种 domain shift，并在一个独立模型家族或量化器上复现；配置键也应避免使用真实 benchmark 名称指代模板代理。
- 验证标准：在冻结的真实语料 split 上重算 predictor 和 damage，报告跨 prompt bootstrap 与跨 checkpoint 结果，且模板与真实域结论明确区分。
- 仍需证据：真实 prompt-domain 实验与至少一个正交模型/quantizer 复现。
- 预期影响：high；判断置信度：high。

##### P7-R1-05 · 主要 · 重要性、实验严谨性、可复现性

- 位置：PDF 第 6 页 Limitations C7/F 与第 12 页 Table 4
- 观察证据：masked AUROC 为 0.800/0.567，4B 单个标签翻转可跨越 KILL/MERGE/PASS；注册 gate 还要求 ≥30% damage reduction，但没有运行 fallback-baseline forward，现有可行性界仅约 14%/13.5%。
- 重要性：论文动机是 runtime trigger，但既没有稳定的分类端点，也没有测量触发后的实际策略、损失改善或成本。因而 deployment value 仍完全假设化。
- 必需修复：冻结并执行实际 fallback policy，包括阈值、可用精度层、额外计算/延迟和 damage endpoint；在标签分辨率足够的独立 cells 上结算完整 gate，而不是只报 AUROC 半边。
- 验证标准：完整 gate 在未参与调参的 cells 上同时达到预先规定的 ranking 与 damage-gain 条件，并相对 depth/static fallback 给出配对改善和成本。
- 仍需证据：fallback forward、系统成本和独立 gate settlement。
- 预期影响：high；判断置信度：high。

##### P7-R1-06 · 次要 · 清晰度

- 位置：PDF 第 5-8 页 Limitations 与第 7、11 页 Tables 2-3
- 观察证据：C4-C12、F、E 等大量编号限制与多层 provenance 字段挤入主文；页面渲染显示第 6-8 页文字和表格密度显著高于其余部分。
- 重要性：重要边界虽完整，却掩盖了核心因果链，且增加版本不一致不易被发现的风险。
- 必需修复：主文只保留决定分数的三项限制（非独立层、深度混杂、仅 2 cells），其余审计字段移至结构化附录，并增加一张 canonical result table。
- 验证标准：读者可仅凭摘要、Figure 1、一个主结果表和三段限制准确复述唯一主张与失败门槛。
- 仍需证据：无需新实验，仅需结构重写。
- 预期影响：medium；判断置信度：high。

**给作者的问题：**

- 请给出一个唯一 canonical 的 masked result ledger：为何 §3.1/Table 2 使用 Δz≈0.513，而 Limitation C8 和 Appendix Table 3 又使用 Δz≈0.79/0.794？这些分别对应哪些标签版本与 bootstrap 对象？
- 第 3 页称 Fisher 与 bootstrap interval 在 4B 上对零排除结论不同，但第 7 页 masked Table 2 的两者都包含零；这里是否混入了第 11 页历史 unmasked 数值？
- 为何不在同样的 probed blocks 上比较 turnover 与其他低成本全精度 predictor，例如 activation mean/variance/kurtosis、静态 tail mass、FP loss gradient 或直接 depth？没有这些对照，如何证明 turnover 这一特定构造而非任意 activation shift 有增量？
- 能否量化所有 28/36 层，或至少使用预先固定的 depth-stratified 均匀抽样，以消除 deep-decile 过采样和 range restriction？
- 若目标是 runtime trigger，实际 fallback action、latency/measurement cost 和 damage reduction 如何定义？当前 predictor 仍需收集 64 deployment prompts 的 full-precision hidden states，这一成本是否符合部署场景？
- 第二个真实 prompt domain、不同 quantizer/bit width 或独立 checkpoint 是否会保持 4B 的方向，还是当前关系仅由一个模板 shift 与一个 dominant block 驱动？

**能提高评分的证据：**

- 完整或 depth-balanced 层集上，turnover 相对 depth 和其他合理 activation/sensitivity baselines 具有稳定增量。
- 多个独立 checkpoint、真实 domain shift 和至少另一量化设置复现 4B 方向，并以 checkpoint/domain 为复现单位。
- 统一 canonical masked ledger 消除所有 Δz/CI 版本冲突。
- 实际 fallback policy 在盲结算中同时达到稳定 ranking、显著 damage reduction 和可接受系统成本。

**会降低评分的证据：**

- 全层或均匀深度采样后 4B 相关消失，或简单 depth/activation baseline 始终相当或更好。
- 真实 prompt domains 或其他 checkpoint 上方向频繁翻转。
- canonical artifact 证实主文混用了不可比较的 masked/unmasked outcome，且无法重建 headline。
- 实际 fallback 的成本或 damage gain 无法超过静态策略。

**伦理标记：** 否。研究使用公开模型与程序模板 prompt，未涉及人体参与者或个人数据；当前主要风险是部署可靠性主张不足，论文已避免把候选信号描述为安全证书。

**LLM 使用披露：** 本审稿由一个全新、隔离的 AI 子代理生成，仅用于内部投稿前质量控制；该子代理未与任何其他评审通信，也未接触作者计划、旧评审或目标分数。

**评审限制：**

- 按任务约束未联网核验量化与 activation-outlier 文献；新颖性仅相对于 PDF 的 Related Work 判断。
- 仅审阅冻结 PDF，未运行 Qwen/HQQ 实验，也未读取代码、JSON、匿名 manifest 或 per-layer artifact，因此无法独立验证重放与 SHA 记录。

#### R2（技术正确性）完整评议

**论文概述：** 论文在 Qwen3-4B 和 Qwen3-1.7B 上计算每个被探测块的激活尾部 turnover，即全精度模型在两类模板提示间的 max-to-mean 激活比标准化变化，并把它与只量化该块为 HQQ 4-bit 后的部署代理困惑度退化做 Spearman 相关。校正 pad mask 后，4B 的点相关为 0.559，1.7B 为 0.118。4B turnover 与层深度相关 0.909，偏相关区间跨零；只执行了预注册八个单元中的两个，AUROC gate 未通过且标签对单次翻转敏感。三台主机的结果位级相同仅表明确定性。论文将工作定位为探索性边界研究。

**最强的已核实贡献：** 在明确列出的 12 个 4B 层与 11 个 1.7B 层上，论文给出了 turnover 和单块 HQQ4 损伤的逐层散点及可复算的秩相关，并实质性展示：4B 上观察到的正关系与深度强烈纠缠，而 1.7B 上最大损伤块并非 turnover 最大块。因此，任何候选层级运行时信号都必须先与简单结构基线比较。

**维度理由：**

- Soundness：逐层 turnover 与单块量化损伤的点估计是可定义的，论文也坦率承认只运行两个单元、层不是独立样本、深度近共线且 gate 未通过。但 Fisher 区间、层 bootstrap、置换和 P(ρ>0) 仍把经过偏置抽样的 11-12 个依赖层当作可交换单位，不能提供所暗示的统计稳健性；此外主 masked 结果与附录历史口径存在具体数值冲突。
- Presentation：测量图和散点图较清楚，限制列表覆盖面很广；然而正文、表2和附录表3在 masked/unmasked、Δz 与区间口径上混杂，限制部分过长且多处用工件代号代替自包含解释，使中心证据难以审计。
- Contribution：用无需运行量化模型的激活漂移信号去预测逐块 PTQ 损伤，并把简单深度基线作为必要对照，是有意义的探索问题。但只有同一家族两个检查点、一个量化器/位宽、一个低多样性模板域对，4B 信号又无法与深度分离，当前结果更像试点边界记录而不是可泛化知识或可部署触发器。

**优点：**

- 把 predictor 限定为全精度模型上可计算、outcome 限定为单块量化的真实困惑度变化，避免机械地从量化输出构造预测器。
- 主动报告 pad-mask 修正、历史未遮罩值、leave-one-block-out、深度基线、部分相关和未通过的预注册 gate，而没有把 4B 点值包装成部署证书。
- 正确区分三台主机位级复放的确定性与独立复现，并明确有效跨运行样本为1。
- 图2直接呈现单个主导层和大量近零层，使结果脆弱性可见。

**问题与可验证修复：**

##### P7-R2-001 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第3页§2.4、第5页§4限制C4-C5、第7页表2、第8页限制C10-C12
- 观察证据：主要推断只有一个检查点中按固定规则选出的 12 层，作者明确写明这些层既不独立也不可交换，却仍报告 Fisher-z CI、重采样层的 percentile CI、层置换 p 和 P(ρ>0)约0.998，并以此称 4B 方向对噪声和 prompt bootstrap 稳健。
- 重要性：这些推断程序的抽样单位假设与实际单位冲突。它们最多描述对特定有限层集合进行人为重加权/扰动后的数值稳定性，不能量化模型总体、层总体或检查点总体的不确定性，也不能把一个探索性点相关升级为统计支持。
- 必需修复：把所有层重采样结果明确降格为有限集合敏感性分析，不赋予覆盖率或概率含义；要作统计推断，需要跨独立检查点/训练种子/域采样，或给出尊重层相关结构的层级模型与足够独立单元。
- 验证标准：用预先规定的多个独立模型或检查点作为最高层抽样单位，模型内保留完整层向量，进行 cluster/model-level 推断；检查 4B 方向能否跨独立单元重现。
- 仍需证据：独立检查点/种子/域的完整逐层测量，或有可辩护相关结构的层级推断。
- 预期影响：high；判断置信度：high。

##### P7-R2-002 · 主要 · 技术正确性、实验严谨性、重要性

- 位置：第3页§2.3、第4页表1与§3.2、第5-6页图2-3、第8页限制E
- 观察证据：固定 probe 规则在最深十等分中放入4/12个层；4B turnover 与深度 ρ=0.909，turnover 与损伤的 raw ρ=0.559，而偏相关区间宽且跨零。将所测层重加权后相关降至0.401，未测层没有 outcome。
- 重要性：当前样本设计主动制造了深层过采样，并且 turnover 与深度近共线；因此无法判断候选信号是否提供超越层索引的独立信息，也无法外推到未探测层。这个不确定性直接限制论文的预测器贡献。
- 必需修复：至少在一个模型上执行全层单块量化，或在测量前冻结深度均衡的探测设计；比较 turnover、depth、权重/曲率敏感性等基线的交叉验证排序性能。
- 验证标准：在未参与拟合的层/检查点上比较 turnover 与 depth 的 rank loss、AUROC或 top-k damage recall，并用模型级重复检验增量性能；预先规定标签和噪声容忍区。
- 仍需证据：全层或深度均衡 outcome、独立检查点以及预注册基线比较。
- 预期影响：high；判断置信度：high。

##### P7-R2-003 · 主要 · 技术正确性、清晰度、可复现性

- 位置：第4页§3.1、第7页表2 Panel B及其后正文、第11页附录表3与方法敏感性注释
- 观察证据：主 masked 表2给 Δz=+0.513、八个 seed-envelope 下界范围[-0.88,-0.78]、上界范围[+2.06,+2.17]且0/8排除零；紧随其后的正文却写 Δz约+0.79、下界“grazes zero”。附录表3又给0.794，并把若干 Fisher/Bootstrap/LOO 数值列为未明确标识的旧口径，与主 masked 梯子不一致。
- 重要性：读者无法确定哪一组是当前主分析、何谓一个置信区间的“lower-edge range”，也无法把估计值与区间对应。该混杂影响跨载体异质性、稳健性和附录重建。
- 必需修复：建立唯一的 canonical masked 结果表，逐行给 estimator、标签版本、随机种子、点估计和单一可解释区间；把所有历史未遮罩值移入明确标记的审计表，并删除正文旧数字。
- 验证标准：自动从 canonical JSON 生成正文、表2和附录，执行一致性断言：同一标签口径的点估计/CI/LOO在全稿完全相同，估计值和区间含义可解析。
- 仍需证据：单一冻结 masked 结果工件及自动化数值一致性报告。
- 预期影响：high；判断置信度：high。

##### P7-R2-004 · 主要 · 技术正确性、实验严谨性、可复现性

- 位置：第7页表2 Panel A/B重建说明、第8页限制C12、第5页§3.3
- 观察证据：三次决定性运行在全部字段上位级相同，但 C12 又从“五臂重测”的近零层 spread 构造 outcome-noise envelope；论文未在自包含方法中说明五臂是什么、每层独立重复数、噪声是否同分布/相关，或为何可将该 spread 独立施加到每个 δ。超过一半4B层的|δ|不超过一个该噪声宽度。
- 重要性：P(ρ>0)约0.998 是支持4B“稳健方向”的主要依据，但其概率完全依赖所选人为噪声模型。若层间噪声相关、异方差或近零偏差来自系统误差，该模拟会严重高估稳定性。
- 必需修复：完整定义噪声来源和生成模型，报告逐层独立重复；优先用真实重复测量而非合成包络，并保留层间协方差。prompt bootstrap 也应解释64个单模板句子的抽样总体。
- 验证标准：在新的独立 prompt 集和独立量化前向上重复每层 outcome，多层联合重算 ρ；将真实重复分布与当前包络的分位数和符号稳定性比较。
- 仍需证据：逐层重复 outcome、prompt 级 predictor 值、协方差估计和噪声模型代码。
- 预期影响：high；判断置信度：medium。

##### P7-R2-005 · 次要 · 清晰度、限制与负责任表述

- 位置：第2页图1图注与第1、3、8-9页正文
- 观察证据：图1图注把 Qwen3-1.7B 称为“a null relation”，而摘要、表1和限制C12更准确地称其为弱且区间无信息、P(ρ>0)约0.59。
- 重要性：未能拒绝或区间宽并不等于确认零关系；图示的简化措辞与论文其他部分的负责任表述不一致。
- 必需修复：统一改为“weak and unresolved/uninformative”，避免使用 null 作为已建立结果。
- 验证标准：全文搜索 null/kill/absent，确保都与预注册 band 或统计证据的实际含义一致。
- 仍需证据：修订后的统一术语。
- 预期影响：low；判断置信度：high。

**给作者的问题：**

- 既然作者明确承认层既不独立也不可交换，Fisher、层 bootstrap、层置换及 P(ρ>0) 应被赋予什么频率学含义？为什么这些量还能支持“robust to noise/prompt bootstrap”的表述？
- 表2的 masked Δz=0.513 与第7页正文/附录表3的 Δz约0.79 分别来自哪一标签口径？附录表3的区间/edge 行应如何解析，为什么正文称其下界“grazes zero”？
- C12 的 outcome-noise envelope 来自哪些五个 arm、每层有多少独立重复、噪声分布如何拟合？这与三主机精确复放 max|Δ|=0 的确定性结果是什么关系？
- 深度重加权 ρ=0.401 使用哪一种加权 Spearman 定义和权重？在只有偏置选出的层上重加权为何能代表未观测的均匀深度总体？
- 为何不在至少一个模型上量化全部层，或预先选取真正深度均衡的层，以直接检验 turnover 是否超越 block index？

**能提高评分的证据：**

- 至少一个模型的全层或预先冻结的深度均衡测量，显示 turnover 在独立层/检查点上稳定超越 block index。
- 以独立检查点或模型为最高抽样单位的重复实验，而不是对单模型依赖层重采样。
- 真实重复 outcome 与新 prompt 集上的联合稳定性分析，替代未充分定义的噪声包络。
- 完成更多预注册量化器/域单元，并用唯一 canonical masked ledger 消除所有数值冲突。

**会降低评分的证据：**

- 全层或深度均衡采样后4B相关消失，或 turnover 不优于深度/离线敏感性基线。
- 独立 prompt 或量化重复显示当前 P(ρ>0)主要由噪声模型假设产生。
- canonical 重算不能复现 masked ρ=0.559，或发现表2/表3混用了不可比较标签。
- 额外预注册单元普遍落入 KILL，且没有可解释异质性。

**伦理标记：** 否。研究使用公开模型与程序化模板提示，未报告人体研究或个人信息；当前主要风险是部署触发器证据不足，论文已明确不作安全证书声明。

**LLM 使用披露：** 本审稿由一个全新、隔离的 AI 子代理生成，仅用于内部投稿前质量控制；子代理只接收冻结 PDF、指定的 ICLR 2026 量表与输出 schema，未与任何其他评审通信，也未查看作者计划、旧评审或目标分数。

**评审限制：**

- 仅审阅指定冻结 PDF；未访问公开代码、JSON、模型或 GPU 运行，因此无法重算逐层量化、核对工件哈希或判断脚本与正文的一致性。
- 按任务要求未联网，未核验 2025-2026 引用、Qwen3/HQQ 版本细节或相关 PTQ 触发器文献的完整性。

#### R3（实验严谨性）完整评议

**论文概述：** 论文提出 activation-tail turnover：在全精度模型上比较64条模板叙事提示与64条模板算术提示时，每个 block 输入激活的 max/mean-absolute 比率发生的标准化变化。结果变量是只把该 block 的线性层量化为4-bit HQQ 后，部署提示上的相对 perplexity 增量。作者在 Qwen3-4B 的12个抽样 block 和 Qwen3-1.7B 的11个抽样 block 上计算 Spearman 相关；pad-masked 主读数分别为0.559与0.118。4B 的 turnover 与深度高度相关，partial 相关区间跨零；1.7B 不具信息性。三台物理主机上的结果逐位一致，但只是同一确定性 draw 的重放。论文只完成预注册8个 cell 中2个，AUROC 门槛分辨率不足且30% fallback gain 未测，因此把结果定位为探索性边界，而非部署触发证书。

**最强的已核实贡献：** 最可信的贡献是识别并量化了候选信号的结构性混杂：第4–8页显示 Qwen3-4B 上 turnover 与 block 深度的秩相关达到0.909，深层过采样重加权后主相关从0.559降至0.401，且两个 masked partial 相关的区间都跨零。这支持“任何层级触发信号必须先对照简单深度基线”的实验设计教训。

**维度理由：**

- Soundness：论文对探索性边界披露充分，但中心相关性只来自同一 checkpoint 内11–12个非独立 block，常规 Fisher/Bootstrap 区间均跨零，4B 上 predictor 与深度的 Spearman 相关为0.909且 partial 区间跨零。噪声包络只在固定已测 block 和 checkpoint 条件下扰动数值，不能替代模型级复制、层选择不确定性或深度去混杂。正文 masked 主分析与附录表3还混入一套未标注的旧数值，削弱可核查性。
- Presentation：图1–3清楚，作者反复陈述不主张部署触发器；但正文被大量 C4–C12/F/E 标签、路径名和重复限定压缩，表2与附录表3的 masked/unmasked 数值版本未清楚分离，导致读者难以确认哪一组区间、LOO 和 Δz 是主结果。
- Contribution：用量化前的激活尾部迁移预测单 block PTQ 损伤并强制与深度基线比较，是相关的问题设定；但只执行预注册8个 cell 中的2个，未运行 fallback gain，且一个 checkpoint 弱、另一个深度混杂，当前更像有用的失败模式记录而非可推广的触发信号或机制贡献。

**优点：**

- 预测量完全在全精度模型上计算，结果量通过逐 block 单独量化获得，测量对象与候选部署信号的因果顺序定义清楚。
- 第3–8页公开了原始秩相关、Pearson、Fisher 区间、block bootstrap、置换、LOO、深度 partial、噪声包络与 prompt bootstrap，而非只展示最有利统计量。
- 作者明确承认 layer 不是独立样本、三主机只是确定性重放、只运行2/8 cells、深层过采样和单/双 dominant block 风险。
- pad masking 修正改变了多个符号与标签，论文保留历史未 masked 行作为审计记录，并没有继续把旧结果当主结果。
- 没有把 AUROC=0.800 或未运行的 fallback gain 写成通过预注册安全门。

**问题与可验证修复：**

##### P7-R3-01 · 主要 · 实验严谨性、技术正确性、限制与负责任表述

- 位置：第2–8页，§2.3–3.3，表1–2，限制C4、C9、C10、C12、E
- 观察证据：每个模型只有一个 checkpoint，单位是同一网络中的11或12个相互依赖 block；规则只探测28/36个 block 中的一部分，并故意在最深 decile 过采样。三次主机重放 max |Δ|=0，因此跨运行有效样本仍为1。
- 重要性：block 不是独立、可交换的总体样本，常规 Fisher SE、block bootstrap、置换和 LOO 不能支持关于 checkpoint 或模型总体的推断。选样偏差还使已测层上的相关不能外推到未测层；对已测点重加权无法恢复未观测结果。
- 必需修复：至少测量每个 checkpoint 的全部 block 或使用事前定义的均匀/随机层样本，并在多个独立 checkpoint、模型 family 和 prompt domain 上复制；把 checkpoint/domain 作为高层单位做分层或置换推断，block 只作为嵌套观测。
- 验证标准：在预注册的多 checkpoint 分层分析中，turnover 的增量关联方向在 checkpoint 级重复，且层抽样方案不偏向深层；checkpoint 级置信区间或置换检验支持非零/有用效应，而非仅 block-resampling 区间。
- 仍需证据：全层结果、多个独立 checkpoint/domain 的原始逐层数据、事前层采样规则与分层统计分析。
- 预期影响：high；判断置信度：high。

##### P7-R3-02 · 主要 · 技术正确性、实验严谨性、重要性

- 位置：第3–8页，表1–2、§3.1–3.2、图2–3、限制C5/C12/E
- 观察证据：4B masked ρ=0.559，但 turnover-depth ρ=0.909；Fisher CI [-0.060,0.868] 和 percentile bootstrap [-0.118,0.928] 都跨零。masked partial 为0.266与0.006，附录区间分别[-0.49,0.66]与[-0.15,0.88]。深度重加权把主相关降到0.401。
- 重要性：当前数据既不能证明主相关超出抽样不确定性，也不能区分 turnover 与几乎共线的深度。P(ρ>0)≈0.998 的噪声扰动只检验在固定已测点周围加入指定噪声后符号是否稳定，不处理层选择、深度混杂或 checkpoint 总体不确定性，不能升级为确认性证据。
- 必需修复：在深度覆盖均匀的全层/预注册样本和多个 checkpoint 上，把 turnover 与 depth、权重/激活幅度、Hessian/量化误差等竞争预测量纳入预先规定的增量模型；用 held-out layers/checkpoints 或嵌套交叉验证比较 rank 预测。
- 验证标准：在 checkpoint 级 hold-out 上，加入 turnover 后相对仅深度基线的预注册 AUROC、rank correlation 或损失显著且稳定改善，并在删除 dominant block、不同 domain/quantizer 后保持方向。
- 仍需证据：独立 checkpoint 的全层竞争基线矩阵、预注册增量模型与外样本预测结果。
- 预期影响：high；判断置信度：high。

##### P7-R3-03 · 主要 · 可复现性、清晰度、实验严谨性

- 位置：第7页表2与第10–12页附录表3–4；§3.1及限制C5/C8/C11
- 观察证据：主表2明确把 masked 4B Fisher CI、bootstrap、LOO 和 Δz 报为[-0.060,0.868]、[-0.118,0.928]、[0.427,0.718]、0.513；附录表3却给出[0.031,0.871]、[-0.085,0.921]、[0.473,0.709]、0.794，未将这些行标成 unmasked/superseded。正文限制C8又把 Δz=0.79 指向表2，和表2的0.513冲突。
- 重要性：这些不是舍入差，而是不同 label/分析版本。版本混杂改变是否排除零、LOO范围和跨载体差值，直接阻碍读者识别主分析，也使“所有 headline 数字均为 masked”无法从 PDF 内部核验。
- 必需修复：建立单一数值来源，逐行标注 masked primary 与 unmasked historical；删除或更正所有 stale 数值和错误交叉引用，并增加一张从原始 payload SHA 到最终表格行的版本映射。
- 验证标准：自动一致性检查断言摘要、表1、表2、附录表3–4、图注与结论中的每个主数值都由同一 masked JSON 生成；旧数值只能出现在明确的 superseded 栏并可追踪到不同 SHA。
- 仍需证据：单源生成脚本、完整 SHA、字段级 provenance map 和修订后 PDF 的数值一致性报告。
- 预期影响：high；判断置信度：high。

##### P7-R3-04 · 主要 · 实验严谨性、重要性、可复现性

- 位置：第1、6、8、12页，摘要、限制C7/F/C12、附录表4
- 观察证据：预注册规则覆盖2 carriers×2 quantizers×2 shifts 共8 cells，但只运行一个量化器和一个 shift 的2 cells。AUROC 的4B值0.800可因一次标签翻转跨越0.639–0.889，1.7B为0.567；要求≥30% damage gain 的 fallback forward 完全未运行，仅给出约14%的可行性上界。
- 重要性：注册的触发器证书既未完成 cell family，也未测其部署动作，因而无法回答候选是否通过或失败。选择性完成2/8 cells 还使相关阈值分类只能是探索性读数，不能作为预注册验证。
- 必需修复：在不查看新增结果的前提下冻结修订版协议，完成全部 cells 或给出事前、与结果无关的停止规则；提高 layer/label 分辨率，并实际运行定义清楚的 fallback baseline 与 gain 测量。
- 验证标准：所有预注册 cells 和两个门条件均有可审计输出，family-level 判定可在标签扰动和噪声界下稳定复现；未执行 cell 不再被投影值替代。
- 仍需证据：完整8-cell payload、fallback forward、预注册时间戳和稳定性/错误率分析。
- 预期影响：high；判断置信度：high。

##### P7-R3-05 · 次要 · 限制与负责任表述、清晰度、可复现性

- 位置：第2–3页§2.3及第8页限制C11
- 观察证据：配置名 wikitext/gsm8k 实际各指64条单模板生成的叙事/算术句子，并非 WikiText-2/GSM8K；每族只有一种低多样性模板，真实流量复现未测。
- 重要性：虽然正文已披露，但名称容易造成数据集级覆盖的错觉；模板特征可能同时驱动激活尾部和 PPL 损伤，限制对真实 calibration-to-deployment shift 的解释。
- 必需修复：在所有图表、配置说明和摘要中使用 templated-narrative/templated-arithmetic 名称，并加入多模板、真实语料与交叉 domain 的复验。
- 验证标准：读者无需查限制即可准确识别数据来源；候选信号在预注册的多个真实/模板 domain 配对上报告独立结果。
- 仍需证据：完整提示生成器、去重统计、多 domain 原始结果。
- 预期影响：medium；判断置信度：high。

**给作者的问题：**

- 附录表3中的 Fisher CI [+0.031,+0.871]、bootstrap [-0.085,+0.921]、LOO [+0.473,+0.709] 和 Δz=0.794 分别来自哪一种 label 版本？为何与表2 masked 主结果 [-0.060,+0.868]、[-0.118,+0.928]、[+0.427,+0.718] 和 Δz=0.513 不同而未标注为 superseded？
- 噪声包络 P(ρ>0)=0.998 的概率空间究竟是什么？它是否仅条件于已选的12个 block、同一 checkpoint 与经验噪声模型，因而不包含 layer 选择、checkpoint 或 domain 变异？
- 固定 probe 规则为何在研究假设冻结前选择四个最深 block？是否看过深度与 turnover/损伤的先验趋势？重加权 Spearman 的权重和秩处理如何定义，为什么它能纠正未观测 block 的选择偏差？
- 能否量化同一模型相邻 block 间的依赖，并说明把 block 当可交换 bootstrap 单位时区间具有何种有效解释？
- 主文称 masked partial 不能区分 turnover 与 depth；在完全测量全部28/36个 block、多个 checkpoint 后，作者将用什么预注册模型或分层统计量检验增量预测价值？
- 预注册8个 cell 的完整定义、冻结时间与未执行原因是什么？剩余量化器、shift cells 和 fallback forward 是否会作为同一确认性 family 完成，还是需要重新预注册？

**能提高评分的证据：**

- 在多个独立 checkpoint、模型 family、量化器和真实 prompt shift 上完成全层或事前均匀层采样，并用 checkpoint 为高层单位复现 turnover 的增量预测价值。
- 相对仅深度及其他离线敏感度基线，turnover 在严格 held-out checkpoint/layer 上稳定改善预注册指标。
- 完成全部预注册 cells 和实际 fallback gain 测量，且 AUROC/gain 判定对标签噪声和单层变化稳定。
- 清理 masked/unmasked 数值版本，并由单源脚本自动证明摘要、表格、图和附录完全一致。

**会降低评分的证据：**

- 全层或均匀层采样后4B相关消失，或 turnover 不优于深度基线。
- 在新增 checkpoint/domain 上关联方向不稳定，表明当前读数由单一 checkpoint 或模板特征驱动。
- 无法解释或复现表2与表3的数值版本差异，或 masked payload 与主文不匹配。
- 完成注册门后 AUROC 与 fallback gain 均稳定落入 KILL 区间。

**伦理标记：** 否。冻结 PDF 研究公开模型上的量化测量，未涉及人体参与者或个人数据；没有发现需要单独伦理升级的实验，但“安全触发”措辞必须继续维持当前非部署性边界。

**LLM 使用披露：** 本审稿由全新、隔离的 AI 子代理 R3 生成，仅用于内部投稿前质量控制；该子代理只读取指定的冻结 PDF、审稿协议、量表与 JSON schema，未与任何其他评审通信，也未接触作者计划、历史评审、目标分数或版本历史。

**评审限制：**

- 按隔离要求只读取冻结 PDF，未打开论文所列 JSON、代码、匿名 manifest 或模型输出，因此无法重算 masked/unmasked 版本或确认三主机 payload 哈希。
- 按任务要求未联网，2025–2026文献、Qwen3/HQQ实现细节和引用支持未作外部核验；相关工作判断仅依据 PDF。
- PDF 共12页，文本抽取完整且逐页视觉核查未发现影响内容读取的解析故障。

### S1

#### 交付与来源状态

- 选定输入：`/Users/liuhanzuo/Downloads/S1-paper (1).pdf`
- 源状态：`exact_frozen_latex_snapshot`
- 润色 PDF：`/Users/liuhanzuo/MacLLM-Bench/output/pdf/paper_polish_20260826/S1-polished.pdf`
- 冻结 SHA-256：`a70d0593930038db805ed9ae99c770ea52d4f7efec5155f1a4689d3b61482ac7`
- 总页数：16；主文状态：主文不超过9页。
- 版面核验：pass；构建：pass。
- 旧评分基线：2,4,4；旧中位数：4。

#### 三评量化结果

| 审稿人 | 角色 | Overall | Confidence | 建议 | Soundness | Presentation | Contribution | 当前上限 | 必需修改后预测 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| R1 | 新颖性与定位 | 2 | 4 | 拒绝 | 2 | 2 | 1 | 4 | 6 |
| R2 | 技术正确性 | 2 | 4 | 拒绝 | 1 | 2 | 1 | 2 | 6 |
| R3 | 实验严谨性 | 2 | 4 | 拒绝 | 2 | 2 | 1 | 2 | 6 |

三评中位数为 **2**，均值 2.00，跨度 0，接收侧票数 0/3。

#### 编辑记录

- [结构审计](work/S1/structure_audit.md)
- [语义锁](work/S1/semantic_lock.md)
- [修订日志](work/S1/revision_log.md)
- [待核验事项](work/S1/needs_verification.md)

**修订日志原文：**

> # S1 Revision Log
>
> ## Editorial scope
>
> Evidence-preserving polish of the exact frozen manuscript source corresponding to the supplied PDF. Numerical endpoints, equations, tables, figures, citations, labels, code identifiers, hashes, and archived evidence status were locked before editing.
>
> ## Structural changes
>
> - Reframed the title and opening around the directly observed 0.118 pp post-recovery spread.
> - Compressed the abstract, introduction, contribution list, and conclusion.
> - Split the limitations into evidence, mechanism, export/provenance, and external-validity boundaries.
> - Kept detailed archived analyses in the appendix rather than expanding the main claim.
>
> ## Language and claim changes
>
> - Removed “registered negative result” wording because the manuscript lacks paired predictions, a joint uncertainty estimate, and an equivalence margin.
> - Replaced an unsupported literature-completeness claim with a narrower statement about what the cited and archived materials establish.
> - Kept the 0.52 pp comparison explicitly marginal and per-checkpoint; it is not a pairwise resolution threshold.
> - Presented curvature scoring, annealing, and SLoRB as pipeline background, not isolated causal contributions.
> - Clarified that no folded/reprojected exact-2:4 deployment export was produced.
>
> ## Preserved scientific content
>
> - The ALPS, ELSA-4096, and ProxSparse supports; 624,951,296-token recovery; 2.675 pp and 0.118 pp ranges; 0.52 pp reference; archived 58.47 Avg-9 endpoint; +0.528 pp margin; p=0.47; and all appendix records are unchanged.
> - No new experiment, equivalence conclusion, mechanism claim, citation, or deployment result was added.
>
> ## Build and QA
>
> - Built twice with `pdflatex` through `latexmk` and shell escape disabled.
> - Citation-key, label, and displayed-equation sets match the untouched frozen source exactly.
> - The final log contains no LaTeX errors, unresolved citations/references, fatal errors, or overfull boxes.
> - Extracted the final text and rendered all sixteen pages; no clipping, overlap, or missing figure/table was observed.

**待核验事项原文：**

> # S1 Needs Verification
>
> - Per-item predictions for the three recovered endpoints were not retained, so joint uncertainty, paired differences, equivalence, and residual support distinguishability cannot be recovered from the archived summaries.
> - The 0.52 pp value is a marginal per-checkpoint binomial standard error, not a pairwise or joint resolution threshold.
> - The archived corpus/checkpoint manifest is unavailable in the accessible workspace; corpus identity and provenance remain unverifiable.
> - No matched-budget 625M SparseForge endpoint or component ablation isolates curvature scoring, annealing, or SLoRB.
> - No folded and reprojected exact-2:4 export was produced or evaluated; deployment viability is therefore unresolved.
> - Bibliographic metadata and archived artifact availability were preserved but not independently verified during this editorial pass.

#### R1（新颖性与定位）完整评议

**论文概述：** 论文研究 exact 2:4 LLM 稀疏化中，不同固定 support 在同样 625M-token joint sparse-weight + rank-16 SLoRB recovery 后，AVG-9 点估计差异是否缩小。ALPS、ELSA-4096、ProxSparse 三种 native support 的 range 为 2.675 pp，恢复后为 0.118 pp。由于没有保留逐 item 预测，论文无法计算配对或 joint range uncertainty，也没有多 seed recovery 方差，因此将结果限定为 archived descriptive observation。SparseForge 的 curvature score、mask annealing 和 SLoRB pipeline 仅作为背景，未做 matched-budget ablation；另一个 5B checkpoint 保留 live SLoRB branch，没有 fold/reprojection 的 exact-2:4 export，训练 corpus manifest 与 checkpoint 又不可访问，只作为历史背景。

**最强的已核实贡献：** 最强的已验证贡献是第 11 页 Table 5 所记录的、在同一九任务 harness 下对三个具名固定 support 应用相同 625M-token recovery 后，AVG-9 点估计 range 从 2.675 pp 缩到 0.118 pp；论文同时正确指出单 checkpoint 的 0.52 pp marginal SE 不是 pairwise/joint resolution limit，不能据此宣称等价。

**维度理由：**

- Soundness：从三个 aggregate CSV endpoint 计算 native 与 recovered 点估计 range 的算术观察是清楚的，作者也反复声明它不是等价性或显著性结论。但缺少逐 item 配对预测、joint max-min uncertainty 和 recovery-run 方差，无法判断 0.118 pp 是真实 support compression、评估采样噪声还是单次训练波动。SparseForge 相关结果又是 branch-active training state，而非最终 exact-2:4 export。
- Presentation：主张边界披露充分，Tables 1、5 和 Figure 2 让主要数字可追踪；但论文同时容纳一个很窄的 archived range 观察、一个明确未验证的 SparseForge pipeline、不可访问的 5B checkpoint、RTE 异常、历史跨家族记录和一个未证 conjecture，故事严重分散。Figure 1 的关键注释低于正文尺度。
- Contribution：论文明确不主张新算法优势、机制、统计等价或可部署 checkpoint。剩余贡献只是三个单次 archived endpoint 的点估计 range 从 2.675 pp 变为 0.118 pp，并指出未来应保留 per-item predictions。这个事实可作为内部测量警示，但在当前证据下不足以构成 ICLR 级的新知识或能力。

**优点：**

- 摘要、第 5-7 页和结论反复区分 point-estimate compression、统计不可分辨与等价性，避免把 0.118 pp 误写成显著的 support 等价。
- 第 5 页 Table 1 统一报告九任务与 WikiText-2 指标，第 10-11 页明确说明为何 aggregate counts 不能恢复 McNemar discordant pairs。
- 第 4、7、15-16 页明确披露 SLoRB branch 仍在推理、fold/reprojection 未执行、groupwise exact-2:4 assertion 未接入、5B corpus/checkpoint 不可访问。
- 第 6 页 Figure 2 揭示 +0.53 pp AVG-9 掩盖 RTE -15.52 pp，作者没有把混合任务结果包装成广泛优越性。
- Related Work 覆盖 one-shot pruning、semi-structured mask learning、AST/CAST 和 fixed-support recovery 的基本邻域。

**问题与可验证修复：**

##### S1-R1-01 · 主要 · 技术正确性、实验严谨性、重要性

- 位置：PDF 第 1-2 页摘要/贡献、第 5-7 页 §5.2-5.3，以及第 10-11 页 Appendix A.1-A.2/Table 5
- 观察证据：核心结果只有三个 native aggregate 与三个单次 recovered aggregate；per-item predictions 未保留，无法得到 checkpoint 间协方差、paired tests 或 max-min joint interval。0.52 pp 只是单 checkpoint marginal binomial SE，且没有 recovery training-run variance。
- 重要性：从 2.675 到 0.118 的点估计变化不等于 underlying support differences 被 recovery 压缩。range 是选择统计量，可能受评估噪声、协方差和单次训练随机性强烈影响；没有不确定性就无法判断观察是否超出噪声。
- 必需修复：重新评估并保存每 item 预测，使用配对 bootstrap/多任务 joint procedure 为 pre/post range 和所有 support contrasts 给出区间；每个 support 还需多个独立 recovery seed。
- 验证标准：预注册的 joint interval 显示 post-recovery range 相对 pre-recovery 有实质收缩，且该结论在独立 recovery seeds 上复现；同时报告失败或 rank flip。
- 仍需证据：逐 item predictions、多个 recovery seed、joint range/contrast analysis。
- 预期影响：high；判断置信度：high。

##### S1-R1-02 · 主要 · 技术正确性、重要性、限制与负责任表述、可复现性

- 位置：PDF 第 1-2 页、 第 4-5 页 §4.3-5.1/Table 1，以及第 14-15 页 Appendix C.4-C.5
- 观察证据：所有 SparseForge/625M recovered 评估均保持 rank-16 SLoRB branch live；frozen finalizer 没有把 BP fold 入权重，也没有 fold 后再做 exact 2:4 projection，甚至 groupwise sum=2 assertion 尚未接入 validator。
- 重要性：有效推理模型包含一个稠密低秩支路，因而不是论文标题容易让人理解的原生、可部署 exact-2:4 模型。support compression 可能由额外稠密容量吸收差异，而不是 2:4 support 本身变得不重要。
- 必需修复：完成 branch fold、exact-2:4 reprojection、逐组验证和 pre/post-export 质量/支持变化测量；同时报告额外分支的参数、FLOPs、内存和延迟。
- 验证标准：导出模型在每个 group 上严格保留 2/4，推理不再调用 SLoRB branch，且三个 support 的 post-export range 与绝对质量在预先规定容差内保留。
- 仍需证据：可下载 export、自动 group assertion、pre/post metrics 与系统成本。
- 预期影响：high；判断置信度：high。

##### S1-R1-03 · 主要 · 新颖性、重要性、实验严谨性

- 位置：PDF 第 1-4 页 §1-4 与第 7 页 §6 Budget and mechanism scope
- 观察证据：论文用大量篇幅描述 curvature score、continuous mask、annealing、quenching 和 SLoRB，但明确说三者从未在 matched budget 下隔离，fixed-support recovery launcher 甚至设置 change_mask=False；没有有效的 625M SparseForge matched-budget endpoint。
- 重要性：因此不能把任何增益或 range compression 归因于 SparseForge 算法组件，也无法判断其相对 AST/CAST 或简单 fixed-mask healing 的新颖贡献。方法章节成为未验证背景，而非论文证据。
- 必需修复：要么执行 matched-budget component ablations 与强基线，要么删除/大幅压缩 SparseForge 方法主线，把投稿明确改为固定 support recovery 的测量说明。
- 验证标准：每个核心机制至少有一个同 token、同数据、同 branch 容量的消融，且主张只覆盖显著/稳定的差异；若不运行，标题、摘要和结论不得暗示算法贡献。
- 仍需证据：matched-budget SparseForge endpoint、component ablations 与重复 seed。
- 预期影响：high；判断置信度：high。

##### S1-R1-04 · 主要 · 可复现性、引文完整性、限制与负责任表述、重要性

- 位置：PDF 第 4-7 页 §5.1-5.4/Table 1-2/Figure 2 与第 15-16 页 Appendix C.6-C.7
- 观察证据：5B headline checkpoint、args.json 和 recovery-corpus tokenization 位于不可访问 archive host；此前 corpus 归因互相冲突而被撤回。RTE 注册值 49.82% 低于 majority baseline，原因未通过 prompt/label ablation 解析；不存在该 checkpoint 的 exact-2:4 export。
- 重要性：该比较不可独立复现，数据治理和许可无法审计，且一个极端任务退化会显著影响 aggregate。即使作为 supporting context，它也降低论文可信度并模糊中央 625M 观察。
- 必需修复：在可访问、来源清楚的 corpus 上重新训练/评估可导出的 checkpoint，或从科学证据与 headline tables 中删除不可验证的 5B 结果，仅保留为非证据历史记录。
- 验证标准：匿名审稿人可下载 checkpoint、manifest 与 corpus provenance，复算九任务和 export；RTE 异常有预注册诊断。若做不到，主文不再使用该行作任何比较。
- 仍需证据：可访问 checkpoint/manifest、数据许可证与来源、RTE 重跑、exact export。
- 预期影响：high；判断置信度：high。

##### S1-R1-05 · 主要 · 新颖性、重要性

- 位置：PDF 第 2 页 §2 Related Work 与第 1-2 页贡献陈述
- 观察证据：论文声称现有比较没有固定并标注多个 native supports、施加同一 recovery recipe 再测 cross-support spread，但只提供单模型、三 support、单次 archived endpoints，且没有统计解析。
- 重要性：即便该表格组合在 PDF 引用中未出现，单个未解析观察也难以形成新的普遍知识。新颖性不能仅以“此前没有完全相同表格”成立，还需说明它改变了何种方法选择或理论理解。
- 必需修复：把 closest-work 比较扩展为支持选择×恢复策略矩阵，并通过多模型/多预算实验说明何时 support identity 会或不会被 recovery 吸收；明确可证伪的决策含义。
- 验证标准：至少两个模型/预算上复现预先定义的 compression pattern，或发现并解释反例；相关工作逐项说明与 AST/CAST/MaskLLM/固定 mask healing 的差异。
- 仍需证据：经核验的 closest-work matrix 和跨设置实证。
- 预期影响：high；判断置信度：medium。

##### S1-R1-06 · 次要 · 技术正确性、清晰度

- 位置：PDF 第 11-12 页 Appendix A.2，Conjecture (C1)
- 观察证据：稿件引入未拟合的 attractor ϕ_s、transient g_s(T) 和 recovery-invariant subspace，并仅能证明 range 的极限是 intercept range；没有多时间点、多 seed 或 intercept 估计支持“iff” conjecture。
- 重要性：这一段不能解释当前单一 T 的三个点，反而给描述性结果附加未经验证的机制语言，并显著增加篇幅与认知负担。
- 必需修复：删除该 conjecture，或把它转为有独立数据和可证伪预测的正式模型。
- 验证标准：若保留，至少三个恢复预算上的 trajectory 能估计 ϕ_s/g_s(T)，并预先检验 shared-attractor 与 distinct-intercept 假设；否则全文不使用 recovery-invariant-subspace 术语。
- 仍需证据：多预算恢复轨迹和模型检验；当前证据不足。
- 预期影响：medium；判断置信度：high。

##### S1-R1-07 · 次要 · 清晰度

- 位置：PDF 第 2 页 Figure 1 与第 13-15 页 Appendix C
- 观察证据：Figure 1 caption 自己承认 in-figure annotations 低于正文尺度；页面渲染中这些标签确实难以在单栏打印尺寸阅读。附录算法细节又占据大量版面，而核心统计证据很少。
- 重要性：视觉层级强化了一个未验证 pipeline，而弱化真正的 archived measurement，影响论文身份与可读性。
- 必需修复：简化 Figure 1、提高所有标签字号，并把未用于解释结果的实现细节移至工件文档。
- 验证标准：在 100% 单栏打印尺寸下所有关键标签可读，且主文篇幅主要服务于核心 estimand、证据和不确定性。
- 仍需证据：无需新实验，仅需版式与结构修订。
- 预期影响：low；判断置信度：high。

**给作者的问题：**

- 能否重新运行三个 recovered checkpoints 的九任务推理并保存逐 item predictions，以对每个 task 做配对检验并对 max-min range 构造 joint interval？
- 三个 625M recovery endpoint 是否各有独立训练 seed？如果只有 base seed 0，range compression 在 recovery-run 方差下是否复现？
- joint SLoRB branch 在推理时增加的稠密参数、FLOPs 与延迟是多少？在 fold + exact-2:4 reprojection 后，0.118 pp range 和绝对质量是否保留？
- 既然 curvature、annealing 与 SLoRB 从未在 matched budget 下隔离，SparseForge pipeline 相对于 AST/CAST/SLoRB 的算法新颖性究竟是什么，还是应从本稿完全移除？
- 中央 625M 三个 endpoint 的训练 corpus、checkpoint、launcher 和原始日志是否均可匿名访问？请把它们与不可访问的 5B archive 明确分开。
- 第 11-12 页的 asymptotic attractor/recovery-invariant-subspace conjecture 对当前三点观察提供了什么可检验增量？若无运行轨迹或截距估计，为何不删除？
- 5B corpus 无法审计时，作者能否确认其许可、隐私与数据治理状态；若不能，为何保留该 checkpoint 的比较结果？

**能提高评分的证据：**

- 三个或更多 support 的逐 item 配对评估与 joint range interval 证实 recovery compression，并在多个 recovery seed 上复现。
- fold + exact-2:4 reprojection 后仍保留质量与窄 range，且报告真实硬件吞吐/内存收益。
- matched-budget SparseForge component ablations 显示 curvature、annealing 或 SLoRB 中至少一项具有独立增量。
- 可访问且许可清楚的 checkpoint/corpus manifest 取代不可验证的 5B archive。
- 跨模型与恢复预算结果给出 support identity 被吸收或保留的可证伪规律。

**会降低评分的证据：**

- paired/joint 重评后 0.118 pp range 与更大真实 support 差异相容，无法支持 compression。
- 不同 recovery seed 的 post range 大幅变化或恢复 rank 不稳定。
- SLoRB fold/reprojection 导致质量显著下降或 support spread 重新扩大。
- 中央 625M endpoint、训练语料或 CSV 也无法从匿名工件复现。
- RTE 等任务异常在重跑中显示 harness 配置或数据污染问题。

**伦理标记：** 是。第 8、15-16 页明确说明 5B recovery corpus 的 manifest 与 checkpoint 不可访问，早期 corpus 归因互相冲突，因而许可、隐私和数据治理无法审计。论文已披露此事，但若继续把 5B checkpoint 作为科学比较，作者需恢复可核验 provenance；否则应删除该证据。

**LLM 使用披露：** 本审稿由一个全新、隔离的 AI 子代理生成，仅用于内部投稿前质量控制；该子代理未与任何其他评审通信，也未接触作者计划、旧评审或目标分数。

**评审限制：**

- 按任务约束未联网检索或核验稀疏化文献；新颖性只相对于 PDF 的 Related Work 评价。
- 仅审阅冻结 PDF，未访问匿名仓库、aggregate CSV、checkpoint、训练日志或逐 item predictions，无法独立确认中央 endpoints 与 provenance。

#### R2（技术正确性）完整评议

**论文概述：** 论文汇总 LLaMA-2-7B exact-2:4 稀疏化的归档实验。ALPS、ELSA-4096、ProxSparse 三种支持在原生 AVG-9 上的范围为2.675个百分点，经相同标称625M token 的稀疏权重加 rank-16 SLoRB 恢复后，三个端点范围为0.118个百分点。由于未保留逐项预测，作者将其限定为描述性观察而非等价性结论。一个5B SparseForge 检查点被作为背景：其恢复语料清单与权重不可访问，SLoRB 分支仍在推理中，尚未折叠并重新投影成可部署 exact-2:4。论文还给出未隔离的曲率/退火管线、一个发散的 ELSA 长预算运行和多项历史记录。

**最强的已核实贡献：** 就 PDF 中呈现的聚合数值而言，三条标称625M恢复行的 AVG-9 分别为55.82、55.94、55.91，所显示范围确为约0.12个百分点，且论文没有把这个小范围错误宣称为统计等价；它清楚指出需要逐项配对和联合 max-min 不确定性才能比较残余差异。

**维度理由：**

- Soundness：中心结论是三个支持在恢复前后的 AVG-9 range 从2.675收缩到0.118，但表1把 ALPS/ELSA 的原生值列为三种子均值、ProxSparse 为官方单点，而恢复后是各支持的单个恢复端点；论文没有证明这些原生数值就是每条恢复运行的精确起点。即使算术无误，缺少匹配起终点、训练重复和逐项配对输出，使“同一恢复压缩支持差异”的核心解释不可验证。附加的未配对 z 检验还对同一测试样本错误采用独立比例近似。
- Presentation：论文非常坦率地罗列数据、导出和机制边界，主表也包含大量原始数值；但中心贡献被大量未验证的归档材料、未执行的导出流程和一个未证明的渐近猜想淹没。图1文字低于正文可读字号，附录与正文反复强调不可验证工件，整体更像审计记录而非完成的会议论文。
- Contribution：在当前证据下，可核实的新知识主要是一组三个归档点估计的数值范围变化；论文自己承认不能做等价性、机制、样本效率、部署或跨模型结论。没有配对不确定性、没有有效的 SparseForge 625M 对照、没有可部署 exact-2:4 导出，尚不足以构成 ICLR 级贡献。

**优点：**

- 对5B检查点的语料不可验证、SLoRB分支仍活跃、未执行fold/reprojection、组件未隔离等问题作了明确披露。
- 没有把0.52个百分点的单检查点边际标准误称为等价界、检测下限或联合range置信界。
- 表1给出九个任务和困惑度的完整同地评测，图2揭示平均值掩盖了RTE的巨大反向变化。
- 把长预算 ELSA 运行的发散限定为该恢复配方失败，而不是 ELSA 方法普遍失败。

**问题与可验证修复：**

##### S1-R2-001 · 致命 · 技术正确性、实验严谨性、可复现性

- 位置：第5页表1及§5.2、第10-11页附录A.2表5
- 观察证据：表1明确说 ALPS 和 ELSA 原生行是三种子均值，ProxSparse 是 official 单点；恢复后每个支持只有一个625M端点。表5直接把这些异质聚合的 native 数值与单个 recovered 数值并列，却没有给出每条恢复运行的精确起点 checkpoint、mask 哈希或逐运行 before 值。
- 重要性：若恢复行并非从表5所列的精确原生端点开始，2.675→0.118不是同三个实验单位的配对range变化，可能由种子平均、起点选择或支持实例差异造成。该缺口直接决定论文唯一的 load-bearing 结论是否成立。
- 必需修复：为每个支持提供同一 mask/weight 实例的0-token和625M配对评测、不可变哈希与运行映射；更理想的是对多个独立支持/训练种子执行匹配恢复，并把支持和训练随机性分层。
- 验证标准：逐支持从已哈希的0-token起点启动冻结625M流程，保存 pre/post 逐项预测；核对三条恢复的起点值确实形成报告的 native range，并在独立种子上重复 range 收缩。
- 仍需证据：逐运行起点/终点哈希、支持映射、0-token同实例评测和多种子恢复记录。
- 预期影响：high；判断置信度：high。

##### S1-R2-002 · 主要 · 技术正确性、实验严谨性、限制与负责任表述

- 位置：第1-2页摘要与贡献、第5页§5.2、第7页局限、第10-11页附录A.1-A.2
- 观察证据：0.118范围只有三个归档端点；没有训练重复、逐项预测或支持间联合协方差。0.52是单个固定检查点的边际二项标准误，不是两个检查点差异或max-min选择后的不确定性。论文虽反复披露这一点，标题、摘要和中心段落仍突出“低于0.52噪声尺度”。
- 重要性：在联合不确定性未知时，小于边际SE既不能说明残余差异不可分，也不能证明恢复真正压缩了潜在支持效应；它只是对三组有抽样误差点估计的描述。没有训练方差时，收缩也可能不稳定。
- 必需修复：获取逐项预测并对预先定义的range或支持对比做配对联合推断；增加多个独立恢复种子，把评价抽样和训练方差分开。若无法取得，应删除噪声尺度比较，只保留三点算术事实。
- 验证标准：对每个种子用相同评价项目计算支持间配对差，进行层级/多重比较校正的联合 bootstrap 或预注册等价检验；报告range收缩在种子间的区间。
- 仍需证据：逐项预测、独立训练种子和预定义联合推断协议。
- 预期影响：high；判断置信度：high。

##### S1-R2-003 · 主要 · 重要性、技术正确性、可复现性、限制与负责任表述

- 位置：第1页§1、第4-5页§4.3-5.1与表1标题、第7页局限、第13-15页附录C.1-C.5
- 观察证据：所有 SparseForge/恢复评测都保留 rank-16 SLoRB 分支参与推理；冻结 finalizer 不把 BP 折叠进权重，也未执行fold后的2:4重投影、组和断言或pre/post-export质量检查。论文没有对应检查点的吞吐/内存测量。
- 重要性：训练状态的稀疏基权重加稠密低秩分支不等同于已部署的原生 exact-2:4模型。支持差异可能正被共同分支补偿，且折叠/重投影可能改变支持、质量和系统收益，因此当前证据不能回答部署层面的半结构化稀疏化问题。
- 必需修复：实现并冻结 SLoRB fold→exact-2:4 reprojection，加入逐组2/4断言，报告分支活跃、折叠前、折叠后三种状态的 AVG-9/PPL、支持变化、吞吐和内存。
- 验证标准：对每个恢复支持执行 EXT-M3，逐层验证每组恰有两个非零，比较导出前后逐项预测与质量，并在目标GPU上测量系统指标。
- 仍需证据：可用权重/分支参数、导出器、支持验证日志、pre/post-export评测和系统基准。
- 预期影响：high；判断置信度：high。

##### S1-R2-004 · 主要 · 技术正确性、实验严谨性

- 位置：第10页附录A.1行524-534、第11页表4
- 观察证据：SparseForge与AST在同一任务的同一评价项目上预测，天然是配对数据；因逐项日志丢失，论文改用两个独立二项比例的未合并 z 检验，并据此报告四个未校正显著、两个Holm显著。
- 重要性：同项目预测的协方差未知，独立比例方差假设没有依据。仅凭两个边际正确数无法唯一确定McNemar方差，因此表4的p值不是当前设计下有效的配对推断，可能过保守也可能反保守。
- 必需修复：删除这些显著性结论，或明确标为假设独立的描述性近似并给出在所有可行discordance计数下的敏感性界；最终应重新保存逐项预测并使用McNemar/配对bootstrap。
- 验证标准：枚举与两个边际计数一致的可行(b,c)，展示p值范围；在新评测中用真实discordant pair重算并比较表4结论。
- 仍需证据：逐项预测或基于可行配对表的严格界。
- 预期影响：medium；判断置信度：high。

##### S1-R2-005 · 主要 · 可复现性、引文完整性、限制与负责任表述

- 位置：第4-7页§5.1-6、第15-16页附录C.6-C.7与D-E
- 观察证据：5B headline checkpoint、args.json、恢复语料tokenization和分支参数都在不可访问主机上，本地无副本；早期语料归因互相冲突后被撤销。lm-eval只给下限版本，历史跨家族和系统结果缺失完整套件/检查点映射。
- 重要性：5B对AST的比较、语料归因和任何导出复现都无法独立审计。虽然被降为背景，它仍占据表1、图2和大量论述，并可能误导读者把不可重现归档点当作方法证据。未知语料还带来许可和数据治理不确定性。
- 必需修复：移除无法核验的5B/历史结果出中心叙事，或恢复完整匿名权重、manifest、精确软件环境、语料来源和运行映射；重新执行可访问的匹配实验。
- 验证标准：第三方从匿名包定位所有输入，重建5B评测与导出；核对语料许可/来源、checkpoint哈希和表1/图2数值。无法通过则只保留为非证据历史注记。
- 仍需证据：可访问检查点、语料manifest、许可信息、精确环境锁和端到端重现。
- 预期影响：high；判断置信度：high。

##### S1-R2-006 · 次要 · 清晰度、技术正确性

- 位置：第11-12页附录A.2的Conjecture (C1)
- 观察证据：所谓猜想先假设每个支持收敛到φ_s且瞬态g_s(T)趋零，再得到range趋向φ_s的range；随后关于“recovery-invariant subspace”的iff陈述没有定义可检验变换、没有证明，也没有由三端点数据识别任何φ_s。
- 重要性：目前可证明部分近乎由假设直接得到，其余部分不可证伪，不能解释已观察的有限预算压缩，反而模糊了论文仅有的描述性结论。
- 必需修复：删除该猜想，或给出形式化定义、非平凡命题与证明，并设计多预算轨迹来估计/反驳不同支持的渐近截距。
- 验证标准：在至少三个预注册恢复预算和多个种子上拟合预先规定的收敛模型，检验共享/不同截距预测；同时对命题进行形式证明或给出反例边界。
- 仍需证据：多预算配对轨迹、可识别模型和完整证明。
- 预期影响：low；判断置信度：high。

**给作者的问题：**

- 表5中每个 native AVG-9 是否来自随后进入对应625M恢复运行的同一个具体 mask/weight checkpoint？若 ALPS/ELSA 是三种子均值、ProxSparse 是官方单点，如何构成逐支持匹配的 before/after range？
- 三条625M恢复运行使用的确切恢复语料、样本顺序、初始支持哈希和起点权重是什么？为何这些信息未在表6或工件声明中固定？
- SLoRB分支在推理时保持活跃后，有效算子和内存/延迟开销是多少？在未fold且未重新投影时，为什么主表标题仍称所有结果为 exact 2:4 sparsity？
- 在同一评价样本上比较两个检查点时，为什么使用独立两比例 z 检验？在未知配对协方差下，作者能否给出合法的最坏/最好方差界，而不是单一 p 值？
- 如果5B检查点、语料清单和分支参数均不可访问，哪些核心结果能够由匿名工件从头独立重现？

**能提高评分的证据：**

- 同一具体支持实例的0-token/625M配对起终点、多独立恢复种子和逐项预测，给出联合range不确定性。
- 完成SLoRB fold、exact-2:4重投影与逐组验证，并证明导出前后质量和系统收益。
- 可访问且许可清楚的恢复语料manifest、检查点和精确环境，使核心结果可端到端复现。
- 有效的SparseForge 625M匹配预算对照及组件消融，能够支持方法而非归档端点叙事。

**会降低评分的证据：**

- 恢复运行的真实0-token起点不对应表5 native行，导致2.675→0.118配对压缩不成立。
- 多种子或逐项联合分析显示0.118范围不稳定，或支持差异仍可可靠区分。
- fold/reprojection后质量显著下降、支持大幅改变或系统收益消失。
- 恢复语料来源存在许可/隐私问题，或5B数值无法从可访问工件复现。

**伦理标记：** 是。归档5B检查点的恢复语料身份、tokenization清单与许可均不可访问，论文也承认早期语料归因互相冲突；因此无法审计数据治理、版权/许可、隐私或内容风险。该检查点目前只是背景，但只要保留其结果，就应把未知语料来源作为明确伦理与可复现性风险。

**LLM 使用披露：** 本审稿由一个全新、隔离的 AI 子代理生成，仅用于内部投稿前质量控制；子代理只接收冻结 PDF、指定的 ICLR 2026 量表与输出 schema，未与任何其他评审通信，也未查看作者计划、旧评审或目标分数。

**评审限制：**

- 仅审阅了指定冻结 PDF；按隔离要求未访问源代码、CSV、匿名镜像、归档主机或任何作者侧材料，因此无法独立重算表格或验证检查点/支持映射。
- 按任务要求未联网，未核验2025-2026文献、语料许可、匿名工件可访问性或外部系统基准。
- 由于逐项预测、运行映射和5B检查点在审稿快照中不可用，部分关键判断必须标为无法外部验证；本审稿针对论文当前可见证据而非潜在隐藏工件。

#### R3（实验严谨性）完整评议

**论文概述：** 论文汇总 LLaMA-2-7B exact-2:4 稀疏化的归档结果。三个命名 support（ALPS、ELSA-4096、ProxSparse）在 native AVG-9 点估计上的范围为2.675个百分点，经过相同625M-token 的 sparse-weight+rank-16 SLoRB 恢复后，三个 recovered AVG-9 点估计范围为0.118个百分点。因为没有保留 per-item 预测、没有恢复训练的多 seed、也无法构造 joint max-minus-min 不确定性，作者把它定位为描述性归档观察而非等价性结论。论文另描述 SparseForge 的曲率/软硬 mask/SLoRB 流程，但这些组件未在匹配预算下隔离；5B checkpoint 保持 SLoRB 分支活跃、没有 fold/reprojection 导出，checkpoint 与恢复语料 manifest 均不可访问。与 AST 的+0.53 pp聚合差在采样噪声内且任务级高度混合，RTE 还存在历史日志冲突。

**最强的已核实贡献：** 最可信的贡献不是算法优势，而是一个负面的测量审计：第5、10–11页明确说明0.52 pp只是单 checkpoint 的边际二项 SE，不能充当三 support 范围的联合分辨率；没有逐 item 配对输出就不能宣称 recovered supports 等价。这一限制陈述是正确且有用的。

**维度理由：**

- Soundness：0.118 pp 是三行聚合点估计的可复算算术范围，作者也正确承认缺少联合/配对不确定性；但中心的“同一三个固定 support 前后恢复压缩”叙事没有清楚对齐实验单位：native ALPS/ELSA 是三 seed 均值、ProxSparse 是官方单点，而 recovered 行看起来是单个 seed=0 端点。没有逐 support 的同一初始 checkpoint 映射、恢复重复或 per-item 输出，无法区分真实收敛、训练随机性、回归均值与评测噪声。
- Presentation：论文在诚实披露历史记录缺口方面很强，但16页内容将一个非常有限的描述性范围，与未验证的5B checkpoint、未执行导出、未隔离算法、猜想、旧吞吐记录及大量代码细节混在一起。图1注释小于正文且难读，主贡献与 SparseForge 方法叙事明显失衡。
- Contribution：当前唯一负载贡献是三个归档聚合端点的范围从2.675降到0.118 pp，但既无可辨识的不确定性，也无 matched multi-seed/paired 复验。SparseForge 本身没有有效625M matched endpoint、组件消融或可部署 exact-2:4 导出；5B checkpoint 与语料 manifest 又不可访问。因此尚无足够的新方法或可靠经验知识达到 ICLR 门槛。

**优点：**

- 作者没有把0.118 pp小于0.52 pp误写成等价或统计阴性结果，并解释了 paired McNemar/joint range 所需的缺失数据。
- 第5–7页完整展示九任务数值与 RTE 异常，不以 AVG-9 掩盖 RTE 的-15.52 pp或早期69.82日志冲突。
- 第7、13–16页明确披露5B checkpoint/语料不可访问、SLoRB 分支仍活跃、fold/reprojection 未执行、exact group-sum validator 缺失和版本下界未完全固定。
- 附录提供了较详细的训练流程、mask 更新、Hutchinson 估计、tie rule 和环境边界，便于理解归档实现。
- 伦理段落主动指出不可核验恢复语料的治理、许可、隐私和计算成本风险。

**问题与可验证修复：**

##### S1-R3-01 · 致命 · 技术正确性、实验严谨性、可复现性

- 位置：第5页表1与§5.2；第11页表5；第13页表6
- 观察证据：表1明确称 ALPS 与 ELSA native 行为三 seed 均值，ProxSparse 为 official 单点；表5却把这三个 native 聚合值与三个 recovered 端点并列为“fixed supports”的0-token/625M前后结果。表6只给625M arm 的 base seed 0，PDF 没有逐 recovered 行到某个确切 native mask/checkpoint/seed 的映射。
- 重要性：如果 before 是跨 seed 方法均值，而 after 是单个固定 mask 的恢复结果，那么2.675→0.118不是同一三个实验单位的配对前后变化，中心的“identical recovery compresses fixed-support spread”解释就不成立。即使数字抄录无误，也可能混合方法均值、support 选择和恢复随机性。
- 必需修复：为每个 recovered 行提供同一初始模型、确切 mask/support SHA 和 seed 的 native before 值，再从该 checkpoint 用完全相同恢复协议得到 after；至少跨多个独立恢复 seeds 重复。若无法恢复映射，应删除配对压缩措辞，只报告两组不可配对的历史横截面范围。
- 验证标准：匿名工件中每个 support 都有唯一 before checkpoint/mask digest、after digest、恢复配置和日志；脚本逐 pair 重算范围，且在预注册的多 seed 分层分析中恢复后的范围收缩稳定出现。
- 仍需证据：逐 support/seed 的 checkpoint 与 mask SHA、before/after per-task predictions、恢复日志和多 seed 重复。
- 预期影响：high；判断置信度：high。

##### S1-R3-02 · 主要 · 实验严谨性、技术正确性、限制与负责任表述

- 位置：第1、5–7、10–11页，摘要、§5.2–5.3、附录A.1–A.2、表3–5
- 观察证据：中心结果只有三个 recovered 聚合点，无 per-item predictions、无训练重复、无 joint max-minus-min 区间。0.118 pp 被与单 checkpoint AVG-9 的0.52 pp边际二项 SE 比较；作者承认该 SE 不是 pairwise/joint 界，也拒绝给 compression ratio 或等价结论。
- 重要性：range 是经最大/最小选择后的联合统计量，受评测协方差、任务加权和训练方差共同影响；边际 SE 不能说明0.118是否可分辨或2.675→0.118是否超出随机变化。当前证据只支持算术记录，无法支持恢复使 supports 收敛的经验规律。
- 必需修复：重新评测并保留逐 item 预测，对每个 support/seed 使用相同任务样本；以 seed 为高层单位，用 paired item bootstrap 或层次 bootstrap 估计每个差值和 max-minus-min 范围的联合区间，并预先定义等价/实质收缩标准。
- 验证标准：预注册的多 seed paired 分析表明 after-range 相对 before-range 的收缩区间排除零或达到事前实质门槛，且结论对任务加权、leave-one-task-out 和 multiple comparisons 稳定。
- 仍需证据：逐 item correctness、逐 seed checkpoint、paired/joint bootstrap 和预设 equivalence/compression estimand。
- 预期影响：high；判断置信度：high。

##### S1-R3-03 · 主要 · 重要性、实验严谨性、可复现性

- 位置：第1、4–7、13–16页，图1、§4.3、§5.1、限制、附录C.4–C.7
- 观察证据：所有 SparseForge 与625M recovery 数字都在 rank-16 SLoRB 分支活跃时评测；finalizer 不把 BP fold 入 W，也未做 fold 后2:4 reprojection、每组sum=2验证、pre/post quality 或对应 checkpoint 的吞吐。现有 validator 只检查全局稀疏度/重叠。
- 重要性：活跃低秩分支改变推理结构、容量、内存与速度，也可能吸收不同 support 的误差并人为压缩 cross-support spread。因而当前结果不是可部署 exact-2:4 模型的质量或系统结果，方法标题和大量 pipeline 讨论缺少执行终点。
- 必需修复：对每个已研究 support 实现并冻结 BP fold→exact-2:4 reprojection→groupwise assertion 的导出；在同一导出上重跑 PPL、AVG-9、support-change、内存与吞吐，并加入禁用/等容量 SLoRB 控制。
- 验证标准：每个导出逐组恰有2个非零，SLoRB 分支被移除；pre/post-export 的质量变化有配对区间，实际目标硬件吞吐来自同一 checkpoint digest；范围压缩在无活跃分支的导出上仍存在。
- 仍需证据：导出代码、groupwise 验证日志、同一 digest 的质量/系统结果及 SLoRB 消融。
- 预期影响：high；判断置信度：high。

##### S1-R3-04 · 主要 · 可复现性、引文完整性、限制与负责任表述

- 位置：第4–7页表1–2、图2、§5.1–5.4；第15–16页附录C.6–C.7、D–E
- 观察证据：5B headline checkpoint、args.json 和恢复语料均位于不可访问主机；早期材料对语料归属相互冲突并已撤回。RTE 曾有69.82与注册49.82两个日志值，当前选择49.82主要依据整数分母可表示与同一行 WinoGrande=69.85 的近似。旧吞吐和 cross-family suite 的 checkpoint/套件身份也不可核验。
- 重要性：这些历史点无法独立确认来自所述 checkpoint、代码或语料，RTE 冲突又显著改变任务行为和聚合。即使被称为背景，它们占据主要表图并影响对方法质量的印象，超出可审计证据。未知语料还带来数据治理与污染风险。
- 必需修复：若不能恢复 checkpoint、manifest、原始 predictions、完整日志与 digest，应从证据性表图中删除5B/旧吞吐/cross-family 数字，只在历史局限中简述；若恢复，则按同一冻结 harness 重评并公开可审计 provenance。
- 验证标准：独立审计可从 checkpoint digest、args、语料 manifest 和 evaluator outputs 重建每个单元；RTE 原始逐 item 输出唯一支持一个值，且所有系统数字绑定同一导出 digest。
- 仍需证据：可访问的 checkpoint、完整训练/语料 manifest、逐 item predictions、harness 日志和硬件 benchmark provenance。
- 预期影响：high；判断置信度：high。

##### S1-R3-05 · 主要 · 新颖性、重要性、实验严谨性

- 位置：第1–4页§1、§3–4；第7页限制；第11页猜想C1
- 观察证据：曲率分数、软硬退火、mask track 与 SLoRB 从未在匹配 token/数据/seed 预算下分别消融；没有有效的625M SparseForge endpoint。附录C1的 asymptotic-intercept 仅为未验证猜想，论文自己称唯一可证明步骤只是范围极限的恒等式。
- 重要性：无法把任何质量、范围压缩或系统性质归因于 SparseForge 的新机制；大量方法描述因此没有对应实验证据，当前工作也不能与 AST/CAST/固定支持恢复做公平方法比较。
- 必需修复：在同一恢复数据、token、优化器、SLoRB 容量、初始 checkpoint 与多 seeds 下运行曲率/随机/幅度分数、无退火、固定 mask、无 SLoRB、AST/CAST 等消融；把猜想移至清晰标注的展望，除非有可证条件与实证轨迹支持。
- 验证标准：预注册的 factorial/最小消融能对每个主组件给出配对增量与不确定性，并在 deployable export 上复现；未被支持的机制语言被删除。
- 仍需证据：匹配预算多 seed 消融、训练轨迹、组件交互分析和导出后结果。
- 预期影响：high；判断置信度：high。

##### S1-R3-06 · 次要 · 清晰度

- 位置：第2页图1及全文结构
- 观察证据：图1图内标注明确小于正文且在单栏尺寸下难读；16页中核心结果只占表5附近少量内容，其余混合方法背景、历史不可验证结果、统计附录、代码规范与开放猜想。
- 重要性：过量背景降低了主张层级的可见性，并可能让读者误把未执行的 pipeline/export 与已验证贡献混为一谈。
- 必需修复：把正文压缩为可审计的三 support 测量、统计边界与必要实现说明；将不可验证历史记录和代码级规范移至补充材料，重绘图1以正常印刷字号标注 executed versus planned。
- 验证标准：正文每个主图表都直接支持一个已执行主张，图内最小字号与正文可读，planned/archived/unverified 状态一眼可区分。
- 仍需证据：重构后的稿件与印刷尺寸视觉检查。
- 预期影响：low；判断置信度：high。

**给作者的问题：**

- 表5的 native ALPS=52.12 与 ELSA=54.59 是三 seed 均值，而 recovered 行是否来自其中某一个确切 support/checkpoint？请给出每个 recovered endpoint 对应的初始 mask SHA、native seed、恢复 seed 和逐一 before/after 数值。
- 三个625M recovered 行是否都只用 base seed 0？是否有调度、数据顺序、初始化或 SLoRB 随机性重复？如果没有，如何排除范围压缩只是单次训练噪声或回归均值？
- SLoRB 分支在推理时活跃且 fold/reprojection 未测。三个 support 的差异是否可能主要被共同的稠密低秩分支吸收？去掉、冻结或匹配分支容量后范围如何变化？
- 既然5B checkpoint、args.json 和语料均不可访问，表1的 SparseForge 5B 数字和附录代码规范如何证明属于同一 frozen commit/配置，而不是不同历史代的记录？
- RTE 的69.82与49.82冲突仅靠整数分母和与 WinoGrande 的近似推断为单数字损坏。是否存在原始 predictions、harness stdout、checkpoint digest 或 evaluator cache 可确认49.82确为目标 checkpoint？
- 为什么保留没有可验证 checkpoint 身份的旧 H800 吞吐与未命名七任务 cross-family 表，而不把论文压缩为可核验的三 support 测量？
- 要支持 exact-2:4 部署主张，作者能否完成 SLoRB fold、每组sum=2断言、reprojection 前后 support 变化、PPL/AVG-9 和实际 TensorRT 吞吐的一致 checkpoint 评测？

**能提高评分的证据：**

- 以同一三个确切 support/checkpoint 为单位提供配对 before/after、多恢复 seeds 和逐 item predictions，并用联合层次分析证实范围实质收缩。
- 完成无活跃 SLoRB 分支的 fold+reprojection exact-2:4 导出，在同一 digest 上复现质量与真实硬件吞吐。
- 在匹配625M token、数据、优化器和 seeds 下完成 SparseForge 组件及强基线消融，显示可归因增量。
- 恢复并公开5B checkpoint/语料/评测 provenance，或完全删除不可验证历史点，形成只由可重放证据支撑的论文。

**会降低评分的证据：**

- 确认 native 三 seed 均值与 recovered 单 seed 端点不是同一固定 support，且无法恢复逐一映射。
- 多 seed paired 重跑显示0.118范围压缩不稳定或处于训练/评测噪声内。
- fold/reprojection 后质量显著下降、support 大幅改变，或实际吞吐不支持2:4部署叙事。
- 恢复的原始记录证明 RTE、语料身份或5B checkpoint 与表1叙述不一致。

**伦理标记：** 是。5B 恢复语料及其 manifest 不可访问，早期材料对语料身份还相互冲突，因此无法审计许可、隐私、治理、污染或数据使用约束。论文已披露该风险，但在恢复语料 provenance 解决前，不应把该 checkpoint 作为可复现或可治理的方法证据。

**LLM 使用披露：** 本审稿由全新、隔离的 AI 子代理 R3 生成，仅用于内部投稿前质量控制；该子代理只读取指定的冻结 PDF、审稿协议、量表与 JSON schema，未与任何其他评审通信，也未接触作者计划、历史评审、目标分数或版本历史。

**评审限制：**

- 按隔离要求仅审阅冻结 PDF，未读取匿名 artifact、CSV、checkpoint、代码或历史日志；因此无法验证表1/表5的 support 映射、RTE 原始输出或实现规范与归档 checkpoint 的一致性。
- 按任务要求未联网，ELSA/ProxSparse/CAST 等2025–2026引用及 NVIDIA/数据集元数据未做外部核验；citation 风险只按 PDF 内部 provenance 评估。
- PDF 共16页，文本抽取完整且逐页视觉核查未发现内容解析故障；图1的小字号是稿件本身的可读性问题，而非解析失败。

## 五、如何使用本报告

1. 先按总评分表筛选：中位数 2 的论文应优先重构贡献或补关键证据；中位数 4 的论文优先处理三位审稿人重复指出的 major/critical 问题；中位数 6 也不代表可直接投稿，仍需清除格式与证据阻断项。
2. 每篇先修复可用现有证据完成的 claim narrowing、标签一致性、表图叙事和复现说明，再决定是否投入新实验。
3. 新实验以 issue 中的 verification test 为验收标准，避免只增加规模而没有解决识别问题。
4. 修改后应重新冻结新 SHA，并由全新审稿上下文进行第二轮盲审；不要把本轮分数作为下一轮审稿人的先验。

附属机器可读文件：`score_summary.tsv`、`review_issue_index.tsv`、`reviews/R1..R3/*.json`、`review_snapshot_manifest.tsv`。
