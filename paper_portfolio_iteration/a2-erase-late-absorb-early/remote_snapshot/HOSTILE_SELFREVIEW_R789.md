# Hostile 自我预审（r789）— paper/ 当前稿（12页编译通过）

MGR 组织正式 hostile reviewer 模拟前的自查。逐项从挑剔 ICLR 审稿人角度攻，核对当前文件实际状态（非记忆）。分级：[blocking] 必须修 / [non-blocking] 宜修 / [noted] 已充分披露无需动作。

## R1 理论严谨性

1. **T1 的 "bitwise identical" 措辞**（§6.1 T1段）：合成凸格三shape逐种子相同是真（pilot14 JSON 极差0），但 "bitwise" 暗示浮点逐位相同。实则是"到报告精度相同（极差<1e-4）"。[non-blocking] 宜把 bitwise 软化为 "identical up to reported precision (range <1e-4)"，防 reviewer 用浮点不结合性挑刺。附录B proof skeleton 里 "bitwise (shape)" 同。
2. **T2 吸引子回归的唯一性依赖强凸**：limitations(b) 已明确 T1/T2 需强凸、MLP推广open。[noted]
3. **T3 鞅条件**：shuffle 无放回 E[ξ_t|F_t]=0 + Azuma-Hoeffding 已写入附录B(2)。A3 F2 已审过此条。[noted]
4. **"endpoint attribution 是 K=1 饱和极限"**（§unify）：这是定性重构句非定理——措辞是 "reappear as the K=1 saturated special case"，无误认为严格定理的风险。[noted]
5. **λ(B) 深穿越低估**：已在 T3 附录 proof + §6.1 机制证伪段 + limitations(d) 三处一致披露 "ordinal/switch robust, pointwise needs λ(B)"。[noted]

## R2 实验/可复现

6. **预注册时间序**：PREREGISTER 文件（SALVAGE_LOSS_PREREG.md 12:34、DIAG20_PREREG 等）均在对应 run 前落盘；EQRA-loss 的 P1/P2/P3 阈值未看结果改。审计 trail 在附录 repro 节列明。[noted]
7. **种子数**：合成 6 种子、real 臂 6 种子、diag20 表 3 种子（表注已写 "medians over 3 seeds"）。real 臂 per-seed 配对差（§repro）。足够。
8. **artifact 链**：lr_phase_datavalue_r1/ 39 脚本 + 34 JSON + GATEC/DIAG20/EQRA verdict 全在。附录 repro 节路径正确（lr_phase_datavalue_r1/ 前缀）。[noted]
9. **CIFAR-10N 引用**：\citep{wei2022cifar10n} 在 ref.bib；数据来自本机缓存 CIFAR-10_human.pt（r210 lane 官方缓存，integrity 已核 clean==tarball）。可复现命令在 RESEARCH_SUMMARY_ZH。[noted]
10. **图1(c) 黑虚线 1-exp(-0.2023B)**：λ̄=0.2023 是 pilot11 梯形积分无参数预测（A3 cond_iv 复算逐位吻合 0.20），非拟合。caption 已写 "parameter-free"。[noted]

## R3 novelty/citation

11. **近邻审计 15 篇一手**：附录A 表 15 行，每行 their object + our delta，含 Safran-Shamir/Nagaraj/Mishchenko RR谱系（r789 新增）、Xie 纠错记录（r774误述→r775更正）、MiniCPM §4.4 future-work缺口。表头数字已同步15。[noted]
12. **无虚构引用**：ref.bib 10+条全部真实一手（koh2017/park2023/ilyas2022/toneva2019/bourtoule2021/hu2024minicpm/mandt2017/smith2017/smith2021/dataprophet2026/arazo2019/jiang2018/han2018/wei2022）。无DOI/页码虚构。RR三篇走 arXiv ID 内联（与既有 Xie/TracIn/Carlini/GradientStarvation 同体例）。[noted]

## R4 结构/呈现

13. **uncited \label{}**：tab:budget、tab:diag20 有 label 但正文无 \ref{}（用 "pilot 15" / "diag20" 文字引用）。ICLR 惯例表应被 \ref 引用。[non-blocking] 宜在 §6.1 T3段加 "(Table~\ref{tab:budget})"、§6.2 diag20段加 "(Table~\ref{tab:diag20})"。fig:main 同。
14. **正文页数**：12页 = 正文9页满（§8 Conclusion 终于 p9）+ refs ~1 + 附录 ~2。守则 ≥8.5 满足，无凑页空白。tab:diag20 数值实时从 JSON 提取核对（r788 已修正表注中位数笔误）。[noted]
15. **双盲**：无作者名/机构/致谢泄漏；ai statement 待 MGR 统一模板（r788 已列待办）。[blocking→待MGR]
16. **abstract**：未单独核查（在 \begin{document} 前），下轮通读确认与正文贡献一致。[non-blocking]

## R5 诚实性（守则核心）

17. **失败不包装**：R2失败(pilot19 recovery 0.24)、R3不过(R3 report-only)、P3失败(EQRA-loss −0.026)、cos负结果——全部如实写且未隐去。boundary/negative 专节 §6.3。[noted]
18. **条件性成功不包装普适**：EQRA-loss 明确 "conditionally, not universally applicable: noise rate non-negligible and estimable"（limitations(e)）。[noted]

## 处置

- [blocking] 15（AI声明）等 MGR 统一模板，非我自主可定。
- [non-blocking] 1（bitwise软化）、13（表/图 \ref）、16（abstract通读）本轮自主修。
- 其余 [noted] 已充分，正式 hostile 模拟时可作为答辩依据。

正式模拟建议由 MGR 指派 1 名非作者 worker 按 ICLR review form（soundness/excitement/reproducibility）独立打分，本自审不替代。
