# 投稿前核对表 — FINITE-SAMPLE SAFE MODEL RANKING UNDER SUBGROUP-MIX TURNOVER

- 唯一主稿：`subgroup_mix_ranking/paper/paper.tex` → `paper.pdf`（r1904 增补 AI-use 声明 + 引用核验后 9 页正文）。
- 当前坐标：M5-C-content-checked。此表为补齐 ICLR 投稿硬性交付物而建，不是推进门禁。

## A. 页数与编译
- [x] **页数**：pdflatex×2+bibtex → `Output written on paper.pdf (12 pages)`。正文恰 9 页
  （Conclusion 收尾于 p9，References 在 p10，附录 A-E + AI-use 声明 p11-12）。ICLR 2027 主文上限 9 页，在限内。
- [x] **编译健康**：0 LaTeX Error / 0 Overfull / 0 undefined（ref+citation）。双盲匿名 `\author{Anonymous}`。
- [x] **模板**：官方 `iclr2027_conference.sty` + `iclr2027_conference.bst`；近期编译 log：`paper/c_r1904*.log`。

## B. 引用核验（ICLR 要求逐条）— `code/verify_citations_r1904.py` 前台 EXIT=0 = 8/8 OK
- [x] 两条 arXiv 用官方页面逐字核验（2026-08-18）：
  - `maurer2009bernstein` = arXiv:0907.3740，Maurer & Pontil，**《Empirical Bernstein Bounds and Sample
    Variance Penalization》**（r1904 把旧 "Penalties" 更正为官方 "Penalization"）。
  - `sagawa2019distribution` = arXiv:1911.08731，Sagawa/Koh/Hashimoto/Liang，2019 题目与作者逐字匹配。
- [x] 其余 6 条为教科书/期刊登记经典（non-arXiv）：Clopper–Pearson Biometrika'34、Dunnett JASA'55、
  Hsu Ann.Stat'84、Duchi Namkoong JMLR'19、Saerens et al.'02、Vovk et al.'05(Springer)。
- [x] **零虚构引用**：删除了无关的 `hsu2016generalized` 死条目（未被正文引用且标题截断错误）——
  所有被引用键均已核验。`\bibcite` 实际引用键为 8 条全部覆盖。

## C. 匿名性
- [x] `\author{Anonymous}`；双盲无作者名/机构/可识别措辞。
- [x] PDF 无 model 水印/作者元数据。

## D. 理论与证明可复现
- [x] P1 线性带联合覆盖（Lem.1/Prop.P1，Bonferroni+Clopper–Pearson，≥1−δ）。
- [x] P2 minimax-regret 上界（Prop.P2）——证明在正文，`argmin_i U_i(w)` 与回退门。
- [x] P3/Thm.1 配对差证书（MCB 风格，连续单纯形同时覆盖，Def:paired）。
- [x] M1 机制警（Prop.metalanti）：带宽做门不做箭；实际 165/352 diverge、+0.0019（本文档约束处内部一致）。
- [x] M3 标签预算 + time-uniform 自适应证书（Lem.2，empirical-Bernstein CS）。
- [x] M3.5 凸 minimax 联合解 + 活跃集，r1897 修正对偶 `值=2λR`；确定性非对称反例。
- [x] 全部承重数字可追溯冻结 JSON（results/SUBGMIX_*.json），数字逐值断言 verifier `results/M5_VERIFY_R1902.py` 前台 EXIT=0 = **105/105 PASS**。

## E. 实验真实性
- [x] 4 个公开 carrier：Fashion-MNIST(MIT)、MNIST、KMNIST(CC-BY-SA-4.0，REPRO_README 标注 license)，
  sklearn digits(BSD-3)、20News（本地 sklearn 缓存，零网络）。
- [x] 主比较 M2 vs M2.5 为 **full 5-seed matched**（r1903：M2 0.269 / MPB 0.260 / Hoeffding 0.097 /
  Normal 0.503，全 cov 1.0，350 行/方法）；M3/M3.5 报 3-seed，已在 Limitations 如实披露。
- [x] 负结果如实保留：M1 anti-selective、digits/news capacity-wall（committed 0）、Normal 仅渐近诊断
  不作有限样本 claim、低覆盖 cell 未隐藏；`BoLEGACY_SUPERSEDED`/`INVALID_DIAGNOSTIC` 已显式作废。
- [x] 零 GPU、纯 CPU、全部前台 readback；无脱离 Harness 后台执行。

## F. AI 使用声明
- [x] `\section{AI use statement}` 附录末（r1904 新增）：注明 LLM 辅助实现/证明核验/稿件准备，科学声明归作者。

## G. 复现材料
- [x] `REPRO_README.md`：环境/依赖、各回合复现命令、产物 JSON、诚实边界。
- [x] `code/`：全部 runner/聚合/verifier，标注回合；论文数字来源可逐项重跑。
- [x] `CHINESE_SUMMARY.md`：中文研究总结（贡献/假设/结论/局限/复现/风险）。
- [x] 数据许可：`duplicate_sel/data/DATA_LICENSE.md`（Fashion/MNIST/KMNIST）。

## H. 剩余待办（非门禁，同题复现/写作增强）
- [ ] M3/M3.5 完整 **5-seed** 复验：m3_cache 仅存 3 seed（s0-s2），补 seed 需 news 重新训练
      ~3.5h/seed，与 M2/M2.5 主表（已 5-seed）无涉；诚实保留为复现项。
- [x] 「τ 选择协议」已闭合（r1906/r1907 + r1908 并入主稿附录 app:tau）：M6 消除 τ
      （status-quo F0 相对门）、M7 保留绝对 τ 有限菜单 CAL-only 选择；冻结结论 L682 已
      改为两种互补闭合。
- [x] 「strictly finite-sample tightened band」remaining 已闭（r1909 M8 + r1910 M9）：绝对门 exact
      MPB/Hoeffding 保留真内容（0.260/0.097，sound）；相对门 M6 在严格带下 vacuous-collapse
      （0.663→0.646，全部 trivial keep-F0，6 个真实切换提案被拒绝）；M9 补 budget-priced 刻画：
      6 提案在有效校准倍增 N*∈{2..20}（MPB 中位 5 / Hoeffding 中位 10）下全数重开、OUTER sound 保
      持（oracle gain 0.034–0.130>0）——相对门严格性非不可行而是预算计价。边界刻画+M9 frontier
      已并入 app:tau。主稿重编译 13 页（正文恰 9）0 err/0 overfull/0 undef；
      verifier `results/M8_VERIFY_R1909.py` EXIT=0 = 17/17、`results/M9_VERIFY_R1910.py` EXIT=0。
- [x] 「相对门 exact 带非空性的可证前沿」（r1911 M10，MGR 指令 7cc94318db8d）：在 fresh
      seed{10..14}、OUTER-exclusive FIT/CAL（同一 F0/i*/subgroup）上预固定每组 CAL 预算网格
      b∈{0.25,0.5,1.0} 并真实按组子采样重算，得到**可证空证书**：全部 125 真实切换行在满预算
      b\*_hoef>1（中位 273 / min 3.7，Δ 中位 0.014），Hoeffding exact 相对门在整个可行预算轴
      b≤1 无可证非空内容；exact hoef/mpb 在每 carrier×预算 admit_real=0.0。**严格分报平凡行**
      （i*==F0）与真实切换行，不掩盖（digits@0.25 triv_frac=0.92、news 0.43）。**可执行混合规则**：
      安全部署用 M2.5 exact 绝对门（fashion@b=1 abs_commit=1.0·no_worse_cov=1.0，随 b 单调增强），
      相对门 M6 仅作渐近/描述性诊断；弱域 news=预算墙（0.136→0）/digits=容量墙（0.093）如实保留。
      理论：`THEORY_EXACTBAND_FRONTIER_R1911.md`；verifier `results/M10_VERIFY_R1911.py` EXIT=0
      = **ALL PASS**。主稿并入 app:tau M10 段，重编译 13 页（正文恰 9）0 err/0 overfull/0 undef；
      只读候选 r1905 包不动。
- [x] 唯一 M5 作者候选已装配且只读：
      `paper/A2_subgroup_mix_ranking_author_candidate_r1905/`（MANIFEST.md 锁不可变发布包，
      r1904 M5 content-complete 状态，105/105 + 8/8 + clean compile 12 页全部从候选根前台重放）。
## r1915（MGR 指令 f147b3bf33e9）职责地图与新增
- **Canonical portal 职责地图**：唯一 mutable 主稿=workspace `subgroup_mix_ranking/paper/paper.tex`；
  当前唯一后继候选=run-root `paper/A2_subgroup_mix_ranking_author_candidate_r1915`（只读发布包+manifest）；
  历史只读=`..._r1905`（最早 frozen）、`..._r1911`（前继）。可变文件不哈希冻结。
- **新增 M12 真采样审计**：`results/SUBGMIX_M12_FRESH5_BUDGET_R1915.json` + `M12_VERIFY_R1915.py`
  （EXIT=0 245/245）。fresh seed{10..14} 在 M3/M3.5 预算总点对比严格相对/绝对/statusquo/baseline。
- **候选内部核对（r1915 装配当日前台重放）**：M5/M7/M8/M9/M10 verifier 原样运行 + M11/M12 新 verifier，
  全部 EXIT=0；干净编译 paper.pdf 14 页（正文恰 9）。
