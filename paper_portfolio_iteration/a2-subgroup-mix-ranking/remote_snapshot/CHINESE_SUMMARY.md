# 中文研究总结 — FINITE-SAMPLE SAFE MODEL RANKING UNDER SUBGROUP-MIX TURNOVER

> **Canonical portal（职责地图，2026-08-18 r1915 更新）**
> - **唯一 mutable 主稿（可继续编辑的唯一源）**：`agents/A2/workspace/subgroup_mix_ranking/paper/paper.tex` → `paper.pdf`。所有方法/实验/写作继续在此推进；可变 tex/脚本/配置不做哈希冻结。
> - **当前唯一后继候选（作者稿，只读发布包）**：run-root `paper/A2_subgroup_mix_ranking_author_candidate_r1915`。由 mutable 主稿整体装配，含 r1912 首图 + M10/M11/M12 段；只对不可变发布资产（paper.tex/pdf/bib + 承重证据 JSON + runner + verifier 定稿件）写 SHA-256 manifest。
> - **历史只读版本（exact bytes 保留，不得修改）**：`paper/A2_subgroup_mix_ranking_author_candidate_r1905`（最早 frozen M5）、`..._r1911`（上一前继）。
> - 问题→权威来源：正文数字→`subgroup_mix_ranking/results/SUBGMIX_*.json`；复现→`REPRO_README.md`；逐值断言→`subgroup_mix_ranking/results/M*_VERIFY_*.py`；中文总结→本文档；投稿核对→`SUBMISSION_CHECKLIST.md`。

坐标：M5-C-content-checked（正文 ≤9 页 + 主表/附录数字全部由 M5_VERIFY_R1902.py 逐值锁冻结
JSON，现含 r1903 新增 full 5-seed M2.5 断言块，105/105 PASS；2 处过度陈述由 r1901 修正，
5-seed 复验缺口由 r1903 收口；r1904 补齐 AI-use 声明、引用核验与投稿核对表；
r1915 增 M12 fresh-seed 真采样审计+M11 保守后方）。纯 CPU、前台、零 GPU。

## 贡献
给定固定候选模型池，在**子组/类别混合比例迁移（label-prior shift）**下，以有限样本方式对每个未来
mixture 权重 $w$ 做"安全模型选择"：要么提交选择并附带"真 regret ≤ τ"的证书，要么诚实弃答而非硬选可能翻车的
点估计最优。核心理论把每个模型的 mixture 风险写成逐组风险的线性带，用 Bonferroni+Clopper–Pearson 给出
联合覆盖 ≥1−δ 的带（Lemma 1 / P1）、sound 的 minimax regret 上界（P2）、以及**配对差（MCB 风格）证书**（P3 / Thm 2）——
后者在 CAL 上把所选模型与最优模型"配对"同点比较，抵消共享误差，得到比"减两个绝对带"严格更紧的界。

## 理论假设
- A1 label-prior shift：$P(X|Y=g)$ 不变，只 $w$ 移动；评测用组内重加权实现。
- A2 各候选在各组的错误率为可估计 Bernoulli（组内 iid）。
- 证书为逐 $w$ 逐点覆盖（≥1−δ），非全 $w$ 网格同时联合（诚实披露为限制）。

## 实验结论（4 个公开 carrier × 4 分类器，δ=0.1）
1. **证书健全**：committed 集合上真(测试) regret ≤ τ 的比例，M2 与 M2.5 全变体 = 1.000（跨整个 τ 扫描）。
2. **弃答有效**：abstained 行若硬选点估计，regret 均值 0.0044、最大 0.0546，远高于 committed 的
   0.00001/0.0005（matched 3-seed M2; 5-seed 视图 0.0043 vs 0.0001）；弃答门精准抓 turnover-skew 危险区。
3. **M1 机制警（门不是箭）**：`argmin_i U_i(w)` 因 `max_j(U_i−L_j)=U_i−min_j L_j` 退化为
   argmin U_i，均质宽度下 vacuous（350/350 ≡ cal-prior）；mixture-ball 下按盘面 JSON 逐行核对为
   **165/352 diverge**（非旧记 88/352），diverged 子集 regret diff **−0.0021**（mrr 略优，
   93 better/62 worse，无一致方向），overall +0.0019——即悲观化器既不全恒同也不恒反选，
   "带只做门不做箭"是承重设计结论（Prop metalanti 证的是构造性反选域，非数据恒差）。
4. **证书价格前沿（r1903 起 full 5-seed matched {0,1,2,3,4}）**：M2.5 现已在与冻结 M2
   相同的 5 seed 上重跑（确定性 random_state=seed+同源 split，零网络，~60min CPU），
   组成严格全额 matched 比较：M2 精确 0.269 / M2.5-MPB 精确 0.260（≈等量，诚实写
   "约等"，不再称非劣也不称超）/ Hoeffding 精确 0.097 / Normal 0.503（**仅渐近诊断**，
   不作有限样本 headline）。全部 cert_cov=1.0。r1901 记的"5-seed M2.5 复验为复现项"
   已收口，r1903 用新冻结 JSON SUBGMIX_M25_PAIRED_R1885_5SEED.json 落盘；M5_VERIFY 增
   5-seed 断言块现 105/105。配对差从"减两个绝对带"收紧证书，选择规则不变（i* 同）。

## 局限（诚实披露）
- **配对证书为 simplex-同时，非逐 $w$ 逐点**：UCB[(i,j)][g] 独立于 $w$，只在有序对×组上一次
  Bonferroni，同一事件同时覆盖整个连续单纯形上所有 $w$ 与数据依赖选择 $i^*(w)$；r1898 以每 cell
  1000 离网格 Dirichlet 扫描核验 committed 覆盖仍 1.0（mnist 0.5 离网格 2816/3000、fashion 0.95
  70/70）。旧"全网格同时联合待做（网格再 Bonferroni）"是过度保守的自我降级，已更正。
- Normal 变体为渐近（CLT），非有限样本；有限样本 claim 用精确 MPB/Hoeffding。
- τ 为操作参数（报告权衡曲线，非定理预测）。
- **M2 vs M2.5 主比较现为 full 5-seed matched**（r1903）：M2 5-seed 0.269、M2.5-MPB 0.260、
  Hoeffding 0.097、Normal 0.503，同 seed/split/model/w-grid/delta/gate，全 cov 1.0
  （350 行/方法）。r1895-95 legacy 3-seed matched 数值保留在 M5_VERIFY 作为历史断言，
  不再承重。每载体 6 折的完整网格、M3 full 5-seed 复验为复现项（与 5-seed M2.5 主表无涉）。
- 覆盖 covariate shift-in-group（非 label-prior）不在证书范围。
- **M3 标签预算与逐样本标签审计的 endpoint 区分**：我是在若干候选模型间做 ranking 选择并对
  mixture-indexed regret 出证书，CAL 标签预算用于"跨哪些 subgroup 花以保住 regret 门健全"；
  与逐样本 label-audit / 点贡献估计（如 A6 相关性感知 pruning 类）对象不同——我审模型候选，
  它审模型内样本点。此区分已写入主稿 Related Work。

## 复现命令
```bash
# 首轮 M1 负结果
python3 subgroup_mix_ranking/code/subgmmix_pilot_r1881.py
python3 subgroup_mix_ranking/code/subgmmix_diag_r1881b.py
python3 subgroup_mix_ranking/code/subgmmix_r1881c_mixtureball.py
# M2 证书门（r1884，5 seed，~1h）
python3 subgroup_mix_ranking/code/subgmmix_m2_gate_r1884.py
# M2.5 配对差证书 / 证书价格前沿（r1885，3 seed，~27min；++SEEDS 到 5 得全网格）
python3 subgroup_mix_ranking/code/subgmmix_m25_paired_r1885.py
```
证据 JSON：`results/SUBGMIX_*.json`；主题论文档在 `paper/paper.tex`（11 页，0 error/0 undef/0 overfull，双盲）。

## 未解决风险
- 精确变体覆盖仍偏低（MPB 0.257 略低 M2），Normal 0.495 提升大但仅渐近诊断——
  下一步同题方向：更紧的精确界（逐组 pooled / 半参 / logloss-risk）、τ 选择协议。
  （"网格联合 Bonferroni"已由 r1898 关闭：配对证书本身即 simplex-同时。）
- fashion训练较慢（受 MLP/SVC 影响），完整 5 seed 复验需注意前台时长。
## r1895 更新：固定预算凸 minimax 分配的正式化（MGR 指令）

把 r1893/r1894 的"uniform=maximin 恒支配"从现象升级为**可核验定理**并把凸 minimax 作强基线：

**形式化**（`THEORY_MINIMAX_R1895.md`）：固定预算对象 min_{n_g,Σ=R} max_j Σ_g w_g r_g(n_g)，
r_g(n_g)=半宽∝β_g/√n_g。KKT/epigraph 给**水填充 n_g ∝ (w_g·β_g)^(2/3)**；uniform 是最优
minimax 解 ⟺ w_gβ_g 恒等（W 对称 spanning + 组方差对称的可核验条件）；对偶闭式给容量墙判定。
> ⚠️ **r1897 对偶修正**：旧 §2.3 的 `d*=λR` 维度错误（正确 `值=2λR`，∝R^{-1/2}），旧
> minimax 还误把"所有候选等权"当活跃集。修正见 r1897 更新段。r1895 的 `SUBGMIX_MINIMAX_R1895.json`
> 与对偶一律标 INVALID-DIAGNOSTIC。

**确定性非对称反例**（`SUBGMIX_MM_COUNTER_R1895.json`）：非对称（非 spanning）部署 W={e_0} 上，
convex-minimax 的 UB 是 uniform 的 ~0.3× 且 **minimax committed、uniform 不 committed** →
uniform 不是普适 minimax 最优，此前"恒支配"仅是评估网格对称性的产物（MGR 判断成立）。

**四 carrier 真实比较**（`SUBGMIX_MINIMAX_R1895.json`，复用 m3_cache 零再训练，同网格同证书）：
mnist 上 uniform 仍胜（0.778 vs minimax 0.533，对称 grid 无菌）；fashion 低预算 minimax 追平
neyman；digits/news 容量墙（n<680）minimax 亦 0。paid-CS 负结果保留。条件化：对称 spanning W→
uniform；非对称 W→凸 minimax（反例证实严格更优）。

证书健全（cv=1.0, 0 viol）；3-seed；配对证书 simplex-同时（TBD-1 由 r1898 关闭：UCB 独立于 w，
单次有序对×组 Bonferroni 即同时覆盖连续单纯形，离网格 Dirichlet 扫描覆盖仍 1.0）。主稿已加 §M3.5。

## r1896 更新：条件选择门端到端执行（§M3.5 从陈述变实跑）
把 r1895 的"非对称非 spanning W 下凸 minimax 严格优于 uniform"从理论+合成反例推进到四真实 carrier
前台实跑：部署集收窄到**单个最高方差顶点 W'={e_g0}**（g0=argmax_g β_g, β 仅 FIT），同一静态配对-MPB
证书/预算/网格/种子，只改部署集与揭示分配（零重训，冻结 m3_cache）。
- **fashion π=0.5：uniform 弃答 2/3 行（medUB 0.041），convex-minimax 全 commit（medUB 0.000）**
  ——水填充把预算集中到被查询的高方差组，收紧该组 UCB→0，恰好买下 uniform 弃答的行。这是
  Prop uniform-opt 预测的首个真实 carrier 直接印证，也是反例（而非对称 M3 网格）捕捉的失败模式。
- mnist 两规则下均全 commit（低噪声统一胜）；digits/news 两下均 capacity wall（minimax 把 UB 减半
  但未过 τ）。全 committed 行 true regret=0（cert 健全，0 viol）。
- 诚实边界：minimax 优势只在非 spanning/窄支撑部署集出现；对称 spanning 网格（M3）仍 uniform=maximin 胜。
  CV 用非零组 population CV（.42/.33/.24/.16，与 r1895 一致）。
- 证据 `results/SUBGMIX_CONDGATE_SINGLE_R1896.json`；主稿加 §M3.5 条件化实跑 + 表 tab:m35gate。

## r1897 更新：minimax 对偶/齐次性修正（MGR 593c907d2ccd）+ 联合凸解 + all-active 消融
MGR 指正 r1895 两处错误：(a) §2.3 `d*=λR` 维度错（λ 已含 R^{-3/2}）；(b) 把"所有候选等权"
当活跃集。本回合用**引擎矩阵** C_{kg}=w_gβ_{jg}（孩子=(候选 j, mixture w)）修正：
- **修正对偶**（`THEORY_MINIMAX_R1895.md` §2.3bis，代码 `code/minimax_core_r1897.py`）：
  对象 $\min_n\max_{\mu}$ 是 convex-vs-concave，Sion 强对偶
  `d*=P*=R^{-1/2}[max_μ S(μ)]^{3/2}`，$S(\mu)=\sum_g a_g(\mu)^{2/3}$（凹）。固定 μ 水填充
  `n_g=R a_g^{2/3}/S`；预算乘子 λ=½S^{3/2}R^{-3/2} ⇒ **值=2λR、∝R^{-1/2}**，与半径目标齐次。
  活跃集由互补松弛=承重引擎，**绝非全部候选**。
- **数值 fixture 核验**（`results/MM_FIXTURE_R1897.json`，前台 EXIT=0）：四类 (M,G)/β 连续对偶
  间隙 <1e-6；`值=2λR` 到 1e-9；R∈{400..4800} 上 V·√R 平坦 <1e-16；互补松弛活跃集正确；
  整数化代价 1e-6–1e-3（floor+n_min，诚实披露）。
- **四 carrier 真实 minimax 重算**（`results/SUBGMIX_MINIMAX_R1897.json`，修正联合求解，
  同 m3_cache/证书/网格/3seed）：mnist committed_rate 0.400/0.711/0.889/1.000（medUB
  0.048→0.000），fashion 0.067/0.067/0.089/0.178，digits/news 容量墙全 0 不变。**数字比旧 r1895
  更保守且正确**（真 worst-case 混合使保证更紧）。
- **all-active 消融**（`results/SUBGMIX_MINIMAX_ABLATION_R1897.json`）：真活跃引擎仅 **K 的
  6–25%**（digits .10/fashion .25/mnist .25/news .20），修正 d*（worst-case）为 uniform-mixing
  旧值的 1.17–4.19×。旧 r1895 全候选作活跃集显著高估保证，已作废。
- 主稿 §M3.5 对偶改写为联合凸解+活跃集(itemize+eq:duality)，加 all-active 消融表 tab:m35ab，
  M3.5 minimax 列换成 r1897 修正数字；paper.pdf 前台编译 EXIT=0。
- r1896 条件化门（单顶点 minimax 集中到 g0）在修正对偶下机制/数字不受影响（该顶点即活跃引擎）；
  容量墙结论不变。
- 与 top-3 顶点探针（`SUBGMIX_CONDGATE_R1896.json`, near-tie）对照：顶点越少、非 span 越强，把"探索
  全体组的代价"显式化后 minimax 优势越明显——r1894 网格探全体支配的另一个侧面。

## r1898 更新：配对证书 = simplex-同时（TBD-1 关闭，覆盖口径更正的诚实性修补）
- 发现问题：paper **自己的** Thm2/Def2 已建立单事件 E（UCB[(i,j)][g] 独立于 w、只在有序对×组上
  Bonferroni δ/(M(M−1)G)），在该事件上 UB(w)=max_j Σ_g w_g·UCB 对**任意** w 与数据依赖选择 i*(w)
  同时有效——即证书是**连续单纯形同时**，根本不需要"再对网格 Bonferroni"。但正文 Honest Scope 段/
  Limitations/Conclusion 旧文句自我降级为"per-w 逐点、网格联合待做"，与自身定理矛盾。
- 实证关闭（`results/VERIFY_GRID_JOINT_R1898.json`，前台 EXIT=0）：每 (carrier,seed,frac) 1000 个
  离网格 Dirichlet(ones(G)) w，同一 UCB 族扫描。全部 commit 区域覆盖=**1.000, 0 violation**：
  mnist π=0.5 离网格 2816/3000（on-grid 0.778）、0.65 3000/3000、0.8/0.95 3000/3000；fashion 0.8
  5/5、0.95 70/70；digits/news 容量墙仍 0 commit。off-grid CR≈on-grid CR ⇒ 非网格采样幸运。
- 写作/文档同步：paper Honest Scope/Limitations/Conclusion → "simplex-同时 + 离网格扫描核验"；
  THEORY_SKELETON TBD-1 标已解决；THEORY_MINIMAX §6 诚实边界；RESULT_MATRIX/本总结 r1898 段。
  顺带修掉 pre-existing eq:duality 排版 0.9–2pt overfull。11pp clean compile（0 overfull/0 undef/0 err）。
- 边际更正：M2.5 原"6页"整数更新为 11 页；Normal 变体仍标渐近（有限样本=MPB/Hoeffding）。

## r1899 条件化门 operating characteristic（next-step c 落地）
- 把 M3.5"条件化选择"里 CV(β̂)+spanning 的门从单点 τ + 3 seed 真实表，升级为**受控合成族
  operating characteristic**（G=4、同一 paired-MPB 证书、R=1200、20 seed、τ 扫描，
  `results/SUBGMIX_GATE_OC_R1899.json`），独立扫 deployed set ∈ {spanning, non-spanning} ×
  CV(β̂) ∈ {低, 高}。
- **判定**（驱动源=deployed set 是否 spanning，非 CV(β̂)）：非 span → minimax committed-rate
  曲线在两 CV 水平都支配 uniform（signed area +0.25，minimax τ=0.08 达 1.0 而 uniform 停
  0.25–0.93）；spanning 低 CV 持平（+0.000=uniform=maximin 成立处）；**spanning 高 CV 时
  minimax 反而更差**（−0.08，worst −0.49@τ=0.12，因集中预算到高 β 组、饿死 spanning 集低方差
  顶点）。⇒ 门"非 span→convex-minimax、否则 uniform"=safe（从不比均匀 baseline 差）+effective
  （恰在理论预测处增益）；CV(β̂) 只当定位承重高方差组的诊断、非第二驱动。与 r1896/r1897 真实
  落点一致。
- 写作：paper §5 conditional selection 段落改写 + 新增 Table gate-oc，12pp clean compile
  （0 overfull/0 undef/0 err）。合成 OC（20 seed）比先前 3 seed 真实表更稳。
- **最大缺口（移入 next-step）**：正文已达 12 页，超过 ICLR 9 页正文上限 — 后续块需
  subsection→paragraph 压缩 + 附录迁移（A5 r804 已示范 9 页无内容删除），并把主表数字锁定到
  冻结 JSON。

## r1900 论文压至 ICLR 9 页正文（零内容删除）
- **缺口**：r1899 正文 12 页，超 ICLR 2027 严格 9 页正文上限（超即 desk-reject）。
- **做了什么**（paper.tex 全文压缩 + 附录迁移）：
  1. **压缩抽象**（原~30 行→~22 行）、Contributions 7 条（3/4/6/7 收紧，去冗余）。
  2. **证明/定义**：Def 变体清单去重复括号、paired-cancellation 段落压到 2 行、
     joint-solve/duality 段压到 6 行、Prop uniform-opt 语句去重复列。
  3. **实验叙述**：M1/M2 Interp/M3 Findings 收紧，tab:m3 长 caption 压缩。
  4. **Related Work** 4 段→3 段（删 negative-caution 冗余段）、Limitations 5 条→3 条、
     Conclusion 5 行、Closing-one-liners 精简。
  5. **附录迁移**（无限页，references 之后）：App.A M1 诊断表（tab:m1）、App.B all-active
     消融（tab:m35ab）、App.C 确定性反例、App.D M3 defer-costs、App.E gate-OC 表
     （tab:gateoc）。
  6. 修 eq:duality 0.9pt overfull（本质排版工具，内容/数字未动）。
- **结果**：正文恰好 9 页（References 干净落在 p10，附录 A-E 在 p11-12，全 12 页）；
  0 error / 0 overfull / 0 undefined ref。所有 tab: 跨引用（Table 1-7、App.A-E）完整解析。
  主表（tab:main/frontier/m3/m35/m35gate/gateoc）数字未改、与冻结 JSON 一致。
- **代价声明**：压缩删的是措辞冗余与移到附录的细节，正文强调的关键数字/表（frontier 0.269→
  0.495、M3 uniform-vs-neyman、M3.5 minimax、条件化门驱动源）全部保留在 9 页内。
- **副本**：原始 12 页版备份于 paper_backup_r1900/。

## r1901 M5 content-complete 自查：86 项数字逐值锁定 + 2 处诚实修正
- **做法**：新建 `results/M5_VERIFY_R1901.py`（只读冻结 JSON，逐个 headline/appendix 数字与
  paper/paper.tex 断言精确匹配），前台 EXIT=0 = **86/86 PASS**。覆盖 tab:main/frontier
  (M2 0.269/1.0/0.0001/0.0019/0.0043/0.0546/0.0031、M2.5 0.495/0.257/0.090/1.0×3、210 行)、
  App.A M1 diag（350/352 顶点、165 diverge、93/62、−0.0021/+0.0019、880 行、15 diverge_U）、
  tab:m3（τ=0.04 评估单元格恰 4200、0 cert-violation cell、adaptive=0、MNIST 0.778/0.400/0.578、
  Fashion 0.067、0.95 1.0/0.133）、tab:m35（minimax 0.400/0.889/0.067/0.089、digits/news=0）、
  tab:m35gate（Fashion 0.333→1.0、medUB 0.041→0.000、digits 1.07–2.06/1.05、news 0.27–0.50/0.24）、
  App.B ablation（25/25/7/20%、1.30/1.33/4.19/1.17）、App.C counter（0.094/0.029、n_g 300/1197、
  no/yes）、App.E gate OC（areas +0.25/+0.25/0.00/−0.08、−0.49@τ=0.12）。
- **2 处过度陈述已诚实修正**（paper + RESULT_MATRIX + THEORY_SKELETON/CHINESE：由本轮直读盘面 JSON 发现）：
  1. **gate-OC "minimax τ=0.06 达 1.0"**：冻结 JSON non-spanning minimax 曲线
     [0,0,0.025,0.675,1.0] → 达 1.0 在 **τ=0.08**（≠0.06）。已改为 τ=0.08，表 tab:gateoc 的
     area +0.25/−0.08 数字未动（那些本就正确）。
  2. **App.A diag "15/880 全 tied"**：实际 15 diverge 中 **11 tied、4 行 M1 严格更差**
     （fashion interp7，mrr 0.0024 vs 0.0）。已改为"11 tied; 4 worse for M1 (diff +0.0024)"，
     App.A 表对应行改为 "$0$ (11 tied; 4 worse for M1, diff +0.0024)"。M1 机制结论
     （门不是选择器、分歧无一致方向）不受影响：unchanged。
- **paper.pdf 重新编译 EXIT=0**：12 页 = 正文恰 9 页 + refs p10 + App.A-D p11 + App.E p12；
  0 overfull / 0 undefined / 0 error。两处修正未把正文顶出 9 页。
- **M2 vs M2.5 seed 差**：旧 headline 曾把 M2(0.269, 5-seed) 与 M2.5(0.495/0.257, 3-seed)
  并列。r1902 已按 MGR 指令 `ea0889891916` 执行 matched-seed 修复：主文承重比较统一到
  matched 3-seed {0,1,2}——M2 0.267 / M2.5-MPB 0.257 / Hoeffding 0.091 / Normal 0.495，
  同 seed/split/model/w-grid/delta/gate（确定性 random_state=seed，零重训）。5-seed M2
  0.269 完整保留但标为独立鲁棒性视图，不作 3-seed 方法的直接 baseline；完整 5-seed 网格
  复验仍是 TBD-4 同题复现增强（news 单 seed 训练 ~3.5h，前台整轮难及，诚实保留）。

## r1902 matched-seed 可比性修复（MGR 指令 ea0889891916）
- **问题**：旧 headline 把 M2(5-seed) 0.269 与 M2.5(3-seed) 0.495/0.257 并列，seed/budget
  不可比；Limitations 披露不能替代可比性。
- **做法**：因全模型由 `random_state=seed` 确定性训练、split/数据/w-grid/门同源于 seed，
  把 M2 在 M2.5 使用的完全同 3 seed {0,1,2} 上从冻结 JSON 重聚合 = 严格 matched baseline
  （同 trained model/CAL 池/w-grid/delta/gate，零重训）。新增可变工具
  `code/subgmmix_matched_r1902.py`（只读冻结 JSON，重聚合 M2/M2.5-paired/M2.5-MPB/M2.5-Hoeffding
  的 matched 端点与 per-carrier）。结果落盘见模块 docstring。
- **matched 关键值（3-seed，τ=0.04, δ=0.1, 210 行/方法）**：
  - M2 absolute：commit 0.267，cov 1.0，comm 0.00001/0.0005，abst 0.0044/0.0546，no-gate 0.0032。
  - M2.5-paired(normal)：0.495，cov 1.0，comm 0.0007/0.0348，abst 0.0057/0.0546。
  - M2.5-MPB：0.257，cov 1.0；M2.5-Hoeffding：0.091，cov 1.0。
  - **exact 序仍诚实**：MPB 0.257 < M2 0.267（W1 保留，只写"略低"）。
  - 5-seed M2 0.269/1.0/0.0001/0.0019/0.0043/0.0546/0.0031 完整保留，标为独立鲁棒性视图。
- **同步更新**：paper.tex（abstract/tab:main/tab:frontier/contribution/interpretation/
  Limitations/新增 matched-seed 方法段）、RESULT_MATRIX、CHINESE_SUMMARY、REPRO_README、
  M5_VERIFY_R1902.py（94 项断言前台 EXIT=0）。checker/源码/config 保持可变工具、不哈希冻结。
- **双编译**：正文仍为 9 页（0 error/0 overfull/0 undefined），从 mutable 形成 r1902 单调后继候选。

## r1904 投稿 prep 收口（AI-use 声明 + 引用核验 + 核对表；零实验改动）
- **坐标**：M5-C-content-checked 保持，正文 9 页、M5_VERIFY 105/105 PASS 维持不变。
- **引用核验**（ICLR 逐条要求）：新建 `code/verify_citations_r1904.py` 前台 EXIT=0 = **8/8 OK**。
  - arxiv 两条用官方页面逐字核验：maurer2009bernstein=arXiv:0907.3740（标题由旧 "Penalties"
    更正为官方 **"Penalization"**）、sagawa2019distribution=arXiv:1911.08731 题目/作者/年逐字匹配。
  - 其余 6 条为教科书/期刊登记经典（Biometrika'34/JASA'55/Ann.Stat'84/JMLR'19/NC'02/Springer'05）。
  - references.bib 删去未被引用且标题截断错误的死条目 hsu2016generalized；零虚构引用。
- **AI 使用声明**：paper.tex 附录末新增 `\section{AI use statement}`（LLM 辅助实现/证明核验/稿件
  准备，科学声明归作者）。重编译 12 页 = 正文恰 9 页 + refs p10 + 附录 A-E + AI 声明 p11-12，
  0 err/0 overfull/0 undef，正文未顶出 9 页。
- **核对表**：新增 `SUBMISSION_CHECKLIST.md`（页数/编译/引用/匿名/证明可复现/实验真实性/AI 声明/
  复现材料/剩余待办 各节均逐条勾选）；REPRO_README 补 r1903+r1904 复现命令与产物。
- **诚实边界**：正文 9 页、headline 数字均未因 prep 改变；r1901 两处过度陈述修正与 r1903
  full 5-seed matched 结论原样保留。M3/M3.5 完整 5-seed 复验仍为非阻塞复现项（m3_cache 仅 3 seed）。
  剩余待办 H 项与 Manager/human 迁入正式投稿主稿目录决策保持一致。

## M6（r1906 τ-free 升级门）与 M7（r1907 τ-菜单 CAL-only 选择协议）后续同题增量
- 冻结候选 `paper/A2_subgroup_mix_ranking_author_candidate_r1905/`（M5-C，r1904 状态）已装配且只读；
  以下为 mutable 主稿上的同题 follow-up，不入冻结候选 bytes，写作待并回后续章节。
- **M6**：把冻结结论明示的"τ-selection protocol"做成消除 τ 的**相对 status-quo F0 安全升级证书**
  （`decision(w)=i*(w) iff Σ_g w_g UCB[(i*,F0)][g] ≤ 0，否则 F0`），5-seed REG_sq max=0、no-worse cov 1.0，
  升级率 0.663，naive 恒切有 19% 真变差被门挡住。注记 `THEORY_TAU_FREE_R1906.md`。
- **M7**：保留绝对-τ 语义的**有限 τ 菜单 CAL-only 选择协议，全新 seed {5,6,7,8}**。Prop M7：
  band τ-agnostic → τ 选择零证书多重性代价；选择代价只在性能层。CAL-select CR 0.357, mean_reg 0.0003,
  max_reg 0.0053, coverage 1.0（比冻结默认 0.04 更紧且更安全），test-snooping 臂 CR 虚高 +0.182 且 0.7% 违例。
  注记 `THEORY_TAU_CAL_R1907.md`。
- 三者互补：绝对证书 M2.5（regret≤τ）/ 相对升级 M6（no-worse-F0）/ τ-菜单选择 M7（CAL 选紧 τ 且零证书代价）。
- **r1909 M8 严格有限样本带（收口唯一 remaining="strictly finite-sample tightened band"）**：
  相对门 M6 在严格 Hoeffding/Maurer-Pontil 带下 commit 0.663→0.646（no-worse cov 仍 1.0），但机制为
  **vacuous-collapse**：226 行 i*==F0 平凡 commit，6 个 i*≠F0 真实切换提案（normal 带 certified 且
  OUTER 全 sound）被 exact 带全拒绝。⇒ 相对门的非平凡内容由渐近带承载，严格有限样本使其为空；
  要严格严格性应用绝对门（M2.5 exact MPB 保留 0.260 真内容）。已并入主稿 app:tau 段 + 结论/局限性。
  证据 `results/SUBGMIX_M8_FINITESAMPLE_RELGATE_R1909.json`，verifier `results/M8_VERIFY_R1909.py` 17/17。
- **r1910 M9 N*-frontier（把相对门严格性从"空洞"升级为"预算计价"）**：在有效样本倍增模型
  （固定实证 per-group 统计，band 在 n_g→N·n_g 下估值）下，6 个真实切换提案在 Maurer-Pontil
  exact 带下 N\*∈{2..10}（中位 5）、Hoeffding 下 N\*∈{2..20}（中位 10）全部重新开通，且
  OUTER soundness 在每个 opening 点保持（oracle_switch_gain 0.034–0.130>0，D 单调非增）。
  ⇒ 相对门严格有限样本坍缩是**廉价可逆**的：校准预算放大 ~5–20× 即可恢复其非平凡内容，
  前沿 N\* 可在证书时逐 mixture 测量。已并入主稿 app:tau 段末尾。证据
  `results/SUBGMIX_M9_NSTAR_FRONTIER_R1910.json`，verifier `results/M9_VERIFY_R1910.py` ALL PASS。
- **r1911 M10 relative-gate exact 带的预算非空性前沿（真实采样 + 可证空证书，MGR 指令 7cc94318db8d）**：
  M8/M9 的 vacuous-collapse 在 fresh seed{10..14}、真实按组子采样 CAL（预算网格 b∈{0.25,0.5,1.0}）
  上得到可证空证书：全部 125 真实切换行在满预算 b\*_hoef>1（中位 273/min 3.7，选择器余量 Δ 中位
  0.014），Hoeffding exact 相对门在整个可行预算轴 b≤1 可证地无真实切换内容；exact hoef/mpb 在每
  carrier×预算 admit_real=0.0。卡②严格分报平凡行（digits@0.25 92%、news 43%）与真实切换行，绝不用
  总 commit 率掩盖。**可执行混合规则**（卡③/④）：安全部署用 M2.5 exact 绝对门（fashion@b=1
  abs_commit=1.0·no_worse_cov=1.0，随 b 单调增强 0.49→0.96→1.0；mnist 全程 1.0/1.0），相对门 M6
  只作渐近/描述性诊断；弱域 news=预算墙（0.136→0）/digits=容量墙（0.093）如实保留。理论
  `THEORY_EXACTBAND_FRONTIER_R1911.md`；verifier `results/M10_VERIFY_R1911.py` EXIT=0 = ALL PASS。
  已并入主稿 app:tau M10 段，重编译 13 页（正文恰 9）0 err/0 overfull/0 undef；候选 r1905 只读包不动。

## r1915（M12）补充结论
- fresh seed{10..14} 真采样下，M3/M3.5 预算总点严格相对带（hoef/mpb）**0 真实切换**（100 配置全 admit
  平凡 keep-status-quo；M10/M11 全轴空证书由真实采样复证），绝对门 M2.5 全 37 commit cell coverage 1.0 且
  内容随预算单调（MNIST uniform 0.773→1.0；Fashion 0→0.20@pi0.95）。digits/news 容量/预算墙如实保留。
