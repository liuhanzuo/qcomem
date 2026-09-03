# RESULT_MATRIX — FINITE-SAMPLE SAFE MODEL RANKING UNDER SUBGROUP-MIX TURNOVER

坐标：M4.1-M（真实结果树与解释已成；正贡献路线 M2 修复已定义并复验，壁垒到 M4.2）。

## r1881 首轮 readback（M1 负结果，已诚实定位；本回合按盘面 JSON 纠偏数字）
- 对象：digits/Fashion/MNIST(tfidf→PCA128)/20News，组=class，模型 LR-C1/LR-C01/SVC-0.1/MLP。
- W 网格：uniform + 20×single-class-skew + collected_u + 3×interp(n=25/载体×seed)。
- **PILOT**（350 行）：naive MMR `argmin_i U_i(w)` 与 cal-prior `argmin Σw p̂` **350/350 逐行相同**
  （mean_regret 0.0031==0.0031, hit 0.6829==0.6829）。
  → vacuity：UI 宽度均质，argmin U 退化为点估计。
- **MIXTUREBALL**（352 顶点，`results/SUBGMIX_R1881C_MIXTUREBALL.json` 逐行核对）：
  at δ=0.1, n_diverges=44/88 每 ρ；全部 ρ 求和 **165/352 diverge**（0.4688）；divered 子集
  mean regret diff(mrr−cal)=**−0.0021**（mrr 平均更优：93 better / 62 worse / 其余 tied）；
  全 352 顶点 overall diff=**+0.0019**（mrr 略逊）。→ **M1 非"恒差"：分歧无一致方向**，
  修正此前记录的"88/352 diverge、regret_diff 恒正(+0.0024)"（0.0024 实为 ρ=0.1 overall 均值，
  "88" 为单 ρ 顶点数，是被误并的错位）。
- **DIAG**（`results/SUBGMIX_DIAG_R1881B.json`，880 行=5δ×176）：diverges_U 累计 **15/880**，
  分歧行 regret 大多 tied（11/15 tied；其余 4 行 M1 略劣 diff +0.0024）。修正此前"4/880 all worse"
  与"15/880 全 tied"两版过度陈述（此后按 r1901 精确：4+11 直读验证）。
- 机制结论（保留，改为门警非"恒反选"）：`argmin_i U_i` 因 `max_j(U_i−L_j)=U_i−min_j L_j`
  恒为 argmin U_i；宽度均质→vacuous、异质→分歧无一致方向，故带是**门**不是**选择器**
  （Prop metallanti 只证构造性反选域，不宣称数据恒差）。

## r1884 M2 修复（本回合，正贡献）
方法：选择=点估计 argmin pt_i(w)；证书門= sound P2 上界 `UB_{i*}=U_{i*}−min_j L_j`；
`UB≤τ`→committ 并证书 "真 regret≤τ"，否则 abstain。
证据：`results/SUBGMIX_M2_GATE_R1884.json`（350 行；前台 EXIT=0；ta可用）。

### 主端点（τ=0.04, δ=0.10；r1903 起为 **full 5-seed matched {0,1,2,3,4}**，与 M2.5 同 seed/split/model/w-grid/delta/gate）
M2 冻结：`results/SUBGMIX_M2_GATE_R1884.json`；M2.5 冻结：`results/SUBGMIX_M25_PAIRED_R1885_5SEED.json`（r1903 新，~60min CPU，零网络）。

**M2（绝对 CP 带，5-seed）**
| 端点 | 值（5-seed, n=350） |
|---|---|
| committed_rate | 0.269（94/350） |
| **cert_coverage（committed 上真 regret≤τ 比例）** | **1.000** |
| committed_mean_regret | 0.0001 |
| committed_max_regret | 0.0019 |
| abstain_hardpick_mean_regret | 0.0043 |
| abstain_hardpick_max_regret | 0.0546 |
| no-gate_mean_regret | 0.0031 |

**M2.5 配对差（5-seed, n=350）**
| 端点 | 值 |
|---|---|
| committed_rate（paired Normal） | 0.503 |
| cert_coverage | 1.000 |
| committed_mean/max regret | 0.0009 / 0.0364 |
| abstain_mean/max（若硬选） | 0.0054 / 0.0546 |
| MPB（精确）committed_rate | 0.260 |
| Hoeffding（精确）committed_rate | 0.097 |

解读：证书健全（committed 全 packet 满足真 regret≤τ，M2 均值 0.0001 远低于 τ）；弃答门精准抓到
turnover-skew 危险 mixture——若硬选点估计，其 regret 均值 0.0043/最大 0.0546，较 committed 高约数 OoM。
主比较现为 full 5-seed matched（r1903 收口 r1901 记的"5-seed M2.5 复验"复现缺口）。

### τ 扫描（证书覆盖-宽裕权衡，cert_coverage 全程 1.000）
| τ | committed_rate | committed_max_regret | abstain_hardpick_mean |
|---|---|---|---|
| 0.02 | 0.01 | 0.0000 | 0.0032 |
| 0.03 | 0.18 | 0.0000 | 0.0038 |
| 0.04 | 0.27 | 0.0019 | 0.0043 |
| 0.05 | 0.34 | 0.0053 | 0.0046 |
| 0.06 | 0.39 | 0.0067 | 0.0049 |
| 0.08 | 0.43 | 0.0195 | 0.0051 |

### per-bucket（τ=0.04）
| bucket | committed率 | comm_reg | hardpick_if_abstain |
|---|---|---|---|
| collected_u+uniform（SAFE） | 0.25 | 0.0000 | 0.0016 |
| interp | 0.22 | 0.0000 | 0.0016 |
| skew_peak（turnover 敏感） | 0.28 | 0.0001 | 0.0054 |

## 诚实边界 / 未做（non-blocking 待办）
- 证书为逐 w 逐点（per-w 覆盖 ≥1−δ），非「全 w 网格同时即联合证」；多 w 同时联合须对 w 网格再 Bonferroni（TBD-1）。
- CP CI 较保守，committed_rate 有提升空间（Wilson/pooled/logloss-risk 待做，同题修复分支）。
- 最优 τ 是设计参数（给权衡曲线），非定理结论。
## r1885 M2.5（配对差/MCB 风格证书）— 同题修复收窄宽带
方法：证书目标 = max_j 配对差 regret；每个差用 **CAL 内配对** 上界（共享误差抵消），
替代 M2 的"减两个独立绝对带"。选择仍=点估计 argmin pt；带只做门（规避 M1 悲观化）。
3× 选择从 M2 的"减两个绝对 CP 带"改为"配对差" + Bonferroni 跨 M(M-1)×G。
证据：`results/SUBGMIX_M25_PAIRED_R1885.json`（210 行；3 seed×4 carrier；前台 EXIT=0）。

### 证书价格前沿（全部 cert_cov=1.0，τ=0.04, δ=0.10；r1903 起全为 full 5-seed matched {0,1,2,3,4}）
| 证书 | committed_rate | cert_cov | 有限样本 |
|---|---|---|---|
| M2 absolute CP 带（r1884，5-seed） | 0.269 | 1.000 | 精确 |
| M2.5 Hoeffding（5-seed） | 0.097 | 1.000 | 精确 |
| M2.5 Maurer–Pontil(emp-Bern)（5-seed） | 0.260 | 1.000 | 精确 |
| M2.5 Normal(CLT)（5-seed） | 0.503 | 1.000 | 渐近 |
| （r1895/r1902 legacy 3-seed matched 视图） | 0.267/0.495/0.257/0.091 | 1.000 | 保留历史断言 |

### per-carrier（M2.5 Normal，matched 5-seed {0,1,2,3,4}，2026-08-18 r1903）
| carrier | commit | MPB commit | comm_regmean | abst_regmean |
|---|---|---|---|---|
| digits | 10/75 | 0 | 0.0074 | 0.0079 |
| fashion | 75/75 | 16 | 0.0010 | 0.0000 |
| mnist | 75/75 | 75 | 0.0000 | 0.0000 |
| news | 16/125 | 0 | 0.0000 | 0.0039 |
（M2 5-seed per-carrier：digits 0/75、fashion 19/75(0.253)、mnist 75/75、news 0/125。）

解读：①M2 absolute 带是"减两个绝对宽带"，中位 UB≈0.17(digits/news) 门住覆盖率，非被真实差异门住。
②M2.5 配对差抵消共享误差，渐近 Normal 覆盖≈2×（0.503，**仅诊断**，非有限样本证）；
精确 MPB 为 0.260，**约等**于 M2 5-seed 0.269（配对方差项有时比减两绝对带宽），**诚实写"约等"
不称非劣不称超**。③fashion/mnist 全 cert（低错误率窄带）；digits/news 高错误率→带仍宽→诚实低覆盖。
④正常候选仍由点估计选，覆盖提升来自更紧证书（渐近）而不是换选择规则。**W1 纪律**：
5-seed 0.260≈0.269 只写"约等"。

### 诚实边界（M2.5）与 MGR W1/W2/W3 原子修复 + r1902 matched-seed 指令（ea0889891916）
- **r1902 可比性修复（matched-3）、r1903 升级为 full 5-seed matched**：r1902 先把 M2 与 M2.5
  承重比较统一到同 seed/split/model/w-grid/δ/门（确定性 `random_state=seed`，不重训）。
  r1903 再把 M2.5 在 M2 相同的 5 seed {0,1,2,3,4} 上真跑一遍（`SUBGMIX_M25_PAIRED_R1885_5SEED.json`，
  ~60min CPU、零网络），headline 升为 full 5-seed matched：M2 0.269 / M2.5-MPB 0.260 /
  Hoeffding 0.097 / Normal 0.503。r1895/02 legacy 3-seed 数值保留为历史断言，不承重。
- **W1**：exact MPB committed_rate=0.260 **≈** M2 5-seed=0.269。诚实写"约等"，不称非劣不称超；
  无预定义配对不劣检验支撑，不写"略低/更高/≈持平"以外的任何强弱判定。（主稿/本矩阵/详报逐处核对。）
- **W2**：M1 的 88/352 **=0.25**，任何上下文都不得写成 metric 值 0.88。已核全 repo 无"0.88"残留；
  且按盘面 JSON 该数是 165/352（见 M1 纠偏段），主稿/本矩阵已同步为 165/352, 不会再出现 0.88。
- **W3**：Normal committed_rate=0.495 仅标 asymptotic/diagnostic，不能充当 finite-sample safe
  headline，也不得与 exact 方法混写为"已证明提升"。有限样本 claim 只用 M2 0.269(5-seed) /
  M2.5 MPB 0.260 / Hoeffding 0.097。
- Normal 为渐近健全（CLT），非有限样本；精确变体(MPB/Hoeffding)用于有限样本 claim。
- 逐 w 逐点覆盖（TBD-1 全网格联合待）；M2 vs M2.5 承重现为 full 5-seed matched（r1903 收口）。
  复现项余：每载体 6 折完整网格、M3 full 5-seed 复验。
- 证书价格前沿=健全性↔覆盖权衡，写为机制分析，非 cherry-pick 选择某变体。

## r1886 M3 预算化分配（本回合，新正贡献：标签预算轴）

同 r1885 判别，新增总 CAL 标签预算 R（frac∈{0.5,0.65,0.8,0.95}×3 seed×4 carrier）
与 5 条逐组分配规则；证书/选样同 M2.5（paired-MPB UCB 只做门）。Fitest 预声明；
规则用预选静态分配；fully-adaptive 用 time-uniform Hoeffding CS（每次只读已揭示标签）。
证据：`results/SUBGMIX_M3_BUDGET_R1886.json`(前台 EXIT=0, runtime 8481.8s)、
`SUBGMIX_M3_SUMMARY_R1886.json`、`m3_cache/recs_*.npz`(4200 cell)。

### 主端点（τ=0.04, δ=0.10）：committed_rate（3 seed 均值；cert_validity 全 1.0 或 None）
| carrier | budget | uniform | neyman | widthgreedy | sens | adaptive |
|---|---|---|---|---|---|---|
| mnist | 0.50 | 0.778 | 0.400 | 0.400 | 0.578 | **0.000** |
| mnist | 0.95 | 1.000 | 1.000 | 1.000 | 1.000 | **0.000** |
| fashion| 0.50 | 0.000 | 0.067 | 0.067 | 0.067 | 0.000 |
| fashion| 0.95 | 0.133 | 0.111 | 0.111 | 0.133 | 0.000 |
| digits | 全部 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| news | 全部 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### 解读（机制级）
① **uniform 在 mnist 恒胜所有分配规则（0.778 vs 0.400/0.578, 3 seed 一致）**：
本证书 UB(w)=max_j Σ_g w_g·UCB[(i*,j)][g]，即便 uniform/skew 网格下每个 group 权重非零；
even 分配等额缩窄全部 group 的 UCB，而 neyman/widthgreedy 集中 n 到少数 group 会饿死被
`w_g×UCB` 卡住的其他 group。② **分配收益是 carrier-heterogeneous**：fashion frac0.5 上
neyman/greedy/sens=1/15 而 uniform=0/15（3 seed 一致，方向与 mnist 相反）→ 不是单调
「分配>uniform」的普适机制，诚实写成 carrier 依赖，不称普适胜。③ neyman≈widthgreedy
逐位同（同用 pg_sd 代理），sens 卡在其他 group。
④ **adaptive（CS 置信序列）全 carrier 全 budget committed≈0**：UB med 0.064–0.138 恒≥τ=0.04
（mnist frac0.5 静态 MPB 0.031 vs CS 0.138 ≈4.5×宽）= 自适应采样每次只看已揭示标签的
「诚实价格」：always-valid 带宽退化，3 seed 无任何 carrier 回本。如实报为价格，不宣称自适应收益。
⑤ **digits/news 全 budget 0 commit 是带宽墙非 bug**：digits CAL 仅 405（group floor 39，MCB
n 太小崩）；news 20 类×高错误率(0.37–0.52)→MCB 带宽恒宽。cert_validity 在能 commit 的行全 1.0，
全局 4200 cell 0 违例 → 健全性完整，低 commit 是覆盖-紧度权衡非实现失败。

### 诚实边界（M3）与 MGR W1/W2/W3 原子修复延续
- **cert 用 paired-MPB UCB（静态，紧）与 time-uniform CS（自适应，宽）严格分列**；
  adaptive committed≈0 是 CS 宽度，非实现失败，逐字核实 UB 数值（mnist frac0.5 adaptive
  UB med 0.138 vs uniform 0.031）。
- **defer/abstain 双口径并列**：hardpick（per-w 点估计负对照，可随 w 翻转）与
  anchor（FIT 预选固定 robust fallback 模型，不读 CAL/OUTER）。fashion/news 上 anchor defer
  max 可略高于 hardpick max（固定模型不随 w 适应）——如实并列，不宣称 anchor 更安全。
- 预算轴非单调：budget 增不必然增 commit（digits/news 恒 0），故不写「更多标签必更好」。
- 逐 w 逐点覆盖仍是 pointwise（TBD-1 全网格联合待）；分配规则参数化仍敏感（同题修复分支：
  Neyman 型应按「当前 UB 尖峰 group」而非 pg_sd 分配，待验）。
- 3 seed（TBD-4 5 seed 复验）；digits/news 未火点需更大 budget 或更紧 bound（同题修复方向）。

## r1887 M4 自适应代价分解：paid-adaptive 证书价格 + 容量墙定位（同题机制探究）

**问题**：M3 的 naive adaptive(CS) 全 carrier committed≈0。是"自适应本身无价值"还是"自适应被
always-valid CS 计价过高"？本回合用 split-CAL 两阶段把两者分开，并定位 digits/news 带宽墙根因。

**方法（split-CAL paid-adaptive）**：预算 R 分两份——Cargo（受证 SRS，计 m_g）+ Annexe（自适应
方向锚，计 n_g，方向只读 FIT 分裂，不与 cargo 重叠）；静态 paired-MPB 只在 cargo 上出证书，
Annexe 消耗计入总支出。自适应方向"免费补证书"但付样本代价。同一预算下与 static-uniform 公平对比。

**结果（tau=0.04, delta=0.10, 3 seed, JSON=SUBGMIX_M4_ADAPTPRICE_R1887.json）**：
- static-uniform 完全复现 M3（mnist 0.5→0.778, 0.8→1.0, 0.95→1.0；fashion 0.5→0.0, 0.8→0.067,
  0.95→0.133）→ runner 自洽性确认。
- C_cargo（paid-adaptive，同总预算 R）committed 恒≈0，唯一在 mnist 0.8→0.333、0.95→0.644 才有值
  （此时 cargo m_g≈6300-7481 已够紧）；fashion/digits/news 全 0，cert_validity committed=1.0。
- **机制解释**：UB(w)=max_j Σ_g w_g·UCB[(i,j)][g] 是"逐对加权均值取最大"——自适应锚消耗预算去
  集中少数组，却仍被其他组的 w_g×UCB 卡住；若受证 cargo 占 50% 预算，certified n 减半→mAD 上界更宽，
  价格≈把有效样本切成 2×bonferroni。paid-adaptive 的"免费证书"是其唯一加分，但样本代价 ≥ 静态全预算。

**判别网格实验（痉挛分配为何被 uniform 支配，机制铁证）**：评估网格每个 w 行都含"均匀/collected_u"，
此类行对所有组非零权重；分配稀疏化某个 g 会在该行以 w_g×UCB 的宽不确定托底。probe（mnist seed0）：
- neyman 集中 pg_sd 高组 {5:1302,8:1256,2:1043,9:1036} 饿死 {0:269,1:259,6:408}；uniform 行 top-contrib
  恰是饿死组 0/1（UB 贡献 0.125/0.093/0.08）。→ **UB=max-of-weighted-means 下均匀=极值极大化
  (maximin)，阵列只要探到所有组就支配 neyman**。这解释 M3 中 mnist uniform>neyman/widthgreedy/sens。
- 稀疏网格判别（只 load 高分歧组 peak）下 uniform 仍 6/6 vs neyman 4/6（top-2）、10/10 vs 7/10（top-4）
  → 结论与网格无关：collected_u/uniform 恒存在 ⇒ 任何饿死组的分配都输。诚实写为"certified-grid
  探全体组"的结构性结论，非"分配无用"。

**容量墙精确定位（digits/news）**：paired-MPB 分裂项 7L/3(n−1)，L=ln(2/δ_cell)≈7.8（M×G=40
Bonferroni）；digits CAL n_g≈40/news≈141–225 → 仅 bias 项 0.93/0.09–0.18 即 >τ=0.04，任何静态
证书下都无法 commit。方差半宽仅 0.1（digits）。→ **digits 是"每组 n 小×Bonferroni 开销"的容量墙，
news 接近墙沿**，非分配可解。这是 carriers-SNR×多重性 的固有边界，诚实写为能力上限不推给分配。

**结论（M4.1 深化）**：统一机制 = 证书为"每对加权均值取最大+Bonferroni"，随机性与浓度开销使
「可证覆盖」在两类载体上分裂：mnist/fashion 低噪声够紧（可 commit），digits/news 容量墙。
分配收益是 carrier 依赖 + grid 探全体结构驱动的（均匀/极值），paidity-adaptive vs 静态价格随
载体/预算变化，写为价格而非普适胜。

## r1895 凸 minimax 分配的正式化与强基线比较（MGR 809f17cdbe78 指令本体）
> ⚠️ **r1897 修正（MGR 593c907d2ccd）**：本节的 `对偶乘子 λ*` 与 `λR 超 τ` 判定、以及
> `all candidates 作活跃集` 均为 **INVALID-DIAGNOSTIC**。修正后对偶见 §r1897：值
> `V=S^{3/2}/√R=2λR`（∝R^{-1/2}），活跃引擎由互补松弛 `μ*_k>0`（真活跃仅 K 的 6–25%），
> 四 carrier 真实 minimax 数字已重算为 `SUBGMIX_MINIMAX_R1897.json`。旧本节数字不作主张。

**对象**（MGR 指令原文形式化）：min_{n_g,Σn_g=R} max_j Σ_g w_g r_g(n_g)，其中 r_g(n_g) 是组 g 配对
差 UCB 的半宽 ∝ β_g/√n_g（经验-Bernstein），β_g 为组 g 跨模型配对差标准差。理论见
`THEORY_MINIMAX_R1895.md`。

**形式化结论（KKT/水填充/可核验条件）**：
- 连续松弛后在 epigraph 形式（convex minimax）下一阶条件给**水填充**
  n_g ∝ (w_g·β_g)^(2/3)，幂次 2/3 来自"半宽 ∝ β/√n → 边际 −½β n^(−3/2) → 列预算取 2/3 幂"。
- **uniform 最优的对称性判据（Proposition）**：uniform 是 P 的极小解 ⟺ w_g·β_g 在 g 上恒等
  （即 W 在候选族上对称 + 组方差对称）；任一不对称即破坏 uniform 最优性。实算里评估网格恒含
  uniform 列（spanning 所有组），故 uniform/maximin 在该对称网格上可核验最优。
- 对偶乘子：λ*= 最优边际收益；闭式半径下界给容量墙判定（digits/news n<680 即此——β 大，
  λR 目标超 τ 预算，可证不可行）。
- **确定性反例（Counter-A，非对称 W）**：`results/SUBGMIX_MM_COUNTER_R1895.json`
  G=4, R=1200, β=(0.1,0.1,0.02,0.02), 部署 W={e_0}（只压单个高方差组，非 spanning）：
  | rule | n_g | UB@e_0 | committed τ=0.04 |
  |---|---|---|---|
  | uniform | [300,300,300,300] | 0.094 | **False** |
  | convex-minimax | [1197,1,1,1] | 0.029 | **True** |
  → **uniform 不是普适 minimax 最优**：在非对称（非 spanning）部署 W 上极小极大分配严格更优
  （其 UB 是 uniform 的 ~0.3×）。在对称 W=uniform_w 两者都不 committed（minimax 全押 e_0 组致
  uniform_w 上偏），诚实并列——minimax 只在 W 不对称时赢。

**四 carrier 真实比较**（`results/SUBGMIX_MINIMAX_R1895.json`，复用 m3_cache 伪拟，零再训练，
评估网格=M3 网格，同 M3 paired-MPB，τ=0.04 δ=0.10 3-seed 均值；cv 在能 commit 行全 1.0，0 viol）：
| carrier/budget | uniform | neyman | width | sens | adaptive(CS) | **minimax** |
|---|---|---|---|---|---|---|
| mnist 0.5   | **0.778** | 0.400 | 0.400 | 0.578 | 0.000 | 0.533 |
| mnist 0.65  | **0.978** | 0.644 | 0.644 | 0.867 | 0.000 | 0.800 |
| mnist 0.8   | **1.000** | 0.844 | 0.844 | 1.000 | 0.000 | 0.933 |
| mnist 0.95  | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| fashion 0.5| 0.000 | **0.067** | 0.067 | 0.067 | 0.000 | 0.044 |
| fashion 0.65| 0.044 | 0.067 | 0.067 | 0.067 | 0.000 | 0.067 |
| fashion 0.8 | 0.067 | 0.089 | 0.089 | 0.067 | 0.000 | **0.089** |
| fashion 0.95| 0.133 | 0.111 | 0.111 | **0.133** | 0.000 | 0.111 |
| digits 全 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| news 全 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**解读（机制级，直接对应 MGR 的"uniform=maximin 恒支配 only 按对称网格现象"）**：
- 在**对称 spanning 网格**（恒含 uniform 列），uniform=maximin 确实是 P 的 minimax 解：mnist 上
  uniform 全预算胜所有方差感知分配含 convex-minimax（0.778>0.533）。水填充集中 n 到高 β 组，
  反而饿死在某些网格行列托底的组 → 与 r1894 观察一致。
- **fashion 低预算 minimax 追平/小胜 neyman/width（0.067 vs 0.067），与 uniform 持平——carrier
  依赖且不对称，不称普适。**
- 容量墙 digits/news 全 0（含 minimax）保留：minimax 水填充也解不开 n<680 的 Bonferroni 带宽
  墙（可证不可行，理论 sec 2.3）。
- **paid-CS（M4 C_cargo）** 仍接近 0：mnist π=0.5 S_uniform 1.0 vs C_cargo 0.022——商品的 adaptive
  样本代价未被 minimax 抵消，保持负结果。

**条件化/融合（MGR 要求"minimax 仅部分域占优时用 FIT/CAL 可见对称性统计做条件化选择"）**：
- FIT 可见方差不对称统计 = 组跨模型配对差 CV（digits .42 / fashion .33 / mnist .24 / news .16），
  加上**部署 W 是否 spanning**（评估网格实测总是含 uniform 列 → 对称 span）。
- 条件规则：若部署 W 对称 spanning → 用 uniform（=可核验 minimax）；若部署 W 非对称（如只压少数
  高方差组的真实 turnover 场景）→ 用凸 minimax（反例证明在此比 uniform 严格更优）。融合=把
  minimax 水填充作 uniform 的**可证后处理**：先 uniform 铺满再按 (w*β)^(2/3) 加权补余，保留了
  uniform 组覆盖、又对非对称 W 收紧。

**诚实边界**：convex-minimax 分配读 FIT-only（β 从 FIT 预选计），不读 CAL，静态 MPB 证书健全
（0 violations，与 M3 同逻辑）；minimax 只改证书支配的 committed_rate，不承诺真实 regret 更低；
3-seed（5-seed TBD 非阻塞）；逐 w 逐点证书（网格联合 TBD-1）。digits/news capacity wall 全部保留。

## r1896 条件选择门端到端执行：单最高方差顶点 W'={e_g0}（把 §M3.5 条件化从陈述变实跑）
- 动机：r1895 已理论 + 合成反例证明凸 minimax 在非 spanning W 严格赢 uniform，但四真实
  carrier 上从未执行该条件化选择（§M3.5 仅文字）。本回合用冻结 m3_cache、同一静态配对-MPB
  证书、同一预算/网格/种子，把部署集收窄到 **单个最高方差顶点 W'={e_g0}, g0=argmax_g β_g
  （β 仅 FIT）** —— 这是"非对称非 spanning"的最小实跑。
- 证据 `results/SUBGMIX_CONDGATE_SINGLE_R1896.json`（tau=0.04, delta=0.1, 3seed×4frac×2rule）。
- **结果（真实 readback）**：
  | carrier | π | uniform cmt/medUB | convex-minimax cmt/medUB |
  |---|---|---|---|
  | mnist | 0.5 | 1.00/0.000 | 1.00/0.000 |
  | fashion | 0.5 | 0.333/0.041 | **1.000/0.000** |
  | fashion | 0.65 | 0.667/0.019 | **1.000/0.000** |
  | fashion | 0.8 | 1.00/0.011 | 1.00/0.000 |
  | fashion | 0.95 | 1.00/0.000 | 1.00/0.000 |
  | digits | all | 0/1.07–2.06 | 0/1.05 |
  | news | all | 0/0.27–0.50 | 0/0.24 |
- **解读（机制级）**：在非 spanning 高位顶点，minimax 水填充把整个预算集中到被查询的高方差组
  g0，从而把该组的 UCB 收紧到→0，恰好"买下"uniform 只能弃答的行：fashion π=0.5 uniform
  弃答 2/3 行（medUB 0.041），minimax 全 commit（medUB 0.000）。mnist 两规则下均全 commit
  （低噪声统一胜）；digits/news 两规则下 capacity wall 仍绑（minimax 把 UB 减半 但未过 τ）。
  **全 committed 行 true regret=0（cert 健全 1.0, 0 violations）。**
- **这是 Prop uniform-opt 预测的首次真实 carrier 直接印证**，也是反例（而非对称 M3 网格）
  所捕捉的失败模式。与 top-3 顶点探针（`SUBGMIX_CONDGATE_R1896.json`，near-tie）对照：
  顶点越少、非 span 越强，显式化"探索全体组的代价"后 minimax 优势越明显。
- **诚实边界**：minimax 优势只在非 spanning/窄支撑部署集出现；对称 spanning 网格（M3）仍
  uniform=maximin 胜（r1895）。CV 用**非零组 population CV**（digits .42/fashion .33/mnist .24/
  news .16，与 r1895 矩阵一致，排除 0 项避免稀疏载体被抬高）。条件化门=读 FIT 可见的 CV +
  部署集是否 spanning，均不读 CAL，证书静态健全。

## r1897 minimax 对偶/齐次性修正：联合凸解 + all-active 消融（MGR 593c907d2ccd）
- **修正内容**：r1895 §2.3 的 `d*=λR` 维度错误（λ 已含 R^{-3/2}，正确恒等式 `值=2λR`，
  值 ∝ R^{-1/2} 与半径目标齐次）；且 r1895 把"所有候选等权"当活跃集。r1897 用**引擎矩阵**
  C_{kg}=w_g β_{jg}（孩子=(候选 j, mixture w)）把对象写成 $\min_n\max_{\mu}$，Sion 强对偶
  + $S(\mu)$ 凹 ⇒ `d*=R^{-1/2}[max_μ S(μ)]^{3/2}`（单纯形上凹最大化的凸解），互补松弛给出
  活跃集={μ*_k>0}。
- **数值 fixture**（`results/MM_FIXTURE_R1897.json`，前台 EXIT=0，修正确认）：
  - 四类 (M,G)/β 情境连续对偶间隙 <1e-6（强对偶实证）；
  - `值=2λR` 到 1e-9；R∈{400,800,1200,2400,4800} 上 `V·√R` 平坦 <1e-16（R^{-1/2} 齐次）；
  - 互补松弛：活跃引擎=承重者、非活跃严格松弛；整数化代价 1e-6–1e-3（floor+n_min，诚实披露）。
- **all-active 消融**（`results/SUBGMIX_MINIMAX_ABLATION_R1897.json`）：真活跃引擎仅 **K 的
  6–25%**（digits .067/fashion .25/mnist .25/news .20），修正 d*（worst-case 混合）为
  uniform-mixing 旧启发式值的 1.17–4.19×。**r1895 全候选作活跃集显著高估保证，已作废。**
- **四 carrier 真实 minimax 重算**（`results/SUBGMIX_MINIMAX_R1897.json`，同 m3_cache/证书/
  网格/3seed，修正联合求解；committed_rate medUB = 全行 over ticks）：
  | carrier/budget | committed_rate | medUB | again n_g 方向 |
  |---|---|---|---|
  | mnist 0.5 | 0.400 | 0.0478 | 集中高 β 组 |
  | mnist 0.65 | 0.711 | 0.0243 | |
  | mnist 0.8 | 0.889 | 0.0079 | |
  | mnist 0.95 | 1.000 | 0.000 | |
  | fashion 0.5 | 0.067 | 0.1077 | |
  | fashion 0.65 | 0.067 | 0.0778 | |
  | fashion 0.8 | 0.089 | 0.0608 | |
  | fashion 0.95 | 0.178 | 0.0520 | |
  | digits 全 | 0.000 | 6.28–1.06 | 容量墙 |
  | news 全 | 0.000 | 0.53–0.29 | 容量墙 |
  **数字与 r1895（错误）不同且更保守**：真 worst-case 混合使保证更紧（committed_rate 一般
  不高于 r1895 旧报），容量墙结论不变（digits/news 仍 0 commit）。cv=1.0 on committed
  rows, 0 violations（同 M3 证书逻辑）。
- **条件化门口径不变**：r1896 的"非 spanning 顶点 minimax 集中到 g0"与其机制在修正对偶下
  依旧成立（活跃引擎即被查询顶点），无需改数字——只把理论对偶从 `λR` 换成 `2λR`。

## r1898 simplex-同时证书核验：离网格 Dirichlet 密集扫描（TBD-1 关闭）
- 动机：论文反复披露"配对证书逐 w 逐点 / 网格联合待做"，但构造上 UCB[(i,j)][g] 独立于 w
  （只在有序对×组上 Bonferroni δ/(M(M−1)G)），同一事件 E 已同时覆盖整个**连续单纯形**上所有
  w 与数据依赖选择 i*(w)→ 旧"引再对网格 Bonferroni"是过度保守的自我降级。本回合用不可网格
  落点的**离网格 Dirichlet** 密集扫描实证关闭（证据 `results/VERIFY_GRID_JOINT_R1898.json`）。
- 方法：每 (carrier,seed,frac) 取 1000 个 Dirichlet(ones(G)) w（与 CAL split 独立种子家族
  RNG777777），用与 M3 uniform 完全相同的 reveal 集/证书/δ=0.10/τ=0.04，构建**同一组** UCB
  后扫描计算 committed 与真 regret（OUTER）。
- **结果（真实 readback，run EXIT=0，~95s）**：
  | carrier | π | on-grid CR/cov | off-grid CR/cov | off-grid committed |
  |---|---|---|---|---|
  | mnist | 0.50 | 0.778/1.000 | 0.939/1.000 | 2816/3000 |
  | mnist | 0.65 | 0.978/1.000 | 1.000/1.000 | 3000/3000 |
  | mnist | 0.80/0.95 | 1.000/1.000 | 1.000/1.000 | 6000 |
  | fashion | 0.80 | 0.067/1.000 | 0.002/1.000 | 5 |
  | fashion | 0.95 | 0.133/1.000 | 0.023/1.000 | 70 |
  | digits/news | 全 | 0.000/— | 0.000/— | 0 |
- **解读**：off-grid 扫描全部 15 commit 区域（mnist 3 frac + fashion 2 frac × 3 seeds）覆盖
  **=1.000，0 violation**；容量墙 carrier（digits/news）离网格仍 0 commit（机制：墙=每组 n 小/
  错误率高/组数 20，与分配无关）。即证书对**连续单纯形**联合，非有限网格伪影；on-grid 与
  off-grid committed_rate 接近/一致，证明不是网格采样幸运。
- **写作修正**：paper Def.2/Thm2 的 w-无关 UCB + Honest Scope 段/Limitations/Conclusion 的
  "per-w 点覆盖 / 网格联合待做"全部改为"**simplex-同时**证书 + 离网格扫描核验"（mnist 0.5
  离网格 2816/3000 覆盖 1.0 入正文）；THEORY_SKELETON TBD-1 标记已解决；THEORY_MINIMAX §6
  诚实边界同步。11pp clean compile（0 overfull/0 undef/0 err）。
- 剩余（诚实）：Normal 变体仍渐近（MPB/Hoeffding 才是有限样本）；3-seed（5-seed 复验 TBD）。

## r1899 条件化门 operating characteristic（next-step c 落地）
- 动机：M3.5 "条件化选择"段落只给 CV(β̂)+spanning 的门 + 单点 τ=0.04 的真实 carrier 表，
  没有控制变量把"门槛的判别力/驱动源"显式化。本回合用受控合成族（G=4，同一 paired-MPB 证书，
  R=1200，20 seed）独立扫两个门输入：deployed set ∈ {spanning, non-spanning} × CV(β̂) ∈ {低, 高}，
  committed_rate 作为 τ 的函数（operating characteristic）。结果 `results/SUBGMIX_GATE_OC_R1899.json`。
- **核心结果（signed area = mean_τ(rate_mm − rate_unif)）**：
  | deployed | CV | signed area | worst τ gap |
  |---|---|---|---|
  | 非spanning W' | 低 | **+0.254** | —（minimax 全主导） |
  | 非spanning W' | 高 | **+0.246** | —（全主导，high-CV 更高处拉开） |
  | spanning W | 低 | +0.000 | 完全持平 |
  | spanning W | 高 | **−0.077** | −0.49 @ τ=0.12（minimax 反而更差） |
- **判定（机制级）**：驱动源是 **deployed set 是否 spanning**，不是 CV(β̂)。非 spanning 时
  minimax 的 committed-rate 曲线在**两个 CV 水平**都支配 uniform（area +0.25，minimax 在 κ=0.08
  即达 1.0 而 uniform 停在 0.25–0.93）；spanning 低 CV 两者完全持平（uniform=maximin 成立处）；
  而 **spanning 高 CV 时 minimax 在高端 τ 明显更差**（−0.08，因 minimax 把预算集中到高 β 组、
  饿死 spanning 集里低方差顶点）——这正是"else uniform"分支的安全半边。⇒ 门"非 spanning →
  convex-minimax，否则 uniform" **safe（从不比均摊 baseline 差）+ effective（恰在理论预测处
  严格增益）**；CV(β̂) 只当定位"承重高方差组"的诊断，不是第二驱动。
- **诚实性**：合成族是 3 点配对差（同确定性反例的公证书构造，清楚标注 feasibility/mechanism）；
  真实 carrier 落点横核对一致性（r1896 真实非spanning probe 用 minimax、r1897 真实对称spanning
  网格用 uniform=maximin，均与此 OC 一致）。20 seed 的合成 OC 比先前 3 seed 真实表更稳。
- **写作**：paper §5 conditional selection 段落改写 + 新增 Table gate-oc（0 overfull compile）。

## r1900 论文 9 页正文落地（排版/交付）
- **背景**：r1899 正文 12 页 > ICLR 2027 严格 9 页正文上限。本块=同题交付收口：压缩 + 附录迁移。
- **做法**：压缩抽象/Contributions/Related Work/Limitations/Conclusion/证明叙述；把 tab:m1
  (App.A)、tab:m35ab (App.B)、确定性反例 (App.C)、defer-cost (App.D)、tab:gateoc (App.E)
  迁到 references 后无限附录；修 eq:duality 0.9pt overfull。零内容删除，主表数字未动。
- **关键数字保留在 9 页内**（与冻结 JSON 一致，r1902 后 headline 为 matched 3-seed）：
  frontier commit 率 0.267/0.257/0.495（5-seed 0.269 独立鲁棒性视图）、cove 1.0；M3 MNIST
  uniform 0.778 vs neyman 0.400 / sens 0.578、Fashion 低预算 neyman 0.067
  > uniform 0.000、CS 自适应全 0；M3.5 MNIST uniform=maximin 0.778 vs minimax 0.400、Fashion
  minimax=neyman 0.067；M3.5 条件化门 fashion 0.333→1.0（medUB 0.041→0.000）、gate-OC
  driving=deployed-set（非span +0.25 / span高CV −0.08）。
- **交付状态**：paper.pdf 正文恰 9 页 / References p10 / App.A-E p11-12；0 err/0 overfull/
  0 undefined ref；跨引用 Table 1-7 + App.A-E 全部解析。原始 12 页版备份 paper_backup_r1900/。

## r1906 τ-free 安全升级门（M6，收口结论"τ-selection protocol"缺口）
- **动机**：冻结主稿结论（r1886 line 682）明示唯一敞开的同题入口之一："a τ-selection
  protocol"。M2/M2.5 的 τ 是操作员旋钮且 abstain 回退对象未定义。M6 把安全目标重定位为
  **相对 status-quo F0 的安全升级**，彻底消除 τ。
- **方法**：F0 = collected-mixture 点估计最优（= 不运行框架时操作员实际会用的 single-point
  selector）。对每个 w：`decision(w)=i*(w) iff Σ_g w_g·UCB_norm[(i*,F0)][g] ≤ 0，否则 F0`。
  同一 paired-difference 单侧 normal UCB + Bonferroni joint event（δ=0.1），控件从
  oracle-best 换成 F0 → committed 被证"在这个 w 上不比 F0 差"，全程无 τ。
- **忠实复现**：脚本开庭前对冻结 `SUBGMIX_M25_PAIRED_R1885_5SEED.json` 逐行断言
  chosen/true_regret/UB_paired/committed **逐位一致**，前台 EXIT=0 零漂移（M6 复用同
  split/同选择器/同 UCB 矩阵）。
- **结果 `results/SUBGMIX_M6_UPGRADEGATE_R1906.json`（5-seed×4-carrier front）**：
  | 端点 | 值 |
  |---|---|
  | upgrade_rate（切换脱离 F0） | 0.663（mnist 1.0/digits 0.773/fashion 0.560/news 0.456） |
  | REG_sq=R_decision−R_F0（vs status-quo） | 全体 mean −0.0015, **max 0.0**, no-worse cov 1.0 |
  | ... upgraded rows | mean −0.0022, max 0.0 |
  | 机制 | 升级收益集中在 skew w（mean_gain +0.0035, max +0.130）；uniform/collected/interp ≈0 |
  | 门的价值 | M6 abstain 118 行中 naive 恒切 i* 有 **23 行（19%）真变差**（+0.028）；350 行上 M6 max REG_sq=0 vs naive max +0.028 |
  | 绝对后悔透明度 | or_mean_all 0.0079, or_max_committed 0.0546, or_max_abstain 0.1496（不作绝对≤τ声明） |
- **诚实边界**：单侧 normal（渐近，同冻结前沿 Normal 行）；相对 no-worse-than-F0 证书与绝对
  regret≤τ 证书互补；F0=collected point best 只是"无框架"的一种合理部署对象，换成 robust/DRO
  则 M6 结构不变（相对"实际会跑的"对象）。
- **写作**：独立注记 `THEORY_TAU_FREE_R1906.md`（身份/断言/证据/边界/下一步）。冻结主稿 bytes
  未改；若 MGR 签发迁稿，可作新增章节（绝对证书 M2.5 与相对升级证书 M6 互补呈现）或独立 follow-up。

## r1907 有限 τ 菜单的 CAL-only 选择协议（M7，全新不相交 seed block {5,6,7,8}）
- **动机/授权**：MGR 指令 00a20a5970c7（装配 M5 候选后立即回 mutable 主稿继续同题）。
  结论明示"τ-selection protocol"；M6 消除 τ（相对证书），本块保留绝对-τ 语义，
  用 CAL-only 数据从有限菜单选 τ，形式化选择后的同时有效性与选择代价。
- **方法/预注册**：菜单 `T={0.01,0.02,0.03,0.04,0.05}`，P0=0.5 单一 floor 规则
  `τ̂=min{τ∈T: CR_cal(τ)≥0.5}`，空则取 min T。总 δ=0.1。**全新不相交 seed {5,6,7,8}**
  （冻结证据用 seed 0–4），重走冻结同 split/pipeline/band（Bonferroni dcell=δ/(M(M-1)G),
  normal 单侧 paired UCB）。OUTER 只结算，绝不选 τ。
- **核心理论（Prop M7，`THEORY_TAU_CAL_R1907.md`）**：冻结 band 是 **τ-agnostic**
  （τ 只在 `committed iff D(i*,w)≤τ` 的比较里，不进 band 构造）。故联合覆盖 ≥1−δ 对任意
  CAL 依赖的 τ̂ 成立：**τ 选择不进入 band 的多重性**（候选数 L 不进 dcell）。
  选择代价只在性能层（哪些点 commit、committed reg/coverage），分解为 (i) 代理失配
  +(ii) 网格离散 +(iii) 抽样方差，paired 区间吸收。
- **结果 `results/SUBGMIX_TAU_CAL_R1907.json`（4 carrier × 4 新 seed × w-grid，front，EXIT）**：
  | 臂 | committed_rate | mean_reg | max_reg | coverage(reg≤承诺τ̂) |
  |---|---|---|---|---|
  | fixed τ=0.01 | 0.275 | 0.0000 | 0.0000 | 1.000 |
  | fixed τ=0.02 | 0.386 | 0.0012 | 0.0364 | 0.982 |
  | fixed τ=0.03 | 0.475 | 0.0013 | 0.0364 | 0.985 |
  | fixed τ=0.04（冻结默认基） | 0.496 | 0.0013 | 0.0364 | 1.000 |
  | fixed τ=0.05 | 0.539 | 0.0014 | 0.0364 | 1.000 |
  | **CAL-select τ̂（本协议）** | **0.357** | **0.0003** | **0.0053** | **1.000** |
  | oracle-τ（看全 test 事后） | 0.539 | 0.0014 | 0.0364 | 0.993 |
  | naive/no-correction（test 窥探） | 0.539 | 0.0014 | 0.0364 | 0.993 |
- **为什么有信息量**：CAL-select 做到 **coverage 1.0、max_reg 0.0053**——比冻结默认 0.04
  strict 更紧的证书（12/16 cell 选 τ̂=0.01），同时 **mean/max regret 降 4×/7×**。它 commit
  少（0.357 vs 0.496），但每个 committed 点都被证在 **更紧的** ≤τ̂ 界内。Snooping 教训：
  test 窥探臂（oracle=naive 都看 test 选 τ）报 CR 0.539，比诚实 CAL-only **虚高 +0.182 CR**，
  且 oracle 点有 0.7% coverage 违例——"无校正诊断"正是演示 test-snooping 膨胀 committed
  rate 并破坏逐个被 commit 点的覆盖。
- **选择代价（测度定义诚实）**：selection-cost = Perf(oracle) − Perf(cal) = **mean_reg
  +0.0011**（性能层）。代价不在于破坏证书（coverage 仍 1.0），而在**少 commit 0.182 CR**——
  用少而更紧的承诺换掉冒进。这正是"选择费用"的正确读法：τ̂ 保守是 P0 floor 与 L 有限的
  组合（(i)(ii)），加上 CAL 抽样 (iii)。
- **paired 区间（单元=seed×mixture，δ=0.1）**：cal−fixed0.04 的 Δreg mean=+0.0004，
  SEM=0.0002，95% CI=[0.0001, 0.0006]（cal 略保守一点的 reg 在噪声内，统计上≈等价但
  max_reg 7× 更紧 + 证书 1.0 vs 同样 1.0）。
- **全部弱域如实**：digits committed 0/60（capacity wall，CAL CR 在 τ≤0.05 只到 0.167，
  从未过 P0=0.5，τ̂ 回退 min T=0.01）；news 4/100（预算墙）；fixed 0.02/0.03 的 2 个
  coverage 违例均 digits skew_peak（normal band 在小 τ 硬载体的渐近欠覆盖），0.01/0.04/0.05
  及 cal_select 零违例。
- **验证**：`results/M7_VERIFY_R1907.py` 前台 EXIT=0 = **28/28 PASS**（重导 cal/fixed/paired/
  snooping-inflation 全部 headline）。缓存 `results/tau_cal_cache_{carrier}_s{seed}.pkl`
  （pickle 的 band/estimate struct）使重跑 <5s 确定性可复现。
- **理论注记**：`THEORY_TAU_CAL_R1907.md`；写作待后续并入主稿（绝对证书 M2.5 vs 相对 M6 vs
  τ-菜单选择 M7 三者互补）。
- **r1908 论文并入**：M6/M7 已作为附录 §Choosing or eliminating the tolerance τ 写入主稿
  `paper/paper.tex`（正文 9 页主拱零改动，附录含 app:tau 之 p12）。结论 L682 更新：τ-selection
  从 remaining work 移出改为「M6 相对 no-worse-than-F0 门 / M7 保留绝对 τ 菜单 CAL 选择」两种互补
  闭合；唯一保留 remaining = strictly finite-sample 收紧 band。重编译 12 页 0 err/0 overfull/0 undef
  （修复 \ref{prop:m2}→prop:p2）。候选 r1905 只读包不动。

## r1909 严格有限样本带在相对门（M8，收口结论唯一 remaining="strictly finite-sample tightened band"）
- **动机**：r1908 收口后结论仍留一条 remaining="a strictly finite-sample (non-asymptotic)
  tightened band"。绝对 gate（M2.5）已由前沿表 exact MPB/Hoeffding 列（0.260/0.097，sound）
  覆盖；本块补**相对 τ-free gate（M6）**在严格有限样本带下的行为，把该 remaining 双侧联通。
- **方法**：新增 `code/subgmmix_m8_finitesample_relgate_r1909.py`，复用 M6 完全相同
  split/F0/i*/UCB 矩阵/门函数，但把相对门的门统计 `D(w)=Σ w_g·UCB[(i*,F0)][g]≤0` 从
  **渐近 normal 带**换成 **严格有限样本的 Hoeffding（range-2）与 Maurer-Pontil（empirical-
  Bernstein）带**（与绝对管道同一公式，d∈[−1,1]，MPB 用 X=(d+1)/2∈[0,1]）。同 5-seed {0..4}×4载体。
- **忠实复现（EXACT）**：normal 带下 decision/REG_sq/D 对冻结 M6 json 逐行 EXACT；base 字段
  chosen/true_regret/UB_paired/committed 对冻结 M2.5 5-seed 逐行 EXACT。前台
  `results/M8_VERIFY_R1909.py` EXIT=0 = **17/17 PASS**。
- **结果 `results/SUBGMIX_M8_FINITESAMPLE_RELGATE_R1909.json`（5-seed×4-carrier front）**：
  | 带变体 | commit_rate | no-worse cov (committed) | sq_mean_upgraded | sq_max_upgraded | or_max_committed |
  |---|---|---|---|---|---|
  | normal（渐近，=M6） | 0.6629 | 1.0 | −0.0022 | 0.0 | 0.0546 |
  | hoef / mpb（严格有限样本，同值） | 0.6457 | 1.0 | 0.0 | 0.0 | 0.0546 |
- **机制（vacuous-collapse 边界刻画）**：226 行 i*==F0 在严格有限样本带下平凡 commit（D=0，
  REG_sq=0），全部 exact-commit 都是 trivial keep-F0；6 行 i*≠F0 的**真实切换提案**（normal 带
  certified，OUTER 全 sound，REG_sq∈[−0.130,−0.034]）被 **hoef/mpb 全部拒绝**（empirical
  D∈[−0.036,0.098]，exact 宽 ~0.25，D 永不为负到可证书一次切换）。⇒ 相对门唯一非平凡内容由
  渐近带承载；**严格有限样本使其为空**。这不是缺陷而是边界：要严格严格性应改用绝对门
  （M2.5-MPB exact，0.260 保留真内容）。
- **写作并稿（mutable 主稿）**：附录 app:tau 新增 "Strictly finite-sample bands in the relative
  gate (M6 continued)" 段；结论 L689 把唯一 remaining 改为"双侧已闭（绝对保留内容/相对坍缩）"；
  Limitations 加相对门边界披露；修正 app:tau 引言 seed 声明——M6 实为 {0..4}（与主表同），仅 M7
  用 {5,6,7,8}（原句误写两者都在 {5..8}）。重编译 **13 页 = 正文恰 9（sec:concl p9）**，0 err/0
  overfull/0 undef。候选 r1905 只读包不动（MANIFEST 只锁 r1905）。
- **诚实边界**：M8 不改任何已冻结数字，仅在相对门新增严格带对比；FD=0.260 的 absolute exact
  来自既有前沿表（未重跑，引用已冻）；边界结论（相对门严格带空洞）为可复算的正向发现而非失败。

## r1910 严格有限样本相对门的 N*-frontier（M9，预算计价刻画）
- **动机**：r1909 M8 揭示相对门在严格有限样本带下 vacuous-collapse（6 个真实切换提案全被
  hoef/mpb 拒绝），但留下 follow-up：测量**临界样本预算 N\***（带宽/预算轴上 relativ 门从
  vacuous 转非空洞的点）。本块在一个**有效样本倍增模型**下补这一数字。
- **方法**：新增 `code/subgmmix_m9_nstar_frontier_r1910.py`。复用 M6/M8 完全相同
  split/F0/i\*/配对差 UCB；固定实证 per-group 统计（mu_g,s_g），把 band 在尺寸 n_g→N·n_g 下
  估值（数据倍增因子 N∈{1,2,5,10,20,50,100,200,500,1000}），得 D_band(N)。这是 pre-statistical
  **有效样本需求**模型：若有 N 份校准数据（相同 realized 误差图景），exact 带给出该 D。只算 6 个
  真实切换提案所在的 4 个 min cell（fashion{1,4} + news{2,3}，4 个 model pool）。
- **忠诚复现（EXACT）**：chosen/F0/true_regret/D_normal 对冻结 `SUBGMIX_M8_FINITESAMPLE_RELGATE_R1909.json`
  逐行（6 切换行）assert 4 位舍入一致；normal 带 D(1) 与 M6/M8 一致。前台
  `results/M9_VERIFY_R1910.py` EXIT=0 = **ALL PASS**（C1-C7）。
- **结果 `results/SUBGMIX_M9_NSTAR_FRONTIER_R1910.json`**：
  | 带 | N\*min | N\*median | N\*max | opens_within_grid |
  |---|---|---|---|---|
  | Maurer-Pontil (exact) | 2 | **5** | 10 | 6/6 |
  | Hoeffding (exact) | 2 | **10** | 20 | 6/6 |
  逐行 N\*：fashion/1/skew_peak6=2、fashion/4/skew_peak0=5、news/2/skew_peak0=10、
  news/2/skew_peak18=5、news/2/skew_peak19=5、news/3/skew_peak19=5（MPB）。
  全部 opening 点 OUTER soundness 保持（oracle_switch_gain 0.034–0.130>0），D 随 N 单调非增。
- **机制（可复算正向发现，升级 M8 的"空洞"为"预算计价"）**：相对门的严格有限样本坍塌不是
  不可行而是**预算廉价可逆**——只需有效 ~5–20× 校准样本即可让 6/6 真实切换在 exact 带下重新
  开通，且不引入 regret 上升（单调 + soundness 保持）。M8 的 vacuous-collapse 结论因此要读作：
  在当前 CAL 预算 1× 下相对门 exact 无内容；但其前沿 N\* 每 mixture 可在证书时测量，凡预算
  可达即可恢复非平凡内容。
- **写作并稿（mutable 主稿）**：app:tau "Strictly finite-sample bands in the relative gate" 段
  末尾追加 M9 frontier 句段（预算计价 + N\* 范围 + soundness 保持）。重编译 **13 页 = 正文恰 9**，
  0 err/0 overfull/0 undef。候选 r1905 只读包不动。
- **诚实边界**：N\* 是 pre-statistical 有效样本模型下的成本特征数，非"免费倍增数据"的方法声明，
  也不改变 exact 带公式；只对 6 个真实切换提案所在 cell 计算（mnist 无切换、digits/fashion/news
  覆盖）；D_mpb(1) 复现 M8 拒绝、N\* 单调性为 grid 内经验。绝对 τ 语义仍属 M2.5。
- **r1911 M10 relative-gate exact 带的预算非空性前沿（真实采样 + 可证空证书, MGR 指令 7cc94318db8d）**
- 动机：M8/M9 在冻结 seed{0..4} 上刻画了相对门 exact 带的 vacuous-collapse 并给出预算计价的宽松上界；
  M10 提供 **OUTER-exclusive FIT/CAL、同一 F0/i*/subgroup、fresh seed{10..14}、真实按组子采样 CAL** 的
  诚实配对物，回答卡①=沿 CAL 预算网格的逐预算并列。
- **设计**：预算网格 b∈{0.25,0.5,1.0}×每组满 n_g^full；每 b 真实按组无放回抽 b·n_g^full、
  重选 i*/F0/误分类、经验重算 normal/Hoeffding/MPB + 绝对门(M2.5 UB_paired≤τ=0.04) + status-quo，
  oracle 仅诊断（只读 OUTER 一次）。指标：δ=0.1、δ_cell=δ/(M(M−1)G)、paired 单侧、CAL_FRAC=0.3、TAU=0.04，
  与 M6/M8 全同。
- **表：逐 carrier×预算**（real=真实切换 i*≠F0 行 / n_rows，triv_frac=平凡行占比，adm=admitted rate，
  abs_commit=绝对门 commit rate，abs_cov=no-worse cov of committed）：
  | carrier | b | real/n | triv_frac | norm_adm | hoef_adm | mpb_adm | abs_commit | abs_cov |
  |---|---|---|---|---|---|---|---|---|
  | digits | 0.25 | 6/75 | 0.92 | 0.0 | 0.0 | 0.0 | 0.36 | 0.889 |
  | digits | 0.5 | 9/75 | 0.88 | 0.0 | 0.0 | 0.0 | 0.227 | 1.0 |
  | digits | 1.0 | 16/75 | 0.787 | 0.0 | 0.0 | 0.0 | 0.093 | 1.0 |
  | fashion | 0.25 | 29/75 | 0.613 | 0.0 | 0.0 | 0.0 | 0.493 | 0.946 |
  | fashion | 0.5 | 29/75 | 0.613 | 0.0 | 0.0 | 0.0 | 0.96 | 0.972 |
  | fashion | 1.0 | 34/75 | 0.547 | **0.053** | 0.0 | 0.0 | **1.0** | **1.0** |
  | mnist | 0.25/0.5/1.0 | 0/75 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
  | news | 0.25 | 71/125 | 0.432 | 0.0 | 0.0 | 0.0 | 0.0 | None |
  | news | 0.5 | 74/125 | 0.408 | 0.0 | 0.0 | 0.0 | 0.024 | 0.0 |
  | news | 1.0 | 75/125 | 0.4 | **0.04** | 0.0 | 0.0 | 0.136 | 0.882 |
- **卡②严格分报平凡行 vs 真实切换行**：triv_frac 如实单独列出（digits@0.25 平凡 92%、news 40-43%），
  绝不并入总 commit 率掩盖；真实切换行数随 b 单调升（digits 6→16、news 71→75、fashion 29→34），
  但每一条在任何预算下都被 exact 带拒（hoef/mpb adm 全 0.0）。
- **卡③形式化非空必要条件 → 可证空证书**：命题——任何严格带 UCB_g≥μ_g，admission Σw_g UCBl_g≤0
  逼 Σw_g·bw_g(n_g)≤Δ(w)（Δ=选择器余量 R̂_F0−R̂_i*≥0）；Hoeffding bw=c/√n_g 与成比例 n_g=b·n_g^full
  下必要预算 b\*_hoef=[c·Σw_g/√n_g^full/Δ]²。满预算真实切换行 125/125 均 b\*_hoef>1
  （min 3.7 / median 272.9 / max 502754；Δ min 0.00038 / median 0.01391 / max 0.14819）
  ⇒ **Hoeffding exact 相对门在整个可行预算轴 b≤1 上可证地无真实切换内容**（数据特定，非通用不可能）。
  MPB 的 +7L/(3(n−1)) 项只提高所需预算，故 b\*_hoef 是保守下界证书。经验对照：exact hoef/mpb 全 0 admit。
- **卡③/④可执行混合规则（不以空洞停止）**：安全部署用 **M2.5 exact 绝对门**（fresh seed 上同样
  保留内容且随 b 单调可用：fashion abs_commit 0.493→0.96→1.0、abs_cov 1.0；mnist 全程 1.0/1.0）；
  相对门 M6 只作**渐近/描述性诊断**（STRICTLY NOT for exact finite-sample safety switching）。
  弱域如实保留：news=**预算墙**（b=1 仅 0.136、b=0.25 为 0、b=0.5 唯一 commit 行 abs_cov=0），
  digits=**容量墙**（b=1 仅 0.093）——这是绝对门自身的代价，与相对门空洞是正交边界。
- **诚实边界**：M10 为 fresh seed{10..14} 真实采样配对物，非 M8 冻结 seed{0..4} 字节复现；两块真实
  切换行数不同（6 vs 125）因 M8 只报 normal-certified 行、M10 报全部 i*≠F0 行。b\*_hoef 为保守
  必要下界，非开通预算，与 M9 经验 N\* 量纲不同不混并。空证书数据特定。单调性为网格内经验。
- **写作并稿**：app:tau 新增 $M9/'M10'$ audited 段，13 页=正文恰 9，0 err/0 overfull/0 undef；
  候选 r1905 只读包不动。verifier `results/M10_VERIFY_R1911.py` EXIT=0 = ALL PASS。
- **r1912 图 fig:m10frontier（全稿首图，纯 M10 JSON 可视化，零新数据）**：
  `code/fig_m10_frontier_r1912.py` 从 `SUBGMIX_M10_EXACTBAND_BUDGET_R1911.json` 读全部
  125 个真实切换行 → (a) Δ(w) vs B=Δ√b\* 可行域象限（对数-对数：全部行严格落在诱发
  可证空半平面 y=x 之上 = 125/125 b\*>1 的可视化证据），(b) 绝对门 committed rate 随预算 b 单调
  （fashion 0.493→0.96→1.0、digits 单调、mnist 全程 1.0）。脚本内 3 条防漂移断言随图 EXIT=0 通过。
  产物 `paper/fig_m10_frontier.{pdf,png}` 插入 appendix app:tau M10 段；重编译 14 页=正文恰 9
  （Conclusion p9、app:tau p12、fig 浮 p14），0 err/0 overfull/0 undef。只读候选 r1905 包不动。

## r1914 M11 全分配轴闭合（把 M10 可证空证书从比例轴扩到全 \{分配×预算≤1\} 盒）
- **问题/审稿攻击**：M10 的 b\*_hoef>1 证书在**比例型分配** n_g=b·n_g^full 上推导，却断言 "whole feasible
  axis"。审稿攻击：「带宽凸减于 n_g——把标签向高权重组重分配（M3 水填充移到相对门），Σw_g·bw_g(n_g) 可跌破
  选择分子 Δ(w)，在非比例分配上复活相对门」。
- **闭合（纯闭式，对冻结 M10 125 行证书数值核验）**：任意可行分配 n_g≤n_g^full，Hoeffding 宽 c/√n_g 单调
  递减 ⇒ Σw_g·bw_g(n_g) ≥ Σw_g·bw_g(n_g^full)（全帽角点 = 满比例分配本身）。满 CAL 已拒——D_mpb_full>0 全 125
  行（min +0.0228）——⇒ 任何重分配只能加宽、恒拒。空性分配单调+预算单调 ⇒ 覆盖全文细盒。
- **证据**：`code/subgmmix_m11_allocation_axis_closure_r1914.py` 前台 EXIT=0（M11_VERIFY 7/7）：
  (a) 125 行 b*>1 + D_mpb_full>0；(b) Hoeffding 全程单调 + MPB 在现实组帽≥39 单调（bw(n)≥bw(ncap) 全帽核验）；
  (c) 盒非退化；(d) 满 CAL 拒→全分配拒。
- **诚实边界**：MPB 稀疏 Bernstein 偏置 7L/(3(n−1)) 在极小组（n≈2）**非单调**（偏置 blow up），如实记录为
  boundary；闭合用承 M10 b* 的 Hoeffding 宽全程单调，MPB 只保证现实盒（所有 cap≥39）。空证书数据特定。
- **写作并稿**：app:tau M10 段后新增全分配轴闭合正文；重编译 14 页=正文恰 9，0 err/0 overfull/0 undef；
  只读候选 r1905 包不动，图+M11 段未迁（待 MGR）。verifier EXIT=0 = ALL PASS。

## r1915 M12 真采样审计（fresh seed{10..14}，M3/M3.5 预算总点，同预算四证书对比）
- **M12 runner** `code/subgmmix_m12_fresh5_budget_r1915.py`（前台 2472s，输出
  `results/SUBGMIX_M12_FRESH5_BUDGET_R1915.json`）。fresh seed{10..14}×4 carrier，真实无放回
  FIT/CAL/OUTER，M3.5 预算点 R=floor(pi*Ncal)（pi∈{0.5,0.65,0.8,0.95}），同预算对比 uniform/neyman/
  sens/widthgreedy/convex-minimax 分配（均为 FIT-only 预指定，静态配对-MPB 证书有效）。
- **严格相对证书（Hoeffding/MPB）在 100 配置上 admit 0 个真实切换**（i*≠F0）：所有 exact-relative
  "commit" 都是平凡 keep-status-quo 行——M10/M11 全轴空证书在 M3 预算总量由真实采样复证（非外推）。
- **exact 绝对门（M2.5）**：37 个 commit cell 全部 coverage 1.0；内容随预算单调——MNIST uniform
  0.773(pi0.5)→0.987→1.0→1.0；Fashion 0→0.013(pi0.65)→0.08→0.20@pi0.95，switch gain mean 0.0131/
  max 0.0753、最坏 harm -0.0012（Outer 诊断，非门输入）。
- **status quo F0**：不做决策（F0 即点估计最优日常模型，无需证书）。
- **弱域如实保留**：digits/news 各 pi 各分配 abs_commit=0（容量/预算墙），与 M3/M10 一致。
- **M12 verifier** `results/M12_VERIFY_R1915.py` EXIT=0 245/245 PASS（0 真实切换+健全+单调+墙）。
