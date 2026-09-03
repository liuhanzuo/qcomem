# A11_correlated_mv_earlystop — 复现与数字溯源（v5.22，r508）

主稿 `paper.tex`（v5：正文 6.5 页 + References + 附录；v5 相对 v4 增量：§5 新增「Third model–carrier pair: RLVE (Qwen3-4B, N=8)」段、abstract/contribution-4/Limitations-(iv) 同步三载体括弧证据）。全部实验 CPU-only、确定性种子、前台运行。

## 载体（pinned 原始数据）

- OpenMathReasoning CoT shard0：`agents/A11/workspace/earlystop_drift_r467/cot_shard0.parquet`（220MB，CC-BY-4.0，sha256 见同目录 `SHA256SUMS.txt`，6640e85f…）。取 `pass_rate_72b_tir` 列 × 11607 唯一问题 → 每题精确通过计数 K∈{0..32}（Qwen2.5-Math-72B-TIR，N=32）。
- OpenMathReasoning CoT shard1（第二 shard 稳健性）：`agents/A11/workspace/earlystop_drift_r471/cot_shard1.parquet`（222MB，CC-BY-4.0，sha256=11a74afa…，见同目录 `SHA256SUMS.txt`）。同列 × 11454 唯一问题。
- OpenR1-Math-220k `all/default-00000-of-00010.parquet`（第二模型载体，r473）：`agents/A11/workspace/earlystop_drift_r473/all/default-00000-of-00010.parquet`（204MB，MIT，sha256=ccc3a95e… 见同目录 `SHA256SUMS.txt`，HF snapshot e4e141ec）。官方 `correctness_math_verify` 布尔 × 9374 唯一问题；去重后恰好 2 条 rollout 的问题 8853 个进入 M=2 replay。
- τ²-bench 六 carrier（机制诊断 Remark 3.1 用）：`agents/A11/workspace/earlystop_drift_r464/data/` + `SHA256SUMS.txt`（MIT）。

## 论文数字 ↔ artifact

| 论文位置 | 数字 | artifact（agents/A11/workspace/ 下） |
|---|---|---|
| Abstract/§5 主表 | BAYES-H α=.05 flip .0249 / k 6.1 / 省 80.9%；α=.02 省 73.5%；FIXED-HOEF 省 15.6%@.05 / null@.02；FIXED-EB 46.9%@.05（r480 修正 flip .0324，权威 artifact 0.03236）/ 3.1%@.02；α=.10 全行（r475 修正：HOEF flip .0645 / EB .0794 / BAYES-H .0370 / k 5.1）；表注撤回「cheaper than every baseline」并如实披露 α=.10 时 FIXED-EB 84.4% 略高于 BAYES-H 84.0%（描述性差 0.4pt） | `earlystop_drift_r469/fit_cal_test_r469.py` → `fit_cal_test_r469_result.json`（FIT4000/CAL4000/TEST3607，seed 20260815；EB+Bonferroni J=64，δ=.05/64；TEST 单次读出；DP vs MC 自检 max .0431 ≤ 3σ=.075 PASS） |
| §5 fair gap | +0.058±0.029（α=.10）；+0.652±0.029（α=.05）vs FIXED-HOEF | 同上 `fair_gap_bayesh_vs_hoeffding` |
| §5 transfer | 四源 transfer flip .021–.033、省 79–83% | `earlystop_drift_r468/robust_src_r468.py` → `_result.json` |
| §5 第二 shard 复现 | shard1 within：α=.05 省 80.6%/flip .0246、α=.02 省 73.1%/flip .0088、HOEF 15.6%@.05/null@.02；het_excess .125、frac_extreme .444 | `earlystop_drift_r471/shard1_robust_r471.py` → `shard1_robust_r471_result.json`（`R1_within`；FIT4000/CAL4000/TEST3454，seed 20260815） |
| §5 跨 shard 先验失配 | 先验=shard0 FIT、CAL/TEST=shard1：α=.05 省 80.6%/flip .0247、α=.02 省 73.2%/flip .0089；先验 TV 距离 0.042、读数移动 ≤0.4pt（α=.10 格 0.39pt，r472 全 α 复核） | 同上 `_result.json`（`R2_transfer`） |
| §5 WINDOW3 无效 | flip .0577>α=.05，k=4.4；α=.10 时 certified UCB .061（barely misses） | `earlystop_drift_r468/adaptive_r468.py`、adaptive_sweep_r468.py → 对应 json |
| §3 Remark（pooled vs FE） | ρ̂ .47–.59 vs FE −.33≈−1/(N−1)；het_excess +.072–.095 | `earlystop_drift_r465/v3_r465.py` → `_result.json` |
| §3 异质超基线 | p_var .129 vs 二项 .0038（34×）；45% 极端、11.6% 中段 | `earlystop_drift_r467/passrate_r467.py` → `_result.json` |
| §6 合成欠覆盖 | iid-CP agree .983<.99（N=40,p=.6,ρ=0 异质格） | `earlystop_drift_r464/synth_r465.py` → `_result.json` |
| §6 plug-in 欠覆盖 | boot p95 .0511>α=.05 @k=11 | `earlystop_drift_r467/v5_contrast_r467.py` → `_result.json` |
| §4 Prop TV 鲁棒性证书（r481） | τ*(.10/.05/.02/.01)=.144/.078/.074/.052（shard0）、.157/.082/.076/.024（shard1）；R=.042 处最坏 flip .039≤.05/.016≤.02 双 shard；R=.05 带 α=.10→α′=.05 worst .042≤.10、省 80.3%（vs 83.6%）；zero-g 原子质量 .36–.47；闭式 vs simplex LP 48 格全一致 | `earlystop_drift_r481/tv_robustness_r481.py` → `tv_robustness_r481_result.json`（双 shard 同 seed/split；零新数据零 TEST 读出） |
| §4 Prop TV 跨载体（r482 RLVE；r483 OpenR1 三原子边界） | RLVE τ*(.10/.05/.02/.01)=.199/.551/.213/.449（zero-g 74–87%、het_excess .0999、非单调 τ*(.05)>τ*(.10)）；OpenR1 单边泵闭式 V(R)=flip̂+g_max·R：τ*(.10)=τ*(.05)=.168（g_max .25→.125 随规则切换）、τ*(.02)=τ*(.01)=1（全单纯形，never-stop 规则 V≡0）；OpenR1 zero-g 质量 76.8%（OMR 36–47% / RLVE 74–87% 之间）；闭式 vs 精确可行分割扫描 ≤1e-9、vs HiGHS LP ≤1e-6（6×4 格） | `earlystop_drift_r482/tv_robustness_rlve_r482.py` → `tv_robustness_rlve_r482_result.json`；`earlystop_drift_r483/tv_robustness_openr1_r483.py` → `tv_robustness_openr1_r483_result.json`（均零新数据/零 TEST 读出；r483 自查披露：先验重导必须复现 r473 的 coin-stream RNG 顺序否则 FIT 切分漂移 0.012/0.016，容差 5e-5 因 r473 JSON 4dp 舍入） |
| App D 守恒曲线（r484） | τ*(α) 连续曲线 α∈[1e-3,0.2]（step 5e-4）：阶梯定律——常规则区间内非减、跳跃恰在证书阈值（四载体全格核验零违例）；规则切换数 96/93/7/2（OMR s0/s1/RLVE/OpenR1）；OpenR1 解析断点 Δ=−0.314@.07852（g_max 翻倍 0.125→0.25，τ* .396→.082）、Δ=−0.864@.04598（α 增大过阈值时 pass-prefix 停止退出、规则退化为 never-stop，全单纯形平台 τ*=1 只在该阈值**以下**成立，τ* 1→.136 同为**下跳**——r488 按 A6 审计把正文/附录方向措辞统一为双下跳）；全单纯形平台 RLVE α<.002（严格，0.002 本身是首个规则切换断点，τ*(.002)=0.06）、OpenR1 α<.0460；zero-g 对齐 vs τ* Spearman 0.75（16 格，描述性）；参考 α 逐位复现 r481/482/483 | `earlystop_drift_r484/tv_conservation_r484.py` → `tv_conservation_r484_result.json` + `fig_tau_conservation.png`（零新数据/零 TEST 读出；自查披露：初版断点检测用 g_max 漏掉 g_max 不变的规则切换，被 RLVE 两处区间内下降抓获，改全 g 向量检测后 P1 全绿） |
| §7 drift 失败域 | E1/E2/E3 × α × δ 全表 | `earlystop_drift_r469/drift_stress_r469.py` → `drift_stress_r469_result.json`（预固定机制与 δ 网格，2000 题×500 序） |
| §7 margin 修复 | E2@.05 γ=.03 覆 δ≤.20（flip .0491，省 71.8% vs 79.7%）；E3@.05 γ=.025 覆 δ≤.15（.0495，省 77.2% vs 81.0%）；δ=.20 需 γ>.03 | `earlystop_drift_r469/margin_repair_r469.py`、`margin_e3_seedfix_r469.py`（共享 MC 种子版）→ `margin_repair_r469_result.json` |
| App A.1 单调性闭式证明 | 边界原子比 (K−t)/(N−K−t)，穷举核验 33×15 零违反 | `earlystop_drift_r468/L2_lemmas_r468.py` → `L2_lemmas_r468_result.json` |
| App B 强制停止占比 | TEST 上精确 0.59%@α=.05、1.10%@α=.02 | `earlystop_drift_r469/fit_cal_test_r469.py` 同一 DP 前向 pass（r470 用同种子/同分割复算） |
| §4 Remark（连续-α 嵌套族，r478 新增；r479 收紧） | 断点 267（精确上界 555 可停状态+1，r479 修正 561+1 计数）、J_CONT=9339 vs J_GRID=64；α∈{.10,.05,.02} 连续族选中规则与网格逐位相同（mean k 5.19/6.21/8.61）；α=.01 收紧 10.3→11.9；FIXED-EB α=.02 两种族大小下均失预算；**r479 收紧**：α∈(0,0.10] 的精确有效族仅 181 个 ≤0.10 的正断点+1=182 条规则（J=6501），4 个参考 α 选中规则不变、UCB 收紧 ≤6e-4 | `earlystop_drift_r478/alpha_continuous_r478.py` → `alpha_continuous_r478_result.json`；`earlystop_drift_r479/prop_562bound_r479.py` → `prop_562bound_r479_result.json`（同 r469 FIT/CAL 切分与 seed；无新 TEST 读出，选择侧声明） |
| 定理 L2d/L4b | 塔性质、EB+Bonferroni 界、无 Jensen 论证 | `earlystop_drift_r468/L2_L4_posterior_flip_theorems.md`、`earlystop_drift_r469/L4b_bound_r469.md` |
| §5 第二模型载体（OpenR1, M=2） | 先验原子 25%/23%/52%（p=0/.5/1，pair-type 精确矩匹配）；cert 达有序前沿：α=.20 全停（flip .059≤α、省 50%）；α=.10 全停且 FIXED-1 带最紧无模型 UCB .0797≤α（certified）；α=.05 部分停（省 32%、flip .029≤α）；α=.02 不停（省 0、flip 0）；FIXED-1 flip 实测 .059=解析 E_H[p(1−p)] .058；α=.02 无任何合法证书可省钱 | `earlystop_drift_r473/openr1_m2_pilot_r473.py` → `openr1_m2_pilot_r473.json`（FIT/CAL/TEST 3000/3000/2853，seed 20260815；公平抛硬币平局；replay 精确。r473 自查修复：初版把混合对直接归属 p=.5 原子导致矩失配 2×，被 claim_check 解析-vs-实测比对抓获，改为 pair-type 矩匹配后解析 .058 与实测 .059 一致） |
| §5 第三模型载体（RLVE, N=8） | 混合 42% 极端、het_excess 0.0999；BAYES-H 省 57.6/50.9/48.2/45.4% @α=.10/.05/.02/.01（flip .031/.010/.005/.002 全 ≤α）；α≤.05 时 FIXED-EB/HOEF 均无预算；α=.10 paired gap +0.20±0.03 显著；生成序 slope −7.8e-5/trial 近平坦 | `earlystop_drift_r474/rlve_n8_r474.py` → `rlve_n8_r474_result.json`（FIT/CAL/TEST 3000/3000/3000，seed 20260815；replay 精确 DP + 3000 次 MC 自检 flip≤.004/k≤.04；6 shard sha256 钉于 SHA256SUMS.txt；success=reward>0，fractional partial credit 计失败已披露） |
| App E 证据矩阵（tab:matrix，r492 新增） | 全部 in-distribution TEST 读出一表：OMR s0 9 行 + shard1 6 行（within/transfer）+ OpenR1 4 行 + RLVE 7 行（cert/flip/mean k/saving 逐格）+ OMR α=.01 CAL 侧描述行（cert .0119）；统一机制四条（g-profile 组织全部面板；zero-g 质量跨载体 Spearman 1.0；适应性本身是 E2/E3 鲁棒通道；失败边界在每条轴上锋利） | 矩阵单元格全部复用既有 artifact（r469/r471/r473/r474/margin/r478/r486），零新实验；聚合文档 `earlystop_drift_r492/EVIDENCE_MATRIX_r492.md`（build_matrix_r492.py 机械生成） |
| App D(f) 规则通道分解（r494 新增） | τ*_m 双反事实曲线分解：τ*_pf=τ*(H_full,g_m)（冻先验只动规则）、τ*_rf=τ*(H_m,g_full)（冻规则只动先验）。重算 both-moving 曲线与 r491 网格逐位一致（0/72 不符）。(i) 规则通道驱动非单调：冻先验后 s1@.01 仍 0.000(m125)→0.060(m1000)→0.024(m4000) 上后下，因重拟合规则 g_max 本身非单调（0.563→0.075→0.098）；(ii) 估计通道非一律良性：冻规则下更粗先验在 31/72 格把半径压到全拟合之下（全在 OMR s1/RLVE；最大赤字 0.053 @RLVE m=93 α=.05），机制=粗前缀欠覆盖低-g 区、过权高-g 质量抬高基线 flip；(iii) RLVE m=187@.05 陷落（both 0.235）为纯规则通道（pf 0.222 仍陷、rf 0.530 消失；g_max 0.179 vs 全规则 0.071）。可操作读法：买半径靠缩小诱导 g-profile（margin 修复）而非单纯加 FIT 数据 | `earlystop_drift_r494/rule_channel_r494.py` → `rule_channel_r494_result.json`（同 r491 冻结 FIT 序/嵌套前缀/精确 LP；零新数据、零 GPU、零 TEST 读出） |
| App D(g) flip-budget 状态子集修复（r499 新增，实验 r498） | 构造：每原子 K 按证书值升序保留原规则停止状态的累计 flip≤cap 最长前缀，未保留状态跑满 n；domination g_S(K)≤min(g_orig(K),cap) 逐点成立（inline assert 全过）、停止集为原规则子集（只晚不早）、精确 LP 重认证。全网格 72 格 × 12 cap：S1 regen 0/72、S2 domination 逐点、best cap 全 τ*=1——70/72 严格增益（votes +1%~+11%，max +11.3% @OMR s0 m=500 α=.10；RLVE ≤+3.3%），2 例外 RLVE α=.02 原已 τ*=1 不选 cap。陷落格 RLVE m=187@.05 0.235→1.0 @cap.07（+0.5% votes）；粗格 s1 m=125@.01 0.0095→1.0。trivial-validity 诚实声明：τ*=1 格按构造满足 max g_S≤α∧base_S≤α，代价由 base_S 实际水平背书（m=4000 α=.05 cap=.02 base_S=0.0054≪α）。两负结果：support truncation 诱导 cert 表与原 g 完全相同（ndiff=0，P1 FAIL）；deadline 实现 flip-by-deadline 关于 κ 非单调且认证半径更差（0.235→0.133）。k=3 baseline 对照：base 0.093（超 α=.01 九倍 τ*=0）vs 修复后 base_S 0.0036——修复非退化早停 | `earlystop_drift_r498/full_sweep_r498e.py` → `full_sweep_r498e_result.json`（主扫描）；`statecap_repair_r498d.py` → `_result.json`（6 格构造细节+原子级 g_S）；`support_trunc_repair_r498.py` / `efficiency_r498c.py`（两负结果）；`cap_repair_r498b.py`（profile-cap 中间步）。同 r491 冻结 FIT 序/嵌套前缀/精确 LP；零新数据、零 GPU、零 TEST 读出 |
| App D(g) Universal budget（r500 新增，纯重分析） | 通用修复预算律：cap=0.01（全文最小 α 格）在 **72/72 格**（3 载体 × 6 m × 4 α）达 τ*=1——代价 votes +1.0%~+22.3%（OMR；RLVE ≤+5.2%）、realized base flip ≤0.0038（RLVE 恒 0）。边界在扫描网格内锋利：cap=0.015 时恰 18 格（全部 α=0.01、每载体 6 格全 m）仍未全修复，其余 54 格（α≥.02）已修复；临界 cap（最小达标扫描 cap）72 格全=0.01。即修复律是 cap 相对 α 的性质，与 m、载体无关；不超过最严格名义水平的预算可修复全部拟合尺寸 | `earlystop_drift_r500/universal_cap_r500.py` → `universal_cap_r500_result.json`（纯重分析 r498e 全扫描 JSON；自带 S1/S2 自检全过；零新数据、零 GPU、零 TEST 读出） |
| App D(g) edge law（r503 新增，精细扫描+二分） | 有效临界 cap c*=sup{c:τ*(c)=1}：因 domination 逐点 g_S≤cap，cap≤α 机械 τ*=1（P1 零违例），故 c*≥α 全 72 格且严格超 α 全 72 格。OMR 余量 ≤+0.8%@α≥.02（s1@.05 单格 +2.8% 例外）、α=.01 最大 +10.6%；RLVE 实测 c*/α∈{15/14(.10), 10/7(.05/.01)}，α=.02 在扫描带 (α,2α] 端点 c*≥2α（未 bracket）。**r503 自查披露**：预注册 P5 网格 {8/7,10/7,2} 系算术错误（8/7 应为 15/14），按描述性 FAIL 如实记录；α=.02 的 2α 为删失下界非真实 edge | `earlystop_drift_r503/critical_cap_r503.py` → `critical_cap_r503_result.json`（细网格 0.004–0.022 step 5e-4 + 前缀 24 次二分 tol≈6e-7；gmax_S 沿 cap 单调不降 inline 2592/2592 验证；P4 与 r498e 54 锚点 0 不符；零新数据/零 TEST 读出） |
| App D(g) edge law 解析化（r504 新增，闭式推导+勘误） | **闭式 edge law**：原子 K 上停止态 (k,x) 的 flip 质量=超几何到达概率 C(K,x)C(n−K,k−x)/C(n,k)（有理数、与拟合先验无关），故「α 之上最小 cert-greedy 可达 flip 和」由证书表单独决定，且在全部 18 个扫描 bracket 格等于扫描测得 edge（D1）。RLVE（N=8）量子：c*=3/28@α=.10（15/14·α）、1/14@.05（10/7·α）、1/70@.01（10/7·α），六个 m 全同（D3 m-不变——edge 是 binding 原子的组合学而非先验）。α=.02 按证书 straddle 分裂：4 个 m（(4,1) 证书≤.02）edge=1/14（25/7·α，超出 r503 扫描带），m∈{93,750}（(4,1) 证书>.02）无可达 flip 和>α → edge=+∞，τ*=1 存至 cap=0.5（max g_S=1/56）永不陷落（D2，精确 LP 认证）。OMR（N=32）无 O(α) 量子：per-atom 闭式 edge 上界预算约束扫描 edge（48 格中 12 格取等），全格 ≤+10.6%（D4）。**勘误（如实披露）**：v5.17 正文「c*/α∈{8/7,2}」有误——8/7 在 r503 测量中从不出现（承自 P5 算术错误），α=.02 的 2α 为删失下界；v5.18 已按本 artifact 修正为派生值，claim_check 由硬编码「8/7」改为 CED.* 工件驱动锚（含 8/7 禁现断言）。范围仅主线 v5.17；冻结候选 r499_v5_15/r500_v5_16 不含 edge-law 段、不受影响 | `earlystop_drift_r504/edge_law_r504.py` → `edge_law_r504_result.json`（同 r503 冻结 FIT 序/证书表；闭式 c* + 精确 LP 边缘认证 τ(c*−1e-7)=1/τ(c*+1e-7)<1；零新数据/零 TEST 读出） |

| App D(g) edge 紧性刻画（r505 新增，同题后续） | **紧性双 regime 刻画 + 取等计数勘误**：r504 的 per-atom 闭式 edge 是预算约束扫描 edge 的上界，但「取等格数」依赖核验容差。r505 在判别容差 1e-6（r503 c* 6 位舍入噪声 5e-7、二分容差 ~6e-7）下重算得 **11/48 取等**，而非核验器 1e-4 容差下的 12/48——第 12 格（shard1 m=2000 α=.10）closed−scan=8.22e-05 超舍入噪声两个数量级，被 1e-4 容差误吸（P1，双向计数与第 12 格身份均断言可核）。机制：以「闭式 edge 处认证 slack 的符号」把 48 格分成两 regime——tight 11 格在 cap=c*−ε 时 realized max g_S 仍 ≤α（slack −1.26e-03..−2.43e-05，crossing 态未进预算，扫描 edge 与闭式 edge 重合），strict 37 格在 c*−ε 时已超 α（slack +2.91e-05..+3.71e-03，预算装不下 binding 原子的 crossing 前缀，扫描必须丢它故严格小于）。两 regime 不重叠、间隔 ≥2.4×，紧性由符号判定而非拟合阈值（P3/P4 双条件 48/48 全过，regime_separation PASS）。**勘误（如实披露）**：v5.18 正文与 CED.omr.nogap「equality at 12 of 48」修为 11（判别容差），CED.omr.nogap 同时断言 1e-6→11 与 1e-4→12 两个计数使披露自身可核 | `earlystop_drift_r505/edge_tightness_r505.py` → `edge_tightness_r505_result.json`（同 r503/r504 冻结 FIT 序/证书表；零新数据/零 TEST 读出） |
## 复现命令（均前台，总 CPU <25 min）

| App D(g) 离散停止集几何统一注记（r506 新增，同题后续） | **三个非单调现象归一到同一对象「离散停止状态集」**：(i) τ*(cap) 非单调（粗网格 323/792 相邻对违反，r501 C4 证伪复现）与其全半径穿越集恒为前缀（细网格 0 洞 0 悬垂，72/72 格）的解析和解——cap↓ 排除 cert 升序前缀，两个标量化 max_K g_S 与 base_S 对 cap 均单调（细网格 2592 对 0 违反、粗网格 792 对 0 违反），τ*=1 一旦成立必在更小 cap 保持，非单调只能活在 τ<1 内部；(ii) deadline flip 序列（probed RLVE 原子 κ=8..3：0.0143/0.3571/0.2143/0.5/0.2429/0.5）与 deadline 修复更差（0.235→0.133）是同一离散性沿时间轴的读法；(iii) r504 闭式 edge（超几何 flip 质量=量子）与 r505 slack 符号紧性（11 tight/37 strict）是同一对象的精确读法。**housekeeping 勘误（如实披露）**：r501 工件/memo 的「1728」分母有误（真实 72×11=792；分子 323 精确复现；分母从未进正文、不影响 r501 checks 门），v5.20 正文写 323 of 792 并括注 misprint。**溯源闭环**：r503 日志「gmax_S 单调不降 inline 2592/2592 验证」的 inline 断言未留在 r503 脚本中，本 artifact U2 在 r503 存储曲线上真实执行了同一核验（2592 对 0 违反），事后闭合该溯源缺口 | `earlystop_drift_r506/discrete_geometry_r506.py` → `discrete_geometry_r506_result.json`（全部读冻结 r498e/r501/r503/r505/r498c 工件；预注册 U1–U6 全过；零新数据/零 TEST 读出） |

```
cd agents/A11/workspace/earlystop_drift_r469
python3 fit_cal_test_r469.py        # ~3.5 min：FIT/CAL/TEST + 选择 + 单次 TEST 读出
python3 drift_stress_r469.py        # ~14.5 min：漂移压力测试
python3 margin_repair_r469.py       # ~7 min：margin 扫描（E2/E3）
python3 margin_e3_seedfix_r469.py   # ~40 s：E3 共享种子修正表
cd ../earlystop_drift_r471
python3 shard1_robust_r471.py       # <1 min：shard1 独立复现 + shard0→shard1 先验失配 transfer
cd ../earlystop_drift_r473
python3 openr1_m2_pilot_r473.py     # ~1.5 min：OpenR1 M=2 跨模型/跨载体前沿实验（含下载后 sha256 核对）
cd ../earlystop_drift_r474
python3 rlve_n8_r474.py           # ~2 min：RLVE N=8 第三载体主链（FIT/CAL/TEST + DP + 生成序 drift 诊断）
cd ../earlystop_drift_r481
python3 tv_robustness_r481.py     # <1 min：prop:tv TV 鲁棒性（OMR 双 shard）
cd ../earlystop_drift_r482
python3 tv_robustness_rlve_r482.py  # <10 s：prop:tv 跨载体（RLVE）
cd ../earlystop_drift_r483
python3 tv_robustness_openr1_r483.py  # <1 min：prop:tv 三原子边界（OpenR1，含 LP 交叉验证）
cd ../earlystop_drift_r484
python3 tv_conservation_r484.py     # ~2 min：τ*(α) 守恒曲线（三载体连续网格 + App D 图）
cd ../earlystop_drift_r486
python3 spearman_decomposition_r486.py  # <1 min：对齐声明分解审计（pooled/within/between/LOCO/平台剔除/het 混杂，v5.9 措辞修复的证据链）
cd ../earlystop_drift_r491
python3 prior_fit_size_r491.py        # <3 min：FIT 池大小敏感性 τ*_m(α)（嵌套前缀子拟合；prop:tv 沿 m 轴闭环 51 格零违例；App D(e) 证据链）
cd ../earlystop_drift_r494
python3 rule_channel_r494.py          # <3 min：τ*_m 规则通道双反事实分解（冻先验/冻规则；regen anchor vs r491 0/72；App D(f) 证据链）
cd ../earlystop_drift_r498
python3 full_sweep_r498e.py           # <3 min：flip-budget 状态子集全网格 72 格 ×12 cap（App D(g) 主证据链；regen 0/72）
python3 statecap_repair_r498d.py      # <1 min：6 格状态子集构造细节（原子级 g_S / domination assert）
python3 support_trunc_repair_r498.py  # <1 min：负结果 1（support truncation 无效）
python3 efficiency_r498c.py           # <1 min：负结果 2（deadline 实现更差）
cd ../earlystop_drift_r500
python3 universal_cap_r500.py         # <1 min：通用修复预算律（App D(g) Universal budget；纯重分析 r498e JSON，S1/S2 自检全过）
cd ../earlystop_drift_r503
python3 critical_cap_r503.py          # <1 min：edge law 精细扫描+二分（App D(g) edge law；P1/P2/P4 PASS，P5 算术错误如实披露 FAIL）
cd ../earlystop_drift_r504
python3 edge_law_r504.py              # <10 s：edge law 闭式推导 + 精确 LP 边缘认证（D1–D5 全过；v5.17 量化勘误的工件依据）
cd ../earlystop_drift_r505
python3 edge_tightness_r505.py        # <30 s：edge 紧性两 regime 刻画 + 12→11 取等计数勘误（P1–P4 + regime 分离全过）
cd ../earlystop_drift_r506
python3 discrete_geometry_r506.py     # <30 s：离散停止集几何统一注记（U1–U6 全过；含 r501「1728→792」勘误与 r503 inline 单调核验的溯源闭环）
cd ../earlystop_drift_r507
python3 discriminant_r507.py          # <1 min：slack 符号分类器判别容差头对头审计（D1–D6 全过；11TP/37TN、draft1 22 错、固定阈值 11 错、容差带 [5e-7,8.22e-05)、v5.19/v5.20「≥2.4×→1.20×」勘误的工件依据）
cd ../earlystop_drift_r508
python3 prefix_prop_r508.py           # <2 min：前缀律+双标量单调正式 Proposition（prop:prefix）机器见证（V1–V7 全过；含 V3 r503 inline 断言落盘见证、V5 base_S 精确重算 30/30、V6 323=253+70+0 洞分解；两处首跑 FAIL 自查披露见 run_r508.log 与 AUDIT_README r508 节）
cp fig_tau_conservation.png ../../../paper/A11_correlated_mv_earlystop/
cd ../earlystop_drift_r470
python3 gen_drift_tables.py drift_tables.tex   # <1 s：附录 drift 全表（从 json 机械生成）
cp appendix_proofs.tex appendix_dp.tex drift_tables.tex ../../../paper/A11_correlated_mv_earlystop/
cd ../../../paper/A11_correlated_mv_earlystop
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

## 边界声明（与论文 §limitations 一致）

1. 主结果是 count-exchangeable replay 模型下的**精确**量（非 MC）；carrier 无 rollout 顺序，有序部署有效性只经 §7 预固定压力机制界定。
2. TEST 仅读一次；所有有效性宣称在 CAL 以 EB+Bonferroni（J=64）认证。
3. L2c 单调性：闭式证明已完成（App A.1，r470），并保留 N=32 全格穷举核验作为独立证据。
4. 第二 shard 稳健性（r471 已闭环）：shard1 独立复现 + shard0→shard1 先验失配 transfer 双实验，全部定性结论逐项复现（读数移动 ≤0.4pt；α=.10 格 0.39pt，r472 全 α 复核）。
5. Figure 1（r471 已落地）：`fig1_earlystop.png`（三面板概念图，AutoFigure-Edit / Nano Banana 2 Pro），嵌入 §1。生成脚本/编辑史/双盲与灰度与 PNG 二进制核验见 `figures/FIG1_HISTORY.json`。已知小瑕疵：panel (c) 注解括号被图像模型渲染为未闭合（3 轮修复未果），gap +0.652 与 significant 数字正确可读，如实记录于 FIG1_HISTORY。
6. 审计包（r472 落地，r473–r507 连续刷新）：`audit_pack/`（v5.21 快照 + 430 项机器声明核验 430 PASS/0 FAIL，claim_check_r507_inplace.log；clean-room 421 PASS/0 FAIL/3 EXT，3 EXT=W1 外部 provenance 设计如此）。r475 hostile 自审修复清单与覆盖空洞披露见 audit_pack/AUDIT_README.md「r475 更新」节；r480 双审定点合并修复与 paper↔artifact 直解核验层见「r480 更新」节；r488 A6 审计（r484_v5_8 理论半审）定点合并与方向谓词负对照见「r488 更新」节；r492 证据矩阵 App E 与 MX.*/X.mx.* 核验层见「r492 更新」节；r493 真实路径溯源核验层 W1.*（r491 prior-fit 实验脚本/JSON/前台日志三 artifact 的存在性、字节一致性、内容锚断言）见「r493 更新」节；r494 规则通道分解 App D(f) 与 FSD.*/X.fsd.* 核验层（含 Q2 证伪的诚实披露）见「r494 更新」节；r496 外部 provenance 分层（EXT 层）与 r499 flip-budget 修复 App D(g) FSG.*/X.fsg.* 核验层（含两负结果披露）见「r499 更新」节；r500 通用修复预算律与 FSU.*/X.fsu.* 核验层（含负对照覆盖洞披露与修复）见「r500 更新」节；r503 edge law 与 r504 闭式推导/勘误见「r504 更新」节（含 v5.17 「8/7」量化错误的完整勘误记录）；r505 edge 紧性两 regime 刻画与取等计数 12→11 勘误见「r505 更新」节；r506 离散停止集几何统一注记与 r501「1728→792」分母勘误见「r506 更新」节；r507 slack 符号分类器判别容差头对头审计与「≥2.4×→1.20×」正文勘误见「r507 更新」节。r508 前缀律+双标量单调正式 Proposition（prop:prefix）与 CPF.* 核验层（含 r503 inline 溯源披露与 V6 分解证伪披露）见「r508 更新」节。
7. 跨模型证据现为三载体括弧：OMR(N=32, Qwen2.5-Math-72B) / RLVE(N=8, Qwen3-4B) / OpenR1(M=2, DeepSeek-R1)，分别省 73–81% / 45–58% / 32–50%（各 α 格实测 flip 恒 ≤ 名义）。更大 M（如 N=16/64）跨模型组合留作后续，不是当前主稿宣称。
