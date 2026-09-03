# THEORY: 相对门严格有限样本带的预算非空性前沿（M10, r1911）——可证空证书与可执行混合规则

**项目**：A2_SAFE_MODEL_RANKING_SUBGROUP_MIX
**回合**：r1911 | **坐标**：M5-C (mutable 主稿继续同题)
**唯一主稿**：`subgroup_mix_ranking/paper/paper.tex`（frozen 候选 r1905 exact bytes 只读）
**证据**：`subgroup_mix_ranking/results/SUBGMIX_M10_EXACTBAND_BUDGET_R1911.json` + `results/M10_VERIFY_R1911.py`

## 0. 承前（M8/M9 边界）
- M8（r1909）在**冻结 seed {0..4}** 下证明：相对 τ-free 门（M6）在严格有限样本带
  （Hoeffding/Maurer-Pontil）下发生 vacuous-collapse——全部 226 行 i\*==F0 平凡 commit，
  6 个真实切换提案（i\*≠F0，渐近 normal certified，OUTER sound）被 exact 带全拒。
- M9（r1910）用 **pre-statistical 有效样本放大模型**（固定冻结实证统计，n_g→N·n_g）测出
  该 6 行在 N\*∈{2..10}(MPB)/{2..20}(Hoeffding) 重新开通，即"预算计价"的宽松上界。
- M10 提供卡（MGR 7cc94318db8d）要求的**诚实真实采样配对物**：
  **在 OUTER-exclusive 的 FIT/CAL 上、沿用同一 F0/i\*/subgroup 定义，预先固定每组 CAL 标签预算网格，
  在 fresh seed {10..14} 上真实按组子采样 CAL、重选 i\*/F0/误分类、经验重算每条带**。无任何放大。

## 1. 形式化（命题：exact 相对门非空的必要预算条件）

门统计（与 M6/M8 同公式）：对候选 i\*、status-quo F0、混合权重 w：

```
D_band(w) = Σ_g w_g · UCB_band[(i*, F0)][g],   commit_band(w) ⇔ D_band(w) ≤ 0
```

**命题（M10，严格有限样本下的可证必要预算）**。设真实选择器余量
Δ(w) = R̂_F0(w) − R̂_i*(w) ≥ 0（i\*≠F0 时 >0）。因为每条严格带满足 UCB_g ≥ μ_g
（带宽项非负），D_band ≤ 0 的**必要条件**是带宽被余量覆盖：

```
Σ_g w_g · bw_g(n_g) ≤ Δ(w).          (NEC)
```

在成比例逐组分配 n_g = b·n_g^full 与 Hoeffding 带宽 bw_g = c/√(n_g)
（c = √(2 ln(1/δ_cell))）下，(NEC) 改写为必要预算：

```
b ≥ b*_hoef(w) := [ c · Σ_g w_g / √(n_g^full) / Δ(w) ]².     (b*)
```

由于带宽随 √b 收缩且余量估计只随 b 变噪变窄（最有利点=满预算 b=1），若 **b*_hoef(w) > 1**，
则**可行轴 b∈(0,1] 上不存在任何非空预算**能使 exact-Hoeffding 相对门证书一次真实切换——
这是**数据特定的可证空证书**。MPB/Bernstein 的 +7L/(3(n−1)) 项只会（在同等置信水平）不降低
所需预算，故 b*_hoef 是保守下界证书：一旦已超 1，对该行的 exact-相对内容在本次 CAL 中不可行。

**注意（诚实量纲）**：b* 是保守必要下界，非"实际开通预算"。M9 的 N\*（种子 {0..4} 6 行，
用实际实证 D 直接找 D_band(N)≤0）给出小得多的开通点（N\*≤20），两者度量不同：
- M9 = 该 6 行**经验**带在某放大 N 下实际 ≤0 的点；
- M10 = 对**全部真实切换行、满预算最有利点**给出的**可证必要下界**。
两者不矛盾：M9 说明"机制层面相对门省内容、放大即可恢复"；M10 说明"对当前真实切换行集合，
Hoeffding exact 带在可行预算内没有一个能被**可证**地开通"。课堂一致性：对 125 行 b*_hoef>1 的
行，其经验 D_hoef(1)>0（该行在满预算下实际也不 admitted），两口径一致。

## 2. 数据（fresh seed {10,11,12,13,14} × 4 carrier，真实按组子采样）

预算网格 b∈{0.25, 0.5, 1.0} × 每组满 n_g^full，确定性 RNG 按组无放回抽 b·n_g^full，
重选 i\*/F0/误分类，经验重算 normal/Hoeffding/MPB 三条带 + 绝对门（M2.5 UB_paired≤τ=0.04）+ status-quo；
oracle 仅诊断（只读 OUTER 一次）。

| carrier | b | n_rows | triv_frac | real_switch | norm_admit_real | exact_hoef | exact_mpb | abs_commit | abs_no_worse_cov |
|---|---|---|---|---|---|---|---|---|---|
| digits | 0.25 | 75 | 0.92 | 6 | 0.0 | 0.0 | 0.0 | 0.36 | 0.889 |
| digits | 0.5 | 75 | 0.88 | 9 | 0.0 | 0.0 | 0.0 | 0.227 | 1.0 |
| digits | 1.0 | 75 | 0.787 | 16 | 0.0 | 0.0 | 0.0 | 0.093 | 1.0 |
| fashion | 0.25 | 75 | 0.613 | 29 | 0.0 | 0.0 | 0.0 | 0.493 | 0.946 |
| fashion | 0.5 | 75 | 0.613 | 29 | 0.0 | 0.0 | 0.0 | 0.96 | 0.972 |
| fashion | 1.0 | 75 | 0.547 | 34 | **0.053** | 0.0 | 0.0 | **1.0** | **1.0** |
| mnist | 0.25/0.5/1.0 | 75 | 1.0 | 0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| news | 0.25 | 125 | 0.432 | 71 | 0.0 | 0.0 | 0.0 | 0.0 | None |
| news | 0.5 | 125 | 0.408 | 74 | 0.0 | 0.0 | 0.0 | 0.024 | 0.0 |
| news | 1.0 | 125 | 0.4 | 75 | **0.04** | 0.0 | 0.0 | 0.136 | 0.882 |

**triv_frac 显式分报（卡②要求，绝不用总 commit 率掩盖平凡行）**：digits@b=0.25 平凡行占 92%，
news 占 40-43%；正是这些平凡行让 naive total commit 看似高。真实切换行随 b 增大而增多
（digits 6→16，news 71→75，fashion 29→34），但**每条真实切换行在任何预算下都被 exact 带拒**。

## 3. 可证空证书（卡③形式化）

`results/SUBGMIX_M10_EXACTBAND_BUDGET_R1911.json` → `emptiness_certificate`:

- 满预算真实切换行：**125**（5 fresh seed ×4 carrier 的 b=1.0 cell，i\*≠F0 全部）。
- b\*_hoef > 1 的行：**125/125 = 100%**；b\* 分布 min 3.7 / median 272.9 / max 502754.3；
  Δ（选择器余量）min 0.00038 / median 0.01391 / max 0.14819。
- ⇒ 对**本次数据与 fresh seed**，Hoeffding exact 相对门在**整个可行预算轴 b≤1** 上
  **可证地没有非空真实切换内容**（每一行 b\*_hoef 都超出现有 CAL 能提供的最大预算）。
  这是强于 M8（仅"6 行满预算被拒"的界刻画）的可证空证书。
- 经验对照：exact_hoef/exact_mpb 在全部 carrier×预算 admit_real=0.0，与可证下界一致。

## 4. 可执行混合规则（卡③："不以空洞停止"，转正同题修复）

形式化非空条件失败（当前数据 exact 相对门空）后，把证据转成**可部署混合规则**：

```
安全部署：用 M2.5 exact 绝对门（对 oracle-best 证书 regret ≤ τ，exact MPB 保留真内容
          —— fashion@b=1 abs_commit=1.0·no_worse_cov=1.0，mnist 1.0/1.0）。
相对门 M6：只作渐近/描述性诊断；不得用于严格有限样本下的安全切换。
```

绝对门在 fresh seed{10..14} 上同样保留内容，且**随预算单调可用**（fashion abs_commit
0.493→0.96→1.0 随 b 增强），与卡④"绝对门做安全部署"一致。弱域如实保留：
news 为**预算墙**（b=1 仅 0.136，b=0.25 为 0，且 b=0.5 唯一 commit 行 no_worse_cov=0），
digits 为**容量墙**（b=1 仅 0.093）。这是绝对门自身的代价，与相对门空洞是两个正交边界。

## 5. 诚实边界
- M10 为 **fresh seed{10..14} 真实采样的同题配对物**，非 M8 冻结 seed{0..4} 的字节复现；
  两 seed 块真实切换行数不同（6 vs 125），因为 M8 只报 normal-certified 的 6 行、M10 报全部
  i\*≠F0 行。D/h 口径（δ=0.1、δ_cell=δ/(M(M−1)G)、paired 单侧、CAL_FRAC=0.3、τ_abs=0.04）与
  M6/M8 完全一致。
- b\*_hoef 是必要（保守）下界，非开通预算；M9 的经验 N\*（冻结 6 行）与 M10 的可证 b\*
  （fresh 125 行）量纲不同，已在 §1 说明，不做混并。
- 空证书是**数据特定**的（依赖本数据集余量与组容量的相对大小），非通用不可能性；理论层面
  该必要条件是通用的，实证值是本数据。
- OUTER soundness 只读 OUTER 一次、从不为门输入；绝对门 no_worse_cov 为已 commit 行内
  REG_or≤τ 占比（b=0.5 news 唯一 commit 行 no_worse_cov=0，如实披露）。
- 单调性（随 b 增平凡分下降、绝对门增强）为网格内经验（{0.25,0.5,1.0}），非定理。

## 6. 下一步（同题）
- 可把 b\*_hoef 前沿做成"证书时预算/带宽象限图"（JSON 已含 Delta/n_full/逐行 b*）。
- 对弱域 news/digits 做**条件预算修复**：仅在绝对门已开的安全子域内允许相对门做诊断。
- 不阻塞复现项 M3/M3.5 补 5-seed（内部自比，ROI 低，沿用诚实 3-seed 披露）。