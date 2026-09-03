# THEORY: 严格有限样本带在相对门（M8, r1909）——vacuous-collapse 边界刻画

**项目**：A2_SAFE_MODEL_RANKING_SUBGROUP_MIX
**回合**：r1909 | **坐标**：M5-C (mutable 主稿继续同题)
**唯一主稿**：`subgroup_mix_ranking/paper/paper.tex` | **证据**：`results/SUBGMIX_M8_FINITESAMPLE_RELGATE_R1909.json` + `results/M8_VERIFY_R1909.py`(17/17)

## 1. 要回答的问题
冻结主稿结论在 r1908 收口 τ-selection 协议后仍唯一 remaining="a strictly finite-sample
(non-asymptotic) tightened band"。绝对 gate（M2.5）已在该前沿表 exact 列（Hoeffding 0.097 /
Maurer-Pontil 0.260）覆盖，故 remaining 的未覆盖半边=**相对 τ-free gate（M6）**在严格有限样本带下的行为。

## 2. 形式化
门统计（M6，`i*`=CAL 点估计最优，`F0`=collected-mixture 点估计最优=status-quo）：
```
D_band(w) = Σ_g w_g · UCB_band[(i*, F0)][g],   commit_band(w) ⇔ D_band(w) ≤ 0
```
带变体（全部同一 Bonferroni split δ_cell=δ/(M(M−1)G)，paired 单侧）：
- normal：U = μ + z_{1−δ_cell}·ŝ/√n（渐近，收紧）
- Hoeffding：U = μ + √(2 ln(1/δ_cell))/√n（range-2，严格）
- Maurer-Pontil (empirical-Bernstein)：d∈[−1,1]→X=(d+1)/2∈[0,1]，
  U_X = mX + √(2 v̂X ln(2/δ_cell)/n) + 7 ln(2/δ_cell)/(3(n−1))，再 2U_X−1 映射回 d（严格）

**命题（M8 边界）**：当行满足 i*==F0（D≡0），mouvband 平凡 commit；当 i*≠F0（真实切换提案），
exact 带宽度 ∝ O(1/√n) + O(ln(2/δ)/n) 相对 empirical D（本数据集 ~[−0.036,0.098]）过大，
故无任何行可被严格有限样本带证书为一次真实切换。

（经验验证，非定理；作为边界刻画披露。）

## 3. 结果（5-seed×4-carrier，front）
| 带 | commit_rate | no-worse cov | sq_mean_upgraded | sq_max_upgraded | or_max_committed |
|---|---|---|---|---|---|
| normal(=M6) | 0.6629 | 1.0 | −0.0022 | 0.0 | 0.0546 |
| hoef / mpb | 0.6457 | 1.0 | 0.0 | 0.0 | 0.0546 |

- n_f0=226 rows i*==F0；mpb/hoef 全部 commit 都 i*==F0（trivial）；6 个 i*≠F0 切换提案被
  normal certified（OUTER 全 sound，REG_sq∈[−0.130,−0.034]）被 exact 带全拒绝。

## 4. 解释（为什么这是边界而非缺陷）
- 绝对门 M2.5 的 exact MPB 带保留真内容（0.260 committed，sound），因为其比较对象是
  oracle-best，regret=0 的天然量级小。
- 相对门的受控对象是 F0——本身已接近最优，i*−F0 的 paired 差很小，exact 带宽度相对
  empirical D 不可忽略 → 只能证书 trivial keep-F0。
- 结论：**相对 no-worse-than-F0 证书的意义恰恰在于它允许渐近带**；一旦强制严格有限样本带，
  选择它的唯一理由（比绝对证书省内容/更可 deploy）消失。操作员要严格严格性就用绝对门（M2.5-MPB）。

## 5. 诚实边界
- M8 不新增论文数字，仅相对门加严格带对比；absolute exact=0.260 引用已冻前沿表。
- FD=0.260 未重跑（已冻）；M8 只复用 M6 既有 trained/oracle（前台重训练，但与 M6 EXACT 复现，
  非新数据泄漏）。
- D∈[−0.036,0.098] 为本数据集经验范围，非最坏界；边界结论在 n_g 极大/极小端可能改变，如实标注。

## 6. 下一步（同题）
- 若 MGR 签发迁稿，M8 随 app:tau 已并入章节迁入；否则保留为 follow-up 注记。
- 对偶地：可测"带宽/预算轴"上相对门的严格带从 vacuous 转非空洞的临界 n_g（经验 frontier），
  作为后续同题增强。