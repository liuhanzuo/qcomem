# N\*-Frontier of the Strictly Finite-Sample Relative Gate (M9, r1910)

身份
- 承r1909 M8：相对 τ-free gate（M6）在严格有限样本带（Hoeffding/Maurer-Pontil）下 vacuous-collapse，
  6 个非平凡切换提案（i\*≠F0，渐近 normal 带证书，OUTER sound）被 exact 带全拒（D_normal<0 但 D_hoef/D_mpb>0）。
- 本块补 M8 明录的同题 follow-up（b）：**测量相对门严格带从 vacuous 转非空洞的临界样本预算 N\***。
- 模型：门统计 D(w)=Σ_g w_g·UCB_g，UBB_g 在 group g 的配对差 UCB，n_g=CAL 组样本数。设预算放大 N
  （数据倍增因子），把实证 per-group 统计（mu_g,s_g）与混合 dcell 冻结，UBB 改在尺寸 N·n_g 下估值 →
  D_band(N)。这是 pre-statistical 模型：相同 realized 误差图景下若有 N 份校准数据，exact 带会给出该值。
  由此把相对门的严格有限样本代价读作**有效样本需求**。

断言（应为）
- (A1) 6 个切换提案的 D_normal<0（复现 M8）且 D_hoef/D_mpb(1)>0（M8 拒绝）。
- (A2) 每个提案存在 N\*_band=min{N∈grid: D_band(N)≤0}；随 N 增大 band 收紧，D 单调降到 0 以下。
- (A3) OUTER soundness：在 N\* 开点切到 i\* 时 REG_sq=R(i\*)-R(F0)≤0 全承（相对门保证 sound 的开关）。

证据
- results/SUBGMIX_M9_NSTAR_FRONTIER_R1910.json；复现断言 vs 冻结 SUBGMIX_M8_FINITESAMPLE_RELGATE_R1909.json
  （chosen/F0/true_regret/D_normal 逐位 4 位舍入一致）。
- verifier：results/M9_VERIFY_R1910.py 前台 EXIT=0。

边界
- pre-statistical 有效样本模型非真实数据扩充；N\* 是成本特征数，不是免费增强数据的方法声明。
- D_mpb(1) 复现 M8 拒绝；N\* 是否落入操作员可达区间（N∈{1..200}）决定该门在真实校准预算下的取舍。
- 只对 6 个真实切换提案刻画；mnist 无切换（F0≡i\*），digits/fashion/news 覆盖。
- 单侧 normal/exact 带与 M6/M8 同口径；绝对 τ 语义仍属 M2.5。

下一步
- 若 N\* 全在可达区间：结论补"相对门严格带的预算代价为 O(几十-几百)× 校准样本, 若手头可达则该门非空洞"。
- 否则：诚实写"该 6 提案需大到不可达的预算才 open，即使收敛到 exact 带相对门在这些行上无内容，
  唯一保留严格内容=绝对门 M2.5"，与 M8 边界刻画互补。