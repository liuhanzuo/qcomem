# REPRO_README — FINITE-SAMPLE SAFE MODEL RANKING UNDER SUBGROUP-MIX TURNOVER

本项目（A2_SAFE_MODEL_RANKING_SUBGROUP_MIX）理论为核，轻量真实实验验证。纯 CPU、前台、零 GPU。

## 依赖
- python3.10 + numpy, scipy, scikit-learn, torch, torchvision
- 数据缓存 `duplicate_sel/data/`（FashionMNIST/MNIST/KMNIST，含 DATA_LICENSE.md）；使用前与 run 冻结缓存核对。

## 复现命令（前台，任选）
```bash
# r1881 首轮：M1 负结果定位（naive argmin_U minimax-regret 退化）
python3 subgroup_mix_ranking/code/subgmmix_pilot_r1881.py
python3 subgroup_mix_ranking/code/subgmmix_diag_r1881b.py
python3 subgroup_mix_ranking/code/subgmmix_r1881c_mixtureball.py

# r1884（本回合）：M2 修复 = 点估计选择 + sound P2 证书门 + 诚实回退
python3 subgroup_mix_ranking/code/subgmmix_m2_gate_r1884.py
```
每次输出写入 `subgroup_mix_ranking/results/SUBGMIX_*.json`。

## 已验证产物（r1884）
- `results/SUBGMIX_M2_GATE_R1884.json` — 本回合主证据（350 行；cert_coverage=1.0；τ 扫描）。
- `results/SUBGMIX_PILOT_R1881.json` — M1 负结果（MRR≡cal_prior 350/350）。
- `results/SUBGMIX_DIAG_R1881B.json`、`SUBGMIX_R1881C_MIXTUREBALL.json` — 异质宽度下 M1 反效果。
- 理论：`THEORY_SKELETON_R1884.md`；结果矩阵：`RESULT_MATRIX.md`。

## 诚实边界 / 复现注意
- 20News 需联网下载，run 冻结缓存已有；若环境无网先取缓存或改用剩余三载体。
- cert 为逐 w 逐点覆盖（per-w ≥1−δ）；全 w 网格同时联合证未做（TBD）。
- 每载体 seed 变化会改变 CAL 组样本量 → committed_rate 波动属预期；cert_cov（健全性）跨 seed 应稳定为 1。
### r1885（M2.5 配对差证书；同题修复，r1884 M2 的 TBD-2 前导）
```bash
python3 subgroup_mix_ranking/code/subgmmix_m25_paired_r1885.py   # 前台，~27min(3 seed)
```
输出 `subgroup_mix_ranking/results/SUBGMIX_M25_PAIRED_R1885.json`。
- 证据：`results/SUBGMIX_M25_PAIRED_R1885.json`（210 行；cert_cov_pair/mpb/hoef 全=1.0；
  committed_rate_pair=0.495, mpb=0.257, hoef=0.09）。
- 与 r1884 REPRO 注意一致：20News 需缓存；cert 逐 w 逐点；Normal 变体渐近健全（精确用 MPB/Hoeffding）。
- **r1902 matched-seed 比较**：先把 M2 与 M2.5 统一到同 3 seed {0,1,2}（M2 在该 seed 上
  由 `random_state=seed` 确定性重聚合，零重训）：M2 0.267 / M2.5-MPB 0.257 / Hoeffding 0.091 /
  Normal 0.495，全 cov 1.0。
- **r1903 full 5-seed matched（新增，收口 r1901 复验缺口）**：M2.5 现已在 M2 相同的 5 seed
  {0,1,2,3,4} 上真跑一遍，headline 升为严格全额 matched（~60min CPU、零网络；news 走本地
  sklearn 缓存）：
  ```bash
  python3 subgroup_mix_ranking/code/subgmmix_m25_paired_5seed_r1903.py   # 前台, ~60min
  ```
  输出 `subgroup_mix_ranking/results/SUBGMIX_M25_PAIRED_R1885_5SEED.json`（350 行）。
  5-seed headline：M2 0.269 / M2.5-MPB 0.260 / Hoeffding 0.097 / Normal 0.503，全 cov 1.0；
  per-carrier 见 RESULT_MATRIX r1903 段。r1902 3-seed matched 数值保留于 M5_VERIFY 作历史断言。
  r1885 原生 3-seed 文件 `SUBGMIX_M25_PAIRED_R1885.json` 保持未改（存档切片）。

### r1899（条件化门 operating characteristic；next-step c 落地，纯 CPU 前台 ~0.1s）
```bash
python3 subgroup_mix_ranking/code/subgmmix_gate_oc_r1899.py
```
输出 `subgroup_mix_ranking/results/SUBGMIX_GATE_OC_R1899.json`。
- 受控合成族（G=4、同一 paired-MPB 证书、R=1200、20 seed、τ 扫描）independent 扫
  deployed set ∈ {spanning, non-spanning} × CV(β̂) ∈ {低, 高}。
- 判定：驱动源=deployed set 是否 spanning（非 span → minimax committed-rate 曲线在
  两 CV 水平都支配 uniform，signed area +0.25）；spanning 低 CV 持平（+0.000）；
  **spanning 高 CV 时 minimax 反而更差（−0.08）**→ "else uniform" 分支的安全半边。
  门=非 span→convex-minimax、否则 uniform，safe+effective。
- 合成构造（3 点配对差）与确定性反例同，清楚标注 feasibility/mechanism；真实落点与
  r1896（非spanning probe 用 minimax）/ r1897（对称spanning 网格用 uniform）一致。

## r1900 论文排版复现
- 正文压缩 + 附录迁移后，复现 `paper/` 下 `pdflatex paper && bibtex paper && pdflatex paper
  (×2)` 得 paper.pdf = 12 页（正文 9 + refs p10 + 附录 A-E p11-12），0 err/0 overfull/0 undef。
- 主表数字均锁冻结 JSON（tab:main/frontier/m3/m35/m35gate/gateoc），本块未改任何实验数字。
- 原始 12 页字排版版存 `paper_backup_r1900/` 以备对照。

## r1901/r1902 M5 content-complete 自查（数字逐值锁定，含 matched-seed）
- 复现：`python3 results/M5_VERIFY_R1902.py`，前台 EXIT=0 = 94/94 PASS（只读冻结 JSON，不写盘）。
- 覆盖：tab:main/frontier（r1902 起为 **matched 3-seed**：M2 0.267 系列、M2.5 0.495/0.257/0.091，
  另断言 5-seed M2 0.269 独立鲁棒性视图）、App.A M1 diag（350/352/165/93/62）、
  tab:m3（4200 cell、0 viol、0.778/0.400/...）、tab:m35（minimax 0.400/0.889/...）、tab:m35gate（0.333→1.0、UB range）、
  App.B ablation（active%/d* ratio）、App.C counter（0.094/0.029、n_g）、App.E gate OC（area ±0.25/−0.08、worst −0.49@0.12）。
- **诚实修正（r1901）**：(1) gate-OC minimax 达 1.0 是 τ=0.08 非 0.06；(2) App.A diag 15/880 是 11 tied + 4 worse 非全 tied。
  两者均已在 paper.tex + CHINESE_SUMMARY + RESULT_MATRIX + THEORY_*.md 同步，paper.pdf 重编译 12 页（正文 9）EXIT=0。
   paper_backup_r1900/ 是历史快照，保持未改。

## r1903 full 5-seed M2.5（收口 r1901 复验缺口）
- 复现：`python3 code/subgmmix_m25_paired_5seed_r1903.py`（~60min CPU、零网络、news 走本地 sklearn 缓存）
  → `results/SUBGMIX_M25_PAIRED_R1885_5SEED.json`（350 行/方法）。
- headline（τ=0.04, δ=0.1, 全 cov 1.0）：M2 0.269 / M2.5-MPB 0.260 / Hoeffding 0.097 / Normal 0.503。
- M5_VERIFY_R1902.py 增 10 项 5-seed 断言 → **105/105 PASS**（前台 EXIT=0）。r1885 原生 3-seed 文件未改。

## r1904 投稿 prep（AI-use 声明 + 引用核验 + 核对表；零实验改动）
- 引用核验：`python3 code/verify_citations_r1904.py` 前台 EXIT=0 = 8/8 OK。
  - maurer2009bernstein=arXiv:0907.3740（标题更正为官方《...Penalization》）、
    sagawa2019distribution=arXiv:1911.08731 均用 arXiv 官方页面逐字核验；其余 6 条为经典期刊登记。
  - references.bib 删去未被引用且标题截断错误的死条目 `hsu2016generalized`。
- 论文：附录末新增 `\section{AI use statement}`；重编译 12 页（正文 9 + refs p10 + App p11-12）
  = 0 err/0 overfull/0 undef。
- 新增 `SUBMISSION_CHECKLIST.md` 投稿前核对表；本 README 与 CHINESE_SUMMARY/RESULT_MATRIX 同步 r1904 段。

### r1906 τ-free 安全升级门（M6）
```
python3 subgroup_mix_ranking/code/subgmmix_m6_upgradegate_r1906.py   # 5-seed×4-carrier, front, 纯CPU
```
运行前先断言冻结 `SUBGMIX_M25_PAIRED_R1885_5SEED.json` 共享字段逐位一致（EXIT=0 方可信；
M6 全部"升级收益/不伤害"对应 `results/SUBGMIX_M6_UPGRADEGATE_R1906.json`）。
产物：frozen-matched 复现断言 + agg/per_carrier/350 row（chosen,F0,decision,REG_sq,REG_or,D_F0）。

### r1907 有限 τ 菜单 CAL-only 选择协议（M7，全新 seed block）
```
python3 subgroup_mix_ranking/code/subgmmix_tau_cal_r1907.py   # 前台, ~40min 首次(16 cell 训练)
```
- 产物：`results/SUBGMIX_TAU_CAL_R1907.json`（4 carrier × 新 seed {5,6,7,8} × w-grid，
  五臂固定-τ / CAL-select / oracle-τ / naive-snooping + 全部 weak 域）。**重跑 <5s**（
  `results/tau_cal_cache_{carrier}_s{seed}.pkl` pickle 缓存 band/estimate，确定性可复现）。
- 数字逐值验证：`python3 results/M7_VERIFY_R1907.py` 前台 EXIT=0 = **28/28 PASS**。
- 新增理论注记：`THEORY_TAU_CAL_R1907.md`（Prop M7：band τ-agnostic → τ 选择零证书多重性代价，
  选择代价只在性能层）。
- 数据/依赖与冻结一致（Fashion/MNIST/KMNIST/digits/news；纯 CPU、零网络、零 GPU）。
- 注意：先清一次缓存或直接跑，runner 会自建缓存；删除 `tau_cal_cache_*.pkl` 可强制重训复现。
- **r1908 论文并入**：M6/M7 已写入主稿附录 `paper/paper.tex` §Choosing or eliminating the tolerance τ
  （label app:tau）。复现主稿：`cd subgroup_mix_ranking/paper && pdflatex×3+bibtex` → 12 页
  （正文恰 9 + refs p10 + 附录含 app:tau p12），0 err/0 overfull/0 undef。附录数字全部引用
  M6/M7 冻结 JSON，无新数据。正文主拱未改动；只读候选 r1905 包不动。
- **r1909 严格有限样本带在相对门（M8）**：
  `python3 subgroup_mix_ranking/code/subgmmix_m8_finitesample_relgate_r1909.py`
  后台─前台（~2min，复用 M6 训练）→ `results/SUBGMIX_M8_FINITESAMPLE_RELGATE_R1909.json`；
  对冻结 M6（normal 带）与 M2.5 5-seed（base 字段）EXACT 复现断言内建于 runner。
  数字逐值锁定：`python3 results/M8_VERIFY_R1909.py` EXIT=0 = **17/17 PASS**。
  理论边界：`THEORY_RELGATE_FINITESAMPLE_R1909.md`。主稿并入 app:tau 新增段 + 结论/局限性更新，
  重编译 13 页（正文恰 9）0 err/0 overfull/0 undef；只读候选 r1905 包不动。
- **r1910 严格有限样本相对门的 N*-frontier（M9，预算计价刻画）**：
  `python3 subgroup_mix_ranking/code/subgmmix_m9_nstar_frontier_r1910.py`
  前台（复用 M6/M8 split/F0/i\*；只算 6 切换行所在 fashion{1,4}+news{2,3} 4 cell）
  → `results/SUBGMIX_M9_NSTAR_FRONTIER_R1910.json`；
  base 字段对冻结 M8 EXACT 复现断言内建，数字逐值锁定：`python3 results/M9_VERIFY_R1910.py`
  EXIT=0 = **ALL PASS**。理论：`THEORY_NSTAR_FRONTIER_R1910.md`。主稿并入 app:tau M9 段，
  重编译 13 页（正文恰 9）0 err/0 overfull/0 undef；只读候选 r1905 包不动。
- **r1911 相对门 exact 带的预算非空性前沿（M10，真实采样 + 可证空证书）**：
  `python3 subgroup_mix_ranking/code/subgmmix_m10_exactband_budget_r1911.py`
  前台（fresh seed{10..14}×4 carrier，OUTER-exclusive FIT/CAL，同一 F0/i*/subgroup；预算网格
  b∈{0.25,0.5,1.0} 真实按组无放回子采样 CAL、经验重算 normal/Hoeffding/MPB + 绝对门 + status-quo，
  oracle 仅诊断；~10min 纯 CPU）
  → `results/SUBGMIX_M10_EXACTBAND_BUDGET_R1911.json`。
  数字逐值锁定：`python3 results/M10_VERIFY_R1911.py` EXIT=0 = **ALL PASS**（125 可证不可行行）。
  理论：`THEORY_EXACTBAND_FRONTIER_R1911.md`。主稿并入 app:tau M10 段（$M9/'M10'$ audited paragraph），
  重编译 13 页（正文恰 9）0 err/0 overfull/0 undef；只读候选 r1905 包不动。
- **r1912 图 fig:m10frontier（正文首图，纯 M10 JSON 可视化）**：
  `python3 subgroup_mix_ranking/code/fig_m10_frontier_r1912.py`
  → 从 `results/SUBGMIX_M10_EXACTBAND_BUDGET_R1911.json` 读全部 125 个真实切换行，画两面板：
  (a) Δ(w) vs B=Δ√b* 可行域象限（全行在 y=x 之上 = 可证空），(b) 绝对门 committed rate 随预算 b 单调。
  脚本内含 3 条断言（125 行、Δ中位 0.0139、b*中位 272.9、b*全>1）随图 EXIT=0 通过，防漂移。
  产物 `paper/fig_m10_frontier.{pdf,png}` 插入 appendix app:tau M10 段；重编译 14 页=正文恰 9
  （Conclusion p9、fig 浮 p14），0 err/0 overfull/0 undef。只读候选 r1905 包不动。

### r1915（M12 真采样审计；MGR 指令 f147b3bf33e9）
```bash
python3 subgroup_mix_ranking/code/subgmmix_m12_fresh5_budget_r1915.py   # 前台 ~40min（fresh seed 重训）
python3 subgroup_mix_ranking/results/M12_VERIFY_R1915.py                # EXIT=0 245/245 PASS
```
输出 `subgroup_mix_ranking/results/SUBGMIX_M12_FRESH5_BUDGET_R1915.json`（+ m12_cache/*/cell 断点）。
- fresh seed{10..14}，M3/M3.5 预算总点 R=floor(pi*Ncal)（pi∈{0.5,0.65,0.8,0.95}），真实无放回
  FIT/CAL/OUTER；同预算严格相对(hoef/mpb)×exact 绝对+statusquo F0 × uniform/convex-minimax 等分配。
- **如实边界**：M12 用 fresh seed{10..14}（5-seed）；M3 原表 seed{0..2}，数值块不同不直接并表；空证书
  数据特定；绝对 gains 为 OUTER 诊断非门输入。
