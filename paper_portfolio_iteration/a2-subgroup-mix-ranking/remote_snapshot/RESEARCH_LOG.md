
## r1903 (2026-08-18) — FULL 5-seed MATCHED M2 vs M2.5：收口 r1901 复验缺口
**坐标**：M5-C。纯 CPU 前台（~60min），零 GPU，零网络（news 走本地 sklearn 缓存）。

**动机**：r1901 记的诚实缺口——主比较承重是 matched 3-seed {0,1,2}，5-seed M2（0.269）只作
"独立鲁棒性视图"，完整 5-seed M2.5 网格为复现待办。因模型由 `random_state=seed` 确定性训练、
各 split 同源，把 M2.5 在 M2 相同的 5 seed {0,1,2,3,4} 上重跑即成严格全额 matched 比较。

**做了什么**：
1. 新建 `code/subgmmix_m25_paired_5seed_r1903.py`：复用冻结的 `run_carrier`（逻辑零改动），
   仅把 seed 循环扩到 5。写新冻结 JSON `results/SUBGMIX_M25_PAIRED_R1885_5SEED.json`（350 行）；
   r1885 原生 3-seed 文件保持未改（存档切片）。
2. **5-seed headline（τ=0.04, δ=0.1, 全 cov 1.0）**：M2 0.269 / M2.5-MPB 0.260 /
   Hoeffding 0.097 / Normal(渐近) 0.503。原 3-seed matched 的定性结论在 5-seed 原样成立：
   MPB 精确 ≈0.260（诚实"约等"，不称非劣不称超）、Normal 约 2×、弃答门精准、fashion/mnist
   全带窄→全 commit、digits/news 带宽→诚实低覆盖。
3. M5_VERIFY_R1902.py 增 5-seed 断言块（10 项：n/paired/MPB/Hoeffding/cov/序 MPB<M2），
   前台 EXIT=0 = **105/105 PASS**。
4. paper.tex：abstract/intro/tab:main/tab:frontier/limitations 从 matched 3-seed 升为
   full 5-seed（M2 0.269、M2.5-MPB 0.260、Hoeffding 0.097、Normal 0.503；no-gate 均 0.0031）；
   删去"5-seed M2.5 网格复验待"的 limitation。重编译 12 页（正文 8 + refs p9 + App p10-12，
   保持 ≤9 页正文）、0 err/0 overfull。
5. 文档同步：CHINESE_SUMMARY（坐标/前沿项4/局限）、RESULT_MATRIX（主端点、per-carrier r1903、
   W1 改为"约等"、W3 数字）、REPRO_README（新命令+产物）。

**诚实边界**：MPB 0.260 与 M2 0.269 差 0.009，诚实写"约等"；不称非劣不称超。每载体 6 折完整
网格、M3/门 OC 的 full 5-seed 复验仍为复现项（与 M2/M2.5 主表无涉）。

## r1901 (2026-08-18) — M5 content-complete 自查：86/86 数字锁定 + 2 处诚实修正
**坐标**：M5-C (content-checked)。纯 CPU 前台，零 GPU。

**动机**：r1900 正文压到 9 页，主表数字称"与冻结 JSON 一致"但无逐值可追溯断言。本块把
"一致"升级为**机器可复验**：worker 不得凭记忆宣称，须落盘 verifier。

**做了什么**：
1. 新建 `results/M5_VERIFY_R1901.py`：只读全部冻结结果 JSON，逐个断言 paper 每个 headline /
   appendix 数字 == JSON 值。前台 EXIT=0 = **86/86 PASS**（范围见 CHINESE_SUMMARY r1901 段）。
2. **抓到 2 处过度陈述并修正**（直读盘面 JSON 发现，非会话记忆）：
   a. gate-OC "minimax τ=0.06 达 1.0" → 实际 τ=0.08（non-spanning 曲线 0.675@0.06）。
      改 paper/CHINESE/RESULT_MATRIX/THEORY_MINIMAX 四处。
   b. App.A diag "15/880 全 tied" → 实际 11 tied + 4 行 M1 严格更差(0.0024)。改 paper App.A
      叙述+表、RESULT_MATRIX、THEORY_SKELETON。M1 机制结论（门非选择器）不变。
3. paper.pdf 重编译 EXIT=0：12 页（正文 9 + refs p10 + AppA-D p11 + AppE p12），0 over/
   undef/err。两处修正未破 9 页。
4. 文档同步：CHINESE_SUMMARY 坐标→M5-C + r1901 段；RESULT_MATRIX/THEORY 两处已改。

**诚实边界**：M2(0.269) 5 seed vs M2.5(0.495/0.257) 3 seed 的差已在 Limitations 披露；
完整 5-seed 主表复验 TBD-4 因 news 训练 ~3.5h/seed 前台整轮难及，诚实保留为非阻塞复现项。
**下一步**：(a) 把 M5_VERIFY_R1901.py 结果写进 REPRO_README；(b) 逐步向 M5 content-complete 收敛
（对照 PROJECT_DESIGN/CHINESE_SUMMARY/RESULT_MATRIX 逐项核）；(c) 如又有前台预算再啃 5-seed M2.5 主表。

## r1904 (2026-08-18) — 投稿 prep 收口：AI-use 声明 + 引用核验 + 提交核对表
**坐标**：M5-C。主稿不变 `subgroup_mix_ranking/paper/paper.tex`。纯文档/排版，零实验。
**动机**：主稿已达到 M5 content-checked（105/105 数字断言、9 页正文），但缺 ICLR 投稿硬性
prep 交付物：AI 使用声明、逐条引用核验、提交前核对表（其他显影房间 r1830 均已齐备）。
**做了什么**：
1. `code/verify_citations_r1904.py`：只读 references.bib + 手工 arXiv 官方逐字核验，前台
   EXIT=0 = **8/8 OK**。
   - arxiv 两条：maurer2009bernstein=0907.3740（标题更正 **Penalties→Penalization** 官方版）、
     sagawa2019distribution=1911.08731，均与 arXiv 页面逐字匹配。
   - 其余 6 条经典期刊登记（ClopperPearson/Dunnett/Hsu/Duchi/Saerens/Vovk）。
   - 删死条目 `hsu2016generalized`（未引用且标题截断错误）。
2. paper.tex 附录末新增 `\section{AI use statement}`；重编译 12 页 = 正文 9 + refs p10 +
   附录 A-E + AI声明 p11-12，0 err/0 overfull/0 undef，正文未顶出 9 页。
3. 新增 `SUBMISSION_CHECKLIST.md`（A 页数/编译、B 引用、C 匿名、D 理论/证明、E 实验真实性、
   F AI声明、G 复现材料逐条勾选；H 剩余待办= M3/M3.5 完整 5-seed + τ选择协议 + MGR 迁稿）。
4. REPRO_README（补 r1903+r1904 命令/产物）、CHINESE_SUMMARY（坐标 + r1904 段）同步。
**验证**：M5_VERIFY_R1902.py 前台 EXIT=0 = 105/105 PASS（数字零改动）；paper 重编译干净。
**诚实边界**：prep 未改任何 headline 数字；r1901 修正与 r1903 full 5-seed matched 结论原样。
M3/M3.5 完整 5-seed 复验仍为非阻塞复现项（m3_cache 仅 3 seed，补 seed 需 news 重训 ~3.5h/seed）。
**下一步**：a) 若前台预算把 M3/M3.5 补 5-seed；b) 监督签发后迁入正式投稿主稿目录；c) 结语
「更紧精确界 + τ 选择协议」同题研究入口。
相关记忆：[[a2-arxiv-citation-verify-api]]、[[a2-r1901-m5-verify-frozen-json]]。

## r1905 (2026-08-18) — 投稿 prep 终验闭环：逐项核实声明 vs 磁盘（零改动）
**坐标**：M5-C。主稿不变 `subgroup_mix_ranking/paper/paper.tex`。纯验证，零改稿、零重训。
**动机**：r1904 的 prep 声明（105/105、8/8 引用、12 页/0 err、9 页正文、AI 声明）必须落到当前磁盘
实际状态复核，不能凭会话记忆断言"已验证"。
**逐项复核结果**（全部前台 readback）：
1. `results/M5_VERIFY_R1902.py` 前台 EXIT=0 = **105/105 PASS**（headline 全冻结 JSON 逐值断言）。
2. `code/verify_citations_r1904.py` 前台 EXIT=0 = **8/8 OK**（arXiv 两篇+经典期刊登记，零虚构）。
3. 重编译（pdflatex×3+bibtex）→ `Output written on paper.pdf (12 pages)`；
   **0 Error / 0 Overfull / 0 undefined**（ref+citation）。
4. 用 aux 的 \newlabel 精确定位：Intro p1、Sec7(M3.5) p7-8、Related p8、Limitations p9、
   **Conclusion p9** ⇒ 正文严格结束于 p9，符合 ICLR 9 页硬上限；References p10、附录+AI 声明 p11-12。
5. 附录末 `\section{AI use statement}` 在 tex L792 确认存在。
**判断**：唯一剩余非阻塞项 M3/M3.5 补 5-seed —— M3/M3.5 是**内部自比**（adaptive vs uniform vs neyman
等策略全在同一 3-seed 集内），无跨 seed 集不可比问题（不同于 r1903 修的 M2 5seed vs M2.5 3seed 硬伤）；
且需 news 重训 ~3.5h/seed 且 M2.5/M3 split 不同无法复用缓存，ROI 远低于 r1903 那次修复。按 ponytail 跳过，
维持 Limitations 中的 3-seed 诚实披露。
**下一步（提交 MGR）**：a) 正式 author_candidate 迁移决策（worker 无权自行迁入 paper/，提请主管终审）；
b) 结语「更紧精确界 + τ 选择协议」为后续同题研究入口。
相关记忆：[[a2-r1904-submission-prep-ai-citations-checklist]]、[[a2-r1901-m5-verify-frozen-json]]、[[a2-honest-scope-theorem-vs-evidence]]。

## r1907（2026-08-18）— M5 候选装配完成 + τ 菜单 CAL-only 选择协议（M7）
- **M5 候选装配（指令 00a20a5970c7）**：一次性装配只读候选
  `paper/A2_subgroup_mix_ranking_author_candidate_r1905/`，自含（tex/pdf/bib/style/code/results/
  REPRO/CHINESE_SUMMARY/CHECKLIST/MANIFEST），无 hardlink/symlink。候选根前台重放：
  M5_VERIFY 105/105、引用 8/8、clean compile 12 页（正文9+refs p10+附录 p11-12）全 EXIT=0。
  `\author{Anonymous}`。MANIFEST.md 只锁该不可变发布包（SHA-256），不冻 mutable 主稿/源码/runner/config。
  canonical SUBMISSION_CHECKLIST 加唯一候选反链。完成后返回 mutable 主稿继续同题。
- **M7 τ-菜单 CAL-only 选择协议**（新 seed {5,6,7,8}，与冻结证 seeds 0-4 完全不相交）：
  - 预注册：菜单 T={0.01..0.05}, P0=0.5 floor `τ̂=min{τ: CR_cal(τ)≥0.5}`；总 δ=0.1；band 复用冻结
    Bonferroni normal 单侧 paired UCB（dcell=δ/(M(M-1)G)）。
  - **Prop M7**：band τ-agnostic ⇒ τ 选择零证书多重性代价，联合覆盖 ≥1−δ 对任意 CAL 依赖 τ̂ 成立；
    选择代价只在性能层（(i) 代理失配 +(ii) 网格离散 +(iii) 抽样方差，paired 区间吸收）。
  - 主数字（OUTER 结算，`results/SUBGMIX_TAU_CAL_R1907.json`）：CAL-select **CR 0.357, mean_reg
    0.0003, max_reg 0.0053, coverage 1.0**，τ̂∈{0.01×12, 0.02×4}（保守多选最紧档）。对比固定 0.04
    (CR 0.496, max_reg 0.0364)；test-snooping（oracle=naive）CR 0.539 虚高 +0.182 且 0.7% coverage 违例。
  - weak 域如实：digits 0/60 capacity 墙、news 4/100 预算墙、fixed 0.02/0.03 的 digits skew 违例。
  - 验证 `results/M7_VERIFY_R1907.py` 前台 EXIT=0 = **28/28 PASS**。缓存 pickle 重跑 <5s。
  - 写作：`THEORY_TAU_CAL_R1907.md` + RESULT_MATRIX r1907 段 + REPRO README 段。
- 下一步：把 M7 写作并入主稿（绝对证书 M2.5 vs 相对 M6 vs τ-菜单 M7 三端点互补），继续同题。

## r1908（2026-08-18）— 主稿关闭 τ 缺口：M6/M7 并入附录（三端点互补）
**坐标**：M5-C。主稿不变 `subgroup_mix_ranking/paper/paper.tex`。纯写作/排版，零新实验、零重训。
**动机**：r1907 记的下一步「把 M7 写作并入主稿（绝对 M2.5 vs 相对 M6 vs τ-菜单 M7 三端点互补）」。
冻结主稿结论 L682 仍把「tighter exact bounds + τ-selection protocol」并列为 remaining work，但
M6（r1906 τ-free 相对门）与 M7（r1907 τ-菜单 CAL-only 选择）均已完成验证、弥补了 τ 半边。
**做了什么**：
1. 新增附录 `\section{Choosing or eliminating the tolerance τ}`（`\label{app:tau}`），正文页数零增加：
   - **M6 消除 τ**：把关证书的受控对象从 oracle-best 移位到 status-quo F0（collected-mixture 点估计
     最优，操作员无框架也会跑的那个），只证「不比我本来要跑的差」，τ 彻底消失。upgrade_rate 0.663
     （mnist 1.0/digits 0.773/fashion 0.560/news 0.456）；REG_sq mean −0.0015 max 0.0 no-worse cov 1.0
     （upgraded mean −0.0022）；门价值=abstain 118 行中 naive 恒切 i* 有 23 行（19%）真变差（+0.028），
     350 行 M6 max REG_sq=0 vs naive max +0.028。证据 `results/SUBGMIX_M6_UPGRADEGATE_R1906.json`。
   - **M7 保留绝对 τ 从有限菜单 CAL-only 选**：菜单 T={0.01..0.05}，`τ̂=min{τ:CR_cal(τ)≥0.5}`（only
     CAL，绝不读 outer）；band τ-agnostic ⇒ τ̂ 零证书多重性代价，联合覆盖 ≥1−δ 对任意 CAL 依赖 τ̂ 成立。
     OUTER 结算 CR 0.357、coverage 1.0（mean_reg 0.0003/max 0.0053），选保守最紧档（τ̂=0.01×12/0.02×4）；
     test-snooping 对照 CR 0.539 虚高 +0.18 且 0.7% coverage 违例=数据窥探的诚实计价。weak 域如实：
     digits 0/60 容量墙、news 4/100 预算墙。证据 `results/SUBGMIX_TAU_CAL_R1907.json`。
2. **结论 L682 重写**：把「τ-selection protocol」从 remaining work 移出，改为「两种互补闭合」叙事
   （M6 相对 no-worse-than-F0 门 / M7 保留绝对 τ 菜单 CAL 选择），唯一保留的 remaining 是 strictly
   finite-sample 收紧 band。
3. **验证**：M7_VERIFY_R1907.py 前台 EXIT=0 = **28/28 PASS**（附录数字逐值锁定）；paper 重编译
   **12 页 = 正文恰 9 + refs p10 + 附录（含新 app:tau p12）p11-12**，**0 err / 0 overfull / 0 undef ref**
   （修复了 `\ref{prop:m2}` 为正确 `prop:p2`）。正文逐标 title 未动，正文页数未顶出 9 上限。
**诚实边界**：附录只把 M6/M7 的已冻结结果写入，未新造任何数字；M6 F0=collected-mixture 点估计最优
  只是「无框架」的一种合理部署对象；M7 band 仍为单侧 normal（渐近），绝对 τ 语义与 M6 相对语义互补
  不替代。正文 9 页主拱不受影响。MANIFEST 只锁 r1905 只读候选，本块改的是 mutable 主稿，不迁稿。
**教训**（关联 r1906/r1907）：冻结稿结论里「remaining work」若已被同题 follow-up 闭合，应回写主稿
  把该条目转为已闭合陈述，避免审稿读到过时缺口；附录纵深补齐而不动正文行数，是 page-limit 安全收口。
**下一步**：a) 若 MGR 签发迁稿，M6/M7 已并入的 appendix 随迁；b) strictly finite-sample 收紧 band
  （MPB exact 变体）作为剩余同题对照查 upgrade_rate 代价；c) 逐项过 SUBMISSION_CHECKLIST H 待办。

## r1909（2026-08-18）— 严格有限样本带在相对门（M8）：vacuous-collapse 边界刻画 + 主稿收口
**坐标**：M5-C。主稿 `subgroup_mix_ranking/paper/paper.tex`。纯 CPU 前台 ~2min（复用 M6 训练），零 GPU/零网络。
**动机**：r1908 收口 τ-selection 后结论唯一 remaining="a strictly finite-sample (non-asymptotic)
tightened band"。绝对 gate 已由前沿表 exact 列（MPB 0.260/Hoeffding 0.097）覆盖；补相对 τ-free gate（M6）
在严格有限样本带下的行为，把 remaining 双侧全闭。
**做了什么**：
1. 新 `code/subgmmix_m8_finitesample_relgate_r1909.py`：复用 M6 完全相同 split/F0/i*/UCB/门，只把相对门
   门统计 D(w)=Σw_g UCB[(i*,F0)][g]≤0 的 paired 带从渐近 normal 换成严格 Hoeffding/Maurer-Pontil
   （与绝对管道同公式）。5-seed{0..4}×4 载体。
2. **EXACT 复现**：normal 带下 decision/REG_sq/D 对冻结 M6 逐行 EXACT；base 字段对冻结 M2.5 5-seed 逐行 EXACT。
   修复了坑：M6 frozen 实为 5-seed{0..4}（350 行），但 on-disk runner import r1885 的 3-seed SEEDS；
   我在 M8 显式 `SEEDS=[0,1,2,3,4]` 对齐。
3. **结果 `results/SUBGMIX_M8_FINITESAMPLE_RELGATE_R1909.json`**：
   normal 0.6629（=M6）/ hoef·mpb 0.6457，no-worse cov 全 1.0，sq_mean_upgraded normal −0.0022/exact 0.0。
   **机制 vacuous-collapse**：226 行 i*==F0 平凡 commit；6 个 i*≠F0 真实切换提案（normal certified，
   OUTER 全 sound，REG_sq∈[−0.130,−0.034]）被 hoef/mpb 全拒绝（empirical D∈[−0.036,0.098]，exact 宽 ~0.25）。
   ⇒ 相对门唯一非平凡内容由渐近带承载；**严格有限样本使其为空**。要严格严格性用绝对门（M2.5 exact MPB 0.260 保留真内容）。
4. **写作并稿（mutable 主稿）**：app:tau 新增 "Strictly finite-sample bands in the relative gate (M6 continued)" 段；
   结论 L689 把唯一 remaining 改为"双侧闭"；Limitations 加相对门边界；**修正 app:tau 引言 seed 声明**：
   M6 实为 seeds{0..4}（与主表同），仅 M7 用{5,6,7,8}（原句误写两者都在 {5..8}）。重编译 **13 页=正文恰 9**
   （sec:concl p9），0 err/0 overfull/0 undef（修复后 paper.log 验证）。
5. **文档同步**：RESULT_MATRIX r1909 段、REPRO_README r1909 段、CHINESE_SUMMARY r1909 段、SUBMISSION_CHECKLIST H
   标 closed、新 `THEORY_RELGATE_FINITESAMPLE_R1909.md`；二级 RESEARCH_LOG 同步记录。
6. **验证**：`results/M8_VERIFY_R1909.py` 前台 EXIT=0 = **17/17 PASS**（normal=M6、exact=0.6457、n_F0=226、
   全 exact-commit trivial、6 切换提案被拒、normal 6 切换全 sound）。
**诚实边界**：M8 不新增论文数字，仅相对门加严格带对比；absolute exact=0.260 引用已冻前沿表未重跑；
D∈[−0.036,0.098] 为本数据集经验范围非最坏界，n 极小时边界可能改变；verifier 按存储精度断言（关联 a2-r1704）。
**教训**：冻结 runner 的 SEEDS 常量可能与其产出 JSON 的 seed 范围不一致（M6 runner 显式 3-seed 但 gen 了
5-seed 文件）——复现断言前先对冻结 JSON 读实际 seed 集，不要信 import 常量。相对门在严格有限样本带下
vacuous-collapse 是"带越紧越无内容"的对称物：证书的意义由允许的带宽决定，收紧到极限会删掉唯一的
非平凡信号；对审稿的诚实应答是把这写成边界刻画而非隐去。
**下一步**：a) 若 MGR 签发迁稿，M8 已并入的 app:tau/结论/局限性随迁；b) 可测"带宽/预算轴"上相对门严格带
从 vacuous 转非空洞的临界 n_g（经验 frontier）作后续同题增强；c) 余下非阻塞复现项= M3/M3.5 补 5-seed。
相关记忆：[[a2-r1906-tau-free-upgrade-gate-m6]]、[[a2-r1904-submission-prep-ai-citations-checklist]]、[[a2-r1704-repair-core-numeric-audit]]。
```

### 文件清单（r1909）
- 新：`code/subgmmix_m8_finitesample_relgate_r1909.py`、`results/M8_VERIFY_R1909.py`、
  `results/SUBGMIX_M8_FINITESAMPLE_RELGATE_R1909.json`、`results/m8_r1909.log`、
  `THEORY_RELGATE_FINITESAMPLE_R1909.md`
- 改：`paper/paper.tex`（app:tau 段/结论/limitations/引言 seed 修正）、`RESULT_MATRIX.md`、
  `REPRO_README.md`、`CHINESE_SUMMARY.md`、`SUBMISSION_CHECKLIST.md`、`RESEARCH_LOG.md`
- 不动：候选 r1905 只读包、M2/M2.5/M3/M6/M7 全部冻结 JSON。

## r1910（2026-08-18）— 相对门 N*-frontier（M9）：vacuous-collapse 升级为"预算计价"刻画
**坐标**：M5-C。主稿 `subgroup_mix_ranking/paper/paper.tex`。纯 CPU 前台（4 cell），零 GPU。
**动机**：r1909 M8 闭合并 reveals 相对 τ-free gate 在严格有限样本带下 vacuous-collapse，记录 follow-up
(b)=测临界样本预算 N*。本块补 M9。
**做了什么**：
1. 新 `code/subgmmix_m9_nstar_frontier_r1910.py`：复用 M6/M8 相同 split/F0/i*/配对差 UCB；固定实证
   per-group 统计，band 在 n_g→N·n_g（N∈{1..1000} 对数网格）下估值 → D_band(N)。pre-statistical 有效
   样本需求模型。只算 6 切换行所在 4 cell（fashion{1,4}+news{2,3}）。
2. **EXACT 复现**：chosen/F0/true_regret/D_normal 对冻结 M8 逐行（6 行）assert 4 位一致；前台 EXIT=0。
3. **结果 `results/SUBGMIX_M9_NSTAR_FRONTIER_R1910.json`**：
   MPB N\*∈{2..10}(中位 5)、Hoeffding N\*∈{2..20}(中位 10)，6/6 全在 grid 内开通、OUTER soundness
   全保持（oracle_switch_gain 0.034–0.130>0）、D 单调非增。
4. **写作并稿**：app:tau 相对门段末尾追加 M9 budget-priced 句段。重编译 13 页（正文恰 9）0 err/0
   overfull/0 undef。候选 r1905 只读包不动。
5. **文档同步**：RESULT_MATRIX/REPRO_README/CHINESE_SUMMARY/SUBMISSION_CHECKLIST 增 r1910；
   新 THEORY_NSTAR_FRONTIER_R1910.md。verifier `results/M9_VERIFY_R1910.py` EXIT=0 ALL PASS。
**诚实边界**：N* 为有效样本模型成本特征数，非免费倍增数据；只覆盖 6 切换行所在 cell（mnist 无切换、
digits/fashion/news 覆盖）；D_mpb(1) 复现 M8 拒绝；单调性为 grid 内经验；pre-statistical 假设=相同
realized 误差图景下 N 份校准。
**教训**（关联 r1909/r1895）：M8 的"严格有限样本使其为空"是预算依赖的真实现象——补一个带宽/预算轴的
N* 扫描把它转成可测量的成本；审稿攻击"exact 带删唯一非平凡信号"的正应答是前者量化了恢复该信号所需
预算，而非反驳带公式。冻结 runner retrain 全部 20 pool 在多核竞争下极慢，restrict 到 claim 所需 min cell
即可（ponytail）。
**下一步**：a) 若 MGR 签发迁稿 M9 随 app:tau 迁；b) 可把 M9 的 N* 前沿做成"证书时预算/带宽象限图"
（M9 已完成 JSON，可 inf 用）；c) 不阻塞复现项 M3/M3.5 补 5-seed。
相关记忆：[[a2-r1906-tau-free-upgrade-gate-m6]]、[[a2-r1904-submission-prep-ai-citations-checklist]]。

## r1911（2026-08-18）— 相对门 exact 带的预算非空性前沿（M10）：真实采样 + 可证空证书 + 混合规则（MGR 指令 7cc94318db8d）
**坐标**：M5-C。主稿 `subgroup_mix_ranking/paper/paper.tex`。纯 CPU 前台 ~10min（fresh seed 重训练），零 GPU/零网络。
**动机**（MGR 卡 7cc94318db8d）：M8/M9 已在冻结 seed{0..4} 上刻画相对 τ-free 门（M6）在严格有限样本带下的
vacuous-collapse 并给出预算计价上界；卡要求做 EXACT-BAND NONTRIVIALITY FRONTIER 的诚实真实采样配对物，并以
"若预算空洞则转可执行混合规则、不以空洞停止"收口。
**做了什么**：
1. 新 `code/subgmmix_m10_exactband_budget_r1911.py`：OUTER-exclusive FIT/CAL（同一 F0/i*/subgroup 定义），
   预算网格 b∈{0.25,0.5,1.0}×每组满 n_g^full，fresh seed{10..14}。每 b 真实按组无放回子采样 CAL、重选
   i*/F0/误分类、经验重算 normal/Hoeffding/MPB 三条带 + 绝对门(M2.5 UB_paired≤TAU=0.04) + status-quo；
   oracle 仅诊断（只读 OUTER 一次）。口径 δ=0.1、δ_cell=δ/(M(M−1)G)、paired 单侧、CAL_FRAC=0.3 与 M6/M8 全同。
2. **17 项断言全 PASS**：`results/M10_VERIFY_R1911.py` EXIT=0。
3. **结果 `results/SUBGMIX_M10_EXACTBAND_BUDGET_R1911.json`**：
   - 需要（agents/A2）assemble r1910 那个量级，但这里是 exact-band frontier。
   - **可证空证书**：满预算真实切换行 125/125 均 b\*_hoef>1（min 3.7/median 272.9/max 502754；
     Δ(选择器余量) min 0.00038/median 0.01391）。形式化=任何严格带 UCB_g≥μ_g ⇒ admission 逼
     Σw_g·bw_g(n_g)≤Δ(w)；Hoeffding bw=c/√n_g 与 n_g=b·n_g^full 下必要预算 b\*_hoef=[c·Σw_g/√n_g^full/Δ]²。
     ⇒ **Hoeffding exact 相对门在整个可行预算轴 b≤1 可证地无真实切换内容**（数据特定）。
   - **严格分报平凡行 vs 真实切换行**（卡②）：triv_frac 如实单独列出（digits@0.25 92%、news 40-43%），
     真实切换行随 b 单调升（digits 6→16、news 71→75、fashion 29→34）但每一条在任何预算下被 exact 带拒
     （hoef/mpb adm 全 0.0）。绝不用总 commit 率掩盖平凡行。
4. **可执行混合规则**（卡③/④）：安全部署用 **M2.5 exact 绝对门**（fresh seed 上同样保留内容且随预算单调可用：
   fashion abs_commit 0.493→0.96→1.0/abs_cov 1.0；mnist 全程 1.0/1.0）；相对门 M6 只作渐近/描述性诊断。
   弱域如实保留：news=预算墙（b=1 0.136、b=0.25 0、b=0.5 唯一 commit 行 abs_cov=0）、digits=容量墙（0.093）。
5. **写作并稿（mutable 主稿）**：app:tau 新增 `$M9/'M10'$ audited` 段（可证必要预算 b\*_hoef 条件 + 125 行
   全不可行 + 混合规则 + 弱域）。重编译 **13 页 = 正文恰 9**（Conclusion p9，app:tau p12），0 err/0 overfull/0 undef。
6. **文档同步**：RESULT_MATRIX/REPRO_README/CHINESE_SUMMARY/SUBMISSION_CHECKLIST 增 r1911；
   新 `THEORY_EXACTBAND_FRONTIER_R1911.md`；verifier 落盘。
**诚实边界**：M10 为 fresh seed{10..14} 真实采样配对物，非 M8 冻结 seed{0..4} 字节复现；两块真实切换行数
不同（6 vs 125）因 M8 只报 normal-certified 行、M10 报全部 i*≠F0 行。b\*_hoef 为保守必要下界，非开通预算，
与 M9 经验 N\* 量纲不同（M9=6 行经验 D_band(N)≤0 点、M10=125 行可证必要下界），已在 THEORY 说明不混并。
空证书数据特定，非通用不可能。单调性为网格内经验。OUTER 只读一次、从不为门输入。
**教训**（关联 r1909/r1910 记忆）：MGR 卡要求的"诚实真实采样配对物"应直接做真实按组子采样 + 可证必要下界，
把 M9 的 pre-statistical 舒适模型升级为可证空证书；"不以空洞停止"的正解=把空洞转成可部署混合规则（绝对门做
安全部署+相对门做诊断），并把弱域（news 预算墙/digits 容量墙）写成绝对门自身的正交边界。compile 一度因 cd
路径混错到 workspace 根而失败，注意 cwd=subgroup_mix_ranking。
**下一步**：a) 若 MGR 签发迁稿，M10 随 app:tau 迁；b) 可把 b\*_hoef 前沿做成证书时预算/带宽象限图（JSON 已含
Delta/n_full/逐行 b*）；c) 对弱域 news/digits 做条件预算修复（仅绝对门已开的安全子域允许相对门诊断）；
d) 不阻塞复现项 M3/M3.5 补 5-seed（沿用诚实 3-seed 披露）。相关记忆：[[a2-r1906-tau-free-upgrade-gate-m6]]、
  [[a2-r1904-submission-prep-ai-citations-checklist]]、[[a2-r1909-relgate-finite-sample-vacuity]]。

## r1912 (2026-08-18) — 全稿首图 fig:m10frontier：r1911 M10 可证空证书的纯 JSON 可视化
**坐标**：M5-C。主稿 `subgroup_mix_ranking/paper/paper.tex`。纯可视化/排版块，零新实验、零重训。
**动机**：r1911 记的下一步(b)「把 b\*_hoef 前沿做成证书时预算/带宽象限图」。主稿此前全稿 0 图
（全是表格 caption）。本块为 M10 的**机制刻画**配正文首图，纯读冻结 M10 JSON，零造假数据。
**做了什么**：
1. 新 `code/fig_m10_frontier_r1912.py`：仅读 `results/SUBGMIX_M10_EXACTBAND_BUDGET_R1911.json`。
   Panel (a)=125 个真实切换行（b=1.0 全 split、fresh seed{10..14}）在 Δ(w) vs B=Δ√b\* 象限
   （对数-对数，marker 按 carrier）；因准入逼 Δ≥B（即 b\*≤1），y=x 之上即**可证空区域**，全部行
   严格落在其上=125/125 b\*>1 的可视证据。Panel (b)=绝对门 committed rate 随预算 b 单调
   （fashion 0.493→0.96→1.0、digits 单调、mnist 全程 1.0），展示 M10 混合规则的安全部署半边保持内容。
2. **防漂移断言随图 EXIT=0 通过**：脚本内 3 断言（行数==125、Δ中位==0.0139、b\*中位==272.9、
   b\*全>1）对冻结 JSON 逐值校验，防止 future 漂移破坏图。
3. **并稿**：paper.tex appendix app:tau M10 段后新增 Figure（fig:m10frontier），caption 双语机制说明。
   重编译 **14 页=正文恰 9**（Conclusion p9、app:tau p12、fig 浮 p14），0 err/0 overfull/0 undef。
   修正 caption 一处拼写 emptineess→emptiness。只读候选 r1905 包不动。
4. **文档同步**：RESULT_MATRIX r1912 段、REPRO_README r1912 命令。
**诚实边界**：图为同一 M10 JSON 的纯可视化，未新增任何数值；125 行/b\*median 272.9/Δmedian 0.0139
  均在标题文字已有；图不外推数据。fig 浮在 appendix p12 段后至 p14，正文页数不变。
**教训**：figureless 主稿的"M10 可证空证书"机制最适合配一张可行域象限图——把最紧 reviewer 攻击
  （"exact 带删唯一非平凡信号"）的正应答直接用可视证据钉死；纯 read-backs from frozen JSON 的图
  零造假风险，且脚本自带断言防漂移。
**下一步**：a) 若 MGR 签发迁稿，此项图与 M10 段随 app:tau 迁；b) 余下非阻塞复现项 M3/M3.5 补 5-seed；
  c) 若 page 预算允许，可在 panel (b) b=1.0 处标 news 预算墙/ digits 容量墙，但当前 9 页正文已满。
相关记忆：[[a2-r1906-tau-free-upgrade-gate-m6]]、[[a2-r1911-exactband-provable-emptiness-hybrid-rule]]。

## r1913 (2026-08-18) — r1912 图复现闭环 + REPRO 收口 + 候选迁移状态厘清
坐标 M5-C。mutable 主稿不变 `subgroup_mix_ranking/paper/paper.tex`。纯验证/文档收口，零新实验。
**动机**：r1912 新增全稿首图后，本块做两条闭环：1) E3 一次性前台重放 r1912 图脚本核实可复现且断言
防漂移（避免"图只在某次运行生成、无法重放"的复现风险）；2) 修正 monitor 指出的上一份报告路径笔误——
r1911 报告里把证据路径误写成 `/SUBGMIX_M10_EXACTBAND_BUDGET_R1911.json`（根目录斜杠开头），真实路径
为 `subgroup_mix_ranking/results/SUBGMIX_M10_EXACTBAND_BUDGET_R1911.json`（该文件确实存在，笔录以核实，
非未生成）。
**做了什么**：
1) 前台重放 `code/fig_m10_frontier_r1912.py`：读冻结 M10 JSON 全 125 真实切换行，两面板均还原，
   输出 paper/fig_m10_frontier.{pdf,png}，脚本内 3 条防漂移断言（125 行、Δ中位 0.0139、b*中位 272.9、
   b*全>1）随 EXIT=0 全部通过。
2) 修正 `REPRO_README.md` r1912 条目：补上缺失的编译结果行（重编译 14 页=正文恰 9，fig 浮 p14，
   0 err/0 overfull/0 undef），与 RESULT_MATRIX r1912 条目一致收口。
3) 厘清候选迁移状态：run-root `paper/A2_subgroup_mix_ranking_author_candidate_r1905`（frozen M5，exact
   bytes 只读）与 `A2_subgroup_mix_ranking_author_candidate_r1911`（successor，已折叠 M6-M11，13 页）。
  确认 r1911 successor 与当前 mutable 主稿唯一差= r1912 图块（17 行，其余逐字一致）。r1912 图尚未迁入
   任何候选（r1911 装配 11:18 早于图 11:44）；依协议 worker 不自装候选，迁稿由 MGR 终审（SUBMISSION_
   CHECKLIST 已记录迁稿为待办）。
**诚实边界**：零新数据、零重训；全部重放读冻结 JSON；图脚本断言与原 RESULT_MATRIX 数字一致，无漂移。
**教训**：报告给监控路径必须写完整真实相对路径（含子目录），根目录 `/...` 前缀会触发机械自查误报；
   新增图表后应同一回合跑一次确定性重放把 E3 证据落盘。
**下一步**：a) 若 MGR 签发迁稿，r1912 图 + M10 段随 app:tau 迁入新 successor 并重算 hash；b) 非阻塞复现项
   M3/M3.5 补 5-seed 仍保留。相关记忆：[[a2-r1906-tau-free-upgrade-gate-m6]]、[[a2-r1911-exactband-provable-emptiness-hybrid-rule]]、[[a2-r1912-m10-emptiness-frontier-figure]]。

## r1914 (2026-08-18) — M11 全分配轴闭合：M10 可证空证书从比例轴扩到全 \{分配×预算≤1\} 盒
坐标 M5-C。mutable 主稿 subgroup_mix_ranking/paper/paper.tex 同题增强（+1 段落）。纯分析/闭式,零新训练。
**动机**：M10 的 b*>1 可证空证书只在**比例型分配** n_g=b·n_g^full 推导,却断言"whole feasible axis"。
审稿可攻击：「带宽凸减——把标签向高权重组重分配（M3 水填充移到相对门）,Σw_g·bw_g(n_g) 可跌破选择
margin Δ(w),在非比例分配上复活相对门」。
**闭合（纯代数,对冻结 125 行证书数值核验）**：任意可行分配 n_g≤n_g^full, Hoeffding 宽 c/√n_g 单调递减
⇒ Σw_g·bw_g(n_g) ≥ Σw_g·bw_g(n_g^full)（全帽角点=满比例分配本身）。满 CAL 已拒（D_mpb_full>0 全 125 行,
min +0.0228）⇒ 任何重分配只能加宽、恒拒。空性是分配上单调 + 预算上单调 ⇒ 证书覆盖全文细盒非仅比例切片。
**M11 verifier**（code/subgmmix_m11_allocation_axis_closure_r1914.py, EXIT=0）：(a)125 行 b*>1+D_mpb_full>0;
(b)Hoeffding 宽全程单调 + MPB 在现实组帽≥39 单调（bw(n)≥bw(ncap) 全帽核验）;(c)盒非退化;(d)满 CAL 拒⇒全
分配拒。7/7 断言过。
**诚实边界**：MPB 稀疏 Bernstein 偏置 7L/(3(n-1)) 在极小组（n≈2）**非单调**（偏置 blow up）,如实记录:
闭合用 Hoeffding 宽（承 M10 b*）全程单调,MPB 只保证现实盒（所有 cap≥39）。未归并候选（图也未迁,待 MGR）。
重编译 14 页 = 正文恰 9,0 error/0 overfull/0 undef。
**教训**：M10"whole feasible axis"原谓词只在比例轴推导,交稿前须把如此"轴级断言"逐条对照推导域;闭式
单调性（凸减带宽·全帽角点=极小）是把比例证书免费升级为全盒证书的标准手法,零新实验纯代数。
**下一步**：a)迁稿待 MGR（图+M10+M11 随 app:tau）;b)非阻塞 M3/M3.5 补 5-seed 保留。相关记忆:
[[a2-r1911-exactband-provable-emptiness-hybrid-rule]]、[[a2-r1906-tau-free-upgrade-gate-m6]]。

## r1915 (2026-08-18) — 单调后继候选装配 + 新 M12 真采样审计（MGR 指令 f147b3bf33e9）
坐标 M5-C。mutable 主稿 subgroup_mix_ranking/paper/paper.tex。纯 CPU 前台，零 GPU/零网络。
**动机**（MGR 卡 f147b3bf33e9）：以唯一 mutable 主稿直接装配单调后继 author candidate，把 r1912 首图与
M10 段纳入；在 canonical portal 明确"当前唯一后继/历史只读/可变主稿"职责地图；预声明 fresh 5 seeds 在
M3/M3.5 预算点重跑真实无放回 FIT/CAL/OUTER，同预算比较严格相对证书、exact 绝对证书、status quo F0 与
强简单 baseline；候选内重跑承重 verifier 与 clean compile；只对不可变发布包写 manifest。
**做了什么**：
1. **M12 runner** `code/subgmmix_m12_fresh5_budget_r1915.py`：fresh seed{10..14}×4 carrier，在 M3/M3.5
   预算点 R=floor(pi*Ncal)（pi∈{0.5,0.65,0.8,0.95}）真实无放回 FIT/CAL/OUTER，同预算对比 uniform/
   neyman/sens/widthgreedy + convex-minimax 分配；逐行结算 OUTER true_regret（只读一次，非门输入）。
   结果 `results/SUBGMIX_M12_FRESH5_BUDGET_R1915.json`（前台读回，runtime 2472s）。
2. **恰 Positive 结果**：100 配置（4C×4pi×5alloc×5seed）下 exact 相对带（Hoeffding/MPB）**0 个真实切换**
   （i*≠F0 被 admit）——M10/M11 全轴空证书在 M3 预算总量处由真实采样复证（非外推）。exact 绝对门 M2.5
   全 37 个 commit cell coverage 1.0；内容随预算单调（MNIST uniform 0.773→1.0；Fashion 0→0.20@pi0.95，
   switch gain mean 0.0131/max 0.0753，最坏 harm -0.0012）；digits/news 全 0（容量/预算墙，如实保留）。
3. **M12 verifier** `results/M12_VERIFY_R1915.py` EXIT=0 245/245 PASS：0 真实切换+绝对门健全+单调+墙。
4. **写作并稿**：mutable 主稿 app:tau M10 段后新增 M12 段；重编译 14 页=正文恰 9，0 err/0 overfull/0 undef。
5. **装配**：整体拷贝 -> run-root `paper/A2_subgroup_mix_ranking_author_candidate_r1915`（r1905 frozen
   只读、r1911 前继保留），写 MANIFEST 只锁不可变发布包 SHA-256，候选根前台重放承重 verifier。
6. **canonical portal 职责地图**：CHINESE_SUMMARY 顶部与 SUBMISSION_CHECKLIST 增明确段。
**诚实边界**：M12 为 fresh seed{10..14} 真实采样；M3 原表用 seed{0..2}（3-seed），此处 reported 为
5-seed fresh，非同一 seed 块直接数值可并；绝对门 gains 为 OUTER 诊断（不在门输入）。空证书仍数据特定。
**教训**：fresh-seed 重跑把 M10 的逐组标签预算 b 换成 M3/M3.5 总量预算 pi，两者量纲不同但结论一致
（相对门真空、绝对门守住部署），这正是"相对预算墙+混合规则修复"的诚实答案：不是用相对门救内容，而是
绝对门做部署、相对门明确定义为渐近诊断。相关记忆[[a2-r1911-exactband-provable-emptiness-hybrid-rule]]、
[[a2-r1912-m10-emptiness-frontier-figure]]、[[a2-m11-allocation-axis-closure-provability]]。
