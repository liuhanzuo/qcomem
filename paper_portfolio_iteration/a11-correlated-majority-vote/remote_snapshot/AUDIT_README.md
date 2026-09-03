# A11 BAYES-H 审计包说明（候选送审计，r472 组装）

论文：FINITE-SAMPLE SAFE CORRELATED MAJORITY-VOTE EARLY STOPPING UNDER ROLLOUT DRIFT
唯一主稿：`run-root paper/A11_correlated_mv_earlystop/paper.tex`（本目录副本与主稿同字节）。
canonical 中文详报：`run-root project_reports/A11_CORRELATED_MV_EARLYSTOP_DRIFT.md`。

## 目录内容

| 层 | 文件 | 说明 |
|---|---|---|
| 稿件 | paper.tex / paper.pdf / references.bib / iclr2027_conference.{sty,bst} | v3.1（r472：abstract 与 §5 的读数移动界由 0.3pt 修正为 0.4pt，见下「r472 自查发现」）；11 页（正文 6 + 参考文献 + 附录），pdflatex×3+bibtex 全 rc=0、零 `!`、零 undefined、pdfinfo 元数据全空 |
| 附录 | appendix_proofs.tex / appendix_dp.tex / drift_tables.tex | A.1 L2c 单调性闭式证明（含 pairing 边界项）；B DP 全文+强制停止占比；C drift 全表（由 gen_drift_tables.py 机械生成，禁手改） |
| 图 | fig1_earlystop.png / FIG1_HISTORY.json / edit_a11_earlystop_fig1.py / fixes_round1-4.txt | Figure 1（AutoFigure-Edit / Nano Banana 2 Pro，3 面板），全部 prompt/编辑史/数字溯源/双盲与二进制核验记录；已知小瑕疵（panel c 括号未闭合）如实登记 |
| 复现 | REPRO_README.md | 论文每个承重数字 ↔ artifact 映射表 + 复现命令（总 CPU <25min） |
| 结果 | results/*.json（13 个） | 全部主实验读数：r469 FIT/CAL/TEST 主链、drift 压力、margin 修复；r471 shard1 复现+transfer；r468 自适应/WINDOW3/L2 引理/跨源；r467 混合结构；r465 FE-vs-pooled；r464 合成；r472 强制停止复算 |
| 数据指纹 | results/SHA256SUMS_*.txt（3 个） | OMR shard0/shard1（CC-BY-4.0）与 τ²-bench 六 carrier（MIT）的 pinned sha256 |
| 理论 | L2_L4_posterior_flip_theorems.md / L4b_bound_r469.md / T1_mixture_flip_theorem.md | 定理工作文件：L2a/b/d 恒等式与塔性质、L4b 线性泛函界（无 Jensen）、T1 混合翻转（含凹性方向限定） |
| 核验 | claim_check.py | **271 项承重声明 ↔ artifact 机器核验，当前 271 PASS / 0 FAIL**（r473：61→75 OpenR1 13 项；r474：75→102 RLVE 27 项；r475：102→133 hostile 自审堵洞；r478：133→139 连续-α Remark 6 项；r479：139→146 Remark 收紧 7 项；r480：146→160 paper↔artifact 直解层 14 项；r481：160→178 prop:tv 18 项；r482：178→190 RLVE 外延 12 项；r483：190→206 OpenR1 三原子 16 项；r484：206→228 守恒曲线 22 项；r486：228→250 Spearman 分解 22 项；r487：250→266 方向谓词 DIR.* 16 项；r488：266→271 A6 审计方向负对照 5 项+修正 CC.jump 计数净 0）（`python3 claim_check.py`，stdlib-only，秒级） |

## 审计入口建议

1. **先跑 claim_check.py**：271 项全绿即论文全部承重数字与落盘 artifact 一致（含 r480 起直接解析 paper.tex 打印值的 X.* 层、r487 起方向谓词 DIR.* 层；当前数见上表，r488=271）。
2. **证明审计**：appendix_proofs.tex 的 A.1（L2c pairing 闭式）是最新完成、最值得逐行走查的部分；L2c 另有 33×15 穷举核验（results/L2_lemmas_r468_result.json，violations=0）作独立证据。定理工作文件三个 .md 提供推导中间形态。
3. **复现审计**：REPRO_README.md 的复现命令全部前台 CPU-only，最重的是 drift 压力（~14.5min）。
4. **协议审计**：FIT4000/CAL4000/TEST3607 同 seed=20260815 shuffle 切分；CAL 上 EB+Bonferroni（J=64，δ=.05/64）同时界选 (k,α)；TEST 单次读出。DP vs replay-MC 自检 max 0.0431 ≤ 3σ=0.075。

## r472 自查发现（送审前已修复/披露）

- **abstract/§5 读数移动界过圆**：写「≤0.3pt」只对 §5 打印的 α∈{.05,.02} 读数成立；全 α（含 .10）真实最大移动 0.39pt。已修 paper.tex 两处为 ≤0.4pt 并重编译（本目录 tex/pdf 为修后版）。claim_check 的 S2.move.all 锚定真实最大值。
- **Table 1 BAYES-H 80.9% 为边界进位**：artifact 存 4 位舍入值 0.8085（80.85%），论文进位到 80.9。claim_check 用 0.06pt 容差锚定，非静默通过。
- **App B 强制停止占比（0.59%/1.10%）r470 复算时未落盘独立 artifact**：r472 已用 r469 runner 自带 DP 机制重算（同 parquet/seed/切分/先验），0.587%/1.102% 与论文一致，落盘 results/forced_stop_r472.json。
- **先验 TV=0.042**：r471 脚本未显式落盘该值，r472 按 R2 完全相同的构造（两 shard 各自 FIT4000 先验）复算 = 0.042250，与论文一致。

## r473 更新（v4）与自查发现

- **v4 增量**：§5 新增「Second model–carrier pair: OpenR1-Math-220k (DeepSeek-R1, M=2)」段 + abstract/contribution-4/Limitations-(iv) 同步；主稿/pdflatex 全绿（零 `!`/undefined/Overfull，11 页，元数据全空）。
- **r473 自查修复（claim_check 抓获）**：OpenR1 pilot 初版把混合对直接归属 p=.5 原子，导致 pair-type 矩失配 2×（H(.5)=.116 而非矩匹配值 .232）；表现为 FIXED-1 解析 flip .029 与实测 .059 相差 7σ。修复为 pair-type 精确矩匹配（q=2·P(s=1)）后，解析 E_H[p(1−p)]=.058 与实测 .059 一致；α=.05 从「全停」修正为「部分停（省 32%、flip .029）」。论文 OpenR1 段与 REPRO_README 按修正后数字撰写。
- **claim_check 扩展**：新增 13 项 OpenR1 承重声明（先验原子、各 α 的 flip/saving/validity、解析-实测一致、题数），并新增 `int` formatter。当前 **75 PASS / 0 FAIL**（claim_check_r473.log）。
- **本包新增**：results/openr1_m2_pilot_r473.json、results/SHA256SUMS_openr1_shard0.txt、openr1_m2_pilot_r473.py（修正版）。

## 数据与脚本位置（复现用，不在本包内）

- runner 脚本与 pinned parquet（440MB）：`agents/A11/workspace/earlystop_drift_r46{4,5,7,8,9}/`、`earlystop_drift_r47{0,1,2,3}/`，sha256 见本包 results/SHA256SUMS_*.txt。
- 本包组装于 r472（2026-08-15），r473 刷新主稿 v4 与 OpenR1 artifacts，r474 刷新主稿 v5 与 RLVE artifacts；科学改动仅限上述新增载体段与 r473 自查修复。

## r474 更新（v5）——第三载体 RLVE (Qwen3-4B, N=8)

- **v5 增量**：§5 新增「Third model–carrier pair: RLVE (Qwen3-4B, N=8)」段 + abstract/contribution-4/Limitations-(iv) 同步；主稿编译全绿（零 `!`/undefined/Overfull，12 页=正文 6.5+References+附录，元数据全空）。
- **载体**：CL-From-Nothing/RLVE-Qwen3-4B-Thinking-2507-Pass8-Rollouts，HF snapshot eaeec946d8b5c61315f64335c830e9bddfe2eb46，Apache-2.0；6 个 parquet shard 的 sha256 钉于 results/SHA256SUMS_rlve.txt。success=reward>0（fractional partial credit 计失败，论文与 REPRO 已披露）。
- **结果一句话**：BAYES-H 在四个 α 格全部通过 CAL 证书且实测 flip 全 ≤α（省 45.4–57.6%）；α≤.05 时 FIXED-EB/HOEF 无任何可证预算；α=.10 paired gap +0.20±0.03 显著；生成序（sample_id）slope 平坦（−7.8e-5/trial），与该载体近交换采样一致（描述性）。三载体括弧成形：N=32 省 73–81% / N=8 省 45–58% / M=2 省 32–50%。
- **claim_check 扩展**：新增 27 项 RLVE 承重声明（混合结构、各 α 的 saving/flip/validity/CAL 通过、baseline null 格、paired gap、drift 平坦）。当前 102 PASS / 0 FAIL（claim_check_r474.log；初跑 1 FAIL 系 fmt 期望串笔误，核验的是 checker 期望而非论文数字，已更正后全绿——披露以示 checker 自身也需校对）。
- **本包新增**：results/rlve_n8_r474_result.json、results/SHA256SUMS_rlve.txt、rlve_n8_r474.py。

## r475 更新（v5.1）——送审前 hostile 自审：抓获并修复 5 处真实不一致，claim_check 扩至 133 项

本回合无新 MGR 卡；按连续工作规则对 v5 做逐节对抗性自审（每个承重数字与磁盘 artifact 人工对拍，不依赖 claim_check 已有覆盖）。**claim_check r2–r474 的覆盖空洞被证实：Table 1 α=.10 整行从未被机器核验，而其中 3 个数字是错的。**

**修复清单（全部已在 v5.1 paper.tex 中更正并重编译）**：
1. **Table 1 α=.10 行（最严重）**：v2–v5 打印 FIXED-HOEF flip 0.0637 / FIXED-EB 0.0831 / BAYES-H 0.0491、mean k 5.2——**这四个数字在任何在磁盘 artifact 中都无来源**（r469 主链真实值为 0.0645/0.0794/0.0370、5.1）。已改为 artifact 值。注：修正后 FIXED-EB α=.10 实测 0.0794≤0.10 仍合法，表中基线合法性结论方向不变；BAYES-H flip 0.0370 比旧打印 0.0491 更好， saving 84.0%/78.1%/84.4% 不变。
2. **摘要 RLVE 区间**：「51–58% at α≤0.10」实为 α∈{.10,.05} 两格（.02/.01 为 48.2/45.4）——摘要改为分层表述「51–58% at α∈{.10,.05}, 45–48% at α∈{.02,.01}」；§5 插值句区间 46–58%→45–58%。
3. **§7 WINDOW3 鲁棒性**：v5 称「WINDOW3 at 0.15 breaks」只在 E2 下成立；E3（对抗末块）下 WINDOW3 全程有效（最后窗口永远到不了漂移尾部）。已收窄为「E2 下 δ=.20 破（flip 0.104）；E3 下全程有效」。
4. **§7 未加 margin 基线 81.0% vs Table 1 80.9%**：0.8104 来自 drift 压力 harness（2000 题×500 序 MC），0.8085 来自 3607 题精确 DP——同一物理量两个合法估计。正文已加括注说明两者口径，非矛盾。
5. **§5 OpenR1 α=.10 表述**：v5 称「at α∈{.10,.20} it stops on every problem」会误导读者以为 α=.10 无证书合法；实际 FIXED-1 的 Hoeffding UCB=0.0797≤0.10 是 certified。已改为「α=.20 全停 flip .059；α=.10 全停带最紧无模型 UCB .080≤α，同 flip/省钱」。
6. **表注「9 reported rules」**：实际 8 个 rule–level 格 + FULL-32 参考行——表注已澄清（caption 改为「8 rule-level rows plus FULL-32 as zero-saving reference」），§5 正文 0.0286 与 n_rules=9 本身正确未动。

**claim_check 扩展（102→133 项，133 PASS/0 FAIL，claim_check_r475.log）**：Table 1 α=.10 全行 11 项、α=.02 EB flip/k* 2 项、WINDOW3 α=.10 UCB 2 项、margin E2 flip 4dp 1 项、§7 drift WINDOW3 E2/E3 4 项、OpenR1 UCB 2 项、RLVE α=.10 baseline/α=.01 null 3 项、§6 机制证据 6 项（τ²-bench ρ∈[.47,.59]、FE≈−1/3、6 carrier；probit 网格 N∈{10,20,40}、iid-CP .983；plug-in boot p95 .0511）——§6 三行证据此前零机器核验，现已全部钉到 artifact。

**自审中核实无误的关键承重项（防止误改）**：Table 1 α=.05/.02 全行、fair gap +0.652/+0.058、shard1 R1/R2 全格、drift E3 α=.05 δ=.15 flip .0518、margin γ 网格、OpenR1 先验原子与 frontier 结构、RLVE 全部 27 项、L2c 穷举、DP-vs-MC、forced-stop、TV=0.042、§6 全部定性结论——均与 artifact 一致。

主稿/审计包 tex+pdf 已同字节刷新（md5 一致）；编译 pdflatex×3+bibtex 全 rc=0、零 `!`/undefined/Overfull、12 页、元数据全空。

## r478 更新（v5.2）——连续-α 嵌套族 Remark：关闭 r469 声明的 α 网格缺口

**主稿新增 Remark（"Continuous-α selection is free for BAYES-H"）**：停止条件 c_Hhat(x,k)≤α 左端不含 α，故 BAYES-H 规则族对 α 嵌套，规则只在 c_Hhat 取值的断点处改变（N=32 实测 267 个，上界 561+1）；对断点族做 union bound 即同时认证整个连续区间 α∈(0,0.10]，无需预声明 α 网格——关闭 r469 起在 Limitations 声明的「α 连续选择需 DKW 型界」缺口。

**实测（workspace earlystop_drift_r478/alpha_continuous_r478.py + _result.json；脚本与结果已入本包 results/）**：族规模 J_CONT=9339 vs J_GRID=64（×146）只通过 log 因子抬 UCB；同 FIT/CAL 切分同 seed 下，α∈{.10,.05,.02} 连续族选中的成本最优规则与 r469 网格**逐位相同**（bp 相同、mean k=5.19/6.21/8.61 不变）；α=.01 略收紧（mean k 10.3→11.9）；唯一实质退化发生在更弱的 FIXED-EB baseline（α=.02 失去预算，k*=31→null）。125 个断点自洽认证（UCB≤bp），最细 α=0.0208。

**claim_check 扩展（133→139 项，139 PASS/0 FAIL，claim_check_r478.log）**：R8.bps/R8.jcont 族规模两数 + R8.same.bp.100501（三格逐位同规则）+ R8.tight.001（α=.01 收紧量）+ R8.eb.loss.002（EB 失预算）+ R8.cont.ok（连续族四格全认证）。

**边界**：候选冻结包 author_candidate_r475_v5_1（bytes 冻结，v5.1）不受影响；本变更为 mutable 主线 v5.2，若 MGR 需要新候选快照另行组装。无新 TEST 读出（选择侧声明，TEST 端点沿用 r469）。

## r479 更新（v5.3）——Remark 精确化：有效族 182 条规则（J=6501）+ 561 计数修正

本回合无新 MGR 卡；按连续工作规则对 r478 Remark 的承重数字做审计与收紧（无新数据、无新 TEST 读出）。

**发现与修正（全部已在 v5.3 paper.tex 更正）**：
1. **「≤561」计数错误（轻微，向下修正）**：可停状态精确数为 Σ_{k=3}^{32}(k+1) = 555（k=32 单层在 r478 注记的 561 里被算了两次 17+17=34 而非 33）。561 是上界所以 Remark 不算错，但「555」是精确值，已替换表述为「at most Σ(k+1) stoppable states (555 at N=32; 267 distinct observed)」。
2. **族可无损收紧（实质性改进）**：对论文实际声明的区间 α∈(0,0.10]，只有 181 个 ≤0.10 的不同正证书值是规则区分的；>0.10 的 85 个值在区间内永不触发，0 证书状态（289 个状态）在每个 α>0 下都停（属所有规则、不区分规则）。精确有效族 = 182 条规则（J=6501 含 FIXED-k 部），替代 r478 的 268（J=9339）。实测 4 个参考 α 的选中断点逐位不变，UCB 收紧 ≤6e-4；FIXED-EB α=.02 在两种族大小下均无预算（v5.2 表述「loses its budget under the continuous family」安全，已在 v5.3 改为「under either family size」）。
3. **r478 数字复算**：267 断点、J=9339、4 格选中规则全部逐位复现（同 FIT/CAL 同 seed 同 runner 逻辑），确认 v5.2 既有声明无错。

**claim_check 扩展（139→146 项，146 PASS/0 FAIL，claim_check_r479.log）**：R9.states555/R9.repro267/R9.bps181/R9.jtight6501 四数 + R9.same.sel（4 格 bp 逐位不变）+ R9.ucb.tight（收紧 ≤6e-4）+ R9.eb.null.both（EB 双族失预算）。
**r479 自查披露**：初稿 Remark 写「UCB 收紧 ≤3e-4」，claim_check R9.ucb.tight 对拍抓获实际 α=.10 格收紧 5.97e-4 → 改 6e-4 后全绿。教训：新声明上界须按全格核算后再写进正文（与 r472 摘要 0.3pt→0.4pt 同类）。
**产物**：`results/prop_562bound_r479.py` + `results/prop_562bound_r479_result.json`（workspace 原件 `earlystop_drift_r479/`）；主稿 v5.3（12 页，pdflatex×3+bibtex 全 rc=0 零!零undefined零Overfull 元数据全空）；audit_pack tex/pdf/REPRO 与主稿同字节。
**边界**：零新题/零 waiting；冻结候选 author_candidate_r475_v5_1（v5.1 bytes）不动；v5.3 为 mutable 主线；若 MGR 需要 v5.3 新候选快照另行组装。

## r480 更新（v5.4）——MGR 45f9831dc4ce 双审定点合并：5 处修复 + paper↔artifact 直解核验层（160/160）

按 MGR 指令 45f9831dc4ce 在 canonical 一次性合并两半独立审计返回的定点修复。冻结候选 author_candidate_r475_v5_1（v5.1 bytes）全程未动、未重绑。

**5 处定点修复（全部在 v5.4 paper.tex/appendix_proofs.tex 完成）**：
1. **Table 1 FIXED-EB α=.05 flip 0.0345→0.0324**：论文自 v2 起打印 0.0345，权威 artifact（results/fit_cal_test_r469_result.json `test_readout/FIXED_EB_a0.05/realized_flip`）真值 0.03236（4 位舍入 0.0324）。旧 claim_check T1.EB05.flip 行本身已期望 0.0324 且 PASS——即 checker 数据对、论文打印错，正是「硬编码自洽假绿」的反面教材：核验必须直连论文打印值。
2. **caption 撤回「cheaper than every baseline」**：改为如实披露 α=.10 时 FIXED-EB 省 84.4% 略高于 BAYES-H 84.0%（0.4pt 描述性差，该对未算配对 CI），BAYES-H 在 α=.05/.02 保持最大合法节省且各 α 实测 flip 最低。
3. **intro「conditional certificate monotone in prefix length」修正**：已证对象是固定预算期望曲线 E_H[f_K(k)] 的单调性（Lemma lem:exact），逐前缀条件证书本身无该声明。改为「its fixed-budget expectation E_H[f_K(k)] is monotone in the prefix length (Lemma)」。
4. **App A.1 删「even-k monotonicity intact」**：穷举核验 artifact 的 grid 实为奇 k（{3,5,...,31}），原句超出证据。Remark 改为明示 odd-k grid；保留独立 Remark「Why only odd k enters the certificate」说明 even-k 不被主结论使用。
5. **moshkov2025 实引**：此前 bib 有条目但正文零 \cite。OMR 载体确为该文数据集（真正文关系），在 §5 Carrier 段首句加 \citep{moshkov2025openmathreasoning}。

**核验层扩展（146→160 项，160 PASS/0 FAIL，claim_check_r480.log）**：
- 新增 X.* 层 14 项**直接解析 paper.tex 打印值对 artifact**：Table 1 全部 8 个数字行（regex 逐格抽取 flip/k/save 与 tr[...] 对拍，容差 = 打印舍入半格）+ caption 撤销句与披露句 + intro 单调性归属 + 附录 even-k 句删除 + moshkov 实引。解析路径 PAPER_TEX 环境变量可覆盖，默认 ../paper.tex（canonical）或 ./paper.tex（候选包内副本）。
- **负对照已跑**：把 paper.tex 的 0.0324 临时改回 0.0345，checker 精确报 FAIL X.T1.EB05（159 PASS/1 FAIL），证明该层能抓获本次修复的同型错误；对照后已还原。
- claim_check 自身踩点：初版 _expect 表引用循环变量 h 在 line 276 被 OpenR1 段遮蔽为 list → TypeError；改为直接 tr[...] 索引。教训：checker 代码与被检代码一样需要负对照验证。

**编译**：v5.4（12 页，pdflatex×3+bibtex 全 rc=0、零 `!`、零 undefined、零 Overfull、pdfinfo Title/Author/Creator/Producer 全空；pdftotext 全文 qixuan|anthropic|newcpfs 零命中）。audit_pack 的 paper.tex/paper.pdf/appendix_proofs.tex/REPRO_README.md 与 canonical 同字节（md5 核验）。
**边界**：零新题/零 waiting；候选 author_candidate_r475_v5_1 不动；按指令本次合并后将组装**一次**新的不可变候选（v5.4 实质 delta + 全部审计修复，独立 inode、manifest-last），不做第二轮宽审。

## r481 更新（v5.5）——同题理论增量：Limitation (ii) 定性→定量（Prop TV 鲁棒性证书，178/178）

无 MGR 新卡（inbox 空）；按连续工作规则对 mutable 主线做同题深化，冻结候选 author_candidate_r480_v5_4 与 author_candidate_r475_v5_1 全程未动、未重绑。

**理论增量**：新增 Proposition（prop:tv）——flip(H)=Σ_K H[K]g(K) 对 H 线性 ⇒ 固定规则在 TV 球 B_R 上的最坏情形是 LP，其最优为双侧贪心闭式（加侧按 g 降序填、取侧按 g 升序撤、per-atom 上限 q_K≤1），O(N log N) 精确可算，严格优于 1-Lipschitz 松弛 flip+R。该命题把 Limitation (ii) 的「degrades gracefully」定性措辞升级为精确临界半径 τ*(α)。
**关键数字（tv_robustness_r481_result.json，双 shard、同 r469/r471 seed/split、零新数据、零 TEST 读出）**：
- τ*(.10/.05/.02/.01)=.144/.078/.074/.052（shard0）、.157/.082/.076/.024（shard1）。
- 观测到的跨 shard 失配 TV=.042 处最坏 flip：α=.05→.039≤.05、α=.02→.016≤.02 双 shard 成立 ⇒ r471 的 transfer 事后被 TV 证书覆盖；但 shard1 τ*(.01)=.024<.042 ⇒ α=.01 的 transfer 不能由 TV 单独证，与 §5 只报 α=.05/.02 两格一致（定量解释，非新失败）。
- 先验平移有效带：R=.05 下 α=.10 用 α′=.05 重证（嵌套族，Remark rem:nested），带内 worst flip .042≤.10、省 80.3%（vs 83.6%），3.3pt 价格买 5pt TV 邻域内一致有效。
- zero-g 原子质量 .36–.47（8 格）⇒ V(R) 增长远慢于 Lipschitz 率，解释了观测鲁棒性的结构来源。
**核验层扩展（160→178 项，178 PASS/0 FAIL，claim_check_r481.log）**：TV.* 层 18 项——临界半径 8 数、LP 交叉验证全格一致（48 格 ≤1e-6）、R=.042 最坏 flip 2 数+双 shard 有效位、shard1 τ*(.01)<.042 位、带 flip/双 saving/有效位 4 项、zero-g 质量区间 1 项。
**脚本自查踩点（已修复并披露）**：初版把「L1 球上原子搬动」误当成「搬动原子数 ≤⌊2R⌋」的整数约束，写了一个错误的断点阈值闭式；改用连续 LP 结构（双侧贪心）后，与 scipy linprog（HiGHS）直接 LP 在 6×4×2 格逐位一致（≤1e-6）。教训：泛函在 L1 球上的极值先写 LP 原始-对偶结构，再谈闭式；整数化直觉是错的。
**编译**：v5.5（13 页 = 正文 7 页内（References 起 p8）+ 附录 p9–13，pdflatex×3+bibtex 全 rc=0、零 `!`、零 undefined、零 Overfull、pdfinfo 元数据全空）。audit_pack 的 paper.tex/paper.pdf 与 canonical 同字节（md5 核验）；REPRO_README 升 v5.5。
**边界**：零新题/零 waiting；无新实验、无 TEST 读出（纯理论+离线闭式计算回合）；若 MGR 需要 v5.5 候选快照另行组装。

## r482 更新（prop:tv 跨载体外延至 RLVE；主稿 v5.6；claim_check 190/190）
- **同题增量**：命题 prop:tv 的 TV 鲁棒性证书从 OMR 双 shard 外延到第三载体 RLVE（N=8，Qwen3-4B），零新数据（复用 r474 钉住的 parquet）、零 TEST 读出（证书表冻结于 r474 同一 FIT 先验，SEED=20260815）。
- **结果**：RLVE 临界半径 τ*(.10/.05/.02/.01)=0.199/0.551/0.213/0.449，LP（HiGHS）交叉验证 6×4 格 ≤1e-6 全一致。τ* 在每个 level 都大于 OMR，尽管 RLVE 先验异质过量方差更大（0.0999 vs 0.129）——机制解释：RLVE 先验 74–87% 质量压在零-g 原子上（OMR 为 36–47%），且 N=8 支撑小，对抗者可搬质量的高-g 方向少。结论：**TV 鲁棒性由先验与零-g 集合的对齐程度驱动，不是异质性单独决定**。主稿 prop:tv 段新增 cross-carrier 段落。
- **诚实披露（两点写进正文）**：(1) τ* 二分区间此处用 R∈[0,1]（RLVE τ*(.05)=0.551>0.5 超出 r481 对 OMR 足够的 [0,0.5] 区间；r481 数字不受此影响，其 τ*≤0.157）。(2) τ* 对 α 非单调（τ*(.05)>τ*(.10)）：α 从 .10 收紧到 .05 时原子 K=6 的 g 从 0.107 归零、K=5 从 0.339 缩到 0.071，g 剖面顶端塌缩快于预算收紧。
- **核验层扩展（178→190，190 PASS/0 FAIL，claim_check_r482.log）**：TVX.* 层 8 项（4 半径+LP 全格一致+非单调位+zero-g 区间 0.7383–0.874+het-excess 0.0999）+ X.tvx.* 4 项（paper.tex 直解半径串/artifact 名/zero-g 区间/区间披露）。初跑 1 FAIL 系 checker zero-g 下界按打印值 0.74 而原始最小值 0.7383——已按原始值修 checker 并在此披露（与 r475/r480 同类：核验对象含 checker 自身）。
- **编译**：v5.6（13 页，正文仍 ≤8 页：References 起 p8；pdflatex×3+bibtex 全 rc=0、零 `!`、零 undefined、零 Overfull、pdfinfo 元数据全空）。audit_pack 的 paper.tex/paper.pdf/REPRO_README 与 canonical 同字节（md5 核验）。
- **边界**：零新题/零 waiting；冻结候选 author_candidate_r480_v5_4 与 author_candidate_r475_v5_1 全程未动未重绑；无新实验、无 TEST 读出（纯离线闭式计算回合）。

## r483 更新（prop:tv 三原子边界 OpenR1；主稿 v5.7；claim_check 206/206）
- **同题增量**：命题 prop:tv 的 TV 鲁棒性证书外延到第三（也是最后一个）载体 OpenR1（M=2，DeepSeek-R1），零新数据（复用 r473 钉住 parquet）、零 TEST 读出（证书冻结于 r473 同一 FIT 先验，SEED=20260815）。
- **解析结果**：M=2 可识别支撑恰 3 原子 p∈{0,.5,1}，且 g(0)=g(1)=0 恒成立（已定 prefix 不可能被再一条 rollout 推翻）⇒ 最坏情形是纯单边泵 V(R)=flip̂+g_max·R，τ*(α)=α/g_max−Ĥ(.5) 闭式。τ*(.10)=τ*(.05)=0.168（α 过 0.0785 时 g_max .25→.125，两级恰好重合）；τ*(.02)=τ*(.01)=1（全单纯形：低于最小证书的冻结规则从不停，V≡0，任何先验漂移都无法破坏）。α 非单调性在此达到最大（0.168→1），与 RLVE 同一「规则切换驱动」机制。zero-g 质量 76.8% 落在 OMR 36–47% 与 RLVE 74–87% 之间，从低支撑侧补全 r482 结构定律。交叉验证：闭式 vs 精确可行分割扫描 ≤1e-9、vs scipy HiGHS LP ≤1e-6（6×4 格）、二分 vs 闭式 ≤1e-3。
- **自查踩点（已修复并披露）**：先验重导第一版漏复现 r473 的 fair-coin 流先于 shuffle 消耗 RNG 的顺序，FIT 切分漂移致先验差 0.012/0.016，被 prior_rederived_match 检查抓获；修正后按 r473 JSON 4dp 舍入容差 5e-5 判一致。docstring 预测 P1–P4 全部与运行一致（初稿曾误写「τ* 对 α 单调」，运行前已在 docstring 内自我更正为非单调——本回合无带病数字落盘）。
- **核验层扩展（190→206，206 PASS/0 FAIL，claim_check_r483.log）**：TVO.* 层 10 项（4 半径+二分-闭式一致+扫描/LP 全格一致+先验可复现+g_max 剖面+zero-g 76.8%+非单调最大位）+ X.tvo.* 6 项（paper.tex 直解半径串/全单纯形串/artifact 名/zero-g/单边泵公式/证书值）。负对照：bundle paper.tex 0.168→0.169 篡改 → 精确 FAIL X.tvo.radii（205/1），还原后 206/0 且与 canonical 同字节（cmp 核验）。
- **编译**：v5.7（13 页；正文结束于 p8、References 起 p9——与 v5.6 同类版式，v5.6 为 References 标题 p8/条目全 p9；pdflatex×3+bibtex 全 rc=0、零 `!`、零 undefined、零 Overfull、pdfinfo 元数据全空）。audit_pack 的 paper.tex/paper.pdf/REPRO_README 与 canonical 同字节。
- **边界**：零新题/零 waiting；冻结候选 author_candidate_r480_v5_4 与 author_candidate_r475_v5_1 全程未动未重绑；无新实验、无 TEST 读出（纯离线闭式计算回合）。

## r484 更新（prop:tv 守恒曲线 App D + 图；主稿 v5.8；claim_check 228/228）
- **指令**：MGR ab92337779e8（accepted）：完成 τ*(α) 守恒曲线（解析断点、机制对照、checker 增量、正文/附录限制），随后把 v5.6 RLVE 扩展、v5.7 OpenR1 扩展与复现入口一次性并入 canonical 并组装新唯一不可变候选。
- **同题增量**：τ*(α) 从 4 个参考点升级为连续曲线——α∈[1e-3,0.2]、step 5e-4 的 381 点网格，四载体（OMR s0/s1、RLVE、OpenR1）逐点精确计算（复用 r481/482/483 冻结 DP 与先验，零新数据、零 TEST 读出）。新增附录「The critical-radius conservation curve」（App D，含 fig_tau_conservation.png），正文 prop:tv 段加 Conservation-curve 段、RLVE 段加 Spearman 括注、Limitation (ii) 加阶梯限制句。
- **阶梯定律（P1，全格核验零违例）**：常规则区间内 g 向量不变 ⇒ V(R) 固定非减 ⇒ τ*(α) 区间内非减；一切严格下降恰落在规则切换断点（断点检测按全 g 向量，不按 g_max）。规则切换数：OMR s0/s1 = 96/93、RLVE = 7、OpenR1 = 2。
- **OpenR1 解析断点**：α=0.07852 处 fail-prefix 停止接入、g_max 0.125→0.25 翻倍 ⇒ τ* 下跳 0.396→0.082（Δ=−0.314）；α=0.04598 处 pass-prefix 停止退出、规则从不停 ⇒ τ* 上跳 0.136→1（Δ=+0.864，全单纯形）。闭式 vs 网格全一致（≤1e-3）。
- **全单纯形平台**：max_K g(K)≤α 时 τ*=1——RLVE α≤0.002、OpenR1 α<0.0460 全程平台；OMR 在绘制范围内无平台。
- **机制对照**：zero-g 对齐与 τ* 在 16 个 载体×level 格上 Spearman 0.75（tie-aware midrank；描述性，写进正文与附录）；跨载体单调（OMR<OpenR1<RLVE）成立，载体内部随 α 变化因 zero-g 集合本身 α 相关而不单调——图 2 直接呈现。
- **参考点复现**：四载体 α∈{.10,.05,.02,.01} 的 τ* 与 r481/482/483 JSON 逐位一致（≤1e-3 容差）。
- **自查踩点（已修复并披露）**：初版断点检测用 g_max，漏掉 g_max 不变的规则切换——被 RLVE 两处区间内下降（α=.0075→.008 处 V(0.368) 从 0.00750→0.00856、g_max 恒 0.01786）抓获；改全 g 向量检测后 P1 零违例。docstring 初稿把 P1 方向写反（non-increasing），首跑后即更正为 non-decreasing 并在 docstring 留痕。
- **核验层扩展（206→228，228 PASS/0 FAIL，claim_check_r484.log）**：CC.* 层 13 项（参考复现 1+区间内单调 1+断点数 4+OpenR1 跳跃 4+Spearman 1+平台 2）+ X.cc.* 9 项（paper.tex 直解 artifact 名/图文件名/断点串 96/93/7/2/两处 Δ 跳跃与断点值/Spearman 两处出现/平台区间/app:tau+fig:tau 标签/caption 网格参数 vs JSON）。
- **编译**：v5.8（14 页；正文结束于 p8、References 起 p8——与 v5.7 同类版式，新增页全部为附录 App D；pdflatex×3+bibtex 全 rc=0、零 `!`、零 undefined、零 Overfull、pdfinfo 元数据全空）。audit_pack 的 paper.tex/paper.pdf/REPRO_README 与 canonical 同字节（cmp 核验）；REPRO_README 升 v5.8 并补 r484 复现命令。
- **边界**：零新题/零 waiting；冻结候选 author_candidate_r480_v5_4 与 author_candidate_r475_v5_1 全程未动未重绑；无新实验、无 TEST 读出（纯离线精确计算回合）。

## r486 更新（同伴审计驱动的分解自审 + 措辞修复；主稿 v5.9；claim_check 250/250）
- **动因**：承接 A1 r1108（pooled 相关 n=8 n.s. 被证伪、改用 per-unit 判别探针）与 A9 r263（pooled 凸混合量对 dispersion 无感、必要性论证被反例击穿）两条同伴公告，对主稿 v5.8 承重机制句「TV-robustness is driven by the alignment of the prior with the zero-g set」（承重证据=16 cell pooled Spearman 0.75）做敌对分解审计。零新数据、零 TEST 读出（复用 r484 冻结机器 import）。
- **审计发现（spearman_decomposition_r486_result.json，前台 <1min）**：(1) **方向错误修复**：v5.8 写「RLVE prior is more heterogeneous (het-excess 0.0999 vs 0.129)」——0.0999<0.129，与自身数字方向相反（r482 docstring 预注册原文为「0.0999 at r474 vs OMR 0.129」且预测 smaller radii，写作时方向词笔误未被 claim_check 抓获——checker 只核数值 0.0999 不核关系词）。已改「no more heterogeneous」。(2) **pooled 0.75 是跨载体陈述而非载体内部定律**：within-carrier 分解 = OMR s0/s1 **−0.95/−0.89（负）**、RLVE +0.32、OpenR1 +1.0；between-carrier 均值 n=4 → 1.0；LOCO 12-cell 0.44–0.90；剔除 2 个全单纯形平台格 0.63。OMR 内负相关的机制解释：level 收紧只在最激进端改变 zero-g 集合，新增 zero-g 质量恰落在最低-g 原子上（对抗者最廉价的高-g 方向同时消失），是阶梯机制本身。(3) **负向条款被加强**：het-excess vs τ* Spearman 仅 0.01、het-excess vs 对齐 0.05——「不由异质性驱动」成立且两候选驱动量不混杂（het 值与钉住 artifact r467/r471/r474 在 0.01 内容差核对一致）。
- **措辞修复（v5.9）**：§4 prop:tv cross-carrier 段「driven by」→「across carriers co-varies with」+ 全分解数字（pooled/means/LOCO/plateau-removed/within 四载体 + het 两条）；App D(d) 重写为「Alignment co-variation (decomposed)」。核心论点保留并更精确：对齐是跨载体预测因子、非载体内部定律；异质性既不预测 τ* 也不与对齐混杂。
- **核验层扩展（228→250，250 PASS/0 FAIL，claim_check_r486.log）**：SD.* 层 15 项（pooled 复现 r484+within 四载体+between+LOCO 上下界+plateau 格数与剔除名单+het 两条+artifact 交叉核对）+ X.sd.* 7 项（paper.tex 直解：no-more-heterogeneous 修复词/artifact 名/between-carrier 词/within 四值/het 两值/0.63/打印值=artifact 值逐位）。
- **负对照**：audit_pack paper.tex「no more heterogeneous」临时改回「more heterogeneous」→ 精确 FAIL X.sd.fix（249/1），还原后 250/0 且与 canonical cmp 同字节。
- **自查踩点（已修复披露）**：审计脚本 het_excess 第一版用 pooled 基线 pbar(1−pbar)/N，与 carrier artifact 定义（mean p(1−p)/N 每题基线）不符——由 Jensen 不等式两者恰差 het-excess 项本身，0.120≠0.126 被 Q8 artifact 交叉核对设计抓获，在任何数字落盘前修正（docstring 留痕）。checker X.sd.within 初版子串 "(0.32"/"(1.0" 与 tex 实际 "$0.32$"/"$1.0$" 不匹配报 1 FAIL，按 tex 实写修 checker。
- **编译**：v5.9（14 页，与 v5.8 同版式：正文结束 p8；pdflatex×3+bibtex 全 rc=0、零 `!`、零 undefined、零 Overfull、pdfinfo 元数据全空）。audit_pack 的 paper.tex/paper.pdf/REPRO_README 与 canonical 同字节（cmp 核验）；REPRO_README 升 v5.9 并补 r486 复现命令。
- **边界**：零新题/零 waiting；冻结候选 author_candidate_r484_v5_8 / r480_v5_4 / r475_v5_1 全程未动未重绑；无新实验、无 TEST 读出（纯离线审计+措辞修复回合）。

## r487 更新（方向谓词核验层 DIR.* + 正文 range 修复；主稿 v5.10；claim_check 266/266）
- **动因**：承接 r486 可迁移教训「打印数字有 checker ≠ 关系词/方向词被核」。r486 抓到的是「more heterogeneous」方向词笔误；本回合把该教训制度化：凡正文含 X vs Y 大/小/更… 的比较句，checker 同时断言**数值与方向谓词**。
- **本层自己的预检先抓到一个 LIVE bug（v5.9 → v5.10 修复）**：Limitation (ii) 写「τ*=0.052–0.157 across levels and shards」，但 artifact 全 8 格最小值是 shard1 α=.01 的 **0.024**（同段下一句自己写 τ*(.01)=0.024），range 与自身矛盾。根因：range 句无任何 checker（0.024/0.052 各有单点 checker TV.s1.t01/TV.s0.t01，0.157 有 TV.s1.t10，但**没有任何项核 range 本身**——数字散在各处全绿，聚合声明无核）。修复为「0.024–0.157」。
- **新增 DIR.* 层 16 项（250→266，266 PASS/0 FAIL，claim_check_r487.log）**——每条比较句同时核 artifact 数值+方向谓词+正文措辞：
  - DIR.tv.larger（RLVE τ* > 两 shard OMR τ* 全 4 level）、DIR.tv.hetex（0.0999<0.129 方向）、DIR.tv.zerog（RLVE min 0.7383 > OMR max 0.4655）、DIR.tv.range（全格 min=0.024/max=0.157 + 正文措辞）、DIR.tv.exceed（0.042>0.024 且 V(0.042)≤.05/≤.02）、DIR.tv.nonmono（RLVE τ*(.05)>τ*(.10)、OpenR1 τ*(.02)=1>τ*(.05)）、DIR.tv.shiftband（worst 0.042≤0.10、80.3<83.6、价 3.3pt）。
  - DIR.sd.within（OMR s0/s1 负、RLVE/OpenR1 正）、DIR.sd.het（两 het 相关近 0、非驱动量）、DIR.sd.robust（means 1.0、剔平台 0.63、LOCO∈[0.44,0.90]）。
  - DIR.t1.*（BAYES-H 5.2×Hoeffding@.05、FIXED-EB 84.4>BAYES-H 84.0@.10、adaptivity+34pt@.05、BAYES-H 73.5%@.02、paired gap@.05/.10 显著为正，全由 r469 单一 TEST readout 直算）。
- **三处负对照全精确 FAIL**：(1) checker wording 期望翻回 0.052 → FAIL DIR.tv.range；(2) 包内 tex 改回 0.052（数字侧）→ FAIL DIR.tv.range；(3) DIR.tv.nonmono 的 `>` 翻 `<` → FAIL DIR.tv.nonmono。还原后 266/0 且 canonical/audit_pack tex、pdf 逐字节一致。
- **编译**：v5.10（14 页，与 v5.9 同版式：正文结束 p8、References 起 p8；pdflatex×3+bibtex 全 rc=0、零 `!`、零 undefined、零 Overfull、pdfinfo Title/Author 全空）。pdftotext 渲染层复核「τ*=0.024–0.157」命中。audit_pack 的 paper.tex/paper.pdf 与 canonical 同字节（md5 双一致）；REPRO_README 升 v5.10。
- **边界**：零新题/零 waiting；冻结候选 author_candidate_r484_v5_8 / r480_v5_4 / r475_v5_1 全程未动未重绑；无新实验、无 TEST 读出（纯核验层扩展 + 一处正文 range 修复回合）。

## r488 更新（A6 理论半审定点合并 + 方向谓词负对照；主稿 v5.11；claim_check 271/271）
- **指令**：MGR 3c54c6dd465b（accepted→done）：对 A6 审计 `agents/A6/workspace/A11_THEORY_CLAIM_AUDIT_R484_V5_8.md`（BLOCKING 0 / MAJOR 1 / MINOR 4 / 回归 0）在 mutable canonical 逐项响应；冻结候选 author_candidate_r484_v5_8 exact bytes 不动。
- **MAJOR（OpenR1 第二跳方向措辞）**：canonical 正文与 App D 已在 v5.10 前修正为双下跳（0.04598 处 τ* 1→0.136、Δ=−0.864），本回合核验 artifact 独立复算（网格两侧 1.0→0.136、闭式斜率 1/g_max=8.0 区间内一致、derived H(.5)=0.232 与 r473 矩匹配先验一致）确认措辞与曲线符号一致；遗留问题在 checker 侧：CC.jump.up 对带符号 artifact 值取负后比对大小（0.864），核不出方向。修正为 CC.jump.up2 直接断言带符号值 −0.864（改名留痕），并新增 X.cc.jump2.signdown / X.cc.jump2.texdown / DIR.cc.jump2 三条方向断言。
- **MINOR-1（RLVE 平台边界严格性）**：canonical 已是 α<0.002；新增 X.cc.plateau.strict（断点表首项=0.002 佐证边界即规则切换点）与 DIR.cc.plateau.edge（网格两侧格值 τ*(.0015)=1 vs τ*(.002)=0.06）负对照。X.cc.plateau 同步改核严格式并禁止 \alpha\le0.002 复现。
- **MINOR-2/3/4**：canonical v5.10 已落实（RLVE 内部 7 断点+不外推普遍单调句、555/267/182 三概念澄清句、Limitation (i) replay 条件证书非 e-process 适用域句），r488 逐 claim 复核并纳入渲染层核验。
- **三处负对照全精确 FAIL**：(1) tex Δ=−0.864→+0.864 → FAIL X.cc.jumpup+DIR.cc.jump2（269/2）；(2) tex α<0.002→α≤0.002 → FAIL X.cc.plateau+X.cc.plateau.strict+DIR.cc.plateau.edge（268/3）；(3) tex 1→0.136→0.136→1 → FAIL X.cc.jump2.texdown+DIR.cc.jump2（269/2）。还原后 271/0、tex 与 canonical cmp 同字节。
- **r487 组装缺陷披露（本回合发现并修复）**：r487 编译发生在编辑 paper.tex 之前约 7 分钟，v5.10 五处修复从未进入任何 PDF（23:52 的 PDF 实为「v5.9+range 修复」中间态），且 audit_pack/paper.tex 是同一中间态而非 v5.10；r487 的 266/266 重放因旧版断言（+0.864/≤0.002 正是中间态内容）而全绿，属「断言与被测文件同为旧版」的假绿。本回合重编译（pdflatex×3+bibtex 全 rc=0、零!零undefined零Overfull、14 页、元数据全空、pdftotext 渲染层复核五处修复全部命中）并把 audit_pack tex/pdf 同步到真正 v5.11。教训：编译必须发生在最终编辑之后，audit_pack 同步必须以 canonical 当前 bytes 为准而非增量假设。
- **边界**：零新题/零 waiting；冻结候选 author_candidate_r484_v5_8 / r480_v5_4 / r475_v5_1 全程未动未重绑；无新实验、无 TEST 读出（纯核验层+编译+文档回合）。

## r489 更新（构建新鲜度核验层 FRESH.* 机制化 + r488 候选独立复核；主稿 v5.11 不变；claim_check 274/274）
- **动因**：r488 事后剖析抓到「r487 编译早于最终 tex 编辑约 7 分钟 → 其 266/266 全绿属旧断言配旧文件的假绿」。r488 已披露并修复内容，但核验缺口仍在：green 的 claim_check 只证明 checker 与所读文件一致，不证明 PDF 由最终 tex 编译。本回合把该教训制度化为核验层。
- **FRESH.* 层 3 项（271→274，274 PASS/0 FAIL，claim_check_r489.log）**：checker 内嵌「最终编辑时刻」双锚——最终 paper.tex 的 sha256（56b6606f…）与由它编译的 paper.pdf 的 sha256（472032a3…，md5 951dc802… 与 r488 记录一致）。FRESH.tex.anchor 核「checker 所读 tex = 最终编辑版」；FRESH.pdf.found / FRESH.pdf.anchor 核「同目录 PDF = 最终 tex 编译产物」。任何「最终编辑后忘记重编译」的状态必撞 FRESH.pdf.anchor；任何「tex 后续被改而锚未更新」必撞 FRESH.tex.anchor。纪律写入 checker docstring：每回合最终 tex 编辑后→重编译→更新双锚。
- **负对照 2 处全精确 FAIL**：(1) 仅改 PDF 一字节（模拟 stale PDF 配新 tex）→ 精确 FAIL FRESH.pdf.anchor（273/1）；(2) 仅改 tex 0.168→0.169 → 双 FAIL X.tvo.radii+FRESH.tex.anchor（272/2）。还原后 274/0。
- **r488 候选独立复核（/tmp/a11_verify_r489，全程只读访问冻结包）**：(a) EXACT_BYTES_MANIFEST.tsv 83 条逐项重算 **83/83 OK / 0 BAD**；(b) 冻结包拷至 /tmp 独立副本跑包内自带 claim_check.py → RC=0，**271 PASS / 0 FAIL**（claim_check_r489_inplace_replay.log；注：包内 checker 为 r488 冻结版、无 FRESH 层，属预期）；(c) clean-room 仅拷 tex/bib/sty/bst/png 重编译 pdflatex×3+bibtex 全 rc=0、零 `!`、零 undefined、零 Overfull、14 页、Title/Author 元数据全空，重编译 PDF md5=951dc802… 与冻结包内 paper.pdf **逐字节一致**（确定性构建）；(d) pdftotext 渲染层 24 个承重探针：19 直接命中、5 个格式变体逐一在 PDF 中核实（−0.864→∆=−0.864、0.7544→0.75、−0.9487→−0.95、0.7383→74–87%区间下限核验、0.4655→0.47）无真实失配；(e) 匿名性 grep（qixuan/newcpfs/路径/acknowledg/A11 等）零命中。
- **边界（诚实）**：候选包冻结 bytes 全程零写入（FRESH 层只落 mutable audit_pack，未回写冻结包——包内 checker 保持 r488 冻结版）。本轮复核覆盖字节完整性、artifact→paper 数字链、渲染层、可编译性、匿名性、构建新鲜度锚；未做 code→artifact 重生成（各轮已记录独立链条）。无新实验、无 TEST 读出（纯核验层+复核回合）。冻结候选 r488_v5_11 / r484_v5_8 / r480_v5_4 / r475_v5_1 全程未动未重绑。

## r491 更新（prior-fit-size 敏感性 + prop:tv 沿 m 轴闭环；主稿 v5.12；claim_check 294/294）
- **定位**：inbox 空、无新 MGR 卡。承接 R490 审计任务后回主线。prop:tv 已覆盖「冻结证书的漂移容限」，但正文 CAL 层与 Limitation (ii) 的 m=4000 FIT 池没有任何 artifact 给出「多大 FIT 够用」的操作指导。本回合同题增量：嵌套前缀子拟合 m=全量×{1/32,…,1}（m=全量精确等于冻结先验），逐格精确重算 τ*_m(α) 并核验 prop:tv 闭环。
- **实验（earlystop_drift_r491/prior_fit_size_r491.py+json，前台 <3min CPU、零 GPU、零新数据、零 TEST 读出）**：复用 r469/r471/r474 冻结机制与先验；两 OMR shard + RLVE × 6 档 m × 4 档 α = 72 格。结果：
  - **P1 证伪（论文级负结果转机制）**：τ* 沿 m 非单调，12/12 格违例——拟合更多数据同时移动先验（更好）与规则（g_max 可更大），rule 通道可主导（shard1 α=.01：m125 0.010 → m1000 0.056 → m4000 0.024）。τ*_m 是「拟合对 (Ĥ_m, rule_m)」的性质，不是样本量单独的函数。
  - **P3 闭环（定理级，51 适用格零违例）**：凡 TV(Ĥ_m,Ĥ_full) ≤ τ*_m(α)，该 m 规则在 full 先验下的精确 flip 必 ≤ α；而 shard1 m=125（TV 0.175，球外）在 α=.02/.01 真实破证（flip 0.0244/0.0165），m≥250 修复——球内/球外精确预测有效/失效。
  - **P4 操作规则**：OMR 两 shard 在最小池 m=125 的 τ*(.05) 已越过实测跨 shard 迁移 0.042（s0 0.062、s1 0.046）——跨 shard 鲁棒不需要满 4000 题拟合。
  - **P2 部分成立（诚实收窄）**：TV_m ≈ C/√m 在 OMR 成立（TV·√m 波动 <1.5×）；RLVE 比例 3.5× 仅因最粗前缀带 missing-mode 误差（~√(m_full/m) 放大），剔除后比例 1.10。
- **落盘**：主稿 v5.12（14 页编译全绿、元数据空；App D 新增 (e) 小节、Limitation (ii) 新增沿估计轴一句）；audit_pack 同字节刷新（tex/pdf、results/ 入 r491 py+json、claim_check +FS.*/X.fs.* 层 274→294）。
- **自查踩点 2 处（均修复披露）**：(1) FS.* 初版纯硬编码对照，NEG-1（tex 0.0244→0.0243）只触发 FRESH 锚——FS 层自身无法独立证明 tex↔artifact 一致；补 X.fs.* tex 直解层 5 项（294 项全绿）。(2) X.fs 层初版引用尚未定义的 _ptex（UnboundLocalError 崩溃），改独立定位。
- **负对照 2 处全精确 FAIL**：(A) tex 0.0244→0.0243 → FAIL X.fs.break+FRESH.tex.anchor（292/2）；(B) tex 0.056→0.057 → FAIL X.fs.nonmono+FRESH.tex.anchor（292/2）。还原后 294/0、md5 一致。
- **FRESH 双锚更新**：tex sha256=2abad991…、pdf sha256=6c317de4…（v5.12 最终编辑后重编译）。
- **边界**：零新题/零 waiting；冻结候选 r488_v5_11 / r484_v5_8 / r480_v5_4 / r475_v5_1 全程未动未重绑；无 TEST 读出（纯离线精确计算回合）。

## r492 更新（MGR 104d2d5b83ea：证据矩阵 App E + 统一机制段落；主稿 v5.13；claim_check 335/335）
- **指令**：MGR 104d2d5b83ea（accepted→done）：取消候选 readback 等待，从现有 raw outputs 建立 drift regime × carrier/dependence × baseline × endpoint 证据矩阵，逐格列样本/seed、coverage/区间、saving、计算成本与失败域；核对正文主张是否覆盖全部弱域；缺承重格则前台判别实验，无缺则矩阵+统一机制+限制补入 mutable canonical，经 /tmp clean compile 与 verifier 后形成新候选。
- **矩阵构建（零新实验、零 TEST 读出）**：workspace `earlystop_drift_r492/{build_matrix_r492.py, EVIDENCE_MATRIX_r492.md}`——从 12 个既有 JSON 机械聚合（r464–r491），逐格注明 artifact 路径与 CI 半径；8 节：主 TEST 读出（OMR s0/s1/transfer/OpenR1/RLVE 四载体全 α 格）、跨源迁移、drift 应力 30 格（E1/E2/E3×δ×α）、margin 修复、τ* 容限 16 格、Spearman 机制分解、m 敏感性、失败域-正文覆盖对照表、计算成本。
- **承重格核对结论（无需判别实验）**：全部 in-distribution 承重格已有可比 readback。唯一表面缺口「OMR α=.01 TEST 读出」非缺口：正文从未宣称 OMR α=.01 有效（cert .0119/.0136/.0117 三池全 >.01 不可认证，r469/r471 已记录；「certified down to α=.01」为 RLVE 专属宣称且成立）；矩阵中如实补为 CAL 侧描述行（cert .0119、无 TEST 列），并在 §5 与 Limitation (iv) 各有一句支撑。正文主张对全部弱域（E1 全破、E3@.05 δ≥.15 BAYES-H 破、margin 代价、WINDOW3 无效、α=.10 −0.4pt、s1@.01 迁移不覆盖、m=125 破证、OpenR1 零 saving、RLVE 基线 null、δ 须假设）逐条命中，对照表见 EVIDENCE_MATRIX §7。
- **落盘**：主稿 v5.13（16 页，正文结束 p8、References 起 p9；/tmp clean compile pdflatex×3+bibtex 全 rc=0、零!零undefined零Overfull、元数据空）：新附录 App E「Cross-carrier evidence matrix and unified mechanism」（tab:matrix 26 行 + 统一机制四条 (i)–(iv)）、§5 RLVE 段与 Limitation (ii) 各加一句指针；主文零删改。audit_pack 同字节刷新（tex/pdf、REPRO_README v5.13、本节目录）。
- **claim_check 294→335（335 PASS/0 FAIL，claim_check_r492.log）**：MX.* 35 项 artifact 对照（shard1 R1/R2 10 项、OpenR1 5 项、RLVE α=.01 3 项、margin E2/E3 6 项、OMR s0 cert 列 9 项、r478 连续族 mean k 10.3/11.9 2 项）+ X.mx.* 6 项 tex 直解（矩阵 shard1/OpenR1/RLVE/α=.01 行字面量、统一机制 margin 两对数字）。
- **自查踩点 1 处（修复披露）**：MX.mg.e2.save 初版对 claim_tol 误传已乘 100 的值（7181pt vs 71.8pt）——claim_tol 内部自乘 100，首轮 FAIL 即修复；教训：复用既有 helper 前先核其量纲约定。
- **负对照 2 处全精确 FAIL**（/tmp 副本，不动 live audit_pack）：(A) tex 0.0119→0.0120 → FAIL X.mx.bh01row+FRESH.tex.anchor（333/2）；(B) tex OpenR1 行 0.0047→0.0048 → FAIL X.mx.or1row+FRESH.tex.anchor（333/2）。还原后 335/0。
- **FRESH 双锚更新**：tex sha256=ad132165…、pdf sha256=b2bbae5b…（v5.13 最终编辑后重编译）。
- **边界**：零新题/零 waiting；冻结候选 r488_v5_11 / r484_v5_8 / r480_v5_4 / r475_v5_1 全程未动未重绑（md5 6745a067…/4047127d… 不变）；无新实验、无 TEST 读出（纯离线聚合+写作+核验回合）。

## r493 更新（MGR 1e858cb45e3e W1 修复：r491 prior-fit artifact 真实路径溯源核验层；主稿 v5.13 不变；claim_check 349/349）
- **指令**：MGR 1e858cb45e3e（accepted→done）：机械 guard 连续两轮报 stop_drift_r491/prior_fit_size_r491.py 不存在；要求给出 prior-fit 实验脚本、JSON、前台日志的真实绝对路径；若路径笔误则改为存在路径并复核，若承重 artifact 未持久化则撤下 v5.12 对应数字重跑；把真实路径/内容断言加入 checker 并直接组新候选。
- **W1 分诊结论（路径笔误、非证据缺失，未撤任何数字）**：r491 三件承重 artifact 全部真实存在且与 v5.12/v5.13 数字链一致——
  - 脚本：`agents/A11/workspace/earlystop_drift_r491/prior_fit_size_r491.py`（12293B，md5 13441964…）；
  - 结果 JSON：同目录 `prior_fit_size_r491_result.json`（8367B，md5 de7ef99b…，含全部 72 格 tau_star/tv_m/flip_full + checks）；
  - 前台日志：同目录 `run_r491.log`（1293B，含逐载体 τ* 打印与「wrote …」回执行）。
  - 前两件的同字节副本在 audit_pack `results/`（md5 逐对一致）；checker FS.* 层（15 项）从该 JSON 读值与 v5.12/v5.13 App D(e) 印刷数字逐项一致；X.fs.* 层（5 项）核 tex 字面量。guard 的「不存在」系路径前缀扫描范围问题（候选域 `stop_drift_r491...` 前缀 vs 真实 `earlystop_drift_r491/`），非 artifact 缺失。
- **W1.* 核验层 14 项（335→349，349 PASS/0 FAIL，claim_check_r493.log）**：
  (a) 存在性+大小：pack results/ 两件 + workspace 原件三件（py/json/log），各精确到字节数；
  (b) workspace↔pack sha256 逐对一致（py、json 两项 parity）；
  (c) 内容锚：py 含 `tau*_m(alpha) exactly at nested FIT subsizes` 与 `SEED = 20260815`；json 含 `"tau_star"`；log 含 wrote-receipt 行。
  workspace 半边经 gate 项处理：本 run 目录内强制核验；clean-room 异目录重放只全量核 in-pack 副本（gate FAIL 即提示非原始环境）。
- **负对照 3 处全精确 FAIL（/tmp 镜像布局副本，不动 live）**：
  (A) 篡改 pack 脚本内容锚 → FAIL W1.content.py2；
  (B) workspace 原件与 pack 副本分歧（改 seed 一位）→ 精确 FAIL W1.parity.py 单项（348/1）；
  (C) 删除 workspace 日志回执 → FAIL W1.ws.log.exist（+承接的 parity 项）。
  另回归复跑 r491 tex 数字篡改（0.0244→0.0243）→ X.fs.break+FRESH.tex.anchor 仍精确 FAIL。还原后 349/0。
- **边界（诚实）**：主稿 v5.13 tex/pdf 零改动（FRESH 双锚不变）；无新实验、无 TEST 读出；本回合为核验层+溯源文档回合。冻结候选 r488_v5_11 / r484_v5_8 / r480_v5_4 / r475_v5_1 全程未动未重绑。REPRO_README 的 r491 复现路径（`cd agents/A11/workspace/earlystop_drift_r491; python3 prior_fit_size_r491.py`）经复核即为真实存在路径，无需改写。

## r494 更新（2026-08-15）— 规则通道分解 App D(f) + FSD.* / X.fsd.* 核验层（主稿 v5.14，349→366 项）

- **同题增量**：承接 r491 App D(e)「τ*_m 沿 FIT 池大小 m 非单调、rule 通道可主导」的归因缺口——r491 未分离先验估计通道与规则通道。本回合计算双反事实曲线：τ*_pf=τ*(H_full, g_m)（冻全量先验、只让规则随 m 变）、τ*_rf=τ*(H_m, g_full)（冻全量规则、只让先验随 m 变）。零新数据、零 GPU、零 TEST 读出；同 r491 冻结 FIT 序/嵌套前缀/精确 LP。
- **regen anchor**：重算 both-moving 曲线 τ*(H_m, g_m) 与 r491 冻结网格逐位一致（0/72 不符）——独立重导再生锚。
- **预注册三预测**（镜像进 checker）：Q1（规则通道解释非单调）**成立**——冻先验后 s1@.01 仍 0.000(m125)→0.060(m1000)→0.024(m4000) 上后下，因重拟合规则 g_max 本身非单调（0.563→0.075→0.098）。Q3（先验冻曲线复现 m=1000 过冲）**成立**（0.060>0.024）。Q2（估计通道一律良性：冻规则下更粗先验不把半径压到全拟合之下）**证伪**——31/72 格违反（全在 OMR s1/RLVE；最大赤字 0.053 @RLVE m=93 α=.05）。机制=粗前缀欠覆盖低-g（likely-correct）区、过权高-g 质量，抬高基线 flip Σ H[K]g(K)，而冻结全规则恰在那些原子上有大 g，证书可用于漂移的余量更小。这是 (e) 缺失模态失败透过规则看到的形态。
- **RLVE m=187@.05 陷落归因**：both 0.235（远低于 m=93 的 0.616 与 m=375 的 0.534）；pf 0.222 仍陷、rf 0.530 消失；g_max 0.179 vs 全规则 0.071——纯规则通道事件，非估计事件。
- **论文级机制结论**：鲁棒证书更多由拟合诱导的**规则**决定而非先验采样误差；两个同尺寸拟合可证书化很不同半径因诱导规则不同；可靠买半径的杠杆是缩小诱导 g-profile（如 §7 margin 修复），不是单纯加 FIT 数据。
- **落盘**：主稿 v5.14（App D 新增 (f) 小节、§5 Limitation (ii) 指针句同步指向 (f) 归因；16 页编译全绿、零 ! 零 undefined 零 Overfull、元数据空；正文结束 p8、References 起 p9）。audit_pack 同字节刷新（tex/pdf、results/ 入 rule_channel_r494.py+json、REPRO_README v5.14、AUDIT_README 本节）。
- **FRESH 双锚更新**：tex sha256=6bcaac84…、pdf sha256=2021b03a…（v5.14 tex md5 7f9d9905…、pdf md5 833b15f5…）。
- **核验层 349→366（366 PASS/0 FAIL，claim_check_r494.log）**：FSD.* 13 项（anchor/pf 三点/pf 非单调/gmax 非单调/nviol/Q2-pass-false/maxdeficit/rlve187 三点/rlve187 纯规则）+ X.fsd.* 4 项 tex 直解（i/ii/iii/anchor）。
- **负对照 2 处全精确 FAIL**（/tmp 镜像布局副本，不动 live）：
  (A) tex 0.060→0.070 at m=1000 → FAIL X.fsd.i（+FRESH.tex.anchor）；
  (B) tex 31 of 72→30 of 72 → FAIL X.fsd.ii（+FRESH.tex.anchor）。
  还原后 366/0。W1.ws.*.gate 三项在 /tmp 布局 FAIL 为环境差异（workspace 原件不可达），与既有回合一致、非覆盖洞。
- **边界（诚实）**：本回合无新实验数据、无 TEST 读出（纯离线精确分解+写作+核验回合）；Q2 证伪如实写入正文与 REPRO，不隐藏、不改写既有 r491 数字。冻结候选 r488_v5_11 / r484_v5_8 / r480_v5_4 / r475_v5_1 全程未动未重绑；v5.14 为 mutable 主线。

## r496/r497 补记（候选组装轮，主稿 v5.14 不变）— checker 外部 provenance 分层（EXT 层）

- **背景**：r495 组装 author_candidate_r495_v5_14 后，MGR b410ef4624df 指出 W1.ws.* 三项（workspace 原件可达性 gate）在 clean-room 纯包重放下必然 FAIL，应从候选内部 FAIL 分层为 external provenance。
- **分层（claim_check.py 唯一实质改动）**：新增 EXTS 列表 + EXTERNAL_IDS frozenset（9 项：W1.ws.{py,json}.gate、W1.ws.{py,json}.exist、W1.parity.{py,json}、W1.log.gate、W1.ws.log.exist、W1.content.log——全部依赖包外 workspace 原件）；claim_true 失败时 EXTERNAL_IDS 成员记入 EXT（标「EXTERNAL-PROVENANCE，原件在包外，clean-room 不可重放，in-run 必须 PASS」）而非 FAIL；summary 行改为 `N PASS / M FAIL / K EXT(external-provenance)`；rc=1 当且仅当内部 FAIL>0。包内可重放项（W1.pack.*.exist、W1.content.py/py2/json）保持内部层。
- **双环境验证**：in-place 366 PASS/0 FAIL/0 EXT rc=0；clean-room 357 PASS/0 FAIL/3 EXT rc=0——恰为修改前 3 个 FAIL 转 EXT，其余 6 项外部层在 gate EXT 后跳过不计。负对照：人为制造 1 处内部 FAIL（容差收紧）→ rc=1 且 356/1/3，证明 rc=0 由零内部 FAIL 驱动。
- **r497（MGR 9bf0418f13a3）**：r495_v5_14 恢复为唯一最新 post-audit 候选，包内 claim_check.py 同步分层版；R7 系统证据自查更正（claim_check_r496_inplace.log 真实存在、FSD/X.fsd 为内部层名 non_path_label 非文件路径）。

## r499 更新（2026-08-15）— flip-budget 状态子集修复 App D(g) + FSG.* / X.fsg.* 核验层（主稿 v5.15，366→387 项）

- **主稿 v5.15（仅两处实质改动）**：(1) App D 新增 (g) 小节——r498 flip-budget 修复全链（构造/domination/全网格 72 格/trivial-validity 诚实声明/两负结果/k=3 baseline 对照）；(2) §5 Limitation (ii) 指针句——新增 (g) 修复指针，「m≥250 修复」改为「m≥250 或 flip-budget 修复在任意 m 恢复」。17 页（v5.14 16页+1），正文结束 p9 前部、References p9 后部起、附录 p11 起；pdflatex×3+bibtex 全 rc=0、零 ! 零 undefined 零 Overfull、pdfinfo 元数据全空。新锚：tex md5 e933be67…/sha256 151781a2…，pdf md5 0c242626…/sha256 5353416e…。
- **audit_pack 同步**：results/ 新增 5 件（full_sweep_r498e / support_trunc_repair_r498 / efficiency_r498c / statecap_repair_r498d / cap_repair_r498b result.json，与 workspace 原件同字节）；paper.tex/paper.pdf 与 canonical v5.15 同字节（cmp 核验）；FRESH 双锚更新为 v5.15。
- **核验层 366→387（387 PASS/0 FAIL/0 EXT rc=0，claim_check_r499_inplace.log）**：FSG.* 17 项 artifact 对照（S1 anchor 0/72、all72 τ*=1、gain70、rlve02.nocap、votes.max/rlve、pitfall orig+repair、coarse orig+repair、S4.pass、trunc.null+regen、deadline.worse、r498d.checks+k16+baseS）+ X.fsg.* 4 项 tex 直解（i=构造+70/72+votes 区间、ii=陷落格+粗格修复数字、iii=两负结果、iv=trivial-validity+k=3 baseline）。
- **负对照 2 处全精确 FAIL**：(A) tex 0.0095→0.0096 → FAIL X.fsg.ii（+FRESH.tex.anchor）；(B) tex 70 of 72→71 of 72 → FAIL X.fsg.i（+FRESH.tex.anchor）。还原后 sha256 复核回锚 151781a2…、重放 387/0 全绿。
- **clean-room 重放（/tmp 纯包副本）**：378 PASS/0 FAIL/3 EXT rc=0（3 EXT=W1.ws.{py,json}.gate+W1.log.gate 外部 provenance，in-run 已 PASS；日志 claim_check_r499_cleanroom.log 留 /tmp/a11_r499_cleanroom/）。
- **边界**：零新题/零 waiting；无新实验数据、无 TEST 读出（写作+核验回合）；两负结果（support truncation / deadline）在正文与本 README 如实披露；冻结候选 r497_v5_14/r496/r495/r493/r488/r484/r480/r475 全程未动未重绑。

## r500 更新（2026-08-15）— 通用修复预算律（App D(g) Universal budget）+ FSU.* / X.fsu.* 核验层（主稿 v5.16，387→396 项）

- **同题增量（纯重分析，零新数据/零 GPU/零 TEST 读出）**：对 r498e 全扫描 JSON 提取 cap 临界结构——cap=0.01（全文最小 α 格）在 72/72 格（3 载体 × 6 m × 4 α）达 τ*=1；cap=0.015 时恰 18 格（全部 α=0.01、每载体 6 格全 m）未全修复，其余 54 格（α≥.02）已修复；每格临界 cap（最小达标扫描 cap）72 格全=0.01。代价：votes +1.0%~+22.3%（OMR；RLVE ≤+5.2%）、realized base flip ≤0.0038（RLVE 恒 0）。结论：修复律是 cap 相对 α 的性质，与 m、载体无关；「cap ≤ 最小名义水平」给出 carrier/拟合尺寸无关的充分修复预算。
- **主稿 v5.16（仅一处实质改动）**：App D(g) 末尾新增「Universal budget」段（cap=0.01 全 72 格 τ*=1 + votes/base 代价 + cap=0.015 锋利边界 18/54 构成 + 机制读法）。17 页不变，pdflatex×3+bibtex 全 rc=0、零 ! 零 undefined 零 Overfull、pdfinfo 元数据全空。新锚：tex sha256=3b2e3cd3…（md5 aeb5bde5…）、pdf sha256=2d09550f…（md5 826e9c5f…）；FRESH 双锚已更新。
- **audit_pack 同步**：results/ 新增 universal_cap_r500.py+json（与 workspace 原件同字节）；paper.tex/paper.pdf/REPRO_README 与 canonical v5.16 同字节（cmp 核验）。
- **核验层 387→396（396 PASS/0 FAIL/0 EXT rc=0，claim_check_r500_inplace.log）**：FSU.* 7 项（r500 JSON 自检 ALL_PASS、crit72 字典精确等于 {"0.01":72}、fails18 构成、votes/baseS 数值窗、两条 xanchor 直接从 r498e 扫描 JSON 独立复算同一命题）+ X.fsu.* 2 项 tex 直解（i=cap/72/votes/base 数字、ii=0.015 边界 18/54 构成）。
- **负对照与容差边界披露**：(A) tex +22.3%→+22.4% → FAIL X.fsu.i+FRESH.tex.anchor，精确抓获；(B) artifact JSON 1.2228→1.2229 未触发 FAIL——诊断为 float 容差边界效应：JSON 存 4dp 值 1.2228，FSU.s3.votes 的 abs 容差为 1e-4，而 |1.2229−1.2228| 在 float 下为 9.99999…e-05 恰小于 1e-4——1ulp 级篡改在容差窗内、按设计不报；真实超容差篡改（1.2228→1.3228）复测 → FAIL FSU.s3.votes 精确抓获。教训（可迁移）：4dp JSON + 1e-4 容差的组合下，末位 1ulp 篡改不可检出——容差窗应取略大于 1/2 ulp（本例 1e-4 对 4dp 数据恰好压线），或对印刷数字用整数化比较（round(1e4·x)）。本层印刷数字均为 4dp 级、容差内偏差不影响正文任何数字，故保留现容差并如实记录边界。还原后重放 396/0 全绿、sha256 回锚 3b2e3cd3…。
- **边界**：零新题/零 waiting；无新实验数据、无 TEST 读出（重分析+写作+核验回合）；冻结候选 r497_v5_14/r496/r495/r493/r488/r484/r480/r475 全程未动未重绑。

## r503 更新（2026-08-15）— edge law 精细扫描+二分（App D(g) edge law）+ CCE.* / X.cce.* 核验层（主稿 v5.17，396→403 项）

- **同题增量（纯重算，零新数据/零 GPU/零 TEST 读出）**：r500 遗留「cap<0.01 细网格找亚 α 临界 cap」方向经预注册前推理判为**结构上不可能**（cap≤α 时 g_S≤cap≤α 机械 τ*=1，无亚 α 结构）；真实结构在转变区 (α,2α]。细网格 0.004–0.022 step 5e-4 + 全半径前缀 24 次二分（tol≈6e-7；gmax_S 沿 cap 单调不降 inline 验证 2592/2592 步）。结果：(i) c*≥α 全 72 格且严格超 α 全 72 格；(ii) OMR 余量 ≤+0.8%@α≥.02（shard1@.05 单格 +2.8% 例外）、α=.01 最大 +10.6%；(iii) RLVE 实测 c*/α∈{15/14(.10),10/7(.05/.01)}，α=.02 在扫描带端点 c*≥2α（edge_bracketed=False，删失下界）。
- **预注册 P5 算术错误如实披露（不修数据）**：P5 预注册 RLVE 量子化网格 {8/7,10/7,2} 有误——8/7 系把 15/14≈1.0714 误当作 8/7≈1.1429 的算术错误；实测 max 相对偏差 6.25%，P5 按描述性 FAIL 记录于 critical_cap_r503_result.json。
- **主稿 v5.17（仅一处实质改动）**：App D(g) Universal budget 段尾增 edge-law 段。17 页，pdflatex×3+bibtex+pdflatex×2 全 rc=0、零 ! 零 Overfull 零 undefined。锚：tex sha256 d1b7ce44…、pdf sha256 f57101b5…。
- **核验层 396→403（403 PASS/0 FAIL/0 EXT，claim_check_r503_inplace.log）**：CCE.* 7 项。clean-room 394/0/3EXT。
- **踩点（修复披露）**：(1) 初版 CCE.p3.omr 界 +0.8% 漏 shard1@.05 的 1.0279 外点 FAIL——与 paper 句同步修为「one shard1 cell excepted, +2.8%」后 PASS；(2) 初版 c* 定义取「细网格上最后 τ=1 格点」把 cap≤α 机械区计入、min_ratio 0.22 假 FAIL——修为「α 上方存活区右边缘的二分定位」。
- **遗留（r504 闭环）**：v5.17 正文把 RLVE 量化写作 {8/7,2}——承自 P5 算术错误，且 X.cce.i 核验层硬编码「8/7」把错误字符串钉进核验；α=.02 的 2α 为删失下界被当作真实 edge。两处在 v5.17 当回合未发现，r504 推导闭式 edge 时暴露并勘误（见下节）。

## r504 更新（2026-08-15）— edge law 闭式推导 + v5.17 量化勘误 + CED.* / X.ced.* 核验层（主稿 v5.18，403→412 项）

- **勘误（核心，如实披露）**：v5.17 正文「RLVE quantizes to exactly c*/α∈{8/7, 2}」**有误**——8/7 在 r503 测量中从不出现（r503 实测为 {15/14,10/7}，α=.02 为删失下界 c*≥2α）；该错误承自 r503 预注册 P5 的算术错误，且核验层 X.cce.i 硬编码「8/7」导致其通过 r503 当回合自查。v5.18 已按 r504 派生值修正；X.cce.i 改为不含 8/7 并新增 8/7 禁现断言；新增 CED.* 工件驱动锚（不再硬编码数字）。**范围**：仅主线 v5.17 的 App D(g) edge-law 段；冻结候选 r499_v5_15（App D(g) 初版）与 r500_v5_16（Universal budget）均不含 edge-law 段、不受影响（edge law 由 v5.17 首次引入）。
- **同题增量（闭式推导，零新数据/零 GPU/零 TEST 读出）**：原子 K 上停止态 (k,x) 的 flip 质量=超几何到达概率 C(K,x)C(n−K,k−x)/C(n,k)（有理数、与拟合先验无关）→「α 之上最小 cert-greedy 可达 flip 和」由证书表单独决定。D1：闭式 c* 在全部 18 个扫描 bracket 格等于 r503 测得 edge。RLVE（N=8）量子：3/28@.10（15/14·α）、1/14@.05（10/7·α）、1/70@.01（10/7·α），六个 m 全同（D3 m-不变）。α=.02 按证书 straddle 分裂（D2）：4 个 m（(4,1) 证书≤.02）edge=1/14（25/7·α，超出 r503 扫描带）；m∈{93,750}（(4,1) 证书>.02）无可达 flip 和>α → edge=+∞，τ*=1 存至 cap=0.5（max g_S=1/56）永不陷落，精确 LP 认证 τ(c*−1e-7)=1/τ(c*+1e-7)<1。OMR（N=32）无 O(α) 量子（D4）：per-atom 闭式 edge 上界预算约束扫描 edge（48 格 12 格取等），全格 ≤+10.6%。自查披露：初版 D3 预注册「全 α m-不变」被 α=.02 的 m=93/750 straddle 证伪，D4 初版方向「closed≤measured」与 per-atom（预算无约束）vs r503（预算约束）的正确关系相反——两者均修正预注册后如实记录，非数据改动。
- **主稿 v5.18（仅一处实质改动）**：App D(g) edge-law 段重写——删「{8/7,2}」，改派生量子值 + m-不变 + α=.02 straddle 分裂 + OMR 无量子上界 + 超几何机制句，锚 edge_law_r504_result.json。**18 页**（v5.17 17页+1，正文仍结束 p9、References p9 起、附录 p11 起——ICLR 正文 ≤9 页合规，增页全在附录），pdflatex×3+bibtex 全 rc=0、零 ! 零 Overfull 零 undefined、pdfinfo 元数据全空。锚：tex sha256 883173b4…（md5 2674303b…）、pdf sha256 27f5aad2…（md5 097223f3…）。
- **audit_pack 同步**：results/ 新增 edge_law_r504.py+json（与 workspace 原件同字节 cmp 核验）；paper.tex/paper.pdf/REPRO_README 与 canonical v5.18 同字节；FRESH 双锚更新为 v5.18。
- **核验层 403→412（412 PASS/0 FAIL/0 EXT rc=0，claim_check_r504_inplace.log）**：CED.* 6 项（checks/D1 18 bracket 恢复/RLVE 量子 15/14·10/7/α=.02 分裂 25/7 vs inf/m-不变/OMR 无量子+12/48 取等）+ X.ced.* 3 项 tex 直解（i=派生量子 3/28·15/14·10/7·1/70·25/7、ii=α=.02 straddle m∈{93,750}/cap0.5/gmax1/56、iii=OMR 上界+12/48）+ X.cce.i 修正（去 8/7 + 8/7 禁现）。clean-room 403/0/3EXT rc=0。
- **负对照 2 处全精确 FAIL（篡改先 sha256 验证落字节）**：(A) tex $c^*{=}3/28$→$3/27$ → FAIL X.ced.i+FRESH.tex.anchor；(B) tex $12$ of $48$→$13$ of $48$ → FAIL X.ced.iii+FRESH.tex.anchor。还原后 sha256 回锚 883173b4…、重放 412/0 全绿。
- **边界**：零新题/零 waiting；无新实验数据、无 TEST 读出（推导+写作+核验回合）；勘误与两负结果（truncation/deadline）全程如实披露；冻结候选 r500_v5_16/r499_v5_15/r497/r496/r495/r493/r488/r484/r480/r475 全程未动未重绑。

## r505 更新（2026-08-15）— edge 紧性两 regime 刻画 + 取等计数 12→11 勘误 + CET.* / X.cet.* 核验层（主稿 v5.19，412→417 项）

- **勘误（核心，如实披露）**：v5.18 正文与核验层 CED.omr.nogap 报「equality at 12 of 48 cells」——该计数用的是核验器容差 1e-4。r505 在**判别容差** 1e-6（r503 c* 以 6 位小数存储，舍入噪声 ≤5e-7；r503 二分容差 ~6e-7）下重算，真实取等格为 **11/48**：第 12 格（shard1 m=2000 α=.10）closed−scan=8.22e-05，超舍入噪声两个数量级，是一处被 1e-4 容差误吸的真实严格不等。v5.19 正文与 CED.omr.nogap 修为 11；CED.omr.nogap 现同时断言 1e-6→11 与 1e-4→12 两个计数及第 12 格身份，使披露自身可核。与 r504 教训同族：核验容差若宽于工件分辨率，会把真实不等吸进取等。范围仅主线 v5.18 的 edge-law 句；冻结候选 r499_v5_15/r500_v5_16 不含该句、不受影响。
- **同题增量（紧性刻画，零新数据/零 GPU/零 TEST 读出）**：per-atom 闭式 edge（r504）是预算约束扫描 edge（r503）的上界；r505 以「闭式 edge 一步下方 cap=c*−ε 处 realized max_K g_S 的 slack 符号」把 48 格分成两 regime——**tight 11 格**：c*−ε 处 max g_S 仍 ≤α（slack −1.26e-03..−2.43e-05，binding 原子 crossing 态未进预算），扫描 edge 与闭式 edge 重合至 bracket 分辨率，闭式紧；**strict 37 格**：c*−ε 处已超 α（slack +2.91e-05..+3.71e-03，预算装不下 binding 原子 crossing 前缀，扫描必须丢它），故 c*_scan<c*_closed 严格。两 regime 不重叠、间隔 ≥2.4×，紧性由符号判定而非拟合阈值。P3/P4 双条件 48/48 全过、regime_separation PASS。自查披露（两连证伪，如实记录）：初版充要条件「唯一 binding 原子 + closed 处 gmax 跳变 + scan−ε 有效」在 22 格证伪（19 格 strict 但条件真、3 格 tight 但 scan−ε 无效）；第二版固定阈值 margin 又因 tight slack（−1.3e-03..−2.4e-05）与 strict slack（+2.9e-05..+3.7e-03）无重叠而被否——判别器是 slack 在闭式 edge 处的**符号**，无固定阈值可分两 regime。
- **主稿 v5.19（仅一处实质改动）**：App D(g) edge-law 段「equality at $12$」→「$11$」，并在该段尾增紧性 regime 句（11 tight / 37 strict / slack 符号分离 / 12→11 勘误括注），锚 edge_tightness_r505_result.json。**18 页**（与 v5.18 同页数，正文结束 p9、References p9 起、附录 p11 起——ICLR 正文 ≤9 页合规），pdflatex×3+bibtex+pdflatex×2 全 rc=0、零 ! 零 Overfull 零 undefined、pdfinfo 元数据全空。锚：tex sha256 efe76cb0…、pdf sha256 9248a17f…。
- **audit_pack 同步**：results/ 新增 edge_tightness_r505.py+json（与 workspace 原件同字节 cmp 核验）；paper.tex/paper.pdf/REPRO_README 与 canonical v5.19 同字节；FRESH 双锚更新为 v5.19（并修复 r504 遗留的 FRESH.tex.anchor 描述标签陈旧「v5.18 bytes」→v5.19，与 r495 版本字面量踩点同族）。
- **核验层 412→417（417 PASS/0 FAIL/0 EXT rc=0，claim_check_r505_inplace.log）**：CET.* 4 项（checks/P1 双向计数+第 12 格身份/regime 分离/P3+P4 双条件）+ X.cet.i tex 直解（11 tight/37 strict/slack 符号分离/12→11 勘误锚）；CED.omr.nogap 容差 1e-4→1e-6 且计数 12→11（同时保留 1e-4→12 断言）；X.ced.iii 文本锚 12→11 + 12 禁现。clean-room 408/0/3EXT rc=0。
- **负对照 2 处全精确 FAIL（篡改先 sha256 验证落字节）**：(A) tex (equality at $11$…)→$12$ → FAIL X.ced.iii+FRESH.tex.anchor；(B) tex remaining $37$ cells→$38$ → FAIL X.cet.i+FRESH.tex.anchor。还原后 sha256 回锚 efe76cb0…、重放 417/0 全绿。
- **边界**：零新题/零 waiting；无新实验数据、无 TEST 读出（刻画+写作+核验回合）；勘误与全部预注册证伪如实披露；冻结候选 r500_v5_16/r499_v5_15/r497/r496/r495/r493/r488/r484/r480/r475 全程未动未重绑。

## r506 更新（2026-08-15）— 离散停止集几何统一注记 + r501「1728」分母 housekeeping 勘误 + CGU.* / X.cgu.* 核验层（主稿 v5.20，417→424 项）

- **housekeeping 勘误（如实披露）**：r501 工件 JSON（fit_pareto_r501_result.json C4_refuted 文本与脚本 docstring）及 R501/R502/R505 日志/状态报告把相邻 cap 比较分母写作「1728」——真实比较数为 72 格 × 11 对相邻（12 cap 降序）= **792**（r506 独立重算核实，U3 项）。分子 323 精确复现；分母从未被正文引用、r501 检查门只看 C2/C3/C5，故属 housekeeping 级；v5.20 正文统一注记写「323 of 792」并括注 misprint 披露，r501 冻结工件原文保留未改（历史证据不回写）。
- **同题增量（统一机制注记，零新数据/零 GPU/零 TEST 读出）**：`earlystop_drift_r506/discrete_geometry_r506.py`（预注册 U1–U6 全过）把三个非单调现象归一到同一对象「离散停止状态集」：(i) τ*(cap) 非单调（粗网格 323/792 对违反）与其全半径穿越集恒为前缀（细网格 0 洞 0 悬垂，72/72 格）的**解析和解**——cap↓ 排除 cert 升序前缀，故两个标量化 max_K g_S 与 base_S 对 cap 均单调（细网格 2592 对 0 违反、粗网格 792 对 0 违反），τ*=1 一旦成立必在更小 cap 保持，非单调只能活在 τ<1 内部；(ii) deadline flip 序列（probed RLVE 原子 κ=8..3：0.0143/0.3571/0.2143/0.5/0.2429/0.5，≥2 次严格上跳）与 r498c deadline 修复更差（0.235→0.133）是同一离散性沿时间轴的读法；(iii) r504 闭式 edge（超几何 flip 质量=量子）与 r505 slack 符号紧性（11 tight/37 strict）是同一对象的精确读法。
- **主稿 v5.20（仅一处实质改动）**：App D(g) 段尾增「One geometry」段（统一注记 + 两操作端 cheapest/universal 读法），锚 discrete_geometry_r506_result.json。**18 页不变**（正文结束 p9、References p9 起、附录 p11 起——ICLR 正文 ≤9 页合规），pdflatex×3+bibtex 全 rc=0、零 ! 零 Overfull 零 undefined、pdfinfo 元数据全空。锚：tex sha256 c304aa7f…（md5 80fe97fb…）、pdf sha256 7d1e509e…（md5 899c239f…）。
- **audit_pack 同步**：results/ 新增 discrete_geometry_r506.py+json（与 workspace 原件同字节 cmp 核验）；paper.tex/paper.pdf 与 canonical v5.20 同字节；FRESH 双锚更新为 v5.20。
- **核验层 417→424（424 PASS/0 FAIL/0 EXT rc=0，claim_check_r506_inplace.log）**：CGU.* 6 项（checks/72格0洞0悬垂/双标量单调 2592+792 对/323-of-792+勘误披露/κ 序列锚+非单调/deadline 0.235→0.133）+ X.cgu.i tex 直解。clean-room 415/0/3EXT rc=0。
- **负对照 2 处全精确 FAIL（篡改先 sha256 验证落字节）**：(A) tex $323$ of $792$→$324$ → FAIL X.cgu.i+FRESH.tex.anchor；(B) tex $0$ holes→$1$ holes → FAIL X.cgu.i+FRESH.tex.anchor。还原后 sha256 回锚 c304aa7f…、重放 415/0/3EXT 全绿。
- **边界**：零新题/零 waiting；无新实验数据、无 TEST 读出（统一注记+写作+核验回合）；勘误与全部预注册证伪如实披露；冻结候选 r500_v5_16/r499_v5_15/r497/r496/r495/r493/r488/r484/r480/r475 全程未动未重绑。

## r507 更新（2026-08-15）— slack 符号分类器判别容差头对头审计 + 「≥2.4×→1.20×」正文勘误 + CED2.* / X.ced2.* 核验层（主稿 v5.21，424→430 项）

- **勘误（核心，如实披露）**：v5.19/v5.20 正文 App D(g) 写「two regimes are disjoint with a ≥2.4× margin gap」——r507 独立重算核实 inner margin ratio（min strict slack / |max tight slack|）实测 **1.195×**（最近 slack −2.431e-05 对 +2.906e-05）。「2.4」承自 r505 预注册文档的口头近似（tight 端 |−2.43e-05| 与 strict 端 +2.91e-05 的口头比误植为「≥2.4」），进入正文后未经 artifact 复核。v5.21 修为「inner margin ratio of 1.20×」并括注 correcting 披露；核验层 CED2.margin 把 1.195 从 discriminant_r507_result.json 派生断言（不硬编码正文数字）。与 r504「8/7」、r505「12→11」同族教训第三例：正文定性量级词必须由 artifact 派生核验，反向硬编码无法抓正文↔工件不一致。范围仅主线 v5.19/v5.20 的 tightness 句；冻结候选 r499_v5_15/r500_v5_16 不含该句、不受影响。
- **同题增量（判别审计，零新数据/零 GPU/零 TEST 读出）**：MGR 2d502ff5db3e 要求的「v5.19 11/48 勘误转同题几何闭环」。在 r505 同一冻结 FIT/证书表上，以判别容差 1e-6（工件分辨率以下）把三个提议判别器与扫描真值头对头（48 OMR 格全格）：(i) slack 符号分类器 11 TP/37 TN/0 FP/0 FN 完美，探针偏移 eps∈{5e-8,1e-7,5e-7,1e-6} 不变；(ii) 初版充要条件（唯一 binding 原子+jump_at_closed+valid_at_scan_minus，r505 已证伪）22 错=19 FP+3 FN；(iii) 固定阈值 margin 最优 11 错（=always-strict 常数 77.1%）。**容差敏感性**：tight 计数在 tol∈{5e-7,1e-6,2e-6} 平台 11、2.5e-7 退化 3（6dp 舍入把 8 精确等差散到 ±4e-7）、≥8.22e-05 吸入真实 strict 第 12 格——可用判别带 [5e-7, 8.22e-05)。±5e-7 舍入扰动下符号规则与扫描裁决 48 格全同；±6e-7（二分容差）各翻转 1 边界格（shard1 m=500 α=.02 / shard1 m=125 α=.10），strict 格从不被吸入。
- **主稿 v5.21（仅一处实质改动）**：App D(g) tightness 段「≥2.4×」→「1.20×（correcting）」+ 段尾增判别审计句，锚 discriminant_r507_result.json。**18 页不变**（正文 p9 止/References p9 起/附录 p11 起），pdflatex×3+bibtex+pdflatex×2 全 rc=0、零 ! 零 Overfull 零 undefined、元数据全空。锚：tex sha256 52c86559…、pdf sha256 9c8fc153…。
- **audit_pack 同步**：results/ 新增 discriminant_r507.py+json（与 workspace 原件同字节）；paper.tex/paper.pdf/REPRO_README 与 canonical v5.21 同字节；FRESH 双锚更新为 v5.21。
- **核验层 424→430（430 PASS/0 FAIL/0 EXT rc=0，claim_check_r507_inplace.log）**：CED2.* 5 项（checks/sign 完美/draft1 22 错+固定阈值 11 错/margin 1.195× 派生锚/容差带 [5e-7,8.22e-05)+D6 扰动）+ X.ced2.i tex 直解（discriminant_r507 锚+11TP/37TN+1.20×+容差带+旧「≥2.4× margin gap」句禁现）。clean-room 421/0/3EXT rc=0。
- **负对照 2 处全精确 FAIL（篡改先 sha256 验证落字节，/tmp 副本执行）**：(A) tex $11$~TP~/~$37$~TN→$12$~TP → FAIL X.ced2.i+FRESH.tex.anchor；(B) tex $1.20\times$→$1.30\times$ → FAIL X.ced2.i+FRESH.tex.anchor。还原后 audit_pack 重放 430/0 全绿。
- **边界**：零新题/零 waiting；无新实验数据、无 TEST 读出（判别审计+写作+核验回合）；勘误与全部预注册证伪如实披露；冻结候选 r500_v5_16/r499_v5_15/r497/r496/r495/r493/r488/r484/r480/r475 全程未动未重绑。

## r508 更新（2026-08-15）— 前缀律+双标量单调升格正式 Proposition（prop:prefix 短证）+ CPF.* / X.cpf.* 核验层（主稿 v5.22，430→435 项）

- **同题增量（形式化，零新数据/零 GPU/零 TEST 读出）**：MGR 5a29a028d9a3 要求把 r506 的「全半径穿越集恒为前缀 + 双标量单调」从机器核验经验命题升格为正式 Proposition 附短证。`earlystop_drift_r508/prefix_prop_r508.py`（V1–V7 全过，ALL_PASS=True）为该命题提供可重放机器见证：V1 base_S 粗网格单调 0 违反/792 对；V2 gmax_S 粗网格 0/792；V3 gmax_S 细网格 0/2592（r503 inline 断言的**落盘可重放见证**，见溯源披露）；V4 细网格前缀 0 洞 0 悬垂（72 格）；V5 穿越驱动普查——30 个前缀端案例在 r503 细网格前缀端 cap 处逐一用同一冻结 subset_profile 机器精确重算 base_S，全部 base_S≤α（30/30），证实 τ=1 的退出由 base_S 条件驱动（r503 细网格只存 tau/gmax_S/votes，base_S 重算路径与 r503 原跑同码同先验）；V6 粗网格 323/792 组合分解 253 严格子单位摆动 + 70 前缀边上跳 + 0 洞；V7 冻结链锚（r501 C4/r505 P3/r506/r507 ALL_PASS）。
- **主稿 v5.22（仅一处实质改动）**：App D(g)「One geometry」段后增正式 **Proposition（Prefix law and two-scalar monotonicity of the full-radius crossing set，\label{prop:prefix}）+ 短证**：(i) 双标量对 cap 单调（嵌套保持集+非负贡献）；(ii) τ(c)=1 ⟺ base_S≤α ∧ gmax_S≤α（prop:tv 双侧贪婪给出 V(R)=base_S+R(gmax−base_S)，R∈[0,1] 线性，sup 塌缩到两端点条件）；(iii) 穿越集=两个非减函数下水平集之交 ⟹ 前缀；c>α 处唯一可绑条件为 base_S≤α（crossing 由 base_S 驱动）。证明尾如实写入机器分解（253+70+0 洞）并锚 prefix_prop_r508_result.json。**19 页**（v5.21 为 18 页；命题+短证入附录使附录增 1 页——正文仍 p9 止、References p9 起、附录 p10 止/p11 起，ICLR 正文 ≤9 页合规不受影响；页数变化如实披露），pdflatex×3+bibtex 全 rc=0、零 ! 零 Overfull 零 undefined、pdfinfo 元数据全空。锚：tex sha256 0a09788c…（md5 207c7137…）、pdf sha256 540052f2…（md5 7a920d02…）。
- **自查披露（如实记录，不修数据）**：r508 首跑 2 项 FAIL——(a) V3 比较方向系**本脚本作者笔误**（升序曲线上断言「非降」却写成 b<a−1e-12 判违反的反号），1615 假违反，改正后 0/2592（与 r506 U2 同结论）；(b) V6 初版 claim「全部 323 违反严格位于 τ<1 内部」被数据证伪：70/323 为前缀边上跳（较大 cap τ<1 → 较小 cap τ=1，是前缀律的合法边界穿越、非洞），按惯例修为如实分解（253 内部+70 边跳+0 洞）并锚定。run_r508.log 保留首跑 FAIL 记录。
- **溯源披露（保留，入工件 provenance_disclosure 字段）**：r503 日志「gmax_S 单调不降 inline 2592/2592 验证」的 inline 断言从未存入 critical_cap_r503.py；r506 U2 在 r503 存储曲线上重跑过一次；本轮 V3 在同样冻结字节上第三次执行并存入本工件，使命题前提具有落盘可重放见证。
- **audit_pack 同步**：results/ 新增 prefix_prop_r508.py+json（与 workspace 原件同字节 cmp 核验）；paper.tex/paper.pdf 与 canonical v5.22 同字节；FRESH 双锚更新为 v5.22（描述标签 v5.21→v5.22 同步）。
- **核验层 430→435（435 PASS/0 FAIL/0 EXT rc=0，claim_check_r508_inplace.log）**：CPF.* 4 项（checks/双标量单调 792+792+2592 对/前缀 0 洞+V6 分解 253+70+0/V5 base 驱动普查）+ X.cpf.i tex 直解（prop:prefix 标签+prefix_prop_r508 锚+253/70 分解+downward 字样）。clean-room（/tmp/a11_r508_cleanroom 纯包副本）426/0/3EXT rc=0（3 EXT=W1 外部 provenance 设计如此）。
- **clean compile（/tmp/a11_r508_compile 异目录）**：pdflatex×3+bibtex 全 rc=0、19 页、零 ! 零 Overfull 零 undefined、元数据全空；重编译 PDF 与 canonical 逐字节一致（md5 7a920d02…）。
- **匿名性**：tex/bib grep 与 pdftotext 全文 qixuan|anthropic|newcpfs 零命中。
- **负对照 2 处全精确 FAIL（篡改先 sha256 验证落字节，/tmp/a11_r508_negctrl 副本执行）**：(A) tex $253$ strictly sub-unit wobbles→$254$ → FAIL X.cpf.i+FRESH.tex.anchor；(B) tex label{prop:prefix}→label{prop:prefixx} → FAIL X.cpf.i+FRESH.tex.anchor。还原后重放 426/0/3EXT 全绿。
- **边界**：零新题/零 waiting；无新实验数据、无 TEST 读出（形式化+写作+核验回合）；全部自查证伪如实披露；冻结候选 r505_v5_19/r500_v5_16/r499_v5_15/r497–r474 全程未动未重绑（md5 复核不变）。
