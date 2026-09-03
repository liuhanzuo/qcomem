# 投稿前核对表 — A2 主稿（2026-08-10, r855 维护更新；r850 AI-use 合规重锁后当前态）

主稿：`paper/A2_lrphase/paper.tex`（run 根，r792 迁移）→ `paper.pdf`（pdflatex+bibtex+pdflatex×2 全绿；workspace/paper/ 为同步镜像）

> **当前冻结身份（r854 现场 sha256sum 实测，MANIFEST 双端 0 非 OK，勿从历史日志复制）**：
> tex=`e9d9a84c…3b3b` / pdf=`bd715137…e0c` / bib=`93afcf65…9555` / bbl=`ff69217c…d171` / ai_use=`89101b36…9941`。
> ⚠️ r843–r850 日志反复记录的 `tex=e099465d / pdf=3b1a9a4c` 等组为**记录层历史错误**（第 15 例已登记，r851 勘误），文件层完整性由 MANIFEST 保障、双端 cmp 全 IDENTICAL，科学内容零变化。

| 项 | 状态 | 证据 |
|---|---|---|
| 编译 0 错误 | ✅ | r850 重编译 16 页 0 错（r854 复核 `grep "^!"` = 0） |
| 未定义引用 | ✅ | r789 python 核对 refs−labels = ∅ |
| 图/表被引用 | ✅ r789 补 | fig:main/tab:budget/tab:diag20 现均 \ref 引用（原仅文字引用，hostile 可挑刺） |
| 正文页数 | ✅ 9/9 页满（上限内，满足守则 ≥上限−0.5）；**计数口径细粒度 ≈8.88 页**（≤9 硬门，余量 0.12）∈ [8.5, 9.0] | r863 更正锚点（pdftotext 行位法三口径：Conclusion 终 8.70（p9 行68/97）/ 首个豁免节 Reproducibility 起 8.80（p9 行78/97）/ **References 起 8.88（p10 行86/99）——计数正文=到 References 起点**，编号 Limitations 起 p8、Conclusion 起 p9 均 span 内）。r859「≈8.74」系 Conclusion 终口径且未计 References 的 p10 溢出，已更正（A1 r743 官方豁免口径：编号 Limitations 不豁免，references 豁免的是条目本身；详见 PAGE_COUNT_SCOPE_AUDIT_R863.md） |
| 总页数 | 16（正文9 [实 8.88] + 声明~2.88 [三豁免 statement 合计：Reproducibility/Ethics/AI-use，p9 行78→p10 行86；r850 AI-use 任务级披露扩段 +1] + refs~1 + 附录~4 [p11 起]；r863 细则同步，pdfinfo 实测 16 页锚定不变） | pdfinfo |
| 模板 | ✅ iclr2027_conference.sty/bst（本目录） | paper.tex 头部 |
| 双盲匿名 | ✅ 无作者信息、无致谢、无机构名 | 全文 grep 自查 |
| 遗留 TODO/stub | ✅ 已清零 | grep TODO/stub |
| 措辞精确性 | ✅ r789 软化 | "bitwise identical"→"identical up to reported precision range<1e-4"（浮点不结合性可被挑刺），§6.1+附录B两处 |
| 定理-证明一致性 | ✅ T1/T2+C1/T3 附录 B proof skeleton 与正文一致；完整版 PROP*_FORMAL_R778/R780.md | 附录 B |
| 独立证明审计 | T2/T3 ✅（A3，F1/F2 修复经复核关闭）；T1 ⏳ 待 A1 复审（MGR 调度中） | GATEB_FIX_RESPONSE_R787.md |
| 实验数字可追溯 | ✅ r790 机器核对：consistency_r790.py 把 132 项 pilot/diag/verify 锚定数字逐一对 *_out.json，0 fail；修 2 处真实错误（tab:diag20 单格 −0.05→+0.08；§6.2 pilot17 部分跑 0.0718/1.012/0.877→全量 6种子 0.0696/1.014/0.894，结论不变） | consistency_r790.py |
| 预注册完整性 | ✅ 阈值先于运行固定，失败如实报（R2 0.240、P3 −0.026、EQRA-cos 双负） | 各 *_VERDICT.md |
| 引用真实性 | ✅ ref.bib 全一手，无虚构 DOI；近邻审计 **15 篇**表（r789 补 RR 谱系 Safran-Shamir 1908.00045 / Nagaraj 1903.01463 / Mishchenko 2006.05988，均 abs 一手核验不占对象）；r802 漂移终检补核第 16 篇 2604.13627（锚前 3.5 个月）；r805 PDF 全文四维核验闭环（lit_2604_13627/2604.13627v2.pdf，theorem/per-example/absorb-erase/data-selection 关键词全文零命中，不占对象） | NEAR_NEIGHBOR_AUDIT_R774.md + 附录A表 + code/arxiv_drift_r802_out.json + lr_phase_datavalue_r1/lit_2604_13627/ |
| 负结果诚实呈现 | ✅ §6.3 专节 + salvage cos/loss 对比 + P3 边界 + limitations 实测值 | §6.2/§6.3/§7 |
| hostile 自我预审 | ✅ r789 HOSTILE_SELFREVIEW_R789.md（18 项逐项对当前文件实际状态；[blocking]仅 AI声明待 MGR；3 non-blocking 已修；14 noted 已充分披露） | paper/HOSTILE_SELFREVIEW_R789.md |
| AI 使用声明 | ✅ r850 闭环（MGR 8f935df34e73）：`ai_use_statement.tex` 三段任务级披露（AI辅助6项/纯人工3项/共享3项，含 A3 r787/A6 r786 证明审计、A2+A6 引用互查），已 `\input` 进 PDF；science 零改动 | ai_use_statement.tex（sha=`89101b36…`） |
| Figure 1 | ⚠️ pgfplots 真实数据三联图（科学准确）；r848 备用预案就绪：共享 AutoFigure-Edit/edit_a2_fig1.py 已从退役 LOO 论文刷新至当前 EraseLateAbsorbEarly 三面板（py_compile 过，未调 API），MGR 裁决后一键生成 | fig:main + lr_phase_datavalue_r1/FIGURE1_AUTOFIGURE_FALLBACK_R848.md |
| 中文研究总结 | ✅ RESEARCH_SUMMARY_ZH.md | — |
| 可运行代码+冻结结果 | ✅ r796 补 | code/ = 39 脚本（全 py_compile 过）+ 34 冻结 JSON；根+code 双层 MANIFEST md5sum -c 全 OK |
| 复现 README | ✅ r796 补 | REPRO_README.md：编译链/checker(JDIR=code 实测154 pass)/逐脚本重跑表/外部数据出处(CIFAR-10N CC-BY)/非确定性说明 |
| 主稿迁移 run 根目录 paper/ | ✅ r792 已执行（MGR 5c5bd6630e1f 授权，迁至 `paper/A2_lrphase/`，编译全绿） | r854 勘误：此前本行停在授权前状态属记录层 stale，迁移早已完成 |
| hostile 正式模拟 | ⏳ MGR 组织（建议指派 1 名非作者 worker 按 ICLR form 独立打分；本自审不替代） | — |

## 本稿的诚实声明要点（供 MGR/hostile review 参考）

1. R2 在真实人类噪声臂预注册失败，主结果已转为"可逆性边界刻画"，未包装成通过。
2. EQRA-cos 负结果与 EQRA-loss 正结果同框架并列报告，贡献声明落在理论闭环而非检测启发式本身。
3. P3 失败已转化为定量精确性边界（q_frac/噪声率匹配）+ 可操作路由规则，未隐去。
4. 凸 cell 定量结论对真实像素只定性迁移（R3 report-only 预注册如此声明）。


## r793 增量：附录B证明自包含化
- T1/T2+C1/T3 附录证明从 3 段 skeleton 扩为完整证明（假设显式化、逐步推导、JSON 锚定数字、诚实边界段：pilot11 V1/V3 预注册失败、pilot17_budget_mid M_fit 失败、OU 协方差缩放失败、C1 结构性不紧）。
- 数字链 consistency checker 扩至 154 项（+22 附录新锚），run 根与 workspace 双跑全过；checker 修 CONSISTENCY_JDIR 环境锚定 bug。
- 附录证明不再依赖 workspace 中文 write-up（PROP1/2/3 仅作内部轨迹，审稿可见版本已进 paper.tex 附录B）。

## r797 增量：镜像 manifest 闭环
- workspace/paper/ 镜像的 MANIFEST 此前沿用 run根 的 paper.pdf md5（c710…），而镜像侧 PDF 是独立编译（0729…，同字节 367650）——md5sum -c 在镜像侧对 paper.pdf 报 FAILED。本轮按"清单匹配本目录派生物"原则，镜像 manifest 的 paper.pdf 行改为镜像自身 md5，两端 md5sum -c 现均 10/10 全 OK（code/ 子清单两端 73/73 不变）。
- 复核：checker 双 JDIR（workspace 原始目录 / run根 code/）均 154 pass / 0 fail；paper.log 0 错 0 undefined 14 页不变（本轮零编辑 paper.tex）。
- 教训(f)：凡含派生物（PDF）的 md5 清单跨目录镜像时，派生物行必须按各目录实际文件重算，不能照抄——源文件（.tex/.bib/.sty）才可跨目录共享 md5。

## r798 增量：MGR排版定点修复（指令 6e548a3fa396 done）
- Figure 1 通栏三个 minipage 内 tikzpicture scale=0.62→0.52（仅 3 行），overfull hbox 17.41/19.56/17.29pt 清零；errors/undefined 均为 0，总页数 14、正文 9 页不变。
- 内容完整性：T1/T2/T3 计数 22/16/25、105.6×7 次、R²=0.971 在，diff 仅 3 行 scale。
- **流程缺口（本轮 r799 修复）**：r798 改 tex/pdf 后未同步刷新两端 MANIFEST 与 workspace 镜像——run根清单 paper.tex/paper.pdf 两行 stale（md5sum -c 2 FAILED），镜像侧 tex/pdf 滞后 r798 前版本。

## r799 增量：r798 后续同步补漏（纯交付物，零研究主张变更）
- workspace 镜像补拷 r798 版 paper.tex/pdf/aux/bbl/blg/log/out + build_r798*.log，两端 md5 逐位一致（tex 10b6a010…，pdf 6a8e36fa…）。
- 两端根 MANIFEST_md5.txt 按本目录实际文件重算（含本 CHECKLIST 自身），md5sum -c 10/10 OK；code/ 子清单 73 项不变。
- 教训(g)：编辑主稿源文件后的收尾清单必须是固定四步——编译验证→镜像同步→MANIFEST 两端重算→md5sum -c 实测，缺任一步都会留下 stale（本轮即漏后三步）。

## r800 增量：编译确定性核验 + PDF 可复现（纯交付物，零研究主张变更）
- 触发：r796 自述纪律「PDF 派生物 md5 跨编译不稳」，但 r798 改 tex 后只做了一次性重编译验证，未实测核验该不稳是否真实、以及 MANIFEST 登记 PDF md5 的跨轮有效性。
- 实测诊断：零编辑 3-pass 全链重编译后 `md5sum paper.pdf` 6a8e36fa→fb0ae858（**变**），字节数不变 367606、编译全绿（0 错/0 undefined/0 overfull/14 页）——根因 pdflatex 嵌入 CreationDate/ID 时间戳。
- 修复：编译前置 `export SOURCE_DATE_EPOCH=1754762400`（r800 全链时间戳），两次独立全链重编译 `md5sum paper.pdf` 逐位一致（bd6b5bfa…）——**PDF 派生物恢复可复现**，可继续纳入 md5 清单核验；REPRO_README §1/§5 同步更新方法。
- 复核：pdftotext 提取正常，锚点数字 105.6/R²/0.971 共 7 次在文，pdfinfo 14 页 letter 不变；两端 MANIFEST 按新 PDF md5 + 两文档新 md5 重算。
- 教训(h)：派生物 md5 若要进清单，必须先实测其生成过程的确定性；不确定的派生物要么固定其全部输入（此处时间戳），要么从清单剔除并改核源文件。「它应该不稳」只是假设，实测两次重编译才能定性。

## r801（2026-08-09）— 编译确定性跨轮复验 + 镜像完整性机器核对

- **确定性复验**：零编辑 SOURCE_DATE_EPOCH=1754762400 全链重编译（pdflatex+bibtex+pdflatex×2，build_r801a/b/c.log），PDF md5=bd6b5bfa… 与 r800 登记值逐位一致——PDF 派生物跨轮可复现成立。
- **镜像完整性脚本化**：check_mirror_parity_r801.py 核对权威端全部常规文件（含未进 MANIFEST 的 build log，根目录+code/ 两级）。首跑发现 15 处镜像缺口（build_mig1-5、build_r793a-e、r801 build log 未拷、paper.log stale、paper.log.bak_r798 缺）——均非 MANIFEST 登记文件故 md5sum -c 全绿但镜像不完整。已补拷，复跑 115/115 OK exit=0。
- **教训(i)**：收尾四步的镜像同步只覆盖 MANIFEST 登记文件；目录级完整性须用专门 parity 脚本核对，md5sum -c 全绿不代表镜像无缺口。脚本两端各存一份，今后每轮编辑后跑一遍。
- 本回合零研究主张变更；tex/pdf 未动。

## r802（2026-08-09）— 投稿日终检：arXiv 漂移复跑 + 锚前近邻补核（零论文主张变更）

- **漂移终检脚本化**：code/arxiv_drift_r802.py（预注册：锚 2026-08-06、关线规则照 2026-08-07 增补
  「仅四维全同构才关线」、id 探针=近邻表 15 篇全部确定 ID）。7 查询 0 条锚后（>=2026-08-06）新进
  条目；Q1/Q4/Q5/Q6/Q7 五查询零命中稳定；探针 15/15 解析成功（API 链路正常，空结果可信非故障）。
  冻结输出 code/arxiv_drift_r802_out.json，同日重跑 totalResults 应逐位一致（A5 r375 同制）。
- **锚前近邻补核（诚实披露）**：Q3 命中的 arXiv:2604.13627v2 Rofin/Varre/Flammarion "(How)
  Learning Rates Regulate Catastrophic Overtraining"（v1 2026-04-15, COLM 2026）为此前 15 篇表
  外近邻（原检索漏检）。一手 abs 核验：四维均不占（无 per-example 相位价值分解/无吸收擦除
  不同构/机制在模型侧锐度通道非数据层/不涉数据选择利用），可诚实引为相关动力学背景。
  **限制**：仅核 abs；PDF 全文是否含数据层主张列入 MGR 移交前待办（同 MiniCPM #10 处理制）。
- 本回合未动 paper.tex/pdf；REPRO_README §3/§4 增漂移脚本一行；NEAR_NEIGHBOR_AUDIT 追加 r802 节。

## r803（2026-08-09）— A5 hostile review 4 项 must-fix 全部修复（MGR b4038f9ebc75）

- 来源：A5_HOSTILE_REVIEW_A2_R379.md（blocking=0，总评 5 borderline 修后 lean accept；
  15 组 headline 独立复算 12 组 PASS + lampath=0.202296 复现 + pilot14 完整重跑字节一致）。
- **修复 1（算术错误）**：B_crit≈3.8 → ≈4.9 三处（正文 honest-boundary 段、附录 parameter-free
  段、附录 seam 段）。真值 1/0.202296=4.94；3.8=1/0.262 系误用被论文自己否定的 self-fit λ̂。
  实验两臂 B=1.2/6.0 对 3.8 与 4.94 均跨骑，预注册 straddle 表述与序律结论不受影响（A5 报告
  第 26 行确认）。
- **修复 2（bitwise 失实）**：fig 注 (b)、§3.1 Evidence、§5 T1-invariance 段、附录 T1 数值验证段
  共 4 处——per-seed 层面改为「identical up to one test point (max 2.5×10⁻⁴=1/4000 测试集量化粒度；
  medians bitwise-identical)」；§5「range<10⁻⁴」删除替换。与 A5 逐种子复算（max 2.5e-4、
  中位数三者 0.34574999999999995 逐位一致）吻合。
- **修复 3（T3 鞅句过强）**：正文 T3 proof skeleton 与附录 Step 2 两处——「E[w_t] obeys the
  noiseless recursion」限定为「at epoch boundaries (and over the shuffle-noise terms within an epoch)」。
- **修复 4（approaching 0.087 措辞）**：改为「moving in the direction of the GD first-cause anchor
  0.087 as K→1 (the smallest scanned K≈4 still gives 0.84, an order of magnitude above the anchor)」
  ——方向/阶证据，非数值趋近。
- **验证**：SOURCE_DATE_EPOCH=1754762400 全链重编译（pdflatex+bibtex+pdflatex×2，
  build_r803a/b/c/bib.log 全 exit=0）——0 错/0 undefined/0 overfull/14 页（368254B）。
  pdftotext 复核：4.9×18 处、残留「3.8」×0（tex 层）、残留「approaching the GD」×0、
  「bitwise-identical」仅剩 medians 限定两处（合规）。checker 两端各 154 pass/0 fail。
- **收尾（教训 g/i 五步）**：编译验证→镜像同步（tex/pdf/aux/bbl/blg/log/out+4 个 build log）→
  parity 脚本核对→两端 MANIFEST 重算→md5sum -c 实测。
- 新 SHA：tex=f60def8adf29a3a4…（完整值见两端 MANIFEST），pdf=f24699c075dc9e21…。
- 零研究主张变更：定理量词、预注册 verdict、冻结结果、headline 数字全部未动。

## r805（2026-08-09）— 2604.13627 PDF 全文四维核验闭环（r802 自列移交前待办清零；零论文主张变更）

- 触发：r802 披露锚前近邻 2604.13627v2 时仅核 abs，自列「PDF 全文是否含数据层主张」为移交前待办。
- Artifact：workspace/lr_phase_datavalue_r1/lit_2604_13627/2604.13627v2.pdf
  （md5 8a1bb6baeeb43a34f00bf96675639772，2026-08-09 抓自 arxiv.org，CC-BY 4.0，COLM 2026）+ pdftotext 全文。
- 核验（grep 系统扫全文非抽样）：theorem/proposition/lemma/corollary 零命中（纯实证优化视角，
  机制在模型侧锐度通道）；per-example/influence/data value/exposure 零命中；absorb/erase/unlearn
  零命中；data selection/pruning/reweighting 零命中。r802 abs 级四维划界全部成立，无一上修。
- 近邻门维持不关线（15+1 篇均无四维同构覆盖）。NEAR_NEIGHBOR_AUDIT_R774.md 追加 r805 节；
  本清单 §引用真实性行同步刷新。paper.tex/pdf 科学 bytes 未动（零研究主张变更）。

## r806（2026-08-09）— r803 must-fix 闭环备忘交付 + 镜像 MANIFEST 不同文件集教训(l)

- 新交付物：`A2_R803_MUSTFIX_CLOSURE_MEMO.md`——把 A5 hostile review（A5_HOSTILE_REVIEW_A2_R379.md，
  blocking=0/总评5）4 项 must-fix 逐项对**冻结 PDF** 再核验：B_crit `4.9`×3、残留 `1/0.262`/`3.8`×0；
  `bitwise-identical` 仅 2 处均 medians-qualified；`epoch boundaries`×2；`approaching the GD`×0、
  `moving in the direction`×1。附 reviewer 一页对照表。供 A5/MGR 正式复核直接对照。
- 一致性：consistency_r790.py（JDIR=code/）154 pass/0 fail；paper.tex/pdf 科学 bytes 未动（零研究主张变更）。
- 教训(l)：双端 MANIFEST 文件集不同（镜像多 GATEB_FIX_RESPONSE/SALVAGE_LOSS_PREREG/自身 RESEARCH_LOG；
  权威端独有 paper.log.bak_r798），两清单必须各自从自身 `ls` 重新生成、不能互拷；parity 对 MANIFEST
  自身报 stale 是预期（文件集不同故内容本应不同）。空目录（已删的 figures/）会被 `ls|xargs md5sum` 当条目报错。
- 实测终态：权威端 md5+sha256 各 45/45 OK；镜像端各 76/76 OK；parity 121/123（唯二 stale=两个 MANIFEST
  自身，合法）；共享内容文件两端 cmp 逐字节一致。

## r811（2026-08-10）— dataprophet2026 引文元数据修复（MGR 授权 f102a624cfd1 执行闭环）

- 触发：r809 引文元数据逐条核验发现 ref.bib dataprophet2026 为 `author={Anonymous}, title={DataProphet}`，
  一手双源（本地 references/2603.19688.pdf 首页 pdftotext + arXiv API id_list）实测真实为
  Xuan Qi, Luxi He, Dan Roth, Xingyu Fu / "DataProphet: Demystifying Supervision Data Generalization
  in Multimodal LLMs" / ICLR 2026 conference paper。Anonymous 自引在双盲审稿中会被读成自我引用信号。
- MGR 授权 f102a624cfd1（2026-08-09 16:12）后执行五步：ref.bib 改 `@inproceedings`
  （author=Qi/He/Roth/Fu, title 完整, booktitle=ICLR, note=arXiv:2603.19688）→ bibtex →
  SOURCE_DATE_EPOCH=1754762400 全链重编译 → 镜像 → 双端 MANIFEST 各自从自身 ls 重生成。
- 实测终态：14 页/0 错/0 未定义引用/0 overfull；bbl 渲染 "Qi et al. (2026)"；PDF 确定性成立
  （重编译 md5 逐字符一致）；consistency 154 pass/0 fail（数字链未破）；paper.tex md5
  f60def8a… 未动（**零研究主张变更**，仅参考文献元数据）；镜像 cmp 逐字节一致；parity 127/129
  （唯二 stale=两个 MANIFEST 自身，结构性预期）；两端 MANIFEST md5+sha256 全 OK。
- 新身份（双口径）：paper.pdf md5=def48ab384afb20692e2a47ae44ffee6 /
  sha256=4430760ac47c867a9c6f5e5dd6d8a213b96db6315e020c7f3491b4de498e7f3c。

## r813 节 — 指令 8ec15bd300d5 双条 title 修复（wei2022 + ilyas2022）

- **授权**：MGR 指令 8ec15bd300d5（A 全修：①wei2022 标题对齐 DBLP+arXiv 双源；②ilyas2022 标题预印本变体→正式版本）。
- **修复前一手复核**（教训 n：不凭 flag 直接改）：DBLP 镜像完整 bibtex（conf/iclr/WeiZ0L0022、conf/icml/IlyasPELM22）+ arXiv abs 2110.12088 `<title>` + PMLR v162/ilyas22a.html `<title>` 双源本轮实测，目标值与 r812 落盘记录逐字一致。
- **改动**（仅 ref.bib 两条 title，零研究主张变更）：
  - wei2022cifar10n title：`Learning from noisy labels with deep neural networks: a real-world human noise dataset`（CIFAR-10N 网站别名）→ `Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations`（官方论文标题）。
  - ilyas2022datamodels title：`Datamodels: Predicting predictions from training data`（arXiv 预印本标题）→ `Datamodels: Understanding Predictions with Data and Data with Predictions`（DBLP/PMLR 官方 ICML 2022 标题）。
- **五步全实测**：bib 更新 → bibtex+pdflatex×3（SOURCE_DATE_EPOCH=1754762400，0 错/0 undefined/0 overfull/14 页）→ bbl 渲染复核（natbib 句式大小写为 bst 正常行为；正文引用 `(Wei et al., 2022)`/`(Ilyas et al., 2022)` 渲染不变）→ consistency_r790.py 154 pass/0 fail（数字链未破）→ PDF 确定性（重编译 md5 逐字符一致）→ 镜像同步 cmp 逐字节一致（含 paper.log/paper.blg）→ 双端 MANIFEST 各自从自身 ls 重生成，md5+sha256 `-c` 各 0 mismatch → parity 134/136（唯二 stale=两 MANIFEST 自身，结构性预期）。
- **paper.tex md5=f60def8adf29a3a4f268469686c7c3a9 未动**（零研究主张变更直接证据）。
- **新冻结身份**：paper.pdf md5=d109f0494f103b610f6371df9c112397 / sha256=8e29e7039fb4ffd9b5be0e27d8dc7b7f0a70785e7a155907be24eb7bbf97d312；ref.bib md5=72076819a84fb5b46c176905f9b5c984；paper.bbl md5=7449f62303fe2dc702cbf59c58a65326。
- **随包记录**：WEI2022_TITLE_FINDING_R812.md、ILYAS2022_TITLE_FINDING_R812.md（核验证据链）。
- **回执**：8ec15bd300d5 accepted→done（双口径）。

## r814 节 — novelty drift 复查（2026-08-10 时点）

- **动机**：距 r774/r805 系统近邻核验约 10 轮，投稿前 novelty 须在提交时点仍成立；对 2026-06~08 新窗口做 drift 复查（同 A6 r451 模式）。
- **方法**：arXiv API（export.arxiv.org，连通实测 HTTP 200）11 组查询按四维核心主张面（数据价值路径泛函/吸收擦除不同构/收缩预算标量律/端点归因失效+EQRA-loss）逐面对应，2026-06+ 命中逐篇取摘要判重叠。S2 仍 429 未用作主源。
- **工作流教训**：arXiv 复合短语查询须 `%22`+`%20` 显式 URL 编码，裸引号/加号静默返回 0（前 5 次 0 entries 实为编码 bug，对照 `all:"data valuation"` total=183 确认）。查询语法须实测连通性+对照已知 total，不能凭 0 命中当"无近邻"（同源教训 m）。
- **结果**：四维主张在 2026-06~08 窗口**无同构近邻**。最接近的 2605.25698（数据质量宏观课程调度+functional scaling law）为 aggregate quality 调度，与我 per-block 暴露归因互补不同层、属同期工作（2026-05-25<3月），诚实引用不阻塞。2605.18814/2606.06892 均改进端点/轨迹归因精度，非暴露时序机制，互补。
- **判定**：novelty 在 2026-08-10 时点维持成立，无 drift。零改动（纯只读核验+落盘记录）。
- **记录**：workspace lr_phase_datavalue_r1/NOVELTY_DRIFT_R814.md（含 11 组查询明细+近邻判定表）。

## r815–r842 压缩节 — Gate-D 锁稿前审计链（细节见各轮审计文档）

r815–r842 完成 28 维终审计并进入锁稿：CROSS_DOC_AUDIT_R834（跨文档一致性）、ENV_REPRO_AUDIT_R833（环境可复现）、HYPERPARAM_CONFIG_AUDIT_R838（超参配置）、table_fig_value_audit_r841（表图数值）、ANONYMIZATION_PACK_AUDIT_R842（匿名化包审计，发现 Z1=checker 未入包/Z2=去归属措辞两缺口）。各轮细节以对应审计文档为准，本节不重复；科学内容自 r813 后零变更（r814 起全部轮次均为只读核验或声明层/工具层修复）。

## r843–r853 节 — Gate-D terminal 锁稿与锁后维护链（当前态）

- **r843 Gate-D terminal 锁稿**：四件套+ai_use 冻结，9 flags+3 声明合并收尾，display-layer 修复闭合 T1 K-invariance/T2 方向性/10-points 声称。MANIFEST 双端各 64 项。
- **r844 匿名打包器**：anon_pack_r844.py 策展发射匿名补充包到 pkg_out/（泄漏扫描+死链检查+覆盖式发射）。
- **r845 Z1/Z2 预备闭合**：checker 参数化（CONSISTENCY_JDIR env）+ `--with-checker` 模式（checker ship 进 scripts/，dead_links 1→0）+ Z2 去归属措辞；clean-room 端到端 shipped checker 对 shipped JSON 154/0 + 重编译 PDF 与冻结包逐字节一致。同轮登记「selftest 10/10 PASS 未实跑」完整性缺陷（教训：审计工具 selftest 必须真实跑看 exit 码）。
- **r846 argv 防护**：未知 arg（含 `--help`）→ stderr usage + exit 2 不发射，防 `--help` 误触发覆盖式发射删除已 ship checker。
- **r847 打包窗口 runbook**：SUBMISSION_WINDOW_RUNBOOK_R847.md——§1 一条命令收口 Z1+Z2、§2 投稿日终检链（每命令当轮实测）、§3 验收表、§4 回滚。
- **r848 Figure 1 备用预案**：共享 AutoFigure-Edit/edit_a2_fig1.py 从退役 LOO 论文刷新至当前 EraseLateAbsorbEarly 三面板（对齐冻结 fig:main caption：absorption 1−e^{−cE} R²=0.971 / T1 形状 K 不变 / T3 B=K·Ewin λ=0.2023）；py_compile 过、未调 API，MGR 裁决后一键。
- **r849 终态就绪核验**：consistency_r790 对冻结包复跑 154/0；MANIFEST 链双端全绿；14 cite=14 bib=14 bibitem；pkg_out=default 回归对照态；登记 2 例证明层工具 exit 语义缺陷（R840 行号漂移 24/2、R836 硬编码 lines[818] 20/1——均工具层非科学层，第 14 例候选登记不修）。
- **r850 AI-use 合规修复（MGR 8f935df34e73）**：ai_use_statement.tex 三段任务级披露；重编译 16 页；四件套+ai_use SHA 重锁；MANIFEST 双端各更新 2 行 64/64 OK；consistency 154/0。science 零变更。
- **r851 短哈希勘误（第 15 例候选，登记不修）**：RESEARCH_LOG r843 起反复记录的五件套短哈希（tex=e099465d 组）与磁盘/MANIFEST 真实值（tex=e9d9a84c 组）不符——文件层零问题（MANIFEST 双端全绿+mtime 停滞+双端 cmp IDENTICAL），缺陷在日志记录层。教训：短哈希必须现场实测禁止跨回合复制；MANIFEST 是权威完整性源。
- **r852 路径勘误**：状态报告引用文件必须用 run 根绝对路径——相对路径在机械核对/同伴复核的错误 cwd 下解析即「不存在」，造成指标不可追溯假象。
- **r853 维护核验**：冻结态现场独立实测全绿（五件套 sha256 双端一致+MANIFEST 0 非 OK+mtime 停锁稿窗口）；打包器三模式回归复确认（default 对照 90 文件/selftest 13/13/argv 防护 exit 2 未发射）。

- **r854 投稿前核对表收口**：本核对表从 r843 时点同步至 r854 当前态（四件套短哈希更正为 MANIFEST 实测值、AI 声明行 ⏳→✅、总页数 15→16、r815–r853 增量链补齐、新增「当前冻结身份」警示块）。
- **r855 AI-use 声明「可签真」标准自查 + 主表迁移行勘误**：
  - **背景**：board 观察到 A6 线 r470→r471（MGR dab0d860a72b supersede 6aaf6780dc9d）把 AI 声明收窄到「可签真」范围——r470 含无法签真句（"authors have reviewed 审计输出" + 把 human review 列为已发生），被收窄为「AI/自动审计做了什么 + human review will occur + take full responsibility」。A2 声明（r850）早于该演进后标准，存在被同口径要求收窄的风险，故本轮预防性逐项自查。
  - **自查方法**：对 `ai_use_statement.tex` 全部 12 个事实主张（AI 辅助 6 + 纯人工 3 + 共享 3）逐条对照 RESEARCH_LOG 事件记录判定可签真性。结果：AI 辅助 6 项全部属实可签；共享 3 项（A3 r787/A6 r786 证明审计、A2+A6 近邻互查、MGR 协调）均有日志事件锚（r792 迁移授权 5c5bd6630e1f、r789 send A6 互查）；纯人工 3 项中任务规范/资源边界、最终责任可签。
  - **发现一处可签真性存疑**："Scope adjudication" 段把 ac2720d7313c 记为 human verdict，但 RESEARCH_LOG r785 两处均记为「MGR 指令 ac2720d7313c 已 ACK accepted」、无 human 字样；dc9fce09a81b 才明确是 human 裁决（r774「MGR dc9fce09a81b human 裁决」）。即 A2 声明把一条 MGR 指令归给了 human——与 A6 r470 被收窄的是**同类归属过强**问题。**本轮不自行改**（五件套冻结、science 零改动原则、声明措辞 MGR 定稿先例），登记为待裁项请 MGR 决定是否微缩措辞（最小改法：ac2720d7313c 改标为 manager-issued directive 或并入 dc9fce09a81b 一条）。
  - **主表勘误**：「主稿迁移 run 根目录 paper/」行由 ⏳（授权前状态残留）更正为 ✅ r792 已执行（MGR 5c5bd6630e1f）——迁移 r792 完成且编译全绿，本行自 r843 起 stale 12 轮未被发现（同 r851 短哈希、r852 路径的第三类记录层 staleness）。
  - 五件套零触碰：本轮仅编辑本 CHECKLIST（+两端 MANIFEST 重算），paper.tex/pdf/bib/bbl/ai_use 字节不变。
- **r859 A6 r487 页数失效规则自查 + 正文细粒度首次实测锚点**：
  - **规则适用性**：A6 r487 发现「锁后编辑点在正文 span 内 ⇒ 冻结时正文细粒度页数自动失效」。A2 线 r850 唯一锁后编辑 = ai_use_statement.tex 扩段，其挂载点 `\input{ai_use_statement}` 在 paper.tex:659，位于 §8 Conclusion(L629)/Reproducibility(L639)/Ethics(L650) **之后**、bibliography(L661) 之前——编辑点在正文 span **外**，本线页数声明**不失效**（trivial-pass）。distinguisher：A6 编辑点在 Limitations 前（span 内），A2 在 Conclusion 后（span 外）。
  - **正文细粒度首次实测**（此前主表只记「§8 Conclusion 终于 p9」页粒度，无细值）：pdftotext 行位法实测——p1–p8 无任何正文后 section 标记（REPRODUCIBILITY/ETHICS/AI USE/REFERENCES 均零命中），§8 Conclusion 终于 p9 非空行 71/96 ≈ 74% 处（其后为 Reproducibility→Ethics→AI Use Statement），**正文细粒度 ≈ 8.74 页**，落在守则窗口 [8.5, 9.0] 内。「正文终于 p9 内」声明精确成立、非高估。
  - 主表「正文页数」行已补该细粒度锚点。五件套零触碰（仅编辑本 CHECKLIST）。

- **r860 A5 r402 协议匿名性全树自查（Z1/Z2 打包预备）**：
  - 扫描：协议正则 `\bA[1-8]\b|qixuan1|agents/A[0-9]|/agents/[AW0-9]` 覆盖 code/ 40 脚本+全部文本文件（双端）。
  - 结果：复现路径 **0 路径/用户名硬泄漏**。唯一命中 = `check_mirror_parity_r801.py` L6-7 硬编码双端绝对路径——该脚本在锁稿清单范围外但**已钉入两端 MANIFEST**，与 A5「范围外+未钉住」先例结构不同，登记待 Z1/Z2 窗口（打包排除或 MGR 批准参数化+重钉），不自行编辑。
  - 次级 token 分类：paper.tex 的 A1-A3 arm 标签/(A1)-(A5) 假设编号=科学内容保留；L317 "A4 audit" 散文归属登记为窗口候选审查项；REPRO_README.md "A2 线" 标签未钉住、打包窗口就地改。全文留痕 `ANON_SCAN_R860.md`。
  - 五件套零触碰：本轮仅新增 ANON_SCAN_R860.md+编辑本 CHECKLIST（+两端 MANIFEST 重算）。

- **r861 同行范式适用性批量自查（第二十九维）+ 第16例记录层登记**：
  - **P1 = A4 r334 Hoeffding range 常数范式**：全文仅 2 处 Azuma–Hoeffding（L300/L923 同一定理步骤），均为**定性调用**（MDS 有界性由有限样本+有界梯度+(A3)稳定性设定保证；常数不进任何数字——定量走 diag8 实测锚 std 0.05–0.08 vs R=2.61，consistency L99-100 机械核验）。A8 式「半径差 2× 欠覆盖」失败模式结构上不可能（无数值半径）。PASS。
  - **P2 = A8 r287 比例/差分 range 混淆模式**：全文 15 组关键词扫描仅 4 命中，零比例/差分置信半径。唯一实例化尾部界 = L831 χ²（‖ε_⊥‖≤2√(d−1) whp，d=40 ⇒ 13σ 事件，余量 ~29 个数量级；方向安全：用于 Lemma C1 **下界**证明的上界步，文中自标 non-tight）。(A5) 的 whp 同族且失败方向被「noise only inflates」明文吸收。PASS/空命中。
  - **P3 = A6 r488 钉住 log 时敏计数**：paper.log Underfull=29/Overfull=0/undefined=0（r861 时点快照）；A2 线无任何文档引用 log 计数为「当前口径」，规则前提不成立。PASS。
  - 留痕 `INEQUALITY_SCOPE_AUDIT_R861.md`（双端）。
  - **第16例记录层登记**：consistency 环境变量实为 `CONSISTENCY_JDIR`（脚本 L12），r858–r860 日志简写 `JDIR=`（裸写无效、回落默认路径即 FileNotFoundError）。登记不修历史日志；本轮起记录写全名。
  - **第17例（工具操作事故，已闭环）**：本轮 MANIFEST 重算首次误用 `ls | xargs sha256sum`——`code/` 目录混入致 xargs 静默跳过多项，run 根 MANIFEST 短暂错为 65 项/镜像 103 项。因 run/ 在 gitignore（无版本回退）按文件名单复原：镜像比 run 根多 39 个历史文件（旧 build log/审计文档等），排除后双端重建为**各 65 项 = 旧 64 项名单 + INEQUALITY_SCOPE_AUDIT_R861.md**，sha256+md5 各 65/65 OK、0 非 OK，五件套 sha 不变（tex=e9d9a84c 等五值）。教训(az2)：MANIFEST 重算必须先 `grep -v` 排除目录再 xargs（或 `find -maxdepth 1 -type f`），且重建后必须对照旧名单 diff——「0 非 OK」只证明清单与当前磁盘一致，不证明名单本身正确。
  - 五件套零触碰：本轮仅编辑本 CHECKLIST+新增审计文档+双端 MANIFEST 按 65 项名单重建。

**待 MGR 外部调度（A2 自主范围无已知缺口）**：Figure 1 重绘裁决、A1 复审 T1、hostile 正式模拟、A5 复核、Z1/Z2 打包窗口、【r855 新增】AI-use 声明 ac2720d7313c 归属措辞裁决（A6 可签真口径）、【r860 新增，r865 更正】Z1/Z2 打包窗口匿名处置：check_mirror_parity_r801.py 路径参数化（须解冻重钉）或打包排除、REPRO_README「A2 线」标签处置（**已钉 MANIFEST，就地改会破坏每回合 md5sum -c 核验**——须 MGR 批准解冻重钉 或 打包排除，同 parity 脚本路径）、paper.tex L317 归属措辞是否脱敏。

- **r862 双端 MANIFEST 文件集对称化（教训 l/az2 未闭环实例修复）**：
  - **发现**：顶层 MANIFEST 只覆盖顶层常规文件，code/ 由独立 code/MANIFEST（75 项，双端 IDENTICAL/0非OK）跟踪——code/ 覆盖完整非缺口。真缺口=顶层不对称：run根顶层未跟踪仅 2 项（MANIFEST 自引用约定），镜像侧多出 41 滞留文件（9 个被本 CHECKLIST 引用的审计产物 R787/R833/R834/R838/SALVAGE/R860-md+env_repro/hyperparam/table_fig_value-py 应在权威端却只在镜像；29 个冗余历史滞留 RESEARCH_LOG r813 快照/bib.log/27 个 r785-r793w 旧 build log）。另发现 r860 自写 ANON_SCAN「双端」实际只放镜像——r851/r852/r855 同族「声明与状态不符」。
  - **处置**：9 审计产物补拷 run根权威端（cmp 一致，五件套 sha 实测不变）；canonical 名单锁定 run根 74 项，镜像先删 29 冗余（备份 frozen_r813_backup/mirror_stragglers_r862/）再按同名单重建双端 MANIFEST；新审计文档 MANIFEST_SYMMETRY_AUDIT_R862.md 双端钉入。
  - **首跑事故实证教训 az2**：直接对镜像 find 重建会把 29 滞留卷入（镜像 103 vs run根 74）——对称性必须用同一 canonical 名单重建双端并显式 diff 名称集。
  - **终验**：双端 MANIFEST 各 75 项、sha256/md5 各 0 非OK、名称集+逐文件哈希双端 IDENTICAL；五件套 sha 权威值不变（tex=e9d9a84c 等五值）；parity 153/153；consistency 154/0；五件套 mtime 停锁稿窗口。
  - 五件套零触碰：本轮仅 9 产物补拷+MANIFEST 重建+镜像冗余清理（备份）+新审计文档。

- **r863 A1 r743「编号 Limitations 豁免失效」规则自查（第三十维）+ 正文计数口径更正**：
  - **规则适用性**：A1 r743 发现 ICLR 2027 官方豁免清单不含编号 Limitations（其线 §10 起 p10 ⇒ 正文计入 9.22>9，desk-reject 级）。A2 稿面实测：编号 `\section{Limitations}` 起 **p8**（layout 层 p8 行106），`\section{Conclusion}` 起 p9，三个豁免 statement 全部 `\section*`（Reproducibility p9 行78 / Ethics p9 行87 / AI-use \input 续至 p10 行~85），References 起 p10 行86——编号 Limitations 完全在 p1–p9 正文 span 内，A1 风险模式结构上不可能成立，**trivial-pass**（distinguisher：A1 的 Limitations 在 p10 span 外，A2 的在 p8 span 内）。附录 p11 起 ~4 页按 A3 r376 官方口径（附录不限页）合规。
  - **正文计数口径更正**：r859 细粒度锚点「≈8.74」止步于「Conclusion 终于 p9」粒度，未计 References 的 p10 溢出。按「计数正文=到 References 起点」口径重算三值：Conclusion 终 8.70 / Reproducibility 起 8.80 / **References 起 8.88（计数口径）**——≤9 合规，余量 0.12 页即锁后正文 span 内编辑的安全垫。主表「正文页数」行已更新；r859 的 8.74 系 Conclusion 终口径+行计数微差（71/96 vs 本轮 68/97，±0.03 行位法噪声），合规结论不变。
  - **页数细则双 stale 修复（教训 bc）**：主表「总页数」行与 RESEARCH_SUMMARY_ZH.md L3 自 r850 +1 页后 12 轮残留旧拆分「声明~2」——实际三豁免 statement 合计 ~2.88（p9 行78→p10 行86）；中文总结 L56 漏列 Ethics Statement。两处已同步为「正文9[实8.88] + 声明~2.88 + refs~1 + 附录~4 = 16 页（pdfinfo 锚定不变）」并补 Ethics。
  - 留痕 `PAGE_COUNT_SCOPE_AUDIT_R863.md`（双端钉入 MANIFEST，75→76 项）。
  - 五件套零触碰：本轮仅编辑本 CHECKLIST 两行+本节、RESEARCH_SUMMARY_ZH.md L3/L56、新增审计文档；编辑后五件套 sha 现场复测不变（tex=e9d9a84c 等五值），MANIFEST 双端按教训(az2)规程重建（find -maxdepth 1 -type f 排除目录与自引用→名称集 diff 仅新增本文档→sha256/md5 各 0 非 OK）。

- **r865 第18例记录层登记：r860 REPRO_README 钉住状态误判更正**：
  - **发现**：r860 节与 ANON_SCAN_R860.md L37/L46 登记「REPRO_README.md 未被 MANIFEST 钉住、打包窗口就地改」——r865 现场实测该文件**双端均在 MANIFEST**（sha256=682af719 前缀，双端同行同值）。若按错误登记执行「就地改」，md5sum -c/sha256sum -c 每回合核验立即 FAIL——即 r860 自建的「范围外/内×钉住/未钉住」四象限被误用于「已钉住」对象，正确路径应与 parity 脚本同型（打包排除 或 MGR 批准解冻重钉）。
  - **性质**：记录层事实性错误，非科学层；文件层零问题（MANIFEST 双端全绿）。同族于 r851/r852/r855/r862「声明与状态不符」——本例特殊处：错误登记**指向下一个窗口的危险动作**（不只是历史描述失真），属登记错误中后果较重亚型。
  - **处置**：待办行前提已更正（见上，「就地改」→「已钉住，须解冻重钉或打包排除」）；ANON_SCAN_R860.md 加 r865 勘误附注（不动原文留痕）；REPRO_README.md 本体不触碰。登记不修历史。
  - **教训(bf)**：四象限处置框架的**前提值（钉住与否）必须现场 grep MANIFEST 实测**，不能凭扫描时记忆/假设填写——r860 对 parity 脚本实测了钉住状态（正确），对 README 凭假设填「未钉住」（错误），同文档内两对象核验深度不一致。收尾清单增：凡登记处置路径含「未钉住可就地改」字样，必须附 grep MANIFEST 命中计数证据（0 次才可写未钉住）。
  - 五件套零触碰：本轮仅编辑本 CHECKLIST 待办行+本节、ANON_SCAN_R860.md 加附注；编辑后五件套 sha 现场复测不变（tex=e9d9a84c/pdf=bd715137/bib=93afcf65/bbl=ff69217c/ai_use=89101b36），MANIFEST 双端按教训(az2)规程重建。
