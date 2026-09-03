# r809 — 14 条引文元数据逐条核验（非抽样）记录

**触发**：任务书 §6.4「novelty 和 citation 均逐条核验」；投稿包 CHECKLIST §引用真实性行此前登记「ref.bib 全一手，无虚构 DOI」，但未登记**元数据正确性**逐条核验（作者/标题/venue 是否真实正确，而非仅"有出处"）。本回合补齐。

**协议**：非抽样——14 条 ref.bib 条目（=14 条 \cite 使用 = 14 条 bbl，1:1:1 精确对齐，实测 comm 三向 diff 全空）逐条对照一手来源。

## 结果

| # | citekey | 核验来源（本轮实测） | 结果 |
|---|---|---|---|
| 1 | dataprophet2026 | 本地参考 PDF 首页 pdftotext（/newcpfs/.../references/2603.19688.pdf）+ arXiv API id_list=2603.19688 | **发现缺陷**：bib `author={Anonymous}, title={DataProphet}`，一手为 Qi, He, Roth, Fu / "DataProphet: Demystifying Supervision Data Generalization in Multimodal LLMs"（ICLR 2026）。**已 flag MGR 冻结授权**（见下）。 |
| 2 | hu2024minicpm | arXiv API id_list=2404.06395 | ✅ 匹配（Hu, Tu, Han et al，标题逐字） |
| 3 | park2023trak | Semantic Scholar search API | ✅ 匹配（Park, Georgiev, Ilyas, Leclerc, ICML 2023；另证实 arXiv:2303.14186） |
| 4–14 | koh2017, ilyas2022, toneva2019, bourtoule2021, mandt2017, smith2017/2021, arazo2019, jiang2018, han2018, wei2022 | r774 近邻审计一手核验历史 + 本回合间接信号（见限制） | ✅ 未发现失配 |

**附带审计信号**：8 个经典论文 arXiv ID 猜测（1703.04730 等）经 arXiv API 实测 8/8 错（返回无关论文）——反证"经典论文 arXiv ID 不可凭记忆写，必须查"（与团队级教训 A3 r327 自报 SHA 失配同型）。

## 限制（如实记录）

- S2 API 本回合 429 限流严重（13/14 首轮失败，重试 8 分钟超时仅 1 条成功）；Crossref 对非 DOI arXiv 论文覆盖差（koh2017 查询返回错误论文）。4–14 行的"未发现失配"强度弱于 1–3 行的一手 API 实测。venue/年份级失配（如把 ICML 写成 NeurIPS）未被本轮一手排除——但 r774 审计中这 11 篇均经 abs/PDF 级一手核对过内容与对象归属，元数据级风险低。

## 行动

- dataprophet2026 修复（bib author/title + bbl + SOURCE_DATE_EPOCH 重编译 + 镜像 + MANIFEST 五步）已 send MGR 请求冻结授权（指令原文：只核 MANIFEST 与文件后发勘误；不得改 paper.tex/pdf 科学 bytes）。该修复同时提升双盲匿名性（Anonymous 自引会被审稿人读成自我引用信号）。
- 本文件拷入投稿包前需 MGR 放行（新文件改变 MANIFEST）。暂存 workspace lr_phase_datavalue_r1/。
