# r812 补 — ilyas2022datamodels 补核：venue 正确（r811 存疑消解），但发现 title 变体缺陷（同 wei2022 同型）

**触发**：r811 遗留「ilyas2022 S2 恢复后补核 ICML 归属」。S2 仍 429，改用 DBLP 镜像 + PMLR 官网 + arXiv PDF 三源（非限流源）。

## 一手三源核验（本轮实测）

| 源 | 方法 | 结果 |
|---|---|---|
| DBLP 镜像 | search `Datamodels Ilyas` | `conf/icml/IlyasPELM22`：title="Datamodels: Understanding Predictions with Data and Data with Predictions"，venue=**ICML 2022**，type=Conference |
| PMLR 官网 | `proceedings.mlr.press/v162/ilyas22a.html` | `<title>`="Datamodels: Understanding Predictions with Data and Data with Predictions"，Proceedings of Machine Learning Research（=ICML 2022 论文集 v162） |
| arXiv PDF 2202.00622 | pdftotext 首页 | 标题="Datamodels: Predicting Predictions from Training Data"（arXiv v1 预印本标题，**无**会议 banner/Comments/journal-ref） |

## 判定

### venue：bib `booktitle={ICML}, year={2022}` —— **正确** ✅
- DBLP（conf/icml/IlyasPELM22）+ PMLR v162 双源一致指向 ICML 2022。
- r811「未证实 ICML」存疑**消解**，非缺陷。bib venue 无需动。

### title：bib 用 arXiv 预印本标题，DBLP/PMLR 官方用另一标题 —— **缺陷（变体混用，同 wei2022 同型）** ⚠️
- bib title = `Datamodels: Predicting predictions from training data`（= arXiv 预印本标题，PDF 首页实测确认）。
- DBLP/PMLR 官方 ICML 2022 记录 title = `Datamodels: Understanding Predictions with Data and Data with Predictions`。
- 两者是同一工作的**预印本 vs 会议正式版**标题变体（datamodels 论文的预印本与 ICML 正式版标题不同，是该领域已知现象——Datamodels 的 ICML 官方标题是 "Understanding Predictions with Data and Data with Predictions"）。
- 一致性缺陷：bib 声称 ICML 2022（正式 venue）却用 arXiv 预印本标题 → venue 与 title 版本不匹配。同 wei2022 的「网站别名 vs 论文标题」同型（都是引用标题与声称 venue 的官方记录标题不一致）。
- 与 wei2022 的区别：wei2022 的 bib 标题是**非论文别名**（网站描述），连 arXiv 预印本都不是；ilyas2022 的 bib 标题是**真实 arXiv 预印本标题**，只是与 ICML 正式版标题不同。严重度：wei2022 > ilyas2022。

## 处置
- venue：无需动（已证正确）。
- title：若 MGR 授权统一修引用标题，可一并将 ilyas2022 title 对齐 DBLP/PMLR 官方 ICML 2022 标题（`Understanding Predictions with Data and Data with Predictions`）。这同样改 ref.bib/bbl/pdf bytes，tex 正文不动。
- 已 board note flag MGR，与 wei2022 合并为「引用标题与官方记录对齐」一类授权请求，避免逐条反复打扰。

## 一致性审计结论（14 条引文维度收敛）
- 元数据正确性（r809）：dataprophet2026 已修复闭环（r811）。
- venue 级（r811+r812）：wei2022 venue 本正确（r811 假缺陷已证伪）；ilyas2022 venue 本正确（存疑消解）；其余 10 条 r811 实测正确。
- **title 级（r812 新发现，2 条同型）**：wei2022（别名）、ilyas2022（预印本变体）——bib title 与声称 venue 的官方记录标题不一致。这是 r809/r811 未覆盖的维度（此前只核作者/venue，未核 title 与官方记录逐字对齐）。
