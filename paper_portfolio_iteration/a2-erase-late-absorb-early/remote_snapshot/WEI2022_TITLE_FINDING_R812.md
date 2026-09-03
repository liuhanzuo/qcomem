# r812 — wei2022cifar10n 修复执行前一手复核：发现比 r811 更大的缺陷（标题整体错误；r811 "TNNLS 2023" 系假缺陷）

**触发**：MGR 指令 89e8750a2b90 授权「wei2022 venue 修复：booktitle=ICLR/year=2022 → 修复为正确 venue（r811 记录为 IEEE TNNLS 2023），五步同 dataprophet 流程，零研究变更」。
**执行纪律**：修复类操作前必须一手复核（教训(m)：凭记忆/凭单源高危；教训：修复前先确认真实值，不能只凭既有 flag 记录直接改）。本轮复核**推翻了 r811 的判定**，并发现更大缺陷。

## 一手三源核验（本轮实测，全部非抽样）

| 源 | 方法 | 结果 |
|---|---|---|
| DBLP 镜像 (dblp.uni-trier.de) | `rec/conf/iclr/WeiZ0L0022.bib` 完整 bibtex | title=Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations；booktitle=ICLR 2022；publisher=OpenReview.net；authors=Wei/Zhu/Cheng/Liu/Niu/Liu 6位；url=openreview.net/forum?id=TBWA6PLJZQm |
| arXiv abs 页 2110.12088 | curl HTML `<title>` + Comments | title 同上逐字符；Comments: "ICLR 2022" |
| arXiv PDF 2110.12088 首页 | pdftotext 第1页 | 页眉 "Published as a conference paper at ICLR 2022"；标题排版 "LEARNING WITH NOISY LABELS REVISITED: A STUDY USING REAL-WORLD HUMAN ANNOTATIONS"；作者 6 位逐字符匹配 |

三源**完全一致**，交叉收敛。

## 判定

### 缺陷A（真实，且比 r811 记录更严重）：bib 标题整体错误
- 现行 ref.bib `wei2022cifar10n` title = `Learning from noisy labels with deep neural networks: a real-world human noise dataset`
- 真实标题 = `Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations`
- 现行标题是 CIFAR-10N 数据集网站 / DatasetDict 生态中常用的**别名/描述性称呼**，不是论文标题。
- 审稿人核对引用会发现标题对不上原文 → 引用元数据错误（比 venue 错误更显眼，因为标题是引用的第一标识）。

### 缺陷B（r811 记录，本轮证伪为假缺陷）：venue = "IEEE TNNLS 2023" 系误配
- r811 记录「一手双源 Crossref container-title + openreview venue 一致为 IEEE TNNLS 2023」。
- 本轮 Crossref `query.bibliographic` 重测：返回的 TNNLS 条目均为**同主题他文**（Song et al. "Learning From Noisy Labels: A Survey" TNNLS 2023；Liu et al. "A Convergence Path..." TNNLS 2024 等），**没有一条是 Wei et al. CIFAR-10N 论文**。
- DBLP 完整 bibtex + arXiv 双源一致指向 **ICLR 2022**，与 bib 现行 venue **本就正确**。
- 结论：r811 的 venue 判定是把 Crossref 对同关键词返回的他文误配到本条。bib venue=ICLR/2022 **无需修复**。MGR 89e8750a2b90 按 r811 记录的「修复为正确 venue」目标实为假缺陷。

## 授权状态与请示

- 89e8750a2b90 授权的是「venue 修复」，但该 venue 缺陷本轮证实不存在（假缺陷）；真实缺陷是**标题**（r811 未覆盖、超出该指令字面范围）。
- 修复标题会改变 ref.bib / paper.bbl / paper.pdf bytes（tex 正文不动）。
- **关键更正（诚实性）**：本回合初向 MGR 的请示消息中曾推断「bbl 年份字母 2022c→2022d」——经 bbl 实测，两 2022 条目（wei2022 / ilyas2022）首作者不同（Wei vs Ilyas），natbib 不分配消歧字母，均渲染为裸 `(2022)`。故修标题**不改变**正文 `(2022)` 渲染。该推断系凭记忆错误，已在本记录更正，正文引用 `(Wei et al., 2022)` 不受修复影响。
- 已 board note 向 MGR 请示范围：(A)标题+venue 双修复 / (B)只修标题 / (C)冻结不动。待 MGR 复范围后立即按五步执行（bib→bbl→重编译→镜像→MANIFEST，md5+sha256 双口径）。

## 附：数据源可达性（如实）
- OpenReview API 现 403 人机验证（ChallengeRequiredError），v1/v2 均不可达 → 本轮弃用，改用 DBLP+arXiv。
- S2 仍 429 限流 → ilyas2022 ICML 归属补核继续挂起。
- Crossref 对该条不能精确返回（只返回同主题他文）→ 该源对本条不可用作归属证据。

## 本回合未改任何 bytes
权威端与镜像端本轮纯只读核验 + 落盘本记录；ref.bib/paper.bbl/paper.pdf 未动，待 MGR 范围授权。
