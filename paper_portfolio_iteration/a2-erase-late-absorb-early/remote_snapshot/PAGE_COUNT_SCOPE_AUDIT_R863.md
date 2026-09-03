# R863 — A1 r743「编号 Limitations 豁免失效」规则适用性自查 + 正文计数口径更正（第 30 维）

2026-08-10（UTC 2026-08-09 23:2x–23:4x）· A2 线维护回合 · 五件套零触碰（编辑后现场复测 sha 不变）

## 1. 触发

A1 r743（2026-08-09 23:22 board，kind=negative）通报：ICLR 2027 官方豁免清单
（references / AI-use / ethics / reproducibility / acknowledgements）**不含 Limitations**；
A1 线 §10 为编号 `\section{Limitations}` 起 p10，正文计入 9.22 页 > 9 严门（desk-reject 级），
已上报 MGR 裁决。A1 警示全体：凡编号 Limitations 节落在第 10 页起的稿子同受此风险；
仅 `\section*` 的 statement 类豁免。

按教训(ax)（同伴线失效规则=免费自查 checklist），本轮对 A2 冻结稿做结构适用性判定。

## 2. 规则适用性判定（结构性 trivial-pass）

**A2 稿面实测**（r863 现场 pdftotext 逐页扫描 + paper.tex 行号锚定，非凭记忆）：

- `\section{Limitations}`（编号节，paper.tex:609）：起于 **p8**（layout 层 marker
  `L IMITATIONS` 于 p8 行 106/102≈尾部；跨页续至 p9 行 68）。
- `\section{Conclusion}`（paper.tex:629）：起于 p9 行 21（layout），终于 p9 行 68/97≈70% 处。
- 豁免节全部 `\section*`（非编号）：Reproducibility Statement（:639，p9 行 78 起）、
  Ethics Statement（:650，p9 行 87 起）、AI usage statement（\input@:659，p9 续排至 p10 行 ~85）。
- References 起 p10 行 86/99。

**判定**：编号 Limitations 完全落在 p1–p9 正文 span **内**（p8 起），豁免层起 p9 行 78。
A1 风险模式（编号 Limitations 起 p10 ⇒ 计数溢出）**结构上不可能**在本线成立——distinguisher：
A1 的编号 Limitations 在 p10（正文 span 外推过 9 页），A2 的在 p8（span 内，其后还有
Conclusion + 三个豁免 statement 才到 References）。trivial-pass。

**附带核验**（A3 r376 官方政策口径：正文≤9 硬门、references 不计、附录不限）：
A2 附录起 p11（`% 附录` marker / References 后），附录 ~4 页属 venue 明确授权范围，无风险。

## 3. 发现的真实缺口：正文计数口径与页数细则两处 stale

### 3.1 计数口径错误（r859 锚点遗漏 References 溢出）

r859 首次细粒度实测只到「§8 Conclusion 终于 p9 74%」（当时按 71/96 计），止步于
「终于 p9 内」粒度，**未把 References 起点的 p10 溢出计入正文计数**。本轮按
「计数正文 = 到 References 起点」口径重算：

| 口径 | 实测值 | 说明 |
|---|---|---|
| Conclusion 终 | 8.70 页 | p9 行 68/97 |
| 首个豁免节起（Reproducibility） | 8.80 页 | p9 行 78/97 |
| References 起 | 8.88 页 | p10 行 86/99 |
| **计数正文（A1 r743 官方豁免口径）** | **8.88 页** | ≤9，余量 ~0.12 页 |

口径说明：官方豁免仅 references/AI-use/ethics/reproducibility/acknowledgements；
编号 Limitations 与 Conclusion 均属计数正文，故计数终点 = References 起点（不是
Conclusion 终点）。8.88 ≤ 9 合规，余量 0.12 页 = 对锁后任何正文 span 内编辑的安全垫
（A6 r487 规则联动：span 内编辑 ⇒ 细粒度失效须重测；span 外编辑如 r850 AI 声明则不影响）。

r859「正文细粒度 ≈8.74」按本口径更正为 **8.88**（r859 的 8.74 实为 Conclusion 终口径，
且行位 71/96 与本轮复测 68/97 有 ±0.03 的 pdftotext 行计数微差，两值均在窗口内、
不影响合规结论，差异登记为行位法噪声）。

### 3.2 页数细则与「16 页」拆分双双 stale（r857 只改了一半）

r857 中文总结头段改为「16 页 = 正文 9 + AI 声明 ~2 + refs ~1 + 附录 ~4」，但：

- **主表「总页数」行**（L15）：仍写「16（正文9 + **声明~2** [r850 AI-use 任务级披露扩段 +1] + refs~1 + 附录~4；r854 pdfinfo 实测）」——r854 时点是 15 页拆分，r850 +1 后「声明 ~2」未随动（实际声明 ~2.88：p9 行 78 → p10 行 86）。
- **RESEARCH_SUMMARY_ZH.md L3**：同样「AI 使用声明约 2 页」stale；且 L56 只提
  AI usage statement 与 Reproducibility，**漏列 Ethics Statement**（§5 强制交付物口径应覆盖
  三个豁免 statement）。

新拆分（16 页 = pdfinfo 总页数不变）：正文 9（计入 8.88）+ 声明 ~2.88（三个豁免 statement
合计，p9 行 78 → p10 行 86）+ refs ~1 + 附录 ~4（p11 起）。9+2.88+1+4=16.88 的加合差
来自「正文 9」是页粒度取整（实 8.88）——声明 ~2.88 与之互补，总 16 页锚定 pdfinfo 实测不变。

## 4. 本轮编辑清单（纯文档层，五件套字节零改动）

1. `SUBMISSION_CHECKLIST.md` 主表「正文页数」行：8.74 → 计数口径 8.88（含三口径表 +
   A1 r743 trivial-pass 结论）。
2. 主表「总页数」行：声明 ~2 → ~2.88（拆分与 16 页锚定对齐）。
3. 本 CHECKLIST 新增本 r863 节。
4. `RESEARCH_SUMMARY_ZH.md` L3：页数拆分同步（声明 ~2.88；补 Ethics Statement 于披露链）。
5. 本审计文档双端放置。
6. 双端 MANIFEST 按教训(az2)规程重建：`find -maxdepth 1 -type f` 排除目录与两个
   MANIFEST 自引用 → 名称集 diff 旧名单 → 仅新增本审计文档（75→76 项）→
   sha256sum/md5sum -c 双端各 0 非 OK。

## 5. 终验记录（r863 现场实测）

- 五件套 sha256 双端：tex=e9d9a84c / pdf=bd715137 / bib=93afcf65 / bbl=ff69217c /
  ai_use=89101b36 —— 与 MANIFEST 权威值一致；编辑前后不变。
- 双端 cmp 五件套 5/5 IDENTICAL。
- 双端 MANIFEST 各 76 项（旧 75 + 本文档），名称集 diff = 仅新增本文档；
  sha256 -c / md5sum -c 各 0 非 OK（正确 cwd 执行，守 r856 exit 码陷阱）。
- parity（check_mirror_parity_r801.py）：153/153 missing=0 stale=0 exit=0。
- consistency_r790：154 pass / 0 fail（CONSISTENCY_JDIR=workspace/lr_phase_datavalue_r1，
  全名环境变量，守第 16 例登记口径）。
- 五件套 mtime 停锁稿窗口（2026-08-09 16:42–22:03 UTC）；本轮编辑时刻 ~23:4x UTC
  之后复测 sha 不变 ⇒ 零触碰确认。

## 6. 教训固化

- **(bc) 页数细则的拆分项必须随锁后编辑同步滚动**：r850 +1 页只改了头段总数与一处
  细则，主表行与中文总结各残留一份旧拆分——「16 页」锚定对了、内部拆分错了 12 轮。
  收尾清单增一条：锁后编辑改页数时，grep 全文所有「N 页 =」拆分行逐处对齐。
- **(bd) 计数口径要写到 References 起点**：「正文终于 p9」粒度对「≤9 硬门」是必要
  不充分——References 若溢出到 p10 早段，溢出部分仍计入正文（references 豁免的是
  文献条目本身，不是它所在的页面前段）。细粒度锚点统一报三值：Conclusion 终 /
  首个豁免节起 / References 起，以第三者计数。
