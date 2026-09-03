# A2 线匿名性自查 — r860（对接 A5 r402 协议，Z1/Z2 打包预备）

日期：2026-08-09 23:0x UTC。触发：A5 r858 消息——A5 线 82 脚本去标识完成、可对接 Z1/Z2
--with-checker；A2 线此前未做过同款全树扫描，本轮按 A5 协议正则自查。

## 扫描范围与协议

- 目录：`run/paper/A2_lrphase/`（权威端）+ `agents/A2/workspace/paper/`（镜像端）。
- 协议正则（A5 r402 同款扩展）：`\bA[1-8]\b`、`qixuan1`、`agents/A[0-9]`、
  另加 `/agents/[AW0-9]` 绝对路径模式。
- 文件类型：code/*.py（40 脚本）、全部 *.md/*.tex/*.sh/*.txt/*.json（MANIFEST 清单文件
  自身除外——其内容为哈希值，无文本）。

## 结果

### 1. 身份泄漏（qixuan1 / agents/ 绝对路径）

- `code/` 40 脚本：**0 命中**（两端）。
- 全部文本文件全树扫描：唯一命中 = `check_mirror_parity_r801.py` L6–L7 硬编码
  AUTH/MIR 双端绝对路径。
  - 定性：该脚本在 **GATE_D_LOCK_RECEIPT 范围外**（从未进任何锁稿清单；A2 线无
    55 式范围外清单文档），但**已钉入两端 MANIFEST**（sha256+md5 各 1 条目）——
    与 A5 先例的「范围外+未钉住」处置结构不同。
  - 处置：**登记待 Z1/Z2 打包窗口**。打包时该脚本属审计工具类，不入投稿 artifact
    即无泄漏；若决议修复（路径参数化），需走解冻重钉 MANIFEST 流程，须 MGR 批准。
    不自行编辑（钉住态原地编辑会破坏每回合 sha256sum/md5sum -c 漂移核验——
    A6 r486 增量层教训同型约束）。
- workspace 镜像端 code/：0 命中。

### 2. `\bA[1-8]\b` token 命中（按语义分类，非泄漏）

| 位置 | 实例 | 语义 | 判定 |
|---|---|---|---|
| paper.tex L393–395 | A1/A2/A3 | 实验 arm 标签（pilot15 四臂） | 保留（科学内容） |
| paper.tex L811–818 | (A1)–(A5) | 假设编号（假定列表） | 保留（科学内容） |
| paper.tex L317 | "Honest boundary (A4 audit, M5)" | 审计过程归属，正文散文 | 引用性提及，anon 风险低；彻底清零须解冻正文，待 MGR 裁决（登记为 Z1/Z2 窗口候选审查项） |
| REPRO_README.md L1/L3 | "A2 线" | 内部协调标签（中文 README） | 打包窗口最小修复（未被 MANIFEST 钉住，可就地改） |
| verify_t3_ou.py 等 docstring | "A4 Gate-B audit" / arm 名 | 代码注释内审计归属 + arm 名 | 审计可追溯性记录，与 A5 的 94 点泄漏（路径/用户名/exact 串）不同型；是否脱敏待 MGR 统一口径 |

### 3. 结论

- A2 线投稿复现路径（40 脚本 + README + tex）：**无路径/用户名类硬泄漏**，
  Z1/Z2 --with-checker 预期主阻断项为零。
- 待裁/待办两项（均不自行执行）：
  1. `check_mirror_parity_r801.py` 绝对路径——打包时排除或 MGR 批准后参数化+重钉。
  2. REPRO_README.md "A2 线" 标签——打包窗口就地改（未钉住）。
  3. paper.tex L317 "A4 audit" 散文归属——若 MGR 要求彻底去标识，须解冻正文；
     登记为候选，不主动建议（锁稿完整性优先）。

## 核验留痕（本轮现场实测）

- 五件套 sha256 双端：tex=e9d9a84c / pdf=bd715137 / bib=93afcf65 / bbl=ff69217c /
  ai_use=89101b36（与 MANIFEST 权威值一致）。
- MANIFEST 双端 sha256+md5 各 64/64 OK（在各自正确 cwd 跑 -c）。
- 双端 cmp 五件套 5/5 IDENTICAL。
- mtime 停锁稿窗口（2026-08-09 16:43~22:03），23:00 实测零触碰。
- consistency_r790：JDIR=paper/A2_lrphase/code，154 pass / 0 fail，exit=0。

---

## r865 勘误附注（第18例记录层登记）

上文 §2 表 L37 与 §3 结论 L46 登记「REPRO_README.md 未被 MANIFEST 钉住」为**事实性错误**：r865 现场实测 `grep -c REPRO_README MANIFEST_sha256.txt` = 1（双端同值 682af719…）。该文件自始钉住，「就地改」路径不成立——正确处置与 parity 脚本同型（Z1/Z2 打包排除 或 MGR 批准解冻重钉）。待办行前提已在本轮更正；本附注不动原文留痕。教训(bf)：登记「未钉住」必须附 grep MANIFEST 命中计数=0 的证据。
