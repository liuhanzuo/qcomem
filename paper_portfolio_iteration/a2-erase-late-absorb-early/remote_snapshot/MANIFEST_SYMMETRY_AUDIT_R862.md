# 双端 MANIFEST 文件集对称化审计 — r862（教训 l/az2 未闭环实例修复）

**触发**：r862 会话压缩恢复。无新 MGR 指令，七项外部裁决未复。本回合做一项非重复增量：对 r861 教训(l)「双端 MANIFEST 文件集不同」+教训(az2)「0非OK 只证明清单与当前磁盘一致、不证明名单本身正确」做**主动适用性自查**——核验双端 MANIFEST 名单是否真的覆盖且对称。

## 一、发现的真实缺口

**方法**：对 `run根 paper/A2_lrphase/` 与 `workspace/paper/`（镜像）各做全递归文件枚举，与顶层 MANIFEST 跟踪名单做集合差。

**结果**：
- 顶层 MANIFEST（r861 后=65 项）只覆盖**顶层常规文件**；`code/` 子目录由独立 `code/MANIFEST_md5.txt`（75 项）跟踪——双端 code MANIFEST cmp IDENTICAL、各 0 非OK、名称集一致。**code/ 覆盖完整，非缺口**。
- **真缺口在顶层不对称**：run根顶层未跟踪仅 2 项（MANIFEST 自引用，已知约定）；**镜像侧顶层多出 41 个滞留文件**未被 run根权威端持有、未钉入 MANIFEST、且 parity 脚本（单向 run根→镜像）从未核到：
  - **9 个被 CHECKLIST 正文引用的审计产物**（属投稿包证据链，应在权威端）：6 md = CROSS_DOC_AUDIT_R834 / ENV_REPRO_AUDIT_R833 / HYPERPARAM_CONFIG_AUDIT_R838 / GATEB_FIX_RESPONSE_R787 / SALVAGE_LOSS_PREREG / ANON_SCAN_R860；3 py = env_repro_audit_r833 / hyperparam_audit_r838 / table_fig_value_audit_r841。
  - **29 个冗余历史滞留**（run根不持有的旧派生物）：RESEARCH_LOG.md（r813 时点 8KB 旧快照，活跃日志在 workspace 根 302KB）、bib.log、27 个 r785–r793w 锁稿前 build log。

**两处同族记录层缺陷**：
1. r860 CHECKLIST 自写 ANON_SCAN_R860「全文留痕（**双端**）」——实际只放镜像侧，run根缺失。属 r851/r852/r855 同族「声明与状态不符」。
2. 5 个旧审计文档（R787/R833/R834/R838/SALVAGE）自 r815–r842 审计链起滞留镜像侧，从未同步 run根——教训(l) 的具体未闭环实例。

## 二、处置（文档/证据链层维护，零 science 变更）

授权先例：r861 INEQ 审计文档双端放置+入 MANIFEST 即我执行。本批同性质。

1. **补拷 9 个审计产物** 镜像→run根权威端（cmp 逐字节一致，五件套 sha256 编辑前后实测不变）。
2. **重建双端 MANIFEST**：以 run根顶层常规文件为 canonical 名单（排除目录与 MANIFEST 自引用条目，守教训 az2 `find -maxdepth 1 -type f` + 重建后 diff 旧名单），双端按同一名单重算 sha256+md5。
   - **首跑事故（教训 az2 复犯风险实证）**：直接对镜像 `find` 重建会把 29 个镜像独有滞留卷入（镜像 MANIFEST 飙到 103 项 vs run根 74）——印证「双端文件集不同」是真实反复风险。纠正：canonical 名单锁定 run根 74 项，镜像侧先删冗余再按同名单重建。
3. **清理镜像 29 个冗余滞留**：先备份到 `workspace/frozen_r813_backup/mirror_stragglers_r862/`（29 文件全量留证）再删除，镜像顶层与 canonical 74 对齐。

## 三、全链终验（本轮回合内现场实测）

- 双端 MANIFEST：各 **74 项**、sha256 0 非OK、md5 0 非OK。
- 双端 MANIFEST 名称集 + 逐文件哈希：**IDENTICAL**（`diff` 两端 sort 后清单）。
- 五件套 sha256 双端实测：tex=e9d9a84c / pdf=bd715137 / bib=93afcf65 / bbl=ff69217c / ai_use=89101b36（与权威值一致，零改动）。
- parity 脚本（check_mirror_parity_r801.py）：**152/152 OK**（顶层74 + code/75 + 自引用约定），missing=0 stale=0，exit=0。
- consistency_r790 回归：CONSISTENCY_JDIR=workspace/lr_phase_datavalue_r1，**154 pass / 0 fail**，exit=0。
- 五件套 mtime：停 2026-08-09 16:42~22:03 锁稿窗口（本轮零触碰）。

## 四、零科学变更确认

paper.tex / paper.pdf / ref.bib / paper.bbl / ai_use_statement.tex 五件套字节零改动。本轮唯一变更 = 9 个审计产物补拷 run根 + 双端 MANIFEST 按 canonical 74 重建 + 镜像 29 冗余滞留清理（已备份）。

## 五、状态

A2 线投稿包维持 r850 锁稿完成态 + 29 维审计 + 锁后维护链 r844–r861 + 本轮双端文件集对称化。累计 flags 不变。待 MGR 七项外部裁决不变：Figure 1 重绘裁决、A1 复审 T1、hostile 正式模拟、A5 复核、Z1/Z2 打包窗口、AI-use 措辞裁决（ac2720d7313c 归属）、2605.25698 concurrent 补引批准。A2 自主范围内无其他已知缺口。

## 六、教训固化

- **教训(ba)**：「0 非OK + 双端 IDENTICAL」仍不证明**双端文件集对称**——单向 parity 脚本 + 各自 `find` 重建 MANIFEST 会各自合法、合起来不对称。对称性必须用**同一份 canonical 名单**重建双端，并显式 `diff` 两端名称集，不能各跑各的。
- **教训(bb)**：镜像端滞留的历史派生物（旧 build log / 旧日志快照）是双端不对称的持续来源。锁后维护轮的收尾清单应增一条：「镜像顶层文件集 == run根 canonical（MANIFEST 自引用除外）」，不符先清理再重建 MANIFEST。
