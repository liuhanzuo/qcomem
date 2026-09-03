# 第十九维自主审计：环境/可复现包完整性（ENV_REPRO_AUDIT_R833）

**日期**：2026-08-10（第833回合）。**范围**：A2 线投稿包（workspace 镜像 `paper/`，与 run 根 `paper/A2_lrphase` 字节一致，已 cmp 核验）。**性质**：纯只读+临时副本故障注入，冻结字节零接触。

## 动机（为何是第十九维）

前 18 维核「论文内容/数字/披露/渲染/引用/计数」的正确性。本维核**正交攻击面：可复现包关于自身的声明是否为真**——hostile reviewer 对 reproducibility 的核心指控是「README 声称的环境/文件清单/manifest 与实际交付不符」。r816 artifact 追溯核的是脚本→JSON→表格的**内容映射**，r818 核构建**新鲜度**，均未核「README 自我声明的环境版本/文件计数/manifest 完整性」。本维补该缺口。

## 协议

脚本 `env_repro_audit_r833.py`（exit 0=全过）。L0 在**临时副本**做故障注入自校验（证明检测器能抓失败），再对冻结包跑主审计。全程只读冻结文件。

## 结果（20 pass / 0 fail，5 层）

**L0 自校验 2/2**：计数检测器抓 39≠40；manifest 检测器抓字节篡改。

**L1 环境版本声明 vs 真实环境 4/4**：
- README §3 钉 torch 2.3.0a0 / numpy 1.24.4。
- 实测 numpy=1.24.4（精确一致）、torch=2.3.0a0+40ec155e58.nv24.03（前缀一致，本地后缀为 NGC 构建标识）、torch.cuda.is_available()=True（对应「CUDA GPU 一张」声明）。
- 本机 python=3.10.12（README 仅要求 Python 3，满足）。

**L2 编译依赖 style 文件 5/5**：README §1 声明随附 `iclr2027_conference.sty/.bst`、`natbib.sty`、`fancyhdr.sty`、`math_commands.tex`——5 个文件全部存在。

**L3 文件清单计数 vs 磁盘 4/4**：README §4 声明「code/ 40 个实验/验证脚本 + 35 个冻结结果 JSON」。实测 code/ 恰 40 个 .py、35 个 _out.json——两计数精确一致。

**L4 MANIFEST 完整性 4/4**：92 条目全列出文件存在（missing=0）；91/92 校验通过；唯一 mismatch=RESEARCH_LOG.md，经三重独立核验确认为**已登记的活文档例外**——(i) 它是唯一 mismatch（无意外篡改）；(ii) 其 mtime(17:43) 晚于 MANIFEST mtime(16:54)（日志在 manifest 生成后追加）；(iii) SUBMISSION_CHECKLIST 第130行教训(l) 显式登记 manifest 文件集不对称并点名 RESEARCH_LOG。合法，非缺陷。

**L5 README checker 入口实跑 2/2**：README §2 命令 `CONSISTENCY_JDIR=$PWD/code python3 consistency_r790.py` 实跑，输出 `154 pass, 0 fail`，与 README 声称逐字一致。

## 审计器自身修正（2 例，非稿件缺陷）

遵循 r819/r830 教训「先自查审计器」：
1. **L2 检测器 bug**：README 用连写 `iclr2027_conference.sty/.bst` 表两个共享词干的文件，初版正则不拆 `/`，拼成不存在的单文件误判 FAIL。修为展开 `/.<ext>` 简写。两个文件本就存在——检测器错，非稿件错。
2. **L4c 检测器过强**：初版要求 CHECKLIST 含「活」字才认 RESEARCH_LOG 例外，实际 CHECKLIST 用中文「文件集不同」+枚举。改为三重独立证据（唯一 mismatch + mtime 时序 + CHECKLIST 登记），更严且不再误伤。

## 判定

可复现包关于自身的全部机器可核验声明（环境版本/编译依赖/文件计数/manifest/checker 入口）**均与实际交付一致**，0 缺陷 0 新增 flag。RESEARCH_LOG 活文档例外有完整登记链，是诚实性旁证而非漏洞。hostile reviewer 对 reproducibility package 的「声明与交付不符」攻击面无立足点。

## 教训(ac)

可复现包的「自我声明」是独立于内容的出错面——README 可能声称 N 个脚本/M 个 JSON/某环境版本，而磁盘实际不符（文件被增删而 README 滞留旧数）。凡 README 含计数/版本/文件清单，必回磁盘机器数+实测版本比对，不信散文。MANIFEST 的活文档例外需三重证据（唯一性+时序+登记）链闭环，单凭一句「是活文档」不足为凭。审计器对连写简写（`a.sty/.bst`）与中文同义表达（「文件集不同」≈「活文档」）的解析要先故障注入验证，否则把合法行为误判为缺陷。
