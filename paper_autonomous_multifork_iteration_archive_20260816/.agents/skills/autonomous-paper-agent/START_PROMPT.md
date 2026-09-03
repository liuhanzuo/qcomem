# 启动 Prompt

使用 `autonomous-paper-agent` skill，对当前 repository 中的研究项目进行端到端论文构建或改进。

目标是得到**最强但完全可审计、可复现、不过度声称**的 submission，而不是只生成看起来像论文的文字。默认自主推进，不要因为普通的 framing、实验排序、章节组织或措辞决策询问我；请生成备选方案、基于证据选择并记录决策。

## 输入与目标

- 目标 venue：从 repository 中识别；识别不到时采用通用顶级 ML conference 标准并记录假设。
- 研究材料：当前 repository 的代码、配置、日志、结果、notes、已有草稿、图表和 bibliography。
- 允许操作：读取和修改当前 workspace、运行非破坏性脚本与实验、编译 LaTeX、检索并核验文献；不要执行对外投稿、发布或其他不可逆操作。
- reviewer 目标：默认 5 个独立 reviewer subagent，每轮 reviewer median 争取达到 7/10，且不得牺牲真实性、证据链或可复现性。
- 最大完整盲审轮数：5。

## 强制多 Agent 协议

每个完整审稿轮次都必须真实调用 subagent，不得由主 agent 在同一上下文里假装多个 reviewer：

1. 冻结当前 submission snapshot 并计算 hash。
2. 并行启动 5 个全新、只读、互不交流的 reviewer subagent，角色分别为：
   - novelty and positioning；
   - technical soundness；
   - experiments and statistics；
   - clarity and presentation；
   - reproducibility and provenance。
3. Reviewer 只能看到本轮 snapshot、rubric 和自己的角色；不能看到旧 review、旧 score、目标 score、修改记录或作者内部计划。
4. 等全部独立 review 返回后，再启动另一个全新 meta-reviewer subagent，处理共识、分歧、错误审稿意见、meta-score 和修改优先级。
5. 主 agent 根据 claim-evidence 和 method-provenance 修改，不能通过编造实验、引用、数字或实现细节满足 reviewer。
6. 每个 critical issue 和被采用的 major issue 修改后，调用 change-verifier subagent 验收，只有底层证据或逻辑真正修复才能关闭。
7. 冻结新 snapshot，并用全新的 reviewer contexts 重新盲审。不要复用上一轮 reviewer thread。
8. 每轮保存原始 review JSON、meta-review、issue ledger、修改记录、验证报告和 score trajectory。
9. 不要默认最后一轮最好；根据真实性 gate、严重问题、score median、lower quartile、meta-score 和维度评分选择最佳 checkpoint。

## 停止条件

只有在以下之一发生时停止：

- 所有 integrity gate 和 reviewer gate 通过；
- 达到最多 5 个完整审稿轮次；
- 连续两轮 median 提升小于 0.25、严重问题没有减少，且不存在高置信度高收益修改；
- 剩余提升依赖当前无法获得的外部数据、算力、凭据、人体实验或作者私有知识。

达到 plateau 时输出最佳诚实版本和精确 blocker，不要用模拟结果填空。内部 subagent score 只能作为优化信号，最终报告不得声称它保证真实录用。

现在开始：先完成 repository/evidence inventory、paper state、claim-evidence map 和 baseline checkpoint，然后自主执行直到停止条件满足。
