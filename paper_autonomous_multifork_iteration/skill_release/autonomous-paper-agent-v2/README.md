# Autonomous Paper Agent v2.0

这是一个可同时用于 **Codex** 和 **Claude Code** 的论文自主写作 skill 包。相比普通“帮我写 paper”的 prompt，它新增了真正的多审稿人闭环：

1. 主 agent 负责研究、实验、写作和修改；
2. 每轮冻结一个不可混淆的 submission snapshot；
3. 5 个全新、互相隔离、只读的 reviewer subagent 并行盲审；
4. 独立 meta-reviewer 汇总分歧、打 meta-score、生成修改优先级；
5. 主 agent 先把审稿意见分成补实验、重分析、收窄 claim 或不成立；需要补实验时先预注册、再执行、独立验算，之后才修改论文；
6. change-verifier subagent 检查问题是否真的解决；
7. 新一轮 reviewer 不看旧分数和旧意见，重新盲审；
8. 用分数分布、严重问题数量、真实性与可复现性共同选择最佳 checkpoint，而不是默认采用最后一版。

视觉资产遵循“证据图与解释图分离”：数值图表必须由登记证据确定性生成；架构概念图、teaser 和工作流插图可调用 `imagegen`，但必须标为 illustrative/conceptual、保存 prompt 与 SHA，并且不能替代任何实验门禁。

内部评分只是质量优化信号，不代表真实会议录用概率。

## 目录

```text
autonomous-paper-agent-v2/
├── README.md
├── START_PROMPT.md
├── skill/
│   ├── SKILL.md
│   ├── references/
│   │   ├── evidence-contract.md
│   │   ├── review-protocol.md
│   │   └── review-rubric.md
│   ├── templates/
│   │   ├── change-verification.schema.json
│   │   ├── meta-review.schema.json
│   │   ├── paper_state.json
│   │   ├── review.schema.json
│   │   └── visual-asset-provenance.json
│   └── scripts/
│       ├── aggregate_reviews.py
│       ├── make_review_snapshot.py
│       └── select_best_round.py
└── adapters/
    ├── codex/.codex/
    │   ├── config.toml.example
    │   └── agents/
    └── claude/.claude/agents/
```

## 安装到 Codex 项目

在目标论文 repository 根目录执行：

```bash
mkdir -p .agents/skills/autonomous-paper-agent
mkdir -p .codex/agents

cp -R /path/to/autonomous-paper-agent-v2/skill/. \
  .agents/skills/autonomous-paper-agent/
cp -R /path/to/autonomous-paper-agent-v2/adapters/codex/.codex/agents/. \
  .codex/agents/
```

把下面配置合并进 `.codex/config.toml`，不要粗暴覆盖已有配置：

```toml
[agents]
enabled = true
max_concurrent_threads_per_session = 6
default_subagent_reasoning_effort = "high"
```

启动 Codex 后可显式调用：

```text
$autonomous-paper-agent
```

随后粘贴 `START_PROMPT.md` 中的任务模板。

## 安装到 Claude Code 项目

在目标论文 repository 根目录执行：

```bash
mkdir -p .claude/skills/autonomous-paper-agent
mkdir -p .claude/agents

cp -R /path/to/autonomous-paper-agent-v2/skill/. \
  .claude/skills/autonomous-paper-agent/
cp -R /path/to/autonomous-paper-agent-v2/adapters/claude/.claude/agents/. \
  .claude/agents/
```

启动 Claude Code 后可显式调用：

```text
/autonomous-paper-agent
```

如果当前 session 启动时 `.codex/agents/` 或 `.claude/agents/` 尚不存在，而新建后没有被识别，重启一次对应客户端。

## 推荐调用方式

不要只说“写一篇 paper”。至少给出：

- 目标 venue 或通用 ML conference；
- repository、实验日志和草稿的位置；
- 哪些实验允许继续运行；
- 可用计算资源或明确的资源上限；
- 是否允许联网检索文献；
- 最大完整盲审轮数，默认 5；
- 希望达到的内部 reviewer median，默认 8/10（ICLR ``accept, good paper`` 档；总分只允许 2/4/6/8/10）。

未提供的信息由 agent 保守推断并写入 assumptions ledger，不应因为普通决策反复询问人类。

## 审稿闭环

每轮默认角色：

1. novelty and positioning reviewer（只看 PDF）；
2. technical soundness reviewer（只看 PDF）；
3. experiments and statistics reviewer（看 PDF + 匿名仓库）；
4. clarity and presentation reviewer（只看 PDF）；
5. reproducibility and provenance reviewer（看 PDF + 匿名仓库）。

也就是说，默认 3/5 reviewer 只能看到投稿 PDF，2/5 reviewer 才能深入检查匿名代码与证据包。这样既保留 artifact audit，又不会让仓库里的解释替 PDF 补课。每份 reviewer JSON 都必须记录 `access_mode`，5 人面板若不是 3 个 `pdf_only` + 2 个 `pdf_plus_repository`，聚合器会拒绝。

### 收到审稿意见后为什么先补实验

决定性意见不能靠改写措辞“解决”。Skill 会先生成 `review/experiment_response_plan.json`，冻结可证伪假设、主指标、oracle、controls、数据/硬件配置、停止规则和正/空/负结果对应的论文决策；随后先跑 smoke，再跑 formal experiment，保存失败和负结果，并由独立 verifier 从 raw artifacts 重算。只有证据冻结且验收后，writer 才能改摘要、正文、表图与结论。

之后由独立 meta-reviewer 处理共识与分歧。所有 critical issue 和被选中的 major issue 修改后，还要由 change-verifier 单独验收。

### 为什么不让 reviewer 直接改稿

Reviewer 只读可以减少角色污染。它负责判断问题和验收标准；主 writer 负责综合修改；change-verifier 再检查修改是否真正解决问题。若 reviewer 一边打分一边改稿，它会更容易为自己的改法辩护，评分也更不独立。

### 为什么下一轮不能看到旧分数

旧分数会造成锚定和讨好目标阈值。新一轮 reviewer 只看当前 frozen snapshot，主 agent 在本轮所有打分完成后才做跨轮比较。

## 工具脚本

### 冻结审稿快照

```bash
python .agents/skills/autonomous-paper-agent/scripts/make_review_snapshot.py \
  --round 0 \
  --paper paper.pdf \
  --pdf-only-include .agents/skills/autonomous-paper-agent/references/review-rubric.md \
  --include evidence/claim_evidence_map.tsv \
  --include evidence/method_provenance.tsv \
  --include evidence/experiment_registry.json \
  --include literature/citation_lock.json
```

命令会生成两个互相隔离的 view：`pdf_only/` 只含 PDF 与 rubric，`pdf_plus_repository/` 才含匿名证据；两个 view 中的 PDF 必须逐字节相同。

Claude Code 安装时，把 `.agents/skills/...` 换成 `.claude/skills/...`。

### 聚合 reviewer JSON

```bash
python .agents/skills/autonomous-paper-agent/scripts/aggregate_reviews.py \
  review/round_00/reviews
```

### 选择最佳 round

```bash
python .agents/skills/autonomous-paper-agent/scripts/select_best_round.py review
```

最佳版本按真实性、critical/major issue、分数中位数与下四分位数、meta-score 和维度分数综合选择，不保证最后一轮获胜。

## 建议的 Git 策略

每轮至少建立三个 checkpoint：

```text
round-RR-before-revision
round-RR-after-revision
round-RR-reviewed
```

不要让多个 reviewer subagent 写同一套文件；它们应该只返回 JSON。所有写入、合并和修稿由主 agent 串行完成。

## 适合的输入状态

- 已有完整实验 repo：可以直接做 evidence inventory、写作和多轮盲审。
- 已有初稿：先审计 claim-evidence，再进入 Round 0。
- 只有 idea：可以自主检索和设计实验，但最终分数受实际可执行实验与证据限制。
- 缺少关键实验环境：应缩小 claim 或输出 blocker，不能生成模拟结果充数。
