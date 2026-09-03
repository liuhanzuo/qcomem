#!/usr/bin/env python3
"""Build the Chinese audit report from the immutable isolated-review JSON files."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PDF_ROOT = REPO / "output/pdf/paper_polish_20260826"
PAPERS = ["A1", "A10", "A14", "A23", "A31", "A32", "C1", "C2", "C3", "P1", "P3", "P5", "P7", "S1"]
REVIEWERS = ["R1", "R2", "R3"]
ROLE_ZH = {
    "R1": "新颖性与定位",
    "R2": "技术正确性",
    "R3": "实验严谨性",
}
RECOMMENDATION_ZH = {
    "reject": "拒绝",
    "marginally_below": "略低于接收线",
    "marginally_above": "略高于接收线",
    "accept": "接收",
    "strong_accept": "强接收",
}
SEVERITY_ZH = {"critical": "致命", "major": "主要", "minor": "次要"}
DIMENSION_ZH = {
    "novelty": "新颖性",
    "significance": "重要性",
    "technical_soundness": "技术正确性",
    "experimental_rigor": "实验严谨性",
    "clarity": "清晰度",
    "reproducibility": "可复现性",
    "citation_integrity": "引文完整性",
    "limitations_responsible_claims": "限制与负责任表述",
}


def read_tsv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def clean_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def bullets(values, empty="无"):
    if not values:
        return [f"- {empty}"]
    return [f"- {value}" for value in values]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    source = {row["paper_id"]: row for row in read_tsv(ROOT / "source_manifest.tsv")}
    baseline = {row["paper_id"]: row for row in read_tsv(ROOT / "baseline_scores.tsv")}
    frozen = {row["paper_id"]: row["snapshot_sha256"] for row in read_tsv(ROOT / "review_snapshot_manifest.tsv")}
    qa = {row["paper_id"]: row for row in read_tsv(ROOT / "qa_manifest.tsv")}

    reviews = {}
    for paper in PAPERS:
        reviews[paper] = {}
        for reviewer in REVIEWERS:
            path = ROOT / "reviews" / reviewer / f"{paper}.json"
            if not path.exists():
                raise SystemExit(f"Missing review: {path}")
            reviews[paper][reviewer] = json.loads(path.read_text())

    score_rows = []
    issue_rows = []
    dimension_counter = Counter()
    severity_counter = Counter()
    for paper in PAPERS:
        payloads = reviews[paper]
        scores = [payloads[r]["overall_score"] for r in REVIEWERS]
        confidences = [payloads[r]["confidence"] for r in REVIEWERS]
        ceilings = [payloads[r]["score_ceiling_under_current_evidence"] for r in REVIEWERS]
        predicted = [payloads[r]["predicted_score_after_required_changes"] for r in REVIEWERS]
        base = baseline[paper]["baseline_median"]
        delta = "NA" if base == "NA" else str(int(statistics.median(scores)) - int(base))
        score_rows.append({
            "paper_id": paper,
            "R1": scores[0],
            "R2": scores[1],
            "R3": scores[2],
            "median": int(statistics.median(scores)),
            "mean": f"{statistics.mean(scores):.2f}",
            "spread": max(scores) - min(scores),
            "accept_side_votes": sum(score >= 6 for score in scores),
            "median_confidence": int(statistics.median(confidences)),
            "median_current_ceiling": int(statistics.median(ceilings)),
            "median_predicted_after_required_changes": int(statistics.median(predicted)),
            "baseline_median": base,
            "median_delta_vs_baseline": delta,
        })
        for reviewer, payload in payloads.items():
            for issue in payload["issues"]:
                severity_counter[issue["severity"]] += 1
                for dimension in issue["dimensions"]:
                    dimension_counter[dimension] += 1
                issue_rows.append({
                    "paper_id": paper,
                    "reviewer": reviewer,
                    "issue_id": issue["issue_id"],
                    "severity": issue["severity"],
                    "dimensions": ",".join(issue["dimensions"]),
                    "location": issue["location"],
                    "observed_evidence": issue["observed_evidence"],
                    "why_it_matters": issue["why_it_matters"],
                    "required_fix": issue["required_fix"],
                    "verification_test": issue["verification_test"],
                    "evidence_needed": issue["evidence_needed"],
                    "expected_impact": issue["expected_impact"],
                    "confidence": issue["confidence"],
                })

    with (ROOT / "score_summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(score_rows)
    with (ROOT / "review_issue_index.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(issue_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(issue_rows)

    lines = [
        "# 14 篇论文证据保真润色与三路隔离盲审详报",
        "",
        "生成日期：2026-08-26",
        "",
        "## 一、结论先行",
        "",
        "本轮对 14 个唯一论文编号采用最新可用版本进行证据保真润色，并对每份润色后的冻结 PDF 进行三次互相隔离的 ICLR 风格盲审。三条审稿通道分别聚焦新颖性与定位、技术正确性、实验严谨性；审稿人看不到旧分数、编辑日志和彼此意见。评分使用 ICLR 离散档 2/4/6/8/10。",
        "",
        "润色的目标是改善论证结构、段落推进、术语一致性、证据边界、表图叙事和 LaTeX 版式；没有新增实验、虚构结果、改变统计量或把事后分析改写成预注册结果。",
        "",
        "> 重要限制：A23 与 P3 缺少可确认的精确 LaTeX 源，本轮是由 PDF 保守重建；必须在正式投稿前与作者持有的真实源逐句核对。A1 的主文内容仍延至第 13 页，不符合 9 页主文上限，因此不应直接提交。",
        "",
        "## 二、总评分表",
        "",
        "| 论文 | R1 新颖性 | R2 正确性 | R3 实验 | 中位数 | 分歧跨度 | 接收侧票数 | 旧中位数 | 变化 | 当前证据上限中位数 | 完成必需修改后的预测中位数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in score_rows:
        lines.append(
            f"| {row['paper_id']} | {row['R1']} | {row['R2']} | {row['R3']} | {row['median']} | "
            f"{row['spread']} | {row['accept_side_votes']}/3 | {row['baseline_median']} | {row['median_delta_vs_baseline']} | "
            f"{row['median_current_ceiling']} | {row['median_predicted_after_required_changes']} |"
        )

    accept_side = [row["paper_id"] for row in score_rows if row["median"] >= 6]
    borderline = [row["paper_id"] for row in score_rows if row["median"] == 4]
    reject_side = [row["paper_id"] for row in score_rows if row["median"] == 2]
    disagreements = [row["paper_id"] for row in score_rows if row["spread"] >= 4]
    improved = [row["paper_id"] for row in score_rows if row["median_delta_vs_baseline"] not in {"NA", "0"} and int(row["median_delta_vs_baseline"]) > 0]
    declined = [row["paper_id"] for row in score_rows if row["median_delta_vs_baseline"] not in {"NA", "0"} and int(row["median_delta_vs_baseline"]) < 0]
    unchanged = [row["paper_id"] for row in score_rows if row["median_delta_vs_baseline"] == "0"]
    lines += [
        "",
        "按中位数分组：",
        "",
        f"- 接收侧（≥6）：{', '.join(accept_side) if accept_side else '无'}",
        f"- 边缘档（4）：{', '.join(borderline) if borderline else '无'}",
        f"- 拒绝档（2）：{', '.join(reject_side) if reject_side else '无'}",
        f"- 高分歧（跨度≥4）：{', '.join(disagreements) if disagreements else '无'}",
        "",
        "与可用旧盲审中位数比较：",
        "",
        f"- 上升：{', '.join(improved) if improved else '无'}",
        f"- 不变：{', '.join(unchanged) if unchanged else '无'}",
        f"- 下降：{', '.join(declined) if declined else '无'}",
        "- 标为 NA 的论文没有可比的旧三评中位数，不能据此断言升降。",
        "- 该对比是描述性的，不是润色效果的因果估计：新旧轮次的审稿上下文、角色和随机性并不完全相同；分数不升并不等于写作没有改善，分数下降也不能单独归因于本轮编辑。",
        "",
        "## 三、审稿协议与可审计性",
        "",
        "- 每篇论文的评分对象是同一份 SHA-256 冻结快照；冻结 PDF 与编辑工作目录分离。",
        "- 三位审稿人分别写入 R1/R2/R3 独占目录，不读取其他审稿人输出。",
        "- 审稿人只读取 PDF 与 reviewer protocol/rubric/schema；C2、C3 额外按技能要求读取 KV-cache 校准锚点。",
        "- 所有 JSON 均通过严格字段、枚举、分值范围、reviewer/role、round 和快照哈希校验。",
        "- 审稿未联网，因此 closest-work 与引文完整性意见属于稿内定位判断，不等于外部文献核验。",
        "",
        "问题索引总计：",
        "",
        f"- 致命问题条目：{severity_counter['critical']}；主要问题条目：{severity_counter['major']}；次要问题条目：{severity_counter['minor']}。",
    ]
    for dim, count in dimension_counter.most_common():
        lines.append(f"- {DIMENSION_ZH.get(dim, dim)}：{count} 条 reviewer issue 标注。")

    lines += [
        "",
        "## 四、论文级详报",
        "",
    ]
    score_lookup = {row["paper_id"]: row for row in score_rows}
    for paper in PAPERS:
        row = score_lookup[paper]
        pdf_path = PDF_ROOT / f"{paper}-polished.pdf"
        if sha256(pdf_path) != frozen[paper]:
            raise SystemExit(f"Polished PDF changed after freezing: {paper}")
        lines += [
            f"### {paper}",
            "",
            "#### 交付与来源状态",
            "",
            f"- 选定输入：`{source[paper]['input_pdf']}`",
            f"- 源状态：`{source[paper]['source_status']}`",
            f"- 润色 PDF：`{pdf_path}`",
            f"- 冻结 SHA-256：`{frozen[paper]}`",
            f"- 总页数：{qa[paper]['polished_pdf_pages']}；主文状态：{qa[paper]['main_text_status']}。",
            f"- 版面核验：{qa[paper]['visual_qa']}；构建：{qa[paper]['build_status']}。",
            f"- 旧评分基线：{baseline[paper]['baseline_ratings']}；旧中位数：{baseline[paper]['baseline_median']}。",
            "",
            "#### 三评量化结果",
            "",
            "| 审稿人 | 角色 | Overall | Confidence | 建议 | Soundness | Presentation | Contribution | 当前上限 | 必需修改后预测 |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
        for reviewer in REVIEWERS:
            payload = reviews[paper][reviewer]
            ds = payload["dimension_scores"]
            lines.append(
                f"| {reviewer} | {ROLE_ZH[reviewer]} | {payload['overall_score']} | {payload['confidence']} | "
                f"{RECOMMENDATION_ZH[payload['recommendation']]} | {ds['soundness']} | {ds['presentation']} | "
                f"{ds['contribution']} | {payload['score_ceiling_under_current_evidence']} | "
                f"{payload['predicted_score_after_required_changes']} |"
            )
        lines += [
            "",
            f"三评中位数为 **{row['median']}**，均值 {row['mean']}，跨度 {row['spread']}，接收侧票数 {row['accept_side_votes']}/3。",
            "",
            "#### 编辑记录",
            "",
            f"- [结构审计](work/{paper}/structure_audit.md)",
            f"- [语义锁](work/{paper}/semantic_lock.md)",
            f"- [修订日志](work/{paper}/revision_log.md)",
            f"- [待核验事项](work/{paper}/needs_verification.md)",
            "",
        ]
        revision_text = (ROOT / "work" / paper / "revision_log.md").read_text().strip()
        verification_text = (ROOT / "work" / paper / "needs_verification.md").read_text().strip()
        lines += [
            "**修订日志原文：**",
            "",
            *[f"> {item}" if item else ">" for item in revision_text.splitlines()],
            "",
            "**待核验事项原文：**",
            "",
            *[f"> {item}" if item else ">" for item in verification_text.splitlines()],
            "",
        ]
        for reviewer in REVIEWERS:
            payload = reviews[paper][reviewer]
            just = payload["dimension_justifications"]
            lines += [
                f"#### {reviewer}（{ROLE_ZH[reviewer]}）完整评议",
                "",
                f"**论文概述：** {payload['paper_summary']}",
                "",
                f"**最强的已核实贡献：** {payload['strongest_verified_contribution']}",
                "",
                "**维度理由：**",
                "",
                f"- Soundness：{just['soundness']}",
                f"- Presentation：{just['presentation']}",
                f"- Contribution：{just['contribution']}",
                "",
                "**优点：**",
                "",
                *bullets(payload["strengths"]),
                "",
                "**问题与可验证修复：**",
                "",
            ]
            if not payload["issues"]:
                lines.append("- 无结构化问题条目。")
            for issue in payload["issues"]:
                dims = "、".join(DIMENSION_ZH.get(item, item) for item in issue["dimensions"])
                lines += [
                    f"##### {issue['issue_id']} · {SEVERITY_ZH[issue['severity']]} · {dims}",
                    "",
                    f"- 位置：{issue['location']}",
                    f"- 观察证据：{issue['observed_evidence']}",
                    f"- 重要性：{issue['why_it_matters']}",
                    f"- 必需修复：{issue['required_fix']}",
                    f"- 验证标准：{issue['verification_test']}",
                    f"- 仍需证据：{issue['evidence_needed'] or '未另列'}",
                    f"- 预期影响：{issue['expected_impact']}；判断置信度：{issue['confidence']}。",
                    "",
                ]
            lines += [
                "**给作者的问题：**",
                "",
                *bullets(payload["questions"]),
                "",
                "**能提高评分的证据：**",
                "",
                *bullets(payload["evidence_that_would_raise_score"]),
                "",
                "**会降低评分的证据：**",
                "",
                *bullets(payload["evidence_that_would_lower_score"]),
                "",
                f"**伦理标记：** {'是' if payload['ethics_flag'] else '否'}。{payload['ethics_concerns']}",
                "",
                f"**LLM 使用披露：** {payload['llm_usage_disclosure']}",
                "",
                "**评审限制：**",
                "",
                *bullets(payload.get("review_limitations", [])),
                "",
            ]

    lines += [
        "## 五、如何使用本报告",
        "",
        "1. 先按总评分表筛选：中位数 2 的论文应优先重构贡献或补关键证据；中位数 4 的论文优先处理三位审稿人重复指出的 major/critical 问题；中位数 6 也不代表可直接投稿，仍需清除格式与证据阻断项。",
        "2. 每篇先修复可用现有证据完成的 claim narrowing、标签一致性、表图叙事和复现说明，再决定是否投入新实验。",
        "3. 新实验以 issue 中的 verification test 为验收标准，避免只增加规模而没有解决识别问题。",
        "4. 修改后应重新冻结新 SHA，并由全新审稿上下文进行第二轮盲审；不要把本轮分数作为下一轮审稿人的先验。",
        "",
        "附属机器可读文件：`score_summary.tsv`、`review_issue_index.tsv`、`reviews/R1..R3/*.json`、`review_snapshot_manifest.tsv`。",
        "",
    ]
    (ROOT / "detailed_review_report_zh.md").write_text("\n".join(lines))
    print(f"WROTE report with {len(lines)} lines, {len(score_rows)} papers, {len(issue_rows)} issue rows")


if __name__ == "__main__":
    main()
