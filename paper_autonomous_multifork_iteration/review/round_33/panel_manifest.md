# Round 33 PDF-only panel

- Frozen input: `pdf_only_input/forkaudit.pdf`
- SHA-256: `94654c78b8c4baf3fd4721670ef7776e94399e3f5ce3942cfd049ef93c204d96`
- Input directory contents: exactly one PDF
- Reviewers: three independent `gpt-5.6-terra` agents
- Reasoning setting: `high`
- Prompt: identical for all three reviewers; archived in `review_prompt.txt`
- Visibility rule: PDF only, including appendix; no source, artifacts, history, or other reviewer outputs
- Score scale: `{2, 4, 6, 8, 10}`; confidence `1–5`
- Scores: `4, 4, 4`
- Confidences: `4, 4, 4`
- Verdicts: `Reject, Reject, Reject`
- Synthesis: `meta_review.md` (root-agent synthesis, not a fourth review)
