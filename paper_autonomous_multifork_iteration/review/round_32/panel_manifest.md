# Round 32 PDF-only panel

- Frozen input: `pdf_only_input/forkaudit.pdf`
- SHA-256: `a34f319550300d603db259a69c5685112009b2d0a3d92aa3096a121624fb6db3`
- Input directory contents: exactly one PDF
- Reviewers: three independent `gpt-5.6-terra` agents
- Reasoning setting: `xhigh`
- Prompt: identical for all three reviewers; archived in `review_prompt.txt`
- Visibility rule: PDF only, including appendix; no source, artifacts, history, or other reviewer outputs
- Score scale: `{2, 4, 6, 8, 10}`; confidence `1–5`
- Scores: `6, 4, 4`
- Confidences: `4, 4, 4`
- Verdicts: `Accept, Reject, Reject`
- Synthesis: `meta_review.md` (root-agent synthesis, not a fourth review)
