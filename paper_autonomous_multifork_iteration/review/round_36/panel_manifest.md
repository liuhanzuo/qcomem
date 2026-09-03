# Round 36 PDF-only panel

- Frozen input: `pdf_only_input/forkaudit.pdf`
- SHA-256: `bca29bca5065b3367939498d03776382d6251ab12bbddb92b0ea3f18bb5fafb4`
- Input directory contents: exactly one PDF
- Reviewers: three independent `gpt-5.6-terra` agents
- Reasoning setting: `high`
- Context fork: `none`
- Prompt: identical for all three reviewers; archived in `review_prompt.txt` (SHA-256 `20fa0592e0d8bc110516b2124c46dbc556672b7a1daa812b32cba7d03beaa262`)
- Visibility rule: PDF only, including appendix; no source, artifacts, history, or other reviewer outputs
- Isolation note: reviewer 3 was spawned as a context-free child only to bypass the root's direct-child thread limit; `fork_turns=none` and the byte-identical prompt preserved independence
- Score scale: `{2, 4, 6, 8, 10}`; confidence `1–5`
- Reviewer output SHA-256 values:
  - `reviewer_1.md`: `b4abd49260d6e76533ed58b486b7c764ace49ab6ddfa3b9ffb9cf001697130ea`
  - `reviewer_2.md`: `5f6c7478a8478a104d9aea5be70c68575d0dde476523735a3722432c7cc74dc3`
  - `reviewer_3.md`: `c6dcfccb49f916a2e5e211377ee8c0b66d6133dd09f2a05f84abc09b0664df24`
- Scores: `4, 4, 4`
- Confidences: `4, 4, 4`
- Verdicts: `Reject, Reject, Reject`
- Synthesis: `meta_review.md` (separate `gpt-5.6-terra` meta-reviewer, not a fourth review or score; SHA-256 `eca88734d87b0878e1524ff08af914a7b40d651e059c71964f93119f51846e03`)
