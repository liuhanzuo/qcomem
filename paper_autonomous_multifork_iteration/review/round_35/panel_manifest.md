# Round 35 PDF-only panel

- Frozen input: `pdf_only_input/forkaudit.pdf`
- SHA-256: `caab4a98117b0f3148e5a362d5d34972b4595a1ae0ef8154409df3847bbf1cbc`
- Input directory contents: exactly one PDF
- Reviewers: three independent `gpt-5.6-terra` agents
- Reasoning setting: `high`
- Context fork: `none`
- Prompt: identical for all three reviewers; archived in `review_prompt.txt` (SHA-256 `2b90e32bcda7146945bbbb927542f0ed05ba692a0fd4620a01bb124ff0d5297e`)
- Visibility rule: PDF only, including appendix; no source, artifacts, history, or other reviewer outputs
- Score scale: `{2, 4, 6, 8, 10}`; confidence `1–5`
- Reviewer output SHA-256 values:
  - `reviewer_1.md`: `39f22ff1908ff4d83d103c4517f18a428b9e9f6e0698211efc55ec73ca84139a`
  - `reviewer_2.md`: `27853c840fedd6cdd31d27e93c5892ebf9cf0ec22a91d3cccf7b03a4b2b41379`
  - `reviewer_3.md`: `f58385cf2af9c6450e4a55a6b7a20cbe3e6469d14b1487e8a26e7ee923768e03`
- Scores: `4, 4, 4`
- Confidences: `4, 4, 4`
- Verdicts: `Reject, Reject, Reject`
- Synthesis: `meta_review.md` (separate `gpt-5.6-terra` meta-reviewer, not a fourth review or score; SHA-256 `7eb25c70dd4e25010616fc6d0c7dec560925d6bf746db7d5d93a57fd35eea816`)
