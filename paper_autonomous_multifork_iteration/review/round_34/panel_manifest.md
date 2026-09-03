# Round 34 PDF-only panel

- Frozen input: `pdf_only_input/forkaudit.pdf`
- SHA-256: `27c275172ce56adf0c8e086fca3783cfb036502b55e4f5f27f48a7e64e49e6f2`
- Input directory contents: exactly one PDF
- Reviewers: three independent `gpt-5.6-terra` agents
- Reasoning setting: `high`
- Context fork: `none`
- Prompt: identical for all three reviewers; archived in `review_prompt.txt` (SHA-256 `6a6cce3c574078815ace9a35d168431f9db9b9e04013ad4f85b3e63fc441f8f9`)
- Visibility rule: PDF only, including appendix; no source, artifacts, history, or other reviewer outputs
- Score scale: `{2, 4, 6, 8, 10}`; confidence `1–5`
- Reviewer output SHA-256 values:
  - `reviewer_1.md`: `feee317772948be4797e1bcc394e6ce80e0bcc563d13d627232bc7600094f001`
  - `reviewer_2.md`: `1cf70dae0910333128b925f27bf4a461527353ed56f0084cbcc8ff2282f6c117`
  - `reviewer_3.md`: `a5c9a4350d1b336ad6895ac73f6bea04b688e339592fa195aefa35b6da6a030e`
- Scores: `6, 4, 4`
- Confidences: `4, 4, 4`
- Verdicts: `Accept, Reject, Reject`
- Synthesis: `meta_review.md` (root-agent synthesis, not a fourth review)
