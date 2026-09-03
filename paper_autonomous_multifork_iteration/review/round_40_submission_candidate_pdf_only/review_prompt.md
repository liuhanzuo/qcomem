# Round 40 PDF-only blind-review protocol

Review exactly the frozen 29-page PDF at
`build/r40_submission_candidate_v1/main_r40_submission_candidate.pdf`, whose
SHA-256 is
`0906080e3d16c0f8ee071f5d3aa2f6d4d541e7f7d8a2cc3efe9e67c0d0916d5b`.
Use the canonical ICLR-style rubric at
`skill_release/autonomous-paper-agent-v2/references/review-rubric.md`, whose
SHA-256 is
`df368bb0b31b60a75f81d155a6b01962865aedb2b2984443f3ba6cd8c153d874`.

Each reviewer is isolated and may read only the PDF and rubric. Repository
source, evidence packages, logs, prior reviews, other reviewers, and network
search are excluded. Review the complete PDF, including appendices and visual
layout. Report overall score on the 2/4/6/8/10 scale, confidence on 1--5,
soundness/presentation/contribution on 1--4, strongest contribution, major and
minor issues, evidence needed for an 8, and evidence that would reduce the
score to 4. Treat this as an internal simulation, not an acceptance prediction.

One initial third-review attempt used an unresolved rubric path. It is invalid,
excluded from every aggregate below, and replaced by a fresh isolated reviewer
that verified the canonical rubric hash before reading the PDF.
