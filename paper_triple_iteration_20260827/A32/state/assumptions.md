# A32 post-meta assumptions and semantic lock

- Every empirical numerical value is source-reported and cannot be independently
  verified from the current package; no experiment was rerun.
- The central quantity is the cross-level mean recorded commit-score minus
  final-answer-correctness difference. `SCE` is only an internal artifact alias.
- The commit score is not verified as a probability of final-answer correctness.
  Source-labelled Brier/NLL measurements are excluded from the manuscript because
  their definitions and artifacts are unavailable.
- The natural-strata value `+0.2227` and paired filler value `-0.0075` are separate
  source records only. Their samples and estimands differ, and no cross-record
  functional, ordering, joint test, or shared inferential interpretation is available.
- The paired filler target is a finite-pool contrast conditional on the recorded
  single-decode realization, not an expectation over problems or decoder randomness.
- The exact filler, insertion, tokenizer-level dose check, sample manifest/selection,
  overlap, prompt serialization, arm construction, and score aggregation are absent.
- The source-mentioned random-dose regression is excluded because its row data,
  equation, design matrix, and code are absent.
- MATH500 numerical scenarios are excluded because judge, sample, and correction
  artifacts are unavailable.
- Unvalidated source brackets are excluded; the manuscript makes no interval or
  coverage claim.
- No causal natural-length, decoder-average, cross-model, cross-task, or transport
  claim may be introduced.

Baseline PDF SHA-256: `c557217eba583f4b47559e4fcbc1c75c71f688febb1f9c633cbdab4d45906277`.
