# Round 37 PDF-only Terra panel

## Panel result

- Scores: 5 / 5 / 5
- Median: 5
- Mean: 5.0
- Confidence: 4 / 4 / 4
- Recommendations: Weak Reject / Weak Reject / Borderline (lean accept under a systems-artifact valuation)
- No scored meta-reviewer was used.

## Consensus strengths

- The ownership problem is real and output/logit equality is insufficient.
- Coverage-versus-verdict separation, the fail-closed schema, phase-indexed checks,
  storage-range witnesses, and replay packaging are technically careful.
- The paper is candid about the TCB and fixed-stack boundary.
- Figures, tables, and appendices are polished and readable.
- Memory denominators are appropriately separated; no reviewer identified a
  numerical or semantic defect in the highlighted 54.5%, 88.68%, or 93.06%
  results.

## Recurring material limitations

1. Producer-side enumeration and semantic binding remain in the trusted capture
   path; the process-separated observer does not independently establish trace
   completeness or end-to-end execution.
2. Evidence remains one model/runtime/hardware family with constrained schedules,
   no native continuous/ragged batching, and no in-flight cancellation.
3. Mutation results are constructed per-fault controls; one historical regression
   does not establish natural-defect recall, false-positive rates, or superiority
   to conventional invariants.
4. Novelty is viewed as rigorous integration/formalization rather than a broad new
   inference technique.

## Presentation feedback

The memory-first revision is regarded as accurate and visually clear. Two
reviewers nevertheless view the Store/F1 and broad related-system context as
peripheral to the audit contribution, while the third explicitly praises their
separation from the core claim. The manuscript remains dense, but this is a
secondary, correctable issue rather than the score-limiting factor.

## Unscored disposition

The panel is borderline overall: all three scores improved to 5 from Round 36's
4/4/4. Two reviewers recommend weak reject and one is borderline with a conditional
lean accept. The remaining gap is material evidence, not a memory-number,
figure-layout, or abstract/conclusion consistency defect.

