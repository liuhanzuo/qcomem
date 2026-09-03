# Round 38 PDF-only Terra panel

## Panel result

- Official ICLR overall ratings: 4 / 4 / 4
- Median / lower quartile / minimum: 4 / 4 / 4
- Confidence: 4 / 4 / 4
- Dimension medians: Soundness 3/4; Presentation 3/4; Contribution 2/4
- Recommendations: Weak Reject / Weak Reject / Borderline Reject
- Evidence ceiling reported independently by all three reviewers: 6
- Snapshot SHA-256: `064a6fd55eda24a58c082d4ed8893a187df22a1adc9983d7472edf77b03facf3`

This panel uses the official ICLR 2026 discrete ratings, so its numerical values
must not be averaged directly with Round 37's legacy 1–10 scores.  Categorically,
both panels place the paper marginally below the acceptance threshold.

## Consensus strengths

- The ownership problem is real: token/logit agreement does not establish
  isolation of mutable hybrid state.
- Coverage-versus-verdict separation, the fail-closed record schema, transition-
  time ownership checks, and pointer-free storage witnesses are careful and
  useful within the declared trace boundary.
- The threat model, TCB, incomplete compiled-dispatch coverage, and cohort
  boundaries are disclosed unusually clearly.
- Figures and main tables are polished and legible.
- The retained-Store numbers and memory-denominator separation are accurate; no
  reviewer identified a numerical, comparator, or semantic defect.

## Consensus decision-driving limitations

1. Producer-side slot enumeration and semantic binding remain trusted; the
   process-separated observer does not independently establish event/slot
   completeness or end-to-end execution.
2. Evidence remains one Qwen3.5/H20 stack with a constrained schedule; native
   continuous/ragged batching, compiled recurrent kernels, kernel overlap, and
   in-flight cancellation are not demonstrated.
3. Fault evidence is predominantly constructed around the registered predicate
   vocabulary; one historical case does not establish natural-defect recall,
   false-positive behavior, or advantage over conventional invariant suites.
4. Compiled dispatch and autotuning choices remain unbound; captured-boundary
   numerical oracles are not independent end-to-end validation.
5. Contribution is judged as rigorous integration/formalization rather than a
   broadly validated new serving technique.

## Writing-only feedback

- The paper is dense, acronym-heavy, and repetitive about claim boundaries.
- A compact running example should distinguish target, gate, coverage, verdict,
  receipt, and supporting control.
- The appendix and contextual tables could be compressed or organized more
  selectively.
- The memory result is correctly reported, but two reviewers believe its
  eight-item Store–F1 evidence is too small for its current prominence; it
  should remain visibly separate from the ForkAudit validation claim.

## Interpretation

The memory-primary revision did not repair or worsen the main numerical claims.
It also did not raise the decision category because the panel's limiting issues
are evidence breadth, observer independence, fault comparators, and dispatch
coverage.  Editorial revision can improve Presentation, but cannot honestly
close those evidence gaps.
