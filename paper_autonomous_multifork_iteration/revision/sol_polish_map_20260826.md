# Sol/Terra review-guided editorial map — 2026-08-26

Source candidate: `main_memory_primary.tex`

Revised candidate: `main_sol_polished.tex`

Intervention: full-manuscript, review-guided substantive edit at submission depth.
All edits are evidence-preserving; no experiment, citation, numeric result, or
claim scope is added.

| Reviewer concern | Location | Diagnosis | Category | Action | Verification | Status |
|---|---|---|---|---|---|---|
| Store–F1 and ForkAudit read as two competing paper identities | Abstract; Introduction contribution list; Results Store–F1 subsection | The independent deployment result is prominent but its relationship to the primary audit track is announced too late | writing / structure | State the two bounded evidence tracks before the contribution list; retain the memory result as a prominent separate bullet and add an explicit Table 3 pointer | Abstract, Introduction, Results, Discussion, and Conclusion use the same primary-audit/separate-Store boundary | completed |
| Target, receipt, coverage, verdict, and gate are hard to distinguish | Section 3 | Terminology is distributed across several paragraphs | writing / terminology | Add one compact taxonomy before the target list and keep gate terminology limited to fault localization | Full-text terminology scan and 27-page PDF reread passed | completed |
| “Exact semantics” can be misread as end-to-end correctness | Introduction, Results, Discussion, Conclusion | The evidence is canonical equality of protocol-defined registered observables | scope / precision | Replace broad wording with canonical equality of registered observables and strengthen the local definition | Citation/label scan and independent semantic-drift verification | completed |
| Results 5.2 mixes primary evidence, fault evidence, and supporting cohorts | Section 5 | One subsection carries several incompatible paragraph jobs | structure | Separate ownership/allocator, falsification, supporting lifecycle/capture, and Store–F1 tracks with clearer headings | Results hierarchy is visible; main text ends on page 9 | completed |
| “Every phase passes” appears inconsistent with partial dispatch | Results 5.2 | Phase-local ownership predicates and the seven target statuses are different domains | clarity | Name the evaluated ownership/immutability predicates and restate dispatch partial separately | Section 5.1, Section 5.2, Table 6, Abstract, and Conclusion now agree | completed |
| Fault/control names drift and Conclusion omits categories | Abstract, protocol, Results, Conclusion | Different experimental groups are easy to read as one fault population | terminology / synthesis | Stabilize names for 44 seeded wrong-operator controls, nine designed all-gates-on mutations, five designer–executor-separated mutations, and one historical no-injection defect | Numeric and terminology scan passed | completed |
| Repeated Store caveats make the memory result defensive | Abstract, Introduction, Results, Discussion, Conclusion | The same exclusions recur without a distinct paragraph job | concision | Give each location one job: denominator in Abstract, speed/comparator in Introduction and Results, non-additivity in Discussion, bounded synthesis in Conclusion | Full-paper PDF text and visual reread passed | completed |
| Trusted capture, one-stack scope, natural-fault comparison, compiled dispatch, and adoption overhead limit the score | Throughout | These are evidence gaps, not prose defects | evidence / experiment | Preserve explicit limitations; do not mark them resolved or strengthen claims | Fresh verifier and final report | unresolved evidence |

## Final verification

- Safe no-shell-escape LaTeX build succeeds; 27 pages total.
- Main text, including Conclusion, ends on page 9; References begins on page 10.
- All 27 rendered pages were inspected after the final manuscript edit.
- Citation keys are unchanged; the only new label is `sec:target-status`.
- LaTeX environments balance 37/37; no overfull boxes, undefined citations,
  undefined references, or rerun warnings remain.
- Final PDF SHA-256: `f55c0c2dca7201904ff82897af75e6f7fc6a31cbf52a1ee76624280d4cdcb72c`.
