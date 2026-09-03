# R40 observed result report

## Completed comparison

The frozen retrospective analysis completed over 52 archived rows: 22
fault/defect-coordinate rows and 30 matched clean-control rows.  These are fixed
case counts, not detection rates or estimates over a defect population.

| Campaign | Fault/defect rows | Clean rows | Conventional full | Conventional strict-observer subset | ForkAudit | ForkAudit unique | Both | Clean case catches (baseline / ForkAudit) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Primary M1--M9 | 9 | 9 | 8 | 2 | 9 | 1 | 8 | 0 / 0 |
| Designer--executor R33 | 5 | 5 | 5 | 1 | 5 | 0 | 5 | 0 / 0 |
| Historical alias R35 | 8 coordinates | 16 | 8 | 8 | 8 | 0 | 8 | 0 / 0 |
| **All rows** | **22** | **30** | **21** | **11** | **22** | **1** | **21** | **0 / 0** |

The eight R35 fault rows are repeated frozen coordinates of one historical
defect mechanism, not eight independent defect families.  Detector
applicability also varies by case.  The zero false-positive result therefore
means zero case-level catches among these 30 controls; it is not a precision
estimate.  The matrix records 114 evaluated conventional detector decisions on
clean rows, with remaining clean decisions explicitly marked `not_evaluated`
or `not_applicable`.

## Incremental evidence

The one full-suite ForkAudit-unique row is `primary/M8/fault`.  ForkAudit first
rejects at `KERNEL_CALLABLE_ID`, localizing the attention callable identity at
dispatch.  The frozen conventional suite has no callable-provenance rule;
matched output observers were unavailable because execution ended first, and
the deliberate fault-payload abort is correctly excluded from the runtime
assertion baseline because an injector sentinel is not a detector.

The other 21 fault/defect-coordinate rows are caught by both ForkAudit and at
least one conventional rule in the permissive full view.  That result does not
support a claim that ordinary assertions broadly fail.  It instead supports a
narrower incremental claim: ForkAudit unifies typed, stage-specific
localization across the suite and covers the callable-provenance case that the
frozen conventional suite misses.

The full conventional result is deliberately generous to the baseline: it
includes direct observers, checks recomputed from archived raw receipts, and 14
primary decisions projected from target-gate-suppression events.  The latter
are counterfactual projections rather than independent detector executions.
The strict-observer subset counts only decisions labeled
`executed_independent_observer`; it catches 11 of the 22 fixed fault rows.  This
label refers to an explicitly evaluated observer result under the frozen
protocol and must not be expanded into a claim of an independently developed
implementation or independent capture.

## Detector-level counts

Conventional detectors can overlap on one case, so the following counts do not
sum to 22:

| Detector | Fault-row catches | Clean-row catches |
|---|---:|---:|
| B1 runtime assertion | 1 | 0 |
| B2 token equality | 1 | 0 |
| B3 full-logit equality | 2 | 0 |
| B4 structural/sequence/argument | 4 | 0 |
| B5 persistent-base immutability | 10 | 0 |
| B6 simple alias/overlap | 11 | 0 |
| B7 lifecycle/cardinality | 4 | 0 |

## Reproducibility and authoritative artifacts

- `results/baseline_detector_matrix.json`: authoritative per-case detector
  decisions, evidence modes, first localizations, sources, and catch relations.
- `results/baseline_detector_matrix.csv`: flat reviewer-readable view.
- `results/summary.json`: count-only aggregate and claim boundaries.
- `results/input_manifest.json`: SHA-256 bindings for 65 input/source files.
- `results/run_receipt.json` and `results/SHA256SUMS`: product receipt and hashes.

The deterministic replay command and three local unit tests pass.  No model,
GPU, or QuickSilver operation was performed for this analysis.

