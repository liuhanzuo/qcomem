# R40 baseline detector matrix

This directory contains a retrospective, read-only comparison between a frozen
conventional assertion/testing suite and ForkAudit over already archived cases.
It does not run a model and does not access GPU or QuickSilver resources.

Read `FROZEN_PROTOCOL.md` before interpreting any count.  In particular, the
analysis is not outcome-blinded, primary projected assertions are not independent
executions, and the eight historical rows are coordinates of one defect
mechanism rather than eight independent defects.

Generate the immutable first result directory from the repository root:

```bash
python3 paper_autonomous_multifork_iteration/evidence/r40_baseline_detector_matrix_v1/build_matrix.py \
  --repo-root . \
  --output-dir paper_autonomous_multifork_iteration/evidence/r40_baseline_detector_matrix_v1/results
```

Verify deterministic replay without overwriting products:

```bash
python3 paper_autonomous_multifork_iteration/evidence/r40_baseline_detector_matrix_v1/build_matrix.py \
  --repo-root . \
  --output-dir paper_autonomous_multifork_iteration/evidence/r40_baseline_detector_matrix_v1/results \
  --verify-existing
```

Run local tests:

```bash
python3 -m unittest \
  paper_autonomous_multifork_iteration/evidence/r40_baseline_detector_matrix_v1/test_build_matrix.py -v
```

`results/baseline_detector_matrix.json` is the authoritative per-case matrix;
the CSV is a flat view.  `summary.json` reports only fixed-case counts and clean
false-positive counts, never population rates.

`RESULT_REPORT.md` summarizes the observed counts and the narrow supported
incremental claim.  It is interpretive documentation; the JSON matrix and
frozen protocol remain authoritative.
