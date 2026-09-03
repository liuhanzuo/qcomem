# A11 same-endpoint comparator experiment

This directory is an isolated, CPU-only formal comparison of three rules on
the frozen OMR count-replay manifest:

1. `BAYES-H`: the FIT-frozen empirical count-mixture score from A11;
2. `BAYES-UNIF`: the same score rule with a uniform distribution on final
   counts; and
3. `WOR-PPR-CS`: the binary hypergeometric prior-posterior-ratio confidence
   sequence of Waudby-Smith and Ramdas (2020), converted into a full-count
   majority decision rule.

The protocol, alpha grid, primary alpha, working prior, running-intersection
choice, CAL family, uncertainty calculations, and output contract are frozen
in `protocol.md`, `config.json`, and `output_schema.json`.  TEST must not be
aggregated until the preflight, FIT lock, and CAL lock exist and pass their
hash checks.

The experiment is conditional on the supplied anonymous derived manifest.  It
does not reconstruct the source parquet, observe natural rollout order,
measure gold correctness, or measure serving cost.

## Staged command sequence

From this directory:

```bash
python3 run_comparison.py preflight
python3 run_comparison.py fit
python3 run_comparison.py cal
python3 run_comparison.py test
python3 run_comparison.py verify
```

Every stage refuses to overwrite an existing formal output. `verify` does not
aggregate TEST a second time; it checks the immutable stage chain, input hash,
output schema, and fixed-`K` CS audit, then writes a verification record.

The completed run is summarized in `AUDIT_REPORT.md` and registered in
`RUN_REGISTRY.json`.

