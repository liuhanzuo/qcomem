# Q-CoMem 60-item mixed-bit validation evidence package

This non-overwriting package preserves the completed H20 validation run behind
`RESULTS_GPU_MIXED_VALIDATION_2026-08-12_ZH.md`.  The authoritative execution
was QuickSilver Job 234340 / Trial 1830116, which ended `Complete` on
2026-08-12.  The package mirrors all 66 files from the shared result root,
including 48 raw configuration-by-rank shards, eight rank logs, the aggregate,
the cached exactness smoke, GPU inventory, and terminal stage markers.

Run the raw-first offline verifier from this directory:

```bash
./replay/run_replay.sh
```

The verifier checks every package and remote-mirror hash; reconstructs the
six 60-item cohorts; recomputes all 360 stored F1 values from predictions and
references; recomputes the reported means, paired bootstrap intervals, exact
agreement rates, catastrophic-regression counts, retained-state bytes, error
norms, and compression ratios; and requires exact agreement with the archived
aggregate.

## Authorized interpretation

This is a bounded, archival, raw-first validation result on Qwen3.5-35B-A3B,
Qasper and 2WikiMQA source indices 6--35 (30 items each), and eight H20-3e
GPUs.  The frozen-static depth-7 policy stores a mean 9.660873 MiB of tensor
payload per document versus 136.235352 MiB for the same-stack full-prefix
baseline (14.1018x smaller).  Its mean F1 is 0.542368, a delta of -0.000520
versus dense and -0.003870 versus Q16 replay; the paired 95% bootstrap interval
versus Q16 replay is [-0.020365, 0.010756], with no delta <= -0.5 among the 60
items.  This supports an observed validation-set Store--F1 Pareto point, not a
claim of statistical equivalence or universal losslessness.

`Store` is the per-document retained tensor payload recorded by
`stored_persistent_nbytes`: boundary-residual packed data/scales/biases plus
unique lower-cache tensor storage.  It is not process/NVML memory, allocator
capacity, active-request workspace, metadata, model weights, or admitted
serving capacity.

## Provenance boundary

The submitted command, data SHA, raw shards, terminal platform status, and
policy file are retained and independently checked.  However, the 2026-08-12
launcher used a mutable shared `CODE_DIR`; no pre-execution full source ledger
or full model-weight ledger was retained.  Current repository source is not
labelled as executed source.  Therefore this package supports offline replay
of the archived outcomes, but not fresh generation from a source-frozen build.

The validation raw shards contain only source indices 6--35.  Calibration used
indices 4--5 and is represented separately by `calibration/layer_policy.json`.
The submitted command names only the validation input file; no test-v2 index
68--99 appears in any raw validation shard.  This is run-scoped evidence of
split separation, not a filesystem-level proof that no unrelated process ever
read test-v2.  This package does not use or cite any later test-v2 result.

No raw row records TTFT, and no recall metric was computed.  Use the separate
registered deployment benchmark for TTFT/TPOT/throughput, and do not derive an
edge-device, speedup, recall, broad-benchmark, or production-capacity claim from
this package.
