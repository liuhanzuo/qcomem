# R28 post-execution RR2 run-binding correction

The eight H20 ranks and all 18 preregistered cases finished before the frozen
CPU aggregator stopped at `BuildError: M1 RR2 run binding`. The failure was a
projection defect: the detached RR2 manifest schema has no top-level `run_id`,
while each of its eight hash-bound shards embeds the same derivation-verifiable
run-ID receipt.

The post-execution wrapper leaves the frozen builder, replay, preregistration,
rank JSON, and FP32 sidecars untouched. It verifies the original raw-artifact
ledger, RR2 receipt derivation, all eight shard bindings, and the frozen source
hashes. It then changes only the generated in-memory RR2 comparison-row field
`run_id` from null to the preregistered value and delegates every remaining
check to the frozen builder. A second invocation rebuilt a byte-identical
summary.

Commands were run from the repository root with these path aliases:

```bash
R28_RUN=paper_autonomous_multifork_iteration/evidence/r28_full_detector_matrix/formal_run_20260824a
R28_CORRECTION="$R28_RUN/postexecution-correction"
RR2_ROOT=paper_autonomous_multifork_iteration/evidence/round_04_rr2_package/upstream
AMENDMENT=paper_autonomous_multifork_iteration/evidence/r28_full_detector_matrix/postexecution-rr2-run-binding-correction.json
```

Both invocations used
`gpu/postprocess_qcomem_qwen35_forkaudit_detector_matrix_v2.py`, the frozen
source paths in `gpu/`, the preregistration copies under `$R28_RUN`, amendment
SHA `09fec191be366d2f203113f628abf1c3a58c33967c549d204ffe854d7cc6ab96`,
preregistration SHA
`5e23539499ad672661921f7b2967be183acb1caa791876655ecf91e4a51ea9de`,
and runner SHA
`2185088131084661a71ddd177b175736eb02cd6deea532f9b206f5862e34801f`.
The aggregate used `--mode aggregate`; replay used `--mode replay
--recorded-summary "$R28_CORRECTION/detector-matrix-v2-summary.json"`.

Terminal checks:

```bash
cmp "$R28_CORRECTION/detector-matrix-v2-summary.json" \
  "$R28_CORRECTION/replay/detector-matrix-v2-summary.json"
sha256sum -c "$R28_CORRECTION/terminal-products.sha256"
```

The canonical summary is scientifically valid with a mixed designed-fault
outcome: 18 cases, zero operational-invalid cases, and four measured
non-ForkAudit escapes. This is a per-fault comparison, not a population rate.
