# Falcon-H1 formal H20 evidence (20260827a)

Status: **PASS**. This directory is the locally recovered, checksum-verified copy of the non-overwriting Falcon-H1 v2 formal run. It intentionally excludes the approximately 1 GB model snapshot.

## Primary result

Across 8 independent H20 ranks and 8 distinct frozen PG19 sources, both candidate arms matched the independent official Transformers reference at zero tolerance:

- 96/96 generated-token decisions were exact.
- 96/96 full-FP32 logit records were byte-exact (`max_abs = 0`, `relative_l2 = 0`).
- 13,824/13,824 registered semantic state-family rows were exact.
- All 864 observed stride differences were singleton-dimension-only and induced equivalent address mappings; non-equivalent stride differences were 0.
- Cross-arm checks were 16/16 and cross-fanout checks were 8/8.
- Mutable-child ownership checks were 192/192 disjoint.
- All 40 frozen negative controls were detected at their preregistered first failing predicate, including the 8/8 prefix-content mutation detectors.
- Detached replay verified the aggregate independently.

## Integrity

- Formal evidence archive SHA-256: `6cbcf860120078e743eb759e2bead74a3bf980e07c4c16f588ee735d3662d6c3`
- Aggregate SHA-256: `03b2dd60422641ffdd18ec4221a06c295ca36fa3de322dc148a5222a8579888b`
- Terminal manifest SHA-256: `1f58f788fd894a6df4342567b26c965520ee3820ca7c173fefd04b27e2de0f81`
- The `COMPLETE` marker equals the terminal-manifest digest.
- All 72 terminal-manifest entries reverified locally; the artifact ledger contains 71 rows.
- Payload: 8 reference JSON files, 8 candidate JSON files, 16 logit sidecars, 15 receipts, 18 logs, and 5 stage markers.
- Formal package tests: 20 passed, 0 failed. No traceback/error/exception hit was found in the formal logs.

Machine-readable details and every claim boundary are in `validation.json`. The original run payload is under `r39-falcon-h1-transfer-20260827b/`; the sibling `.tar.gz` is the exact remote-recovered archive.

## Claim boundary

This evidence supports only bounded exact relational transfer for `tiiuae/Falcon-H1-0.5B-Base` under the frozen Transformers 5.14.1/H20 naive-path configuration. It does not establish runtime independence, optional/compiled-kernel behavior, performance or memory savings, long-context quality, production scheduling, or generality to other Falcon revisions or architectures.
