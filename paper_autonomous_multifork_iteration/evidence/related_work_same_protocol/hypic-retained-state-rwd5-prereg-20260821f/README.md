# RW-D5 HYPIC retained-state bytes — frozen preregistration F

This affected-only preregistration can fill only the Prefix Cache and HYPIC
`Store (MiB)` cells. It preserves the physical-byte denominator, receipt
topology, and E's bounded server cleanup while adding an external expected-cell
authority to blind replay. A through E are retired; no earlier output or audit
identity carries forward. This is not a result package. The `STOP` file requires
a fresh independent GREEN audit of this exact freeze before GPU submission.

Only 16 GPU cells are authorized: Prefix Cache and HYPIC
`transition_rope_recompute`, each on the same eight frozen Qasper/2WikiMQA
workloads. Full Recompute, CoMem, RR2, GDN, serving controls, and all other
methods remain absent. No GPU work has been submitted for D, E, or F.

## Retired attempt and freezes

Trial `1876986` under C is invalid (`0/16` raw/store cells). Its one-shot
30-second `/server_info` call overlapped roughly 46 seconds of scheduler warmup;
its leaked server groups were recovered from exact PID files and all GPUs were
verified at 0 MiB / 0%. D and E were both retired before GPU submission. D had
an unbounded cleanup ordering and weak readiness-cell binding. E fixed those
but still verified only internal receipt consistency: a fully re-signed rank-1
chain could be moved into the rank-0 file slot because `replay_one` received no
external expected cell. Structured records for C, D, and E are included.

## F external expected-cell closure

`replay_one` now requires caller-supplied `expected_mode`, `expected_rank`,
`expected_snapshot_id`, and `expected_workload_id`. `replay_all` derives these
only from its frozen `MODES`, rank loop, filename convention, and
`EXPECTED_PAIRS`; producer receipts cannot select them. Before append,
`replay_all` independently checks the returned mode, rank, snapshot, and
workload against the file position.

The discriminating regression rewrites worker, server, readiness, target,
store, terminal, raw, process environment, and every downstream SHA into a
complete internally valid rank-1/qasper-7 chain. It first passes when explicitly
replayed under that forged expectation, proving the chain is fully re-signed,
then fails under the actual rank-0/qasper-6 file-position expectation through
both `replay_one` and `replay_all`. The last successful readiness attempt's
`response_sha256` must additionally equal the readiness-level and server-level
`server_info` SHA.

## Preserved lifecycle and Store denominator

Readiness remains evidence-bearing 3-second `/server_info` polls at a 1-second
interval with a 300-second total deadline, bound to mode, rank, endpoint, and
actual launch PID. Cleanup remains `TERM`, bounded `kill -0` polling, `KILL` of
survivors, and final reap. Failure removes `COMPLETED` and writes `FAILED` before
cleanup; success creates `COMPLETED` only after cleanup.

`Store (MiB)` remains the unique overlap-aware union of physical tensor ranges
owned by the frozen document after prime and before measured query. Blind replay
derives ranges from dtype, element size, full shape, stride, storage offset,
pointer-relative identity, and exact target-bound token/slot selections.
Metadata, NVML/process allocation, reservation, and pool capacity are excluded.
All previously audited authority, topology, int8/PIC identity, zero-lock,
pre-allocation, terminal ownership, manifest, and canonical-hash gates remain.

Frozen F validation:

- official HYPIC HEAD `98147c01909004e66d98bcb18b886927d41b0ee5` clean;
- exact overlay apply/reverse, `git diff --check`, and patched compilation pass;
- canonical tracked overlay SHA-256:
  `acd14dbf615b64d4c6fea09681e3bca66bd72eccd88b0be775a478663c4486fe`;
- focused RW-D5 tests: 44/44 passed;
- inherited same-protocol tests: 10/10 passed;
- combined local run: 54/54 passed;
- full-chain rank relocation and outer append-closure regressions: passed;
- delayed readiness and real SIGTERM-ignoring process-group regressions: passed;
- GPU submissions for F: zero;
- `main.tex` or paper tables edited by this workstream: no.

If any listed byte changes, retire F and create a new freeze. Do not launch
while the independent-audit STOP is uncleared.
