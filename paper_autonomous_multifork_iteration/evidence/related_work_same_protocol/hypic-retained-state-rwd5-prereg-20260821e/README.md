# RW-D5 HYPIC retained-state bytes — frozen preregistration E

This is the affected-only preregistration for filling only the Prefix Cache
and HYPIC `Store (MiB)` cells. It preserves C/D's scientific denominator and
all previously audited receipt gates while repairing D's two lifecycle and
readiness-identity blockers. A, B, C, and D are retired; no earlier output or
audit identity carries forward. This is not a result package. The `STOP` file
requires a fresh independent GREEN audit of this exact freeze before any GPU
submission.

Only 16 GPU cells are authorized: Prefix Cache and HYPIC
`transition_rope_recompute`, each on the same eight frozen Qasper/2WikiMQA
workloads. Full Recompute, CoMem, RR2, GDN, serving controls, and all other
methods remain absent. No GPU work has been submitted for D or E.

## Retired attempts and freezes

Trial `1876986` under freeze C is invalid and supplies no paper evidence. It
stopped before `0/16` raw/store cells completed. Although all eight
`/model_info` endpoints were ready, SGLang scheduler-internal 80-token warmup
continued for roughly 46 seconds. C's one-shot `/server_info` request timed out
after 30 seconds. The failed exit also left eight server process groups
resident at 90,968 MiB per H20-3e; exact PID-file-based `TERM` recovery followed
by a 10-second check returned every GPU to 0 MiB / 0% with no SGLang process.
The structured record is `invalid-attempt-trial-1876986.json`.

Freeze D was never submitted to GPU. It is retired because independent static
audit found that cleanup performed a potentially blocking `wait` before its
bounded poll/KILL phase, and that readiness files were not independently bound
to the exact mode/rank/endpoint cell. Its structured disposition is
`retired-freeze-d.json`.

## E readiness and failure lifecycle

After every rank passes `/model_info`, E polls the exact rank's real
`/server_info` endpoint with individual 3-second requests, a 1-second interval,
and a 300-second total deadline. Every failed poll and the terminal success are
retained. Producer, server receipt, and blind replay bind the exact mode, rank,
derived base URL and endpoint, actual launch PID, ordered attempt sequence, and
frozen polling parameters. A fully re-signed cross-rank readiness exchange is
rejected. The subsequent exact receipt/configuration read retains a 120-second
request timeout. This is evidence-bearing readiness, not a fixed sleep.

The launcher uses Bash errtrace (`set -E`) and idempotent
`EXIT`/`ERR`/`INT`/`TERM` handling. Cleanup collects both tracked PIDs and exact
run-directory PID-file fallbacks, sends process-group `TERM`, performs a bounded
`kill -0` poll, sends `KILL` to survivors, and only then performs final reap.
On failure, `COMPLETED` is removed and `FAILED` is written before cleanup. On
success, cleanup returns before `COMPLETED` is created and traps are disarmed.
A real independent process group that ignores `SIGTERM` is forcibly collected
within the bound and leaves only `FAILED`.

## Unchanged Store denominator

`Store (MiB)` remains the overlap-aware union of physical tensor byte ranges
owned by the frozen document after prime and before measured query. Blind
replay derives ranges from dtype, element size, full shape, stride, storage
offset, pointer-relative identity, and exact target-bound token/slot selection.
Metadata, NVML/process allocation, reservation, and pool capacity are excluded.

Prefix tokens must be the target-document prefix. HYPIC must be ordered exact
segments 0 and 1, concatenating to the target document. Selected KV/Mamba slots
must be outside canonical pre-snapshot free/release domains and terminal replay
must prove those exact old slots transition into complete free domains. All C
integrity, authority, topology, int8/PIC identity, zero-lock, manifest,
canonical-hash, and terminal ownership gates remain in force.

Local validation from frozen E code:

- official HYPIC HEAD `98147c01909004e66d98bcb18b886927d41b0ee5` and cleanliness: passed;
- exact instrumentation-only overlay, apply/reverse, `git diff --check`, and
  patched Python compilation: passed;
- canonical tracked overlay SHA-256:
  `acd14dbf615b64d4c6fea09681e3bca66bd72eccd88b0be775a478663c4486fe`;
- focused RW-D5 tests: 42/42 passed;
- inherited same-protocol tests: 10/10 passed;
- combined local run: 52/52 passed;
- delayed `/server_info` greater than 30 seconds: mock regression passed;
- cross-rank exchanged readiness with all downstream hashes re-signed: blind
  replay rejection passed;
- nonzero server-receipt exit with a `SIGTERM`-ignoring process group: bounded
  KILL/reap, `FAILED=37`, and no `COMPLETED`: passed;
- GPU submissions for E: zero;
- `main.tex` or paper tables edited by this workstream: no.

If any listed byte changes, retire E and create a new freeze. Do not launch
while the independent-audit STOP is uncleared.
