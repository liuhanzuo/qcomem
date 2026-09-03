# RW-D5 HYPIC retained-state bytes — frozen preregistration D

This is the affected-only preregistration for filling only the Prefix Cache
and HYPIC `Store (MiB)` cells.  It preserves C's scientific denominator and
authority checks while repairing two pre-evidence operational failures.  A,
B, and C are retired; no earlier output or audit identity carries forward.
This is not a result package.  The `STOP` file requires a fresh independent
GREEN audit of this exact freeze before any GPU submission.

Only 16 GPU cells are authorized: Prefix Cache and HYPIC
`transition_rope_recompute`, each on the same eight frozen Qasper/2WikiMQA
workloads.  Full Recompute, CoMem, RR2, GDN, serving controls, and all other
methods remain absent.  No GPU work has been submitted for D.

## Invalid C attempt

Trial `1876986` is invalid and supplies no paper evidence.  It stopped before
`0/16` raw/store cells completed.  Although all eight `/model_info` endpoints
were ready, SGLang scheduler-internal 80-token warmup continued for roughly 46
seconds.  C's one-shot `/server_info` request timed out after 30 seconds.  The
failed exit also left eight server process groups resident at 90,968 MiB per
H20-3e; exact PID-file-based `TERM` recovery followed by a 10-second check
returned every GPU to 0 MiB / 0% with no SGLang process.  The structured record
is `invalid-attempt-trial-1876986.json`.

## D readiness and failure lifecycle

After every rank passes `/model_info`, D polls the real `/server_info` endpoint
with individual 3-second requests, a 1-second interval, and a 300-second total
deadline.  Every failed poll and the terminal success are retained in a
readiness receipt.  The server-receipt stage cannot start before all readiness
receipts exist; it binds the exact readiness hash and still performs the full
server-configuration check.  Its subsequent endpoint read has a 120-second
request timeout.  This is evidence-bearing readiness, not a fixed sleep.

The launcher uses Bash errtrace (`set -E`) and idempotent
`EXIT`/`ERR`/`INT`/`TERM` handling.  Cleanup collects both live tracked PIDs and
exact run-directory PID-file fallbacks, signals process groups, reaps children,
and escalates only after a bounded wait.  Every unsuccessful terminal path
removes `COMPLETED` and writes `FAILED`; the success path cleans remaining
servers and explicitly disarms the traps.

## Unchanged Store denominator

`Store (MiB)` remains the overlap-aware union of physical tensor byte ranges
owned by the frozen document after prime and before measured query.  Blind
replay derives ranges from dtype, element size, full shape, stride, storage
offset, pointer-relative identity, and exact target-bound token/slot selection.
Metadata, NVML/process allocation, reservation, and pool capacity are excluded.

Prefix tokens must be the target-document prefix.  HYPIC must be ordered exact
segments 0 and 1, concatenating to the target document.  Selected KV/Mamba
slots must be outside canonical pre-snapshot free/release domains and terminal
replay must prove those exact old slots transition into complete free domains.
All C integrity, authority, topology, int8/PIC identity, zero-lock, manifest,
canonical-hash, and terminal ownership gates remain in force.

Local validation from frozen D code:

- official HYPIC HEAD `98147c01909004e66d98bcb18b886927d41b0ee5` and cleanliness: passed;
- exact instrumentation-only overlay, apply/reverse, `git diff --check`, and
  patched Python compilation: passed;
- canonical tracked overlay SHA-256:
  `acd14dbf615b64d4c6fea09681e3bca66bd72eccd88b0be775a478663c4486fe`;
- focused RW-D5 tests: 40/40 passed;
- inherited same-protocol tests: 10/10 passed;
- combined local run: 50/50 passed;
- delayed `/server_info` greater than 30 seconds: mock regression passed;
- nonzero server-receipt exit reaped all mock process groups and produced
  `FAILED` without `COMPLETED`: subprocess regression passed;
- GPU submissions for D: zero;
- `main.tex` or paper tables edited by this workstream: no.

If any listed byte changes, retire D and create a new freeze.  Do not launch
while the independent-audit STOP is uncleared.
