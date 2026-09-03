# RW-D5 HYPIC retained-state bytes — frozen preregistration C

This is a new affected-only preregistration for filling only the Prefix Cache
and HYPIC `Store (MiB)` cells.  It supersedes freezes A and B; no prior frozen
identity or audit outcome carries forward.  It is not a result package.  The
separate `STOP` file requires a fresh independent GREEN audit of this exact
freeze before any GPU submission.

The experiment keeps the official HYPIC checkout clean at full commit
`98147c01909004e66d98bcb18b886927d41b0ee5` (SGLang 0.5.14), copies it to a
temporary directory, and admits exactly one instrumentation-only overlay:

- modified tracked files: `python/sglang/srt/mem_cache/common.py` and
  `python/sglang/srt/managers/scheduler.py`;
- sole untracked file: `python/sglang/srt/retained_state_receipt.py`; and
- no other tracked or untracked path.

Only 16 GPU cells are authorized: Prefix Cache and HYPIC
`transition_rope_recompute`, each on the same eight frozen Qasper/2WikiMQA
workloads.  Full Recompute, CoMem, RR2, GDN, serving controls, and all other
methods are absent.  No GPU work has been submitted for this freeze.

## Store denominator and ownership transition

`Store (MiB)` is the overlap-aware union of physical tensor byte ranges owned
by the frozen document after prime and before the measured query.  It includes
the exact full-attention K/V slots and required recurrent/PIC state selected by
the owned radix/PIC entries.  Metadata is measured separately and excluded.
NVML, process allocation, allocator reservation, and pool capacity are not
accepted substitutes.

Blind replay derives every range from dtype, element size, complete shape,
stride, storage offset, pointer-relative storage identity, and independently
bound token/slot selection; producer offsets and totals are assertions only.
It derives layer/component cardinalities from the frozen model configuration.
Prefix tokens must equal the exact prefix of the target document.  HYPIC must
contain exactly ordered segment entries 0 and 1, each equal to its target
segment and concatenating to the target document.  Cache-hit counts are
derived independently from target semantics and the preregistered seam.

At the pre-query snapshot, duplicate-free canonical KV free/release and Mamba
free domains are recorded.  Every selected KV and Mamba slot must belong to
the complementary allocated domain.  Terminal replay then proves that these
same old slots transition into exact, duplicate-free full free domains after
target-entry removal.  This prevents an already-free stale cache entry from
vacuously satisfying the terminal check.

Prefix Cache additionally requires int8 Mamba checkpointing disabled, no int8
checkpoint pool, and PIC/cache Mamba-pool object identity.  HYPIC requires
nonempty transition and `conv_tails` components.  All post-prime target lock
references must be zero.

## Authority and frozen code

The launcher accepts the independently recorded SHA-256 of `SHA256SUMS`, then
checks every listed file before creating run output or starting a server.
Blind replay additionally canonicalizes manifest paths (`./code/x` to
`code/x`); absolute, parent-traversal, empty, and duplicate-canonical paths
fail closed.  The same manifest identity binds static, server,
scheduler-worker, target, raw, store, terminal, and blind-replay receipts.

Scheduler-worker PID, PPID, command line, environment, ancestry, and actual
front-end lineage are bound.  RW-D5 process and server-configuration hashes
use one frozen JSON-plus-newline canonicalization in runner, receipt producer,
and replay; the older formal helper's compact convention is not modified or
silently mixed into this authority family.

Local validation from the frozen C code:

- official checkout full HEAD and clean status: passed;
- patch apply, exact overlay status, `git diff --check`, reverse apply, and
  patched Python compilation: passed;
- canonical tracked overlay SHA-256:
  `acd14dbf615b64d4c6fea09681e3bca66bd72eccd88b0be775a478663c4486fe`;
- focused producer/replay/static/launcher tests: 38/38 passed;
- focused plus inherited HYPIC protocol tests: 48/48 passed;
- GPU submissions: zero;
- `main.tex` or paper tables edited by this workstream: no.

If any listed byte changes, retire C and create a new freeze.  Do not edit an
output into conformance and do not launch while the independent-audit STOP is
uncleared.
