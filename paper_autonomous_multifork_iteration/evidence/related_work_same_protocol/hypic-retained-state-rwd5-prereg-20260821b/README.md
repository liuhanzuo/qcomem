# RW-D5 HYPIC retained-state bytes — audit-remediated frozen bundle B

This directory is a new affected-only preregistration for filling only the
Prefix Cache and HYPIC `Store (MiB)` cells.  It supersedes freeze A; none of
freeze A's identities or audit status carries forward.  This is not a result
package.  The `STOP` file remains in force until a fresh independent audit
explicitly clears this exact manifest.

The experiment keeps the official HYPIC checkout clean at commit
`98147c01909004e66d98bcb18b886927d41b0ee5` (SGLang 0.5.14), copies it to a
temporary directory, and admits exactly one instrumentation overlay:

- modified tracked files: `common.py` and `scheduler.py`;
- sole untracked file: `retained_state_receipt.py`; and
- no other tracked or untracked path.

The frozen launcher validates an externally supplied SHA-256 of this bundle's
`SHA256SUMS`, validates every listed file before creating a run directory, and
binds the same manifest identity into the static preregistration, server,
scheduler-worker, target, raw, store, terminal, and blind-replay receipts.
The scheduler receipt records its own PID, PPID, command line, environment,
and front-end lineage; replay verifies that the front end actually owns the
scheduler child.

Only 16 GPU cells are authorized: Prefix Cache and HYPIC
`transition_rope_recompute`, each on the same eight frozen Qasper/2WikiMQA
workloads.  Full Recompute, CoMem, RR2, GDN, serving controls, and all other
methods are absent.  No GPU cell has been submitted for this freeze.

## Frozen Store denominator

`Store (MiB)` is the overlap-aware union of physical tensor byte ranges owned
by the frozen document after prime and before the measured query.  It includes
the exact full-attention K/V slots and required recurrent/PIC state selected by
the owned radix/PIC entries.  Metadata is measured separately and excluded.
NVML, process allocation, allocator reservation, and pool capacity are not
accepted substitutes.

Blind replay does not trust producer offsets or totals.  It derives each
expected byte range from frozen dtype, element size, complete shape, stride,
storage offset, pointer-relative storage identity, and the owned token/slot
selection.  It independently checks exact model-derived layer/component
cardinalities and layer-by-slot coverage.  Prefix Cache requires int8 Mamba
checkpointing to be disabled, `int8_ckpt_pool` absent, and PIC/cache Mamba-pool
object identity.  HYPIC requires nonempty transition and `conv_tails` state.
All post-prime target lock references must be zero.

Terminal replay requires duplicate-free free-list domains equal exactly to
`1..pool_size`, target entries removed, every old owned slot returned, and
closed whole-pool ownership counts.  Any missing authority field, structural
component, slot, or exact frozen hash fails closed.

## Local freeze validation

- official checkout HEAD and cleanliness: passed;
- patch apply, reverse apply, `git diff --check`, exact overlay status, and
  patched Python compilation: passed;
- canonical tracked overlay SHA-256:
  `acd14dbf615b64d4c6fea09681e3bca66bd72eccd88b0be775a478663c4486fe`;
- new producer/replay/static/launcher tests: 23/23 passed;
- new plus inherited HYPIC protocol tests: 33/33 passed;
- GPU jobs submitted: zero;
- `main.tex` or paper tables edited by this workstream: no.

Use `launch-plan.json`, `prelaunch-static-validation.json`, and `SHA256SUMS`
as one frozen object.  If any listed byte changes, retire this freeze and make
a new one.  Do not edit an output into conformance and do not launch while the
independent-audit STOP remains uncleared.
