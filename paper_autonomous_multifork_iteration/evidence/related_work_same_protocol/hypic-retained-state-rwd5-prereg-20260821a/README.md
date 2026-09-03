# RW-D5 HYPIC retained-state bytes — frozen prelaunch bundle

This directory freezes the affected-only experiment that can replace `n/r` in
the Prefix Cache and HYPIC `Store (MiB)` cells.  It is not a result package.
The `STOP` file is intentional: an independent audit must clear the code and
launch contract before any GPU submission.

The experiment keeps the official HYPIC checkout clean at commit
`98147c01909004e66d98bcb18b886927d41b0ee5` (SGLang 0.5.14), copies it to a
temporary worktree, and applies the hash-bound read-only overlay in
`code/hypic_retained_state_instrumentation.patch`.  The patch adds two hooks:
one after request/cache release and one after terminal cache flush.  The actual
receipt implementation is frozen separately and copied into the temporary
worktree.  The patch does not alter model forward, cache matching, slot
allocation, or output generation.

Only 16 GPU cells are authorized: Prefix Cache and HYPIC
`transition_rope_recompute`, each on the same eight Qasper/2WikiMQA workloads.
Full Recompute, CoMem, RR2, GDN, vLLM, SGLang controls, and every unaffected
experiment are explicitly absent from the launcher.

The result denominator is not NVML, process memory, `memory_reserved`, or a
pool-capacity delta.  Immediately after the formal prime and before the
measured query, the receipt binds the exact target radix path or HYPIC segment
entries, records their KV and Mamba slots, and maps those slots to live tensor
dtype/shape/stride/element-size/backing-storage byte ranges.  `Store (MiB)` is
the overlap-aware union of full-attention K/V and recurrent/PIC payload ranges.
Index/token/hash metadata is reported separately and excluded.  A standalone
blind replay recomputes all token/segment hashes and byte-range unions from raw
JSON.  `/flush_cache` must then remove the target entries and return every old
KV/Mamba slot to the appropriate free list.

CPU/static status at freeze:

- official checkout commit and cleanliness: passed;
- patch apply check: passed (5 lines in `common.py`, 7 in `scheduler.py`);
- temporary patched-worktree reverse check and Python compilation: passed;
- new receipt/client/replay tests: 8 passed;
- new plus inherited HYPIC protocol tests: 18 passed;
- GPU jobs submitted: zero.

Use `launch-plan.json` and `prelaunch-static-validation.json` for the exact
runtime contract and `SHA256SUMS` for frozen file identity.  If any hash changes,
discard this freeze and create a new preregistration instead of editing outputs.
