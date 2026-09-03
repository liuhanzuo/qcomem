# RW-D5 HYPIC retained-state bytes — frozen preregistration K

K authorizes only the affected Prefix Cache and official HYPIC
`transition_rope_recompute` Store-byte measurement: 2 modes × the same 8
frozen Qasper/2WikiMQA workloads = 16 cells. Full Recompute, CoMem, RR2, GDN,
serving controls, and all other methods must not run. This is a stopped
preregistration, not a result package; GPU execution requires fresh independent
GREEN review of this exact manifest.

## Why F was invalid and what K changes

Formal F passed static review but was invalid before `0/16` raw/store results.
The official live `MambaPool` stores convolution state in BF16 and temporal
state in FP32, while F imposed one BF16 dtype on both. All Prefix schedulers
therefore failed closed with `ReceiptError: temporal dtype`. This is a receipt
contract defect, not a scientific negative result.

K changes only the component dtype authority and the evidence that binds it:

- full-attention K/V: BF16, 2 bytes;
- convolution and HYPIC convolution tails: BF16, 2 bytes;
- temporal and HYPIC transition state: FP32, 4 bytes.

All other scientific choices remain unchanged: official commit, model, TP=1,
eight workloads, prime/measured prompts, seam, cache-hit coverage, selected
document-owned slots, unique overlap-aware physical-byte union, metadata
exclusion, terminal ownership removal, and blind replay.

## Live component authority

The complete J debug-only GPU0 mirror is included under
`live-debug-j-trial-1879097/`. Its 19-file local ledger SHA-256 is
`59530c0c8bc10cedbf4b0bde51d04e5490adeaf369e8738d9df363fc83941026`;
all 19 entries pass and its copied remote absolute artifact ledger maps exactly
to the local files. It has `COMPLETED_DEBUG_ONLY`, no failure marker, no formal
receipt, and terminal GPU0 at 0 MiB / 0%.

Observed live inventories:

- Prefix (`MambaRadixCache`/`MambaPool`): `conv[0]` BF16/2B with shape
  `[30,364,8192,3]`; temporal FP32/4B with shape
  `[30,364,32,128,128]`.
- HYPIC (`PICache`/`MambaPool`): conv and tail BF16/2B with shape
  `[30,184,8192,3]`; temporal and transition FP32/4B with shape
  `[30,184,32,128,128]`.

All tensors are C-contiguous on `cuda:0`; the environment is explicitly
`SGLANG_MAMBA_CONV_DTYPE=bfloat16` and
`SGLANG_MAMBA_SSM_DTYPE=float32`; the official commit is exact. Raw receipt
hashes are `83dbc66e...` and `017ee1c6...`; independent validation receipt
hashes are `51b0c17e...` and `0bfd3be1...`.

The K static builder does not merely cite these values. Before output it checks
the immutable local mirror manifest, replays the remote-to-local hash mapping,
requires exact file/terminal sets, and independently validates every raw,
validation, run-summary, and target hash plus cache class, environment,
component keys, dtype, element size, rank, recurrent-layer/slot axes, shape,
stride, contiguity, and HYPIC tail/transition topology. The resulting binding
is embedded into the model storage contract, preregistration, producer
receipt, and blind replay. Terminal static verification recomputes it again.

## Preserved integrity and lifecycle gates

The producer still derives ranges from full tensor metadata and exact selected
slots; blind replay distrusts producer byte ranges/totals and recomputes them.
Prefix requires its exact KV/Mamba ownership and disabled int8 checkpoint pool;
HYPIC requires its exact two segment entries, shared MambaPool identity,
transition, and tails. Pre-free allocator domains prove selected slots were
allocated; terminal domains prove the same slots were returned. Lock refs must
be zero. Scheduler child lineage, exact server cell/endpoint, external expected
cell, code/model/data/environment hashes, and instrumentation-only overlay all
remain closed.

Readiness remains bounded evidence-bearing `/server_info` polling. Cleanup
remains TERM → bounded liveness polling → KILL survivors → final reap; failure
removes `COMPLETED` and writes `FAILED`, and success creates `COMPLETED` only
after cleanup.

Frozen K validation:

- focused RW-D5 tests: 63/63;
- inherited same-protocol tests: 10/10; combined 73/73;
- exact J mirror validation and immutable-replacement rejection pass;
- mixed component producer/replay and toy-byte totals pass;
- Python compilation and both Bash launchers' syntax checks pass;
- GPU submissions from K before STOP: zero;
- `main.tex` and paper tables changed by this workstream: no.

If any listed byte changes, retire K and refreeze. Never use F or the debug-only
J run as a paper Store number.
