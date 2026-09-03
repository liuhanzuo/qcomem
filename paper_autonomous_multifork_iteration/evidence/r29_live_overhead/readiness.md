# Round 29 live ForkAudit overhead readiness

Status: feasible, dependency-closed, hash-consistent, and frozen before any GPU
execution. No candidate overhead result exists.

The experiment reuses the successful frozen `7620...` Qwen3.5 lifecycle
runtime directly. It does not introduce a toy attention path. The baseline is
the existing `Qwen35VllmPagedHitLedger` with optional ForkAudit append/call
capture, KV/GDN ownership witnesses, content hashing, and artifact persistence
disabled. Mandatory Q16 functional-adapter shape/operator checks remain; this
is the only honest meaning of “audit off” on this live stack. The instrumented
arm uses the existing `Q16PagedSequence.append_observer`,
`MultiForkHitLedger.call_observer`, live KV ownership validator, and persistent
and request GDN guards.

## Paired boundary

Each pair rebuilds one persistent 4,033-token PG19 document, Q16 conversion,
and two distinct shared-reuse requests before either timer. Both requests use
the same frozen 16-token query. There is one discarded warmup pair, then five
measured pairs with orders B/I, I/B, B/I, I/B, B/I and preregistered request
slots. The odd pair count and slot imbalance are disclosed rather than hidden.

Work outside both timers is common: model load, token/input validation,
document prefill, Q16 conversion, two-request construction, pair-wide source
KV/GDN immutability validation, and the final float32 full-vocabulary logit
copy used by the semantic oracle. Thus the overhead timing cannot include the
common final-logit D2H copy.

Work inside both timers is arm adapter/ledger construction and registration,
one full-model query step, CUDA synchronization, ten-layer fused-operator
completion verification, backend restoration/unregistration, and a final CUDA
synchronization. The instrumented timer additionally includes opening capture
files; persistent KV before/after hashing; KV and GDN ownership capture and
verification; fifty synchronous tensor captures across ten full-attention
layers; capture-file hashing; and atomic receipt write/hash.

Immediately before each arm, with that pair's persistent cache and both
requests still live, the runner executes GC, releases only free cached CUDA
blocks, synchronizes, resets peak-memory stats, and records allocated/reserved
bytes. Reported incremental peak is `max_memory_allocated - allocated_before`;
absolute peak and before/after counters are retained. The first completed
request remains live for the second arm, so execution order and slot accompany
every raw pair.

## Validity and claim boundary

All warmup and measured pairs require exact `torch.equal` float32 full-vocab
logits and identical generated tokens. The instrumented cell additionally
requires 10 append events, 10 call events, 50 hash-covered binary tensor
records, source immutability, live KV/GDN ownership receipts, and independent
artifact replay. Any semantic/receipt failure invalidates the overhead
estimand but remains preserved. Any positive, negative, or counterintuitive
wall/peak delta is retained when those validity gates pass.

The claim is limited to paired live request-step overhead for this one frozen
Qwen3.5-35B-A3B/vLLM-Q16/H20/PG19 configuration. It does not measure model
load or document prefill, production server latency, continuous batching,
QPS, throughput, capacity, or cross-runtime generality.

## Frozen checks and resources

- preregistration SHA:
  `2114d2cd85bedc1eafa5d1398fd0afd0d57819c0360c3be3f9ec20f1b2878939`;
- executable source-ledger SHA:
  `402c7006e0cb36dc51bab5cf172996c96a44b9c4e48fccb2ed5d4d1c7af387f7`;
- all six focused tests pass, including negative-delta preservation, full fake
  formal replay, 50-record capture replay, and capture/semantic tamper gates;
- all Python sources compile; the launcher passes `bash -n`; CPU/mock preflight
  passes;
- every nonstandard imported implementation is present byte-identically in
  the frozen `7620...` upstream code ledger.

Resource request: one exclusive H20 exposed as logical GPU7, one process, no
server port, no collective, approximately 15 minutes, read-only shared
model/PG19 assets, and private output/Triton/TorchInductor paths. The launcher
requires a fresh `RUN_DIR`; set its asset variables and `R29_GPU_UUID`, then
run `bash gpu/launch_r29_live_overhead_1gpu.sh` from the frozen package.
