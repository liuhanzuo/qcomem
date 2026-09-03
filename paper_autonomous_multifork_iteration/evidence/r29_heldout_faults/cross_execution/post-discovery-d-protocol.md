# Post-discovery D protocol (development plan; not preregistered or executed)

## Status and naming boundary

This document was authored after seeing attempt C.  It is a development
protocol for a repaired system version, not an amendment to C and not a
preregistration.  Attempt C is immutable and must not be overwritten or rerun.
H01--H03 are known development cases and may not be described as held out in D.

No D execution has occurred.  The only authorized QS trial, 1898483, is
terminal, and this protocol does not authorize creating another resource.

## D1: implementation repair, with the audit rule unchanged

Add a fault-ID-blind transition helper to the resident runtime.  Immediately
before the first cached single-token call for one borrowed-base request, it
must:

1. identify the 30 configured Qwen3.5 GDN convolution states;
2. require that each still exactly aliases the persistent coordinate expected
   by the borrowed-base policy;
3. clone and rebind only the selected request's convolution states;
4. require identical shape, dtype, device, tensor bytes, and content digest;
5. require byte-range disjointness from the persistent base and every peer;
6. leave the 30 recurrent states borrowed until the existing functional
   recurrent update rebinds them; and
7. become an audited no-op after that request's convolution states are private.

Do not weaken `gdn_completed_binding_rebound`, change any storage-overlap
definition, suppress a clean failure, alter semantic estimators, or specialize
the repair by fault ID.  Freeze the repair source, unit tests, exact
Transformers source identity, model/data ledgers, and clean runner before any
clean GPU output is inspected.

## D2: independent clean regression gate

An agent independent of the repair implementation must freeze and score a
clean-only regression before any new fault suite is authored.  At minimum it
must run the repaired borrowed-base path with two simultaneously resident
requests and a separately rebuilt materialized-state control from identical
frozen inputs.

Required cells:

- one-token transition of request 0 while request 1 remains live and
  unadvanced;
- then one-token transition of request 1 while request 0 remains live;
- eight consecutive one-token decode steps per request, to verify that the
  first transition privatizes state once and later steps do not clone or leak;
- the reverse request order in a separately rebuilt cache, to rule out an
  order-dependent clean result.

Mandatory clean acceptance rules, fixed before execution:

- At borrowed setup, all 60 tensors per request exactly alias the persistent
  coordinate, as required by the original policy.
- Immediately before the first single-token kernel for a request, its 30
  convolution tensors are content-exact and byte-range-disjoint from base and
  all peers; the unadvanced peer remains an exact base alias.
- Immediately after the call, all 60 completed-request tensors are disjoint
  from base and every peer.  The peer and persistent binding/content manifests
  remain unchanged.
- After both requests complete, their 60-tensor sets are pairwise disjoint and
  both are disjoint from the persistent base.
- At every step, greedy token and canonical full-vocabulary FP32 logits are
  byte-exact versus the separately rebuilt materialized-state control.
- Terminal logical KV digests and all final GDN content digests are exact versus
  control.
- KV construction binding, append horizon, source immutability, allocator
  cleanup, backend restoration, and all existing GDN receipts pass without
  suppression.
- The transition receipt reports exactly 30 convolution clones on the first
  one-token transition of a borrowed request and zero clones thereafter.

Any clean mismatch blocks D3.  The raw clean failure must be retained as
development evidence; no fault run may be used to compensate for it.

## D3: independently authored new realistic-pattern suite

Only after D2 passes may a new isolated fault-author agent receive the frozen
post-fix system specification.  That agent must not receive C outputs, H01--H03
definitions, expected ForkAudit gates, or candidate detector outcomes.  Its
prompt/input closure and source hashes must be archived.  The author must draw
new consequence-level patterns from independently selected upstream serving
histories and freeze them before the executor sees candidate outputs.

The new fault IDs must be fresh (for example, a `Dxx` namespace), and every
intervention must be demonstrably different in mutation coordinate and action
sequence from H01--H03.  This document deliberately does not propose the new
patterns because its author has already seen H01--H03 and is therefore not an
eligible independent fault author.

Accurate terminology after such a freeze is “prospective, independently
authored realistic-pattern faults on the post-fix system.”  They are held out
from post-fix candidate outputs, but they do not erase the post-discovery nature
of the system repair.  H01--H03 remain excluded from D's formal evidence.

## D4: fresh execution and detached replay

The executor must use a new code directory, a new run directory, and a new run
ID.  It must never reuse the C paths.  Before execution it must freeze:

- fault suite bytes and canonical digest;
- rank/fault assignment and fresh-case lane order;
- model, data, query coordinate, runtime, GPU assignment, and action horizon;
- repair source, runner, tests, launcher, and scorer hashes;
- operational-validity rules, tri-state rules, semantic estimators, receipt
  order, and no-pooled-rate policy.

Each rank must preserve raw JSON, canonical FP32 sidecars, pointer-free storage
snapshots, request/base/peer relation rows, mutation/restoration receipts,
terminal logs, and a raw-artifact SHA-256 ledger before aggregation.

The formal scorer and replay package must be candidate-import-free: they may
read only frozen schemas/rules and raw artifacts, and must import no candidate
runtime, repair, mutation, or witness module.  Detached replay must independently
recompute:

- all file sizes and hashes;
- sidecar dtype, shape, byte count, token, argmax, exactness, maximum absolute
  difference, and relative L2;
- normalized-storage equality, byte intervals, exact aliases, overlaps, and
  request/base/peer disjointness;
- frozen action order and semantic horizon;
- mutation delta and byte-exact restoration; and
- per-fault tri-state classification and fail-closed operational validity.

Negative, escaped, wrong-gate, cleanup-failure, and operational-invalid results
must remain in the internal raw package.  A failed D run is not imported into
the manuscript.  Even a valid D run supports only per-pattern outcomes on the
one fixed post-fix model/runtime/hardware configuration; it does not support a
population detection rate, completeness, or cross-runtime generality.

## Execution blocker

At the time of this draft, QS Job 249885 / Trial 1898483 is terminated and its
pod is not live.  No stop, eviction, or deletion was requested by this
executor.  The minimum next step is explicit authority for a fresh GPU resource
after D1 source and D2 clean inputs have been frozen.  Creating a new resource
is outside this document's authority.
