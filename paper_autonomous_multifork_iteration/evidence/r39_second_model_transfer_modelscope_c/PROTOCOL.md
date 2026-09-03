# Frozen protocol: R39 Qwen3.5-0.8B Transformers transfer — official ModelScope C

This protocol is frozen before any R39 GPU output.  There is one formal run,
no smoke, no adaptive retry, no threshold change, and no selective rank rerun.

The original A run failed before model or GPU execution because the official
Hugging Face Xet/CAS connection exhausted its fixed retries.  Mirror B also
failed before model execution when its mandatory LFS redirect target could
not complete a TLS handshake.  Those failed runs and their partial model
directories remain immutable and are not evidence for or against the
scientific hypothesis.  ModelScope C is a new non-overwriting run.  Its only
change is model acquisition provenance: the official Qwen ModelScope repo at
full commit `4d58a7b524cd33ed843d5125be8cd8f0a452d9bf`, whose weight and tokenizer
SHA-256 values exactly equal those frozen for canonical Hugging Face commit
`2fc06364715b967f1860aea9cf38778875588b17`.  A pinned 14-file tree, each local
file, and the read-only model authority are closed before and after the
unchanged eight-rank cell.

## Identity and cell

- Model: `Qwen/Qwen3.5-0.8B`.
- Revision: `2fc06364715b967f1860aea9cf38778875588b17`.
- Evaluated path: text-only language model; vision is out of scope.
- Required geometry: dense `qwen3_5_text`, 24 layers, hidden size 1024, and
  exact repeating layer pattern `(linear, linear, linear, full) x 6`.
- Runtime: the already provisioned Transformers 5.14.1 environment used by
  A4, BF16 weights, `DynamicCache`, one process and one CUDA stream per rank.
- Acquisition: fresh ModelScope-C model root; all A/B partials are forbidden;
  official endpoint, token-free policy, both full commits, cross-source weight
  and tokenizer equivalence, the pinned remote tree, and every local file
  digest are recorded in the model authority and final aggregate.  Every file
  uses an independent resumable temporary and exact size/SHA-256 gate.
- Hardware: exactly eight selected, distinct H20 devices when at least eight
  are available; otherwise fail before a shard is started.
- Inputs: eight distinct A4-preregistered PG-19 books, deterministically
  truncated without reselection to 64 document tokens and two 8-token query
  windows per rank.
- Formal factors: ranks 0--7, `N={1,2}`, depth 7, two greedy steps.

## Arms and execution order

For each `N`, `deep_materialized` independently invokes
`write_lower_replay(document, depth=7)` once per request.  `persistent_q16`
constructs one lower state, encodes its residual and every active cache leaf at
16 bits, and calls `PackedLowerReplayState.fork()` once per request.  The Q16
representation is lossless for the stored BF16/FP32 tensors.  The boundary
residual is immutable and may remain shared; every mutable cache leaf must own
disjoint storage.

All request states remain live.  In each arm the launcher starts requests in
index order, then advances semantic steps in `(step, request_index)` order on
one stream.  A suffix cache is seeded with the stored document boundary before
the query is appended.  There is no batching, scheduler, overlap, or timing
claim.

## Registered semantic relations

Every compared logit is stored as canonical contiguous little-endian CPU
float32 over the full 248,320-token vocabulary.  Detached replay recomputes
SHA-256, finiteness, argmax, max-absolute error, and relative L2 from those raw
bytes.

- Cross-arm: for the same `(rank,N,request,step)`, generated token and FP32
  sidecar bytes must be exactly equal.
- Cross-N: request 0 at `N=1` and request 0 at `N=2` must be exactly equal
  within each arm.
- Reference numeric gate: top-1 equality and relative L2 at most 0.005.
- Manual-wrapper validation gate: top-1 equality and relative L2 at most
  0.001 between the official one-shot wrapper and manual one-shot split.
- Standard-cache validation gate: top-1 equality and relative L2 at most
  0.005 between official one-shot recomputation and independently rebuilt
  official full-model DynamicCache execution.

The standard cached full-model reference may authorize split-path semantic
accuracy only if both validation gates pass for every rank/request/step.  If a
validation gate fails, the run remains a scientifically valid negative but the
standard reference is marked unauthorized.  No alternate oracle is selected.

## Ownership predicates

The persistent Q16 state is content-hashed before and after every formal cell.
At setup, first-query transition, and final state, normalized contiguous tensor
byte ranges must show:

- every mutable request cache disjoint from the persistent packed cache;
- all `N=2` mutable request caches pairwise disjoint;
- deep-materialized request caches pairwise disjoint; and
- the persistent boundary residual unchanged, with sharing permitted only for
  that explicitly immutable tensor.

Empty/non-contiguous authorizing inventories, duplicate paths, malformed
ranges, or vacuous `N=2` comparisons fail closed.

## Frozen targeted controls

Each rank executes four controls on fresh state, never on an arm used for the
clean result:

1. `MUTABLE_CACHE_ALIAS`: matched forks are disjoint; assigning request 1's
   cache object to request 0 must first fail `PRIVATE_MUTABLE_STORAGE`.
2. `PREFIX_CONTENT_MUTATION`: matched content equals its frozen digest; a
   one-bit mutation of a fresh Q16 residual must first fail
   `PERSISTENT_PREFIX_IMMUTABLE` while storage identity stays stable.
3. `POSITION_OFFSET_DRIFT`: matched `current_length` equals document length;
   adding one must first fail `POSITION_CANONICAL`.
4. `DENSE_MASK_ROUTE_RELABEL`: the exact 3:1 route passes; relabeling layer 0
   as full attention must first fail `LAYER_TYPE_MASK_ROUTE`.

All four matched-clean predicates must pass and all four mutants must fail
their registered first predicate.  Escapes and wrong-gate detections are
preserved as valid negative outcomes; no control is retuned.

## Seven ForkAudit targets

1. Frozen source/input/model/runtime/GPU identity: applicable, maximum full.
2. Persistent-prefix immutability: applicable, maximum full.
3. Private mutable ownership: applicable, maximum full.
4. Paged partial-tail copy-before-append: not applicable to DynamicCache.
5. Dispatch provenance: applicable, maximum partial (Python source/class/call
   receipts only; compiled binary/autotune/instruction evidence is absent).
6. Deep-materialized versus persistent-Q16 semantic equivalence: applicable,
   maximum full.
7. Cross-N request-0 consistency: applicable, maximum full.

## Validity, stopping, and claims

Missing ranks, reused GPU UUIDs, source/static drift, wrong model revision or
geometry, mutable model files, pre/terminal model-authority mismatch, malformed
or non-finite sidecars, missing raw records, or detached-replay disagreement
invalidate the run as infrastructure/evidence failure.  A clean predicate,
reference gate, or targeted-control failure after valid completion is a valid
negative scientific result.

A positive run supports only a bounded second-model/second-runtime transfer of
the listed ownership and relational targets on these eight short-input cells.
It does not establish vision-path behavior, tail safety, compiled dispatch,
continuous batching, concurrency, performance, capacity, memory saving,
scheduler behavior, other models, other revisions, or production portability.
