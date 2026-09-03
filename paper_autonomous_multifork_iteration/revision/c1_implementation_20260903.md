# C1 implementation: one packed entry shared across N>1 requests, audited

Date: 2026-09-03. Status: **code only, nothing executed on a GPU.** No number
in this document is a measurement. Everything below is what was built, how to
run it, and what it will and will not establish once it runs.

C1 is the meta-review's single Contribution-lifting action: *"One execution in
which the packed Q4/Q4/Q8 entry at j=7 is shared across N > 1 resident requests
under the Sec. 4.3 ownership discipline — shared immutable dequantized document
tensors, request-local mutable state, copy-on-write tails, recurrent rebinding —
with ForkAudit coverage and verdicts, Store, F1 and allocator endpoints all
reported from that single run."*

---

## 1. The gap this closes, restated precisely

Two things are true of the submitted paper and each undoes the other.

* The Read path behind Tables 1 and 2 is `PackedLowerReplayState.fork` in
  `gpu/qcomem_torch.py`. It calls `PackedResidual.dequantize()` and
  `PackedCache.dequantize()` and returns a fresh, wholly private
  `LowerReplayState`. Sec. 4.3 says so in the paper's own words: it "shares
  nothing and exercises neither borrowing nor copy-on-write."
* Every ForkAudit verdict comes from vLLM plus Transformers on a full-prefix
  BF16 KV configuration with no split depth and no quantization.

So the composed system — quantize a depth-split hybrid entry **and** share it
safely across concurrent requests — has no end-to-end evidence. This deliverable
builds a Read path where the two halves coexist, and instantiates the contract
on that path.

---

## 2. Files

### New (all additive; nothing existing was edited)

| Path | Torch? | Purpose |
|---|---|---|
| `gpu/qcomem_multifork_accounting.py` | no | Torch-free bookkeeping: the 10-target contract table, mandatory-slot coverage, ownership byte-range algebra, transient working-set arithmetic, token-trace comparison, row validation, shard summarization. |
| `gpu/qcomem_shared_packed_fork.py` | yes | The `shared-packed-view` fork mode, the `BorrowedPrefixKVLayer`, the registered-transition rebind, and the N-request interleaved driver for both the Q-CoMem and full-prefix arms. |
| `gpu/qcomem_shared_packed_forkaudit.py` | yes | ForkAudit on that path: digests, receipts, the ten predicates, the audited-run wrapper, and the per-rank preflight gate. |
| `gpu/run_shared_packed_multifork.py` | yes | The GPU runner (`--gate-only` supported). |
| `gpu/aggregate_shared_packed_multifork.py` | no | Torch-free aggregator and blind contract replay. |
| `gpu/launch_shared_packed_multifork_8gpu.sh` | — | 8-GPU launcher on the `CODE_DIR / MODEL_DIR / DATA_FILE / RUN_DIR / ENV_DIR` contract, with `GATE_ONLY=1`. |
| `gpu/test_qcomem_multifork_accounting.py` | no | 72 torch-free tests. |
| `gpu/test_aggregate_shared_packed_multifork.py` | no | 31 torch-free tests. |
| `gpu/test_qcomem_shared_packed_fork.py` | torch-gated | 71 tests; skip on the laptop. |
| `paper_autonomous_multifork_iteration/revision/c1_implementation_20260903.md` | — | this document. |

### Read but **not modified**

`gpu/qcomem_torch.py`, `gpu/qcomem_paged.py`, `gpu/qcomem_deployment.py`,
`gpu/qcomem_deployment_arms.py`, `gpu/qcomem_eq3_accounting.py`,
`gpu/qcomem_paged_attention.py`, `gpu/run_deployment_bench.py`,
`gpu/run_deployment_length_sweep.py`,
`gpu/launch_deployment_length_sweep_8gpu.sh`,
`gpu/qcomem_transformers_forkaudit_transfer.py`.

`PackedCache.nbytes`, `cache_nbytes`, `tensor_nbytes`, `capacity_estimate`,
`run_incremental_generation`'s three published modes, `run_exactness_gate`,
`run_cow_vs_deep_clone_gate` and `run_dense_semantics_gate` are all
byte-for-byte as published. Their file mtimes are unchanged by this work. The
new code **imports and calls** the published Read path; it never reimplements
or edits it.

### Reused rather than reinvented

`qcomem_paged.analyze_cache_for_cow` (fail-closed cache classification),
`qcomem_paged._safe_dynamic_cow_update` (the audited read-and-rebind append),
`qcomem_paged.SharedTensorRecord` (storage pointer + version counter + 16-point
sample guard), `qcomem_paged._iter_tensors` / `_storage_key`, and the whole
ownership field vocabulary (`fork_strategy_requested` /
`fork_strategy_effective` / `fallback_reason`, `initial_shared_nbytes` /
`initial_private_nbytes`, `memory_breakdown()`, `verify_shared_immutable()`,
`deployment_memory_components()`). The seven-target/coverage-versus-verdict
shape is copied from `qcomem_transformers_forkaudit_transfer.TARGET_CONTRACT`
and `build_target_rows`, deliberately re-expressed torch-free so an archived
shard replays without a GPU stack.

---

## 3. What the new modes do

### 3.1 `share_mode`

| Mode | Behaviour |
|---|---|
| `private-materialize` | Exactly the published path. `PackedLowerReplayState.fork` called unchanged, once per request. Retained so Tables 1/2 stay reproducible, and so the shared mode has a reference it must equal token for token. |
| `shared-packed-view` | The entry is dequantized **once** into one `LowerReplayState` view. Every request forks that view. Immutable document tensors are shared by reference; mutable state is request-local. |

The entry **falls back to private materialization with a recorded reason** if
`analyze_cache_for_cow` rejects the cache (any leaf whose mutation semantics it
cannot classify). It never shares a cache it does not understand.

### 3.2 `tail_policy` — how the attention prefix is combined with the tail

| Policy | Behaviour | Sharing window |
|---|---|---|
| `borrowed-prefix` (default) | Each active attention layer becomes a `BorrowedPrefixKVLayer` that retains **only** the tokens this request appended and returns, per call, a transient `torch.cat([borrowed_prefix, private_tail], dim=-2)`. The attention module receives exactly the single contiguous tensor a stock `DynamicLayer` would have returned, so the arithmetic is unchanged; the cache simply does not retain it. | The whole request lifetime. |
| `materialized-tail` | The frozen `_safe_dynamic_cow_update`: read the shared prefix, bind a newly concatenated private tensor on first append. | Fork to first append only, zero thereafter. |

`materialized-tail` is the conservative fallback. It touches no cache-layer
interface beyond `update`, so if `BorrowedPrefixKVLayer` fails its preflight on
a given Transformers build, `TAIL_POLICY=materialized-tail` still produces a
publishable (weaker) result.

The audit records which window it evaluated aliasing at, and the emitted
contract row's `scope_note` says so in words. Under `materialized-tail` the row
reads *"this target does not establish steady-state sharing."*

### 3.3 `rebind_policy` — when the mutable base becomes private

| Policy | Behaviour |
|---|---|
| `transition` (default) | The GatedDeltaNet convolution and recurrent buffers are **borrowed read-only** at fork and rebound to private storage at the registered transition: the `rebind_mutable_state()` call the driver makes immediately before the request's first `continue_lower_replay`. This is the discipline Sec. 4.3 describes. |
| `setup` | The mutable base is cloned at fork time (what the existing paged COW staging path does). |

A second `rebind_mutable_state()` raises. Under `transition`, the setup capture
shows every request with `private_nbytes == 0` (everything borrowed) and the
transition capture shows the rebound private storages — that transition is
visible in the receipt, not asserted.

### 3.4 The N-request driver

`run_shared_packed_multifork` creates all N forks **before** any of them
executes, keeps every request's suffix cache live for the whole run, and
interleaves decode round-robin on one CUDA stream. It invokes a `capture`
callback at three points — `setup` (all forks live, none transitioned),
`transition` (all rebinds and query prefills done), `final` (after decode) —
which is the only way the borrow window is observable, because it closes as
soon as the first request transitions.

`run_full_prefix_multifork` runs the identical protocol on
`FullPrefixState.fork` (deep clone, nothing shared) so the working set is
measured for **both** methods.

---

## 4. The contract: 7 + 3 targets, coverage separate from verdict

| # | Target | Predicate | Applicability | Max status |
|---|---|---|---|---|
| 1 | `frozen_identity` | `FROZEN_ENTRY_POLICY_AND_INPUT_BINDINGS` | applicable | full |
| 2 | `prefix_immutability` | `PERSISTENT_PREFIX_CONTENT_UNCHANGED` | applicable | full |
| 3 | `private_ownership` | `ALL_MUTABLE_CACHE_STORAGE_PAIRWISE_DISJOINT` | applicable | full |
| 4 | `tail_safe_append` | `SHARED_PREFIX_NOT_WRITTEN_ON_APPEND` | applicable | **partial** |
| 5 | `dispatch_provenance` | `BOUNDED_HOST_SIDE_CALL_PROVENANCE` | applicable | **partial** |
| 6 | `cross_arm_equivalence` | `SHARED_FORK_EQUALS_PRIVATE_MATERIALIZATION` | applicable | full |
| 7 | `cross_n_prefix_consistency` | `FIRST_REQUEST_PREFIX_INVARIANT_ACROSS_N` | applicable | full |
| 8 | `dequantized_view_immutability` | `SHARED_DEQUANTIZED_VIEW_CONTENT_UNCHANGED` | applicable | full |
| 9 | `residual_chunk_binding` | `DOCUMENT_AND_QUERY_RESIDUAL_CHUNKS_BOUND_DISTINCTLY` | applicable | full |
| 10 | `packed_entry_lifetime` | `PACKED_ENTRY_CONTENT_AND_REFERENCE_LIFETIME` | applicable | full |

Targets 8–10 are the three the manuscript names as untested. The task statement
named two (dequantized-view immutability, residual-chunk binding); the
meta-review's verification criterion for C1 requires three ("all seven targets
plus the three packed-entry obligations"), so packed-entry lifetime is
implemented as well. **It is the weakest of the three** — see §8.

Target 4 changes applicability relative to the earlier Transformers transfer,
which marked it `not_applicable` because a `DynamicCache` has no fixed-size
paged partial tail. That is still true. What is new is a genuinely *shared*
attention prefix with an append that must not write it, which is a real
copy-before-append obligation. It is therefore `applicable` but capped at
`partial`, with the exact missingness printed on the row: *"no fixed-size page
granularity; the tail is a whole-tensor concatenation"* and *"no partial-page
copy-before-append event exists in a Transformers DynamicCache."*

### Coverage versus verdict

Every target declares mandatory receipt slots (`MANDATORY_SLOTS`). A slot must
be present, unique, bound to the live object its receipt names, and unmodified.
`build_multifork_target_rows` enforces:

* incomplete coverage ⇒ `status="open"`, `predicate_passed=None`. **A missing
  or unbound mandatory receipt cannot produce a pass, even when the predicate
  is `True`.** Tested (`test_missing_receipt_cannot_pass_even_with_a_true_predicate`,
  `test_unbound_receipt_cannot_pass`).
* complete coverage with a missing or non-boolean predicate **raises**: a
  covered target with no verdict is an evidence defect, not a result.
* the aggregator re-derives every status from the row's own coverage and
  predicate and reports drift as a defect, so a producer that inflated a status
  is caught.

### Non-vacuity

Each predicate carries a `non_vacuous` flag and refuses to pass when the
evidence is empty. Concretely: `private_ownership` needs at least two requests
and therefore at least one pairwise comparison; `dequantized_view_immutability`
needs ≥ 2 requests *and* every request to hold non-zero aliasing bytes at the
sharing window; `residual_chunk_binding` needs ≥ 2 requests and exactly one
binding event each, all naming the same document-chunk storage;
`cross_n_prefix_consistency` needs ≥ 2 fanouts; `tail_safe_append` needs at
least one append event per request. An N=1 run therefore reports four open
targets rather than a green contract — this is tested.

---

## 5. Transient working set, both arms

Emitted as first-class row fields by `working_set_row`, for every arm at every
fanout — not as diagnostics:

* `shared_dequantized_view_nbytes` — the one view (0 for the private and
  full-prefix arms, which have nothing to share).
* `per_request_materialized_nbytes` / `transient_materialized_nbytes_total` /
  `_max` / `_mean` — what each request brings into existence for itself.
* `peak_transient_allocation_nbytes` — `torch.cuda.max_memory_allocated()`
  minus the pre-fork baseline.
* `steady_state_resident_nbytes` and `per_request_steady_resident_nbytes`.
* `resident_model = {intercept_nbytes, slope_nbytes_per_request}` — the affine
  fit `intercept + slope·N`, with intercept = retained entry + one shared view
  and slope = mean per-request steady resident state.
* `transient_concat_peak_nbytes_max` — under `borrowed-prefix`, the largest
  prefix+tail concatenation handed to attention. This is the price of keeping
  the prefix shared and it is measured, not assumed away.
* `resident_crossover_vs_full_prefix` — the smallest integer N at which the
  Q-CoMem arm's fitted resident line exceeds full prefix's, or `null`. This is
  the arithmetic the meta-review asked for in TS-02, answered from measured
  slopes instead of asserted.

This is exactly what falsifies Eq. 1's premise that `M_active` is
method-independent, and it is now measured symmetrically: the full-prefix arm's
per-request deep clone is charged the same way the Read path's dequantization is.

---

## 6. Semantic equivalence — the experiment's own correctness gate

For every request, the N>1 shared run must emit token-for-token what the
**published** N=1 private-materialization path emits on the same document and
the same query. The reference comes from
`qcomem_deployment.run_incremental_generation` called unchanged, once per
query, through `published_private_reference_traces`.

`compare_token_traces` records, per request: identity, the first divergence
step, both full token sequences, and whether the request was present in each
trace at all (a missing request is a discrepancy, not a skipped comparison).
Discrepancies are surfaced in the row, in the gate, in target 6's verdict, and
in the aggregate's `semantic_equivalence.discrepancies` — never dropped.

The `qcomem-private-materialize` arm at N>1 is compared the same way and is
also gating. The `full-prefix` arm is compared and reported as a **diagnostic
that never gates** — see §7.

---

## 7. The preflight gate, and what it deliberately does not assert

`run_shared_packed_multifork_gate` is the `GATE_ONLY=1` per-rank preflight. It
builds a 256-token entry, forks it twice, decodes 4 tokens, and passes **iff**:

1. the shared mode took effect (one view materialized, no fallback);
2. sharing is non-vacuous — ≥ 2 requests, and at the tail policy's own sharing
   window every request holds non-zero shared bytes **and** non-zero private
   bytes;
3. the N>1 shared traces are token-identical, per request, to the published N=1
   private path;
4. every applicable contract target has complete coverage and a passing
   predicate.

It **does not** require agreement with the full-prefix arm. That comparison
crosses the document/query chunk boundary the Qwen3.5 GatedDeltaNet and
convolution states are sensitive to; the A2 re-analysis measured dense and
full-prefix diverging on 6 of 60 archived items. Requiring it would reproduce
the 2026-09-03 `run_dense_semantics_gate` defect that silently discarded an
eighth of a registered cohort. The full-prefix comparison is executed (so its
working-set fields exist) and recorded as `full_prefix_token_agreement`, a
diagnostic.

Every gate and validator here asserts exactly what its docstring claims:

* `evaluate_private_ownership`'s docstring says it evaluates the transition and
  final captures and records-but-does-not-assert the setup capture — because
  under `rebind_policy=transition` the mutable base is *deliberately* borrowed
  at setup. Asserting disjointness there would assert the opposite of the
  contract under audit. The implementation evaluates exactly those two captures
  and sets `setup_capture_recorded_not_asserted: true`.
* `validate_multifork_row`'s docstring says it checks the schema, not the
  science. It has an explicit test
  (`test_validator_does_not_require_a_passing_result`) that a row whose
  equivalence failed and whose sharing saved nothing is still *valid*.
* The aggregator's docstring says it does not enforce that any predicate
  passed. A negative ownership result aggregates and is reported; only an
  incomplete or internally inconsistent record is a defect. Tested
  (`test_a_negative_scientific_result_still_aggregates`).

---

## 8. Exact invocations

Environment contract is identical to `launch_deployment_length_sweep_8gpu.sh`.

### Gate (run this first — minutes, fails fast and cheaply)

```bash
CODE_DIR=/path/to/gpu \
MODEL_DIR=/path/to/Qwen3.5-35B-A3B \
DATA_FILE=/path/to/longbench_validation.jsonl \
RUN_DIR=/path/to/runs/c1-multifork-gate-20260903 \
ENV_DIR=/path/to/venv \
GATE_ONLY=1 \
bash /path/to/gpu/launch_shared_packed_multifork_8gpu.sh
```

Success prints, per rank, the tail policy, the sharing window, and the ten-entry
`status_vector` and `coverage_vector`, then
`C1 shared-packed multifork gates passed on every rank`.

If the borrowed-prefix layer is rejected by this Transformers build, rerun the
gate with `TAIL_POLICY=materialized-tail` before the formal run and record that
the weaker sharing window applies.

### Formal run

```bash
CODE_DIR=/path/to/gpu \
MODEL_DIR=/path/to/Qwen3.5-35B-A3B \
DATA_FILE=/path/to/longbench_validation.jsonl \
RUN_DIR=/path/to/runs/c1-multifork-formal-20260903 \
ENV_DIR=/path/to/venv \
CONFIG=qcomem-d7-frozen-static \
FANOUTS=1,2,4 \
MAX_NEW_TOKENS=32 \
TAIL_POLICY=borrowed-prefix \
REBIND_POLICY=transition \
QUERY_SOURCE=cross-item \
LIMIT_PER_DATASET=4 \
MAX_INPUT_TOKENS=4096 \
EOS_POLICY=ignore \
REPEATS=1 \
SEED=20260903 \
EXPECTED_DATA_SHA256=<frozen validation SHA> \
EXPECTED_SOURCE_REVISION=<frozen revision> \
bash /path/to/gpu/launch_shared_packed_multifork_8gpu.sh
```

Add `DROP_RECEIPT_DETAILS=1` if shard size becomes a problem; verdicts, contract
rows and the ownership ledger survive, only per-tensor detail is dropped.

Aggregate standalone (torch-free, runs anywhere):

```bash
PYTHONPATH=/path/to/gpu python -m aggregate_shared_packed_multifork \
  /path/to/runs/c1-multifork-formal-20260903 \
  --expected-shards 8 --require-complete-record
```

Both launcher paths already run the aggregator; the standalone form is for
re-validating an archived run.

The launcher's own preflight compiles all eight touched Python files and runs
`test_qcomem_multifork_accounting`, `test_aggregate_shared_packed_multifork`,
`test_qcomem_shared_packed_fork`, plus `test_qcomem_deployment` and
`test_qcomem_paged` — the published paths this work must not have changed —
before a GPU is claimed.

---

## 9. Local validation performed

```
python3 -m py_compile   →  OK on all 7 new Python files
python3 -m py_compile   →  OK on qcomem_torch.py, qcomem_paged.py,
                           qcomem_deployment.py, qcomem_deployment_arms.py,
                           qcomem_eq3_accounting.py (unchanged, re-checked)
bash -n                 →  OK on launch_shared_packed_multifork_8gpu.sh
python3 -m unittest     →  174 tests: 103 pass, 71 skipped (torch absent), 0 fail
torch-stub import check →  all 5 new modules import cleanly; every __all__ name resolves
```

* `test_qcomem_multifork_accounting.py` — 72 tests, all torch-free, all pass.
  Covers the ten-target table shape; every coverage failure mode (missing,
  unbound, duplicated, modified, non-boolean); the must-not-silently-pass rule
  in both directions; contract summarization with coverage separated from
  verdict; byte-range normalization and its out-of-bounds rejections; empty and
  adjacent views never aliasing; storage-id deduplication matching
  `cache_nbytes`; the ownership ledger's shared/private split, its overlap
  detection, and its vacuity at N=1; sharing efficiency (N=1 avoids zero
  copies); every transient working-set field and the affine fit; the crossover
  search in both outcomes; token-trace comparison including missing requests
  and length differences; cross-N vacuity and mismatch; row validation; and
  summarization that counts rather than drops discrepant rows.
* `test_aggregate_shared_packed_multifork.py` — 31 tests, all torch-free, all
  pass. Builds synthetic shards and asserts a good run aggregates and each
  specific defect is caught: inflated status, incomplete coverage carrying a
  predicate, missing/duplicate/unknown target, contract-table drift, summary
  drift, gate failure, non-completed status, schema drift, malformed row,
  missing arm at a fanout, policy disagreement between shards. Also asserts a
  negative scientific result still aggregates and a semantic discrepancy is
  surfaced.
* `test_qcomem_shared_packed_fork.py` — 71 tests, torch-gated, **skipped
  locally**. They need torch but not CUDA, not Transformers and not a
  checkpoint: caches are `SimpleNamespace` layers with real tensors (one
  attention layer, one GDN layer with FP32 recurrent state), and execution runs
  against a fake adapter that appends through the cache-layer `update` contract
  and mutates conv/recurrent buffers in place. They cover the tensor-slot
  walker, storage inventories, a regression guard that
  `PackedLowerReplayState.fork` still produces fully private copies, the
  shared/borrowed/private byte split at fork and after the transition, the
  double-rebind error, two forks' disjointness, the borrowed-prefix layer's
  bit-identical return value and its never-written prefix, `get_mask_sizes` in
  both call forms and its fail-loud rejection of a third, the fallback on an
  unclassified cache leaf, an end-to-end audit in which all ten targets are
  covered and pass, an N=1 audit in which four targets are correctly open, and
  the missing-capture error.

The torch-stub import check reuses this repository's existing pattern (a
minimal `sys.modules["torch"]` stub, as in the A4/A5 tick): it executes module
bodies, which catches import-time `NameError`s and bad `from`-import lists that
`py_compile` cannot. The stub lives in the scratchpad and is not committed.

---

## 10. Untested on GPU — what the first run must verify

Ordered by how likely each is to bite.

1. **`BorrowedPrefixKVLayer`'s mask-sizing contract on this Transformers
   build.** `qcomem_torch._layer_context` calls `create_causal_mask(...,
   past_key_values=cache, layer_idx=full_attention_layer)`, which consults the
   cache's mask-size path for that layer. Our layer supplies
   `get_seq_length()` and `get_mask_sizes(*args)` accepting either an integer
   query length or a 1-D `cache_position` tensor, mirroring
   `qcomem_paged_attention.PagedKVLayer.get_mask_sizes(query_length)`. If this
   build's `DynamicCache` reads `layer.keys.shape[-2]` directly instead of
   delegating, our `keys is None` will raise — loudly, at the gate, in minutes.
   The receipt records `mask_size_call_forms` so the observed convention is in
   the record. **If this fails, rerun with `TAIL_POLICY=materialized-tail`,
   which touches no interface beyond `update`.**
2. **Bit-identity of the borrowed-prefix return value.** The claim is that
   `torch.cat([prefix, tail])` returned per call is the same tensor a stock
   `DynamicLayer` would have returned, so the attention GEMM is unchanged. This
   is why the page-wise reference in `qcomem_paged_attention` is *not* used:
   its per-page online softmax has a different reduction order and is already
   recorded as failing the exactness gate on this checkpoint
   (`RESULTS_GPU_QWEN35_PAGED_REFERENCE_NEGATIVE_2026-08-13_ZH.md`). Gate
   criterion 3 tests this directly against the published N=1 path.
3. **`CacheLayerMixin` is concrete for our subclass.** `BorrowedPrefixKVLayer`
   implements `offload`, `prefetch`, `reset`, `reorder_cache`, `crop`,
   `batch_repeat_interleave` and `batch_select_indices` as raising stubs,
   mirroring `PagedKVLayer`. If this build's mixin declares an abstract method
   we have not overridden, instantiation fails at the first fork.
4. **At j=7 only one full-attention layer sits below the split.** On Qwen3.5 the
   full-attention layers are 3, 7, 11, …, 39, so the depth-7 lower cache has
   exactly one shareable attention layer; layers 0,1,2,4,5,6 are GDN. The entry
   reports `shared_attention_layer_count`. If a future depth is shallower than
   layer 3, the shared attention bytes are zero, the borrowed-prefix policy
   shares only the boundary residual, and gate criterion 2 (non-vacuous
   sharing at the window) fails rather than passing quietly. Confirm the count
   is 1 on the first shard.
5. **`analyze_cache_for_cow` accepts the real 40-layer `DynamicCache`.** It
   fails closed on any unclassified non-empty leaf. It is already exercised by
   the published `paged-cow-staging` path, but never on a cache produced by
   `PackedCache.dequantize()`. A rejection shows up as an entry falling back to
   private materialization with a reason, which the gate treats as a failure of
   criterion 1.
6. **Cost of the full-content digests.** Four `tensor_tree_digest` passes per
   audited run over the packed entry and the view (~30 MiB dequantized at 4k
   tokens). Expected to be seconds, but `audit_seconds` is recorded; check it
   on the first shard before scaling `FANOUTS` or `LIMIT_PER_DATASET`.
7. **Peak memory with all fanouts of one document alive.** The runner holds one
   `audit_inputs` per fanout for the whole document so the cross-N target is
   non-vacuous, which keeps one dequantized view alive per fanout (three views
   at `FANOUTS=1,2,4`). Confirm `phase_allocated_nbytes` and
   `peak_transient_allocation_nbytes` on the first shard rather than assuming.
8. **Wall clock.** Per document per repeat: N private reference runs
   (`published_private_reference_traces`) plus, per fanout, one audited shared
   run, one private multifork run and one full-prefix multifork run. At
   `FANOUTS=1,2,4` and `MAX_NEW_TOKENS=32` that is roughly 7 reference requests
   plus 21 multifork requests per document. Start with the default
   `LIMIT_PER_DATASET=4` smoke and read the per-row timings before enlarging.
9. **Transformers `DynamicCache` layer list is mutable.** The borrowed-prefix
   path assigns `local_cache.layers[index] = BorrowedPrefixKVLayer(...)`, as
   `qcomem_paged_attention.replace_dynamic_cache_layer` already does. If
   `layers` is a tuple in this build, the assignment raises at the first fork.

---

## 11. Assumptions a reviewer could challenge, and what this still will not prove

Stated plainly, because disclosure of a *measured* limitation is different from
disclosure of an *unmeasured* gap, and the panel was explicit that more of the
latter cannot move Contribution.

**Assumptions.**

* *Concurrency is interleaving on one CUDA stream, not multi-stream or
  multi-process.* All N requests are resident and their state is live
  simultaneously, and decode steps are interleaved round-robin. That is the
  same concurrency model the earlier A4 transfer used. It is **not** a serving
  system, and nothing here measures scheduler behaviour, queuing, batching, or
  admitted throughput.
* *The N distinct queries against one document come from the rank's other
  LongBench items* (`QUERY_SOURCE=cross-item`), falling back to distinct
  truncated windows of the item's own question when the rank holds fewer items
  than the fanout. Only request `r00` carries the document's matched question,
  so **F1 is meaningful only for `r00`**; every row records
  `matched_request_id` and per-request query provenance. A reviewer could
  reasonably say the other requests are memory-realistic but not
  quality-realistic. They are there to make the sharing non-vacuous, not to
  produce F1.
* *The registered transition is a driver call, not a mutation trap.* We do not
  intercept a fused GatedDeltaNet kernel to detect the first in-place write. We
  rebind at a point recorded in the receipt, immediately before the first
  lower-layer call. If some code path mutated the borrowed base between fork
  and that call, the sampled view guard would catch it only if the change is
  visible at one of the 16 sample points or changes storage identity.
* *`forks_released` is bookkeeping.* Target 10 observes that the driver dropped
  its Python references and that the packed entry's content digest replays; it
  is **not** an allocator-level proof of deallocation. This is printed in that
  target's `exact_missingness`. Of the three packed-entry obligations it is the
  weakest, and a reviewer is entitled to say so.
* *Digests are in-process.* Capture and replay run in the same process as the
  execution they observe. A kernel that wrote a shared tensor and restored it
  before the next digest would not be caught. This is an offline,
  non-adversarial regression contract, printed as
  `trusted_computing_base` on every audit.
* *The affine resident model is fitted from a small number of fanouts.* The
  crossover number is a property of that fit, not an extrapolated measurement,
  and is labelled as such in the payload's `semantic` field.
* *`cache_nbytes`-style deduplication.* Shared and private byte totals
  deduplicate by storage id, matching the frozen accountant. A tensor that is a
  non-contiguous view has its whole storage charged, flagged `contiguous:
  false` on the inventory row.

**What this experiment still will not prove, even if every target passes.**

1. **Nothing about a paged attention kernel.** No fixed-size pages, no
   partial-page tail, no page table, no fused kernel. Target 4 is capped at
   `partial` for exactly this reason.
2. **Nothing about vLLM or SGLang.** The result transfers to the Transformers
   split-replay stack only. It does not retroactively validate — or invalidate
   — the existing vLLM ForkAudit verdicts.
3. **Nothing about throughput, TTFT, TPOT or admitted serving capacity.**
   Timings are recorded, but the interleaved single-stream protocol is not a
   serving benchmark and no latency claim may be made from it. In particular
   `borrowed-prefix` pays a transient concatenation per attention call; its
   TPOT is expected to be *worse*, and that is measured and reported, not
   hidden.
4. **Nothing about a second backbone or a second checkpoint.** One model, one
   depth, one frozen bit vector.
5. **Nothing about the quality of shared execution beyond token identity.**
   Target 6 establishes that sharing does not change the output. It does not
   establish that the output is good; F1 is reported for `r00` only and is not
   a new quality result.
6. **Nothing about security.** The contract is a regression check on a
   cooperative runtime, not an attestation, and the TCB statement says so.
7. **It does not by itself fix ISS-01, ISS-03, ISS-05, TS-04 or TS-05.** C1
   closes the composition gap. A1–A8 remain the paper's other required work and
   this deliverable does not touch them.
8. **If the gate forces `TAIL_POLICY=materialized-tail`, the result is
   materially weaker.** Sharing would then be established only between fork and
   each request's first write, the emitted `scope_note` would say so, and the
   honest reading is "N resident-but-not-yet-writing requests share one entry",
   not "N concurrent requests share one entry throughout." That distinction
   must reach the manuscript verbatim if that is the path taken.

---

## 12. What the paper can claim if the run comes back clean

One table, from one execution, at N ∈ {1, 2, 4} on the quantized j=7 split
path, reporting per arm: retained Store, the shared dequantized view, transient
materialized bytes, peak transient allocation, steady-state resident bytes, the
fitted resident line, allocator phase endpoints, F1 for the matched request,
token-identity against the published Read path, and the ten-target ForkAudit
coverage and status vectors. That is precisely the C1 verification criterion.

If it comes back with open targets or a semantic discrepancy, that is a valid
negative and the record still aggregates: the aggregator is built to report it,
not to average it away.
