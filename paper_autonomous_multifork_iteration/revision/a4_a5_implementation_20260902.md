# A4 / A5 — implementation report

Date: 2026-09-02
Actions: **A4** (honest dense baseline + generation-length sweep) and **A5**
(quantized exact-cache baselines), both `code`, both blocking for R45.
Status: **code complete, validated locally, never executed on a GPU.**
No numbers are produced or claimed in this document.

Live source: `/Users/liuhanzuo/MacLLM-Bench/gpu`.
Nothing under `/mnt/tidal-alsh-hilab` was read or written; no job was submitted.

---

## 0. Headline

Eight new files. **Zero existing files modified.** Confirmed by mtime: the only
files under `gpu/` newer than 2026-09-01 are the eight listed in §1.

The two new modes live in a new module that *delegates* the three published
modes (`dense_recompute`, `full_prefix`, `qcomem`) to
`qcomem_deployment.run_incremental_generation` unchanged, so a sweep row for a
published arm executes byte-for-byte the published code path.
`PackedCache.nbytes`, `cache_nbytes`, `tensor_nbytes`, `persistent_components`
and `capacity_estimate` are untouched; the new accounting is additive and is
*reconciled against* the frozen accountant on every row rather than replacing
it.

`quantize_tensor` at `bits=16` still has the known dtype defect (it clones each
leaf in its source dtype, so Qwen3.5 FP32 GDN `recurrent_states` are counted at
4 bytes/element). The new code **does not replicate it and does not route
around it silently**: a `bits=16` quantized full-prefix arm is refused with an
explanatory error, and the dtype-consistent reference the paper needs is
computed analytically from element counts and emitted on every row as *two*
numbers — a native-dtype count and an all-BF16 count — so both ratios can be
formed downstream without guessing.

---

## 1. Every file touched

### Added (8)

| file | sha256 (first 16) | lines | torch needed? |
|---|---|---|---|
| `gpu/qcomem_eq3_accounting.py` | `8045cdd9ab6c4199` | 769 | **no** |
| `gpu/qcomem_deployment_arms.py` | `4166b69ad61da612` | 1198 | yes |
| `gpu/run_deployment_length_sweep.py` | `a8058121858b9cde` | 742 | yes |
| `gpu/aggregate_deployment_length_sweep.py` | `93231183b8233aa7` | 477 | **no** |
| `gpu/launch_deployment_length_sweep_8gpu.sh` | `af0d4ab1a62401a9` | 238 | n/a |
| `gpu/test_qcomem_eq3_accounting.py` | `bca3acda7287d8f6` | 586 | **no** |
| `gpu/test_aggregate_deployment_length_sweep.py` | `10cd47ebb477e4bb` | 399 | **no** |
| `gpu/test_qcomem_deployment_arms.py` | `9cda7ad8f5e919a2` | 710 | yes (skips) |

### Modified (0)

In particular the following were **read but not edited**: `qcomem_torch.py`,
`qcomem_deployment.py`, `qcomem_paged.py`, `run_deployment_bench.py`,
`aggregate_deployment.py`, `run_downstream.py`, `launch_deployment_8gpu.sh`,
`test_qcomem_deployment.py`, `test_qcomem_torch.py`.

### Roles

* **`qcomem_eq3_accounting.py`** — deliberately torch-free. The Eq. 3 byte
  identity and its assertion, both reference counts, the per-component /
  per-state-type / per-layer aggregation, decode-latency and throughput
  statistics, the sweep planner, and the row validator. Everything a
  reproducibility reviewer would want to re-check is here and runs on a laptop.
* **`qcomem_deployment_arms.py`** — the two new modes, the packed full-prefix
  state, the cache walker that turns real tensors into the plain-dict component
  records the torch-free module consumes, the strided memory recorder, and the
  two new gates.
* **`run_deployment_length_sweep.py`** — the runner. Reuses
  `run_deployment_bench.longbench_workloads` / `synthetic_workloads` verbatim,
  so the LongBench slicing, prompt protocol, SHA-256 and source-revision guards
  are literally the same code as the published deployment bench.
* **`aggregate_deployment_length_sweep.py`** — torch-free aggregator that
  re-derives every check from the raw per-row fields.
* **`launch_deployment_length_sweep_8gpu.sh`** — 8-GPU launcher, same
  `CODE_DIR` / `MODEL_DIR` / `DATA_FILE` / `RUN_DIR` / `ENV_DIR` contract as
  `launch_deployment_8gpu.sh`.

---

## 2. A4.1 — `dense_prefill_once`

New configuration name: **`dense-prefill-once`**, mode `dense_prefill_once`.

### What it does

For **every** query, independently:

1. concatenate `document_tokens` and `query_tokens` into one prefix;
2. call `adapter.prefill_full_prefix(prefix)` — one forward pass, `use_cache=True`,
   producing final-token logits and a `FullPrefixState`;
3. decode greedily with `adapter.continue_full_prefix(state, [token])`, one
   position at a time, against that within-request KV cache;
4. drop the cache at the end of the request.

This is exactly the semantics of `qcomem_torch.py:978 greedy_generate_dense`
(`prefill_full_prefix` then repeated `continue_full_prefix`) — the path the
60-item quality cohort used — and it is what Section 3 of the manuscript
defines as the dense baseline.

### What it does NOT do

* It does **not** retain anything between requests. `persistent_document_nbytes`
  is `0` and `store_breakdown` is the explicit empty breakdown carrying the
  string `"this arm retains no cross-request state"`. It is not a store-size
  competitor; it is a *latency and throughput* competitor.
* It does **not** replace `dense-recompute`. That mode is untouched and stays in
  the default arm list at `n = 8`. The manuscript reports both and explains that
  Table 2's 648.75 ms TPOT (≈ its own 0.649 s TTFT) is the recompute arm, not
  this one.
* It is **not** claimed to be bit-identical to `full-prefix-q16`. It consumes
  document+query as **one** chunk; the incremental full-prefix arm consumes them
  as **two**. Qwen3.5 GatedDeltaNet and convolution states are sensitive to that
  boundary, which is why the existing `run_exactness_gate` already labels the
  single-chunk dense path `dense_single_chunk_diagnostic_only`. The new gate
  keeps that classification (see §7).
* It does **not** measure a fresh-model cold start. Model load and CUDA warmup
  are amortized exactly as for every other arm.

### Row fields specific to this arm

`fork_memory.prefill_cache_nbytes` records the request-local prefix cache size
(document+query) so a reviewer can see what dense actually holds per active
request. `selected_fork_active_state_*` is that same number, and
`decode_kv_*` counts only what the generated tokens add on top of it.

---

## 3. A4.2 — what every row now records about throughput

The published bench already emitted `tpot_seconds` as a list and
`generated_tokens`; what was missing was a real wall clock. Every row now
carries, in addition to the existing Store/F1/memory fields:

| field | meaning |
|---|---|
| `generated_tokens` | the true per-item token count, never imputed |
| `max_new_tokens_requested` / `max_new_tokens_effective` | the swept cap, and the cap after any deliberate dataset clamp |
| `reached_cap`, `eos_stopped`, `natural_eos_step` | whether the cap was reached, and where a stopping run *would* have stopped |
| `ttft_seconds` | unchanged semantics (synchronized `_timed` around the prefill) |
| `tpot_seconds` | the **full per-step list**, not a mean |
| `decode_latency` | `mean / median / min / max / p90 / p99 / first / last / first-quarter mean / last-quarter mean` over that list |
| `request_wall_seconds` | CUDA-synchronized wall clock around the whole request, instrumentation included |
| `end_to_end_including_build_seconds` | wall clock around build + request |
| `throughput.online_tokens_per_second` | `n / (TTFT + sum(TPOT))` — instrumentation-free |
| `throughput.wall_tokens_per_second` | `n / request_wall_seconds` — the real end-to-end number |
| `throughput.end_to_end_tokens_per_second` | `n / end_to_end_including_build_seconds` |
| `throughput.reconstructed_tokens_per_second` | `n / (TTFT + n * median TPOT)` — **the model Table 2's tok/s column was derived from**, emitted beside the measurement so its error is visible instead of assumed |
| `throughput.reconstructed_over_measured` | the ratio of the two, per item |
| `throughput.instrumentation_overhead_seconds` | `request_wall - online` |

The reconstructed model is structurally biased: it charges `n` decode steps
when only `n - 1` are ever paid (the loop never advances after the last emitted
token). At `n ≈ 8` that is a ~12% overcharge on the decode term. Emitting both
numbers is the point of A4.2 — the aggregator's `throughput_model_audit`
section reports the ratio per (config, length).

`decode_latency.first_quarter_mean` vs `last_quarter_mean` is the field that
answers the actual open question in `derived_vs_measured_20260902.md`: whether
TPOT stays constant as the KV grows. It is measured, not modelled.

---

## 4. A4.3 — the generation-length sweep

`--max-new-tokens-sweep 8 128 512` (default). Each (configuration, length) pair
is an **arm**, named `config@nLENGTH`, e.g. `full-prefix-q4@n512`. Arms are
interleaved with the same randomized-order discipline as the published bench:
`qcomem_eq3_accounting.shuffled_arm_orders` uses the identical
`random.Random(seed + repeat).shuffle` construction as
`qcomem_deployment.shuffled_config_orders`, applied to arms instead of configs.
Every (workload, repeat) cell contains a full permutation of all arms, and the
aggregator refuses a run where any cell is incomplete.

### Reaching the cap

`--eos-policy ignore` (**default**) passes an empty EOS set into the decode
loop, so generation is never cut short and `generated_tokens == max_new_tokens`
on every row. The declared EOS ids are still recorded, and `natural_eos_step`
records the index of the first generated token that *is* an EOS id — i.e. where
a stopping run would have ended — so nothing about EOS behaviour is lost.

`--eos-policy stop` honours EOS. In that mode `generated_tokens` is the true
per-item count, `eos_stopped` is set, and the row validator *requires* a short
generation to be marked as such. Nothing is imputed in either mode: the
aggregator's arm summaries report `generated_tokens_min/max` and
`reached_cap_fraction` rather than assuming `n`.

### `dense-recompute` is capped

`--config-length-limit dense-recompute=8` (default). `dense_recompute` re-runs
the whole document/query/generated sequence for every token, so its cost is
quadratic in the generation length; at `n = 512` on a 4k document it is not a
measurement, it is a way to lose the job. It stays in the comparison at `n = 8`,
where Table 2 measured it. If `--configs` omits `dense-recompute`, the default
limit is narrowed away and recorded in
`protocol.config_length_limits_ignored` rather than raising.

### `--generation-limit-policy`

`fixed` (default) runs every arm at the swept length.
`dataset` additionally clamps to `run_downstream.DATASET_MAX_NEW_TOKENS`
(`qasper: 128`, `2wikimqa: 32`). Both the requested and the effective cap, and
whether the clamp fired, are recorded per row.

---

## 5. A5 — quantized exact-cache baselines

New mode `full_prefix_quantized`, built by `build_packed_full_prefix`:

```
exact  = adapter.write_full_prefix(document_tokens)          # all 40 layers
packed = quantize_transformers_cache(exact.cache, ...)       # the SAME packer
del exact
```

`quantize_transformers_cache` is the identical function, with the identical
default `group_size=64` and the identical BF16 scale/bias metadata format, that
`LowerReplayState.quantize` calls for the split-replay rows. There is no
parallel implementation. Read is `PackedFullPrefixState.fork()` →
`PackedCache.dequantize()` → an ordinary `FullPrefixState`, after which decode
is the *same* `continue_full_prefix` loop the exact arm uses.

### Arm names

| name | attention (K/V) | linear (conv/recurrent) | note |
|---|---|---|---|
| `full-prefix-q16` | — | — | **unchanged published exact arm**, mode `full_prefix` |
| `full-prefix-q8` | 8 | 8 | uniform Q8 |
| `full-prefix-q4` | 8 → 4 | 4 | uniform Q4 |
| `full-prefix-q2` | 2 | 2 | available, not in the default list |
| `full-prefix-frozen-static` | 4 | 8 | full-prefix analogue of the frozen policy |
| `full-prefix-aA-lL[-layers=…]` | A | L | general state-type / per-layer form |

### Why `full-prefix-frozen-static` is `a4 / l8`

The frozen split policy `qcomem-d7-frozen-static` declares
`attention_bits=4, linear_bits=8, cache_layer_bits=(8,8,8,4,8,8,8)`. Per A11
§1.3 the layer vector **overrides** the state-type widths for the actual
packing, and per A11 §3.2 layer 3 is the sole full-attention layer among the
lower seven, the other six being GDN. So the frozen vector is, read as a
state-type policy, exactly *attention at Q4, linear at Q8*. The full-prefix
analogue therefore sets `attention_bits=4, linear_bits=8` with
`cache_layer_bits=None`, which makes `quantize_transformers_cache` resolve each
of the 40 layers by whether it exposes `conv_states`/`recurrent_states`.
**This is an interpretation, and it is the single most challengeable choice in
A5** — see §10.

### A `bits=16` quantized full-prefix arm is refused

`_validate_quantized_full_prefix` raises if any declared width is 16, with a
message naming the defect and pointing at `full-prefix-q16` (the genuinely
exact arm) as the reference. Reason: `quantize_tensor` at `bits=16` does
`data = flat.clone()` preserving `tensor.dtype`, so on Qwen3.5 such an arm
would count FP32 GDN `recurrent_states` at 4 bytes/element and inflate its own
denominator — the exact mechanism A11 identified. The dtype-consistent
reference is produced analytically instead (§6).

### What A5 does NOT claim

* It does **not** claim token or logit equality with the exact arm. A Q4 cache
  is lossy by construction; the gate records the divergence
  (`vs_exact_full_prefix`) and explicitly does not gate on it.
* It does **not** implement a paged / borrowed / copy-on-write read for the
  quantized full prefix. Read is a **full private materialization** per query,
  the same discipline as `PackedLowerReplayState.fork` (A11 finding C4).
  `fork_memory.materialized_read_nbytes` reports that cost explicitly so the
  comparison is not quietly flattering.
* It does **not** re-tune the policy. `full-prefix-frozen-static` is a
  transplant of an existing policy, not a new search; no calibration was run.

---

## 6. Accounting: per-component bytes and two references, on every row

Every row carries `store_breakdown`:

```
store_breakdown
├─ format            {equation, group_size, metadata_nbytes_per_group, metadata_dtype, ...}
├─ components[]      one per (layer, state type) leaf:
│                      leaf_path, layer_index, state_type, elements, bits,
│                      floating, quantized, is_packed_width, group_size, groups,
│                      code_nbytes, scale_nbytes, bias_nbytes, metadata_nbytes,
│                      total_nbytes, storage_nbytes,
│                      native_itemsize, native_reference_nbytes, bf16_reference_nbytes,
│                      dtype_inconsistent_reference,
│                      eq3_expected_nbytes, eq3_identity_ok, eq3_identity_checked
├─ by_state_type     {document_residual | attention_key | attention_value |
│                     conv_state | recurrent_state | other → summed fields}
├─ by_layer[]        same fields, keyed by layer_index (None = boundary/root)
├─ totals
├─ packed_store_nbytes, packed_store_storage_nbytes
├─ native_dtype_reference_nbytes, bf16_reference_nbytes
├─ native_dtype_ratio, bf16_ratio
├─ dtype_inconsistent_components
├─ eq3_identity_ok, eq3_identity_violations[]
└─ reconciliation    {frozen_accountant_nbytes, breakdown_storage_nbytes, delta_nbytes, matches}
```

### The assertion

`assert_eq3_component_identity` raises `Eq3IdentityError` unless

```
total_nbytes == ceil(elements / group_size) * (group_size * bits / 8 + 4)
```

for every genuinely packed component (`bits ∈ {2, 4, 8}`), naming the component,
its layer, its state type, its width, and the code/metadata split in the
message. It is applied to every component of every row by default
(`--no-strict-accounting` downgrades it to a recorded violation instead of a
raise), again inside both gates, and a third time by the torch-free aggregator
over the archived shards. Group padding is handled correctly: a 100-element
component at Q8 is charged two whole groups (136 bytes), which is what
`quantize_tensor`'s edge-value padding actually allocates.

Verbatim components (`bits = 16`, and the unquantized leaves of the exact
reference arm) are **not** subject to that identity — they carry no group
metadata. They are checked against the weaker verbatim identity
`elements * native_itemsize`, and any floating one whose itemsize is not 2 is
flagged `dtype_inconsistent_reference`. On Qwen3.5 that flag fires on exactly
the FP32 GDN `recurrent_states`, in the exact arm, per layer, visibly.

### Both references

* `native_dtype_reference_nbytes` = Σ `elements × native itemsize` — what the
  component costs dense **in the dtype the model actually produced** (FP32 for
  GDN recurrent state).
* `bf16_reference_nbytes` = Σ `elements × 2` for floating components (and native
  bytes for non-floating leaves such as integer counters, which cannot be
  BF16).

Both are emitted per component, per state type, per layer and in total, so
`native ratio` and `all-BF16 ratio` are both computable afterwards with no
guessing. The aggregator's `paired_against_reference` forms three ratios per
arm against the *same item's* `full-prefix-q16` row: against its stored bytes,
against its BF16 reference, and against its native reference.

### Reconciliation with the frozen accountant

The walker deduplicates storage by `(device, data_ptr, storage nbytes)`, exactly
as `qcomem_torch.cache_nbytes` does, and the summed `storage_nbytes` is compared
to `PackedCache.nbytes` / `state.stored_nbytes` on every row
(`reconciliation.matches`). This is the artifact A1 §"Status" asked for: a
per-component table that provably adds up to the number Table 2 printed.

---

## 7. Gates

Run once per rank, before any timed measurement, and archived in the shard.

1. **`published_exactness_gate`** — the existing
   `qcomem_deployment.run_exactness_gate`, unchanged and unskipped by default.
   Keeps the published incremental full-prefix vs Q16 hard gate in force.
2. **`dense_semantics_gate`** (new) — runs `dense-recompute`,
   `dense-prefill-once` and `full-prefix-q16` on the gate-sized document/query
   **with EOS disabled**, so all three emit the same number of positions.
   * **Asserts:** all three agree on the **first** token (computed from the
     identical prefix in every arm); and
     `median TPOT(dense-prefill-once) < median TPOT(dense-recompute)` — one
     decodes a single position against a KV cache, the other re-runs the whole
     sequence, so if that inequality fails the new arm is not doing what its
     name says.
   * **Records but does not assert:** full token-sequence and logit comparisons
     in `diagnostics`, plus per-arm TTFT, median TPOT and `TPOT/TTFT`. Token
     equality across different chunk boundaries is *not* required, for the same
     GDN chunk-sensitivity reason the published gate already cites.
3. **`full_prefix_quant_gate`** (new) — for each quantized full-prefix arm.
   * **Asserts:** every packed component satisfies the Eq. 3 identity; the
     breakdown reconciles with `PackedCache.nbytes`; the packed store is
     strictly smaller than the exact store; and the dequantized Read restores
     the same component shapes and dtypes as the exact cache.
   * **Records but does not assert:** the token/logit divergence from the exact
     arm.

`--gate-only` / `GATE_ONLY=1` runs the gates and stops, writing
`status: gate_passed` shards. That is the recommended first cluster action.

---

## 8. How to invoke each run

### Gate-only smoke (do this first)

```bash
CODE_DIR=/path/to/gpu \
MODEL_DIR=/path/to/qwen35 \
DATA_FILE=/path/to/longbench.jsonl \
RUN_DIR=/path/to/runs/a4a5-gate \
ENV_DIR=/path/to/venv \
GATE_ONLY=1 \
bash "$CODE_DIR/launch_deployment_length_sweep_8gpu.sh"
```

### A4 + A5 formal run (60-item cohort, all arms, all lengths)

```bash
CODE_DIR=... MODEL_DIR=... DATA_FILE=... RUN_DIR=... ENV_DIR=... \
LIMIT_PER_DATASET=30 \
SOURCE_INDEX_START=6 SOURCE_INDEX_END=35 \
MAX_INPUT_TOKENS=4096 \
MAX_NEW_TOKENS_SWEEP=8,128,512 \
CONFIG_LENGTH_LIMITS=dense-recompute=8 \
EOS_POLICY=ignore \
REPEATS=3 WARMUPS=1 SEED=20260902 \
EXPECTED_DATA_SHA256=<frozen> EXPECTED_SOURCE_REVISION=<frozen> \
PROTOCOL_LABEL=a4-a5-length-sweep \
bash "$CODE_DIR/launch_deployment_length_sweep_8gpu.sh"
```

Defaults if unset: `LIMIT_PER_DATASET=4` (a smoke: 8 workloads, one per rank),
`MAX_NEW_TOKENS_SWEEP=8,128,512`, `EOS_POLICY=ignore`,
`GENERATION_LIMIT_POLICY=fixed`, `REPEATS=3`, `SEED=20260902`,
`GROUP_SIZE=64`, `FORK_STRATEGY=deep-clone`,
`CONFIGS=dense-recompute,dense-prefill-once,full-prefix-q16,full-prefix-q8,full-prefix-q4,full-prefix-frozen-static,qcomem-d7-frozen-static`.
That default is 19 arms (dense-recompute only at `n=8`).

### A4 only (drop the quantized full-prefix arms)

```bash
CONFIGS=dense-recompute,dense-prefill-once,full-prefix-q16,qcomem-d7-frozen-static \
... bash "$CODE_DIR/launch_deployment_length_sweep_8gpu.sh"
```

### A5 only (single length, quality-protocol generation limit)

```bash
CONFIGS=full-prefix-q16,full-prefix-q8,full-prefix-q4,full-prefix-frozen-static,qcomem-d7-frozen-static \
MAX_NEW_TOKENS_SWEEP=128 \
GENERATION_LIMIT_POLICY=dataset EOS_POLICY=stop \
... bash "$CODE_DIR/launch_deployment_length_sweep_8gpu.sh"
```

`GENERATION_LIMIT_POLICY=dataset` + `EOS_POLICY=stop` reproduces the LongBench
generation protocol, which is the configuration whose F1 is comparable to the
published table. See §10 on F1.

### One rank by hand

```bash
CUDA_VISIBLE_DEVICES=0 python run_deployment_length_sweep.py \
  --model "$MODEL_DIR" --run-dir "$RUN_DIR" --rank 0 --world-size 1 \
  --workload longbench --data "$DATA_FILE" \
  --configs dense-prefill-once full-prefix-q16 full-prefix-q4 \
  --max-new-tokens-sweep 8 128 --eos-policy ignore --repeats 1
```

### Aggregation (torch-free; runs anywhere)

```bash
python aggregate_deployment_length_sweep.py "$RUN_DIR" --expected-shards 8
# writes $RUN_DIR/length-sweep-aggregate.json
```

The launcher already does this. Sections produced: `consistency`, `gates`,
`row_audit`, `coverage`, `arm_summary`, `arm_summary_by_dataset`,
`paired_against_reference`, `throughput_model_audit`, `store_reference_audit`.

### Other useful switches

| flag / env | effect |
|---|---|
| `--config-length-limit NAME=LIMIT` / `CONFIG_LENGTH_LIMITS` | cap the lengths one config runs at |
| `--decode-sample-stride` / `DECODE_SAMPLE_STRIDE` | `0` = auto (≤ 64 decode memory samples); `1` = sample every step like the published recorder |
| `--drop-memory-samples` / `DROP_MEMORY_SAMPLES=1` | omit the raw sample list from the shard |
| `--no-strict-accounting` | record byte-identity violations instead of raising |
| `--skip-published-exactness-gate` / `SKIP_PUBLISHED_EXACTNESS_GATE=1` | skip the pre-existing gate (not recommended) |
| `--mixed-policy-file` / `MIXED_POLICY_FILE` | same mixed-policy loading as the published bench |

Per-layer full-prefix policies (`full-prefix-a4-l8-layers=8,8,4,…`) must be
passed directly to the runner, not through the comma-separated `CONFIGS` env
var, because the `layers=` value itself contains commas.

---

## 9. Local validation performed

```
python3 -m py_compile  →  OK on all 7 new python files
python3 -m py_compile  →  OK on qcomem_torch.py, qcomem_deployment.py,
                          run_deployment_bench.py, aggregate_deployment.py
bash -n                →  OK on launch_deployment_length_sweep_8gpu.sh
python3 -m unittest    →  97 tests, 56 pass, 41 skipped (torch absent), 0 fail
```

* `test_qcomem_eq3_accounting.py` — 39 tests, all torch-free, all pass. Covers
  the group arithmetic; the Q8/Q4/Q2 byte table (68/36/20); the compression
  ceilings (1.8824 / 3.5556 / 6.4000); padding; the archived-cohort residual
  identity (4096×2048 elements → 16,777,216 B at Q16 and 4,718,592 B at Q4,
  ratio exactly 3.555556); the FP32 GDN inflation (+1 MiB per layer, +6 MiB at
  `j=7`, +30 MiB full prefix); **the exact failure mode A1 hypothesised** (a
  counter omitting the 4 metadata bytes per group is caught, with the right
  delta); the identity assertion's message; summarization and reconciliation;
  the decode/throughput statistics including the reconstructed-model bias; the
  sweep planner and its rejections; and every branch of the row validator.
* `test_aggregate_deployment_length_sweep.py` — 17 tests, all torch-free, all
  pass. Builds synthetic shards and asserts that a good run aggregates and that
  each specific defect is caught: incomplete shard, divergent arm list,
  divergent protocol, failed or missing gate, byte-identity violation, short
  generation under `eos-policy ignore`, missing arm coverage, wrong arm list.
  Also asserts the Q4 arm's BF16 ratio comes out at exactly 32/9.
* `test_qcomem_deployment_arms.py` — 41 tests, torch-gated, **skipped locally**.
  They need torch but not CUDA and not a checkpoint: caches are
  `SimpleNamespace` layers holding real tensors (including an FP32 recurrent
  state and a deliberately ragged 100-element component), and generation runs
  against a fake adapter that counts prefill / continue / full-recompute calls.

Additionally, all new and touched modules were **import-checked** against a
minimal torch stub, which catches import-time name errors that `py_compile`
cannot; and the CLI was exercised end-to-end without torch to confirm the
default arm plan resolves to the expected 19 arms with the expected modes and
widths.

---

## 10. Untested on GPU — what the first run must verify

Nothing in this deliverable has run on a GPU or against a real checkpoint.
Ordered by how likely each is to bite:

1. **`quantize_transformers_cache` on a model-produced full-prefix cache.**
   It requires `cache.layers`. The published code only ever hands it a cache
   built by `adapter.make_cache()`; A5 hands it `output.past_key_values` from
   `write_full_prefix`. Both are `DynamicCache` in the target build, and the
   published `full_prefix` arm already deep-copies that object, but the
   `.layers` walk over a model-produced 40-layer cache is genuinely new.
   **`full_prefix_quant_gate` will fail loudly on rank 0 within minutes if this
   is wrong.** Run `GATE_ONLY=1` first.
2. **Every leaf of the full 40-layer cache is reached and is either floating or
   deliberately verbatim.** The gate asserts the packed component shapes and
   dtypes match the exact arm's leaf-for-leaf; a leaf the walker misses would
   show up there and in `reconciliation.matches`.
3. **Group padding never fires on the real components.** A11 §3.1 reports every
   element count in this cohort is a multiple of 64. If that holds, the packed
   codes are exactly `n·b/8`; if some full-prefix leaf is ragged, the identity
   still holds (it is written with `ceil`) but the ratio is slightly below the
   format ceiling. Check `groups × 64 == elements` per component in the first
   shard.
4. **`dense_semantics_gate`'s decode inequality.** Expected to hold by a wide
   margin (one token vs a full 320-token recompute), but it is a timing
   comparison and has never been observed on this hardware.
5. **Peak memory during a quantized full-prefix build.** `write_full_prefix`
   and the packed copy are both alive for the duration of
   `quantize_transformers_cache`, and `quantize_tensor` holds FP32 intermediates
   for the leaf it is packing. Estimated worst case is roughly 2× the exact
   cache (~150 MiB/doc at 4k tokens) plus ~24 MiB of FP32 scratch — negligible
   on H20, but confirm `build_memory.cuda_peak_allocated_bytes` on the first
   shard rather than assuming.
6. **Wall-clock cost of the full sweep.** 19 arms × 3 repeats × workloads-per-rank,
   with eight arms running at `n=512`. Estimate 1.5–2.5 h per rank for the
   60-item cohort. The default `LIMIT_PER_DATASET=4` is a smoke; do not launch
   the 30-item version without checking the smoke's per-row timings first.
7. **Shard size.** Full per-step `tpot_seconds` at `n=512` is ~10 kB/row. With
   the auto decode-sample stride the memory samples are capped at ~64/row.
   If shards get unwieldy, `DROP_MEMORY_SAMPLES=1`.

---

## 11. Assumptions a reviewer could challenge

1. **`full-prefix-frozen-static = attention Q4 / linear Q8` is an
   interpretation of the frozen split policy, not a measurement.** The frozen
   policy is a *per-layer* vector over seven layers; mapping it onto forty
   layers requires deciding what it "means". I read it as a state-type policy
   because layer 3 — the one layer at Q4 — is the sole full-attention layer
   below `j=7`. A reviewer could reasonably ask for a per-layer full-prefix
   policy instead (supported: `full-prefix-a4-l8-layers=…`), or could argue no
   such transplant is meaningful and only uniform Q8/Q4 should be reported.
   **`full-prefix-q8` and `full-prefix-q4` carry no such interpretation and are
   the safe rows to headline.**
2. **The quantized full-prefix Read is a full private materialization.** It
   dequantizes the entire 40-layer cache per query. A reviewer could object that
   a serious quantized-KV system would dequantize lazily, per layer, inside the
   attention kernel, and that this baseline therefore overstates the Read cost
   of quantized exact caching. That objection is correct and unanswered by this
   code; `materialized_read_nbytes` is emitted so the objection can at least be
   quantified. Note the split-replay arm has the identical limitation (A11 C4),
   so the *comparison* is like-for-like even though neither is state of the art.
3. **F1 in the sweep is not the LongBench quality protocol.** With
   `--eos-policy ignore` and `--generation-limit-policy fixed` the model is
   forced to emit exactly `n` tokens, which at `n = 512` produces long
   post-answer continuations and depresses F1 for every arm equally. Sweep F1 is
   a *sanity* signal, not a quality result. The quality-comparable configuration
   is `EOS_POLICY=stop GENERATION_LIMIT_POLICY=dataset`, and that is what should
   be cited for any F1 claim. This is stated per row: `eos_policy`,
   `generation_limit_policy` and `dataset_generation_limit_applied` are all
   recorded.
4. **The dense arms carry `Store = 0`, which is a semantic claim, not a
   measurement.** They retain nothing between requests by construction. A
   reviewer comparing "memory" across arms must compare the *right* columns:
   Store for the retaining arms, `fork_memory.prefill_cache_nbytes` /
   `cuda_peak_allocated_bytes` for the dense arms.
5. **`dense_prefill_once` uses a one-chunk document+query prefill; the
   incremental full-prefix arm uses two chunks.** Their token sequences are not
   guaranteed identical on Qwen3.5. This is a property of the model's GDN state,
   not of the implementation, and it is the reason the gate asserts first-token
   agreement rather than full-sequence agreement. Any claim that the two dense
   arms and the cached arm "compute the same thing" must be qualified by the
   recorded `diagnostics` divergence.
6. **Each row rebuilds its persistent state from scratch**, including the same
   config at three different lengths. This preserves the published randomized
   interleave discipline and prevents cross-arm cache contamination, but it
   means Write cost is paid 3× per config per item. `write_build_seconds` is
   recorded separately and is excluded from `online_seconds` and
   `request_wall_seconds`; only `end_to_end_including_build_seconds` includes it.
7. **The BF16 reference for non-floating leaves is their native size.** An
   integer length counter cannot be stored as BF16. These leaves are a handful
   of bytes and the choice is recorded per component (`floating: false`), but it
   does mean `bf16_reference_nbytes` is not literally `2 × total elements`.
8. **Decode memory samples are thinned at long lengths** (auto stride, first and
   last decode step always kept). `steady_state_*` medians are computed over the
   thinned set. `--decode-sample-stride 1` restores the published behaviour;
   the stride and the number of skipped samples are recorded on every row.
9. **The reconstructed-tok/s comparison assumes the published column was
   `n / (TTFT + n · TPOT)`** — as stated in the task brief. If Table 2's column
   was in fact computed some other way, `reconstructed_over_measured` is
   answering a slightly different question; the raw `ttft_seconds`,
   `tpot_seconds` and `request_wall_seconds` are all present so any other model
   can be re-derived from the shards.
10. **Cache-holding arms pay a `cache_nbytes` walk inside the timed decode
    step; `dense-recompute` does not.** This asymmetry is inherited verbatim
    from the published `full_prefix` branch of `run_incremental_generation`,
    and `dense_prefill_once` was written to match it rather than to be quietly
    faster. Cost is a Python walk over ~80 cache leaves per step, order 0.1–0.5
    ms against a ~25 ms decode, i.e. roughly 1–2% pessimism on every cached
    arm and 0% on `dense-recompute` (whose TPOT is ~649 ms, so the walk would
    be noise there anyway). `throughput.online_tokens_per_second` includes it;
    `--decode-sample-stride` does not affect it. If the paper needs a decode
    number free of it, use `decode_latency.decode_seconds_min` as a lower
    bound, or re-run with the instrumentation removed — but then the cached
    arms are no longer comparable to the published Table 2 rows.
11. **`capacity_estimate` is reused unchanged**, so its known framing
    (allocator-level bytes, a fixed safety headroom, no fragmentation model)
    carries over. A5 does not attempt to answer A13.

---

## 12. What was deliberately not done

* No modification to any published mode, accounting function, launcher or
  aggregator. A13 (residency at fixed VRAM), A14 (the `j` sweep) and A15
  (ownership discipline on the quantized Read path) are out of scope and
  untouched.
* No `bits=16` packer arm (§5).
* No new quantization policy search; `full-prefix-frozen-static` is a transplant.
* No results, no tables, no projected numbers. Per
  `derived_vs_measured_20260902.md`, the A4 and A5 rows stay in the
  "NOT admissible until measured" column until these runs return.
