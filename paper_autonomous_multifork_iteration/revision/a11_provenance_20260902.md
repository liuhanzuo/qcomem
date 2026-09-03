# A11 — Method provenance for the named contribution, and the Store-accountant defect

Date: 2026-09-02
Action: A11 (`code`), issues R44-5-01 … R44-5-04, R44-4-15. Depends on A1.
Deliverable 1: `revision/a11_method_provenance_rows_20260902.tsv` — 25 rows in the
exact schema of `evidence/method_provenance.tsv`
(`method_id`, `method_statement`, `source_path`, `symbol_or_lines`,
`configuration_or_runtime`, `manuscript_locations`), ready to append.
Nothing under `gpu/`, `main_r44_structure.tex` or `evidence/method_provenance.tsv`
was modified.

---

## 0. Headline

Two findings, in order of importance.

**(1) The Store-accounting defect is real, but it is not the defect A1 hypothesised,
and it sits in the opposite half of the fraction.** `PackedCache.nbytes` does *not*
undercount. For every one of the 60 items and all three quantized policies the
lower-cache byte count equals the Eq. 3 format identity
`ceil(n/64) * (64b/8 + 4)` summed over components — 180/180 exact, zero
mismatches. The inflation is in the **Q16 reference**: `quantize_tensor` at
`bits=16` clones each cache leaf **in its source dtype instead of casting to
BF16**, and Qwen3.5 GDN `recurrent_states` are FP32. The reference therefore
counts 4 bytes/element for 524,288 elements per GDN layer. This adds exactly
**6.0000 MiB/doc** to the split-Q16 lower cache (6 GDN layers below `j=7`) and
exactly **30.0000 MiB/doc** to the full-prefix baseline (30 GDN layers of 40).
Under a consistent BF16 reference the headline moves **14.10× → 11.00×** and
**3.93× → 3.70×**; the quantized rows do not move at all.

**(2) The provenance chain can be fully restored from code, but three of the
files that produced Table 1 are not archived anywhere in this repository**:
`run_replay_diagnostic.py`, `analyze_validation.py`, `aggregate_replay.py`
(and `launch_mixed_validation_8gpu.sh`). They exist only in the live
implementation tree. This is a genuine artifact gap for a reproducibility
reviewer and is flagged in every affected row.

Every number quoted in this report was re-derived from
`evidence/qcomem_mixed_validation_60item_20260812d/artifacts/raw/` (48 shards)
and `.../calibration/layer_policy.json`, not taken from the manuscript.

---

## 1. What was found, method element by method element

Source hashes of the live files read (SHA-256):

| file | sha256 | archived in this repo? |
|---|---|---|
| `qcomem_torch.py` | `5901f153…4118a7373c` | yes, byte-identical at 4 paths incl. `supplement_anonymous/code/` |
| `run_downstream.py` | `95a7d0d1…eb14c8d5` | yes, byte-identical at 4 paths incl. `supplement_anonymous/code/` |
| `run_replay_diagnostic.py` | `7c38720f…0a8af3760` | **no** |
| `analyze_validation.py` | `82f46b45…59b1a001` | **no** |
| `aggregate_replay.py` | `b879aef9…56733ede` | **no** |
| `aggregate_layer_sensitivity.py` | `ae750b17…5f94a855` | **no** (its output `layer_policy.json` is archived) |
| `qcomem_deployment.py` | `2f2c7b3e…977bafa6` | **no** |
| `qcomem_paged.py` | `0b7135d5…24f607d0` | **no** |
| `launch_mixed_validation_8gpu.sh` | `cf404265…6ddd1de5` | **no** (its invocation is recorded in the submission YAML) |

### 1.1 Eq. 3 quantizer and dequantizer — FOUND

`qcomem_torch.py:98` `quantize_residual` and `qcomem_torch.py:186`
`quantize_tensor`. Both compute `biases = amin`, `maxima = amax`,
`scales = (max-min)/(2^b-1)`, `q = round((x-m)/s).clamp(0, 2^b-1).to(uint8)`,
then `_pack_unsigned`. This matches Eq. 3 exactly, with one undocumented guard:
degenerate groups (`u == m`) get `s = 1`, not `s = 0`
(`qcomem_torch.py:126`, `qcomem_torch.py:234`).

Dequantizers: `PackedResidual.dequantize` (`:78`) and `PackedTensor.dequantize`
(`:163`), both unpacking via `_unpack_unsigned` (`:37`) and reconstructing in
FP32 before a final cast.

One structural difference the manuscript does not mention: the **residual** is
grouped *within the hidden dimension* (`hidden % 64 == 0` is required,
`:105-106`), whereas **cache leaves** are grouped over the *flattened element
order* with edge-value padding to a whole group (`:224-228`). For cache leaves
a group therefore may straddle head and token boundaries.

### 1.2 Group size 64 and BF16 scale/bias metadata — FOUND

`group_size: int = 64` is the default at `qcomem_torch.py:102`, `:190`, `:402`,
`:548` and is passed explicitly at `run_replay_diagnostic.py:531`. Scales and
biases are cast to BF16 at `:136-137` (residual) and `:246-247` (cache leaf).
So a `b`-bit group is `64b/8 + 4` bytes: 68 at Q8, 36 at Q4, 20 at Q2.

Confirmed empirically per item, not in aggregate — see §3.1.

### 1.3 Offline Write enumerating the Eq. 2 components — FOUND, but reflective

`TorchSplitCausalLM.write_lower_replay` (`:796`) runs layers `[0, j)`, keeps the
boundary residual and the `DynamicCache` the forward pass populated;
`LowerReplayState.quantize` (`:541`) then calls
`quantize_transformers_cache` (`:396`).

The Eq. 2 enumeration is **not** an explicit list of `K^D_ℓ, V^D_ℓ, C^D_ℓ,
G^D_ℓ`. `seed()` (`:420`) walks each layer's `dict` / `list` / `__dict__` graph
and packs *every floating-point tensor leaf* at that layer's width, cloning
non-floating leaves verbatim. In this cohort the walk reaches exactly K, V,
`conv_states` and `recurrent_states` and nothing else — confirmed because the
reconstructed byte identity in §3.1 is exact with no residual term.

A behaviour worth recording: when `cache_layer_bits` is supplied it **overrides**
`attention_bits` / `linear_bits` for the actual packing (`:454-479`); those two
fields survive only as `PackedCache` metadata and as the attention/linear
error-category split. A 7-entry vector is expanded to the 40 allocated Qwen3.5
layers by `active_cache_layer_indices` (`:297`), filling inactive layers with 16.

### 1.4 Online Read — FOUND, but it is a full materialize, not a shared fork

- dequantize + fork: `PackedLowerReplayState.fork` (`:605`) → `PackedCache.dequantize` (`:368`)
- lower-layer replay of the query: `continue_lower_replay` (`:847`)
- merge at split depth `j` and suffix execution: `run_suffix_cached_last_logits` (`:898`),
  seeded with the document residual chunk at `position_offset=0` and then the
  query residual chunk at `position_offset=document_length`
- greedy decoding: `greedy_generate_replay` (`:1062`), mirrored for timing by
  `generate_replay_with_timing` (`run_replay_diagnostic.py:297`) and
  `_finish_incremental_generation` (`:217`), `torch.argmax` with an EOS stop.

The two-chunk seeding is deliberate and documented in-code
(`qcomem_torch.py:1073-1080`): Qwen3.5 GatedDeltaNet/conv state is sensitive to
whether document and query positions arrive in one chunk or two.

### 1.5 Policy definitions and bit vectors — FOUND

All three vectors are in `run_replay_diagnostic.py` `LAYER_VALIDATION_POLICIES`
(`:116-154`):

| manuscript name | config name | line | residual | cache_layer_bits |
|---|---|---|---|---|
| frozen Q4/Q4/Q8 | `replay-d7-frozen-static` | `:141` | 4 | `(8,8,8,4,8,8,8)` |
| same-memory per-layer | `replay-d7-same-memory-mixed` | `:153` | 4 | `(8,8,4,4,8,8,8)` |
| aggressive mixed | `replay-d7-minus25-mixed` | `:159` | 4 | `(8,8,2,2,2,8,2)` |

Their provenance is the four-prompt calibration knapsack in
`aggregate_layer_sensitivity.py` (`optimize` `:20`, budgets `:80-92`), whose
output and receipt are archived:
`evidence/qcomem_mixed_validation_60item_20260812d/calibration/layer_policy.json`
(SHA-256 `f34d4b89e9936c8d58d27df69268250f7985c9f4db1d9cef4d3041a06df36e87`,
written 2026-08-11T15:52:45Z, i.e. before the validation run) and `receipt.json`
(job 233909 / trial 1827870, `policy_before_validation: true`, calibration
indices 4–5, validation indices 6–35).

### 1.6 Split depth j = 7 and the 4,096-token rule — FOUND

`j` is carried in the config name and parsed in `resolve_config`
(`run_replay_diagnostic.py:365`, depth at `:370-374`); the exactness smoke gate
runs `--depth 7` (`launch_mixed_validation_8gpu.sh:89`).

The 4,096-token cap is `prompt_parts` (`run_downstream.py:182`). The truncation
is a **middle-drop**: when the context exceeds the budget left by the rendered
chat template, the first `available//2` and the last `available - available//2`
tokens are kept (`:210-213`). Document = template prefix + truncated context;
query = template suffix. `DATASET_MAX_NEW_TOKENS = {"qasper":128, "2wikimqa":32}`
(`:53`). All cohort parameters (`MAX_INPUT_TOKENS=4096`, `LIMIT_PER_DATASET=30`,
`SOURCE_INDEX_START=6`, `SOURCE_INDEX_END=35`, `EXCLUDE_SOURCE_INDICES=4,5`) are
recorded verbatim in the archived submission YAML.

### 1.7 LongBench F1 scorer — FOUND, in two independent implementations

`run_downstream.py:84` `normalize_answer` / `:91` `answer_f1` / `:105`
`max_reference_f1`, with the per-item max over references taken at
`run_replay_diagnostic.py:567`. The archived verifier
`evidence/qcomem_mixed_validation_60item_20260812d/replay/verify.py:179,186`
reimplements it and recomputes all 360 stored F1 values from predictions and
references, requiring exact agreement. Note that punctuation stripping uses
Python `string.punctuation`, i.e. ASCII only.

### 1.8 Paired bootstrap — FOUND, in two independent implementations

`analyze_validation.py:37` `bootstrap_mean_ci`, items matched by
`(dataset, id, source_index)` in `paired_rows` (`:19`), 10,000 resamples from a
seeded `random.Random`, order statistics `int(0.025*(n-1))` and
`int(0.975*(n-1))`. Call sites with deterministic seed offsets are in
`aggregate_replay.py:351,358,377,559` (base seed 20260811). `verify.py:207`
reimplements it and requires exact equality of every archived interval.

Because indices are drawn one at a time from a seeded `Random`, the intervals
are reproducible only for this exact seed sequence *and* repetition count.

### 1.9 Store accountant — FOUND; see §3 for the defect

`stored_persistent_nbytes` = `PackedResidual.nbytes` + `PackedCache.nbytes`
(`PackedLowerReplayState.stored_nbytes`, `qcomem_torch.py:588`), recorded at
`run_replay_diagnostic.py:544-554, 591-593`.

---

## 2. Manuscript claims I could NOT find in the code — claim-narrowing candidates

These are listed prominently because each is a place where the text asserts
something the implementation does not do.

**C1 — "Q16 is the BF16 unpacked reference" (Sec. 4.2) is false for cache leaves.**
`quantize_residual` at `bits=16` *does* cast: `data=residual.to(torch.bfloat16)`
(`qcomem_torch.py:116`). `quantize_tensor` at `bits=16` does **not**: it is
`data = flat.clone()` with `original_dtype=tensor.dtype` (`:201-210`). The
`FullPrefixState` baseline never quantizes at all (`:627-629`). Qwen3.5 GDN
`recurrent_states` are FP32 by construction (see
`qcomem_qwen35_gdn_functional.py:534-540`, whose own comment says
"Transformers' torch gated-delta fallbacks accumulate recurrent state in float32
even when hidden/conv inputs are bf16"). This is the mechanism in §3 and it
propagates to the Table 1 and Table 2 row labels "Full-prefix Q16/BF16" and
"Q-CoMem split Q16/BF16".

**C2 — Table 2's caption claim that Store is "the physical byte-range union owned
by one retained entry" does not describe the printed column.** The printed
column is `persistent_document_nbytes`
(`scripts/generate_h20_deployment_table.py:109`), which
`qcomem_deployment.py:316` `persistent_components` computes as
`residual.nbytes + PackedCache.nbytes` — the *same* formula as the 60-item panel.
The byte-range union does exist (`persistent_total_resident_nbytes`, backed by
`qcomem_paged.py:380` and `_storage_nbytes` `:30-33`) but is not what Table 2
prints. Verified by reconstruction: at a 4,016-token median document the printed
15.89 / 10.01 / 9.74 / 8.41 MiB rows are exactly the format identity. The
Sec. 5.1 sentence "the accounting implementations differ" is therefore
overstated: they differ in aggregation (mean vs. median), not in formula.

**C3 — "Q16 projections" (Sec. 4.2 and the Table 2 caption) has no referent in
the code.** There is no projection-level quantization anywhere in
`qcomem_torch.py`. The per-layer policies set `linear_bits=None`, which becomes
16 only as unused metadata once `cache_layer_bits` is supplied. Nothing named
"projection" is quantized, retained or measured.

**C4 — the Sec. 4.3 ownership discipline is not the Table 1/Table 2 Read path.**
"A borrowed recurrent base is read-only at setup and must rebind to private
storage when the registered transition first mutates it; a partial KV tail is
copied before append" describes the vLLM paged implementation
(`qcomem_vllm_paged_kernel.py`), which is what the Sec. 5.5 ownership evidence
covers. The path that produced Tables 1 and 2 is
`PackedLowerReplayState.fork` (`qcomem_torch.py:605`): a full dequantize and
memo-substituted deepcopy, i.e. a *complete private materialization per query*
with no sharing, no borrowing and no copy-on-write. Both are legitimate; the
manuscript currently presents one Read section as if it described both.

**C5 — "same-memory per-layer" is not equal-memory.** `layer_policy.json` records
`budget_bytes = 6,956,256` and `predicted_bytes = 6,677,728` (96.0% of the cap);
the measured panel Store is 9.3952 vs. 9.6609 MiB/doc, i.e. 2.75% *smaller*. The
manuscript already reports the 2.75% figure in Sec. 5.4, so only the policy name
is loose.

**C6 — provenance gap.** `run_replay_diagnostic.py` (the Table 1 runner),
`analyze_validation.py` and `aggregate_replay.py` (the bootstrap and the
aggregate), `aggregate_layer_sensitivity.py` (the policy search),
`qcomem_deployment.py` / `qcomem_paged.py` (the Table 2 accountant) and
`launch_mixed_validation_8gpu.sh` are **not present in any evidence package, nor
in `supplement_anonymous/code/`**. Only `qcomem_torch.py` and
`run_downstream.py` are archived (byte-identical at four paths). A
reproducibility reviewer with artifact access cannot currently read the code
that produced Table 1's numbers, only the code it calls.

---

## 3. The Store accountant: exact mechanism, with code lines

### 3.1 What `PackedCache.nbytes` actually does — and what it does *not* do wrong

```
qcomem_torch.py:364  @property
qcomem_torch.py:365  def nbytes(self) -> int:
qcomem_torch.py:366      return cache_nbytes(self.cache)
```

`cache_nbytes` (`:266-294`) is indeed a *different* accounting path from
`PackedResidual.nbytes` / `PackedTensor.nbytes` (`:63-69`, `:156-161`), exactly
as A1 said:

```
qcomem_torch.py:276          if isinstance(value, torch.Tensor):
qcomem_torch.py:277              key = (
qcomem_torch.py:278                  str(value.device),
qcomem_torch.py:279                  value.untyped_storage().data_ptr(),
qcomem_torch.py:280                  value.untyped_storage().nbytes(),
qcomem_torch.py:281              )
qcomem_torch.py:282              if key in seen_tensors:
qcomem_torch.py:283                  return 0
qcomem_torch.py:284              seen_tensors.add(key)
qcomem_torch.py:285              return value.untyped_storage().nbytes()
```

versus

```
qcomem_torch.py:15   def tensor_nbytes(tensor: torch.Tensor | None) -> int:
qcomem_torch.py:18       return tensor.numel() * tensor.element_size()
```

So `cache_nbytes` measures *physical storage extents, deduplicated by
`data_ptr`*, while the residual accountant measures *logical element payload*.
The hazard A1 identified is real in principle. **It does not fire here**, for two
reasons that I checked in the code and then confirmed numerically:

1. **Nothing is deduplicated away.** After `quantize_transformers_cache`, every
   packed leaf's `data`, `scales` and `biases` are freshly allocated tensors that
   are all simultaneously alive inside the returned `PackedCache`, so no two can
   share a `data_ptr`. `PackedTensor` is an unslotted dataclass, so `visit()`
   descends into it through the `hasattr(value, "__dict__")` branch (`:290-291`)
   and reaches all three sidecars.
2. **Storage extent equals logical payload.** `_pack_unsigned` allocates exactly
   `ceil(n/64)*64*b/8` bytes and the BF16 sidecars exactly `2*ceil(n/64)` bytes
   each. In this cohort every component's element count is a multiple of 64, so
   even the padding path (`:224-228`) never fires.

**Numerical proof.** I reconstructed every archived byte field from first
principles and compared to the shards:

```
Q16   lower cache  = 2*(512T)*2  +  6*(32768*2 + 524288*4)
Qb    component    = ceil(n/64) * (64b/8 + 4)
full-prefix        = 10*(2*(512T)*2) + 30*(32768*2 + 524288*4)
```

with `T` = `document_tokens`, hidden = 2048, layer 3 the sole lower
full-attention layer (`num_kv_heads * head_dim = 512`), and six lower GDN layers
each holding 32,768 conv elements and 524,288 recurrent elements.

Result over `evidence/qcomem_mixed_validation_60item_20260812d/artifacts/raw/`:

| field | items checked | mismatches |
|---|---|---|
| `stored_residual_nbytes`, Q16 (`= 2048T*2`) | 60 | **0** |
| `stored_residual_nbytes`, Q4 (`= ceil(n/64)*36`) | 60 | **0** |
| `stored_lower_cache_nbytes`, split Q16 | 60 | **0** |
| `stored_lower_cache_nbytes`, frozen `[8,8,8,4,8,8,8]` | 60 | **0** |
| `stored_lower_cache_nbytes`, same-memory `[8,8,4,4,8,8,8]` | 60 | **0** |
| `stored_lower_cache_nbytes`, aggressive `[8,8,2,2,2,8,2]` | 60 | **0** |
| `stored_persistent_nbytes`, full prefix | 60 | **0** |

**Therefore A1's stated mechanism is refuted.** The lower-cache counter does not
omit the 4 bytes per group of BF16 scale and bias. It includes them, exactly, in
every quantized arm, on every item. (A1's *observation* — that 3.6164× and
3.8007× are impossible against a BF16 reference — is correct and important; only
the attribution was wrong.)

### 3.2 Where the bytes actually come from

```
qcomem_torch.py:110      if bits == 16:                        # quantize_residual
qcomem_torch.py:111          return PackedResidual(
...
qcomem_torch.py:116              data=residual.to(torch.bfloat16),      # <-- CASTS
```

```
qcomem_torch.py:201      if bits == 16:                        # quantize_tensor
qcomem_torch.py:202          data = flat.clone()                        # <-- DOES NOT CAST
qcomem_torch.py:203          return PackedTensor(
...
qcomem_torch.py:207              original_dtype=tensor.dtype,
```

```
qcomem_torch.py:627      @property
qcomem_torch.py:628      def stored_nbytes(self) -> int:       # FullPrefixState
qcomem_torch.py:629          return cache_nbytes(self.cache)   # raw, never quantized
```

Qwen3.5 GDN recurrent state is FP32:

```
qcomem_qwen35_gdn_functional.py:534   # Transformers' torch gated-delta fallbacks accumulate recurrent state in
qcomem_qwen35_gdn_functional.py:535   # float32 even when hidden/conv inputs are bf16.
qcomem_qwen35_gdn_functional.py:536   recurrent_state = torch.zeros(
qcomem_qwen35_gdn_functional.py:538       dtype=torch.float32,
```

So the Q16 arm counts `524,288 × 4 = 2,097,152` bytes per GDN recurrent state,
while every quantized arm counts `ceil(524288/64) × (64b/8 + 4)` — a number
derived from the *element count*, which implicitly assumes a 2-byte reference.
The mismatch is **exactly 524,288 × 2 = 1,048,576 bytes = 1.0000 MiB per GDN
layer**.

- below the split (`j=7`, 6 GDN layers): **+6.0000 MiB/doc**
- full prefix (40 layers, 30 GDN): **+30.0000 MiB/doc**

### 3.3 Independent, self-contained corroboration inside an archived artifact

`evidence/qcomem_mixed_validation_60item_20260812d/calibration/layer_policy.json`
records `mean_component_nbytes` per component per width. It shows the defect on
its own, with no reconstruction required:

| component | Q16 | Q8 | Q4 | Q2 | Q16/Q4 | Q16/Q8 |
|---|---|---|---|---|---|---|
| `residual` (BF16) | 8,071,168 | 4,287,808 | 2,270,016 | 1,261,120 | **3.5556** | 1.8824 |
| `cache.3` (full attention, BF16) | 4,035,584 | 2,143,904 | 1,135,008 | 630,560 | **3.5556** | 1.8824 |
| `cache.0/1/2/4/5/6` (GDN) | 2,162,688 | 591,872 | 313,344 | 174,080 | 6.9020 | **3.6540** |

The Eq. 3 ceilings are 3.5556× at Q4 and 1.8824× at Q8. The BF16 components hit
them exactly. The GDN rows exceed both — 3.6540× for a *16→8 bit* step is
arithmetically impossible for a 2-byte reference. `2,162,688 = 32,768×2 +
524,288×4`, which is the FP32 recurrent state, not a BF16 one.

Note that the policy *search* is unaffected: `option_bytes`
(`aggregate_layer_sensitivity.py:53`) only ever compares the `b < 16` rows, which
are format-correct. The three bit vectors stand.

### 3.4 Corrected formula

The accountant should measure one thing. The consistent definition, matching
Eq. 3 and Figure 2c, is a BF16 reference:

```
Store_BF16(policy) =  sum over components c:
                        ceil(n_c / 64) * (64 * b_c / 8 + 4)      if b_c < 16
                        n_c * 2                                   if b_c == 16
```

where `n_c` is the element count of the boundary residual and of each
floating-point cache leaf. The two code changes that implement it are:

1. `qcomem_torch.py:202` → `data = flat.to(torch.bfloat16).clone()`, mirroring
   `quantize_residual` at `:116`. (`PackedTensor.original_dtype` must then record
   the reference dtype, or `dequantize` will no longer round-trip FP32 leaves —
   which is itself a numerical decision the paper should state, since it makes
   the Q16 arm lossy for GDN recurrent state.)
2. `FullPrefixState.stored_nbytes` (`:627-629`) must use the same
   BF16-normalised count instead of raw `cache_nbytes`, or Table 1's baseline row
   stays on a different footing from every other row.

If instead the paper prefers to keep physical bytes, then `Store_phys` must be
used on **both** sides and the Q16/BF16 labels must go. That option is discussed
in §3.6.

### 3.5 Corrected numbers (recomputed from the 48 archived shards)

Cohort: 60 items, mean document length 3807.25 tokens (min 1146, max 4050).

| row | published Store MiB/doc | BF16-normalised | published ×vs prefix | corrected ×vs prefix | published % reduction | corrected % reduction |
|---|---|---|---|---|---|---|
| Full-prefix Q16/BF16 | 136.2354 | **106.2354** | 1.00× | 1.00× | — | — |
| Q-CoMem split Q16/BF16 | 34.6831 | **28.6831** | 3.9280× | **3.7038×** | 74.54% | 73.00% |
| Q-CoMem frozen Q4/Q4/Q8 | 9.6609 | 9.6609 | 14.1018× | **10.9965×** | 92.9087% | **90.9062%** |
| Q-CoMem same-memory mixed | 9.3952 | 9.3952 | 14.5005× | **11.3073×** | 93.10% | 91.16% |
| Q-CoMem aggressive mixed | 7.5361 | 7.5361 | 18.0778× | **14.0969×** | 94.47% | 92.91% |

Lower-cache-only compression, which is what A1 flagged as impossible:

| policy | published | BF16-normalised | finest-layer Eq. 3 ceiling | verdict after correction |
|---|---|---|---|---|
| frozen `[8,8,8,4,8,8,8]` | 3.6164× | **2.5211×** | 3.5556× | within format |
| same-memory `[8,8,4,4,8,8,8]` | 3.8007× | **2.6496×** | 3.5556× | within format |
| aggressive `[8,8,2,2,2,8,2]` | 5.9079× | **4.1187×** | 6.4000× | within format |

Per-component decomposition at the cohort mean (this is the table A1 said the
archived shards could not produce; it is recoverable analytically and was
verified item-by-item):

| component | elements | Q16 as counted | Q16 if BF16-referenced | Q8 | Q4 | Q2 |
|---|---|---|---|---|---|---|
| boundary residual `h_7^D` | 2048·T | 14.8721 | 14.8721 | 7.9008 | 4.1828 | 2.3238 |
| layer-3 K | 512·T | 3.7180 | 3.7180 | 1.9752 | 1.0457 | 0.5809 |
| layer-3 V | 512·T | 3.7180 | 3.7180 | 1.9752 | 1.0457 | 0.5809 |
| GDN `conv_states` (per layer) | 32,768 | 0.0625 | 0.0625 | 0.0332 | 0.0176 | 0.0098 |
| GDN `recurrent_states` (per layer) | 524,288 | **2.0000** | **1.0000** | 0.5312 | 0.2812 | 0.1562 |

(MiB, T = 3807.25.)

Table 2 is affected identically. Reconstructing at its 4,016-token median
document reproduces the printed 15.89 / 10.01 / 9.74 / 8.41 MiB rows exactly and
shows that 30.0000 MiB of the 140.34 MiB reference is FP32 recurrent state.
BF16-normalised, the reference is **110.34 MiB** and the reductions become
85.60% (Q8), 90.93% (Q4/Q4/Q8), 91.17% (per-layer) and 92.38% (Q4), against the
printed 88.68 / 92.87 / 93.06 / 94.01%.

### 3.6 Bounding the claim honestly

Two readings are defensible, and the paper must pick one explicitly.

**(A) BF16 reference (recommended, and the one the paper's own Eq. 3 and Fig. 2c
imply).** Apply §3.4. Headline becomes **11.00×** / 90.91%, split-Q16 **3.70×**.
Row labels "Q16/BF16" then become true.

**(B) Physical native-dtype reference.** 136.2354 and 9.6609 MiB are both honest
counts of bytes actually resident, so 14.1018× is an arithmetically correct
*physical* ratio. But then the reference is not Q16/BF16, and the paper must say
that roughly 22% of the 136.235 MiB baseline (30.0 of 136.2) is FP32 GDN
recurrent state, and that the corresponding part of the reduction is an
FP32→low-bit dtype narrowing rather than the Eq. 3 Q16→Q4/Q8 step the method
section describes. Under (B) the sentence "Q16 is the BF16 unpacked reference"
must be deleted and both table row labels changed.

**What is not defensible is the current combination**: BF16 labels, an Eq. 3
narrative, and a native-dtype denominator.

### 3.7 Where this leaves A1

- A1's finding that the frozen and same-memory lower-cache ratios are impossible
  against a BF16 reference: **confirmed**, and now explained.
- A1's residual-side verification (180/180 format identities, exact 3.555556
  Q16/Q4 ratio): **confirmed**, and extended — the lower-cache counter satisfies
  the same identity, which A1 assumed it did not.
- A1's proposed mechanism (lower-cache counter omits the 4 bytes/group of BF16
  scale and bias): **refuted**, 0/180 mismatches.
- A1's corrected range "about 13.2–13.6×" and its strict bound "≤ 13.9662×":
  **superseded**. Those follow from the metadata-omission hypothesis. The
  consistent-reference value is **10.9965×**, and 14.1018× is not "wrong" so much
  as measured against an undisclosed denominator.
- A1's handoff request — "emitting the authoritative per-component table requires
  the Store accountant itself" — is **discharged** by §3.5 without re-running any
  GPU work: the per-component breakdown is uniquely determined by the archived
  shards plus the archived calibration table, and was verified with zero
  mismatches on 60/60 items and 5/5 configurations.

---

## 4. Verification test for this document

Re-run against `evidence/qcomem_mixed_validation_60item_20260812d/artifacts/raw/`
and require:

1. `stored_lower_cache_nbytes` equals the component sum of
   `ceil(n/64)*(64b/8+4)` for **180/180** quantized item-policy pairs (0 mismatches).
2. `stored_lower_cache_nbytes` for `replay-d7-layer-q16` equals
   `2*(512T)*2 + 6*(32768*2 + 524288*4)` for **60/60** items.
3. `stored_persistent_nbytes` for `prefix` equals
   `10*(2*(512T)*2) + 30*(32768*2 + 524288*4)` for **60/60** items.
4. The difference between (2) and its BF16 normalisation is exactly
   `6 * 524288 * 2 = 6,291,456` bytes per item; for (3), exactly
   `30 * 524288 * 2 = 31,457,280` bytes per item.
5. `layer_policy.json` GDN Q16/Q8 ratio = 3.6540 > 1.8824, while its `residual`
   and `cache.3` Q16/Q4 ratios both equal 3.5556.

Then require the corrected means: 106.2354, 28.6831, 9.6609, 9.3952, 7.5361
MiB/doc, and 10.9965× for the frozen headline.
