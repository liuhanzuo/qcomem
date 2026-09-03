# A2 -- Single-reference-arm re-analysis of the 60-item Q-CoMem validation cohort

**Action:** `A2` of `review/round_44/meta_review.json` (issues `R44-4-01`, `R44-4-08`, `R44-4-16`, `Q5`, `T-11`).  
**Date:** 2026-09-02.  
**Nature:** pure re-analysis of already-archived item-level F1. No new experiment, no GPU execution, no new measurement of any kind.  
**Scope guard:** this file and `revision/a2_reference_arm_20260902.json` are the only artifacts produced. `main_r44_structure.tex`, `tables/`, and `evidence/` are untouched.

---

## 0. Headline finding

The comparison a deployment actually faces -- frozen Q4/Q4/Q8 versus full-prefix Q16 -- is

> **-0.4455 F1 points, 95% paired bootstrap interval [-2.0586, +0.9907]**, at **14.10x** smaller retained state (136.235 -> 9.661 MiB/document).

Both halves of that sentence are now measured against the same arm. The published headline paired the 14.10x (measured against full-prefix Q16) with -0.39 [-2.04, +1.08] (measured against split Q16).

**None of the 24 paired-bootstrap intervals in the archived analysis file uses full-prefix Q16 as its reference arm.** The archived reference arms are ['dense', 'split Q16']. The interval above did not previously exist anywhere in the artifact or the manuscript; it is computed here for the first time.

---

## 1. Method

| Item | Value |
|---|---|
| Data source | `evidence/qcomem_mixed_validation_60item_20260812d/artifacts/raw, 48 shards` |
| Shards read | 48 (8 shards x 6 configurations) |
| Shard-set SHA-256 | `55862618fc56f5cdbc05fc4f513127e1e38681bdbc7b02dd5fe93aaa4c59de3e` |
| Item key | `(dataset, source_index)`; 60 keys common to all six arms |
| Cohort | 30 Qasper + 30 2WikiMQA |
| Scale | archived `f1` x 100 (the paper's 0--100 scale) |
| **Resampling unit** | **item** (paired: one index vector drawn per resample, applied to both arms) |
| **Resamples** | **10000** |
| **Seed** | **20260902** |
| RNG | `numpy.random.default_rng (PCG64)` |
| Interval | percentile, 95% (2.5th / 97.5th percentiles of the resampled mean paired difference) |
| Reference arm | `prefix` = full-prefix Q16/BF16, for every row of every table below |

The estimand is `mean(F1[config] - F1[reference])` over the 60 paired items. The bootstrap resamples the 60 item indices with replacement and recomputes that mean; both arms are indexed by the same draw, so the pairing is preserved. This matches the description in the paper ("paired 10,000-resample bootstrap intervals"), and Section 3.3 below confirms it reproduces the archived intervals to within Monte-Carlo noise.

A seed is stated because the archived analysis does not record one; the interval endpoints are therefore Monte-Carlo random variables. Section 3.4 quantifies how much they move across seeds.

---

## 2. Reproduction check against the published values

### 2.1 Mean F1 (0--100)

| Configuration | Recomputed mean F1 | Rounded | Published | Match |
|---|---:|---:|---:|:--:|
| No-cache dense recompute | 54.2888 | 54.29 | 54.29 | yes |
| Full-prefix Q16/BF16 | 54.6823 | 54.68 | 54.68 | yes |
| Q-CoMem split Q16/BF16 | 54.6238 | 54.62 | 54.62 | yes |
| Q-CoMem frozen Q4/Q4/Q8 | 54.2368 | 54.24 | 54.24 | yes |
| Q-CoMem same-memory mixed | 53.3648 | 53.36 | 53.36 | yes |
| Q-CoMem aggressive mixed | 49.1862 | 49.19 | 49.19 | yes |

**All six published means reproduce exactly.**

### 2.2 Store (MiB/document)

| Configuration | Recomputed | Published |
|---|---:|---:|
| No-cache dense recompute | -- | -- |
| Full-prefix Q16/BF16 | 136.2354 | 136.235 |
| Q-CoMem split Q16/BF16 | 34.6831 | 34.683 |
| Q-CoMem frozen Q4/Q4/Q8 | 9.6609 | 9.661 |
| Q-CoMem same-memory mixed | 9.3952 | 9.395 |
| Q-CoMem aggressive mixed | 7.5361 | 7.536 |

All Store values reproduce. Compression of frozen Q4/Q4/Q8 versus full-prefix Q16 is 14.101763x, i.e. the published 14.10x. Dense recompute has no Store value (`stored_persistent_nbytes` is null in every dense row), which is why the paper prints `--`.

### 2.3 The published Table 1 deltas, against the arm the paper actually used (split Q16)

| Configuration | Recomputed delta | Published delta | Recomputed 95% CI | Archived 95% CI | Published 95% CI |
|---|---:|---:|---|---|---|
| No-cache dense recompute | -0.3350 | -0.33 | [-4.6063, +2.6634] | [-4.6063, +2.6650] | [-4.61, +2.67] |
| Full-prefix Q16/BF16 | +0.0585 | +0.06 | [+0.0000, +0.1754] | [+0.0000, +0.1754] | [+0.00, +0.18] |
| Q-CoMem split Q16/BF16 | +0.0000 | +0.00 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | [+0.00, +0.00] |
| Q-CoMem frozen Q4/Q4/Q8 | -0.3870 | -0.39 | [-1.9943, +1.0496] | [-2.0365, +1.0756] | [-2.04, +1.08] |
| Q-CoMem same-memory mixed | -1.2590 | -1.26 | [-4.2734, +0.9853] | [-4.2793, +0.9775] | [-4.28, +0.98] |
| Q-CoMem aggressive mixed | -5.4376 | -5.44 | [-11.5195, -0.1093] | [-11.5615, -0.1713] | [-11.56, -0.17] |

**Point estimates reproduce to machine precision** (max absolute difference from the archived analysis: 6.11e-16 F1 points).

**The published interval endpoints are the archived endpoints, correctly rounded** -- for all six rows (`published_matches_archived_to_2dp` is true for every row in the JSON).

**My independently seeded intervals differ from the archived ones by at most 0.0621 F1 points** (largest on the aggressive-mixed row). This is Monte-Carlo resampling noise, not a disagreement: the archived analysis does not record its bootstrap seed, so exact endpoint reproduction is not possible. Section 3.4 shows the across-seed spread is of the same magnitude.

### 2.4 T-11's arithmetic note -- resolved, and it is a rounding artifact

REV-2 observed that `54.24 - 54.62 = -0.38` while Table 1 prints `-0.39`, and `49.19 - 54.62 = -5.43` while Table 1 prints `-5.44`. From the unrounded values:

- frozen: `54.2368 - 54.6238 = -0.3870` -> rounds to `-0.39`. The printed delta is correct.
- aggressive: `49.1862 - 54.6238 = -5.4376` -> rounds to `-5.44`. The printed delta is correct.

The table is arithmetically sound; it is only non-reproducible from its own two-decimal columns. Printing four decimals in the mean-F1 column, or printing the delta with its own precision, removes the objection.

---

## 3. The single-reference-arm table (full-prefix Q16 throughout)

### 3.1 Pooled and per-dataset, every configuration versus full-prefix Q16

All deltas and intervals below use **one** reference arm: full-prefix Q16/BF16. Store and compression already used that arm, so memory and quality are now commensurable.

| Configuration | Store (MiB/doc) | vs prefix | Mean F1 | dF1 | 95% CI |
|---|---:|---:|---:|---:|---|
| No-cache dense recompute | -- | -- | 54.2888 | -0.3934 | [-4.6648, +2.6049] |
| Full-prefix Q16/BF16 | 136.235 | 1.00x | 54.6823 | +0.0000 | [+0.0000, +0.0000] |
| Q-CoMem split Q16/BF16 | 34.683 | 3.93x | 54.6238 | -0.0585 | [-0.1754, +0.0000] |
| Q-CoMem frozen Q4/Q4/Q8 | 9.661 | 14.10x | 54.2368 | -0.4455 | [-2.0586, +0.9907] |
| Q-CoMem same-memory mixed | 9.395 | 14.50x | 53.3648 | -1.3174 | [-4.3322, +0.9193] |
| Q-CoMem aggressive mixed | 7.536 | 18.08x | 49.1862 | -5.4961 | [-11.5805, -0.1984] |

**Per-dataset breakdown** (n=30 each; the paper currently reports none, despite 360 archived item-level F1 values):

| Configuration | Qasper mean | Qasper dF1 | Qasper 95% CI | 2WikiMQA mean | 2WikiMQA dF1 | 2WikiMQA 95% CI | Signs agree |
|---|---:|---:|---|---:|---:|---|:--:|
| No-cache dense recompute | 43.6094 | +1.2131 | [-0.2339, +3.3305] | 64.9683 | -2.0000 | [-10.0000, +4.0000] | **NO** |
| Full-prefix Q16/BF16 | 42.3962 | +0.0000 | [+0.0000, +0.0000] | 66.9683 | +0.0000 | [+0.0000, +0.0000] | n/a |
| Q-CoMem split Q16/BF16 | 42.2793 | -0.1170 | [-0.3509, +0.0000] | 66.9683 | +0.0000 | [+0.0000, +0.0000] | yes |
| Q-CoMem frozen Q4/Q4/Q8 | 42.0609 | -0.3354 | [-3.3818, +2.3932] | 66.4127 | -0.5556 | [-1.6667, +0.0000] | yes |
| Q-CoMem same-memory mixed | 42.1423 | -0.2539 | [-3.3148, +2.3720] | 64.5873 | -2.3810 | [-7.1429, +0.0000] | yes |
| Q-CoMem aggressive mixed | 38.4518 | -3.9445 | [-11.4709, +1.5691] | 59.9206 | -7.0476 | [-17.7143, +1.3333] | yes |

Two things follow directly.

1. **The frozen policy's sign is consistent across the two datasets** (-0.3354 on Qasper, -0.5556 on 2WikiMQA). R44-4-08's worry that a pooled small negative might hide opposite-signed per-dataset effects does **not** materialise for the headline row. Both per-dataset intervals cross or touch zero.
2. **It does materialise for the dense arm**: +1.2131 on Qasper versus -2.0000 on 2WikiMQA -- opposite signs, pooling to -0.3934. Any statement about dense based on the pooled number alone is uninterpretable. See Section 4.

Arithmetic sanity: because the two datasets have equal n, the pooled delta is exactly the mean of the two per-dataset deltas; for the frozen row `(-0.335375 + -0.555556)/2 = -0.445465` = the pooled -0.445465.

### 3.2 The two deltas compound, they do not cancel

- full-prefix Q16 -> split Q16: `-0.0585`
- split Q16 -> frozen Q4/Q4/Q8: `-0.3870`
- full-prefix Q16 -> frozen Q4/Q4/Q8: `-0.0585 + -0.3870 = -0.4455`

Means are linear, so this is exact, not approximate. REV-4's reading is correct: the split step carries its own small non-positive cost and the quantization step adds to it. The headline figure quoted against split Q16 (-0.3870) understates the deployment-relevant loss by 0.0585 points, about 13% of the true value.

### 3.3 All intervals in the archive, and which arm each uses

The Reproducibility Statement says 24 paired-bootstrap intervals were computed. I enumerated `replay_analysis.json` and found exactly **24**, distributed as:

| Reference arm | Pooled | Per-dataset | Total |
|---|---:|---:|---:|
| dense | 6 | 12 | 18 |
| split Q16 | 6 | 0 | 6 |
| **full-prefix Q16** | **0** | **0** | **0** |

So the 24 intervals are: 6 configurations x (pooled-vs-dense, pooled-vs-split-Q16) = 12, plus 6 configurations x 2 datasets vs dense = 12. Table 1 displays six of them (the pooled vs-split-Q16 column). The 12 per-dataset intervals that exist in the archive are all against **dense**, and none of the 24 is against full-prefix Q16.

Full enumeration (values x100, as printed in the paper's scale):

| # | Configuration | Reference arm | Split | 95% CI | Shown in Table 1 |
|---:|---|---|---|---|:--:|
| 1 | dense | dense | pooled | [+0.0000, +0.0000] | no |
| 2 | dense | split Q16 | pooled | [-4.6063, +2.6650] | yes |
| 3 | dense | dense | 2wikimqa | [+0.0000, +0.0000] | no |
| 4 | dense | dense | qasper | [+0.0000, +0.0000] | no |
| 5 | prefix | dense | pooled | [-2.6066, +4.5088] | no |
| 6 | prefix | split Q16 | pooled | [+0.0000, +0.1754] | yes |
| 7 | prefix | dense | 2wikimqa | [-4.0000, +10.0000] | no |
| 8 | prefix | dense | qasper | [-3.3305, +0.2339] | no |
| 9 | replay-d7-layer-q16 | dense | pooled | [-2.6650, +4.4572] | no |
| 10 | replay-d7-layer-q16 | split Q16 | pooled | [+0.0000, +0.0000] | yes |
| 11 | replay-d7-layer-q16 | dense | 2wikimqa | [-4.0000, +10.0000] | no |
| 12 | replay-d7-layer-q16 | dense | qasper | [-3.4475, +0.0000] | no |
| 13 | replay-d7-frozen-static | dense | pooled | [-3.3045, +4.2800] | no |
| 14 | replay-d7-frozen-static | split Q16 | pooled | [-2.0365, +1.0756] | yes |
| 15 | replay-d7-frozen-static | dense | 2wikimqa | [-4.3333, +9.4444] | no |
| 16 | replay-d7-frozen-static | dense | qasper | [-4.6073, +0.7317] | no |
| 17 | replay-d7-same-memory-mixed | dense | pooled | [-5.2845, +3.7838] | no |
| 18 | replay-d7-same-memory-mixed | split Q16 | pooled | [-4.2793, +0.9775] | yes |
| 19 | replay-d7-same-memory-mixed | dense | 2wikimqa | [-8.4762, +8.6667] | no |
| 20 | replay-d7-same-memory-mixed | dense | qasper | [-4.9649, +1.6662] | no |
| 21 | replay-d7-minus25-mixed | dense | pooled | [-11.9941, +1.5786] | no |
| 22 | replay-d7-minus25-mixed | split Q16 | pooled | [-11.5615, -0.1713] | yes |
| 23 | replay-d7-minus25-mixed | dense | 2wikimqa | [-16.7619, +6.6667] | no |
| 24 | replay-d7-minus25-mixed | dense | qasper | [-12.9395, +1.0689] | no |

Cross-check of the 12 vs-dense intervals I recomputed against the archived ones:

| Configuration | Split | Recomputed (seed 20260902) | Archived |
|---|---|---|---|
| No-cache dense recompute | qasper | [+0.0000, +0.0000] | [+0.0000, +0.0000] |
| No-cache dense recompute | 2wikimqa | [+0.0000, +0.0000] | [+0.0000, +0.0000] |
| Full-prefix Q16/BF16 | qasper | [-3.3305, +0.2339] | [-3.3305, +0.2339] |
| Full-prefix Q16/BF16 | 2wikimqa | [-4.0000, +10.0000] | [-4.0000, +10.0000] |
| Q-CoMem split Q16/BF16 | qasper | [-3.4475, +0.0000] | [-3.4475, +0.0000] |
| Q-CoMem split Q16/BF16 | 2wikimqa | [-4.0000, +10.0000] | [-4.0000, +10.0000] |
| Q-CoMem frozen Q4/Q4/Q8 | qasper | [-4.5209, +0.7275] | [-4.6073, +0.7317] |
| Q-CoMem frozen Q4/Q4/Q8 | 2wikimqa | [-4.3333, +10.0000] | [-4.3333, +9.4444] |
| Q-CoMem same-memory mixed | qasper | [-5.0123, +1.6878] | [-4.9649, +1.6662] |
| Q-CoMem same-memory mixed | 2wikimqa | [-8.4762, +8.6667] | [-8.4762, +8.6667] |
| Q-CoMem aggressive mixed | qasper | [-13.1148, +1.0039] | [-12.9395, +1.0689] |
| Q-CoMem aggressive mixed | 2wikimqa | [-16.7619, +6.6667] | [-16.7619, +6.6667] |

Agreement is exact or within Monte-Carlo noise on every row.

### 3.4 Seed sensitivity of the interval endpoints

Because a seed had to be chosen, here is the same pooled-vs-prefix interval under five seeds:

| Configuration | seed 20260902 | seed 1 | seed 7 | seed 12345 | seed 99991 |
|---|---|---|---|---|---|
| No-cache dense recompute | [-4.665, +2.605] | [-4.607, +2.607] | [-4.606, +2.547] | [-4.665, +2.607] | [-4.605, +2.663] |
| Full-prefix Q16/BF16 | [+0.000, +0.000] | [+0.000, +0.000] | [+0.000, +0.000] | [+0.000, +0.000] | [+0.000, +0.000] |
| Q-CoMem split Q16/BF16 | [-0.175, +0.000] | [-0.175, +0.000] | [-0.175, +0.000] | [-0.175, +0.000] | [-0.175, +0.000] |
| Q-CoMem frozen Q4/Q4/Q8 | [-2.059, +0.991] | [-2.068, +1.006] | [-2.043, +1.013] | [-2.049, +1.040] | [-2.118, +1.028] |
| Q-CoMem same-memory mixed | [-4.332, +0.919] | [-4.396, +0.909] | [-4.428, +0.928] | [-4.309, +0.891] | [-4.394, +0.909] |
| Q-CoMem aggressive mixed | [-11.581, -0.198] | [-11.802, -0.180] | [-11.797, -0.232] | [-11.779, -0.126] | [-11.768, -0.259] |

For the headline frozen row the across-seed spread is 0.075 points on the lower endpoint and 0.049 on the upper. The reported interval should therefore be quoted to two decimals at most, and the seed should be stated in the caption.

---

## 4. Dense versus full-prefix Q16: investigating the CI-width anomaly (Q5)

REV-1 flagged that dense recomputation and exact prefix caching should be numerically near-equivalent, yet Table 1 shows paired CI widths of 7.28 (dense) versus 0.18 (full-prefix) against split Q16. Here is what the raw rows say.

### 4.1 The direct paired comparison, never previously reported

| Comparison | Paired delta | 95% CI | CI width | SD of per-item delta |
|---|---:|---|---:|---:|
| dense - full-prefix Q16 | -0.3934 | [-4.6648, +2.6049] | 7.2698 | 14.5124 |
| full-prefix Q16 - split Q16 | +0.0585 | [+0.0000, +0.1754] | 0.1754 | 0.4530 |
| dense - split Q16 | -0.3350 | [-4.6063, +2.6634] | 7.2698 | 14.5067 |

Answering REV-1's verification test directly: the point estimate **is** near zero (-0.3934) and the interval contains zero, but the interval is **not** narrow -- its width is 7.27 points, wider than the frozen policy's interval against the same arm (3.05). So exact prefix caching does reproduce dense recomputation in expectation on this stack, but not item-by-item. Sections 4.2--4.3 establish why.

### 4.2 It is not an execution-boundary artifact

The Table 18 footnote attributes the dense arm's behaviour to "a different document/query execution boundary." **The archived rows do not support that explanation.** I compared, item-by-item across all 60 items and all six arms, every field that would record such a difference:

`prefix_tokens`, `context_tokens`, `original_context_tokens`, `document_tokens`, `query_tokens`, `input_tokens`, `max_new_tokens`, `context_truncated`, `id`, `question`, `references`

| Configuration | All boundary fields identical to dense |
|---|:--:|
| No-cache dense recompute | yes |
| Full-prefix Q16/BF16 | yes |
| Q-CoMem split Q16/BF16 | yes |
| Q-CoMem frozen Q4/Q4/Q8 | yes |
| Q-CoMem same-memory mixed | yes |
| Q-CoMem aggressive mixed | yes |

Every arm, including dense, runs identical per-item boundaries: prefix token counts of 85 (Qasper) and 33 (2WikiMQA), the same per-item document, query and input token counts (input_tokens ranges 1207--4096 across items and is identical across arms item-by-item), the same 49-of-60 truncation decisions, and the same generation caps (128 Qasper / 32 2WikiMQA). Whatever the footnote means, it is not visible as a document/query boundary difference in the archived rows, and the footnote as written is not a supported explanation of the CI width.

### 4.3 What actually drives the width: a handful of greedy-decoding token flips

Dense and full-prefix produce **identical generated token sequences on 54 of 60 items**. They diverge on 6. Those 6 items:

| Item | Dense F1 | Prefix F1 | dF1 (dense - prefix) | First divergent token position | Dense tokens / Prefix tokens |
|---|---:|---:|---:|---:|---:|
| `2wikimqa#12` | 100.00 | 60.00 | +40.00 | 0 | 7 / 7 |
| `2wikimqa#33` | 0.00 | 100.00 | -100.00 | 0 | 1 / 1 |
| `qasper#8` | 55.17 | 38.89 | +16.28 | 7 | 21 / 32 |
| `qasper#10` | 63.16 | 66.67 | -3.51 | 6 | 17 / 15 |
| `qasper#30` | 47.62 | 24.00 | +23.62 | 0 | 51 / 39 |
| `qasper#31` | 14.29 | 14.29 | +0.00 | 13 | 17 / 16 |

Two of them are short-answer 2WikiMQA items where a single flipped token moves F1 by the maximum possible amount:

- `2wikimqa#12` -- gold answer `Guy II, Count of Soissons`: dense answers `Guy II, Count of Soissons` (F1 100); full-prefix answers `John I, Count of Soissons` (F1 60); dF1 +40. The two arms disagree on the single generated token.
- `2wikimqa#33` -- gold answer `yes`: dense answers `No` (F1 0); full-prefix answers `Yes` (F1 100); dF1 -100. The two arms disagree on the single generated token.

By contrast, full-prefix and split Q16 diverge on only 2 items, both Qasper, both long-form, with a maximum absolute per-item swing of 3.51 points:

| Item | Prefix F1 | Split Q16 F1 | dF1 | First divergent token position |
|---|---:|---:|---:|---:|
| `qasper#10` | 66.67 | 63.16 | +3.51 | 6 |
| `qasper#31` | 14.29 | 14.29 | +0.00 | 13 |

### 4.4 The explanation, quantified

| | dense vs full-prefix | full-prefix vs split Q16 |
|---|---:|---:|
| Items with divergent token sequences | 6 / 60 | 2 / 60 |
| Max absolute per-item dF1 | 100.00 | 3.51 |
| SD of per-item dF1 | 14.5124 | 0.4530 |
| 95% CI width | 7.2698 | 0.1754 |

The SD ratio is 32.0x and the CI-width ratio is 41.4x. A paired bootstrap interval scales with the SD of the per-item difference, so a ~32x larger SD producing a ~41x wider interval is expected behaviour, not an anomaly in the statistics. The residual gap between 32 and 41 comes from the shape of the difference distribution: for full-prefix vs split Q16, 58 of 60 per-item differences are exactly zero and the remaining mass is one-sided, so the percentile interval is truncated at zero on one end.

### 4.5 Conclusion of the investigation, stated plainly

REV-1's premise is right and the paper's implicit story is wrong in an interesting way:

- Exact prefix caching is **not** producing bit-identical decoding to dense recomputation on this stack. With the document/query boundary, inputs, references and generation caps all verified identical (Section 4.2), the only mechanism the archived rows leave available is floating-point nondeterminism between the two execution paths (single full prefill versus cached-prefix continuation) flipping a near-tie argmax under greedy decoding. This is an inference from the archived data, not a directly measured quantity -- confirming it would need logit-level instrumentation, which is not archived. What is directly measured is that the two arms emit different token sequences on 6 of 60 items with everything else held identical.
- The dense arm's wide interval is therefore a **decoding-sensitivity** effect, concentrated in a few items, amplified by 2WikiMQA's short answers where per-item F1 is nearly binary. It is not evidence that dense and exact caching differ in expectation: the point estimate (-0.3934) is small and the interval contains zero.
- The genuinely surprising fact -- worth stating in the paper -- is the **opposite** one: Q-CoMem split Q16 agrees with full-prefix Q16 on 58/60 generated token sequences, which is **more** agreement than dense recomputation achieves with full-prefix Q16 (54/60). On both items where split Q16 differs from full-prefix, its output is the one that matches dense.
- The correct framing for Section 5.1 is that **full-prefix Q16 is the semantically matched comparator for the split-replay arms**, and dense recomputation is a third numerical path whose per-item outputs are not reproducible from either. Promoting the Table 18 footnote is necessary but insufficient: the footnote's stated reason (a different document/query execution boundary) is not what the archived rows show.

---

## 5. Prediction-identity counts (requested item 6)

| Pair | Identical predictions | Identical generated token sequences |
|---|---:|---:|
| full-prefix Q16 vs split Q16 | **58 / 60** | 58 / 60 |
| full-prefix Q16 vs frozen Q4/Q4/Q8 | **51 / 60** | 51 / 60 |
| full-prefix Q16 vs dense | 54 / 60 | 54 / 60 |
| split Q16 vs frozen Q4/Q4/Q8 | 51 / 60 | -- |

Prediction identity and token-sequence identity coincide on every pair, so no two arms reach the same string by different token paths.

---

## 6. Per-item regression counts under one arm (R44-4-16, constructive half)

The registered catastrophic rule is per-item dF1 <= -50 points **against dense**. That rule is preregistered and is reported unchanged. Alongside it, the same counts against the full-prefix Q16 reference arm, and a more informative -10-point threshold:

| Configuration | <= -50 vs dense (registered) | <= -50 vs full-prefix | <= -10 vs dense | <= -10 vs full-prefix |
|---|---:|---:|---:|---:|
| No-cache dense recompute | 0 / 60 | 1 / 60 | 0 / 60 | 1 / 60 |
| Full-prefix Q16/BF16 | 0 / 60 | 0 / 60 | 3 / 60 | 0 / 60 |
| Q-CoMem split Q16/BF16 | 0 / 60 | 0 / 60 | 3 / 60 | 0 / 60 |
| Q-CoMem frozen Q4/Q4/Q8 | 0 / 60 | 0 / 60 | 4 / 60 | 2 / 60 |
| Q-CoMem same-memory mixed | 1 / 60 | 1 / 60 | 5 / 60 | 2 / 60 |
| Q-CoMem aggressive mixed | 5 / 60 | 4 / 60 | 7 / 60 | 6 / 60 |

The frozen policy's safety claim survives the change of reference arm: **0/60 against full-prefix Q16 as well as 0/60 against dense.** At the -10-point threshold it is 2/60 against full-prefix. Note that the dense arm itself has 1/60 items below -50 against full-prefix (`2wikimqa#33`), which is the same decoding-flip item from Section 4.3 -- a further reason not to treat dense as a stable per-item reference.

Selection note (T-11): six policies were compared on this one 60-item panel and the headline policy was chosen by inspecting those comparisons. No multiplicity adjustment is applied and the interval is not selection-corrected. This must be stated, not fixed by re-analysis.

---

## 7. What the manuscript must now say

Drafted replacements. **I have not edited `main_r44_structure.tex`; these are for separate integration.**

### 7.1 Abstract -- replacement for the quality sentence

Current:

> Its mean F1 (0--100) is $54.24$ versus $54.62$ for split Q16, a paired difference of $-0.39$ points with a 95\% bootstrap interval of $[-2.04,1.08]$; this is a selected trade-off point among the evaluated split-replay policies, not a statistical-equivalence claim.

Replacement:

```latex
Against the same full-prefix Q16 reference used for the memory reduction, its mean F1 (0--100)
is $54.24$ versus $54.68$, a paired
difference of $-0.45$ points with a 95\% bootstrap interval of
$[-2.06,0.99]$ (10,000 paired item-level
resamples, seed 20260902); the sign is consistent across both datasets
($-0.34$ Qasper, $-0.56$ 2WikiMQA). This is a
selected trade-off point among the evaluated split-replay policies, chosen on this same
validation panel and not selection-corrected, and it is not a statistical-equivalence claim.
```

The point of the edit is that `14.10x` and the F1 delta in the preceding and following sentences now both refer to full-prefix Q16.

### 7.2 Section 5.4 (Answer quality) -- replacement for the opening paragraph

Current:

> On the 60-item panel, the frozen Q4/Q4/Q8 point has mean F1 $54.24$ versus $54.62$ for split Q16, a paired difference of $-0.39$ points with 95\% bootstrap interval $[-2.04,1.08]$. ...

Replacement:

```latex
On the 60-item panel we report every quality difference against a single reference arm,
full-prefix Q16, the same arm the Store column and the compression ratios use. The frozen
Q4/Q4/Q8 point has mean F1 $54.24$ versus $54.68$ for full-prefix Q16, a paired
difference of $-0.45$ points with 95\% bootstrap interval
$[-2.06,0.99]$. This decomposes into two steps that compound rather than cancel:
splitting costs $-0.06$ points and quantizing the split state costs a further
$-0.39$ points. The sign is consistent across both datasets:
$-0.34$ $[-3.38,2.39]$ on Qasper and
$-0.56$ $[-1.67,0.00]$ on 2WikiMQA (30 items each), so the pooled
figure is not masking opposite-signed per-dataset effects. No item crosses the registered
catastrophic threshold against dense, and none crosses it against full-prefix either
(0/60 under both arms); at a $-10$-point threshold the frozen policy regresses on 2 of 60
items relative to full-prefix. The headline policy was selected by comparing six policies on
this same validation panel; the interval is therefore not selection-corrected and carries no
multiplicity adjustment, and an interval crossing zero does not prove equivalence or
noninferiority.
```

### 7.3 Section 5.1 -- new paragraph promoting and correcting the Table 18 footnote

```latex
All arms in the 60-item panel share one execution boundary: the archived rows record identical
prefix, document, query and input token counts, identical truncation decisions, and identical
items, questions and references across dense, full-prefix and every split-replay arm. Full-prefix
Q16 is nonetheless the semantically matched comparator for the split-replay arms. With inputs
and boundaries held identical, dense recomputation and cached-prefix continuation nevertheless
emit different greedy token sequences on 6 of 60 items; logits are not archived, so we
report the divergence rather than asserting its mechanism. Dense and full-prefix Q16 emit
identical token sequences on 54 of 60 items, and their paired mean-F1
difference is $-0.39$ $[-4.66,2.60]$. The interval is wide not because the
two endpoints differ in expectation but because two of the flipped items are short-answer
2WikiMQA questions whose per-item F1 is nearly binary, giving a per-item difference standard
deviation of $14.51$ against $0.45$ for full-prefix versus split Q16. By the same
measure Q-CoMem split Q16 tracks full-prefix Q16 more closely (58/60 identical token
sequences) than dense recomputation does, and on both items where it differs its output is the
one that matches dense. We therefore report all quality differences against full-prefix Q16 and
retain the dense row as a no-retention reference rather than as a per-item baseline.
```

### 7.4 Table 1 -- change the dF1 column's reference arm

Change the header from `$\Delta$F1 vs. split Q16 [95\% CI]` to `$\Delta$F1 vs. full-prefix Q16 [95\% CI]` and use the pooled column of Section 3.1. Keep the split-Q16 column as an adjacent ablation column if space permits, clearly labelled. Add the per-dataset intervals as an appendix table (Section 3.1, second table). Add to the footnote:

```latex
All intervals in this table use one reference arm, full-prefix Q16/BF16, the same arm as the
compression column; they are paired 10,000-resample item-level bootstrap percentile intervals
(seed 20260902). The archival package contains 24 further intervals computed against dense
and against split Q16; Appendix~\ref{app:all-intervals} lists all of them with their reference
arms. The catastrophic-regression rule is preregistered against dense; the frozen policy scores
0/60 against dense and 0/60 against full-prefix Q16.
```

### 7.5 Contribution bullet 2 and the Conclusion

Both currently pair `14.10x` with `-0.39 [-2.04, 1.08]`. Replace the quality half in each with:

```latex
$-0.45$ points $[-2.06,0.99]$ against the same full-prefix Q16 reference
```

After the edit, a text search for `14.10` should find no sentence pairing it with a delta measured against any other arm.

### 7.6 New appendix section: all intervals and the display rule

```latex
\section{Complete interval inventory}
\label{app:all-intervals}
The archival analysis computed 24 paired-bootstrap intervals: six configurations against dense
and against split Q16 pooled (12), and six configurations against dense on each of the two
datasets (12). None used full-prefix Q16 as reference. This revision adds 18 intervals against
full-prefix Q16 (six configurations x pooled/Qasper/2WikiMQA). Table~\ref{tab:main-quality}
displays the six pooled full-prefix intervals; Table~\ref{tab:all-intervals} lists all 42 with
their reference arms, so no interval is computed but unreported.
```

---

## 8. Disagreements with published values

**None.** Every published mean F1, every published Store value, every published Table 1 delta and every published Table 1 interval endpoint reproduces from the archived shards, exactly or to correct rounding of the archived analysis. The two caveats worth recording are not disagreements about numbers:

1. The archived analysis records no bootstrap seed, so interval endpoints are reproducible only to within Monte-Carlo noise (observed max deviation 0.0621 F1 points, comparable to the across-seed spread in Section 3.4). A seed should be recorded going forward; this re-analysis uses 20260902.
2. The Table 18 footnote's stated reason for the dense arm's behaviour -- a different document/query execution boundary -- is **not supported by the archived rows**, which show identical boundaries across all arms (Section 4.2). The observed behaviour has a different and directly observable signature -- a small number of greedy token flips (Section 4.3). Promoting the footnote verbatim into Section 5.1, as A2's description proposes, would promote an unsupported explanation; Section 7.3 drafts a corrected version that reports what is measured and does not assert an unarchived mechanism.

---

## 9. Replay

The script below is exactly what produced `revision/a2_reference_arm_20260902.json` and every number in this document. It reads only the archived shards and the archived `replay_analysis.json`; it writes only the JSON. Two independent runs produced bitwise-identical numeric output.

```python
#!/usr/bin/env python3
"""A2 reference-arm re-analysis (ICLR round 44).

PURE RE-ANALYSIS of already-archived item-level F1.  No new execution.

Input : evidence/qcomem_mixed_validation_60item_20260812d/artifacts/raw/
        48 shards, shard-{0..7}-{dense,prefix,replay-d7-frozen-static,
        replay-d7-layer-q16,replay-d7-minus25-mixed,replay-d7-same-memory-mixed}.json
        plus the archived replay_analysis.json (for reconciliation only).
Output: revision/a2_reference_arm_20260902.json

Every configuration is compared against ONE reference arm: full-prefix Q16.
"""
import json, os, hashlib
import numpy as np

ROOT = "/Users/liuhanzuo/MacLLM-Bench/paper_autonomous_multifork_iteration"
RAW = os.path.join(ROOT, "evidence/qcomem_mixed_validation_60item_20260812d/artifacts/raw")
OUT = os.path.join(ROOT, "revision")

SEED, N_RESAMPLES, ALPHA = 20260902, 10000, 0.05

CONFIGS = [
    ("dense",                       "No-cache dense recompute"),
    ("prefix",                      "Full-prefix Q16/BF16"),
    ("replay-d7-layer-q16",         "Q-CoMem split Q16/BF16"),
    ("replay-d7-frozen-static",     "Q-CoMem frozen Q4/Q4/Q8"),
    ("replay-d7-same-memory-mixed", "Q-CoMem same-memory mixed"),
    ("replay-d7-minus25-mixed",     "Q-CoMem aggressive mixed"),
]
REF, PAPER_REF = "prefix", "replay-d7-layer-q16"

def load(config):
    rows = []
    for shard in range(8):
        with open(os.path.join(RAW, f"shard-{shard}-{config}.json")) as fh:
            rows.extend(json.load(fh)["rows"])
    return {(r["dataset"], r["source_index"]): r for r in rows}

data = {c: load(c) for c, _ in CONFIGS}
keys = sorted(set.intersection(*[set(d) for d in data.values()]))
assert len(keys) == 60
f1 = {c: np.array([data[c][k]["f1"] * 100.0 for k in keys]) for c in data}   # 0-100 scale
ds = np.array([k[0] for k in keys])
MASKS = {"pooled": np.ones(60, bool), "qasper": ds == "qasper", "2wikimqa": ds == "2wikimqa"}

def paired_bootstrap(a, b, seed=SEED, n=N_RESAMPLES):
    """95% percentile CI of mean(a-b); resamples ITEMS with replacement, paired."""
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n, d.size))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(d.mean()), float(lo), float(hi)

def store_mib(c):
    v = [data[c][k]["stored_persistent_nbytes"] for k in keys]
    return None if any(x is None for x in v) else float(np.mean(v)) / 1024 ** 2

PUB_MEAN = {"dense": 54.29, "prefix": 54.68, "replay-d7-layer-q16": 54.62,
            "replay-d7-frozen-static": 54.24, "replay-d7-same-memory-mixed": 53.36,
            "replay-d7-minus25-mixed": 49.19}
PUB_DELTA_SPLIT = {"dense": (-0.33, -4.61, 2.67), "prefix": (0.06, 0.0, 0.18),
                   "replay-d7-layer-q16": (0.0, 0.0, 0.0),
                   "replay-d7-frozen-static": (-0.39, -2.04, 1.08),
                   "replay-d7-same-memory-mixed": (-1.26, -4.28, 0.98),
                   "replay-d7-minus25-mixed": (-5.44, -11.56, -0.17)}
PUB_STORE = {"prefix": 136.235, "replay-d7-layer-q16": 34.683,
             "replay-d7-frozen-static": 9.661, "replay-d7-same-memory-mixed": 9.395,
             "replay-d7-minus25-mixed": 7.536}

results = {
    "method": {"resampling_unit": "item (paired; the 60 common (dataset, source_index) keys)",
               "n_resamples": N_RESAMPLES, "seed": SEED,
               "rng": "numpy.random.default_rng (PCG64)",
               "interval_type": "percentile, 95%", "scale": "F1 x 100 (0-100 paper scale)",
               "reference_arm": "prefix (full-prefix Q16/BF16)",
               "source": "evidence/qcomem_mixed_validation_60item_20260812d/artifacts/raw, 48 shards"},
    "cohort": {"n_items": len(keys), "n_qasper": int(MASKS["qasper"].sum()),
               "n_2wikimqa": int(MASKS["2wikimqa"].sum())},
    "configs": {},
}

for c, label in CONFIGS:
    e = {"label": label, "published_mean_f1": PUB_MEAN[c], "store_mib_per_doc": store_mib(c),
         "published_store_mib_per_doc": PUB_STORE.get(c), "by_split": {}}
    for split, m in MASKS.items():
        a, b = f1[c][m], f1[REF][m]
        pt, lo, hi = paired_bootstrap(a, b)
        e["by_split"][split] = {"n": int(m.sum()), "mean_f1": float(a.mean()),
                                "ref_mean_f1": float(b.mean()), "delta_vs_prefix": pt,
                                "ci95_vs_prefix": [lo, hi],
                                "sd_per_item_delta": float((a - b).std(ddof=1))}
    pt, lo, hi = paired_bootstrap(f1[c], f1[PAPER_REF])
    e["vs_split_q16_pooled"] = {"delta": pt, "ci95": [lo, hi],
                                "published": list(PUB_DELTA_SPLIT[c])}
    d = f1[c] - f1[REF]
    e["per_item_vs_prefix"] = {"sd": float(d.std(ddof=1)), "min": float(d.min()),
                               "max": float(d.max()), "n_identical_f1": int((d == 0).sum()),
                               "n_below_minus10": int((d <= -10).sum()),
                               "n_below_minus50": int((d <= -50).sum())}
    e["identical_predictions_vs_prefix"] = int(sum(
        data[c][k]["prediction"] == data[REF][k]["prediction"] for k in keys))
    e["identical_predictions_vs_dense"] = int(sum(
        data[c][k]["prediction"] == data["dense"][k]["prediction"] for k in keys))
    dd = f1[c] - f1["dense"]
    e["catastrophic_vs_dense_le_minus50"] = int((dd <= -50).sum())
    e["regression_vs_dense_le_minus10"] = int((dd <= -10).sum())
    e["regression_vs_prefix_le_minus10"] = int((d <= -10).sum())
    ent = {}
    for split, m in MASKS.items():
        pt2, lo2, hi2 = paired_bootstrap(f1[c][m], f1["dense"][m])
        ent[split] = {"delta": pt2, "ci95": [lo2, hi2]}
    e["vs_dense"] = ent
    results["configs"][c] = e

# ---- dense-vs-prefix investigation (issue Q5) ----
dvp = {}
for name, (x, y) in {"pooled": ("dense", "prefix"),
                     "prefix_vs_split_q16": ("prefix", PAPER_REF),
                     "dense_vs_split_q16": ("dense", PAPER_REF)}.items():
    pt, lo, hi = paired_bootstrap(f1[x], f1[y])
    dvp[name] = {"delta": pt, "ci95": [lo, hi], "ci_width": hi - lo,
                 "sd_per_item": float((f1[x] - f1[y]).std(ddof=1))}
dvp["n_identical_predictions_dense_prefix"] = int(sum(
    data["dense"][k]["prediction"] == data["prefix"][k]["prediction"] for k in keys))
results["dense_vs_prefix_investigation"] = dvp

results["prediction_identity"] = {
    "prefix_vs_split_q16": int(sum(data["prefix"][k]["prediction"] == data[PAPER_REF][k]["prediction"] for k in keys)),
    "prefix_vs_frozen": int(sum(data["prefix"][k]["prediction"] == data["replay-d7-frozen-static"][k]["prediction"] for k in keys)),
    "prefix_vs_dense": dvp["n_identical_predictions_dense_prefix"],
    "split_q16_vs_frozen": int(sum(data[PAPER_REF][k]["prediction"] == data["replay-d7-frozen-static"][k]["prediction"] for k in keys)),
    "n_items": 60,
    "token_ids_prefix_vs_split_q16": int(sum(data["prefix"][k]["generated_token_ids"] == data[PAPER_REF][k]["generated_token_ids"] for k in keys)),
    "token_ids_prefix_vs_frozen": int(sum(data["prefix"][k]["generated_token_ids"] == data["replay-d7-frozen-static"][k]["generated_token_ids"] for k in keys)),
    "token_ids_prefix_vs_dense": int(sum(data["prefix"][k]["generated_token_ids"] == data["dense"][k]["generated_token_ids"] for k in keys)),
}

results["seed_stability_vs_prefix_pooled"] = {
    c: [{"seed": s, "ci95": list(paired_bootstrap(f1[c], f1[REF], seed=s)[1:])}
        for s in (20260902, 1, 7, 12345, 99991)] for c, _ in CONFIGS}

results["item_level"] = {"keys": [f"{k[0]}#{k[1]}" for k in keys],
                         "f1_by_config": {c: [float(x) for x in f1[c]] for c, _ in CONFIGS}}

# ---- reconciliation against the archived analysis (issue R44-4-16) ----
arch = json.load(open(os.path.join(RAW, "replay_analysis.json")))
by_cfg = {s["config"]: s for s in arch["summary"]}
inv = []
for s in arch["summary"]:
    for key, ref in (("paired_bootstrap_95_ci_vs_dense", "dense"),
                     ("paired_bootstrap_95_ci_vs_q16_replay", "split Q16")):
        if s.get(key) is not None:
            inv.append({"config": s["config"], "reference_arm": ref, "split": "pooled",
                        "ci95_x100": [s[key][0] * 100, s[key][1] * 100]})
    for key, ref in (("dataset_paired_bootstrap_95_ci_vs_dense", "dense"),
                     ("dataset_paired_bootstrap_95_ci_vs_q16_replay", "split Q16")):
        for dsn, ci in sorted((s.get(key) or {}).items()):
            inv.append({"config": s["config"], "reference_arm": ref, "split": dsn,
                        "ci95_x100": [ci[0] * 100, ci[1] * 100]})
results["archived_interval_inventory"] = {
    "n_intervals": len(inv),
    "n_with_full_prefix_reference": sum(i["reference_arm"] == "full-prefix Q16" for i in inv),
    "reference_arms_present": sorted({i["reference_arm"] for i in inv}),
    "displayed_in_table1": 6, "intervals": inv}
for c, _ in CONFIGS:
    a = by_cfg[c]
    results["configs"][c]["vs_dense"]["archived_pooled_ci_x100"] = [
        x * 100 for x in a["paired_bootstrap_95_ci_vs_dense"]]
    results["configs"][c]["vs_dense"]["archived_per_dataset_ci_x100"] = {
        k: [x * 100 for x in v]
        for k, v in (a.get("dataset_paired_bootstrap_95_ci_vs_dense") or {}).items()}
rec = []
for c, label in CONFIGS:
    a = by_cfg[c]
    ap, ac = a["mean_f1_delta_vs_q16_replay"] * 100, [x * 100 for x in a["paired_bootstrap_95_ci_vs_q16_replay"]]
    mine = results["configs"][c]["vs_split_q16_pooled"]
    rec.append({"config": c, "label": label, "archived_delta": ap,
                "recomputed_delta": mine["delta"], "delta_abs_diff": abs(ap - mine["delta"]),
                "archived_ci": ac, "recomputed_ci": mine["ci95"],
                "ci_lo_abs_diff": abs(ac[0] - mine["ci95"][0]),
                "ci_hi_abs_diff": abs(ac[1] - mine["ci95"][1]),
                "published_ci": mine["published"][1:],
                "published_matches_archived_to_2dp": (
                    round(ac[0], 2) == mine["published"][1] and round(ac[1], 2) == mine["published"][2])})
results["reconciliation_vs_archived"] = rec

# ---- per-item divergence detail ----
def divergence(cx, cy):
    out = []
    for k in keys:
        x, y = data[cx][k], data[cy][k]
        if x["generated_token_ids"] != y["generated_token_ids"]:
            xt, yt = x["generated_token_ids"], y["generated_token_ids"]
            n = min(len(xt), len(yt))
            out.append({"key": f"{k[0]}#{k[1]}", "f1_a": x["f1"] * 100, "f1_b": y["f1"] * 100,
                        "delta": (x["f1"] - y["f1"]) * 100,
                        "first_divergent_token_position": next((i for i in range(n) if xt[i] != yt[i]), n),
                        "gen_tokens_a": x["generated_tokens"], "gen_tokens_b": y["generated_tokens"],
                        "pred_a": x["prediction"][:400], "pred_b": y["prediction"][:400]})
    return out
results["divergences"] = {"dense_vs_prefix": divergence("dense", "prefix"),
                          "prefix_vs_split_q16": divergence("prefix", PAPER_REF),
                          "prefix_vs_frozen": divergence("prefix", "replay-d7-frozen-static")}
results["dense_vs_prefix_investigation"]["n_items_differing"] = len(results["divergences"]["dense_vs_prefix"])

# ---- execution-boundary audit (Q5 / Table 18 footnote) ----
BFLDS = ["prefix_tokens", "context_tokens", "original_context_tokens", "document_tokens",
         "query_tokens", "input_tokens", "max_new_tokens", "context_truncated",
         "id", "question", "references"]
results["execution_boundary_audit"] = {
    "fields_checked": BFLDS,
    "all_identical_to_dense": {c: all(data[c][k].get(fl) == data["dense"][k].get(fl)
                                      for k in keys for fl in BFLDS) for c, _ in CONFIGS},
    "note": ("Every archived row of every arm carries identical document/query token counts, "
             "identical truncation flags, and identical item ids, questions and references. "
             "No document/query execution-boundary difference is visible in the archived rows.")}

results["ci_width_explanation"] = {
    "dense_vs_prefix": {"n_divergent_token_sequences": len(results["divergences"]["dense_vs_prefix"]),
                        "sd_per_item_delta": dvp["pooled"]["sd_per_item"],
                        "ci_width": dvp["pooled"]["ci_width"],
                        "max_abs_per_item_delta": max(abs(x["delta"]) for x in results["divergences"]["dense_vs_prefix"])},
    "prefix_vs_split_q16": {"n_divergent_token_sequences": len(results["divergences"]["prefix_vs_split_q16"]),
                            "sd_per_item_delta": dvp["prefix_vs_split_q16"]["sd_per_item"],
                            "ci_width": dvp["prefix_vs_split_q16"]["ci_width"],
                            "max_abs_per_item_delta": max(abs(x["delta"]) for x in results["divergences"]["prefix_vs_split_q16"])}}
results["ci_width_explanation"]["sd_ratio"] = dvp["pooled"]["sd_per_item"] / dvp["prefix_vs_split_q16"]["sd_per_item"]
results["ci_width_explanation"]["width_ratio"] = dvp["pooled"]["ci_width"] / dvp["prefix_vs_split_q16"]["ci_width"]

h = hashlib.sha256()
for c, _ in CONFIGS:
    for s in range(8):
        with open(os.path.join(RAW, f"shard-{s}-{c}.json"), "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
results["input_shard_set_sha256"] = h.hexdigest()
results["n_input_shards"] = len(CONFIGS) * 8

with open(os.path.join(OUT, "a2_reference_arm_20260902.json"), "w") as fh:
    json.dump(results, fh, indent=2, sort_keys=False)
print("wrote", os.path.join(OUT, "a2_reference_arm_20260902.json"))
```

---

## 10. Files

| File | Contents |
|---|---|
| `revision/a2_reference_arm_20260902.md` | this document |
| `revision/a2_reference_arm_20260902.json` | all computed statistics, including the 360 item-level F1 values under `item_level`, the 24-interval inventory, per-item divergences, and the seed-stability grid |

Input shard-set SHA-256 (SHA-256 over the concatenated per-file SHA-256 digests of the 48 shards, in the config/shard order defined in the script): `55862618fc56f5cdbc05fc4f513127e1e38681bdbc7b02dd5fe93aaa4c59de3e`

