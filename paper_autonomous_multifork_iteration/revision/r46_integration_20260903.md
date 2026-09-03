# R46 — integrating the A16 operating-point run (quantized exact-cache baseline)

Date: 2026-09-03
SOURCE: `main_r45_evidence.tex` (untouched)
NEW: `main_r46_baseline.tex`
NEW TABLE: `tables/qcomem_validation60_r46.tex` (R45's `qcomem_validation60_r42.tex` untouched;
`qcomem_tradeoff_r42.tex` reused unchanged)
BUILD: `build/r46_baseline_v1/main_r46_baseline.pdf`, 37 pp.

Evidence integrated: `revision/a16_operating_point_gate_20260903.md` — QS Trial 1943447, completed
by 1943737, extracted 1943780. 60-item validation cohort (Qasper/2WikiMQA source indices 6--35),
real LongBench protocol, one warm-up, three repeats, seed 20260903, paired item-level bootstrap
against `full-prefix-q16` with 10,000 resamples.

---

## 1. What was added

### 1.1 The quantized exact-cache baseline — closes Q1 and R44-4-02

Reported for the first time, at five sites, all in the main text:

| Site | PDF p. | What it says |
|---|---|---|
| Abstract (L64--67) | 1 | "A separately executed run on the same items adds the quantized exact-cache arm: full-prefix Q8 retains $56.438$ MiB/document at $-3.78$ points $[-9.12,0.32]$, so the frozen policy is $5.84\times$ smaller than that arm with a $3.41$-point higher mean F1." |
| Contribution bullet 2 | 2 | "a separately executed run on the same items supplies the quantized exact-cache arm, $5.84\times$ larger in retained state and $3.41$ F1 points lower than the frozen point." |
| Table 1, lower block | 7 | `Full-prefix Q8 · 56.438 · 2.41× · 0.182 s · 50.89 · −3.78 [−9.12, 0.32]` |
| Sec. 5.2 | 7 | "full-prefix Q8, an exact cache packed with the same codec, retains $56.438$ MiB/document, $2.41\times$ below the native-dtype reference but $5.84\times$ above the frozen point." |
| Sec. 5.4 | 8 | the paired-difference paragraph (below) |
| Conclusion | 9 | "$5.84\times$ smaller than a quantized exact cache at $3.41$ F1 points better" |

Sec. 5.4's new paragraph states it plainly and bounds it:

> The operating-point run measures what an exact cache costs once quantized with the same codec.
> Against that run's own full-prefix reference of 54.668, full-prefix Q8 scores 50.885 (−3.78
> [−9.12, 0.32]) and the frozen j=7 point 54.297 (−0.37 [−1.97, 1.08]), so the frozen point is both
> 5.84× smaller than the strongest quantized exact cache we measured and 3.41 points higher in mean
> F1; we computed no paired interval between those two arms and their intervals against the shared
> reference overlap, so the retained-state ordering is the firmer of the two.

The last clause is deliberate. The bootstrap was run against `full-prefix-q16`, so no paired
interval exists for `d7` versus `q8`, and `q8`'s own interval touches zero at the top. The Store
ordering (5.84×) is unambiguous; the 3.41-point F1 gap is a difference of measured means with
overlapping reference-arm intervals, and the paper now says so rather than leaving a reviewer to
notice it.

### 1.2 The depth control — j = 7 is now measured, not assumed

Table 1's lower block carries `Q-CoMem frozen Q4/Q4/Q8, j = 13 · 16.101 · 8.46× · 0.582 s · 51.27 ·
−3.40 [−8.67, 0.52]` beside the `j = 7` row. Sec. 5.4:

> The j=13 control loses 3.40 points [−8.67, 0.52], about nine times that loss, fixing the split
> depth at j=7 for this policy — a control on one parameter, and we report nothing about depths this
> cohort did not measure.

It is presented as a control, not a contribution: no subsection, no sweep framing, one sentence.
Sec. 5.6 states the corresponding limitation, "the split depth is validated against one alternative
($j=13$), not swept".

**Correction to the task brief.** R45 contains no sentence conceding that `j=7` was never justified;
`grep` over `main_r45_evidence.tex` for `depth` returns only method text and appendix cohort rows.
The unmeasured-`j` entry lived in `revision/derived_vs_measured_20260902.md`, not the manuscript.
There was therefore no concession to replace — the measurement was added, and the honest residual
limitation (one alternative, not a sweep) was added with it. Nothing was deleted here.

### 1.3 The independent replication — one sentence

Sec. 5.4: "The frozen point's −0.37 [−1.97, 1.08] also reproduces the archival cohort's −0.45
[−2.06, 0.99] against the same reference arm in an independent run." Stated once, nowhere else.

### 1.4 Latency: the concession is now stronger, not weaker

Sec. 5.3 gained the 60-item numbers and the dense comparison:

> The operating-point run sharpens this on all 60 items: full-prefix Q16 and Q8 reach the first
> token in 0.181 and 0.182 s, dense recomputation in 0.636 s, and the frozen j=7 point in 0.656 s,
> so Q-CoMem is slower to the first token than honest dense recomputation as well as than exact
> reuse, while TPOT stays between 54.4 and 56.0 ms at every arm.

Sec. 5.3 closes "no latency advantage over either endpoint" (R45 said only "not a universal latency
improvement"). Sec. 5.6 now reads "TTFT is substantially higher than full-prefix reuse and, on the
operating-point run, than dense recomputation too". The Conclusion adds "dense recomputation reaches
the first token sooner". The capacity-first framing is unchanged.

### 1.5 Protocol and provenance for the second execution

- Sec. 5.1 "Quality cohort" now names the operating-point run: same 60 items, same stack, one
  warm-up, three repeats, dataset-length generation, seed 20260903; adds full-prefix Q8 and the
  $j=13$ control; "The two executions occupy separate blocks of Table 1, are never pooled, and each
  is differenced against its own full-prefix arm."
- Table 1's caption and both in-table block headers carry the separation and the two seeds.
- Sec. 5.6 ¶1 now describes three unpooled executions.
- The Reproducibility Statement says the operating-point block "comes from a separate registered
  execution (one warm-up, three repeats, seed 20260903) recorded under its own run identifier; it is
  not part of that archival package and the two are never pooled."
- Appendix Table `tab:cohorts` gained an `H20 operating-point run` row whose exclusion column reads
  "not pooled with the archival panel, not Recall, total memory, **admitted capacity, resident-set**,
  edge, or ForkAudit evidence".

---

## 2. Hard constraints — how each was honoured

**Residency is not reported.** The words *resident documents*, *residency*, and `capacity_estimate`
appear nowhere in `main_r46_baseline.tex` or in either table (`grep`: 0 hits; the only survivor of
`resident` is Introduction "other resident documents", which is R45 text about a deployment's
working set, and the ForkAudit factorial's "resident counts $N$"). The 14.18× / 5.83× residency
ratios were not written anywhere, so no labelling question arises, and no derived residency figure
sits beside a Store ratio. The A16 correction section's reasoning is instead encoded as an explicit
*exclusion* in the new Appendix cohort row ("not ... admitted capacity, resident-set"), which is
consistent with Sec. 5.6's standing statement that reducing Store "does not by itself prove lower
peak VRAM or serving capacity". A13 is not marked closed anywhere.

**No latency claim.** See §1.4. Nothing in the manuscript claims a TTFT or throughput advantage over
either endpoint.

**No projections.** `grep` for `3.1\times`, `3.5--3.9`, `3.9\times`: 0 hits. The reviewers' 3.5--3.9×
and the author-side 3.1× appear nowhere in the manuscript, the tables, or the appendices.

**No non-claim, scope statement, or limitation weakened.** Verified by word-diff of R45 → R46. All
nine ForkAudit scope sites survive verbatim or strengthened, and A11 finding C4 survives at both its
sites (Sec. 4.3 "materializes a full private copy of the dequantized entry per query, so it shares
nothing and exercises neither borrowing nor copy-on-write"; Sec. 5.6 "$j=7$ packed Q4/Q4/Q8 path
whose Read step materializes a full private copy rather than borrowing it, so ownership there
transfers by design argument only"). Four wordings that a first pass had shortened were restored
after the diff review, because each carried scope rather than emphasis:

1. Introduction: "The timing execution uses an eight-item subset, the operating-point run a separate
   execution, and the ownership study a third system entirely" (a compressed "Each result names its
   own execution" had lost *a third system entirely*).
2. Sec. 5.5: "The live-binding rerun is **scientifically valid** on all eight ranks" (bare "valid"
   would have read as a stronger, undefined verdict).
3. Sec. 5.6: "so **local or** edge use is motivation" (the caveat covers memory-constrained local
   use as well as edge).
4. Sec. 5.6: "measure a **$j=7$ packed Q4/Q4/Q8** path" and Conclusion "localizes **request-local**
   ownership violations" (C4 and the ownership-scope qualifier).

One deliberate *narrowing* was made because R45's wording became false: Sec. 5.1 "Executed systems"
no longer says the deployment tables are executions "at $j=7$", because Table 1 now also contains a
$j=13$ row.

---

## 3. Also fixed

### T-02 — the per-component breakdown reached the manuscript

Two changes, both aimed at the printed check that triggered T-02.

1. **Table 1's footnote now prints the BF16-normalised Store column**, including the value R45 never
   printed: "$30.000$ MiB/document of that $136.235$ MiB is FP32 GDN recurrent state in excess of a
   BF16 encoding, as is $6.000$ MiB of the split row's $34.683$ MiB, so the all-BF16 column reads
   $106.235/28.683/9.661/9.395/7.536$ MiB/document ($1.00/3.7038/10.9965/11.3073/14.0969\times$);
   normalised, the split-to-frozen step is $2.97\times$ rather than the $3.59\times$ the printed
   column shows." A reader no longer has to divide 106.235 by 3.7038 to recover 28.683.
2. **New Appendix C, "Per-Component Store Accounting and the Eq. 3 Ceiling"** (PDF p. 17), carrying
   A11 §3.5 into the paper: the identity $\mathrm{bytes}(n,b)=\lceil n/64\rceil(8b+4)$ with its
   3.5556×/1.8824× ceilings and the 180/180 conformance count; a per-component table (residual,
   layer-3 K, layer-3 V, GDN conv, GDN recurrent) at native / BF16-ref / Q8 / Q4 / Q2; the
   composition of all five Table 1 rows as explicit sums; and the resolution of the apparent
   violation, "with the split row's 6.000 MiB of FP32 excess removed, the step is
   $28.683/9.661=2.969\times$, inside the ceiling", plus the lower-cache-only sub-ratios
   $2.5211\times$ / $2.6496\times$ / $4.1187\times$ against their own bounds.

Every figure in Appendix C was recomputed independently before printing:
`10×2×3.7180 + 30×0.0625 + 30×2.0000 = 136.235` (BF16 106.235); split `= 34.683` (BF16 28.683);
frozen `= 9.661`; same-memory `= 9.395`; aggressive `= 7.536`; printed step 3.590×, normalised
2.969×; lower-cache sub-ratios 2.5213 / 2.6497 / 4.119. All match the printed values.

Sec. 5.1 and Sec. 5.2 both point to Appendix C. `11.73×` — which the meta-review said the authors
must **not** be required to print — appears nowhere.

### The Introduction's ForkAudit scope qualifier — the ninth site

R45's L114 read "we pair the design with ForkAudit, a phase-aware ownership trace over immutability,
copy-on-write (COW), recurrent rebinding, and selected call provenance." R46 adds ", evaluated on a
full-prefix BF16 configuration rather than on the quantized Read path." (PDF p. 2, margin 070--071).
This was the one main-text pairing site `verify_r45_scope_20260903.md` flagged as lacking the
qualifier.

---

## 4. Page budget — exactly what was cut

R45 ended the main text at margin line 481 of page 9 with ~4 lines of slack. R46 adds ~35 typeset
lines of new prose plus ~15 lines of table growth (6 new tabular rows, a second block header, a
longer caption and footnote). Every page of the main text is full at 54 line slots, so the whole
addition had to be recovered from the same nine pages.

**Relocated to the appendix (no information lost):**

| Where | What moved | Now in |
|---|---|---|
| Sec. 5.1 / 5.2 | per-component Store decomposition detail | new Appendix C (expanded, not just relocated) |
| Table 1 footnote | the catastrophic-regression counts and the multiplicity caveat, which duplicate Sec. 5.4 word for word | Sec. 5.4 (already there) |
| Table 1 footnote | lower-block TPOT range, which duplicates Sec. 5.3 | Sec. 5.3 (already there) |

**Deleted as duplication of text elsewhere in the paper:**

| Where | Cut | Where it still stands |
|---|---|---|
| Abstract | "the sign consistent across both datasets ($-0.34$ Qasper, $-0.56$ 2WikiMQA)" | Sec. 5.4 |
| Intro bullet 1 | trailing "(Section 4)" pointer | — |
| Sec. 5.1 quality cohort | stack versions, GPU count, generation caps; the six-policy enumeration | Table 2 caption; Table 1 rows |
| Sec. 5.1 latency execution | the metric list (Store/TTFT/TPOT/tok-s/F1) | Table 2's column headers |
| Sec. 5.2 | "Table 1 reports the broadest current cohort"; the two-reductions-are-complementary sentence; the eight-item Q8 15.89/9.74 restatement; the 92.91% and 90.91% percentages (ratios retained) | Table 1, Table 2 |
| Sec. 5.4 | "On the eight-item execution Q8 records the same measured mean F1 of 39.137" | Table 2 footnote ("Q8's is 0.000") |
| Sec. 5.5 scope | the shard-verdict term definitions; "39 state-appended tokens exercise COW on the 127-token final page"; "each rebuilt as one allocator and one witness execution" | Appendix A, Table 5 |
| Sec. 5.5 what it found | "including all 288 adjacent-fan-out comparisons"; "32-token" boundary | Appendix A |
| Sec. 5.5 falsification | "whereas the repaired path is storage-clean in 8/8" (a positive result, not a caveat) | Appendix F/H |
| Sec. 5.6 ¶2 | "which the 60-item panel also decomposes into stored components and which both implementations count identically on the eight shared documents" | Sec. 5.1, Appendix C |
| Sec. 4.3 | "The design exchanges suffix reconstruction and dequantization for a smaller retained entry" | Abstract, Sec. 3 |
| Sec. 4.4 | the workflow enumeration ("freezes code, model, data, schedule, and runtime roles ... reporting coverage separately from verdict") | Appendix E; the preceding sentence still states coverage-vs-verdict separation |
| Sec. 5 opener | three-sentence roadmap compressed to one | — |
| Conclusion | "the one term of the deployment budget in Eq. 1 that scales with the hot document set" | Abstract opening, Sec. 3 ¶1 |

**Structural:** the Read-section fork definition (R45's unnumbered Eq. 4,
$\mathcal{F}_r(D)=(\{K^D_\ell,K^r_\ell\},\{G^D_\ell,G^r_\ell\})$) was inlined rather than displayed.
It carried no label and nothing referenced it; Eq. 1--3 keep their numbers, so the hardcoded
"Eq. 3" references in Sec. 5.2, Table 1 and Appendix C remain correct.

**Compression without deletion:** roughly 120 words of redundant phrasing across Related Work,
Sec. 3, the two figure captions, Sec. 5.1, 5.5, 5.6 and the Conclusion. No caveat, exclusion,
non-claim or scope statement was among them (§2).

Result: Conclusion heading on page 9 (margin 478), main text ending margin 485; Ethics, LLM-use,
Reproducibility and References all on page 10, exactly as in R45.

---

## 5. Build QA

```
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/r46_baseline_v1 main_r46_baseline.tex
exit 0
```

| Gate | Result |
|---|---|
| undefined references | **0** |
| undefined citations | **0** |
| multiply-defined labels | **0** |
| Overfull boxes | **0** |
| LaTeX warnings (any package) | **0** |
| BibTeX `warning$` calls | **0** |
| citations cited / resolved | **40 / 40** |
| Underfull `\hbox` | 91 — identical to R45's 91; no regression (all from `raggedright` appendix longtables) |
| pages | 37 |
| main text ends | page 9 (verified with `pdftotext -layout`, not line count) |
| References begin | page 10 |

Files verified untouched: `main_r45_evidence.tex`, `main_r44_structure.tex`,
`tables/qcomem_validation60_r42.tex`, `tables/qcomem_tradeoff_r42.tex`, everything under `review/`
and `state/` (all mtimes predate this session).

---

## 6. What I judged should not be written, and why

1. **Any residency or resident-document figure.** Not even labelled as derived. The A16 correction
   is right that `capacity_estimate` is Store divided by a budget, and the cheapest way to guarantee
   a reader never reads the two as independent corroboration is for the second one not to exist in
   the paper. The capacity claim is already concrete without it: 9.661 versus 56.438 MiB/document is
   the same fact in the units the paper's own denominator is defined in. I recorded the exclusion in
   the Appendix cohort row instead, so the artifact-side reader knows the run measured it and the
   paper declined to use it.

2. **A13 marked closed.** It is not, and nothing in R46 implies it is. Eq. 1's question still needs
   an admission experiment that raises the resident set until failure.

3. **A paired interval, or any significance language, for d7 versus q8.** The bootstrap targeted
   `full-prefix-q16`; the item-level rows for the A16 run are not in this repository, so the
   interval cannot be recomputed here. Sec. 5.4 states the difference of means and says explicitly
   that no paired interval was computed and that the two arms' intervals against the shared
   reference overlap. Recomputing it later would be admissible re-analysis and would strengthen the
   claim; asserting it now would not.

4. **Restating the headline compression relative to the quantized cache**, as Q1's `required_action`
   asked. The meta-review partially rejected that instruction, and the two ratios answer different
   questions: 14.10× is what the stack actually stops retaining, 5.84× is the margin over the best
   competing policy. Both are printed, in the abstract and the conclusion, with their reference arms
   named. Demoting 14.10× to "a secondary reference" would have traded one under-specified headline
   for another.

5. **Harmonising Table 2's 42.8% with Table 1's 22.0%** (T-02 residual item 4). Table 2's footnote
   reports the *total* FP32 recurrent share of its own 140.34 MiB reference; Table 1 reports the
   *excess over BF16*. Both are individually correct, both name their quantity, and fixing the
   asymmetry needs either a second percentage in Table 2 or a BF16-normalised ratio set for the
   eight-item cohort. On a page with zero slack, and with Appendix C now giving a reader the general
   rule, I left Table 2 alone and am flagging it as still open.

6. **Anything about the eight-item A14b depth signal** (d20/d26/d33, the 0.340 plateau, the
   synthetic-harness latency win). Superseded by A16 on the 60-item cohort; A16's own note records
   that the earlier reading did not survive. None of it is in the manuscript and none was added.

---

## 7. Issue status after R46

| Issue | Before | After | Basis |
|---|---|---|---|
| Q1 (critical, R44-REV-1) | open | **closed** | quantized exact-cache arm measured and reported in the abstract, contribution bullet 2, Table 1, Sec. 5.2, Sec. 5.4 and the Conclusion; a reader can read $56.438/9.661$ off Table 1 directly |
| R44-4-02 (major, R44-REV-4) | open | **closed** | full-prefix Q8 row added to Table 1 with the same packer, cohort and denominator; its Store and F1 both reported; the frozen policy's ratio and F1 gap relative to it stated. Its `required_action` also asked for a full-prefix Q4 row — not run, so not reported |
| T-02 (major, R44-REV-2) | partially_resolved | **resolved**, pending verification | per-row component bytes and the $\sum_c\lceil n_c/64\rceil(8b_c+4)$ identity published (Appendix C); the 28.683 MiB BF16-normalised split value printed in Table 1's footnote; the apparent super-ceiling ratio named and explained |
| Intro ForkAudit qualifier (residual of R44-REV-3-C1 / T-03 / Q3) | non-blocking residual | **closed** | ninth site now carries the scope |
| A13 | open | **open** | unchanged; explicitly excluded in the new cohort row |
| A15 (ForkAudit on the quantized Read path) | open | **open** | unchanged; all ten C4/scope sites intact |
| R44-5-02 (method_provenance rows) | not_resolved | **not_resolved** | out of scope for this task; `evidence/method_provenance.tsv` untouched |
| T-02 residual item 4 (Table 2 framing) | — | **open** | see §6.5 |
