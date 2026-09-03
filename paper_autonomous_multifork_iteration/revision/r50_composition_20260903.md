# R50 — the composition run lands, and the memory claim becomes total memory

Date: 2026-09-03
Source manuscript: `main_r49b_ownership.tex` (untouched; md5 `81402a26b870a6e24c11cf4ed5c9864a`,
mtime unchanged)
Produced: `main_r50_composition.tex`
Bibliography: `\bibliography{references_r47}` retained unchanged. `references.bib` and
`references_r47.bib` are byte-identical to R49b; BibTeX emits the same **39** entries and
the cited-key set diffs empty against R49b's `.bbl`.
New table files: `tables/qcomem_transient_r50.tex`, `tables/qcomem_composition_r50.tex`,
`tables/qcomem_validation60_r50.tex` (R48's Table 1 with three duplicated note sentences
trimmed; every number and every non-claim in the notes preserved).
Untouched inputs: `figures/qcomem_pipeline_r47.pdf`, `tables/qcomem_tradeoff_r48.tex`,
`tables/qcomem_validation60_r48.tex` (md5s unchanged; the R48 files remain the ones
`main_r49b_ownership.tex` reads).
Build: `build/r50_composition_v1/main_r50_composition.pdf` — 42 pp (R49b: 41 pp).
Main text ends **page 9**; Ethics/LLM/Reproducibility and References begin **page 10**.

Primary sources: `revision/c1_composition_results_20260903.md` and
`revision/total_memory_model_20260903.md`. Standing admissibility rule:
`revision/derived_vs_measured_20260902.md`. Panel: `review/round_46/meta_review.json`.

---

## 1. Issue IDs closed

| ID | What R50 does |
|---|---|
| **C1** (`R46-REV-1-01`, `TS-01`, `R46-REV-3-01`, `R46-REV-5-07`) | **Closed by execution.** Section 5.6 reports the composition run: one packed depth-7 Q4/Q4/Q8 entry shared across N ∈ {1,2,4,8} concurrent requests; shared mode effective with no fallback; sharing non-vacuous at the policy's window *final*; the N>1 shared traces token-identical, per request, to the published N=1 private-materialization path; every applicable ForkAudit target — the seven of Sec. 4.4 plus the three packed-entry obligations previously named untested — covered and passing on that path. Agreement with the full-prefix arm is a recorded diagnostic that gates nothing. The disclosure-only treatment the meta-review said "cannot move Contribution" is replaced by the run. |
| **A3 / TS-02** | **Closed, and against us.** The transient term of Eq. 1 is now *measured* for both arms at N ≤ 8 rather than derived from Table 4. The measurement contradicts the arithmetic the meta-review used to reject a reviewer's concurrency concern: Q-CoMem's peak transient exceeds full-prefix at every fanout run, fitted crossing at N = 12.5, outside the measured range. Appendix C keeps the arithmetic verbatim, labelled as refuted, and says why the two disagree. |

Nothing else changes status. A1, A2, A4–A8, B1–B7 remain as R45–R49 left them; C2–C5 remain open.

---

## 2. What was written

**2.1 Section 5.6, new — "Composition: one packed entry shared across concurrent requests"**
(37 rendered lines, three run-in paragraphs, page 8–9). Placed after Sec. 5.5 and before
Limitations, which becomes 5.7.

* ¶1 reports the run and, **in the same paragraph, three sentences later**, the sharing
  coverage: "at the sharing window 2 of 16 tensors and 0.500 of 12.938 MiB per request are
  shared, about 4% of per-request state. This demonstrates the ownership discipline of
  Section 4.3 on the packed path, not that sharing materially reduces per-request memory."
  The good result and the 4% are not separated by a paragraph, a page, or a caption.
* ¶2 reports peak transient allocation, the three fits, the sharing saving (214.87 MiB at
  N=4, 263.83 at N=8; slope 148.38 → 109.76), the crossing at N = 12.5 outside the measured
  range, and the retraction of our own prediction. It closes by *strengthening* Sec. 5.7's
  Store admission rather than qualifying it.
* ¶3 gives `M_total(D,N) = S·D + T(N)` with S = 136.235 / 9.661 MiB/document and T measured,
  the three assumptions, the four crossover values, two worked ratios, the losing direction
  stated as plainly as the winning one, and the regime described **qualitatively only**.

**2.2 The central claim is now total memory.** Reframed in five places, all with 14.1x kept
and correctly labelled as the retained term:

* Abstract now leads the results with "We report *total* memory rather than retained state
  alone, `M_total(D,N) = S·D + T(N)` … both terms measured", then S (10.9965x like-for-like
  first, 14.10x native second — the R47 ordering is preserved), then T running the other way,
  then the regime, then the three assumptions.
* Intro bullet 2 retitled "Total memory, not the retained term alone."
* Sec. 3 (Motivation) now says we do not report the retained term by itself and points to
  Sec. 5.6 for M_active and the sum. The existing admission that the workspace term is not
  method-independent, and that our own Read path is why, is kept verbatim.
* Sec. 5.2 opens "Retained state is one of the two method-dependent terms of Eq. 1;
  Section 5.6 measures the other and reports the sum", and closes both Store columns with
  "both are the retained term only".
* Conclusion opens with the total form and states the regime in both directions.

**2.3 Every scope statement rewritten from "absent" to "narrower residual".** None deleted.

| Where | R49b | R50 |
|---|---|---|
| Abstract | "that factorial runs full-prefix BF16 KV … so it audits the ownership discipline and not the quantized Read path" | "the primary ForkAudit factorial still runs full-prefix BF16 with no split or quantization, so ownership on the exact code path behind the deployment tables transfers by argument" (after the composition paragraph) |
| Fig. 1 caption | "the packed Read path drawn here is unaudited" | "the packed Read path drawn here is covered only by the separate eight-item composition run of Section 5.6, and not on the code path behind Tables 1 and 22" |
| Sec. 4.3 | "shares nothing and exercises neither borrowing nor copy-on-write" (kept verbatim) | + "Section 5.6 implements and audits the borrowing path itself at N ≤ 8, token-identically to that private path" |
| Sec. 4.4 | "the packed-entry obligations … are untested" | the three are now named as targets 8–10; the factorial "covers the first seven"; the composition run "covers all ten … at one entry, one depth, one bit vector and N ≤ 8" |
| Sec. 5.1 | "no ForkAudit verdict here covers those packed entries" | "no verdict of that factorial covers those packed entries, which only the composition run covers" |
| Sec. 5.5 | "audits the ownership discipline …, not the quantized Read path of Tables 1 and 2" | + "which Section 5.6 audits separately" |
| Sec. 5.7 | "ForkAudit has not been run on the quantized Read path … ownership there transfers by design argument only" | "The primary ForkAudit factorial has not been run on the quantized Read path … The composition run closes that gap **only on its own terms**: one entry, one depth, one frozen bit vector, one Transformers stack, eight items, N ≤ 8, and a Read path built to borrow rather than the one those tables ran, to which it is token-identical but not identical in implementation — so ownership on the exact code path behind those tables still transfers by design argument, and that run establishes nothing about paged kernels, vLLM or SGLang, a second backbone, throughput, admitted capacity, or security." |

The Sec. 5.7 rewrite is **net more restrictive than R49b**: it adds six new non-claims
(one entry / one depth / one bit vector / eight items / N ≤ 8 / different implementation)
and the paged-kernel, vLLM-SGLang, second-backbone, throughput, capacity and security
exclusions from the C1 note's "what this still will not prove" list.

**2.4 The transient record corrected in three places.**

* Sec. 5.2 no longer says the transient term "is arithmetic on Table 4, not an allocator
  measurement"; that sentence is superseded.
* Sec. 5.7: "The per-request transient is now measured at N ≤ 8 (Section 5.6) and is higher
  for Q-CoMem than for the exact cache throughout that range; it is a CUDA-allocator peak,
  not process or NVML memory, and neither it nor Store fixes a concurrency limit for either
  arm, since we evaluate no admission or eviction policy." The Store admission preceding it
  ("reducing it does not by itself prove lower peak VRAM or serving capacity", "a
  post-priming delta against a full-copy control, neither additive with Store nor evidence
  of higher admitted capacity") is byte-identical to R49b.
* Appendix C, "The transient term: the arithmetic, and the measurement refuting it": the
  R49b arithmetic (9.661 + 28.683N against 106.235 + 31.875N, "smaller intercept and smaller
  slope … no crossing at any N") is reprinted verbatim and then explicitly refuted, with the
  reason the two disagree (entry-owned payload versus every allocation the query makes,
  including dequantization and concatenation buffers).

**2.5 New tables.**

* **Table 4** (Appendix C, p. 20) — median peak transient allocation per query, three arms
  × four fanouts, plus the three least-squares fits. Defined as "the CUDA-allocator peak
  above the pre-fork baseline, not process or NVML memory".
* **Table 5** (Appendix C, p. 20) — `M_total(D,N)`, with the four crossover values on a
  spanning row and the four worked deployment points. Its notes carry the three assumptions
  verbatim, plus "The transient term is an allocator peak on this stack, is not additive
  with Store, and fixes no admitted capacity" and "No workload of the many-warm-document
  shape these rows describe was executed."

**2.6 Registration.** A cohort row for the composition run added to Appendix D's
authorization matrix (frozen scope, authorized claims, and the explicit exclusion list), a
row added to Appendix J's artifact/claim map describing the gate, shards, ownership
accounting, token traces and the blind contract replay, and a sentence added to the
Reproducibility Statement.

---

## 3. Numeric audit — every number traces

All new values come from the two primary notes; none is projected.

| Value(s) | Source |
|---|---|
| 2 of 16 tensors; 0.500 of 12.938 MiB; ~4% | C1 note, "Result 2", gate record at the sharing window |
| shared_mode_effective / no fallback / non_vacuous / window `final` / token identity / all applicable targets covered and passing / full-prefix agreement a non-gating diagnostic | C1 note, "Result 1" |
| 286.44 / 427.49 / 752.09 / 1441.93 (full-prefix) | C1 note, transient table |
| 930.91 / 1062.73 / 1178.41 / 1713.15 (shared) | C1 note, transient table |
| 954.09 / 1063.35 / 1393.28 / 1976.98 (private) | C1 note, transient table |
| fits 103.45+166.28N, 809.68+109.76N, 790.50+148.38N | C1 note |
| 214.87 @ N=4, 263.83 @ N=8; slope 148.38 → 109.76 | C1 note |
| crossing at N = 12.5 | C1 note; total-memory note |
| "intercept roughly eight times larger" | 809.68 / 103.45 = 7.83, stated as "roughly eight" |
| S = 136.235 / 9.661 MiB/document | total-memory note; already in Table 1 |
| crossover D = 5.13 / 4.69 / 3.79 / 2.01 | total-memory note, crossover table |
| (4,8) 1978.6 / 1726.4 / 1.15x; (20,4) 3493.3 / 1441.9 / 2.42x; (100,1) 13893.2 / 1885.5 / 7.37x; (100,8) 15057.2 / 2653.9 / 5.67x | total-memory note, worked points |
| N ∈ [1,8], four points | total-memory note, assumption 3 |

Independent re-derivation performed while writing: the worked points and crossovers
reproduce from the *fitted* transient lines and the retained means to the printed digits
(e.g. 136.235·20 + 103.45 + 166.28·4 = 3493.27 → 3493.3; 9.661·20 + 809.68 + 109.76·4 =
1441.94 → 1441.9; ratio 2.423 → 2.42x). The manuscript therefore says "from the (a) fits",
not "measured", for every cell of Table 5.

No pre-existing number changed. Every number printed in R49b still appears at least once in
R50 (verified by counting all 24 distinctive values across the rendered text; several lost
*redundant* instances, none lost its last instance).

---

## 4. Non-claim / limitation audit

Method: both PDFs rendered with `pdftotext -layout`, page furniture and ICLR line numbers
stripped, whitespace and punctuation normalised, then a 54-phrase checklist of R49b's
scope statements, non-claims and limitations matched against R50's full rendered text.

**Result: 54 / 54 present.** The single initial miss — "the column must not be inverted for
a token count" — is a false negative caused by an ICLR line number falling inside the phrase
in R50's rendering; the phrase is present verbatim in the relocated Table 22 note.

Additionally verified byte-identical or strengthened, by source comparison:

* Sec. 5.7 ¶1 (three panels, archival replay boundary, edge, split depth, TTFT, dispersion,
  dense-TPOT discrepancy, continuous batching/QPS/admission/eviction/scheduler) —
  **unchanged from R49b**.
* Sec. 5.7 ¶2 first sentence (Store exclusions and the "does not by itself prove" clause) —
  **unchanged from R49b**.
* Sec. 4.4's TCB paragraph — unchanged except for the seven/ten target restructure.
* Table 1's notes — three sentences duplicated verbatim in Sec. 5.1/5.2 were removed; all
  five non-claims in the notes ("an interval crossing zero is not statistical equivalence",
  "selected without multiplicity adjustment", "the ten pairs carry none either", "Recall was
  measured in no cohort", "blank TTFT cells mark a metric the archival run did not record")
  are kept verbatim.
* Table 22's caption and both note blocks (including the `†` throughput note) moved to the
  appendix **verbatim**, not edited.

No non-claim was weakened. Nine new ones were added (§2.3, §2.4, Table 5's notes).

---

## 5. What was cut — exactly

The 9-page gate is hard and R49b's main text already filled exactly 9 pages, so ~120
rendered lines of new and reframed material had to be paid for. Cuts are listed in
descending cost.

### 5a. Relocated to the appendix, every word preserved

| # | Moved | From → to | Body keeps |
|---|---|---|---|
| 1 | **Table 2, the latency panel** (`tab:qcomem-tradeoff`), both panels, caption and all notes | Sec. 5.3 body → Appendix F, immediately before the expanded deployment table that already repeated panel (a). Now **Table 22, p. 34** | Sec. 5.3 cites it twice; all of panel (a)'s numbers and all of panel (b)'s dispersion facts stay in Sec. 5.3 prose and Sec. 5.7. Appendix F's lead-in was updated from "is printed in the body" to "printed here for space rather than in the body". **This is the single largest and most contestable cut: 27 rendered lines. It is the first thing to restore if the page budget frees up.** |
| 2 | Related Work's three `\paragraph` headings and the "one state family — attention keys and values — produced on a single execution path …" contrast sentence | Sec. 2 → Appendix G, "The established quantization surfaces" | Every citation (13 keys, unchanged set), and all four disclaimers verbatim: "we build on that interface rather than claiming a new split-replay primitive", "Q-CoMem claims none of its mechanisms", "not a new selection method", "We claim the hybrid entry and its accounting, not quantize-and-recompute". XQuant keeps its own sentences and the concession that quantize-and-recompute "is already its thesis". |
| 3 | The calibrated/aggressive policies' calibration set and byte budgets ("selected on a disjoint four-item calibration set under a byte budget set by the frozen policy's predicted size"; "under a budget 25% below it") | Sec. 4.2 → Appendix C, beside the per-component reconstruction | Both bit vectors, the two-bit width disclosure, and "Each is one frozen operating point, not a claim that layer-wise selection is optimal". |
| 4 | The reason the three retained-state levels are printed rather than ratioed (SGLang's 128-token page quantization vs our element-exact payload) | Sec. 5.3 → pointer to Appendix F, where the same explanation already stood at greater length | "three levels we print rather than ratio, for the reason Appendix F gives". |

### 5b. Deleted from the body, still printed elsewhere in the paper

Each was verified to survive at least once in the final PDF.

1. Sec. 5.4: the two neighbouring paired intervals' values (`+0.3708 [−1.0533,+2.0345]`,
   `−0.1173 [−4.4576,+3.1106]`). Appendix B's inventory prints both. The body keeps the
   reading they license: "Both neighbouring pairs from that pass cross zero, so the frozen
   point separates from the quantized exact cache and from neither unquantized reference."
2. Sec. 5.4: the eight-item inversion's numbers (`9.74 MiB, 39.11, Δ=−0.022` vs
   `10.01, 39.01, Δ≈−0.13`). Table 22a and Appendix F's Table 20 notes print them. The body
   keeps the inversion itself and "we report the inversion rather than the favourable half of it".
3. Sec. 5.4 ¶2: the absolute F1s' per-arm intervals in prose; Table 1 prints them.
4. Sec. 5.3 ¶1: the four 60-item TTFT values, replaced by a Table 1 pointer; Table 1 prints
   them, and the adverse ordering sentence is unchanged.
5. Sec. 5.2: "agreeing with the 3.93x split step above" kept, but "product 5.8416" and the
   explicit fractions dropped; Appendix C derives all of it.
6. Sec. 3 ¶2: the historical-alias clause ("its tokens and terminal request state exact while
   a persistent base is corrupted"), which was the fourth of four instances. Kept in intro
   bullet 3, Sec. 5.5, the abstract and Appendix H.
7. **Abstract, numeric detail.** Dropped: `0.181 / 0.656 / 0.636` TTFT; `13.38 → 7.15–7.19`
   throughput; `54.11` ms TPOT; SGLang/HYPIC `0.146 / 0.072` and `139.53 / 324.09 / 15.89`;
   `56.438` and `−3.78 [−9.12, 0.32]`; the four allocator GiB endpoints. **Every statement
   they supported survives in the abstract in qualitative form** — "exact reuse, honest dense
   recomputation and two published same-stack prefix caches all reach the first token
   sooner", "throughput roughly halves", "the frozen policy is 5.84x smaller at 3.41 points
   higher mean F1", "the audited fork ties and is worse on generation increment" — and every
   number is printed in the body or appendix. The abstract retains `404.01` / `53.45`, the
   `−3.4122 [−8.4296,−0.0390]` interval with its near-zero qualification, the width-confound
   decomposition, and the full non-claim list. Net: abstract 53 → 61 rendered lines while
   absorbing ~25 lines of new content.
8. Conclusion: shortened 14 → 10 rendered lines. Dropped from it: the `404.01 / 53.45`
   restatement and "not Recall, process memory or faster inference" (both in the abstract and
   Sec. 5.7); "no width-matched arm executed" moved out of the conclusion but kept in the
   abstract, Sec. 5.2 and Table 1's notes.

### 5c. What was NOT cut

* **Figure 1 stays in the body**, page 3, at `0.81\textwidth` — the R47 asset, byte-identical,
  the width R49 measured at 8.10 pt labels. R49's restoration is not undone. It is the only
  figure in the main text and the gate is met.
* Table 1 stays in the body (page 6) with all eleven rows and both Store conventions.
* Sec. 5.7 Limitations: **not one clause removed**; two clauses rewritten to be *more*
  restrictive and six new exclusions added.
* Sec. 5.3's dispersion paragraph and its three "each against us" consequences: intact.
* Sec. 5.3's published-systems paragraph: intact, with its numbers.
* Sec. 5.5's "Against the appropriate reference the ownership component saves nothing" and
  the 2.229/2.843/1.950/0.019 tie: intact.

---

## 6. Build QA

```
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=build/r50_composition_v1 main_r50_composition.tex     → exit 0
```

| Gate | Result |
|---|---|
| Undefined references | **0** |
| Undefined citations | **0** |
| Multiply-defined labels | **0** |
| Overfull boxes | **0** |
| `LaTeX Warning:` lines | **0** |
| BibTeX warnings | **0** (39 entries, cited-key set identical to R49b) |
| Underfull `\hbox` | 92 — **identical count to the R49b baseline** (92); these are `\raggedbottom` justification slacks the template has carried since R42 and are not a regression |
| Main text ends | **page 9** (verified with `pdftotext -layout`: Sec. 6 heading on p. 9 line 476, last line "total memory for a large warm set." on p. 9) |
| References begin | **page 10** (Ethics / LLM / Reproducibility also on p. 10, as in R49b) |
| Figure in body | **yes** — Figure 1, page 3 |
| Total pages | 42 (R49b: 41) |

Rendered main text: 476 lines against R49b's 479, i.e. the paper absorbed the composition
section, the total-memory reframe and two new tables at a net reduction of three lines.

---

## 7. What I judged should not be written, and why

1. **No benchmark, task, trace or measurement for the multi-document / agentic regime.** The
   total-memory note forbids it explicitly and none was run. Sec. 5.6 says "The setting it
   suits is the opposite, many documents warm and each queried intermittently, which we
   describe qualitatively only, having run no benchmark, task or trace of that shape", and
   Table 5's notes repeat "No workload of the many-warm-document shape these rows describe
   was executed." One sentence, no citation, no implied evaluation.
2. **No statement about N > 8.** The fits rest on four points. The prohibition is written
   into Sec. 5.6, Table 5's notes and the abstract ("its fits license no N above 8"). The
   fitted crossing at N = 12.5 is reported only as evidence that the crossing lies *outside*
   the measured range — never as a prediction that Q-CoMem wins there.
3. **No claim that T is independent of D.** Stated as an assumption, three times, always
   with "unverified".
4. **No latency claim of any kind.** Frozen depth-7 remains behind exact caching *and* honest
   dense recomputation everywhere it is mentioned. The composition run records timings; the
   C1 note says its interleaved single-stream protocol is not a serving benchmark, so no
   timing from it is printed anywhere in the manuscript.
5. **No claim that sharing reduces per-request memory.** 4% is reported in the same paragraph
   as the audit result, and the licensed reading is written out.
6. **No F1 result from the composition run.** Sec. 5.1 says "its F1 is not a new quality
   result".
7. **No invented evidence-registry identifier.** The C1 note supplies internal QS trial
   numbers, not a registry alias. Minting an `E-…` ID would have put an unverifiable string
   into the artifact map, and printing raw trial numbers in an anonymous submission is both
   uninformative to a reviewer and a deanonymisation risk. The artifact-map row and the
   Reproducibility Statement therefore describe the artifacts — preflight gate, per-rank
   shards with per-target coverage and predicate rows, ownership byte accounting, token
   traces, and the blind replay that re-derives every status and reports drift as a defect —
   without an ID.
8. **No claim that the composition run validates the code path behind Tables 1 and 22.** It
   is a Read path built to borrow; the deployment tables ran the private-materialization
   path. Sec. 5.7 states this and keeps "ownership on the exact code path behind those tables
   still transfers by design argument".
9. **The 15.4% / 13.3% percentage form of the sharing saving** was not printed; the absolute
   MiB savings and the slope change carry the same content without a second denominator.
10. **Section 5.7's Store admission was not softened.** The C1 note's instruction was that
    the transient result *strengthens* it; Sec. 5.6 says so in those words and Sec. 5.7's
    sentence is unchanged.

---

## 8. Residual risk, stated for the next pass

* The composition and transient tables (Tables 4 and 5) and the latency panel (Table 22) are
  in the appendix, cited from pages 8–9. A reviewer reading only the main text sees the
  total-memory claim in prose with every number present but must turn to Appendix C for the
  crossover table. This is a page-budget consequence, not a judgement about importance; if a
  page is freed, restore Table 5 to Sec. 5.6 first and Table 22 to Sec. 5.3 second.
* The composition run is one entry, one depth, one bit vector, one stack, eight items,
  N ≤ 8. Sec. 5.7 says so. It does not close ISS-01, ISS-03, ISS-05, TS-04 or TS-05.
* Table 5's cells are arithmetic on two measurements, admissible under
  `derived_vs_measured_20260902.md`, but they are not themselves measurements and the
  manuscript never calls them one.
