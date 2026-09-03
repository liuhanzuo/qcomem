# R47 — panel-and-meta-review corrections pass

Date: 2026-09-03
Source manuscript: `main_r46_baseline.tex` (untouched)
Produced: `main_r47_corrections.tex`, `references_r47.bib`,
`tables/qcomem_validation60_r47.tex`, `tables/qcomem_tradeoff_r47.tex`,
`figures/qcomem_figures_r47.py` → `figures/qcomem_pipeline_r47.pdf`,
`figures/qcomem_quantization_map_r47.pdf`
Build: `build/r47_corrections_v1/main_r47_corrections.pdf` (40 pp)

Scope: meta-review R46 `prioritized_actions` A1, A3, A6, A7 and B1–B7. A2, A4, A8
and C4 were left untouched by instruction; no text bearing on them was altered
except where a shared sentence had to move, and no number was invented for them.

---

## A1 — bit-width decomposition of the 5.84x (ISS-01)

Closes: ISS-01.

Every site that carried `5.84x` now carries the decomposition, and each states
that no width-matched exact cache was executed.

| site | what it now says |
|---|---|
| Abstract | 5.84x, then "That arm stores every attention layer at 8 bits while the frozen policy stores attention at 4 … rebuilding a width-matched exact cache from the same component inventory gives 37.85 MiB/document, and the measured 5.84x factors exactly into 3.918x from the split and 1.491x from bit width. No width-matched exact cache was executed." |
| Intro, bullet 2 | "5.84x larger and 3.41 F1 points lower — a ratio that is not width-matched and factors into 3.918x from the split and 1.491x from bit width" |
| Sec. 5.2 | full arithmetic: `20x1.9752 + 30x0.5644 = 56.436` against the printed 56.438; `20x1.0457 + 16.932 = 37.846`; `37.846/9.661 = 3.9174`, `56.436/37.846 = 1.4912`, product `5.8416`; the 3.918x agrees with the panel's own 3.93x split step; italicised "No width-matched exact cache was executed" |
| Sec. 5.4 | "that 5.84x is the width-confounded ratio Section 5.2 decomposes, and no width-matched arm exists to say whether the F1 ordering survives at matched precision" |
| Conclusion | "a width-confounded ratio that factors into 3.918x from the split and 1.491x from bit width, with no width-matched arm executed" |
| Table 1 footnote | "Full-prefix Q8 packs the unsplit entry at a uniform 8-bit width, *not* the frozen policy's bit vector, so the 5.84x … is width-confounded" |
| Appendix C (new paragraph) | "The exact-cache arm, and the width-matched point it is not" — the identity, the displayed factorisation, and the explicit statement that the 37.846 MiB arm was never executed and carries no F1, no interval, and no quality claim |

Also renamed the arm everywhere from "an exact cache packed with the same codec"
to "the unsplit entry under the same codec **at a uniform 8-bit width**", and
added the same qualifier to Sec. 5.1's description of the operating-point run.

Admissibility. 37.846 MiB/document is stated as a **byte count fixed by the
Eq. 3 format identity once the bit vector is given**, in the same class as the
all-BF16 re-expression the paper already prints (106.235 MiB/document is likewise
not an executed configuration). No quality, latency, or capacity property is
claimed for it. This is the only reading under which `derived_vs_measured_20260902.md`
permits it: the packer's byte identity is verified 180/180 item-policy pairs, so
the byte count is an identity, not a projection; anything about its F1 remains a
C2 experiment.

## A3 — transient working set, and Eq. 1's false premise (TS-02)

Closes: TS-02 (motivation-level residual only; the reviewer's crossover bound is
*not* reproduced, per the meta-review's rejection of it).

- Sec. 3, after Eq. 1: the sentence claiming workspace is set by request shape is
  replaced with "The workspace term is not method-independent, and our own Read
  path is why: it materializes a private dequantized copy of the entry for each
  concurrent request (Sec. 4.3), so `M_active` scales with the entry a method
  chose to retain, and Sec. 5.2 prints it for both arms instead of assuming it
  cancels."
- Sec. 5.2, new paragraph "The transient term, for both arms": 28.683 MiB/request
  for the implemented Read; 31.875 MiB/request in BF16 and 61.875 in the stack's
  FP32 for the full-prefix reference's own mutable GDN state. Under one dtype
  convention the resident-byte lines are `9.661 + 28.683N` against
  `106.235 + 31.875N`; Q-CoMem has **both the smaller intercept and the smaller
  slope, so they do not cross at any N**.
- Both lines are labelled arithmetic on Table 4, not allocator measurements, and
  Sec. 5.6 adds that the transient term "fixes no concurrency limit for either arm".

The rejected one-sided bound (N ≈ 4.4) appears nowhere.

## A6 — Store estimand (ISS-05, R46-REV-5-02)

Closes: ISS-05, R46-REV-5-02.

All three PDF definitions of Store were byte-range-union; all three now name what
the code computes:

- Sec. 5.1 "Memory denominators": "the deduplicated sum of the bytes of the tensor
  storages one retained entry owns: the residual term is numel x element size, the
  cache term sums untyped-storage bytes deduplicated by (device, data pointer,
  storage size)… It is **not** the entry's physical byte-range union, which the
  same code computes but does not print; that distinction was previously stated
  the other way round."
- Table 2 (now Table 20) footnote: same wording, pointing at Sec. 5.1.
- Sec. 5.6: "Store, the deduplicated sum of the tensor-storage bytes one entry owns".
- Intro closing scope sentence: same.
- Appendix H.1 cross-runtime comparison restated: the SGLang-side receipts measure
  a byte-range union while our printed column is the owned-storage sum, and it is
  the **measured coincidence** on these eight documents (byte-exact at all five
  retained configurations) that licenses printing the levels side by side, not a
  shared definition.

No number moved. The agreement statement is the one already in the manuscript and
in `revision/a8_a10_drafts_20260902.md` (residual unexplained 0 B on every matched
row; 780 byte-exact closed-form identities, 0 mismatches).

## A7 — F1 scorer attribution (R46-REV-5-05)

Closes: R46-REV-5-05.

- Sec. 5.1: "F1 is our reimplementation of the LongBench-v1 `qa_f1_score`
  protocol, not the upstream module", with both documented deviations in the main
  text (empty prediction/reference → `float(pred tokens == ref tokens)`;
  ASCII-only punctuation stripping) and the explicit note that the citation is the
  protocol source, not the code that produced the numbers.
- Appendix A: "its official F1 implementation" replaced with the same attribution.
- `references_r47.bib` (a copy; `references.bib` untouched): the `bai2024longbenchcode`
  annotation no longer says "F1 computed by LongBench/metrics.py::qa_f1_score via
  LongBench/eval.py"; it now reads "cited as the protocol source for the
  `qa_f1_score` metric definition, not as the code that produced the F1 values
  reported here", with the reimplementation noted.

## B1 — generation budgets, F1 level gap, policy-ordering inversion (ISS-08, R46-REV-3-05)

- Sec. 5.1 heading renamed "Quality cohort and generation budgets"; states greedy
  decoding with an EOS stop and a per-dataset budget of 128 new tokens (Qasper) /
  32 (2WikiMQA) for both 60-item panels, 32 for both datasets on the timing panel,
  "so the three panels' F1 levels are not interchangeable". Also adds the
  first-half/last-half middle-drop truncation rule and the 256-token minimum.
- Sec. 5.1 "Latency execution": the eight items are "a subset of the quality
  cohort's *items* but not of its measurement protocol", and the 39.14-versus-54.68/54.67
  gap is named. We say we do **not** decompose the gap into budget and
  item-difficulty components, because per-item F1 was not extracted under both
  protocols — see "not written" below.
- Sec. 5.4: the eight-item panel's inversion is stated with both deltas
  (calibrated 9.74 MiB / 39.11 / −0.022 versus frozen 10.01 / 39.01 / ≈ −0.13 from
  the printed means), with the reason the 60-item cohort is the selector.
- Both table captions carry the budgets.

## B2 — abstract cohort mixing (ISS-04)

The abstract now reports the **60-item** TTFT (0.181 s full-prefix, 0.656 s frozen,
0.636 s dense) and marks the eight-item numbers as such ("on an eight-item subset,
median throughput falls from 13.38 to 7.15–7.19 tokens/s at TPOT within 2.5% of
54.11 ms"). No eight-item median appears under a 60-item framing.

## B3 — published same-stack systems promoted (R46-REV-3-06)

- Sec. 5.3 gains a paragraph opening "The comparison a reader should hold us to is
  against deployed systems, not only our two in-house endpoints, and it is
  harder": SGLang 0.5.17 Radix at 0.146 s / 15.83 tok/s and the published HYPIC
  codebase's prefix cache at 0.072 s / 19.07 tok/s, both at F1 39.14, against
  0.673–0.674 s and 7.15–7.19 tok/s at 39.01–39.14 here; retained state 139.53 and
  324.09 MiB/document against 15.89, printed as three levels rather than a ratio.
  Sec. 5.3 now cites Tables 20, 21 and 22 from the main text.
- The abstract carries the same comparison.
- Sec. 5.6 adds the published prefix caches to the TTFT limitation.

## B4 — rebalanced headline presentation (R46-REV-3-04, R46-REV-1-03, TS-03)

(i) The all-BF16 column is out of the footnote and into Table 1 proper, at body
table size, as two columns ("Store, BF16 ref." MiB/doc and ratio) printed **first**;
Sec. 5.2 leads with it ("We lead with the all-BF16 column, because it measures what
this paper contributes rather than the stack's dtype choices"), and the abstract,
intro bullet and conclusion all lead with 10.9965x and give 14.10x second.

(ii) Sec. 5.5's allocator paragraph now opens "Against the appropriate reference the
ownership component saves nothing", states the tie (2.229/2.843 GiB) and the
1.950-versus-0.019 GiB generation-increment regression first, and demotes the
54.5%/42.2% to "measured against a different arm". The abstract and intro bullet 3
carry the tie and the regression in the same sentence as the percentages.

## B5 — reproduction specification (TS-06, R46-REV-5-03/04/06)

- Sec. 4.2: grouping axis (inside the hidden dimension for the residual, requiring
  hidden size divisible by 64; flattened element order with edge-value padding for
  the cache leaves), the degenerate-group convention (`s = 1` when `u = m`), and
  the dequantizer (`x̂ = qs + m` in FP32 from BF16 `s`, `m` rounded after `q` is
  computed, cast back to the leaf's original dtype, so an FP32 leaf round-trips as
  FP32).
- Sec. 4.2 also prints all three bit vectors as configuration values.
- Sec. 5.1 / Table 20 caption: model revision `59d61f3`, PyTorch 2.11.0+cu129,
  Transformers 5.14.1, CUDA 12.9; the PDF states plainly that the interpreter
  version was not recorded at execution time.
- Appendix B: the bootstrap generator (Python `random.Random`, indices drawn one at
  a time, order statistics at floor(0.025(n−1)) and floor(0.975(n−1))), with the
  warning that endpoints reproduce only for this generator, seed sequence and
  repetition count.
- Appendix A, new paragraph "What a reader receives, and what is missing": the
  package accompanies the submission as anonymous supplementary material; eleven
  provenance rows are not archived in any package (split depth, three bit policies,
  policy search, group-size verification, panel runner and launcher, Store
  accountant, deployment accountant, timing driver, bootstrap and aggregation), and
  those are exactly the components that select the operating point and produce the
  uncertainty, so Tables 1 and 20 are replayable-from-outcomes and not re-executable.
- The Reproducibility Statement carries the same availability/omission statement,
  moved out of Appendix A per R46-REV-5-03.

## B6 — figures and nomenclature (R46-REV-3-02, R46-REV-3-03)

Figures regenerated from a deterministic script, not hand-edited:
`figures/qcomem_figures_r47.py` imports the R43 palette and drawing primitives
unchanged and reflows the two single-row compositions into bands so the labels can
grow. Measured from the rendered PDF with `pdftotext -bbox`:

| | R46 (reviewer's measurement) | R47 (measured here) |
|---|---|---|
| body text | 8.9 pt bbox | 8.91 pt bbox |
| Figure 1 labels | 4.8–5.1 pt | **8.40 pt** (all label words) |
| Figure 2 labels | 4.5–5.0 pt | **8.68 pt** (all label words) |

Sub/superscripts inside figure math (6.2–7.1 pt) are the only smaller glyphs and
match body-text math behaviour. Figure 1 is included at 0.84\textwidth and
Figure 2 at 0.85\textwidth; the r43 assets are left in place and unused.

Content changes to the figures, both authorised by R46-REV-3-02: Figure 1's
"retention–online-work trade-off" band and Figure 2's scope banner are removed as
verbatim restatements of the surrounding prose. Figure 2(b) now prints the frozen
bit vector `[8,8,8,4,8,8,8]` and its caption names the calibrated vector
`[8,8,4,4,8,8,8]` as a different policy that is not drawn — the one-position
confusion REV-3 flagged.

Nomenclature, one name per arm and per policy:

| object | canonical name | removed aliases |
|---|---|---|
| unpacked reference | **full-prefix, native dtype** | "full-prefix Q16" (Sec. 4.2 now says explicitly that we do not write "Q16", because that arm asserts no uniform width) |
| unpacked split arm | **Q-CoMem split, native dtype** | "Q-CoMem split Q16" |
| quantized exact cache | **full-prefix Q8 (uniform 8-bit)** | bare "Q8" |
| split arms in the timing panel | **Q-CoMem split Q8 / split Q4** | bare "Q-CoMem Q8 / Q4" |
| calibrated policy | **calibrated per-layer** | "same-memory mixed", "per-layer mixed", "the same-memory per-layer policy", "the calibrated per-layer row", "the same-memory policy" |
| audit-stack KV | **BF16** | "BF16/Q16", "vLLM Q16 adapter", "vLLM-Q16 stack", "persistent-Q16" |

The token "Q16" now appears exactly twice in the source, both in the sentence and
caption that retire it. Sec. 4.2 carries the one-line policy glossary.

## B7 — XQuant positioning (R46-REV-1-04)

- The sentence "each method compresses one homogeneous cache along a single
  execution path" is gone. It now reads: "these methods pack one state family ---
  attention keys and values --- produced on a single execution path, whereas a
  hybrid split-replay entry mixes a boundary residual, lower-layer attention KV,
  and mutable convolution and recurrent state".
- New Sec. 2 paragraph "Quantized intermediates with recomputation": XQuant is
  named the closest prior work to our core mechanism, its thesis is conceded
  ("persisting a quantized intermediate and paying recomputation to avoid
  retaining downstream state is already its thesis"), and the distinction is made
  on the two grounds the meta-review names — the cut point is a depth-*j* boundary
  residual replayed for a later query rather than a layer input rematerialised
  inside its own decode step, and the retained object is heterogeneous and raises
  an ownership question a dense activation cache does not.
- Table 14 (`tab:closest`) gains an XQuant row with mechanism and relation, and
  states that no head-to-head comparison is reported.

---

## Build QA

`latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/r47_corrections_v1 main_r47_corrections.tex`
from a clean output directory: **exit 0**.

| gate | result |
|---|---|
| undefined references | 0 |
| undefined citations | 0 |
| multiply-defined labels | 0 |
| overfull boxes | 0 |
| LaTeX/package warnings (final pass) | 0 |
| underfull boxes | 92 (R46 baseline: 92; not in the gate list, unchanged in kind) |
| main text ends | Conclusion completes on **page 9** |
| References | begin on **page 10** (page 10 opens with the Ethics Statement) |
| total | 40 pages |

Verified from the rendered PDF with `pdftotext` and `pdftotext -bbox`.

## What was cut for space

R46's main text ended exactly on page 9. The corrections added roughly 2.5 pages,
so the following was removed or relocated. Nothing here is a claim, non-claim,
scope statement, limitation, or number.

1. **Table 2 (eight-item latency panel) moved from Sec. 5.3 to Appendix H**, where
   it is now Table 20 and sits immediately before the expanded serving table that
   already repeated its rows. Sec. 5.3 quotes every one of its values in prose and
   cites it in its first sentence, and the appendix text says in terms that it is
   printed there "rather than in the body only for space". This is the largest
   single cut (~0.5 page) and the one most worth flagging to the next reviewer: the
   adverse latency evidence is fully stated in the main text but its table is not.
2. **Figure 2 (quantization map) moved to Appendix C**, next to the component
   accounting it illustrates, with all three bit vectors now printed in Sec. 4.2
   prose so nothing depends on reading the figure.
3. **Eq. 4 (the `Pass_i` predicate) moved from Sec. 4.4 to Appendix E**; Sec. 4.4
   states the same condition in one sentence and keeps the coverage/verdict
   separation and the whole TCB sentence verbatim.
4. Sec. 5.1's "Executed systems" paragraph folded into "Memory denominators"
   (the ForkAudit-does-not-cover-packed-entries sentence is preserved word for word).
5. Sec. 5.5's "What it found" and "Falsification beyond output equality" merged
   into one paragraph; every count and every non-claim survives.
6. The Sec. 5 preamble sentence ("Sections 5.2–5.4 report capacity, its online
   cost, and answer quality…") deleted as pure signposting.
7. Related Work's "Split-depth reuse" and "Paged and prefix-aware inference"
   paragraphs merged; "Evaluation and metamorphic testing" demoted from a
   `\paragraph` to a plain paragraph. All citations retained.
8. The Recall column was dropped from both tables (it was blank in every row); the
   statement "Recall was measured in no cohort" is printed in both captions,
   Sec. 5.4, Sec. 5.6, the intro and the abstract.
9. One favourable claim removed: Sec. 5.4's "The frozen point's −0.37 also
   reproduces the archival cohort's −0.45 against the same reference arm in an
   independent run." It is the only substantive sentence deleted rather than
   compressed, and it was deleted because it argues *for* the paper.
10. Roughly 120 further lines of wording compression across the abstract, intro,
    Sec. 3, Sec. 4.1–4.4, Sec. 5.1–5.6, the conclusion and the three statements.
    Table 1's footnote lost its restatement of the 2.97x normalised step (still in
    Appendix C and now derivable from the two printed columns).

Scope audit. Every ForkAudit scope site, every non-claim and every limitation
present in R46's rendered text was checked for survival in R47's rendered text by
string match, including "audits the ownership discipline and not the quantized Read
path", "materializes a full private copy … shares nothing and exercises neither
borrowing nor copy-on-write", "the quantized Read path is not audited", the three
untested packed-entry obligations, "no ForkAudit verdict here covers those packed
entries", "ownership there transfers by design argument only", "not a
statistical-equivalence claim", "an interval crossing zero is not statistical
equivalence", "none was computed between two non-reference arms", "selected without
multiplicity adjustment", "no recall claim", "nor evidence of higher admitted
capacity", "trusted computing base (TCB)", "not security attestation", "not
exclusive detection", "packed Read path drawn here is unaudited", "H20 is not an
edge device", "not source-frozen regeneration", "controlled ownership factor", "not
an optimized baseline", "neither additive with Store", "do not evaluate continuous
batching". **Zero losses.**

## What I judged should not be written, and why

1. **No F1, interval, or quality statement for the 37.846 MiB width-matched arm.**
   The byte count is an identity; anything about its quality is C2 and would be a
   projection of an unrun configuration under `derived_vs_measured_20260902.md`.
   The manuscript says so in Sec. 5.2 and Appendix C.
2. **No decomposition of the 54.68-versus-39.14 F1 gap into budget and
   item-difficulty shares.** The generation-budget difference is stated as the
   mechanism that makes the levels incomparable, but the per-item F1 for the eight
   indices under both protocols has not been extracted, so any share would be
   invented. The PDF says this in Sec. 5.1 in terms. R46-REV-3-05 also asked for
   per-item F1 for the eight indices to be printed; that needs the extraction and
   is left for the A2/A4 pass.
3. **No XQuant row in Table 19 (published context) and no XQuant compression or
   perplexity numbers.** `literature/reported_system_context.json` contains no
   XQuant entry, so the only available source for "~7.7x at <0.1 ppl" was the
   reviewer's own text. The meta-review's B7 requires only the Sec. 2 paragraph;
   the positioning is made on mechanism, and Table 14 gains a numberless row.
4. **No residency figure anywhere.** Per the correction section of
   `revision/a16_operating_point_gate_20260903.md`, resident-document counts are an
   analytic corollary of Store under a budget model. None appears in R47, and the
   per-request transient of A3 is explicitly labelled arithmetic on the component
   table and explicitly does not fix a concurrency limit.
5. **No claim that the 60-item cohort was preregistered as the policy selector.**
   R46-REV-3-05 asked for "the pre-registered reason"; I could not verify a
   preregistration of that choice, so Sec. 5.4 gives the defensible reason instead
   (larger cohort, paired intervals, full generation budget).
6. **A2, A4, A8, C4 text left exactly as R46 had it.** In particular the
   136.235-versus-140.34 reconciliation still says "3.8% longer" and "0.000 MiB left
   to the accounting" (A4 disputes the percentage), the frozen-minus-Q8 paired
   interval is still absent with the existing "we computed no paired interval
   between those two arms" disclosure, and the latency panel's tok/s column is
   unrepaired. None of these was touched, and no number for them was estimated.
7. **`references.bib` was not modified.** The corrected annotation lives in a copy,
   `references_r47.bib`, so R46's build output is unchanged.
