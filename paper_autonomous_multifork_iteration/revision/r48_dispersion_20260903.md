# R48 — dispersion, the missing paired interval, and the Store reconciliation

Date: 2026-09-03
Source manuscript: `main_r47_corrections.tex` (untouched)
Produced: `main_r48_dispersion.tex`, `tables/qcomem_tradeoff_r48.tex`,
`tables/qcomem_validation60_r48.tex`
Bibliography: `\bibliography{references_r47}` retained unchanged (R47's arrangement
carried forward; `references.bib` and `references_r47.bib` both untouched, no new
citations were needed)
Figures: unchanged assets `figures/qcomem_pipeline_r47.pdf` and
`figures/qcomem_quantization_map_r47.pdf`
Build: `build/r48_dispersion_v1/main_r48_dispersion.pdf` (40 pp)

Scope: meta-review R46 `prioritized_actions` **A2, A4, A8, C4** only. Every number
is re-analysis already recorded in `revision/a2a4a8c4_analysis_20260903.md` or
already printed in R47. No new measurement, no projection, no residency figure,
no latency claim.

---

## A2 — the paired interval exists; the concession is replaced, not softened

Closes: **ISS-02** (meta-review A2).

The obsolete concession appeared in two places. Both are replaced, and the
near-zero upper bound travels with the number at every site.

| site | R47 | R48 |
|---|---|---|
| Sec. 5.4 | "We computed no paired interval between those two arms and their intervals against the shared reference overlap, so the retained-state ordering is the firmer." | "The paired interval between those two arms now exists: over the 60 common item keys, full-prefix Q8 minus frozen $j=7$ is $-3.4122$ $[-8.4296,-0.0390]$ (10,000 paired item-level resamples, seed 20260903), excluding zero, so the ordering is supported on this cohort rather than merely observed. We claim that and no more: the upper endpoint is $-0.0390$, near enough to zero that a different resample sequence could carry it across, so the interval is evidence neither that the effect is large nor that it is robust." |
| Table 1 footnote | "none was computed between two non-reference arms" | "All ten pairwise comparisons among the operating-point block's five arms, including the frozen-versus-Q8 pair, are now computed and listed in Appendix B" |

Both supporting readings from the note are in Sec. 5.4, written with explicit
sign direction so the reader cannot invert them by accident:

- full-prefix at native dtype minus frozen $j=7$ = $+0.3708$ $[-1.0533,+2.0345]$,
  flagged as the sign-reversed recomputation of the $-0.37$ row printed just above;
- dense minus frozen $j=7$ = $-0.1173$ $[-4.4576,+3.1106]$;
- both crossing zero, so "the frozen point separates from the quantized exact
  cache and from neither unquantized reference."

**Full ten-pair inventory** is in Appendix B (`tab:all-intervals`), as a third
labelled block "Operating-point run; all ten pairwise comparisons, pooled (seed
20260903)"; the caption now reads "All 52 paired-bootstrap intervals". A new
paragraph reconciles the pass against the printed rows: the four pairs that
duplicate Table 1 reproduce its point estimates **exactly** and its endpoints to
within **0.20 F1 points** — the resampling noise of an independently drawn
resample sequence, of the same order as the 0.075-point five-seed movement the
appendix already records. It states that one pair excludes zero and the other
nine do not, that the excluding pair's upper endpoint is 0.0390 from zero, and
that "None of the ten carries a multiplicity adjustment, and none is an
equivalence or noninferiority test."

The interval and its qualification also appear in the abstract, intro bullet 2
and the conclusion, each in the same breath as the 3.41-point difference:
"excluding zero by $0.0390$ points --- enough to support the ordering, not to
call the effect large or robust" (abstract); "(paired $-3.4122$
$[-8.4296,-0.0390]$, excluding zero at its edge)" (intro); "excluding zero by
$0.0390$ and no more" (conclusion).

Table 1's printed intervals were **not** overwritten with the ten-pair pass's
endpoints — see "what should not be written", item 4.

## A8 and C4 — the latency panel returns to the body, repaired

Closes: **TS-04, R46-REV-3-07** (A8) and **ISS-07** (C4).

**The eight-item table is back in the main body**, as Table 2 on page 7 beside
Table 1, and it is now a two-panel float (`tables/qcomem_tradeoff_r48.tex`):

- **Panel (a)** — the eight-item execution exactly as R47 printed it (six arms;
  Store, Reduction, TTFT, TPOT, tok/s, F1), with the throughput column carrying a
  dagger.
- **Panel (b)** — new: per-arm **TTFT min/median/max** and **TPOT
  min/median/max** over the operating-point run's three declared repeats, plus a
  **median generated-token count** column headed "tokens (cap 32)": 4, 4, 5, 4, 4
  against a 32-token cap, and a full-prefix TPOT maximum of **404.01 ms** against
  a 53.45 ms median.

**The aggregation rule is stated explicitly** in the footnote: block (a) is
medians over the eight documents with no dispersion (none was recomputed for that
execution); block (b) is the minimum, median and maximum of the three per-repeat
values. It also says plainly that Table 1's operating-point timing cells are that
run's recorded aggregates "under a rule it did not record", are therefore **not**
the block-(b) medians (full-prefix reads 0.181 s there against 0.161 s here), but
do each fall inside the matching min–max range.

**The reporting failure is named, not papered over.** The dagger note states that
throughput is recorded, not derived, and that no single generated-token count $n$
satisfies $n/(\mathrm{TTFT}+(n-1)\mathrm{TPOT})$ across panel (a) — $n\approx5.3$
full-prefix, $7.3$–$7.4$ on the four Q-CoMem rows, and none at or above one token
for dense, whose own timings give 1.541 tok/s at the cap against the 1.36 printed
— and that in the operating-point run, where per-repeat values can be matched,
the reconstruction still misses recorded throughput by a median **7.6–18.8%** per
arm. It closes: "the column must not be inverted for a token count."

**Sec. 5.3 gains "What the dispersion does to all of that."** It concedes that
readers were right, gives the two measured causes (generation halting after four
or five tokens, so the throughput denominator is four tokens and not the cap; and
a per-repeat spread of TTFT maxima 1.09–1.57 s against medians 0.16–0.67 s and a
404.01 ms TPOT maximum against a 53.45 ms median), then draws three consequences
"each against us":

1. the 2.5% TPOT band "compares point estimates whose repeat-level dispersion is
   unmeasured on that panel and, where measured, far outside the band, so it is
   not a tolerance";
2. "No TTFT ordering is resolved by three repeats, every pair of min–max ranges in
   panel (b) overlapping";
3. "the ordering is unchanged at every aggregation we computed and stays adverse:
   frozen $j=7$ is behind both exact caching and honest dense recomputation."

It ends "We make no latency claim, and record that this dispersion makes the
timing evidence weaker, not stronger."

**The two dense TPOT values are flagged, not reconciled.** Sec. 5.6 now records
that the eight-item panel's dense TPOT of 648.75 ms "equals its own TTFT to three
digits, the re-prefill signature, against 53.46 ms for the operating-point run's
prefill-once dense arm, a discrepancy we flag and do not reconcile, its shards not
being re-analysed."

The abstract's throughput sentence now rests on stated metric behaviour: the
13.38→7.15–7.19 tok/s and 2.5%-TPOT figures are called "point estimates over wide
per-repeat spread", followed by the four-to-five-token median, the 404.01 ms
maximum, the non-reconciling column, and "so we print minimum, median and maximum
per timing cell and read this evidence as weaker, not stronger."

Sec. 5.6 also carries the standing limitation: three repeats, per-repeat maxima of
404.01 ms TPOT against a 53.45 ms median, four-to-five generated tokens against a
32-token cap, "so no timing difference is resolved and the eight-item throughput
column reconciles with no generated-token count." Appendix I's limitation on the
eight-item execution gains "its throughput column reconciles with no
generated-token count, no per-repeat dispersion was recomputed for its cells."

## A4 — the length explanation is withdrawn, not defended

Closes: **TS-05, ISS-06**.

R47 (Sec. 5.1): "the whole 136.235-versus-140.34 and 9.661-versus-10.01
difference is cohort and statistic --- a 60-item mean against a median over
documents 3.8% longer --- with 0.000 MiB left to the accounting."

R48 (Sec. 5.1): "…is cohort and statistic, not length: document token counts are
identical across the operating-point run's arms and its own full-prefix mean Store
matches the archival cohort exactly, so we withdraw the earlier explanation of
documents 3.8% longer: 69.3% of the gap is which eight of the sixty documents the
timing panel draws and 30.7% is a 60-item mean against an eight-item median, with
0.0% left to the estimand (Appendix C)."

Table 2's footnote no longer contains R47's "the 60-item panel reports means over
a cohort with 3.8% shorter documents, which is why its full-prefix figure is
136.235 rather than 140.34 MiB"; that sentence is gone entirely.

The document-length statistics A4's verification criterion asks for are printed in
Appendix C, beside the component identity they belong to: "the operating-point
run's document token counts are identical across its five arms (minimum 1,146,
median 4,000, maximum 4,050), its own full-prefix mean Store is 136.2354
MiB/document, matching the archival cohort exactly, and the archived per-item rows
attribute 69.3% of the gap to which eight documents the timing panel draws and
30.7% to mean versus median, 0.0% to a difference of estimand." The 42.8%
FP32-GDN total-versus-excess sentence moved there from the Table 2 footnote and is
preserved word for word in substance ("a total, where the 30.000 MiB here is an
excess over BF16, and not two readings of one quantity").

---

## Build QA

`latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/r48_dispersion_v1 main_r48_dispersion.tex`
from a clean output directory: **exit 0**.

| gate | result |
|---|---|
| undefined references | 0 |
| undefined citations | 0 |
| multiply-defined labels | 0 |
| overfull boxes | 0 |
| LaTeX/package/class warnings | 0 |
| underfull boxes | 92 (R47 baseline: 92; not a gate, unchanged in kind) |
| main text ends | Conclusion **starts and completes on page 9** |
| References | begin on **page 10** (page 10 opens with the Ethics Statement, as in R47) |
| Tables 1 and 2 | both in the body, page 7 |
| total | 40 pages (R47: 40) |

Verified from the rendered PDF with `pdftotext`.

### Evidence-survival audits against R47's rendered text

1. **Curated scope audit (41 strings).** Every ForkAudit scope site, non-claim,
   scope statement and limitation R47's report listed was string-matched in R48's
   rendered text, including "audits the ownership discipline and not the quantized
   Read path", "materializes a full private copy of the dequantized entry per
   query, so it shares nothing and exercises neither borrowing nor copy-on-write",
   "the quantized Read path is not audited", the three untested packed-entry
   obligations, "no ForkAudit verdict here covers those packed entries",
   "ownership there transfers by design argument only", "not a
   statistical-equivalence claim", "an interval crossing zero is not statistical
   equivalence", "selected without multiplicity adjustment", "we make no recall
   claim", "Recall was measured in no cohort", "nor evidence of higher admitted
   capacity", "trusted computing base (TCB)", "not security attestation", "not
   exclusive detection", "packed Read path drawn here is unaudited", "H20 is not an
   edge device", "not source-frozen regeneration", "controlled ownership factor",
   "not an optimized baseline", "neither additive with Store", "do not evaluate
   continuous batching", "No width-matched exact cache was executed", "not a
   detection study or an error rate", "proves neither equivalence nor
   noninferiority", "none is computed but unreported". **Zero losses.** The single
   intentional replacement is A2's "none was computed between two non-reference
   arms", which is now false.
2. **Numeric audit.** Every numeric token in R47's rendered text (2,660 distinct)
   was checked for survival in R48. **Zero losses.** One token, `0.5644`, was
   briefly dropped when Sec. 5.2's component arithmetic was compressed and was
   restored in Appendix C's identity (`30×0.5644`).
3. **Negation-sentence audit.** All 222 negation-bearing sentences in R47's
   rendered text were checked for a surviving 7-gram. Six flagged; four are
   rewordings or page-break artefacts whose content was confirmed present
   ("not just larger", "never pooled", the 42.8% total-versus-excess sentence,
   "length/input robustness"); the remaining two are the intentional A2 replacement
   and one deleted signpost (below).

---

## What was cut, exactly

R47's main text ended exactly on page 9. R48 adds roughly 1.5 pages (Table 2 in
the body with its second panel and repair footnote, Sec. 5.3's dispersion
paragraph, Sec. 5.4's A2 passage, and the A2/A8 additions to the abstract, intro,
conclusion and limitations). The following was removed or relocated to pay for it.

### Relocated to the appendix — nothing deleted, every word preserved

1. **Figure 1 (protocol geometry) moved from the body to Appendix C**, beside
   Figure 2, caption verbatim including the scope sentence "the packed Read path
   drawn here is unaudited". The intro's reference now reads "(Figure 1, in
   Appendix C for space)". This is the largest single cut (~0.45 page) and it is a
   pure exposition asset that argues *for* the paper; the body now carries no
   figure, which is the change most worth flagging to the next reviewer.
2. **Sec. 5.2's "The transient term, for both arms" paragraph (the A3 correction)
   moved verbatim to Appendix C.** The body keeps a pointer that preserves both of
   its scope clauses: "The method-dependent transient term of Eq. 1 is reported for
   both arms in Appendix C rather than assumed to cancel; it is arithmetic on
   Table 4, not an allocator measurement, and fixes no concurrency limit." This
   paragraph argues *for* the paper (Q-CoMem has both the smaller intercept and the
   smaller slope), which is why it was chosen.
3. **Sec. 5.5's allocator derivation moved verbatim to Appendix G** (under
   "Primary allocator endpoints"), including the 4.901→2.229 GiB / 4.920→2.843 GiB
   figures, the 54.5%/42.2% percentages and the block-sharing citation. The body
   keeps B4's adverse-first framing intact: the tie at 2.229/2.843 GiB, the
   1.950-versus-0.019 GiB generation-increment regression, "it buys auditable
   request-local state, not fewer allocator bytes", and "a controlled ownership
   factor and not an optimized baseline". The favourable half is what moved.
4. **Sec. 5.5's per-target pass detail moved to Appendix A** as a new paragraph
   "What the seven targets establish" (semantic-observable match across the 96
   configurations, the rebind statement, the $N=8$ selected-cell limitation with
   Table 6, and the primary rerun's stack/PG-19 citations). The body keeps "All
   seven targets have complete mandatory-trace coverage and pass at their declared
   fixed-stack scope, with the live-binding rerun's stronger owner-row binding
   covering only one selected cell (Section 5.6)", and Sec. 5.6 states the
   selected-cell limitation in full.
5. **Sec. 5.5's fixed-case detection counts moved to Appendix G** (0 of 5
   semantics-completing mutants for token equality, 1 of 5 for exact logits, 8/8
   cells in the no-injection defect). The body keeps the qualitative statement,
   "not exclusive detection", and "not a detection study or an error rate".
6. **Sec. 5.4's two ablation details moved to Appendix B** (the calibrated policy's
   2.75% Store saving with one of 60 items crossing the threshold; uniform Q4's
   −3.047 points on the eight-item execution). The body keeps the eight-item
   inversion with both deltas, the selector rationale, and "we report the inversion
   rather than the favourable half of it".
7. **Sec. 5.1's document-length statistics moved to Appendix C** (min 1,146 /
   median 4,000 / max 4,050 and 136.2354 MiB/document). The body keeps the
   withdrawal of the 3.8% explanation and the 69.3/30.7/0.0 decomposition.
8. **Sec. 5.2's component arithmetic reduced to its factors.**
   `20×1.9752+30×0.5644=56.436` and `20×1.0457+16.932=37.846` are printed in
   Appendix C only; the body keeps 37.846/9.661 = 3.9174, 56.436/37.846 = 1.4912,
   product 5.8416, the 37.85 MiB point, and "No width-matched exact cache was
   executed", so A1's verification criterion still holds at every 5.84× site.
9. **Table 2's 42.8%/140.34 total-versus-excess sentence moved to Appendix C.**

### Deleted outright — three sentences, none a claim, non-claim, scope statement, limitation or number

1. Sec. 5.2: "Uniform Q4 and the aggressive mixed policy go further still, but
   Section 5.4 shows why neither is the headline point." Pure signposting.
2. Sec. 5.1: "The two blocks are never pooled and each is differenced against its
   own full-prefix arm." A verbatim duplicate of Table 1's caption ("blocked and
   never pooled, each differenced against its own full-prefix row") and of Sec.
   5.1's own "every quality difference uses the full-prefix arm of its own
   execution"; both survive.
3. Sec. 5.4: "We therefore use frozen Q4/Q4/Q8 at $j=7$ as the headline point."
   The selection rationale immediately preceding it survives verbatim, and "the
   headline policy was selected among six on this panel" and Table 1's footnote
   both remain.

### Compressed

Roughly 60 further lines of wording across the abstract, introduction, Related
Work, Motivation, Sec. 4.2–4.4, Sec. 5.1–5.6, the Conclusion and the two table
captions/footnotes. Motivation's two run-in `\paragraph` headings were dropped and
its second paragraph folded into a single sentence opening; the Conclusion's
opening method sentence was merged into its first result sentence. Every citation
was retained. The numeric and negation audits above are the check that no number,
claim, non-claim, scope statement or limitation was lost in this pass.

---

## What I judged should not be written, and why

1. **No per-repeat dispersion for the eight-item panel.** The analysis note
   re-analyses only `runs/qcomem/r45-d7d13-60item-20260903a`. Importing that run's
   spread onto the eight-item cells would be a projection under
   `derived_vs_measured_20260902.md`. Panel (a) therefore prints medians and the
   footnote says in terms that no dispersion was recomputed for that execution.
2. **No claim that the 2.5% TPOT band survives the dispersion.** C4's verification
   criterion asks for the band and the TTFT ordering to be "shown to survive". The
   band is not shown to survive and the paper now says so: it is "not a tolerance",
   and no TTFT ordering is resolved by three repeats. The ordering itself is
   retained only in its adverse direction, which is what the measurement supports.
3. **No reconciliation of the two dense TPOT values.** The note does not address
   the eight-item dense arm. Sec. 5.6 gives the one mechanical observation the
   printed numbers license (648.75 ms ≈ its own 0.649 s TTFT, the re-prefill
   signature) against the operating-point run's prefill-once dense arm, and says
   explicitly that we cannot say which behaviour the eight-item row measured.
   A8 permits "explain or withdraw"; withdrawing an adverse row would remove
   evidence against the paper, so it stays, flagged.
4. **No statement of what statistic Table 1's operating-point TTFT/TPOT cells
   are, and no overwriting of Table 1's intervals.** Neither
   `evidence/experiment_registry.json` nor `revision/a16_operating_point_gate_20260903.md`
   records an aggregation rule for the timing columns (both record only "item F1
   averaged over repeats"), so the paper says the rule was not recorded rather than
   guessing "mean". Likewise Table 1's printed intervals remain the A16 registered
   extraction; the ten-pair pass's endpoints differ by up to 0.20 F1 points as
   Monte-Carlo noise on the same point estimates, and Appendix B reconciles them
   instead of silently replacing them.
5. **No upgrade of the −3.4122 interval.** It is reported as excluding zero and
   nothing more, with the −0.0390 upper endpoint quoted at every one of the four
   sites, no equivalence or noninferiority reading, and an explicit statement that
   none of the ten pairs carries a multiplicity adjustment.
6. **No latency claim, no projection, no residency figure.** Verified by probe:
   the resident-document counts (7659.7 / 1314.7 / 4595.6 / 540.1), the retired
   ~3.1× and ~29.88 MiB projections, and the n=128/512 throughput projections
   (91.8%, 96.5%, −1.74%) appear nowhere. The only "resident documents" occurrence
   is R47's own intro phrase about competing for device memory, and the only "3.1×"
   is Prompt Cache's published-context row, both pre-existing.
7. **No per-item F1 for the eight indices under both protocols** (R46-REV-3-05's
   remaining half). That extraction still does not exist, so the budget-versus-
   item-difficulty decomposition is still declined in Sec. 5.1, as in R47.
8. **`references.bib` and `references_r47.bib` were not modified**, and R48 builds
   against `references_r47.bib`, so R47's build output is bit-identical to before
   this pass. `main_r47_corrections.tex`, all earlier manuscripts, `review/` and
   `state/` are untouched.
