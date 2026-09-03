# R45 integration — `main_r45_evidence.tex`

Date: 2026-09-03
Source: `main_r44_structure.tex` (unmodified; SHA/MD5 `e57fda0a00acb2eb8a7be045d72ebc95` unchanged)
Target: `main_r45_evidence.tex`
Build: `build/r45_evidence_v1/`

Integrated: **A2, A3, A6, A8, A9, A10, A11, A12**. Nothing that depends on a
running experiment was touched: the `j=7` framing, the split-depth sweep, any
residency figure, any quantized-exact-cache comparison, and any new latency
claim are all absent. Every number written traces to
`revision/derived_vs_measured_20260902.md`'s admissible table, to one of the
A2/A8/A9/A11 notes, or to a value already in R44.

Files changed: `main_r45_evidence.tex` (new), `references.bib`,
`tables/qcomem_validation60_r42.tex`, `tables/qcomem_tradeoff_r42.tex`,
`tables/h20_deployment_table_r40_candidate.tex`,
`tables/mac_m4_motivation_table_r39_revised.tex`,
`tables/mac_m4_motivation_table.tex`,
`tables/related_work_reported_context.tex`,
`tables/target_coverage_compact.tex` (new). Nothing under `review/` or
`state/` was written.

---

## 1. Build QA (hard gates)

```
latexmk -pdf -interaction=nonstopmode -halt-on-error \
        -outdir=build/r45_evidence_v1 main_r45_evidence.tex     -> exit 0
```

| Gate | Result |
|---|---|
| Undefined references | **0** |
| Undefined citations | **0** |
| Multiply-defined labels | **0** |
| Overfull boxes | **0** |
| BibTeX errors/warnings | 0 |
| Pages | 35 (R44: 31) |
| Underfull hboxes | 91 (R44 baseline: 85; not a gate) |

Cited keys vs. `.bbl`: **40 cited, 40 resolved, 0 uncited entries emitted.**
All 16 new keys resolve.

**Label survival.** Verified defined exactly once in the `.aux` and resolving:
`sec:contract`, `sec:correctness`, `sec:results`, `sec:target-status`,
`sec:limitations`, plus `sec:intro`, `sec:related`, `sec:motivation`,
`sec:method`, `sec:decomposition`, `sec:write`, `sec:read`, `sec:experiments`,
`sec:setup`, `sec:denominators`, `sec:memory`, `sec:latency`, `sec:quality`,
`app:reproducibility`, `app:protocol-geometry`, `app:storage-schema`,
`app:invariants`, `app:novelty-boundary`, `app:controls`,
`app:expanded-limitations`, `app:artifact`, and every `tab:`/`fig:` label used.
Two new labels added, no collisions: `app:all-intervals`, `tab:all-intervals`,
`tab:target-coverage`.

**Page structure.** Page 9 spans margin lines 432–485.

| | R44 | R45 |
|---|---|---|
| Section 6 Conclusion heading | line 450 (p.9) | line 470 (p.9) |
| End of main text (Conclusion body) | line 468 (p.9) | **line 481 (p.9)** |
| Ethics Statement | Appendix B (p.11) | p.10, line 486 |
| LLM-use statement | absent | p.10, line 494 |
| Reproducibility Statement | line 465 (p.9) | p.10, line 502 |
| References begin | p.9, line 474 | **p.10, line 513** |

Main text ends on page 9; References start on page 10. Both gates hold.

---

## 2. What was integrated, per action

### A2 — single reference arm (closes `R44-4-01`, `R44-4-08`, `R44-4-16`, `Q5`, `T-11`)

* Abstract quality sentence rewritten to full-prefix Q16: `-0.45` `[-2.06,0.99]`,
  seed 20260902, per-dataset signs `-0.34` / `-0.56`, selection caveat added.
* Section 5.4 rebuilt on one reference arm, with the compounding decomposition
  (`-0.06` splitting + `-0.39` quantizing = `-0.45`) and the 0/60 catastrophic
  count under *both* arms.
* Contribution bullet 2 and the Conclusion now pair `14.10x` with `-0.45
  [-2.06,0.99]` against the same arm. A text search for `14.10` finds no
  sentence pairing it with a delta measured against another arm.
* Table 1: header `ΔF1 vs. split Q16` → `ΔF1 vs. full-prefix`; all six rows
  replaced with the A2 §3.1 pooled values; footnote states the seed, the arm,
  the 0/60 counts under both arms, and that deltas come from unrounded means
  (this is T-11's resolution — the printed deltas were always correct, only
  non-reproducible from the two-decimal columns).
* New **Appendix B, `app:all-intervals`**, with `tab:all-intervals`: all **42**
  intervals — the 18 recomputed against full-prefix Q16 and the 24 archival ones
  with the reference arm each actually used. So no interval is computed but
  unreported.
* A2 §7.3's dense-vs-prefix investigation is in that appendix, in the corrected
  form A2 §8 item 2 demanded: it reports the 6/60 divergent greedy sequences,
  the 14.51-vs-0.45 per-item SD, and 58/60 vs 54/60 token-sequence agreement,
  and explicitly **does not** assert the Table 18 footnote's unsupported
  "different document/query execution boundary" mechanism. The main text keeps
  only the one-clause version in Section 5.1.

### A3 — ForkAudit scope in the main text (closes `R44-REV-3-C1`, `T-03`, `Q3`)

The scope formula (*full-prefix BF16 KV, no split depth, no quantization,
audits the ownership discipline and not the quantized Read path*) now appears at
**eight** sites — seven in main text plus Table 5:

1. Abstract (A3-1)
2. Contribution bullet 3, retitled "Ownership-safe reuse, audited off the quantized path" (A3-2)
3. Figure 1 caption — "the packed Read path drawn here is unaudited" (A3-3)
4. Section 4.4 opening (A3-4), naming the three untested packed-entry obligations
5. Section 5.1 "Executed systems" (A3-8, prose form, compressed — see §4)
6. Section 5.5 opening paragraph (A3-5)
7. Section 5.6, as a named limitation opening its own paragraph (A3-6)
8. Table 5 `tab:cohorts`, primary-factorial row: "not the quantized Read path" (A3-7)

Per A3's flag 5 the wording says *no ForkAudit/ownership cell* runs split replay
or packed state — not "no experiment in this paper", which would be false
because of the Mac common-mode control.

### A6 — two narrowed claims (closes `Q4`, `T-04`, `R44-REV-3-M5`, `R44-4-05`, `R44-4-11`, `R44-5-11`)

* **Allocator.** Abstract, Section 5.5 and Section 5.6 now carry both endpoints
  (`54.5%` final / `42.2%` peak), name the full-copy arm a *controlled ownership
  factor rather than an optimized baseline*, attribute the saving to block
  sharing `\citep{kwon2023pagedattention}`, and state that the audited fork ties
  the paper's own paged-prefix baseline on both endpoints and is worse on
  generation increment (`1.950` vs `0.019` GiB).
* **Alias.** Abstract, Section 3, Section 5.5 and contribution bullet 3 now say
  the persistent-base content invariant also catches the historical alias, so
  the contract's increment is transition-time owner/layer/family *localization*,
  not exclusive detection; the same caveat is attached to the `0/5` and `1/5`
  figures. Conclusion changed to "localizing request-local ownership violations".
* **A6's flag 1 honoured.** The `42.2%` peak figure is reported, but the
  manuscript does **not** say peak determines OOM or admission — on a torch
  allocator delta above a frozen post-priming baseline that is not established,
  and Section 5.6 continues to disclaim it.
* **A6's flag 2 honoured.** The tie is stated factually rather than as "zero
  benefit", and the sentence "the ownership mechanism buys auditable
  request-local state, not fewer allocator bytes" makes the point without
  implying a failed optimization.
* **A6's flag 4 acknowledged, not acted on.** The 21/22-vs-22/22
  conventional-detector matrix is still excluded; it would need its own evidence
  path and claim-map entry. This remains the paper's weakest asymmetry and
  should be answered in rebuttal.

### A8 — Store estimand reconciliation (closes `R44-REV-3-M1`, `R44-5-05`, `R44-4-09`)

* Section 5.1 "Memory denominators" replaced: one estimand (the entry-owned
  physical byte-range union), of which the 60-item panel prints a component
  decomposition; the two implementations agree **to the byte** at all five
  retained configurations on the eight shared documents; the tables differ only
  in cohort and statistic, with `0.000` MiB left to the accounting.
  Section 5.6's "component accounting versus byte-range union" sentence replaced
  with the proven identity.
* Cohort/statistic annotations added at each ratio site (`60-item mean`,
  `8-item median`) in the abstract, Section 5.2 and both table footnotes.
* Appendix G.1: the cross-runtime `8.78x` / `20.39x` ratios are **removed** and
  replaced by the three levels side by side (15.89 / 139.53 / 324.09 MiB) plus
  the page-quantization caveat and the `0.58%` full-prefix calibration point,
  exactly as A8.6 recommends.

### A9 — KV/state-quantization literature (closes `Q2`, `R44-5-12`, `Q6`)

* New Related-Work paragraph **"KV and state quantization"** citing
  `jacob2018quantization`, `sheng2023flexgen`, `liu2024kivi`, `hooper2024kvquant`,
  `tomar2025xquant`, `li2025kvtuner`, `tao2025asymkv`, `yang2024mikv`,
  `chiang2025quamba`, `chiang2025quamba2`, `yue2025mambaquant`, alongside the
  existing `chang2025palu` / `wang2025squeezeattention` / `fu2025headkv`. It
  states plainly that Q-CoMem claims none of these mechanisms. FlexGen and
  KVTuner — the two A9 said must not be cut — are kept with their specific
  content (64-element groups; offline layer-wise bit-pair search).
* Contribution bullet 1 rewritten from a mechanism claim to an
  object-and-denominator claim, including the verified CoMem-is-BF16 fact.
* Dataset citations added at Section 5.1 (`bai2024longbench`, `dasigi2021qasper`,
  `ho2020multihopqa`) and release/code pins cited in Appendix A
  (`bai2024longbenchrelease`, `bai2024longbenchcode`), which is where A9 says the
  commit SHA belongs.
* New row for KVTuner in `tables/related_work_reported_context.tex`.
* **CoMem → `\qcomem{}` rename applied where the configuration was measured in
  this work**: manuscript lines for the Mac common-mode control, the H20
  deployment scope, the matched-serving-controls row, and the G.1 ratio sentence
  (the fourth site, rewritten wholesale by A8.6); table files
  `mac_m4_motivation_table_r39_revised.tex`, `mac_m4_motivation_table.tex`,
  `h20_deployment_table_r40_candidate.tex`, and both main tables (already
  `\qcomem{}`). Every remaining bare "CoMem" refers to the *prior work* and was
  deliberately left: abstract line 44, introduction line 100, Related Work
  line 181, bullet 1, the quantization paragraph, the `tab:closest` prior-work
  row, and the Hydragen/Palu-plus-CoMem hypothetical.

### A10 — Section 5.5 rebuild (closes `R44-REV-3-M2`, `T-12`, `R44-4-17`, `R44-5-10`, `R44-REV-3-m1` partly, `R44-REV-3-m2` not)

* Section 5.5 rebuilt into four paragraphs: *Scope, cohort and procedure* (also
  carrying `\label{sec:target-status}` and the A3-5 scope sentence), *What it
  found*, *Allocator endpoints under a controlled ownership factor* (A6a-2), and
  *Falsification beyond output equality* (A6b-2).
* Every undefined count cascade moved to a new **Appendix D paragraph
  "Enumerated evidence units and call counts"** with its execution named:
  209,920 / 635,520 receipts and their factorization; 144 / 12,960 / 3,840 / 24
  with the factor-3 phases disambiguated from the three resident counts;
  60 = 30x2, 480 = 8x60, 540 = (1+8)x60, six anchors; the 96 clean-memory calls;
  72 rank–query identities; the borrowed-GDN endpoints `4.890` / `2.229` GiB; the
  GDN oracle layers 0,10,20,38; 1,080 observations and 96,660 relations.
  **No number was dropped.**
* `tables/target_coverage_compact.tex` created as drafted (`tab:target-coverage`).
  See §4 for its placement.
* Verdict vocabulary (*scientifically valid*, *terminally complete*,
  *operationally invalid*) is now defined before first use, via Appendix A.
* "same-run hook" replaced by "live-binding rerun" throughout, resolving the
  contradiction with Appendix A.
* Previously uncited floats now cited from the new 5.5 / Appendix D:
  `fig:architecture`, `tab:protocol`, `tab:v29-evidence-units`,
  `tab:r33-fresh-heldout`, `tab:gdn-oracle`.

### A11 — provenance and the accounting decision (closes `R44-5-01`…`R44-5-04`, `R44-4-15`; and `T-02`, `Q7`, `R44-REV-3-M4`, `R44-5-06`, `R44-4-13` via A1)

The adjudicated decision is implemented exactly as instructed: **keep 14.1018x
as the physical ratio, relabel the reference, disclose the FP32→low-bit share,
and report 10.9965x alongside.**

* Abstract, Section 5.2, Table 1's footnote and the Conclusion all now say the
  reference is *what the unmodified stack retains, in its native dtypes*; that
  `30.000` MiB/document of the `136.235` MiB is FP32 GDN recurrent state in
  excess of a BF16 encoding; that part of the gain is therefore dtype narrowing
  rather than the Eq. 3 packing step; and that against an all-BF16 reference of
  `106.235` MiB/document the same policy is `10.9965x` smaller. Both figures are
  labelled exact re-analyses of the same archived per-item rows on 60/60 items.
* Table 1's compression column footnote lists the full BF16-normalised set
  `1.00 / 3.7038 / 10.9965 / 11.3073 / 14.0969x`; Section 5.2 gives `3.7038x`
  and `10.9965x` (`90.91%`).
* Table 2's footnote states `42.8%` of its `140.34` MiB reference is FP32 GDN
  recurrent state — the value A8.5 computed at that cohort's median document,
  which is the row it annotates.
* Row labels changed: Table 1 `Full-prefix Q16/BF16` → `Full-prefix, native
  dtype`; Table 2 likewise; `\qcomem{} split Q16/BF16` → `\qcomem{} split Q16`.
  No "Q16/BF16" label survives on a row whose leaves are not all BF16.

**Claim-narrowing items C1–C6:**

| | Where implemented |
|---|---|
| **C1** "Q16 is the BF16 unpacked reference" is false for cache leaves | Section 4.2 sentence rewritten; Figure 2(c) caption rewritten; `h20_deployment_table_r40_candidate.tex` footnote's "Q16 denotes BF16" rewritten |
| **C2** Table 2's Store is the same formula, not a different one | Section 5.1 denominators paragraph and Section 5.6; the "the accounting implementations differ" sentence is gone |
| **C3** "Q16 projections" has no referent in the code | Deleted from Section 4.2, from `tables/qcomem_tradeoff_r42.tex`, and from `tables/h20_deployment_table_r40_candidate.tex` |
| **C4** the measured Read path is **full private materialization**, not the borrow/COW discipline of Sec. 4.3, and ForkAudit has never run on the quantized Read path | Stated plainly **twice** in main text: Section 4.3 ("materializes a full private copy of the dequantized entry per query, so it shares nothing and exercises neither borrowing nor copy-on-write") and Section 5.6 ("ForkAudit has not been run on the quantized Read path… whose Read step materializes a full private copy of the entry rather than borrowing it, so ownership there transfers by design argument only"); reinforced at the other six A3 scope sites |
| **C5** "same-memory per-layer" is not equal-memory | Section 5.4: "named for its calibration budget, not measured parity" |
| **C6** three files that produced Table 1 are unarchived | Appendix A names them by role (run driver, aggregation/bootstrap, layer-policy search, deployment accountant, launch script) and says they must be released before Table 1 is independently re-executable; Section 5.6 and the Reproducibility Statement point there |

A1's superseded content is not carried anywhere: no "13.2–13.6x", no
"≤13.9662x", no metadata-omission mechanism.

### A12 — compliance and hygiene (closes `Q11`, `R44-REV-3-m6`, `R44-5-13`, `T-13` mechanical part, `R44-REV-3-m3`)

* **A12-1, blocking.** `\section*{Use of Large Language Models}` added to the
  statements block, beside Ethics and Reproducibility. **The wider variant was
  used**, honouring A12's flag 7: the repository shows LLM-driven reviewer
  panels, an issue ledger and revision planning, so "drafting and revision" alone
  would be under-scoped. The disclosure covers drafting, LaTeX/tables, related-
  work triage, and internal critiques/revision plans that the authors adjudicated,
  while attributing all experiments, measurements and evidence packages to the
  authors.
* **A12-2.** Ethics Statement moved from Appendix B to a `\section*` in the
  statements block; appendices relabel C→B…I→H automatically (all references are
  `\label`-based). **Response-letter note: former "Appendix H" is now
  "Appendix G", and the new interval appendix takes the letter B.**
* **A12-3(i).** QS Job/Trial identifier replaced by archive alias `QP-60` plus
  the real, reviewer-checkable manifest SHA-256
  `24ac6952…c9952`.
* **A12-3(ii).** `Apple Mac16,8` → `Apple-silicon (M4-class) laptop CPU host` at
  both sites; "Apple M4 Pro" kept, as A12 instructs.
* **A12-3(iii).** The undefined `V29` token removed from Table 4's caption.
* **A12-3(iv).** Artifact-alias paragraph added at the head of Appendix H, and
  the mapping applied mechanically: 13 round-numbered evidence paths →
  `package/fa-0N-…` / `package/qp-…` / `package/rw-…`, 8 registry IDs
  de-round-numbered (incl. `E-R40-INDEPENDENT-LIVE-BINDING-V29-A` →
  `E-LIVE-BINDING-A`), and `designer_input/round32_input.pdf` →
  `designer_input/prior-submission.pdf` (the path `R44-REV-3-m3` singles out).
  The now-redundant "the anonymous package assigns a neutral artifact alias"
  clause was deleted.

---

## 3. Flags carried forward from the drafts

1. **A12's LLM-disclosure flag honoured** — the wider variant is in the paper. If
   the authors disagree, the narrow variant is in `a3_a6_a12_drafts` §A12-1.
2. **A12-4's flag** — the response letter must **not** repeat T-08's
   "syntactically identical" characterization of Eq. 4; it is refutable. The
   substantive complaint (the old equation carried no ownership, mutability or
   phase semantics) stands and is unfixed this round (see §4).
3. **A6's flag 1** — `42.2%` peak is reported but never described as determining
   OOM or admission.
4. **A6's flag 4** — the 21/22 conventional-detector matrix is still excluded.
5. **A12-3's separate anonymity finding is not a manuscript item and was not
   actioned here**: `evidence/qcomem_mixed_validation_60item_20260812d/platform_receipt.json`
   contains a corporate registry hostname, cluster/queue IDs and a username in
   cleartext. Renaming manuscript strings does not anonymize the supplement.
   Scrub or exclude that receipt before any artifact ships.
6. **A9's open item** — the LongBench commit SHA and dataset revision hash are
   referenced as "pinned in the package receipt" but the authors must supply
   them.

---

## 4. Deferred, and why

R44's main text ended at margin line 474 (last line before References) with
page 9 running 432–485, i.e. **11 typeset lines of slack**. The
full drafted package as written measured **+113 typeset lines** — the A3/A6/A12
drafts' own flag 8 predicted this ("the package does not fit in 9 pages as
specified"). The cut menu recovers ~17 lines and the 5.5 rebuild ~6 net (its
promoted table costs back what the relocation frees). That is not close.

Every substantive item was therefore integrated and paid for by compression
elsewhere; three items were deferred, in the order the brief specifies.

**Deferred (all non-blocking):**

| Item | Cost | Status |
|---|---|---|
| **A12-4, the Eq. 4 rewrite** (`T-08` half) | 551 chars ≈ 5 lines | **Deferred.** Eq. 4 keeps R44's form. Consequently `\label{eq:store}` and `\label{eq:fork}` were not added and A3-4 does not reference `eq:fork`. The two real defects the draft found — the missing `ℓ<j` restriction and the missing residual/V/C tensors — are therefore still open, and A12's verification criterion for Eq. 4 still fails. |
| **A3-8's Section 5.1 executed-system map** | 945 chars (table) / ~545 (prose) | **Partially deferred.** The table form was dropped; the prose form was integrated and then compressed to a two-sentence `\paragraph{Executed systems.}` (~410 chars). A reader of Sections 5.1–5.5 can still answer "no" to *was Table 1's Q4/Q4/Q8 configuration ever passed through ForkAudit?* — so A3's verification criterion is met — but the per-table map is gone. |
| **A3-4's Eq. 5 half** (`T-08` other half) | ~300 chars | **Deferred.** Eq. 5 keeps `[Coverage_i(τ)=complete]`; the dangling `M_i(Σ)` is instead bound by a following clause ("where completeness quantifies over `M_i(Σ)` — every mandatory slot present, unique, unmodified — and `Bind_Σ` requires each slot to resolve to the live tensor its receipt names"). This closes the *dangling-symbol* complaint in prose at ~a third of the cost, but not by rewriting the formula. |

**Also deferred, beyond the three named:**

* **`tab:target-coverage` was not promoted to the main text.** The file
  `tables/target_coverage_compact.tex` was created exactly as A10 drafted it and
  is `\input` in the appendix beside `tab:witness`, cited from Section 5.5 as
  "Appendix Table 7". The main-text float cost ~16 lines against an 11-line
  budget. A10's freed space comes from relocating the count cascades to
  Appendix D, and that part is fully applied; only the promotion is deferred.
  This is not a regression against R44, which had no such table anywhere.
* **A9's optional concurrent-work sentence** (`zhang2026damp`, `yu2026dasc`) —
  omitted, with the BibTeX entries also left out so no uncited keys enter
  `references.bib`. The meta-review explicitly called it optional.
* **A12's `T-09`** (Eq. 3's degenerate `s=0` case and the dequantization map) —
  not drafted by A12 and not written here. Still open; the fix is ~1 line.
* **A10.5's remaining uncited floats** — Tables 10, 12, 13, 20, 22, 23, 24, 25,
  26 in R44's numbering are still uncited, and Tables 10 and 12 still carry no
  `\label`. Five of the thirteen were fixed by the new 5.5/Appendix D.
* **A10.7's residual items** — the blank TTFT/Recall column deletion
  (`R44-REV-3-m1`), the `tab:cohorts` `longtable` caption-margin fix
  (`R44-REV-3-m2`), and the two Store-vs-F1 / Store-vs-TTFT scatter figures
  (`R44-4-17`) were not done. The scatters would need the space the page budget
  does not have.

### Page-budget accounting

| Stage | Effect on main text |
|---|---|
| R44 baseline | Conclusion body ends 468; Reproducibility Statement ends 474; References start p.9; slack 11 lines |
| All A2/A3/A6/A8/A9/A10/A11/A12 additions as drafted | **+113 lines** (measured, first full build: Reproducibility ended at line 587) |
| A10 relocation of count cascades to Appendix D | −6 lines net |
| `tab:target-coverage` moved to the appendix | −16 lines |
| Cut menu CUT-A…CUT-H equivalents applied across §1, §3, §5.1–5.6 | −38 lines |
| Compression of the R45 additions themselves (Related Work, Sec. 5.1/5.2/5.4/5.5/5.6, abstract, both table footnotes and captions, Figures 1–2 captions) | −34 lines |
| Figures 1 and 2 set to `0.86\textwidth` / `0.9\textwidth` | −4 lines |
| The three sanctioned deferrals above | −11 lines |
| **Result** | **Conclusion ends line 481 (page 9); statements on page 10; References begin page 10, line 513** |

No number, table row, `\ref`, `\cite` or `\label` was lost to compression; the
cuts removed duplicated narration (software versions, count restatements already
in Appendix A, per-dataset intervals now in Appendix B) and tightened prose.
The one measurable content loss is the "at a −10-point threshold the frozen
policy regresses on 2 of 60 items" sentence, cut from Section 5.4; it survives
in `revision/a2_reference_arm_20260902.md` §6 and can be restored if the
statements turn out to be page-limit-exempt.

---

## 5. Verification against the standing rule and the brief

**Nothing from the not-admissible table entered the manuscript.** Grep-verified
absent: `3.1x` advantage over a quantized cache, `29.88` MiB, `91.8%` / `96.5%` /
`-1.74%` throughput figures, any residency claim, any depth-sweep result, any
statement that `j=7` is a good operating point, and any new latency claim. The
`j=7` framing is unchanged from R44. Trial 1943158 is not referenced.

**No claim was strengthened.** Every claim edit moves in the conservative
direction:

* `14.10x` kept but relabelled and disclosed; `10.9965x` added beside it.
* Headline F1 delta moves from `-0.39 [-2.04, 1.08]` (vs. split Q16) to
  `-0.45 [-2.06, 0.99]` (vs. full-prefix) — a *larger* reported loss.
* Allocator claim gains the peak figure, the "controlled ownership factor"
  label, and the concession that the audited fork ties paged-prefix sharing and
  is worse on generation increment.
* Alias claim narrowed from "invisible to output equality" to
  "transition-time localization, not exclusive detection".
* Contribution bullet 1 drops the mechanism claim entirely.
* `8.78x` / `20.39x` cross-runtime ratios removed rather than caveated.
* ForkAudit's coverage narrowed at eight sites.

Numeric diff of the main text against R44: the only numbers removed are
`1.08`, `2.04`, `54.62` (superseded by the reference-arm change), `0.5` (a
scale note in Table 1's footnote), and `4.890`, `648.75`, `8.41`, `93.06`
(relocated to Appendix D or already present in Table 2). Every number added is
listed in the admissible table of `derived_vs_measured_20260902.md` or already
in R44.

**Abstract non-claims list intact and stronger.** The original sentence survives
verbatim ("We do not claim lower total process memory, measured recall,
edge-device performance, or broad quality preservation"), and the abstract now
adds four further non-claims: the ForkAudit scope exclusion, the dtype-narrowing
share of the memory gain, the selection/no-multiplicity caveat, and the
localization-not-detection caveat.

**Three memory denominators stay separated.** Section 5.1 and Section 5.6 both
keep (i) Store — the entry-owned byte-range union, (ii) the post-priming torch
allocator delta, named "a third denominator", and (iii) process/NVML memory,
unmeasured. A fourth distinction is now explicit *inside* Store: the
native-dtype reference versus the all-BF16 reference.

**ForkAudit scope appears at every site where ownership safety is claimed** —
including the abstract, Figure 1's caption and contribution bullet 3, plus
Sections 4.4, 5.1, 5.5, 5.6 and Table 5.

**A11 finding C4 is stated plainly** in Section 4.3 and again in Section 5.6, in
both places naming full private materialization and the absence of borrowing and
copy-on-write on the measured path.
