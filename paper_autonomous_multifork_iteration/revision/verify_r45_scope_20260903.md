# R45 change verification — scope, claim narrowing, citations, compliance

Verifier role: change verifier (did not author the revision; not a paper review).
Date: 2026-09-03
OLD: `main_r44_structure.tex` (build `build/r44_structure_v1/`, 31 pp.)
NEW: `main_r45_evidence.tex` (build `build/r45_evidence_v1/`, 35 pp.)
Method: `grep`/`sed` diff of the two `.tex` sources, `pdftotext -layout` on both PDFs
for placement, `.blg`/`.bbl`/`.log` for citation resolution. Revision notes were read
for orientation only; every verdict below rests on the manuscript or the built PDF.

Build gates independently re-checked on `build/r45_evidence_v1/`:
undefined references 0, undefined citations 0, BibTeX `warning$` calls 0,
40 cited / 40 resolved. Main text still ends on p.9 (`6 CONCLUSION` p.9);
statements and references begin p.10. No page-limit regression.

---

## Issue 1 — R44-REV-3-C1 / T-03 / Q3: ForkAudit scope not stated where ownership safety is claimed

**Verdict: `resolved`**

### OLD (all sites, `main_r44_structure.tex`)

Abstract (L64-66):
> A fail-closed \method{} trace validates immutability, copy-on-write, recurrent
> rebinding, and call provenance because output equality alone can miss state aliasing.

Contribution bullet 3 (L124-130, quoted in full):
> \textbf{Ownership-safe reuse.} The seven-target \method{} contract couples semantic
> relations to phase-indexed storage evidence across 96 ownership configurations.

Figure 1 caption (L149-150):
> The narrow \method{} rail validates the ownership boundary; it is not an additional
> execution stage.

Section 4.4 first sentence (L331):
> \method{} validates the Section~\ref{sec:read} lifecycle with seven targets: ...

Section 5.5 first paragraph (L483-486):
> Primary feasibility uses one fixed Qwen3.5-35B-A3B configuration ...: ten vLLM-0.26
> Triton full-attention layers with BF16 KV (the protocol's ``Q16'' storage label, not
> quantization) and 128-token pages, plus thirty Transformers-torch GDN layers.

Section 5.6, Conclusion, Appendix Table 5 primary row (L839):
> Primary ownership factorial & 8 books; $N=1,8,32$; four KV$\times$GDN cells; 8 decode
> steps & ... not transfer or scheduler evidence

Grep of the OLD file for `no split depth`, `no quantization`, `quantized Read path`,
`unaudited`: **0 hits.** Confirmed against the OLD PDF as well.

### NEW (`main_r45_evidence.tex`), nine sites, eight of them main text

1. **Abstract, L49-53** (PDF p.1, margin 026-028) — "...; that factorial runs full-prefix
   BF16 KV with no split depth and no quantization, so it audits the ownership discipline
   and not the quantized Read path."
2. **Contribution bullet 3, L138-142** (PDF p.2, 083-086) — retitled "Ownership-safe reuse,
   **audited off the quantized path**": "...96 ownership configurations, all full-prefix
   BF16 KV with no split depth and no quantization; the quantized Read path is not audited."
3. **Figure 1 caption, L165-168** (PDF p.3) — "...it ran only on a full-prefix BF16
   configuration with no split depth and no quantization, so the packed Read path drawn
   here is unaudited."
4. **Section 4.3 (Read), L359-362** — "This discipline is what \method{} audits; the
   Transformers implementation behind Tables 1 and 2 instead materializes a full private
   copy of the dequantized entry per query, so it shares nothing and exercises neither
   borrowing nor copy-on-write."
5. **Section 4.4 opening, L372-380** (PDF p.5) — "It runs on a full-prefix BF16
   configuration with no split depth and no quantization ..., so it validates that
   discipline and not the dequantize-then-fork Read path; the obligations specific to a
   packed entry --- dequantized-view immutability, residual-chunk binding, packed-entry
   lifetime --- are untested."
6. **Section 5.1 `\paragraph{Executed systems.}`, L460-464** (PDF p.6) — "Tables 1 and 2
   are Transformers-only executions of packed split-replay entries at $j=7$, whereas
   Table 17 and every \method{} verdict come from a different system --- vLLM 0.26 plus
   Transformers, full-prefix BF16 KV in 128-token pages, no split depth, no quantization,
   PG-19 --- so no \method{} verdict here covers those packed entries."
7. **Section 5.5 first sentence, L543-547** (PDF p.8) — "The primary factorial runs
   full-prefix BF16 KV on a vLLM-plus-Transformers stack with no split depth and no
   dequantization step, and therefore audits the ownership discipline of Section 4.3
   rather than the quantized Read path measured in Tables 1 and 2."
8. **Section 5.6 limitations, own paragraph, L638-643** (PDF p.9) — "\method{} has not
   been run on the quantized Read path: every ownership cell uses full-prefix BF16 KV with
   no split depth and no dequantization, whereas Tables 1 and 2 measure a $j=7$ packed
   Q4/Q4/Q8 path ... so ownership there transfers by design argument only."
9. **Conclusion, L666-667** — "\method{} complements the capacity result by localizing
   request-local ownership violations **on a full-prefix BF16 configuration**."

Appendix Table 5 (`tab:cohorts`) primary row, L1023 (PDF p.18):
> Primary ownership factorial & 8 books; PG-19; **full-prefix BF16 KV; no split depth; no
> quantization**; ... & ...; **not the quantized Read path**, and not transfer or scheduler evidence

### Reasoning

The meta-review's specific ruling was that appendix disclosure does not make the abstract,
Figure 1 and contribution bullet 3 accurate. All three now carry the scope statement in the
compiled PDF (verified by page: p.1, p.2, p.3), not only in the source.

R44-REV-3-C1's verification test — a reader with no appendix access must be able to answer
"was Table 1's Q4/Q4/Q8 configuration ever passed through ForkAudit?" from Sections 5.1-5.5
alone — is met explicitly by site 6: "no \method{} verdict here covers those packed entries."
T-03's extra requirement (a Table 5 row making the exclusion explicit in the
"Authorized claim / explicit exclusion" column) is met. Q3's verification test (either a
quantized ForkAudit cell with its Table 6 verdict, or "quantized Read path" listed among
Table 5's primary-factorial exclusions) is met by the second branch.

No new ForkAudit experiment was run on the quantized path — the revision took the
claim-narrowing branch that all three reviewer records explicitly authorize as the
alternative. That is a legitimate closure of these issues, not a partial one.

### Residual (non-blocking, does not change the verdict)

One main-text sentence mentions the pairing without the qualifier: **Introduction L113-116**,
"Because equal outputs can still hide a mutable-state alias, we pair the design with
\method{}, a phase-aware ownership trace over immutability, copy-on-write (COW), recurrent
rebinding, and selected call provenance." This is a definition of the tool rather than a
claim of ownership safety for the quantized path, and the correction lands 25 lines later in
bullet 3 on the same PDF page-2 spread. Adding "(evaluated on a full-prefix BF16
configuration)" would close it. Separately, R44-REV-3-C1's `required_action` item (1) asked
for a table or paragraph mapping Tables 1, 2 and 15 to model / runtime / KV backend / split
depth / quantization policy / dataset; the delivered `Executed systems` paragraph covers
runtime, KV backend, split depth, quantization and dataset but not the model, and the
per-table tabular form was dropped. The verification test still passes without it.

---

## Issue 2 — Q4 / T-04: 54.5% allocator result measured against a non-baseline

**Verdict: `resolved`**

### OLD

Abstract (L65-66):
> Separately, shared-document KV lowers the post-priming allocated delta from $4.901$ to
> $2.229\gib$ at $N=32$, and a fail-closed ownership audit exposes a historical alias
> invisible to output equality.

Contribution bullet 3 (L128-130):
> At $N=32$, shared-document KV separately reduces the post-priming allocator delta by
> $54.5\%$ relative to full-copy KV (Section~\ref{sec:correctness}).

Section 5.5 `\paragraph{Allocator reduction with canonical equality.}` (L543-552):
> ... reduces final allocated delta from $4.901$ to $2.229\gib$, a $54.5\%$ reduction above
> the frozen post-priming baseline ... Borrowed GDN changes allocation timing rather than
> the shared-KV end state ... Appendix Table~\ref{tab:rr2-memory} reports all four endpoints ...

No peak figure, no `42.2`, no "controlled ownership factor", no tie concession anywhere in
the OLD main text (grep for `42.2`, `2.843`, `4.920`: 0 hits in `main_r44_structure.tex`).

### NEW

Abstract (L74-79, PDF p.1 margin 040-042):
> Separately, sharing one document's KV rather than copying it per request lowers the
> post-priming allocator delta from $4.901$ to $2.229\gib$ final ($54.5\%$) and $4.920$ to
> $2.843\gib$ peak ($42.2\%$) at $N=32$, **against a full-copy arm that is a controlled
> ownership factor rather than an optimized baseline**; this **reproduces paged-prefix
> block sharing, which our audited fork ties on both endpoints**.

Contribution bullet 3 (L145-148, PDF p.2 margin 088-090):
> At $N=32$ shared-document KV cuts the post-priming allocator delta by $54.5\%$ final and
> $42.2\%$ peak against a **full-copy ownership control, matching this paper's own
> paged-prefix baseline**.

Section 5.5, retitled `\paragraph{Allocator endpoints under a controlled ownership factor.}`
(L578-590, PDF p.8 margin 421-429):
> ... reduces the final allocated delta from $4.901$ to $2.229\gib$ ($54.5\%$) and the peak
> from $4.920$ to $2.843\gib$ ($42.2\%$) ... **The full-copy arm is a controlled ownership
> factor, not an optimized baseline: no serving engine copies a shared prefix per request,
> and the saving is block sharing~\citep{kwon2023pagedattention}. Against the closest
> same-stack reference, the paged-prefix baseline of Table 17, our audited hybrid fork is
> even on both endpoints and worse on generation increment ($1.950$ versus $0.019\gib$).
> The ownership mechanism buys auditable request-local state, not fewer allocator bytes.**

Section 5.6 (L633-636):
> The separate $54.5\%$ final / $42.2\%$ peak result is a post-priming allocator delta
> against a full-copy control, neither additive with Store nor evidence of higher admitted capacity.

### Reasoning

Every one of the four occurrences of `54.5` in the NEW file (L75, L146, L580, L634) carries a
qualifier in the same sentence; the claim is nowhere stated at its original strength.
Q4's literal verification test — "the abstract and Sec 5.5 must both carry the phrase
identifying full-copy KV as a controlled ownership factor rather than an optimized baseline,
at the sentence where 4.901 -> 2.229 GiB appears" — is satisfied verbatim in both places.
The 42.2% peak is now reported at all four sites. The tie against the paper's own
paged-prefix baseline is conceded in the abstract, in bullet 3, and stated numerically in
Sec 5.5, which additionally discloses the *regression* on generation increment
(1.950 vs 0.019 GiB) that the reviewer flagged. `tables/rr2_memory_table.tex` retains the
`Paged-prefix baseline` / `Audited hybrid fork` arm labels, and Sec 5.5 now uses them in body text.

T-04's `verification_test` asked for a fourth allocator arm (N requests sharing one packed
Q-CoMem entry at j=7). That arm was not run. It is not required here: T-04's own
`required_action` offers restatement as the alternative, and the adjudicated must-resolve
text for A6 asks exactly for what was delivered — "the abstract's allocator result must be
labelled as reproducing paged-prefix sharing against a controlled full-copy factor, with the
peak figure and the audited fork's zero delta against the paper's own paged-prefix baseline
stated where the claim is made."

Minor note, not a downgrade: bullet 3 concedes the tie ("matching this paper's own
paged-prefix baseline") but is the one of the three sites that does not also say
"not an optimized baseline"; it says "full-copy ownership control".

---

## Issue 3 — Alias claim: localization, not unique detection

**Verdict: `resolved`**

### OLD

Abstract (L66-67):
> ... and a fail-closed ownership audit exposes a historical alias **invisible to output equality**.

Section 5.5 (L565-568):
> In the historical no-injection case, output and terminal request state remain exact in 8/8
> cells despite persistent-base corruption, whereas the repaired path is storage-clean in 8/8.

Conclusion (L626-628):
> ... \method{} complements the capacity result by validating request-local ownership and
> **exposing a historical silent alias**.

The qualification existed only in Appendix H and in the Table 14 caption, exactly as
R44-5-11 described.

### NEW — four main-text sites, all logically identical to Appendix H

Abstract (L79-81, PDF p.1 margin 042-045):
> The audit exposes a historical alias that output and terminal-state equality miss **but a
> persistent-base content invariant also catches, so its increment is transition-time
> localization, not exclusive detection.**

Contribution bullet 3 (L142-145, PDF p.2):
> A historical alias corrupts the persistent base while output and terminal request state
> stay exact, **but a persistent-base content invariant also catches it, so the contract's
> increment is transition-time owner/layer/family localization, not exclusive detection.**

Section 3 Motivation (L272-278):
> ... we observe a historical configuration whose tokens and terminal request state stay
> exact while a persistent base is corrupted, **which a persistent-base content invariant
> also catches, but only at final capture and without naming the owner, layer, or state
> family responsible**.

Section 5.5 `\paragraph{Falsification beyond output equality.}` (L596-604):
> ... **but the persistent-base content invariant of Table 14 also fails on all eight pre-fix
> cells, so \method{}'s increment here is authenticated transition-time owner/layer/family
> localization, not exclusive detection, and the $0/5$ and $1/5$ figures are likewise
> fixed-case comparisons against two registered observables rather than a head-to-head
> detection study.**

Conclusion (L665-667): the "exposing a historical silent alias" clause is gone, replaced by
"localizing request-local ownership violations on a full-prefix BF16 configuration".

Appendix H (L1663-1666) is unchanged and now says the same thing as the main text.

### Reasoning

R44-5-11's verification test is that the main text's statement be logically identical to
Appendix H's with no qualification present only in the appendix. Verified: the appendix
sentence ("Output and terminal-state differentials miss the alias, while a persistent-base
content invariant catches it; ForkAudit contributes authenticated transition-time
owner/layer/family localization rather than exclusive detection") and the four main-text
sites now assert the same proposition. I found no remaining site implying unique detection —
notably the 0/5 and 1/5 mutant figures, the other place the paper could imply it, are
explicitly hedged as fixed-case rather than head-to-head. Appendix Table 14's own "Base inv."
column (fail 3/3, fail 5/5) is now consistent with what the main text says about it.

---

## Issue 4 — Q2 / contribution bullet 1: quantizer and per-layer bit vector are prior art

**Verdict: `resolved`**

### OLD

Contribution bullet 1 (L110-114):
> \textbf{Complete quantized hybrid split state.} \qcomem{} stores the depth-$j$ residual
> together with the necessary lower full-attention KV and lower convolution/recurrent state.
> **Real groupwise Q4/Q8 packing and a state-type/per-layer bit assignment** reduce the whole
> retained entry rather than counting the residual alone.

Related Work, "Hybrid state and cache management" (L179-187): cites only Palu,
SqueezeAttention, HeadKV for compression. The OLD reference list contains **no** KV-cache
quantization work of any kind (verified against `build/r44_structure_v1/`).

### NEW

Contribution bullet 1 (L129-137, PDF p.2):
> \textbf{Whole-entry accounting for a hybrid split-replay entry.} \qcomem{} **applies
> group-wise asymmetric quantization and per-layer, per-state-type width assignment ---
> neither claimed as new (Section~\ref{sec:related})** --- to an entry those methods do not
> address ... **CoMem retains only the residual, unquantized in BF16**, so counting it alone
> understates a hybrid deployment ...

New Related Work paragraph `\paragraph{KV and state quantization.}` (L206-229):
> **Low-bit storage of reusable state is established prior art and \qcomem{} claims none of
> its mechanisms.** The packer of Section 4.2 is the standard asymmetric affine
> quantizer~\citep{jacob2018quantization} applied group-wise --- **the form FlexGen already
> used for weights and the KV cache as 4-bit codes over 64-element groups** with per-group
> min/max metadata ...~\citep{sheng2023flexgen} --- and KIVI, KVQuant, and XQuant fix the
> grouping axis or the tensor quantized~\citep{liu2024kivi,hooper2024kvquant,tomar2025xquant}.
> **Per-layer width assignment is likewise prior art**~\citep{li2025kvtuner,tao2025asymkv,yang2024mikv}:
> KVTuner searches layer-wise mixed-precision KV bit pairs offline ... and Quamba, Quamba2,
> and MambaQuant quantize selective-SSM activations and per-state-group recurrent inputs,
> including for a hybrid backbone~\citep{chiang2025quamba,chiang2025quamba2,yue2025mambaquant}.
> **Our bit vector $\mathbf b$ is an application of an established design surface, not a new
> selection method.** What this literature does not cover is the object quantized ...

Also new: a KVTuner row in Table 11 (`tables/related_work_reported_context.tex`) stating the
relation — "Layer-wise bit search for a dense KV cache; it does not quantize a split residual
or recurrent state" — absent from the r44 PDF (grep KVTuner in `r44` PDF: 0 hits).

### Citation resolution

11 new quantization keys added to `references.bib` and all resolve in the build:
`jacob2018quantization`, `sheng2023flexgen` (ICML 2023, PMLR v202 pp. 31094-31116),
`liu2024kivi`, `hooper2024kvquant`, `tomar2025xquant`, `li2025kvtuner` (ICML 2025, PMLR v267),
`tao2025asymkv`, `yang2024mikv`, `chiang2025quamba`, `chiang2025quamba2`, `yue2025mambaquant`.
`main_r45_evidence.blg`: 40 entries used, `warning$ -- 0`; `.log`: zero undefined citations.
Spot-checked venues and page ranges against the entries: FlexGen ICML 2023 and KVTuner
ICML 2025 are correctly attributed to the venues Q2 named.

### Reasoning

The three things this assignment asked me to confirm are all present: bullet 1 no longer
claims the packing scheme or the per-layer bit assignment as novel (it explicitly disclaims
both), the KV/state-quantization literature is cited, and the citations resolve. Q2's own
verification test additionally asks that Section 2 name the group-wise, per-channel/per-token,
and layer-wise mixed-precision families. Group-wise and layer-wise are named explicitly; the
per-channel/per-token family is present by work (KIVI, KVQuant, XQuant) and by paraphrase
("fix the grouping axis or the tensor quantized") rather than by the family name, and the
"one sentence each" of differentiation is delivered as one collective sentence covering all
of them. These are wording gaps against the letter of the test, not a residue of the
overclaim the issue was raised about, so they do not hold the verdict below `resolved`.

Worth noting for a copy pass: Section 4.2 still presents the quantizer equation with no
citation at the equation itself; the attribution lives in Section 2, reached from bullet 1's
cross-reference.

---

## Issue 5 — A12 LLM-use disclosure (blocking venue-compliance item)

**Verdict: `partially_resolved`**

### OLD

Absent. Grep of `build/r44_structure_v1/main_r44_structure.pdf` for
"Large Language Models" / "LLM-based assistance" / "use of large language": **0 hits across
all 31 pages.** Confirms R44-REV-3-m6.

### NEW — `\section*{Use of Large Language Models}` (L679-687), PDF p.10

> The authors used LLM-based assistance throughout the preparation of this manuscript:
> drafting and revising prose, LaTeX, and tables; triaging related work; and **generating
> internal critiques and revision plans that the authors adjudicated**. All experiments,
> measurements, and evidence packages were designed and executed by the authors, who
> verified every numerical claim against the archived artifacts and take full responsibility
> for the content, including any remaining errors.

Placement: after `\section*{Ethics Statement}` and before `\section*{Reproducibility
Statement}`, all three on p.10, i.e. after the main text ends on p.9. The Ethics Statement
also moved out of Appendix B (it was on p.13 of the r44 PDF), which was the second half of
R44-REV-3-m6's placement complaint.

### Reasoning — presence: closed; scope: one sentence overreaches

The blocking compliance gap is closed: the disclosure exists, is correctly placed, and is
materially broader than "drafting assistance". The clause "generating internal critiques and
revision plans that the authors adjudicated" is a fair functional description of the
LLM-driven reviewer panels and the revision planning this project ran, even though it does
not name them as such.

What does not hold up is the next sentence. `review/experiment_response_plan.json` is an
LLM-produced artifact under a stated "Phase 10.5 of the autonomous-paper-agent contract"
policy; it classifies reviewer findings into `analysis_required` / `experiment_required` /
`claim_narrowing_required`, enumerates actions A1-A16 with per-action experiment
specifications, and sorts them into `runnable_now` and `needs_new_execution`. That is
experiment *specification*, and the revision followed it. The categorical claim that
"All experiments, measurements, and evidence packages were **designed** and executed by the
authors" is therefore stronger than the record supports. `review/issue_ledger.json` likewise
carries a five-reviewer panel with numeric scores and a meta-recommendation — an
LLM-simulated peer review, which a reader of the disclosure would not infer.

### What remains

One sentence edit. Either soften "designed and executed by the authors" to acknowledge that
LLM-generated plans proposed and prioritized the analyses and runs the authors then
commissioned and adjudicated, or name the mechanism directly ("LLM-simulated reviewer panels,
an issue ledger, and revision/experiment plans"). No new evidence is needed.

Also still open in the same ledger record, though explicitly outside the "A12 (LLM-use
disclosure only)" must-resolve scope: `references.bib` `liu2026comem` still lists `{Key}` and
`{Rayying}` as authors.

---

## Issue 6 — LongBench / Qasper / 2WikiMultihopQA uncited (R44-5-12)

**Verdict: `partially_resolved`**

### OLD

Section 5.1 (L376-379): "or 30 examples per dataset, under official LongBench-v1 prompts and a
4,096-token input cap" — no citation. Grep of the r44 PDF reference list for
`Bai` / `Dasigi` / `Xanh Ho`: **0 hits.** Confirms R44-5-12.

### NEW

Section 5.1 `\paragraph{Quality cohort.}` (L422-427), PDF p.6 margin 290-291:
> ... under official LongBench-v1 prompts, a 4,096-token input cap, and generation caps of
> 128 and 32 tokens~\citep{bai2024longbench,dasigi2021qasper,ho2020multihopqa} ...

Rendered as "(Bai et al., 2024a; Dasigi et al., 2021; Ho et al., 2020)". All three entries
are in `references.bib` with correct venues (LongBench = ACL 2024 long papers pp. 3119-3137;
Qasper = NAACL 2021 pp. 4599-4610; 2WikiMultihopQA = COLING 2020 pp. 6609-6625) and all
resolve — zero undefined citations, zero BibTeX warnings.

Two further pinning entries were added and are cited from Appendix A (L723-726, PDF p.14):
> The cohort is indexed against the LongBench-v1 release and its official F1
> implementation~\citep{bai2024longbenchrelease,bai2024longbenchcode}; the dataset revision
> and code commit hashes are pinned in the package receipt.

### What remains

R44-5-12's `required_action` is "Cite LongBench, Qasper, and 2WikiMultihopQA, **and pin the
dataset release/revision plus the F1 implementation (repository and commit)** used for the
reported scores." The citation half is done; the pinning half is not, and the shortfall is
printed in the submission:

`references.bib` L423-441 carry literal placeholder notes, and both are typeset into the
reference list on **PDF p.10** (the first page of references, margin lines 524-525 and 530-531):

> ... https://github.com/THUDM/LongBench. Accessed 2026-09-02; **commit SHA to be pinned by
> the authors.**
> ... https://huggingface.co/datasets/THUDM/LongBench. Accessed 2026-09-02; **dataset
> revision hash to be pinned by the authors.**

This also contradicts the Appendix A sentence that cites them, which asserts the hashes "are
pinned in the package receipt". Two fixes: supply the dataset revision hash and the LongBench
commit SHA in the two `note` fields (or drop the note and point at the receipt), and
reconcile the appendix sentence with whatever the bib then says.

---

## Summary

| # | Issue | Verdict |
|---|---|---|
| 1 | R44-REV-3-C1 / T-03 / Q3 — ForkAudit scope at every ownership-safety site | `resolved` |
| 2 | Q4 / T-04 — 54.5% allocator claim narrowing | `resolved` |
| 3 | R44-5-11 — alias reframed to localization | `resolved` |
| 4 | Q2 — bullet 1 novelty and KV-quantization citations | `resolved` |
| 5 | R44-REV-3-m6 / A12 — LLM-use disclosure | `partially_resolved` |
| 6 | R44-5-12 — LongBench / Qasper / 2WikiMQA citations | `partially_resolved` |

No regressions found in the assigned scope. Both partial verdicts turn on small,
non-experimental edits: one sentence in the LLM-use statement, and two hash values plus one
reconciling clause for the dataset pinning.
