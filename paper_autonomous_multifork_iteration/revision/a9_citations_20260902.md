# A9 — Citation repair for the KV/state-quantization gap

**Action:** `review/round_44/meta_review.json` -> `prioritized_actions` -> `A9` (issues Q2, R44-5-12, Q6).
**Date:** 2026-09-02. **Author of this note:** revision agent.
**Scope:** research + drafts only. `main_r44_structure.tex` and `references.bib` were **not** edited.

---

## 0. How each citation was verified

Two independent checks were required before a work appears below.

* **Metadata verification (M).** The title, author list, venue, year and pages were read off the
  *publisher's own* record — PMLR paper page, NeurIPS/ICLR proceedings BibTeX file, ACL Anthology
  entry, dblp record, or the arXiv abstract page. Nothing below is from memory.
* **Support verification (S).** The claim I attach the citation to was checked against the paper's
  own text (PDF/HTML pulled and read) or against the publisher-hosted abstract. Where I could only
  confirm the abstract, the table says so.

Nothing in this note relies on the three arXiv identifiers the meta-review rejected as
unverifiable (`2608.11231`, `2608.30310`, `2608.30386`). Two preprints I *did* independently
resolve are quarantined in §5 as **optional** concurrent work, consistent with the meta-review's
ruling that "a concurrent-work sentence is optional, the quantization-literature paragraph is not."

---

## 1. Verified literature and novelty-boundary consequence

### 1a. Group-wise / per-channel / per-token KV quantization

| Key | Work, venue (verified) | Verification | What it already does | Consequence for the Q-CoMem novelty boundary |
|---|---|---|---|---|
| `jacob2018quantization` | Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, CVPR 2018, pp. 2704–2713 | M: dblp record `conf/cvpr/JacobKCZTHAK18` (DOI 10.1109/CVPR.2018.00286). S: abstract + canonical status | Defines the standard asymmetric affine quantizer (scale + zero-point, integer codes, dequantize before compute). | Section 4.2's formula is a textbook instance of this. The paper must not present the quantizer form as its own. |
| `sheng2023flexgen` | Sheng et al., *FlexGen: High-Throughput Generative Inference of LLMs with a Single GPU*, ICML 2023, PMLR 202:31094–31116 | M: PMLR v202/sheng23a page + BibTeX. **S: full PDF pulled and read, §"Group-wise Quantization"** | Verbatim: "fine-grained group-wise asymmetric quantization method… we choose g contiguous elements along a certain dimension as a group. For each group, we compute the min and max of the group elements and quantize each element x into b-bit integers by `x_quant = round((x-min)/(max-min) × (2^b − 1))`… The tensors are stored in the quantized format and converted back to FP16 before computation… we compress both to 4 bits **with a group size of 64**." | **This is the single decisive citation.** It is Section 4.2's Eq. (3) — same asymmetric min–max form, *same group size of 64*, same store-packed/dequantize-before-compute design — published in 2023 for weights and KV. Contribution bullet 1 cannot claim the packing scheme in any form. |
| `liu2024kivi` | Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*, ICML 2024, PMLR 235:32332–32344 | M: PMLR v235/liu24bz page. S: publisher abstract + GitHub README | Group-based asymmetric 2-bit KV quantization with the axis chosen per tensor: keys per channel, values per token; tuning-free. | Establishes that *asymmetric group quantization of the KV cache*, including the choice of grouping axis, is settled prior art. Q-CoMem's Q4/Q8 KV packing is an application. |
| `hooper2024kvquant` | Hooper et al., *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization*, NeurIPS 2024, vol. 37, pp. 1270–1303 | M: NeurIPS proceedings BibTeX file (DOI 10.52202/079017-0040). S: publisher abstract | Per-channel and pre-RoPE key quantization, non-uniform codebooks, per-vector dense-and-sparse handling, sub-4-bit KV. | Second independent anchor that low-bit KV storage with per-channel grouping is prior art; also shows the field's accuracy bar (<0.1 ppl at 3-bit) that Q-CoMem does not target. |
| `tomar2025xquant` | Tomar et al., *XQuant: Breaking the Memory Wall for LLM Inference with KV Cache Rematerialization*, arXiv:2508.10395 (14 Aug 2025) | M: arXiv abs page (title, 10 authors, date, cs.LG, "24 pages"). S: abstract. **Preprint — cite as such** | Quantizes and caches **layer input activations** rather than K/V, and regenerates K and V on read; XQuant-CL exploits cross-layer activation similarity. | The closest published precedent for *quantizing a cached hidden state instead of a KV cache*. It weakens any claim that quantizing a residual (rather than KV) is itself novel — but it quantizes per-layer activations inside one dense path, not a depth-split entry, and does not touch recurrent state. |

### 1b. Layer-wise / head-wise mixed-precision bit allocation

| Key | Work, venue (verified) | Verification | What it already does | Consequence for the Q-CoMem novelty boundary |
|---|---|---|---|---|
| `li2025kvtuner` | Li et al., *KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference*, ICML 2025, PMLR 267:36451–36485 | M: PMLR v267/li25dd page + BibTeX (9 authors, pages). S: publisher abstract quoted verbatim ("nearly lossless 3.25-bit mixed precision KV cache quantization for LLMs like Llama-3.1-8B-Instruct and 4.0-bit for sensitive models like Qwen2.5-7B-Instruct"; "throughput can be improved by 21.25% compared with KIVI-KV8") | Offline, sensitivity-driven **search over per-layer KV bit-width pairs**, calibrated once and then frozen for inference. | **This is the decisive citation for the `[8,8,4,4,8,8,8]` vector.** Per-layer bit assignment selected on a calibration set and frozen is exactly KVTuner's contribution, published at ICML 2025. Q-CoMem may claim the *instance* (which layers, which widths, for this hybrid entry) but not the idea or the method of choosing it. |
| `tao2025asymkv` | Tao, Yu & Zhou, *AsymKV: Enabling 1-Bit Quantization of KV Cache with Layer-Wise Asymmetric Quantization Configurations*, COLING 2025, pp. 2316–2328, ACL, Abu Dhabi | M: ACL Anthology 2025.coling-main.158 entry + its BibTeX. S: Anthology abstract | Assigns **different bit-widths per layer and separately to keys vs. values**, pushing 75% of decoder layers to 1 bit. | Directly anticipates "assign the bit width per state type *and* per layer". Q-CoMem's `b` is indexed by (state type, layer); AsymKV's is indexed by (K/V, layer). The design surface is not new. |
| `yang2024mikv` | Yang et al., *No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization* (MiKV), arXiv:2402.18096 | M: arXiv abs page + dblp (**dblp lists CoRR only — no conference venue**; cite as preprint). S: abstract | Mixed precision *within* the KV cache by token importance: important pairs FP16, unimportant pairs INT2/INT4 instead of evicted. | Third granularity of mixed-precision KV (token). Included for completeness of the thread; optional if the paragraph must be shortened. Must be cited as an arXiv preprint, not as a conference paper. |
| `chang2025palu`, `wang2025squeezeattention`, `fu2025headkv` | already in `references.bib` | already locked in `literature/citation_lock.json` | Low-rank latent KV; per-layer KV *budget*; per-head KV *budget*. | These three are budget/rank allocation, **not** bit allocation. They are the paper's current sole compression citations and they do not cover the quantization thread — which is precisely the gap Q2 identified. They stay, but they must no longer stand in for the quantization literature. |

### 1c. Recurrent / SSM / linear-attention state quantization

| Key | Work, venue (verified) | Verification | What it already does | Consequence for the Q-CoMem novelty boundary |
|---|---|---|---|---|
| `chiang2025quamba` | Chiang et al., *Quamba: A Post-Training Quantization Recipe for Selective State Space Models*, ICLR 2025, pp. 101328–101354 | M: proceedings.iclr.cc BibTeX file (5 authors, pages, volume 2025). S: OpenReview/publisher abstract | 8-bit static per-tensor PTQ for selective SSMs, incl. outlier handling in the selective-scan path; also quantizes **Jamba**, a hybrid attention+SSM model. | Establishes that quantizing recurrent/SSM state paths — including in a *hybrid* model — is prior art. Q-CoMem cannot claim "quantizing GDN state" as a new idea. |
| `chiang2025quamba2` | Chiang et al., *Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models*, ICML 2025, PMLR 267:10411–10427 | M: PMLR v267/chiang25a page + BibTeX (6 authors, pages). S: publisher abstract | W8A8/W4A8/W4A16 for Mamba and Mamba2, with **per-state-group quantization** of the recurrence inputs B and C and sorting/clustering for x; 4× memory reduction. | The closest published work on *low-bit recurrent state*. Q-CoMem's Q8 on lower GDN convolution/recurrent state is an application of an established capability, not a new one. |
| `yue2025mambaquant` | Yue et al., *MambaQuant: Quantizing the Mamba Family with Variance Aligned Rotation Methods*, ICLR 2025, pp. 33231–33250 | M: proceedings.iclr.cc BibTeX file (9 authors, pages, volume 2025). S: publisher abstract | Rotation-based W8A8 PTQ framework for the Mamba family. | Weights/activations rather than persisted state; include only to show the SSM-quantization thread is mature. **Droppable** if the paragraph must be trimmed. |

### 1d. Datasets used throughout but never cited (Q6)

| Key | Work, venue (verified) | Verification | Use in the manuscript |
|---|---|---|---|
| `bai2024longbench` | Bai et al., *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding*, ACL 2024 (Vol. 1: Long Papers), pp. 3119–3137, Bangkok, DOI 10.18653/v1/2024.acl-long.172 | M+S: ACL Anthology entry and its official BibTeX | Section 5.1 ("official LongBench-v1 prompts", the 4,096-token cap, the F1 metric) and the appendix related-work scope caveats. |
| `dasigi2021qasper` | Dasigi et al., *A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers* (Qasper), NAACL-HLT 2021, pp. 4599–4610, DOI 10.18653/v1/2021.naacl-main.365 | M+S: ACL Anthology entry and its official BibTeX | The 30 Qasper items of the 60-item quality cohort and the 4 Qasper timing items. |
| `ho2020multihopqa` | Ho et al., *Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps* (2WikiMultihopQA), COLING 2020, pp. 6609–6625, DOI 10.18653/v1/2020.coling-main.580 | M+S: ACL Anthology entry and its official BibTeX | The 30 "2WikiMQA" items and the 4 timing items. Note the paper's short name `2WikiMQA` is LongBench's config name, not the dataset's own title. |

**Release / implementation pinning (A9 asks for this explicitly).** "Source indices 6–35" is only
meaningful against a fixed release ordering. Verified pins:

* Dataset release: HuggingFace `THUDM/LongBench`, configs `qasper` and `2wikimqa`, split `test`
  (verified from the official repo README's own loading snippet:
  `load_dataset('THUDM/LongBench', dataset, split='test')`).
* F1 implementation: `LongBench/metrics.py`, function `qa_f1_score` — verified by fetching the raw
  file; it normalises both strings (`normalize_answer`), whitespace-splits, and calls `f1_score` on
  token multisets. Driver: `LongBench/eval.py`.
* Both live under `LongBench/` in `github.com/THUDM/LongBench` after the v2 reorganisation
  (verified from the repo README: "The original LongBench v1 related files are moved under
  `LongBench/`").
* **Still needed from the authors:** the specific commit SHA of `THUDM/LongBench` and the dataset
  revision hash used to enumerate indices 6–35. I can verify that the files and configs exist; I
  cannot verify *which revision* the archived run consumed. Put the SHA in the reproducibility
  statement, not in the .bib.

---

## 2. The three boundary questions, answered

1. **Does prior work already do group quantization of KV?** — **Yes, identically.** FlexGen (ICML
   2023) publishes the same asymmetric min–max group quantizer at the same group size of 64, for
   both weights and the KV cache, with dequantization before compute. KIVI and KVQuant add the
   per-channel/per-token axis choice. Verified from FlexGen's own §"Group-wise Quantization".
2. **Does prior work already do per-layer bit assignment?** — **Yes.** KVTuner (ICML 2025) searches
   layer-wise mixed-precision KV bit-width pairs offline on calibration data and freezes them;
   AsymKV (COLING 2025) assigns per-layer widths separately to K and V. SqueezeAttention and HeadKV
   (already cited) do the analogous thing for *budget* rather than *bits*. The `[8,8,4,4,8,8,8]`
   vector is an instance of a known design surface.
3. **Has anyone quantized a depth-split residual together with lower-layer hybrid state?** —
   **No verified work does.** The nearest points, and why each falls short:
   * `tomar2025xquant` quantizes cached *layer-input activations* and rematerialises K/V — a
     hidden-state cache rather than a KV cache, but per-layer inside one dense path, with no split
     depth and no recurrent state.
   * `chiang2025quamba2` quantizes recurrent state, but only the recurrent state, and for a full
     execution, not a persisted reusable entry.
   * `pan2025marconi` (cited) and the sparse-checkpoint line store hybrid attention+recurrent state
     for reuse but store it **unquantized**.
   * `liu2026comem` stores the depth-split residual and nothing else, **in BF16** (see §3).

   **This is the defensible delta and the only one:** the *composite object* — boundary residual +
   lower full-attention KV + lower convolution/recurrent state, packed as one entry and accounted
   under one denominator. Bullet 1 must be rewritten around the object and the accounting, not the
   packer and not the bit vector.

---

## 3. Did CoMem already quantize its retained residual? — **RESOLVED: No. BF16.**

The reviewer could not resolve this; it is resolvable, and the answer is definite.

**Evidence 1 — the exact archived primary source.** `literature/citation_lock.json` records
`liu2026comem` with `source_pdf_sha256 = 9451a471b0e007616452f99545ba861712d81c01a176dd5761c0445e01ef9c97`.
I located a local PDF whose SHA-256 matches that value byte-for-byte:
`/Users/liuhanzuo/Downloads/112_CoMem_Reusing_Transformer_.pdf`
(header: "Published at the COLM 2026 Workshop on Efficient Reasoning. CoMem: Reusing Transformer
Depth across Queries with Persistent Intermediate Residuals"). Extracted text shows:

* Artifact table, `Write / Cached state` row: *"Split j=12; disjoint 512-token chunks; chunk-local
  positions restart at zero; **one bf16 residual tensor h_j per token**."*
* §5.3: *"The residual store costs **8,192 bytes/token**"* — i.e. 4096 dims x 2 bytes, exactly BF16,
  with no packing overhead and no scale/bias sidecar.
* Storage equation §3: *"For Qwen3-8B this is 1/18: 8 KiB rather than 144 KiB/token **in bf16**."*
* Limitations, verbatim: *"**Quantization**, eviction, update contention, and production
  multi-tenancy **are not evaluated**."*
* Full-text search of the extracted PDF for `quant`: the only hits are the words "quantifies" /
  "quantified" and that one limitations sentence. No INT4/INT8/FP8/low-bit anywhere.

**Evidence 2 — the expanded arXiv version.** `arxiv.org/abs/2607.28263` (`liu2026understanding`,
"Understanding Is Done Early…", submitted 30 Jul 2026) was fetched. Its HTML contains **zero**
occurrences of `quantiz`, `INT8`, `INT4`, `8-bit`, `4-bit`, `FP8`, `low-bit`. Storage comparisons
are stated "in bf16"; no experiment stores the residual below BF16.

**Consequence, and it is large in the paper's favour if stated plainly.** CoMem retains **only**
`h_j`, **unquantized**, on a **dense** Qwen3-8B (36 layers, GQA) — it explicitly discards the
document prefix and stores no lower-layer KV and no recurrent state, because a dense model does
not need them for suffix replay. Q-CoMem's delta over CoMem is therefore *both* halves of the
retained-state decomposition:

* the low-bit packing of the residual (CoMem: BF16, never quantized, quantization explicitly listed
  as future work), **and**
* the additional lower full-attention KV and lower convolution/recurrent state terms, which exist
  only because the backbone is hybrid and which CoMem never had to account for.

This is a clean, checkable claim. **It is safe to state it in the paper**, and it should be stated,
because it converts the reviewer's open question into a positive, verifiable delta.

**One residual caveat for the authors — please confirm, do not let me assert it for you:** I
verified the *workshop* PDF and the *arXiv* version, which are the two artifacts `references.bib`
cites. If any newer CoMem revision exists that adds a quantized arm, this paragraph must be
re-checked before submission. The authors are the only ones who can rule that out.

---

## 4. Drafted LaTeX — Section 2 (Related Work)

Replace the existing `\paragraph{Hybrid state and cache management.}` block
(`main_r44_structure.tex` lines 179–189) with the following **two** paragraphs. The compression
works move out of the hybrid paragraph and into the new one, where they belong.

> **Note on `\ref` targets:** neither the retained-state equation (Section 4.1) nor the quantizer
> equation (Section 4.2) carries a `\label`, so the draft refers to `\ref{sec:decomposition}` and
> `\ref{sec:write}` rather than to equation numbers. If you add `\label{eq:decomposition}` and
> `\label{eq:quantizer}` during integration, swap them in — that would read better.

```latex
\paragraph{Hybrid state and cache management.}
Gated Delta Networks maintain mutable compact state~\citep{yang2025gateddeltanet};
Jamba, Marconi, and HYPIC address hybrid architectures or reusable hybrid
prefixes~\citep{lieber2025jamba,pan2025marconi,liu2026hypic}, and store that
hybrid state unquantized.  \qcomem{} differs by jointly accounting for the
split residual and the lower hybrid state under one denominator.  \method{}
validates the immutable-document/mutable-request boundary rather than replacing
these cache policies; Appendix
Table~\ref{tab:published-context} separates published numbers from our controls.

\paragraph{KV and state quantization.}
Low-bit storage of reusable state is established prior art, and \qcomem{}
claims none of its mechanisms.  The packer of Section~\ref{sec:write} is the
standard asymmetric affine quantizer~\citep{jacob2018quantization} applied
group-wise, the form FlexGen already used to hold both weights and the KV cache
as 4-bit codes over 64-element contiguous groups with per-group min/max
metadata, dequantized before computation~\citep{sheng2023flexgen}.  KIVI and
KVQuant fix the grouping axis, quantizing keys per channel and values per token
at 2--4 bits~\citep{liu2024kivi,hooper2024kvquant}, and XQuant caches quantized
layer inputs instead of keys and values and rematerializes them on
read~\citep{tomar2025xquant}.  Assigning different widths per layer is likewise
prior art: KVTuner searches sensitivity-aware layer-wise mixed-precision KV bit
pairs offline on calibration data, AsymKV gives keys and values separate
per-layer widths, and MiKV mixes precision by token
importance~\citep{li2025kvtuner,tao2025asymkv,yang2024mikv}, while Palu,
SqueezeAttention, and HeadKV allocate rank or per-layer/per-head budget rather
than bits~\citep{chang2025palu,wang2025squeezeattention,fu2025headkv}.  For
recurrent state, Quamba and Quamba2 quantize selective-SSM activations and
per-state-group recurrent inputs, including for a hybrid backbone, and
MambaQuant supplies a rotation-based recipe for the Mamba
family~\citep{chiang2025quamba,chiang2025quamba2,yue2025mambaquant}.  Our bit
vector $\mathbf b$ is therefore an application of an established design surface
and not a new selection method.  What this literature does not cover is the
object being quantized: each of these methods compresses one homogeneous cache
along a single execution path, whereas a hybrid split-replay entry is
heterogeneous by construction --- a depth-$j$ boundary residual, the lower
full-attention KV, and the lower convolution/recurrent state --- and we report
Store for that whole entry (Section~\ref{sec:decomposition}) rather than for the
residual alone, which CoMem itself retains unquantized in
BF16~\citep{liu2026comem}.
```

**Optional trimmings**, in the order I would cut if space is tight:
`yue2025mambaquant` -> `yang2024mikv` -> `tomar2025xquant`. Do **not** cut `sheng2023flexgen` or
`li2025kvtuner`: those two are the exact prior art the reviewer named, and dropping either
re-opens Q2.

**Do not add** a sentence here about the absence of a quantized exact-cache baseline. That is A5's
territory and a Related Work paragraph is the wrong place to concede it. If A5 lands, the natural
cross-reference sentence is:
`A quantized exact-cache policy is the matching capacity baseline and is reported in Section~\ref{sec:memory}.`

---

## 5. Optional concurrent-work sentence (authors' discretion)

The meta-review explicitly rejected requiring the reviewer's arXiv identifiers and called a
concurrent-work sentence optional. I nevertheless independently resolved two of them on arXiv, so
the authors can make an informed choice. **Both postdate the submission and neither is prior art.**

| Key | Work | Verification | Relation |
|---|---|---|---|
| `zhang2026damp` | Zhang, Tan, Sun, Yu, Jiang, Xie, Cai, Zeng, *DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization*, arXiv:2608.27513, 27 Aug 2026, cs.LG | M+S: arXiv abs page (title, 8 authors, date, abstract) | Channel-level mixed precision for **recurrent-state storage** in GDN/KDA-style models: high-risk channels kept above INT8, rest INT8; 69.1% recurrent-state storage reduction. Closest work to Q-CoMem's Q8 GDN term. Posted ~1 week before this revision. |
| `yu2026dasc` | Yu, Sun, Tan, Zhang, Xie, Cai, Liu, *DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving*, arXiv:2608.30386, 31 Aug 2026, cs.LG | M+S: arXiv abs page | 2.63x compression of KDA/GDN recurrent-state checkpoints for serving. This is one of the identifiers the meta-review declined to require; it does exist, and it is concurrent, not prior. |

If used, one sentence, clearly marked concurrent, at the end of the quantization paragraph:

```latex
Concurrent preprints posted after this submission compress the recurrent term
alone: DAMP keeps high-risk recurrent-state channels above INT8, and DASC
compresses Gated DeltaNet and Kimi Delta Attention state for
serving~\citep{zhang2026damp,yu2026dasc}.
```

I recommend **including it**. It costs one sentence, it is verifiable, and it pre-empts the
reviewer re-raising the same identifiers next round while making clear they are not prior art.

---

## 6. Drafted LaTeX — Section 1, contribution bullet 1

Replace bullet 1 (`main_r44_structure.tex` lines 110–114).

**Current text claims the mechanism** ("Real groupwise Q4/Q8 packing and a state-type/per-layer bit
assignment"). Both halves of that phrase are FlexGen (2023) and KVTuner/AsymKV (2025) respectively.
The replacement claims the *object* and the *denominator*:

```latex
  \item \textbf{Whole-entry accounting for a hybrid split-replay entry.}
  \qcomem{} applies group-wise asymmetric quantization and per-layer,
  per-state-type width assignment --- neither of which we claim as new
  (Section~\ref{sec:related}) --- to an entry those methods do not address:
  the depth-$j$ boundary residual \emph{together with} the lower
  full-attention KV and the lower convolution/recurrent state that a hybrid
  backbone needs to replay a later query.  CoMem retains only the residual and
  retains it unquantized in BF16, so counting the residual alone understates a
  hybrid deployment; every number we report is measured under the whole-entry
  denominator of Section~\ref{sec:decomposition}
  (Section~\ref{sec:method}).
```

Notes for integration:
* The bullet now makes exactly two claims, both checkable: (i) the entry is the composite object,
  (ii) the denominator is the whole entry. Neither is a mechanism claim.
* The CoMem/BF16 clause is verified in §3 above and is what sizes the delta. Keep it.
* `\ref{sec:related}` forward-references the new paragraph; the paper already forward-references
  across sections in this list, so this is consistent.
* If the abstract repeats the "Real groupwise Q4/Q8 packing" phrasing, it needs the same edit —
  grep for `groupwise` and `packing` before you finish.

---

## 7. Suggested row for the published-systems context table

`tables/related_work_reported_context.tex` (`tab:published-context`). One row is enough; KVTuner is
the right choice because it is the work whose contribution most nearly overlaps the bit vector.
Numbers below are quoted from KVTuner's publisher-hosted abstract and were verified verbatim.

```latex
KVTuner~\citep{li2025kvtuner} & Offline sensitivity-aware search over layer-wise mixed-precision KV bit pairs\newline \textit{Setting:} Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct; mathematical- and general-reasoning suites & Nearly lossless 3.25-bit mixed-precision KV for Llama-3.1-8B-Instruct and 4.0-bit for Qwen2.5-7B-Instruct; up to 21.25\% higher maximum inference throughput vs. KIVI-KV8 & Layer-wise bit search for a dense KV cache; it does not quantize a split residual or recurrent state, and no common Qwen3.5 slice exists \\
```

Optionally a second row for `chiang2025quamba2` covering the recurrent-state side ("4x memory
reduction with 1.6% average accuracy drop; 1.3x prefill and 3x generation speedup for Quamba2-8B",
verified from the publisher abstract) — add it only if the table is not already at its page budget.

---

## 8. Naming consistency (the rest of A9, flagged not fixed)

A9 also requires: *"no row reporting a configuration measured in this paper may be labelled
'CoMem'."* Confirmed present in `main_r44_structure.tex`:

* line 854 — "vanilla dense plus **CoMem** BF16/Q8/Q4 at depth 7" (Mac motivation table caption)
* line 856 — "**CoMem** Q8/Q4/mixed variants (six configurations total)" (H20 deployment scope)
* line 1142 — "retain 8.78x and 20.39x as much Store as **CoMem Q8**"
* plus the corresponding cells inside `tables/mac_m4_motivation_table*.tex`,
  `tables/h20_deployment_table_r40_candidate.tex`, `tables/related_serving_table.tex`

These are configurations measured *in this work*; they must read `\qcomem{}` (e.g. `Q-CoMem Q8`).
This is a mechanical rename and is **not** included in this note's drafts — flagging it so it is
not lost, since leaving it undone defeats the positioning the new bullet 1 depends on.

---

## 9. BibTeX — ready to paste

Style matches `references.bib`: aligned `field     = {...}`, braced acronyms/model names to protect
capitalisation, `url` on every entry, `doi` where the publisher provides one, ICLR entries in the
repo's existing `proceedings.iclr.cc` form.

```bibtex
% ============================================================
% A9 -- KV/state quantization thread (all metadata publisher-verified 2026-09-02)
% ============================================================

@inproceedings{jacob2018quantization,
  author    = {Jacob, Benoit and Kligys, Skirmantas and Chen, Bo and Zhu, Menglong and Tang, Matthew and Howard, Andrew G. and Adam, Hartwig and Kalenichenko, Dmitry},
  title     = {Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference},
  booktitle = {2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {2704--2713},
  year      = {2018},
  publisher = {IEEE},
  doi       = {10.1109/CVPR.2018.00286},
  url       = {https://doi.org/10.1109/CVPR.2018.00286}
}

@inproceedings{sheng2023flexgen,
  author    = {Sheng, Ying and Zheng, Lianmin and Yuan, Binhang and Li, Zhuohan and Ryabinin, Max and Chen, Beidi and Liang, Percy and R{\'e}, Christopher and Stoica, Ion and Zhang, Ce},
  title     = {{FlexGen}: High-Throughput Generative Inference of Large Language Models with a Single {GPU}},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {202},
  pages     = {31094--31116},
  publisher = {PMLR},
  year      = {2023},
  url       = {https://proceedings.mlr.press/v202/sheng23a.html}
}

@inproceedings{liu2024kivi,
  author    = {Liu, Zirui and Yuan, Jiayi and Jin, Hongye and Zhong, Shaochen and Xu, Zhaozhuo and Braverman, Vladimir and Chen, Beidi and Hu, Xia},
  title     = {{KIVI}: A Tuning-Free Asymmetric 2bit Quantization for {KV} Cache},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  pages     = {32332--32344},
  publisher = {PMLR},
  year      = {2024},
  url       = {https://proceedings.mlr.press/v235/liu24bz.html}
}

@inproceedings{hooper2024kvquant,
  author    = {Hooper, Coleman and Kim, Sehoon and Mohammadzadeh, Hiva and Mahoney, Michael W. and Shao, Yakun Sophia and Keutzer, Kurt and Gholami, Amir},
  title     = {{KVQuant}: Towards 10 Million Context Length {LLM} Inference with {KV} Cache Quantization},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  pages     = {1270--1303},
  publisher = {Curran Associates, Inc.},
  year      = {2024},
  doi       = {10.52202/079017-0040},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html}
}

@inproceedings{li2025kvtuner,
  author    = {Li, Xing and Xing, Zeyu and Li, Yiming and Qu, Linping and Zhen, Hui-Ling and Yao, Yiwu and Liu, Wulong and Pan, Sinno Jialin and Yuan, Mingxuan},
  title     = {{KVTuner}: Sensitivity-Aware Layer-Wise Mixed-Precision {KV} Cache Quantization for Efficient and Nearly Lossless {LLM} Inference},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  pages     = {36451--36485},
  publisher = {PMLR},
  year      = {2025},
  url       = {https://proceedings.mlr.press/v267/li25dd.html}
}

@inproceedings{tao2025asymkv,
  author    = {Tao, Qian and Yu, Wenyuan and Zhou, Jingren},
  title     = {{AsymKV}: Enabling 1-Bit Quantization of {KV} Cache with Layer-Wise Asymmetric Quantization Configurations},
  booktitle = {Proceedings of the 31st International Conference on Computational Linguistics},
  address   = {Abu Dhabi, UAE},
  publisher = {Association for Computational Linguistics},
  pages     = {2316--2328},
  month     = jan,
  year      = {2025},
  url       = {https://aclanthology.org/2025.coling-main.158/}
}

@article{yang2024mikv,
  author        = {Yang, June Yong and Kim, Byeongwook and Bae, Jeongin and Kwon, Beomseok and Park, Gunho and Yang, Eunho and Kwon, Se Jung and Lee, Dongsoo},
  title         = {No Token Left Behind: Reliable {KV} Cache Compression via Importance-Aware Mixed Precision Quantization},
  journal       = {arXiv preprint arXiv:2402.18096},
  year          = {2024},
  eprint        = {2402.18096},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  doi           = {10.48550/arXiv.2402.18096},
  url           = {https://arxiv.org/abs/2402.18096}
}

@article{tomar2025xquant,
  author        = {Tomar, Aditya and Hooper, Coleman and Lee, Minjae and Xi, Haocheng and Tiwari, Rishabh and Kang, Wonjun and Manolache, Luca and Mahoney, Michael W. and Keutzer, Kurt and Gholami, Amir},
  title         = {{XQuant}: Breaking the Memory Wall for {LLM} Inference with {KV} Cache Rematerialization},
  journal       = {arXiv preprint arXiv:2508.10395},
  year          = {2025},
  eprint        = {2508.10395},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  doi           = {10.48550/arXiv.2508.10395},
  url           = {https://arxiv.org/abs/2508.10395}
}

@inproceedings{chiang2025quamba,
  author    = {Chiang, Hung-Yueh and Chang, Chi-Chih and Frumkin, Natalia and Wu, Kai-Chiang and Marculescu, Diana},
  title     = {{Quamba}: A Post-Training Quantization Recipe for Selective State Space Models},
  booktitle = {International Conference on Learning Representations},
  volume    = {2025},
  pages     = {101328--101354},
  year      = {2025},
  url       = {https://proceedings.iclr.cc/paper_files/paper/2025/hash/fb4b2fb2434f7cce5cb5ab50271296ee-Abstract-Conference.html}
}

@inproceedings{chiang2025quamba2,
  author    = {Chiang, Hung-Yueh and Chang, Chi-Chih and Frumkin, Natalia and Wu, Kai-Chiang and Abdelfattah, Mohamed S. and Marculescu, Diana},
  title     = {{Quamba2}: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  pages     = {10411--10427},
  publisher = {PMLR},
  year      = {2025},
  url       = {https://proceedings.mlr.press/v267/chiang25a.html}
}

@inproceedings{yue2025mambaquant,
  author    = {Yue, Yuxuan and Hu, Xing and Yang, Dawei and Yuan, Zhihang and Jiang, Zixu and Chen, Zhixuan and Yu, Jiangyong and Chen, Xu and Zhou, Sifan},
  title     = {{MambaQuant}: Quantizing the {Mamba} Family with Variance Aligned Rotation Methods},
  booktitle = {International Conference on Learning Representations},
  volume    = {2025},
  pages     = {33231--33250},
  year      = {2025},
  url       = {https://proceedings.iclr.cc/paper_files/paper/2025/hash/51ba8a68f471d952af625d1faf55e6c6-Abstract-Conference.html}
}

% ============================================================
% A9 -- datasets used throughout but previously uncited (Q6)
% ============================================================

@inproceedings{bai2024longbench,
  author    = {Bai, Yushi and Lv, Xin and Zhang, Jiajie and Lyu, Hongchang and Tang, Jiankai and Huang, Zhidian and Du, Zhengxiao and Liu, Xiao and Zeng, Aohan and Hou, Lei and Dong, Yuxiao and Tang, Jie and Li, Juanzi},
  title     = {{LongBench}: A Bilingual, Multitask Benchmark for Long Context Understanding},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  address   = {Bangkok, Thailand},
  publisher = {Association for Computational Linguistics},
  pages     = {3119--3137},
  month     = aug,
  year      = {2024},
  doi       = {10.18653/v1/2024.acl-long.172},
  url       = {https://aclanthology.org/2024.acl-long.172/}
}

@inproceedings{dasigi2021qasper,
  author    = {Dasigi, Pradeep and Lo, Kyle and Beltagy, Iz and Cohan, Arman and Smith, Noah A. and Gardner, Matt},
  title     = {A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  address   = {Online},
  publisher = {Association for Computational Linguistics},
  pages     = {4599--4610},
  month     = jun,
  year      = {2021},
  doi       = {10.18653/v1/2021.naacl-main.365},
  url       = {https://aclanthology.org/2021.naacl-main.365/}
}

@inproceedings{ho2020multihopqa,
  author    = {Ho, Xanh and Duong Nguyen, Anh-Khoa and Sugawara, Saku and Aizawa, Akiko},
  title     = {Constructing A Multi-hop {QA} Dataset for Comprehensive Evaluation of Reasoning Steps},
  booktitle = {Proceedings of the 28th International Conference on Computational Linguistics},
  address   = {Barcelona, Spain (Online)},
  publisher = {International Committee on Computational Linguistics},
  pages     = {6609--6625},
  month     = dec,
  year      = {2020},
  doi       = {10.18653/v1/2020.coling-main.580},
  url       = {https://aclanthology.org/2020.coling-main.580/}
}

% ---- Release/implementation pins.  Add the commit SHA in the note field
% ---- before submission; I could verify the files and configs exist but not
% ---- which revision the archived 60-item run consumed.

@misc{bai2024longbenchrelease,
  author       = {Bai, Yushi and Lv, Xin and Zhang, Jiajie and Lyu, Hongchang and Tang, Jiankai and Huang, Zhidian and Du, Zhengxiao and Liu, Xiao and Zeng, Aohan and Hou, Lei and Dong, Yuxiao and Tang, Jie and Li, Juanzi},
  title        = {{LongBench} v1 Dataset Release},
  year         = {2024},
  howpublished = {Hugging Face dataset repository \texttt{THUDM/LongBench}, configurations \texttt{qasper} and \texttt{2wikimqa}, split \texttt{test}},
  note         = {Accessed 2026-09-02; revision hash to be pinned by the authors},
  url          = {https://huggingface.co/datasets/THUDM/LongBench}
}

@misc{bai2024longbenchcode,
  author       = {Bai, Yushi and Lv, Xin and Zhang, Jiajie and Lyu, Hongchang and Tang, Jiankai and Huang, Zhidian and Du, Zhengxiao and Liu, Xiao and Zeng, Aohan and Hou, Lei and Dong, Yuxiao and Tang, Jie and Li, Juanzi},
  title        = {{LongBench} v1 Official Evaluation Code},
  year         = {2024},
  howpublished = {GitHub repository \texttt{THUDM/LongBench}; F1 computed by \texttt{LongBench/metrics.py::qa\_f1\_score} via \texttt{LongBench/eval.py}},
  note         = {Accessed 2026-09-02; commit SHA to be pinned by the authors},
  url          = {https://github.com/THUDM/LongBench}
}

% ============================================================
% OPTIONAL -- concurrent work, posted after submission.  Include only with the
% concurrent-work sentence of section 5; the meta-review does not require these.
% ============================================================

@article{zhang2026damp,
  author        = {Zhang, Tao and Tan, Jianchao and Sun, Pingwei and Yu, Yanqi and Jiang, Zixu and Xie, Yuchen and Cai, Xunliang and Zeng, Ziqian},
  title         = {{DAMP}: Decay-Aware Mixed-Precision Recurrent-State Quantization},
  journal       = {arXiv preprint arXiv:2608.27513},
  year          = {2026},
  eprint        = {2608.27513},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  url           = {https://arxiv.org/abs/2608.27513}
}

@article{yu2026dasc,
  author        = {Yu, Yanqi and Sun, Pingwei and Tan, Jianchao and Zhang, Tao and Xie, Yuchen and Cai, Xunliang and Liu, Yao},
  title         = {{DASC}: Decay-Aware State Compression for Hybrid Linear-Attention Serving},
  journal       = {arXiv preprint arXiv:2608.30386},
  year          = {2026},
  eprint        = {2608.30386},
  archiveprefix = {arXiv},
  primaryclass  = {cs.LG},
  url           = {https://arxiv.org/abs/2608.30386}
}
```

---

## 10. Candidates considered and EXCLUDED, with reasons

Honest short list. These were surfaced during searching and dropped.

| Candidate | Why excluded |
|---|---|
| The reviewer's `2608.11231` and `2608.30310` | I could not resolve either to a matching arXiv record in the time available, and the meta-review already rejected requiring them. Excluded on the verification rule. (`2608.30386` *did* resolve and is quarantined as optional concurrent work in §5.) |
| MiKV as an "ICML 2024" paper | Widely described as ICML 2024 in secondary sources, but **dblp lists only `CoRR abs/2402.18096`** and the arXiv page carries no `Comments:` or `Journal ref:` venue field. Cited as a preprint or not at all — never as a conference paper. |
| Atom, QServe/QoQ, GEAR, ZipCache, SKVQ, Coupled Quantization, MiniKV, RotateKV, SQuat, QJL | All plausible and several are certainly real, but each would add a fourth or fifth redundant anchor for a point FlexGen/KIVI/KVQuant already establish. Not verified, therefore not proposed. Adding unverified breadth is what created Q2's sibling problems. |
| "Sparse Prefix Caching for Hybrid and Recurrent LLM Serving", arXiv:2605.05219 | Metadata verified on arXiv (Shirokikh & Nikolenko), but it stores recurrent state at sparse checkpoints **without quantization**, so it does not support any sentence in the quantization paragraph, and Marconi already anchors the hybrid-prefix thread. Excluded on support, not on metadata. |
| Any TurboQuant/"KV Cache is 1 Bit Per Channel"/MiniCache row | Same reason: redundant with the three group-quantization anchors, and not verified to publisher level. |
| Q-BERT (Shen et al. 2020), which FlexGen cites for group-wise quantization | One hop too far. FlexGen is the direct precedent for the exact scheme *and* the group size of 64; adding its ancestor buys nothing at this page budget. |

---

## 11. Open items for the authors

1. **Confirm no newer CoMem revision quantizes the residual.** §3 is definitive for the two cited
   artifacts (workshop PDF, SHA-256 matched to the locked hash; arXiv:2607.28263). Only the authors
   can rule out a newer revision.
2. **Pin the LongBench commit SHA and dataset revision** for "source indices 6--35". I verified the
   repo, the configs, the split and the F1 function; I cannot verify which revision the archived run
   consumed, and the index range is meaningless without it.
3. **Do the `CoMem` -> `\qcomem{}` rename** in the four manuscript locations and the four table
   files listed in §8. Not drafted here.
4. **Re-check the abstract** for mechanism language mirroring the old bullet 1 ("groupwise",
   "packing", "per-layer bit assignment") and apply the same claim demotion.
