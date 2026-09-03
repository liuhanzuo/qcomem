# Round 30 PDF-only Terra panel: archival meta-review

This document is an archival synthesis of three independent reviews, not a fourth review. All three reviewers used fresh isolated `gpt-5.6-terra` contexts, received the same prompt, and were restricted to the same frozen PDF. The raw returned review text is preserved in the three files under `reviews/`.

## Frozen review object

- PDF: `/tmp/forkaudit-final-pdf-only.xs72Hx/ForkAudit.pdf`
- SHA-256: `408f7d495a383cd40df6a4bbe49dbf6ef6e732bb3663eae791a4b96000bfcd39`
- Access mode: PDF only; the frozen directory contained exactly one file.
- Model: `gpt-5.6-terra` for each of three fresh independent runs.

## Identical prompt

The following prompt was used verbatim for all three reviewers:

```text
Act as an independent ICLR conference reviewer. Review only this frozen PDF: /tmp/forkaudit-final-pdf-only.xs72Hx/ForkAudit.pdf. The directory contains exactly one file. Do not open, search, or infer from any other file, repository source, evidence package, conversation history, prior review, or author note. Judge the submission solely from what is visible in the PDF, as a fresh reviewer with no assigned specialty. Read the entire PDF, including figures, tables, references, and appendices. Return an official-style review with: (1) concise summary and claimed contributions; (2) strengths; (3) weaknesses, distinguishing fatal from fixable; (4) questions for authors; (5) numerical/evidence/scope consistency audit; (6) presentation and figure/table readability; (7) overall score on the ICLR 1–10 scale with label, confidence 1–5, and accept/reject recommendation. Be critical and calibrated. Do not coordinate with other reviewers and do not modify any file.
```

## Scores and confidence

| Review | Archived file | Score | Label / recommendation | Confidence |
|---|---|---:|---|---:|
| 1 | `reviews/terra_pdf_review_1.md` | 4/10 | Weak Reject | 4/5 |
| 2 | `reviews/terra_pdf_review_2.md` | 4/10 | Borderline reject | 4/5 |
| 3 | `reviews/terra_pdf_review_3.md` | 4/10 | Below acceptance threshold / Reject | 4/5 |

Panel median, lower quartile, and minimum are all 4; score dispersion is zero. Median confidence is 4. All three recommendations are on the reject side of the threshold.

## Consensus strengths

1. The paper is unusually candid about the assurance boundary and does not silently convert trusted-capture replay into independent recapture, production scheduling, capacity, or end-to-end correctness.
2. The receipt contract is concrete and technically useful: record classes, phase semantics, storage/range witnesses, target-to-record dependencies, and the KV-by-GDN ownership factorial are clearly specified.
3. The main reported counts and allocator arithmetic visible in the PDF are internally consistent. No reviewer found a numerical contradiction in the headline evidence.
4. The target-gate suppression matrix and positive controls show that output equality alone is insufficient for the designed violations.
5. Memory denominators and unpooled cohorts are carefully scoped, even though the volume of contextual material weakens focus.

## Consensus acceptance-critical issues

1. **Trusted-producer/common-mode ceiling.** All three reviewers regard the honest-capture-producer assumption as the central weakness. Replay checks consistency of producer-supplied records but cannot exclude coherent omission or fabrication, transient mutation between snapshots, unrecorded semantics, or compiled-dispatch divergence. The source-distinct observer remains in the same process and shares candidate-created tensors, phase labels, and PyTorch storage semantics.
2. **No independent end-to-end recapture or differential implementation.** Selected numerical sidecars start from captured inputs and therefore do not validate upstream construction or full-model behavior. Cross-arm and cross-fan-out equality can preserve common-mode defects.
3. **Narrow empirical scope.** Evidence comes from one model and software stack, one H20 family, one BF16 KV/page geometry and partial-tail shape, a fixed 32-token transition, `N<=32`, eight greedy steps, and sequential batch-one execution. Native ragged/continuous batching and real scheduler interleavings are not demonstrated.
4. **Designed-fault evidence is not realistic fault coverage.** Nine hand-crafted positive controls show gate reachability, not sensitivity, specificity, false-positive rate, false-negative rate, or robustness to held-out/naturally occurring faults.
5. **Contribution boundary remains integration-heavy.** Reviewers agree that the ingredients are largely established and want a sharper threat model and a clearer distinction between a reusable protocol contribution and a bespoke validation harness.

## Other consensus issues

- The many explicitly unpooled CoMem/HYPIC/Hydragen/Palu/Marconi and lifecycle/context cohorts dilute the central paper and create excessive cognitive load.
- The abstract, acronyms, appendices, and several tables/labels are too dense at ordinary reading scale.
- The live full-audit overhead and replay-package size do not yet establish deployment practicality; the likely operational role may be offline or debugging-oriented.
- `RC` / “receipt-complete” and “exact” can be read more strongly than intended unless their producer-trusted and canonicalization boundaries are foregrounded.

## Material disagreements and reviewer-specific observations

- **Figure readability:** Review 1 found Figures 1–3 clear; Review 2 found Figure 2 clearest but Figures 1 and 3 small at normal scale; Review 3 found the conceptual diagrams strong but the overall paper and appendix labels too dense. This is a degree-of-readability disagreement, not a disagreement about correctness.
- **Transition-token accounting:** Review 2 uniquely flagged the apparent `32 + 8 = 40` versus reported 39 appended-token ambiguity. The other reviewers did not call it inconsistent. This should be resolved by an explicit counting convention/equation rather than treated as a demonstrated numerical error.
- **Postexecution provenance:** Review 3 uniquely emphasized the raw-versus-derived GDN schema correction, detached-manifest run-ID issue, and the meaning of CPU-FP32 canonicalized “exact” equality.
- **Artifact access:** Review 2 noted that no artifact could be assessed under PDF-only access. This is a limitation of the mandated review view, not evidence that the artifact is absent, and it must not be converted into an artifact-availability allegation.
- **Severity labels:** The wording ranges from “Weak Reject” to “Borderline reject” to “Reject,” but there is no score disagreement: all three assign 4/10 with confidence 4/5.

## Items fixable by manuscript text or layout alone

1. Reframe the central claim explicitly as **trusted-capture trace-consistency / ownership-trace validation on one fixed stack**, or define a precise trusted computing base and adversary that justifies retaining “audit.”
2. Rename or visually qualify `RC` so it cannot be mistaken for independent evidence completeness; foreground that completeness is only relative to the mandatory producer-trusted trace schema.
3. Add a compact, explicit accounting equation for the 39 appended-state tokens and distinguish consumed tokens, predicted tokens, and state transitions.
4. Sharpen the novelty statement: identify the reusable contract-level integration, separate it from stack-specific instrumentation, and avoid implying novelty of established constituent mechanisms.
5. Move or compress unpooled deployment/related-work/context cohorts, streamline the abstract, and reduce acronym/cohort proliferation.
6. Foreground that “exact” refers to the declared canonicalization/digest relation rather than bitwise identity of original device tensors.
7. Present live capture as the measured narrow audit/debug mode unless further evidence supports a deployment claim; explain the 4.321x timing and approximately 851 MiB replay package at the point of use.
8. Enlarge or simplify the densest figure labels and appendix tables, particularly the areas identified around Figures 1 and 3 and Tables 6, 13, and 23.
9. Clarify the chronology and non-effect of postexecution schema/manifest corrections to the extent already supported by frozen provenance. Independent confirmation of that chronology is not a text-only fix.

These changes can improve accuracy, focus, and presentation, but the panel does not indicate that wording alone would remove the common-mode evidence ceiling.

## Items requiring new evidence or analysis

1. A genuinely independent live recapture or differential implementation that does not share candidate-created tensor objects, candidate phase labels, runtime-side hooks, or the same storage-view API.
2. Independent end-to-end validation from inputs through relevant hybrid transitions and outputs, rather than operator checks that begin from candidate-captured intermediates.
3. Native ragged/continuous batching with real scheduler admission/interleaving, plus relevant cancellation, eviction/re-admission, multiple-document, and transition-length cases. If these are not run, all corresponding motivation and applicability claims must remain explicitly out of scope.
4. Held-out, naturally occurring, blind, or adversarially selected ownership faults, with clean negative controls and false-positive/false-negative characterization.
5. An ablation against a simpler trusted trace validator and standard differential/metamorphic tests, showing the incremental detection or diagnostic value of each receipt family.
6. A formal soundness/completeness analysis relative to a precise runtime and threat model, or an explicit statement that no such guarantee is claimed.
7. Independently checkable provenance for postexecution corrections and, if claimed, a closed immutable timeline tying executed sources, capture artifacts, corrections, and replay results together.
8. If practicality is a claim, cost/coverage measurements for reduced audit configurations and broader workloads; if not, narrow the operational claim to the measured audit/debug scenario.

## Revision gate

**Current gate: FAIL for acceptance-quality blind review.** The panel is unanimous at 4/10, below both the skill's minimum individual target of 6 and preferred median target of 8. No reviewer found evidence fabrication or headline arithmetic failure; the failed gate is contribution/soundness strength under the visible evidence boundary.

Before another score-bearing blind round, the minimum defensible path is:

1. complete the text-only reframing and the 39-token/`RC`/“exact” clarity fixes;
2. add at least one score-changing independent evidence path—preferably independent recapture or differential end-to-end validation;
3. add realistic held-out/negative-control fault evidence and either native scheduler batching evidence or an unambiguous fixed-sequential scope boundary;
4. rebuild and visually audit the full PDF, verify all new claims against registered evidence, and then freeze a new PDF-only snapshot for three fresh identical independent reviews.

If only wording and layout are changed, the revision may be clearer and more honest, but the Round 30 consensus predicts that the acceptance-critical trusted-producer/common-mode objection will remain. Internal subagent reviews are diagnostic signals and do not predict actual venue acceptance.
