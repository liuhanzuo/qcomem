# Round 40 independent meta-review

## Identity and validity

- PDF SHA-256 verified:
  `0906080e3d16c0f8ee071f5d3aa2f6d4d541e7f7d8a2cc3efe9e67c0d0916d5b`
- Canonical rubric SHA-256 verified:
  `df368bb0b31b60a75f81d155a6b01962865aedb2b2984443f3ba6cd8c153d874`
- The meta-reviewer read the complete 29-page PDF, the three valid isolated
  reviewer records, their protocol, and the panel summary. The invalid initial
  third-review attempt is not included.

## Decision

- Overall: **6 / 10**
- Confidence: **4 / 5**
- Soundness: **3 / 4**
- Presentation: **3 / 4**
- Contribution: **2 / 4**
- Recommendation: marginally above the acceptance threshold
- Current evidence ceiling: **6 / 10**

The paper credibly supports its explicitly conditional conclusion on the
declared Qwen3.5/H20 stack, honest capture, fixed schedules, and bounded dispatch
surface. Its strongest evidence is the historical alias: the defective path
preserves tokens, FP32 logits, terminal GDN, and logical KV in 8/8 cells while
damaging persistent-base storage; the repaired path is storage-clean in 8/8.
This directly shows why semantic equality alone is insufficient.

The conventional persistent-base invariant also catches all eight cases, so
ForkAudit's demonstrated unique increment is earlier phase/owner/layer/family
localization and unified fail-closed accounting, not exclusive detection. The
panel contribution median of 2 is therefore appropriate.

## Evidence boundary

Three material gaps prevent an 8 but do not invalidate the current 6:

1. Correct slot-ID-to-live-tensor binding remains in the TCB. The independent
   preproducer census closes expected-slot enumeration but does not independently
   validate live-object binding.
2. There is no blind, matched head-to-head against a frozen strong conventional
   suite that reports unique detection, first failure, localization, false
   positives, or maintenance and runtime cost.
3. The main experiment is batch-one and sequential round-major. The two-stream
   cohort does not establish native continuous/ragged batching, true in-flight
   cancellation, or matched H20 audit-on/off cost.

There is no supported critical scientific defect that must be resolved before a
marginal score-6 acceptance under the paper's narrow claim. If the internal
target is a stable score 8, all three evidence gaps require new experiments.
The canonical workflow quality-pass gate is also unmet because this panel has
three rather than five reviewers, contribution median 2 rather than at least 3,
and major evidence gaps remain.

## Revisions possible with current evidence

- Separate attention kernel-cache artifact/configuration provenance from GDN
  eager-route provenance; avoid suggesting compiled GDN or whole-path compiled
  execution attestation.
- State more prominently that the base invariant also catches the historical
  defect and that ForkAudit's demonstrated increment is transition-time
  localization.
- Promote the post-execution run-ID correction into the main reproducibility
  explanation and state that it changes only the comparison-row ID, not
  candidate bytes, outputs, or frozen source.
- Explain the 0.005 tolerance and row sampling, or label the threshold clearly
  as a preregistered engineering choice.
- Reduce count density and the prominence of allocator/serving context, and
  compress the artifact-path appendix to stable evidence IDs and replay roots.

## New experiments required for a score-8 case

1. A source-distinct live-binding witness with blind slot-swap, stale-handle,
   semantic-relabel, and wrong-live-tensor challenges.
2. A frozen strong conventional suite evaluated on a blind fault set with
   detection, localization, false-positive, runtime, artifact, and maintenance
   comparisons.
3. Native continuous/ragged batching and true in-flight cancellation with
   matched H20 audit-off/on latency, throughput, memory, and perturbation.

R40 is the retained candidate within the isolated materials supplied to this
meta-reviewer, but the reviewer was not given prior checkpoints and therefore
cannot make a valid cross-round best-checkpoint selection. Repository-level
selection must retain the stronger completed historical checkpoint if its
lexicographic review gates remain superior.

This is an internal automated review simulation, not an ICLR acceptance
prediction or an external reviewer decision.
