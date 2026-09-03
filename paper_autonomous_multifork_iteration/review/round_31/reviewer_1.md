## 1. Summary and claimed contributions

This paper presents ForkAudit, a trusted-capture, offline trace-validation protocol for ownership isolation in hybrid LLM inference with both paged KV state and mutable recurrent GDN state. Its key idea is to record typed, phase-indexed lifecycle, storage, call-contract, and semantic receipts, then replay registered predicates under an explicit honest, mandatory-event-complete capture assumption.

On one Qwen3.5-35B-A3B/H20 configuration, the paper reports complete replay coverage for identity, prefix immutability, private ownership, tail COW, cross-arm equality, and cross-fan-out consistency; dispatch provenance is explicitly only partial because compiled binary/autotuning identity is not bound. It further reports selected captured-boundary numerical checks, nine designed live faults, a target-gate suppression matrix, allocator measurements, and several explicitly unpooled support cohorts.

## 2. Strengths

- The assurance boundary is unusually clear. The paper repeatedly distinguishes trace-relative validation from independent live monitoring, compiled-dispatch attestation, unseen-fault coverage, and end-to-end correctness. This is a substantive strength, not merely cautious wording.
- The protocol targets a real gap in hybrid inference: output equivalence does not establish isolation of mutable recurrent state. The phase-conditioned request/base and request/peer ownership relations, registered transition-boundary check, and tail-COW treatment are concrete and well specified.
- The paper has strong methodological discipline: typed mandatory records, fail-closed coverage semantics, normalized pointer-free storage ranges, predeclared gates, and separation of coverage from verdicts make the proposed validation target precise.
- The empirical evidence is detailed for the claimed fixed-stack setting. The 2x2 KV-by-GDN ownership design, 96 configurations, 288 cross-N comparisons, exact canonical observables, and carefully separated memory denominators are more informative than reporting token equality alone.
- The fault analysis is appropriately qualified. In particular, the suppressed-gate matrix shows that token equality misses all five completing faults and full logits catch only M5, while correctly avoiding an invalid “detection rate” interpretation.
- The paper provides meaningful reproducibility metadata, including a claim/artifact map, replay scope, hashes, postexecution corrections, and explicit distinctions between raw, derived, and replay-complete evidence.

## 3. Weaknesses

- The central guarantee is only as strong as an honest, mandatory-event-complete producer. Detached replay cannot detect coherent omission/fabrication, transient writes restored before snapshots, or unrecorded execution semantics. The source-distinct observer still shares process, objects, labels, and PyTorch storage APIs. These limitations are disclosed, but they substantially constrain the practical assurance value of the proposal.
- The validation is highly specialized: one model, one hardware family, BF16 KV, one page size, a 4,095-token document with one partial-tail geometry, a registered 32-token transition, short generation, and sequential batch-one round-major execution. The paper does not demonstrate that the protocol adapts robustly to actual native continuous/ragged batching, eviction, multi-document serving, varied recurrent implementations, or production schedulers.
- Novelty appears primarily integrative. The paper itself attributes paging/COW, hashing, metamorphic relations, differential checking, and memory accounting to prior work. The integration may be useful, but the evidence does not yet establish that this particular protocol is broadly needed or meaningfully better than a carefully engineered combination of existing tracing and testing techniques.
- The numerical checks remain captured-boundary tests: attention begins after candidate-produced RoPE inputs and GDN after candidate-produced normalization inputs; only 12 of 30 GDN layers are covered in the expanded sweep. The seeded wrong-operator controls demonstrate sensitivity to known, designed mutations, not independent end-to-end correctness or realistic bug coverage.
- Several results are deterministic single-run or tiny-cohort observations rather than robust evaluations. For example, the 4.321x audit-cost result comes from five pairs on one frozen request-step setting, and the memory results do not measure process-level memory or service capacity. The authors disclose these boundaries, but the resulting empirical significance is narrow.
- The submission describes extensive replay artifacts, but the frozen PDF itself cannot substantiate their execution or availability. A usable artifact release and an independent re-execution would be especially important given that the proposed contribution concerns trustworthy evidence capture.

## 4. Questions for the authors

1. What is the intended deployment/adoption model for ForkAudit? How much adapter-specific engineering is required to instantiate the schema for a new hybrid architecture or serving stack?
2. Can the authors demonstrate a stronger observer boundary, e.g., a separate-process recorder, lower-level allocator/kernel telemetry, or hardware-supported event capture, to reduce reliance on the candidate capture producer?
3. How would the phase-indexed protocol handle true continuous batching, eviction/re-admission, variable prefix lengths, and in-flight cancellation rather than only fixed synchronized or sequential schedules?
4. Can the authors provide evidence from at least one materially different hybrid architecture/runtime, especially one using an optimized recurrent kernel rather than the Transformers Torch GDN path?
5. What are the false-positive behavior and engineering overhead of mandatory trace coverage under normal runtime variability, version changes, and realistic long-running service workloads?
6. Will the full replay package, environment specification, and raw data needed to reproduce the numerical and fault results be released in a form that an independent reviewer can execute?

## 5. Reproducibility and ethics

The paper is strong on documenting reproducibility boundaries: it specifies manifests, hashes, replay commands/artifact locations, raw-versus-derived distinctions, and disclosed replay-only corrections. However, based on this PDF alone, the actual package cannot be inspected or rerun; independent artifact availability is therefore central to confidence in the reported results.

The ethics discussion is appropriately modest: there are no stated human-subject experiments, and the work does not claim safety or downstream-quality evaluation. The authors should nevertheless clarify data licensing/provenance and retention practices for PG-19-derived inputs and archived traces, particularly if captured prompts, model outputs, or tensors could be distributed.

## 6. Overall score (integer 1–10, using ICLR-style meaning; state score and concise justification)

**Score: 4 (Borderline Reject).** The paper is exceptionally careful about what its fixed-stack evidence does and does not show, and the trace contract is technically thoughtful. However, its novelty is mainly an integration of established mechanisms, while the assurance remains dependent on trusted capture and the evaluation is too narrow to establish broad practical impact or generality for an ICLR acceptance.

## 7. Confidence (integer 1–5; state confidence and concise justification)

**Confidence: 4.** The methodology, experiment boundaries, and limitations are described in substantial detail throughout the paper. My uncertainty concerns the uninspectable artifact execution and the extent to which the integration would transfer beyond the single demonstrated stack.

## 8. Verdict (Accept or Reject)

**Reject.**
