## 1. Summary and claimed contributions

This paper presents ForkAudit, a trusted-capture, offline trace-validation protocol for verifying ownership and lifecycle properties when a shared-prefix hybrid LLM forks requests. Its case study combines paged Transformer KV state with mutable GDN recurrent state. The protocol records typed, phase-indexed identity, storage, lifecycle, dispatch, and semantic receipts, then replays predicates for prefix immutability, private ownership, tail copy-on-write, and cross-arm/cross-fan-out equivalence.

On a fixed Qwen3.5-35B-A3B/H20 stack, the paper reports exact equality across 96 ownership configurations and 288 cross-N comparisons; selected captured-boundary attention/GDN numerical checks; nine designed faults that reach prespecified gates; and memory reductions from shared KV. The paper explicitly limits its conclusion to registered trace predicates under an honest, mandatory-event-complete capture producer, and disclaims independent recapture, compiled-kernel attestation, fault coverage, serving performance, and end-to-end correctness.

## 2. Strengths

- The assurance boundary is unusually explicit. The paper carefully distinguishes complete trace coverage from a replay pass, relational output checks from ownership checks, and captured-boundary numerical validation from end-to-end validation.
- The phase-indexed schema is technically clear and well motivated for hybrid state: immutable document KV, private append state, persistent recurrent bases, transition-time rebinding, and tail COW are all concrete obligations rather than vague “cache correctness.”
- The pointer-free normalized storage-range witness is a thoughtful design that makes the claimed storage predicates portable and replayable without relying on raw addresses.
- The paper performs more than output comparison. In particular, the target-gate suppression study usefully demonstrates that four designed violations can preserve both generated tokens and canonical logits over the tested horizon while being caught by trace gates.
- Evaluation reporting is careful about denominators and unpooled cohorts. The distinction between final allocation delta, allocator peaks, logical payload, and unmeasured service capacity is commendable.
- Limitations are not buried: the authors explicitly acknowledge the shared-process observer limitation, absent per-call compiled-dispatch binding, narrow numerical-oracle boundary, sequential schedule, and lack of fault-rate interpretation.

## 3. Weaknesses

- The central limitation is fundamental rather than incidental: the validator trusts the capture producer to faithfully emit all mandatory events. Consequently, replay establishes consistency of a producer-supplied trace, not that the live execution actually satisfied the predicates. Coherent omissions/fabrication, transient mutation restored between captures, and unobserved compiled behavior remain invisible. This substantially limits the practical meaning of “trusted-capture validation” for the ownership failures the system is intended to assure.
- The main evaluation is extremely narrow: one model/adapter/hardware family, one partial-tail geometry, one document/query length, eight generated tokens, \(N\leq32\), and sequential batch-one calls on a single CUDA stream. The paper discloses this, but it still leaves unclear whether ForkAudit is usable for the dynamic ragged batching, eviction, multi-document, and concurrency patterns that motivate prefix sharing in practice.
- The empirical checks are largely self-consistency checks within the same implementation. Materialized/shared arms, cross-N equality, and source-distinct observation share important runtime objects, labels, storage API, and much of the execution stack. The selected NumPy oracles improve confidence at their boundaries, but begin from producer-captured post-RoPE or post-normalization intermediates and cover only 12/30 recurrent layers in the expanded sweep. They do not resolve common-mode implementation errors upstream or end-to-end.
- The nine mutations are targeted positive controls designed around the proposed gates. Their successful localization demonstrates expected sensitivity, but gives little evidence on naturally occurring bugs, false positives, interactions between faults, or detection beyond the hand-designed taxonomy. The M8 sentinel is explicitly not a production detector.
- Novelty appears principally to be an integration and disciplined reporting framework. The paper itself attributes paging, prefix reuse, COW, hashing, metamorphic relations, and storage checks to prior work. The submission needs a sharper argument—and preferably comparison to a credible alternative auditing approach—that its integrated schema yields materially stronger or more economical assurance than well-engineered invariant logging plus existing testing.
- The paper is overextended. Large unpooled Mac, deployment, HYPIC, Hydragen/Palu, and policy-simulator sections consume substantial space without strengthening the primary ForkAudit claim. This makes the contribution harder to evaluate and does not compensate for the limited main study.
- The memory result is easy to overread despite careful caveats: the prominent final allocated-delta reduction at \(N=32\) coexists with much smaller full-lifecycle allocated-peak differences, retained-source controls, and no capacity/NVML measurement. The paper should foreground the operationally relevant limitation more strongly.

## 4. Questions for the authors

- What deployment threat model makes an honest, mandatory-event-complete producer a realistic trust anchor? Can capture be moved beneath, or independently checked against, the candidate runtime so that a faulty implementation cannot simply omit or fabricate receipts?
- What is the incremental assurance or cost benefit of ForkAudit relative to a baseline composed of standard lifecycle assertions, storage-range checks, and artifact hashing? A direct comparison of bugs caught, false alarms, engineering effort, and overhead would clarify the contribution.
- Can the protocol be exercised under native continuous/ragged batching, eviction, multiple shared documents, and genuinely in-flight cancellation? Which schema predicates would require redesign rather than merely additional experiments?
- Why is the recurrent state checked at one registered 32-token transition boundary? How should a practitioner select transition points and ensure coverage for arbitrary query lengths or recurrent updates?
- Can the authors provide an external or lower-level dispatch witness—e.g., kernel/binary/autotuning provenance—or explain why Python-call provenance is sufficient for intended users?
- How should readers interpret the allocator gains for production planning given the retained-source control and the lack of process-level or admitted-capacity measurements?

## 5. Reproducibility and ethics

The paper gives an unusually detailed artifact/claim map, manifest-first replay description, typed record schema, and explicit package-size/replay-cost information. This supports reproducibility of the reported trace replay, assuming the accompanying artifacts are released and the pinned environment remains usable. However, replay cannot independently reproduce or validate the original live capture; several supporting summaries also lack raw trials. Reproducibility of the core *claim* is therefore conditional on the trusted-capture assumption, not merely on artifact availability.

Ethical risk appears low: no human-subject study is reported, and the workload uses PG-19. The ethics statement appropriately notes that it does not evaluate model safety, bias, privacy, or deployment impacts. It would nevertheless benefit from addressing data licensing/provenance and the possibility that more efficient serving can amplify downstream model harms.

## 6. Overall score (integer 1–10, using ICLR-style meaning; state score and concise justification)

**4 — Borderline reject.** The paper is exceptionally transparent and technically careful, with a useful ownership-trace framework and well-documented bounded experiments. However, the trusted-producer assumption undercuts the main assurance claim, and the evidence is too narrow and self-referential to establish a broadly useful validation method for practical hybrid-LLM serving.

## 7. Confidence (integer 1–5; state confidence and concise justification)

**4 — High confidence.** The submission clearly specifies its protocol, experimental scope, artifacts, and limitations, allowing a well-grounded assessment. My uncertainty is mainly about the novelty and practical value that an independently accessible implementation and broader deployment evaluation could clarify.

## 8. Verdict (Accept or Reject)

**Reject**
