## 1. Summary and claimed contributions

ForkAudit is a trace-validation protocol for hybrid LLM inference that combines shared attention KV state with mutable recurrent GDN state. It defines phase-indexed ownership, lifecycle, storage-range, copy-on-write, dispatch, and semantic receipts; replay is fail-closed for missing mandatory records. On a fixed Qwen3.5/H20 implementation, it reports cross-arm equality, storage isolation at a registered transition, selected captured-boundary numerical checks, designed-fault checks, and separated memory denominators. Its assurance is explicitly conditional on an honest, mandatory-event-complete capture producer.

## 2. Strengths

- The paper is unusually clear about what its evidence does and does not establish. Separating trace coverage from replay verdicts, and labeling dispatch coverage partial, is commendable.
- The phase-aware contract addresses a real gap in output-only validation: output equality cannot demonstrate non-aliasing of mutable recurrent state.
- The schema is technically concrete: it specifies lifecycle phases, mandatory records, conservative byte-range overlap rules, COW requirements, and target-to-record dependencies.
- The fixed-stack evaluation is carefully controlled. The ownership factorial, selected numerical sidecars, and all-gates-on versus gate-suppression controls provide useful evidence that the stated checks can catch their intended injected violations.
- Memory accounting is responsibly scoped; logical state, allocator deltas, and service capacity are not conflated.

## 3. Weaknesses

- The central ownership conclusion depends on the same capture producer that observes and serializes the candidate state. Since storage identifiers and phase labels originate there, replay can detect malformed or inconsistent traces but cannot independently establish that the runtime did not omit, mislabel, or transiently mutate state. The paper states this limitation, but it materially limits the strength of the main contribution as a validator.
- The empirical evidence is extremely narrow: one model family, one hardware family, a short fixed continuation, limited fan-outs, sequential batch-one calls, and largely deterministic observations rather than replicated measurements. The supporting cohorts do not close the gap to native ragged/continuous batching, production scheduling, or in-flight cancellation.
- The numerical references begin from candidate-produced intermediate tensors, and recurrent coverage remains partial. Thus they provide bounded operator checks rather than strong end-to-end independent validation.
- Novelty is mainly an integration and instrumentation protocol over established ownership, metamorphic-testing, replay, and cache-management concepts. The paper does not convincingly show that this protocol transfers across materially different hybrid runtimes or that it discovers non-designed defects.
- The nine positive controls are targeted to named gates. Their success demonstrates expected sensitivity, but cannot estimate fault coverage, false positives, or practical bug-finding value.
- The manuscript is admirably cautious but overly dense and repetitive; extensive caveats, cohort distinctions, and auxiliary tables obscure the central novelty and result.

## 4. Questions for the authors

- Can a replay-only third party validate that two normalized storage IDs correspond to distinct actual backing allocations without trusting producer-assigned identity labels? What concrete mechanism would strengthen this beyond the current honest-capture assumption?
- Please evaluate at least one substantially different hybrid runtime/model and a native continuous or ragged batching setting, including longer and varied transition/tail geometries.
- Can you provide a blind or naturally occurring fault study, plus false-positive analysis, rather than only gate-targeted injected controls?
- Is the full replay package publicly available to reviewers, including the environment specification and sufficient raw evidence to reproduce all central tables without the original producer?

## 5. Reproducibility and ethics

The protocol description, schemas, tables, artifact map, and stated replay boundaries support partial reproducibility. However, the PDF alone does not provide a directly accessible artifact or independent recapture path, and major conclusions remain dependent on the producer and pinned environment. Ethically, the work has limited direct human-subject risk, but it uses PG-19 training-split books without discussing licensing/data-governance considerations and does not report the resource cost of its substantial H20 experimentation.

## 6. Overall score

4 — Weak reject. This is a careful and potentially useful systems-validation methodology with unusually honest claim boundaries, but the trusted-producer dependency, narrow fixed-stack evidence, and limited demonstrated practical bug-finding/generalization leave the contribution insufficiently validated for acceptance.

## 7. Confidence

4 — The paper is detailed, internally coherent, and explicit about its limitations; the remaining uncertainty concerns the unavailable artifacts and how the protocol performs beyond the reported frozen setting.

## 8. Verdict

Reject
