# R28-TERRA-B

## 1. Concise summary

ForkAudit is a receipt-based auditing protocol for shared-prefix inference in a hybrid attention/GDN LLM stack. It records and replays ownership, lifecycle, storage-range, tail-COW, and Python-call receipts, supplementing output equality with explicit state-isolation checks. On one fixed Qwen3.5-35B-A3B / H20 sequential configuration, it evaluates a 2×2 KV/GDN ownership factorial across 96 configurations, selected numerical sidecars, and targeted fault injections/suppression experiments. The paper deliberately limits its claims to trusted capture of this one stack and excludes compiled-dispatch attestation, concurrency, capacity, and independent recapture.

## 2. Strengths

- The paper is unusually clear about its assurance boundary. It explicitly states that trusted capture is required and does not oversell receipt replay as independent runtime verification.
- The contract is concrete: mandatory record types, ownership predicates, pointer-free interval schema, target-to-record matrix, and decision statuses make the audit operational rather than merely conceptual.
- The experimental design is well organized for its stated narrow goal: four ownership cells, cross-N comparisons, phase-specific state checks, and designed fault-to-gate localization.
- The gate-suppression matrix is a useful demonstration that token equality—and sometimes canonical full-logit equality—can miss particular ownership/contract violations.
- Memory metrics are carefully separated from service capacity and process/NVML memory. The paper consistently avoids inappropriate cross-framework comparisons.
- Limitations are prominent, specific, and credible. Visual presentation, tables, and figures are polished and readable.

## 3. Weaknesses

Must-fix:

- The central correctness conclusion remains substantially circular: the capture producer that may embody the bug is trusted to emit complete and truthful lifecycle/storage/call records. Hashing and offline replay protect archived bytes, but cannot independently establish that observed storage IDs, ranges, calls, or events correspond to actual runtime behavior. This is stated, but the title and “ownership audit” framing should more consistently foreground that this is trace-consistency auditing under a trusted-producer assumption, not an audit of a potentially faulty runtime in the usual independent-observer sense.
- Empirical support for practical/general utility is very narrow: one model, one hardware family, sequential single-stream execution, one partial-tail geometry in the primary result, N≤32, and deterministic single runs per rank/configuration. The separate cancellation extension helps, but does not address concurrent kernels, realistic schedulers, ragged batching, eviction, or multi-document workloads. The contribution should either be positioned as a tightly scoped systems case study or expanded with materially broader validation.
- The mutation evidence is not strong evidence of fault-detection coverage. All nine faults are hand-designed around predeclared gates, and several alternatives are redundant or engineered to preserve outputs. The paper properly disclaims rates, but should provide at least some held-out or naturally occurring regressions, or clearly reduce any implication that the suite establishes broad sensitivity.

Optional improvements:

- The selected independent numerical evidence is limited: eight attention rows and four GDN transitions, with GDN checking beginning after native q/k normalization. More layers, sequence positions, and independently reconstructed upstream inputs would better substantiate the “hybrid stack” framing.
- Reproducibility would be easier to assess with a concise artifact-availability statement in the main text: exact release status, licenses/model access requirements, expected replay time/storage, and which claims can be reproduced CPU-only versus recaptured.
- The extensive unpooled CoMem/HYPIC/related-work context is carefully caveated but distracts from the core audit contribution. A shorter main-paper treatment would sharpen the narrative.
- Terms such as RR2/R28 and the distinction between formal run, capture, replay, wrapper correction, and cohort would benefit from a compact timeline/terminology box.

## 4. Questions for authors

1. What independent mechanism, if any, validates that capture events and storage-range observations faithfully reflect live runtime state rather than a coherent instrumentation/runtime defect?
2. Can the authors evaluate the audit under true concurrent CUDA streams or continuous batching, where ownership bugs and scheduling interleavings are more consequential?
3. Are any of the nine mutations derived from real historical bugs? If not, can the authors add held-out mutations or regressions not used to define the gate map?
4. Why is relative-L2 ≤0.005 an adequate attention-oracle threshold, and how sensitive are conclusions to stricter tolerances or error distributions across more rows?
5. Is the complete artifact package available to reviewers, including raw sidecars and replay scripts? What can an independent reviewer reproduce without the eight-H20 capture environment?

## 5. Reproducibility assessment

Moderate to good for offline verification of the reported bounded results: the PDF specifies manifests, artifact counts, schemas, replay commands, selected raw numerical sidecars, and a detailed artifact map. Reproducibility of the original live capture is weaker because it requires a highly specific Qwen/vLLM/Triton/H20 setup and because replay cannot independently recapture or attest compiled dispatch. Availability of the referenced accompanying package is essential but cannot be verified from the PDF alone.

## 6. Overall score: 6 / 10

A technically careful and transparently scoped systems/auditing paper with a useful concrete protocol and strong documentation. However, its central evidence is conditional on trusted instrumentation, the empirical domain is extremely narrow, and the targeted mutation suite does not establish broad detection efficacy. I would lean weak accept if ICLR values narrowly scoped, artifact-grounded systems methodology; otherwise this needs stronger independent validation and generality.

## 7. Confidence: 4 / 5

