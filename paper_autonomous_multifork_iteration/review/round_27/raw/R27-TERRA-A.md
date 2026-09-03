# R27-TERRA-A

Access mode: PDF only. Model: `gpt-5.6-terra`.

## Summary

ForkAudit proposes a receipt-based audit protocol for verifying ownership/isolation of shared token KV caches and mutable GDN recurrent states after prefix forking. On one Qwen3.5 hybrid runtime, it evaluates a 2x2 KV/GDN ownership factorial across fan-outs, receipt-replays lifecycle/storage predicates, runs selected FP32/NumPy operator checks, and injects nine targeted faults. It carefully separates these bounded audit claims from performance and capacity claims.

## Strengths

- The paper identifies a real and under-examined systems-correctness problem: output agreement alone cannot establish isolation of mutable hybrid-model state.
- The ownership contract is unusually explicit: roles, phases, storage-range normalization, tail COW, and claim boundaries are specified clearly.
- Experimental disclosure is commendably disciplined. The paper repeatedly distinguishes receipt-level evidence, selected operator corroboration, positive controls, and non-comparable context.
- The factorial design, fixed fan-outs, lifecycle checks, and targeted mutations give useful diagnostic coverage for the stated frozen configuration.
- Memory accounting is substantially more careful than typical serving papers; it distinguishes logical state, allocator counters, and service-capacity claims.

## Weaknesses

Must fix:

- The central receipt-complete evidence remains conditional on an honest producer and self-captured records. The paper does not offer independent recapture, trusted instrumentation, or an end-to-end independently implemented check, so the practical assurance improvement over a carefully logged internal test suite is not convincingly quantified.
- External validity is very limited: one model/configuration, one page size and partial-tail geometry, one sequential one-stream schedule, N<=32, and no concurrent kernels, continuous/ragged batching, eviction, or capacity study. This is especially consequential for a method intended to audit serving behavior.
- The numerical checks cover only selected captured-input boundaries. They cannot validate upstream activations, full-model semantics, all attention/GDN layers, or compiled dispatch; the missing per-call compiled-binary/autotuning binding is a material remaining gap.
- Novelty is primarily an integration and formalization of known techniques. The paper needs a sharper comparison against existing testing/auditing frameworks and evidence that the integrated contract discovers or prevents failures that a strong conventional test methodology would not.

Optional improvements:

- Reduce repetition and expand adoption cost, implementation burden, and audit overhead.
- Add an ablation showing which predicates/faults output equality, digest-only checks, or conventional assertions miss.
- Report audit overhead and artifact size and make external byte bindings concrete.
- Shorten the unpooled systems-context material.

## Questions

1. What concrete trust mechanism makes producer capture more credible than runtime-generated logs?
2. How often does ForkAudit reveal a real defect missed by standard unit/integration tests?
3. What is the runtime, memory, and engineering overhead of collecting receipts at realistic serving scale?
4. Can the contract cover concurrent kernels, continuous batching, eviction, and multiple shared documents?
5. Why is the fixed 0.005 operator tolerance discriminative for the BF16/fused-attention setting?
6. Which artifact components let a third party re-run capture rather than only replay author-produced hashes?

## Reproducibility

Moderate. The PDF describes a detailed manifest-first replay package, explicit artifacts, frozen configurations, and replayable numerical sidecars. Most evidence is digest-based and replay does not independently recapture the producer, so receipt checks appear reproducible while independent execution validation remains incomplete.

## Rating

Overall: **4/10** (borderline reject). Confidence: **4/5**.
