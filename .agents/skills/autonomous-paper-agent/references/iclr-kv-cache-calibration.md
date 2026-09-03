# ICLR KV-Cache and LLM-Inference Calibration Set

Use this file only for papers about KV-cache management, shared contexts, prefix reuse, long-context inference, or LLM serving. Read the papers and their public OpenReview discussions as qualitative anchors. Do not copy their scores, treat acceptance as proof of correctness, or infer that matching one surface feature deserves an 8.

## Closest Accepted ICLR Anchors

### SCBench: A KV Cache-Centric Analysis of Long-Context Methods

- Venue: ICLR 2025 Poster
- Proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/hash/a540b17fb2295c736d5afd6c507acf66-Abstract-Conference.html
- OpenReview: https://openreview.net/forum?id=gkUyYcY1W9
- Calibration value: shared-context and multi-request framing, full KV-cache lifecycle, breadth across tasks and models, and clear distinction between single-request and reuse settings.

### Palu: KV-Cache Compression with Low-Rank Projection

- Venue: ICLR 2025 Poster
- Proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/hash/7da6e0e00702c60607a6ae05c802ef85-Abstract-Conference.html
- OpenReview: https://openreview.net/forum?id=LWMS4pk2vK
- Calibration value: method novelty, optimized-kernel evidence, accuracy/memory/speed trade-offs, multi-model evaluation, and quantization compatibility.

### SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget

- Venue: ICLR 2025 Poster
- Proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/hash/3b0a8df568ec496a717566a7f8158aaa-Abstract-Conference.html
- OpenReview: https://openreview.net/forum?id=9HK2rHNAhd
- Calibration value: explicit closest-baseline comparison, memory and throughput evaluation, multiple models and benchmarks, and decomposition of the design dimensions.

### Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning

- Venue: ICLR 2025 Poster
- Proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/hash/f649556471416b35e60ae0de7c1e3619-Abstract-Conference.html
- OpenReview: https://openreview.net/forum?id=FJFVmeXusW
- Calibration value: head-level contribution boundary, retrieval/reasoning evaluation, strong baselines, and quality preservation under tight cache budgets.

### RazorAttention: Efficient KV Cache Compression Through Retrieval Heads

- Venue: ICLR 2025 Poster
- Proceedings: https://proceedings.iclr.cc/paper_files/paper/2025/hash/2a98af4fea6a24b73af7b588ca95f755-Abstract-Conference.html
- Calibration value: training-free cache design, information-preservation argument, FlashAttention compatibility, broad model evaluation, and explicit system-efficiency claims.

### Efficient Streaming Language Models with Attention Sinks

- Venue: ICLR 2024 Poster
- OpenReview: https://openreview.net/forum?id=NG7sS51zVF
- Paper: https://openreview.net/pdf?id=NG7sS51zVF
- Calibration value: a sharply defined phenomenon, mechanism-to-system connection, multiple model families, very long sequence evaluation, and an end-to-end speed claim matched to evidence.

## Pairwise Calibration Questions

For the manuscript under review, compare against the anchors on:

1. Is the paper proposing a mechanism, an implementation, an evaluation protocol, an empirical finding, or an artifact? Is that contribution type explicit?
2. Is the closest baseline a real competing system, an ablation, or only a causal control?
3. Does the evaluation vary models, workloads, context geometry, precisions, request lifecycles, and quality metrics in proportion to the breadth of the claim?
4. Are memory quantities analytical storage, framework allocator counters, or process/device measurements? Are they kept separate?
5. Are speed, throughput, scheduler, and deployment claims supported by end-to-end measurements?
6. Does exactness cover only tokens, or also logits and internal state where relevant?
7. Does the paper discover new knowledge beyond a formula implied directly by allocation policy?
8. Could an ICLR reader reproduce the central result from the paper and artifact?

## Calibration Discipline

- A narrow but rigorous artifact may be Soundness 4 and Contribution 2; do not inflate Contribution because the audit is meticulous.
- A broad result with weak provenance may have high apparent significance but low Soundness.
- An accepted anchor is not a minimum checklist. Use it to identify the evidence gap created by the manuscript's own claims.
- Do not demand throughput, downstream quality, or multiple models when the manuscript explicitly makes no such claim unless their absence leaves no meaningful ICLR contribution.
- Conversely, a paper claiming practical serving value must include realistic lifecycle and end-to-end evidence comparable in spirit to the closest anchors.

