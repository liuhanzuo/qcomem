# ICLR 2026 review-score calibration

This project uses the public ICLR 2026 review form and no continuous or
probabilistic surrogate score.

Primary sources:

- Reviewer Guide: <https://iclr.cc/Conferences/2026/ReviewerGuide>
- Author Guide: <https://iclr.cc/Conferences/2026/AuthorGuide>
- Code of Ethics: <https://iclr.cc/public/CodeOfEthics>

## Allowed fields and values

- Soundness, Presentation, Contribution: integer `1` (poor), `2` (fair),
  `3` (good), or `4` (excellent).
- Overall Rating: exactly one of:
  - `2`: reject, not good enough;
  - `4`: marginally below the acceptance threshold;
  - `6`: marginally above the acceptance threshold;
  - `8`: accept, good paper (poster);
  - `10`: strong accept, should be highlighted.
- Confidence: integer `1` through `5`.

The autonomous review pipeline rejects decimal scores, odd overall scores,
acceptance probabilities, and weighted pseudo-scores.  A simulated score is an
internal revision signal, not a prediction of acceptance.

## Public related-paper anchors

These papers are used to calibrate expected breadth and evidence, not to copy a
numerical score mechanically across ICLR years.

| Paper | Public decision | Calibration use |
|---|---|---|
| [Preble](https://openreview.net/forum?id=meKEKDhdnx) | ICLR 2025 Poster | Prefix reuse at scheduler/distributed-serving scope; mean and tail latency evidence. |
| [SqueezeAttention](https://openreview.net/forum?id=9HK2rHNAhd) | ICLR 2025 Poster | KV-memory method with layer/token budgeting, quality, memory, and throughput evidence. |
| [Palu](https://openreview.net/forum?id=LWMS4pk2vK) | ICLR 2025 Poster | KV compression plus kernels; public final ratings were reported as 6/6/6/5 under the 2025 form. |
| [HeadKV](https://openreview.net/forum?id=FJFVmeXusW) | ICLR 2025 Poster | Long-context quality and task breadth for a cache-management contribution. |
| [Multi-Head Low-Rank Attention](https://openreview.net/forum?id=vBJKZ19XGY) | ICLR 2026 Poster | Current-year evidence around KV traffic, tensor parallelism, quality, and decoding speed. |
| [TurboQuant](https://openreview.net/forum?id=tO3ASKZlok) | ICLR 2026 Poster | Current-year combination of quantization theory and inference evidence. |

## Implication for the current manuscript

The frozen ForkAudit evidence is strong on bounded soundness and auditability,
but materially narrower than the accepted system/method anchors above.  Before
blind review, the conservative prior is:

- Soundness: `3/4`
- Presentation: `3/4` if the paper is clear and visually audited
- Contribution: `2/4`
- Overall: `4/10`
- Confidence: `4/5`

The likely ceiling without new GPU evidence is `6`, not `8`: the paper has one
model/hardware configuration, no real scheduler lifecycle, no optimized
production baseline, no cross-model validation, and no latency/throughput or
quality claim.  Reviewers must nevertheless score the submitted paper from
scratch rather than inherit this prior.
