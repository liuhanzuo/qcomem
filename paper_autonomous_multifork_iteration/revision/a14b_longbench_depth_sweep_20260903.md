# A14b — split depth on real LongBench items

Date: 2026-09-03
Producing run: QS Trial 1943158 (`qcomem-r45-jsweep-longbench-20260903b`),
`failed=0`, eight shards, 17:31:54--17:35:17 UTC 2026-09-02.
Extraction: QS Trial 1943289, 384 rows.
Predecessor: Trial 1943130, which failed on all eight ranks at argument parsing
before any scientific execution and is recorded as an infrastructure failure, not
evidence.

## Protocol

Real LongBench items, source indices 6--9 of Qasper and 2WikiMQA (eight items),
`--eos-policy stop` and `--generation-limit-policy dataset`, so generation ends
naturally and F1 is a quality measurement rather than the fixed-length sanity
signal the A4/A5 sweep produced. One warm-up, three repeats, seed 20260903,
group size 64, 4,096-token input cap. Configurations `dense-prefill-once`,
`full-prefix-q16`, `full-prefix-q8`, and `qcomem-d{7,13,20,26,33}-r4-a4-l8`, at
requested generation lengths 8, 32 and 128.

**Sample size is the governing limitation.** Eight items. The n=8 cells carry 24
rows and the n=32 and n=128 cells carry 12 each. No interval is computed here.
Every F1 difference below is a signal to be confirmed on the 60-item cohort, not
an established effect, and nothing in this note may be written into the
manuscript as a quality result without that confirmation.

## Result 1: the harness disagreement resolves against a latency win at j=7

The deployment bench again places Q-CoMem at `j = 7` slightly *behind* an honest
dense baseline, now on real items: TTFT 0.6682 s versus 0.6549 s at n=8, a 2.0%
deficit. This reproduces the A4 result and contradicts the synthetic capacity
harness, which had `j = 7` 6% ahead of dense at 4,096 tokens.

The two harnesses differ in query length (13 synthetic tokens versus real
LongBench queries), in document composition, and in bit widths (`a8` versus
`a4`), so they are not measuring the same thing and the sign flip is not a
defect in either. The defensible conclusion is the one both support: **at
`j = 7` and roughly 4k documents, Q-CoMem and honest dense recomputation are
within a few percent of each other, and no latency advantage may be claimed at
that operating point.** The manuscript must not cite the synthetic harness for a
latency comparison.

## Result 2: TTFT falls steeply with depth, on real items

TTFT at n=8, against dense 0.6549 s and full-prefix Q16 0.1713 s:

| config | TTFT (s) | vs dense | Store (MiB/doc) | F1 @ n=8 | F1 @ n=128 |
|---|---:|---:|---:|---:|---:|
| `qcomem-d7` | 0.6682 | 1.020 | 9.901 | 0.340 | 0.312 |
| `qcomem-d13` | 0.5824 | 0.889 | 16.501 | 0.340 | 0.316 |
| `qcomem-d20` | 0.4757 | 0.726 | 23.666 | 0.313 | 0.254 |
| `qcomem-d26` | 0.3791 | 0.579 | 28.660 | 0.313 | 0.254 |
| `qcomem-d33` | 0.3177 | 0.485 | 35.825 | 0.313 | 0.254 |
| `full-prefix-q8` | 0.1714 | 0.262 | 57.949 | 0.340 | 0.309 |
| `full-prefix-q16` | 0.1713 | 0.262 | 139.080 | 0.340 | 0.318 |
| `dense-prefill-once` | 0.6549 | 1.000 | 0 | 0.354 | 0.359 |

## Result 3: quality appears to fall off a cliff between d13 and d20

`d7` and `d13` sit at the full-prefix F1 (0.340 at n=8; 0.316 and 0.312 versus
full-prefix's 0.318 at n=128). From `d20` onward every depth drops to 0.313 at
n=8 and 0.254 at n=128, and does not fall further with additional depth.

The flatness of the degraded value across `d20`, `d26` and `d33` is what makes
this look like a threshold rather than noise: three independent configurations
landing on exactly the same F1 is more consistent with a shared failure mode than
with sampling variation. It is nevertheless eight items, the identical n=32
column (every arm at 0.464) shows the cohort is small enough for arms to
coincide by accident, and no interval was computed. Treat it as a strong signal
requiring confirmation.

The mechanism is plausible and was anticipated: deeper splits quantize more
layers of attention KV and recurrent state at Q4/Q8, and the A4/A5 sweep already
produced one case where widening quantization coverage destroyed quality
outright, full-prefix Q4 reaching F1 0.000--0.005.

## Result 4: residency *decreases* with depth --- correcting an earlier reading

Maximum resident documents, measured, at n=128:

| config | store only | with one active request | decode KV peak (MiB) |
|---|---:|---:|---:|
| `qcomem-d7` | 7496.0 | 7405.7 | 119.7 |
| `qcomem-d13` | 4497.5 | 4439.9 | 96.3 |
| `qcomem-d20` | 3134.0 | 3093.8 | 71.1 |
| `qcomem-d26` | 2585.2 | 2555.5 | 53.2 |
| `qcomem-d33` | 2068.0 | 2041.4 | 27.8 |
| `full-prefix-q8` | 1282.2 | 1277.9 | 2.3 |
| `full-prefix-q16` | 531.8 | 529.9 | 2.3 |

An earlier author-side reading treated "deeper is better" as unqualified. It is
not. Depth buys reconstruction speed and *costs* retained state, so residency
falls monotonically as `j` grows: from 7496 documents at `j = 7` to 2068 at
`j = 33`. Depth trades the paper's strongest measured axis against its weakest.

Note also that Q-CoMem's decode-time KV peak is one to two orders of magnitude
above full-prefix's, because the suffix is reconstructed per request. Residency
here is measured with a single active request; it says nothing about how many
requests can be served concurrently, and the manuscript must not conflate the
two.

## What this selects

`d13` is the only depth that is simultaneously (i) at full-prefix quality on both
generation lengths, (ii) faster than honest dense recomputation, and (iii) far
ahead of every exact-cache arm on retained state and residency:

- versus dense: 11.1% lower TTFT, at a cost of 16.5 MiB/document
- versus `full-prefix-q8`: 3.51x smaller store, 3.51x more resident documents,
  equal or better F1 (0.340 vs 0.340 at n=8, 0.316 vs 0.309 at n=128), but 3.4x
  higher TTFT
- versus `full-prefix-q16`: 8.43x smaller store, 8.46x more resident documents,
  equal F1 at n=8 and 0.316 vs 0.318 at n=128

`d7`, the published operating point, is dominated by `d13` on every axis except
retained state and residency, where it leads by 1.67x, and it buys that lead by
giving up the latency argument entirely.

## Decision and its gate

The operating point is **not** moved on this evidence alone. The quality
comparison that would justify moving it rests on eight items with no intervals,
and the manuscript's existing quality claims rest on the 60-item cohort. Moving
the headline on a strictly weaker cohort would be exactly the substitution the
provenance contract forbids.

Registered next step: run `d7` and `d13` on the full 60-item validation cohort
under the archival protocol, with paired item-level bootstrap intervals against
full-prefix, seed recorded. If `d13` holds full-prefix quality there, the
operating point moves to `d13` and the latency claim becomes admissible for the
first time. If it does not, `j = 7` stands and the paper keeps its current
capacity-only framing.
