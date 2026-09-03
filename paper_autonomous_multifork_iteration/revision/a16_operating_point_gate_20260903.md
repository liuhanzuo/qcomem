# A16 — operating-point gate, 60 items, decided

Date: 2026-09-03
Producing run: QS Trial 1943447 (seven ranks) completed by Trial 1943737 after
the `dense_semantics_gate` contract fix; 900 rows = 5 configurations x 60 items
x 3 repeats, all eight shards `completed`.
Extraction: QS Trial 1943780.

## Protocol

The full 60-item validation cohort, Qasper and 2WikiMQA source indices 6--35,
real LongBench protocol with `--eos-policy stop` and
`--generation-limit-policy dataset`, 4,096-token input cap, one warm-up, three
repeats, seed 20260903. Item F1 is averaged over repeats, then differences are
taken pairwise against `full-prefix-q16` on the 60 common items, with 10,000
paired item-level bootstrap resamples at seed 20260903 and percentile intervals.

## Result

| config | Store (MiB/doc) | TTFT (s) | TPOT (ms) | F1 (0--100) | dF1 vs full-prefix | 95% CI | resident docs |
|---|---:|---:|---:|---:|---:|---|---:|
| `dense-prefill-once` | 0.000 | 0.6358 | 54.42 | 54.180 | -0.4881 | [-4.566, 2.551] | n/a |
| `full-prefix-q16` | 136.235 | 0.1811 | 56.00 | 54.668 | 0 | -- | 540.1 |
| `full-prefix-q8` | 56.438 | 0.1819 | 54.45 | 50.885 | **-3.7831** | [-9.121, 0.321] | 1314.7 |
| `qcomem-d13` | 16.101 | 0.5815 | 54.95 | 51.273 | **-3.3952** | [-8.668, 0.516] | 4595.6 |
| `qcomem-d7` | 9.661 | 0.6564 | 54.89 | 54.297 | **-0.3708** | [-1.969, 1.079] | 7659.7 |

## The gate's verdict: j = 7 stands

A16 was registered in advance with an explicit decision rule: move the operating
point to `d13` only if `d13` holds full-prefix quality on 60 items, otherwise
`j = 7` stands. **It does not hold.** `d13` loses 3.40 F1 points against
full-prefix, roughly nine times `d7`'s 0.37, and the two intervals barely
overlap at their edges.

The eight-item sweep had shown `d7` and `d13` both at 0.340 and was read as a
quality plateau through `d13` with a cliff at `d20`. That reading does not
survive the larger cohort. The earlier note flagged exactly this risk --- eight
items, no intervals, and an n=32 column in which every arm coincided at 0.464 ---
and recorded the finding as a signal rather than an effect. It was a signal, and
it was wrong. The operating point never moved, no manuscript text was written
against it, and nothing needs to be retracted.

## The result that actually matters: the missing baseline resolves in our favour

The panel's most-cited critical gap was the absence of a quantized exact-cache
baseline, with the reviewers' arithmetic suggesting the honest advantage would
collapse to roughly 3.5--3.9x once one existed. Author-side planning had
projected about 3.1x. All of those were projections, and the standing
admissibility rule kept every one of them out of the manuscript.

Measured, `full-prefix-q8` costs **3.78 F1 points** against `full-prefix-q16`.
Quantizing an exact cache to Q8 is not free, which the earlier eight-item run
could not see because `q8` and `q16` both scored 0.340 there. So `qcomem-d7`
does not merely trade capacity for quality against the strongest quantized exact
cache --- it wins on both axes at once:

- **5.84x smaller retained state** (9.661 versus 56.438 MiB/document)
- **5.83x more resident documents** (7659.7 versus 1314.7)
- **3.41 F1 points better** (54.297 versus 50.885)

Against `full-prefix-q16` it is 14.10x smaller, holds 14.18x more documents, and
its F1 difference of -0.37 [-1.97, 1.08] is indistinguishable from zero. This is
the same -0.45 [-2.06, 0.99] the archival cohort gives under the same reference
arm, measured independently on a fresh run.

## What did not change

Latency. `qcomem-d7` TTFT is 0.6564 s against `full-prefix-q16`'s 0.1811 s, a
3.6x deficit, and `dense-prefill-once` at 0.6358 s remains marginally ahead of
it. The deployment bench has now placed Q-CoMem behind honest dense
recomputation three times on real items. No latency advantage may be claimed,
and the capacity-first framing the manuscript already uses is the correct one.

TPOT is flat across every arm at 54.4--56.0 ms, so the cost is confined to
reconstruction and does not touch steady-state decoding.

## Admissibility

Every figure here is a direct measurement from a completed run under a
preregistered protocol with the seed recorded, so all of it is admissible. The
projections it replaces --- the 3.1x planning figure, the reviewers' 3.5--3.9x,
and the eight-item quality plateau --- are now superseded and must not appear.

## Manuscript consequences

1. The quantized exact-cache baseline can be reported for the first time, and it
   strengthens rather than weakens the contribution. This closes the panel's
   Q1/R44-4-02 critical gap.
2. `j = 7` keeps its place, and Section 5.6 can now say the depth was validated
   against `d13` on the full cohort rather than admitting it was never justified.
3. The `full-prefix-q8` quality cost is a new result and belongs in Table 2.
4. No latency claim. No change to the ForkAudit scope statements.

## Correction added on review of the instrument: residency is not an independent axis

`capacity_estimate` computes

    max_resident_documents_store_only
      = (total_device_bytes - model_allocated_bytes - safety_headroom_bytes)
        // persistent_document_bytes

That is a measured budget divided by a measured per-document payload. It is an
analytic corollary of Store under an explicit budget model, **not** a
demonstration that any number of documents was held resident simultaneously. It
assumes retained tensor payload is the only quantity scaling with document
count, and therefore excludes allocator fragmentation, per-document non-tensor
metadata, and any pool behaviour --- the same exclusions Section 5.6 already
lists when it says reducing Store does not by itself prove serving capacity.

The consequence for the manuscript is a reporting rule, not a retraction. The
14.18x residency ratio and the 14.10x Store ratio are the same measurement
expressed twice; presenting them side by side as two results would inflate one
finding into two. Residency may be used to make the capacity claim concrete
under a named budget model, and must be labelled as derived from Store rather
than measured alongside it.

This also means **A13 is not closed by this run.** Eq. 1's question --- whether
the retained-byte reduction actually converts into more documents held resident
--- still requires an admission experiment that increases the resident set until
failure and observes where it fails. Marking A13 resolved on a division would be
exactly the substitution the provenance contract forbids.
