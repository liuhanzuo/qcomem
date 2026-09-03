# A14 — split depth by document length, measured

Date: 2026-09-03
Action: A14 (`experiment_required`), also bearing on A13 and on the Section 5.6
limitation that the paper never justified `j = 7`.
Producing runs: QS Trial 1941241 (`qcomem-r45-deepj-longdoc-20260902a`), two
sweeps, exit 0 on both, eight rank shards each.
Extraction: QS Trial 1942890, 288 rows.

## Status: admissible as a mechanism/cost result, NOT as a quality result

This is a synthetic-document harness. Documents are a repeated technical
paragraph produced by `repeated_document`, the query is a fixed 13-token
question, and `measure_config` generates exactly one token, so what is measured
is reconstruction cost, not answer quality and not throughput. No F1 exists here
and none may be inferred.

The numbers below are therefore admissible for statements about how retained
state and reconstruction time move with split depth and document length. They
are **not** admissible for any claim about answer quality at a given depth, and
they cannot by themselves select the paper's operating point. The LongBench
sweep (Trial 1943158, extraction pending) is what closes that gap.

## Method

`run_capacity_scaling_r45.py`, a thin wrapper that makes depth and document
length command-line arguments. It imports `measure_config` and
`repeated_document` unchanged from `run_capacity_scaling.py`, so a row here is
directly comparable to a row from the published `extreme`/`quality` suites. One
document length per rank, eight ranks, three repeats per configuration, medians
reported. Bit widths held fixed at residual Q4 / attention Q8 / linear Q8 across
every depth, so depth is the only variable that moves within a column.

## Result 1: deeper splits are monotonically faster, at every length

Generation time as a fraction of the dense baseline at the same length (lower is
better):

| `j` | layers skipped | 4k | 8k | 16k | 32k |
|---:|---:|---:|---:|---:|---:|
| 7 | 17.5% | 0.938 | 0.920 | 0.865 | 0.850 |
| 13 | 32.5% | 0.813 | 0.779 | 0.727 | 0.702 |
| 20 | 50.0% | 0.661 | 0.620 | 0.553 | 0.527 |
| 26 | 65.0% | 0.522 | 0.455 | 0.397 | 0.370 |
| 30 | 75.0% | 0.433 | 0.358 | 0.297 | 0.268 |
| 33 | 82.5% | 0.367 | 0.287 | 0.226 | 0.196 |
| 36 | 90.0% | 0.307 | 0.219 | 0.156 | 0.125 |

Both directions help and they compound. The published operating point `j = 7`
is the worst row in the table at every length, and at 4,096 tokens it is within
6% of dense, which is the regime in which the deployment bench found Q-CoMem to
be slightly slower than an honest dense arm.

## Result 2: the cost is retained state, and the trade is continuous

At 32,768 tokens, against a full-prefix baseline of 701.88 MiB/document and a
dense baseline of 5.8997 s:

| `j` | generation | vs dense | Store (MiB/doc) | vs full-prefix |
|---:|---:|---:|---:|---:|
| 7 | 5.017 s | 0.850 | 73.39 | 0.105 |
| 13 | 4.139 s | 0.702 | 143.64 | 0.205 |
| 20 | 3.109 s | 0.527 | 214.47 | 0.306 |
| 26 | 2.181 s | 0.370 | 251.29 | 0.358 |
| 30 | 1.581 s | 0.268 | 286.98 | 0.409 |
| 33 | 1.156 s | 0.196 | 322.11 | 0.459 |
| 36 | 0.735 s | 0.125 | 357.24 | 0.509 |

`j` interpolates continuously between the two endpoints the Motivation section
already names: at `j -> 0` the method approaches dense recomputation, retaining
nothing and recomputing everything; at `j -> L` it approaches exact prefix
caching, retaining everything and recomputing nothing. Every row beats dense on
time and full-prefix on retained state, at every length. That is a weaker
statement than it sounds, because it is what interpolation between two endpoints
guarantees; the substantive question is where on the curve the quality holds,
and this harness cannot answer it.

## What this changes, and what it does not

It supplies the justification for `j` that Section 5.6 currently admits is
missing, and it shows the published choice is the least favourable point on the
curve for the latency argument. It does not license moving the operating point
on its own: deeper splits quantize more layers of KV and recurrent state, and
the A4/A5 sweep already showed one case where widening quantization coverage
destroys quality outright (full-prefix Q4 reaching F1 0.000-0.005 while every
other arm stayed near 0.34). A deep `j` could be a prefix cache wearing
Q-CoMem's name, and only a real-protocol quality measurement can distinguish
those.

## Caveat carried forward

The full-prefix arm's generation time differed substantially between the two
published suites in the earlier capacity run (1.407 s versus 0.224 s at the same
4,096 tokens), which is most likely warm-up. Timing from this harness should be
read as within-column comparisons at a fixed length, not as absolute latencies,
until that variance is characterised.

## Registered follow-up

Trial 1943158 runs `d7/d13/d20/d26/d33` on real LongBench items under
`--eos-policy stop` and `--generation-limit-policy dataset`, with warm-ups,
three repeats and seed 20260903, so its F1 is a quality measurement. It also
re-measures the dense-versus-Q-CoMem comparison at 4,096 tokens inside the
deployment harness, which is what resolves the sign disagreement between the two
harnesses. No manuscript statement about the operating point or about latency
may be written before it is harvested.
