# C1 — the composition experiment, measured

Date: 2026-09-03
Runs: gate QS Trial 1944157 (8/8 ranks passed); formal QS Trial 1944486, exit 0,
288 rows across 8 shards, all `completed`; extractions Trials 1944868 and
1945503.
Protocol: one packed depth-7 Q4/Q4/Q8 entry, fanouts N in {1,2,4,8}, eight
LongBench items at source indices 6--9, 32 generated tokens, cross-item queries,
one warm-up, three repeats, seed 20260903, strict accounting.

## What C1 was for

Four of five round-46 reviewers independently found that the paper's two halves
are each measured where the other is absent: the Read path behind the headline
tables materializes a full private copy per request and shares nothing, while
every ForkAudit verdict comes from a full-prefix BF16 stack with no split and no
quantization. The composed system had no end-to-end evidence, and the
meta-review judged this the only change able to lift Contribution above 2.

## Result 1: the composed system runs, and the audit holds on it

The gate reports `shared_mode_effective: true`, `share_mode_effective:
shared-packed-view`, `fallback_reason: null`, `non_vacuous_sharing: true`, and
`sharing_window: final` -- sharing persists to the end of the request rather
than collapsing at the first append, so the strong policy was exercised, not the
narrowed fallback. Its stated semantic is that shared mode took effect, that
sharing is non-vacuous at the policy's window, that the N>1 shared run is
token-identical to the published N=1 private-materialization path, and that
every applicable contract target is covered and passing. Agreement with the
full-prefix arm is recorded as a diagnostic and never gates, which is the
correct treatment of the cross-chunk-boundary sensitivity established earlier.

This is the first end-to-end evidence for the system the paper describes.

## Result 2: the sharing is real but small

Per-request ownership at the sharing window, from the gate record: of 16 tensors
per request, 2 are shared and 14 are private; of 12.938 MiB per request, 0.500
MiB is shared and 12.438 MiB is private.

So sharing is non-vacuous, audited, and semantically exact -- and it covers
**about 4 percent of per-request state**. The honest statement is that the
ownership discipline is now demonstrated on the packed path, not that sharing
substantially reduces per-request memory.

## Result 3: transient memory contradicts an adjudication that favoured the paper

Peak transient allocation, medians over the run, in MiB:

| N | full-prefix | Q-CoMem shared | Q-CoMem private |
|---:|---:|---:|---:|
| 1 | 286.44 | 930.91 | 954.09 |
| 2 | 427.49 | 1062.73 | 1063.35 |
| 4 | 752.09 | 1178.41 | 1393.28 |
| 8 | 1441.93 | 1713.15 | 1976.98 |

Least-squares fits: full-prefix 103.45 + 166.28N; Q-CoMem shared 809.68 +
109.76N; Q-CoMem private 790.50 + 148.38N.

Sharing helps: it saves 214.87 MiB at N=4 and 263.83 MiB at N=8 against private
materialization, 15.4 and 13.3 percent, and it lowers the slope from 148.38 to
109.76 MiB per request.

But **Q-CoMem's peak transient allocation exceeds full-prefix's at every fanout
measured**, by 644 MiB at N=1 falling to 271 MiB at N=8, with a fitted crossover
at N = 12.5 that lies outside the measured range.

This matters beyond the number. In round 46 a reviewer argued that per-request
transient state bounds the capacity claim to roughly four or five concurrent
requests. The meta-review rejected that, on the ground that by the paper's own
component table the full-prefix arm carries the larger per-request mutable state,
so Q-CoMem has both the smaller intercept and the smaller slope and never
crosses. **Measurement contradicts the rejection.** Q-CoMem does have the smaller
slope, but its intercept is roughly eight times larger, and the two lines cross
above the fanouts actually run. The rejection was itself derived from unmeasured
Table 4 arithmetic; the reviewer's direction of concern was right even though the
specific one-sided bound was not.

## What this licenses, and what it does not

Licensed: the ownership discipline of Section 4.3 is now demonstrated on the
packed Read path at N>1, token-identically to the published path, with every
applicable contract target covered and passing. The composition gap that four
reviewers raised is answered by execution rather than by disclosure.

Not licensed: any claim that sharing materially reduces per-request memory, at
4 percent of per-request state; any capacity claim that ignores transient
allocation, which is higher than the exact-cache baseline throughout the
measured range; and any statement that Q-CoMem never crosses full-prefix on
total memory, which the fits place at N = 12.5 and which was asserted from
arithmetic rather than measurement.

Also not established: anything about paged kernels, vLLM, throughput, a second
backbone, or security. Eight items, one checkpoint, one stack.

## Required manuscript consequences

1. Report the composition run and its audit outcome; this is the answer to the
   round-46 critical.
2. Report the 4 percent sharing coverage in the same breath, not as a footnote.
3. Report measured peak transient allocation for both arms with its fits, and
   retract nothing from Section 5.6's existing admission that reducing Store
   does not by itself prove serving capacity -- this strengthens that admission.
4. Correct the record on the concurrency question: state that Q-CoMem's
   transient allocation is higher than the exact cache throughout the measured
   range, with the fitted crossover outside it.
