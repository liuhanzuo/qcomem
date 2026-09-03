# The total-memory model — what is measured, what is assumed

Date: 2026-09-03
Purpose: state the deployment-regime result the paper should report, and fix
exactly how much of it is measurement.

## The two measured terms

**Retained state per document**, measured on the 60-item validation cohort
(evidence `E-QCOMEM-60-OPPOINT-20260903A`): full-prefix 136.235 MiB/document,
frozen depth-7 Q4/Q4/Q8 9.661 MiB/document.

**Peak transient allocation during a query**, measured in the C1 composition run
at fanouts N in {1,2,4,8} (Trial 1944486): full-prefix 286.44 / 427.49 / 752.09
/ 1441.93 MiB; Q-CoMem shared-packed 930.91 / 1062.73 / 1178.41 / 1713.15 MiB.
Least-squares fits over those four points: full-prefix `103.45 + 166.28N`,
Q-CoMem `809.68 + 109.76N`.

Both terms are direct measurements. Neither is projected.

## The model, and its three assumptions

    total(D, N) = retained_per_document * D + peak_transient(N)

for D warm documents and N concurrent requests. Combining two measured
quantities by addition is arithmetic on measurements and is admissible, but the
model carries three assumptions that were NOT measured and must be stated
wherever it appears:

1. **Retained state scales linearly in D.** Each document's entry is
   independent, so this is structural rather than empirical, but no run held
   more than one document resident.
2. **Peak transient does not depend on D.** The C1 run had one document
   resident, so the model assumes serving a request costs the same whether five
   or five hundred documents are warm. Plausible, since transient cost is
   per-request reconstruction work, but unverified.
3. **The fits hold only inside N in [1, 8].** Four points. Any statement about
   N beyond 8 is extrapolation and is not admissible.

## Result

Q-CoMem's total is lower when

    D > (peak_transient_qcomem(N) - peak_transient_fullprefix(N)) / (136.235 - 9.661)

| concurrent requests N | Q-CoMem wins once warm documents exceed |
|---:|---:|
| 1 | 5.13 |
| 2 | 4.69 |
| 4 | 3.79 |
| 8 | 2.01 |

Worked points, total MiB:

| warm docs D | concurrency N | full-prefix | Q-CoMem | ratio |
|---:|---:|---:|---:|---:|
| 4 | 8 | 1978.6 | 1726.4 | 1.15x |
| 20 | 4 | 3493.3 | 1441.9 | 2.42x |
| 100 | 1 | 13893.2 | 1885.5 | 7.37x |
| 100 | 8 | 15057.2 | 2653.9 | 5.67x |

## Why this is the right thing to report

The manuscript currently reports 14.1x, which is the retained term alone. That
is exactly the omission the round-46 panel attacked: Eq. 1 assumes active
workspace is method-independent, and this method's own Read falsifies that. The
total-memory form answers the objection instead of restating the ratio, and it
is weaker but more useful: it names the regime in which the method helps and the
regime in which it does not.

The crossover is low. Above roughly five warm documents at low concurrency, and
roughly two at N=8, the method wins on total memory, and the margin grows with
the warm set. Below it, it does not. A deployment that keeps a handful of
documents warm and serves many concurrent requests against them is outside the
method's useful range, and the paper should say so rather than let a reader
discover it.

## What must NOT be claimed

- No specific agentic or multi-document workload was measured. The regime may be
  described qualitatively as one where many documents stay warm and each is
  queried intermittently; no benchmark, task, or trace is cited as evidence for
  it, because none was run.
- No N beyond 8.
- No claim that the transient term is independent of D; it is assumed.
- The 14.1x retained ratio remains correct for what it measures and should be
  kept, subordinated to the total-memory form rather than replaced by it.
