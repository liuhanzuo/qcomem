# Derived versus measured — admissibility rule for R45

Date: 2026-09-02
Standing instruction from the user: every projected figure must be confirmed by an
actual experiment before it is stated as a result.

## The rule

Two categories, and only one of them may enter the manuscript as a result.

**ADMISSIBLE — re-analysis of archived per-item measurements.**
The quantity was measured in a registered run; we are recomputing a different
statistic or a different reference arm from the archived per-item rows. No new
configuration is imagined. These may enter the manuscript with their method and
seed stated.

**NOT ADMISSIBLE — projection of an unrun configuration.**
The quantity describes what some configuration *would* cost or achieve if it were
run. It rests on assumptions the archive cannot test. These may be used for
planning and for choosing which experiments to run, and they may appear in
internal notes, but they must NOT appear in the manuscript, the abstract, a
figure, or a response letter until the corresponding run exists.

## Inventory of every number currently in play

### Admissible (re-analysis of archived per-item data)

| quantity | value | basis |
|---|---|---|
| frozen vs full-prefix Q16 paired F1 | -0.4455, 95% CI [-2.0586, +0.9907] | A2, 360 archived item-level F1 values, seed 20260902, 10,000 paired resamples |
| all six published mean F1 values | reproduce exactly | A2, max delta 6.11e-16 |
| headline under a consistent all-BF16 reference | 10.9965x | A11/A8, exact reconstruction on 60/60 items, 5 configurations |
| FP32 GDN share of the full-prefix baseline | 42.8% | A11, exact; +6.0000 MiB/doc at j=7, +30.0000 MiB/doc full prefix |
| the two Store estimands are the same estimand | 0 B residual | A8, 780 byte-exact identities, and bit-exact reproduction of all five Table 2 values |
| the 136.235 vs 140.34 gap decomposition | 69.3% cohort, 30.7% mean-vs-median, 0.0% estimand | A8, archived per-item rows |
| prediction identity full-prefix vs split Q16 | 58/60 | A2 |
| residual counter conforms to Eq. 3 | 180/180, 0 mismatches | A1 |

### NOT admissible until measured

| quantity | current projected value | required experiment | what the projection assumes |
|---|---|---|---|
| advantage over a quantized full-prefix cache | ~3.1x | **A5** | that the Eq. 3 packer applies uniformly to a full 40-layer cache, that the Q4 format ratio holds there, and that quality is unchanged. None is tested. A quantized full-prefix cache may also lose quality, which would change the comparison in our favour, or may not, which would change it against us. |
| full-prefix Q4 retained state | ~29.88 MiB/doc | **A5** | same |
| throughput ratio at n=128 | 91.8% | **A4** long-generation arm | that TPOT stays constant as the KV grows. The tok/s model is validated only at n≈8. |
| throughput ratio at n=512 | 96.5% | **A4** long-generation arm | same |
| asymptotic throughput gap | -1.74% | **A4** long-generation arm | same, and that no other cost appears at length |
| resident documents at fixed VRAM | unmeasured | **A13** | that retained-byte reduction converts into residency. Fragmentation, workspace, and allocator behaviour may prevent this entirely. |
| j = 7 is a good operating point | unmeasured | **A14** | never swept |
| ownership discipline holds on the quantized Read path | unmeasured | **A15** | A11 finding C4 shows the measured Read path is full private materialization, not the borrow/COW discipline the method section describes |

## Consequence for R45

R45 states the admissible rows only. Every not-admissible row is either omitted or
written as an explicitly labelled open question with the experiment named. In
particular the manuscript must not claim any advantage over a quantized exact
cache, and must not claim throughput neutrality, until A5 and A4 return.

Where a reviewer asked for a comparison we cannot yet make, the honest response is
that the experiment is registered and pending, not a projected number.
