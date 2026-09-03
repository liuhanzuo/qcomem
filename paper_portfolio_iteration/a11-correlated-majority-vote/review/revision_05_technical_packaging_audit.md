# Round-5 technical, packaging, and clarity audit

Date: 2026-08-22  
Scope: current-evidence technical/package/clarity revision only. No new
scientific experiment, GPU/remote job, fitting, selection, baseline,
`remote_snapshot/`, or frozen review/submission was changed.

## Resolved reviewer items

| Issue | Current repair | Verification |
| --- | --- | --- |
| A11-R04-TS01 | Lemma 1 and Appendix A.1 prove the even-$N$, $K=N/2$ center separately; reflection is restricted to $K<N/2$. An odd early budget exists only for a nonempty feasible set. FULL-$N$ is a deterministic fallback, not a CAL-selected member of $J=64$. | Exact rational audit: 21,824 reversal cases, 11,904 zero-$F_0$ support cases, 528 centers, and 32 $\alpha<1/2$ no-feasible-odd/FULL fallback cases all pass. |
| A11-R04-TS02 | Remark 1 retains the zero-sum demeaning identity and count-mixture functional but removes any source or dominance attribution for correlation. | Hash-bound claim audit requires mechanical-identity/nonidentifiability wording. |
| A11-R04-TS03 | Proposition 1 defines $\tau^\star$ only if $V(0)\leq\alpha$; otherwise output is `infeasible_at_R0`, distinct from a feasible $\tau^\star=0$. | Dedicated CPU semantic audit passes; stored numeric sensitivity displays are unchanged. |
| A11-R04-RP01 | Figure 1 renderer reads `evidence/repro_bundle_round4/recovered_outputs/drift_stress_r469_result.json`, not a frozen-snapshot path. | The recovered input has SHA-256 `1208deade2cb42a324bb948c93bf66ee72e68a93e78815aadcfbeb19880c3163`; an isolated reviewer-package-shaped copy rendered twice byte-identically under fixed metadata, and active assets/provenance are hash bound. |
| A11-R04-CP01 | Table 5 now labels outcomes as descriptive TEST, names `CAL EB UCB`, marks OMR $\alpha=.01$ CAL-only, and Section 6 points to that status. | Static audit and final page-16 visual QA pass. |
| A11-R04-MR01 | Related work/matrix name the Waudby-Smith--Ramdas without-replacement confidence-sequence majority stopper as a direct same-endpoint comparator not evaluated here. | Claim audit, closest-work records, and citation lock require direct-comparator/no-separation wording. |

## Reproducible checks

- Current hash-bound claim audit: **54 PASS / 0 FAIL**.
- Main replay: `replay_fit_cal_test_stdlib.py` reproduced the main JSON
  byte-for-byte conditional on the derived manifest:
  `b114c72d9ab1cf1a6ba1d2bd734433c06bd4d5cbd19bf93be0964edf6fc8a5f7`.
- Frozen read-only checks: remote claim check **426 PASS / 0 FAIL / 3
  external-provenance**; Round-3 snapshot **113 files** at
  `683d2d58f3eead73dba6344efefe02a17d9904f9170b77fb6364a6e1735dc6ee`;
  Round-4 snapshot **133 files** at
  `36a6e8c45f3c2aed3763f2ebb1efa07641e83957120cb31ac9cdba1bb234fd7d`.
- Two fresh noaux isolated source builds are byte-identical:
  `e5d9f7839b6b627570b63be172fa34b7b2d8340024e99479d5ef18c2812b2c89`.
  They have 16 letter pages; Conclusion ends on page 9; final logs contain no
  TeX error, undefined citation/reference, or overfull box.
- Visual QA at 144 dpi: pages 2 (Figure 1), 4 (Lemma/proof), 5 (TV semantics),
  6 (protocol), 9 (Conclusion), and 16 (Table 5). No clipping, overlap, stray
  mark, or unreadable affected element was observed. Figure 1 remains readable
  at 100% page width (minimum approximately 6.9pt effective type).

## Final active hashes

| Item | SHA-256 |
| --- | --- |
| `manuscript/paper.tex` | `b0e1207a223b9b884abc206a47b842f8e2b2a057a45cbd3e34f31d2992e54ffa` |
| `manuscript/paper.pdf` | `e5d9f7839b6b627570b63be172fa34b7b2d8340024e99479d5ef18c2812b2c89` |
| Figure 1 renderer | `a209fd417a6292a0f7a61dc3ce7b4a3d5b0eff692942aec8f81085057b2ba1b7` |
| Figure 1 PDF / PNG | `a7c0f72f3ea179d0a2e597e751598bc7ce64047e4586d211744c06e4abe035ac` / `a5e3ed1605c0fcf6e1dfa1868c1400eea83592c54be3c38db611d76251c6a7d6` |
| Figure 1 input manifest | `6562da13845dabdc4c18eb298346da1dce76f2d8a1fca55c3343bafe14232f02` |

## Residual evidence boundaries

- No same-endpoint comparison against the confidence-sequence-derived stopper;
  no efficiency separation is claimed.
- No naturally chronological rollout, stopped-answer gold-correctness result,
  generated-token/latency/cancellation telemetry, raw-parquet clean-room
  reconstruction, secondary-runner local rerun, target-population TV coverage,
  or executable $K$-independent Appendix E(g) repair.
- The replay guarantee remains conditional on the derived manifest and the
  stated random-prefix/count-exchangeable replay model.
