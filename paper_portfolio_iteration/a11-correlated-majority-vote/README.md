# A11: count-exchangeable majority-vote replay

The active manuscript is Revision 05 of *Finite-Sample Certificates for
Adaptive Stopping in Count-Exchangeable Binary Pass-Count Replay*.

## Current checkpoint

- Paper: [`manuscript/paper.pdf`](manuscript/paper.pdf)
- TeX SHA-256: `b0e1207a223b9b884abc206a47b842f8e2b2a057a45cbd3e34f31d2992e54ffa`
- PDF SHA-256: `e5d9f7839b6b627570b63be172fa34b7b2d8340024e99479d5ef18c2812b2c89`
- 16 pages; Conclusion ends on page 9.
- Frozen Round-5 snapshot: 134 files, root
  `b889ed62195b1b38ffe21b5b846eed1b5b60688d19b7cab12cb18b6961a6b9d7`.
- Independent change verification: `resolved_for_the_selected_round_05_change_scope`.

Revision 05 is the best verified artifact on integrity/build grounds, not a
new score round. The latest formal panel is Round 04:
`[6,4,4,6,4]`, median 4, meta 4, evidence ceiling 6.

## Supported claim boundary

For the true count law, the oracle score is the exact conditional replay-flip
probability and supports the tower theorem. BAYES-H instead uses a FIT-frozen
plug-in score; its operational statement is only the finite-family CAL
empirical-Bernstein/Bonferroni marginal replay screen under independent i.i.d.
FIT/CAL tasks. TEST is one descriptive readout.

The endpoint is binary pass-count replay agreement with full-`N`, and the
reported saving is rollout-count reduction. Neither is chronological online
validity, stopped-answer gold correctness, token/latency/cancellation cost, or
target-population shift coverage.

## Round-5 verification

- Active claim audit: 54/54 PASS.
- Exact CPU checks: 21,824 reversal cases, 528 even-`N` centers, and 32 empty
  odd-feasible-set to FULL-`N` fallback cases; TV null-versus-zero sentinel PASS.
- Main JSON: byte-exact replay conditional on the reviewer-safe derived manifest,
  SHA-256 `b114c72d…c8a5f7`.
- Figure 1: code-native, hash-pinned, package-local, and deterministic in an
  isolated sibling manuscript/evidence package.
- Two fresh noaux builds: byte-identical; affected pages visually inspected.

See [`review/revision_05_independent_verification.json`](review/revision_05_independent_verification.json),
[`review/best_checkpoint.json`](review/best_checkpoint.json), and
[`state/score_trajectory.json`](state/score_trajectory.json).

## Next score-moving evidence

1. Same-endpoint WoR-CS majority-stopper plus BAYES-UNIF comparison on the
   frozen manifest; no efficiency separation is currently claimed.
2. Authorized chronological rollout study with gold correctness and real
   token/latency/cancellation telemetry.
3. Raw-to-manifest clean-room provenance and local secondary-runner replay.

No new scientific experiment or remote/GPU job was run in Revision 05.
