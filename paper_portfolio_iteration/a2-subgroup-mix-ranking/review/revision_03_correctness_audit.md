# Round-3 corrective correctness audit

Date: 2026-08-22  
Scope: allowed mutable manuscript, evidence, state, build, and issue-ledger files only. No
baseline, remote snapshot, frozen submission, or historical review artifact was modified. No
scientific experiment was run.

## Correctness repairs

1. **Certificate contract.** Assumption 2 now requires FIT-only fixed candidates and design
   choices, FIT--CAL independence, and conditionally i.i.d. within-group CAL samples. It makes
   individual errors i.i.d. Bernoulli for CP and same-point transformed differences
   `X=(d+1)/2 in [0,1]` conditionally i.i.d. for paired Hoeffding/MPB. It excludes
   without-replacement, CAL-adaptive, and CAL-trained/refit designs unless separately justified.
2. **Multiplicity and theorem.** Lemma 1 explicitly uses `kG` CP cells with per-cell failure
   `delta/(kG)`. Definition 1/Theorem 1 explicitly use `k(k-1)G` ordered-pair/group cells with
   failure `delta/[k(k-1)G]`; MPB also requires `n_g >= 2`. The proof uses `E_pair`, and the
   one event remains simultaneous over all simplex weights and CAL-data-dependent selections.
3. **E03 boundary.** The manuscript and evidence maps now say that E03's aggregate rows lack a
   split manifest, CAL sampling law, paired sufficient statistics, executed UCB implementation,
   and environment lock. It cannot demonstrate that the historical split met the theorem contract
   or provide realized coverage.
4. **Proposition 3.** The invalid width premise and reversed regret sign were replaced by a valid
   two-candidate condition: `U_a<U_b`, point estimates prefer `b`, and true risk prefers `b`.
   M1 selects `a`, with correctly signed regret `R_a-R_b>0`. The included witness has regret
   `0.02`.
5. **M3/M3.5.** All retained M3/M3.5 text is oracle-stratified constructed-mixture diagnostic
   language. True class cannot target an unlabeled point absent an independent external stratified
   frame, which E04/E08 do not provide. The width surrogate remains separate from actual UCB
   optimization, field label acquisition, and a deployment allocation policy.
6. **Story boundary.** The exact paired MPB endpoint (`0.260`) is the primary empirical
   narrative. The normal `0.503` endpoint remains asymptotic only. The abstract now claims only
   an aggregate abstention/regret association, not exact abstention behavior or fallback utility.

## Verification status

- Static source audit: pass. Searches found no former ``abstaining exactly''/static-policy phrase,
  no former `Delta_a` premise or reversed `R_b-R_a` regret, and no stale proof event. They find the
  explicit contract, `kG`/`k(k-1)G` counts, MPB `n_g >= 2`, E03 boundary, and oracle scope.
- Frozen artifact audit: pass. The paired self-comparator audit passed all 350 stored rows at
  `tau={0,0.02,0.04,0.10}` and its strict-dominance boundary; M5/M7/M8/M9/M10/M12 respectively
  passed `105/105`, `28/28`, `17/17`, `ALL`, `ALL`, and `245/245`. This is artifact consistency,
  not a scientific rerun or evidence that the new sampling contract held.
- Build: pass. In-place and two independent fresh source-copy builds have PDF SHA-256
  `5bfc127ea8b8114f0594ac658ebee5e019c5f09f7665b76d2fe0c40a4d0beba9`; source SHA-256 is
  `dbeda9c4316a8635a2a1c0038a34268f0a00639bcfb65c0203f670536a501cd0`. The PDF has 14 pages;
  the Conclusion ends before References on page 9. Rendered pages 3--10 and 14 passed visual QA
  without clipping, overlap, missing glyphs, or broken tables/figures.

## Residual gaps

- E03 remains non-replayable and cannot establish applicability of the exact finite-sample
  contract.
- M3/M3.5 lack an externally observable stratified frame, collection trace, and seed-level
  allocation outcomes.
- No natural time/geographic shift, within-group invariance audit, or costed fallback result is
  claimed or newly supplied.
