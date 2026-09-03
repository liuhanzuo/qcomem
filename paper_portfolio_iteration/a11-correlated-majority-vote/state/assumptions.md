# Assumptions and boundaries

1. **Venue.** The target is ICLR 2027. Official Author and Reviewer Guidelines were verified on 2026-08-22: initial main text is at most 9 pages; submission is double blind; the AI-use statement is mandatory and the reproducibility statement recommended; the abstract and paper deadlines are 2026-09-18 and 2026-09-25 AOE. The local ICLR 2026 discrete rubric remains an internal review proxy, not a claim about an official 2027 score form.
2. **Frozen object and revision boundary.** `baseline/paper.tex`, `artifacts/checkpoints/pre_revision_01/paper.tex`, and `remote_snapshot/paper.tex` retain SHA-256 `0a0978…9acb6`; their supplied PDFs retain `540052…39b7`. Revisions modify only `manuscript/` and associated state/evidence/review/build records; their output hashes are recorded in `build/build_record.json`.
3. **Validity scope.** The theorem and existing numerical evidence are assessed only under the stated count-exchangeable replay model.  The public OMR carrier contains per-task pass counts, not chronological rollout outcomes.
4. **Endpoint distinction.** In the current replay paper, `full-N` is the binary
   pass-count decision $\mathbf{1}\{K>N/2\}$, so `prefix majority != full-N
   majority` is an agreement/flip endpoint. It is not an aggregation of
   delivered textual answers, task correctness, or a substitute for gold
   accuracy and user utility.
5. **Oracle/fitted terminal convention.** At a reachable replay endpoint, all
   binary outcomes are observed and $x_N=K$; therefore the oracle
   $c_{H_\star}(x_N,N)=0$, and the fitted-score DP endpoint is also zero.
   This endpoint fact does not identify $s_{\widehat H_{\rm FIT}}(x,k)$ with
   the true conditional probability at earlier prefixes.
6. **Cost distinction.** Savings reported as `1 - mean(k)/N` are counterfactual rollout-count savings.  They are not verified generated-token savings, wall-clock latency, queueing cost, or cancellation cost.
7. **Existing artifacts and recovery.** Result JSON and check logs are treated
   as supplied artifacts, not new scientific experiments. The Round-4 bundle
   provides a derived anonymous OMR manifest (22,230 source rows, 12,423 valid
   count rows, 11,607 deduplicated problems; FIT/CAL/TEST 4000/4000/3607,
   seed 20260815, $K=32$) and byte-exact main-JSON replay conditional on that
   manifest. It does not clean-room reconstruct the manifest from raw parquet
   or locally rerun secondary runners.
8. **Audit separation.** `remote_snapshot/claim_check.py` verifies only the
   frozen source/artifacts. `manuscript/claim_audit.py` is hash-bound to the
   Round-5 manuscript snapshot and checks claim scope, active Figure 1
   provenance, replay-bundle references, and current source semantics; neither
   audit validates deployment claims or replaces a proof assistant.
9. **Visual role and provenance.** The active Figure 1 is a code-native
   Matplotlib schematic rendered from hash-pinned frozen JSON, not a generated
   image. Its drift input is reviewer-package-local at sibling
   `evidence/repro_bundle_round4/recovered_outputs/`; fixed
   `SOURCE_DATE_EPOCH` and metadata make the renderer deterministic. The old
   generated raster is retained as historical-only provenance.
   The active figure separates true $H_\star$ oracle theory from
   $\widehat H_{\rm FIT}$ plus CAL screening and does not independently
   establish an empirical, online, correctness, or cost claim.
10. **CAL population theorem.** Theorem 2 assumes that FIT and CAL are
    independent i.i.d. task samples from one count-exchangeable replay
    population, and conditions on the entire FIT-frozen rule family. For the
    reported protocol $J=64$ and $\delta_r=0.05/64$. Its
    empirical-Bernstein/Bonferroni result is a CAL selection guarantee for a
    marginal replay loss; the fixed benchmark TEST split is a one-time
    descriptive readout, not a second population guarantee or fitted
    conditional-identity statement.
11. **Appendix E(g) oracle boundary.** The supplied certificate-ordered
    subset calculation is indexed by the unobserved final count $K$. It is an
    oracle K-wise profile-capping sensitivity diagnostic, not a common
    prefix-observable table $S(x,k)$, a stopping policy, or an instance of
    Theorem 1. It cannot support executable/deployable repair, policy-level
    extra-vote/cost, 70-of-72, universal-budget, domination, edge-law,
    crossing-set, or full-simplex claims without a newly constructed shared
    table and full recomputation.
12. **Closest-work scope.** Waudby-Smith and Ramdas (2020) supply a direct
    same-endpoint WoR-CS majority-stopper comparator under the random-prefix
    finite-population model; it is not evaluated here. Rossell and M\"uller
    (2013) and Novikov (2010) are Bayesian sequential-stopping antecedents.
    The supported novelty is only model-assisted independent FIT/CAL screening
    of a finite FIT-frozen plug-in score family for exact task-level replay
    loss, with no CS or efficiency separation claim.
13. **Formal edge semantics.** Lemma~1 is monotone on odd early budgets, but
    an odd $k^*$ exists only for a nonempty feasible set; the protocol's
    deterministic FULL-$N$ fallback is outside its $J=64$ CAL family. For the
    TV sensitivity proposition, $\tau^*(\alpha)$ is numeric only if
    $V(0)\le\alpha$; otherwise the record is explicitly infeasible rather
    than $\tau^*=0$. CPU-only boundary scripts check both distinctions.
