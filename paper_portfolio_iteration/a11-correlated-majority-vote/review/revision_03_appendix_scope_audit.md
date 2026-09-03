# A11 Round-3 Appendix scope audit

Date: 2026-08-22  
Scope: evidence-only correction of the Round-2 Appendix E(g) findings. No
experiment, fitting, selection, frozen numerical artifact, baseline, remote
snapshot, or frozen review was modified.

## Adjudicated finding and manuscript action

`review/round_02/adjudication/appendix_repair.json` finds that the submitted
certificate-ordered subset is selected separately after conditioning on final
count $K$. Because $K$ is unobserved at every nonterminal prefix, the supplied
family is not one shared state table $S(x,k)$ and does not establish a
prefix-observable stopping policy.

The active Round-3 manuscript therefore calls it an **oracle K-wise
profile-capping sensitivity calculation** only. It explicitly says that it is
not a stopping policy, not an instance of Theorem 1, and cannot support
executable/deployable repair or policy-level extra-vote/cost, 70-of-72,
universal-budget, domination, edge-law, crossing-set, or full-simplex claims.
The historical E(g) text has been removed from the live source so it cannot
confuse source review; its exact submitted form remains preserved in the
immutable Round-2 snapshot and adjudication record.

The same adjudication invalidates the former all-radius identity
$V(R)=\mathrm{base}+R(g_{\max}-\mathrm{base})$. Round 3 withdraws Proposition
2 and its base-driven crossing conclusion. The retained TV statement is that
the general capacity-constrained LP is piecewise linear and that $B_1$ is the
simplex, so $V(1)=\max_K g(K)$.

## Theorem, protocol, and endpoint scope

Theorem 1 is retained only for genuinely prefix-measurable rules and does not
depend on the K-indexed calculation or the withdrawn proposition. Theorem 2
now states i.i.d. task sampling, FIT/CAL independence, conditional freezing of
the rule family on FIT, and CAL-side Bonferroni selection. The fixed benchmark
TEST split is explicitly descriptive. The title, abstract, contributions, and
Table 1 identify the object as count-exchangeable binary pass-count replay;
they do not claim ordered-online validity, delivered-answer correctness, gold
correctness, or operational cost.

## Verification protocol and result

1. `python3 manuscript/claim_audit.py` verifies the active source, exact
   manifest hashes, the oracle-only boundary, absence of the active affine
   identity/Proposition 2, theorem/protocol wording, Table 1 status labels,
   and pre-existing endpoint/provenance checks.
2. Two controlled isolated builds use the same fixed environment variables and
   must yield identical PDFs. The final log must have no TeX error or undefined
   citation/reference.
3. PDF text is searched for the withdrawn policy language, and the affected
   theorem, Table 1, Appendix E(g), and page-break locations are visually
   inspected after rendering.

Final result, hashes, page count, and visual-page list are recorded in
`build/build_record.json` under `revision_03`.

## Remaining evidence gap

An executable Appendix E(g) repair remains unproved. It requires a single
K-independent serialized $S(x,k)$ based only on prefix-observable information,
mechanical measurability/consistency checks, and a complete recomputation of
the profiles and capacity-aware TV LP. Separately, ordered rollout traces,
gold stopped-answer correctness, and token/latency/cancellation telemetry are
still absent; this revision does not claim to repair those gaps.
