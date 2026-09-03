# Story architecture — revision 01

## Candidate stories

| ID | Problem / insight | Strongest evidence | Main objection / forbidden claim |
|---|---|---|---|
| S1 | A model ranking can be certified over every subgroup-mixture weight by simultaneous paired regret bounds. | P1/P2/P3 plus matched exact-certificate frontier. | Conditional on within-group invariance; do not claim natural-shift validation or operational utility. |
| S2 | Budget allocation can be solved optimally for candidate/mix regret UCBs. | M3/M3.5 snapshot comparisons and Eq. (mm). | Unsupported: Eq. (mm) removes candidate-specific means/widths and is only a width proxy. Must not be used. |
| S3 | Exact certification exposes abstention's practical value under real temporal/geographic shift. | Constructed-mixture rates and forced-pick diagnostics. | No natural-shift run or precommitted fallback/cost measurement exists. Must remain planned. |

## Selection

**Selected story: S1.** A conditional, simultaneous paired-regret certificate for fixed
candidate ranking over the continuous subgroup-mixture simplex, with exact methods explicitly
separated from an asymptotic diagnostic and abstention honestly left cost-unpriced.

S1 has a direct proof and an implemented snapshot. S2 has the highest overclaim risk and is
retained only as a clearly labelled certificate-width diagnostic. S3 is the right next
experiment, not current evidence.

## Paper identity and claims

**Identity.** Given a held-out calibration split and a stated subgroup-mix-turnover assumption,
pair candidate errors on the same examples to give one simultaneous, finite-sample regret
certificate for every mixture weight; commit only when its exact certificate is below an
operator tolerance.

1. P1/P2 provide simultaneous absolute mixture-risk/regret certificates by exact CP bands.
2. P3/Thm. 1 gives the central simultaneous paired-regret certificate for any selected candidate.
3. Exact paired and CP variants have a reported certificate-price frontier on constructed
   class-mixture shifts; normal/CLT is asymptotic diagnostic evidence only.
4. Budget allocation comparisons are descriptive. Eq. (mm)/Prop. 5 concern only the
   one-mixture subproblem of a width surrogate: for $W=\{w\}$, all-$G$ uniformity is optimal
   only when every $w_g\beta_g$ is equal and strictly positive; zero coefficients receive zero
   continuous allocation. General multi-mixture $W$ uses a separate dual calculation. None of
   this is actual regret-UCB optimization, safe allocation, or an allocation policy.

Each claim is falsified by, respectively: failure of simultaneous cell/pair coverage; a valid
committed violation; rerun/audit disagreement; or showing that prose assigns an actual-UCB
meaning to the surrogate.

## Evaluation map and section budget

| Section | Function / evidence | Main-text budget |
|---|---|---:|
| Introduction + setup | Conditional problem and abstention boundary | 1.4 pages |
| Theory | P1/P2/P3 and simultaneous scope | 2.0 pages |
| M1/M2 | Gate-not-select caution and exact frontier | 1.8 pages |
| M3/M3.5 | Budget outcomes plus width-surrogate boundary | 1.5 pages |
| Related work, limitations, conclusion | positioning and unresolved natural-shift/cost gap | 1.3 pages |
| Total | ICLR 2027 main text target | 8.0 pages |

Tables: `tab:main` distinguishes exact M2 from normal diagnostic; `tab:frontier` has the
finite-sample split; M3.5 tables are descriptive width-surrogate comparisons. Figure
`fig:m10frontier` remains an exact-relative-gate boundary visual, not allocation evidence.
