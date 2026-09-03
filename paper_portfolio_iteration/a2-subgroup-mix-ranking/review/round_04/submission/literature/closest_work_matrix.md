# Closest-work matrix — revision 01

Metadata and local-support verification completed 2026-08-22. ``Relationship'' is a
scope statement, not a novelty claim.

| Work | Verified primary source | Task / assumptions | What it provides | Difference / limitation relative to this paper |
|---|---|---|---|---|
| Clopper & Pearson (1934) | *Biometrika* 26(4):404–413, DOI `10.1093/biomet/26.4.404` | Binomial interval estimation | Exact binomial intervals used in P1/P2. | Does not rank candidates or give mixture-simplex regret certificates. |
| Maurer & Pontil (2009) | arXiv:0907.3740 / COLT paper | Bounded independent observations; variance-sensitive empirical bounds | The empirical-Bernstein ingredient for exact paired bounds. | Does not supply the paper's pair/group Bonferroni or mixture-regret construction. |
| Dunnett (1955); Hsu (1984) | JASA 50(272):1096–1121; *Annals of Statistics* 12(3):1136–1153 | Multiple comparisons with a best/control | Conceptual paired best-vs-rest comparison ancestry. | No continuous mixture index or abstaining deployment decision. |
| Saerens, Latinne & Decaestecker (2002) | *Neural Computation* 14(1):21–41, DOI `10.1162/089976602753284446` | Prior shift with fixed within-class densities | Adjusts a chosen classifier under changed priors. | This work chooses from a fixed pool and certifies candidate regret; it does not recalibrate outputs. |
| Sagawa et al. (2020) | ICLR 2020 OpenReview / ML Anthology | Group shifts and worst-group training loss | Group-DRO baseline perspective. | Optimizes training models; it is not a post-training, mixture-indexed regret certificate. |
| Christie et al. (2018), fMoW | CVPR 2018, DOI `10.1109/CVPR.2018.00646` | Satellite imagery with temporal and geographic metadata | Planned natural-shift execution candidate. | Candidate only: no experiment is run or claimed in this paper. |

## Support audit

- The CP sentence only claims exact cell intervals; it is supported by Clopper--Pearson.
- The Maurer--Pontil sentence only claims a variance-sensitive empirical-Bernstein ingredient;
  it does not attribute the paper's simultaneous mixture result to that source.
- The related-work contrast to Saerens is supported because its source adjusts a classifier under
  changed priors, while the manuscript's formal object is a selected-candidate regret.
- The contrast to group DRO is supported because Sagawa et al. optimize worst-group training
  loss. No claim of being the first such method is made.
- fMoW is not yet cited in the manuscript as evidence; the original CVPR paper documents
  timestamps and UTM geographic metadata, sufficient to lock it as the planned protocol's
  candidate subject to executable download/version recording.
