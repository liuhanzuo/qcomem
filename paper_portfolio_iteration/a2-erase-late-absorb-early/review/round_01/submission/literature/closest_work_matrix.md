# Closest-work matrix and source verification

Audit date: 2026-08-22. “Verified” means title/authors/venue were checked against a primary or official bibliographic record; it does not make an empirical claim in this manuscript true.

| work | verified source | task / assumptions | relation and boundary |
|---|---|---|---|
| Koh & Liang (2017), influence functions | PMLR v70 | influence of training points; differentiability/curvature assumptions | Relevant attribution baseline; this paper does not claim it assumes path-independent value. |
| Ilyas et al. (2022), Datamodels | PMLR v162 | predict trained-model behavior across training subsets | Related training-data/model relation; distinct from a fixed-quadratic time intervention. |
| Park et al. (2023), TRAK | PMLR v202 | scalable data attribution for differentiable models | Related attribution method; no comparative result is claimed. |
| Bourtoule et al. (2021), SISA | IEEE S\&P DOI `10.1109/SP40001.2021.00019` | sharded/sliced retraining for machine unlearning | Distinct retraining construction; T2 is not an unlearning guarantee. |
| Arazo et al. (2019) | PMLR v97 | loss-mixture noisy-label modeling | Motivation for a planned loss-mixture policy only. |
| Wei et al. (2022), CIFAR-10N | ICLR 2022 official/ML Anthology record | real human-noise benchmark with clean/noisy labels | Defines the planned N scenario; no present result is claimed. |
| Jiang et al. (2018), MentorNet; Han et al. (2018), Co-teaching | original ICML/NeurIPS proceedings metadata pending final BibTeX lock | robust learning with noisy labels | Related detectors/training methods; not claimed as a policy validation. |
| Mandt et al. (2017); Smith & Le (2018); Smith et al. (2021) | original JMLR/ICLR metadata pending final BibTeX lock | SGD noise / implicit regularization | Motivates explicit T3 assumptions; does not justify nonlinear affine transfer. |

Primary metadata records consulted: PMLR pages for [Koh--Liang](https://proceedings.mlr.press/v70/koh17a.html), [Ilyas et al.](https://proceedings.mlr.press/v162/ilyas22a.html), [Park et al.](https://proceedings.mlr.press/v202/park23c.html), and [Arazo et al.](https://proceedings.mlr.press/v97/arazo19a.html); the [CIFAR-10N ICLR record](https://mlanthology.org/iclr/2022/wei2022iclr-learning/); and the [SISA DOI record](https://doi.org/10.1109/SP40001.2021.00019). The copied bibliography is a build dependency whose entries still require per-entry final lock before submission, especially `dataprophet2026`.
