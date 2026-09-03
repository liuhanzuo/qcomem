# Closest-work matrix and source verification

Audit date: 2026-08-22, revision 02. The status of every active citekey below
is identical to `literature/citation_lock.json`. ``Locked'' means that the
metadata source and deliberately narrow local-sentence support relation were
checked; it is not a novelty finding.

| active source(s) | lock status | task / assumptions | relationship to this paper |
|---|---|---|---|
| Koh & Liang (2017), `koh2017understanding` | primary metadata verified; support narrowed | Influence of training points under differentiability/curvature assumptions | Context for data/model relationships only; not a theorem antecedent for T1--T3. |
| Ilyas et al. (2022), `ilyas2022datamodels` | primary metadata verified; support narrowed | Predict trained-model behavior across training subsets | Context only; no path comparison or datamodel result is claimed. |
| Park et al. (2023), `park2023trak` | primary metadata verified; support narrowed | Scalable data attribution for differentiable models | Context only; no comparative result is claimed. |
| Bourtoule et al. (2021), `bourtoule2021sisa` | primary metadata verified; support narrowed | Sharded/sliced retraining for machine unlearning | Different construction; this manuscript makes no unlearning result. |
| Mandt et al. (2017), `mandt2017sgd`; Smith & Le (2018), `smith2017bayesian`; Smith et al. (2021), `smith2021origin` | primary metadata verified; support narrowed | SGD noise / implicit regularization | Motivate stating T3's affine conditional-mean recursion as an assumption; they do not establish that premise for general SGD here. |
| Arazo et al. (2019), `arazo2019unsupervised`; Jiang et al. (2018), `jiang2018mentornet`; Han et al. (2018), `han2018coteaching`; Wei et al. (2022), `wei2022cifar10n` | primary or official metadata verified; support narrowed | Noisy-label modeling, curricula, sample selection, and a human-noise benchmark | Broad context only. No empirical policy is defined or evaluated in this paper. |

## Theorem-level positioning boundary

No already locked source in this package supplies a verified theorem-level
antecedent comparison for all three active statements: the fixed-quadratic
product/log remainder (T1), ambient-Hessian full-norm contraction (T2), and
affine scalar conditional-mean product (T3). It would be misleading to infer
novelty from that absence. Revision 02 therefore makes no novelty claim and
does not present the juxtaposition as a new unifying theorem. The novelty
ceiling remains until a source-verified distinction from direct optimization
and stochastic-recursion antecedents, a nontrivial common theorem, or new
evidence is added.

Primary metadata records used by the lock include [Koh--Liang](https://proceedings.mlr.press/v70/koh17a.html), [Ilyas et al.](https://proceedings.mlr.press/v162/ilyas22a.html), [Park et al.](https://proceedings.mlr.press/v202/park23c.html), [Arazo et al.](https://proceedings.mlr.press/v97/arazo19a.html), [MentorNet](https://proceedings.mlr.press/v80/jiang18c.html), [Co-teaching](https://proceedings.neurips.cc/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html), [CIFAR-10N](https://mlanthology.org/iclr/2022/wei2022iclr-learning/), [Mandt et al.](https://www.jmlr.org/papers/v18/17-214.html), the two [Smith sources](https://openreview.net/forum?id=BJij4yg0Z) [respectively](https://openreview.net/forum?id=rq_Qr0c1Hyo), and the [SISA DOI record](https://doi.org/10.1109/SP40001.2021.00019).
