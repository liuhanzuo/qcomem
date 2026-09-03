# Closest-work check — Round 4

This is a scoped closest-work check, not an exhaustiveness claim.  Each entry
was checked against an original paper or its official proceedings/journal page;
the descriptions below state only the observation model, target, and guarantee
visible in those primary sources.  It distinguishes two adjacent literatures:
Bayesian decision-theoretic sequential stopping, and finite-population
without-replacement sequential inference.

## Comparison matrix

| Primary source | Observation model | Terminal target | Stopping guarantee in the checked source | Treatment of prior / fitted model | Residual difference from A11 |
| --- | --- | --- | --- | --- | --- |
| Rossell & Müller (2013), [official article](https://academic.oup.com/biostatistics/article/14/1/75/250357), DOI [10.1093/biostatistics/kxs026](https://doi.org/10.1093/biostatistics/kxs026) | Batches of high-throughput measurements under GaGa or NN hierarchical models; latent differential-expression indicators. | Classify genes as equally or differentially expressed at stopping; the paper fixes a terminal rule with posterior expected FDR control. | Fully Bayesian stopping compares posterior expected utilities, with an approximate boundary implementation because exact backward induction is costly.  This is a decision-theoretic/model-based guarantee, not a held-out marginal error certificate. | Hyperparameters are estimated by EM and treated as fixed for posterior calculation; examples also examine model misspecification.  The checked source does not establish a separate FIT/CAL finite-population calibration certificate for a fitted score. | A11 has a fixed set of 32 binary rollouts per task, samples their random order without replacement, targets disagreement with that task's full-32 majority decision, freezes a task-count mixture on independent FIT tasks, then CAL-screens its selected policy by exact per-task flip risks. |
| Novikov (2010), [official paper](https://www.kybernetika.cz/content/2010/4/754/paper.pdf) / [journal record](https://www.kybernetika.cz/content/2010/4/754) | General discrete-time stochastic processes, including dependent observations. | A sequential statistical decision under a Bayesian loss. | Characterizes optimal sequential procedures minimizing average sample number subject to a bound on Bayesian incorrect-decision risk; also treats observation cost in Bayes risk. | A Bayesian risk and decision rule are part of the stipulated problem.  The official record checked here does not document a held-out fitted-prior calibration mechanism. | It is a broad Bayes-decision antecedent, but not a fixed finite population with hypergeometric prefix law, a full-budget replay target, or an independent FIT/CAL guarantee for a plug-in posterior score. |
| Ankirchner & Klein (2020), [official article](https://www.numdam.org/articles/10.1051/cocv/2019045/), DOI [10.1051/cocv/2019045](https://doi.org/10.1051/cocv/2019045) | Sequential observations of the drift of Brownian motion under two simple hypotheses. | Stop, continue, or in one case abandon a two-hypothesis test. | Under an expectation constraint, gives an optimal stopping rule characterized through barriers of the posterior probability; it also shows the constrained rule need not be an interval-exit time. | Posterior probability is the state variable of the Bayesian test.  The primary-source description does not supply a finite-population or separate fitted-prior calibration device. | It is a sharp posterior-boundary analogue, but A11's state is a hypergeometric prefix count from a finite set and its reported guarantee is task-distributional CAL screening rather than an optimal Brownian stopping rule. |
| Waudby-Smith & Ramdas (2020), [official proceedings paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/e96c7de8f6390b1e6c71556e4e0a4959-Paper.pdf) / [record](https://papers.nips.cc/paper/2020/hash/e96c7de8f6390b1e6c71556e4e0a4959-Abstract.html) | A fixed finite population observed in a uniformly random order without replacement; for binary data, the prefix count is hypergeometric. | Time-uniform confidence sets for finite-population totals/means and associated hypothesis tests. | Frequentist confidence-sequence validity at arbitrary stopping times, including Hoeffding- and empirical-Bernstein-type constructions. | Uses a prior-posterior-ratio martingale and explicitly calls the prior/posterior a working model; the paper states the CS remains frequentist and valid for any data-independent positive prior choice. | This is the closest finite-population inference precursor.  A11 changes the terminal object to a full-budget majority replay decision and uses an independently FIT-frozen plug-in score only as a candidate policy whose marginal flip risk is CAL-certified across tasks; it should not be presented as a new general confidence sequence. |

## What the primary sources support

- Rossell & Müller explicitly separate sequential stopping from terminal
  classification, formulate the ideal rule with posterior expected utility, and
  use a fitted hierarchical model.  The paper therefore supports positioning
  A11 against practical Bayesian decision-theoretic stopping, but not claiming
  that their guarantee transfers under A11's finite-population replay target.
- Novikov's official abstract states the average-sample-number objective and
  Bayesian-risk constraint for general discrete-time processes.  It supports a
  broad Bayes-risk antecedent, not an assertion about finite-population
  hypergeometric sampling.
- Ankirchner & Klein's official article states the Brownian-drift observation
  model, posterior-probability barriers, and expectation-constrained optimum.
  It supports a posterior-threshold comparison only.
- Waudby-Smith & Ramdas specify fixed finite entities, uniformly random
  without-replacement order, hypergeometric binary observations, working
  priors, and frequentist time-uniform confidence sequences.  It is direct
  support for the finite-population side of the comparison.

## Conservative positioning and citation recommendation

The defensible residual is narrow: **under an iid task-distribution assumption
across independent FIT and CAL task sets, a frozen plug-in count-mixture score
can be converted into a selected early-stop rule whose exact task-level replay
loss is screened by a finite-family, held-out marginal calibration bound.**
The terminal loss is disagreement with the full-32 binary majority decision,
not model truth, a posterior credible event, or an online service-quality
claim.

The manuscript should cite Waudby-Smith & Ramdas whenever it invokes
finite-population without-replacement or posterior-like hypergeometric
reasoning, and cite Rossell & Müller and/or Novikov when contrasting the
method with Bayesian decision-theoretic sequential stopping.  Ankirchner &
Klein is useful only if the text discusses posterior-probability boundary
methods specifically.  Avoid claims of being the first Bayesian sequential
stopping method, the first finite-population sequential method, or a
distribution-free guarantee for a fitted posterior.  The evidence supports a
specific FIT/CAL calibration contribution, subject to the manuscript proving
its stated task-iid and selection-family assumptions.
