#!/usr/bin/env python3
"""A11 r473 exp-B: OpenR1-Math-220k (DeepSeek-R1, M=2) risk-cost frontier.

Cross-model / cross-carrier evidence for the paper's central claim:
conditional-posterior certificates (BAYES-H) extract saving only when the
mixture has exploitable structure. OpenR1's M=2 per problem is the
structurally opposite carrier to OMR's N=32 counts.

Exact replay semantics (matches paper's shared-prefix replay model):
  - per problem, we observe exactly two iid rollouts (X1,X2) with dedup.
  - FULL-2 verdict = side(X1+X2, 2) with fair-coin tie-break (fixed per problem).
  - prefix-1 stop verdict = side(X1,1). Flip = P(verdict_1 != verdict_2).
    P(flip|p) = p(1-p) exactly (tie breaks contribute (2p-1)^2/4 - ... = algebra;
    we compute it in closed form: flip = p(1-p) + P(tie)/2 - |P(tie)*(2p-1)/2|...
    -- simpler: enumerate the 8 equally-weighted outcome patterns).
  - certificate for stopping at k=1 given x: c_H(1,x) = P_H(p<=.5 | x) (posterior
    that the full-2 majority disagrees, including fair tie-break mass).

Rules compared (all alpha-grid, selection on CAL with EB/Hoeffding UCB,
Bonferroni J-family, TEST single readout — same machinery as r469/r471):
  FIXED-1  (always stop at 1), FULL-2 (never stop), BAYES-H (stop at 1 iff
  c_H(1,X1) <= alpha), and a degenerate FIXED-2 = FULL-2 reference.

Splits: problems with exactly 2 deduped rollouts; FIT 3000 / CAL 3000 / TEST rest.
Pre-registered deliverable: realized flip vs alpha and saving=1 - E[k]/2 per rule.
Expected outcome (hypothesis, not gate): BAYES-H chooses ~never stop because
posterior c_H(1,x) stays high for observed p ~ near-uniform-ish middle mass;
the certificate correctly refuses to stop where structure is missing — i.e. the
method's value is carrier-aware, not a generic stop-early trick.
"""
import json, random
from math import log, sqrt

OUT = "openr1_m2_pilot_r473.json"
SHARD = "all/default-00000-of-00010.parquet"
SEED = 20260815
AGRID = [0.20, 0.10, 0.05, 0.02]
DELTA_CAL = 0.05


def side(cnt, k):
    return 1 if cnt > k / 2 else 0


def flip_prob_p(p):
    """P(verdict at k=1 != full-2 verdict with fair coin tie) for Bernoulli p."""
    # flip needs a split pair (X1 != X2), prob 2p(1-p); then the fair coin
    # assigns the full-2 verdict to the side opposite X1 with prob 1/2.
    # => flip = p(1-p).
    return p * (1 - p)


def eb_ucb(vals, delta):
    m = len(vals)
    mu = sum(vals) / m
    var = sum((v - mu) ** 2 for v in vals) / (m - 1) if m > 1 else 0.0
    return mu + sqrt(2 * var * log(4 / delta) / m) + 7 * log(4 / delta) / (3 * (m - 1))


def hoef_ucb(vals, delta):
    m = len(vals)
    return sum(vals) / m + sqrt(log(1 / delta) / (2 * m))


def mean_ci(vals, delta):
    m = len(vals)
    mu = sum(vals) / m
    return mu, sqrt(log(2 / delta) / (2 * m))


def main():
    import pyarrow.parquet as pq, hashlib
    t = pq.read_table(SHARD, columns=["problem", "correctness_math_verify",
                                      "generations"]).to_pylist()
    probs = []  # (x1, x2) with dedup
    for row in t:
        seen = set()
        obs = []
        for gen, ok in zip(row["generations"], row["correctness_math_verify"]):
            gh = hashlib.sha1(gen.encode()).hexdigest()
            if gh in seen:
                continue
            seen.add(gh)
            obs.append(int(ok))
        if len(obs) == 2:
            probs.append(tuple(obs))
    n = len(probs)
    print(f"problems with exactly 2 deduped rollouts: {n}")

    rnd = random.Random(SEED)
    coins = [rnd.random() for _ in range(n)]       # fair tie-break coin per problem
    idx = list(range(n))
    rnd.shuffle(idx)
    fit_idx, cal_idx, test_idx = idx[:3000], idx[3000:6000], idx[6000:]

    # FIT prior on coarse identifiable support p in {0, .5, 1}, by EXACT moment
    # matching of the three pair-type rates (the only identifiable functionals
    # at M=2). With q = H(.5):  P(s=1) = q/2,  P(s=0) = H0 + q/4,
    # P(s=2) = H1 + q/4.  =>  q = 2*P(s=1); H0 = P(s=0) - q/4; H1 = P(s=2) - q/4.
    # (r473 fix: naive attribution H(.5)=P(s=1) implies mixed-pair rate .058
    # instead of the observed .116 -- factor-2 moment mismatch, caught by the
    # claim_check analytic-vs-realized flip comparison.)
    c0 = sum(1 for i in fit_idx if probs[i][0] + probs[i][1] == 0) / len(fit_idx)
    c1 = sum(1 for i in fit_idx if probs[i][0] + probs[i][1] == 1) / len(fit_idx)
    c2 = sum(1 for i in fit_idx if probs[i][0] + probs[i][1] == 2) / len(fit_idx)
    q = 2 * c1
    H = [c0 - q / 4, q, c2 - q / 4]
    assert all(h >= -1e-12 for h in H), (c0, c1, c2, H)
    H = [max(h, 0.0) for h in H]

    # posterior c_H(1,x): P(full-2 verdict != side(x,1) | X1=x) under prior.
    # given X1=1: flip if (X2=0 and coin picks side0) => prob .5*E[p(1-p)|X1=1]*... compute:
    # P(X2=0|X1=1) = E[(1-p)*p | X1=1]/E[p|X1=1]... direct Bayes on atoms:
    def cert1(x):
        # likelihood of X1=x under atoms p in {0,.5,1}
        lik = [0.0, 0.5, 1.0] if x == 1 else [1.0, 0.5, 0.0]
        den = sum(H[j] * lik[j] for j in range(3))
        if den == 0:
            return 1.0
        # flip at prefix x: flip iff X2 = 1-x (pair becomes a tie) AND the fair
        # coin assigns the full-2 verdict to side 1-x. Under the coarse prior the
        # only atom reaching a tie is p=.5; there P(X2=1-x)=1/2 and the coin
        # opposes with prob 1/2 => P(flip | X1=x, p=.5) = 1/4.
        num = H[1] * 0.5 * 0.25  # weight * lik(p=.5) * flip-prob
        return num / den

    # rules
    rules = [("FULL2", "FULL", None), ("FIXED1", "FIXED1", None)]
    for a in AGRID:
        rules.append((f"BAYESH_a{a}", "BH", a))
    # CAL selection: for FIXED1 the cert is just P(flip); family = 1 fixed + 4 BH
    J = 1 + len(AGRID)
    d_sel = DELTA_CAL / J

    def replay(i, rule, par, coin):
        x1, x2 = probs[i]
        if rule == "FULL":
            return 0.0, 2.0  # never flips vs itself
        if rule == "FIXED1":
            stop = True
        else:  # BAYESH
            stop = cert1(x1) <= par
        if not stop:
            s = x1 + x2
            full = side(s, 2) if s != 1 else int(coin < 0.5)
            return 0.0, 2.0
        # stopped at 1
        s = x1 + x2
        full = side(s, 2) if s != 1 else int(coin < 0.5)
        flip = float(side(x1, 1) != full)
        return flip, 1.0

    cal_vals = {name: [replay(i, kind, par, coins[i])[0] for i in cal_idx]
                for name, kind, par in rules if name != "FULL2"}
    sel = {}
    for name, kind, par in rules:
        if name == "FULL2":
            continue
        sel[name] = {"eb": round(eb_ucb(cal_vals[name], d_sel), 5),
                     "hoef": round(hoef_ucb(cal_vals[name], d_sel), 5),
                     "alpha": par}

    d_rule = 0.05 / len(rules)
    test_res = {}
    for name, kind, par in rules:
        flips, kss = [], []
        for i in test_idx:
            fl, ek = replay(i, kind, par, coins[i])
            flips.append(fl)
            kss.append(ek)
        mf, rf = mean_ci(flips, d_rule)
        mk, rk = mean_ci(kss, d_rule)
        test_res[name] = {"alpha": par, "realized_flip": round(mf, 5),
                          "flip_ci": round(rf, 5), "mean_k": round(mk, 3),
                          "saving_vs_full": round(1 - mk / 2, 4)}

    out = {"seed": SEED, "n_problems": n, "prior_H_p0_p5_p1": [round(h, 4) for h in H],
           "cert1_x0": round(cert1(0), 5), "cert1_x1": round(cert1(1), 5),
           "E_flip_FIXED1_theory": round(sum(H[j] * flip_prob_p(p)
               for j, p in enumerate([0.0, 0.5, 1.0])), 5),
           "cal_selection": sel, "test_readout": test_res}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"prior H (p=0,.5,1): {[round(h,3) for h in H]}")
    print(f"cert1(x=0)={cert1(0):.4f} cert1(x=1)={cert1(1):.4f}")
    for name, kind, par in rules:
        t_ = test_res[name]
        s_ = sel.get(name, {})
        print(f"{name}: flip={t_['realized_flip']}±{t_['flip_ci']} k={t_['mean_k']} "
              f"save={t_['saving_vs_full']} calEB={s_.get('eb')}")


if __name__ == "__main__":
    main()
