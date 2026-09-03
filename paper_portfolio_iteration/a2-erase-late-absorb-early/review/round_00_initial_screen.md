# Round 00 — external initial screen (context only)

This is not a synthetic five-reviewer round. It records only the relevant finding from `INITIAL_ICLR_PORTFOLIO_SCREEN_ZH.md`, dated 2026-08-22.

- Target scored object: frozen public portfolio draft.
- Reported cross-review distribution: `6/6/4` for novelty/positioning, technical soundness/experiments, and impact/reproducibility respectively.
- Reported meta score: `6` on the local ICLR 2026 discrete scale.
- Stated strongest point: fixed-quadratic product law, strong-convex tail contraction, and matched-path evidence form a coherent story.
- Stated decisive blocker: there is no deployment-visible way to decide whether a high-loss example is harmful/noisy or benign-hard/valueful; the real positive domain is narrow.
- Named minimum next step: freeze an observable enable signal and test it with matched-path multi-seed evaluation in at least two real noisy/benign settings.

The screen also says it is a portfolio triage rather than a per-paper five-agent full review, did not rerun experiments or fully verify citations, and does not predict acceptance. Consequently it is not included in full-panel statistics, does not select a checkpoint, and creates no fabricated reviewer JSON.

The following audit records add constraints not supplied by the screen: public baseline and remote snapshot have different source hashes; the public baseline is missing build/citation/evidence dependencies; and ICLR 2027 rules remain unverified, so the local ICLR 2026 rubric is a proxy only.
