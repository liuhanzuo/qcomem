# Fresh change verification — revision 01b

Date: 2026-08-22  
Verifier: isolated GPT-5.6 Terra technical verifier  
Verdict: **resolved** for the requested terminal-semantics and visual-provenance changes; no new in-scope blocker.

## Verified changes

- `manuscript/paper.tex` defines the reachable terminal certificate as `c_H(x,N)=0`; Theorem 1 applies the ordinary threshold rule at `k=N`.
- `manuscript/appendix_proofs.tex` uses `x_N=K` and a zero terminal contribution; `manuscript/appendix_dp.tex` uses the same terminal convention and stop test.
- The full-budget endpoint is explicitly the binary pass-count replay decision, not a delivered-answer majority.
- The adaptive statement is confined to prefix-filtration stopping times that stop only at certified states; the manuscript disclaims arbitrary ordered-online or e-process validity.
- Figure 1 is disclosed as an AI-composed conceptual layout whose quantitative labels were independently checked against frozen artifacts. `evidence/visual_asset_provenance.json` binds the current PNG, frozen generation/edit history, source result hashes, and label-by-label checks.
- CAL selection versus the single TEST read is stated consistently. The verifier did not inspect TEST labels beyond the frozen checker summaries; this is not independent proof of historical single access.

## Read-only checks

- `python3 manuscript/claim_audit.py`: 21 PASS / 0 FAIL, hash-bound to current `paper.tex` SHA-256 `f89c07a8aacf76f12ff892a8a70ab7f87238cfc8ba6653c0fe2f8f3828c06248`.
- `python3 remote_snapshot/claim_check.py`: 426 PASS / 0 FAIL / 3 external-provenance items. This checks frozen historical source/artifacts only, not the current manuscript.
- Current PDF SHA-256 matches the isolated build record: `8138d3434d8886d299b402d2fe73837eadb87e57ea256cd12d975573fc3f0dad`.

The broader evidence gaps remain unchanged: no naturally ordered online rollout, gold-correctness, token/latency, or cancellation-cost result is supplied or claimed.
