# R40 live-binding V29 independent post-run audit report

Audit date: 2026-09-02 (Asia/Shanghai)  
Decision: **PASS; fresh admissible formal result**

## Execution identity and audit disposition

The frozen V29 launcher completed its terminal path on QS Job `256220`, Trial
`1936087`, Pod `qs-256220-1936087-ai-1482497-master-0`. The result has run ID
`71391b1a7ce85c4dfa8beb18f3c2189a`, eight committed shards, scientific outcome
`valid_positive`, empty root and formal `COMPLETE` markers, and no failure
ledger. The read-only attempt-2 post-run audit reports `audit_status=pass`.

Attempt 1 was a checker defect: it treated the declared numerical-oracle
threshold `0.005` as if it were the observed maximum. Attempt 2 corrected the
predicate to require `threshold == 0.005` and every observed rank value, hence
the vector maximum, to be at most that threshold. The observed maximum is
`0.0017432502481433169`, so the corrected check passes. This distinction does
not reclassify a failed scientific run; the formal launcher had already
completed successfully before either post-run checker invocation.

## Scientific and live-binding result

- N=32 materialized final-memory reduction, full-copy to shared-document:
  `54.531038401%` (display `54.5%`).
- Numerical-oracle maximum relative L2: `0.0017432502481433169`, bounded by
  the preregistered `0.005` threshold on all eight ranks.
- Live-binding closure: 144 selected rows, 12,960 storage rows, 3,840 clone
  edges, 24 stable phase artifacts, 96 primary calls, and zero global primary
  memory-hook events.
- All 24 phase paths use stable `primary/raw/rank-*` publication paths; no
  `.forkaudit-rank-*` temporary staging path remains.
- The result-sink bytecode authority contains exactly 31 authorized
  `.cpython-311.pyc` files and 13 descendant directories, derived from 31
  Python sources in the 34-row source ledger.
- The final terminal tree contains 1,367 nodes. Primary reached `99_done`;
  both root and formal completion markers are empty; failure ledger is absent.

## Cryptographic anchors

- Primary summary: `d49f25ddef31d8a0afffeccba855b05123210b1b1ccdcdc364ebef56ae3e298c`
- Primary scientific ledger: `ffdd40f02d114ce2a50ddd042701ae4282177de87c3e32875b90bc598e66fd13`
- Formal aggregate: `feae2481a4cf9e6a45135896741b08a4529d9b264a63622e5e8004cfe766c1fb`
- Formal terminal ledger: `d814ffa69d9bb1fcb502fa8704edb351606cf1ccba147bd1376caa1ee98f4a10`
- R40 aggregate: `40e1b45d715a20222fff6d85344d8fbbd06dbeae6a7d0056462e5d90af53d4fa`
- R40 CUDA smoke: `2ac8aa3c6b8324d10ebd01e7bded1631e11d7e1aeb058899204969ffb666a4e9`
- Terminal closure: `7ba11f6a71e8558eabd82af742e7f4c901ba8ceb9ce9ccd6a3d15e3f9c9610bf`
- Terminal tree: `6aadf2d4e066f0e78978c6e216be3ef1ad34f46959f74cba3be79dde91a1f72a`
- V29 source ledger: `4d0563a99997a6d2c0a76ee6694b195599fff0caaa1119d22b4e78ad3ad489b0`
- V29 archive: `893202582f3cac7ef9f8b61fc2d5c574c7609c51aa811cf518c488a1f1efd297`

## Evidence boundary

V29 is the fresh admissible successor. V26, V27, and V28 remain inadmissible
post-science governance failures and must not be cited or pooled. This mirror
contains the exact compact audit extraction, not all remote shards and terminal
products; full replay still requires the immutable remote result tree.

