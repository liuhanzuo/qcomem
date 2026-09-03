# R40 compiled-dispatch v11 post-run minimal mirror

Status: **read-only mirror of a completed formal result; frozen source package
left unchanged**.

The v11 source package was intentionally frozen before formal execution, so its
own `README.md`, `acceptance.json`, and `preregistration.json` remain pre-run
HOLD records. This separate directory mirrors the smallest useful set of
post-run products without rewriting that history.

Formal execution:

- Job / trial: `253976 / 1911962`
- Fixed result root:
  `/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r40-primary-compiled-dispatch-v11-20260827k`
- Frozen source archive SHA-256:
  `0013e1e458711263342b37c1a274b6a36d227a602a885201f12892a8968b3641`
- Minimal mirror archive SHA-256:
  `c9ef02c21ce782bef65dde1ad76fd18e8fda233e7d97d6fd20ea22428c99929d`

The mirror was read from the shared result mount without modifying the result.
It contains both empty `COMPLETE` markers, the formal aggregate, the primary
aggregate, the runtime preflight, the aggregate final-audit marker, the raw and
scientific ledgers, and both terminal ledgers.

Key checks:

- `formal-binding/formal-aggregate.json`: `status=pass`,
  `formal_evidence_eligible=true`,
  `target_5_status_at_declared_scope=pass`; file SHA-256
  `04b5ae63dc2f2dbe7c116a7136c2cdda2d9cab2e433b72b31d57cd28125c7a1f`.
- `primary/forkaudit-summary.json`: `passed=true`,
  `scientific_outcome=valid_positive`, `scientific_run_valid=true`, eight
  ranks, exact factorial, all-rank oracle pass, no negative reasons; file
  SHA-256
  `5221d9ae0eb12092e311929fed6269122c290baddc97b6014a69f0266e634353`.
- Root terminal ledger: 949 entries; file SHA-256
  `909d47d38ba3e37f196ceca340b4a0d2e40bbe6b8c494f63bc78286ca217fa5d`.
- Formal terminal ledger: 169 entries; file SHA-256
  `b01d76704b4155d826ebc21fdce8abe85a9ed8ce9aac64c9308feecf49b4e525`.
- Raw/scientific ledgers: file SHA-256
  `cc8a39aedd87ee196dd6424db5403c3b5ac7cc2b86c68b089dfa730989b780de`
  and
  `50097b75ea925cc4ef7b6393113e10bcdf78d1508573dac218652d0270cc4758`.
- Runtime preflight file SHA-256:
  `e4467acfbf440fff5b9a4c4ca99b46a282edea79f31d9cd44ef5d90036991651`.

This is a minimal local mirror, not a copy of the approximately 73 GB model
view or all raw shards. The independent post-run audit replayed the complete
remote ledgers and all raw records without sampling; this mirror preserves the
terminal aggregates and the ledgers that name those products.

