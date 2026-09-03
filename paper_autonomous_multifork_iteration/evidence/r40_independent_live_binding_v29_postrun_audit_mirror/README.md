# R40 live-binding V29 post-run audit mirror

Status: **PASS; fresh admissible formal result**.

This directory is a compact, read-only local mirror of the authoritative
post-run audit produced from the completed remote result. It does not modify
the frozen V29 source package or the remote result tree.

- QS Job / Trial: `256220 / 1936087`
- Pod: `qs-256220-1936087-ai-1482497-master-0`
- Run ID: `71391b1a7ce85c4dfa8beb18f3c2189a`
- Result: `r40-v29-result-pycache-whitelist-fix-20260902a`
- Scientific outcome: `valid_positive`
- Authoritative remote audit: `/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_debug/qcomem-r40-v29-result-pycache-whitelist-fix-8h20-20260902a/v29-postrun-audit-attempt2.json`
- Frozen source archive SHA-256: `893202582f3cac7ef9f8b61fc2d5c574c7609c51aa811cf518c488a1f1efd297`
- Source-ledger-file SHA-256: `4d0563a99997a6d2c0a76ee6694b195599fff0caaa1119d22b4e78ad3ad489b0`

The formal launcher completed successfully before the independent post-run
audit. The first audit invocation failed only because its checker compared the
observed oracle maximum to `0.005` as though `0.005` were the observed maximum.
That was a checker defect, not a formal-run failure. Attempt 2 checked the
declared threshold as exactly `0.005` and the observed vector maximum as
`0.0017432502481433169 <= 0.005`; it passed.

`POSTRUN_INDEPENDENT_AUDIT.json` is the exact JSON copied from the authoritative
attempt-2 artifact. `POSTRUN_INDEPENDENT_AUDIT_REPORT.md` records its bounded
interpretation. `COMPLETE` is a local textual mirror marker, not a claim that
the remote root's empty `COMPLETE` file was copied byte-for-byte.

