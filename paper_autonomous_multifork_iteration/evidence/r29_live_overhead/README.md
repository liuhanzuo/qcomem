# R29 live overhead package

This directory retains the immutable scientific design and the full execution
history of the paired H20 experiment for the live request-step increment of
the existing ForkAudit capture and ownership/receipt path.

`preregistration.json`, `source-code.sha256`, and `readiness.md` are the v1
pre-attempt records. Attempt a failed during the first common document prefill,
before either comparison arm or any candidate result. Its exact disposition,
log hashes, inference-mode root cause, and the separately discovered arm
fairness defect are recorded in `pre-second-execution-amendment-v2.json`.

`source-code-v2.sha256` and `readiness-v2.md` froze attempt b. That formal GPU
execution completed one discarded warmup and five measured pairs, and its
runner marked the result scientifically valid and formally eligible. The
formal result and raw artifacts are immutable inputs to offline verification.

The first offline replay then failed because it required the derived
`rebound_tensor_count` field on a legal raw request-GDN witness. No replay JSON
or terminal stage was produced. This verifier-only bug, the already-read
outcome disclosure, and the prohibition on a GPU rerun are frozen in
`postexecution-replay-only-amendment-v2.json`.

`source-code-replay-v2.sha256` is the only executable ledger for the repaired
offline verifier. `readiness-replay-v2.md` gives the exact CPU-only command and
immutability gates. Do not cite the overhead result until replay v2 succeeds
against the frozen formal result and the new raw/replay ledger is independently
hash-closed.
