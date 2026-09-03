# R30 expanded oracle evidence

The local directory contains the frozen producer/reference protocol, exact
pre-execution manifests, the captured manifest, NumPy result, raw-file ledger,
terminal ledger, compact validation report, and all 272 numerical sidecars
(140,199,936 bytes).  The sidecars were copied byte-for-byte from the immutable
new-output run directory and every local SHA-256 was checked against
`raw-artifacts.sha256`.  `raw/capture-manifest.json` is an exact copy of the
root capture manifest so its relative `sidecars/` receipts resolve locally.

Run the candidate-import-free CPU/NumPy replay from the paper directory with:

```bash
python3 evidence/r30_expanded_oracle_sweep/r30_expanded_oracle_reference.py \
  --capture-manifest evidence/r30_expanded_oracle_sweep/raw/capture-manifest.json \
  --preregistration evidence/r30_expanded_oracle_sweep/preregistration.json \
  --output /tmp/forkaudit-r30-local-replay.json
```

All 44 clean and 44 seeded-control gates reproduce locally.  Floating-point
metrics can differ in their last digits across NumPy/BLAS platforms; the
archived `oracle-result.json` remains the canonical execution-environment
result, while local replay must preserve the preregistered decisions and
tolerances.

The result is deliberately conditional on captured operator inputs.  It is not
an honest-capture proof, end-to-end model oracle, unseen-fault estimate, or
cross-stack result.
