# R33 executor preparation validation

This directory contains execution preparation only. No R33 held-out fault was run locally and no QS resource was used by the executor-preparation agent.

Frozen inputs verified before mapping:

- `FAULTS.json`: `b1f4d6c544c30fccc32370a03e170aee38596a370d02c0db4a6748c83cc34dff`
- `PROTOCOL.md`: `b85995e180732588ac6ee09fc33181d9c276980795a7288a23feb4c94ad3925c`
- `MANIFEST.sha256`: `132b794be970a61025dd1d63f26e9b0fbc978b00f2a9f94bf5864f5bb8f8c548`
- reused fixed-stack execution input: `5a522e48650e3010621e6e06c7c8bbab67c074bca1b6a6a0c70aa50133b4e98d`
- generated formal protocol: `7d0cc087b6b529aa41a8003cf748121b3adedc7433cb6c75dbde50da5ba62fb7`

Static validation command:

```bash
PYTHONPATH=paper_autonomous_multifork_iteration/scripts \
python3 -m unittest -v \
  paper_autonomous_multifork_iteration/scripts/r33_test_executor_core.py \
  paper_autonomous_multifork_iteration/scripts/r33_test_fault_mapping.py
```

Result on 2026-08-25: `Ran 14 tests ... OK`. The tests cover the exact five author-row hashes and rank/policy/gate mapping, clean-before-mutant gating, correct HF03 discarded-first ordering, all five expected-primary synthetic replays, missing/tampered sidecars, earlier unrelated failure, missing rank artifact, sentinel handling, mutation restoration, receipt-chain tampering, and strict aggregate refusal.

Candidate-free clean-only local validation command:

```bash
PYTHONPATH=paper_autonomous_multifork_iteration/scripts \
python3 paper_autonomous_multifork_iteration/scripts/r33_clean_only_dry_run.py \
  --output-dir /tmp/r33-clean-only-parent-TTH3qV/out

(cd /tmp/r33-clean-only-parent-TTH3qV/out && sha256sum -c artifacts.sha256)
```

Result: `status=passed`, eight archived artifacts verified `OK`, `fault_module_loaded=false`, `faults_executed=false`, and `scientific_result=false`. This is an engineering validation of gate/receipt/cleanup mechanics, not paper evidence.

Operational behavior of the formal executor is fail closed:

1. Each rank rebuilds and disposes a matched clean case.
2. A separate `python -I` replay validates byte-bound clean records without importing candidate modules.
3. Only a passing clean replay authorizes that rank's mutant.
4. The mutant is rebuilt fresh, executed to the full eight-output horizon, restored, disposed, and replayed.
5. Any unregistered exception or cleanup mismatch produces `operational-invalid.json` and a nonzero rank exit.
6. The launcher does not run the aggregate after any rank failure.
7. The aggregate requires exactly five byte-bound pair artifacts and preserves caught and escaped outcomes.

The formal executor has not been GPU-executed in this preparation record. HF01 and HF03 retain explicit operational-invalidity risks in `MAPPING.json`; those risks are not converted to successful detections.
