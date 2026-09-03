# R40 independent live-binding formal local result

Status: **PASS**.  The non-overwriting local launcher completed naturally on
2026-08-27.  It touched no QS or GPU resource.

## Frozen provenance

- preregistration SHA-256:
  `3cb819008191370c8fd1db1a203d6ca288d8ce7cabb46ca8e3946560aed87511`
- 22-file executed-source ledger SHA-256:
  `e5d059091c4c72a348867e13a56be83fd5e48a9f2a5f696e1705c0ece79634c3`
- freeze receipt SHA-256:
  `96029139106abcb7f7cbfe542bf506e20f1462f5c9ae9b909006c82aadd4acd3`
- formal campaign JSON SHA-256:
  `f7809358ece681f39c089ca8906e4450381f4d909eb648d367e749c8831fa923`
- independent verification JSON SHA-256:
  `f4bfe3d5038b5fd11c132568040df735823915756f98e69ca4211998d3c38454`
- seven-row formal terminal ledger SHA-256:
  `75367fd70e7eff7abf8bdbdacb19d4a066bc57c10d22308e0adda023357b10d4`

The preregistration and source ledger were frozen before campaign outputs.
All 22 frozen files were reverified after execution; all seven terminal-ledger
rows were independently rehashed successfully.

## Outcome

| Fault | Actual binding operation | Matched clean | Mutant | Fail-closed codes |
|---|---|---:|---:|---|
| R40-LB01 | coherent same-geometry handle swap | pass | rejected | `challenge_response_mismatch` |
| R40-LB02 | stale handle after semantic-owner rebind | pass | rejected | `challenge_response_mismatch` |
| R40-LB03 | one-way cross-layer handle substitution | pass | rejected | `challenge_response_mismatch`, `storage_relation_mismatch` |
| R40-LB04 | request-private role bound to persistent base | pass | rejected | `challenge_response_mismatch`, `storage_relation_mismatch` |

The aggregate is four of four matched clean controls accepted and four of four
fixed live-handle mutants rejected.  All eight producer PIDs were distinct;
every lane used separate oracle and observer PIDs; all clean/mutant oracle
projections matched; exactly four faults changed live tensor references; and
zero fault changed a serialized schema or semantic label.  The independent
standard-library replay reproduced all eight decisions exactly.  Four unit
tests also passed.

## Environment and boundary

The formal fixture used Python 3.9.6, PyTorch 2.8.0, and CPU tensors on an
Apple-arm64 macOS host (`cuda_available=false`).  This result validates the
source/process-separated challenge mechanism against four fixed live-handle
faults.  It does not independently attest the current Qwen/H20 producer, does
not execute a model, does not resist a malicious producer that controls
semantic registration, does not attest OS/driver/allocator/runtime state, and
does not establish a detection rate or fault-set completeness.  A real-model
closure still requires an independent pre-binder semantic-registration hook
and a fresh authorized H20 run.

