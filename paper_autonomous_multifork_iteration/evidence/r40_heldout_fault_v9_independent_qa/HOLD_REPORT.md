# R40 held-out fault method v9 independent QA HOLD report

Audit date: 2026-08-28 (Asia/Shanghai)  
Decision: **HOLD -- method freeze is not eligible for operator binding or H20 execution**

This is a read-only QA report for the adjacent immutable method freeze
`evidence/r40_heldout_fault_v9_method_freeze`.  The audit used only local CPU
fixtures and temporary directories.  It did not modify v9, bind a real
operator/fault/configuration, access QS, initialize CUDA, execute on a GPU, or
read/modify the manuscript.

## Frozen-object bindings

- Method archive SHA-256:
  `d98eb61b763531bf622652e3db7724185da988f3a49332711d8b9d9ba0e3848f`
- `source-ledger.json` SHA-256:
  `a6662f99cab059bce6a259f4d082e9edf7bd02bad733dbc5a60adeb08fa57c02`
- `METHOD_FROZEN.json` SHA-256:
  `ba4abbdae73c6c87a9e35146e2ad167b54594099d05e3000e2a0cec68c9f9b0f`
- `OPERATOR_TRUST_ROOT.json` SHA-256:
  `3833a61d8320cc6ac7b510cd51268e73daedb52499a0baf7ee5522b9e9c78440`

## Blocking findings

1. **The complete packaged test set does not pass.** `test_v9.py` passes 11/11
   with zero skips, but `fixtures_v9.py` contains 36 additional executable
   tests.  Across all 47 packaged test methods, 15 methods do not pass (one
   assertion failure and 18 error events).  `METHOD_FROZEN.json` reports only
   11 tests and therefore does not describe the package's complete test
   surface.
2. **One-shot consumption is not bound to a durable directory object.**  The
   signed binding omits the terminal root's device/inode identity.  The same
   binding/nonce/attempt can be accepted again after replacing the root at the
   same pathname or after removing the nine terminal outputs from the same
   root.
3. **The executable receipt is a pre-launch pathname measurement.**  The
   launcher measures the Python pathname before creating the child and writes
   that earlier value into the receipt.  It does not obtain the executable
   identity actually consumed by the child after launch.
4. **Terminal typed-spec closure is incomplete.**  Validation accepts
   self-consistent hashes without reconstructing the actual typed argv/env
   specifications from all eight worker receipts and cross-binding that
   aggregate to provenance.
5. **The public process-launch surface is larger than the audited formal
   worker entry.**  A public `isolated_torch_probe` exposes two additional
   process-launch calls that are not enumerated by `static_audit_v9.py` and do
   not share the formal operator-binding gate.

## Passing subchecks that do not override HOLD

- Frozen archive, source ledger, snapshot, and canonical build products are
  internally hash-consistent and deterministic.
- The no-overwrite checks preserve pre-existing target bytes/inodes.
- Local SIGINT/SIGTERM fixture checks clean up already-started workers and do
  not leave orphan processes.

These are useful implementation checks, but the blockers prevent the freeze
from authorizing an external fault campaign or any paper claim.

## Required successor gate

A successor must be a new version.  It must accurately discover all packaged
tests, provide durable one-shot authority outside the replaceable terminal
root, obtain child-reported post-launch executable identity, reconstruct and
cross-bind typed worker specifications, and enumerate every process-launch
entry under one authorization policy.  It then requires a fresh independent
audit before external binding or H20 execution.
