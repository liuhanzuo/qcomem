# R39 dual-producer repeat package

Status: preregistered, unit-tested, packaged, and **not launched**.

This evidence package wraps the frozen R33 independent-capture experiment in a
prospective two-producer repeat. One R39 semantic-slot census is frozen before
both producer starts; two fresh R33 runs are then executed serially on one GPU,
audited separately, and compared exactly.

Key files:

- `preregistration.json`: frozen equality rules, counts, inputs, failure policy,
  and claim boundary.
- `slot_protocol.json`: independent 180-slot census derivation.
- `scripts/verify_dual_producer_repeat.py`: fail-closed cross-producer verifier.
- `tests/test_dual_producer_repeat.py`: clean closure plus resealed content,
  relation, process-reuse, and input-binding negative tests.
- `formal/launch_dual_producer_h20.sh`: formal serial one-GPU pipeline.
- `formal/launch_existing_h20_node.sh`: path wrapper for an already-created H20
  node; it contains no QS command.
- `formal/build_execution_bundle.py`: deterministic self-contained archive
  builder.
- `source-code.sha256`: package source ledger.

The source bundle vendors the unchanged R33 producer/receiver/replay code, the
unchanged R39 census generator/auditor, and the unchanged R29 runtime adapter.
The launcher creates no resource and must only be invoked on an existing node.

The formal success condition is 2 fresh producer PIDs, 4 fresh receiver
sessions, 6 captures per producer, and exact cross-producer closure of 1,080
semantic coordinates/content digests/stable descriptors and 96,660 relation
labels. Receiver-local HMAC tokens are intentionally not compared across
processes. Tolerance is zero; no post-hoc fallback is allowed.

