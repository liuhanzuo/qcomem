# R40 v14 local closure and CUDA-gated staging

Status: **HOLD_PENDING_FRESH_AUDIT; local closure PASS; H20/CUDA not run**. Even after audit release, science is ineligible until the same-run CUDA smoke passes.

V14 integrates the final hash-bound passive-lineage mechanism around the
unchanged real builder. It observes every persistent-rooted `aten.clone`/copy
edge, requires exact N×60 direct clone closure for materialized policy, supports
borrowed zero-edge exact alias, holds exact source/destination objects, seals
the private dispatch ledger, and rejects wrong-source offset views, returned
views, unused edges, and forged events.

The faithful 30-layer/8-request CPU E2E test exercises install, unchanged
builder signature, 480 edges, setup/transition/generation phase tuples, atomic
artifact write/read/hash binding, lifecycle, restore, exact build count,
primary-memory absence counters, tamper cleanup, and orphan rejection. Exact
row universe/schema/descriptors/intervals and aggregate predicates are tested.

The hash-bound real `_gpu_round_robin_generate` call path is wrapped as well.
For all 64 calls (eight rounds by eight requests), the verifier records an
ordered 60-coordinate functional-rebind edge from the prior live object to the
new object. The transition phase therefore closes after call 1, while the final
phase requires eight rebinds per request; missing, extra, repeated, or
out-of-order calls fail closed.

`formal/launch_h20.sh` is non-overwriting and atomically one-shot. Without both
`R40_H20_EXECUTION_AUTHORIZED=yes` and `R40_V14_FRESH_AUDIT_APPROVED=yes` it
exits before marker/result action. It additionally requires externally supplied,
lowercase SHA-256 approvals for both the frozen source ledger and deterministic
archive; in-package source plus ledger cannot self-authorize. An authorized run
verifies sources, makes staging read-only, executes the PyTorch 2.11+cu129 CUDA
smoke before science, then finalizes and terminally rehashes sources. The
archive is staging material, not GPU evidence.

LB01--LB04 remain local mechanism tests and are not H20 faults, a production
fault campaign, or formal sensitivity evidence.

The sealed in-process lineage mechanism is scoped to integration bugs in the
hash-bound honest producer. It is not a security boundary against malicious
same-process reflection or mutation.
Per-call isolation is an endpoint guarantee at the real generation callback;
it does not claim visibility into an alias created and fully removed inside an
opaque forward before that callback.

After an independent operator records the two approved hashes, the exact
preflight environment is:

```bash
env -u R40_LAUNCHER_HANDLER_SELFTEST -u R40_LAUNCHER_ATOMIC_GATE_SELFTEST \
  R40_H20_EXECUTION_AUTHORIZED=yes R40_V14_FRESH_AUDIT_APPROVED=yes \
  R40_V14_APPROVED_SOURCE_LEDGER_SHA256=<externally-approved-ledger-sha256> \
  R40_V14_APPROVED_ARCHIVE_SHA256=<externally-approved-archive-sha256> \
  bash formal/launch_h20.sh
```
