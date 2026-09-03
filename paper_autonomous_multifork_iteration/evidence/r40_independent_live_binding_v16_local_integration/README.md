# R40 v16 exact terminal closure and CUDA-gated staging

Status: **HOLD_PENDING_FRESH_AUDIT_AND_H20**. Local CPU, static, and formal-preflight checks are complete; fresh independent audit and H20/CUDA execution remain outstanding. Science is ineligible until both gates close and the same-run CUDA smoke succeeds.

V16 preserves the hash-bound passive-lineage mechanism around the
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

Freshness is cumulative, not merely relative to the current live group. Before
the first generation callback, strong references retain exactly 540 initial
endpoints: 60 persistent endpoints plus 8 requests times 60 endpoints. Callback
0 contributes 60 new endpoints, bringing the retained historical universe to
600; every later successful callback contributes another 60. A superseded
object or storage allocation therefore cannot be rotated into another request
or become acceptable through allocator identity reuse.

Capture finalization and terminal closure walk lexical paths with `lstat` and
`scandir`. A root is rejected before resolution when it is a symlink, and its
canonical path must equal its lexical absolute path. Every terminal output is
normalized, must contain no `..`, and must be strictly inside that root before
any write. The closure binds an exact path whitelist, exact node/regular-file/
directory counts, and an exact per-file content schema. Before `prepare`, the
launcher reconstructs the predetermined existing-path list from fixed preflight,
log, and stage paths; the primary scientific ledger; the private-model-view
manifest; the formal-binding terminal ledger; the exact capture projection; and
the fixed R40 subtrees. It passes every reconstructed path explicitly as an
`--expected-existing-path`; `prepare` independently rescans and requires exact
set equality. Thus a regular file or empty directory inserted even before
expectation publication cannot be blessed by a dynamic snapshot. Symlinks,
hardlinks, FIFOs, sockets, devices, missing nodes, and extra regular files or
directories fail closed.

`cuda-smoke.json`, `aggregate.json`, `terminal-closure.json`, `COMPLETE`, and
`terminal-tree.json` are each published with exclusive creation. The final tree
ledger is written only after the sealed expectation and empty completion marker
exist, then the whole tree and exact final counts are rescanned.

`formal/launch_h20.sh` is non-overwriting and atomically one-shot. Without both
`R40_H20_EXECUTION_AUTHORIZED=yes` and `R40_V16_FRESH_AUDIT_APPROVED=yes` it
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
  R40_H20_EXECUTION_AUTHORIZED=yes R40_V16_FRESH_AUDIT_APPROVED=yes \
  R40_V16_APPROVED_SOURCE_LEDGER_SHA256=<externally-approved-ledger-sha256> \
  R40_V16_APPROVED_ARCHIVE_SHA256=<externally-approved-archive-sha256> \
  bash formal/launch_h20.sh
```
