# R40 v20 preflight-environment isolation successor

Status: **HOLD_PENDING_FRESH_AUDIT_AND_H20**. This is a controlled
successor to v18 that repairs one deterministic terminal-governance defect.
The fixed producer writes
`compiled-dispatch-capture/rank-{0..7}/invocation.json`, but v18 omitted those
eight files from its terminal expected-path set. V18 was blocked before formal
science launch and must not be used as scientific evidence. V19 has completed
its pre-freeze package and clean-stage validation, but it has not yet been
independently approved or executed on H20/CUDA; it is not H20 evidence.

The canonical v6 base archive remains fixed at SHA-256
`306daba7b79b045a306f0b22d6434143dd568cf1f3b6af7114ad1a4ebe1d6f82`.
Cross-platform enumeration exposes 260 members: 130 logical members and 130
regular AppleDouble `._*` metadata members. GNU tar/Python materialize the
latter, while the original macOS extraction consumed them as metadata. V19
excludes only a basename beginning `._` after exact AppleDouble validation,
requires a retained logical companion with the same mode, verifies every
retained byte/mode/type against `v6-clean-members.json`, and requires zero
`._*` anywhere in the final exact stage tree. Unsafe paths, duplicate names,
links, special nodes, false AppleDouble files, missing companions, extra nodes,
and any source-ledger/archive/tree drift fail closed before result action.

V19 keeps 9 of the 10 v18 payload files byte-identical. The sole controlled
change is `executed_source/r40_tree_closure.py`; it adds the eight fixed
invocation paths to terminal closure, validates each invocation's exact schema
and rank, and cross-binds its runner SHA-256, argv, shard SHA-256, and canonical
argv SHA-256 to the corresponding formal receipt's `execution_binding`.
Preregistration, CUDA smoke, rank entrypoint, binding hook/verifier, lineage
mechanism, and scientific finalizer remain byte-identical to v18. The external
runner and builder remain fixed at
SHA-256 `9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775`
and `546efd59e2833034bc2e24d4cc0e6077f5a408275e359af43cd96f7f71cad16e`.
The 10-file current manifest is pinned at
`6007133c45dc24f51f1482598e17c739eb8eae7f4579bed6eb101d3a249896ab`.
The v18/v19 controlled-diff record is pinned at
`fc81a6a0b6cbac46708ef4f9680db9829ba7c2e22f437d9e8f8629be81c76f9c`
with payload seal
`73fae1086b003a4233546b6d6c179bf0cac4e44445efb790e9aa54801d44b6f6`.
Verification checks schemas, path order, file count, every current hash and
size, and the exact 9-unchanged/1-controlled-change boundary without opening
or requiring a sibling v18 package.
The mechanism preserves the hash-bound passive-lineage integration around the
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
the fixed R40 subtrees. The projection now includes the fixed per-rank
`invocation.json` files, and their content must match the already validated
formal receipt binding. It passes every reconstructed path explicitly as an
`--expected-existing-path`; `prepare` independently rescans and requires exact
set equality. Thus a regular file or empty directory inserted even before
expectation publication cannot be blessed by a dynamic snapshot. Symlinks,
hardlinks, FIFOs, sockets, devices, missing nodes, malformed or mismatched
invocations, and extra regular files or directories fail closed.

`cuda-smoke.json`, `aggregate.json`, `terminal-closure.json`, `COMPLETE`, and
`terminal-tree.json` are each published with exclusive creation. The final tree
ledger is written only after the sealed expectation and empty completion marker
exist, then the whole tree and exact final counts are rescanned.

`formal/launch_h20.sh` is non-overwriting and atomically one-shot. Without both
`R40_H20_EXECUTION_AUTHORIZED=yes` and `R40_V20_FRESH_AUDIT_APPROVED=yes` it
exits before marker/result action. It additionally requires externally supplied,
lowercase SHA-256 approvals for the frozen source ledger, deterministic v20
overlay archive, and fixed v6 archive. It verifies the sealed stage receipt and
the entire clean stage before atomic one-shot ownership; in-package source plus
ledger cannot self-authorize. An authorized run
verifies sources, makes staging read-only, executes the PyTorch 2.11+cu129 CUDA
smoke before science, then finalizes and terminally rehashes sources. The
launcher builder opens the v6 launcher exactly once with `O_NOFOLLOW`, checks
stable `fstat` metadata around a descriptor read, and uses that same byte
snapshot for SHA-256, strict UTF-8 decoding, and transformation. It never
reopens the v6 pathname. The archive is staging material, not GPU evidence.

The complete v19 package suite passed 87/87 with zero skips on both the source
tree and a pre-freeze exact clean stage; the static audit passed 131/131. The
deterministic source/archive freeze and fresh independent audit remain external
gates and must be recorded before formal launch. The inherited same-Python
raw/clean replay enumerates 162 tests per tree; the formal launcher continues
to require 162/162 with zero skips before any GPU action. Historical v18
validation results are not silently promoted to v19 results.

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
python3 -B scripts/stage_v6_clean.py prepare \
  --output-root /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r40_v19_clean_20260828a \
  --v6-archive <canonical-v6-archive> --overlay-archive <v19-overlay-archive> \
  --clean-ledger v6-clean-members.json --exclusion-ledger v6-appledouble-exclusions.json \
  --expected-v6-sha256 306daba7b79b045a306f0b22d6434143dd568cf1f3b6af7114ad1a4ebe1d6f82 \
  --expected-overlay-sha256 <externally-approved-v19-archive-sha256>

env -u R40_LAUNCHER_HANDLER_SELFTEST -u R40_LAUNCHER_ATOMIC_GATE_SELFTEST \
  R40_H20_EXECUTION_AUTHORIZED=yes R40_V19_FRESH_AUDIT_APPROVED=yes \
  R40_V19_APPROVED_SOURCE_LEDGER_SHA256=<externally-approved-ledger-sha256> \
  R40_V19_APPROVED_ARCHIVE_SHA256=<externally-approved-v19-archive-sha256> \
  R40_V19_APPROVED_V6_ARCHIVE_SHA256=306daba7b79b045a306f0b22d6434143dd568cf1f3b6af7114ad1a4ebe1d6f82 \
  R40_V19_CANONICAL_V6_ARCHIVE=<canonical-v6-archive> \
  R40_V19_OVERLAY_ARCHIVE=<v19-overlay-archive> \
  bash /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r40_v19_clean_20260828a/paper_autonomous_multifork_iteration/evidence/r40_independent_live_binding_v19_terminal_closure_fix/formal/launch_h20.sh
```
