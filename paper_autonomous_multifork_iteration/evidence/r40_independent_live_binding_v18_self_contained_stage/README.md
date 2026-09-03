# R40 v18 self-contained clean staging for unchanged v16 science

Status: **HOLD_PENDING_FRESH_AUDIT_AND_H20**. This is a staging-only
successor to v17 that closes its clean-stage sibling dependency and launcher
snapshot TOCTOU blockers. Fresh independent audit and H20/CUDA execution
remain outstanding; this package is not H20 evidence.

The canonical v6 base archive remains fixed at SHA-256
`306daba7b79b045a306f0b22d6434143dd568cf1f3b6af7114ad1a4ebe1d6f82`.
Cross-platform enumeration exposes 260 members: 130 logical members and 130
regular AppleDouble `._*` metadata members. GNU tar/Python materialize the
latter, while the original macOS extraction consumed them as metadata. V18
excludes only a basename beginning `._` after exact AppleDouble validation,
requires a retained logical companion with the same mode, verifies every
retained byte/mode/type against `v6-clean-members.json`, and requires zero
`._*` anywhere in the final exact stage tree. Unsafe paths, duplicate names,
links, special nodes, false AppleDouble files, missing companions, extra nodes,
and any source-ledger/archive/tree drift fail closed before result action.

V18 preserves the complete v16 scientific payload byte-for-byte, including
`preregistration.json`, `absorbed-lineage.json`, and all eight
`executed_source/*.py` files. The external runner and builder remain fixed at
SHA-256 `9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775`
and `546efd59e2833034bc2e24d4cc0e6077f5a408275e359af43cd96f7f71cad16e`.
The frozen 10-file manifest and sealed v16/v18 equivalence record have exact
SHA-256 pins. Verification checks their schemas, payload seal, path order,
file count, and every current payload hash and size without opening or
requiring a sibling v16 package.
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
`R40_H20_EXECUTION_AUTHORIZED=yes` and `R40_V18_FRESH_AUDIT_APPROVED=yes` it
exits before marker/result action. It additionally requires externally supplied,
lowercase SHA-256 approvals for the frozen source ledger, deterministic v18
overlay archive, and fixed v6 archive. It verifies the sealed stage receipt and
the entire clean stage before atomic one-shot ownership; in-package source plus
ledger cannot self-authorize. An authorized run
verifies sources, makes staging read-only, executes the PyTorch 2.11+cu129 CUDA
smoke before science, then finalizes and terminally rehashes sources. The
launcher builder opens the v6 launcher exactly once with `O_NOFOLLOW`, checks
stable `fstat` metadata around a descriptor read, and uses that same byte
snapshot for SHA-256, strict UTF-8 decoding, and transformation. It never
reopens the v6 pathname. The archive is staging material, not GPU evidence.

Local package and exact clean-stage formal-preflight suites each complete all
86 tests with zero skips. The same-Python raw/clean upstream replay enumerates
all 162 tests on each tree and reproduces exactly two raw AppleDouble errors
versus zero clean errors; this Mac lacks exact Transformers 5.14.1, so both
local 162-test runs report the same one environment-only skip. The inherited
formal launcher still requires 162/162 with zero skips before any GPU action.

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
  --output-root /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r40_v18_clean_20260828a \
  --v6-archive <canonical-v6-archive> --overlay-archive <v18-overlay-archive> \
  --clean-ledger v6-clean-members.json --exclusion-ledger v6-appledouble-exclusions.json \
  --expected-v6-sha256 306daba7b79b045a306f0b22d6434143dd568cf1f3b6af7114ad1a4ebe1d6f82 \
  --expected-overlay-sha256 <externally-approved-v18-archive-sha256>

env -u R40_LAUNCHER_HANDLER_SELFTEST -u R40_LAUNCHER_ATOMIC_GATE_SELFTEST \
  R40_H20_EXECUTION_AUTHORIZED=yes R40_V18_FRESH_AUDIT_APPROVED=yes \
  R40_V18_APPROVED_SOURCE_LEDGER_SHA256=<externally-approved-ledger-sha256> \
  R40_V18_APPROVED_ARCHIVE_SHA256=<externally-approved-v18-archive-sha256> \
  R40_V18_APPROVED_V6_ARCHIVE_SHA256=306daba7b79b045a306f0b22d6434143dd568cf1f3b6af7114ad1a4ebe1d6f82 \
  R40_V18_CANONICAL_V6_ARCHIVE=<canonical-v6-archive> \
  R40_V18_OVERLAY_ARCHIVE=<v18-overlay-archive> \
  bash /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r40_v18_clean_20260828a/paper_autonomous_multifork_iteration/evidence/r40_independent_live_binding_v18_self_contained_stage/formal/launch_h20.sh
```
