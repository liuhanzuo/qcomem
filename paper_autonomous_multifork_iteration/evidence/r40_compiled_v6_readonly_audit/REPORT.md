# R40 read-only audit of R39 primary compiled-dispatch v6

## Verdict

**HOLD.**  The package is internally consistent and its exact-rank/call/hash
closure is substantially stronger than the current paper evidence, but it is
not sufficient to upgrade target 5 to “per-call compiled-binary and explicit
autotuning provenance complete.”  No formal GPU run should be started from v6
for that stronger claim.

V6 can support a narrower statement after a successful authorized run: in one
frozen, honest in-process execution, every registered formal attention call
reached exactly one observed Triton `CompiledKernel` Python launcher, and the
corresponding final cache bundle plus normalized launch metadata were hashed
and replayed.  It does **not** independently attest a driver/device binary
launch, all autotuner choice fields, or a malicious producer/runtime.

This audit did not edit v6, use a GPU, contact QS, use SSH, or run a formal
experiment.  All adversarial probes wrote only to temporary CPU directories.

## Re-executed checks

| Check | Independent result |
|---|---|
| v6 `source-code.sha256` | PASS, every listed file |
| `dependency-files.sha256` | PASS in the source repository |
| focused source-fixture ledger | PASS |
| archive SHA-256 | PASS: `306daba7b79b045a306f0b22d6434143dd568cf1f3b6af7114ad1a4ebe1d6f82` |
| archive safety | PASS: 260 members, 228 regular files, no absolute/`..` paths, duplicates, symlinks, hard links, or devices |
| all four ledgers from a fresh archive extraction | PASS |
| v6 unit suite, source tree | PASS: 13/13, zero skips |
| v6 unit suite, fresh archive extraction | PASS: 13/13, zero skips |
| Python compilation and shell syntax | PASS; pycache was redirected outside the target/archive tree |
| frozen TF-5.14.1 route static replay | PASS on the archived runtime source at SHA-256 `688d9a8f...9716b9f` |
| detached 162-test RR2 suite on this Mac | 161 PASS, one environment skip because real Transformers is unavailable |

The local one-skip result is not treated as a formal pass.  The H20 launcher
correctly rejects any skip and specifically requires the real-Transformers
test to report `ok` before a shard can start
(`r39_primary_formal_h20.sh:169--191`).

The focused archive ledger is intentionally meaningful only after extraction:
its 11 archive-projection paths are absent from the smaller source-tree view,
but all 11 pass in the fresh extracted package.  This is packaging structure,
not an archive failure.

## What the implementation really proves

The attention wrapper opens a context around the actual imported vLLM Python
call.  Inside that context, the global Triton `CompiledKernel.run` property is
wrapped, the selected kernel metadata is matched to exactly one cache bundle,
and the bundle contents, PTX, cubin, metadata, and normalized configuration are
hashed (`r39_compiled_dispatch_receipts.py:378--398, 734--757`).  The compact
recorder requires exactly one such event for every primary attention call and
the replay closes exact cell/rank call geometry.  The finalizer then loops over
all ranks 0--7, checks the receipt's shard path and SHA-256 against the actual
shard, replays the receipt and shard together, copies the cache/sources, emits
a terminal ledger, and creates `COMPLETE` last
(`r39_primary_finalize.py:263--370`).

The GDN side is explicitly eager rather than compiled.  V6 source-binds the
accelerate wrapper, the MoE forward, both chunk/recurrent rules, the in-place
conv update, and both qcomem functional cache rebinds; exact mutually exclusive
route counts are required.  Its claim boundary correctly excludes compiled GDN
and underlying ATen/CUDA identity.

Accordingly, ordinary failures are well handled:

- a missing rank, wrong rank, wrong shard path/hash, missing call, extra call,
  wrong call shape, wrong GDN route count, or direct artifact-byte tamper fails;
- pre-factorial priming/warmup alone cannot satisfy the formal-cell closure;
- a CUDA-graph replay that bypasses the per-call Python `CompiledKernel.run`
  hook would produce zero launcher events and fail, although v6 does not
  positively identify graph nodes;
- the frozen primary shard separately requires `dense_fallback_calls == 0`.

## Blocking findings

### H1 — no explicit execution-authorization gate (critical operational)

The launcher starts with fixed stage/result roots and a non-overwrite check,
but it has no required `...EXECUTION_AUTHORIZED=yes` predicate
(`r39_primary_formal_h20.sh:1--40`).  A direct `bash` invocation therefore
begins the run without an in-package confirmation of the user's authorization.

Minimum repair: require a narrowly named environment value before creating
`RESULT_ROOT`, and record that authorization-gate success without storing any
credential or user secret.

### H2 — selected cache artifact is not independent device-binary attestation (major claim boundary)

The compiled wrapper calls `recorder.record_compiled_kernel(kernel)` and only
then calls `original_launcher` (`r39_compiled_dispatch_receipts.py:748--755`).
A completed formal process implies the Python launcher returned normally, but
the receipt contains no post-launch/device/driver module identity.  A no-op or
decoy launcher that returns normally is not distinguished at this layer.

This does not invalidate the narrow “honest in-process selected launcher/cache
artifact” claim.  It does invalidate wording such as “the cubin was
independently attested as executed on every call.”

Minimum repair: either (a) keep the paper claim at selected Python launcher and
hashed cache-artifact/configuration scope, or (b) add a separately justified
post-launch/device-level witness.  A `post_launcher_returned` bit alone is
useful failure plumbing but is not device attestation.

### H3 — vLLM/Triton dispatch source is not source-bound (major provenance)

The source-binding set contains only the Transformers/qcomem GDN callables.
`vllm_module.unified_attention`, `CompiledKernel.run`, and `Autotuner.run` are
imported and patched but not added to the source snapshot
(`r39_compiled_dispatch_receipts.py:605--707`).  The inherited runner pins
package versions and the callable name, which is useful, but same-version file
drift or a decoy compiled call followed by an unobserved fallback remains
outside detached replay.

Minimum repair: source-bind and snapshot the exact vLLM unified-attention
callable plus the Triton launcher/autotuner implementation files, with frozen
module/qualname/file hashes and exact schema checks.

### H4 — explicit autotuning choice is not verified (major claim boundary)

The recorder serializes `best_config.kwargs`, `num_warps`, `num_stages`, and
`num_ctas` (`r39_compiled_dispatch_receipts.py:400--417`).  Replay checks only
the final event's three numeric fields against the compact configuration; it
never validates `selected_kwargs`.  Counterexample CE1 changes
`selected_kwargs` to impossible/sentinel values while replay still passes.

Minimum repair: define the exact allowed autotune-event schema and bind every
selected kwarg to the selected artifact metadata/launcher arguments.  If the
fixed path emits no autotuner event, state “no autotuner observed; selected
compiled metadata bound” and do not claim an explicit autotuning decision.

### H5 — v6 observer delta lacks a frozen-environment integration smoke (major readiness)

The 13-test suite checks the route counter logic and searches the hook source
for the required assignments, but it does not install v6 into the actual
PyTorch-2.11/Transformers-5.14.1/vLLM-0.26/Triton-3.6 environment.  The formal
preflight parses the frozen Transformers source and then proceeds to the full
eight-rank runner (`r39_primary_formal_h20.sh:194--214`).  This is precisely the
class of boundary at which v5 previously failed.

Minimum repair: before any model load or scientific shard, run a no-CUDA
frozen-environment smoke that installs/restores the hook and proves instance
binding for chunk, recurrent, in-place conv, functional conv rebind, recurrent
rebind, vLLM unified attention, `CompiledKernel.run`, and `Autotuner.run`.

### H6 — outer failure provenance is incomplete (moderate readiness)

The inner immutable launcher has its own failure markers, and successful output
has good terminal-ledger/`COMPLETE` ordering.  The outer wrapper itself has no
supervisor transcript or `ERR` trap.  A failure in outer preregistration,
static/v6 preflight, or the finalizer leaves a partial non-overwritable root but
no outer phase receipt or terminal hash of that partial attempt.

Minimum repair: add an outer phase variable, persistent stdout/stderr log, and
an `ERR` trap that writes a non-overwriting failure receipt plus a partial-file
ledger.  Never write `COMPLETE` on failure.

## CPU counterexamples

`counterexamples.py` reruns five probes against the unmodified v6 verifier;
all five are accepted.  Exact output is frozen in
`counterexample_results.json`.

1. **CE1:** arbitrary `selected_kwargs` with matching warp/CTA/stage values.
2. **CE2:** self-consistent substitution from the original selected bundle to
   an unrelated decoy kernel bundle.
3. **CE3:** contradictory unknown fields claiming zero device launches, dense
   fallback, and a bogus vLLM source hash; outer receipt/scope schemas ignore
   them.
4. **CE4:** self-consistent receipt/shard relabel to another rank without any
   process/GPU identity in the compiled layer.
5. **CE5:** duplicate, unreferenced rows in every compact table.

CE2 and CE4 require self-consistent producer-side rewriting.  They therefore
demonstrate the current trusted-producer boundary; they do **not** show that a
simple missing rank or a casual postcapture byte edit passes the full finalizer.
Those simpler cases are rejected.

## Minimal non-overwriting v7 repair plan

1. Add the explicit authorization gate and outer failure transcript/ledger.
2. Source-bind vLLM unified attention and Triton launcher/autotuner files.
3. Add the actual frozen-environment no-CUDA hook installation/restoration
   smoke before any shard.
4. Make receipt/scope/table schemas exact; reject unreferenced/duplicate rows.
5. Validate the full selected autotune configuration, or explicitly freeze the
   claim to “no autotuner observed + selected compiled metadata.”
6. Add post-launch-return receipt semantics and keep device-binary attestation
   out of the claim unless a genuinely independent device witness is added.
7. Bind the compiled receipt to the inherited GPU-assignment row/UUID and
   require eight unique rank/process/GPU tuples in the finalizer.
8. Independently rerun this audit on the fresh archive.  Only then request
   explicit permission for the one-shot 8×H20 execution.

Until those repairs pass, retain the current paper's partial dispatch wording.
Do not change the abstract/results/table to “seven of seven complete” on the
basis of v6.
