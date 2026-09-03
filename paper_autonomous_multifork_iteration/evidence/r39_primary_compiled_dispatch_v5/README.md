# Round-39 primary-factorial compiled-dispatch closure (`20260827e`, v5)

This non-overwriting package replaces the failed-closed `20260827d` capture
attempt without changing the scientific experiment.  Version `20260826a`
remains ineligible because its GDN verifier did not bind each semantic key to
one preregistered callable identity.  Version `20260826b` fixed that verifier
and passed an independent package audit, but its detached archive omitted
repository-relative fixtures consumed by the immutable RR2 focused tests.
That attempt stopped in pre-GPU focused tests and is not scientific evidence.

Version `20260827c` closed that fixture defect and passed both detached and
immutable-launcher focused tests, but its wrapper selected an obsolete
`preregister-probe` release digest tied to an older runner ledger.  The
launcher regenerated a manifest byte-identical to the already-existing RR2
`upstream/preregistration/release-manifest.json`, then correctly stopped before
any GPU shard because the selected expected digest was stale.

Version `20260827d` preserved the v3 immutable runner, old launcher, Round-39
scientific factors, inputs, thresholds, and execution order byte for byte.  Its
only preregistration repair was to include and select the preexisting
current-ledger upstream preregistration authority, check its raw and semantic
digests and embedded code identity before the launcher, and use its semantic
release digest.  The complete historical test-fixture projection and detached
162-plus-7 no-skip gates also passed remotely.

That d attempt then exposed a separate entrypoint instrumentation defect: the
capture wrapper sent discarded pre-factorial priming/warmup memory calls into a
recorder whose factorial rank had not yet been opened.  All eight ranks failed
closed on the same first priming-call invariant before any formal factorial
cell or receipt was produced.  Version `20260827e` changes only the rank
entrypoint capture scope and its tests: memory/witness calls are captured only
inside the dynamic scope of the immutable `_run_formal_factorial_cells` call.
Calls outside that scope still execute on the original runner path.  Inside the
factorial, the unchanged compact recorder retains exact rank, order, nesting,
and 24-cell closure checks.  The immutable runner/launcher, compiled-dispatch
hooks, compact recorder/finalizer, inputs, factors, thresholds, and execution
order remain byte-identical to d.

## Disposition of the v2 infrastructure failure

The v2 focused-test log contains exactly 162 tests and eight errors (raw log
SHA-256 `4e6c01e72db929872c8e87cafb11028ca327caaf3b8eda6a07d98b1a82c3d80a`).
All eight observed errors arise from one detached-fixture closure defect:

- two calibration tests cannot find the archived `pg19-gate-shards` tree;
- six tests first fail while opening the same nested
  `forkaudit_fp32_calibration_manifest.json`; and
- after that manifest is restored, one of those tests also consumes the
  nested `review/experiment_response_plan.json`, so v3 projects the immutable
  RR2 preregistration sidecar there rather than waiting for a second failure.

The nested response-plan fixture is deliberately the historical RR2 input with
SHA-256 `e2be05e198d6c86276f229d4e862c3579c65e311d07c219e9e9c792a390cbfbb`,
which is the identity frozen in the byte-identical runner.  It is not the later
expanded working-tree file at the same logical repository path (SHA-256
`785d59f76e7d96600e4ced13d3e6476726f88efd0dea0dfd469034b0e7beae09`).
The formal wrapper requires the projected bytes to match the authoritative
historical sidecar under `executed_inputs`, in addition to checking both
fixture ledgers.  Thus “projection” means historical repository-relative test
layout, not equivalence to the current evolving review-plan document.

No shard process, model load, or GPU scientific cell ran in v2.  Its immutable
failure record is `v2-pre-gpu-infrastructure-failure.json`; the original remote
log remains at the path and digest recorded there.  This is an infrastructure
preflight failure, not a negative compiled-dispatch result.  The canonical
RR2 source tree and the v2 package remain unchanged; all v5 layout construction
uses the copied fixture projection and fresh temporary payloads.

## Disposition of the v3 preregistration-pin failure

The `20260827c` attempt passed the detached 162-test suite, all seven Round-39
tests, the bytecode-absence gate, and the immutable launcher's independent
162-test suite.  It then stopped in phase `preregistration` at
`2026-08-27T04:15:40Z` with `release manifest differs from the externally
pinned preregistration`.  The manifest builder reported
`gpu_initialized=false`; no primary shard or scientific cell started.

Forensic comparison found exactly three semantic JSON-path differences between
the obsolete executed-input probe and the generated manifest: two copies of
the code-ledger digest and the derived frozen-identity digest.  The old and
current 34-entry ledgers differ in exactly one source entry: the immutable
runner changed from historical SHA-256 `8200b08b...e394` to the already-frozen
current SHA-256 `9da619fc...67775`.  All other manifest fields agree.

Crucially, the c-generated release manifest (raw SHA-256
`05465256c451b14a65ede6329d56cedb56e70388c4ff7ec064bdfd6d4c7f3fcb`,
semantic SHA-256
`201b15c945676db1924bcf2e197ee93a10078feda81ec0f4a8da113c56fac456`)
is byte-identical to the RR2 package's preexisting
`upstream/preregistration/release-manifest.json`.  Version d therefore does not
derive a new authority from a failed run or tune a digest to an output; it
selects the existing authority that already binds code ledger
`837f7a488d75cbedbc01e35a236a97f00b85259746e6a368b7aeec873045e94a`.
The complete evidence and exact three-path diff are machine-readable in
`c-pre-gpu-preregistration-failure.json` and
`preregistration-authority-repair.json`.

## Disposition of the v4 capture-scope failure

The `20260827d` attempt passed the detached 162-test suite, all seven then-
registered Round-39 tests, the bytecode-absence gate, the immutable launcher's
independent 162-test suite, static preregistration, private-model-view checks,
and formal GPU preflight.  Eight rank processes then started and all eight
naturally failed on their first discarded priming memory call with
`PrimaryDispatchError: cell/rank factorial binding drift`.

This was not a rank assignment mismatch.  The entrypoint had globally wrapped
`_run_clean_memory_cell`, but the immutable runner calls it once for priming and
four times for arm warmup before entering `_run_formal_factorial_cells`, where
`recorder.begin_factorial(rank)` occurs.  Thus the recorder rank was still
unset.  The eight logs contain the same exception once each; there are zero
dispatch-capture files, zero completed formal cells, zero raw shards, zero
aggregates, and no `COMPLETE`.  Exact remote paths, log hashes, failure-marker
bytes, and the evidence boundary are frozen in
`d-gpu-capture-scope-failure.json`.

The v5 gate is dynamic-scope based rather than result based: five
pre-factorial memory calls per rank are ignored by capture, the 24 formal cells
are captured, and post-factorial controls are ignored by capture.  Wrong-rank,
nested-cell, repeated-factorial, exception, and short-factorial paths still
fail closed.  The repair changes no factor, measurement, threshold, or formal
cell order.

The package freshly reruns the immutable RR2 W-run primary runner and original
launcher on the same frozen model, PG19 inputs, thresholds, factor levels, rank
assignment, and execution order. R29 remains supporting evidence and is not
retrofitted as primary evidence.

## Why a fresh primary run is necessary

The existing Round-39 compiled-dispatch result binds 120 attention calls and
540 GDN calls in the separate R29 overhead experiment. It cannot upgrade target
5 for the paper's 96-configuration primary factorial. This package therefore
routes only the eight `--stage shard` processes of the byte-identical primary
launcher through mandatory hooks. Static preparation, source replay,
aggregation, and all other phases still use the unmodified launcher and runner.

The route uses a transparent Python proxy rather than editing the frozen code
snapshot. The old launcher recursively audits its own read-only 34-file closure
against raw code-ledger SHA-256
`837f7a488d75cbedbc01e35a236a97f00b85259746e6a368b7aeec873045e94a`;
the proxy and hooks have a separate source ledger.

## Exact preregistered closure

Each rank runs 12 factor configurations and rebuilds each as a formal-memory
cell and an ownership-witness cell. The immutable order is N-major, arm-major,
memory then witness. Per rank the hook must close exactly:

- 24 primary execution cells;
- 26,240 vLLM unified-attention calls;
- 720 document-prefill GDN calls;
- 78,720 request-cell GDN calls; and
- 79,440 total GDN calls.

Across eight ranks this is 209,920 attention and 635,520 GDN calls. Every
attention call contains integer references to one fully hashed selected Triton
artifact, its selected compile configuration, a content-hashed call shape, and
an autotune observation. Every referenced shape is also compared call by call
with the corresponding immutable primary request-ledger row. The table column
schemas are frozen exactly. Artifact objects are stored once per rank, avoiding
multi-gigabyte repetition without sampling calls.

Each GDN call records cell-local order, actual layer, sequence length,
previous-state flag, selected eager `torch_chunk_gated_delta_rule`, and one
functional conv plus recurrent cache rebind. Six GDN source keys are frozen to
their exact root kind, relative path, source SHA-256, module, and qualname;
cross-key substitution and duplicate callable identities are rejected. The
accepted phase geometry is 30 layer-ordered document-prefill calls at sequence
length 4095, followed by round-major/request-major 30-layer blocks at sequence
length 32 for round 0 and length 1 for rounds 1--7.

Each receipt is additionally bound to its exact rank, immutable runner path and
SHA-256, complete runner argv and argv SHA-256, `--stage shard`, `--rank`,
`--output`, and the corresponding shard path and SHA-256. This closes simple
cross-rank receipt substitution; it is still trusted producer capture rather
than runtime attestation.

## Claim boundary

Only if the fresh primary aggregate, all call closures, source snapshots,
artifact replays, exact shape/shard bindings, and all controls pass may target 5
be reported as passing at the declared scope: per-call compiled Triton
selection for paged attention and exact eager-source/cache-rebind selection for
GDN. The result does **not** establish a compiled GDN binary, the identity of
underlying ATen/CUDA operators, runtime attestation, malicious-producer
resistance, or generality across models, runtimes, or hardware.

## Local gates

From the repository root:

```bash
PYTHONPATH=paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v5/executed_source:paper_autonomous_multifork_iteration/evidence/r39_compiled_dispatch/executed_source \
  python3 -m unittest discover \
  -s paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v5/tests \
  -p 'test_r39_*.py' -v

PYTHONPATH=paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v5/executed_source:paper_autonomous_multifork_iteration/evidence/r39_compiled_dispatch/executed_source \
  python3 -m py_compile \
  paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v5/executed_source/*.py

bash -n \
  paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v5/executed_source/python_proxy_env/bin/python \
  paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v5/executed_source/r39_primary_formal_h20.sh
```

Eleven Round-39 tests cover frozen counts, compact replay, omitted-call rejection,
artifact-reference tampering, exact key-to-callable GDN identity, scope/column/
shape/rank substitution, and binding all 26,240 rank calls to the immutable
primary request ledger.  Four capture-scope regressions additionally execute
the real entrypoint wrappers across all eight ranks: 40 total warmup calls stay
uncaptured, 192 formal cells are captured, post-factorial calls stay
uncaptured, and wrong-rank/nested/reentry/error/short-count paths fail closed
without leaking the scope token.  A fresh archive extraction must additionally pass the
immutable 162-test RR2 suite from that exact detached layout.  The formal
wrapper repeats both gates before invoking the immutable launcher; the launcher
then independently repeats its own 162-test preflight.  The formal finalizer
applies 21 refusal controls to each rank's actual receipt, selected artifacts,
and snapshotted sources.  The final top-level terminal ledger binds the v5
preflight logs and stage markers together with the primary, dispatch-capture,
and formal-binding artifacts before `COMPLETE` is created.

## Resource estimate and launch boundary

The formal run requires all eight H20 GPUs concurrently. Likely wall time is
20--45 minutes, with a conservative 90-minute upper bound because N=32 uses
sequential round-major generation and the primary runner emits about 0.85 GiB
of raw evidence. Building and auditing this directory does not stage or launch
the run and does not operate any QS resource.

After independent package audit, the staged entry point will be:

```bash
bash /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r39_primary_compiled_dispatch_20260827e/paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v5/executed_source/r39_primary_formal_h20.sh
```

The intended non-overwriting result root is
`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r39-primary-compiled-dispatch-20260827e`.
