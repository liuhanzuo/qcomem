# R40 primary compiled-dispatch v11 (`20260827k`)

This is a new, non-overwriting bytecode-isolation repair derived from the
canonical frozen v10 archive. It modifies neither v10 nor earlier evidence,
their consumed formal result roots, the paper, the immutable RR2 runner, nor
its original launcher. No v11 GPU result is claimed here.

V10 completed all eight H20 ranks, emitted eight raw shards and 192 primary
cells, passed the detached receipt stage and blind aggregate stage 06, and then
failed its terminal code-snapshot audit. The terminal audit found
`round_04_rr2_package/executed_source/gpu/__pycache__`. Consequently v10 did
not write `COMPLETE` and its one-shot result remains a sealed infrastructure
failure rather than admissible formal evidence.

The v10 outer preflight isolated its own Python calls, but the complete old
primary launcher process tree was not closed as an explicit invocation
contract. In particular, the proxy's non-rank branch forwarded Python argv
without forcing `-B` or rechecking the bytecode environment. The old launcher
also contains an explicit `py_compile`, for which `-B` alone is insufficient;
its `PYTHONPYCACHEPREFIX` must remain bound to a result-owned directory.

V11 changes only the formal supervisor, Python proxy, tests, and package
metadata:

- the exact primary `bash` argv, launcher/proxy/Python hashes, and
  `PYTHONDONTWRITEBYTECODE=1` plus result-owned `PYTHONPYCACHEPREFIX` are sealed
  in a pre-launch contract;
- every proxy branch rechecks that contract and environment, then forces
  `-B` before the real interpreter;
- source-visible Python grandchildren inherit the same two variables, while
  explicit `py_compile` output is redirected beneath the primary result;
- the detached source is scanned before result creation, after outer
  preflight, and after the complete old primary launcher returns; and
- a command-level regression runs a successful aggregate-style import and
  Python grandchild, the actual byte-identical RR2 aggregate entrypoint through
  its expected missing-input gate, and explicit `py_compile`, leaving zero
  bytecode paths in frozen source.

`python-invocation-audit.json` enumerates all 20 executing `$PYTHON` command
sites (35 direct Python processes on a successful path) and the three
source-visible Python grandchild sites. `v10-v11-scientific-equivalence.json`
binds the unchanged scientific bytes and counts.

## Evidence boundary

V11 preserves v10's honest-process, trusted pinned PyTorch/CUDA boundary. For
every registered primary `unified_attention` call it:

1. creates an exact rank/cell/local call ID;
2. resolves the one expected `kernel_unified_attention` cache bundle and full
   compile configuration immediately before the original Triton launcher;
3. calls that original launcher;
4. emits no success evidence on exception; and
5. seals a receipt only after normal return on the same assigned CUDA device
   and stream.

The sealed digest covers the call ID and shape, selected artifact ID, complete
selected configuration, exact autotuner `selected_kwargs` plus
`num_warps`/`num_ctas`/`num_stages` (or the exact statement that no autotuner
was observed), device/stream identity, and both post-return predicates.

This is not driver/device binary attestation and is not malicious-runtime
resistance. It establishes an exact selected-launcher-to-successful-return
receipt only within the declared trusted runtime. GDN remains explicitly eager
and keeps the frozen mutually exclusive route accounting from v6.

## Immutable scientific payload

The v11 deterministic builder accepts only the exact v10 archive at SHA-256
`be206a1dba2442421f71ed8d83f4f29e52bc302017152d69fe8c555fcf7cb99e`,
removes the v10 observer package, and imports its complete 70-file RR2
dependency snapshot byte-for-byte. The immutable runner, original launcher,
model and input ledgers, factorial arms/order, quantization policies, metrics,
thresholds, receipt recorder, rank wrapper, compact verifier, and finalizer are
unchanged. The frozen totals remain eight ranks, 96 configurations, 192 cells,
209,920 attention calls, 635,520 GDN calls, and zero dense fallback calls.

## Local gates

The source tree must pass 28 CPU tests with zero skips, Python AST compilation,
Bash syntax, all source/dependency/focused-fixture ledgers, the complete
process-tree audit, and the v10-v11 scientific equivalence audit. A fresh
normalized archive extraction must repeat these gates, contain only safe
regular-file/directory members, contain zero `__pycache__`, `.pyc`, or `.pyo`
paths, and rebuild byte-identically twice. Package construction remains HOLD
until these checks and a fresh independent archive audit pass.

## Formal execution remains HOLD

Only after an independent archive PASS may an authorized operator use the new
one-shot path:

```bash
R40_H20_EXECUTION_AUTHORIZED=yes bash \
  /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r40_primary_compiled_dispatch_v11_20260827k/paper_autonomous_multifork_iteration/evidence/r40_primary_compiled_dispatch_v11/executed_source/r39_primary_formal_h20.sh
```

Its fixed result root is
`.../runs/qcomem/r40-primary-compiled-dispatch-v11-20260827k`. A failed or
successful root is never reused. Package construction does not grant launch
authority.
