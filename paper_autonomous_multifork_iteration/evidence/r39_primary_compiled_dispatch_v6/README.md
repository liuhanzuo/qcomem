# Round-39 primary-factorial compiled-dispatch closure (`20260827f`, v6)

This is a non-overwriting local repair package.  It preserves v5 and every
earlier package and does not create, retry, stop, evict, or release any QS
resource.  No formal GPU result is claimed by this directory.

## Why v6 exists

The v5 formal attempt failed closed in the observer during the first formal GDN
call.  The frozen Transformers 5.14.1 Qwen3.5-MoE forward has two mutually
exclusive routes, but the v5 observer assumed every call used the multi-token
route:

- multi-token: `torch_chunk_gated_delta_rule`, functional
  `update_conv_state`, functional `update_recurrent_state`;
- cached single-token: `torch_recurrent_gated_delta_rule`,
  `torch_causal_conv1d_update` in place, functional
  `update_recurrent_state`.

V6 changes only the observer and its fail-closed replay.  It intercepts and
source-binds all four Transformers route callables and the two qcomem
functional cache callables.  Each formal call must have exactly one of these
five-tuples, ordered as `(chunk rule, recurrent rule, functional conv rebind,
in-place conv update, recurrent rebind)`:

- multi-token: `(1, 0, 1, 0, 1)`;
- cached single-token: `(0, 1, 0, 1, 1)`.

Cross-route contamination, an omitted event, or a duplicate event fails the
rank before a receipt can be accepted.  A separate pre-GPU static gate checks
the frozen runtime file SHA-256
`688d9a8f2830d6729cd2945563f38b710100c086565b97c27c94c96bd9716b9f`,
its three eager fallback definitions, and the two single-token branch gates.

## Scientific identity is unchanged

The scientific runner remains
`run_qcomem_qwen35_forkaudit_review_revision.py` at SHA-256
`9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775`.
The launcher remains SHA-256
`077a876b9849661135044c50cfdea272d302a48af0bb4e21ec640eca2ca85460`.
The model revision, PG19 inputs, 8 generation steps, 8-rank assignment,
N-major/arm-major/memory-then-witness factor order, thresholds, measurements,
and primary shard schema are unchanged.  The v5 rank-scope wrapper is also
byte-identical in v6.

Per rank the immutable runner still closes 24 cells, 26,240 attention calls,
720 GDN document-prefill calls, 78,720 GDN request calls, and 79,440 total GDN
calls.  The eight-rank totals remain 192 cells, 209,920 attention calls, and
635,520 GDN calls.  V6 changes only which route events the receipt records for
the already-existing sequence-length schedule.

## Local CPU/static gates

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/executed_source \
python3 -m unittest discover \
  -s paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/tests \
  -p 'test_r39_*.py' -v

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/executed_source \
python3 -m py_compile \
  paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/executed_source/*.py

bash -n \
  paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/executed_source/python_proxy_env/bin/python \
  paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh
```

The 13 CPU tests preserve the 11 v5 gates and add exact route-count refusal and
four-callable hook-coverage checks.  The static runtime verifier was also run
locally against the already snapshotted frozen runtime source; its result is
recorded in `local-detached-validation.json`.

## Launch draft only

No command below has been executed by this package-preparation task.  If a
fresh eight-H20 run is separately authorized and the archive is staged at the
declared non-overwriting root, the entry point is:

```bash
bash /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r39_primary_compiled_dispatch_20260827f/paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch_v6/executed_source/r39_primary_formal_h20.sh
```

Its non-overwriting result root is
`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r39-primary-compiled-dispatch-20260827f`.

## Claim boundary

Only a fresh completed formal aggregate may support target 5 at the declared
scope: per-call selected Triton attention artifact/configuration and exact
eager GDN route/cache-source selection in this frozen configuration.  It does
not establish a compiled GDN binary, underlying ATen/CUDA operator identity,
runtime attestation, malicious-producer resistance, or cross-model/runtime/
hardware generality.
