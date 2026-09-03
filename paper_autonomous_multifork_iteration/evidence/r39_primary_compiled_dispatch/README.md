# Round-39 primary-factorial compiled-dispatch closure (`20260826a`)

This package closes a narrower provenance gap without changing the scientific
experiment. It freshly reruns the immutable RR2 W-run primary runner and its
original launcher on the same frozen model, PG19 inputs, thresholds, factor
levels, rank assignment, and execution order. R29 is not treated as primary
evidence.

## Why a fresh primary run is necessary

The existing Round-39 compiled-dispatch result binds 120 attention calls and
540 GDN calls in the separate R29 overhead experiment. That result is useful
supporting evidence, but it cannot upgrade target 5 for the paper's
96-configuration primary factorial. The present package therefore routes only
the eight `--stage shard` processes of the byte-identical primary launcher
through mandatory hooks. Static preparation, source replay, aggregation, and
all other launcher phases still execute through the original unmodified
launcher and runner.

The route uses a transparent Python proxy rather than editing the old code
snapshot. The old launcher still recursively audits its own read-only 34-file
closure against raw code-ledger SHA-256
`837f7a488d75cbedbc01e35a236a97f00b85259746e6a368b7aeec873045e94a`;
the proxy and hooks have a separate ledger.

## Exact preregistered closure

Each rank runs 12 factor configurations and rebuilds each as a formal-memory
cell and an ownership-witness cell. The immutable order is N-major, arm-major,
memory then witness. Per rank the hook must close exactly:

- 24 primary execution cells;
- 26,240 vLLM unified-attention calls;
- 720 document-prefill GDN calls;
- 78,720 request-cell GDN calls; and
- 79,440 total GDN calls.

Across eight ranks this is 209,920 attention and 635,520 GDN calls. Each
attention call contains integer references to a fully hashed selected Triton
artifact, selected compile configuration, call shape, and autotune observation.
The artifact objects themselves are stored once per rank, which avoids the
multi-gigabyte repetition produced by embedding a full cubin/PTX manifest in
every call row. This is lossless call closure, not sampling.

Each GDN call records its cell-local order, actual layer, sequence length,
previous-state flag, selected eager `torch_chunk_gated_delta_rule`, and one
functional conv plus recurrent cache rebind. The exact accepted phase geometry
is 30 layer-ordered document-prefill calls at sequence length 4095, followed by
round-major/request-major 30-layer blocks at sequence length 32 for round 0 and
length 1 for rounds 1--7.

## Claim boundary

If the fresh primary aggregate, all call closures, source snapshots, artifact
replays, and bound controls pass, target 5 may be reported as passing at the
declared scope: per-call compiled Triton selection for paged attention and
exact eager-source/cache-rebind selection for GDN. The result does **not**
establish a compiled GDN binary, the identity of underlying ATen/CUDA
operators, runtime attestation, malicious-producer resistance, or generality
across models, runtimes, or hardware.

## Local gates

From the repository root:

```bash
PYTHONPATH=paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch/executed_source:paper_autonomous_multifork_iteration/evidence/r39_compiled_dispatch/executed_source \
  python3 -m unittest discover \
  -s paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch/tests \
  -p 'test_r39_*.py' -v

PYTHONPATH=paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch/executed_source:paper_autonomous_multifork_iteration/evidence/r39_compiled_dispatch/executed_source \
  python3 -m py_compile \
  paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch/executed_source/*.py

bash -n \
  paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch/executed_source/python_proxy_env/bin/python \
  paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch/executed_source/r39_primary_formal_h20.sh
```

The five local tests cover exact frozen counts, compact replay, omitted-call
rejection, artifact-reference tamper rejection, and binding to the primary
runner's request ledgers. The formal finalizer additionally applies eight
negative controls to each rank's actual captured receipt, artifacts, and
source snapshots.

## Resource estimate and launch boundary

The formal run requires all eight H20 GPUs concurrently. The likely wall time
is 20--45 minutes, with a conservative 90-minute upper bound because N=32 uses
sequential round-major generation and the primary runner emits roughly 0.85
GiB of raw evidence. The build in this directory does not stage or launch the
run and does not operate any QS resource.

After independent package audit, the staged entry point is:

```bash
bash /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r39_primary_compiled_dispatch_20260826a/paper_autonomous_multifork_iteration/evidence/r39_primary_compiled_dispatch/executed_source/r39_primary_formal_h20.sh
```
