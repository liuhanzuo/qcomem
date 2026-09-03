# R39 source-aware blind-fault executor

This directory executes the immutable eleven-row designer freeze in
`../designer_freeze/`.  It does not edit the freeze, manuscript, or existing
evidence.  The scientific order is fixed before any candidate lane:

1. verify both designer files and the source ledger;
2. seal an exact-selector feasibility receipt;
3. run an independent reference and matched clean in separate processes;
4. authorize the mutant only after detached clean replay passes;
5. run the mutant in a third fresh process;
6. replay output equality, persistent-base, allocator, and unmodified
   ForkAudit outcomes without importing Torch/model/runtime modules.

Every lane writes sixteen complete-vocabulary contiguous CPU-FP32 logit
sidecars, two eight-token sequences, persistent document-KV and GDN snapshots,
synchronized allocator endpoints, a byte-bound locus receipt, full ForkAudit
trace, and cleanup receipt.  Exceptions, misses, and valid negative outcomes
are retained individually.  The aggregate deliberately computes no detection
rate.

## Exact-selector boundaries

- `R39-BF02` is pre-output ineligible in this source: a layer owns one
  monolithic K/V arena and its page table contains only local block integers;
  there is no operation for rebinding exactly one page to a different layer's
  byte interval without adding other overlaps.
- `R39-BF09` is pre-output ineligible in this source: the existing two-stream
  replacement path globally synchronizes before reclamation and exposes no
  per-request last-use event wait that could be omitted alone.
- `R39-BF11` is resolved from the warmup Triton cache.  Absence of a distinct
  ABI-compatible, already loadable alternate artifact is sealed as ineligible;
  disk-byte substitution is never used.

These are frozen-selector absences, not substituted faults or detector passes.

## GPU mapping

The launcher requires an exclusive eight-H20 node.  Each physical GPU runs its
tuple serially while all eight workers run in parallel:

```text
GPU0 BF01 -> BF09       GPU1 BF02 -> BF10
GPU2 BF03 -> BF11       GPU3 BF04
GPU4 BF05               GPU5 BF06
GPU6 BF07               GPU7 BF08
```

Each eligible fault performs one feasibility probe plus three fresh model
loads.  Do not start while another experiment owns any GPU, including GPU0.

## Local validation

```bash
python3 -m py_compile executor/*.py executor/tests/*.py
bash -n executor/r39_launch_8gpu.sh executor/r39_formal_existing_trial.sh
PYTHONPATH=executor python3 -m unittest discover -s executor/tests -v
sha256sum -c executor/source-code.sha256   # from package root
```

The formal wrapper refuses existing stage/output paths and verifies the archive
SHA before extraction.  It does not create, stop, evict, or delete QS
resources.
