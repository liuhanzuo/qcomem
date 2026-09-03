# ForkAudit held-out fault method v3

Status: **method-freeze candidate only; no v3 fault set, GPU run, or scientific
result**.

This package hardens the method-v2 design against its independent audit.  The
formal verifier has one zero-argument entry point, loads the authoritative
method and formal configuration from fixed package-relative paths, enumerates
the exact disk inventory itself, rereads every receipt and full-vocabulary
sidecar, and never accepts caller-supplied observations or configuration
mappings.

The capture path synchronizes through its own backend interface and computes
canonical KV/GDN digests and scalar state directly from bound live tensors.
Formal execution rejects the CPU test backend.  Receipts bind campaign, run,
lane, case, GPU, call schedule, model, policy, preregistration, and frozen
method hashes.

The evaluator applies:

1. exact token and complete-vocabulary FP32 semantic comparison;
2. exact paired structural pre/post comparison plus per-call atomic coherence;
3. exact synchronized allocator endpoints, monotone peak, and restoration.

The future public designer receives only `designer_snapshot/`.  This directory
contains no v3 case definition.  `formal/formal-execution.json` is deliberately
absent; the executor is therefore fail-closed until a later independently
audited case/binding freeze creates it in a separate formal campaign package.

No file outside this directory is written by local validation or freezing.

