# V7 threat boundary and counterexample disposition

V7 deliberately does not attempt hostile-runtime attestation. Its trust root is
the honest rank process plus the hash/version-bound PyTorch/CUDA, vLLM, Triton,
Transformers, qcomem, immutable runner, and original launcher. Within that
boundary, a receipt means: this exact formal call selected this exact compiled
launcher artifact/configuration, invoked the original launcher, and observed
its normal Python return without changing the assigned device or stream.

The following are fail-closed:

- a decoy or wrong kernel name, a second launcher, an unreturned or failed
  launcher, a missing formal call, a warmup-only capture, a CUDA-graph replay
  that bypasses the per-call launcher wrapper, or a dense fallback;
- forged/partial autotune fields, unknown provenance fields, duplicate or
  unreferenced tables, cleared return predicates, or a recomputed call digest
  inconsistent with any bound component;
- rank relabeling without the independently reopened proxy PID receipt and
  immutable H20 assignment row, missing/duplicate ranks, duplicate processes
  or GPUs, or aggregate/raw-shard omission;
- postcapture mutation of bound sources, metadata, PTX, cubin, runner,
  launcher, model/code/weight ledgers, protocol, preflight, assignment, or
  terminal artifacts.

The following remain explicitly out of scope and must not be claimed:

- a malicious same-user process rewriting all evidence and trusted files;
- a malicious or no-op CUDA/Triton runtime that returns normally while not
  enqueueing the intended device work;
- independent driver module, device program-counter, or completed-kernel
  attestation (normal launcher return is an enqueue/return boundary, not a
  device synchronization witness);
- compiled identity of eager GDN's underlying ATen/CUDA operators; or
- generality beyond this model, protocol, pinned stack, and H20 setup.

Those exclusions are intentional. Wording such as “device binary independently
attested on every call” is not supported. “Exact selected compiled launcher to
post-return receipt in the declared trusted runtime” is the maximum supported
claim after a fresh formal PASS and independent audit.
