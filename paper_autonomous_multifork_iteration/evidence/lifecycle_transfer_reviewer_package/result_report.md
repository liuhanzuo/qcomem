# Lifecycle-transfer formal result

Job 243408 / Trial 1859580 completed successfully on eight H20 GPUs. The formal run moved from `Uncommit` to `Pending` at 00:56:53, `Running` at 00:57:10, and `Complete` at 01:01:17 on 2026-08-19. No GPU smoke run was used.

The frozen static manifest (`c59b66df...`), code ledger (`7620f058...`), model artifact ledger (`c0a23e9d...`), and canonical 14-shard model-weight ledger (`8314a82c...`) were independently rechecked. All 14 model-weight shards and all eight raw-result hashes passed. The eight ranks used eight distinct PG-19 train books and eight distinct H20 UUIDs.

All eight ranks passed the preregistered lifecycle predicates. Against the uninterrupted control, all four requests and four semantic rounds passed full-vocabulary `torch.equal` checks and had identical recorded logit hashes and generated tokens, identical final logical KV digests, identical final GDN-state digests, and unchanged document-KV digests. The 4096-token prefix was page aligned with zero partial-tail staging bytes. Slots 2 and 3 were cancelled after two rounds, zero-scrubbed, and their exact private physical reservations were reassigned. Every stale epoch-0 handle was rejected at `STALE_SLOT_LEASE`, every matched epoch-1 replacement was accepted, and there were zero mutant escapes or wrong-gate outcomes.

The defensible claim is limited to aligned-page geometry plus cancellation, zero-scrub, exact reclamation, and stale-lease rejection on the existing Qwen3.5/vLLM Q16 adapter. This result is not evidence for a second independently implemented model/runtime, a different recurrent backend, true concurrent kernel execution, production scheduler integration, latency, throughput, or NVML capacity.

The authoritative aggregate is `artifacts/qwen35-forkaudit-lifecycle-transfer-20260819c/forkaudit-lifecycle-summary.json` (SHA-256 `4d0a3b0a44f125a1c18e2c0e6f9644843585fdc34244a6fbf20ad58a84cb760d`).
