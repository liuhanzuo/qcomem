# Held-out fault v9 method freeze

Status: `HOLD_PENDING_FRESH_AUDIT_AND_EXTERNAL_OPERATOR_BINDING`.

This is a non-overwriting CPU-only security-method artifact. It releases no operator binding, designer snapshot, held-out fault identities, formal configuration, GPU result, or paper claim. The sole authorized entry point is `run_authorized_campaign`; its caller supplies only signed artifact locations plus the signed attempt, run nonce, and exact terminal root. Worker specifications are re-derived from the signed execution contract immediately before each launch.

V9 removes every module-level authority token, plan object, lifecycle object, and callable worker-spawn method. The only `Popen` site is lexically inside the authorized call. SIGINT/SIGTERM remain blocked from immediately before process creation until PID, PGID, executable inode/hash, argv, typed argv, cwd inode, environment, CUDA UUID, and actual-spec digest are registered. A no-replace consumption receipt atomically consumes `(binding, nonce, terminal_root, attempt)` before launch, preventing replay to the same or a fresh root.

Run `python3 -B static_audit_v9.py` and `python3 -B test_v9.py`. Freezing is permitted only after both pass and deterministic builds match. This artifact does not defend against a malicious kernel/root actor.
