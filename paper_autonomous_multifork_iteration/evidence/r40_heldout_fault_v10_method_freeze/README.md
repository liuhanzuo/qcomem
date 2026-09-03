# Held-out fault v10 method freeze

Status: `HOLD_PENDING_FRESH_INDEPENDENT_AUDIT_AND_EXTERNAL_BINDING`.

This is a non-overwriting method artifact with local CPU-only regression fixtures. It contains no operator binding, designer snapshot, held-out fault identities, formal configuration, GPU result, or paper claim. No packaged test contacts a network, QS, or a GPU.

The sole formal entry point is `run_authorized_campaign`. Its caller supplies signed artifact locations, an exact signed attempt and nonce, and two pre-existing signed directory objects: the terminal root and an independent durable consumption root. The binding commits the canonical paths and the device/inode pair of both roots. A record named by the canonical `(terminal object, durable object, nonce, attempt)` run-identity hash is atomically created without replacement in the durable root before any probe or worker can start; an identical local copy is then placed in the terminal root. Clearing or replacing only the terminal root cannot restore authority, and re-signing the same run identity does not create a second authority. The durable store must itself be retained by the independent operator.

Production process creation consists of one fixed provenance-probe `subprocess.run` site and one worker `subprocess.Popen` site, both lexically inside the formal entry point. There is no module-level authority token, plan/lifecycle object, spawn helper, or public probe launcher. Each worker first reports its child-observed logical argv, typed argv, exact environment and CUDA UUID, cwd device/inode, PID/PGID, and the loaded Python executable's path/device/inode/SHA-256. Darwin measures the mapped executable vnode through libproc; Linux uses `/proc/self/exe`. The parent accepts the worker only after this handshake matches the signed runtime and re-derived spec.

SIGINT and SIGTERM remain blocked across durable consumption and across each Popen-to-handshake registration window. Finalization owns and reaps every recorded process group, rehashes all inputs, recomputes the eight typed actual specs from child receipts, cross-checks their aggregate against signed provenance, and publishes exactly eight no-replace terminal records plus the local consumption record. Terminal validation enforces that exact nine-name closure and no replacement.

Run `python3 -B static_audit_v10.py` and `python3 -B -m unittest discover -v -s . -p 'test*.py'`. The packaged discovery count is 18 tests, with no skip or expected-failure mechanism. Freezing is permitted only after raw and clean-stage suites pass and two independent deterministic builds are byte-identical.

Trust boundary: the method fails closed on unsupported executable-identity kernels and does not defend against a malicious kernel/root actor or deletion/corruption of the independently retained durable authority store by an actor already authorized to rewrite that store.
