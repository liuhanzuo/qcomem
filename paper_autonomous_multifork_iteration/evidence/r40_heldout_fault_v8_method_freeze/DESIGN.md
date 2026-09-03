# V8 security design

## Trust and binding

`v8_guard.verify_operator_binding` accepts canonical JSON bytes, not a Python
approval object. It verifies an Ed25519 signature under the one compiled public key
and signed trust-root identifier. The signed payload binds the exact archive,
embedded source ledger, snapshot byte commitment, snapshot inventory commitment,
execution contract, runtime expectation, canonical runner-manifest bytes, and an
independently recomputed runner-tree inventory, plus an HTTPS publication URI. The
execution contract repeats both runner commitments and the expectation commitment.
The manifest is loaded only from anchored canonical bytes; a caller-supplied Python
manifest object is not an accepted interface. Archive, ledger, snapshots, and the
runner inventory are independently recomputed from regular unique files. No
callback or caller-selected public key is accepted.

## Lifecycle and output

`authorized_launcher_v8.py` is the sole formal end-to-end entrypoint. It first
loads the canonical signed binding and verifies, in order, the archive and embedded
ledger, external ledger equality, both snapshot commitments, both runner
commitments, runtime expectation, and execution contract. It then runs the fixed
isolated torch probe. Only after every step passes can it construct an opaque,
immutable `AuthorizedPlan`, install lifecycle handlers, and spawn workers. Missing
binding, a manifest not covered by the binding, or an exact common-system-shell
transplant at the signed Python path therefore fails before any subprocess.

`Lifecycle.install_signal_handlers` installs actual SIGINT and SIGTERM handlers.
The handler finalizes one transactional set of eight failure terminals and exits
with 130 or 143. `Lifecycle` owns the real process handles, spawns and waits each
worker itself, performs required termination/escalation, and derives the only gate
from those actual receipts. There is no caller gate setter. Overall kill fields are
checked against all eight per-worker receipts. Success is impossible before
`start`, before eight actual zero exits, or after a hash change. Every terminal
carries exact pre/post hash maps, the complete lifecycle receipt, and its digest.
The gate and every terminal also commit the binding, execution-contract,
runtime-expectation, isolated-probe-report, and exact derived spawned-spec digests.
Validation rejects extra, missing, symlinked, non-regular, or multiply-linked files.
Signal fields accept only exact integer `2` or `15`, never booleans or floats.

`ProtectedParent` retains a directory descriptor and device/inode identity. Batch
publication stages and fsyncs every file, links without replacement, fsyncs and
rechecks the retained parent, and rolls back every link made by the transaction if
that final check fails.

## Process and command provenance

Torch is imported only by `torch_probe_v8.py` in `python -I -S`. Before launch, the
parent validates an exact signed runtime expectation and independently rescans the
canonical runner manifest. All probe parameters are derived from that expectation.
The child rescans the same complete staged runtime tree, pins `sys.executable`, its
digest, CPython implementation/version/cache tag, the fixed probe digest, the torch
root/version/digest, and the device UUID/index/visibility. It also requires every
observed torch module file to stay inside the signed runtime tree. A `/bin/sh` file
declared as Python, a transplanted probe, or a caller-fabricated report is rejected.

Execution contracts are canonical signed bytes containing exactly the ordered
workers V8F01--V8F08, a signed timeout, exact argv/environment/cwd for each worker,
runner commitments, and the runtime-expectation commitment. Every argv index has
exactly one `literal`, `option`, `path`, or `option-path` row; no path heuristic is
used. Environment keys are an exact allowlist and each entry is typed as literal,
UUID list, or inventory-bound path. Relative, absolute, separate-option, and
`--config=value` path spellings resolve to regular unique files in the exact runner
inventory. Independently of those syntactic checks, every worker must use the
expected Python as `argv[0]`, the expected cwd, and the exact visibility whose
expected index resolves to the expected physical UUID. `Lifecycle.spawn_workers()`
accepts no arguments and derives every actual `Popen` token solely from this
validated contract.

## Freeze and archive

The ledger, deterministic tar.gz, and `METHOD_FROZEN.json` are built in memory and
published as one no-replace transaction. A rerun fails without changing bytes or
inodes. Verification requires one canonical gzip member with no trailing bytes and
a byte-for-byte canonical USTAR reconstruction; traversal names, alternate mode or
mtime, duplicate names, and non-regular members are rejected.

V8 does not release the snapshot, choose faults, choose a formal configuration,
bind an external operator, or invoke QS/GPU resources. Its terminal status remains
`HOLD_PENDING_FRESH_AUDIT_AND_EXTERNAL_OPERATOR_BINDING`.
Malicious concurrent same-user mutation and power-loss/crash atomicity are declared
out of scope; the host kernel/dynamic loader is part of the fixed trusted stack.
