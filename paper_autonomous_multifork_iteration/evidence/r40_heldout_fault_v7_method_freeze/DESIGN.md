# V7 security design

## Trust and binding

`v7_guard.verify_operator_binding` accepts canonical JSON bytes, not a Python
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

`Lifecycle.install_signal_handlers` installs actual SIGINT and SIGTERM handlers.
The handler finalizes one transactional set of eight failure terminals and exits
with 130 or 143. `Lifecycle` owns the real process handles, spawns and waits each
worker itself, performs required termination/escalation, and derives the only gate
from those actual receipts. There is no caller gate setter. Overall kill fields are
checked against all eight per-worker receipts. Success is impossible before
`start`, before eight actual zero exits, or after a hash change. Every terminal
carries exact pre/post hash maps, the complete lifecycle receipt, and its digest.
Validation rejects extra, missing, symlinked, non-regular, or multiply-linked files.
Signal fields accept only exact integer `2` or `15`, never booleans or floats.

`ProtectedParent` retains a directory descriptor and device/inode identity. Batch
publication stages and fsyncs every file, links without replacement, fsyncs and
rechecks the retained parent, and rolls back every link made by the transaction if
that final check fails.

## Process and command provenance

Torch is imported only by `torch_probe_v7.py` in `python -I -S`. Before launch, the
parent validates an exact signed runtime expectation and independently rescans the
canonical runner manifest. All probe parameters are derived from that expectation.
The child rescans the same complete staged runtime tree, pins `sys.executable`, its
digest, CPython implementation/version/cache tag, the fixed probe digest, the torch
root/version/digest, and the device UUID/index/visibility. It also requires every
observed torch module file to stay inside the signed runtime tree. A `/bin/sh` file
declared as Python, a transplanted probe, or a caller-fabricated report is rejected.

Execution contracts are canonical signed bytes containing exact argv, environment,
cwd, runner commitments, and runtime-expectation commitment. Every argv index has
exactly one `literal`, `option`, `path`, or `option-path` row; no path heuristic is
used. Environment keys are an exact allowlist and each entry is typed as literal,
UUID list, or inventory-bound path. Relative, absolute, separate-option, and
`--config=value` path spellings resolve to regular unique files in the exact runner
inventory.

## Freeze and archive

The ledger, deterministic tar.gz, and `METHOD_FROZEN.json` are built in memory and
published as one no-replace transaction. A rerun fails without changing bytes or
inodes. Verification requires one canonical gzip member with no trailing bytes and
a byte-for-byte canonical USTAR reconstruction; traversal names, alternate mode or
mtime, duplicate names, and non-regular members are rejected.

V7 does not release the snapshot, choose faults, choose a formal configuration, or
invoke QS/GPU resources. Its terminal status remains HOLD pending external work.
Malicious concurrent same-user mutation and power-loss/crash atomicity are declared
out of scope; the host kernel/dynamic loader is part of the fixed trusted stack.
