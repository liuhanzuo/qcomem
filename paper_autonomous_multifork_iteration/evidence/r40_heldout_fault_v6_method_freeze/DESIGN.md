# V6 security design

## Trust and binding

`v6_guard.verify_operator_binding` accepts canonical JSON bytes, not a Python
approval object. It verifies an Ed25519 signature under the one compiled public key
and signed trust-root identifier. The signed payload binds the exact archive,
embedded source ledger, snapshot byte commitment, snapshot inventory commitment,
execution contract, and torch expectation, plus an HTTPS publication URI. Archive,
ledger, and both snapshot commitments are independently recomputed from regular,
unique files. No callback or caller-selected public key is accepted.

## Lifecycle and output

`Lifecycle.install_signal_handlers` installs actual SIGINT and SIGTERM handlers.
The handler finalizes one transactional set of eight failure terminals and exits
with 130 or 143. Success is impossible before `start`, and requires the exact gate
schema including overall kill completion and an exact status object for every
worker. Each terminal carries exact pre/post hash maps. Validation rejects extra,
missing, symlinked, non-regular, or multiply-linked terminal files.

`ProtectedParent` retains a directory descriptor and device/inode identity. Batch
publication stages and fsyncs every file, links without replacement, fsyncs and
rechecks the retained parent, and rolls back every link made by the transaction if
that final check fails.

## Process and command provenance

Torch is imported only by `torch_probe_v6.py` in `python -I -S`. The probe requires
an empty pre-import `sys.modules` slot, verifies PathFinder/import/module/spec/loader
agreement and the exact `torch/__init__.py` digest, and returns canonical JSON. The
parent checks the report, nonempty canonical GPU UUID visibility, and a nonnegative
in-range logical index. A detached object or a caller-fabricated `sys.modules`
entry is not an input to this API.

Execution contracts are canonical signed bytes containing exact argv, environment,
cwd, and an exhaustive typed path-binding list. Path-looking bare arguments,
relative paths, separate path options, and `--config=value` forms must all appear in
that list and resolve to regular unique files in the exact runner inventory.

## Freeze and archive

The ledger, deterministic tar.gz, and `METHOD_FROZEN.json` are built in memory and
published as one no-replace transaction. A rerun fails without changing bytes or
inodes. Verification requires one canonical gzip member with no trailing bytes and
a byte-for-byte canonical USTAR reconstruction; traversal names, alternate mode or
mtime, duplicate names, and non-regular members are rejected.

V6 does not release the snapshot, choose faults, choose a formal configuration, or
invoke QS/GPU resources. Its terminal status remains HOLD pending external work.
