# V17 Linux clean-stage boundary and unchanged v16 scientific boundary

V17 changes staging only. A Python `tarfile`/GNU-tar-equivalent inventory of
the canonical v6 archive is checked against its fixed SHA-256 before any
extraction. Only exact regular AppleDouble metadata basenames are excluded;
all retained logical members are frozen by path, type, mode, size, and digest.
The v17 overlay is independently hash-bound and may contain only its exact
package root with regular files. Both inputs are written into a private sibling
directory using exclusive file creation and atomically renamed to the exact
new stage name only after full receipt/tree verification. Verification repeats
the archive/ledger proofs and exact lexical tree closure. The launcher repeats
verification and a zero-`._*` scan before marker or result action.

The v16 mechanism, runner-facing code, preregistration, and claim boundary are
byte-identical; their hashes are frozen in `v16-scientific-payload.sha256`.

The source registry is frozen before `original_build`. A private nonce-sealed
dispatch ledger records strong source/destination handles and full storage
descriptors for persistent-rooted clone/copy edges. The returned group must
consume every relevant edge exactly once; semantic origin, exact destination
object, interval, content, and storage are checked.

Each real phase serializer runs first. Its artifact reference is then resolved
under the artifact root, bytes and SHA are reread, the on-disk GDN object is
canonically hashed against the returned object, and all 540 rows are checked.
On post-write verification failure the exact new phase artifact is removed, so
the finalizer cannot accept an orphan. Completed sets are independently derived
from phase and compared with call arguments and GDN rows.

Global primary absence is derived from instrumented primary call/event counters,
not a hard-coded aggregate zero. Final aggregation derives all totals from eight
rank receipts and rejects temp/rejected orphan files.

Terminal publication first rejects a symlinked or non-canonical lexical root
and normalizes all output paths without permitting `..` or root escape. A
predetermined whitelist is reconstructed from fixed producer paths and the
already-validated primary/model/formal manifests. The launcher passes every
path explicitly and `prepare` requires an independent exact rescan before it
publishes a sealed expectation with exact node/file/directory counts, bytes,
hashes, and per-file content schema. The completion marker and all R40 terminal
JSON files use exclusive creation; a final lexical rescan must equal the
expectation plus the self-excluded terminal-tree ledger.

Only the CUDA smoke and subsequent clean scientific execution remain external.
