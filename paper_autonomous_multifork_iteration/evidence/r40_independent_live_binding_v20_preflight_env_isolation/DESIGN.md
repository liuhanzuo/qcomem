# V19 terminal-closure repair and controlled v18 scientific boundary

V19 inherits v18's self-contained staging and changes one terminal-governance
file. A Python
`tarfile`/GNU-tar-equivalent inventory of
the canonical v6 archive is checked against its fixed SHA-256 before any
extraction. Only exact regular AppleDouble metadata basenames are excluded;
all retained logical members are frozen by path, type, mode, size, and digest.
The v19 overlay is independently hash-bound and may contain only its exact
package root with regular files. Both inputs are written into a private sibling
directory using exclusive file creation and atomically renamed to the exact
new stage name only after full receipt/tree verification. Verification repeats
the archive/ledger proofs and exact lexical tree closure. The launcher repeats
verification and a zero-`._*` scan before marker or result action.

V18 is not scientific evidence: its producer deterministically writes eight
per-rank `invocation.json` files that its terminal expected-path projection did
not admit. V19 keeps 9/10 payload files byte-identical and changes only
`executed_source/r40_tree_closure.py`. The pinned
`v19-current-payload.sha256` and payload-sealed
`v18-v19-controlled-diff.json` bind the exact paths, sizes, hashes, and sole
controlled change without consulting a sibling package. The normalized v18
and v19 generated-launcher hashes are respectively
`8d5ba77f9b61b760346334b4bca041e1ac0176719c5b8bd2e616a29b24226636`
and `ef1f68028fbec4180925c60701ed6d850975c0eeeff541964cab60fafa2e20ed`.

The launcher builder obtains the v6 launcher through one `O_NOFOLLOW` file
descriptor, checks stable `fstat` identity and metadata before and after the
read, and derives its hash, strict UTF-8 text, and transformed output from the
same byte snapshot. A pathname replacement cannot substitute bytes between a
hash check and a second open because no second open exists.

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

V19 adds the fixed
`compiled-dispatch-capture/rank-{0..7}/invocation.json` paths to that
projection. Each file must be canonical JSON with the exact producer schema,
rank, runner SHA-256, argv list, shard SHA-256, and canonical argv SHA-256; all
binding fields must equal the corresponding formal receipt's
`execution_binding`. Missing, extra, malformed, or cross-rank/mismatched
invocations fail before terminal publication.

The complete package suite passed 87/87 with zero skips on the source tree and
a pre-freeze exact clean stage, and the static audit passed 131/131. The
deterministic archive freeze, fresh independent audit, same-run 162/162
zero-skip inherited preflight, CUDA smoke, and subsequent clean scientific
execution remain gates.
