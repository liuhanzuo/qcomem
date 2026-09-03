# Round 29 independent GDN observer

This package records a locally prospectively frozen, source-distinct,
same-process implementation of the GDN storage/lifecycle observer on one
H20-3e.  The candidate capture and the second observer both read the same live
candidate-created PyTorch tensors and trust the same tensor/storage API, but
the second implementation does not import the candidate capture rows or
verdict booleans.

The first source freeze was superseded before GPU execution because its
launcher digest did not match the source ledger.  The disclosed
`preexecution-amendment-v2.json`, `preregistration-v2.json`, and
`source-code-v2.sha256` are the only execution authority.  No candidate GPU
output existed before that correction.

The formal run is under `formal_run_20260825a/`.  It completed two fresh
N=2 policy cells (shared GDN base and materialized GDN), each at setup,
post-transition, and post-generation.  Across those six phase points, the
second observer matched all 1,080 candidate phase-row descriptors and all
96,660 paired ownership-relation coordinates (zero mismatches).  Using its own
cross-phase opaque identities, it also passed all six phase and both lifecycle
checks.  The raw result contains 2,160 second-observer before/after rows plus
1,080 candidate rows, or 3,240 serialized rows in total.  The CPU replay
imports no candidate capture/replay module, recomputes the archived comparison,
and reproduced the shipped replay byte-for-byte.  The 16-entry terminal ledger
verifies the 16 artifacts it lists; it does not include itself, the final stage
marker, or the later local transfer receipt.

Key raw SHA-256 values:

- preregistration v2: `90decffb732d50ec04fbebe5f34d8d5fb7acb0fabec88f1ba6ce53c9e262984c`
- source ledger v2: `9b9a135fff63bbd4bd363d9c2d341e593ddef776d4f07dc0088333baaef91818`
- formal result: `5d8edf442f9dedd1b3e7e2b338a324b2c30f9001df2fcbba5ce4f6ad2f42c0df`
- independent replay: `65435c367bab0b22e55c2b79eb91eb3ef1a6ca62ee56b578b93c7920edfd8e29`
- terminal file ledger: `9b39b9c291b173496aa7ca6d52b01fdf3405c0c900f1b95084496d3558b239cc`

Validation from this repository root:

```bash
cd paper_autonomous_multifork_iteration/evidence/r29_independent_observer/formal_run_20260825a
shasum -a 256 -c receipts/terminal-files.sha256
cd ../../../..
PYTHONPATH=gpu python3 gpu/r29_replay_independent_gdn_observer.py \
  --input paper_autonomous_multifork_iteration/evidence/r29_independent_observer/formal_run_20260825a/raw/independent-gdn-observer-result.json \
  --expected-input-raw-sha256 5d8edf442f9dedd1b3e7e2b338a324b2c30f9001df2fcbba5ce4f6ad2f42c0df \
  --output /tmp/r29-independent-observer-replay.json
cmp /tmp/r29-independent-observer-replay.json \
  paper_autonomous_multifork_iteration/evidence/r29_independent_observer/formal_run_20260825a/replay/independent-gdn-observer-replay.json
```

Claim boundary: this is cross-implementation PyTorch-level agreement in the
same process, not independent recapture or external ground truth.  Both paths
inspect the same candidate-created objects, receive phase/completion labels
from the same runner, and trust PyTorch's `untyped_storage`/`data_ptr` API.  It
is not a second producer process, an end-to-end recapture, a KV observer,
compiled-binary attestation, a live injected-negative campaign, or evidence for
arbitrary models, runtimes, schedules, or hardware.  Snapshot equality rules
out observed net mutation during capture but cannot exclude a transient write
followed by restoration.
