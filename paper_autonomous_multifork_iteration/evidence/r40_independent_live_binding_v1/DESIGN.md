# R40 design and falsification contract

Date frozen: 2026-08-27

## Independence split

- `r40lib/oracle_worker.py` is the pre-injection authority.  It receives the
  correct fixture bindings before the candidate binder runs and emits salted
  payload challenges, descriptors, and storage relations.
- `r40lib/candidate_binder.py` is the only module that traverses the candidate
  graph after injection.  Faults change live `torch.Tensor` references in the
  graph or substitute a retained stale handle.  It never changes a slot ID or
  semantic manifest.
- `r40lib/observer_worker.py` is a separate implementation in a separate
  process.  It receives only opaque slot/tensor pairs and derives observations
  without importing the candidate binder or oracle worker.
- `scripts/verify_campaign.py` uses only the Python standard library and does
  not import any executed campaign module.

The oracle and observer intentionally share PyTorch tensor/storage semantics;
that remains in the trusted computing base.  Source and process separation are
not a security boundary against a malicious package author.

## Why these are live-binding faults

The fault injector operates before serialization and changes actual Python
tensor references.  The injection receipt HMAC-obscures producer-local object
and storage identities, records the exact changed slot set, and proves that the
semantic manifest hash and live-wire field set stayed fixed.  The observer
then imports the substituted handles and derives their payload and storage
relations.  No campaign fault edits an observer row, slot label, content hash,
or relation vector after capture.

## Frozen acceptance

The campaign passes only when all of the following are true:

1. all four matched clean lanes pass with no detector code;
2. all four mutant lanes reach the observer and fail closed;
3. each mutant changes exactly its preregistered live-handle slot set;
4. every clean/mutant pair has an identical pre-injection oracle projection;
5. slot IDs, semantic manifest hash, and wire field sets are unchanged;
6. coherent swap and stale-handle mutants fail the content challenge;
7. cross-layer and request/base mutants fail both the content challenge and
   storage-relation check;
8. producer, oracle, and observer PIDs are distinct within every lane;
9. all authored source hashes still match the pre-execution ledger; and
10. the standard-library replay independently reproduces all eight verdicts.

Any unexpected pass, missing lane, wrong failure code, wrong changed-slot set,
oracle drift, label mutation, PID reuse within a lane, or provenance drift
fails the campaign.  There is no threshold, retry, fallback, or target
substitution.

## Residual boundary

The fixture uses deterministic CPU tensors with unique payloads and known
private/disjoint ownership.  This validates a possible live-binding challenge
mechanism and its sensitivity to four fixed handle faults.  It does not close
the real-model trusted-producer boundary: a Qwen/H20 integration would still
need an independent pre-binder semantic-registration hook and a fresh formal
run.  It also cannot distinguish a byte-identical disjoint clone from the
intended tensor unless an allocation-lineage witness is added.

