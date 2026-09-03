# R39 independent expected-slot census

Status: **clean archived-H20 audit passed; three resealed negative controls
failed closed as preregistered in the R39 protocol.**

## Question addressed

R33 moved GDN descriptor and relation reconstruction into a separate process,
but its replay still took the producer-emitted slot manifest as the source of
which semantic slots should exist.  R39 removes that circularity for slot-set
coverage.  A source-distinct standard-library verifier derives the expected
slot census from:

- 40 layers with full attention at indices congruent to 3 modulo 4, yielding
  30 independently derived linear/GDN layer indices;
- one persistent owner and two scheduled request owners;
- frozen conv and recurrent state families at state index 0; and
- the three-capture completion schedule.

This produces exactly `3 owners x 30 layers x 2 state families = 180` slots
per capture without reading `cell.slot_manifest.slots`, capture rows, or a
producer row count.  The opaque slot identifier is independently recomputed
from each semantic coordinate, and each archived receiver row must match that
binding and the frozen family-specific tensor geometry.

## Outcome

The clean hash-bound audit passed both policy cells and all six captures:

- 180/180 independently expected slots per capture;
- 1,080/1,080 receiver rows across the experiment;
- 96,660/96,660 receiver-derived pair relations;
- no missing or duplicate slot ids;
- exact slot-id-to-semantic-coordinate binding for every row; and
- exact frozen conv/recurrent descriptor geometry for every row.

Three controls modify a deep copy of one clean capture and then recompute its
row digest, relation digest, row count, and relation count before auditing:

| Control | Intended corruption | Observed fail-closed code |
|---|---|---|
| `C-OMIT-ONE-SLOT` | remove one row | `slot_set_mismatch` |
| `C-DUPLICATE-ONE-SLOT` | duplicate one row | `duplicate_slot_id` |
| `C-SEMANTIC-RELABEL` | swap labels between two same-family rows while preserving their slot ids and tensor descriptors | `semantic_binding_mismatch` |

All three failed closed with their expected codes.  Both internal digests were
valid after every mutation, so these controls exercise the independent census
rather than only stale checksums.

Authoritative machine-readable outputs:

- `artifacts/clean_audit.json`
- `artifacts/expected_slot_census.json`
- `artifacts/negative_controls.json`

## Reproduction

From `paper_autonomous_multifork_iteration`:

```bash
python3 -m unittest discover \
  -s evidence/r39_independent_slot_census/tests \
  -p 'test_*.py' -v

python3 evidence/r39_independent_slot_census/scripts/audit_independent_slot_census.py \
  --protocol evidence/r39_independent_slot_census/protocol.json \
  --input evidence/r33_independent_capture/formal_h20/result/raw/out-of-process-gdn-capture.json \
  --preregistration evidence/r33_independent_capture/formal_h20/result/preregistration/preregistration.json \
  --output evidence/r39_independent_slot_census/artifacts/clean_audit.json \
  --census-output evidence/r39_independent_slot_census/artifacts/expected_slot_census.json

python3 evidence/r39_independent_slot_census/scripts/run_negative_controls.py \
  --protocol evidence/r39_independent_slot_census/protocol.json \
  --input evidence/r33_independent_capture/formal_h20/result/raw/out-of-process-gdn-capture.json \
  --output evidence/r39_independent_slot_census/artifacts/negative_controls.json
```

No GPU is required because the audited tensors were already captured by the
formal R33 H20 run.  `h20/run_read_only_audit.sh` supplies the equivalent safe,
read-only command for an existing H20 host.  It launches no GPU kernels and
does not create, stop, or delete any QS resource.

## Fresh-H20 formal binder

The executable formal path is `formal/launch_r39_h20.sh`; the one-command
Trial-1907358 wrapper is `formal/launch_trial_1907358.sh`.  Its order is fixed:

1. hash-freeze the R39 verifier and exact frozen R33/R29 sources;
2. derive and serialize the 180-slot census before starting any producer;
3. invoke the unchanged, hash-pinned R33 H20 launcher and preregistration;
4. bind the fresh raw capture and passing R33 lifecycle replay to the
   preexecution census semantic hash;
5. run all controls on deep copies and prove the clean raw SHA is unchanged;
6. aggregate only if the clean audit and all exact control codes pass; and
7. recheck all sources and write a terminal artifact ledger.

The fresh run makes no change to R33 scientific configuration, model, data,
policy cells, capture ids, phase schedule, or pass criteria.  R39 adds only
preexecution census freezing plus output/terminal binding and copy-only
negative controls.  Build the transfer package with
`formal/build_execution_bundle.py`; `packages/package_build.json` records its
digest and entrypoint.

## Exact claim boundary

The pass removes producer-emitted row enumeration as the source of expected
slot coverage for the archived R33 experiment and demonstrates fail-closed
handling of emitted-row omission, duplication, and semantic relabeling.

It does **not** prove that a malicious producer put the correct live tensor
under an otherwise correct slot id.  It is not OS/driver allocation ground
truth, independent model execution, KV recapture, transient-write monitoring,
kernel/compiled-binary attestation, or cross-stack evidence.  This is a
retrospective audit of a prospectively frozen R33 H20 capture; the R39 controls
were frozen before their clean/control execution but after the R33 raw result
already existed.
