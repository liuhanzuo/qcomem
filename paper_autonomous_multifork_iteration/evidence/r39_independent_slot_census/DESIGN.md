# Design and falsification contract

Date: 2026-08-26

## Dependency split

The verifier uses only Python's standard library and its own `protocol.json`.
It imports no R33 producer, slot-manifest builder, observer, ForkAudit replay,
candidate runtime, Transformers, vLLM, or PyTorch module.  The archived R33
producer manifest is treated as an untrusted receipt and compared against the
independently derived census only after the census exists.

The R39 protocol restates the frozen model geometry and schedule needed for an
expected-slot census.  Linear/GDN layers are generated from a 40-layer periodic
architecture rule rather than copied as an emitted 30-index list.  Owner
coordinates are generated from resident count two, not copied from capture
rows.  State families and their allowed geometry are fixed explicitly.

## Acceptance rules

A clean audit passes only if:

1. the archived raw result and R33 preregistration have their fixed SHA-256
   digests;
2. the result binds the fixed preregistration;
3. the independently derived census has exactly 180 unique opaque slots;
4. the two policy cells and three capture ids exactly match the independent
   schedule;
5. every capture contains exactly the independently expected slot-id set with
   no duplication;
6. every slot id binds the expected owner, request, layer, family, and state
   coordinate;
7. every row matches its family's frozen descriptor geometry;
8. receiver row and pair-relation digests recompute; and
9. all six captures total exactly 1,080 rows and 96,660 pair relations.

The audit never substitutes the producer's `row_count`, manifest slots, or
capture rows for an expectation.

## Negative-control construction

Controls operate after the pristine raw input digest is verified.  Each makes
one structural mutation in memory, recomputes the capture's row and relation
digests and cardinality receipts, and invokes the same structural audit.  The
test harness accepts only the exact registered failure code.  A control that
passes, fails at a different earlier check, or retains a stale internal digest
fails the control campaign.

The semantic relabel uses two conv rows, which share the same tensor geometry.
Only their semantic coordinate labels are exchanged; slot ids and tensor
descriptors remain fixed.  It therefore cannot be rejected merely because a
conv tensor was renamed recurrent or because its shape changed.

## Fresh-run temporal binding

For the new H20 execution, `generate_preexecution_census.py` runs before the
unchanged R33 launcher.  Its receipt explicitly records that no producer
manifest or row existed as an input to derivation.  The terminal aggregator
requires the clean audit's census semantic hash to equal this preexecution
receipt and requires the R33 replay's canonical input hash to equal the fresh
raw capture.  It also verifies the raw file hash before and after controls.

This wrapper does not revise the R33 scientific configuration: it uses the
same preregistration SHA, eight frozen R33 source hashes, R29 runtime adapter,
model revision, data, query bank, two policies, capture ids, and completion
schedule.  Its only additions are preexecution census generation and
postexecution/terminal binding.

## Residual trusted computing base

The independent census says which semantic rows must be emitted and rejects
wrong emitted labels.  It cannot independently identify the semantic meaning
of a live tensor handle before the producer assigns its opaque slot id.  A
malicious producer could potentially substitute a same-geometry tensor under a
correct slot id.  Closing that stronger boundary would require independent
live model execution or allocator/runtime instrumentation that binds live
allocations to semantic coordinates outside the producer.
