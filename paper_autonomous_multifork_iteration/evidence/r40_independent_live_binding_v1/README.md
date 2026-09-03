# R40 independent live-binding challenge campaign

Status: local CPU mechanism/fault campaign prepared.  Formal result is written
only by `formal/run_local_formal.sh`; no QS or GPU command exists in this
package.

## Question

Earlier evidence independently derives the semantic slot census and observes
the tensors delivered under those slots, but it still trusts the producer to
associate each opaque slot ID with the intended live tensor.  This package
tests a narrower, source-separated challenge mechanism against actual
`torch.Tensor` handle substitutions.  It never edits emitted JSON labels to
simulate a binding fault.

An oracle process observes the correct semantic graph before fault injection
and freezes per-slot content challenges, descriptors, and pairwise storage
relations.  A source-distinct candidate binder then emits live handles to a
different observer process.  The observer receives only `slot_id` and
`tensor`, independently recomputes the same challenge responses and storage
relations, and a fail-closed detector compares them with the frozen oracle.

The four prospectively fixed faults are:

1. a coherent swap of two same-geometry live slot handles;
2. a stale handle retained across a real semantic-owner rebind;
3. a one-way cross-layer live-handle substitution; and
4. request/private-to-persistent/base role misbinding.

Each fault has a fresh-process matched clean lane.  See `preregistration.json`
and `DESIGN.md` for exact targets, precedence, and claim boundary.

## Run locally

From this directory:

```bash
formal/run_local_formal.sh formal_result_20260827a
```

The launcher refuses to overwrite an existing output directory.  It verifies
the frozen source ledger, runs the unit tests, executes eight fresh producer
lanes (four clean and four mutant), independently replays the result, and
writes a terminal SHA-256 ledger.

## Evidence boundary

This is CPU synthetic mechanism and fixed-fault evidence.  It demonstrates
that this challenge design rejects the four implemented live-handle faults
while accepting their matched controls.  It is not a Qwen/H20 result, does not
independently execute a model, does not attest the allocator/driver/runtime,
does not resist a malicious producer that controls both semantic registration
and tensor construction, and does not prove a population detection rate.

