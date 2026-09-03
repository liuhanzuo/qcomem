# R33 Fresh Held-Out Fault Protocol - Author Freeze

## Source and independence declaration

I designed this freeze from one project artifact only:

`/Users/liuhanzuo/MacLLM-Bench/paper_autonomous_multifork_iteration/main.pdf`

Its SHA-256 is `a34f319550300d603db259a69c5685112009b2d0a3d92aa3096a121624fb6db3` and `pdfinfo` reports 24 pages. I read the complete PDF text and visually checked relevant rendered pages. I did not inspect LaTeX, implementation code, scripts, pre-existing evidence, old fault files, reviews, or state files. I did not run experiments. The only non-PDF project files I accessed are the new files I authored in this `author_freeze` directory.

## Purpose and boundary

This document freezes five held-out live faults before execution. They are intended to be realistic, implementable on the paper's publicly described fixed RR2 case, and discriminative between a matched clean execution and a one-fault mutant. They do not estimate a detection rate and do not extend the paper's claims beyond honest, mandatory-event-complete, byte-bound trace validation at registered observation points.

The execution boundary is the PDF-declared fixed stack and geometry: Qwen3.5-35B-A3B, ten full-attention plus thirty GDN layers, BF16/Q16 paged KV with 128-token pages, a 4,095-token document with a 127-token valid tail, a 32-token registered transition, eight generated outputs, 39 state-appended inputs, and batch-one sequential round-major calls on one CUDA stream. Each held-out pair uses the PDF's fault-campaign fan-out of N=2.

No fault requires compiled-binary identity, autotuning attestation, continuous/ragged batching, kernel concurrency, in-flight cancellation, independent recapture, an end-to-end correctness oracle, or detection of a mutation restored between registered snapshots.

## Pair construction

For each fault independently:

1. Freeze the full identity ledger and the concrete request, ownership cell, phase, call/layer ordinal, and mutation payload stated in `FAULTS.json` before either pair member executes.
2. Rebuild a fresh matched clean member and mutant from the same frozen identity. Do not reuse live state across pair members.
3. Apply exactly one named mutation to the mutant. Keep all audit gates enabled; do not perform target-gate suppression.
4. Capture mandatory setup, registered-transition, ordered execution, and final records honestly. Every required slot must have exactly one schema-valid, byte-bound record.
5. Replay the clean member first. Only if its clean gate passes may the mutant be classified.
6. Replay the mutant with the frozen predicate precedence. The expected primary predicate must be the first relevant failed predicate; earlier unrelated failures invalidate the trial rather than count as detection.
7. Preserve every outcome, including invalid or operationally failed cases. Do not replace, rerun selectively, or tune the payload after observing results without issuing a new freeze version.

## Global clean gate

A matched clean member passes only when all of the following hold:

- all applicable mandatory record slots exist exactly once and bind to archived bytes;
- identity, prefix immutability, private ownership, tail-safe append, cross-arm, and cross-N predicates applicable to the case pass;
- dispatch passes at the PDF's Python-call scope and remains labeled partial for compiled-binary/autotuning coverage;
- the frozen sequential round-major call and append ledger has the exact expected order and cardinality;
- no unallowlisted exception or missing record substitutes for a verdict.

If the clean gate fails, the paired mutant is not evidence for or against the fault.

## Frozen faults and first detection ports

### HF01 - Delayed tail detach

At the first full-attention layer in frozen model order, request r0 performs its first append write to the still-shared partial document page and only then executes the otherwise correct 127-token private-tail copy. This differs from the disclosed omitted-COW mutant because a copy exists; only ordering is wrong.

Primary invariant: `copy_ordinal < first_write_ordinal`. The redundant port is physical document-page digest equality across setup, transition, and final. The mutant succeeds as a held-out test only if the honest trace shows write-before-copy and earlier identity/storage-role checks pass.

### HF02 - Inactive document-lane scribble

After setup and before tail copying, flip one bit in the one inactive token lane of the shared 127-of-128 K tail page at the first full-attention layer. Valid document tokens, the V page, page tables, private reservations, calls, and GDN state remain unchanged.

Primary invariant: the byte-bound physical document K page, including inactive backing bytes, has identical setup, transition, and final digests. Logical valid-token KV and outputs may remain identical; that is expected and does not rescue the physical immutability violation.

### HF03 - Duplicate committed dispatch

For r0's first generated-token feedback call, commit the correct call once, discard only its returned output, and issue the same call again before r1's scheduled call. Both committed transitions must remain in the honest call and append ledgers.

Primary invariant: the ordered call ledger has exactly the frozen request-round order and cardinality, with one committed transition per scheduled state-appended input. This is a retry/double-commit fault, not a sequence-ID swap.

### HF04 - Effective scale drift

For r0's first full-attention call at the registered transition, use exactly twice the frozen effective scale and keep callable identity, query, mask, positions, append history, and paged-view type unchanged.

Primary invariant: the call receipt's effective scale equals the frozen protocol value in its declared scalar encoding. Semantic or selected numerical mismatch is secondary; the fault is decided at Python-call-contract scope.

### HF05 - Stale GDN binding token after correct rebind

After r1 correctly allocates and populates all 60 private GDN tensors at the registered transition, retain its setup-phase binding token while preserving its new storage token, private ranges, and exact tensor contents.

Primary invariant: a completed borrowed request changes both binding and storage tokens at transition. The physical disjointness predicates should pass, making the stale lifecycle token - rather than base/peer aliasing - the isolated failure.

## Classification

A fault trial is a success only if:

- its matched clean member passes the global clean gate;
- the exact frozen injection receipt is present and unambiguous;
- every earlier unrelated predicate passes;
- the expected primary predicate rejects the mutant;
- no missing record, capture ambiguity, or unallowlisted exception substitutes for rejection.

A fault trial fails if the mutant passes its expected primary predicate, if the injection is absent or materially different, or if the trial is invalid for any reason above. Secondary semantic outcomes must be reported as `same`, `changed`, or `not_observed`; they never override the primary classification.

## Freshness check

These five faults do not reuse the PDF-disclosed live mutations: reservation alias, sequence swap, omitted COW, GDN-base alias, GDN-peer alias, position offset, materialized mask, wrong callable, dense KV view, stale cancelled lease, wrong-request lease, or reclaim-before-scrub. They also do not reuse the disclosed GDN or R30 wrong-operator controls.

The distinctions are intentional: HF01 tests COW ordering with a present copy; HF02 tests physical inactive bytes; HF03 tests retry double-commit cardinality; HF04 changes only effective scale; and HF05 preserves correct physical GDN isolation while making lifecycle metadata stale.

## Freeze rule

`FAULTS.json` is the normative machine-readable definition. `PROTOCOL.md` is the human-readable interpretation. `MANIFEST.sha256` binds both. Any change to fault locus, payload, invariant, detection port, clean gate, or decision criterion requires a new freeze version and a regenerated manifest before execution.
