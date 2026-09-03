# HYPIC Mamba allocator read-only diagnostic D

Debug A is retired without GPU execution after independent audit found that
its replay accepted a jointly forged allocator/tensor/cache receipt. B keeps
that fix but is also retired without GPU execution because its structured P
receipt mislabeled Trial 1879097 as the later debug Trial 1879456. C corrects
only that provenance: Recovery P's asset receipt binds Job 247512 / Trial
1879097 and its launch plan binds the exact `20260822p` run directory. C is
retired without GPU execution because its correction record imprecisely called
the erroneous receipt "unfrozen" even though B had frozen that exact file; D
uses the precise frozen-B provenance label and changes nothing else.

Formal P is invalid partial execution evidence. Prefix completed, but all eight
HYPIC schedulers failed before any HYPIC raw/store receipt because the formal
receipt found duplicate entries in `MambaSlotAllocator.free_slots`. Prefix-only
outputs are excluded from aggregation.

This freeze runs one HYPIC GPU0 cell through P's exact warm-prime, warm-hit,
target-publication, and formal-prime boundary. The receipt branch then records
the raw free tensor and any release representation without sorting,
deduplicating, freeing, allocating, or otherwise modifying them. It also binds
every cache entry's raw token ids, the target segments, and exact live
allocator aliases. A separate client validator rederives all duplicate
positions, counts, domains, tensor byte/pointer relationships, allocator
fields, cache hashes/counts/locks, and target-slot membership from the raw
record and exact target file.

The launcher pins K, the official repository, model ledgers, data, code, cwd,
environment, GPU, mode, literal qasper-6 workload/rank, and fresh paths. Both
model ledgers are checked entry-by-entry (14 weights and 9 artifacts). The instrumentation
overlay is limited to K's two tracked hook files plus the receipt module. No
formal receipt member of any type may be emitted. Completion requires a
reproduced nonempty duplicate representation, exact raw/validation/run-summary
hash closure, full artifact ledger, bounded PID/PGID cleanup, and GPU0 at 0 MiB.

This is debug-only evidence. It does not report Store(MiB), throughput, F1, or
a scientific negative result. A formal invariant may change only after the
captured ownership representation is interpreted and independently audited.
