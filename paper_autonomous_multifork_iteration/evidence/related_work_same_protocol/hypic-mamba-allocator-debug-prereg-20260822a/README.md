# HYPIC Mamba allocator read-only diagnostic A

Formal P is invalid partial execution evidence. Prefix completed, but all eight
HYPIC schedulers failed before any HYPIC raw/store receipt because the formal
receipt found duplicate entries in `MambaSlotAllocator.free_slots`. Prefix-only
outputs are excluded from aggregation.

This freeze runs one HYPIC GPU0 cell through P's exact warm-prime, warm-hit,
target-publication, and formal-prime boundary. The receipt branch then records
the raw free tensor and any release representation without sorting,
deduplicating, freeing, allocating, or otherwise modifying them. It also binds
the target cache entries and exact live
allocator aliases. A separate client validator rederives all duplicate
positions, counts, domains, and tensor metadata from the raw record.

The launcher pins K, the official repository, model ledgers, data, code, cwd,
environment, GPU, mode, workload rank, and fresh paths. The instrumentation
overlay is limited to K's two tracked hook files plus the receipt module. No
formal receipt may be emitted. Completion requires a reproduced nonempty
duplicate representation, full artifact ledger, server cleanup, and GPU0 at
0 MiB.

This is debug-only evidence. It does not report Store(MiB), throughput, F1, or
a scientific negative result. A formal invariant may change only after the
captured ownership representation is interpreted and independently audited.
