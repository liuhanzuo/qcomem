# Lifecycle raw-first replay

From the lifecycle package root, run:

```bash
./replay/run_replay.sh --self-test
```

The entry point first verifies `replay/MANIFEST.json` and its detached digest,
then verifies the frozen raw-shard ledger before parsing any shard.  It
reconstructs lease epochs/owners, reservation disjointness and exact
reassignment, zero-scrub receipts, document immutability, aligned append,
full-vocabulary hash/token equality, logical-KV equality, GDN equality, and the
stale-handle expected-gate result for all eight ranks.  The self-test mutates a
raw byte and, after an in-memory rebind, independently corrupts the stale gate,
scrub receipt, and reservation reassignment; all four must be rejected.

This is an offline validation of frozen raw artifacts, not a fresh GPU run or
a second model/runtime implementation.
