# V9 authority and lifecycle design

The signed binding commits the canonical archive and ledger, snapshot and inventory, runner manifest and inventory, runtime expectation, execution contract, unique run nonce, positive attempt, and exact existing terminal-root inode/path. Verification, isolated provenance probe, and complete typed contract checks occur before consumption or worker creation.

Authority is function-local and consume-once. After atomic no-replace publication of `AUTHORIZED_CONSUMPTION.json`, each V9F01--V9F08 spec is freshly re-derived and all signed anchors are re-read. Signals are masked across `Popen` and complete receipt registration. Finalization kills owned process groups, waits for death, rehashes inputs, and publishes exactly eight immutable terminals. Every terminal commits the consumption, binding, contract, expectation, probe, intended-spec set, and every worker's actual process/materialized command identity.
