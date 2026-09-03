# V27 source-ledger-bound no-bytecode interpreter shim

V27 changes Python invocation hygiene only. Scientific runner inputs, model
inputs, measurement code, ownership verifier predicates, producer receipt
equalities, allocator measurement, and terminal closure rules are unchanged.

## Failure mechanism

The immutable launcher invokes the model-load lease keeper as `$PYTHON -I -c`.
The R39 proxy deliberately passes non-shard commands transparently to
`R39_PRIMARY_REAL_PYTHON`. Environment-only `PYTHONDONTWRITEBYTECODE` is not a
sufficient invariant under isolated mode. The v26 run therefore created a
primary-code `__pycache__` after its preflight absence check; the terminal code
snapshot correctly rejected that writable entry.

## Controlled repair

The generated launcher now freezes two bindings:

```text
R40_ACTUAL_REAL_PYTHON = original real-environment interpreter
R39_PRIMARY_REAL_PYTHON = R40 source-ledger-bound no-bytecode wrapper
```

The wrapper performs only:

```text
exec "$R40_ACTUAL_REAL_PYTHON" -B "$@"
```

Command-line `-B` precedes upstream `-I`, so it cannot be lost through ignored
environment variables. The exact R39 transparent-versus-shard routing decision
is preserved. The wrapper is hash-bound by `source-code.sha256`, staged
read-only with the rest of the overlay, and contains no scientific logic.
