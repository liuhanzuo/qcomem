# Eight-H20 one-shot executor skeleton

This skeleton is intentionally non-executable as frozen: every external
binding in `formal-execution.template.json` is null and the template is not
sealed.  After a separate fault-design freeze and independent audit, an
authorized operator may create a new formal configuration without editing this
method package.

The driver enforces:

- explicit `R40_H20_EXECUTION_AUTHORIZED=yes` authorization;
- one new, nonexistent output root and an exclusive one-shot lock;
- exactly eight distinct configured UUIDs, all present on an eight-GPU H20
  node;
- hash verification of the method manifest, fault set, runner manifest, and
  every file named by the runner manifest;
- exactly eight unique frozen cases;
- concurrent one-case-per-GPU execution;
- one ordered reference, clean, and mutant lane attempt per case, subject to a
  hard 900-second whole-case deadline;
- no shell command expansion, retry, payload change, or case replacement; and
- a terminal JSON record for every case, including lanes not started because a
  preceding attempt exhausted the deadline.

The driver only records execution terminals.  It does not calculate or select
scientific outcomes; the frozen verifier must do that later over the complete
ledger.  A failed reference or clean lane is retained as an operationally
invalid case, not converted into a favorable detection.

Formal invocation, only after all bindings and authorization exist:

```bash
R40_H20_EXECUTION_AUTHORIZED=yes \
  ./launch_h20_one_shot.sh /absolute/formal-execution.json \
  /absolute/new-output-root
```

