# Round-29 held-out fault cross-execution handoff

This workstream is frozen but intentionally not executed by the fault author.
It contains three **historical-pattern-inspired, independently authored
mutations**. It does not contain a naturally occurring bug, and it does not
reproduce or evaluate the upstream vLLM implementations cited as provenance.

## Frozen inputs

- `preregistration/heldout-fault-suite.json`: fault definitions, sources,
  lanes, uniform baselines, uniform receipt battery, and decision rules.
- `preregistration/author-freeze-receipt.json`: raw/canonical suite digests and
  author/executor separation receipt.
- `preregistration/fault-author-code.sha256`: fault module, unit test, launcher,
  and suite raw-byte ledger.
- `gpu/r29_heldout_fault_suite.py`: the only fault implementation authority.
- `gpu/r29_launch_heldout_faults.sh`: generic three-rank launcher.

The executor must not edit any of these files. If a frozen file is defective,
the attempt must be recorded as a superseded preregistration rather than
silently patched in place.

## Required executor interface

The separate executor is passed the following arguments by the launcher:

```text
--suite PATH
--expected-suite-raw-sha256 SHA256
--expected-suite-canonical-sha256 SHA256
--execution-input PATH
--expected-execution-input-sha256 SHA256
--fault-id H01|H02|H03
--rank 0|1|2
--expected-gpu-uuid GPU-...
--output PATH
--sidecar-root PATH
```

Each rank must build three fresh disposable cases in this order: `clean`,
`fault_conventional`, and `fault_forkaudit`. H01 and H03 must call
`apply_state_fault`; H02 must consume `h02_action_sequence` exactly. The
executor may only adapt the frozen operations to the already-pinned R28 model,
data, and resident-cache context.

The execution-input JSON is authored by the executor and frozen before output.
It must bind model/data ledgers, frozen query-bank digest, imported RR2 code
ledger, environment, executor/aggregator code digests, and output root. The
launcher separately binds its raw SHA-256.

## Uniform execution and scoring

For every fault, persist the intervention/restoration or disposable-case
receipt, then apply the same baseline and receipt rules from the suite. Do not
add a fault-specific detector expectation. In particular:

1. Generic crashes are operational failures, not catches.
2. Missing token/logit outputs are `not_evaluated`, never `not_caught`.
3. The conventional lane reports completion, exact production assertion if
   any, greedy-token equality, and full FP32-logit equality/differences.
4. The ForkAudit lane runs the same ordered receipt battery for all faults and
   records the first authenticated rejection plus all receipts completed before
   it.
5. Valid escapes and negative outcomes remain in raw and summary artifacts.
6. Report three per-fault rows; do not pool them into a detection rate.

## Aggregator contract

The separate aggregator must accept the launcher arguments and emit
`forkaudit-r29-heldout-fault-summary-v1`. It must verify:

- exactly ranks 0/1/2 and faults H01/H02/H03;
- exactly three fresh lanes per fault;
- source/input/suite/code bindings and distinct H20 UUIDs;
- intervention non-no-op plus restoration for H01/H03;
- exact H02 action-sequence digest and fresh-case disposal;
- clean completion, finite FP32 sidecars, and matched advertised horizons;
- evaluated/not-evaluated semantics for every detector cell;
- zero operationally invalid cases before `scientific_valid=true`;
- `detection_rate_reported=false` and `naturally_occurring_claimed=false`.

No QS resource was created, submitted, stopped, or modified by this authoring
workstream.
