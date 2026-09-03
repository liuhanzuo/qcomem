# R40 held-out fault evidence audit

Status: **local audit PASS; positive held-out claim HOLD**.

This directory is a non-overwriting, read-only-input audit of the fault
evidence already present in the repository.  It does not alter the paper,
`experiment_registry.json`, any earlier evidence package, or any remote/GPU
resource.

The important result is negative but precise.  The R39 PDF-only blind campaign
is the only completed campaign in the repository that is both fault-definition
separated and outcome-held-out relative to the frozen R39 executor.  Seven of
its eleven frozen rows produced valid reached pairs, but the unchanged
ForkAudit replay rejected none of the seven.  Output/logit comparison exposed
four, allocator comparison exposed two, and one row (BF03) escaped all four
registered observers.  Three rows were pre-execution ineligible and one was
operationally invalid.  These bytes are useful as development/audit evidence,
not as positive held-out sensitivity evidence.

The R35 experiment remains useful evidence for one organically encountered
integration defect.  Its eight cells are input coordinates for the same defect
mechanism, not eight independent defects.  R33 is a valid
designer--executor-separated, preregistered fixed-fault campaign, but its
designer used the public predicate contract and froze an expected primary gate
for every row.  It is therefore not an unseen-fault cohort.  R29 is
operationally invalid, R30 never reached clean inference, and R28 reuses the
original designed M1--M9 mechanisms.  The R23 scheduler faults and the seeded
attention/GDN wrong-operator rows are likewise predicate/oracle design
controls, not held-out implementation defects; the R40 binding substitutions
are fixed mechanism tests.

Run the local audit with:

```bash
cd /Users/liuhanzuo/MacLLM-Bench/paper_autonomous_multifork_iteration/evidence/r40_heldout_fault_v1
./run_local_audit.sh
```

The command verifies frozen input hashes, checks all 173 locally retained R39
files against the 605-entry terminal ledger, reproduces the R39 aggregate
byte-for-byte from its eleven terminal outcomes, and writes/validates
`audit_result.json`.  The 432 omitted ledger entries include the full FP32
sidecars and compiled artifacts, so this package cannot honestly claim a full
pair replay.

`V2_PROTOCOL.md` and `v2_campaign_plan.json` describe the smallest defensible
next step.  They are a design, not an executed result or an authorized H20
launcher.

`REGISTRY_SUGGESTION.json` supplies inactive/internal entries for the R39
negative campaign and this audit.  It is intentionally not applied here,
because this task forbids changing the existing registry.
