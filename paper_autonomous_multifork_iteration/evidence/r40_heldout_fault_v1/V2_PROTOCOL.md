# Scientifically honest v2 path after the R39 blind campaign

Status: **design only; detector-v2 bytes, fresh fault bytes, and H20 execution
are not yet frozen or authorized**.

## 1. What R39 may and may not do

R39 becomes development/audit evidence for v2.  Its eleven fault definitions,
all seven valid outcomes, three ineligible outcomes, one invalid outcome, and
all earlier M/H/HF/R30 definitions belong permanently to the **design set**.
They may motivate general contract changes, tests, and implementation repair.
They may never be rescored or relabeled as v2 held-out evidence.

The frozen R39 method itself has no positive held-out result: unchanged
ForkAudit replay rejected zero of seven valid reached mutants.  Four were
exposed by exact output/logit comparison, two by allocator comparison, and
BF03 escaped all four R39 observers.  This must remain in the internal evidence
record even if v2 later succeeds.

## 2. Legitimate v2 contract expansion

The revised contract may add the following **general**, always-on obligations
before any new fault specification is created:

1. **Paired semantic differential envelope.**  For every registered call in a
   matched reference/clean/candidate protocol, bind call cardinality, exact
   generated tokens, and shape/dtype-bound complete-vocabulary CPU-FP32 logit
   bytes.  Missing comparands are `open`, never pass.
2. **Paired allocator envelope.**  Bind synchronized current and peak allocated
   bytes at H0/H1/H4/H6/H7 and exact H7-to-H0 restoration.  The denominator is
   PyTorch allocated bytes only; it is not NVML memory or service capacity.
3. **Hybrid atomic-version coherence.**  At every scheduled state-changing
   call, bind a single commit ID to request ID, call/round ordinal, input token,
   pre/post logical KV length and digest, pre/post GDN digest, KV and GDN
   storage/binding versions, and the surfaced logit sidecar.  A valid commit
   advances the scheduled call exactly once and publishes KV and GDN under the
   same commit ID.  No subsystem may advance, roll back, or publish alone.
   Post-state is re-read from the actual live state after the call; producer
   self-reported version strings alone are insufficient.

The first two gates are conventional paired baselines incorporated into the
v2 validation envelope.  They must remain separately attributed in every
table.  They are not evidence that ForkAudit's structural receipts uniquely
found an error.  The third is a new structural receipt motivated by BF03 and
must be implemented generically for every call, not keyed to a fault ID or
payload.

Additional R39-specific predicates may be added only if expressed as uniform
schema obligations over all clean and candidate executions.  Fault-ID branches,
payload sentinels, or a list of R39 hashes in the detector are forbidden.

## 3. Freeze order that makes the next campaign truly held-out

The next campaign is genuinely outcome-held-out for **method v2** only if this
order is hash-bound and enforced:

1. Freeze the v2 paper contract, executable detector/replay source, predicate
   registry, thresholds, schemas, capture points, clean gates, comparison
   baselines, and the no-fault clean package.  Run clean-only local and H20
   preflight.  No fault designer has been spawned yet.
2. Record the v2 detector/package SHA-256 values in a non-overwriting
   preregistration.  From this point, any predicate or threshold change creates
   v3; it may not consume the coming v2 outcomes.
3. Spawn a fresh isolated fault designer.  Give it only the frozen revised
   contract/PDF, the fixed stack/geometry, and an injection-adapter interface.
   Do not give it R39/R29/R30/R33 fault files, outcomes, reviews, detector
   implementation, or author plans.  The designer freezes a fixed set of
   faults, selectors, eligibility rules, horizons, and independent injection
   witnesses before the executor reads them.  It must not assign or optimize
   an expected detector gate.
4. A separate executor maps the frozen payloads without changing detector-v2.
   Static/unit/CUDA-smoke review may reject an impossible selector as
   pre-execution ineligible; it may not substitute a target.  All eligibility
   decisions are hashed before candidate output.
5. Run all matched reference/clean/mutant rows once.  Preserve every valid,
   escaped, ineligible, invalid, and clean-false-positive row.  No selective
   rerun or post-output predicate amendment is allowed.
6. A detached scorer receives only frozen bytes and produces per-gate outcomes.
   Report structural ForkAudit, semantic differential, allocator, production
   assertion, and injection-validity outcomes in separate columns.

Seeing the revised public contract does not invalidate held-out status: test
cases can be adversarially designed against a public specification.  The key
is that the method bytes and decision rules precede the new fault bytes and do
not change afterward.  Calling the new campaign “detector-blind” would be too
strong because the designer sees the public contract; “method-v2 held-out,
designer--executor separated” is accurate.

## 4. Reporting and success gates

Operational success is independent of how many faults are rejected.  It
requires a passing matched clean for every executed row, exact injection
witnesses, complete artifacts, detached agreement, and retention of all
outcomes.  Scientific outcomes are reported per fault; no population recall,
accuracy, or prevalence estimate follows from a small constructed set.

If v2 rejects a new valid fault, the narrow claim is that the named frozen v2
gate rejected that independently frozen case.  If only output/logit or
allocator comparison rejects it, attribution stays with that baseline.  If a
fault escapes, it is a v2 scope boundary.  A positive paper claim remains HOLD
until these bytes exist and pass the integrity audit.

Constructed held-out faults alone still do not establish natural-defect recall.
The existing R35 case may remain the one natural historical case.  A stronger
future study should prospectively retain every integration defect encountered
after v2 freeze in an independent port, with no outcome-based selection.

## 5. One-shot H20 feasibility

The path is feasible on one eight-H20 node once the v2 package passes local and
CUDA smoke review.  The prior R39 eight-GPU campaign started at 22:25:04Z and
completed at 22:29:31Z in its retained stage timestamps (267 seconds), showing
that the matched full-logit/allocator envelope is operationally feasible on
the same bounded stack.  This is not a runtime guarantee for v2.

Use eight frozen fault rows, one per H20, each with fresh reference, clean, and
mutant processes.  Budget one setup/smoke phase plus a 15-minute formal timeout;
OOM, timeout, or an early exception is operational-invalid rather than a
detection.  The launcher must require the exact authorization variable
`R40_H20_EXECUTION_AUTHORIZED=yes`, exactly eight distinct H20 UUIDs, frozen
detector/fault/package hashes, a non-existing output root, and no retry flag.
No launcher is supplied in this v1 directory because no audited v2 executable
package exists; emitting a runnable wrapper around unfrozen code would weaken,
not strengthen, the evidence chain.

