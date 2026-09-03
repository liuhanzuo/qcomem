# R35 historical single-token GDN alias regression

## Why this experiment

Round 34's two rejecting PDF-only reviewers both asked for evidence from an
organically encountered defect and for a same-defect comparison with ordinary
output or differential checks.  Adding another predicate-designed mutation
would not answer that request.

The selected case already exists in the immutable development history.  On
2026-08-25, all three **clean**, no-fault lanes of R29 attempt C reached their
one-token model horizon and then failed at
`gdn_completed_binding_rebound`.  The failure occurred after 24 focused local
tests had passed and before any R29 fault outcome could be accepted.  The
postexecution diagnosis found an integration defect: borrowed request, base,
and unadvanced peer reused the same convolution-state tensor, while the
Transformers 5.14.1 `seq_len == 1` GDN path updated that tensor in place and
bypassed `Cache.update_conv_state`.  Relaxing the audit rule would therefore
have hidden a real cross-owner alias.

The repair, written only after that observation, clones the selected request's
30 borrowed convolution states immediately before its first cached one-token
call.  A prior post-discovery development run showed that the repair preserved
one-step logits and terminal state, but it was not a frozen same-defect
comparison and was not manuscript-authorized.

R35 therefore performs a retrospective, hash-bound **reproduction of the
historical pre-fix path** beside the repaired path and a materialized-state
reference.  It does not byte-for-byte re-execute the archived R29 executor:
the new runner adds outcome-preserving capture needed for the comparison.
There is no injected mutation.  The R29 held-out fault suite and executor are
hard-blocked, while the frozen RR2 dependency passively imports its generic
mutant-definition module; no mutant is requested or applied.  Eight frozen
PG19 rank inputs are run independently on eight H20 GPUs.  The purpose is to
measure detection and localization for this one historical defect, not to
estimate a defect population.

Ranks 0--2 reproduce the three input coordinates on which Attempt C originally
exposed the defect.  Ranks 3--7 are five additional frozen inputs executed
through the same pre-fix path; they are not described as historical outcomes
or as independent samples from a defect population.  Alternating lane order
is a fixed execution control, not evidence that order effects are eliminated.

Attempt C remains operationally invalid under its own frozen protocol, and
none of its rank outcomes is rehabilitated or imported as a paper result.
Only a scientifically valid R35 execution may support the retrospective case
study; Attempt C is used solely to bind the pre-discovery chronology and the
historical failing behavior that R35 attempts to reproduce.

## Immutable historical provenance

- Attempt-C disposition:
  `evidence/r29_heldout_faults/cross_execution/attempt-c-internal-operational-invalid.json`,
  SHA-256 `de1e8e52c36d18f229e9b46a59602935220e7aafacb0de4392b9e6b5e77d8472`.
- Postexecution root-cause analysis:
  `evidence/r29_heldout_faults/cross_execution/attempt-c-root-cause.md`,
  SHA-256 `8945b6f09f1da7d40a1de817b5f56f47cb97d8a614aa3ed9aca47c0e13af677a`.
- Pre-output readiness record showing 24/24 focused tests passed:
  `evidence/r29_heldout_faults/cross_execution/readiness_report-v3.json`,
  SHA-256 `bf5a91c0c4432d7597cbdad1e4ea64eb343dad2c3409b01f35f74d339ce65e01`.
- Archived pre-fix executor:
  `gpu/r29_execute_heldout_faults.py`, SHA-256
  `b5efd926cef0dc6505a8d710af453c38499784491f807ef63a5b3d9bfd63d360`.
- Borrowed-state request builder: SHA-256
  `546efd59e2833034bc2e24d4cc0e6077f5a408275e359af43cd96f7f71cad16e`.
- Functional cache adapter: SHA-256
  `2ede63c74e4799316cc179cd3900f1e26e8dc284da326233376b2ed4c79d3a84`.
- Storage witness: SHA-256
  `57c0dfe457abf165f058faac57200173f8e75f874cd3220510e4ac676a9fc520`.
- Post-discovery pointer-free diagnostic source: SHA-256
  `c0bfd465c78643d711b2b645f9573f9ba0b33a95a000059159fc7f2a8c731435`.
- Post-discovery repair helper: SHA-256
  `4a2938cc99503f54abf91f780034e08ae64e4105a51c0736433b84ff363bad7a`.
- Repair unit test: SHA-256
  `6a8bb7325560378333d750404f9ef1d5f3e4530957e72e14a9f42c362b7d8e56`.
- Prior repair-only development summary:
  `evidence/r30_postdiscovery_d/clean-c-local-mirror/result-and-replay-summary.json`,
  SHA-256 `625fc9f592a6429984d1fbb7ca3d857819937a5a08cbe52408365f628fe01eee`.

The installed Transformers tag `v5.14.1` resolves to commit
`a08ace4bbd97e721c98751deec37d87b026acadc`; the relevant
`modeling_qwen3_5.py` has SHA-256
`0e2cd8dc50885b2701d26b116c585eedcdc62a24080ec34345af55b963126ded`.
This provenance supports the mechanism of **our borrowed-state integration
bug**; it does not attribute a standalone defect to Transformers.

## Frozen comparison

Each rank constructs three fresh lanes from the same frozen document and
boundary token:

1. `historical_pre_fix`: execute the archived borrowed-base one-token path
   without the later repair.  This is the historical implementation behavior,
   not a mutation.  ForkAudit's authenticated first rejection is recorded; the
   lane is allowed to serialize outputs only for retrospective baseline
   comparison and is never treated as a clean pass.
2. `repaired_borrowed`: apply the fault-ID-independent 30-convolution-state
   privatization once, then execute the otherwise identical one-token call.
3. `materialized_control`: rebuild the same case with request-materialized GDN
   state and execute the same token.

Every lane starts from a fresh case and must return the CUDA allocator to the
post-warmup baseline.  The detached replay imports no candidate modules.

The comparison matrix and pair mappings are fixed before execution:

- **output-only:** greedy-token and complete finite FP32-logit equality for
  historical--materialized, historical--repaired, and
  repaired--materialized pairs;
- **state differential:** request-0 terminal GDN content and terminal logical
  KV content for the same three pairs;
- **state invariant:** persistent-base immutability within each lane; and
- **ForkAudit:** lane-local storage intervals, owner relations, and the
  setup-to-transition binding relation.

Counts are reported per frozen cell; no pooled detection rate or false-positive
estimator is computed.

## Acceptance and manuscript policy

Operational validity is independent of whether the expected scientific outcome
occurs.  A run is valid if exactly eight rank artifacts, 24 FP32-logit
sidecars, and 24 unique fresh-case nonces are present and hash-bound; every
lane reaches the one-token horizon; every cleanup is exact; there are no
missing, duplicate, or extra paths; and detached replay agrees with the online
classification.  A failure to reproduce the gate or a repair/control semantic
mismatch is a retained valid negative result when these operational conditions
hold.

The intended headline-positive case additionally requires:

- all eight historical lanes reproduce the authenticated
  `gdn_completed_binding_rebound` first rejection with the R29 held-out fault
  suite blocked and without any requested or applied mutation;
- all eight repaired lanes pass the unchanged storage/binding predicates;
- repaired logits, tokens, terminal GDN content, and logical KV are exact
  against their materialized controls; and
- historical versus materialized tokens and logits remain equal at the point
  where ForkAudit rejects, which supports only the narrow statement that
  output-only checking missed this one ownership defect; and
- every state-invariant catch is reported.  If persistent-base immutability
  catches the defect, ForkAudit may claim earlier or finer owner/layer/family
  localization, not exclusive detection by all conventional testing.

All observed baseline catches and misses remain in the aggregate.  If the
historical behavior does not reproduce or the repair changes semantics, the
valid negative result is retained and excluded from positive manuscript
claims.  If a lane does not reach its horizon, cleanup fails, an artifact is
missing, or an operational error substitutes for an authenticated gate, the
affected execution is operationally invalid.  A mixed but valid result may be
reported only per cell with a narrower claim.

This experiment can support one retrospective historical-defect case study and
a same-defect baseline comparison on the fixed Qwen3.5/H20 stack.  It cannot
support natural-bug prevalence, unseen-fault recall, a detection or false-
positive rate, superiority over all conventional test suites, cross-runtime or
cross-model generality, adversarial capture, independent slot enumeration,
continuous batching, or production-serving correctness.
