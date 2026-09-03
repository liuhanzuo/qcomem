# Round 34 synthesis (not an additional reviewer score)

## Outcome

The three identical, independent PDF-only Terra reviews score the frozen paper
`6, 4, 4`, all at confidence `4`; verdicts are one Accept and two Reject.  This
improves over Round 33's unanimous `4, 4, 4`: the accepting reviewer explicitly
finds the fixed-stack conditional claim aligned with the evidence.  Consensus
does not cross the acceptance boundary.

## What the revision resolved

- All reviewers understand ForkAudit as a non-adversarial offline-CI contract
  check rather than security or runtime attestation.
- Trusted producer enumeration and capture semantics are recognized as an
  explicit TCB, not an undisclosed assumption or runtime-independence claim.
- The contribution is read as a method-first, fail-closed ownership contract;
  the designer--executor-separated fault protocol and process-separated GDN
  reconstruction are understood correctly.
- No reviewer reports a numerical inconsistency, incorrect comparison,
  unreadable table/figure, round-label problem, or PDF layout defect.  One calls
  the writing, tables, and cohort authorization exceptionally clear.

## Remaining decision blockers

- The ownership conclusion is still conditional on producer-side slot
  enumeration and paused snapshots; the receiver does not independently
  establish capture completeness, model semantics, KV state, or compiled
  dispatch.
- One Qwen3.5/H20 stack with sequential batch-one calls, bounded geometry, and
  selected captured-boundary oracles remains too narrow for two reviewers'
  desired external significance.
- The five newly frozen faults strengthen evaluation but remain constructed
  fixed-stack mutations rather than natural defects, an unseen-fault estimate,
  or a comparison with a conventional testing baseline.
- The method is still judged an integrative systems-validation contribution;
  broader adoption cost and realistic defect-finding value are not established.
- One reviewer finds the volume of explicitly unpooled related-work context
  distracting, although no reviewer disputes its labels or numerical validity.

## Plateau decision

The repairable presentation and threat-model issues are resolved.  The two
remaining Reject decisions ask for materially different evidence: independent
capture/enumeration, a broader or production-like execution setting, natural or
historical defects, a conventional-testing baseline, or scaling overhead.  No
further prose-only pass can honestly resolve these requests.  Noncentral
unfavorable overhead measurements remain preserved in the internal registry but
are not restored to the paper's claim hierarchy; the PDF retains the qualitative
offline-CI cost limitation.  Round 34 is therefore the stopping point for this
iteration rather than another cosmetic revision.
