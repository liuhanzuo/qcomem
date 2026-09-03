# Round 32 synthesis (not an additional reviewer score)

## Outcome

The three identical, independent PDF-only Terra reviews score the frozen paper
`6, 4, 4`, all at confidence `4`; verdicts are one Accept and two Reject.  This
is a modest improvement over Round 31's `4, 4, 4`, but not a consensus change.

## Shared positives

- All three recognize the output-equality gap and the value of explicitly
  witnessing mutable recurrent-state ownership.
- All three view coverage-versus-verdict separation, mandatory fail-closure,
  byte-bound records, and phase-aware storage/lifecycle obligations as concrete
  technical strengths.
- No reviewer reports a numerical inconsistency, invalid comparison, missing
  result, unreadable figure, or PDF layout defect.
- The accepting reviewer finds the fixed-stack evidence sufficiently useful and
  well scoped for a weak accept.

## Remaining decision blockers

- The dominant blocker remains the honest, event-complete capture-producer
  assumption; the same-process observer does not establish independent live
  truth.
- Two reviewers still find one model/hardware/schedule and sequential execution
  too narrow for acceptance-level methodological impact.
- The designed faults demonstrate expected sensitivity but not held-out or
  naturally occurring bug coverage.
- Reviewers continue to see the novelty as an integration unless a stronger
  independent observer, different execution setting, or blind fault baseline is
  added.

## Effect of the claim-hierarchy revision

The exact negative cost results no longer dominate any review.  Reviewers now
mention overhead only as an unanswered practicality question, while their core
score rationale is trusted capture, external validity, and comparative novelty.
The rewrite therefore improved emphasis as intended, but further score movement
likely requires new central evidence rather than reinserting noncentral cost
numbers or adding more caveats.
