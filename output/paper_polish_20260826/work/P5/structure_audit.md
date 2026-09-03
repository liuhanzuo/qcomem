# P5 Structure Audit

## Paper identity

This paper studies a synthetic late-training label shock and tests whether a FIT-calibrated last-margin alarm can detect the shock and roll deployment back from the final checkpoint to the already available best-validation checkpoint.

## Claim thread

`late endpoint shock -> last-margin alarm -> frozen hold-out settlement -> benefit versus last -> exact equality with best-val/early-stop -> narrow false-alarm and deployment scope`

The thread is scientifically stable, but its last three links are repeated in the abstract, introduction, results interpretation, discussion, limitations, and conclusion. The distinct Route-3 diagnostic interrupts the E1 narrative in the main text and belongs in the appendix.

## Priority repairs

- Major: remove repeated explanations that the gate beats `last` but not best-val; retain one full treatment in Results and one bounded synthesis in Discussion/Conclusion.
- Major: move the Route-3 table out of the main argument and keep it as a clearly separate appendix diagnostic.
- Minor: define the E1 scope in plain language before using the internal label.
- Minor: remove forced page breaks that fragment sections and waste the ICLR main-text budget.
- Minor: shorten the abstract and contribution list without dropping the frozen threshold, paired effect, equality result, or false-alarm boundary.

## Promise-evidence closure

- Detection promise -> 64/64 shocked runs and exact lower bound in Table 4: supported in the registered synthetic carrier.
- Rollback value versus `last` -> paired outer-loss contrast -0.3395 with stored interval: supported.
- Checkpoint novelty -> explicitly not supported; best-val/early-stop equal the gate 64/64.
- Deployment false-alarm behavior -> not supported; healthy trajectories are structurally monotone and `m=0`.

## Main-text constraint

The manuscript already uses an ICLR template. Editorial changes should reduce, not increase, the main text and should leave the appendix outside the bibliography boundary.
