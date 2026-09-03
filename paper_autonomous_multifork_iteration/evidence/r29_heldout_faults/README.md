# R29 held-out realistic-pattern fault suite

Status: **frozen; awaiting independent cross-execution**.

The suite adds three faults not used to define the original nine R28
mutations: stale prefix identity/page order (H01), hybrid recurrent
resume-boundary double application (H02), and a latent future allocator
reservation rebound to a peer (H03). They were selected from upstream serving
bug patterns first, then transferred as small reversible interventions to the
fixed Qwen3.5/H20 stack. No detector outcome is encoded in the suite.

The source reports are provenance for the *fault patterns*, not experimental
subjects. The paper may describe these only as historical-pattern-inspired,
independently authored held-out mutations unless a later experiment separately
reproduces an upstream bug.

See `EXECUTION_HANDOFF.md` for the cross-run interface and strict scoring
contract. Candidate results must be written under a new formal-run subdirectory
and must not overwrite `preregistration/`.
