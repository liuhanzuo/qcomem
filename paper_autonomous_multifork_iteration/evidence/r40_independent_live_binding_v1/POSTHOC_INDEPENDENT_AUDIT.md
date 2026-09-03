# Post-hoc independent read-only audit

Date: 2026-08-27

This note was added after the frozen v1 execution and is intentionally not part
of the pre-execution source ledger. It narrows interpretation; it does not
change any frozen result.

## Verdict

PASS for a deterministic CPU/PyTorch fixed-fault mechanism claim. FAIL for any
claim that the experiment closes trusted producer capture or independently
attests Qwen/H20 live binding.

The auditor reproduced 4/4 matched-clean acceptances and 4/4 mutant
fail-closed decisions, reverified all frozen-source and terminal-ledger hashes,
and obtained the same outcome in a fresh no-write eight-lane execution. The
four mutants change actual live \`torch.Tensor\` references while leaving the
manifest, semantic slot set, and serialized live-item field set unchanged.

## Required interpretation boundary

- Semantic \`bind_oracle_items\` executes in the producer process. The separate
  oracle worker performs measurement. The accurate description is therefore
  **source-distinct traversal with process-separated oracle and observer
  measurements**, not an independently secured semantic oracle.
- Candidate and oracle share the frozen manifest, fixture, and coordinate
  convention. A common semantic-mapping error can affect both.
- Standard-library replay independently recomputes verdicts over saved
  artifacts; it does not reobserve live tensors.
- The freeze chronology is supported by filesystem birth times and embedded
  timestamps, but the receipt field
  \`campaign_outputs_existed_at_freeze=false\` is not an external
  tamper-resistant timestamp.
- The detector's general gate requires the preregistered failure codes to be
  present; it does not generally forbid additional codes. The four executed
  mutants happened to produce exactly the preregistered code sets.

## Narrow paper-safe statement

> On a deterministic 18-slot CPU/PyTorch fixture, source-distinct reference
> and candidate traversal code, together with process-separated oracle and
> observer measurements, accepted four matched clean controls and failed
> closed on four preregistered pre-serialization live tensor-reference
> substitutions: a same-geometry swap, a stale reference after owner rebind, a
> cross-layer substitution, and a request-to-persistent-role misbinding. The
> semantic manifest, slot set, and live-item field set remained unchanged.
> This is fixed-fault mechanism evidence and does not independently attest the
> Qwen/H20 producer binding.

