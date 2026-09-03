# Round 30 revision plan

## Frozen input

- PDF SHA-256: `408f7d495a383cd40df6a4bbe49dbf6ef6e732bb3663eae791a4b96000bfcd39`
- Panel: three independent, identical-prompt, PDF-only `gpt-5.6-terra` reviews
- Scores: `4/10, 4/10, 4/10`; confidence `4/5, 4/5, 4/5`
- Archival synthesis: `meta_review.md`

## Text-only changes required in one coherent manuscript patch

1. Reframe the title, abstract, introduction, discussion, and conclusion as
   trusted-capture ownership-trace validation on one fixed stack.
2. Add a formal conditional trace-validity statement.  Its conclusion is only
   that registered predicates held at captured observation points, assuming an
   honest and mandatory-event-complete producer plus correct byte binding and
   replay.  It makes no claim about coherent omission/fabrication, transient
   restored writes, compiled dispatch, common-mode semantics, or unseen faults.
3. Replace `receipt-complete (RC)` with two separate concepts: mandatory-trace
   coverage (`complete`, `partial`, or `open`) and replay verdict (`pass`,
   `fail`, or `not evaluated`).
4. State the 39-token convention explicitly: the first model call consumes the
   32-token query into state; the next seven calls consume generated tokens
   1--7; generated token 8 is output but is not fed into another call, so the
   state-appended count is `32 + 7 = 39`.
5. Put the primary RR2 result before Mac/deployment/related-work context, retain
   the user-requested Mac and H20 tables, and mark the context as unpooled.
6. Shorten the abstract and distinguish canonical CPU-FP32/digest equality from
   bitwise identity of unarchived device tensors.
7. Position the measured `4.321x` full-capture path as an offline/debug/CI mode;
   move the five-pair detail out of the main result flow if layout permits.
8. Sharpen novelty as a reusable typed trace schema plus heterogeneous
   KV/recurrent-state ownership predicates and fail-closed coverage semantics;
   do not claim novelty for paging, COW, metamorphic testing, or cache policy.

## New evidence lanes

Only a lane that passes its frozen clean/control/replay gates may enter the
manuscript, claim map, integrated results, or validator.

| Lane | GPU | Frozen purpose | Entry gate |
|---|---:|---|---|
| Native batching | 1 | Real vLLM V1 scheduler with ragged admission, decode overlap, turnover, and sequential controls | Clean run, output/logit comparisons, scheduler-visible KV ownership, and independent replay all pass |
| Post-discovery faults | 2 | Repair the discovered real cross-owner alias and then test independently frozen new patterns | Clean regression passes before any new fault is executed; new faults remain distinct from discovered H01--H03 |
| Expanded operator oracle | 3 | Increase attention/GDN captured-boundary coverage with candidate-import-free FP32 replay | Pre-execution source/config freeze, clean tolerance gates, positive controls, and independent replay all pass |

## Explicit exclusions

- Candidate C from the held-out-fault lane is operationally invalid because its
  clean path aliases recurrent state across owners; it remains internal raw
  evidence and is not a manuscript result.
- Failed uploads, infrastructure interruptions, and invalid clean lanes are not
  scientific negative results and cannot be converted into support.
- No result from these lanes authorizes runtime independence, arbitrary
  production scheduling, unseen-fault rates, end-to-end correctness, or
  compiled-kernel attestation unless a separately frozen protocol directly
  measures that claim.

## Final gates

1. Apply the accepted evidence and all text-only changes in one manuscript
   patch.
2. Synchronize the claim map, integrated evidence, and fail-closed validator.
3. Compile and render the PDF; inspect every page and read the full paper from
   abstract through appendices.
4. Freeze a detached PDF-only directory and run three fresh independent Terra
   reviews with the same prompt and no assigned specialties.
