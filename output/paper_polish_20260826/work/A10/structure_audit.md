# A10 Structure Audit

## Paper identity

This paper uses a matched 135M masked-diffusion negative-control pilot to show that two degenerate emissions can differ on audited emission coordinates even when per-token scores do not supply an independent quality ranking.

## Claim thread and closure

- Problem → per-token metrics can obscure materially different degenerate emissions.
- Design → matched AR-init and fixed random-init checkpoints under the same short training and generation protocol.
- Evidence → 2.262-nat last-logged loss gap across four paired data-order seeds; one-prompt emission audit; MAUVE not recorded.
- Boundary → no quality anchor, metric-capability claim, new metric, training discovery, or generalization beyond the pilot.

## High-impact findings

- Major: the abstract and introduction repeat the same “not a new metric/method/discovery” boundary several times, delaying the measured result.
- Major: the title uses per-word `\mbox` wrappers that impede readability and are unnecessary for the scientific identity.
- Minor: the three-category taxonomy of reproduction/diagnosis/new method is useful but overexplained; compress it into one contribution paragraph.
- Minor: keep the limitations table because it efficiently aligns each support boundary with its inferential ceiling.

## Priority edits

1. Simplify the title without changing scope.
2. Rewrite the abstract in result-first order.
3. Compress the introduction's repeated taxonomy and scope statements.
4. Keep the exact numerical and “MAUVE not recorded” boundaries unchanged.
