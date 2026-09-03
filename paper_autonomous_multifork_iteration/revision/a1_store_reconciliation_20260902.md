# A1 — Store accounting reconciliation against Eq. 3

Date: 2026-09-02
Action: A1 (blocking, `analysis_required`, no new execution)
Issues: T-02, Q7, R44-REV-3-M4, R44-5-06, R44-4-13
Source: `evidence/qcomem_mixed_validation_60item_20260812d/artifacts/raw/` (48 shards,
60 items x 6 configurations), fields `stored_residual_nbytes`,
`stored_lower_cache_nbytes`, `stored_persistent_nbytes`.

## Result: the defect is real, and it is confined to one component

The manuscript's format is fixed by Eq. 3 and Figure 2c: groups of 64 values,
packed unsigned `b`-bit codes plus BF16 scale and BF16 bias. One group therefore
occupies `64b/8 + 4` bytes against `128` bytes for the BF16 reference, so the
achievable compression is

| width | bytes/group | max compression vs BF16 |
|---|---|---|
| Q8 | 68 | 1.8824x |
| Q4 | 36 | 3.5556x |
| Q2 | 20 | 6.4000x |

### The residual counter is exactly conformant

Checked per item, not in aggregate:

- `stored_residual_nbytes == ceil(elements/64) * (64b/8 + 4)` for **180/180**
  item-configuration pairs across the frozen, same-memory, and minus25 policies.
  Zero mismatches.
- The Q16 residual equals `elements * 2` bytes exactly, so the reference arm
  carries no metadata and is a sound denominator.
- The per-item Q16/Q4 residual ratio is exactly `3.555556` for all 60 items
  (min == max), i.e. precisely the format ceiling.

There is nothing wrong with the residual half of the accounting.

### The lower-cache counter is not reconstructible

| policy | lower-cache bit vector | observed lower-cache compression | ceiling of its **finest** layer | verdict |
|---|---|---|---|---|
| frozen-static | `[8,8,8,4,8,8,8]` | **3.6164x** | 3.5556x | **impossible** |
| same-memory-mixed | `[8,8,4,4,8,8,8]` | **3.8007x** | 3.5556x | **impossible** |
| minus25-mixed | `[8,8,2,2,2,8,2]` | 5.9079x | 6.4000x | within format |

The first two rows are impossible in the strongest sense: 3.5556x is the ceiling
that would apply if *every* lower layer were Q4, whereas only one layer (frozen)
or two layers (same-memory) actually are, and the remainder are Q8 at 1.8824x.
A payload that is five-sevenths to six-sevenths Q8 cannot out-compress uniform
Q4.

### Mechanism

The two counters disagree in a specific, testable way. If the lower-cache
counter omits the 4 bytes per group of BF16 scale and bias that the residual
counter demonstrably includes, its apparent ceilings become `16/b`, i.e. 2.0000x
at Q8, 4.0000x at Q4, 8.0000x at Q2. Every observed lower-cache ratio falls
under those ceilings, while two of three violate the correct-format ceilings.
The hypothesis discriminates cleanly and no observation contradicts it.

Independent corroboration: applying the implied per-layer correction factor
`(8b+4)/(8b)` brings the frozen policy's total compression against split Q16
from the impossible 3.5901x down to roughly 3.35--3.47x, which is inside the
format for the first time. The correction is self-consistent.

## Corrected headline, bounded

Only the lower-cache term moves; the residual and the Q16 reference are sound.

| basis | lower cache (MiB/doc) | total (MiB/doc) | vs full-prefix Q16 |
|---|---|---|---|
| as published | 5.4781 | 9.6609 | 14.1018x |
| correction if all lower layers Q8 (x1.0625) | 5.8205 | 10.0033 | **13.619x** |
| correction if all lower layers Q4 (x1.125) | 6.1629 | 10.3456 | **13.168x** |
| strict floor: every lower layer at Q4 | >= 5.5719 | >= 9.7546 | **<= 13.9662x** |

The published 14.1018x exceeds even the strict upper bound of 13.9662x, so it is
wrong regardless of which correction applies. The realistic corrected value is
**about 13.2--13.6x**, an overstatement of roughly 4--7 percent.

## Status and what is still required

This analysis establishes that the number is wrong and bounds the correction. It
does **not** produce the authoritative corrected value, because the per-layer
Q16 byte breakdown for the lower cache is not in the archived shards; only the
aggregate is. Emitting the authoritative per-component table requires the Store
accountant itself, which is the artifact A11 must restore to
`method_provenance.tsv`. A1 therefore hands off to A11, and A5 and A13 inherit
the corrected denominators from it.

Until the accountant is re-run, the manuscript must not quote 14.10x. The
defensible interim statement is the bounded one above.

## Verification test

Re-run this analysis against the archived shards and require: 180/180 residual
format identities; the Q16 residual identity; and the two impossibility findings
for the frozen and same-memory lower-cache ratios. Then re-run the Store
accountant with metadata included and require every per-component value to equal
`ceil(n/64) * (64b/8 + 4)` bytes for its declared width.
