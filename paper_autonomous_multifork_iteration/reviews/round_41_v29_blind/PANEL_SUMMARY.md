# Round 41 V29 PDF-only blind-review panel

Date: 2026-09-02 (Asia/Shanghai)

Reviewed artifact: `build/r41_v29_live_binding_v1/main_r41_v29_live_binding.pdf`

The three reviewers were instructed to read only the rendered PDF and not the
TeX, repository evidence, handoff notes, or prior reviews.  No reviewer edited
the manuscript or evidence.

## Scores

| Reviewer | Overall | Confidence | Disposition |
|---|---:|---:|---|
| Technical correctness | 6/10 | 4/5 | weak accept / borderline accept |
| Systems / ML infrastructure | 4/10 | 4/5 | marginal reject |
| Editorial / presentation | 6/10 | 4/5 | blocker until page-limit repair |

## Consensus strengths

- The coverage-versus-verdict distinction and fail-closed contract are useful.
- The threat model and TCB exclusions are unusually explicit and honest.
- The selected-cell run is a genuine improvement over caller-supplied receipts:
  it reaches the actual builder, direct `aten.clone` lineage, and production
  serializer for its declared materialized-GDN cell.
- The arithmetic for 144 selected rows, 12,960 full rows, 3,840 clone edges,
  24 artifacts, 1,080 observations, 96,660 relations, and the 54.5% allocator
  result is internally consistent.
- The historical alias demonstrates that output and terminal-state equality can
  miss a real ownership defect.

## Consensus concerns

1. The V1 conclusion and reproducibility statement spilled onto page 10.
2. The PDF did not clearly distinguish the six owner-addressed selected anchors,
   the complete 540-row serializer universe, the 480 direct clone edges, and the
   separate 96 clean-memory calls.
3. “96 calls with zero hook events” was easy to misread as evidence that the
   selected hook never ran.  The actual meaning is an isolation proof: all
   clean-memory calls were observed and none ran inside the selected ownership
   context.
4. “Dispatch provenance” could be read as executed-kernel provenance although
   the evidence stops at host-side selected artifact/configuration and normal
   launcher return.
5. The strongest live binding covered the materialized cell, whereas the
   historically risky borrowed-to-private transition remained trace-relative.
6. The event sequence and capability semantics were under-specified relative to
   the amount of artifact-count detail.
7. The pointer-free “cannot miss a shared byte” statement needed a local
   condition on correct storage-ID/backing-storage mapping and the dense-strided
   schema.

## Actions taken after the panel

- Compressed the main text so Conclusion and Reproducibility remain on page 9
  and References start on page 10.
- Replaced ambiguous zero-event wording with “zero selected-context overlaps”
  into the separately enumerated clean-memory cells.
- Defined the evidence units and added a field-level appendix table.
- Stated that the hook wraps the selected ownership-witness invocation in the
  same formal run, not a shadow invocation.
- Renamed target 5 to host-side dispatch-selection provenance and retained the
  device/driver/ATen/CUDA exclusions.
- Added the five-step builder/clone/capability/serializer/publication sequence
  and an operational TCB decomposition.
- Conditioned pointer-free interval completeness on the storage-ID and
  dense-strided assumptions.
- Added the exact attention/GDN call-count factorization.
- Began a non-overwriting V30 successor for the borrowed-to-private transition;
  no V30 result is treated as evidence until its formal terminal path and an
  independent post-run check pass.

## Current panel disposition

V29 is admissible bounded evidence and materially improves the manuscript, but
the panel is not treated as a final acceptance gate because the systems review
identified the borrowed-transition gap.  A fresh PDF-only review is required
after V30 either passes and is integrated, or fails and the claim is explicitly
left at the V29 materialized-cell boundary.
