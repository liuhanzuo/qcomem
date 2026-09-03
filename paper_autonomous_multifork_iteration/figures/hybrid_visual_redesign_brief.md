# Hybrid teaser and architecture redesign brief

## Decision

Use the visual grammar of the `figures4papers` README's miscellaneous examples
as a high-level reference, not as reusable source material.  The upstream
repository is pinned at commit
`6790a93af3552539d955d77181c818916e1700b7` and has no repository-level license
file.  Do not copy its images, icons, source code, or panel layouts.

The final ForkAudit visuals should combine:

1. deterministic, evidence-bound text, arrows, tensor/state geometry, and
   numerical annotations;
2. manually authored composition, spacing, emphasis, and panel balance; and
3. optional raster texture only for non-authority illustrative elements.

Every output keeps an editable source and a flattened PDF/PNG review asset.

## Teaser: one causal story

The teaser should be read left to right in roughly ten seconds.

### 1. Ambiguity

Show one document KV region and one persistent GDN base forking into three
requests.  Use blue solid arrows for document sharing and green dashed arrows
for setup-time GDN borrowing.  Highlight a single post-write alias in muted red
with the short question: `Who owns mutable state after the fork?`

### 2. Contract

Make this the visually dominant region.  Use two horizontal lanes:

- `KV`: immutable document region plus private append tail;
- `GDN`: setup-time local/read-only alias, then request-private mutable state.

Use one orange vertical divider labelled `first write: rebind`.  Keep
`Setup`, `First write`, and `Generation` as the only phase labels.  Arrows must
encode actual state transitions, not decorative flow.

### 3. Bounded evidence

Use an understated evidence strip rather than a dashboard.  Retain only the
registered values needed for the paper identity:

- `96` exact factor trajectories;
- `8` dense-attention oracle rows, maximum relative L2 `0.001743`;
- `9` intended mutant gates; and
- at `N=32`, final allocated memory `4.90 -> 2.23 GiB`.

End with the explicit boundary: `one model · one schedule · no speed claim`.
The figure is a presentation of registered evidence, never its authority.

## Architecture: lifecycle before audit

Use three unequal panels, with the lifecycle panel occupying about half of the
width.

### A. Ownership policy

Render the 2x2 KV/GDN policy factorial as four compact state thumbnails rather
than four equally large cards.  Show the two axes once and use solid/dashed
line style redundantly with color.

### B. Request lifecycle

Render the exact two-lane transition described above.  Include `document
region`, `private tail`, `persistent GDN base`, and `private mutable state`.
The orange first-write boundary is the main visual anchor.

### C. Audit replay

Use a thin vertical audit spine with four small, distinct glyphs labelled
`ownership`, `call contract`, `FP32 oracle`, and `live mutants`.  Dotted guide
lines may connect the replay class to the relevant lifecycle state, but the
spine must not look like an additional execution stage.

## Visual grammar

- white background with generous margins;
- normal-width Arial/Helvetica-like typography;
- sentence case, no all-caps display text;
- blue = KV/document, green = GDN, orange = transition, muted red = violation
  or claim boundary, neutral gray = context;
- thin consistent strokes, at most two arrow families;
- use object scale, whitespace, and grouping for hierarchy rather than large
  title blocks or rounded dashboard cards;
- no gradient UI cards, stock icons, glossy effects, drop shadows, badge rows,
  or decorative arrows;
- all distinctions remain legible in grayscale through outline, dash, hatch,
  or shape.

## Acceptance checks

1. Every visible label matches the manuscript terminology character-for-
   character.
2. Every arrow and ownership relation matches the implementation and method
   provenance map.
3. Every number maps to a registered evidence identifier.
4. The teaser remains legible at full two-column width; the architecture
   remains legible at its actual PDF size.
5. Grayscale preserves policy and state distinctions.
6. No generated or decorative element implies an unexecuted runtime,
   additional model, throughput result, or completeness claim.
7. The replacement is not merged until it is visually inspected against the
   current PDF and is clearly better than the existing assets.
