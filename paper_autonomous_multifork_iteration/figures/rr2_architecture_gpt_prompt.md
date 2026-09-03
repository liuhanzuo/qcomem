# RR2 architecture ImageGen record

- Date: 2026-08-19
- Generator: GPT Image 2 through the built-in Codex ImageGen reference-image edit path (backend identity confirmed by the user)
- Historical selected asset: `rr2_architecture_imagegen_v4.png`
- Selected asset SHA-256: `798006944f2971c33cb735dd65d56dcfd868b46b832d5479bce63ff05895c7e7`
- Layout reference: `rr2_architecture_layout_reference.png`
- Layout-reference SHA-256: `389a610153ef0cb7a33899c4327cb9f9ba83ce15dc2ce7ae29fd087e2d11e64a`
- Status: superseded in `main.tex` by the deterministic figures4papers-style
  vector asset recorded in `rr2_figures4papers_record.md`; retained only as
  provenance for the earlier rendering iteration
- Execution note: CLI `gpt-image-2` was not used because `OPENAI_API_KEY` was unavailable; the built-in reference-edit path was used instead

## Superseded exploratory prompt

> Create a camera-ready academic architecture diagram that looks as if it was carefully drawn in LaTeX TikZ for an ICLR or systems paper. Wide landscape, 2.25:1. Pure opaque white background. Sparse black and gray line art with only two very restrained accents: muted navy for shared state and muted teal for private state. No illustration style.
>
> The diagram has one left-to-right reading path and exactly four regions.
>
> LEFT REGION: two orthogonal setup-policy axes shown only with text, thin braces, and straight lines. Top: label "KV setup" with alternatives "full copy" and "shared paged". Bottom: label "GDN setup" with alternatives "materialized" and "borrowed immutable". A small multiplication sign between the two axes indicates the 2 x 2 factorial. These alternatives control the whole request group.
>
> MIDDLE REGIONS: three phase columns separated by thin vertical gray rules and titled "Setup", "First transition", and "Generation". There are exactly two horizontal lanes, titled "KV cache" and "GDN state". At the left of the KV lane is a plain rectangle labeled "Document KV". At the left of the GDN lane is a separate plain rectangle labeled "Persistent GDN base". Do not put icons inside either rectangle.
>
> Within each lane, represent the whole request group with three aligned rows labeled "r0", "r1", and "rN", with one centered ellipsis. In Setup, indicate the two allowed group-wide relations using small brace annotations rather than mixing policies across requests: KV is either request-local full copy or shared read-only paged document; GDN is either request-local materialized state or a shared read-only borrowed base. At First transition and Generation, KV preserves a shared document rectangle followed by a small outlined rectangle labeled "private tail". GDN becomes a separate outlined rectangle labeled "private mutable" for each completed request. Use a single thin orange tick at the first-transition boundary. All arrows are straight, thin, and left to right.
>
> BOTTOM REGION: one plain thin horizontal rule labeled "ForkAudit replay", with four unobtrusive labels below it: "ownership", "call contract", "FP32 oracle", "live mutants". Use thin dashed vertical connectors to the relevant phase columns. At far right, a thin brace spanning the request group is labeled "same execution schedule".
>
> Use only these exact visible labels: "KV setup", "full copy", "shared paged", "GDN setup", "materialized", "borrowed immutable", "Document KV", "Persistent GDN base", "Setup", "First transition", "Generation", "KV cache", "GDN state", "r0", "r1", "rN", "shared read-only", "request-local", "private tail", "private mutable", "ForkAudit replay", "ownership", "call contract", "FP32 oracle", "live mutants", "same execution schedule". No title, subtitle, figure number, caption, equations, result values, or additional text.
>
> Visual constraints: pure flat vector linework; unfilled or extremely pale flat rectangles; 1 pt consistent strokes; square corners; conventional compact sans-serif type; strict grid alignment; large whitespace; no background texture. No icons, document symbols, database cylinders, locks, shields, bugs, magnifying glasses, checkmarks, badges, cards, panels with shadows, gradients, glow, bevels, 3D, rounded app UI, presentation-slide styling, decorative arrows, saturated colors, or oversized text. The result must resemble a restrained black-and-white TikZ method figure, not an AI-generated infographic.

## Superseded reference-edit prompt

> Use case: scientific-educational. Asset type: final full-width architecture figure for an ICLR two-column paper. Image 1 is the composition and relationship reference. Recreate its scientific content as a polished high-resolution raster figure; do not copy its clipping, overlaps, cramped text, line numbers, or caption fragment.
>
> Preserve the reference's four-part structure exactly: two orthogonal whole-group setup-policy axes at left; distinct Document KV and Persistent GDN base sources; three lifecycle columns Setup, First transition, Generation with KV above GDN; and a bottom ForkAudit replay rail. Keep the repeated request rows r0, r1, ellipsis, rN, but increase spacing enough that every label is readable. The two policy axes control the entire request group, never individual requests.
>
> KV setup alternatives are full copy and shared paged. The setup document region is request-local under full copy and shared read-only under shared paged. At later phases, label the large KV segment "document region" because its ownership remains policy-dependent; label only the appended small segment "private tail". GDN setup alternatives are materialized and borrowed immutable. A materialized setup state is request-local; a borrowed setup state is a read-only alias to the Persistent GDN base. At First transition and Generation, completed GDN request states are "private mutable". Document KV must never feed GDN. Persistent GDN base must never feed KV. A thin brace at right indicates "same execution schedule". The replay rail contains ownership, call contract, FP32 oracle, live mutants.
>
> Use only these exact labels: "KV setup", "full copy", "shared paged", "GDN setup", "materialized", "borrowed immutable", "KV cache", "GDN state", "Document KV", "Persistent GDN base", "Setup", "First transition", "Generation", "r0", "r1", "rN", "full copy: request-local", "shared paged: shared read-only", "materialized: request-local", "borrowed: read-only alias", "document region", "setup state", "private tail", "private mutable", "ForkAudit replay", "ownership", "call contract", "FP32 oracle", "live mutants", "same execution schedule". Render the ellipsis as a vertical three-dot mark. Do not invent a title, result number, pass/fail phrase, figure number, or caption.
>
> Use a sober camera-ready academic schematic resembling a carefully typeset TikZ figure but rendered as a high-resolution bitmap: pure opaque white background; thin crisp charcoal arrows and rules; muted navy with extremely pale blue for document/shared state; muted teal with extremely pale teal for private state; one muted-orange transition line; conventional compact sans-serif type; square corners; strict alignment; generous whitespace. Wide 2.4:1 composition with 5% outer margin. All labels must remain readable at 6.5 inches wide. Keep the replay rail and right brace fully inside the frame.
>
> Avoid marketing infographic, corporate slide, dashboard, cards, rounded UI, icons, document icon, database cylinder, locks, shields, bugs, magnifying glasses, checkmarks, badges, gradients, glow, shadows, pseudo-3D, photorealism, decorative background, oversized headings, dense callout bubbles, and watermark.

## Localized edit prompts

1. Add a correctly separated `GDN state` lane label and thin source-to-request branching connectors from `Document KV` and `Persistent GDN base`, without cross-lane connections or other changes.
2. Delete only the accidentally duplicated top-row `GDN state` label; retain the correct lower label and every other element.

## Selected three-region redesign

The repeated request-row composition was rejected as visually dense and too
close to a generated infographic.  The selected redesign uses three regions:
a compact $2\times2$ ownership matrix, one representative request lifecycle,
and a narrow audit-replay spine.  It follows the prompt hierarchy recommended
by academic-figure skills: global composition, section-by-section content,
global annotations, and an explicit style/negative block.  The palette is
limited to navy, teal, one orange transition line, and neutral gray.

## Typography-only reference edit

> Edit this existing academic architecture figure. Preserve the exact
> scientific structure, positions, arrows, labels, colors, boxes, and
> landscape composition. Change only the typography and minor spacing needed
> to fit it. Use one neutral Swiss sans-serif family throughout, visually
> matching Helvetica Neue, Arial, or Liberation Sans. Never use condensed,
> narrow, handwritten, rounded-display, comic, decorative, or humanist poster
> fonts. Set the region headings in sentence case: "Ownership factorial",
> "Request lifecycle", and "Audit replay". Use semibold for region and stage
> labels, regular weight for body labels, dark text, normal tracking, and only
> three levels of hierarchy. Keep the background white and retain all linework.
> Do not add, remove, paraphrase, or misspell scientific content. No all-caps
> display typography, shadows, outlines, slide-deck styling, or marketing
> infographic styling.

The built-in image editor does not expose or guarantee a font-file identity.
Accordingly, the selected bitmap is described only as using
Helvetica/Arial-like neutral sans-serif typography; it is not claimed to embed
Helvetica.

## GPT Image 2 fixed-slot refinement

The selected v3 was produced by rephrasing the typography edit into the
fixed-slot structure used by GPT Image 2 scientific-schematic skills:
`SCENE/USE`, `PRESERVE`, `CHANGE ONLY`, `EXACT TEXT`, `MUST KEEP`, and `AVOID`.
Every visible label was quoted, while the hierarchy was specified numerically:
body regular, stage labels roughly 8% larger, and region headings roughly 15%
larger.  The prompt prohibited condensed/DIN-like fonts, all-caps display text,
extra-bold titles, slide-deck styling, and any topology change.  The model
rendered the mathematical ellipsis as three closely spaced periods; this is
treated as a visual abbreviation rather than machine-readable mathematics.

## Round-4 publication-style rerender

The v4 candidate used v3 as an edit target and applied the paper-figure prompt
structure supplied by the user: intended artifact, preserved topology, exact
text whitelist, academic typography, layout constraints, scientific
invariants, and an explicit avoid list.  The primary prompt was:

> Re-render Image 1 as a restrained, publication-quality systems-paper
> architecture diagram. Preserve the exact scientific meaning, three-region
> layout, object count, connections, and reading order. Improve only
> typography, alignment, whitespace, line weight, and print polish. Use a
> neutral academic sans-serif similar to Helvetica/Arial, regular weight, with
> modest semibold only for the three region headers. Keep a flat white
> background, muted blue KV outlines, muted green GDN outlines, one thin orange
> first-transition line, hairline rules, and no gradients, shadows, icons,
> decorations, fake equations, performance claims, or extra text. Preserve the
> four whole-group setup cells, the policy-dependent document region, private
> append tail, setup-only read-only GDN alias, post-transition private mutable
> GDN state, and the audit spine as replay classes rather than execution stages.

The first render omitted the row-axis label.  A single-change reference edit
then added exactly `GDN setup` vertically at the far left while preserving all
other words, geometry, colors, and connections.  The final image is a raster
asset; the font is described by appearance only and no exact font embedding is
claimed.

## Verification boundary

The image is explanatory, not evidence. Its visible topology was checked against the frozen method and experiment plan before inclusion: policy choices apply to the entire request group; source lanes do not cross; KV document ownership remains policy-dependent; private append tails are request-local; borrowed GDN aliasing is setup-only; and completed GDN state is private after first transition. Quantitative outcomes remain in deterministic figures and tables.
