# Round 5 text-free image-generation candidates

Generated with the built-in `image_gen` tool on 2026-08-19. The v1 images are retained as rejected candidates. The text-free v2 bases were cropped and combined with deterministic Latin Modern Sans labels by `scripts/build_round5_imagegen_figures.py`; those checked composites are the teaser and architecture merged into the paper. The generated bases themselves never supply technical text or numerical evidence.

## Teaser

Output: `teaser_textfree_v1.png` (1774 × 887 px)

Exact prompt:

~~~text
Use case: scientific-educational
Asset type: text-free academic paper teaser base artwork, wide landscape raster for later deterministic label overlay
Primary request: Create a publication-quality three-act left-to-right technical teaser about isolating forked generation requests and validating that isolation.
Scene/backdrop: pure white background, generous margins and large negative space, no border around the full canvas.
Composition/framing: ultra-wide landscape, approximately 3:1. Three balanced visual acts separated only by subtle thin light-gray vertical dividers. Reserve a completely empty horizontal band across the top for later headings, plus small clean white label zones under major modules.
Act 1, left: show one shared immutable document-memory object as a solid pale-blue rectangular stack feeding three forked request cards through clean blue arrows. Beside it show one persistent GDN base as a pale-green rectangular stack feeding the same three requests through green paths. The document branches remain visibly shared; the GDN paths visually create ambiguity/tension at the request forks, using one restrained dashed orange-red caution bracket near the mutable branch point. Do not depict accidental sharing as the solution.
Act 2, center: show a compact phase-aware ownership contract across three successive stations. Use two aligned horizontal state lanes: upper blue document lane, lower green GDN lane. At setup, the upper object is shared and immutable and the lower object is a persistent base. At the first transition, place one thin vertical orange boundary and an orange switch/rebinding glyph; after it, each request has a clearly private green append/tail while the blue document core remains shared and immutable with a small request-private blue tail appended at the right edge. At the later station, preserve this split. Include a small four-cell whole-group policy matrix motif beneath or beside the stations, visually indicating that a single paired policy choice governs the entire request group, not individual cells.
Act 3, right: depict bounded validation as a restrained audit/evidence panel, not a celebratory result. Show four distinct evidence tokens entering a vertical audit rail: a blue circle for ownership, a blue diamond for call-contract evidence, a teal outlined square for FP32-oracle evidence, and a green triangle for live-mutant evidence. The rail terminates in a modest dark-gray bracketed evidence container. Beside it show a small tidy set of finite sample tiles and a thin enclosing frame to communicate bounded evidence. Do not imply universal proof, speed, or completeness.
Style/medium: flat vector-like academic paper illustration; figures4papers miscellaneous-figure sensibility; authored visual hierarchy; crisp geometric modules; clean black and dark-gray outlines; consistent 2–3 px visual stroke rhythm; simple arrowheads; no photographic content.
Color palette: muted academic blue (#0F4D92, #3775BA, pale blue fills), muted green (#4F7F34, #8BCF8B, pale green fills), neutral grays, exactly one restrained orange transition accent (#D97724). A small muted red-orange caution mark is allowed only in Act 1.
Text: none.
Constraints: technically faithful topology; shared immutable document KV plus request-private append/tail; persistent GDN base becomes private after the first transition; whole-group policy axes; separate audit rail receiving ownership, call-contract, FP32-oracle, and live-mutant evidence; bounded-evidence framing. Keep designated clean zones truly empty for later labels.
Avoid: all words, letters, digits, legends, axes, tick marks, equations, pseudo-text, fake typography, watermarks, logos, decorative circuitry, gears, robots, brains, clouds, locks, people, 3D, isometric perspective, glossy AI infographic styling, gradients, glow, drop shadows, clutter, excessive rounded cards, copied motifs from any specific reference.
~~~

Inspection notes:

- Strong three-act composition, white background, generous top label band, clean blue/green/orange semantic hierarchy, and no generated text or watermark.
- The center expands the state into three request rows, so the shared-document-plus-private-tail relation is distributed across request instances rather than expressed as one compact pair of lanes.
- The upper request-private tails and lower private states are legible through dashed boundaries, but deterministic labels will still be needed to distinguish append ownership from immutability.
- The evidence rail correctly receives four shaped tokens and terminates in a bounded sample panel; the bracketed four-dot intermediary is more abstract than the authority figure.

## Architecture schematic

Output: `architecture_textfree_v1.png` (1666 × 944 px)

Exact prompt:

~~~text
Use case: scientific-educational
Asset type: text-free academic architecture schematic base artwork, wide landscape raster for later deterministic label overlay
Primary request: Create a publication-quality architecture schematic for phase-aware ownership in forked generation requests, with one policy block, exactly two state lanes, a clear first-transition boundary, and a separate audit rail.
Scene/backdrop: pure white background with generous outer margins and large negative space. No full-canvas border.
Composition/framing: wide landscape, approximately 2.5:1 to 3:1. Reserve a completely empty top band for later section headings and clean white label zones around modules.
Left zone — whole-group policy: a compact 2-by-2 policy matrix of four small cells inside one larger subtle frame. Columns represent two document ownership choices visually: solid blue local core versus dashed blue shared immutable core. Rows represent two GDN ownership choices visually: solid green local/materialized base versus dashed green borrowed/read-only alias. Use only icons and line treatments, no text. A single dark-gray arrow exits the entire matrix toward the lifecycle, showing that one policy pair controls the whole request group.
Center zone — request lifecycle: exactly two long aligned horizontal lanes, upper blue document-memory lane and lower green GDN-state lane, spanning three phase stations. At the left station, show a pale-blue immutable shared document core in the upper lane and a pale-green persistent GDN base in the lower lane. In the middle, place one unmistakable thin vertical orange first-transition boundary cutting between stations, with a restrained orange switch/rebinding glyph centered near the boundary. To the right of the boundary, the upper lane must preserve the same blue shared immutable document core while adding a small dashed-outline blue request-private append/tail at its right edge. The lower lane must change from the persistent base/read-only setup state into a clearly request-private green mutable block, using a light dotted fill or repeated green dots. At the final station, preserve the same relations: shared immutable blue core plus private blue append/tail above, private mutable green state below. Use clear directional arrows from left to right. Show subtle grouping braces or parallel request sprouts only at the private append/tail ends, but keep exactly two principal state lanes.
Right zone — audit replay rail: place a visually detached vertical dark-gray audit rail with generous white gap from the lifecycle; connect via four thin dotted evidence lines. Arrange four distinct evidence markers on the rail from top to bottom: blue outlined circle for ownership, blue outlined diamond for call-contract, teal outlined square for FP32-oracle, green outlined triangle for live-mutant evidence. End the rail in a modest downward arrow entering a thin dark-gray bounded-evidence bracket/container. This is evidence collection, not a new execution phase. Do not connect the rail back into lifecycle arrows.
Bounded-evidence framing: near the bottom of the lifecycle and rail, include a subtle finite bracket or clipped frame enclosing a small, countable set of sample tiles; visually communicate limited audited coverage without axes, numbers, or text.
Style/medium: flat vector-like academic paper schematic; figures4papers miscellaneous-figure sensibility; authored hierarchy; crisp geometric modules; clean black and dark-gray outlines; consistent 2–3 px visual stroke rhythm; simple arrowheads; print-safe line encodings; no photographic content.
Color palette: muted academic blue (#0F4D92, #3775BA with pale blue fills), muted green (#4F7F34, #8BCF8B with pale green fills), teal (#42949E), neutral grays, exactly one restrained orange transition accent (#D97724).
Text: none.
Constraints: topology must be unambiguous: shared immutable document KV plus request-private append/tail; persistent GDN base becomes request-private after the first transition; whole-group policy axes represented by the 2-by-2 matrix; separate audit rail receiving ownership, call-contract, FP32-oracle, and live-mutant evidence; bounded-evidence framing. Preserve clean empty zones for deterministic labels.
Avoid: all words, letters, digits, legends, axes, tick marks, equations, pseudo-text, fake typography, watermarks, logos, decorative circuitry, gears, robots, brains, clouds, locks, people, 3D, isometric perspective, glossy AI infographic styling, gradients, glow, drop shadows, clutter, excessive rounded cards, copied motifs from any specific reference.
~~~

Inspection notes:

- Two principal lifecycle lanes, the orange first-transition boundary, shared blue core with dashed private tail, and dotted private green state are all visually clear.
- The detached audit rail has the correct ordered marker vocabulary and does not feed back into execution.
- The 2×2 policy matrix is visually clean, but its cells simplify the factorial choices: it reads as blue options in the top row and green options in the bottom row rather than explicitly combining both axes inside every cell. Deterministic row/column labels or a redrawn overlay would be needed for strict factorial semantics.
- The bounded-evidence strip contains seven finite tiles and reads clearly as a bounded set, though the three small timeline nodes above it could be mistaken for an additional axis if left unlabeled.
- The generated aspect ratio is about 1.76:1 rather than the requested 2.5–3:1, so it is best treated as a candidate for cropping or recreation rather than a drop-in final architecture figure.

## Teaser v2 targeted edit

Output: `teaser_textfree_v2.png` (1774 × 887 px)

Exact edit prompt:

~~~text
Use case: precise-object-edit
Asset type: targeted v2 edit of a text-free academic paper teaser base artwork
Input image: Image 1 is the edit target and must remain the compositional and stylistic source.
Primary request: Make only two targeted edits in the center lifecycle of Image 1.
Edit 1: replace the center orange circle containing crossed arrows with a neutral first-transition marker consisting only of one thin orange vertical line and one small solid orange dot centered on that line. No circle, no crossed arrows, no switch, no toggle, no extra orange marks.
Edit 2: simplify the center lifecycle from three blue request rows and three green request rows to exactly two representative blue rows in the upper lane and exactly two representative green rows in the lower lane. Preserve their left-to-right state progression, arrows, dashed private append/tails, blue shared-document semantics, green persistent-to-private semantics, phase spacing, and the small whole-group policy matrix below. Rebalance the center spacing cleanly so the two-row-per-lane schematic looks intentional and airy.
Invariants: preserve everything outside these two edits exactly as closely as possible, including the three-act left-to-right composition, left fork/tension panel, right four-marker audit rail, bounded evidence panel, subtle vertical act dividers, generous empty top label band, clean white background, flat vector-like figures4papers academic style, geometry, stroke rhythm, muted blue and green palette, restrained orange color, and large negative space.
Text: none.
Constraints: remain absolutely text-free and publication-clean.
Avoid: all words, letters, digits, pseudo-text, fake typography, legends, axes, tick marks, equations, logos, watermarks, gradients, glow, 3D, glossy infographic styling, decorative circuitry, clutter, new icons, or any unrelated redesign.
~~~

QA notes:

- The crossed-arrows symbol is gone; the first transition is now a single thin orange vertical line with one small orange dot.
- The center lifecycle now contains exactly two representative blue rows and two representative green rows, improving the audit-schematic reading and negative space.
- The left fork, bottom policy matrix, right audit rail, bounded evidence panel, palette, and overall three-act layout are preserved closely.
- The image remains fully text-free with no pseudo-text or watermark.
- Limitation: the green private state is now conveyed mainly by the dashed private tail rather than the dotted mutable fill used in v1, so deterministic labeling remains important.

## Architecture v2 targeted edit

Output: `architecture_textfree_v2.png` (1665 × 944 px)

Exact edit prompt:

~~~text
Use case: precise-object-edit
Asset type: targeted v2 edit of a text-free academic architecture schematic base artwork
Input image: Image 1 is the edit target and must remain the compositional and stylistic source.
Primary request: Make only two targeted edits to Image 1.
Edit 1: replace the center orange circle-and-toggle/switch glyph with a neutral first-transition marker consisting only of one thin orange vertical line and one small solid orange dot centered on that line. No circle, no switch arm, no toggle, no crossed arrows, and no extra orange marks.
Edit 2: simplify and clarify the left 2-by-2 panel as a compact whole-group policy matrix. Retain exactly four cells inside one shared outer frame, but make every cell visibly represent one paired policy for the entire request group by containing a tiny blue upper state strip and a tiny green lower state strip together. Use row-wise green line treatment and column-wise blue line treatment to reveal two orthogonal policy axes: solid versus dashed styling only, with no labels. Do not make the cells look like four independently assigned requests. Keep one single dark-gray arrow exiting the matrix as a whole and feeding the shared two-lane lifecycle, emphasizing one paired matrix selection controlling the whole group.
Invariants: preserve everything else exactly as closely as possible, including the two principal horizontal lifecycle lanes, three phase stations, blue shared immutable document cores, dashed blue request-private append/tails, green persistent base before transition, dotted private green mutable blocks after transition, left-to-right arrows, private tail sprouts, detached right audit rail with circle/diamond/square/triangle markers, dotted evidence inputs, bounded seven-tile evidence strip, generous empty top label band, white background, flat vector-like figures4papers academic style, clean outlines, muted blue/green/teal palette, restrained orange color, and large negative space.
Text: none.
Constraints: remain absolutely text-free and publication-clean; the matrix must read as two whole-group policy axes and the lifecycle must remain exactly two principal state lanes.
Avoid: all words, letters, digits, pseudo-text, fake typography, legends, axes, tick marks, equations, logos, watermarks, gradients, glow, 3D, glossy infographic styling, decorative circuitry, clutter, new icons, new execution stages, or any unrelated redesign.
~~~

QA notes:

- The switch/toggle glyph is gone; the first transition now uses the same thin orange line-and-dot vocabulary as teaser v2.
- The four-cell policy block remains compact inside one outer frame and has a single matrix-level arrow into the shared lifecycle, reducing the independent-request-cell reading.
- The two lifecycle lanes, shared blue cores with dashed private tails, dotted private green states, detached audit rail, and seven-tile bounded evidence strip are preserved.
- The image remains fully text-free with no pseudo-text or watermark.
- Limitation: the generated matrix only partially realizes the requested paired-state encoding—its top cells show blue-over-green pairs, while its bottom cells are green-over-green. Deterministic axis labels or a small code-drawn overlay would still be needed for strict factorial semantics.
- The v1 aspect-ratio limitation remains: this is approximately 1.76:1 rather than the originally requested 2.5–3:1.

## Architecture v3 layout correction

The v2 base was edited twice with GPT Image 2 after manuscript-scale visual
QA showed that the bottom evidence bracket was optically anchored to the
far-right audit rail instead of centered beneath the whole architecture. The
first edit preserved the upper diagram, centered and evenly spaced the seven
evidence cards, and replaced the long vertical arrow with a clean elbow
connector. The second edit made the final leftward alignment correction so the
outer bracket has visually balanced canvas margins. Both prompts prohibited
all text and pseudo-text; technical labels remain deterministic overlays from
`scripts/build_round5_imagegen_figures.py`.
