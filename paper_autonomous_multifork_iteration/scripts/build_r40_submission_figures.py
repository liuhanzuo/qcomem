#!/usr/bin/env python3
"""Add deterministic, verified labels to the text-free ImageGen figure bases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "figures" / "imagegen_round5_candidates"
FONT_REGULAR = Path("/usr/local/texlive/2026/texmf-dist/fonts/opentype/public/lm/lmsans10-regular.otf")
FONT_BOLD = Path("/usr/local/texlive/2026/texmf-dist/fonts/opentype/public/lm/lmsans10-bold.otf")
BASES = {
    "teaser": {
        "path": CANDIDATES / "teaser_textfree_v2.png",
        "sha256": "cfbfecfc53a34d1bd19e2a247fe0700174dd42e66fcfce9b71fe060e0463dd57",
        "crop": (0, 80, 1774, 830),
        "output": ROOT / "figures" / "rr2_teaser_r40_dispatch.png",
    },
    "architecture": {
        "path": CANDIDATES / "architecture_textfree_v3.png",
        "sha256": "10b58c061546da1f27ec932acad6c1cb3dabee510c4749d04526262f67e3ade1",
        # Retain the full ownership circle at the top of the audit rail.  The
        # previous 50 px top crop cut the circle and made the glyph mapping
        # disagree visually with the teaser.
        "crop": (0, 0, 1665, 935),
        "output": ROOT / "figures" / "rr2_architecture_r39_rebuilt.png",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    size: int,
    bold: bool = False,
    fill: str = "#20262D",
    anchor: str = "mm",
    box: bool = True,
) -> None:
    chosen = font(size, bold=bold)
    bounds = draw.textbbox(xy, text, font=chosen, anchor=anchor)
    if box:
        padding_x, padding_y = 10, 5
        draw.rounded_rectangle(
            (
                bounds[0] - padding_x,
                bounds[1] - padding_y,
                bounds[2] + padding_x,
                bounds[3] + padding_y,
            ),
            radius=7,
            fill=(255, 255, 255, 238),
        )
    draw.text(xy, text, font=chosen, fill=fill, anchor=anchor)


def dashed_segment(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: str,
    width: int = 3,
    dash: int = 10,
    gap: int = 7,
) -> None:
    """Draw an axis-aligned dashed segment with deterministic geometry."""
    x1, y1 = start
    x2, y2 = end
    length = abs((x2 - x1) or (y2 - y1))
    for offset in range(0, length, dash + gap):
        stop = min(offset + dash, length)
        if x1 == x2:
            direction = 1 if y2 >= y1 else -1
            draw.line((x1, y1 + direction * offset, x2, y1 + direction * stop), fill=fill, width=width)
        else:
            direction = 1 if x2 >= x1 else -1
            draw.line((x1 + direction * offset, y1, x1 + direction * stop, y2), fill=fill, width=width)


def policy_strip(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str,
    dashed: bool,
) -> None:
    x1, y1, x2, y2 = bounds
    draw.rounded_rectangle(bounds, radius=6, fill=fill)
    if not dashed:
        draw.rounded_rectangle(bounds, radius=6, outline=outline, width=3)
        return
    dashed_segment(draw, (x1, y1), (x2, y1), fill=outline)
    dashed_segment(draw, (x2, y1), (x2, y2), fill=outline)
    dashed_segment(draw, (x2, y2), (x1, y2), fill=outline)
    dashed_segment(draw, (x1, y2), (x1, y1), fill=outline)


def arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    fill: str = "#20262D",
    width: int = 4,
    head: int = 12,
) -> None:
    """Draw a deterministic orthogonal arrow ending at ``points[-1]``."""
    if len(points) < 2:
        raise ValueError("an arrow needs at least two points")
    draw.line(points, fill=fill, width=width, joint="curve")
    x0, y0 = points[-2]
    x1, y1 = points[-1]
    if x0 == x1 and y1 > y0:
        polygon = [(x1, y1), (x1 - head, y1 - head), (x1 + head, y1 - head)]
    elif x0 == x1 and y1 < y0:
        polygon = [(x1, y1), (x1 - head, y1 + head), (x1 + head, y1 + head)]
    elif y0 == y1 and x1 > x0:
        polygon = [(x1, y1), (x1 - head, y1 - head), (x1 - head, y1 + head)]
    elif y0 == y1 and x1 < x0:
        polygon = [(x1, y1), (x1 + head, y1 - head), (x1 + head, y1 + head)]
    else:
        raise ValueError("only orthogonal final arrow segments are supported")
    draw.polygon(polygon, fill=fill)


def ownership_cell(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    *,
    kv_shared: bool,
    gdn_borrowed: bool,
) -> None:
    """Draw one paired KV/GDN ownership cell with redundant line coding."""
    x1, y1, x2, y2 = bounds
    height = y2 - y1
    kv_top = y1 + round(height * 0.14)
    kv_bottom = y1 + round(height * 0.43)
    gdn_top = y1 + round(height * 0.57)
    gdn_bottom = y1 + round(height * 0.86)
    draw.rounded_rectangle(bounds, radius=9, fill="#FAFBFC", outline="#6B7177", width=2)
    policy_strip(
        draw,
        (x1 + 11, kv_top, x2 - 11, kv_bottom),
        fill="#DCEBFA",
        outline="#0F4D92",
        dashed=kv_shared,
    )
    policy_strip(
        draw,
        (x1 + 11, gdn_top, x2 - 11, gdn_bottom),
        fill="#E1F1DA",
        outline="#2F6F2B",
        dashed=gdn_borrowed,
    )


def build_teaser(base: Image.Image) -> Image.Image:
    image = base.crop(BASES["teaser"]["crop"]).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    label(draw, (285, 27), "1  Forked hybrid state", size=40, bold=True)
    label(draw, (885, 27), "2  Phase-aware ownership", size=40, bold=True)
    label(draw, (1480, 27), "3  Bounded validation", size=40, bold=True)
    label(draw, (125, 91), "shared document KV", size=27, fill="#0F4D92")
    label(draw, (125, 357), "persistent GDN base", size=27, fill="#2F6F2B")
    label(draw, (274, 316), "mutable alias?", size=22, fill="#B64342")
    label(draw, (808, 75), "first write", size=26, fill="#B95717")
    # Replace the ambiguous generated mini-grid with four explicit paired
    # cells.  Every cell contains both a KV strip (blue) and a GDN strip
    # (green); columns encode copy/share and rows encode own/borrow.
    draw.rounded_rectangle((630, 474, 1138, 742), radius=14, fill="white", outline="#5C6268", width=3)
    label(draw, (884, 494), "Primary ForkAudit 2 x 2 factorial", size=23, bold=True, box=False)
    label(draw, (812, 523), "KV copy", size=19, fill="#0F4D92", box=False)
    label(draw, (972, 523), "KV share", size=19, fill="#0F4D92", box=False)
    label(draw, (704, 570), "GDN own", size=18, fill="#2F6F2B", anchor="rm", box=False)
    label(draw, (704, 636), "GDN borrow", size=18, fill="#2F6F2B", anchor="rm", box=False)
    for row, (y1, borrowed) in enumerate(((538, False), (604, True))):
        for col, (x1, shared) in enumerate(((727, False), (887, True))):
            ownership_cell(
                draw,
                (x1, y1, x1 + 142, y1 + 58),
                kv_shared=shared,
                gdn_borrowed=borrowed,
            )
    label(draw, (884, 682), "Relational verdict", size=19, bold=True, box=False)
    label(draw, (884, 715), "tokens / logits / KV / GDN match", size=19, fill="#2F6F2B", box=False)
    # Align each replay label with its vertical glyph instead of presenting a
    # misleading horizontal list above vertically arranged markers.
    label(draw, (1322, 96), "ownership", size=22)
    label(draw, (1322, 205), "call contract", size=22)
    label(draw, (1322, 312), "FP32 oracle", size=22)
    label(draw, (1322, 414), "slot census", size=22)
    label(draw, (1480, 73), "declared cohorts", size=19, fill="#6B7177")
    # Make the meaning of the right-hand verdicts explicit.  These cards
    # distinguish registered primary checks from the separate bounded census;
    # they do not absorb deployment/related-work runs.
    draw.rounded_rectangle((1582, 146, 1766, 542), radius=12, fill="white", outline="#6B7177", width=2)
    label(draw, (1674, 169), "REGISTERED RESULTS", size=14, bold=True, box=False)
    verdicts = [
        (192, 282, "4 cells", "registered match", "#E8F1FB", "#0F4D92"),
        (306, 396, "209,920 calls", "attention bound", "#E7F5F7", "#16788A"),
        (420, 510, "180 slots", "preproducer census", "#EDF6E9", "#2F6F2B"),
    ]
    for y1, y2, count, outcome, face, edge in verdicts:
        draw.rounded_rectangle((1594, y1, 1754, y2), radius=9, fill=face, outline=edge, width=2)
        label(draw, (1674, y1 + 27), count, size=19, bold=True, fill=edge, box=False)
        if outcome == "preproducer census":
            label(draw, (1674, y1 + 58), "preproducer", size=15, fill="#20262D", box=False)
            label(draw, (1674, y1 + 77), "census", size=15, fill="#20262D", box=False)
        else:
            label(draw, (1674, y1 + 63), outcome, size=15, fill="#20262D", box=False)
    arrow(draw, [(1572, 344), (1588, 344)], fill="#20262D", width=3, head=7)
    label(draw, (1515, 604), "primary: 96 configs; 7 / 7 targets", size=19)
    label(draw, (1515, 632), "dispatch: 209,920 attention calls bound", size=18)
    label(draw, (1515, 663), "GDN: 635,520 eager-route calls bound", size=18)
    label(draw, (1515, 689), "historical alias: base corrupt 8 / 8; outputs exact", size=16)
    label(draw, (1515, 721), "N=32 allocator: 4.90 → 2.23 GiB (−54.5%)", size=17)
    return image.convert("RGB")


def build_architecture(base: Image.Image) -> Image.Image:
    image = base.crop(BASES["architecture"]["crop"]).convert("RGBA")
    source = image.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    # Replace the imperfect generated policy thumbnails with a deterministic
    # paired 2x2 glyph: every cell contains both a KV and a GDN strip, with
    # column-wise KV and row-wise GDN line encodings.
    draw.rounded_rectangle((32, 105, 365, 548), radius=18, fill="white")
    draw.rounded_rectangle((42, 150, 352, 535), radius=18, fill="white", outline="#363B40", width=4)
    label(draw, (210, 126), "KV columns / GDN rows", size=19, fill="#5C6268", box=False)
    label(draw, (145, 180), "KV copy", size=24, fill="#0F4D92", box=False)
    label(draw, (270, 180), "KV share", size=24, fill="#0F4D92", box=False)
    label(draw, (63, 285), "own", size=18, fill="#2F6F2B", box=False)
    label(draw, (63, 439), "borrow", size=16, fill="#2F6F2B", box=False)
    for row, (y1, gdn_dashed) in enumerate(((207, False), (361, True))):
        for col, (x1, kv_dashed) in enumerate(((85, False), (210, True))):
            ownership_cell(
                draw,
                (x1, y1, x1 + 112, y1 + 126),
                kv_shared=kv_dashed,
                gdn_borrowed=gdn_dashed,
            )
    label(draw, (210, 31), "Whole-group policy", size=36, bold=True)
    label(draw, (885, 31), "Request lifecycle", size=36, bold=True)
    label(draw, (1400, 31), "Audit rail", size=34, bold=True)
    # Lifecycle checkpoints are explicit and correspond to the recorded
    # setup--post-first-transition--final receipt sequence.
    label(draw, (525, 78), "Setup", size=23, bold=True)
    label(draw, (855, 78), "Post-first-transition", size=22, bold=True)
    label(draw, (1215, 78), "Final", size=23, bold=True)
    label(draw, (690, 130), "first write", size=24, fill="#B95717")
    label(draw, (510, 169), "document KV", size=26, fill="#0F4D92")
    label(draw, (510, 367), "GDN state", size=26, fill="#2F6F2B")
    label(draw, (949, 166), "private append KV", size=23, fill="#0F4D92")
    label(draw, (954, 389), "request-private state", size=23, fill="#2F6F2B")
    label(
        draw,
        (210, 574),
        "4 primary ownership cells",
        size=23,
        bold=True,
    )
    # Same glyph semantics as Figure 1: circle=ownership, diamond=call,
    # square=FP32 oracle, triangle=live fault.  With the uncropped top, all
    # four glyphs and their aligned labels remain visible.
    label(draw, (1452, 76), "ownership", size=24, anchor="rm")
    label(draw, (1452, 202), "call contract", size=24, anchor="rm")
    label(draw, (1452, 346), "FP32 oracle", size=24, anchor="rm")
    label(draw, (1452, 482), "live faults", size=24, anchor="rm")

    # Rebuild the entire bottom bundle from the original card crops, shifted
    # 27 px left so the seven-card group and black bracket are centered on the
    # full 1665 px canvas.  This also lets the audit connector target the
    # bundle header rather than only the final fault card.
    draw.rectangle((175, 640, 1555, 934), fill="white")
    card_boxes = [
        (282, 752, 404, 875),
        (454, 752, 585, 875),
        (628, 752, 751, 875),
        (792, 752, 915, 875),
        (964, 752, 1102, 875),
        (1138, 752, 1271, 875),
        (1314, 752, 1446, 875),
    ]
    shift_x = -27
    card_y_shift = 15
    for box in card_boxes:
        tile = source.crop(box)
        image.alpha_composite(tile, (box[0] + shift_x, box[1] + card_y_shift))

    bundle_left, bundle_right = 190, 1475
    bundle_center = (bundle_left + bundle_right) // 2
    draw.line((bundle_left, 720, bundle_left, 925), fill="#20262D", width=5)
    draw.arc((bundle_left, 706, bundle_left + 28, 734), 90, 180, fill="#20262D", width=5)
    draw.line((bundle_left + 14, 720, bundle_left + 34, 720), fill="#20262D", width=5)
    draw.line((bundle_right, 720, bundle_right, 925), fill="#20262D", width=5)
    draw.arc((bundle_right - 28, 706, bundle_right, 734), 0, 90, fill="#20262D", width=5)
    draw.line((bundle_right - 34, 720, bundle_right - 14, 720), fill="#20262D", width=5)
    draw.line((bundle_left, 925, bundle_right, 925), fill="#20262D", width=5)

    draw.rounded_rectangle((655, 650, 1010, 696), radius=9, fill="#F2F4F6", outline="#5C6268", width=2)
    label(draw, (bundle_center, 673), "Recorded evidence bundle", size=25, bold=True, box=False)
    # Route from the whole audit rail to the bundle header, not to one card.
    draw.rectangle((1522, 515, 1540, 700), fill="white")
    draw.rectangle((1360, 625, 1540, 705), fill="white")
    arrow(draw, [(1531, 510), (1531, 630), (bundle_center, 630), (bundle_center, 648)], width=4, head=10)

    # Group headers make the seven-card semantics readable at manuscript size;
    # each card still receives an individual short label.
    label(draw, (572, 700), "STATE SNAPSHOTS", size=22, bold=True, fill="#44505A", box=False)
    label(draw, (1092, 700), "RECEIPTS", size=22, bold=True, fill="#44505A", box=False)
    label(draw, (1352, 700), "FAULT MAP", size=22, bold=True, fill="#44505A", box=False)
    labels = [
        ((315, 746), "document KV"),
        ((491, 746), "private append\nKV"),
        ((661, 746), "GDN base"),
        ((825, 746), "private GDN"),
        ((1005, 746), "storage"),
        ((1177, 746), "call / semantics"),
        ((1353, 746), "fault / gate"),
    ]
    for xy, text in labels:
        label(draw, xy, text, size=21, box=True)
    return image.convert("RGB")


def main() -> None:
    for config in BASES.values():
        if sha256_file(config["path"]) != config["sha256"]:
            raise ValueError(f"ImageGen base drift: {config['path']}")
    teaser = build_teaser(Image.open(BASES["teaser"]["path"]))
    teaser.save(BASES["teaser"]["output"], format="PNG", optimize=True, dpi=(300, 300))
    prompt_path = CANDIDATES / "PROMPTS_v1.md"
    record = {
        "deterministic_overlay": {
            "font_bold": FONT_BOLD.name,
            "font_regular": FONT_REGULAR.name,
            "script": Path(__file__).relative_to(ROOT).as_posix(),
            "script_raw_sha256": sha256_file(Path(__file__)),
            "technical_labels_are_not_image_generated": True,
            "visual_mapping_revision": "r40-seven-target-dispatch-v1",
        },
        "prompt_record_raw_sha256": sha256_file(prompt_path),
        "schema_version": "figures4papers-imagegen-deterministic-overlay-v3",
        "teaser": {
            "base_raw_sha256": BASES["teaser"]["sha256"],
            "crop_pixels": list(BASES["teaser"]["crop"]),
            "final_raw_sha256": sha256_file(BASES["teaser"]["output"]),
            "output": BASES["teaser"]["output"].relative_to(ROOT).as_posix(),
        },
    }
    (CANDIDATES / "IMAGEGEN_ASSET_RECORD_R40.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
