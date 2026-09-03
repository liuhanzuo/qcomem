#!/usr/bin/env python3
"""Generate / refine Figure 1 concept figure (correlated-MV early-stopping project).

Paper: "Finite-Sample Safe Adaptive Early Stopping for Correlated Majority-Vote Rollouts"

Three-panel concept figure, all numbers REAL from on-disk artifacts
(fit_cal_test_r469_result.json, drift/margin jsons, passrate_r467_result.json):
  (a) The mixture is the certificate object (bimodal pass rates, count-exchangeable replay)
  (b) BAYES-H per-prefix posterior conditional certificate; adaptivity is free (tower)
  (c) Certified dominance: rollout saving vs strongest baselines + drift margin repair

Usage (keys come from the environment via: source <AutoFigure-Edit>/set_api_keys.sh):
  AUTOFIGURE_EDIT_DIR=<AutoFigure-Edit> python3 edit_a11_earlystop_fig1.py gen  <dst_png> pro
  AUTOFIGURE_EDIT_DIR=<AutoFigure-Edit> python3 edit_a11_earlystop_fig1.py edit <src_png> <dst_png> pro [fixes_file]
"""
import os
import sys

from PIL import Image

sys.path.insert(0, os.environ["AUTOFIGURE_EDIT_DIR"])
from autofigure2 import (
    call_llm_image_generation,
    check_and_fix_svg,
    generate_svg_template,
    svg_to_png,
)

IMAGE_MODELS = {
    "flash": "google/gemini-3.1-flash-image-preview",
    "pro": "google/gemini-3-pro-image-preview",
}
SVG_MODEL = "google/gemini-3.1-pro-preview"
BASE_URL = "https://openrouter.ai/api/v1"

GEN_PROMPT = r"""Render ONE publication-grade ACADEMIC CONCEPT FIGURE to serve as Figure 1 of a
theoretical machine-learning paper titled "Finite-Sample Safe Adaptive Early Stopping for
Correlated Majority-Vote Rollouts".
It MUST look like a serious, flat, modern schematic from a top theory/ML venue
(NeurIPS / ICML / ICLR): NOT a cartoon, NOT an infographic, NO mascots, no people,
no robots, no emojis, no photorealism, no 3D, no heavy gradients. This is a DOUBLE-BLIND
submission: absolutely NO author names, institution names, or logos anywhere.

================ ABSOLUTE HARD CONSTRAINTS (override everything else) ================
- ONE wide landscape banner, aspect about 2.5:1, WHITE background, generous white space,
  NOTHING overlapping anywhere, no clipped words at the edges, no arrow crossing another arrow.
- THREE panels left-to-right, separated by thin light-gray vertical divider lines, with small
  bold panel tags "(a)", "(b)", "(c)" at their top-left corners:
    (a) "The mixture is the certificate object"      (~30% width)
    (b) "Per-prefix posterior certificate"           (~38% width, the theory core)
    (c) "Certified dominance + drift margin"         (~32% width)
- Use a consistent flat macaron palette: soft blue #83B8E5 for data/rollout elements,
  muted green #8FD6BD for certified/valid elements, muted red #E79B9B for invalid/exceeds,
  warm orange #F4C7A0 ONLY for the selected stopper box, dark slate text #2F3B4C.
  Rounded rectangles, 2px outlines, ALL text horizontal, sans-serif, short labels only.

================ PANEL (a) — THE MIXTURE IS THE CERTIFICATE OBJECT ================
- Top: a minimal flat histogram of per-task pass rate p, strongly U-shaped (bimodal):
  tall bars near p=0 and p=1, shallow in the middle. X-axis labeled "per-task pass rate p".
  A small annotation: "45% of tasks at p<=0.1 or p>=0.9 (bimodal)".
- Middle: an arrow down to a small box labeled "pass count K ~ H (mixture over tasks)".
- Below: a flat row of N=32 small dots (soft blue) = "N rollouts of one task"; a bracket
  labeled "majority vote"; the first k dots highlighted warm orange with a stop bar,
  labeled "stop after k". Caption: "replay: uniform random prefix, count-exchangeable".
- Small caption inside panel bottom: "pooled correlation is the wrong statistic; the flip
  probability is a functional of the task-difficulty mixture H".

================ PANEL (b) — PER-PREFIX POSTERIOR CERTIFICATE (theory core) ================
- A clean statement box (light gray fill, thin border) containing exactly three lines:
    Stop when  c_H(x,k) = P_H( MV(N) != side(x,k) | x,k ) <= alpha
    exact hypergeometric posterior (no approximation)
    Theorem: P(flip) = E[ c_H(x_tau,tau) ] <= alpha  -- no union bound
- Below the box, a small left-to-right path diagram: a horizontal chain of 5 prefix states
  (small circles) with the running majority side marked, and a green "STOP" chip where the
  posterior certificate first drops to alpha. Annotation above: "stop early exactly on
  bimodal mass; wait only on mid-difficulty tasks".
- A small tag chip: "adaptivity is free (tower property); calibration = one
  empirical-Bernstein step over tasks + Bonferroni over the rule family".

================ PANEL (c) — CERTIFIED DOMINANCE + DRIFT MARGIN ================
- A minimal flat horizontal bar chart (NO gridlines) of rollout SAVING at alpha = 0.05,
  four bars, larger bar = better:
    Bar "FULL-32" (muted red): labeled "0%".
    Bar "FIXED-HOEF" (muted red): labeled "15.6%".
    Bar "FIXED-EB" (soft blue): labeled "46.9%".
    Bar "BAYES-H" (muted green): clearly longest, labeled "80.9%, flip 0.025 <= 0.05".
  A dashed vertical reference line is FORBIDDEN; the green bar must be obviously longest.
  Annotation to the right of the green bar: "valid; dominates the strongest
  distribution-free baseline (paired gap +0.652, significant)".
- Small side note box at panel bottom: "ordered drift: a certificate margin gamma=0.025
  restores validity up to drift delta<=0.15 (saving 77.2% vs 81.0%); replicates on a
  disjoint second shard and under cross-shard prior transfer".
- All numbers above are REAL verified artifacts; do NOT invent additional numbers.

================ STYLE / QUALITY BAR ================
Flat vector look, crisp 2px outlines, consistent corner radii, aligned boxes, generous
padding, high contrast text, every arrow with a clear head, all labels spelled EXACTLY
as written above (check spelling character by character). NO other text beyond the
labels specified. Title text inside the figure is FORBIDDEN. Colorblind-safe: the red
and green bars must also differ in length/position so the distinction survives grayscale.
A single unified caption strip at the very bottom, small italic font:
  "Treat across-task heterogeneity as the certificate object (a): stop when the exact
   posterior flip probability drops to alpha (b) -- adaptivity is free and the result
   saves 81% of rollouts where the strongest distribution-free baseline saves 16% (c)."
No fabricated data beyond the numbers given; no watermarks; no 3D; no shadows.
"""

EDIT_PROMPT_HEADER = r"""You are refining an academic concept figure (Figure 1 of a theory paper on
finite-sample safe adaptive early stopping for correlated majority-vote rollouts:
treating across-task heterogeneity as the certificate object, stopping when the exact
posterior flip probability drops to alpha, with adaptivity free by the tower property).
Keep the overall composition, palette, and three-panel structure EXACTLY as in the input
image. Apply ONLY the fixes listed below. Flat modern vector style, white background, no
mascots/emojis/logos, double-blind safe. Do not add new panels or new text beyond what
the fixes request. Preserve aspect ratio.

FIXES TO APPLY:
"""


def do_gen(dst_png: str, model_key: str = "pro") -> None:
    model = IMAGE_MODELS[model_key]
    print(f"[gen] model={model}")
    img = call_llm_image_generation(
        prompt=GEN_PROMPT, model=model, base_url=BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"], provider="openrouter",
    )
    if img is None:
        print("FAILED: no image returned"); sys.exit(1)
    img.convert("RGB").save(dst_png)
    print(f"[gen] saved {dst_png} size={img.size}")


def do_edit(src_png: str, dst_png: str, model_key: str = "pro",
            fixes_file: str | None = None) -> None:
    model = IMAGE_MODELS[model_key]
    fixes = open(fixes_file).read() if fixes_file else "- overall polish only"
    prompt = EDIT_PROMPT_HEADER + fixes
    print(f"[edit] model={model} fixes={fixes_file}")
    img = call_llm_image_generation(
        prompt=prompt, model=model, base_url=BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"], provider="openrouter",
        reference_image=Image.open(src_png),
    )
    if img is None:
        print("FAILED: no image returned"); sys.exit(1)
    img.convert("RGB").save(dst_png)
    print(f"[edit] saved {dst_png} size={img.size}")


def do_svg(src_png: str, dst_svg: str) -> None:
    svg = generate_svg_template(
        image_path=src_png, model=SVG_MODEL, base_url=BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    svg = check_and_fix_svg(svg)
    with open(dst_svg, "w") as f:
        f.write(svg)
    print(f"[svg] saved {dst_svg}")


def do_render(src_svg: str, dst_png: str, scale: float = 2.0) -> None:
    svg_to_png(src_svg, dst_png, scale=scale)
    print(f"[render] saved {dst_png}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen":
        do_gen(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "pro")
    elif cmd == "edit":
        do_edit(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "pro",
                sys.argv[5] if len(sys.argv) > 5 else None)
    elif cmd == "svg":
        do_svg(sys.argv[2], sys.argv[3])
    elif cmd == "render":
        do_render(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 2.0)
    else:
        print(__doc__)
