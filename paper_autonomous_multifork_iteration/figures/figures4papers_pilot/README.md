# figures4papers compatibility pilot

This directory tests whether the public visual conventions in
[`ChenLiu-1996/figures4papers`](https://github.com/ChenLiu-1996/figures4papers)
fit the ForkAudit manuscript.  The repository was inspected at commit
`6790a93af3552539d955d77181c818916e1700b7`.

The upstream repository is used as a style and example reference only.  No
license file was present at the inspected commit, so this pilot does not copy
upstream code.  `generate_memory_phase_pilot.py` is an original deterministic
Matplotlib renderer that follows the documented typography, palette, spine,
legend, hatch, and PDF/PNG export conventions.

The plotted values are copied exactly from
`tables/rr2_memory_table.tex`; this candidate is not merged into `main.tex`.

Run:

```bash
python3 figures/figures4papers_pilot/generate_memory_phase_pilot.py
```

Expected outputs:

- `memory_phase_pilot.pdf` — editable vector PDF;
- `memory_phase_pilot.png` — 300-dpi review preview.
