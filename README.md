# MacLLM-Bench

Reproducible local-LLM inference experiments for Apple Silicon. The first
milestone benchmarks MLX-LM on a single 3B 4-bit model before adding
llama.cpp, additional quantization formats, or long-context workloads.

## Host baseline

- MacBook Pro, Apple M4 Pro
- 24 GB unified memory
- arm64
- macOS 26.4.1

## Run

From this directory, run one command:

```bash
make run
```

It installs `uv` in the user account when needed, installs a managed Python
3.12, creates `.venv`, installs MLX-LM, downloads the configured model, records
the host snapshot, and runs the benchmark. Homebrew and `sudo` are not needed.
The initial run downloads model weights and therefore takes longer.

Individual commands remain available after setup:

```bash
make system-info
make smoke
source .venv/bin/activate
```

Run the same-process 512/2048/8192-token context benchmark:

```bash
make context
```

## First experiment

The smoke configuration uses
`mlx-community/Llama-3.2-3B-Instruct-4bit`, one warm-up run, and three
measured 64-token generations:

```bash
make smoke
```

Outputs are written under `results/`:

- `system_info.json`: hardware, OS, Python, memory, and power settings
- `runs.jsonl`: one record per run
- `summary.json`: median, mean, minimum, and maximum
- `raw/`: complete MLX-LM output for audit/debugging

The initial parser records MLX-LM's prompt throughput, generation throughput,
peak memory, and end-to-end wall time. TTFT will be added in the next milestone
using an instrumented Python generation loop; it is deliberately not inferred
from total wall time.

## Experimental rules

1. Pin model repository and revision before publishing results.
2. Record actual prompt/output token counts, not character counts.
3. Separate cold starts, warm-up runs, and measured runs.
4. Keep sampling parameters fixed.
5. Report median and dispersion, not only the fastest run.
6. Note power mode, background load, swap, and thermal state.
7. Do not treat unrelated 4-bit formats as equivalent quantization.

## Roadmap

- Instrument true TTFT and tokenizer time.
- Add 512/2048/8192-token prompt fixtures.
- Add a 7B/8B model and controlled memory-pressure runs.
- Add llama.cpp build, GGUF metadata capture, and `llama-bench` ingestion.
- Add process-level memory/swap sampling and sustained thermal runs.
- Add Core ML only after the runtime benchmark is trustworthy.
