.PHONY: bootstrap run system-info smoke context manual-context comem-smoke comem-multidoc comem-multidoc-diagnostic mlx-replay mlx-replay-diagnostic test

bootstrap:
	bash scripts/bootstrap.sh

run:
	bash scripts/run_first_benchmark.sh

system-info:
	.venv/bin/macllm-system-info

smoke:
	.venv/bin/macllm-run-mlx --config configs/smoke.json

context:
	.venv/bin/macllm-context-bench

manual-context:
	.venv/bin/python -m macllm_bench.manual_mlx

comem-smoke:
	.venv/bin/python -m macllm_bench.comem_bench

# Formal entry point: exits before model loading unless the machine is on AC,
# low-power mode is off, and the preflight checks pass.
comem-multidoc:
	.venv/bin/python -m macllm_bench.comem_multidoc_bench

# Implementation/debug run only. Its JSON is always assessed against the same
# formal criteria and may be marked formal_result_eligible=false.
comem-multidoc-diagnostic:
	.venv/bin/python -m macllm_bench.comem_multidoc_bench --power-policy record-only --output results/q_comem_multidoc_diagnostic.json

# Frozen H20-selected state policy on Qwen3.5 hybrid layers. The formal entry
# exits before model loading unless AC/power/thermal checks pass.
mlx-replay:
	.venv/bin/python -m macllm_bench.mlx_replay_bench

mlx-replay-diagnostic:
	.venv/bin/python -m macllm_bench.mlx_replay_bench --power-policy record-only --context-lengths 128 --runs 1 --max-new-tokens 2 --no-save-store --output results/qcomem_mlx_hybrid_diagnostic.json

test:
	.venv/bin/python -m compileall -q src
	.venv/bin/python -m unittest discover -s tests -v
