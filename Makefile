.PHONY: bootstrap run system-info smoke context test

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

test:
	.venv/bin/python -m compileall -q src
