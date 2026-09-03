# R40 CI-cost accounting

This directory separates what is cheap to replay locally from what remains
unmeasured.  It contains no GPU execution and does not modify earlier evidence.

Run the unit tests from the paper directory:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s evidence/r40_ci_cost_accounting_v1/tests -p 'test_*.py' -v
```

The frozen measurement command is:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  evidence/r40_ci_cost_accounting_v1/measure_replays.py --run --repetitions 3
```

Inventory and aggregation are produced with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  evidence/r40_ci_cost_accounting_v1/audit_inventory.py
PYTHONDONTWRITEBYTECODE=1 python3 \
  evidence/r40_ci_cost_accounting_v1/aggregate_results.py
```

All commands are fail-closed and refuse to replace an existing output.

