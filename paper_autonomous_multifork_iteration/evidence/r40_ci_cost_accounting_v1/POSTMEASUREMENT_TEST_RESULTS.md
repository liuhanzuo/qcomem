# R40 post-measurement verification tests

Command, run from the paper directory after inventory and aggregation:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s evidence/r40_ci_cost_accounting_v1/tests -p 'test_*.py' -v
```

Result: 16 tests run in 1.264 seconds; all passed; zero failures and zero
errors.  Coverage includes frozen protocol boundaries, safe preparation,
Attempt A/B separation, exact artifact counts, three rows per component,
unmeasured-cost prohibitions, and terminal-manifest bulk-input bindings.

