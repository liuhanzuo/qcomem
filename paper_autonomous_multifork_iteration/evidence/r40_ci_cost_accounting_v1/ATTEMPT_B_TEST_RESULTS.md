# Attempt B pre-execution tests

Command, run from the paper directory before any Attempt B timing row:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s evidence/r40_ci_cost_accounting_v1/tests -p 'test_*.py' -v
```

Result: 10 tests run in 0.723 seconds; all passed; zero failures and zero
errors.  This includes the six original tests and four Attempt B isolation,
path-safety, exclusive-copy, and frozen-repeat tests.

