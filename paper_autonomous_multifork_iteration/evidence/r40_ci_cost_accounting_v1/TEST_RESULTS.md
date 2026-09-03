# Pre-execution test result

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s evidence/r40_ci_cost_accounting_v1/tests -p 'test_*.py' -v
```

Result before the first timing row: 6 tests run, 6 passed, 0 failed, 0 errors.
The tests cover the frozen component order, Darwin resource-report parsing,
unsafe-tar rejection, aggregate statistics, explicit Falcon/blind-fault
blockers, and the prohibition on interpreting local replay as H20 overhead.

