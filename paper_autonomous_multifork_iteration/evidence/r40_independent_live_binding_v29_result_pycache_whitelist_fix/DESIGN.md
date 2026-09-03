# V29 exact result-pycache authority

The immutable primary launcher sets `PYTHONPYCACHEPREFIX` to
`primary/pycache`. V28 proved that the successful formal protocol produces 31
CPython 3.11 bytecode files plus 13 parent directories there. These are
non-scientific result-sink artifacts, but terminal closure must still enumerate
and hash them.

V29 derives the only accepted file set from the 31 flat `.py` entries in
`primary/preregistration/code.sha256`. Every source maps to exactly one file
under the fixed staged `PRIMARY_CODE` projection with suffix
`.cpython-311.pyc`; the accepted directory set is exactly their parent closure.
The final lexical-tree equality continues to reject every unlisted node.

Scientific execution and R40 binding/finalizer semantics are byte-identical to
v28. Only `preregistration.json` and `executed_source/r40_tree_closure.py`
change in the scientific payload.
