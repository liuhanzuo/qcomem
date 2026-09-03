# Falcon-H1 result-count clarification

This note corrects one wording ambiguity in the preserved human-readable
`RESULT.md`; it does not modify the formal archive or any machine-readable
result.

The authoritative `validation.json` and formal aggregate report two disjoint
sets of decisions:

- 40 registered control decisions: 5 registered control IDs on each of 8
  ranks; and
- 8 separately reported prefix-content mutation-detector decisions: 1 per
  rank.

Accordingly, the phrase “including the 8/8 prefix-content mutation detectors”
in `RESULT.md` should be read as “plus 8/8 separately reported prefix-content
mutation detectors.” The manuscript uses the latter, machine-readable count.

Bindings:

- `validation.json` SHA-256:
  `f2ece92098d9c9a3009354d9a5bbf228fefeb29fdee7bbe14672d37838674df9`
- formal aggregate SHA-256:
  `03b2dd60422641ffdd18ec4221a06c295ca36fa3de322dc148a5222a8579888b`
- preserved formal archive SHA-256:
  `6cbcf860120078e743eb759e2bead74a3bf980e07c4c16f588ee735d3662d6c3`
