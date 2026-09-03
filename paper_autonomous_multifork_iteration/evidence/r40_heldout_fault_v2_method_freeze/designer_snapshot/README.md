# Fresh-designer input boundary

This directory is the complete permitted input for a future isolated fault
designer.  The designer must not receive repository access, prior campaign
materials, detector source, test source, review discussion, or any execution
result.  The public method contract is intentionally visible; implementation
details and historical examples are not.

The designer's sole output is one sealed `fault-set.json` satisfying
`FAULT_SET_OUTPUT_SCHEMA.json`.  The output must be frozen before an executor
sees it.  The designer must not predict which gate a case should trigger and
must not revise, replace, or withdraw a case after any execution begins.

This snapshot defines a constructed, designer--executor-separated evaluation.
It does not turn injected cases into naturally occurring defects.

