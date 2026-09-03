# Internal-only closure: valid negative

The frozen R30 end-to-end semantic control is an infrastructure-valid
**negative**. It must not be used in the paper, abstract, tables, figures,
claim map, rebuttal, or reviewer packet as evidence of semantic equivalence,
quality preservation, runtime independence, hardware portability, or a passed
end-to-end control.

The preregistered exact-token primary gate failed: 48/64 candidate decisions
and 12/16 candidate trajectories matched the source-distinct dense
Transformers reference. All four ownership arms failed on the same previously
unseen case/request and produced the same divergent trajectory. Ownership and
lifecycle replay passed, but that does not rescue the semantic gate.

The history-matched full-vocabulary measurements are secondary diagnostics,
not acceptance criteria: top-1 agreement was 56/64, relative-L2 error had mean
0.0405989 and maximum 0.0669337, and maximum absolute error had mean 0.661331
and maximum 1.16406. These measurements may not be hidden, thresholded after
the fact, renamed as a pass, or replaced by a cherry-picked input.

No rerun with another input, altered seed, shorter horizon, removed request,
changed arm, or tuned numeric threshold is authorized from this attempt. The
only defensible internal reading is that the four ownership implementations
behaved consistently and passed their ownership gates, while the approximate
retained-state path did not universally preserve the dense reference's greedy
trajectory on the frozen output-unseen inputs.

The complete immutable artifacts, including all 144 FP32 logit sidecars, remain
on the remote shared filesystem. The local mirror contains the input, source
and artifact ledgers, logs, reference result, full candidate result, and
independent replay; it intentionally omits the large sidecars while preserving
their exact entries in `final-artifacts.sha256`.
