# RW-D5 operational recovery P

P supersedes O before GPU submission. O closed the cwd and import-authority
chain, but wrote the recovery pass marker before those probes. A probe failure
could therefore leave a false pass marker and make the fresh receipt path
unusable. O is retired and supplies no execution.

P preserves the exact O authority checks and unchanged K science. All model,
manifest, root-stat, shadow, frozen safe-path, focused-test origin, and SGLang
origin checks complete before any recovery receipt directory is created. Only
after all probes pass does P create the directory, copy the repair receipt,
write the frozen-authority ledger, and publish the precise status
`all_recovery_preflight_passed_before_k_exec`. The controlled K exec is the
immediately following stage.

`test_recovery_p_publication.py` retains the O negative and authority tests,
checks the exact source-stage order, and executes guard-failure and
origin-failure gates. Both failures must return nonzero with neither receipt
directory nor pass marker present.

P is operational authority, not a new science freeze and not a result. Fresh
independent GREEN is required before it may launch K's 16 affected cells.
