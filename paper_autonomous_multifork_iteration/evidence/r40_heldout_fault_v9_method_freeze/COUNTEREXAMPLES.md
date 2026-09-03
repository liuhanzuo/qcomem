# V9 regression counterexamples

The test suite independently rejects: missing/unsigned binding; signed CUDA/typed-token semantic drift; binding replay to the same root; replay to a fresh root; wrong nonce or attempt; terminal actual-environment forgery; and no-replace overwrite. Static and reflection tests prove there is no accessible authority token, AuthorizedPlan, Lifecycle, private/public spawn method, caller worker-spec parameter, or second Popen site.

The signal-window regression wraps the real Popen, sends SIGTERM after the child exists but before Popen returns, and verifies exit 143, eight failure terminals, and no surviving worker. A valid control launches eight real workers and validates actual argv/executable inode+SHA, cwd inode, environment/CUDA UUID, PID/PGID, and spawned-spec commitments. The suite also exercises canonical binding/archive, signed artifact, typed contract, snapshot, terminal, and transactional freeze primitives inherited from v6--v8.
