# RW-D5 operational recovery N

N supersedes M before GPU submission. M pinned the K inputs and cleared the
ambient environment, but `env -i` retained the caller's working directory.
Because K invokes Python modules with `-m`, an attacker-controlled cwd remained
ahead of the frozen code on `sys.path`. M also retained an empty trailing
`LIBRARY_PATH` component. M is retired and supplies no execution.

N preserves the same completed asset repair and unchanged K science. Its
preflight uses pinned command paths, verifies both exact manifests and every
post-repair file stat/hash, requires zero writable top-level model files and
fresh N paths, and copies the frozen repair receipt. Before the controlled
exec it rejects the exact root-level module shadows, performs a real `cd /`,
and verifies the resulting cwd. It then executes `/usr/bin/env -i` with a
closed environment, `PYTHONSAFEPATH=1`, no empty library-path component, and
explicit absolute values for every K-overridable input. The K launcher is
invoked by exact absolute path with `--noprofile --norc`.

`test_recovery_n_cwd_env.py` parses the actual exec block. It injects malicious
ambient values and launches a real child from a malicious cwd containing both
`sglang` and the focused unittest module. The child must run from `/`, exclude
the attacker cwd and empty string from `sys.path`, and observe only the frozen
environment whitelist.

N is operational authority, not a new science freeze and not a result. Fresh
independent GREEN is required before it may launch K's 16 affected cells.
