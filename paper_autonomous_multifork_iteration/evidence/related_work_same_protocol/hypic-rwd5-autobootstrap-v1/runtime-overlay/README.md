# RW-D5 Trial1880346 runtime overlay

This is a minimal operational adapter for the independently GREEN Z science
bundle. It is not a new freeze, STOP, or independent audit round. The source Z
identity is recorded in `runtime-overlay.json`.

Only these authority/operation surfaces differ from Z:

- `platform-execution-authority-runtime-trial1880346.json` binds the exact
  Job247699/Trial1880346 queue358 platform configuration;
- `code/build_hypic_retained_state_static.py` validates that exact receipt and
  PID-1 identity while retaining the historical Z invalid-attempt boundary;
- `code/launch_hypic_retained_state_bytes_8gpu.sh` accepts only the runtime
  driver's sentinel and uses fresh Trial1880346 run/instrumented paths;
- `code/launch_hypic_retained_state_bytes_runtime_trial1880346.sh` verifies an
  externally supplied overlay-manifest SHA, all member hashes, exact PID-1
  Job/Trial/scope, the prep asset observation, and fresh paths before controlled
  internal launch.

All producer, receipt, replay, workload, topology, ownership, physical-range,
and terminal-closure files listed in `runtime-overlay.json` are byte-identical
to Z. The execution scope remains exactly two modes by eight workloads. No
remote staging or GPU execution is allowed until Trial1880346 is Running with
exactly one Pod and the read-only inventory closes.
