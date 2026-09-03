# R40 v10 local closure and CUDA-gated staging

Status: **local closure PASS; H20/CUDA not run; science ineligible until same-run CUDA smoke passes**.

V10 integrates the final hash-bound passive-lineage mechanism around the
unchanged real builder. It observes every persistent-rooted `aten.clone`/copy
edge, requires exact N×60 direct clone closure for materialized policy, supports
borrowed zero-edge exact alias, holds exact source/destination objects, seals
the private dispatch ledger, and rejects wrong-source offset views, returned
views, unused edges, and forged events.

The faithful 30-layer/8-request CPU E2E test exercises install, unchanged
builder signature, 480 edges, setup/transition/generation phase tuples, atomic
artifact write/read/hash binding, lifecycle, restore, exact build count,
primary-memory absence counters, tamper cleanup, and orphan rejection. Exact
row universe/schema/descriptors/intervals and aggregate predicates are tested.

`formal/launch_h20.sh` is non-overwriting and one-shot. Without
`R40_H20_EXECUTION_AUTHORIZED=yes` it exits before mutation. An authorized run
verifies sources, makes staging read-only, executes the PyTorch 2.11+cu129 CUDA
smoke before science, then finalizes and terminally rehashes sources. The
archive is staging material, not GPU evidence.

LB01--LB04 remain local mechanism tests and are not H20 faults, a production
fault campaign, or formal sensitivity evidence.
