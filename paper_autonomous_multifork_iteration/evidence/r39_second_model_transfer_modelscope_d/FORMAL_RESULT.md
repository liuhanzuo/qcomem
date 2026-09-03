# R39 ModelScope-D second-model formal result

Status: **valid negative; not a positive second-model/second-runtime transfer**.

The frozen ModelScope-D package was executed once on Trial 1907355 with eight
distinct NVIDIA H20-3e devices.  The formal interval was
`2026-08-26T15:28:09Z`--`2026-08-26T15:29:33Z`.  All eight raw shards and all
eight full-vocabulary FP32 sidecars completed, source/static/model authorities
were byte-identical before and after execution, and detached replay verified
the frozen aggregate.

## What passed

- All 8/8 ranks were scientifically valid and used distinct frozen inputs and
  GPU UUIDs.
- The seven-target status vector was
  `[full, full, full, not_applicable, partial, full, full]`.
- Prefix immutability, private mutable ownership, cross-arm exactness, and
  cross-N request-0 exactness passed on 8/8 ranks.
- All 32/32 preregistered targeted controls (four per rank) failed first at
  their expected predicate.
- The manual one-shot split agreed exactly with the official one-shot wrapper
  on 32/32 full-vocabulary rows (relative L2 and max-absolute error both zero).
- The deep-materialized and persistent-Q16 arms agreed exactly with the
  official `DynamicCache` chunked path on 96/96 full-vocabulary rows.
- The eight sidecars contain 192 FP32 records and 190,709,760 bytes in total.

## Why the global result is negative

The independent standard-cache authorization control failed its frozen
numeric gate.  Official `DynamicCache` document/query chunking and official
one-shot recomputation had identical top-1 tokens on 32/32 rows, but all 32/32
rows exceeded the preregistered relative-L2 threshold of 0.005.  The observed
range was `0.005182636097508993`--`0.013834544925893082` (maximum absolute
error `0.28125`).  Consequently `reference_authorized=false` on all 8 ranks,
the aggregate has `passed=false`, and its frozen scientific outcome is
`valid_negative_second_model_second_runtime_transfer`.

This evidence may support only the bounded structural observations listed
above.  It must not be cited as a positive semantic transfer, runtime
independence, portability, performance, capacity, memory-saving, compiled
dispatch, paged-tail, scheduler, concurrency, vision-path, or production
result.

## Acquisition and provenance

The official ModelScope acquisition downloaded 14/14 pinned files
(1,769,980,952 bytes) from official ModelScope hosts.  Every file succeeded on
its first zero-origin HTTP-200 attempt; no range request, append, partial reuse,
or retry occurred.  The weight SHA-256 is
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`
and the tokenizer SHA-256 is
`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`.
The resulting 14 repository files plus source marker were fully hashed and
closed read-only.  Runtime identity was Python 3.11.13, PyTorch 2.11.0+cu129,
Transformers 5.14.1, and Qwen3.5-0.8B revision
`2fc06364715b967f1860aea9cf38778875588b17`.

## Evidence closure

- Execution package SHA-256:
  `4cbf057580a1d4aeb55a1c2f79eefefb02ff06f495f0804771700fa3bb32198e`.
- Full evidence archive SHA-256:
  `13cbda7d30060608a49a7ac4d9b948a2ab3f73efce8ee6f50cc14f5a2406531a`.
- Metadata archive SHA-256:
  `a836576953940030ce1ec4cb1505c13a57586c1f61ac78ddf204ef62da9071e5`.
- Aggregate SHA-256:
  `2354141432a78bebb87324dd7147774847b9a2a9e19e76e1b01d078c5e058a7f`.
- Artifact ledger SHA-256:
  `4569df28b4b8d95ac202be834cf903b256fa860ad16d74fcf2442f8344a1e87f`
  (41 rows).
- Terminal ledger SHA-256:
  `d65a9a84400b70f0b8d1628d935a4cff301836d4f6157facac41e05f450bf706`
  (42 entries: the 41 ledger rows plus the artifact ledger itself).
- Acquisition receipt SHA-256:
  `b7880a9e64841d3c0ed71e3269cc0400856a428277a5dab6cf0a7e75f876d0fb`.
- Pre/terminal model-authority SHA-256:
  `6a445d1bdabab3e820e7595c39675a7e9ef5507d6e9e0ca2d002bc6780b18646`.

The full archive was downloaded to `formal_h20/`, extracted into a fresh local
directory, checked against every terminal-ledger entry (including all eight
sidecars), and replayed locally with the detached no-ML replay implementation.
The replay reproduced the aggregate SHA-256 and the valid-negative outcome.

