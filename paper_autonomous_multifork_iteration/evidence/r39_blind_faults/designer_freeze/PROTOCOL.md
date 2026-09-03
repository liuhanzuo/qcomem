# ForkAudit R39 blind fault protocol

Status: **immutable designer freeze**  
Frozen: 2026-08-26T11:04:56Z  
Designer role: isolated PDF-only fault-set designer

This protocol fixes eleven new hybrid KV/GDN ownership, lifecycle, and dispatch faults and a matched-clean comparison against four observers: output equality, persistent-base invariants, allocation assertions, and ForkAudit. It is a fixed-fault sensitivity study, not a defect-population sample, detection-rate estimate, security exercise, or manuscript evaluation.

## Blind-design boundary

The only task-content input read by the designer was the frozen 27-page PDF:

`/Users/liuhanzuo/MacLLM-Bench/paper_autonomous_multifork_iteration/output/pdf/ForkAudit_sol_polished_20260826.pdf`

Its SHA-256 is `f55c0c2dca7201904ff82897af75e6f7fc6a31cbf52a1ee76624280d4cdcb72c` and its size is 2,645,908 bytes. Venue-generic systems/testing knowledge was also permitted. Repository source, existing evidence, earlier fault files, reviews, implementation, and candidate outputs were not inspected. PDF-derived text and page renders were used only to read the frozen PDF.

The PDF-visible live faults were treated as exclusions: inter-request KV reservation alias, sequence swap, omitted/delayed tail COW, GDN request-base or request-peer alias, position offset, materialized mask, wrong Python callable, dense KV view, inactive-page scribble, duplicate commit, effective-scale drift, stale GDN binding token, stale/wrong slot lease, reclaim before zero scrub, and the four seeded GDN recurrence changes. None of the faults below is a mere payload variation of those mechanisms.

## Frozen system geometry and selectors

Unless a row says otherwise, use the PDF's fixed Qwen3.5-35B-A3B geometry: ten full-attention layers, thirty GDN layers, BF16/Q16 KV, 128-token pages, a 4,095-token document with a 127-token final page, a 32-token query, eight argmax outputs, seven feedback calls, N=2, shared-document KV, borrowed GDN, batch-one sequential round-major execution, and one CUDA stream.

Every clean and mutant lane starts in a fresh process from the same identity-bound input and synchronized post-priming state. Selection is deterministic and precedes mutation:

1. Order requests by frozen request identity; call the first two A and B.
2. Order attention/GDN layers by global model-layer index.
3. Order tensor coordinates by global layer, frozen role/family, then coordinate index.
4. If equal geometry is needed, choose the first eligible ordered tuple and record why earlier tuples were ineligible.
5. If two documents are needed, choose the two lowest distinct document-identity hashes in the already frozen primary eight-book set.
6. Never use tokens, logits, terminal state, allocator results, or detector verdicts to select a target.

## Event horizons

- H0: identity bind, load, warmup, and synchronized post-priming allocator baseline.
- H1: post-fork setup inventory before query execution.
- H2: partial-tail detach/copy and first query-token append.
- H3: first new request-private KV page allocation/publication after the detached tail fills.
- H4: registered transition after 32 query tokens, before feedback calls.
- H5: seven ordered feedback-token calls and per-call receipts.
- H6: terminal capture after eight output tokens, before teardown.
- H7: quiescent scrub, release, reclamation, and synchronized allocator restoration.
- H8: a fault-specific replacement, eviction, or compiled-dispatch confirmation event.

## Four frozen observers

Output equality compares mutant and matched clean at identical ordered call indices. Tokens require exact integer equality. Logits require exact equality of shape/dtype-bound, contiguous CPU-FP32 bytes for the complete vocabulary. Terminal KV/GDN state is deliberately excluded from this baseline. Unequal call cardinality is `not_evaluable`, never a pass.

The persistent-base invariant compares each lane's setup digest with every later registered digest for all persistent document-KV and persistent GDN-base physical byte ranges, including padding/inactive capacity when exposed. It does not inspect request-private state or ownership. Missing mandatory snapshots make it `open`.

Allocation assertions compare synchronized current allocated bytes, reset peak allocated bytes, and post-cleanup restoration with the matched clean lane at every applicable horizon. Registered endpoints require exact byte equality and H7 must restore exactly to H0. Native allocator assertions are retained. This is allocator accounting only, not NVML/process memory, capacity, or ownership provenance.

ForkAudit captures all existing mandatory identity, lifecycle, execution, and fault records and then runs unmodified offline replay with every pre-existing predicate enabled. Coverage and verdict remain separate. No predicate may be added after outcomes. In particular, per-call compiled-binary/autotuning coverage must remain partial unless that predicate already existed before this freeze.

## Paired execution and precedence

For each opaque fault ID, run matched clean first and mutant second in fresh processes. Both traverse the same hook: clean records one eligible no-op; mutant records exactly one authenticated payload application. ForkAudit is replayed after execution and does not short-circuit this comparison lane; existing production assertions remain enabled. Selective suppression and post-failure continuation reruns are forbidden.

Required sidecars are the complete token sequence, complete per-call FP32 logits, persistent-base digests, synchronized allocator endpoints/restoration, full ForkAudit trace/replay, and a byte-bound eligibility/injection/consequence receipt. The latter validates that the fault occurred; it is not counted as one of the four observers.

Localization uses chronological event order. At the same event, precedence is:

1. invalid eligibility, injection, or operational condition;
2. production exception/native assertion;
3. ForkAudit mandatory-record, schema, or byte-binding failure;
4. ForkAudit identity/lifecycle/storage/call predicate failure;
5. persistent-base invariant failure;
6. allocation assertion failure;
7. full-logit equality failure;
8. token equality failure.

Later observer outcomes are still reported if the lane reaches its expected horizon. Precedence labels the first consequence; it does not erase later evidence.

## Global validity requirements

A clean lane is valid only if it reaches the row's expected horizon; has complete applicable ForkAudit coverage with no existing predicate failure (while preserving the PDF-declared dispatch partial status); keeps document KV and GDN bases byte-identical; has consistent allocator endpoints and exact H7 restoration when applicable; exactly matches an independently sealed identical-protocol clean reference in tokens and complete logits; and emits exactly one eligibility/no-op receipt at the mutant locus.

A mutant is valid only if the clean is valid; the selector resolves before mutation without outcome information; the mutant differs only by the frozen payload and its unavoidable consequences; exactly one authenticated receipt binds selected objects, pre-state digests/ranges/versions, payload, and event index; the frozen first-consequence witness is present; and the expected horizon completes. An exception, timeout, OOM, or missing mandatory sidecar before the horizon is invalid for all eleven rows. No payload adjustment, target substitution, gate suppression, selective rerun, or post-output amendment is permitted.

## Immutable fault set

### R39-BF01 - wrong-source private-tail copy

- Class/locus: KV ownership and COW content lineage, H2, default arm.
- Payload: after allocating and binding the correct private tail for request A at the lowest full-attention layer, but before any query write, copy K and V for all 127 positions from positions 0..126 of that layer's immediately preceding full document page. The correct source would be the actual 127-token document tail. Keep detach timing, owner, valid length, append position, and every other layer clean.
- Clean requirements: one 127-token detach occurs before first append; destination pre-append digest equals the true tail digest.
- First consequence: before the first append write, the private-tail digest differs from the persistent tail although ownership/order are correct.
- Expected horizon: H6.
- Expected comparison: logits fail (token divergence not required); base and allocation pass; ForkAudit should fail at copied-tail content/lineage if represented, otherwise at terminal logical-KV/cross-arm semantics.
- Validity/success: preceding-page and true-tail digests must differ; destination is private/disjoint; no persistent page is written; the exact wrong-source digest fills 127 pre-append positions and H6 completes.
- Failure: wrong length, late detach, base write, another mutation, or early stop.
- Novelty: unlike omitted or delayed COW, detach and ordering are correct; only copy provenance is wrong.

### R39-BF02 - same-request cross-layer KV page collision

- Class/locus: KV ownership/allocator cross-layer collision, H3, request A.
- Payload: bind one identical backing byte range to the first new private KV page roles of the first two full-attention layers. Preserve request/sequence identity, paged-view type, position, and inter-request reservation disjointness. Both layers perform normal writes into the shared range.
- Clean requirements: selected layer-page roles have equal geometry, disjoint clean ranges, and both publish a new page at H3.
- First consequence: two distinct layers of one request overlap in physical bytes; after the second writer, at least one page digest differs from clean.
- Expected horizon: H6.
- Expected comparison: logits fail; persistent bases pass; allocation current or peak fails by one missing page allocation; ForkAudit fails on cross-layer overlap if registered, otherwise on logical-KV/semantics. A complete ForkAudit pass exposes a predicate gap.
- Validity/success: prove nonempty overlap only between selected private roles, no peer/base alias, and both normal writes; preserve the overlap through a write pair and reach H6.
- Failure: metadata-only alias, inter-request alias, or early stop.
- Novelty: the PDF reservation fault aliases requests; this aliases layers within one request.

### R39-BF03 - torn final hybrid commit with KV rollback

- Class/locus: hybrid KV/GDN atomic lifecycle commit, final H5 feedback call.
- Payload: after logits for output token eight are materialized, publish the correct next GDN state and return the logits, but suppress that call's KV publication. Restore the written private-KV slot, page-valid count, logical length, and KV version to pre-call values. Call cardinality remains one.
- Clean requirements: the clean call advances KV and GDN exactly once and emits output token eight.
- First consequence: KV remains at version t while GDN/call receipt advance to t+1; emitted logits are unchanged.
- Expected horizon: H6.
- Expected comparison: exact tokens/logits pass; base and allocation pass; ForkAudit fails on ordered append/cardinality, hybrid-version coherence, or terminal logical-KV equality.
- Validity/success: a real clean KV append exists, GDN equals the clean next state, call cardinality/logits equal clean, and only KV publication rolls back.
- Failure: skipping/duplicating the whole call, rolling back GDN, or changing output.
- Novelty: the PDF duplicates a whole commit; this commits one hybrid subsystem once and rolls back the other after output production.

### R39-BF04 - wrong-source GDN privatization

- Class/locus: GDN ownership/lifecycle content lineage, H4, borrowed-to-private transition.
- Payload: for the first frozen GDN coordinate, allocate correct disjoint private storage and correctly advance its binding token, but initialize bytes from its H1 persistent-base snapshot rather than its correct H4 transition state. All other coordinates are clean.
- Clean requirements: clean H4 source differs from H1 base; clean storage is disjoint and token advances.
- First consequence: disjointness/token checks pass, but private content equals stale H1 base rather than H4 state.
- Expected horizon: H6.
- Expected comparison: logits fail after H4; base and allocation pass; ForkAudit fails on transition content binding, terminal GDN, or cross-arm semantics.
- Validity/success: no alias, correct token advance, unequal stale/correct source digests, exactly one stale-source coordinate, H6 completes.
- Failure: alias, stale token, or recurrence mutation.
- Novelty: unlike PDF alias/stale-token/wrong-recurrence faults, ownership metadata is correct and copy provenance is wrong.

### R39-BF05 - disjoint GDN role permutation

- Class/locus: GDN role binding across layers, immediately after H4.
- Payload: select the first pair of request-A private GDN coordinates with identical geometry and distinct clean content digests. Swap role-to-tensor bindings without copying bytes or aliasing storage. Owners, peer disjointness, tokens, and allocation remain unchanged.
- Clean requirements: eligible equal-geometry, unequal-content pair exists; clean role map is bound.
- First consequence: all ranges remain private/disjoint, but two role-bound content digests are transposed.
- Expected horizon: H6.
- Expected comparison: logits fail; bases and allocation pass; ForkAudit fails on role/content binding or terminal GDN/cross-arm equality.
- Validity/success: selection uses only frozen geometry/order and pre-injection digest inequality; exactly two disjoint bindings transpose; no bytes change at injection; H6 completes.
- Failure: unequal geometry, storage alias, or output-guided selection.
- Novelty: the PDF aliases bases/peers; this misbinds disjoint same-request coordinates.

### R39-BF06 - cross-family KV/GDN arena collision in unused KV capacity

- Class/locus: hybrid allocator/ownership, H4.
- Payload: choose the first private GDN tensor that fits in an aligned unused suffix of request A's first private KV page for the whole horizon. Back the GDN tensor with that exact KV suffix, initialize correct GDN bytes, and run normal updates. Never advance KV valid length into the overlap through H6.
- Clean requirements: GDN storage is exclusive; KV suffix is allocated but logically unused through H6; clean ranges are disjoint.
- First consequence: one byte interval simultaneously has request-private KV-capacity and GDN roles while both logical values can remain clean.
- Expected horizon: H6.
- Expected comparison: exact outputs are expected but not required for validity; bases pass; allocation current/peak fail by the omitted GDN allocation; ForkAudit fails if cross-family all-role overlap is enforced. A complete pass exposes a contract gap.
- Validity/success: overlap is nonempty/aligned/contained/outside valid KV bytes at every call; no persistent storage or peer is involved; same interval retains both roles through H6.
- Failure: overlap touches valid KV, introduces base/peer alias, or no frozen eligible interval exists.
- Novelty: all PDF ownership mutations are within KV or within GDN; this crosses allocator families.

### R39-BF07 - wrong-target teardown scrub of persistent GDN base

- Class/locus: GDN lifecycle and persistent-base integrity, H7 after correct private scrub.
- Payload: issue one extra scrub through a stale persistent-base handle. Select the first base coordinate containing a nonzero aligned slice and zero the lowest-offset such slice of length `min(4096, tensor_nbytes)`. Do not alter bindings or allocation metadata.
- Clean requirements: private state alone is scrubbed and every base remains unchanged; selected base slice contains a nonzero byte.
- First consequence: the selected persistent-base digest changes after terminal outputs are sealed.
- Expected horizon: H7.
- Expected comparison: exact outputs pass; persistent-base invariant fails; allocation passes; ForkAudit fails on prefix/base immutability if H7 is registered, otherwise records insufficient post-terminal scope.
- Validity/success: private scrub remains correct; no private/base alias exists; exact base slice changes; H7 completes.
- Failure: dependency on aliasing, pre-output corruption, or allocation-metadata change.
- Novelty: PDF base corruption uses alias/in-place update; this dispatches cleanup to the wrong disjoint owner.

### R39-BF08 - post-teardown private-GDN retention leak

- Class/locus: GDN reclamation/allocation lifecycle, H7 release.
- Payload: after correct zero-scrub/lifecycle completion, retain exactly the first exclusive request-private GDN backing storage in the retired-request registry. Release all KV and all other GDN storage normally and never reuse the retained storage.
- Clean requirements: the storage is exclusive and clean releases it; clean returns exactly to H0 bytes.
- First consequence: a zeroed retired-owner storage remains live; current allocated bytes exceed clean by its allocator-accounted size.
- Expected horizon: H7.
- Expected comparison: outputs and bases pass; allocation current/restoration fail; ForkAudit fails if cleanup inventory/reclamation is registered, otherwise a primary pass exposes lifecycle scope.
- Validity/success: exactly one zeroed nonpersistent backing storage is retained and measured excess equals its size.
- Failure: unsanitized/reused storage, multiple leaks, or unsynchronized H7.
- Novelty: unlike reclaim-before-zero, this zeroes correctly and fails to release afterward.

### R39-BF09 - reassignment without last-use event fence

- Class/locus: concurrent hybrid lifecycle, H8 cancellation/replacement in the PDF's bounded two-stream N=2 geometry.
- Payload: suppress only request A's last-use-event wait. Correctly zero-scrub, increment lease, reassign, and initialize the exact KV/GDN reservation bundle for replacement B. Then release a prearranged event so A's already enqueued state writer completes after B initialization and before B's first call.
- Clean requirements: clean waits for A's event; event schedule proves `A enqueued -> B initialized -> A late write -> B call`; clean B matches serialized reference.
- First consequence: metadata/zero/epoch are correct at reassignment but the happens-before relation is false; A's late operation changes B-owned bytes.
- Expected horizon: H8 after B's first call/capture.
- Expected comparison: B logits fail; bases and allocation pass; ForkAudit may fail only on later state/semantics and may miss the event-lifetime relation without per-kernel evidence.
- Validity/success: scrub/epoch/identity/reservation equal clean; A event is incomplete at reassign and complete before B call; late write changes B storage; only wait is omitted.
- Failure: stale/wrong lease, omitted scrub, quiescent reuse, or nondeterministic ordering.
- Novelty: unlike stale/wrong lease or reclaim-before-scrub, lease/scrub are correct and the untested last-use fence is wrong.

### R39-BF10 - hybrid document-component split brain

- Class/locus: hybrid identity/dispatch at H1.
- Payload: with documents A and B pre-resident, assemble request A using document A's immutable KV object and document B's immutable persistent GDN base. Preserve request/query identity A, correct component storage ownership, and all call contracts.
- Clean requirements: both components bind A; A/B identity and GDN-base digests differ; both bases are resident before allocator comparison.
- First consequence: two valid immutable component bindings carry different document identities in one logical fork.
- Expected horizon: H6.
- Expected comparison: logits fail; both bases and allocation pass; ForkAudit fails on object-to-role/document identity before semantics.
- Validity/success: no incorrect private/peer alias, no document writes, only cross-component association changes, H6 completes.
- Failure: whole request/sequence swap, mutated component, or nonresident comparison skew.
- Novelty: unlike a KV sequence swap, this forms one request from two internally valid document identities across state families.

### R39-BF11 - unbound compiled-selection drift

- Class/locus: compiled dispatch/autotuning provenance, first H5 feedback full-attention call.
- Payload: after the correct Python callable and argument bytes are recorded, choose the lowest-SHA256 noncanonical ABI-compatible compiled/autotuned artifact already available for the identical signature, rather than clean's artifact. Selection uses pre-call artifact metadata only. Python callable, query, mask, position, append, scale, schedule, and candidate source stay exact.
- Clean requirements: clean compiled identity is externally captured only for injection validation; a distinct compatible artifact exists; Python receipts pass.
- First consequence: executed artifact hash differs while every PDF-described Python-level receipt is exact.
- Expected horizon: H6.
- Expected comparison: exact outputs and allocation are expected for a legal alternate autotune variant but any differences are reported; bases pass; ForkAudit remains partial rather than failing unless a per-call compiled predicate independently pre-existed this campaign.
- Validity/success: external receipt proves different artifact hashes for the same signature; selection precedes outcomes; Python bytes remain exact; alternate executes through H6; partial coverage is reported honestly.
- Failure: Python-level change, outcome-guided selection, ABI stop, or post-hoc ForkAudit extension.
- Novelty: the PDF changes the Python callable; this changes only the explicitly unbound compiled/autotuning selection.

## Success, failure, and reporting rules

A pair is valid only when every global and row-specific criterion holds and both bundles bind this frozen plan. A fault is reached only when its authenticated injection receipt and first-consequence witness pass and its horizon completes. An observer detects a fault only if that observer gives its frozen failing outcome on a valid mutant while the matched clean passes. `open`, `partial`, `not_evaluable`, and production-only stops are not detections by that observer.

The campaign succeeds when all eleven rows are individually reported as valid-reached or invalid with the frozen reason; all four observer outcomes are retained; no clean false positive, payload tuning, selective rerun, or rate claim occurs. It fails if any result lacks a passing matched clean, any target/payload changes after outputs, a mandatory outcome is omitted, dispatch partial coverage is upgraded without evidence, or outcomes are pooled into recall/accuracy.

A miss is a boundary of this fixed implementation/predicate set, not proof of undetectability. A detection is one fixed positive-control result, not an estimate of unseen-fault sensitivity.

Only one amendment is allowed: before any candidate execution, the executor may hash an amendment marking a row ineligible because its frozen selector has no eligible object/artifact. The amendment may not substitute another target or payload.

## Integrity block

- Source PDF SHA-256: `f55c0c2dca7201904ff82897af75e6f7fc6a31cbf52a1ee76624280d4cdcb72c`
- Canonical fault-set SHA-256: `a919c53cda32a1e1089568b340725ff287c3d74ac590e25cf97d124779901ac2`
- Protocol-core SHA-256: `2aa9ca0cc5652591bbee5338abe97436657c14f6c4605bdc89cd73cf69c88b9e`
- Canonical `plan.json` SHA-256: `cfb9f93f5b60377c1db9a3f7cca57d376c657b72e0e9449804166c75a84efe4c`

Hash rules:

- Fault set: SHA-256 of UTF-8 bytes emitted by `jq -S -c '.faults' plan.json`, including its trailing LF.
- Protocol core: SHA-256 of bytes emitted by `sed '/^## Integrity block$/,$d' PROTOCOL.md`, including emitted line endings.
- Canonical plan: SHA-256 of UTF-8 bytes emitted by `jq -S -c '.integrity.plan_canonical_sha256=null' plan.json`, including its trailing LF.

The protocol-core hash intentionally excludes this integrity block, avoiding an impossible self-referential whole-file hash. The canonical-plan rule nulls only its own hash field for the same reason.
