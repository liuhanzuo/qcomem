# Boundary after full independent audit

The verifier now holds initial live tensor objects, reconstructs normalized
serializer storage IDs from all current live rows, rejects cross-owner aliases,
and rechecks persistent object/storage/content/descriptor each phase.

This is still insufficient for a source-independent mapping proof when two
persistent coordinates have identical content and geometry. Materialization
deliberately destroys source/request storage identity, so no remaining passive
observable identifies clone provenance. Mutating values supplies a challenge
but changes scientific state; accepting builder provenance restores common-mode
trust. The package therefore remains HOLD pending a scientifically acceptable
discriminability design and completion of every gate in `formal-blocker.json`.
