# Boundary after full independent audit

The verifier now holds initial live tensor objects, reconstructs normalized
serializer storage IDs from all current live rows, rejects cross-owner aliases,
and rechecks persistent object/storage/content/descriptor each phase.

V7 defines a fail-closed lineage receipt for the ambiguous equal-content case:
each semantic request coordinate must carry an independently captured
`aten.clone.default` edge from the exact persistent source storage/content to
the actual destination live storage/content. The interface and equal-content
CPU fixture pass, but production TorchDispatch capture and PyTorch 2.11/CUDA
behavior are not yet verified. The package therefore remains HOLD pending that
integration and every remaining gate in `formal-blocker.json`.
