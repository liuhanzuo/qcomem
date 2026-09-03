# Frozen-preregistration build location

The canonical preregistration is generated with:

```text
gpu/build_qcomem_transformers_forkaudit_transfer_prereg.py --stage source
gpu/build_qcomem_transformers_forkaudit_transfer_prereg.py --stage static
```

The formal launcher then runs `--stage verify-source`, a single pre-output full model-authority pass, `--stage verify-static` (bytewise reconstruction from the exact PG-19 train64 data and local tokenizer), and `--stage gpu-assignment` before any model output.

The repository-side canonical `source-manifest.json` has raw SHA-256 `a00822ad4c83f2ee93380b3d6164cbca5b69bb0e5a2cda7189896be27fb94ec9` and verifies seven runtime/source files. The frozen formal environment produced `static-preregistration.json` with raw SHA-256 `76c1fcbe0ef620962c1ad92f3f0f16a482bff27b413f4973dfaa7609e4b5d429`. Its path-independent seven-artifact ledger has raw SHA-256 `c0a23e9d3f9d220257af97b78fd97661f315f0c82a3a010b57a771e3eeefbbfb` and normalized-entry SHA-256 `60cb6e7740dcb28f611da18ac863e699143bfa3410c215e59157ba9c8ea084b2`; the fourteen-weight ledger has raw SHA-256 `8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014` and normalized-entry SHA-256 `48cee72f5cd17af2e026712b5916b290dfbf892aaf7774131d9db08d4fae4c5e`. The formal source validator, bytewise static rebuild, Python compilation, ten targeted unit tests, and launcher syntax check all passed before these raw digests replaced the QS YAML placeholders. Model-authority schema v2 requires regular files with no write mode bits and closes full SHA-256 plus size/device/inode/ctime before and after outputs; because the worker runs as root, this is ordinary frozen-workflow mutation evidence and not resistance to a privileged-root or raw-device adversary. No GPU task has been submitted from this preregistration.

The current formal-root receipts are `source-verification.json` SHA-256 `6f6b1ac5fe7e0e36260059651ba753030ae91ab4781e5e8381c8687e185b216f`, `static-bytewise-rebuild.json` SHA-256 `d2ef59e8a21169823331534f0b439b0906e6b0bdfede1738ac4189f3d1c9803b`, and `model-mode-stat-preflight.json` SHA-256 `fb0671f059ab5bb9b1de11ce9d2dd4d51d63f2aa695ec10959ba27e5f294e1c3`. The stat receipt binds all 21 ledger paths and has normalized stat-snapshot SHA-256 `421a32fb4613c5943ba6f9a83443c7f5bb73b7621adb311f4e5a7ee9b49cd553`.

Frozen scientific settings: eight distinct PG-19 train books, document length 256, two 24-token queries per rank, `N={1,2}`, split depth 7, two semantic steps, one CUDA stream with request-index interleaving, and dense-oracle relative-L2 threshold 0.005. The full protocol and exact target/fault missingness are in `gpu/FORKAUDIT_TRANSFORMERS_TRANSFER_PROTOCOL.md`.
