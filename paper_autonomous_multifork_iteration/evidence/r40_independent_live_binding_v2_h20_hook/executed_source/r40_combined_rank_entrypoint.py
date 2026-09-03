#!/usr/bin/env python3
from __future__ import annotations

"""Chain the v2 pre-binder hook around the hash-pinned R39 v6 rank wrapper."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


EXPECTED_V6_ENTRYPOINT_SHA256 = "5c5ffdac992b0ee0e4f5f8a42bba0a7ce25749d187f395924f6b34a43a484365"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("forkaudit_r39_v6_entrypoint", path)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load frozen v6 rank entrypoint")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _runner_option(name: str) -> str:
    matches = [index for index, value in enumerate(sys.argv) if value == name]
    if len(matches) != 1 or matches[0] + 1 >= len(sys.argv):
        raise SystemExit(f"combined wrapper requires exactly one {name}")
    return sys.argv[matches[0] + 1]


def main() -> int:
    v6_path = Path(os.environ["R40_V2_V6_ENTRYPOINT"]).resolve()
    prereg_path = Path(os.environ["R40_V2_PREREGISTRATION"]).resolve()
    capture_root = Path(os.environ["R40_V2_CAPTURE_ROOT"]).resolve()
    source_root = Path(os.environ["R40_V2_SOURCE_ROOT"]).resolve()
    if _sha256_file(v6_path) != EXPECTED_V6_ENTRYPOINT_SHA256:
        raise SystemExit("v6 rank entrypoint hash drift")
    expected_prereg = os.environ["R40_V2_EXPECTED_PREREGISTRATION_SHA256"]
    if _sha256_file(prereg_path) != expected_prereg:
        raise SystemExit("v2 preregistration hash drift")
    sys.path.insert(0, str(source_root))
    from r40_h20_hook import install_h20_live_binding_hooks

    preregistration = json.loads(prereg_path.read_text(encoding="utf-8"))
    rank = int(_runner_option("--rank"))
    v6 = _load(v6_path)
    base_install = v6._install_primary_scope_wrappers

    def combined_install(*, runner_module: Any, recorder: Any):
        restore_v6 = base_install(runner_module=runner_module, recorder=recorder)
        try:
            restore_v2 = install_h20_live_binding_hooks(
                runner_module=runner_module,
                preregistration=preregistration,
                capture_root=capture_root,
                rank=rank,
                process_workers=True,
            )
        except BaseException:
            restore_v6()
            raise

        def restore() -> None:
            restore_v2()
            restore_v6()

        return restore

    v6._install_primary_scope_wrappers = combined_install
    return int(v6.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())

