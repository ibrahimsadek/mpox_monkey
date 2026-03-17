"""Repository sanity checks for local development.

This script performs lightweight checks that do not require the image datasets:
- confirms required repository files exist
- confirms manifests do not contain absolute/private paths
- validates that the example config points to repository-local manifests
- verifies Python source files compile
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import py_compile

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / "main.py",
    ROOT / "configs" / "run_config.json",
    ROOT / "manifests" / "Unified_Monkey_Aug.txt",
    ROOT / "manifests" / "Unified_Monkey_Original.txt",
    ROOT / "manifests" / "Unified_NonMonkey_Aug.txt",
    ROOT / "manifests" / "Unified_NonMonkey_Original.txt",
    ROOT / "scripts" / "prepare_unified_mpox_dataset.py",
]


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def check_required_files() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.exists()]
    if missing:
        fail(f"Missing required files: {missing}")
    ok("required repository files exist")


def check_manifests() -> None:
    bad_lines = []
    for manifest in (ROOT / "manifests").glob("*.txt"):
        for idx, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
            value = line.strip()
            if not value:
                continue
            if value.startswith(("/", "\\")) or ":\\" in value:
                bad_lines.append(f"{manifest.name}:{idx}:{value}")
    if bad_lines:
        fail("Manifest files contain absolute/private paths: " + "; ".join(bad_lines[:5]))
    ok("manifest files are portable and anonymized")


def check_config() -> None:
    cfg = json.loads((ROOT / "configs" / "run_config.json").read_text(encoding="utf-8"))
    for key in ["pos_list_file", "neg_list_file", "orig_pos_list_file", "orig_neg_list_file"]:
        value = Path(cfg[key])
        target = ROOT / value
        if not target.exists():
            fail(f"Config path '{key}={value}' does not exist in the repository")
    ok("run_config.json references repository-local manifests")


def check_python_compile() -> None:
    for path in [ROOT / "main.py", ROOT / "scripts" / "prepare_unified_mpox_dataset.py", ROOT / "scripts" / "sanity_check.py"]:
        py_compile.compile(str(path), doraise=True)
    ok("Python entry points compile successfully")


def main() -> None:
    check_required_files()
    check_manifests()
    check_config()
    check_python_compile()
    print("[OK] repository sanity checks completed")


if __name__ == "__main__":
    main()
