# Repository audit summary

This package was reviewed and cleaned for public release.

## What was fixed

- Removed private local path references from manifests and configuration files.
- Kept the public entry point as `main.py`.
- Added `.gitattributes` to reduce noisy LF/CRLF churn on Windows.
- Added `requirements-optional.txt` for non-core features such as W&B, MLflow, PennyLane, and Zarr caches.
- Added `scripts/sanity_check.py` to validate repository-local configuration and manifest hygiene before pushing changes.
- Improved `scripts/prepare_unified_mpox_dataset.py` so it can write portable manifests and uses OS-agnostic source-prefix detection.
- Improved the import-time error message for incompatible `torch` / `torchvision` installations.
- Added `pyproject.toml` with formatter/linter defaults for easier maintenance.
- Added `run_benchmark_windows.cmd` as a Windows CMD launcher template.

## What was checked

- Python sources compile successfully with `py_compile`.
- The repo-local sanity check passes.
- Example manifests contain portable entries rather than absolute machine-specific paths.
- The example config references repository-local manifests.

## What was not changed

- The core training logic and reported benchmark settings were not altered.
- The checked-in result summaries were preserved.
