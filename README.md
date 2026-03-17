# Mpox Leakage-Aware Benchmark Code

This repository packages the **Python code and reproducibility assets only** for the mpox skin-lesion classification benchmark.

## Included

- `scripts/mpox_mooB_optimized_v7_1.py` — main training, optimization, evaluation, calibration, thresholding, and ensemble pipeline.
- `scripts/prepare_unified_mpox_dataset.py` — dataset-unification and curation script.
- `configs/run_config.json` — executed run configuration.
- `manifests/` — text manifests for original and augmented Monkeypox / non-Monkeypox images.
- `results/` — command summary and result summaries from the reported run.

## What this repository does

The main pipeline implements:

- leakage-aware grouped splitting
- 5-fold stratified group cross-validation
- original-only evaluation
- original+augmented training with augmentation controls
- Optuna / NSGA-II multi-objective optimization
- temperature scaling calibration
- threshold selection using `f1_at_precision`
- weighted ensembling and optional RAG-enhanced ensembling

## Repository structure

```text
.
├── configs/
│   └── run_config.json
├── manifests/
│   ├── Unified_Monkey_Aug.txt
│   ├── Unified_Monkey_Original.txt
│   ├── Unified_NonMonkey_Aug.txt
│   └── Unified_NonMonkey_Original.txt
├── results/
│   ├── cmdSummary_2026-03-15.txt
│   ├── summary_mean_std.csv
│   ├── summary_statistics.json
│   └── fold_*_results.csv
├── scripts/
│   ├── mpox_mooB_optimized_v7_1.py
│   └── prepare_unified_mpox_dataset.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset preparation

Prepare the unified dataset from MSLD v1.0 and MSLD v2.0 using the curation script:

```bash
python scripts/prepare_unified_mpox_dataset.py --help
```

The manifests in `manifests/` correspond to the curated original and augmented image lists used for the benchmark.

## Reproducing the benchmark

A representative run uses the provided configuration file:

```bash
python scripts/mpox_mooB_optimized_v7_1.py --config configs/run_config.json
```

## Reported benchmark characteristics

From the bundled run configuration and command summary:

- original evaluation pool: 876 images
- grouped cross-validation folds: 5
- image size: 224
- batch size: 16
- main epochs: 30
- hyperparameter-evaluation epochs: 6
- original-only fine-tuning epochs: 5
- threshold strategy: `f1_at_precision`
- precision target: 0.80
- test-time augmentation: 16
- Optuna / NSGA-II objectives: AUC, F1@precision, log loss
- ensemble weight optimization: random search with 4096 trials

## Notes

- This repository intentionally excludes the manuscript files.
- The public datasets themselves are not redistributed here.
- `timm` is required for timm-backed backbones.
- GPU execution is recommended for the full benchmark.

## Suggested GitHub repo name

`mpox-leakage-aware-benchmark-code`
