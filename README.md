# Mpox Leakage-Aware Benchmark Code

This repository contains the **code-only** release for a leakage-aware mpox skin-lesion classification benchmark.

## Included

- `main.py` — main training, optimization, calibration, thresholding, and ensemble pipeline.
- `scripts/prepare_unified_mpox_dataset.py` — dataset unification and curation script.
- `configs/run_config.json` — generic example configuration for the reported benchmark settings.
- `manifests/` — anonymized example manifests for original and augmented Monkeypox / non-Monkeypox images.
- `results/` — fold-wise result summaries and aggregate statistics.

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
│   ├── summary_mean_std.csv
│   ├── summary_statistics.json
│   └── fold_*_results.csv
├── scripts/
│   └── prepare_unified_mpox_dataset.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup

### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Expected data layout

The repository does **not** include the public image datasets. The default example configuration assumes this generic layout:

```text
data/
├── augmented_images/
│   ├── Monkeypox_augmented/
│   └── Others_augmented/
└── original_images/
    ├── Monkey Pox/
    └── Others/
```

If your local dataset lives elsewhere, replace the generic paths in the command below with your own directories.

## Reproducing the benchmark on Windows CMD

```cmd
python main.py ^
  --data_root "C:\path\to\data\augmented_images" ^
  --pos_class "Monkeypox_augmented" ^
  --neg_class "Others_augmented" ^
  --pos_list_file "manifests\Unified_Monkey_Aug.txt" ^
  --neg_list_file "manifests\Unified_NonMonkey_Aug.txt" ^
  --orig_root "C:\path\to\data\original_images" ^
  --orig_pos_class "Monkey Pox" ^
  --orig_neg_class "Others" ^
  --orig_pos_list_file "manifests\Unified_Monkey_Original.txt" ^
  --orig_neg_list_file "manifests\Unified_NonMonkey_Original.txt" ^
  --eval_on_original_only ^
  --split_strategy group_stratified ^
  --group_mode regex ^
  --group_regex "^(?:v[0-9]+__)?(.+?)(?:_(?:ORIGINAL|[0-9]+))?$" ^
  --cv_folds 5 ^
  --cv_val_frac 0.20 ^
  --models "efficientnetv2_s,convnext_tiny,vit_b_16,resnet50,densenet121,efficientnet_b0,mobilenet_v3_large" ^
  --img_size 224 ^
  --pretrained ^
  --amp ^
  --tta 16 ^
  --use_weighted_sampler ^
  --train_mix original+aug ^
  --train_orig_ratio 0.7 ^
  --max_aug_per_group 3 ^
  --finetune_on_original ^
  --finetune_epochs 5 ^
  --finetune_lr 1e-5 ^
  --freeze_backbone_epochs 0 ^
  --moo_enable ^
  --moo_objectives "auc,f1_at_precision,logloss" ^
  --moo_directions "maximize,maximize,minimize" ^
  --moo_selection_strategy f1_at_precision_first ^
  --moo_precision_target 0.80 ^
  --moo_constraint_auc_target 0.80 ^
  --moo_constraint_f1_target 0.80 ^
  --moo_eval_epochs 6 ^
  --moo_population_size 8 ^
  --moo_generations 3 ^
  --moo_top_k 3 ^
  --calibrate temperature ^
  --weighted_ensemble ^
  --ensemble_weight_metric f1_at_precision ^
  --ensemble_weight_opt random_search ^
  --ensemble_weight_trials 4096 ^
  --ensemble_weight_seed 42 ^
  --early_stop_patience 4 ^
  --early_stop_metric f1_at_precision ^
  --threshold_strategy f1_at_precision ^
  --precision_target 0.80 ^
  --moo_plot ^
  --rag_enable ^
  --rag_model auto ^
  --rag_index_source train_orig_only ^
  --rag_k 25 ^
  --rag_alpha 0.25 ^
  --rag_metric cosine ^
  --rag_weighted ^
  --rag_temp 0.07 ^
  --rag_chunk 512 ^
  --rag_tta 0 ^
  --plot_dpi 600
```

## Dataset preparation

```cmd
python scripts\prepare_unified_mpox_dataset.py --help
```

## Reported benchmark characteristics

- grouped split strategy: `group_stratified`
- cross-validation folds: 5
- validation fraction within fold: 0.20
- image size: 224
- batch size: 16
- main epochs: 30
- optimization epochs per Optuna/NSGA-II trial: 6
- fine-tuning on original images: 5 epochs at `1e-5`
- threshold strategy: `f1_at_precision`
- precision target: 0.80
- test-time augmentation: 16
- Optuna/NSGA-II objectives: AUC, F1@precision, log loss
- ensemble weight optimization: random search with 4096 trials

## Notes

- The manifests are anonymized to use generic paths and are intended as examples.
- The public datasets themselves are not redistributed in this repository.
- GPU execution is strongly recommended for the full benchmark.
