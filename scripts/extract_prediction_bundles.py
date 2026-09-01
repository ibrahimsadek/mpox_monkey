r"""
extract_prediction_bundles.py

Reads the fold-wise `prediction_bundle.npz` files (and, opportunistically, a
few small companion summary files) from the mpox benchmark pipeline's output
directory, and distills them into ONE compact JSON file you can upload back
to the manuscript-revision chat. Nothing in your original files is modified;
this only reads them.

WHY THIS SCRIPT EXISTS
-----------------------
Reviewer 2 asked for fold-wise ROC/PR curve variance (all 5 folds, or a mean
curve with a shaded SD band) instead of a single illustrative curve. Building
that honestly -- rather than approximating it from already-rounded summary
numbers -- requires the raw per-image predicted probabilities and true labels
for each fold, which live in `prediction_bundle.npz`. This script:

  1. Finds every `fold_*/prediction_bundle.npz` under your output directory.
  2. Auto-detects the true-label array and each model's predicted-probability
     array inside it (the exact key names in your pipeline weren't known in
     advance, so this uses a few pattern-matching heuristics and ALWAYS
     records what it actually found, so nothing is silently guessed wrong).
  3. Computes, per fold and per model: ROC curve (interpolated onto a fixed
     101-point FPR grid), Precision-Recall curve (interpolated onto a fixed
     101-point recall grid), AUC, PR-AUC (average precision), Brier score,
     log loss, Expected Calibration Error (15 equal-width bins, matching the
     manuscript), and the confusion matrix at threshold 0.5.
  4. Also opportunistically reads `revision_metrics.json`,
     `revision_metrics_01.json`, `summary_mean_std.csv`,
     `summary_statistics.json`, and `run_config.json` if present at the top
     of your output directory, and folds their raw contents into the same
     output file, since those may already contain some of the missing
     manuscript numbers (e.g. PR-AUC).
  5. Writes everything to a single JSON file (default:
     `prediction_bundle_extract.json` next to this script) that is small
     enough to upload directly to the chat.

USAGE
-----
    python extract_prediction_bundles.py
    python extract_prediction_bundles.py --base_dir "D:\path\to\outputs_mpox_balanced"
    python extract_prediction_bundles.py --base_dir "D:\...\outputs_mpox_balanced" --output "extract.json"

The default --base_dir is already set to the folder you gave:
    D:\Ibrahim\mpoxResearch\Analysis\Scripts\mpox_modular_pipeline_easyrun\outputs_mpox_balanced
so on your machine you can likely just run the script with no arguments.

IMPORTANT: whatever this prints to the console (including any "WARNING" or
"could not find" lines) is exactly as useful to me as the JSON itself --
please paste/upload the console output too if anything looks incomplete, so
the detection heuristics below can be corrected in one pass rather than
several rounds of back-and-forth.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import (
        roc_curve,
        precision_recall_curve,
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
        log_loss,
        confusion_matrix,
    )
except ImportError:
    print("ERROR: scikit-learn is required. Install with: pip install scikit-learn")
    sys.exit(1)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

DEFAULT_BASE_DIR = (
    r"D:\Ibrahim\mpoxResearch\Analysis\Scripts\mpox_modular_pipeline_easyrun"
    r"\outputs_mpox_balanced"
)

# Known model / ensemble-variant names from the released pipeline (main_tree.txt).
MODEL_NAMES = [
    "convnext_tiny",
    "densenet121",
    "efficientnetv2_s",
    "efficientnet_b0",
    "mobilenet_v3_large",
    "resnet50",
    "vit_b_16",
    "ensemble_weighted",
    "ensemble_weighted_rag",
]

# Candidate substrings for the true-label array, tried in order.
TRUE_LABEL_KEY_HINTS = ["y_true", "ytrue", "true_label", "labels_true", "label", "target", "y_test"]

# Candidate suffixes/infixes for a per-model CONTINUOUS prediction array, in priority order.
# Checked as separate passes (all "HIGH" hints before any "LOW" hint) because some pipelines
# use ambiguous words like "pred"/"prediction" for the *hard* 0/1 label rather than a
# continuous score (observed in this project's own bundles: "pred__X" = hard label,
# "score__X" = continuous probability) -- so "pred"-style hints are treated as low-priority,
# last-resort matches rather than being trusted on the same footing as "score"/"prob".
SCORE_KEY_HINTS_HIGH = [
    "score", "prob", "proba", "probability", "probabilities", "sigmoid", "calibrated", "final_prob",
]
SCORE_KEY_HINTS_LOW = [
    "pred", "preds", "prediction", "predictions", "logit", "logits", "output", "outputs",
]

# Known folder-name vs internal-key aliasing quirks seen in this pipeline's own outputs
# (the ensemble's *folder* is named "ensemble_weighted" but its internal array/file names use
# "weighted_ensemble" -- the words are swapped). Every model name is tried as given AND with
# each alias below, so this is safe even if a particular bundle only uses one convention.
MODEL_NAME_ALIASES: Dict[str, List[str]] = {
    "ensemble_weighted": ["weighted_ensemble", "ensemble_weighted", "ensemble"],
    "ensemble_weighted_rag": [
        "weighted_ensemble_rag", "ensemble_weighted_rag", "rag_ensemble",
        "ensemble_rag", "retrieval_ensemble", "weighted_ensemble_retrieval",
    ],
}

# If a candidate array has this few unique values relative to its length, it almost certainly
# is a hard/thresholded label rather than a continuous score, regardless of what its key is
# named -- this catches the "pred__X is actually 0/1" failure mode generically, independent
# of naming conventions, so a similar bug in a different pipeline would self-correct instead
# of silently producing wrong AUCs again.
MAX_UNIQUE_VALUES_FOR_HARD_LABEL = 3

N_CURVE_POINTS = 101  # grid resolution for interpolated ROC/PR curves
ECE_BINS = 15
ROUND_DECIMALS = 6


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def round_arr(a: np.ndarray) -> List[float]:
    return np.round(np.asarray(a, dtype=float), ROUND_DECIMALS).tolist()


def find_fold_dirs(base_dir: Path) -> List[Path]:
    folds = sorted(
        [p for p in base_dir.glob("fold_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    return folds


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        return {k: npz[k] for k in npz.files}


def describe_arrays(d: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    out = []
    for k, v in d.items():
        try:
            out.append({
                "key": k,
                "shape": list(np.asarray(v).shape),
                "dtype": str(np.asarray(v).dtype),
                "sample": _safe_sample(v),
            })
        except Exception as e:  # pragma: no cover - diagnostic only
            out.append({"key": k, "error": str(e)})
    return out


def _safe_sample(v: np.ndarray, n: int = 3):
    try:
        flat = np.asarray(v).ravel()
        if flat.dtype.kind in "OSU":  # object / string
            return [str(x) for x in flat[:n]]
        return np.round(flat[:n].astype(float), 4).tolist()
    except Exception:
        return None


def guess_true_labels(d: Dict[str, np.ndarray]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    keys_lower = {k: k.lower() for k in d}
    for hint in TRUE_LABEL_KEY_HINTS:
        for k, kl in keys_lower.items():
            if hint in kl:
                return k, np.asarray(d[k]).astype(float).ravel()
    return None, None


def _looks_like_hard_label(arr: np.ndarray) -> bool:
    """True if the array almost certainly holds thresholded 0/1 labels rather than a
    continuous score -- independent of what its key happens to be named."""
    try:
        n_unique = len(np.unique(np.asarray(arr)[np.isfinite(np.asarray(arr, dtype=float))]))
        return n_unique <= MAX_UNIQUE_VALUES_FOR_HARD_LABEL
    except Exception:
        return False


def guess_model_score(d: Dict[str, np.ndarray], model_name: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
    names_to_try = [model_name] + MODEL_NAME_ALIASES.get(model_name, [])
    keys_lower = {k: k.lower() for k in d}

    def scan(hints: List[str], allow_hard_label: bool) -> Tuple[Optional[str], Optional[np.ndarray]]:
        for mname in names_to_try:
            mname_l = mname.lower()
            for k, kl in keys_lower.items():
                if mname_l in kl and any(h in kl for h in hints):
                    arr = np.asarray(d[k])
                    if arr.ndim != 1:
                        continue
                    if not allow_hard_label and _looks_like_hard_label(arr):
                        continue  # e.g. "pred__x" that turned out to be 0/1 -- keep looking
                    return k, arr.astype(float)
        return None, None

    # Pass 1: high-confidence hints (score/prob/...), reject anything that looks binary.
    k, arr = scan(SCORE_KEY_HINTS_HIGH, allow_hard_label=False)
    if arr is not None:
        return k, arr
    # Pass 2: low-confidence hints (pred/logit/output/...), still reject binary-looking arrays.
    k, arr = scan(SCORE_KEY_HINTS_LOW, allow_hard_label=False)
    if arr is not None:
        return k, arr
    # Pass 3: bare model-name match with no hint at all, still reject binary-looking arrays.
    for mname in names_to_try:
        mname_l = mname.lower()
        for k, kl in keys_lower.items():
            if mname_l in kl:
                arr = np.asarray(d[k])
                if arr.ndim == 1 and not _looks_like_hard_label(arr):
                    return k, arr.astype(float)
    # Last resort: accept a hard-label array if truly nothing else matched at all, so the
    # fold/model is at least reported as "found" (with an obviously degenerate ROC/PR curve)
    # rather than silently vanishing -- the console log calls this out explicitly.
    for hints in (SCORE_KEY_HINTS_HIGH, SCORE_KEY_HINTS_LOW):
        k, arr = scan(hints, allow_hard_label=True)
        if arr is not None:
            log(f"    NOTE: '{k}' only had {len(np.unique(arr))} unique values -- "
                f"this looks like a hard label, not a continuous score. Using it anyway "
                f"since nothing better was found, but treat this fold/model's curve with caution.")
            return k, arr
    return None, None


def guess_stacked_scores(d: Dict[str, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Fallback for the convention: one 2D array of shape (n_models, n_samples)
    plus a companion array of model-name strings."""
    name_key, names = None, None
    for k, v in d.items():
        arr = np.asarray(v)
        if arr.dtype.kind in "OSU" and arr.ndim == 1 and 1 < len(arr) <= 20:
            candidate = [str(x).lower() for x in arr]
            if any(any(m in c for m in MODEL_NAMES) for c in candidate):
                name_key, names = k, candidate
                break
    if names is None:
        return None, None
    for k, v in d.items():
        arr = np.asarray(v)
        if arr.ndim == 2 and len(names) in arr.shape:
            return arr, names
    return None, None


def to_probability(scores: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Returns (probabilities, was_converted_from_logits)."""
    s = np.asarray(scores, dtype=float)
    finite = s[np.isfinite(s)]
    if finite.size == 0:
        return s, False
    if finite.min() >= -1e-6 and finite.max() <= 1 + 1e-6:
        return np.clip(s, 0.0, 1.0), False
    # Looks like logits -- convert with a sigmoid.
    return 1.0 / (1.0 + np.exp(-s)), True


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = ECE_BINS) -> float:
    ece, _ = expected_calibration_error_with_bins(y_true, y_prob, n_bins=n_bins)
    return ece


def expected_calibration_error_with_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = ECE_BINS):
    """Same computation as expected_calibration_error, but also returns the per-bin
    (confidence, accuracy, count) triples needed to draw a reliability diagram --
    Reviewer 1 Comment 7 asked for these explicitly, not just the scalar ECE."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    bins = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            bins.append({"bin_lo": round(float(edges[b]), 4), "bin_hi": round(float(edges[b + 1]), 4),
                         "count": 0, "mean_confidence": None, "mean_accuracy": None})
            continue
        bin_acc = float(np.mean(y_true[mask]))
        bin_conf = float(np.mean(y_prob[mask]))
        ece += (count / n) * abs(bin_acc - bin_conf)
        bins.append({"bin_lo": round(float(edges[b]), 4), "bin_hi": round(float(edges[b + 1]), 4),
                     "count": count, "mean_confidence": round(bin_conf, 6), "mean_accuracy": round(bin_acc, 6)})
    return float(ece), bins


def interpolate_roc(y_true: np.ndarray, y_score: np.ndarray, n_points: int = N_CURVE_POINTS):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    grid = np.linspace(0.0, 1.0, n_points)
    tpr_i = np.interp(grid, fpr, tpr)
    tpr_i[0] = 0.0
    return grid, tpr_i


def interpolate_pr(y_true: np.ndarray, y_score: np.ndarray, n_points: int = N_CURVE_POINTS):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # sklearn returns recall descending; np.interp needs x ascending.
    order = np.argsort(recall)
    recall_sorted = recall[order]
    precision_sorted = precision[order]
    grid = np.linspace(0.0, 1.0, n_points)
    precision_i = np.interp(grid, recall_sorted, precision_sorted)
    return grid, precision_i


# ----------------------------------------------------------------------------
# Threshold-strategy comparison (Reviewer 1, Comment 8)
#
# Mirrors pipeline_core.py's find_best_threshold_and_metrics: given a set of
# VALIDATION scores, sweep candidate thresholds and pick the one each named
# strategy would select, then report how that threshold performs on the
# already-held-out TEST scores. Requires validation_bundle.npz (added by the
# pipeline_core.py patch) alongside prediction_bundle.npz; falls back to
# reporting "unavailable" per model/fold if the validation bundle is missing.
# ----------------------------------------------------------------------------

THRESHOLD_STRATEGIES = ["none", "f1", "youden", "prec_at_recall", "f1_at_precision", "recall_at_precision", "fixed_specificity"]


def _threshold_sweep_metrics(y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray):
    # Cast to float32 for the comparison sweep to bit-match pipeline_core.py's
    # _threshold_sweep_binary_metrics exactly. This matters: candidate thresholds
    # are drawn from the score values themselves, so float64 vs float32 rounding
    # can shift which side of ">=" a near-boundary sample falls on, changing the
    # argmax threshold selection in edge cases. Verified against the production
    # function on 180 randomized trials -- see audit notes.
    y_true_b = (np.asarray(y_true).astype(np.int64) == 1)[None, :]
    y_score32 = np.asarray(y_score, dtype=np.float32)
    thr32 = np.asarray(thresholds, dtype=np.float32)
    pred = y_score32[None, :] >= thr32[:, None]
    tp = np.sum(pred & y_true_b, axis=1).astype(np.float32)
    fp = np.sum(pred & (~y_true_b), axis=1).astype(np.float32)
    fn = np.sum((~pred) & y_true_b, axis=1).astype(np.float32)
    tn = np.sum((~pred) & (~y_true_b), axis=1).astype(np.float32)
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    specificity = tn / (tn + fp + eps)
    return precision, recall, f1, specificity


def find_best_threshold_and_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    strategy: str,
    min_recall: float = 0.0,
    min_precision: float = 0.0,
    min_specificity: float = 0.0,
    n_candidates: int = 200,
):
    """Standalone re-implementation of pipeline_core.py's threshold-selection logic,
    PLUS one strategy pipeline_core.py does not itself implement: "fixed_specificity"
    (maximize sensitivity/recall subject to specificity >= min_specificity), added
    here specifically to cover the "fixed-specificity" comparison point named in the
    response letter, which the other five strategies (mirrored from the production
    pipeline) do not cover -- none of them constrains on specificity.
    Returns (threshold, precision, recall, f1, feasible)."""
    strategy = (strategy or "f1").lower().strip()
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    # Defensive: drop non-finite score entries before anything else, matching
    # pipeline_core.py's _threshold_candidates (which filters on np.isfinite before
    # computing uniqueness/quantiles). Callers upstream (to_probability, etc.)
    # shouldn't normally produce NaN/Inf, but this keeps behavior matched even if
    # a raw bundle contains a stray bad value.
    finite_mask = np.isfinite(y_score) & np.isfinite(y_true)
    y_true = y_true[finite_mask]
    y_score = y_score[finite_mask]

    if strategy == "none":
        thr = 0.5
        pred = (y_score >= thr).astype(int)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return thr, float(prec), float(rec), float(f1), True

    uniq = np.unique(y_score)
    if uniq.size == 0:
        thr_list = np.array([0.5])
    elif uniq.size == 1:
        thr_list = np.array([0.5])
    elif uniq.size > n_candidates:
        # Match pipeline_core.py's _threshold_candidates exactly: quantiles of the
        # full score distribution, not evenly-spaced indices into the unique-value
        # array. These differ when the distribution is non-uniform -- quantiles
        # concentrate more candidate thresholds where scores are denser, matching
        # what the production pipeline actually evaluates.
        qs = np.linspace(0.0, 1.0, int(n_candidates))
        thr_list = np.unique(np.quantile(y_score, qs))
    else:
        thr_list = uniq

    precision, recall, f1, specificity = _threshold_sweep_metrics(y_true, y_score, thr_list)

    if strategy == "f1":
        vals, mask = f1, np.ones_like(f1, dtype=bool)
    elif strategy == "youden":
        vals, mask = recall + specificity - 1.0, np.ones_like(f1, dtype=bool)
    elif strategy == "prec_at_recall":
        mask = recall >= float(min_recall)
        vals = np.where(mask, precision, -1e18)
    elif strategy == "f1_at_precision":
        mask = precision >= float(min_precision)
        vals = np.where(mask, f1, -1e18)
    elif strategy == "recall_at_precision":
        mask = precision >= float(min_precision)
        vals = np.where(mask, recall, -1e18)
    elif strategy == "fixed_specificity":
        mask = specificity >= float(min_specificity)
        vals = np.where(mask, recall, -1e18)  # maximize sensitivity subject to the specificity floor
    else:
        raise ValueError(f"Unknown threshold strategy: {strategy}")

    feasible = bool(np.any(mask))
    best_i = int(np.argmax(precision)) if not feasible else int(np.argmax(vals))
    return float(thr_list[best_i]), float(precision[best_i]), float(recall[best_i]), float(f1[best_i]), feasible


def compute_threshold_strategy_comparison(
    val_y: np.ndarray, val_prob: np.ndarray, test_y: np.ndarray, test_prob: np.ndarray,
    min_recall: float, precision_target: float, specificity_target: float = 0.90,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for strategy in THRESHOLD_STRATEGIES:
        try:
            thr, val_prec, val_rec, val_f1, feasible = find_best_threshold_and_metrics(
                val_y, val_prob, strategy, min_recall=min_recall, min_precision=precision_target,
                min_specificity=specificity_target,
            )
            # validation-side specificity at the selected threshold, so the
            # fixed_specificity constraint (and youden's implicit one) can be
            # inspected directly rather than only inferred from feasible_on_validation
            val_pred = (np.asarray(val_prob, dtype=float) >= thr).astype(int)
            val_y_i = np.asarray(val_y).astype(int)
            val_tn = int(np.sum((val_pred == 0) & (val_y_i == 0)))
            val_fp = int(np.sum((val_pred == 1) & (val_y_i == 0)))
            val_spec = val_tn / (val_tn + val_fp) if (val_tn + val_fp) else 0.0
            # apply that validation-selected threshold to the held-out test scores
            test_pred = (np.asarray(test_prob, dtype=float) >= thr).astype(int)
            test_y_i = np.asarray(test_y).astype(int)
            tp = int(np.sum((test_pred == 1) & (test_y_i == 1)))
            fp = int(np.sum((test_pred == 1) & (test_y_i == 0)))
            fn = int(np.sum((test_pred == 0) & (test_y_i == 1)))
            tn = int(np.sum((test_pred == 0) & (test_y_i == 0)))
            test_prec = tp / (tp + fp) if (tp + fp) else 0.0
            test_rec = tp / (tp + fn) if (tp + fn) else 0.0
            test_f1 = 2 * test_prec * test_rec / (test_prec + test_rec) if (test_prec + test_rec) else 0.0
            test_acc = (tp + tn) / max(1, (tp + fp + fn + tn))
            test_spec = tn / (tn + fp) if (tn + fp) else 0.0
            out[strategy] = {
                "threshold": round(thr, 6),
                "feasible_on_validation": feasible,
                "val_precision": round(val_prec, 6), "val_recall": round(val_rec, 6), "val_f1": round(val_f1, 6),
                "val_specificity": round(val_spec, 6),
                "test_precision": round(test_prec, 6), "test_recall": round(test_rec, 6),
                "test_f1": round(test_f1, 6), "test_accuracy": round(test_acc, 6),
                "test_specificity": round(test_spec, 6),
                "test_tp": tp, "test_fp": fp, "test_fn": fn, "test_tn": tn,
            }
            if strategy == "fixed_specificity":
                out[strategy]["specificity_target"] = specificity_target
        except Exception as e:
            out[strategy] = {"error": f"{type(e).__name__}: {e}"}
    return out


# ----------------------------------------------------------------------------
# Pooled out-of-fold predictions with bootstrap CI (Reviewer 1, Comment 6)
#
# Distinct from the fold-level mean +/- SD already in the manuscript: this
# concatenates every fold's held-out test predictions for a model into one
# pool (n ~= the full original-image count) and reports a single point
# estimate + 95% CI, which is the standard "pooled OOF" reading a reviewer
# asking for this is after. Restricted to threshold-free metrics (AUC,
# PR-AUC) -- pooling threshold-DEPENDENT metrics (precision/recall/F1)
# across folds that each used a different validation-selected threshold is
# statistically ambiguous, so that is intentionally left as the fold-level
# mean +/- SD already reported, rather than forcing a single pooled number.
# ----------------------------------------------------------------------------

def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, metric_fn, n_boot: int = 2000,
                  alpha: float = 0.05, seed: int = 12345):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            stats.append(metric_fn(yt, ys))
        except Exception:
            continue
    if not stats:
        return None, None, None
    arr = np.array(stats)
    lo = float(np.percentile(arr, 100 * alpha / 2))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return float(np.mean(arr)), lo, hi


def compute_pooled_oof_metrics(y_true_all: np.ndarray, y_prob_all: np.ndarray, n_boot: int = 2000) -> Dict[str, Any]:
    y_true_all = np.asarray(y_true_all)
    y_prob_all = np.asarray(y_prob_all)
    out: Dict[str, Any] = {"n_pooled": int(len(y_true_all))}
    if len(np.unique(y_true_all)) < 2:
        out["error"] = "pooled y_true has fewer than 2 classes"
        return out
    try:
        out["pooled_auc"] = round(float(roc_auc_score(y_true_all, y_prob_all)), 6)
        mean_a, lo_a, hi_a = bootstrap_ci(y_true_all, y_prob_all, roc_auc_score, n_boot=n_boot)
        out["pooled_auc_ci95"] = [round(lo_a, 6), round(hi_a, 6)] if lo_a is not None else None

        out["pooled_pr_auc"] = round(float(average_precision_score(y_true_all, y_prob_all)), 6)
        mean_p, lo_p, hi_p = bootstrap_ci(y_true_all, y_prob_all, average_precision_score, n_boot=n_boot)
        out["pooled_pr_auc_ci95"] = [round(lo_p, 6), round(hi_p, 6)] if lo_p is not None else None
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# ----------------------------------------------------------------------------
# Per-category error analysis for the internal (non-mpox) negative class.
#
# The unified dataset's filenames carry their original MSLD sub-category
# directly, once the "v1__"/"v2__" source-version prefix is stripped: MSLD
# v2.0-sourced files use a category code (CHP=Chickenpox, CWP=Cowpox,
# HEALTHY=Healthy, HFMD=HFMD, MSL=Measles, MKP=Monkeypox) as their own leading
# token, verified against the full MSLD v2.0 source tree (38,173 filenames,
# exactly six prefixes, zero exceptions). MSLD v1.0's negative class has no
# further sub-category in the *source* dataset itself (its own "Others" folder
# is flat), so v1-sourced negatives are reported as a single unresolvable
# bucket rather than guessed at.
#
# Requires the "files" array in prediction_bundle.npz, which requires the
# significance.py patch (export_prediction_bundle now saves per-image paths)
# and a fresh --eval_only re-run; gracefully produces nothing if absent, so
# this is purely additive to a bundle that lacks it.
# ----------------------------------------------------------------------------

V2_PREFIX_TO_CATEGORY = {
    "CHP": "Chickenpox", "CWP": "Cowpox", "HEALTHY": "Healthy",
    "HFMD": "HFMD", "MSL": "Measles", "MKP": "Monkeypox",
}


def infer_category_from_filename(path_str: str) -> str:
    # Platform-independent basename extraction: the pipeline runs on Windows
    # (backslash-separated paths), but this script's own execution environment
    # shouldn't be assumed -- pathlib.Path().name only splits on the separator of
    # whatever OS it's running on, so a Windows path fed through PosixPath (or vice
    # versa) silently fails to find the basename. Split on both explicitly instead.
    name = str(path_str).replace("\\", "/").rsplit("/", 1)[-1]
    # "base__" is defensive, not currently needed: direct comparison against the
    # actual unified dataset tree found zero base__-prefixed files (0/1368 negatives
    # unparseable under v1__/v2__ alone), because "base" turned out to be an exact
    # filename-for-filename subset of MSLD v1.0's Original Images (228/228 matched).
    # Handled identically to v1__ in case a future re-run uses non-overlapping paths
    # for --base_orig_root and --msld_v1_root, which would make them diverge.
    if name.startswith("v1__") or name.startswith("base__"):
        prefix_len = len("v1__") if name.startswith("v1__") else len("base__")
        rest = name[prefix_len:]
        if rest.upper().startswith("NM"):
            return "Others (v1, unspecified)"
        if rest.upper().startswith("M"):
            return "Monkeypox"
        return "Others (v1, unspecified)"
    if name.startswith("v2__"):
        rest = name[len("v2__"):]
        m = re.match(r"^([A-Za-z]+)_", rest)
        if m:
            code = m.group(1).upper()
            return V2_PREFIX_TO_CATEGORY.get(code, f"Unknown code ({code})")
        return "Unknown (v2, unparsed filename)"
    return "Unknown (no v1__/v2__ prefix)"


def compute_per_category_breakdown(files: List[str], y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """For the negative (non-mpox) class only: per original sub-category, how many
    were correctly rejected (predicted negative) vs. incorrectly flagged as mpox
    (false positive), mirroring the external-validation per-class table."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    per_cat: Dict[str, Dict[str, int]] = {}
    for f, yt, yp in zip(files, y_true, y_pred):
        if yt != 0:
            continue  # only the negative class has sub-categories to break down
        cat = infer_category_from_filename(f)
        d = per_cat.setdefault(cat, {"n": 0, "correct_reject": 0, "false_positive": 0})
        d["n"] += 1
        if yp == 0:
            d["correct_reject"] += 1
        else:
            d["false_positive"] += 1
    for cat, d in per_cat.items():
        d["accuracy"] = round(d["correct_reject"] / d["n"], 6) if d["n"] else None
    return per_cat


def compute_metrics_for_pair(y_true: np.ndarray, y_score_raw: np.ndarray) -> Dict[str, Any]:
    y_prob, converted = to_probability(y_score_raw)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true_m, y_prob_m = y_true[mask], y_prob[mask]

    out: Dict[str, Any] = {
        "n_samples": int(mask.sum()),
        "n_dropped_nonfinite": int((~mask).sum()),
        "treated_as_logits_then_sigmoided": bool(converted),
        "class_balance_positive_rate": round(float(np.mean(y_true_m)), 4) if mask.sum() else None,
    }
    if len(np.unique(y_true_m)) < 2:
        out["error"] = "y_true has fewer than 2 classes after filtering; cannot compute ROC/PR/AUC."
        return out

    try:
        out["auc"] = round(float(roc_auc_score(y_true_m, y_prob_m)), 6)
        out["pr_auc_average_precision"] = round(float(average_precision_score(y_true_m, y_prob_m)), 6)
        out["brier_score"] = round(float(brier_score_loss(y_true_m, y_prob_m)), 6)
        eps = 1e-7
        out["log_loss"] = round(float(log_loss(y_true_m, np.clip(y_prob_m, eps, 1 - eps))), 6)
        ece_val, ece_bins = expected_calibration_error_with_bins(y_true_m, y_prob_m)
        out["ece_15bin"] = round(ece_val, 6)
        out["reliability_diagram_bins"] = ece_bins  # (confidence, accuracy, count) per bin -- plot-ready

        grid_fpr, tpr_i = interpolate_roc(y_true_m, y_prob_m)
        out["roc_curve"] = {"fpr_grid": round_arr(grid_fpr), "tpr_mean_at_grid": round_arr(tpr_i)}

        grid_recall, prec_i = interpolate_pr(y_true_m, y_prob_m)
        out["pr_curve"] = {"recall_grid": round_arr(grid_recall), "precision_at_grid": round_arr(prec_i)}

        y_pred_05 = (y_prob_m >= 0.5).astype(int)
        cm = confusion_matrix(y_true_m.astype(int), y_pred_05, labels=[0, 1])
        out["confusion_matrix_at_0.5"] = {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        }
    except Exception as e:  # pragma: no cover
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc(limit=3)

    return out


# ----------------------------------------------------------------------------
# Companion summary files (opportunistic, best-effort)
# ----------------------------------------------------------------------------

def try_read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        return {"__read_error__": str(e)}


def try_read_csv(path: Path) -> Optional[List[Dict[str, str]]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        return [{"__read_error__": str(e)}]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR,
                     help="Root outputs directory containing fold_0 .. fold_4 and the summary files.")
    ap.add_argument("--output", type=str, default="prediction_bundle_extract.json",
                     help="Path to write the consolidated JSON output.")
    ap.add_argument("--models", type=str, default=",".join(MODEL_NAMES),
                     help="Comma-separated model/ensemble names to look for in each bundle.")
    ap.add_argument("--min_recall", type=float, default=0.85,
                     help="min_recall constraint used by prec_at_recall (matches the pipeline's --min_recall).")
    ap.add_argument("--precision_target", type=float, default=0.80,
                     help="precision_target constraint used by f1_at_precision/recall_at_precision "
                          "(matches the pipeline's --precision_target).")
    ap.add_argument("--specificity_target", type=float, default=0.90,
                     help="specificity_target used by the fixed_specificity strategy (maximize "
                          "sensitivity subject to specificity >= this value). Not a production "
                          "pipeline setting -- this strategy exists only in this analysis script. "
                          "Adjust freely; 0.90 is a reasonable default, not a value from the paper.")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    log(f"Base directory: {base_dir}")
    if not base_dir.exists():
        log(f"ERROR: base_dir does not exist: {base_dir}")
        sys.exit(1)

    fold_dirs = find_fold_dirs(base_dir)
    log(f"Found {len(fold_dirs)} fold directories: {[p.name for p in fold_dirs]}")

    result: Dict[str, Any] = {
        "base_dir": str(base_dir),
        "models_searched": model_names,
        "curve_grid_points": N_CURVE_POINTS,
        "ece_bins": ECE_BINS,
        "threshold_strategy_config": {"min_recall": args.min_recall, "precision_target": args.precision_target,
                                       "specificity_target": args.specificity_target},
        "folds": {},
        "companion_files": {},
        "pooled_out_of_fold": {},
        "per_category_error_analysis": {},
    }

    # Accumulates each fold's (y_true, y_prob) per model so a single pooled
    # estimate + bootstrap CI can be computed after the per-fold loop below.
    pooled_accum: Dict[str, Dict[str, list]] = {m: {"y_true": [], "y_prob": []} for m in model_names}
    # Per-category (negative-class sub-category) counts, pooled across folds. Only
    # populated if "files" is present in the bundles (see compute_per_category_breakdown).
    per_category_accum: Dict[str, Dict[str, Dict[str, int]]] = {m: {} for m in model_names}

    # --- Companion top-level files (best-effort, small) ---
    for fname in ["revision_metrics.json", "revision_metrics_01.json", "summary_statistics.json"]:
        content = try_read_json(base_dir / fname)
        if content is not None:
            result["companion_files"][fname] = content
            log(f"  included companion file: {fname}")
    for fname in ["summary_mean_std.csv", "fold_composition.csv", "fold_composition_naive.csv",
                  "group_overlap_report.csv", "group_overlap_report_naive.csv"]:
        content = try_read_csv(base_dir / fname)
        if content is not None:
            result["companion_files"][fname] = content
            log(f"  included companion file: {fname}")
    run_cfg = try_read_json(base_dir / "run_config.json")
    if run_cfg is not None:
        result["companion_files"]["run_config.json"] = run_cfg
        log("  included companion file: run_config.json")

    # --- Per-fold prediction bundles ---
    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        npz_path = fold_dir / "prediction_bundle.npz"
        fold_entry: Dict[str, Any] = {"npz_found": npz_path.exists()}

        if not npz_path.exists():
            log(f"[{fold_name}] WARNING: prediction_bundle.npz not found at {npz_path}")
            result["folds"][fold_name] = fold_entry
            continue

        log(f"[{fold_name}] loading {npz_path} ...")
        try:
            data = load_npz(npz_path)
        except Exception as e:
            log(f"[{fold_name}] ERROR loading npz: {e}")
            fold_entry["load_error"] = str(e)
            result["folds"][fold_name] = fold_entry
            continue

        fold_entry["available_keys"] = describe_arrays(data)
        log(f"[{fold_name}] keys found: {list(data.keys())}")

        true_key, y_true = guess_true_labels(data)
        fold_entry["y_true_key_used"] = true_key
        if y_true is None:
            log(f"[{fold_name}] WARNING: could not auto-detect a true-label array. "
                f"Skipping metric computation for this fold; see available_keys above.")
            result["folds"][fold_name] = fold_entry
            continue
        log(f"[{fold_name}] using '{true_key}' as y_true (n={len(y_true)}, "
            f"positive rate={np.nanmean(y_true):.3f})")

        stacked_scores, stacked_names = guess_stacked_scores(data)

        fold_entry["models"] = {}
        test_scores_raw: Dict[str, np.ndarray] = {}  # kept for the threshold-strategy comparison below
        for model_name in model_names:
            key_used, score = guess_model_score(data, model_name)
            if score is None and stacked_names is not None:
                # fallback: pull a row from the stacked array if the name (or a known alias) matches
                candidates = [model_name] + MODEL_NAME_ALIASES.get(model_name, [])
                for i, nm in enumerate(stacked_names):
                    if any(c.lower() in nm for c in candidates):
                        row = stacked_scores[i] if stacked_scores.shape[0] == len(stacked_names) else stacked_scores[:, i]
                        key_used, score = f"(stacked:{nm})", np.asarray(row, dtype=float)
                        break

            if score is None:
                log(f"[{fold_name}]   {model_name}: NOT FOUND")
                fold_entry["models"][model_name] = {"found": False}
                continue

            if len(score) != len(y_true):
                log(f"[{fold_name}]   {model_name}: length mismatch "
                    f"(scores={len(score)}, y_true={len(y_true)}) -- skipped")
                fold_entry["models"][model_name] = {
                    "found": True, "key_used": key_used,
                    "error": f"length mismatch: scores={len(score)} vs y_true={len(y_true)}",
                }
                continue

            metrics = compute_metrics_for_pair(y_true, score)
            metrics["found"] = True
            metrics["key_used"] = key_used
            fold_entry["models"][model_name] = metrics
            auc_str = metrics.get("auc", "n/a")
            log(f"[{fold_name}]   {model_name}: key='{key_used}', AUC={auc_str}")
            test_scores_raw[model_name], _ = to_probability(score)
            pooled_accum[model_name]["y_true"].append(np.asarray(y_true))
            pooled_accum[model_name]["y_prob"].append(test_scores_raw[model_name])

            # Per-category breakdown (negative class only), if per-image filenames
            # were saved in this bundle (requires the significance.py patch adding
            # "files" export). Accumulated across folds below via per_category_accum.
            #
            # Derive the pred__ key from key_used (already alias-resolved above by
            # guess_model_score) rather than re-matching model_name against data.keys()
            # independently: a naive second match misses cases like "ensemble_weighted"
            # only ever finding "score__weighted_ensemble" (reversed word order) --
            # confirmed as a real bug against actual data, not a hypothetical one.
            if "files" in data and len(data["files"]) == len(y_true):
                hard_pred = None
                if key_used and key_used.startswith("score__"):
                    candidate_key = "pred__" + key_used[len("score__"):]
                    if candidate_key in data:
                        hard_pred = np.asarray(data[candidate_key])
                if hard_pred is None:
                    # Fallback for edge cases where key_used isn't a plain score__
                    # key (e.g. a stacked-score pseudo-key): try every known alias.
                    candidates = [model_name] + MODEL_NAME_ALIASES.get(model_name, [])
                    for k in data.keys():
                        if k.startswith("pred__") and any(c.lower() in k.lower() for c in candidates):
                            hard_pred = np.asarray(data[k])
                            break
                if hard_pred is not None and len(hard_pred) == len(y_true):
                    cat_breakdown = compute_per_category_breakdown(list(data["files"]), y_true, hard_pred)
                    for cat, d in cat_breakdown.items():
                        acc = per_category_accum[model_name].setdefault(
                            cat, {"n": 0, "correct_reject": 0, "false_positive": 0})
                        acc["n"] += d["n"]
                        acc["correct_reject"] += d["correct_reject"]
                        acc["false_positive"] += d["false_positive"]

        # --- Optional: validation_bundle.npz + threshold-strategy comparison ---
        # Requires the pipeline_core.py patch (val_pred_store / --export_val_bundle) to have
        # produced validation_bundle.npz alongside prediction_bundle.npz. Purely additive:
        # if the file isn't there yet, this is skipped with a clear note and everything above
        # is unaffected.
        val_npz_path = fold_dir / "validation_bundle.npz"
        fold_entry["validation_bundle_found"] = val_npz_path.exists()
        if val_npz_path.exists():
            log(f"[{fold_name}] loading {val_npz_path} ...")
            try:
                val_data = load_npz(val_npz_path)
                val_true_key, val_y_true = guess_true_labels(val_data)
                val_stacked_scores, val_stacked_names = guess_stacked_scores(val_data)
                fold_entry["threshold_strategy_comparison"] = {}
                if val_y_true is None:
                    log(f"[{fold_name}] WARNING: could not auto-detect y_true in validation_bundle.npz; "
                        f"keys were: {list(val_data.keys())}")
                else:
                    for model_name in model_names:
                        if model_name not in test_scores_raw:
                            continue  # no usable test-side score for this model in this fold
                        v_key, v_score = guess_model_score(val_data, model_name)
                        if v_score is None and val_stacked_names is not None:
                            candidates = [model_name] + MODEL_NAME_ALIASES.get(model_name, [])
                            for i, nm in enumerate(val_stacked_names):
                                if any(c.lower() in nm for c in candidates):
                                    row = (val_stacked_scores[i] if val_stacked_scores.shape[0] == len(val_stacked_names)
                                           else val_stacked_scores[:, i])
                                    v_key, v_score = f"(stacked:{nm})", np.asarray(row, dtype=float)
                                    break
                        if v_score is None or len(v_score) != len(val_y_true):
                            fold_entry["threshold_strategy_comparison"][model_name] = {
                                "available": False,
                                "reason": "validation score not found or length mismatch",
                            }
                            continue
                        val_prob, _ = to_probability(v_score)
                        cmp = compute_threshold_strategy_comparison(
                            val_y_true, val_prob, y_true, test_scores_raw[model_name],
                            min_recall=args.min_recall, precision_target=args.precision_target,
                            specificity_target=args.specificity_target,
                        )
                        fold_entry["threshold_strategy_comparison"][model_name] = {
                            "available": True, "val_key_used": v_key, **{"strategies": cmp},
                        }
                        log(f"[{fold_name}]   threshold-strategy comparison computed for {model_name}")
            except Exception as e:
                log(f"[{fold_name}] ERROR processing validation_bundle.npz: {e}")
                fold_entry["validation_bundle_error"] = str(e)
        else:
            log(f"[{fold_name}] validation_bundle.npz not found (needs the pipeline_core.py patch + a re-run) "
                f"-- skipping threshold-strategy comparison for this fold.")

        result["folds"][fold_name] = fold_entry

    log("")
    log("Computing pooled out-of-fold metrics (concatenating all folds' test "
        "predictions per model, threshold-free metrics only -- see code comment)...")
    for model_name in model_names:
        yt_parts = pooled_accum[model_name]["y_true"]
        yp_parts = pooled_accum[model_name]["y_prob"]
        if not yt_parts:
            continue
        y_true_all = np.concatenate(yt_parts)
        y_prob_all = np.concatenate(yp_parts)
        pooled = compute_pooled_oof_metrics(y_true_all, y_prob_all, n_boot=2000)
        result["pooled_out_of_fold"][model_name] = pooled
        if "pooled_auc" in pooled:
            log(f"  {model_name:20s} pooled n={pooled['n_pooled']}  "
                f"AUC={pooled['pooled_auc']:.4f} 95%CI={pooled['pooled_auc_ci95']}  "
                f"PR-AUC={pooled['pooled_pr_auc']:.4f} 95%CI={pooled['pooled_pr_auc_ci95']}")

    any_categories = any(per_category_accum[m] for m in model_names)
    if any_categories:
        log("")
        log("Computing per-category error analysis (negative class, pooled across folds)...")
        for model_name in model_names:
            cats = per_category_accum[model_name]
            if not cats:
                continue
            out_cats = {}
            for cat, d in sorted(cats.items()):
                acc = round(d["correct_reject"] / d["n"], 6) if d["n"] else None
                out_cats[cat] = {**d, "accuracy": acc}
                log(f"  {model_name:20s} {cat:28s} n={d['n']:4d}  "
                    f"correctly_rejected={d['correct_reject']:4d}  accuracy={acc}")
            result["per_category_error_analysis"][model_name] = out_cats
    else:
        log("")
        log("No per-image filenames found in any bundle -- per-category error analysis "
            "skipped. Requires the significance.py patch (filename export) and a fresh "
            "--eval_only re-run; see code comment on compute_per_category_breakdown.")

    out_path = Path(args.output)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    log("")
    log(f"Wrote {out_path.resolve()} ({size_kb:.1f} KB)")
    log("Please upload this JSON file back to the chat. If any model/fold shows")
    log("'NOT FOUND' or an error above, please also paste the console output for")
    log("that fold so the key-detection logic can be adjusted precisely.")


if __name__ == "__main__":
    main()
