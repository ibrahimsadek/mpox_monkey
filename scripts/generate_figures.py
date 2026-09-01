#!/usr/bin/env python3
"""
generate_figures.py

Regenerates every data-driven figure in "A leakage-aware benchmark for mpox
skin lesion classification" (PeerJ Computer Science, Manuscript #135497):
Figures 2 through 6. Figure 1 (the pipeline architecture diagram) is a
hand-drawn diagram edited directly in a PDF editor, not a data visualization,
and is intentionally NOT regenerated here -- see the note printed at runtime.

WHAT EACH FIGURE IS BUILT FROM
-------------------------------
Figures 2, 3, and 5 are built entirely from numbers already published and
independently verified against the manuscript's own tables (each function's
docstring cites the exact table). They need no external data file and will
always reproduce the published figures exactly.

Figures 4 and 6 are built from real per-fold ROC/PR curves and calibration
bins for ConvNeXt-Tiny. By default this script uses the exact verified data
bundled alongside it (curve_data_convnext_tta_on.npz,
reliability_data_convnext.npz) -- the same data underlying the published
Figures 4 and 6 -- so it reproduces them exactly with no extra setup.

If you have re-run the pipeline yourself and want to verify your own run
reproduces the published curves (rather than trusting the bundled data),
pass --extract pointing at the JSON produced by extract_prediction_bundles.py
against your outputs_mpox_balanced directory; the script will compute
Figures 4 and 6 fresh from your data and print a comparison against the
published values.

USAGE
-----
    python generate_figures.py [--outdir DIR] [--extract PATH_TO_JSON] [--model MODEL_NAME]

    --outdir    Where to write the PDFs. Default: current directory.
    --extract   Optional path to a leakage_aware_extract.json-style file
                (see extract_prediction_bundles.py). If omitted, Figures 4
                and 6 use the bundled verified reference data instead.
    --model     Which backbone Figures 4 and 6 are drawn for. Default:
                convnext_tiny, matching the published figures. Only takes
                effect together with --extract (the bundled reference data
                is ConvNeXt-Tiny-only).

DEPENDENCIES
------------
    numpy, matplotlib  (pip install numpy matplotlib)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent


# =============================================================================
# Figure 2: fold-aggregated dot-and-whisker plot
# Data verbatim from Tables 4 and 5 (mean +/- SD across 5 CV folds), cross-
# checked against the manuscript's own Abstract headline numbers.
# =============================================================================

TABLE4_5_DATA = {
    "EfficientNetV2-S":              dict(Accuracy=(0.8137, 0.0291), AUC=(0.9061, 0.0286), F1=(0.7575, 0.0519), Precision=(0.7610, 0.0900), Recall=(0.7726, 0.1235), Specificity=(0.8395, 0.0984)),
    "EfficientNet-B0":                dict(Accuracy=(0.8293, 0.0460), AUC=(0.8927, 0.0358), F1=(0.7736, 0.0656), Precision=(0.7854, 0.0746), Recall=(0.7680, 0.0911), Specificity=(0.8704, 0.0389)),
    "ResNet50":                       dict(Accuracy=(0.8521, 0.0357), AUC=(0.9191, 0.0260), F1=(0.7950, 0.0632), Precision=(0.8440, 0.0726), Recall=(0.7708, 0.1416), Specificity=(0.8952, 0.0709)),
    "DenseNet121":                    dict(Accuracy=(0.8506, 0.0226), AUC=(0.9270, 0.0251), F1=(0.8000, 0.0462), Precision=(0.8164, 0.0692), Recall=(0.7923, 0.0915), Specificity=(0.8863, 0.0482)),
    "ViT-B/16":                       dict(Accuracy=(0.8454, 0.0223), AUC=(0.9192, 0.0286), F1=(0.8018, 0.0513), Precision=(0.7794, 0.0696), Recall=(0.8308, 0.0767), Specificity=(0.8540, 0.0446)),
    "MobileNetV3-Large":              dict(Accuracy=(0.8581, 0.0329), AUC=(0.9221, 0.0253), F1=(0.8053, 0.0498), Precision=(0.8466, 0.0390), Recall=(0.7765, 0.1115), Specificity=(0.9111, 0.0335)),
    "ConvNeXt-Tiny":                  dict(Accuracy=(0.8647, 0.0412), AUC=(0.9284, 0.0303), F1=(0.8159, 0.0727), Precision=(0.8369, 0.0654), Recall=(0.8005, 0.1043), Specificity=(0.9007, 0.0465)),
    "Weighted ensemble + retrieval":  dict(Accuracy=(0.8634, 0.0300), AUC=(0.9307, 0.0171), F1=(0.8179, 0.0588), Precision=(0.8330, 0.0539), Recall=(0.8159, 0.1282), Specificity=(0.8856, 0.0699)),
    "Weighted ensemble":              dict(Accuracy=(0.8729, 0.0232), AUC=(0.9388, 0.0203), F1=(0.8334, 0.0432), Precision=(0.8360, 0.0728), Recall=(0.8449, 0.1119), Specificity=(0.8868, 0.0726)),
}


def fig2_dotwhisker(outdir: Path) -> Path:
    """Figure 2: mean +/- SD dot-and-whisker across 6 metrics and 9 models/variants.
    Source: Table 4 (Accuracy, AUC, F1) and Table 5 (Precision, Recall, Specificity)."""
    order = sorted(TABLE4_5_DATA.keys(), key=lambda k: TABLE4_5_DATA[k]["F1"][0])
    metrics = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC"]
    is_ensemble = lambda name: name.startswith("Weighted ensemble")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.edgecolor": "#444444", "axes.linewidth": 0.6,
    })

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.6), sharey=True)
    axes = axes.flatten()

    color_single, color_ens = "#3B6FA0", "#C1440E"

    for ax, metric in zip(axes, metrics):
        means = [TABLE4_5_DATA[k][metric][0] for k in order]
        stds = [TABLE4_5_DATA[k][metric][1] for k in order]
        colors = [color_ens if is_ensemble(k) else color_single for k in order]
        y = np.arange(len(order))
        ax.errorbar(means, y, xerr=stds, fmt="none", ecolor="#999999", elinewidth=1.0, capsize=2, zorder=1)
        ax.scatter(means, y, c=colors, s=26, zorder=2, edgecolors="white", linewidths=0.4)
        ax.set_title(metric, fontsize=9.5, fontweight="bold")
        ax.set_xlim(0.55, 1.02)
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for ax in axes[0::3]:
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(order, fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7.5)

    legend_elems = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_single, markersize=6, label='Single backbone'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_ens, markersize=6, label='Ensemble variant'),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Fold-aggregated performance (mean $\\pm$ SD across 5 cross-validation folds)", fontsize=9.5, y=1.01)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out_path = outdir / "fold_aggregate_dotwhisker.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================================================================
# Figure 3: row-normalized confusion matrix, weighted ensemble
# Source: Table 8 (mean confusion-matrix components: TP=57.8, FP=12.4, TN=95.0, FN=10.0).
# =============================================================================

CONF_SUMMARY = dict(TP=57.8, FP=12.4, TN=95.0, FN=10.0)  # Table 8


def fig3_normalized_cm(outdir: Path) -> Path:
    """Figure 3: row-normalized (percentage) confusion matrix for the weighted
    ensemble, derived from Table 8's mean TP/FP/TN/FN counts across 5 folds."""
    TP, FP, TN, FN = CONF_SUMMARY["TP"], CONF_SUMMARY["FP"], CONF_SUMMARY["TN"], CONF_SUMMARY["FN"]
    pos, neg = TP + FN, TN + FP

    mat = np.array([[TP / pos, FN / pos], [FP / neg, TN / neg]]) * 100.0
    labels = np.array([[f"TP\n{mat[0,0]:.1f}%", f"FN\n{mat[0,1]:.1f}%"],
                        [f"FP\n{mat[1,0]:.1f}%", f"TN\n{mat[1,1]:.1f}%"]])

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=100)

    for i in range(2):
        for j in range(2):
            txt_color = "white" if mat[i, j] > 55 else "#222222"
            ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=10.5, color=txt_color, fontweight="bold")

    ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted\nMpox", "Predicted\nNon-mpox"], fontsize=8.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual\nMpox", "Actual\nNon-mpox"], fontsize=8.5)
    ax.set_title("Weighted ensemble\n(row-normalized, mean across 5 folds)", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    out_path = outdir / "weighted_ensemble_normalized_cm.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================================================================
# Figure 5: per-fold confusion-matrix grid, weighted ensemble
# Source: reconstructed exactly from Table 14's per-fold precision/recall and
# Table 3's exact per-fold test-set positive/negative counts (TP = recall *
# n_positive; FP = TP/precision - TP; FN = n_positive - TP; TN = n_negative - FP).
# Verified: every fold's four counts sum exactly to that fold's Table 3 test total.
# =============================================================================

CM_GRID_PER_FOLD = {
    0: dict(tp=63, fp=8,  fn=18, tn=99),
    1: dict(tp=60, fp=21, fn=4,  tn=88),
    2: dict(tp=55, fp=6,  fn=8,  tn=93),
    3: dict(tp=76, fp=22, fn=4,  tn=93),
    4: dict(tp=35, fp=5,  fn=16, tn=102),
}
CM_GRID_THRESHOLDS = {0: 0.389, 1: 0.323, 2: 0.365, 3: 0.369, 4: 0.485}  # Table 14


def fig5_cm_grid(outdir: Path) -> Path:
    """Figure 5: five per-fold confusion matrices (raw counts) for the weighted
    ensemble, each at that fold's own validation-selected deployed threshold."""
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 1.85))
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    vmax = max(max(c.values()) for c in CM_GRID_PER_FOLD.values())

    for i, ax in enumerate(axes):
        c = CM_GRID_PER_FOLD[i]
        mat = np.array([[c["tp"], c["fn"]], [c["fp"], c["tn"]]])
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
        for r in range(2):
            for cc in range(2):
                val = mat[r, cc]
                color = "white" if val > vmax * 0.55 else "#222222"
                ax.text(cc, r, str(val), ha="center", va="center", fontsize=10, fontweight="bold", color=color)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred\nMpox", "Pred\nNon"], fontsize=6.3)
        ax.set_yticklabels(["Act\nMpox", "Act\nNon"] if i == 0 else [], fontsize=6.3)
        ax.set_title(f"Fold {i}\nthr={CM_GRID_THRESHOLDS[i]:.3f}", fontsize=7.8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Weighted ensemble: confusion matrix per fold, at each fold's deployed threshold", fontsize=8.6, y=1.10)
    fig.tight_layout()
    out_path = outdir / "weighted_ensemble_cm_grid.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================================================================
# Figures 4 and 6: real per-fold ROC/PR curves and calibration, ConvNeXt-Tiny
# =============================================================================

def load_extraction_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_fold_overlay_data(d: dict, model: str) -> dict:
    fold_names = sorted(d["folds"].keys())
    m0 = d["folds"][fold_names[0]]["models"][model]
    roc_grid = np.array(m0["roc_curve"]["fpr_grid"])
    pr_grid = np.array(m0["pr_curve"]["recall_grid"])
    tprs, precs, aucs, praucs, pos_rates = [], [], [], [], []
    for fname in fold_names:
        mm = d["folds"][fname]["models"][model]
        tprs.append(mm["roc_curve"]["tpr_mean_at_grid"])
        precs.append(mm["pr_curve"]["precision_at_grid"])
        aucs.append(mm["auc"])
        praucs.append(mm["pr_auc_average_precision"])
        pos_rates.append(mm["class_balance_positive_rate"])
    return dict(roc_grid=roc_grid, pr_grid=pr_grid, tprs=np.array(tprs), precs=np.array(precs),
                aucs=np.array(aucs), prauc=np.array(praucs), pos_rates=np.array(pos_rates))


def extract_reliability_data(d: dict, model: str, n_bins: int = 15) -> dict:
    fold_names = sorted(d["folds"].keys())
    pooled_count = np.zeros(n_bins)
    pooled_conf_sum = np.zeros(n_bins)
    pooled_acc_sum = np.zeros(n_bins)
    for fname in fold_names:
        bins = d["folds"][fname]["models"][model]["reliability_diagram_bins"]
        for i, b in enumerate(bins):
            if b["count"] > 0:
                pooled_count[i] += b["count"]
                pooled_conf_sum[i] += b["mean_confidence"] * b["count"]
                pooled_acc_sum[i] += b["mean_accuracy"] * b["count"]
    conf = np.divide(pooled_conf_sum, pooled_count, out=np.full(n_bins, np.nan), where=pooled_count > 0)
    acc = np.divide(pooled_acc_sum, pooled_count, out=np.full(n_bins, np.nan), where=pooled_count > 0)
    return dict(count=pooled_count, conf=conf, acc=acc, n_folds=len(fold_names))


def fig4_fold_overlay(curve_data: dict, outdir: Path, model_label: str = "ConvNeXt-Tiny") -> Path:
    """Figure 4: five-fold ROC and precision-recall curves overlaid with a
    shaded +/-1 SD band, for the main TTA-on benchmark."""
    roc_grid, pr_grid = curve_data["roc_grid"], curve_data["pr_grid"]
    tprs, precs = curve_data["tprs"], curve_data["precs"]
    aucs, prauc, pos_rates = curve_data["aucs"], curve_data["prauc"], curve_data["pos_rates"]

    fold_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))

    ax = axes[0]
    for i in range(len(aucs)):
        ax.plot(roc_grid, tprs[i], color=fold_colors[i % 5], lw=1.0, alpha=0.75, label=f"Fold {i} (AUC={aucs[i]:.3f})")
    mean_tpr, std_tpr = tprs.mean(axis=0), tprs.std(axis=0)
    ax.plot(roc_grid, mean_tpr, color="black", lw=2.2, label=f"Mean (AUC={aucs.mean():.3f}$\\pm${aucs.std():.3f})")
    ax.fill_between(roc_grid, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1),
                     color="black", alpha=0.15, linewidth=0, label="$\\pm$1 SD")
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.9, ls="--", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    ax = axes[1]
    for i in range(len(prauc)):
        ax.plot(pr_grid, precs[i], color=fold_colors[i % 5], lw=1.0, alpha=0.75, label=f"Fold {i} (AP={prauc[i]:.3f})")
    mean_prec, std_prec = precs.mean(axis=0), precs.std(axis=0)
    ax.plot(pr_grid, mean_prec, color="black", lw=2.2, label=f"Mean (AP={prauc.mean():.3f}$\\pm${prauc.std():.3f})")
    ax.fill_between(pr_grid, np.clip(mean_prec - std_prec, 0, 1), np.clip(mean_prec + std_prec, 0, 1),
                     color="black", alpha=0.15, linewidth=0, label="$\\pm$1 SD")
    ax.axhline(pos_rates.mean(), color="#999999", lw=0.9, ls="--", label="Chance (prevalence)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision--Recall", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    handles0, labels0 = axes[0].get_legend_handles_labels()
    n = len(aucs)
    fig.legend(handles0[:n] + [handles0[n], handles0[n + 1]], labels0[:n] + [labels0[n], labels0[n + 1]],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.16), fontsize=7.3, frameon=False,
               columnspacing=1.2, handlelength=1.6)

    fig.suptitle(f"{model_label}: all {n} folds (main TTA-on benchmark)", fontsize=9.5, y=1.03)
    fig.tight_layout()
    out_path = outdir / "fold_roc_pr_overlay_TTAon.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    if model_label == "ConvNeXt-Tiny":
        print(f"  Fold-mean AUC={aucs.mean():.4f} (published: 0.9284), "
              f"PR-AUC={prauc.mean():.4f} (published: 0.9048)")
    else:
        print(f"  Fold-mean AUC={aucs.mean():.4f}, PR-AUC={prauc.mean():.4f} "
              f"(no published comparison for {model_label} -- Figure 4 is published for ConvNeXt-Tiny only)")
    return out_path


def fig6_reliability(rel_data: dict, outdir: Path, model_label: str = "ConvNeXt-Tiny", n_folds: int = 5) -> Path:
    """Figure 6: pooled reliability diagram + bin-occupancy panel."""
    count, conf, acc = rel_data["count"], rel_data["conf"], rel_data["acc"]
    n_folds = rel_data.get("n_folds", n_folds) if isinstance(rel_data, dict) else n_folds
    n_bins = len(count)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    n_total = int(count.sum())

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.3, 1]})

    ax1.plot([0, 1], [0, 1], color="#999999", lw=1.0, ls="--", label="Perfect calibration")
    valid = ~np.isnan(acc)
    ax1.plot(conf[valid], acc[valid], marker="o", ms=4, color="#C1440E", lw=1.5, label=f"{model_label} (pooled)")
    ax1.set_xlabel("Mean predicted probability (confidence)")
    ax1.set_ylabel("Observed frequency (accuracy)")
    ax1.set_title("Reliability diagram", fontsize=10, fontweight="bold")
    ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=7, loc="upper left", frameon=False)
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)

    ax2.bar(bin_centers, count, width=1 / n_bins * 0.9, color="#4C72B0", alpha=0.85)
    ax2.set_xlabel("Predicted probability bin")
    ax2.set_ylabel(f"Count (pooled, n={n_total})")
    ax2.set_title("Bin occupancy", fontsize=10, fontweight="bold")
    ax2.set_xlim(-0.02, 1.02)
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)

    fig.suptitle(f"{model_label} calibration, pooled across {n_folds} folds (n={n_total}, {n_bins} equal-width bins)", fontsize=9, y=1.03)
    fig.tight_layout()
    out_path = outdir / "reliability_diagram.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=str, default=".", help="Directory to write the PDF figures to.")
    parser.add_argument("--extract", type=str, default=None,
                         help="Optional path to a leakage_aware_extract.json-style file "
                              "(from extract_prediction_bundles.py). If omitted, Figures 4 and 6 "
                              "use the bundled verified reference data instead.")
    parser.add_argument("--model", type=str, default="convnext_tiny",
                         help="Backbone for Figures 4/6 when --extract is given (default: convnext_tiny, "
                              "matching the published figures).")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Figure 1 (pipeline.pdf): NOT regenerated by this script -- it is a hand-drawn")
    print("  architecture diagram edited directly in a PDF editor, not a data visualization.")
    print()

    print("Figure 2 (dot-and-whisker, Tables 4-5)...")
    print(f"  saved {fig2_dotwhisker(outdir)}")

    print("Figure 3 (normalized confusion matrix, Table 8)...")
    print(f"  saved {fig3_normalized_cm(outdir)}")

    print("Figure 5 (per-fold confusion-matrix grid, Tables 3 & 14)...")
    print(f"  saved {fig5_cm_grid(outdir)}")

    if args.extract:
        print(f"Loading {args.extract} for Figures 4 and 6 (model={args.model})...")
        data = load_extraction_json(args.extract)
        curve_data = extract_fold_overlay_data(data, args.model)
        rel_data = extract_reliability_data(data, args.model)
    else:
        print("--extract not given: using bundled verified reference data for Figures 4 and 6 "
              "(ConvNeXt-Tiny, matching the published figures exactly)...")
        curve_data = dict(np.load(HERE / "curve_data_convnext_tta_on.npz"))
        rel_data = dict(np.load(HERE / "reliability_data_convnext.npz"))
        rel_data["n_folds"] = 5  # the bundled reference data is the published 5-fold benchmark

    model_label = "ConvNeXt-Tiny" if args.model == "convnext_tiny" else args.model
    print("Figure 4 (fold-overlaid ROC/PR curves)...")
    print(f"  saved {fig4_fold_overlay(curve_data, outdir, model_label)}")

    print("Figure 6 (pooled reliability diagram)...")
    print(f"  saved {fig6_reliability(rel_data, outdir, model_label)}")

    print()
    print(f"Done. 5 figures written to {outdir.resolve()}")


if __name__ == "__main__":
    main()
