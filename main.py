#!/usr/bin/env python3
"""Main training and evaluation entry point for the leakage-aware mpox benchmark.

The script supports grouped cross-validation, original-vs-augmented training
mixes, calibration, threshold selection, ensembling, optional Optuna/NSGA-II
search, and a set of auxiliary analysis features.
"""

from __future__ import annotations

import warnings

# Filter the specific warning about autocast deprecation
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import copy
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from contextlib import nullcontext

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless for terminal runs
import matplotlib.pyplot as plt
from matplotlib import colors

_QUANV_MODEL_CFG = None  # set in main when --use_quanv

# timm is used for EfficientNet/ViT backbones (and quanv-hybrid backbone).
# Keep it optional, but fail loudly *when a timm-backed model is requested*.
def _require_timm():
    try:
        import timm  # type: ignore
        return timm
    except Exception as e:
        raise ImportError(
            "This run requested a timm-backed model (e.g., efficientnet_b0_quanv), but 'timm' could not be imported. "
            "Install/repair it with: pip install -U timm"
        ) from e

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
try:
    from torchvision import transforms, models
    from torchvision.datasets.folder import default_loader
except Exception as e:
    raise SystemExit(
        "Failed to import torchvision. Ensure that torch and torchvision are installed "
        "as a compatible pair for your Python/CUDA build. See https://pytorch.org/get-started/locally/. "
        f"Original error: {e}"
    )

from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import (
accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score, log_loss, brier_score_loss, classification_report
)

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, **kwargs): return iterator



def read_name_list(path: Union[str, Path]) -> List[str]:
    """Read a text file containing image names (one per line).

    Accepts either bare filenames (e.g., M01_01_00.jpg) or relative/absolute paths.
    Returns a de-duplicated list preserving file order.

    Important robustness behavior (fixes common "list-file contains .txt/.csv" mistakes):
    - Ignores empty lines and lines starting with '#'
    - Matches by basename
    - Ignores lines whose basename does NOT look like an image (by extension)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"List file not found: {p}")

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: List[str] = []
    seen: Set[str] = set()
    for ln in lines:
        s = ln.strip().strip('"').strip("'")
        if not s or s.startswith('#'):
            continue
        name = Path(s).name
        ext = Path(name).suffix.lower()
        if ext not in image_exts:
            # Prevent false "missing" warnings when list files accidentally include text filenames.
            continue
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out

# -----------------------------
# v10 additions: SAM optimizer, Focal Loss, optional experiment tracking
# -----------------------------

class FocalLoss(nn.Module):
    """Multi-class Focal Loss (works for binary with class indices 0/1).

    alpha: weight for class 1 (positive). class 0 uses (1-alpha) by default.
    gamma: focusing parameter.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = str(reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B]
        if logits.ndim != 2:
            raise ValueError(f"FocalLoss expects logits [B,C], got {tuple(logits.shape)}")
        targets = targets.long()
        logp = torch.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        # gather per-target
        idx = targets.view(-1, 1)
        logpt = logp.gather(1, idx).squeeze(1)
        pt = p.gather(1, idx).squeeze(1)
        # alpha weighting: binary special-case
        if logits.size(1) == 2:
            at = torch.where(targets == 1, torch.full_like(pt, self.alpha), torch.full_like(pt, 1.0 - self.alpha))
        else:
            at = torch.ones_like(pt)
        loss = -at * (1.0 - pt).pow(self.gamma) * logpt
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class SAM(optim.Optimizer):
    """Sharpness-Aware Minimization (SAM) optimizer wrapper.

    Usage:
      base_opt = torch.optim.AdamW
      optimizer = SAM(model.parameters(), base_opt, rho=0.05, adaptive=False, lr=..., weight_decay=...)

    In training step:
      loss.backward(); optimizer.first_step(zero_grad=True)
      loss2.backward(); optimizer.second_step(zero_grad=True)
    """
    def __init__(self, params, base_optimizer, rho: float = 0.05, adaptive: bool = False, **kwargs):
        if rho <= 0:
            raise ValueError("rho must be positive")
        self.rho = float(rho)
        self.adaptive = bool(adaptive)
        self.base_optimizer = base_optimizer(params, **kwargs)
        defaults = self.base_optimizer.defaults
        super().__init__(self.base_optimizer.param_groups, defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = (p.pow(2) if self.adaptive else 1.0) * p.grad
                e_w = e_w * scale
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = self.state[p].get('e_w', None)
                if e_w is None:
                    continue
                p.sub_(e_w)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure=None):
        raise RuntimeError("SAM requires first_step and second_step; call those from the training loop.")

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]['params'][0].device
        norms = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if self.adaptive:
                    g = g * p.abs()
                norms.append(torch.norm(g, p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)



def make_optimizer(params, lr: float, weight_decay: float, args=None, name=None, **kwargs):
    """Create optimizer.

    Backward/forward compatible wrapper:
      - accepts either `args` (Namespace-like) or explicit `name`
      - tolerates extra kwargs (e.g., fine-tune code passing name=...)
    Supported: adamw, adam, sgd. Defaults to AdamW.
    """
    opt_name = (name or getattr(args, "optimizer", None) or getattr(args, "opt", None) or "adamw")
    opt_name = str(opt_name).lower()

    momentum = float(getattr(args, "momentum", 0.9)) if args is not None else 0.9
    betas = getattr(args, "betas", None) if args is not None else None
    eps = float(getattr(args, "eps", 1e-8)) if args is not None else 1e-8

    if betas is None:
        betas = (0.9, 0.999)

    # Accept either an iterable of parameters OR a torch.nn.Module-like object.
    # (Some code paths may pass the model by mistake; optimizers expect an iterable.)
    if hasattr(params, "parameters") and callable(getattr(params, "parameters")):
        try:
            params = params.parameters()
        except Exception:
            pass

    if opt_name in ("adamw", "adam_w"):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    if opt_name == "sgd":
        nesterov = bool(getattr(args, "nesterov", False)) if args is not None else False
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)

    # Fallback
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)



def maybe_init_wandb(args, config: dict):
    if not getattr(args, 'wandb', False):
        return None
    try:
        import wandb
    except Exception as e:
        print(f"[WARN] wandb requested but not available: {e}")
        return None
    run = wandb.init(project=getattr(args, 'wandb_project', 'mpox'), name=getattr(args, 'wandb_name', None), config=config, reinit=True)
    return run


def maybe_init_mlflow(args):
    if not getattr(args, 'mlflow', False):
        return None
    try:
        import mlflow
    except Exception as e:
        print(f"[WARN] mlflow requested but not available: {e}")
        return None
    uri = getattr(args, 'mlflow_uri', None)
    if uri:
        mlflow.set_tracking_uri(uri)
    exp = getattr(args, 'mlflow_experiment', None)
    if exp:
        mlflow.set_experiment(exp)
    mlflow.start_run(run_name=getattr(args, 'mlflow_run_name', None))
    return mlflow

# -----------------------------
# New: Dataset Balancing Classes
# -----------------------------

class DatasetAnalyzer:
    """Analyze dataset distribution and properties."""
    
    def __init__(self, samples: List[Tuple[Path, int]]):
        self.samples = samples
        self.labels = [y for _, y in samples]
        self.paths = [p for p, _ in samples]
    
    def get_statistics(self) -> Dict:
        """Get comprehensive dataset statistics."""
        
        # Basic counts
        label_counts = Counter(self.labels)
        total_samples = len(self.samples)
        
        # Extract subjects (assuming format: PREFIX_*)
        subjects = {}
        for path, label in self.samples:
            stem = path.stem
            # Try to extract subject ID (first part before second underscore)
            parts = stem.split('_')
            if len(parts) >= 2:
                subject_id = f"{parts[0]}_{parts[1]}"
            else:
                subject_id = parts[0]
            
            if subject_id not in subjects:
                subjects[subject_id] = {'label': label, 'count': 0, 'files': []}
            subjects[subject_id]['count'] += 1
            subjects[subject_id]['files'].append(path.name)
        
        # Per-subject statistics
        subject_stats = {}
        for label in set(self.labels):
            label_subjects = [s for s, data in subjects.items() if data['label'] == label]
            subject_counts = [subjects[s]['count'] for s in label_subjects]
            
            subject_stats[label] = {
                'unique_subjects': len(label_subjects),
                'avg_images_per_subject': np.mean(subject_counts) if subject_counts else 0,
                'std_images_per_subject': np.std(subject_counts) if len(subject_counts) > 1 else 0,
                'min_images_per_subject': min(subject_counts) if subject_counts else 0,
                'max_images_per_subject': max(subject_counts) if subject_counts else 0,
            }
        
        return {
            'total_samples': total_samples,
            'label_distribution': dict(label_counts),
            'class_ratio': label_counts[1] / max(1, label_counts[0]) if 0 in label_counts and 1 in label_counts else float('nan'),
            'subject_statistics': subject_stats,
            'imbalance_ratio': max(label_counts.values()) / min(label_counts.values()) if len(label_counts) >= 2 else 1.0,
        }
    
    def plot_distribution(self, output_path: Path, title: str = "Dataset Distribution") -> None:
        """Plot dataset distribution."""
        stats = self.get_statistics()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot of class distribution
        labels = ['Class 0', 'Class 1'] if len(stats['label_distribution']) == 2 else list(stats['label_distribution'].keys())
        counts = [stats['label_distribution'].get(i, 0) for i in range(len(labels))]
        
        axes[0].bar(labels, counts, color=['skyblue', 'lightcoral'])
        axes[0].set_title(f'Class Distribution (Total: {stats["total_samples"]})')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Number of Images')
        for i, count in enumerate(counts):
            axes[0].text(i, count + max(counts)*0.01, str(count), ha='center')
        
        # Subject distribution
        if stats['subject_statistics']:
            for label in sorted(stats['subject_statistics'].keys()):
                stats_label = stats['subject_statistics'][label]
                axes[1].bar(f'Class {label}', stats_label['unique_subjects'], 
                          alpha=0.7, label=f'Class {label}')
                axes[1].text(f'Class {label}', stats_label['unique_subjects'] + 1, 
                           f"{stats_label['unique_subjects']} subjects", ha='center')
        
        axes[1].set_title('Unique Subjects per Class')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Number of Unique Subjects')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save figure
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        fig.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')
        plt.close(fig)


class DatasetBalancer:
    """Balance dataset using various strategies."""
    
    def __init__(self, samples: List[Tuple[Path, int]], group_ids: List[str] = None):
        self.samples = samples
        self.group_ids = group_ids if group_ids else [self._extract_group_id(p) for p, _ in samples]
        self.labels = [y for _, y in samples]
    
    def _extract_group_id(self, path: Path) -> str:
        """Extract group ID from filename."""
        stem = path.stem
        parts = stem.split('_')
        return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else parts[0]
    
    def analyze_imbalance(self) -> Dict:
        """Analyze dataset imbalance."""
        label_counts = Counter(self.labels)
        return {
            'class_0': label_counts.get(0, 0),
            'class_1': label_counts.get(1, 0),
            'imbalance_ratio': max(label_counts.values()) / min(label_counts.values()) if len(label_counts) >= 2 else 1.0,
            'total_samples': len(self.samples),
        }
    
    def balance_dataset(self, strategy: str = 'undersample', balance_level: str = 'dataset') -> Tuple[List[Tuple[Path, int]], List[str]]:
        """
        Balance the dataset.
        
        Args:
            strategy: 'undersample', 'oversample', or 'none'
            balance_level: 'dataset' (balance before split) or 'train' (balance only training set)
        
        Returns:
            Balanced samples and corresponding group IDs
        """
        if strategy == 'none' or balance_level == 'none':
            return self.samples, self.group_ids
        
        # Separate by class
        class_0_indices = [i for i, (_, label) in enumerate(self.samples) if label == 0]
        class_1_indices = [i for i, (_, label) in enumerate(self.samples) if label == 1]
        
        min_count = min(len(class_0_indices), len(class_1_indices))
        
        if strategy == 'undersample':
            # Randomly sample from the larger class
            if len(class_0_indices) > len(class_1_indices):
                # Class 0 is larger, undersample it
                selected_0 = random.sample(class_0_indices, min_count)
                selected_1 = class_1_indices
            else:
                # Class 1 is larger, undersample it
                selected_0 = class_0_indices
                selected_1 = random.sample(class_1_indices, min_count)
            
            balanced_indices = selected_0 + selected_1
            
        elif strategy == 'oversample':
            # Oversample the smaller class with replacement
            if len(class_0_indices) < len(class_1_indices):
                # Class 0 is smaller, oversample it
                selected_0 = random.choices(class_0_indices, k=len(class_1_indices))
                selected_1 = class_1_indices
            else:
                # Class 1 is smaller, oversample it
                selected_0 = class_0_indices
                selected_1 = random.choices(class_1_indices, k=len(class_0_indices))
            
            balanced_indices = selected_0 + selected_1
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Shuffle indices
        random.shuffle(balanced_indices)
        
        # Create balanced samples and group IDs
        balanced_samples = [self.samples[i] for i in balanced_indices]
        balanced_groups = [self.group_ids[i] for i in balanced_indices]
        
        print(f"Balanced dataset: {len(class_0_indices)}→{sum(1 for _,y in balanced_samples if y==0)} class 0, "
              f"{len(class_1_indices)}→{sum(1 for _,y in balanced_samples if y==1)} class 1")
        
        return balanced_samples, balanced_groups
    
    def create_stratified_split(self, train_ratio: float = 0.7, val_ratio: float = 0.1, 
                               test_ratio: float = 0.2, seed: int = 42) -> Dict[str, List[int]]:
        """Create stratified split indices."""
        from sklearn.model_selection import train_test_split
        
        indices = list(range(len(self.samples)))
        
        # First split: train vs temp (val+test)
        train_idx, temp_idx = train_test_split(
            indices, 
            train_size=train_ratio,
            stratify=self.labels,
            random_state=seed
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size,
            stratify=[self.labels[i] for i in temp_idx],
            random_state=seed + 1
        )
        
        return {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }


# -----------------------------
# Plot saving helpers (PNG + SVG)
# -----------------------------
def save_figure_dual(fig: plt.Figure, base_path: Path, dpi: int = 600) -> None:
    """Save a matplotlib figure to <base_path>.png (dpi) and <base_path>.svg."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Compute common binary classification metrics."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    acc = float(accuracy_score(y_true, y_pred)) if y_true.size else float("nan")
    prec = float(precision_score(y_true, y_pred, zero_division=0)) if y_true.size else float("nan")
    rec = float(recall_score(y_true, y_pred, zero_division=0)) if y_true.size else float("nan")
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if y_true.size else float("nan")

    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    auc = float("nan")
    if y_score is not None:
        ys = np.asarray(y_score).astype(np.float32)
        if y_true.size and len(np.unique(y_true)) == 2:
            try:
                auc = float(roc_auc_score(y_true, ys))
            except Exception:
                auc = float("nan")

    metrics = {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "auc": auc,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }
    extras = {"confusion_matrix": cm}
    return metrics, extras

def save_classification_report_txt(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path,
                                   neg_name: str = "neg", pos_name: str = "pos",
                                   title: str = "Classification report") -> None:
    """Save sklearn classification_report to a text file (safe for imbalanced data)."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    try:
        rep = classification_report(
            y_true, y_pred,
            labels=[0, 1],
            target_names=[str(neg_name), str(pos_name)],
            digits=4,
            zero_division=0
        )
    except Exception as e:
        rep = f"(classification_report failed: {e})"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write(rep)
        f.write("\n")




# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def mean_std(values):
    """Return (mean, sample_std) for a list/iterable of numbers.

    - Uses sample std (ddof=1) when n>1; returns 0.0 when n==1.
    - Returns (nan, nan) when empty.
    """
    vals = list(values) if values is not None else []
    if len(vals) == 0:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std

# -----------------------------
# Integrity helpers (MD5 exact duplicates)
# -----------------------------
def md5_file(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------
# Identity grouping to avoid leakage
# -----------------------------
def infer_group_id(path: Path, group_mode: str, group_regex: Optional[str]) -> str:
    """
    Infer a group/identity id from a filepath.

    Why this matters:
      - When multiple images belong to the same subject/lesion (or are near-duplicates/augmentations),
        they MUST be kept in the same split to avoid leakage.
      - For filenames like `M01_01_00.jpg` / `NM101_02_13.jpg`, a safe default is grouping by the
        first two underscore-separated tokens (e.g., `M01_01`, `NM101_02`).

    group_mode options:
      - regex: capture group id using --group_regex (recommended default)
      - prefix: first token before '_'  (e.g., M01, NM101)
      - prefix2: first two tokens joined by '_' (e.g., M01_01, NM101_02)
      - full: full stem (each file its own group)
      - stem: stem with common augmentation suffixes stripped
    """
    stem = path.stem

    if group_mode == "regex":
        if not group_regex:
            raise ValueError("--group_mode regex requires --group_regex.")
        m = re.search(group_regex, stem)
        if not m or len(m.groups()) < 1:
            raise ValueError(
                f"Regex '{group_regex}' failed to match or capture a group in filename:\n"
                f"  -> {path.name}\n"
                "Check your regex or your filenames."
            )
        return m.group(1)

    if group_mode == "prefix":
        return stem.split("_")[0]

    if group_mode == "prefix2":
        parts = stem.split("_")
        return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]

    if group_mode == "full":
        return stem

    # "stem" mode: strip typical augmentation suffix tokens at end of filename
    s = stem
    s = re.sub(
        r"(_aug\d+|_augment\d+|_flip|_flipped|_rot\d+|_rotate\d+|_crop\d+|_bright\d+|_contrast\d+|_noise\d+)$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"(\-aug\d+|\-rot\d+|\-flip|\-crop\d+)$", "", s, flags=re.IGNORECASE)
    return s


# -----------------------------
# Augmentation / leakage diagnostics helpers
# -----------------------------
_AUG_TOKEN_REGEXES = [
    r"(?:aug\d+|augment\d+)",
    r"(?:flip|flipped|hflip|vflip)",
    r"(?:rot\d+|rotate\d+)",
    r"(?:crop\d+|resize\d+|scale\d+)",
    r"(?:bright\d+|brightness\d+|contrast\d+|sat\d+|saturation\d+|hue\d+)",
    r"(?:noise\d+|blur\d+|sharp\d+)",
    r"(?:gamma\d+|clahe\d+)",
]
_AUG_TOKEN_RE = re.compile(rf"^(?:{'|'.join(_AUG_TOKEN_REGEXES)})$", re.IGNORECASE)
_TRAILING_NUM_RE = re.compile(r"^(.*?)(?:_\d{1,4})$")

def strip_aug_tokens(stem: str) -> str:
    """Repeatedly strip common augmentation tokens from the end of a stem."""
    s = stem.strip()
    changed = True
    while changed:
        changed = False
        parts = s.split("_")
        if len(parts) >= 2 and _AUG_TOKEN_RE.fullmatch(parts[-1] or ""):
            s = "_".join(parts[:-1])
            changed = True
            continue
        if "-" in s:
            dparts = s.split("-")
            if len(dparts) >= 2 and _AUG_TOKEN_RE.fullmatch(dparts[-1] or ""):
                s = "-".join(dparts[:-1])
                changed = True
                continue
    return s

def infer_leakage_id(path: Path) -> str:
    """Infer a *base* identity intended to represent the same original before augmentation.

    Used only for diagnostics: if the same leakage_id appears in multiple splits, you likely have leakage.
    """
    s = strip_aug_tokens(path.stem)
    m = _TRAILING_NUM_RE.match(s)
    if m and m.group(1):
        return m.group(1)
    return s

def detect_filename_suffix_patterns(paths: List[Path], sample_n: int = 50) -> Dict[str, Any]:
    """Detect common filename patterns that look like augmentation suffixes."""
    stems = [p.stem for p in paths]
    n = len(stems)
    base_num = 0
    aug_num = 0
    for st in stems:
        if re.match(r"^.*_[0-9]{1,4}$", st):
            base_num += 1
        if re.match(r"^.*_[A-Za-z]+[0-9]*_[0-9]{1,4}$", st):
            aug_num += 1
    pct_base_num = (base_num / n) * 100.0 if n else 0.0
    pct_aug_num = (aug_num / n) * 100.0 if n else 0.0
    return {
        "n": n,
        "pct_base_num": pct_base_num,
        "pct_aug_num": pct_aug_num,
        "suggested_group_regex": "(.+?)_[0-9]+$" if pct_base_num >= 20.0 else None,
        "examples": stems[:sample_n],
    }

def print_leakage_risk_report(samples: List[Tuple[Path,int]], group_ids: List[str], splits: "SplitIndices", max_examples: int = 15) -> None:
    """Print diagnostics for potential leakage using filename-based near-duplicate heuristics."""
    leakage_ids = [infer_leakage_id(p) for p, _ in samples]

    idx_to_split: Dict[int, str] = {}
    for i in splits.train:
        idx_to_split[i] = "train"
    for i in splits.val:
        idx_to_split[i] = "val"
    for i in splits.test:
        idx_to_split[i] = "test"

    lid_to_splits: Dict[str, set] = {}
    lid_to_examples: Dict[str, List[str]] = {}
    for i, lid in enumerate(leakage_ids):
        sname = idx_to_split.get(i, "unknown")
        lid_to_splits.setdefault(lid, set()).add(sname)
        lid_to_examples.setdefault(lid, [])
        if len(lid_to_examples[lid]) < 3:
            lid_to_examples[lid].append(samples[i][0].name)

    cross = [(lid, sp) for lid, sp in lid_to_splits.items() if len(sp) > 1 and "unknown" not in sp]
    if not cross:
        print("[LeakageCheck] ✅ No base-stem (leakage_id) appears in more than one split (filename heuristic).")
    else:
        print("[LeakageCheck] ⚠️ Potential leakage risk detected!")
        print(f"[LeakageCheck] {len(cross)} base-stems appear across multiple splits.")
        for lid, sp in sorted(cross, key=lambda x: (-len(x[1]), x[0]))[:max_examples]:
            ex = ", ".join(lid_to_examples.get(lid, [])[:3])
            print(f"  - leakage_id='{lid}' in splits={sorted(list(sp))} e.g. {ex}")
        if len(cross) > max_examples:
            print(f"  ... and {len(cross)-max_examples} more.")

    # If leakage_id maps to multiple group_ids, grouping may be too strict
    lid_to_gids: Dict[str, set] = {}
    for lid, gid in zip(leakage_ids, group_ids):
        lid_to_gids.setdefault(lid, set()).add(gid)
    multi_gid = [(lid, gids) for lid, gids in lid_to_gids.items() if len(gids) > 1]
    if multi_gid:
        print("[LeakageCheck] ⚠️ Grouping may be splitting augmentation variants into different group_ids.")
        print(f"[LeakageCheck] {len(multi_gid)} base-stems map to multiple group_ids.")
        for lid, gids in sorted(multi_gid, key=lambda x: (-len(x[1]), x[0]))[:max_examples]:
            gids_list = sorted(list(gids))
            print(f"  - leakage_id='{lid}' -> group_ids={gids_list[:6]}{' ...' if len(gids_list)>6 else ''}")
        print("[LeakageCheck] Recommendation: use a group id equal to leakage_id (e.g., --group_mode stem) or adjust --group_regex.")

@dataclass
class SplitIndices:
    train: List[int]
    val: List[int]
    test: List[int]


def assert_no_group_overlap(group_ids: List[str], splits: SplitIndices) -> None:
    def groups(idxs: List[int]) -> set:
        return set(group_ids[i] for i in idxs)

    g_train, g_val, g_test = groups(splits.train), groups(splits.val), groups(splits.test)
    inter_tv = g_train.intersection(g_val)
    inter_tt = g_train.intersection(g_test)
    inter_vt = g_val.intersection(g_test)
    if inter_tv or inter_tt or inter_vt:
        raise RuntimeError(
            "Group overlap detected across splits (leakage risk).\n"
            f"train∩val={len(inter_tv)}, train∩test={len(inter_tt)}, val∩test={len(inter_vt)}"
        )


# -----------------------------
# Split builders (Enhanced with balancing)
# -----------------------------
def build_splits_group_balanced(
    samples: List[Tuple[Path, int]],
    group_ids: List[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    balance_strategy: str = 'none',
    balance_level: str = 'train'
) -> SplitIndices:
    """Build splits with optional balancing."""
    
    # First create base splits
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.")
    
    idx = np.arange(len(group_ids))
    
    # Group-based split
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(gss1.split(idx, groups=group_ids))
    
    temp_groups = [group_ids[i] for i in temp_idx]
    val_within_temp = val_ratio / (val_ratio + test_ratio)
    
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_within_temp, random_state=seed + 1)
    val_rel, test_rel = next(gss2.split(temp_idx, groups=temp_groups))
    
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]
    
    splits = SplitIndices(train=train_idx.tolist(), val=val_idx.tolist(), test=test_idx.tolist())
    
    # Apply balancing if requested
    if balance_strategy != 'none' and balance_level == 'train':
        # Balance only training set
        train_samples = [samples[i] for i in splits.train]
        train_labels = [samples[i][1] for i in splits.train]
        
        # Separate by class
        class_0_indices = [splits.train[i] for i, label in enumerate(train_labels) if label == 0]
        class_1_indices = [splits.train[i] for i, label in enumerate(train_labels) if label == 1]
        
        if balance_strategy == 'undersample':
            min_count = min(len(class_0_indices), len(class_1_indices))
            if len(class_0_indices) > min_count:
                class_0_indices = random.sample(class_0_indices, min_count)
            if len(class_1_indices) > min_count:
                class_1_indices = random.sample(class_1_indices, min_count)
        
        elif balance_strategy == 'oversample':
            max_count = max(len(class_0_indices), len(class_1_indices))
            if len(class_0_indices) < max_count:
                class_0_indices = random.choices(class_0_indices, k=max_count)
            if len(class_1_indices) < max_count:
                class_1_indices = random.choices(class_1_indices, k=max_count)
        
        # Combine and shuffle
        balanced_train = class_0_indices + class_1_indices
        random.shuffle(balanced_train)
        splits.train = balanced_train
        
        print(f"Balanced training set: {len(class_0_indices)} class 0, {len(class_1_indices)} class 1")
    
    return splits


def build_splits_group_stratified(
    samples: List[Tuple[Path, int]],
    group_ids: List[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    balance_strategy: str = 'none',
    balance_level: str = 'train'
) -> SplitIndices:
    """Build stratified group splits with optional balancing."""
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.")

    group_to_indices: Dict[str, List[int]] = {}
    group_to_labels: Dict[str, set] = {}

    for i, (_, y) in enumerate(samples):
        gid = group_ids[i]
        group_to_indices.setdefault(gid, []).append(i)
        group_to_labels.setdefault(gid, set()).add(int(y))

    groups_sorted = sorted(group_to_indices.keys())
    group_labels: List[int] = []
    for gid in groups_sorted:
        lbls = group_to_labels[gid]
        if len(lbls) != 1:
            raise RuntimeError(
                f"Group id '{gid}' contains multiple class labels {sorted(lbls)}. "
                "Fix --group_mode/--group_regex."
            )
        group_labels.append(next(iter(lbls)))

    g_idx = np.arange(len(groups_sorted))
    y_g = np.asarray(group_labels, dtype=int)

    try:
        sss1 = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
        train_g, temp_g = next(sss1.split(g_idx, y_g))

        val_within_temp = val_ratio / (val_ratio + test_ratio)
        y_temp = y_g[temp_g]
        sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_within_temp, random_state=seed + 1)
        val_rel, test_rel = next(sss2.split(temp_g, y_temp))
        val_g = temp_g[val_rel]
        test_g = temp_g[test_rel]
    except Exception as e:
        print(f"[WARN] Stratified group split failed ({e}). Falling back to non-stratified group split.")
        return build_splits_group_balanced(
            samples, group_ids, seed, train_ratio, val_ratio, test_ratio,
            balance_strategy, balance_level
        )

    def expand(group_indices: np.ndarray) -> List[int]:
        out: List[int] = []
        for gi in group_indices.tolist():
            gid = groups_sorted[int(gi)]
            out.extend(group_to_indices[gid])
        return out

    splits = SplitIndices(train=expand(train_g), val=expand(val_g), test=expand(test_g))
    
    # Apply balancing if requested
    if balance_strategy != 'none' and balance_level == 'train':
        # Balance only training set
        train_samples = [samples[i] for i in splits.train]
        train_labels = [samples[i][1] for i in splits.train]
        
        # Separate by class
        class_0_indices = [splits.train[i] for i, label in enumerate(train_labels) if label == 0]
        class_1_indices = [splits.train[i] for i, label in enumerate(train_labels) if label == 1]
        
        if balance_strategy == 'undersample':
            min_count = min(len(class_0_indices), len(class_1_indices))
            if len(class_0_indices) > min_count:
                class_0_indices = random.sample(class_0_indices, min_count)
            if len(class_1_indices) > min_count:
                class_1_indices = random.sample(class_1_indices, min_count)
        
        elif balance_strategy == 'oversample':
            max_count = max(len(class_0_indices), len(class_1_indices))
            if len(class_0_indices) < max_count:
                class_0_indices = random.choices(class_0_indices, k=max_count)
            if len(class_1_indices) < max_count:
                class_1_indices = random.choices(class_1_indices, k=max_count)
        
        # Combine and shuffle
        balanced_train = class_0_indices + class_1_indices
        random.shuffle(balanced_train)
        splits.train = balanced_train
        
        print(f"Balanced training set: {len(class_0_indices)} class 0, {len(class_1_indices)} class 1")
    
    return splits


def build_splits_random_stratified(
    labels: List[int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    balance_strategy: str = 'none',
    balance_level: str = 'train'
) -> SplitIndices:
    """Build random stratified splits with optional balancing."""
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.")

    idx = np.arange(len(labels))
    y = np.asarray(labels, dtype=int)

    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(sss1.split(idx, y))

    y_temp = y[temp_idx]
    val_within_temp = val_ratio / (val_ratio + test_ratio)

    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_within_temp, random_state=seed + 1)
    val_rel, test_rel = next(sss2.split(temp_idx, y_temp))

    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    splits = SplitIndices(train=train_idx.tolist(), val=val_idx.tolist(), test=test_idx.tolist())
    
    # Apply balancing if requested
    if balance_strategy != 'none' and balance_level == 'train':
        # Balance only training set
        train_labels = [labels[i] for i in splits.train]
        
        # Separate by class
        class_0_indices = [splits.train[i] for i, label in enumerate(train_labels) if label == 0]
        class_1_indices = [splits.train[i] for i, label in enumerate(train_labels) if label == 1]
        
        if balance_strategy == 'undersample':
            min_count = min(len(class_0_indices), len(class_1_indices))
            if len(class_0_indices) > min_count:
                class_0_indices = random.sample(class_0_indices, min_count)
            if len(class_1_indices) > min_count:
                class_1_indices = random.sample(class_1_indices, min_count)
        
        elif balance_strategy == 'oversample':
            max_count = max(len(class_0_indices), len(class_1_indices))
            if len(class_0_indices) < max_count:
                class_0_indices = random.choices(class_0_indices, k=max_count)
            if len(class_1_indices) < max_count:
                class_1_indices = random.choices(class_1_indices, k=max_count)
        
        # Combine and shuffle
        balanced_train = class_0_indices + class_1_indices
        random.shuffle(balanced_train)
        splits.train = balanced_train
        
        print(f"Balanced training set: {len(class_0_indices)} class 0, {len(class_1_indices)} class 1")
    
    return splits




# ============================================================================
# Advanced optional features
# ============================================================================
# NEW: Advanced Imports for Enhanced Features
# ============================================================================
try:
    import zarr
    import numcodecs
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ============================================================================
# Enhanced Data Loading with Smart Caching
# ============================================================================

class SmartImageCache:
    """Memory-mapped caching for faster data loading (3x speedup)."""
    
    def __init__(self, cache_dir=None, max_memory_gb=4, compression_level=3):
        self.cache_dir = Path(cache_dir or f"./image_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.max_memory_gb = max_memory_gb
        self.compression_level = compression_level
        self.cache = {}
        self.access_counts = Counter()
        self.cache_hits = 0
        self.cache_misses = 0
        self.disk_cache_enabled = ZARR_AVAILABLE
        
    def get(self, img_path: Path, transform=None, img_size=224):
        """Get image from cache or load and cache."""
        cache_key = f"{img_path.stem}_{img_size}_{hash(str(transform))}"
        
        # Check memory cache first (LRU)
        if cache_key in self.cache:
            self.cache_hits += 1
            self.access_counts[cache_key] += 1
            return self.cache[cache_key].clone()
        
        # Check disk cache if enabled
        if self.disk_cache_enabled:
            cache_path = self.cache_dir / f"{cache_key}.zarr"
            if cache_path.exists():
                try:
                    # Load from compressed Zarr format (10x smaller)
                    with zarr.open(cache_path, mode='r') as z:
                        img_array = z[:]
                    img = torch.from_numpy(img_array)
                    self._add_to_memory_cache(cache_key, img)
                    self.cache_hits += 1
                    return img.clone()
                except:
                    pass
        
        # Load original image (cache miss)
        self.cache_misses += 1
        img = default_loader(img_path)
        
        # Apply transforms
        if transform:
            img = transform(img)
        else:
            # Default preprocessing
            img = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])(img)
        
        # Cache in memory
        self._add_to_memory_cache(cache_key, img)
        
        # Save to disk cache for future runs
        if self.disk_cache_enabled and not transform:  # Don't cache random augmentations
            self._save_to_disk_cache(cache_key, img)
        
        return img.clone()
    
    def _add_to_memory_cache(self, key: str, tensor: torch.Tensor):
        """Add to memory cache with LRU eviction."""
        # Calculate current memory usage
        tensor_memory = tensor.numel() * tensor.element_size() / 1e9  # GB
        
        # Evict if over memory limit
        while len(self.cache) > 0 and self._get_cache_memory() + tensor_memory > self.max_memory_gb:
            # Remove least recently used
            lru_key = self.access_counts.most_common()[-1][0]
            del self.cache[lru_key]
            del self.access_counts[lru_key]
        
        self.cache[key] = tensor
        self.access_counts[key] = 1
    
    def _save_to_disk_cache(self, key: str, tensor: torch.Tensor):
        """Save tensor to disk in compressed format."""
        try:
            cache_path = self.cache_dir / f"{key}.zarr"
            z = zarr.open(cache_path, mode='w', shape=tensor.shape,
                         chunks=tensor.shape, dtype=tensor.numpy().dtype,
                         compressor=numcodecs.Blosc(cname='zstd', 
                                                   clevel=self.compression_level))
            z[:] = tensor.numpy()
        except Exception as e:
            print(f"Disk cache save failed for {key}: {e}")
    
    def _get_cache_memory(self):
        """Calculate current cache memory usage in GB."""
        return sum(t.numel() * t.element_size() for t in self.cache.values()) / 1e9
    
    def get_stats(self):
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': self.cache_hits / total if total > 0 else 0,
            'memory_mb': self._get_cache_memory() * 1000,
            'unique_cached': len(self.cache),
            'disk_cache_enabled': self.disk_cache_enabled
        }
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_counts.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class CachedPathLabelDataset(Dataset):
    """Enhanced dataset with smart caching."""
    
    def __init__(self, samples: List[Tuple[Path, int]], transform=None,
                 cache_dir=None, img_size=224, use_cache=True, max_memory_gb=4):
        self.samples = [(str(p), int(y)) for p, y in samples]
        self.transform = transform
        self.img_size = img_size
        self.use_cache = use_cache
        self.cache = SmartImageCache(cache_dir=cache_dir, max_memory_gb=max_memory_gb) if use_cache else None
        self.loader = default_loader
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        path = Path(path)
        
        # Use cache for non-random transforms (validation/test)
        if self.use_cache and self.cache and not self._has_random_transform():
            img = self.cache.get(path, self.transform, self.img_size)
        else:
            # Load fresh (for training with augmentation)
            img = self.loader(path)
            if self.transform:
                img = self.transform(img)
        
        return img, y
    
    def _has_random_transform(self):
        """Check if transform includes random operations."""
        if not self.transform:
            return False
        trans_str = str(self.transform)
        random_keywords = ['Random', 'random', 'RandAugment', 'AutoAugment']
        return any(k in trans_str for k in random_keywords)
    
    def get_cache_stats(self):
        """Get cache statistics if using cache."""
        return self.cache.get_stats() if self.cache else {}

# ============================================================================
# Enhanced Learning Rate Scheduler with Gradient Awareness
# ============================================================================

class GradientAwareScheduler:
    """
    Adaptive learning rate scheduler based on gradient statistics.
    Dynamically adjusts LR based on gradient norm, loss landscape, and improvement.
    """
    
    def __init__(self, optimizer, initial_lr, warmup_epochs=5,
                 patience=3, cooldown=5, min_lr=1e-6,
                 max_lr=1e-2, mode='aggressive', factor=None,
                 increase_factor=None, grad_threshold=None, **_ignored_kwargs):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.mode = mode
        
        self.gradient_history = []
        self.loss_history = []
        self.lr_history = []
        self.no_improve = 0
        self.cooldown_counter = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Mode-specific parameters
        self.params = {
            'aggressive': {'increase_factor': 1.2, 'decrease_factor': 0.5, 'grad_threshold': 0.1},
            'conservative': {'increase_factor': 1.05, 'decrease_factor': 0.8, 'grad_threshold': 0.01},
            'adaptive': {'increase_factor': 1.1, 'decrease_factor': 0.7, 'grad_threshold': 0.05}
        }.get(mode, {'increase_factor': 1.1, 'decrease_factor': 0.7, 'grad_threshold': 0.05})
        # Allow external overrides (compat with ReduceLROnPlateau-style args)
        if factor is not None:
            # Plateaus typically multiply LR by `factor` (<1) when reducing
            try:
                self.params['decrease_factor'] = float(factor)
            except Exception:
                pass
        if increase_factor is not None:
            try:
                self.params['increase_factor'] = float(increase_factor)
            except Exception:
                pass
        if grad_threshold is not None:
            try:
                self.params['grad_threshold'] = float(grad_threshold)
            except Exception:
                pass

    
    def step(self, current_loss, gradient_norm=None, gradient_stats=None, **kwargs):
        """Update learning rate based on training dynamics."""
        # Accept legacy/alternate keyword used elsewhere in the codebase
        if gradient_norm is None and 'grad_norm' in kwargs:
            gradient_norm = kwargs.get('grad_norm')
        self.epoch += 1
        self.loss_history.append(current_loss)
        
        if gradient_norm is not None:
            self.gradient_history.append(gradient_norm)
        
        # Warmup phase
        if self.epoch <= self.warmup_epochs:
            lr = self.initial_lr * (self.epoch / self.warmup_epochs)
        else:
            # Adaptive phase
            lr = self._compute_adaptive_lr(current_loss, gradient_norm, gradient_stats)
        
        # Apply LR to all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.lr_history.append(lr)
        return lr
    
    def _compute_adaptive_lr(self, current_loss, gradient_norm, gradient_stats):
        """Compute adaptive learning rate."""
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Check for improvement
        loss_improved = current_loss < self.best_loss - 1e-4
        
        if loss_improved:
            self.best_loss = current_loss
            self.no_improve = 0
            
            # Consider increasing LR if gradients are healthy
            if gradient_norm and gradient_norm < self.params['grad_threshold'] * 10:
                lr_factor = self.params['increase_factor']
            else:
                lr_factor = 1.0
        else:
            self.no_improve += 1
            
            # If gradient is large but no improvement, reduce LR more
            if gradient_norm and gradient_norm > 5.0:
                lr_factor = self.params['decrease_factor'] * 0.7
            else:
                lr_factor = self.params['decrease_factor']
        
        # Apply patience logic
        if self.no_improve >= self.patience:
            lr_factor *= 0.5  # More aggressive reduction
            self.no_improve = 0
            self.cooldown_counter = self.cooldown
        
        # Cooldown period
        if self.cooldown_counter > 0:
            lr_factor = 1.0
            self.cooldown_counter -= 1
        
        # Apply bounds and stability check
        new_lr = current_lr * lr_factor
        new_lr = max(self.min_lr, min(new_lr, self.max_lr))
        
        # Ensure we don't change too drastically
        if abs(new_lr - current_lr) / current_lr > 0.5:  # Max 50% change
            new_lr = current_lr * (1 + 0.5 * np.sign(new_lr - current_lr))
        
        return new_lr
    
    def get_state(self):
        """Get scheduler state."""
        return {
            'epoch': self.epoch,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'best_loss': self.best_loss,
            'no_improve': self.no_improve,
            'loss_history': self.loss_history[-10:],
            'lr_history': self.lr_history[-10:],
            'gradient_history': self.gradient_history[-10:] if self.gradient_history else []
        }
    
    def plot_lr_schedule(self, save_path=None):
        """Plot learning rate schedule."""
        if len(self.lr_history) < 2:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot LR
        ax1.plot(self.lr_history, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Learning Rate')
        ax1.set_title('Learning Rate Schedule')
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        if self.loss_history:
            ax2.plot(self.loss_history, 'r-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Loss History')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# ============================================================================
# Gradient Monitor for Training Stability
# ============================================================================

class GradientMonitor:
    """Monitor gradient flow and statistics for debugging and optimization."""
    
    def __init__(self, model):
        self.model = model
        self.gradient_history = []
        self.hook_handles = []
        self.layer_stats = {}
        
        # Register hooks on all trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(
                    lambda grad, name=name: self._gradient_hook(grad, name)
                )
                self.hook_handles.append(handle)
        
        self.reset_epoch()
    
    def _gradient_hook(self, grad, name):
        """Hook to capture gradient statistics."""
        if grad is not None:
            with torch.no_grad():
                grad_norm = grad.norm().item()
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()
                grad_min = grad.min().item()
                grad_max = grad.max().item()
                
                self.layer_stats[name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std,
                    'min': grad_min,
                    'max': grad_max,
                    'zero_fraction': (grad == 0).float().mean().item()
                }
    
    def reset_epoch(self):
        """Reset statistics for new epoch."""
        self.layer_stats = {}
        self.epoch_stats = []
    
    def record_batch(self):
        """Record batch statistics."""
        if self.layer_stats:
            self.epoch_stats.append(self.layer_stats.copy())
    
    def get_epoch_statistics(self):
        """Get comprehensive gradient statistics for epoch."""
        if not self.epoch_stats:
            return {}
        
        all_norms = []
        all_means = []
        all_stds = []
        zero_fractions = []
        
        for batch_stats in self.epoch_stats:
            for layer_stats in batch_stats.values():
                all_norms.append(layer_stats['norm'])
                all_means.append(layer_stats['mean'])
                all_stds.append(layer_stats['std'])
                zero_fractions.append(layer_stats['zero_fraction'])
        
        return {
            'grad_norm_mean': np.mean(all_norms) if all_norms else 0,
            'grad_norm_std': np.std(all_norms) if len(all_norms) > 1 else 0,
            'grad_norm_max': max(all_norms) if all_norms else 0,
            'grad_norm_min': min(all_norms) if all_norms else 0,
            'grad_mean_abs': np.mean(np.abs(all_means)) if all_means else 0,
            'grad_std_mean': np.mean(all_stds) if all_stds else 0,
            'zero_fraction_mean': np.mean(zero_fractions) if zero_fractions else 0,
            'vanishing_layers': sum(1 for n in all_norms if n < 1e-7),
            'exploding_layers': sum(1 for n in all_norms if n > 10.0),
        }
    
    def get_problematic_layers(self, threshold_norm_low=1e-6, threshold_norm_high=10.0):
        """Identify layers with gradient issues."""
        problematic = []
        if not self.epoch_stats:
            return problematic
        
        # Aggregate across batches
        layer_aggregates = {}
        for batch_stats in self.epoch_stats:
            for name, stats in batch_stats.items():
                if name not in layer_aggregates:
                    layer_aggregates[name] = []
                layer_aggregates[name].append(stats['norm'])
        
        # Check each layer
        for name, norms in layer_aggregates.items():
            avg_norm = np.mean(norms)
            if avg_norm < threshold_norm_low:
                problematic.append((name, 'vanishing', avg_norm))
            elif avg_norm > threshold_norm_high:
                problematic.append((name, 'exploding', avg_norm))
        
        return problematic
    
    def plot_gradient_distribution(self, save_path=None):
        """Visualize gradient distribution."""
        if not self.epoch_stats:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Collect all gradients
        all_gradients = []
        layer_names = []
        layer_norms = []
        
        for batch_stats in self.epoch_stats:
            for name, stats in batch_stats.items():
                if name not in layer_names:
                    layer_names.append(name)
                    layer_norms.append(stats['norm'])
        
        # 1. Gradient norms by layer
        if layer_norms:
            axes[0, 0].bar(range(len(layer_names)), layer_norms[:20], alpha=0.7)
            axes[0, 0].set_title('Gradient Norms by Layer (Top 20)')
            axes[0, 0].set_ylabel('Norm')
            axes[0, 0].axhline(y=1e-6, color='r', linestyle='--', alpha=0.5, label='Vanishing')
            axes[0, 0].axhline(y=10.0, color='orange', linestyle='--', alpha=0.5, label='Exploding')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Histogram of gradient values
        if all_gradients:
            axes[0, 1].hist(all_gradients, bins=100, alpha=0.7, log=True)
            axes[0, 1].set_title('Gradient Value Distribution')
            axes[0, 1].set_xlabel('Gradient Value')
            axes[0, 1].set_ylabel('Frequency (log)')
        
        # 3. Time series of gradient norm
        if len(self.gradient_history) > 1:
            history_norms = [h.get('grad_norm_mean', 0) for h in self.gradient_history]
            axes[1, 0].plot(history_norms, 'b-', alpha=0.7)
            axes[1, 0].set_title('Average Gradient Norm over Time')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Norm')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Layer-wise gradient statistics
        stats = self.get_epoch_statistics()
        if stats:
            metric_names = ['grad_norm_mean', 'grad_mean_abs', 'zero_fraction_mean']
            metric_values = [stats.get(k, 0) for k in metric_names]
            axes[1, 1].bar(metric_names, metric_values, alpha=0.7)
            axes[1, 1].set_title('Aggregate Gradient Statistics')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_report(self, save_path):
        """Save gradient analysis report."""
        stats = self.get_epoch_statistics()
        problematic = self.get_problematic_layers()
        
        report = {
            'statistics': stats,
            'problematic_layers': problematic,
            'total_layers_monitored': len(set().union(*[d.keys() for d in self.epoch_stats]))
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def __del__(self):
        """Clean up hooks."""
        for handle in self.hook_handles:
            handle.remove()

# ============================================================================
# Bayesian Ensemble with Uncertainty Quantification
# ============================================================================

class BayesianEnsemble:
    """
    Bayesian model ensemble with uncertainty quantification.
    Supports MC Dropout, Deep Ensembles, and Snapshot Ensembles.
    """
    
    def __init__(self, models=None, method='mc_dropout', num_samples=30,
                 temperature_scaling=True, uncertainty_type='both'):
        """
        Args:
            models: List of models or model checkpoints
            method: 'mc_dropout', 'deep_ensemble', 'snapshot_ensemble'
            num_samples: Number of forward passes for MC Dropout
            temperature_scaling: Calibrate predictions with temperature scaling
            uncertainty_type: 'epistemic', 'aleatoric', or 'both'
        """
        self.models = models or []
        self.method = method
        self.num_samples = num_samples
        self.temperature_scaling = temperature_scaling
        self.uncertainty_type = uncertainty_type
        self.temperature = 1.0
        self.calibrated = False
        
    def add_model(self, model, weight=1.0, enable_dropout=False):
        """Add a model to the ensemble."""
        model_info = {
            'model': model,
            'weight': weight,
            'enable_dropout': enable_dropout,
            'predictions': None,
            'uncertainty': None
        }
        self.models.append(model_info)
    
    def predict_with_uncertainty(self, dataloader, device='cuda'):
        """Predict with uncertainty quantification."""
        all_predictions = []
        all_uncertainties = []
        
        for model_info in self.models:
            model = model_info['model']
            model.eval()
            
            if self.method == 'mc_dropout' and model_info['enable_dropout']:
                # Monte Carlo Dropout
                predictions, uncertainties = self._mc_dropout_predict(
                    model, dataloader, device
                )
            else:
                # Standard prediction
                predictions, uncertainties = self._standard_predict(
                    model, dataloader, device
                )
            
            model_info['predictions'] = predictions
            model_info['uncertainty'] = uncertainties
            
            all_predictions.append(predictions)
            all_uncertainties.append(uncertainties)
        
        # Combine predictions with weighting
        ensemble_pred, ensemble_uncertainty = self._combine_predictions(
            all_predictions, all_uncertainties
        )
        
        # Temperature scaling calibration
        if self.temperature_scaling and hasattr(dataloader.dataset, 'targets'):
            # This would need validation labels for calibration
            pass
        
        return ensemble_pred, ensemble_uncertainty
    
    def _mc_dropout_predict(self, model, dataloader, device):
        """MC Dropout prediction with multiple forward passes."""
        # Enable dropout at test time
        self._enable_dropout(model, enable=True)
        
        all_mc_preds = []
        for _ in range(self.num_samples):
            batch_preds = []
            for x, _ in dataloader:
                x = x.to(device)
                with torch.no_grad():
                    output = model(x)
                    probs = torch.softmax(output, dim=1)
                    batch_preds.append(probs.cpu().numpy())
            
            all_mc_preds.append(np.concatenate(batch_preds, axis=0))
        
        # Disable dropout
        self._enable_dropout(model, enable=False)
        
        # Statistics over samples
        all_mc_preds = np.stack(all_mc_preds)  # [num_samples, num_data, num_classes]
        mean_pred = all_mc_preds.mean(axis=0)
        std_pred = all_mc_preds.std(axis=0)
        
        # Calculate uncertainties
        uncertainties = self._calculate_uncertainties(mean_pred, std_pred, all_mc_preds)
        
        return mean_pred, uncertainties
    
    def _standard_predict(self, model, dataloader, device):
        """Standard single forward pass prediction."""
        all_preds = []
        for x, _ in dataloader:
            x = x.to(device)
            with torch.no_grad():
                output = model(x)
                probs = torch.softmax(output, dim=1)
                all_preds.append(probs.cpu().numpy())
        
        predictions = np.concatenate(all_preds, axis=0)
        
        # For standard prediction, only aleatoric uncertainty
        eps = 1e-10
        entropy = -np.sum(predictions * np.log(predictions + eps), axis=1)
        
        uncertainties = {
            'epistemic': np.zeros_like(entropy),
            'aleatoric': entropy,
            'total': entropy
        }
        
        return predictions, uncertainties
    
    def _calculate_uncertainties(self, mean_pred, std_pred, mc_preds):
        """Calculate different types of uncertainty."""
        eps = 1e-10
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = std_pred.mean(axis=1)  # Average std over classes
        
        # Aleatoric uncertainty (data uncertainty)
        entropy = -np.sum(mean_pred * np.log(mean_pred + eps), axis=1)
        aleatoric = entropy
        
        # Mutual information (disagreement between samples)
        if mc_preds.shape[0] > 1:
            mean_entropy = entropy
            mean_pred_entropy = -np.sum(mean_pred * np.log(mean_pred + eps), axis=1)
            mutual_info = mean_entropy - mean_pred_entropy
        else:
            mutual_info = np.zeros_like(epistemic)
        
        return {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'mutual_info': mutual_info,
            'total': epistemic + aleatoric,
            'std_per_class': std_pred
        }
    
    def _enable_dropout(self, model, enable=True):
        """Enable or disable dropout layers."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train() if enable else module.eval()
    
    def _combine_predictions(self, all_predictions, all_uncertainties):
        """Combine predictions from multiple models."""
        weights = np.array([m['weight'] for m in self.models])
        weights = weights / weights.sum()
        
        # Weighted average of predictions
        ensemble_pred = np.zeros_like(all_predictions[0])
        ensemble_epistemic = np.zeros(all_predictions[0].shape[0])
        ensemble_aleatoric = np.zeros(all_predictions[0].shape[0])
        
        for idx, (pred, unc) in enumerate(zip(all_predictions, all_uncertainties)):
            w = weights[idx]
            ensemble_pred += w * pred
            ensemble_epistemic += w * unc['epistemic']
            ensemble_aleatoric += w * unc['aleatoric']
        
        ensemble_uncertainty = {
            'epistemic': ensemble_epistemic,
            'aleatoric': ensemble_aleatoric,
            'total': ensemble_epistemic + ensemble_aleatoric,
            'predictive_variance': np.var(all_predictions, axis=0).mean(axis=1)
        }
        
        return ensemble_pred, ensemble_uncertainty
    
    def calibrate(self, val_probs, val_labels, max_iter=100):
        """Calibrate ensemble using temperature scaling."""
        logits = torch.tensor(np.log(val_probs + 1e-10))
        labels = torch.tensor(val_labels)
        
        # Optimize temperature parameter
        temperature = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
        
        def eval():
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(logits / temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        self.temperature = temperature.item()
        self.calibrated = True
        
        # Apply temperature scaling
        calibrated_probs = torch.softmax(logits / self.temperature, dim=1)
        return calibrated_probs.numpy()
    
    def get_confidence_intervals(self, alpha=0.05):
        """Get confidence intervals for predictions."""
        if not self.models or not self.models[0]['uncertainty']:
            return None
        
        # Simple approximation using total uncertainty
        total_uncertainty = self.models[0]['uncertainty']['total']
        z_score = 1.96  # For 95% CI
        
        # Assuming normal distribution (approximation)
        ci_width = z_score * np.sqrt(total_uncertainty)
        
        return {
            'lower_bound': np.maximum(0, -ci_width),
            'upper_bound': np.minimum(1, ci_width),
            'width': ci_width,
            'confidence_level': 1 - alpha
        }

# ============================================================================
# Model Compression for Deployment
# ============================================================================

class ModelCompressor:
    """Model compression with pruning and quantization."""
    
    def __init__(self, method='pruning', pruning_amount=0.3, 
                 quantization_type='int8', calibration_samples=100):
        self.method = method
        self.pruning_amount = pruning_amount
        self.quantization_type = quantization_type
        self.calibration_samples = calibration_samples
        
    def prune_model(self, model, pruning_type='l1_unstructured'):
        """Apply pruning to the model."""
        model = model.clone() if hasattr(model, 'clone') else copy.deepcopy(model)
        
        # Identify layers to prune
        layers_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Don't prune classification head
                if 'classifier' not in name.lower() and 'fc' not in name.lower():
                    layers_to_prune.append((module, 'weight'))
        
        # Apply pruning
        if pruning_type == 'l1_unstructured':
            prune.global_unstructured(
                layers_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.pruning_amount
            )
        
        # Make pruning permanent
        for module, param_name in layers_to_prune:
            prune.remove(module, param_name)
        
        # Calculate sparsity
        total_params = sum(p.numel() for p in model.parameters())
        zero_params = sum((p == 0).sum().item() for p in model.parameters())
        sparsity = zero_params / total_params if total_params > 0 else 0
        
        print(f"Pruning applied: {sparsity:.2%} sparsity ({zero_params:,}/{total_params:,} zero params)")
        
        return model, {'sparsity': sparsity, 'zero_params': zero_params, 'total_params': total_params}
    
    def quantize_model(self, model, calibration_loader=None):
        """Apply quantization to the model."""
        try:
            # Prepare model for quantization
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Fuse layers if possible
            if hasattr(model, 'features'):
                torch.quantization.fuse_modules(model, [['features.0', 'features.1']], inplace=True)
            
            # Prepare and calibrate
            model_prepared = torch.quantization.prepare(model)
            
            if calibration_loader:
                # Calibrate with data
                samples = 0
                with torch.no_grad():
                    for data, _ in calibration_loader:
                        model_prepared(data[:32])
                        samples += 32
                        if samples >= self.calibration_samples:
                            break
            
            # Convert to quantized model
            model_quantized = torch.quantization.convert(model_prepared)
            
            print(f"Model quantized to {self.quantization_type}")
            return model_quantized, {'quantization': self.quantization_type, 'calibrated': calibration_loader is not None}
            
        except Exception as e:
            print(f"Quantization failed: {e}")
            return model, {'quantization': 'failed', 'error': str(e)}
    
    def compress(self, model, calibration_loader=None):
        """Full compression pipeline."""
        compression_stats = {}
        
        # Pruning
        if 'pruning' in self.method.lower():
            model, prune_stats = self.prune_model(model)
            compression_stats.update(prune_stats)
        
        # Quantization
        if 'quantization' in self.method.lower():
            model, quant_stats = self.quantize_model(model, calibration_loader)
            compression_stats.update(quant_stats)
        
        # Calculate compression ratio
        original_size = self._estimate_model_size(model, compressed=False)
        compressed_size = self._estimate_model_size(model, compressed=True)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        compression_stats.update({
            'original_size_mb': original_size,
            'compressed_size_mb': compressed_size,
            'compression_ratio': compression_ratio,
            'method': self.method
        })
        
        return model, compression_stats
    
    def _estimate_model_size(self, model, compressed=False):
        """Estimate model size in MB."""
        total_params = sum(p.numel() for p in model.parameters())
        
        if compressed:
            # Quantized models use less memory per parameter
            if hasattr(model, 'qconfig') and model.qconfig is not None:
                bytes_per_param = 1  # int8
            else:
                bytes_per_param = 4  # float32
        else:
            bytes_per_param = 4  # float32
        
        size_mb = (total_params * bytes_per_param) / (1024 * 1024)
        return size_mb
    
    def export_to_onnx(self, model, input_shape, output_path):
        """Export model to ONNX format for deployment."""
        try:
            model.eval()
            dummy_input = torch.randn(input_shape)
            
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            print(f"Model exported to ONNX: {output_path}")
            return True
        except Exception as e:
            print(f"ONNX export failed: {e}")
            return False

# ============================================================================
# Statistical Model Selection
# ============================================================================


# ============================================================================
# Four-way paired significance testing (McNemar, permutation-F1, Wilcoxon,
# DeLong AUC) and prediction-bundle export, added to support the manuscript's
# "Statistical significance of pairwise differences" analysis. This is a
# separate, independent addition alongside StatisticalModelSelector above,
# which is unchanged and continues to serve its existing --statistical_testing
# role. This block additionally runs when --statistical_testing is enabled.
# ============================================================================

def _sig_validate_store(model_pred_store: list) -> tuple:
    if len(model_pred_store) < 2:
        raise ValueError("Need at least two models for significance testing.")
    normalized: Dict[str, Dict[str, Any]] = {}
    y0 = None
    for record in model_pred_store:
        model = str(record["model"])
        y_true = np.asarray(record["y_true"]).reshape(-1)
        y_pred = np.asarray(record["y_pred"]).reshape(-1)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(f"{model}: y_true/y_pred length mismatch.")
        y_score = record.get("y_score")
        if y_score is not None:
            y_score = np.asarray(y_score).reshape(-1)
            if y_score.shape[0] != y_true.shape[0]:
                raise ValueError(f"{model}: y_true/y_score length mismatch.")
        if y0 is None:
            y0 = y_true
        elif not np.array_equal(y0, y_true):
            raise ValueError("All models must share the same y_true for paired significance testing.")
        normalized[model] = {"y_true": y_true, "y_pred": y_pred, "y_score": y_score}
    return y0, normalized


def _sig_mcnemar_test(y_true, pred1, pred2):
    from scipy.stats import chi2, binomtest
    b = int(np.sum((pred1 != y_true) & (pred2 == y_true)))
    c = int(np.sum((pred1 == y_true) & (pred2 != y_true)))
    if b + c < 25:
        result = binomtest(min(b, c), b + c, 0.5, alternative="two-sided")
        return float(result.pvalue), float(min(b, c))
    stat = ((abs(b - c) - 1) ** 2) / max(1, (b + c))
    return float(1 - chi2.cdf(stat, df=1)), float(stat)


def _sig_permutation_f1_test(y_true, pred1, pred2, n_permutations, rng):
    observed = float(f1_score(y_true, pred1, zero_division=0) - f1_score(y_true, pred2, zero_division=0))
    diffs = np.empty(int(n_permutations), dtype=float)
    n = y_true.shape[0]
    for i in range(int(n_permutations)):
        swap = rng.random(n) > 0.5
        p1 = np.where(swap, pred1, pred2)
        p2 = np.where(swap, pred2, pred1)
        diffs[i] = f1_score(y_true, p1, zero_division=0) - f1_score(y_true, p2, zero_division=0)
    p = float((np.sum(np.abs(diffs) >= abs(observed)) + 1) / (len(diffs) + 1))
    return p, observed


def _sig_wilcoxon_score_test(score1, score2):
    from scipy.stats import wilcoxon
    try:
        stat, p = wilcoxon(score1, score2, zero_method="wilcox", correction=False)
        return float(p), float(stat)
    except Exception:
        return None, None


def _sig_compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def _sig_fast_delong(scores, label_1_count):
    m = label_1_count
    n = scores.shape[1] - m
    positive = scores[:, :m]
    negative = scores[:, m:]
    k = scores.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _sig_compute_midrank(positive[r, :])
        ty[r, :] = _sig_compute_midrank(negative[r, :])
        tz[r, :] = _sig_compute_midrank(scores[r, :])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def _sig_delong_auc_test(y_true, score1, score2):
    import math
    from scipy.stats import norm
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) != 2:
        return None, None
    order = np.argsort(-y_true)
    label_1_count = int(np.sum(y_true))
    stacked = np.vstack([score1, score2])[:, order]
    aucs, cov = _sig_fast_delong(stacked, label_1_count)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0:
        return None, diff
    z = diff / math.sqrt(var)
    p = float(2 * (1 - norm.cdf(abs(z))))
    return p, diff


def _sig_apply_correction(pairs, alpha, method):
    method = str(method).lower()
    if not pairs:
        return
    if method == "none":
        for pair in pairs:
            pair["adjusted_alpha"] = float(alpha)
            pair["significant"] = bool(pair["p_value"] is not None and pair["p_value"] < alpha)
        return
    if method == "bonferroni":
        adjusted = float(alpha / len(pairs))
        for pair in pairs:
            pair["adjusted_alpha"] = adjusted
            pair["significant"] = bool(pair["p_value"] is not None and pair["p_value"] < adjusted)
        return
    if method == "holm":
        valid = sorted([p for p in pairs if p["p_value"] is not None], key=lambda x: x["p_value"])
        n = len(valid)
        for idx, pair in enumerate(valid):
            adjusted = float(alpha / max(1, n - idx))
            pair["adjusted_alpha"] = adjusted
            pair["significant"] = bool(pair["p_value"] < adjusted)
        for pair in pairs:
            if "adjusted_alpha" not in pair:
                pair["adjusted_alpha"] = None
                pair["significant"] = False
        return
    raise ValueError(f"Unknown correction method: {method}")


def build_four_way_significance_report(model_pred_store: list, alpha: float = 0.05,
                                        correction: str = "bonferroni", permutations: int = 2000,
                                        random_seed: int = 42) -> dict:
    """Four complementary paired tests (McNemar, permutation-F1, Wilcoxon,
    DeLong AUC) applied to every pairwise model comparison, matching the
    manuscript's "Statistical analysis" / "Statistical significance of
    pairwise differences" methodology. Independent of StatisticalModelSelector."""
    y_true, store = _sig_validate_store(model_pred_store)
    names = list(store.keys())
    rng = np.random.default_rng(random_seed)

    model_metrics: Dict[str, Dict[str, Any]] = {}
    for name in names:
        pred = store[name]["y_pred"]
        score = store[name]["y_score"]
        row: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, pred)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        }
        if score is not None and len(np.unique(y_true)) == 2:
            row["auc"] = float(roc_auc_score(y_true, score))
        model_metrics[name] = row

    pairwise: Dict[str, Any] = {}
    family_results: Dict[str, list] = {"mcnemar": [], "permutation_f1": [], "wilcoxon_scores": [], "delong_auc": []}

    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i >= j:
                continue
            pred1, pred2 = store[name1]["y_pred"], store[name2]["y_pred"]
            score1, score2 = store[name1]["y_score"], store[name2]["y_score"]
            key = f"{name1}_vs_{name2}"
            p_mcn, stat_mcn = _sig_mcnemar_test(y_true, pred1, pred2)
            p_perm, stat_perm = _sig_permutation_f1_test(y_true, pred1, pred2, permutations, rng)
            p_wil, stat_wil, p_del, stat_del = None, None, None, None
            if score1 is not None and score2 is not None:
                p_wil, stat_wil = _sig_wilcoxon_score_test(score1, score2)
                p_del, stat_del = _sig_delong_auc_test(y_true, score1, score2)
            pair_entry = {
                "models": [name1, name2],
                "metric_summary": {name1: model_metrics[name1], name2: model_metrics[name2]},
                "tests": {
                    "mcnemar": {"p_value": p_mcn, "statistic": stat_mcn},
                    "permutation_f1": {"p_value": p_perm, "statistic": stat_perm},
                    "wilcoxon_scores": {"p_value": p_wil, "statistic": stat_wil},
                    "delong_auc": {"p_value": p_del, "statistic": stat_del},
                },
            }
            pairwise[key] = pair_entry
            family_results["mcnemar"].append({"pair": key, "p_value": p_mcn, "statistic": stat_mcn})
            family_results["permutation_f1"].append({"pair": key, "p_value": p_perm, "statistic": stat_perm})
            family_results["wilcoxon_scores"].append({"pair": key, "p_value": p_wil, "statistic": stat_wil})
            family_results["delong_auc"].append({"pair": key, "p_value": p_del, "statistic": stat_del})

    for family, rows in family_results.items():
        _sig_apply_correction(rows, alpha, correction)
        indexed = {row["pair"]: row for row in rows}
        for pair_key, pair_entry in pairwise.items():
            if pair_key in indexed:
                pair_entry["tests"][family]["adjusted_alpha"] = indexed[pair_key]["adjusted_alpha"]
                pair_entry["tests"][family]["significant"] = indexed[pair_key]["significant"]

    summary = {family: {"n_pairs": len(rows), "significant_pairs": int(sum(1 for r in rows if bool(r.get("significant"))))}
               for family, rows in family_results.items()}

    return {
        "metadata": {"alpha": alpha, "correction": correction, "permutations": permutations, "n_models": len(names), "n_pairs": len(pairwise)},
        "model_metrics": model_metrics,
        "pairwise": pairwise,
        "summary": summary,
    }


def export_prediction_bundle(seed_dir, model_pred_store: list, filename: str = "prediction_bundle.npz"):
    """Exports y_true/pred__X/score__X/files for every model to an npz bundle,
    for post-hoc analysis via scripts/extract_prediction_bundles.py."""
    out_dir = Path(seed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, store = _sig_validate_store(model_pred_store)
    out_path = out_dir / filename
    arrays: Dict[str, Any] = {}
    any_model = next(iter(store))
    arrays["y_true"] = store[any_model]["y_true"]
    arrays["model_names"] = np.array(list(store.keys()), dtype=object)
    for model, rec in store.items():
        safe_model = re.sub(r"[^0-9A-Za-z_]+", "_", model)
        arrays[f"pred__{safe_model}"] = rec["y_pred"]
        if rec["y_score"] is not None:
            arrays[f"score__{safe_model}"] = rec["y_score"]
    for record in model_pred_store:
        files = record.get("files")
        if files:
            n = len(arrays["y_true"])
            if len(files) == n:
                arrays["files"] = np.array([str(f) for f in files], dtype=object)
            break
    np.savez_compressed(out_path, **arrays)
    return out_path


class StatisticalModelSelector:
    """Statistical model selection with hypothesis testing."""
    
    def __init__(self, alpha=0.05, test_type='mcnemar', correction='bonferroni'):
        self.alpha = alpha
        self.test_type = test_type
        self.correction = correction
        self.comparisons = []
        
    def compare_models(self, y_true, predictions_dict, metric='f1'):
        """Compare models using statistical tests."""
        model_names = list(predictions_dict.keys())
        results = {}
        
        # Calculate metrics for each model
        model_metrics = {}
        for name, preds in predictions_dict.items():
            if metric == 'f1':
                score = f1_score(y_true, preds, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, preds)
            elif metric == 'auc':
                # For AUC, need probabilities
                score = roc_auc_score(y_true, preds) if len(np.unique(y_true)) == 2 else 0.5
            else:
                score = f1_score(y_true, preds, zero_division=0)
            model_metrics[name] = score
        
        # Pairwise comparisons
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i >= j:
                    continue
                
                preds1 = predictions_dict[name1]
                preds2 = predictions_dict[name2]
                
                if self.test_type == 'mcnemar':
                    p_value, test_stat = self._mcnemar_test(y_true, preds1, preds2)
                elif self.test_type == 'wilcoxon':
                    p_value, test_stat = self._wilcoxon_test(y_true, preds1, preds2)
                else:
                    p_value, test_stat = self._permutation_test(y_true, preds1, preds2)
                
                comparison = {
                    'model1': name1,
                    'model2': name2,
                    'p_value': p_value,
                    'significant': p_value < self.alpha,
                    'test_statistic': test_stat,
                    'test_type': self.test_type,
                    f'{metric}_1': model_metrics[name1],
                    f'{metric}_2': model_metrics[name2],
                    'difference': model_metrics[name1] - model_metrics[name2]
                }
                
                key = f"{name1}_vs_{name2}"
                results[key] = comparison
                self.comparisons.append(comparison)
        
        # Apply multiple testing correction
        if self.correction == 'bonferroni':
            self._apply_bonferroni_correction()
        elif self.correction == 'holm':
            self._apply_holm_correction()
        
        return results
    
    def _mcnemar_test(self, y_true, preds1, preds2):
        """McNemar's test for paired nominal data."""
        from scipy.stats import chi2
        
        # Contingency table
        a = np.sum((preds1 == y_true) & (preds2 == y_true))
        b = np.sum((preds1 != y_true) & (preds2 == y_true))
        c = np.sum((preds1 == y_true) & (preds2 != y_true))
        d = np.sum((preds1 != y_true) & (preds2 != y_true))
        
        if b + c < 25:
            # Exact binomial test for small samples
            from scipy.stats import binomtest
            result = binomtest(min(b, c), b + c, 0.5, alternative='two-sided')
            p_value = result.pvalue
            test_stat = min(b, c)
        else:
            # Chi-squared test with continuity correction
            chi2_stat = ((abs(b - c) - 1) ** 2) / (b + c)
            p_value = 1 - chi2.cdf(chi2_stat, df=1)
            test_stat = chi2_stat
        
        return p_value, test_stat
    
    def _wilcoxon_test(self, y_true, preds1, preds2):
        """Wilcoxon signed-rank test for paired samples."""
        from scipy.stats import wilcoxon
        
        # Compare prediction scores (e.g., probabilities)
        scores1 = preds1 if preds1.ndim == 1 else preds1[:, 1]
        scores2 = preds2 if preds2.ndim == 1 else preds2[:, 1]
        
        try:
            test_stat, p_value = wilcoxon(scores1, scores2)
        except:
            p_value = 1.0
            test_stat = 0
        
        return p_value, test_stat
    
    def _permutation_test(self, y_true, preds1, preds2, n_permutations=1000):
        """Permutation test for model comparison."""
        # Observed difference
        f1_1 = f1_score(y_true, preds1, zero_division=0)
        f1_2 = f1_score(y_true, preds2, zero_division=0)
        observed_diff = f1_1 - f1_2
        
        # Permutation distribution
        null_diffs = []
        n = len(y_true)
        
        for _ in range(n_permutations):
            # Randomly swap predictions between models
            swap_mask = np.random.rand(n) > 0.5
            perm_preds1 = np.where(swap_mask, preds1, preds2)
            perm_preds2 = np.where(swap_mask, preds2, preds1)
            
            perm_f1_1 = f1_score(y_true, perm_preds1, zero_division=0)
            perm_f1_2 = f1_score(y_true, perm_preds2, zero_division=0)
            null_diffs.append(perm_f1_1 - perm_f1_2)
        
        # p-value = proportion of null differences as extreme as observed
        p_value = (np.sum(np.abs(null_diffs) >= np.abs(observed_diff)) + 1) / (n_permutations + 1)
        
        return p_value, observed_diff
    
    def _apply_bonferroni_correction(self):
        """Apply Bonferroni correction for multiple testing."""
        n_tests = len(self.comparisons)
        corrected_alpha = self.alpha / n_tests if n_tests > 0 else self.alpha
        
        for comp in self.comparisons:
            comp['corrected_alpha'] = corrected_alpha
            comp['significant_corrected'] = comp['p_value'] < corrected_alpha
    
    def _apply_holm_correction(self):
        """Apply Holm-Bonferroni correction."""
        n_tests = len(self.comparisons)
        if n_tests == 0:
            return
        
        # Sort by p-value
        sorted_comps = sorted(self.comparisons, key=lambda x: x['p_value'])
        
        for i, comp in enumerate(sorted_comps):
            corrected_alpha = self.alpha / (n_tests - i)
            comp['holm_corrected_alpha'] = corrected_alpha
            comp['holm_significant'] = comp['p_value'] < corrected_alpha
    
    def select_best_model(self, metrics_dict):
        """Select best model with statistical significance."""
        if not self.comparisons:
            return None, {}
        
        # Sort models by primary metric
        model_names = list(metrics_dict.keys())
        sorted_models = sorted(
            model_names,
            key=lambda x: metrics_dict[x].get('f1', 0),
            reverse=True
        )
        
        best_model = sorted_models[0]
        significance_report = {}
        
        # Check if best is significantly better than others
        for other in sorted_models[1:]:
            key1 = f"{best_model}_vs_{other}"
            key2 = f"{other}_vs_{best_model}"
            
            comp = self.comparisons.get(key1) or self.comparisons.get(key2)
            if comp:
                is_significant = comp.get('significant_corrected', comp['significant'])
                significance_report[other] = {
                    'significant': is_significant,
                    'p_value': comp['p_value'],
                    'difference': comp['difference']
                }
        
        return best_model, significance_report
    
    def generate_report(self, output_path):
        """Generate statistical comparison report."""
        report = {
            'test_settings': {
                'alpha': self.alpha,
                'test_type': self.test_type,
                'correction': self.correction
            },
            'comparisons': self.comparisons,
            'summary': {
                'total_comparisons': len(self.comparisons),
                'significant_pairs': sum(1 for c in self.comparisons if c.get('significant', False)),
                'significant_corrected': sum(1 for c in self.comparisons if c.get('significant_corrected', False))
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

# ============================================================================
# Integration into Existing Codebase
# ============================================================================
# We'll now integrate these features into your existing functions

# Add to the argument parser

def add_enhanced_args(parser):
    """Add enhanced feature arguments to parser."""
    # Smart Caching
    parser.add_argument("--use_smart_cache", action="store_true",
                       help="Use memory-mapped image caching for faster loading")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Directory for image cache")
    parser.add_argument("--cache_memory_gb", type=float, default=4.0,
                       help="Maximum memory for cache in GB")
    
    # Adaptive Learning
    parser.add_argument("--adaptive_lr", action="store_true",
                       help="Use gradient-aware adaptive learning rate")
    parser.add_argument("--lr_mode", type=str, default="aggressive",
                       choices=["aggressive", "conservative", "adaptive"],
                       help="Adaptive LR mode")
    
    # Gradient Monitoring
    parser.add_argument("--monitor_gradients", action="store_true",
                       help="Monitor gradient flow and statistics")
    parser.add_argument("--gradient_log_freq", type=int, default=50,
                       help="Frequency for gradient logging")
    
    # Bayesian Ensemble
    parser.add_argument("--bayesian_ensemble", action="store_true",
                       help="Use Bayesian ensemble with uncertainty quantification")
    parser.add_argument("--ensemble_method", type=str, default="mc_dropout",
                       choices=["mc_dropout", "deep_ensemble", "snapshot_ensemble"],
                       help="Ensemble method")
    parser.add_argument("--uncertainty_samples", type=int, default=30,
                       help="Number of samples for uncertainty estimation")
    
    # Model Compression
    parser.add_argument("--compress_model", action="store_true",
                       help="Apply model compression (pruning + quantization)")
    parser.add_argument("--compression_method", type=str, default="pruning+quantization",
                       choices=["pruning", "quantization", "pruning+quantization"],
                       help="Compression method")
    parser.add_argument("--pruning_amount", type=float, default=0.3,
                       help="Percentage of weights to prune")
    
    # Statistical Testing
    parser.add_argument("--statistical_testing", action="store_true",
                       help="Use statistical testing for model selection")
    parser.add_argument("--test_alpha", type=float, default=0.05,
                       help="Alpha level for statistical tests")
    parser.add_argument("--test_correction", type=str, default="bonferroni",
                       choices=["none", "bonferroni", "holm"],
                       help="Multiple testing correction method")
    
    # ---------------- Multi-Objective Optimization (MOO) ----------------
    moo = parser.add_argument_group("Multi-Objective Optimization (MOO)")

    moo.add_argument("--moo_enable", action="store_true",
                     help="Enable multi-objective HPO (Optuna NSGA-II). If enabled, it runs instead of --hpo.")
    moo.add_argument("--moo_objectives", type=str, default="auc,f1_at_precision,logloss",
                     help="Comma-separated objectives, e.g. 'recall,f1,auc' or 'recall,f1,auc,loss'")
    moo.add_argument("--moo_directions", type=str, default="maximize,maximize,minimize",
                     help="Comma-separated directions for each objective, e.g. 'maximize,maximize,maximize'")
    moo.add_argument("--moo_population_size", type=int, default=40,
                     help="NSGA-II population size (Optuna NSGAIISampler).")
    moo.add_argument("--moo_generations", type=int, default=20,
                     help="Number of NSGA-II generations. Total trials = population_size * generations.")
    moo.add_argument("--moo_eval_epochs", type=int, default=3,
                     help="Epochs per trial (short evaluation training).")
    moo.add_argument("--moo_precision_target", type=float, default=0.90,
                     help="Target precision for recall_at_precision/f1_at_precision constraints (0..1).")

    moo.add_argument("--moo_constraint_auc_target", type=float, default=0.90,
                 help="Feasibility constraint on validation ROC-AUC when selecting Pareto configs.")
    moo.add_argument("--moo_constraint_f1_target", type=float, default=0.90,
                 help="Feasibility constraint on validation F1 (uses threshold maximizing F1 at precision>=moo_precision_target).")
    moo.add_argument("--moo_disable_constraints", action="store_true",
                 help="Disable feasibility constraints during MOO selection (default: constraints enabled).")
    moo.add_argument("--moo_threshold_strategy", type=str, default="f1",
                     choices=["none", "f1", "youden", "prec_at_recall", "f1_at_precision", "recall_at_precision"],
                     help="If objectives include thresholded metrics (recall/f1/etc.), choose threshold on validation inside each MOO trial.")
    moo.add_argument("--moo_min_recall", type=float, default=0.85,
                     help="Used when --moo_threshold_strategy=prec_at_recall inside MOO trials.")
    moo.add_argument("--moo_top_k", type=int, default=3,
                     help="How many Pareto configs to keep (and optionally train).")
    moo.add_argument("--moo_selection_strategy", type=str, default="pr_auc_first",
                     choices=["recall_first", "weighted", "f1_first", "auc_first", "diversity", "pr_auc_first", "recall_at_precision_first", "f1_at_precision_first", "loss_first", "logloss_first"],
                     help="How to pick the final configs from the Pareto front.")
    moo.add_argument("--moo_train_top_k", action="store_true",
                     help="After MOO, train/evaluate the top-k selected configs (otherwise only best-1 is trained).")
    moo.add_argument("--moo_storage", type=str, default="",
                     help="Optuna storage for MOO, e.g. sqlite:///optuna_moo.db (empty = in-memory).")
    moo.add_argument("--moo_study", type=str, default="",
                     help="Optuna study name for MOO (empty = auto).")
    moo.add_argument("--moo_plot", action="store_true",
                     help="Save Pareto front plots (2D/3D) to outputs.")

    return parser

# Enhanced training function
def train_one_epoch_enhanced(model, loader, optimizer, criterion, device, args, 
                           gradient_monitor=None, epoch=0):
    """Enhanced training with gradient monitoring and adaptive features."""
    model.train()
    running_loss = 0.0
    y_true, y_pred, y_score = [], [], []
    
    # Initialize gradient monitor if requested
    if gradient_monitor:
        gradient_monitor.reset_epoch()
    
    pbar = tqdm(loader, desc=f'Train Epoch {epoch}', leave=False)
    
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # Record gradient statistics
        if gradient_monitor:
            gradient_monitor.record_batch()
        
        # Gradient clipping if enabled
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=args.grad_clip_norm
            )
        
        optimizer.step()
        
        # Update running metrics
        running_loss += loss.item() * x.size(0)
        
        # Get predictions
        with torch.no_grad():
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs[:, 1].cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Log gradient issues periodically
        if gradient_monitor and batch_idx % args.gradient_log_freq == 0:
            problematic = gradient_monitor.get_problematic_layers()
            if problematic:
                print(f"  Gradient issues at batch {batch_idx}:")
                for name, issue, norm in problematic[:2]:
                    print(f"    - {name}: {issue} (norm={norm:.2e})")
    
    # Calculate epoch gradient statistics
    grad_stats = None
    if gradient_monitor:
        grad_stats = gradient_monitor.get_epoch_statistics()
        
        # Save gradient visualization every 5 epochs
        if epoch % 5 == 0 and args.output_dir:
            save_path = Path(args.output_dir) / f"gradient_epoch_{epoch}.png"
            gradient_monitor.plot_gradient_distribution(save_path)
    
    avg_loss = running_loss / len(loader.dataset)
    grad_norm = grad_stats['grad_norm_mean'] if grad_stats else 0
    
    return avg_loss, np.array(y_true), np.array(y_pred), np.array(y_score), grad_norm

# Enhanced model creation with compression
def create_model_enhanced(name: str, num_classes: int, pretrained: bool, 
                         drop_path_rate: float = 0.0, args=None):
    """Create model with optional compression preparation."""
    model = create_model(name, num_classes, pretrained, drop_path_rate)
    
    # Prepare for compression if requested
    if args and getattr(args, 'compress_model', False):
        print(f"  Preparing model {name} for compression...")
        # Note: Actual compression happens after training
    
    return model

# Enhanced main training loop integration



# -----------------------------
# Dataset
# -----------------------------
class PathLabelDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples = [(str(p), int(y)) for p, y in samples]
        self.transform = transform
        self.loader = default_loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, y


# -----------------------------
# Models (unchanged from your original)
# -----------------------------
def create_model(name: str, num_classes: int, pretrained: bool, drop_path_rate: float = 0.0) -> nn.Module:
    """Create model by name.

    drop_path_rate: stochastic depth probability for supported models (ViT/ConvNeXt/ConvNeXtV2).
    """
    name = name.lower().strip()

    # --- Quantum-Classical (Quanv) ---
    # 1) Pure Quanv classifier (fast, no CNN backbone)
    if name in {"quanv_mlp", "quanvmlp"}:
        cfg = globals().get("_QUANV_MODEL_CFG", None)
        if cfg is None:
            raise ValueError("Unknown model: quanv_mlp (missing quanv config). Use --use_quanv and quanv args.")
        return QuanvMLP(
            in_ch=int(cfg.get("in_ch", 4)),
            grid=int(cfg.get("grid", 16)),
            hidden=int(cfg.get("hidden", 128)),
            num_classes=num_classes,
            dropout=float(cfg.get("dropout", 0.2)),
        )

    # 2) Quanv + standard backbone: "<backbone>_quanv" (e.g., efficientnet_b0_quanv)
    if name.endswith("_quanv"):
        cfg = globals().get("_QUANV_MODEL_CFG", None)
        if cfg is None:
            raise ValueError(f"Model '{name}' requested but quanv config not set. Use --use_quanv.")
        base_name = name.replace("_quanv", "")
        try:
            timm = _require_timm()
            backbone = timm.create_model(
                base_name,
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=int(cfg.get("in_ch", 4)),
                drop_path_rate=float(drop_path_rate),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create timm backbone '{base_name}' for quanv-hybrid. Ensure 'timm' is installed and the model name is supported. Underlying error: {e}") from e
        return QuanvBackboneWrapper(backbone=backbone, target_hw=224)



    # Optional model fallbacks for older torchvision builds
    # (common on Windows/conda envs). This keeps the CLI usable even if
    # newer model builders (ConvNeXtV2/EfficientNetV2/ViT) are missing.
    _fallbacks = {
        'convnextv2_tiny': 'convnext_tiny',
        'convnext_v2_tiny': 'convnext_tiny',
        'convnextv2tiny': 'convnext_tiny',
        'efficientnetv2_s': 'efficientnet_b0',
        'efficientnet_v2_s': 'efficientnet_b0',
        'efficientnetv2s': 'efficientnet_b0',
    }

    if name in _fallbacks:
        # Only fallback if the requested builder is actually unavailable
        if name.startswith('convnextv2') or name.startswith('convnext_v2'):
            if not hasattr(models, 'convnext_v2_tiny'):
                print(f"[WARN] Requested {name} but torchvision does not provide convnext_v2_tiny; falling back to convnext_tiny.", file=sys.stderr)
                name = _fallbacks[name]
        elif name.startswith('efficientnetv2') or name.startswith('efficientnet_v2'):
            if not hasattr(models, 'efficientnet_v2_s'):
                print(f"[WARN] Requested {name} but torchvision does not provide efficientnet_v2_s; falling back to efficientnet_b0.", file=sys.stderr)
                name = _fallbacks[name]
    # --- Classic CNNs ---
    # Accept a few common aliases to avoid CLI friction
    if name in {"mobilenetv3", "mobilenet_v3", "mobilenetv3_large", "mobilenet_v3_large"}:
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if name in {"mobilenetv3_small", "mobilenet_v3_small"}:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "inception_v3":
        weights = models.Inception_V3_Weights.DEFAULT if pretrained else None
        model = models.inception_v3(weights=weights, aux_logits=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if hasattr(model, "AuxLogits") and model.AuxLogits is not None:
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    # --- SOTA torchvision models (if available) ---
    if name in ("efficientnetv2_s", "efficientnet_v2_s", "efficientnetv2s"):
        if not hasattr(models, "efficientnet_v2_s"):
            raise ValueError("efficientnet_v2_s is not available in your torchvision. Please upgrade torchvision (e.g., >=0.13/0.14+ depending on build).")
        weights_cls = getattr(models, "EfficientNet_V2_S_Weights", None)
        w = weights_cls.DEFAULT if (pretrained and weights_cls is not None) else None
        model = models.efficientnet_v2_s(weights=w)
        # classifier is Sequential, last is Linear
        if hasattr(model, "classifier"):
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        else:
            raise RuntimeError("Unexpected EfficientNetV2 structure")
        return model

    if name in ("convnext_tiny", "convnext-tiny", "convnexttiny"):
        if not hasattr(models, "convnext_tiny"):
            raise ValueError("convnext_tiny is not available in your torchvision. Please upgrade torchvision.")
        weights_cls = getattr(models, "ConvNeXt_Tiny_Weights", None)
        w = weights_cls.DEFAULT if (pretrained and weights_cls is not None) else None
        # stochastic depth supported in newer torchvision
        try:
            model = models.convnext_tiny(weights=w, stochastic_depth_prob=float(drop_path_rate))
        except TypeError:
            model = models.convnext_tiny(weights=w)
        # Replace last Linear
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            for i in range(len(model.classifier) - 1, -1, -1):
                if isinstance(model.classifier[i], nn.Linear):
                    model.classifier[i] = nn.Linear(model.classifier[i].in_features, num_classes)
                    break
            else:
                raise RuntimeError("Unexpected ConvNeXt classifier structure; could not locate Linear.")
        elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
            model.head = nn.Linear(model.head.in_features, num_classes)
        else:
            raise RuntimeError("Unexpected ConvNeXt model structure; could not set head.")
        return model

    if name in ("convnextv2_tiny", "convnext_v2_tiny", "convnextv2tiny"):
        if not hasattr(models, "convnext_v2_tiny"):
            raise ValueError("convnext_v2_tiny is not available in your torchvision. Please upgrade torchvision.")
        weights_cls = getattr(models, "ConvNeXt_V2_Tiny_Weights", None)
        w = weights_cls.DEFAULT if (pretrained and weights_cls is not None) else None
        try:
            model = models.convnext_v2_tiny(weights=w, stochastic_depth_prob=float(drop_path_rate))
        except TypeError:
            model = models.convnext_v2_tiny(weights=w)
        # classifier[-1] is Linear
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            for i in range(len(model.classifier) - 1, -1, -1):
                if isinstance(model.classifier[i], nn.Linear):
                    model.classifier[i] = nn.Linear(model.classifier[i].in_features, num_classes)
                    break
            else:
                raise RuntimeError("Unexpected ConvNeXtV2 classifier structure; could not locate Linear.")
        else:
            raise RuntimeError("Unexpected ConvNeXtV2 model structure; could not set head.")
        return model

    # --- Vision Transformers ---
    if name in ("vit_b_16", "vit-b-16", "vitb16", "vit"):
        if not hasattr(models, "vit_b_16"):
            raise ValueError("vit_b_16 is not available in your torchvision. Please upgrade torchvision.")
        weights_cls = getattr(models, "ViT_B_16_Weights", None)
        w = weights_cls.DEFAULT if (pretrained and weights_cls is not None) else None
        try:
            model = models.vit_b_16(weights=w, stochastic_depth_prob=float(drop_path_rate))
        except TypeError:
            model = models.vit_b_16(weights=w)
        # classifier head lives under model.heads
        if hasattr(model, "heads"):
            if hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
                model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            elif isinstance(model.heads, nn.Sequential):
                for i in range(len(model.heads) - 1, -1, -1):
                    if isinstance(model.heads[i], nn.Linear):
                        model.heads[i] = nn.Linear(model.heads[i].in_features, num_classes)
                        break
                else:
                    raise RuntimeError("Unexpected ViT heads structure; could not locate Linear.")
            else:
                raise RuntimeError("Unexpected ViT heads structure; could not set head.")
        elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
            model.head = nn.Linear(model.head.in_features, num_classes)
        else:
            raise RuntimeError("Unexpected ViT model structure; could not set head.")
        return model


    # --- Extra architectures (available in many torchvision builds) ---
    if name == "densenet169":
        weights = getattr(models, 'DenseNet169_Weights', None)
        w = weights.DEFAULT if (pretrained and weights is not None) else None
        if not hasattr(models, 'densenet169'):
            raise ValueError('densenet169 is not available in your torchvision.')
        model = models.densenet169(weights=w)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if name == "densenet201":
        weights = getattr(models, 'DenseNet201_Weights', None)
        w = weights.DEFAULT if (pretrained and weights is not None) else None
        if not hasattr(models, 'densenet201'):
            raise ValueError('densenet201 is not available in your torchvision.')
        model = models.densenet201(weights=w)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if name == "resnet101":
        weights = getattr(models, 'ResNet101_Weights', None)
        w = weights.DEFAULT if (pretrained and weights is not None) else None
        if not hasattr(models, 'resnet101'):
            raise ValueError('resnet101 is not available in your torchvision.')
        model = models.resnet101(weights=w)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "resnet152":
        weights = getattr(models, 'ResNet152_Weights', None)
        w = weights.DEFAULT if (pretrained and weights is not None) else None
        if not hasattr(models, 'resnet152'):
            raise ValueError('resnet152 is not available in your torchvision.')
        model = models.resnet152(weights=w)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "efficientnet_b4":
        weights = getattr(models, 'EfficientNet_B4_Weights', None)
        w = weights.DEFAULT if (pretrained and weights is not None) else None
        if not hasattr(models, 'efficientnet_b4'):
            raise ValueError('efficientnet_b4 is not available in your torchvision.')
        model = models.efficientnet_b4(weights=w)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if name == "efficientnet_b5":
        weights = getattr(models, 'EfficientNet_B5_Weights', None)
        w = weights.DEFAULT if (pretrained and weights is not None) else None
        if not hasattr(models, 'efficientnet_b5'):
            raise ValueError('efficientnet_b5 is not available in your torchvision.')
        model = models.efficientnet_b5(weights=w)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    if name in ("convnext_small", "convnext-small", "convnextsmall"):
        if not hasattr(models, 'convnext_small'):
            raise ValueError('convnext_small is not available in your torchvision.')
        weights_cls = getattr(models, 'ConvNeXt_Small_Weights', None)
        w = weights_cls.DEFAULT if (pretrained and weights_cls is not None) else None
        try:
            model = models.convnext_small(weights=w, stochastic_depth_prob=float(drop_path_rate))
        except TypeError:
            model = models.convnext_small(weights=w)
        for i in range(len(model.classifier) - 1, -1, -1):
            if isinstance(model.classifier[i], nn.Linear):
                model.classifier[i] = nn.Linear(model.classifier[i].in_features, num_classes)
                break
        return model

    if name in ("convnext_base", "convnext-base", "convnextbase"):
        if not hasattr(models, 'convnext_base'):
            raise ValueError('convnext_base is not available in your torchvision.')
        weights_cls = getattr(models, 'ConvNeXt_Base_Weights', None)
        w = weights_cls.DEFAULT if (pretrained and weights_cls is not None) else None
        try:
            model = models.convnext_base(weights=w, stochastic_depth_prob=float(drop_path_rate))
        except TypeError:
            model = models.convnext_base(weights=w)
        for i in range(len(model.classifier) - 1, -1, -1):
            if isinstance(model.classifier[i], nn.Linear):
                model.classifier[i] = nn.Linear(model.classifier[i].in_features, num_classes)
                break
        return model

    if name in ("vit_l_16", "vit-l-16", "vitl16", "vitl"):
        if not hasattr(models, 'vit_l_16'):
            raise ValueError('vit_l_16 is not available in your torchvision.')
        weights_cls = getattr(models, 'ViT_L_16_Weights', None)
        w = weights_cls.DEFAULT if (pretrained and weights_cls is not None) else None
        try:
            model = models.vit_l_16(weights=w, stochastic_depth_prob=float(drop_path_rate))
        except TypeError:
            model = models.vit_l_16(weights=w)
        if hasattr(model, 'heads') and hasattr(model.heads, 'head') and isinstance(model.heads.head, nn.Linear):
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model

    if name in ("swin_t", "swin-t", "swint", "swin_tiny"):
        if not hasattr(models, 'swin_t'):
            raise ValueError('swin_t is not available in your torchvision.')
        weights_cls = getattr(models, 'Swin_T_Weights', None)
        w = weights_cls.DEFAULT if (pretrained and weights_cls is not None) else None
        model = models.swin_t(weights=w)
        # head lives at model.head
        if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
            model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    if name in ("swin_s", "swin-s", "swins", "swin_small"):
        if not hasattr(models, 'swin_s'):
            raise ValueError('swin_s is not available in your torchvision.')
        weights_cls = getattr(models, 'Swin_S_Weights', None)
        w = weights_cls.DEFAULT if (pretrained and weights_cls is not None) else None
        model = models.swin_s(weights=w)
        if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
            model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    raise ValueError(f"Unknown model: {name}")


def auto_img_size_for_model(model_name: str, default_img_size: int) -> int:
    return 299 if model_name.lower().strip() == "inception_v3" else int(default_img_size)




class WarmupCosineLR:
    """Simple warmup + cosine scheduler stepped once per epoch."""
    def __init__(self, optimizer: optim.Optimizer, total_epochs: int, warmup_epochs: float = 1.0, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.total_epochs = max(1, int(total_epochs))
        self.warmup_epochs = max(0.0, float(warmup_epochs))
        self.min_lr = float(min_lr)
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.epoch = 0

    def step(self):
        self.epoch += 1
        e = self.epoch
        if self.warmup_epochs > 0 and e <= self.warmup_epochs:
            t = e / self.warmup_epochs
            lrs = [lr * t for lr in self.base_lrs]
        else:
            # cosine from warmup_end..total
            t0 = self.warmup_epochs
            t = (e - t0) / max(1.0, (self.total_epochs - t0))
            import math
            lrs = [self.min_lr + 0.5 * (lr - self.min_lr) * (1.0 + math.cos(math.pi * t)) for lr in self.base_lrs]
        for g, lr in zip(self.optimizer.param_groups, lrs):
            g['lr'] = float(lr)


def make_scheduler(
    optimizer,
    args=None,
    total_epochs: int | None = None,
    scheduler_name: str | None = None,
    num_epochs: int | None = None,
    warmup_epochs: float | None = None,
):
    """Create an LR scheduler.

    Backward compatible:
      - old call: make_scheduler(optimizer, args, total_epochs)
    New (for MOO/HPO):
      - make_scheduler(optimizer, args=None, scheduler_name=..., num_epochs=..., warmup_epochs=...)

    Notes:
      - 'warmup_cosine' uses (warmup_epochs, num_epochs/total_epochs) for warmup + cosine decay.
      - 'plateau' uses args.plateau_factor / args.plateau_patience when args is provided.
    """
    # Resolve scheduler name
    if scheduler_name is None and args is not None:
        scheduler_name = getattr(args, "scheduler", None)
    scheduler_name = (scheduler_name or "none").lower()

    # Resolve epochs
    epochs = None
    if total_epochs is not None:
        epochs = int(total_epochs)
    elif num_epochs is not None:
        epochs = int(num_epochs)
    elif args is not None and hasattr(args, "epochs"):
        epochs = int(getattr(args, "epochs"))
    else:
        epochs = 1

    # Resolve warmup epochs
    if warmup_epochs is None and args is not None:
        warmup_epochs = float(getattr(args, "warmup_epochs", 0.0))
    warmup_epochs = float(warmup_epochs or 0.0)

    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if scheduler_name == "warmup_cosine":
        # Linear warmup then cosine
        def lr_lambda(ep: int):
            if warmup_epochs > 0 and ep < warmup_epochs:
                return max(1e-8, float(ep + 1) / float(warmup_epochs))
            # Cosine decay from 1.0 -> 0.0 over remaining epochs
            t = float(ep - warmup_epochs)
            T = float(max(1.0, epochs - warmup_epochs))
            return 0.5 * (1.0 + np.cos(np.pi * min(1.0, max(0.0, t / T))))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if scheduler_name == "plateau":
        # For plateau we require args (or fall back to safe defaults)
        factor = 0.5 if args is None else float(getattr(args, "plateau_factor", 0.5))
        patience = 2 if args is None else int(getattr(args, "plateau_patience", 2))
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience)
    return None

# -----------------------------
# Training functions (unchanged from your original)
# -----------------------------
def _primary_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    return output

def _aux_logits(output):
    if hasattr(output, "aux_logits"):
        return output.aux_logits
    return None

def _one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.zeros((y.size(0), num_classes), device=y.device, dtype=torch.float32).scatter_(1, y.view(-1, 1), 1.0)

def _apply_label_smoothing(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0.0:
        return targets
    n = targets.size(1)
    return targets * (1.0 - smoothing) + (smoothing / n)

def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor, class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    if class_weights is not None:
        w = class_weights.view(1, -1)
        loss = -(soft_targets * w * log_probs).sum(dim=1)
        denom = (soft_targets * w).sum(dim=1).clamp_min(1e-12)
        loss = loss / denom
    else:
        loss = -(soft_targets * log_probs).sum(dim=1)
    return loss.mean()

def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float, num_classes: int, label_smoothing: float) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0.0:
        raise ValueError("mixup alpha must be > 0")
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x2 = x[idx]
    y1 = _apply_label_smoothing(_one_hot(y, num_classes), label_smoothing)
    y2 = _apply_label_smoothing(_one_hot(y[idx], num_classes), label_smoothing)
    x_mix = lam * x + (1.0 - lam) * x2
    y_mix = lam * y1 + (1.0 - lam) * y2
    return x_mix, y_mix

def cutmix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float, num_classes: int, label_smoothing: float) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0.0:
        raise ValueError("cutmix alpha must be > 0")
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x2 = x[idx]
    B, C, H, W = x.size()
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2b = np.clip(cx + cut_w // 2, 0, W)
    y1b = np.clip(cy - cut_h // 2, 0, H)
    y2b = np.clip(cy + cut_h // 2, 0, H)

    x_cut = x.clone()
    x_cut[:, :, y1b:y2b, x1:x2b] = x2[:, :, y1b:y2b, x1:x2b]

    lam_adj = 1.0 - ((x2b - x1) * (y2b - y1b) / (W * H))
    y_a = _apply_label_smoothing(_one_hot(y, num_classes), label_smoothing)
    y_b = _apply_label_smoothing(_one_hot(y[idx], num_classes), label_smoothing)
    y_mix = lam_adj * y_a + (1.0 - lam_adj) * y_b
    return x_cut, y_mix

def diverse_tta_transforms(k: int):
    """Diverse TTA on already-normalized tensors [B,C,H,W]."""
    k = int(k)
    if k == 0:
        return lambda t: t
    if k == 1:
        return lambda t: torch.flip(t, dims=[3])  # hflip
    if k == 2:
        return lambda t: torch.flip(t, dims=[2])  # vflip
    if k == 3:
        return lambda t: torch.rot90(t, k=1, dims=[2, 3])
    if k == 4:
        return lambda t: torch.rot90(t, k=2, dims=[2, 3])
    if k == 5:
        return lambda t: torch.rot90(t, k=3, dims=[2, 3])
    if k == 6:
        # mild brightness/contrast tweak (deterministic)
        def _bc(x):
            # x already normalized; apply affine in pixel space approximation
            return x * 1.05 + 0.02
        return _bc
    if k == 7:
        def _bc2(x):
            return x * 0.95 - 0.02
        return _bc2
    return lambda t: t

# Backward-compatible alias used by eval_epoch_with_logits
def _tta_transform(k: int):
    return diverse_tta_transforms(int(k))



def _threshold_candidates(y_score: np.ndarray, max_points: int = 200) -> np.ndarray:
    """Generate a robust set of candidate thresholds from scores."""
    ys = np.asarray(y_score, dtype=float)
    ys = ys[np.isfinite(ys)]
    if ys.size == 0:
        return np.array([0.5], dtype=float)
    uniq = np.unique(ys)
    if uniq.size == 1:
        return np.array([0.5], dtype=float)
    if uniq.size > int(max_points):
        qs = np.linspace(0.0, 1.0, int(max_points))
        thr_list = np.quantile(ys, qs)
        thr_list = np.unique(thr_list)
    else:
        thr_list = uniq
    return thr_list.astype(float)




def _threshold_sweep_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, thr_list: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized sweep of precision/recall/f1 for many thresholds.

    y_true: shape (N,), values in {0,1}
    y_score: shape (N,), floats
    thr_list: shape (T,), floats
    Returns: (prec[T], rec[T], f1[T])
    """
    y_true = np.asarray(y_true).astype(np.int64).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float32).reshape(-1)
    thr_list = np.asarray(thr_list).astype(np.float32).reshape(-1)
    if y_true.size == 0 or thr_list.size == 0:
        z = np.zeros((max(1, thr_list.size),), dtype=np.float32)
        return z, z, z

    # pred[T,N] = score>=thr
    pred = (y_score[None, :] >= thr_list[:, None])
    yt = (y_true[None, :] == 1)

    tp = np.sum(pred & yt, axis=1).astype(np.float32)
    fp = np.sum(pred & (~yt), axis=1).astype(np.float32)
    fn = np.sum((~pred) & yt, axis=1).astype(np.float32)

    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2.0 * prec * rec / (prec + rec + 1e-12)
    return prec, rec, f1


def find_best_threshold_and_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    strategy: str,
    min_recall: float = 0.0,
    min_precision: float = 0.0,
) -> Tuple[float, float, float, float, bool]:
    """Pick a decision threshold on scores and return (thr, precision, recall, f1, feasible).

    Strategies:
      - f1: maximize F1
      - youden: maximize (sensitivity + specificity - 1)
      - prec_at_recall: maximize precision subject to recall >= min_recall
      - f1_at_precision: maximize F1 subject to precision >= min_precision
      - recall_at_precision: maximize recall subject to precision >= min_precision
      - none: returns 0.5
    """
    strategy = (strategy or "f1").lower().strip()
    if strategy == "none":
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score).astype(float)
        y_pred = (y_score >= 0.5).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0) if y_true.size else 0.0
        rec = recall_score(y_true, y_pred, zero_division=0) if y_true.size else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0) if y_true.size else 0.0
        return 0.5, float(prec), float(rec), float(f1), True

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    thr_list = _threshold_candidates(y_score, max_points=200)
    if thr_list.size == 0:
        thr_list = np.array([0.5], dtype=float)

    # Fast vectorized sweep for binary metrics
    precs, recs, f1s = _threshold_sweep_binary_metrics(y_true, y_score, thr_list)

    feasible_any = True

    if strategy == "f1":
        vals = f1s
        mask = np.ones_like(vals, dtype=bool)
    elif strategy == "prec_at_recall":
        mask = recs >= float(min_recall)
        vals = np.where(mask, precs, -1e18)
        feasible_any = bool(np.any(mask))
    elif strategy == "f1_at_precision":
        mask = precs >= float(min_precision)
        vals = np.where(mask, f1s, -1e18)
        feasible_any = bool(np.any(mask))
    elif strategy == "recall_at_precision":
        mask = precs >= float(min_precision)
        vals = np.where(mask, recs, -1e18)
        feasible_any = bool(np.any(mask))
    elif strategy == "youden":
        # Youden needs specificity too; compute via confusion counts
        y_true_b = (y_true.reshape(-1) == 1)
        pred = (y_score[None, :] >= thr_list[:, None])
        tp = np.sum(pred & y_true_b[None, :], axis=1).astype(np.float32)
        fp = np.sum(pred & (~y_true_b[None, :]), axis=1).astype(np.float32)
        fn = np.sum((~pred) & y_true_b[None, :], axis=1).astype(np.float32)
        tn = np.sum((~pred) & (~y_true_b[None, :]), axis=1).astype(np.float32)
        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        vals = sens + spec - 1.0
        mask = np.ones_like(vals, dtype=bool)
    else:
        raise ValueError(f"Unknown threshold strategy: {strategy}")

    if not feasible_any:
        # Fallback: choose threshold that maximizes precision
        best_i = int(np.argmax(precs))
        return float(thr_list[best_i]), float(precs[best_i]), float(recs[best_i]), float(f1s[best_i]), False

    best_i = int(np.argmax(vals))
    return float(thr_list[best_i]), float(precs[best_i]), float(recs[best_i]), float(f1s[best_i]), True


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    strategy: str,
    min_recall: float = 0.0,
    min_precision: float = 0.0,
) -> float:
    thr, _, _, _, _ = find_best_threshold_and_metrics(
        y_true=y_true,
        y_score=y_score,
        strategy=strategy,
        min_recall=float(min_recall),
        min_precision=float(min_precision),
    )
    return float(thr)


def optimize_ensemble_weights(
    val_scores: dict,
    y_val: np.ndarray,
    metric: str = "f1_at_precision",
    precision_target: float = 0.90,
    trials: int = 2048,
    strategy: str = "random_search",
    seed: int = 123,
    auc_target: float | None = None,
    f1_target: float | None = None,
):
    """
    Choose non-negative weights that sum to 1 for an ensemble of probabilistic scores.
    - val_scores: dict[name -> np.array(prob)]
    - metric: objective on validation used to select weights.
    Returns: dict[name -> weight]
    """
    names = list(val_scores.keys())
    if len(names) == 0:
        return {}
    if len(names) == 1:
        return {names[0]: 1.0}

    y_val = np.asarray(y_val).astype(int)
    probs_mat = np.vstack([np.asarray(val_scores[n]).astype(float) for n in names])  # (M, N)

    def _objective(p: np.ndarray) -> float:
        # higher is better
        m = metric
        if m == "auc":
            try:
                return float(roc_auc_score(y_val, p))
            except Exception:
                return -1e9
        if m == "pr_auc":
            try:
                return float(average_precision_score(y_val, p))
            except Exception:
                return -1e9
        if m == "neg_logloss":
            try:
                return -float(log_loss(y_val, np.clip(p, 1e-7, 1 - 1e-7)))
            except Exception:
                return -1e9
        if m == "neg_brier":
            try:
                return -float(brier_score_loss(y_val, p))
            except Exception:
                return -1e9

        # Precision-constrained objectives
        if m in ("f1_at_precision", "recall_at_precision"):
            _, best_p, best_r, best_f1, feasible = find_best_threshold_and_metrics(
                y_val, p, strategy="f1_at_precision", min_recall=0.0, min_precision=precision_target
            )
            if not feasible:
                return -1e9
            return float(best_f1 if m == "f1_at_precision" else best_r)

        return -1e9

    def _constraints_ok(p: np.ndarray) -> bool:
        if auc_target is None and f1_target is None:
            return True
        ok = True
        if auc_target is not None:
            try:
                ok = ok and (roc_auc_score(y_val, p) >= float(auc_target))
            except Exception:
                return False
        if f1_target is not None:
            _, _, _, best_f1, feasible = find_best_threshold_and_metrics(
                y_val, p, strategy="f1_at_precision", min_recall=0.0, min_precision=precision_target
            )
            ok = ok and feasible and (best_f1 >= float(f1_target))
        return bool(ok)

    # Baseline: AUC-weighted (fast + deterministic)
    if strategy == "auc":
        aucs = []
        for n in names:
            try:
                aucs.append(float(roc_auc_score(y_val, val_scores[n])))
            except Exception:
                aucs.append(0.0)
        w = np.array(aucs, dtype=float)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones(len(names), dtype=float)
        w = w / w.sum()
        return {n: float(w[i]) for i, n in enumerate(names)}

    # Random search on simplex (Dirichlet)
    rng = np.random.default_rng(int(seed))
    best_w = None
    best_obj = -1e18
    best_obj_uncon = -1e18
    best_w_uncon = None

    # Try a reasonable baseline first: uniform
    w0 = np.ones(len(names), dtype=float) / len(names)
    p0 = np.clip((w0[:, None] * probs_mat).sum(axis=0), 0.0, 1.0)
    o0 = _objective(p0)
    best_w_uncon, best_obj_uncon = w0, o0
    if _constraints_ok(p0):
        best_w, best_obj = w0, o0

    # Main loop
    trials = int(max(1, trials))
    for _ in tqdm(range(trials), desc="Ensemble-Weights", leave=False):
        w = rng.dirichlet(np.ones(len(names), dtype=float))
        p = np.clip((w[:, None] * probs_mat).sum(axis=0), 0.0, 1.0)
        obj = _objective(p)

        if obj > best_obj_uncon:
            best_obj_uncon = obj
            best_w_uncon = w

        if _constraints_ok(p) and obj > best_obj:
            best_obj = obj
            best_w = w

    if best_w is None:
        # Fallback to best unconstrained
        best_w = best_w_uncon

    best_w = np.asarray(best_w, dtype=float)
    best_w = np.clip(best_w, 0.0, None)
    if best_w.sum() <= 0:
        best_w = np.ones(len(names), dtype=float) / len(names)
    else:
        best_w = best_w / best_w.sum()

    return {n: float(best_w[i]) for i, n in enumerate(names)}


def _stacking_transform(X: np.ndarray, mode: str = "logit", eps: float = 1e-6) -> np.ndarray:
    """Transform base-model probabilities into stacking features.

    X is expected to be of shape (n_samples, n_models) with values in [0, 1].
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Stacking feature matrix must be 2D, got shape={X.shape}")
    if mode == "proba":
        return X
    if mode == "logit":
        Xc = np.clip(X, eps, 1.0 - eps)
        return np.log(Xc / (1.0 - Xc)).astype(np.float32)
    raise ValueError(f"Unknown stacking_feature_mode: {mode}")


def fit_stacking_meta_learner(
    X: np.ndarray,
    y: np.ndarray,
    meta_model: str = "logreg",
    C: float = 1.0,
    seed: int = 42,
    cv_folds: int = 0,
    standardize: bool = True,
    max_iter: int = 2000,
) -> tuple[object, dict]:
    """Fit a simple meta-learner for stacking (stacked generalization)."""
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y size mismatch: X={X.shape}, y={y.shape}")
    if X.shape[1] < 2:
        raise ValueError("Stacking requires at least 2 base models (n_features >= 2).")

    if meta_model == "logreg":
        clf = LogisticRegression(
            C=float(C),
            penalty="l2",
            solver="lbfgs",
            max_iter=int(max_iter),
            class_weight="balanced",
            random_state=int(seed),
        )
    elif meta_model == "logreg_cv":
        cv = int(cv_folds) if int(cv_folds) > 1 else 5
        Cs = np.logspace(-3, 3, 13)
        clf = LogisticRegressionCV(
            Cs=Cs,
            cv=cv,
            penalty="l2",
            solver="lbfgs",
            max_iter=int(max_iter),
            class_weight="balanced",
            scoring="neg_log_loss",
            random_state=int(seed),
            refit=True,
        )
    else:
        raise ValueError(f"Unknown stacking_meta_model: {meta_model}")

    if standardize:
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", clf),
        ])
    else:
        model = clf

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X, y)

    info: dict = {
        "meta_model": meta_model,
        "C": float(C),
        "standardize": bool(standardize),
    }
    try:
        est = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        coef = getattr(est, "coef_", None)
        intercept = getattr(est, "intercept_", None)
        if coef is not None:
            info["coef"] = [float(x) for x in coef.reshape(-1)]
        if intercept is not None:
            info["intercept"] = [float(x) for x in intercept.reshape(-1)]
    except Exception:
        pass

    return model, info

def freeze_backbone(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    head_modules = []
    if hasattr(model, "fc"):
        head_modules.append(model.fc)
    if hasattr(model, "classifier"):
        head_modules.append(model.classifier)
    if hasattr(model, "head"):
        head_modules.append(model.head)
    # Torchvision ViT keeps the classifier under model.heads (Sequential)
    # e.g., model.heads.head
    if hasattr(model, "heads"):
        head_modules.append(model.heads)
    for hm in head_modules:
        for p in hm.parameters():
            p.requires_grad = True
    if hasattr(model, "AuxLogits") and hasattr(model.AuxLogits, "fc"):
        for p in model.AuxLogits.fc.parameters():
            p.requires_grad = True

def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True

def head_parameters(model: nn.Module):
    params = []
    if hasattr(model, "fc"):
        params += list(model.fc.parameters())
    if hasattr(model, "classifier"):
        params += list(model.classifier.parameters())
    if hasattr(model, "head"):
        params += list(model.head.parameters())
    # Torchvision ViT keeps the classifier under model.heads (Sequential)
    if hasattr(model, "heads"):
        params += list(model.heads.parameters())
    if hasattr(model, "AuxLogits") and hasattr(model.AuxLogits, "fc"):
        params += list(model.AuxLogits.fc.parameters())
    seen = set()
    uniq = []
    for p in params:
        if id(p) not in seen:
            uniq.append(p)
            seen.add(id(p))
    return uniq




# -------------------------------
# Hyperparameter optimization (HPO)
# -------------------------------

def override_namespace(args: argparse.Namespace, overrides: dict) -> argparse.Namespace:
    """Return a shallow copy of argparse.Namespace with overrides applied."""
    new_args = argparse.Namespace(**vars(args))
    for k, v in overrides.items():
        setattr(new_args, k, v)
    return new_args


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    """Sample log-uniform in [low, high]."""
    import math
    if low <= 0 or high <= 0:
        raise ValueError('log-uniform bounds must be > 0')
    lo = math.log(low)
    hi = math.log(high)
    return float(math.exp(lo + (hi - lo) * rng.random()))


def _hpo_sample_params(rng: random.Random, base_args: argparse.Namespace, model_name: str) -> dict:
    """A safe, light-weight random search space (no dataloader rebuild required)."""
    # Conservative ranges that usually work for image fine-tuning.
    params: Dict[str, Any] = {}

    params['lr'] = _log_uniform(rng, 2e-5, 3e-3)
    params['weight_decay'] = _log_uniform(rng, 1e-6, 5e-3)

    # Regularization & loss
    params['label_smoothing'] = float(rng.uniform(0.0, 0.12))
    params['mixup_alpha'] = float(rng.choice([0.0, rng.uniform(0.05, 0.35)]))
    params['cutmix_alpha'] = float(rng.choice([0.0, rng.uniform(0.05, 0.35)]))

    params['loss'] = rng.choice(['ce', 'focal'])
    if params['loss'] == 'focal':
        params['focal_alpha'] = float(rng.uniform(0.15, 0.40))
        params['focal_gamma'] = float(rng.uniform(1.0, 3.0))
    else:
        params['focal_alpha'] = float(getattr(base_args, 'focal_alpha', 0.25))
        params['focal_gamma'] = float(getattr(base_args, 'focal_gamma', 2.0))

    # Optimizer / scheduler
    params['optimizer'] = rng.choice(['adamw', 'adam'])
    params['scheduler'] = rng.choice(['plateau', 'cosine', 'warmup_cosine', 'none'])
    params['warmup_epochs'] = float(rng.choice([0.5, 1.0, 2.0]))

    # Model-specific regularization (safe to pass; create_model may ignore for some backbones)
    params['drop_path_rate'] = float(rng.uniform(0.0, 0.20))

    # Training stability
    params['grad_clip_norm'] = float(rng.choice([0.0, 0.5, 1.0]))

    # Head LR (when using freeze_backbone_epochs)
    params['head_lr'] = float(_log_uniform(rng, 2e-5, 3e-3))

    return params


def _hpo_score_from_val(metric: str, va_y: np.ndarray, va_p: np.ndarray, va_s: np.ndarray, va_loss: float) -> float:
    metric = (metric or 'f1').lower()
    if metric in ('loss', 'val_loss'):
        # lower is better
        return -float(va_loss)
    if metric in ('auc', 'roc_auc'):
        if len(np.unique(va_y)) < 2:
            return float('-inf')
        try:
            return float(roc_auc_score(va_y, va_s))
        except Exception:
            return float('-inf')
    if metric in ('balanced_acc', 'balanced_accuracy'):
        try:
            return float(balanced_accuracy_score(va_y, va_p))
        except Exception:
            return float('-inf')
    # default: f1
    try:
        return float(f1_score(va_y, va_p, zero_division=0))
    except Exception:
        return float('-inf')


def run_hpo_for_model(
    base_args: argparse.Namespace,
    device: torch.device,
    model_name: str,
    full_samples: list[tuple[Path, int]],
    splits: 'SplitIndices',
    img_size: int,
    seed: int,
    out_dir: Path,
) -> dict:
    """Random-search HPO that reuses the existing pipeline (train_one_epoch/eval_epoch).

    It runs a short training (args.hpo_epochs) for each trial and selects the best trial
    by args.hpo_metric. The chosen overrides are returned.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    hpo_trials = int(getattr(base_args, 'hpo_trials', 20))
    hpo_epochs = int(getattr(base_args, 'hpo_epochs', 6))
    hpo_metric = str(getattr(base_args, 'hpo_metric', 'f1'))
    hpo_seed = int(getattr(base_args, 'hpo_seed', 777))

    rng = random.Random(hpo_seed + (abs(hash(model_name)) % 10_000))

    # Build loaders once (do NOT change dataloader-dependent params in the search space)
    need_quanv = ('_quanv' in str(model_name).lower()) or ('quanv_mlp' in str(model_name).lower())
    if ('_quanv' in str(model_name)) and (not bool(getattr(base_args, 'use_quanv', False))):
        print(f"[HPO] Note: '{model_name}' implies quanv features; enabling quanv for HPO loaders.")
    train_loader, val_loader, _test_loader = build_loaders(
        full_samples, splits, img_size, base_args.batch_size,
        base_args.num_workers, base_args.online_augmentation,
        base_args.use_weighted_sampler and base_args.balance_strategy == 'none',
        randaugment=bool(getattr(base_args, 'randaugment', False)),
        autoaugment=bool(getattr(base_args, 'autoaugment', False)),
        randaugment_n=int(getattr(base_args, 'randaugment_n', 2)),
        randaugment_m=int(getattr(base_args, 'randaugment_m', 9)),
        aug_strength=str(getattr(base_args, 'aug_strength', 'light')),
        use_smart_cache=bool(getattr(base_args, 'use_smart_cache', False)),
        cache_dir=(str(getattr(base_args, 'cache_dir', '')).strip() or None),
        cache_memory_gb=float(getattr(base_args, 'cache_memory_gb', 4.0)),
        use_quanv=need_quanv,
        quanv_input_size=int(getattr(base_args, 'quanv_input_size', 32)),
        quanv_patch=int(getattr(base_args, 'quanv_patch', 2)),
        quanv_qubits=int(getattr(base_args, 'quanv_qubits', 4)),
        quanv_depth=int(getattr(base_args, 'quanv_depth', 1)),
        quanv_cache_dir=(str(getattr(base_args, 'quanv_cache_dir', '')).strip() or None),
        train_mix=str(getattr(base_args, 'train_mix', 'aug')),
        train_orig_ratio=float(getattr(base_args, 'train_orig_ratio', 0.7)),
        max_aug_per_group=int(getattr(base_args, 'max_aug_per_group', 0)),
        orig_n=int(getattr(base_args, '_orig_n', 0)),
        group_ids=getattr(base_args, '_full_group_ids', None),
        rng_seed=int(getattr(base_args, 'seed', 42)),
    )

    # CSV log
    csv_path = out_dir / 'hpo_trials.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'trial', 'score', 'metric', 'val_loss', 'val_f1', 'val_auc',
            'lr', 'weight_decay', 'optimizer', 'scheduler', 'warmup_epochs',
            'label_smoothing', 'mixup_alpha', 'cutmix_alpha', 'loss', 'focal_alpha', 'focal_gamma',
            'drop_path_rate', 'grad_clip_norm', 'head_lr'
        ])

    best = {'score': float('-inf'), 'overrides': None, 'summary': None}

    for t in range(hpo_trials):
        trial_params = _hpo_sample_params(rng, base_args, model_name)
        t_args = override_namespace(base_args, trial_params)

        setattr(t_args, 'use_quanv', need_quanv)
        # Trial-level reproducibility
        set_seed(seed + 10_000 + t)

        # Create model & loss
        model = create_model(model_name, 2, bool(getattr(t_args, 'pretrained', False)),
                             drop_path_rate=float(getattr(t_args, 'drop_path_rate', 0.1))).to(device)

        # Class weights (optional)
        weight = None
        if getattr(t_args, 'use_class_weights', False):
            tr_ys = [full_samples[i][1] for i in splits.train]
            n1 = sum(tr_ys)
            n0 = len(tr_ys) - n1
            if n0 > 0 and n1 > 0:
                w0 = (n0 + n1) / (2.0 * n0)
                w1 = (n0 + n1) / (2.0 * n1)
                weight = torch.tensor([w0, w1], device=device)

        if getattr(t_args, 'loss', 'ce') == 'focal':
            criterion = FocalLoss(alpha=float(getattr(t_args, 'focal_alpha', 0.25)), gamma=float(getattr(t_args, 'focal_gamma', 2.0)))
        else:
            criterion = nn.CrossEntropyLoss(
                weight=weight,
                label_smoothing=float(getattr(t_args, 'label_smoothing', 0.0)) or 0.0,
            )

        # Optimizer (handle optional frozen phase in a trial-safe way)
        frozen_phase = bool(getattr(t_args, 'pretrained', False) and getattr(t_args, 'freeze_backbone_epochs', 0) and getattr(t_args, 'freeze_backbone_epochs', 0) > 0)
        if frozen_phase:
            freeze_backbone(model)
            optimizer = make_optimizer(head_parameters(model), lr=float(getattr(t_args, 'head_lr', t_args.lr)), weight_decay=float(getattr(t_args, 'weight_decay', 0.0)), args=t_args)
        else:
            optimizer = make_optimizer(model.parameters(), lr=float(getattr(t_args, 'lr', 1e-3)), weight_decay=float(getattr(t_args, 'weight_decay', 0.0)), args=t_args)

        scheduler = make_scheduler(optimizer, t_args, total_epochs=hpo_epochs)

        # Short training
        for epoch in range(1, hpo_epochs + 1):
            # If freeze_backbone_epochs is large, don't waste all HPO epochs frozen
            if frozen_phase and epoch == min(int(getattr(t_args, 'freeze_backbone_epochs', 0)) + 1, max(2, hpo_epochs // 2)):
                unfreeze_all(model)
                optimizer = make_optimizer(model.parameters(), lr=float(getattr(t_args, 'lr', 1e-3)), weight_decay=float(getattr(t_args, 'weight_decay', 0.0)), args=t_args)
                scheduler = make_scheduler(optimizer, t_args, total_epochs=hpo_epochs)
                frozen_phase = False

            tr_loss, _, _, _ = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                grad_clip_norm=float(getattr(t_args, 'grad_clip_norm', 0.0)),
                mixup_alpha=float(getattr(t_args, 'mixup_alpha', 0.0)),
                cutmix_alpha=float(getattr(t_args, 'cutmix_alpha', 0.0)),
                label_smoothing=float(getattr(t_args, 'label_smoothing', 0.0)),
                class_weights=weight,
                loss_name=str(getattr(t_args, 'loss', 'ce')),
                grad_accum_steps=int(getattr(t_args, 'grad_accum_steps', 1)),
                amp=(False if getattr(t_args, 'no_amp', False) else (bool(getattr(t_args, 'amp', False)) or (device.type == 'cuda'))),
                scaler=None,
                grad_monitor=None,
            )

            va_loss, va_y, va_p, va_s = eval_epoch(model, val_loader, criterion, device)
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(va_loss)
                else:
                    scheduler.step()

        # Metrics
        try:
            va_f1 = float(f1_score(va_y, va_p, zero_division=0)) if len(va_y) else float('nan')
        except Exception:
            va_f1 = float('nan')
        if len(np.unique(va_y)) == 2:
            try:
                va_auc = float(roc_auc_score(va_y, va_s))
            except Exception:
                va_auc = float('nan')
        else:
            va_auc = float('nan')

        score = _hpo_score_from_val(hpo_metric, va_y, va_p, va_s, va_loss)

        # Log
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                t, score, hpo_metric, float(va_loss), va_f1, va_auc,
                trial_params['lr'], trial_params['weight_decay'], trial_params['optimizer'], trial_params['scheduler'], trial_params['warmup_epochs'],
                trial_params['label_smoothing'], trial_params['mixup_alpha'], trial_params['cutmix_alpha'], trial_params['loss'], trial_params['focal_alpha'], trial_params['focal_gamma'],
                trial_params['drop_path_rate'], trial_params['grad_clip_norm'], trial_params['head_lr']
            ])

        if score > best['score']:
            best['score'] = score
            best['overrides'] = trial_params
            best['summary'] = {
                'trial': t,
                'score': score,
                'metric': hpo_metric,
                'val_loss': float(va_loss),
                'val_f1': va_f1,
                'val_auc': va_auc,
            }

        # free memory between trials
        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    # Persist best
    best_path = out_dir / 'hpo_best.json'
    payload = {
        'model': model_name,
        'chosen_overrides': best['overrides'],
        'best_summary': best['summary'],
        'hpo_trials': hpo_trials,
        'hpo_epochs': hpo_epochs,
        'hpo_metric': hpo_metric,
        'hpo_seed': hpo_seed,
    }
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    return best['overrides'] or {}



def _optuna_available() -> bool:
    try:
        import optuna  # noqa: F401
        return True
    except Exception:
        return False


def _hpo_suggest_params_optuna(trial, base_args: argparse.Namespace) -> dict:
    """Optuna search space.

    Notes:
      - Keep it conservative; avoid dataloader-dependent params.
      - IMPORTANT: Keep the search space *static* (no conditional parameters) so that Optuna's multivariate TPE
        works properly without falling back to independent sampling.
    """
    # Log scales for LR/WD are essential.
    params: Dict[str, Any] = {}

    params['lr'] = trial.suggest_float('lr', 2e-5, 3e-3, log=True)
    params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 5e-3, log=True)

    params['label_smoothing'] = trial.suggest_float('label_smoothing', 0.0, 0.12)
    params['mixup_alpha'] = trial.suggest_float('mixup_alpha', 0.0, 0.35)
    params['cutmix_alpha'] = trial.suggest_float('cutmix_alpha', 0.0, 0.35)

    # Loss choice + focal params (ALWAYS present to keep a fixed search space).
    params['loss'] = trial.suggest_categorical('loss', ['ce', 'focal'])
    params['focal_alpha'] = trial.suggest_float('focal_alpha', 0.15, 0.40)
    params['focal_gamma'] = trial.suggest_float('focal_gamma', 1.0, 3.0)

    params['optimizer'] = trial.suggest_categorical('optimizer', ['adamw', 'adam'])
    params['scheduler'] = trial.suggest_categorical('scheduler', ['plateau', 'cosine', 'warmup_cosine', 'none'])

    # Warmup only matters for warmup_cosine; still safe to pass.
    params['warmup_epochs'] = float(trial.suggest_categorical('warmup_epochs', [0.5, 1.0, 2.0]))

    params['drop_path_rate'] = trial.suggest_float('drop_path_rate', 0.0, 0.20)
    params['grad_clip_norm'] = float(trial.suggest_categorical('grad_clip_norm', [0.0, 0.5, 1.0]))

    params['head_lr'] = trial.suggest_float('head_lr', 2e-5, 3e-3, log=True)

    return params




def _make_optuna_sampler(seed: int):
    """Create an Optuna sampler (robust across Optuna versions).

    We prefer multivariate/grouped TPE for mixed continuous/categorical spaces.
    If a given Optuna version doesn't support some arguments, we gracefully fall back.
    """
    import optuna

    # NOTE: multivariate TPE is marked experimental in some Optuna versions.
    # Keeping a *static* search space (no conditional params) avoids independent sampling fallbacks.
    for kwargs in (
        dict(seed=seed, multivariate=True, group=True, warn_independent_sampling=False),
        dict(seed=seed, multivariate=True, group=True),
        dict(seed=seed, multivariate=True),
        dict(seed=seed),
    ):
        try:
            return optuna.samplers.TPESampler(**kwargs)
        except TypeError:
            continue
    # Last resort
    return optuna.samplers.TPESampler(seed=seed)


def _make_optuna_pruner(base_args: argparse.Namespace, hpo_epochs: int):
    """Create an Optuna pruner.

    Recommended (default): Hyperband / ASHA-style pruning for fast, reliable early stopping.
    """
    import optuna

    name = str(getattr(base_args, "optuna_pruner", "hyperband") or "hyperband").strip().lower()
    reduction_raw = getattr(base_args, "optuna_reduction_factor", 3) or 3
    try:
        reduction = int(reduction_raw)
    except Exception:
        reduction = int(float(reduction_raw))
    reduction = max(2, int(reduction))
    # Use epoch as the "step" (we report per-epoch).
    if name in ("hyperband", "hb"):
        # Hyperband needs max_resource. Here: hpo_epochs.
        try:
            return optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=max(1, int(hpo_epochs)),
                reduction_factor=reduction,
            )
        except TypeError:
            # Some versions use different signature; fall back to SHA.
            name = "sha"

    if name in ("sha", "asha", "successive_halving"):
        try:
            return optuna.pruners.SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=reduction,
                min_early_stopping_rate=0,
            )
        except TypeError:
            return optuna.pruners.SuccessiveHalvingPruner()

    # median (safe fallback)
    return optuna.pruners.MedianPruner(n_warmup_steps=max(1, min(2, int(hpo_epochs) // 2)))


def run_hpo_optuna_for_model(
    base_args: argparse.Namespace,
    device: torch.device,
    model_name: str,
    full_samples: list[tuple[Path, int]],
    splits: 'SplitIndices',
    img_size: int,
    seed: int,
    out_dir: Path,
) -> dict:
    """Optuna (TPE) HPO with pruning. Returns chosen overrides dict."""
    if not _optuna_available():
        raise RuntimeError(
            "Optuna is not installed in this environment. Install with: python -m pip install optuna"
        )

    import optuna

    out_dir.mkdir(parents=True, exist_ok=True)

    hpo_trials = int(getattr(base_args, 'hpo_trials', 20))
    hpo_epochs = int(getattr(base_args, 'hpo_epochs', 6))
    hpo_metric = str(getattr(base_args, 'hpo_metric', 'f1'))
    hpo_seed = int(getattr(base_args, 'hpo_seed', 777))

    # Optional persistent study
    storage = str(getattr(base_args, 'optuna_storage', '') or '').strip() or None
    study_name = str(getattr(base_args, 'optuna_study', '') or '').strip() or f"hpo_{model_name}"

    sampler = _make_optuna_sampler(hpo_seed)
    pruner = _make_optuna_pruner(base_args, hpo_epochs)

    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    # Build loaders once
    train_loader, val_loader, _test_loader = build_loaders(
        full_samples, splits, img_size, base_args.batch_size,
        base_args.num_workers, base_args.online_augmentation,
        base_args.use_weighted_sampler and base_args.balance_strategy == 'none',
        randaugment=bool(getattr(base_args, 'randaugment', False)),
        autoaugment=bool(getattr(base_args, 'autoaugment', False)),
        randaugment_n=int(getattr(base_args, 'randaugment_n', 2)),
        randaugment_m=int(getattr(base_args, 'randaugment_m', 9)),
        aug_strength=str(getattr(base_args, 'aug_strength', 'light')),
        use_smart_cache=bool(getattr(base_args, 'use_smart_cache', False)),
        cache_dir=(str(getattr(base_args, 'cache_dir', '')).strip() or None),
        cache_memory_gb=float(getattr(base_args, 'cache_memory_gb', 4.0)),
        train_mix=str(getattr(base_args, 'train_mix', 'aug')),
        train_orig_ratio=float(getattr(base_args, 'train_orig_ratio', 0.7)),
        max_aug_per_group=int(getattr(base_args, 'max_aug_per_group', 0)),
        orig_n=int(getattr(base_args, '_orig_n', 0)),
        group_ids=getattr(base_args, '_full_group_ids', None),
        rng_seed=int(getattr(base_args, 'seed', 42)),
    )

    # CSV log
    csv_path = out_dir / 'hpo_trials.csv'
    if not csv_path.exists():
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                'trial', 'score', 'metric', 'val_loss', 'val_f1', 'val_auc',
                'lr', 'weight_decay', 'optimizer', 'scheduler', 'warmup_epochs',
                'label_smoothing', 'mixup_alpha', 'cutmix_alpha', 'loss', 'focal_alpha', 'focal_gamma',
                'drop_path_rate', 'grad_clip_norm', 'head_lr'
            ])

    def objective(trial: 'optuna.Trial') -> float:
        trial_params = _hpo_suggest_params_optuna(trial, base_args)
        t_args = override_namespace(base_args, trial_params)

        # Trial-level reproducibility
        set_seed(seed + 20_000 + int(trial.number))

        # Create model & loss
        model = create_model(model_name, 2, bool(getattr(t_args, 'pretrained', False)),
                             drop_path_rate=float(getattr(t_args, 'drop_path_rate', 0.1))).to(device)

        # Class weights (optional)
        weight = None
        if getattr(t_args, 'use_class_weights', False):
            tr_ys = [full_samples[i][1] for i in splits.train]
            n1 = sum(tr_ys)
            n0 = len(tr_ys) - n1
            if n0 > 0 and n1 > 0:
                w0 = (n0 + n1) / (2.0 * n0)
                w1 = (n0 + n1) / (2.0 * n1)
                weight = torch.tensor([w0, w1], device=device)

        if getattr(t_args, 'loss', 'ce') == 'focal':
            criterion = FocalLoss(alpha=float(getattr(t_args, 'focal_alpha', 0.25)), gamma=float(getattr(t_args, 'focal_gamma', 2.0)))
        else:
            criterion = nn.CrossEntropyLoss(
                weight=weight,
                label_smoothing=float(getattr(t_args, 'label_smoothing', 0.0)) or 0.0,
            )

        frozen_phase = bool(getattr(t_args, 'pretrained', False) and getattr(t_args, 'freeze_backbone_epochs', 0) and getattr(t_args, 'freeze_backbone_epochs', 0) > 0)
        if frozen_phase:
            freeze_backbone(model)
            optimizer = make_optimizer(head_parameters(model), lr=float(getattr(t_args, 'head_lr', t_args.lr)), weight_decay=float(getattr(t_args, 'weight_decay', 0.0)), args=t_args)
        else:
            optimizer = make_optimizer(model.parameters(), lr=float(getattr(t_args, 'lr', 1e-3)), weight_decay=float(getattr(t_args, 'weight_decay', 0.0)), args=t_args)

        scheduler = make_scheduler(optimizer, t_args, total_epochs=hpo_epochs)

        best_epoch_score = float('-inf')
        last_va_loss = float('inf')
        last_va_y = np.asarray([])
        last_va_p = np.asarray([])
        last_va_s = np.asarray([])

        for epoch in range(1, hpo_epochs + 1):
            if frozen_phase and epoch == min(int(getattr(t_args, 'freeze_backbone_epochs', 0)) + 1, max(2, hpo_epochs // 2)):
                unfreeze_all(model)
                optimizer = make_optimizer(model.parameters(), lr=float(getattr(t_args, 'lr', 1e-3)), weight_decay=float(getattr(t_args, 'weight_decay', 0.0)), args=t_args)
                scheduler = make_scheduler(optimizer, t_args, total_epochs=hpo_epochs)
                frozen_phase = False

            _tr_loss, _, _, _ = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                grad_clip_norm=float(getattr(t_args, 'grad_clip_norm', 0.0)),
                mixup_alpha=float(getattr(t_args, 'mixup_alpha', 0.0)),
                cutmix_alpha=float(getattr(t_args, 'cutmix_alpha', 0.0)),
                label_smoothing=float(getattr(t_args, 'label_smoothing', 0.0)),
                class_weights=weight,
                loss_name=str(getattr(t_args, 'loss', 'ce')),
                grad_accum_steps=int(getattr(t_args, 'grad_accum_steps', 1)),
                amp=(False if getattr(t_args, 'no_amp', False) else (bool(getattr(t_args, 'amp', False)) or (device.type == 'cuda'))),
                scaler=None,
                grad_monitor=None,
            )

            va_loss, va_y, va_p, va_s = eval_epoch(model, val_loader, criterion, device)
            last_va_loss, last_va_y, last_va_p, last_va_s = float(va_loss), va_y, va_p, va_s

            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(va_loss)
                else:
                    scheduler.step()

            score = _hpo_score_from_val(hpo_metric, va_y, va_p, va_s, va_loss)
            best_epoch_score = max(best_epoch_score, score)

            trial.report(score, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Final metrics for logging
        try:
            va_f1 = float(f1_score(last_va_y, last_va_p, zero_division=0)) if len(last_va_y) else float('nan')
        except Exception:
            va_f1 = float('nan')
        if len(np.unique(last_va_y)) == 2:
            try:
                va_auc = float(roc_auc_score(last_va_y, last_va_s))
            except Exception:
                va_auc = float('nan')
        else:
            va_auc = float('nan')

        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                int(trial.number), float(best_epoch_score), hpo_metric, float(last_va_loss), va_f1, va_auc,
                trial_params['lr'], trial_params['weight_decay'], trial_params['optimizer'], trial_params['scheduler'], trial_params['warmup_epochs'],
                trial_params['label_smoothing'], trial_params['mixup_alpha'], trial_params['cutmix_alpha'], trial_params['loss'], trial_params['focal_alpha'], trial_params['focal_gamma'],
                trial_params['drop_path_rate'], trial_params['grad_clip_norm'], trial_params['head_lr']
            ])

        # free memory
        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None

        return float(best_epoch_score)

    study.optimize(objective, n_trials=hpo_trials, show_progress_bar=True, catch=(Exception,))

    # Build overrides from best trial params. Ensure required keys exist.
    best_params = dict(study.best_trial.params)

    # If best loss isn't focal, enforce numeric defaults (keeps rest of code stable)
    if best_params.get('loss', 'ce') != 'focal':
        best_params['focal_alpha'] = float(getattr(base_args, 'focal_alpha', 0.25))
        best_params['focal_gamma'] = float(getattr(base_args, 'focal_gamma', 2.0))

    best_path = out_dir / 'hpo_best.json'
    payload = {
        'model': model_name,
        'chosen_overrides': best_params,
        'best_value': float(study.best_value),
        'hpo_trials': hpo_trials,
        'hpo_epochs': hpo_epochs,
        'hpo_metric': hpo_metric,
        'hpo_seed': hpo_seed,
        'optuna_storage': storage,
        'optuna_study': study_name,
    }
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    return best_params


# ============================================================================
# Multi-Objective HPO (Optuna NSGA-II)
# ============================================================================

def _parse_csv_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def _moo_extract_objectives(
    objectives: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    loss: float,
    precision_target: float = 0.90,
) -> Dict[str, float]:
    """Compute a superset of metrics, then return dict for requested objectives.

    Supports classic metrics (recall/f1/auc/loss) plus discrimination-focused:
      - pr_auc (average precision / PR-AUC)
      - recall_at_precision (max recall s.t. precision >= precision_target)
      - logloss (negative log-likelihood)
      - brier (Brier score)
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    ys = np.asarray(y_score).astype(np.float32) if y_score is not None else None

    m, _ = compute_metrics(y_true, y_pred, ys)
    if not isinstance(m, dict):
        m = {}

    # Always include loss
    m["loss"] = float(loss)

    # Robust ROC-AUC fallback
    if "auc" not in m or not np.isfinite(float(m.get("auc", float("nan")))):
        m["auc"] = 0.5

    # PR-AUC (Average Precision)
    if ys is not None and y_true.size and len(np.unique(y_true)) == 2:
        try:
            m["pr_auc"] = float(average_precision_score(y_true, ys))
        except Exception:
            m["pr_auc"] = 0.0
    else:
        m["pr_auc"] = 0.0

    # Log-loss / NLL (lower is better)
    if ys is not None and y_true.size:
        try:
            ysc = np.clip(ys, 1e-6, 1.0 - 1e-6)
            m["logloss"] = float(log_loss(y_true, ysc, labels=[0, 1]))
        except Exception:
            m["logloss"] = 999.0
    else:
        m["logloss"] = 999.0

    # Brier score (lower is better)
    if ys is not None and y_true.size:
        try:
            ysc = np.clip(ys, 0.0, 1.0)
            m["brier"] = float(brier_score_loss(y_true, ysc))
        except Exception:
            m["brier"] = 1.0
    else:
        m["brier"] = 1.0

    # Recall at (minimum) precision target
    m["recall_at_precision"] = 0.0
    if ys is not None and y_true.size and len(np.unique(y_true)) == 2:
        try:
            p, r, th = precision_recall_curve(y_true, ys)
            # p,r length = len(th)+1; thresholds align to p[:-1], r[:-1]
            if th is not None and len(th) > 0:
                p0 = p[:-1]
                r0 = r[:-1]
                mask = p0 >= float(precision_target)
                if np.any(mask):
                    m["recall_at_precision"] = float(np.max(r0[mask]))
                else:
                    m["recall_at_precision"] = 0.0
            else:
                m["recall_at_precision"] = 0.0
        except Exception:
            m["recall_at_precision"] = 0.0


    # F1 at (minimum) precision target: best-achievable F1 with precision >= precision_target
    m["f1_at_precision"] = 0.0
    m["precision_at_f1_at_precision"] = 0.0
    m["recall_at_f1_at_precision"] = 0.0
    if ys is not None and y_true.size and len(np.unique(y_true)) == 2:
        try:
            p, r, th = precision_recall_curve(y_true, ys)
            if th is not None and len(th) > 0:
                p0 = p[:-1]
                r0 = r[:-1]
                f1s = (2.0 * p0 * r0) / (p0 + r0 + 1e-12)
                mask = p0 >= float(precision_target)
                if np.any(mask):
                    f1_masked = np.where(mask, f1s, -1.0)
                    j = int(np.argmax(f1_masked))
                    m["f1_at_precision"] = float(f1s[j])
                    m["precision_at_f1_at_precision"] = float(p0[j])
                    m["recall_at_f1_at_precision"] = float(r0[j])
        except Exception:
            pass

    # Synonyms
    alias = {
        "prauc": "pr_auc",
        "average_precision": "pr_auc",
        "ap": "pr_auc",
        "nll": "logloss",
        "neg_log_loss": "logloss",
        "recall@precision": "recall_at_precision",
        "rap": "recall_at_precision",
        "f1@precision": "f1_at_precision",
        "f1ap": "f1_at_precision",
        "f1_at_prec": "f1_at_precision",
        "f1_at_p": "f1_at_precision",
    }
    out: Dict[str, float] = {}
    for obj in objectives:
        k = obj.lower().strip()
        k = alias.get(k, k)
        if k in m:
            out[k] = float(m[k])
        else:
            out[k] = 0.0
    return out

def _moo_select_from_pareto(
    pareto_trials: List["optuna.trial.FrozenTrial"],
    objectives: List[str],
    strategy: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Pick configs from Pareto set."""
    if not pareto_trials:
        return []
    top_k = max(1, int(top_k))

    # Convert to list of (metrics_dict, params)
    rows = []
    for t in pareto_trials:
        # multi-objective values order matches objectives list used in study
        vals = list(t.values) if t.values is not None else []
        metrics = {}
        for i, obj in enumerate(objectives):
            if i < len(vals):
                metrics[obj] = float(vals[i])
        rows.append((metrics, dict(t.params)))

    def safe_get(m, k, default=0.0):
        return float(m.get(k, default))

    strategy = (strategy or "recall_first").lower().strip()

    if strategy == "recall_first":
        rows.sort(key=lambda r: safe_get(r[0], "recall", 0.0), reverse=True)
    elif strategy == "f1_first":
        rows.sort(key=lambda r: safe_get(r[0], "f1", 0.0), reverse=True)
    elif strategy == "auc_first":
        rows.sort(key=lambda r: safe_get(r[0], "auc", 0.0), reverse=True)
    elif strategy == "pr_auc_first":
        rows.sort(key=lambda r: safe_get(r[0], "pr_auc", 0.0), reverse=True)
    elif strategy == "recall_at_precision_first":
        rows.sort(key=lambda r: safe_get(r[0], "recall_at_precision", 0.0), reverse=True)
    elif strategy == "f1_at_precision_first":
        rows.sort(key=lambda r: safe_get(r[0], "f1_at_precision", 0.0), reverse=True)
    elif strategy in ("loss_first", "logloss_first"):
        rows.sort(key=lambda r: safe_get(r[0], "logloss", 999.0), reverse=False)
    elif strategy == "weighted":
        # KPI-aligned weights: F1@precision > AUC > (negative) logloss (fallbacks to 0 if missing)
        w = {"f1_at_precision": 0.5, "auc": 0.35, "logloss": -0.15}
        rows.sort(
            key=lambda r: sum(w.get(k, 0.0) * safe_get(r[0], k, 0.0) for k in w.keys()),
            reverse=True,
        )
    elif strategy == "diversity":
        # greedy diversity by L1 distance in param space (simple + robust)
        selected = []
        remaining = rows.copy()

        # start with pr_auc_first (fallback to recall)
        remaining.sort(key=lambda r: (safe_get(r[0], "pr_auc", 0.0), safe_get(r[0], "recall", 0.0)), reverse=True)
        selected.append(remaining.pop(0))

        def dist(p1, p2):
            d = 0.0
            for k in set(p1.keys()) | set(p2.keys()):
                v1, v2 = p1.get(k), p2.get(k)
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    d += abs(float(v1) - float(v2))
                else:
                    d += 0.0 if v1 == v2 else 1.0
            return d

        while len(selected) < top_k and remaining:
            best_i, best_d = None, -1.0
            for i, cand in enumerate(remaining):
                dmin = min(dist(cand[1], s[1]) for s in selected)
                if dmin > best_d:
                    best_d = dmin
                    best_i = i
            selected.append(remaining.pop(best_i))
        rows = selected
    else:
        # default
        rows.sort(key=lambda r: safe_get(r[0], "recall", 0.0), reverse=True)

    return [r[1] for r in rows[:top_k]]

def _moo_plot_pareto(
    pareto_trials: List["optuna.trial.FrozenTrial"],
    objectives: List[str],
    save_path: Path,
    dpi: int = 300,
):
    """Simple 2D/3D Pareto plot."""
    if not pareto_trials:
        return
    vals = np.array([t.values for t in pareto_trials if t.values is not None], dtype=float)
    if vals.size == 0:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if len(objectives) >= 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(vals[:, 0], vals[:, 1], vals[:, 2], s=40, alpha=0.7)
            ax.set_xlabel(objectives[0])
            ax.set_ylabel(objectives[1])
            ax.set_zlabel(objectives[2])
            ax.set_title("Pareto Front (3D)")
        else:
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.scatter(vals[:, 0], vals[:, 1], s=40, alpha=0.7)
            ax.set_xlabel(objectives[0] if len(objectives) > 0 else "obj0")
            ax.set_ylabel(objectives[1] if len(objectives) > 1 else "obj1")
            ax.set_title("Pareto Front (2D)")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=int(dpi), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[MOO] Plot failed: {e}")

def run_moo_optuna_for_model(
    args: argparse.Namespace,
    device: torch.device,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    out_dir: Path,
    seed: int,
    num_classes: int = 2,
) -> Dict[str, Any]:
    """Run multi-objective HPO using Optuna NSGA-II (Pareto optimization).

    Returns:
        dict with:
          - selected_params: list[dict]
          - objectives: list[str]
          - directions: list[str]
          - study_name: str
          - storage: str|None
    """
    import optuna  # local import to keep optuna optional in non-HPO runs

    objectives = _parse_csv_list(getattr(args, "moo_objectives", "recall,f1,auc")) or ["recall", "f1", "auc"]
    directions = _parse_csv_list(getattr(args, "moo_directions", "")) or ["maximize"] * len(objectives)

    if len(directions) != len(objectives):
        print(f"[MOO] directions length != objectives length. Using all 'maximize'.")
        directions = ["maximize"] * len(objectives)

    # Normalize objective names (supports aliases like 'prauc' -> 'pr_auc')
    _alias = {
        "prauc": "pr_auc",
        "average_precision": "pr_auc",
        "ap": "pr_auc",
        "nll": "logloss",
        "neg_log_loss": "logloss",
        "recall@precision": "recall_at_precision",
        "rap": "recall_at_precision",
        "f1@precision": "f1_at_precision",
        "f1ap": "f1_at_precision",
        "f1_at_prec": "f1_at_precision",
    }
    objectives = [_alias.get(o.lower().strip(), o.lower().strip()) for o in objectives]

    pop = int(getattr(args, "moo_population_size", 40))
    gens = int(getattr(args, "moo_generations", 20))
    n_trials = max(1, pop * gens)
    eval_epochs = int(getattr(args, "moo_eval_epochs", 3))

    storage = (str(getattr(args, "moo_storage", "")).strip() or None)
    study_name = (str(getattr(args, "moo_study", "")).strip() or f"moo_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    moo_dir = Path(out_dir) / "moo"
    moo_dir.mkdir(parents=True, exist_ok=True)

    print(f"    [MOO] Optuna(NSGA-II) for {model_name} -> {moo_dir}")
    print(f"    [MOO] objectives={objectives} directions={directions}")
    print(f"    [MOO] pop={pop} gens={gens} trials={n_trials} eval_epochs={eval_epochs}")

    sampler = optuna.samplers.NSGAIISampler(population_size=pop, seed=int(seed))
    # IMPORTANT: keep pruner disabled for multi-objective stability
    pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        directions=directions,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: "optuna.Trial"):
        # reuse same search space as single-objective HPO (safe + proven)
        params = {}
        params["lr"] = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 3e-3, log=True)
        params["label_smoothing"] = trial.suggest_float("label_smoothing", 0.0, 0.15)
        params["mixup_alpha"] = trial.suggest_float("mixup_alpha", 0.0, 0.4)
        params["cutmix_alpha"] = trial.suggest_float("cutmix_alpha", 0.0, 0.4)
        params["loss"] = trial.suggest_categorical("loss", ["ce", "focal"])
        params["focal_alpha"] = trial.suggest_float("focal_alpha", 0.15, 0.45)
        params["focal_gamma"] = trial.suggest_float("focal_gamma", 0.5, 3.0)
        params["optimizer"] = trial.suggest_categorical("optimizer", ["adamw", "adam"])
        params["scheduler"] = trial.suggest_categorical("scheduler", ["none", "plateau", "warmup_cosine"])
        params["warmup_epochs"] = trial.suggest_float("warmup_epochs", 0.0, 2.0)
        params["drop_path_rate"] = trial.suggest_float("drop_path_rate", 0.0, 0.2)
        params["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [0.0, 0.5, 1.0])
        params["head_lr"] = trial.suggest_float("head_lr", 1e-5, 3e-3, log=True)

        # Create model
        model = create_model(
            model_name,
            num_classes=num_classes,
            pretrained=bool(getattr(args, "pretrained", False)),
            drop_path_rate=float(params["drop_path_rate"]),
        ).to(device)

        # Optimizer
        opt = make_optimizer(
            model,
            optimizer_name=str(params["optimizer"]),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            head_lr=float(params["head_lr"]),
        )

        # Scheduler
        sch = make_scheduler(
            optimizer=opt,
            scheduler_name=str(params["scheduler"]),
            num_epochs=int(eval_epochs),
            warmup_epochs=float(params["warmup_epochs"]),
        )

        # Criterion
        if params["loss"] == "focal":
            crit = FocalLoss(alpha=float(params["focal_alpha"]), gamma=float(params["focal_gamma"]))
        else:
            crit = nn.CrossEntropyLoss(label_smoothing=float(params["label_smoothing"]))

        best_vec = None
        best_y = best_p = best_s = None
        best_epoch_loss = None

        # If objectives include thresholded metrics (recall/f1/etc.), tune threshold on validation inside each trial.
        thr_strategy = str(getattr(args, "moo_threshold_strategy", "f1") or "f1").lower().strip()
        thr_min_recall = float(getattr(args, "moo_min_recall", getattr(args, "min_recall", 0.0)) or 0.0)
        prec_target = float(getattr(args, "moo_precision_target", 0.90) or 0.90)

        thresh_metrics = {"recall", "f1", "precision", "acc", "specificity", "balanced_acc", "balanced_accuracy"}
        need_thr = (thr_strategy != "none") and any((o in thresh_metrics) for o in objectives)

        def _is_better(vec, best):
            if best is None:
                return True
            for v, b, d in zip(vec, best, directions):
                d = str(d).lower().strip()
                if d == "minimize":
                    if float(v) < float(b) - 1e-12:
                        return True
                    if float(v) > float(b) + 1e-12:
                        return False
                else:
                    if float(v) > float(b) + 1e-12:
                        return True
                    if float(v) < float(b) - 1e-12:
                        return False
            return False

        for ep in range(int(eval_epochs)):
            _tr_loss, _, _, _ = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=opt,
                criterion=crit,
                device=device,
                grad_clip_norm=float(params["grad_clip_norm"]),
                mixup_alpha=float(params["mixup_alpha"]),
                cutmix_alpha=float(params["cutmix_alpha"]),
            )

            _val_loss, y_true, y_pred, y_score = eval_epoch(
                model=model,
                loader=val_loader,
                criterion=crit,
                device=device,
            )

            if sch is not None:
                if str(params["scheduler"]) == "plateau":
                    sch.step(_val_loss)
                else:
                    sch.step()

            # Optional: threshold tuning inside MOO objective for threshold-sensitive metrics
            if need_thr:
                try:
                    thr = find_best_threshold(y_true, y_score, thr_strategy, min_recall=float(thr_min_recall), min_precision=float(prec_target))
                    y_pred = (np.asarray(y_score) >= float(thr)).astype(np.int64)
                except Exception:
                    pass

            obj_vals = _moo_extract_objectives(
                objectives, y_true, y_pred, y_score, float(_val_loss),
                precision_target=float(prec_target),
            )
            vec = tuple(obj_vals[o] for o in objectives)

            # Keep best epoch by objectives (lexicographic by objective order + directions)
            if _is_better(vec, best_vec):
                best_vec = vec
                best_epoch_loss = float(_val_loss)
                best_y, best_p, best_s = y_true, y_pred, y_score

        if best_y is None:
            # worst-case fallback
            best_y = np.zeros((1,), dtype=int)
            best_p = np.zeros((1,), dtype=int)
            best_s = np.zeros((1,), dtype=float)
            best_epoch_loss = 999.0

        obj_vals = _moo_extract_objectives(objectives, best_y, best_p, best_s, float(best_epoch_loss), precision_target=float(getattr(args, "moo_precision_target", 0.90) or 0.90))

        # Store rich validation diagnostics for constraint-based selection (v5)
        try:
            _prec_target = float(getattr(args, "moo_precision_target", 0.90) or 0.90)
            thr_f1ap, p_f1ap, r_f1ap, f1_f1ap, feas_f1ap = find_best_threshold_and_metrics(
                best_y, best_s, "f1_at_precision", min_precision=_prec_target
            )
            trial.set_user_attr("val_thr_f1_at_precision", float(thr_f1ap))
            trial.set_user_attr("val_precision_at_f1_at_precision", float(p_f1ap))
            trial.set_user_attr("val_recall_at_f1_at_precision", float(r_f1ap))
            trial.set_user_attr("val_f1_at_precision_thr", float(f1_f1ap))
            trial.set_user_attr("val_f1_at_precision_feasible", bool(feas_f1ap))
        except Exception:
            pass

        try:
            # Objectives computed from scores (threshold-free)
            trial.set_user_attr("val_auc", float(obj_vals.get("auc", 0.0)))
            trial.set_user_attr("val_pr_auc", float(obj_vals.get("pr_auc", 0.0)))
            trial.set_user_attr("val_logloss", float(obj_vals.get("logloss", 999.0)))
            trial.set_user_attr("val_brier", float(obj_vals.get("brier", 999.0)))
            trial.set_user_attr("val_recall_at_precision", float(obj_vals.get("recall_at_precision", 0.0)))
            trial.set_user_attr("val_f1_at_precision_prcurve", float(obj_vals.get("f1_at_precision", 0.0)))
            trial.set_user_attr("val_precision_at_f1_at_precision_prcurve", float(obj_vals.get("precision_at_f1_at_precision", 0.0)))
            trial.set_user_attr("val_recall_at_f1_at_precision_prcurve", float(obj_vals.get("recall_at_f1_at_precision", 0.0)))
        except Exception:
            pass

        # Return in the same order as objectives list
        return tuple(obj_vals[o] for o in objectives)

    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=True)

    pareto = list(study.best_trials)  # Pareto front

    # ---------------- Feasibility constraints (enabled by default in v5) ----------------
    constraints_enabled = not bool(getattr(args, "moo_disable_constraints", False))
    feasible_pareto: List["optuna.trial.FrozenTrial"] = []
    if constraints_enabled and pareto:
        prec_target_c = float(getattr(args, "moo_precision_target", 0.90))
        auc_target_c = float(getattr(args, "moo_constraint_auc_target", 0.0))
        f1_target_c = float(getattr(args, "moo_constraint_f1_target", 0.0))

        for t in pareto:
            auc_v = float(t.user_attrs.get("val_auc", 0.0))
            p_v = float(t.user_attrs.get("val_precision_at_f1_at_precision", 0.0))
            f1_v = float(t.user_attrs.get("val_f1_at_precision_thr", 0.0))
            if (auc_v >= auc_target_c) and (p_v >= prec_target_c) and (f1_v >= f1_target_c):
                feasible_pareto.append(t)

        print(
            f"[MOO] Feasible Pareto trials meeting constraints: {len(feasible_pareto)}/{len(pareto)} "
            f"(prec≥{prec_target_c:.2f}, f1≥{f1_target_c:.2f}, auc≥{auc_target_c:.2f})"
        )

    # Select configs
    top_k = max(1, int(getattr(args, "moo_top_k", 3)))
    if constraints_enabled and feasible_pareto:
        # Rank feasible configs: prioritize threshold-based F1 under precision constraint, then AUC, then logloss
        feasible_pareto.sort(
            key=lambda t: (
                -float(t.user_attrs.get("val_f1_at_precision_thr", 0.0)),
                -float(t.user_attrs.get("val_auc", 0.0)),
                float(t.user_attrs.get("val_logloss", 999.0)),
            )
        )
        selected_params = [dict(t.params) for t in feasible_pareto[:top_k]]
    else:
        if constraints_enabled:
            print("[MOO][WARN] No Pareto trials satisfied constraints; falling back to --moo_selection_strategy on the full Pareto front.")
        selected_params = _moo_select_from_pareto(
            pareto_trials=pareto,
            objectives=objectives,
            strategy=str(getattr(args, "moo_selection_strategy", "recall_first")),
            top_k=top_k,
        )

    # Save Pareto trials
    pareto_payload = []
    for t in pareto:
        pareto_payload.append({
            "number": int(t.number),
            "values": list(t.values) if t.values is not None else None,
            "params": dict(t.params),
            "user_attrs": dict(t.user_attrs),
        })
    with open(moo_dir / f"pareto_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "objectives": objectives,
            "directions": directions,
            "study_name": study.study_name,
            "storage": storage,
            "pareto_trials": pareto_payload,
            "selected_params": selected_params,
        }, f, indent=2)

    if bool(getattr(args, "moo_plot", False)):
        _moo_plot_pareto(
            pareto_trials=pareto,
            objectives=objectives[:3],
            save_path=moo_dir / f"pareto_{model_name}.png",
            dpi=int(getattr(args, "plot_dpi", 300)),
        )

    return {
        "selected_params": selected_params,
        "objectives": objectives,
        "directions": directions,
        "study_name": study.study_name,
        "storage": storage,
    }

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    grad_clip_norm: float = 0.0,
    aux_weight: float = 0.4,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    label_smoothing: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
    loss_name: str = 'ce',
    grad_accum_steps: int = 1,
    amp: bool = False,
    scaler=None,
    grad_monitor=None,
):
    """One training epoch.

    - Supports Mixup/Cutmix only for CE-based loss.
    - Supports SAM optimizer (two-step) if optimizer is instance of SAM.
    - Adds optional AMP + gradient accumulation (for non-SAM optimizers).

    Notes:
      * For correctness, SAM disables AMP and accumulation (it needs 2 exact full-batch steps).
    """
    model.train()
    running_loss = 0.0
    y_true, y_pred, y_score = [], [], []

    grad_accum_steps = max(1, int(grad_accum_steps))

    use_mix = (mixup_alpha and mixup_alpha > 0.0) and (loss_name == 'ce')
    use_cut = (cutmix_alpha and cutmix_alpha > 0.0) and (loss_name == 'ce')

    use_sam = isinstance(optimizer, SAM)
    if use_sam:
        if amp:
            print('[WARN] AMP is disabled for SAM to keep steps correct/stable.', file=sys.stderr)
        if grad_accum_steps > 1:
            print('[WARN] grad_accum_steps>1 is disabled for SAM (needs exact two-step batch).', file=sys.stderr)
        amp = False
        grad_accum_steps = 1

    if amp and (device.type != 'cuda'):
        amp = False

    if amp and scaler is None:
        try:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
        except Exception:
            scaler = None
            amp = False

    autocast_ctx = (torch.cuda.amp.autocast if (amp and scaler is not None) else nullcontext)

    def _forward_loss(x_in, y_in, y_soft_in=None):
        out = model(x_in)
        logits = _primary_logits(out)
        if y_soft_in is not None:
            loss = soft_cross_entropy(logits, y_soft_in, class_weights=class_weights)
        else:
            loss = criterion(logits, y_in)
        aux = _aux_logits(out)
        if aux is not None:
            if y_soft_in is not None:
                loss = loss + aux_weight * soft_cross_entropy(aux, y_soft_in, class_weights=class_weights)
            else:
                loss = loss + aux_weight * criterion(aux, y_in)
        return loss, logits

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc='Train', leave=False)

    for step_idx, (x, y) in enumerate(pbar, start=1):
        x = x.to(device)
        y = torch.as_tensor(y, device=device)

        # Prepare batch (mixup/cutmix only for CE)
        x_sam = x
        y_sam = y
        y_soft = None
        use_soft = (x.dim() == 4) and (x.size(0) > 1) and (use_mix or use_cut)
        if use_soft:
            if use_mix and use_cut:
                do_mix = (np.random.rand() < 0.5)
            else:
                do_mix = use_mix
            if do_mix:
                x_sam, y_soft = mixup_batch(x, y, float(mixup_alpha), num_classes=2, label_smoothing=float(label_smoothing))
            else:
                x_sam, y_soft = cutmix_batch(x, y, float(cutmix_alpha), num_classes=2, label_smoothing=float(label_smoothing))

        if use_sam:
            # SAM (no AMP, no accumulation)
            optimizer.zero_grad(set_to_none=True)
            loss, logits = _forward_loss(x_sam, y_sam, y_soft)
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.first_step(zero_grad=True)

            loss2, logits2 = _forward_loss(x_sam, y_sam, y_soft)
            loss2.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.second_step(zero_grad=True)

            loss_to_log = loss2
            logits_used = logits2

        else:
            # Standard optimizer (optionally AMP + accumulation)
            with autocast_ctx():
                loss, logits = _forward_loss(x_sam, y_sam, y_soft)
                loss_scaled = loss / float(grad_accum_steps)

            if amp and scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            do_step = (step_idx % grad_accum_steps == 0) or (step_idx == len(loader))
            if do_step:
                if grad_clip_norm and grad_clip_norm > 0:
                    if amp and scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))

                if amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if grad_monitor is not None:
                    grad_monitor.record_batch()

            loss_to_log = loss
            logits_used = logits

        running_loss += float(loss_to_log.item()) * x.size(0)

        probs = torch.softmax(logits_used, dim=1)[:, 1].detach().cpu().numpy()
        preds = (probs >= 0.5).astype(np.int64)

        y_true.extend(y.detach().cpu().numpy())
        y_pred.extend(preds)
        y_score.extend(probs)

        pbar.set_postfix({'loss': f"{loss_to_log.item():.4f}"})

    return running_loss / len(loader.dataset), np.asarray(y_true), np.asarray(y_pred), np.asarray(y_score)



@torch.no_grad()
def eval_epoch(model, loader, criterion, device, tta: int = 0, threshold: Optional[float] = None):
    model.eval()
    running_loss = 0.0
    y_true, y_pred, y_score = [], [], []

    tta = int(tta) if tta is not None else 0
    n_passes = max(1, tta)

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device)
        y = torch.as_tensor(y, device=device)

        out0 = model(x)
        logits0 = _primary_logits(out0)
        loss = criterion(logits0, y)

        probs_acc = None
        for k in range(n_passes):
            tform = diverse_tta_transforms(k % 8)
            xk = tform(x)
            outk = model(xk)
            logitsk = _primary_logits(outk)
            probsk = torch.softmax(logitsk, dim=1)
            probs_acc = probsk if probs_acc is None else (probs_acc + probsk)
        probs = probs_acc / float(n_passes)

        running_loss += float(loss.item()) * x.size(0)

        p1 = probs[:, 1].detach().cpu().numpy()
        if threshold is None:
            preds = torch.argmax(probs, dim=1).detach().cpu().numpy()
        else:
            preds = (p1 >= float(threshold)).astype(np.int64)

        y_true.extend(y.detach().cpu().numpy())
        y_pred.extend(preds)
        y_score.extend(p1)

    return running_loss / len(loader.dataset), np.asarray(y_true), np.asarray(y_pred), np.asarray(y_score)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def eval_epoch_with_logits(model, loader, criterion, device, tta: int = 0):
    """Evaluate and return (loss, y_true, prob_pos, logit_diff).

    - prob_pos is the uncalibrated softmax probability for class 1.
    - logit_diff is (logit_1 - logit_0) averaged over TTA passes; useful for temperature scaling.
    """
    model.eval()
    running_loss = 0.0
    y_true, y_score, logit_diff_list = [], [], []

    tta = int(tta) if tta is not None else 0
    n_passes = max(1, tta)

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device)
        y = torch.as_tensor(y, device=device)

        out0 = model(x)
        logits0 = _primary_logits(out0)
        loss = criterion(logits0, y)

        logits_acc = logits0.detach()
        for k in range(1, n_passes):
            xk = _tta_transform(k)(x)
            outk = model(xk)
            logitsk = _primary_logits(outk).detach()
            logits_acc = logits_acc + logitsk

        logits_mean = logits_acc / float(n_passes)
        probs = torch.softmax(logits_mean, dim=1)

        running_loss += float(loss.item()) * x.size(0)

        p1 = probs[:, 1].detach().cpu().numpy()
        ld = (logits_mean[:, 1] - logits_mean[:, 0]).detach().cpu().numpy()

        y_true.extend(y.detach().cpu().numpy())
        y_score.extend(p1)
        logit_diff_list.extend(ld)

    return running_loss / len(loader.dataset), np.asarray(y_true), np.asarray(y_score), np.asarray(logit_diff_list)


def fit_temperature_scaling(logit_diff: np.ndarray, y_true: np.ndarray, max_iter: int = 200) -> float:
    """Fit temperature T > 0 on validation set using BCEWithLogitsLoss.

    We calibrate the binary logit difference: p = sigmoid((logit1-logit0)/T).
    Returns T (float). If optimization fails, returns 1.0.
    """
    try:
        ld = torch.tensor(logit_diff, dtype=torch.float32)
        yt = torch.tensor(y_true.astype(np.float32), dtype=torch.float32)
        # parameterize T with softplus to keep it positive
        t_param = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))  # softplus(0)=0.693...
        opt = torch.optim.LBFGS([t_param], lr=0.5, max_iter=max_iter)

        bce = torch.nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            T = torch.nn.functional.softplus(t_param) + 1e-6
            loss = bce(ld / T, yt)
            loss.backward()
            return loss

        opt.step(closure)
        T = float((torch.nn.functional.softplus(t_param) + 1e-6).item())
        if not np.isfinite(T) or T <= 0:
            return 1.0
        return T
    except Exception:
        return 1.0

# -----------------------------
# Retrieval-Augmented Inference (Option A: kNN over train embeddings)
# -----------------------------
def _find_last_linear_module(model: nn.Module) -> Tuple[str, nn.Linear]:
    """Return (name, module) of the last nn.Linear in the model."""
    last_name = ""
    last_mod: Optional[nn.Linear] = None
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_name = n
            last_mod = m
    if last_mod is None:
        raise RuntimeError("RAG: could not find a nn.Linear layer to hook for embeddings.")
    return last_name, last_mod


def _build_embedding_loader(
    samples: List[Tuple[Path, int]],
    abs_indices: List[int],
    img_size: int,
    batch_size: int,
    num_workers: int,
    args: argparse.Namespace,
) -> DataLoader:
    """Build a deterministic (eval-transform) loader over a subset of samples."""
    _, eval_tf = build_transforms(
        img_size=int(img_size),
        online_augmentation=False,
        randaugment=False,
        autoaugment=False,
        randaugment_n=0,
        randaugment_m=0,
        aug_strength=str(getattr(args, "aug_strength", "light")),
    )

    if bool(getattr(args, "use_smart_cache", False)):
        base = CachedPathLabelDataset(
            samples,
            transform=eval_tf,
            cache_dir=(str(getattr(args, "cache_dir", "")).strip() or None),
            max_memory_gb=float(getattr(args, "cache_memory_gb", 4.0)),
            img_size=int(img_size),
        )
    else:
        base = PathLabelDataset(samples, transform=eval_tf)

    ds: Dataset = Subset(base, abs_indices)

    # Keep Quanv wrapping if the training run enabled it (rare; mostly off)
    if bool(getattr(args, "use_quanv", False)):
        try:
            q_cache = Path(str(getattr(args, "quanv_cache_dir", "")).strip()) if str(getattr(args, "quanv_cache_dir", "")).strip() else None
            ds = QuanvDataset(
                ds,
                input_size=int(getattr(args, "quanv_input_size", 32)),
                patch=int(getattr(args, "quanv_patch", 2)),
                qubits=int(getattr(args, "quanv_qubits", 4)),
                depth=int(getattr(args, "quanv_depth", 1)),
                seed=int(getattr(args, "quanv_seed", 42)),
                cache_dir=(q_cache / "rag" if q_cache else None),
            )
        except Exception as _e:
            print(f"[WARN] RAG: could not wrap loader with QuanvDataset; continuing without quanv wrapper. Error: {_e}")

    pin = True
    # On Windows, DataLoader worker processes can occasionally hang late in long runs.
    # For RAG embedding extraction we prefer deterministic, low-friction settings.
    if int(num_workers) <= 0:
        loader = DataLoader(
            ds,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=pin,
            drop_last=False,
        )
    else:
        loader = DataLoader(
            ds,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=pin,
            drop_last=False,
            persistent_workers=False,
            prefetch_factor=2,
        )
    return loader


@torch.no_grad()
def _extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: int = 0,
    amp: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer embeddings (hooked at last Linear input)."""
    model.eval()
    _, lin = _find_last_linear_module(model)

    captured: List[torch.Tensor] = []

    def _pre_hook(_m, inp):
        # inp is a tuple; first element is the features fed to Linear
        try:
            captured.append(inp[0].detach())
        except Exception:
            pass

    handle = lin.register_forward_pre_hook(_pre_hook)

    n_passes = max(1, int(tta) if tta else 0)
    use_amp = bool(amp) and (device.type == "cuda")
    ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    embs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    for x, y in tqdm(loader, desc="RAG-Embed", leave=False):
        x = x.to(device, non_blocking=True)

        emb_acc: Optional[torch.Tensor] = None
        for k in range(n_passes):
            xk = diverse_tta_transforms(k % 8)(x) if n_passes > 1 else x
            captured.clear()
            with ctx():
                _ = model(xk)
            if not captured:
                handle.remove()
                raise RuntimeError("RAG: embedding hook did not capture features; cannot build kNN index.")
            e = captured[-1]
            if e.ndim > 2:
                e = e.view(e.size(0), -1)
            emb_acc = e if emb_acc is None else (emb_acc + e)

        emb_mean = emb_acc / float(n_passes)
        embs.append(emb_mean.detach().cpu().numpy())
        ys.append(np.asarray(y))

    handle.remove()

    emb_np = np.concatenate(embs, axis=0) if embs else np.zeros((0, 1), dtype=np.float32)
    y_np = np.concatenate(ys, axis=0).astype(np.int64) if ys else np.zeros((0,), dtype=np.int64)
    return emb_np.astype(np.float32), y_np


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)


def _knn_positive_rate(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    query_emb: np.ndarray,
    k: int,
    metric: str = "cosine",
    weighted: bool = False,
    temp: float = 0.07,
    chunk: int = 512,
) -> np.ndarray:
    """Compute kNN-based positive probability for each query embedding."""
    n_train = int(train_emb.shape[0])
    if n_train <= 0:
        return np.zeros((query_emb.shape[0],), dtype=np.float32)

    k_eff = int(min(max(1, int(k)), n_train))
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric=str(metric))
    nbrs.fit(train_emb)

    out = np.zeros((query_emb.shape[0],), dtype=np.float32)
    chunk = int(chunk) if chunk else 512
    chunk = max(1, chunk)

    for s in range(0, query_emb.shape[0], chunk):
        e = min(query_emb.shape[0], s + chunk)
        dist, idx = nbrs.kneighbors(query_emb[s:e], n_neighbors=k_eff, return_distance=True)
        y_nb = train_y[idx].astype(np.float32)

        if weighted:
            t = float(temp) if float(temp) > 0 else 1e-6
            if str(metric).lower() == "cosine":
                sim = 1.0 - dist  # higher is better
                w = _softmax_rows(sim / t)
            else:
                # Convert distance to similarity by negative distance
                w = _softmax_rows((-dist) / t)
            out[s:e] = np.sum(w * y_nb, axis=1).astype(np.float32)
        else:
            out[s:e] = np.mean(y_nb, axis=1).astype(np.float32)

    return out


def rag_blend_ensemble_probs(
    base_val: np.ndarray,
    base_test: np.ndarray,
    *,
    args: argparse.Namespace,
    device: torch.device,
    samples: List[Tuple[Path, int]],
    splits: SplitIndices,
    seed_dir: Path,
    models_to_run: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Blend ensemble probabilities with a kNN probability over train embeddings.

    Returns: (val_blended, test_blended, p_knn_val, p_knn_test, meta)
    """
    rag_model = str(getattr(args, "rag_model", "auto")).strip()
    if rag_model.lower() == "auto":
        rag_model = str(models_to_run[0]) if models_to_run else ""
    if not rag_model:
        raise RuntimeError("RAG: could not resolve --rag_model (empty).")

    # Resolve checkpoint: prefer finetuned checkpoint if present.
    ckpt_dir = Path(seed_dir) / rag_model
    ckpt_path = ckpt_dir / "best_finetune.pt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"RAG: checkpoint not found for rag_model='{rag_model}' under {ckpt_dir}")

    img_size = auto_img_size_for_model(rag_model, int(getattr(args, "img_size", 224)))
    rag_net = create_model(
        rag_model,
        2,
        False,  # no need to download pretrained weights when loading a checkpoint
        drop_path_rate=float(getattr(args, "drop_path_rate", 0.1)),
    ).to(device)
    try:
        st = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        st = torch.load(ckpt_path, map_location=device)
    rag_net.load_state_dict(st)
    rag_net.eval()

    print(f"[RAG] Backbone={rag_model} | ckpt={ckpt_path.name} | img_size={img_size}")

    # Build index from TRAIN only (optionally ORIGINAL-only to avoid augmented near-duplicates).
    train_idx = list(map(int, list(splits.train)))
    if str(getattr(args, "rag_index_source", "train_all")).lower() == "train_orig_only":
        orig_n = int(getattr(args, "_orig_n", 0))
        if orig_n > 0:
            train_idx = [i for i in train_idx if int(i) < orig_n]

    if len(train_idx) == 0:
        raise RuntimeError("RAG: empty train index set after applying --rag_index_source filter.")

    bs = int(getattr(args, "batch_size", 16))
    nw = int(getattr(args, "rag_num_workers", 0)) if hasattr(args, "rag_num_workers") else 0
    amp_enabled = bool(getattr(args, "amp", False)) and (not bool(getattr(args, "no_amp", False))) and (device.type == "cuda")

    tr_loader = _build_embedding_loader(samples, train_idx, img_size, bs, nw, args)
    va_loader = _build_embedding_loader(samples, list(map(int, list(splits.val))), img_size, bs, nw, args)
    te_loader = _build_embedding_loader(samples, list(map(int, list(splits.test))), img_size, bs, nw, args)

    rag_tta = int(getattr(args, "rag_tta", 0) or 0)

    print(f"[RAG] Extract embeddings: train_index_size={len(train_idx)} | bs={bs} | workers={nw} | tta={rag_tta}")

    tr_emb, tr_y = _extract_embeddings(rag_net, tr_loader, device, tta=rag_tta, amp=amp_enabled)
    print(f"[RAG] Extract embeddings: val_size={len(list(splits.val))}")
    va_emb, _va_y = _extract_embeddings(rag_net, va_loader, device, tta=rag_tta, amp=amp_enabled)
    print(f"[RAG] Extract embeddings: test_size={len(list(splits.test))}")
    te_emb, _te_y = _extract_embeddings(rag_net, te_loader, device, tta=rag_tta, amp=amp_enabled)

    k = int(getattr(args, "rag_k", 25))
    metric = str(getattr(args, "rag_metric", "cosine")).lower()
    weighted = bool(getattr(args, "rag_weighted", False))
    temp = float(getattr(args, "rag_temp", 0.07))
    chunk = int(getattr(args, "rag_chunk", 512))

    p_knn_val = _knn_positive_rate(tr_emb, tr_y, va_emb, k=k, metric=metric, weighted=weighted, temp=temp, chunk=chunk)
    print(f"[RAG] Build kNN index + query: k={k} metric={metric} weighted={weighted} chunk={chunk}")

    p_knn_test = _knn_positive_rate(tr_emb, tr_y, te_emb, k=k, metric=metric, weighted=weighted, temp=temp, chunk=chunk)

    alpha = float(getattr(args, "rag_alpha", 0.25))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    val_blend = (1.0 - alpha) * base_val.astype(np.float32) + alpha * p_knn_val.astype(np.float32)
    test_blend = (1.0 - alpha) * base_test.astype(np.float32) + alpha * p_knn_test.astype(np.float32)

    meta = {
        "rag_model": rag_model,
        "checkpoint": str(ckpt_path),
        "img_size": int(img_size),
        "index_source": str(getattr(args, "rag_index_source", "train_all")),
        "train_index_size": int(len(train_idx)),
        "k": int(min(max(1, k), len(train_idx))),
        "alpha": float(alpha),
        "metric": metric,
        "weighted": bool(weighted),
        "temp": float(temp),
        "rag_tta": int(rag_tta),
        "rag_chunk": int(chunk),
    }

    # Free memory (best-effort)
    try:
        del rag_net
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    return val_blend, test_blend, p_knn_val, p_knn_test, meta





# -----------------------------
# Plotting (enhanced)
# -----------------------------
def plot_confusion_matrix(cm: np.ndarray, out_base: Path, title: str, dpi: int) -> None:
    class_names = ["Non-Monkeypox", "Monkeypox"]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=90, va="center", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    save_figure_dual(fig, out_base, dpi=dpi)
    plt.close(fig)


def plot_loss_curves(history: Dict[str, List[float]], out_base: Path, title: str, dpi: int) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    save_figure_dual(fig, out_base, dpi=dpi)
    plt.close(fig)


def plot_roc_pr(y_true: np.ndarray, y_score: np.ndarray, out_dir: Path, prefix: str, dpi: int) -> None:
    if len(np.unique(y_true)) == 2:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(fpr, tpr)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            fig.tight_layout()
            save_figure_dual(fig, out_dir / f"{prefix}_roc", dpi=dpi)
            plt.close(fig)

            p, r, _ = precision_recall_curve(y_true, y_score)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(r, p)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision–Recall Curve")
            fig.tight_layout()
            save_figure_dual(fig, out_dir / f"{prefix}_pr", dpi=dpi)
            plt.close(fig)
        except Exception:
            pass


# -----------------------------
# DataLoaders (enhanced)
# -----------------------------
def build_transforms(
    img_size: int,
    online_augmentation: bool,
    randaugment: bool = False,
    autoaugment: bool = False,
    randaugment_n: int = 2,
    randaugment_m: int = 9,
    aug_strength: str = "light",
):
    """Create train/eval torchvision transforms.

    - Train: optional on-the-fly augmentation (recommended when evaluating on original-only).
    - Eval: deterministic resize+center-crop.

    Notes:
    - We keep transforms conservative and stable across torchvision versions.
    - Normalization uses ImageNet stats (works best with pretrained backbones).
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # --- Eval (deterministic) ---
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    if not online_augmentation:
        # Train mirrors eval when augmentation is disabled
        train_tf = eval_tf
        return train_tf, eval_tf

    # --- Train (with augmentation) ---
    # Strength presets
    aug_strength = (aug_strength or "light").lower()
    if aug_strength in ("strong", "heavy"):
        cj = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.35, hue=0.05)
        rot = 20
        scale = (0.75, 1.0)
    else:
        cj = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)
        rot = 15
        scale = (0.85, 1.0)

    t_list = [
        transforms.RandomResizedCrop(img_size, scale=scale),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(rot),
    ]

    # Optional: AutoAugment / RandAugment (if available)
    if autoaugment:
        try:
            t_list.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET))
        except Exception:
            pass
    if randaugment:
        try:
            t_list.append(transforms.RandAugment(num_ops=int(randaugment_n), magnitude=int(randaugment_m)))
        except Exception:
            pass

    t_list.extend(
        [
            cj,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_tf = transforms.Compose(t_list)
    return train_tf, eval_tf


def build_loaders(
    samples: List[Tuple[Path, int]],
    splits: SplitIndices,
    img_size: int,
    batch_size: int,
    num_workers: int,
    online_augmentation: bool,
    use_weighted_sampler: bool = False,
    randaugment: bool = False,
    autoaugment: bool = False,
    randaugment_n: int = 2,
    randaugment_m: int = 9,
    aug_strength: str = "light",
    use_smart_cache: bool = False,
    cache_dir: Optional[str] = None,
    cache_memory_gb: float = 4.0,
    # NEW: domain mixing (original vs augmented) for TRAIN only
    train_mix: str = "aug",
    train_orig_ratio: float = 0.7,
    max_aug_per_group: int = 0,
    orig_n: int = 0,
    group_ids: Optional[List[str]] = None,
    rng_seed: int = 42,
    # NEW: quanv feature extraction
    use_quanv: bool = False,
    quanv_input_size: int = 32,
    quanv_patch: int = 2,
    quanv_qubits: int = 4,
    quanv_depth: int = 1,
    quanv_seed: int = 42,
    quanv_cache_dir: Optional[str] = None,
):
    """
    Build train/val/test loaders.

    Notes:
    - Val/Test follow `splits.val` / `splits.test` exactly.
    - Train set can be re-arranged via `train_mix` when `samples` is a concatenation
      of [original] + [augmented] samples (originals first, length=orig_n).
    - If train_mix == 'original+aug', we enforce a DOMAIN ratio using WeightedRandomSampler.
      If `use_weighted_sampler` is also True, we combine domain weights with class balancing weights.
    """
    train_tf, eval_tf = build_transforms(
        img_size=img_size,
        online_augmentation=online_augmentation,
        randaugment=randaugment,
        autoaugment=autoaugment,
        randaugment_n=randaugment_n,
        randaugment_m=randaugment_m,
        aug_strength=aug_strength,
    )

    if use_smart_cache:
        base_train = CachedPathLabelDataset(
            samples, transform=train_tf, cache_dir=cache_dir, max_memory_gb=cache_memory_gb, img_size=img_size
        )
        base_eval = CachedPathLabelDataset(
            samples, transform=eval_tf, cache_dir=cache_dir, max_memory_gb=cache_memory_gb, img_size=img_size
        )
    else:
        base_train = PathLabelDataset(samples, transform=train_tf)
        base_eval = PathLabelDataset(samples, transform=eval_tf)

    # --- Build TRAIN indices according to train_mix ---
    train_indices = list(splits.train)

    # If we don't have a meaningful original/aug boundary, fall back to existing behavior
    have_domain_split = (orig_n is not None and orig_n > 0 and orig_n < len(samples))

    if train_mix == "original" and have_domain_split:
        train_indices = [i for i in train_indices if i < orig_n]
    elif train_mix == "original+aug" and have_domain_split:
        orig_idx = [i for i in train_indices if i < orig_n]
        aug_idx = [i for i in train_indices if i >= orig_n]

        # Optional: cap augmented samples per group
        if max_aug_per_group and max_aug_per_group > 0 and group_ids is not None and len(group_ids) == len(samples):
            by_g: Dict[str, List[int]] = {}
            for i in aug_idx:
                by_g.setdefault(str(group_ids[i]), []).append(i)
            rng = random.Random(int(rng_seed))
            aug_idx2: List[int] = []
            for g, idxs in by_g.items():
                if len(idxs) <= max_aug_per_group:
                    aug_idx2.extend(idxs)
                else:
                    rng.shuffle(idxs)
                    aug_idx2.extend(idxs[: int(max_aug_per_group)])
            aug_idx = aug_idx2

        # Keep all originals + capped augmented
        train_indices = orig_idx + aug_idx

    train_ds = Subset(base_train, train_indices)
    val_ds = Subset(base_eval, splits.val)
    test_ds = Subset(base_eval, splits.test)

    # --- Samplers ---
    sampler = None

    # Helper: class weights for a list of *absolute* indices into `samples`
    def _class_sample_weights(abs_indices: List[int]) -> np.ndarray:
        train_labels = [samples[i][1] for i in abs_indices]
        class_counts = np.bincount(train_labels)
        class_counts = np.clip(class_counts, 1, None)
        class_weights = 1.0 / class_counts
        return class_weights[train_labels].astype(np.float64)

    if train_mix == "original+aug" and have_domain_split:
        # Domain weights over the train subset (indices in the subset order)
        n = len(train_indices)
        if n == 0:
            domain_w = np.ones(0, dtype=np.float64)
        else:
            # Identify domain per sample in subset order
            is_orig = np.array([1.0 if (idx < orig_n) else 0.0 for idx in train_indices], dtype=np.float64)
            n_orig = float(is_orig.sum())
            n_aug = float(n - n_orig)

            # normalize to ratio across sampling mass
            r = float(train_orig_ratio)
            r = max(0.0, min(1.0, r))
            w_orig = (r / n_orig) if n_orig > 0 else 0.0
            w_aug = ((1.0 - r) / n_aug) if n_aug > 0 else 0.0
            domain_w = np.array([w_orig if idx < orig_n else w_aug for idx in train_indices], dtype=np.float64)

        if use_weighted_sampler:
            class_w = _class_sample_weights(train_indices)
            w = domain_w * class_w
        else:
            w = domain_w

        # Safety
        if w.size == 0:
            sampler = None
        else:
            w = np.clip(w, 1e-12, None)
            sampler = WeightedRandomSampler(weights=w.tolist(), num_samples=len(w), replacement=True)

    elif use_weighted_sampler:
        # Original behavior: class balancing weights on splits.train
        class_w = _class_sample_weights(train_indices)
        sampler = WeightedRandomSampler(weights=class_w.tolist(), num_samples=len(class_w), replacement=True)

    # --- Quanv wrapping (optional) ---
    if use_quanv:
        q_cache = Path(quanv_cache_dir) if quanv_cache_dir else None
        train_ds = QuanvDataset(train_ds, input_size=quanv_input_size, patch=quanv_patch, qubits=quanv_qubits, depth=quanv_depth, seed=quanv_seed, cache_dir=(q_cache / "train" if q_cache else None))
        val_ds = QuanvDataset(val_ds, input_size=quanv_input_size, patch=quanv_patch, qubits=quanv_qubits, depth=quanv_depth, seed=quanv_seed, cache_dir=(q_cache / "val" if q_cache else None))
        test_ds = QuanvDataset(test_ds, input_size=quanv_input_size, patch=quanv_patch, qubits=quanv_qubits, depth=quanv_depth, seed=quanv_seed, cache_dir=(q_cache / "test" if q_cache else None))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
    )

    return train_loader, val_loader, test_loader
def load_names_txt(txt_path: Union[str, Path]) -> List[str]:
    """
    Load a newline-separated list of image filenames.
    Blank lines and comment lines starting with '#' are ignored.
    Returns a de-duplicated list while preserving order.
    """
    p = Path(txt_path)
    if not p.exists():
        raise FileNotFoundError(f"List file not found: {p}")
    names: List[str] = []
    seen: Set[str] = set()
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            continue
        seen.add(line)
        names.append(line)
    return names

# -----------------------------
# Main (enhanced with balancing)
# -----------------------------


def export_failures_csv(out_csv: Path, samples: List[Tuple[Path, int]], y_true: np.ndarray, y_prob: np.ndarray, thr: float, positive_label: int = 1, indices: Optional[List[int]] = None) -> None:
    """Save misclassified samples to CSV for error analysis."""
    if y_true.size == 0:
        return
    y_pred = (y_prob >= float(thr)).astype(int)
    wrong = np.where(y_pred != y_true.astype(int))[0]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'true', 'pred', 'prob_pos', 'threshold'])
        for i in wrong.tolist():
            idx = indices[i] if indices is not None else i
            pth = samples[idx][0]
            w.writerow([Path(pth).name, int(y_true[i]), int(y_pred[i]), float(y_prob[i]), float(thr)])



def parse_seeds(seeds_str: str) -> List[int]:
    """Parse seeds from a string like '42,43,44' or '42-46'.

    Supports commas and/or whitespace as separators. Ranges are inclusive.
    Returns a de-duplicated list preserving order.
    """
    if seeds_str is None:
        return [42]
    s = str(seeds_str).strip()
    if not s:
        return [42]
    parts = re.split(r"[\s,]+", s)
    seeds: List[int] = []
    for part in parts:
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = a.strip(); b = b.strip()
            if a == "" or b == "":
                raise ValueError(f"Invalid seed range: '{part}'")
            start = int(a); end = int(b)
            step = 1 if end >= start else -1
            seeds.extend(list(range(start, end + step, step)))
        else:
            seeds.append(int(part))
    # de-dupe preserving order
    seen = set()
    out: List[int] = []
    for x in seeds:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

# -----------------------------
# Optional Quantum-Classical baseline (Quanvolution + MLP)
# -----------------------------
def _try_import_pennylane():
    try:
        import pennylane as qml  # type: ignore
        return qml
    except Exception as e:
        # Keep this message single-line safe for syntax
        raise ImportError(
            "PennyLane is required for --use_quanv. "
            "If you see an autoray.NumpyMimic error, try: pip install -U 'pennylane>=0.40' 'autoray<0.8'. "
            f"Original import error: {e}"
        )

_QUANV_CACHE: Dict[Tuple[int, int, int], Any] = {}

def _get_quanv_qnode(qubits: int, depth: int, seed: int):
    key = (qubits, depth, seed)
    if key in _QUANV_CACHE:
        return _QUANV_CACHE[key]
    qml = _try_import_pennylane()
    rng = np.random.RandomState(seed)
    dev = qml.device("default.qubit", wires=qubits)
    init_w = rng.normal(loc=0.0, scale=0.5, size=(depth, qubits, 3)).astype(np.float32)

    @qml.qnode(dev, interface="numpy")
    def circuit(x, w):
        for i in range(qubits):
            qml.RY(np.pi * x[i], wires=i)
        for d in range(depth):
            for i in range(qubits):
                qml.Rot(w[d, i, 0], w[d, i, 1], w[d, i, 2], wires=i)
            for i in range(qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if qubits > 1:
                qml.CNOT(wires=[qubits - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(qubits)]

    _QUANV_CACHE[key] = (circuit, init_w)
    return circuit, init_w

def quanv_features_from_tensor(img_t: torch.Tensor, input_size: int, patch: int, qubits: int, depth: int, seed: int) -> torch.Tensor:
    if img_t.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(img_t.shape)}")
    if img_t.shape[0] == 3:
        x = img_t.mean(dim=0, keepdim=True)
    else:
        x = img_t[:1]
    x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(input_size, input_size), mode="bilinear", align_corners=False).squeeze(0)
    x = x.clamp(0, 1)

    circuit, w = _get_quanv_qnode(int(qubits), int(depth), int(seed))
    H = input_size
    P = patch
    out_h = H // P
    out = np.zeros((qubits, out_h, out_h), dtype=np.float32)
    arr = x.squeeze(0).cpu().numpy()

    for i in range(out_h):
        for j in range(out_h):
            patch_arr = arr[i*P:(i+1)*P, j*P:(j+1)*P]
            flat = patch_arr.reshape(-1)
            if flat.size < qubits:
                flat = np.pad(flat, (0, qubits - flat.size), mode="wrap")
            x_in = flat[:qubits].astype(np.float32)
            out[:, i, j] = np.array(circuit(x_in, w), dtype=np.float32)
    return torch.from_numpy(out)

class QuanvDataset(Dataset):
    def __init__(self, base: Dataset, input_size: int, patch: int, qubits: int, depth: int, seed: int, cache_dir: Optional[Path] = None):
        self.base = base
        self.input_size = int(input_size)
        self.patch = int(patch)
        self.qubits = int(qubits)
        self.depth = int(depth)
        self.seed = int(seed)
        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]

        # Try to recover a stable file path for caching (works for Subset(PathLabelDataset/CachedPathLabelDataset))
        path_str = None
        abs_idx = idx
        base = self.base
        try:
            if isinstance(base, Subset):
                abs_idx = int(base.indices[idx])
                ds = base.dataset
            else:
                ds = base
            if hasattr(ds, "samples") and abs_idx < len(getattr(ds, "samples")):
                # samples is list[(str_path, y)]
                path_str = str(ds.samples[abs_idx][0])
        except Exception:
            path_str = None

        xt = x if isinstance(x, torch.Tensor) else transforms.ToTensor()(x)

        fp = None
        if self.cache_dir:
            # Stable cache key: hash(file_path) when available; fallback to absolute idx.
            if path_str:
                import hashlib
                key = hashlib.sha1(path_str.encode("utf-8", errors="ignore")).hexdigest()[:16]
            else:
                key = f"idx{int(abs_idx)}"
            fp = self.cache_dir / f"q_{key}_{self.input_size}_{self.patch}_{self.qubits}_{self.depth}_seed{self.seed}.pt"
            if fp.exists():
                return torch.load(fp), y

        feat = quanv_features_from_tensor(xt, self.input_size, self.patch, self.qubits, self.depth, self.seed)
        if fp is not None:
            try:
                torch.save(feat, fp)
            except Exception:
                pass
        return feat, y

class QuanvMLP(nn.Module):
    def __init__(self, in_ch: int, grid: int, hidden: int = 128, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        in_dim = int(in_ch) * int(grid) * int(grid)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class QuanvBackboneWrapper(nn.Module):
    """Upsample Quanv feature maps and feed into a standard CNN backbone (timm)."""
    def __init__(self, backbone: nn.Module, target_hw: int = 224):
        super().__init__()
        self.backbone = backbone
        self.target_hw = int(target_hw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"QuanvBackboneWrapper expects 4D tensor [B,C,H,W], got shape {tuple(x.shape)}")

        # --- Channel safety ---
        # Quanv models usually produce C=quanv_qubits channels. If, for any reason, the loader outputs RGB (C=3)
        # while the backbone was constructed with in_chans=4, we auto-adapt instead of crashing.
        try:
            expected = int(getattr(getattr(self.backbone, "conv_stem", None), "in_channels", x.shape[1]))
        except Exception:
            expected = x.shape[1]

        if x.shape[1] != expected:
            if not hasattr(self, "_warned_channels"):
                print(f"[Quanv] [WARN] Input channels={x.shape[1]} but backbone expects in_chans={expected}. Auto-adapting channels.")
                self._warned_channels = True

            # Special-case: expected=4, got=3 -> add a grayscale channel for stability (better than zeros)
            if x.shape[1] == 3 and expected == 4:
                gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
                x = torch.cat([x, gray], dim=1)
            elif x.shape[1] < expected:
                pad = expected - x.shape[1]
                z = torch.zeros((x.shape[0], pad, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
                x = torch.cat([x, z], dim=1)
            else:
                x = x[:, :expected, :, :]

        if x.shape[-2:] != (self.target_hw, self.target_hw):
            x = F.interpolate(x, size=(self.target_hw, self.target_hw), mode="bilinear", align_corners=False)

        return self.backbone(x)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI benchmark pipeline for leakage-aware mpox image classification."
    )

    # Data arguments
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--pos_class", type=str, default="monkeypox")
    parser.add_argument("--neg_class", type=str, default="normal")
    

    # Optional: include an additional ORIGINAL (non-augmented) dataset root alongside augmented data
    parser.add_argument('--orig_root', type=str, default=None,
                        help='Optional path to ORIGINAL images root (e.g., .../Original Images). If set, the script can train using both original+augmented, while evaluating on original-only to reduce overfitting.')
    parser.add_argument('--orig_pos_class', type=str, default=None,
                        help='Positive subfolder name under orig_root (e.g., Monkey Pox). If omitted, uses --pos_class.')
    parser.add_argument('--orig_neg_class', type=str, default=None,
                        help='Negative subfolder name under orig_root (e.g., Others). If omitted, uses --neg_class.')
    parser.add_argument('--orig_pos_list_file', type=str, default=None,
                        help='Optional: text file listing POSITIVE ORIGINAL image filenames (one per line) under orig_pos_class folder')
    parser.add_argument('--orig_neg_list_file', type=str, default=None,
                        help='Optional: text file listing NEGATIVE ORIGINAL image filenames (one per line) under orig_neg_class folder')
    parser.add_argument('--eval_on_original_only', action='store_true',
                        help='If set (recommended when orig_root is provided), validation/test will use ORIGINAL images only; augmented images will be used for TRAIN only (by group).')
    parser.add_argument('--include_augmented_in_eval', action='store_true',
                        help='If set, evaluation sets may include augmented images too (not recommended; can inflate validation).')

    # Domain mixing / two-stage training (original vs augmented)
    parser.add_argument('--train_mix', type=str, default='aug',
                        choices=['aug', 'original', 'original+aug'],
                        help="Training data source: 'aug' (default), 'original', or 'original+aug' (mixed with ratio). "
                             "When --eval_on_original_only is used, splits are derived from original groups; augmented samples are aligned by group.")
    parser.add_argument('--train_orig_ratio', type=float, default=0.7,
                        help='When --train_mix=original+aug, probability mass assigned to ORIGINAL samples in the mixed sampler (0..1).')
    parser.add_argument('--max_aug_per_group', type=int, default=3,
                        help='When --train_mix=original+aug, cap the number of augmented samples per group (0 disables cap).')
    parser.add_argument('--finetune_on_original', action='store_true',
                        help='After normal training, run a short fine-tune phase on ORIGINAL-only training samples (recommended for domain shift).')
    parser.add_argument('--finetune_epochs', type=int, default=5,
                        help='Number of fine-tuning epochs when --finetune_on_original is set.')
    parser.add_argument('--finetune_lr', type=float, default=1e-5,
                        help='Learning rate used during the fine-tune phase on ORIGINAL only.')


    parser.add_argument("--pos_list_file", type=str, default=None,

                       help="Optional: text file listing POSITIVE image filenames (one per line) under pos_class folder")

    parser.add_argument("--neg_list_file", type=str, default=None,

                       help="Optional: text file listing NEGATIVE image filenames (one per line) under neg_class folder")

    parser.add_argument("--strict_lists", action="store_true",

                       help="If set, abort when any filename from the list files is missing on disk")
    # New: Dataset balancing arguments
    parser.add_argument("--balance_strategy", type=str, default="none", 
                       choices=["none", "undersample", "oversample"],
                       help="Strategy for balancing classes")
    parser.add_argument("--balance_level", type=str, default="train",
                       choices=["dataset", "train", "none"],
                       help="Level at which to apply balancing (dataset: before split, train: only training set)")
    parser.add_argument("--analyze_dataset", action="store_true",
                       help="Analyze and plot dataset distribution before training")
    parser.add_argument("--use_weighted_sampler", action="store_true",
                       help="Use weighted random sampler for training (alternative to balancing)")

    # Model arguments
    parser.add_argument("--use_quanv", action="store_true", help="Enable a quantum-classical baseline (Quanv+MLP). Requires PennyLane.")
    parser.add_argument("--quanv_input_size", type=int, default=32)
    parser.add_argument("--quanv_patch", type=int, default=2)
    parser.add_argument("--quanv_qubits", type=int, default=4)
    parser.add_argument("--quanv_depth", type=int, default=1)
    parser.add_argument("--quanv_cache_dir", type=str, default="")

    parser.add_argument("--models", type=str, default="efficientnet_b0,efficientnetv2_s,convnext_tiny,convnextv2_tiny,vit_b_16,densenet121,resnet50")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--plot_dpi", type=int, default=600)

    # Split arguments
    parser.add_argument("--split_ratios", type=str, default="0.70,0.10,0.20")
    parser.add_argument("--split_strategy", type=str, default="group_stratified", 
                       choices=["group_stratified", "group", "random_stratified"])
    parser.add_argument("--naive_split_ablation", action="store_true",
                       help="Leakage ablation: use plain StratifiedKFold (ignoring inferred groups) "
                            "instead of StratifiedGroupKFold within the --cv_folds path, with augmented "
                            "images assigned to training at random rather than by group. Quantifies the "
                            "leakage the grouped protocol is designed to prevent (writes "
                            "group_overlap_report_naive.csv) rather than trying to eliminate it. "
                            "Intended for a separate ablation run, not the main benchmark.")
    parser.add_argument("--export_fold_composition", action="store_true", default=True,
                       help="Write fold_composition.csv (per-fold class/source-version/unique-group "
                            "counts for every train/val/test partition). On by default.")
    parser.add_argument("--export_val_bundle", action="store_true", default=True,
                       help="Write validation_bundle.npz alongside prediction_bundle.npz, enabling "
                            "post-hoc threshold-strategy comparison via extract_prediction_bundles.py. "
                            "On by default.")

    # Group arguments
    parser.add_argument("--group_mode", type=str, default="regex",
                       choices=["stem", "prefix", "prefix2", "regex", "full"])
    parser.add_argument("--group_regex", type=str, default=None,
                       help="Used when --group_mode=regex. Must contain ONE capturing group for the group id.")

    # Data quality arguments
    parser.add_argument("--remove_exact_duplicates", action="store_true")
    parser.add_argument("--check_exact_overlap", action="store_true")

    # Augmentation arguments
    parser.add_argument("--online_augmentation", action="store_true")
    parser.add_argument("--aug_strength", type=str, default="light", choices=["none","light","medium","strong"],
                       help="Augmentation preset (applies when --online_augmentation is set, or when aug_strength != none).")
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--grad_clip_norm", type=float, default=0.0)


    # v10: optimizer/loss/model robustness
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sam"],
                       help="Optimizer: adamw (default), adam, or sam (Sharpness-Aware Minimization using AdamW as base).")
    parser.add_argument("--sam_rho", type=float, default=0.05, help="SAM neighborhood size (rho).")
    parser.add_argument("--sam_adaptive", action="store_true", help="Use adaptive SAM (ASAM).")


    # Performance/throughput
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                       help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps). Disabled for SAM.")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP) on CUDA.")
    parser.add_argument("--no_amp", action="store_true", help="Force-disable AMP.")

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine", "warmup_cosine", "none"],
                       help="LR scheduler. plateau=ReduceLROnPlateau (default). cosine=CosineAnnealingLR. warmup_cosine=linear warmup then cosine.")
    parser.add_argument("--warmup_epochs", type=float, default=1.0, help="Warmup epochs for warmup_cosine.")


    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"],
                       help="Loss function: cross-entropy (ce) or focal.")
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument("--drop_path_rate", type=float, default=0.1,
                       help="Stochastic depth probability (ViT/ConvNeXt/ConvNeXtV2 where supported).")

    parser.add_argument("--randaugment", action="store_true", help="Enable RandAugment policy on training set.")
    parser.add_argument("--autoaugment", action="store_true", help="Enable AutoAugment policy on training set.")
    parser.add_argument("--randaugment_n", type=int, default=2)
    parser.add_argument("--randaugment_m", type=int, default=9)

    # Experiment tracking (optional)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases if installed.")
    parser.add_argument("--wandb_project", type=str, default="mpox")
    parser.add_argument("--wandb_name", type=str, default=None)

    parser.add_argument("--mlflow", action="store_true", help="Log metrics to MLflow if installed.")
    parser.add_argument("--mlflow_uri", type=str, default=None)
    parser.add_argument("--mlflow_experiment", type=str, default=None)
    parser.add_argument("--mlflow_run_name", type=str, default=None)

    # Advanced training arguments (from your original)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--cutmix_alpha", type=float, default=0.0)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=3)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--tta", type=int, default=8)
    parser.add_argument("--threshold_strategy", type=str, default="f1_at_precision",
                       choices=["none", "f1", "youden", "prec_at_recall", "f1_at_precision", "recall_at_precision"],
                       help="How to choose the decision threshold on validation scores.")
    parser.add_argument("--min_recall", type=float, default=0.85,
                       help="Used when --threshold_strategy=prec_at_recall: choose threshold maximizing precision subject to recall>=min_recall")
    parser.add_argument("--precision_target", type=float, default=0.90,
                       help="Used when --threshold_strategy is f1_at_precision/recall_at_precision: enforce precision >= precision_target when choosing threshold.")
    parser.add_argument("--calibrate", type=str, default="none", choices=["none", "temperature"],
                       help="Optional post-hoc probability calibration on validation set (temperature scaling)")
    parser.add_argument("--weighted_ensemble", action="store_true")
    parser.add_argument("--ensemble_weight_opt", type=str, default="random_search", choices=["auc", "random_search"],
                        help="How to choose weights for --weighted_ensemble (default: random_search).")
    parser.add_argument("--ensemble_weight_metric", type=str, default="f1_at_precision",
                        choices=["f1_at_precision", "recall_at_precision", "auc", "pr_auc", "neg_logloss", "neg_brier"],
                        help="Metric optimized on validation to pick ensemble weights (default: f1_at_precision).")
    parser.add_argument("--ensemble_weight_trials", type=int, default=2048,
                        help="Number of random weight samples when ensemble_weight_opt=random_search.")
    parser.add_argument("--ensemble_weight_seed", type=int, default=123,
                        help="RNG seed for ensemble weight optimization.")

    # Retrieval-Augmented Inference (RAG): kNN over TRAIN embeddings blended into final ensemble probs
    parser.add_argument("--rag_enable", action="store_true",
                        help="Enable retrieval-augmented inference: kNN over train embeddings blended into final ensemble probabilities before thresholding.")
    parser.add_argument("--rag_model", type=str, default="auto",
                        help="Embedding backbone model name (must be one of --models), or 'auto' to use the first trained model in each fold.")
    parser.add_argument("--rag_index_source", type=str, default="train_all",
                        choices=["train_all", "train_orig_only"],
                        help="Which samples to index: train_all (default) or train_orig_only (avoid augmented near-duplicates).")
    parser.add_argument("--rag_k", type=int, default=25, help="Number of nearest neighbors for kNN retrieval.")
    parser.add_argument("--rag_alpha", type=float, default=0.25,
                        help="Blend factor for kNN: p_final = (1-alpha)*p_ensemble + alpha*p_knn.")
    parser.add_argument("--rag_metric", type=str, default="cosine", choices=["cosine", "euclidean"],
                        help="Distance metric for kNN index.")
    parser.add_argument("--rag_weighted", action="store_true",
                        help="Use distance-weighted neighbors via softmax(sim/temp) (cosine) or softmax(-dist/temp) (euclidean).")
    parser.add_argument("--rag_temp", type=float, default=0.07, help="Temperature used for neighbor weighting.")
    parser.add_argument("--rag_chunk", type=int, default=512, help="Chunk size (#queries per kneighbors call) to limit memory.")
    parser.add_argument("--rag_tta", type=int, default=0, help="TTA passes for embedding extraction (0 disables).")
    parser.add_argument("--rag_num_workers", type=int, default=0, help="Num workers for RAG embedding loaders (default 0 for Windows stability).")

    # Stacking / meta-learning ensemble
    parser.add_argument("--stacking_ensemble", "--staking", action="store_true",
                        help="Enable stacking (meta-learning): train a meta-learner on validation predictions to combine models.")
    parser.add_argument("--stacking_meta_model", type=str, default="logreg",
                        choices=["logreg", "logreg_cv"],
                        help="Meta-learner type for stacking (default: logreg).")
    parser.add_argument("--stacking_feature_mode", type=str, default="logit",
                        choices=["proba", "logit"],
                        help="Transform base model probabilities into stacking features (default: logit).")
    parser.add_argument("--stacking_top_k", type=int, default=0,
                        help="If >0, only use the top-K base models (ranked on validation by --stacking_rank_metric) for stacking.")
    parser.add_argument("--stacking_rank_metric", type=str, default="f1_at_precision",
                        choices=["f1_at_precision", "recall_at_precision", "auc", "pr_auc", "neg_logloss", "neg_brier"],
                        help="Validation metric used to rank base models when --stacking_top_k>0.")
    parser.add_argument("--stacking_C", type=float, default=1.0,
                        help="Inverse regularization strength for the logistic-regression meta-learner (higher = weaker regularization).")
    parser.add_argument("--stacking_cv_folds", type=int, default=0,
                        help="If >1 and stacking_meta_model=logreg_cv, run LogisticRegressionCV with this many folds on the validation set.")
    parser.add_argument("--stacking_no_standardize", action="store_true",
                        help="Disable feature standardization for stacking (not recommended).")


    # Training control arguments
    parser.add_argument("--early_stop_patience", type=int, default=0)
    # Backward-compatible alias (some runs used --early_stopping)
    parser.add_argument("--early_stopping", type=int, default=None, help="Alias for --early_stop_patience.")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0,
                        help="Minimum improvement to reset early-stopping patience.")
    parser.add_argument("--early_stop_metric", type=str, default="val_loss",
                        choices=["val_loss", "f1_at_precision", "recall_at_precision", "auc", "pr_auc", "brier"],
                        help="Metric to monitor for early stopping / best checkpoint (default: val_loss).")
    parser.add_argument("--early_stop_mode", type=str, default="auto", choices=["auto", "min", "max"],
                        help="Whether to minimize or maximize early_stop_metric (default: auto).")
    parser.add_argument("--force_full_epochs", action="store_true")
    parser.add_argument("--plateau_patience", type=int, default=10)
    parser.add_argument("--plateau_factor", type=float, default=0.5)

    # Experiment arguments
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")

    parser.add_argument("--cv_folds", type=int, default=0,
                        help="If >1, run GroupKFold CV on ORIGINAL images (recommended). Replaces --seeds loop.")
    parser.add_argument("--cv_val_frac", type=float, default=0.15,
                        help="Within each CV fold, fraction of ORIGINAL trainval groups used for validation.")
    parser.add_argument("--cv_print_groups", type=int, default=8,
                        help="Leakage sanity check: number of example groups to print before training (0 to disable).")


    # Hyperparameter optimization (optional)
    parser.add_argument("--hpo_method", type=str, default="random", choices=["random", "optuna"],
                        help="HPO method to use when --hpo is set.")
    parser.add_argument("--hpo", action="store_true",
                        help="Run a lightweight random-search hyperparameter optimization per model (cached) before full training.")
    parser.add_argument("--hpo_trials", type=int, default=20,
                        help="Number of HPO trials per model (random search).")
    parser.add_argument("--hpo_epochs", type=int, default=6,
                        help="Epochs per HPO trial (short training; keep small).")
    parser.add_argument("--hpo_metric", type=str, default="f1",
                        choices=["f1", "auc", "balanced_acc", "loss"],
                        help="Metric to optimize during HPO.")
    parser.add_argument("--hpo_seed", type=int, default=777,
                        help="Random seed for HPO sampling (reproducible).")
    parser.add_argument("--optuna_storage", type=str, default="",
                        help="Optuna storage URL for resuming studies (e.g., sqlite:///optuna.db).")
    parser.add_argument("--optuna_study", type=str, default="",
                        help="Optuna study name (default: auto per model).")
    parser.add_argument("--optuna_pruner", type=str, default="hyperband",
                        choices=["hyperband", "hb", "sha", "asha", "successive_halving", "median"],
                        help="Optuna pruning strategy. hyperband/asha are recommended for fast reliable HPO.")
    parser.add_argument("--optuna_reduction_factor", type=float, default=3,
                        help="Reduction factor for Hyperband/SHA (typical: 3).")
    parser.add_argument("--debug", action="store_true")
    # Advanced optional flags
    add_enhanced_args(parser)

    args = parser.parse_args()

    # Apply backward-compatible alias if provided
    if getattr(args, "early_stopping", None) is not None and int(args.early_stop_patience) == 0:
        args.early_stop_patience = int(args.early_stopping)
    # Configure Quanv model (if enabled)
    global _QUANV_MODEL_CFG
    if bool(getattr(args, "use_quanv", False)):
        grid = int(getattr(args, "quanv_input_size", 32)) // max(1, int(getattr(args, "quanv_patch", 2)))
        _QUANV_MODEL_CFG = {
            "in_ch": int(getattr(args, "quanv_qubits", 4)),
            "grid": int(grid),
            "hidden": 128,
            "dropout": 0.2,
        }
    else:
        _QUANV_MODEL_CFG = None


    # Validate and setup
    if args.group_mode == "regex" and (args.group_regex is None or args.group_regex.strip() == ""):
        # Default for filenames like: M01_01_00.jpg  /  NM101_02_13.jpg  -> group id = first two tokens
        args.group_regex = r"^([A-Za-z]+\d+_[0-9]+)_" 

    tr, vr, te = [float(x) for x in args.split_ratios.split(",")]
    if abs(tr + vr + te - 1.0) > 1e-6:
        raise ValueError("split_ratios must sum to 1.")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)


    # Optional experiment tracking
    wandb_run = maybe_init_wandb(args, config={k: v for k, v in vars(args).items() if k not in ['pos_list_file','neg_list_file']})
    mlflow_mod = maybe_init_mlflow(args)


    # Save configuration
    run_cfg = vars(args).copy()
    run_cfg["timestamp"] = datetime.now().isoformat(timespec="seconds")
    with open(out_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset(s)
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def _scan_class_dirs(root: Path, pos_class: str, neg_class: str, pos_list_file: Optional[str], neg_list_file: Optional[str]):
        pos_dir = root / pos_class
        neg_dir = root / neg_class
        if not pos_dir.exists() or not neg_dir.exists():
            raise FileNotFoundError(f"Expected class folders at: {pos_dir} and {neg_dir}")

        pos_files_all = sorted([p for p in pos_dir.rglob('*') if p.suffix.lower() in exts])
        neg_files_all = sorted([p for p in neg_dir.rglob('*') if p.suffix.lower() in exts])
        if not pos_files_all or not neg_files_all:
            raise RuntimeError(f"No images found under {pos_dir} / {neg_dir}")

        def _apply_list_filter(files_all, list_file, class_dir: Path):
            if not list_file:
                return files_all
            wanted = read_name_list(list_file)
            by_name = {}
            for p in files_all:
                by_name.setdefault(p.name, []).append(p)
            kept = []
            missing = []
            dup = []
            for name in wanted:
                if name not in by_name:
                    missing.append(name)
                    continue
                if len(by_name[name]) > 1:
                    dup.append(name)
                kept.append(by_name[name][0])
            if dup:
                print(f"[ListFilter] WARNING: {len(dup)} duplicate filenames under {class_dir}. Using first match.")
            if missing:
                msg = f"[ListFilter] Missing {len(missing)} filenames under {class_dir}. Example: {missing[:5]}"
                if args.strict_lists:
                    raise FileNotFoundError(msg)
                print(msg)
            return kept

        pos_files = _apply_list_filter(pos_files_all, pos_list_file, pos_dir)
        neg_files = _apply_list_filter(neg_files_all, neg_list_file, neg_dir)
        return pos_files, neg_files

    # Primary dataset root (typically augmented)
    data_root = Path(args.data_root)
    pos_files, neg_files = _scan_class_dirs(data_root, args.pos_class, args.neg_class, args.pos_list_file, args.neg_list_file)

    # Optional original dataset root
    orig_root = Path(args.orig_root) if args.orig_root else None
    orig_pos_files = []
    orig_neg_files = []
    if orig_root is not None:
        op = args.orig_pos_class if args.orig_pos_class else args.pos_class
        on = args.orig_neg_class if args.orig_neg_class else args.neg_class
        orig_pos_files, orig_neg_files = _scan_class_dirs(orig_root, op, on, args.orig_pos_list_file, args.orig_neg_list_file)

    # Remove duplicates if requested (within each pool, and cross-class within pool)
    def _dedup_pool(pos_list, neg_list, pool_name: str):
        if not args.remove_exact_duplicates:
            return pos_list, neg_list
        print(f"Removing byte-identical duplicates ({pool_name})...")
        pos_hash: Dict[str, Path] = {}
        neg_hash: Dict[str, Path] = {}

        def unique_files(files, store):
            uniq = []
            for p in files:
                h = md5_file(p)
                if h not in store:
                    store[h] = p
                    uniq.append(p)
            return uniq

        pos_u = unique_files(pos_list, pos_hash)
        neg_u = unique_files(neg_list, neg_hash)

        overlap = set(pos_hash.keys()).intersection(set(neg_hash.keys()))
        if overlap:
            to_remove = set(neg_hash[h] for h in overlap)
            neg_u = [p for p in neg_u if p not in to_remove]
            print(f"Removed {len(to_remove)} cross-class duplicates in {pool_name}.")

        print(f"After de-dup ({pool_name}): pos={len(pos_u)}, neg={len(neg_u)}")
        return pos_u, neg_u

    pos_files, neg_files = _dedup_pool(pos_files, neg_files, pool_name='primary')
    if orig_root is not None:
        orig_pos_files, orig_neg_files = _dedup_pool(orig_pos_files, orig_neg_files, pool_name='original')

    # Build samples with group IDs
    samples_aug = [(p, 1) for p in pos_files] + [(p, 0) for p in neg_files]
    labels_aug = [y for _, y in samples_aug]
    group_aug = [infer_group_id(p, args.group_mode, args.group_regex) for p, _ in samples_aug]

    samples_orig = [(p, 1) for p in orig_pos_files] + [(p, 0) for p in orig_neg_files]
    labels_orig = [y for _, y in samples_orig]
    group_orig = [infer_group_id(p, args.group_mode, args.group_regex) for p, _ in samples_orig]

    # Decide split base and final composition
    if orig_root is not None and (args.eval_on_original_only or not args.include_augmented_in_eval):
        # Recommended: split based on ORIGINAL images only; then add augmented images to TRAIN only (by group)
        base_samples = samples_orig
        base_labels = labels_orig
        base_groups = group_orig
        base_name = 'original'
    else:
        # Legacy behavior: split on primary pool (often augmented)
        base_samples = samples_aug + samples_orig
        base_labels = labels_aug + labels_orig
        base_groups = group_aug + group_orig
        base_name = 'combined'

    if not base_samples:
        raise RuntimeError('No images found after filtering.')

    # Create samples/labels/group_ids arrays used for splitting
    samples = base_samples
    labels = base_labels
    group_ids = base_groups

    print(f"Dataset pools: primary={len(samples_aug)} samples, original={len(samples_orig)} samples | splitting_base={base_name} ({len(samples)})")

    # Group/split leakage diagnostics (pre-split)

    gc = Counter(group_ids)
    n_groups = len(gc)
    avg_per_group = (len(group_ids) / n_groups) if n_groups else 0.0
    max_per_group = max(gc.values()) if n_groups else 0

    print(f"Group mode: {args.group_mode}  |  groups={n_groups}  avg_imgs_per_group={avg_per_group:.2f}  max_imgs_per_group={max_per_group}")

    # Detect filename augmentation suffix patterns (helps validate group_regex)
    fn_pat = detect_filename_suffix_patterns([p for p, _ in samples])
    if fn_pat["pct_base_num"] >= 20.0:
        print(f"[NamePattern] Detected many filenames ending with _<num>: {fn_pat['pct_base_num']:.1f}% of samples.")
        if fn_pat["suggested_group_regex"]:
            if args.group_mode == "regex" and args.group_regex != fn_pat["suggested_group_regex"]:
                print(f"[NamePattern] Suggested safer group_regex: --group_regex \"{fn_pat['suggested_group_regex']}\"")
            elif args.group_mode != "regex":
                print(f"[NamePattern] Suggested safer grouping: --group_mode regex --group_regex \"{fn_pat['suggested_group_regex']}\"")
    if fn_pat["pct_aug_num"] >= 5.0:
        print(f"[NamePattern] Detected filenames like base_<token>_<num>: {fn_pat['pct_aug_num']:.1f}% of samples.")
        print("[NamePattern] If your augmentation is encoded in filenames, consider --group_mode stem (or a regex that strips aug tokens).")

    if n_groups and avg_per_group < 1.05:
        print(
            "WARNING: Almost every image is its own group (avg≈1). "
            "That usually means leakage protection is ineffective.\\n"
            "         For filenames like M01_01_00.jpg / NM101_02_13.jpg, recommended:\\n"
            "           --group_mode regex --group_regex \"(.+?)_[0-9]+$\"\\n"
            "         or simply: --group_mode prefix2"
        )

    # NEW: Dataset analysis
    if args.analyze_dataset:
        print("\n" + "="*60)
        print("DATASET ANALYSIS")
        print("="*60)
        
        analyzer = DatasetAnalyzer(samples)
        stats = analyzer.get_statistics()
        
        print(f"\nTotal samples: {stats['total_samples']}")
        print(f"Class distribution: {stats['label_distribution']}")
        print(f"Class ratio (pos/neg): {stats['class_ratio']:.2f}")
        print(f"Imbalance ratio: {stats['imbalance_ratio']:.2f}")
        
        for label, label_stats in stats['subject_statistics'].items():
            print(f"\nClass {label}:")
            print(f"  Unique subjects: {label_stats['unique_subjects']}")
            print(f"  Images per subject: {label_stats['avg_images_per_subject']:.1f} ± {label_stats['std_images_per_subject']:.1f}")
        
        # Plot distribution
        analyzer.plot_distribution(out_root / "dataset_distribution", "Dataset Distribution Analysis")
        print(f"\nDistribution plot saved to: {out_root / 'dataset_distribution.png'}")
        print("="*60 + "\n")

    # NEW: Dataset balancing at dataset level
    if args.balance_strategy != 'none' and args.balance_level == 'dataset':
        print(f"\nBalancing entire dataset using {args.balance_strategy}...")
        balancer = DatasetBalancer(samples, group_ids)
        samples, group_ids = balancer.balance_dataset(
            strategy=args.balance_strategy,
            balance_level=args.balance_level
        )
        labels = [y for _, y in samples]

    # Runs: either GroupKFold CV on ORIGINAL split base, or repeated random splits across seeds
    metrics_by_model: Dict[str, Dict[str, List[float]]] = {}

    # Cache best HPO params per model name (computed on first encounter and reused across runs)
    hpo_cache: Dict[str, Dict[str, Any]] = {}
    moo_cache: Dict[str, Any] = {}  # per-model MOO results cache
    num_classes: int = 2  # binary classification (neg vs pos)

    # Keep explicit pools for CV logic
    orig_n = len(samples_orig)
    aug_n = len(samples_aug)
    if orig_n > 0 and aug_n > 0:
        full_samples = samples_orig + samples_aug
        full_labels = labels_orig + labels_aug
        full_group_ids = group_orig + group_aug
        aug_offset = orig_n

    else:
        full_samples = samples
        full_labels = labels
        full_group_ids = group_ids
        aug_offset = len(full_samples)
    # Internal helpers for domain-mixing loaders (used by HPO too)
    setattr(args, '_orig_n', int(orig_n))
    setattr(args, '_full_group_ids', full_group_ids)


    fold_splits: list[SplitIndices] = []
    run_ids: list[int] = []
    run_label = 'seed'

    def _make_val_split_from_trainval_orig(trainval_orig_idxs: list[int], val_frac: float, seed: int):
        # split ORIGINAL indices into train/val by groups with stratification at group level
        import numpy as np
        from collections import Counter
        from sklearn.model_selection import StratifiedShuffleSplit

        by_g: dict[str, list[int]] = {}
        for i in trainval_orig_idxs:
            by_g.setdefault(group_orig[i], []).append(labels_orig[i])

        groups = sorted(by_g.keys())
        if len(groups) < 4:
            return trainval_orig_idxs, []

        y_g = np.array([1 if Counter(by_g[g]).get(1, 0) >= Counter(by_g[g]).get(0, 0) else 0 for g in groups])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        tr_g_idx, va_g_idx = next(sss.split(groups, y_g))
        tr_groups = set(groups[i] for i in tr_g_idx)
        va_groups = set(groups[i] for i in va_g_idx)
        tr_idxs = [i for i in trainval_orig_idxs if group_orig[i] in tr_groups]
        va_idxs = [i for i in trainval_orig_idxs if group_orig[i] in va_groups]
        return tr_idxs, va_idxs

    def _augment_indices_for_train_groups(train_groups: set[str]) -> list[int]:
        # add augmented samples whose group id is in train_groups
        if orig_n == 0 or aug_n == 0:
            return []
        extra = []
        for j in range(aug_n):
            idx = aug_offset + j
            if full_group_ids[idx] in train_groups:
                extra.append(idx)
        return extra

    def _augment_indices_random_for_naive(k_folds: int, fold_idx: int, seed: int) -> list[int]:
        # Naive-split-ablation counterpart to _augment_indices_for_train_groups:
        # each augmented image is assigned to this fold's training set at random,
        # independent of which fold its source original was assigned to, with
        # probability matching the nominal (k-1)/k train share of a k-fold split.
        if orig_n == 0 or aug_n == 0:
            return []
        rng_local = np.random.default_rng(seed + fold_idx)
        keep_prob = (k_folds - 1) / float(k_folds)
        extra = []
        for j in range(aug_n):
            if rng_local.random() < keep_prob:
                extra.append(aug_offset + j)
        return extra

    _naive_split_ablation = bool(getattr(args, "naive_split_ablation", False))

    if args.cv_folds and args.cv_folds > 1:
        if orig_n == 0:
            raise RuntimeError('GroupKFold CV requires ORIGINAL images. Provide --orig_root and original list files.')
        run_label = 'fold'
        k = int(args.cv_folds)
        if _naive_split_ablation:
            # Leakage-inflation ablation: ignore inferred group membership entirely,
            # using plain StratifiedKFold on original images instead of
            # StratifiedGroupKFold, so that related images (originals + their
            # augmented derivatives) can land in different partitions.
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=12345)
            splitter = cv.split(range(orig_n), labels_orig)
        else:
            try:
                from sklearn.model_selection import StratifiedGroupKFold
                cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=12345)
                splitter = cv.split(range(orig_n), labels_orig, groups=group_orig)
            except Exception:
                from sklearn.model_selection import GroupKFold
                cv = GroupKFold(n_splits=k)
                splitter = cv.split(range(orig_n), labels_orig, groups=group_orig)

        _group_overlap_rows = []
        for fold_idx, (trainval_idx, test_idx) in enumerate(splitter):
            trainval_idx = list(map(int, trainval_idx))
            test_idx = list(map(int, test_idx))

            tr_orig, va_orig = _make_val_split_from_trainval_orig(trainval_idx, float(args.cv_val_frac), seed=12345 + fold_idx)

            tr_groups = set(group_orig[i] for i in tr_orig)
            va_groups = set(group_orig[i] for i in va_orig)
            te_groups = set(group_orig[i] for i in test_idx)

            if _naive_split_ablation:
                # Quantify, rather than prevent, the leakage this split strategy permits.
                overlap = te_groups & tr_groups
                n_test_groups = max(1, len(te_groups))
                _group_overlap_rows.append({
                    "fold": fold_idx,
                    "train_test_group_overlap": len(overlap),
                    "train_val_group_overlap": len(tr_groups & va_groups),
                    "val_test_group_overlap": len(va_groups & te_groups),
                    "n_test_groups": len(te_groups),
                    "pct_test_groups_also_in_train": round(100.0 * len(overlap) / n_test_groups, 4),
                })
            else:
                if (tr_groups & va_groups) or (tr_groups & te_groups) or (va_groups & te_groups):
                    raise RuntimeError('Internal error: group overlap across CV splits')
                _group_overlap_rows.append({
                    "fold": fold_idx, "train_test_group_overlap": 0, "train_val_group_overlap": 0,
                    "val_test_group_overlap": 0, "n_test_groups": len(te_groups),
                    "pct_test_groups_also_in_train": 0.0,
                })

            tr_idx_full = [i for i in tr_orig]
            va_idx_full = [i for i in va_orig]
            te_idx_full = [i for i in test_idx]

            if _naive_split_ablation:
                tr_idx_full += _augment_indices_random_for_naive(k, fold_idx, seed=12345)
            else:
                tr_idx_full += _augment_indices_for_train_groups(tr_groups)

            fold_splits.append(SplitIndices(train=tr_idx_full, val=va_idx_full, test=te_idx_full))

        try:
            _overlap_csv_path = Path(args.output_dir) / (
                "group_overlap_report_naive.csv" if _naive_split_ablation else "group_overlap_report.csv")
            _overlap_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(_overlap_csv_path, "w", newline="", encoding="utf-8") as _f:
                _writer = csv.DictWriter(_f, fieldnames=list(_group_overlap_rows[0].keys()))
                _writer.writeheader()
                _writer.writerows(_group_overlap_rows)
            print(f"[GroupOverlap] Wrote {_overlap_csv_path}")
        except Exception as e:
            print(f"WARNING: failed to write group overlap report: {e}")

        run_ids = list(range(len(fold_splits)))

        # Per-fold class / source-version / unique-group composition export,
        # for exact reproduction of dataset-composition reporting (Table 3 of
        # the manuscript) directly from the released outputs.
        if bool(getattr(args, "export_fold_composition", True)):
            try:
                _comp_rows = []
                for _fi, _sp in enumerate(fold_splits):
                    for _split_name, _idxs in [("train", _sp.train), ("val", _sp.val), ("test", _sp.test)]:
                        _n_pos = sum(1 for _i in _idxs if full_samples[_i][1] == 1)
                        _n_neg = sum(1 for _i in _idxs if full_samples[_i][1] == 0)
                        _n_v1 = sum(1 for _i in _idxs if Path(str(full_samples[_i][0])).name.startswith("v1__"))
                        _n_v2 = sum(1 for _i in _idxs if Path(str(full_samples[_i][0])).name.startswith("v2__"))
                        _n_groups = len(set(full_group_ids[_i] for _i in _idxs))
                        _comp_rows.append({
                            "fold": _fi, "split": _split_name,
                            "n_positive": _n_pos, "n_negative": _n_neg,
                            "n_source_v1": _n_v1, "n_source_v2": _n_v2,
                            "n_unique_groups": _n_groups,
                        })
                _comp_path = Path(args.output_dir) / "fold_composition.csv"
                _comp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(_comp_path, "w", newline="", encoding="utf-8") as _f:
                    _writer = csv.DictWriter(_f, fieldnames=list(_comp_rows[0].keys()))
                    _writer.writeheader()
                    _writer.writerows(_comp_rows)
                print(f"[FoldComposition] Wrote {_comp_path}")
            except Exception as e:
                print(f"WARNING: failed to write fold_composition.csv: {e}")

        # Leakage sanity check: print example groups (original vs augmented)
        n_show = int(args.cv_print_groups)
        if n_show > 0:
            from collections import defaultdict
            orig_by_g = defaultdict(list)
            aug_by_g = defaultdict(list)
            for i in range(orig_n):
                orig_by_g[group_orig[i]].append(samples_orig[i][0].name)
            for j in range(aug_n):
                aug_by_g[group_aug[j]].append(samples_aug[j][0].name)
            print("\n[LeakageSanity] Example groupings (original vs augmented):")
            for g in list(orig_by_g.keys())[:n_show]:
                olist = orig_by_g[g]
                alist = aug_by_g.get(g, [])
                print(f"  Group '{g}' -> originals={len(olist)} (e.g. {olist[:3]}) | augmented={len(alist)} (e.g. {alist[:3]})")
            print('[LeakageSanity] If augmented counts are not aligned to originals, adjust --group_regex.')

    else:
        seeds = parse_seeds(args.seeds)
        run_ids = list(seeds)
        run_label = 'seed'

    # Main training loop
    for run_id in run_ids:
        # Deterministic randomness per run (seed or fold)
        seed = int(run_id) if run_label == 'seed' else (12345 + int(run_id))
        print(f"\n{'='*60}")
        print(f"{run_label.upper()}: {run_id} (rng_seed={seed})")
        print(f"{'='*60}")
        set_seed(seed)
        # Build splits
        if run_label == 'fold':
            splits = fold_splits[int(run_id)]
        else:
            # Build splits with optional balancing (legacy seed mode)
            if args.split_strategy == "random_stratified":
                splits = build_splits_random_stratified(
                    labels, seed, tr, vr, te,
                    args.balance_strategy, args.balance_level
                )
            elif args.split_strategy == "group":
                splits = build_splits_group_balanced(
                    samples, group_ids, seed, tr, vr, te,
                    args.balance_strategy, args.balance_level
                )
                assert_no_group_overlap(group_ids, splits)
                print('[LeakageCheck] ✅ No GROUP overlap across splits (train/val/test).')
            elif args.split_strategy == "group_stratified":
                splits = build_splits_group_stratified(
                    samples, group_ids, seed, tr, vr, te,
                    args.balance_strategy, args.balance_level
                )
                assert_no_group_overlap(group_ids, splits)
            else:
                raise ValueError(f"Unknown split_strategy: {args.split_strategy}")

        run_samples = full_samples
        run_group_ids = full_group_ids

        # Leakage risk diagnostics (filename heuristic) — always print a concise status.
        print_leakage_risk_report(full_samples, full_group_ids, splits, max_examples=15)

        # Create output directory for this seed
        seed_dir = out_root / (f"seed_{seed}" if run_label == 'seed' else f"fold_{int(run_id)}")
        seed_dir.mkdir(parents=True, exist_ok=True)
        
        # Train each model
        models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
        if (not bool(getattr(args, 'use_quanv', False))) and any('_quanv' in m for m in models_to_run):
            print("[Config] Note: model name contains '_quanv' -> enabling quanv features automatically.")
            args.use_quanv = True
        seed_rows = []

        # Ensemble storage
        ens_val_scores: Dict[str, np.ndarray] = {}
        ens_test_scores: Dict[str, np.ndarray] = {}
        ens_val_y: Optional[np.ndarray] = None
        ens_test_y: Optional[np.ndarray] = None

        model_pred_store: list[Dict[str, Any]] = []
        val_pred_store: list[Dict[str, Any]] = []  # for --export_val_bundle / threshold-strategy comparison

        for mname in models_to_run:
            m_out = seed_dir / mname
            m_out.mkdir(parents=True, exist_ok=True)
            img_size = auto_img_size_for_model(mname, args.img_size)

            print(f"\n  Model: {mname} ({img_size}x{img_size})")

            # HPO runs once per model name (first time seen) and is reused for all subsequent seeds/folds.
            hargs = args
            if bool(getattr(args, 'moo_enable', False)):
                # Build loaders for MOO (uses base args).
                train_loader, val_loader, _ = build_loaders(
                    full_samples, splits, img_size, int(getattr(args, 'batch_size', 16)),
                    int(getattr(args, 'num_workers', 0)), bool(getattr(args, 'online_augmentation', False)),
                    (bool(getattr(args, 'use_weighted_sampler', False)) and str(getattr(args, 'balance_strategy', 'none')).lower() == 'none'),
                    randaugment=bool(getattr(args, 'randaugment', False)),
                    autoaugment=bool(getattr(args, 'autoaugment', False)),
                    randaugment_n=int(getattr(args, 'randaugment_n', 2)),
                    randaugment_m=int(getattr(args, 'randaugment_m', 9)),
                    aug_strength=str(getattr(args, 'aug_strength', 'light')),
                    use_smart_cache=bool(getattr(args, 'use_smart_cache', False)),
                    cache_dir=(str(getattr(args, 'cache_dir', '')).strip() or None),
                    cache_memory_gb=float(getattr(args, 'cache_memory_gb', 4.0)),
                    train_mix=str(getattr(args, 'train_mix', 'aug')),
                    train_orig_ratio=float(getattr(args, 'train_orig_ratio', 0.7)),
                    max_aug_per_group=int(getattr(args, 'max_aug_per_group', 0)),
                    orig_n=int(getattr(args, '_orig_n', 0)),
                    group_ids=getattr(args, '_full_group_ids', None),
                    rng_seed=int(seed),
                    use_quanv=bool(getattr(args, 'use_quanv', False)),
                    quanv_input_size=int(getattr(args, 'quanv_input_size', 32)),
                    quanv_patch=int(getattr(args, 'quanv_patch', 2)),
                    quanv_qubits=int(getattr(args, 'quanv_qubits', 4)),
                    quanv_depth=int(getattr(args, 'quanv_depth', 1)),
                    quanv_seed=int(seed),
                    quanv_cache_dir=(str(getattr(args, 'quanv_cache_dir', '')).strip() or None),
                )

                if mname not in moo_cache:
                    moo_cache[mname] = run_moo_optuna_for_model(
                        args=args,
                        device=device,
                        model_name=mname,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        out_dir=m_out,
                        seed=seed,
                        num_classes=num_classes,
                    )
                _sel = moo_cache[mname].get("selected_params", []) if isinstance(moo_cache.get(mname, None), dict) else []
                if _sel:
                    best_params = _sel[0]
                    print(f"    [MOO] Using selected Pareto config #0 for training: {best_params}")
                    hargs = override_namespace(args, best_params)
                else:
                    print("    [MOO] No Pareto configs selected; using base args.")
            elif bool(getattr(args, 'hpo', False)):
                if mname not in hpo_cache:
                    hpo_out = (out_root / 'hpo' / mname)
                    hpo_method = str(getattr(args, 'hpo_method', 'random')).lower()
                    if hpo_method == 'optuna':
                        print(f"    [HPO] Running Optuna(TPE) HPO for {mname} -> {hpo_out}")
                        hpo_cache[mname] = run_hpo_optuna_for_model(
                            base_args=args,
                            device=device,
                            model_name=mname,
                            full_samples=full_samples,
                            splits=splits,
                            img_size=img_size,
                            seed=seed,
                            out_dir=hpo_out,
                        )
                    else:
                        print(f"    [HPO] Running random-search HPO for {mname} -> {hpo_out}")
                        hpo_cache[mname] = run_hpo_for_model(
                        base_args=args,
                        device=device,
                        model_name=mname,
                        full_samples=full_samples,
                        splits=splits,
                        img_size=img_size,
                        seed=seed,
                        out_dir=hpo_out,
                    )
                    print(f"    [HPO] Best overrides for {mname}: {hpo_cache[mname]}")
                hargs = override_namespace(args, hpo_cache[mname])

            # Build data loaders with optional weighted sampler
            train_loader, val_loader, test_loader = build_loaders(
                full_samples, splits, img_size, args.batch_size,
                args.num_workers, args.online_augmentation,
                args.use_weighted_sampler and args.balance_strategy == 'none',
                randaugment=bool(getattr(args, "randaugment", False)),
                autoaugment=bool(getattr(args, "autoaugment", False)),
                randaugment_n=int(getattr(args, "randaugment_n", 2)),
                randaugment_m=int(getattr(args, "randaugment_m", 9)),
                aug_strength=str(getattr(args, "aug_strength", "light")),
                use_smart_cache=bool(getattr(args, "use_smart_cache", False)),
                cache_dir=(str(getattr(args, "cache_dir", "")).strip() or None),
                cache_memory_gb=float(getattr(args, "cache_memory_gb", 4.0)),
                train_mix=str(getattr(args, 'train_mix', 'aug')),
                train_orig_ratio=float(getattr(args, 'train_orig_ratio', 0.7)),
                max_aug_per_group=int(getattr(args, 'max_aug_per_group', 0)),
                orig_n=int(getattr(args, '_orig_n', 0)),
                group_ids=getattr(args, '_full_group_ids', None),
                rng_seed=int(seed),
                use_quanv=bool(getattr(args, "use_quanv", False)),
                quanv_input_size=int(getattr(args, "quanv_input_size", 32)),
                quanv_patch=int(getattr(args, "quanv_patch", 2)),
                quanv_qubits=int(getattr(args, "quanv_qubits", 4)),
                quanv_depth=int(getattr(args, "quanv_depth", 1)),
                quanv_seed=int(seed),
                quanv_cache_dir=getattr(args, "quanv_cache_dir", None),
            )

            # Create model
            model = create_model(mname, 2, bool(getattr(hargs, "pretrained", False)), drop_path_rate=float(getattr(hargs, "drop_path_rate", 0.1))).to(device)
            
            # Class weights for loss
            weight = None
            if args.use_class_weights:
                tr_ys = [full_samples[i][1] for i in splits.train]
                n1 = sum(tr_ys)
                n0 = len(tr_ys) - n1
                if n0 > 0 and n1 > 0:
                    w0 = (n0 + n1) / (2.0 * n0)
                    w1 = (n0 + n1) / (2.0 * n1)
                    weight = torch.tensor([w0, w1], device=device)

            # Loss
            if getattr(hargs, "loss", "ce") == "focal":
                criterion = FocalLoss(alpha=float(getattr(hargs, "focal_alpha", 0.25)), gamma=float(getattr(hargs, "focal_gamma", 2.0)))
            else:
                criterion = nn.CrossEntropyLoss(
                    weight=weight,
                    label_smoothing=float(getattr(hargs, "label_smoothing", 0.0)) if getattr(hargs, "label_smoothing", 0.0) else 0.0,
                )

            # Training phases
            frozen_phase = bool(args.pretrained and args.freeze_backbone_epochs and args.freeze_backbone_epochs > 0)
            if frozen_phase:
                freeze_backbone(model)
                optimizer = make_optimizer(head_parameters(model), lr=float(getattr(hargs, "head_lr", args.head_lr)), weight_decay=float(getattr(hargs, "weight_decay", args.weight_decay)), args=hargs)
            else:
                optimizer = make_optimizer(model.parameters(), lr=float(getattr(hargs, "lr", args.lr)), weight_decay=float(getattr(hargs, "weight_decay", args.weight_decay)), args=hargs)

            scheduler = make_scheduler(optimizer, hargs, total_epochs=args.epochs)

            # Optional: gradient monitoring + adaptive LR (merged features)
            grad_monitor = GradientMonitor(model) if bool(getattr(args, "monitor_gradients", False)) else None
            adaptive_scheduler = None
            if bool(getattr(args, "adaptive_lr", False)):
                adaptive_scheduler = GradientAwareScheduler(
                    optimizer,
                    initial_lr=float(getattr(hargs, "lr", getattr(args, "lr", 1e-3))),
                    mode=str(getattr(args, "lr_mode", "adaptive")),
                    warmup_epochs=int(round(float(getattr(hargs, "warmup_epochs", 0.0)))),
                    patience=2,
                    factor=0.5,
                    min_lr=1e-7,
                    max_lr=float(getattr(hargs, "lr", getattr(args, "lr", 1e-3))) * 5.0,
                    cooldown=1,
                )
                # Use GradientAwareScheduler instead of the classic scheduler
                scheduler = None


            # Training loop
            patience = args.early_stop_patience
            early_stop_enabled = (not args.force_full_epochs) and (patience > 0)
            best_val_loss = float("inf")
            no_improve = 0
            best_path = m_out / "best.pt"
            history = {"train_loss": [], "val_loss": []}

            # Metric-based early stopping / checkpointing
            monitor_metric = getattr(args, "early_stop_metric", "val_loss")
            monitor_mode = getattr(args, "early_stop_mode", "auto")
            if monitor_mode == "auto":
                monitor_mode = "min" if monitor_metric in ("val_loss", "brier") else "max"
            min_delta = float(getattr(args, "early_stop_min_delta", 0.0))
            best_monitor = float("inf") if monitor_mode == "min" else -float("inf")
            best_epoch = 0

            for epoch in range(1, args.epochs + 1):
                if grad_monitor is not None:
                    grad_monitor.reset_epoch()
                # Unfreeze backbone if needed
                if frozen_phase and epoch == int(args.freeze_backbone_epochs) + 1:
                    print("    Unfreezing backbone for full fine-tuning...")
                    unfreeze_all(model)
                    optimizer = make_optimizer(model.parameters(), lr=float(getattr(hargs, "lr", args.lr)), weight_decay=float(getattr(hargs, "weight_decay", args.weight_decay)), args=hargs)
                    scheduler = make_scheduler(optimizer, hargs, total_epochs=args.epochs)
                    frozen_phase = False
                
                # Train and validate
                tr_loss, _, _, _ = train_one_epoch(
                    model, train_loader, optimizer, criterion, device,
                    grad_clip_norm=float(getattr(hargs, "grad_clip_norm", args.grad_clip_norm)),
                    mixup_alpha=float(getattr(hargs, "mixup_alpha", args.mixup_alpha)),
                    cutmix_alpha=float(getattr(hargs, "cutmix_alpha", args.cutmix_alpha)),
                    label_smoothing=float(getattr(hargs, "label_smoothing", args.label_smoothing)),
                    class_weights=weight,
                    loss_name=str(getattr(hargs, "loss", "ce")),
                    grad_accum_steps=int(getattr(hargs, "grad_accum_steps", getattr(args, "grad_accum_steps", 1))),
                    amp=(False if getattr(args, "no_amp", False) else (bool(getattr(args, "amp", False)) or (device.type=="cuda"))),
                    scaler=None,
                    grad_monitor=grad_monitor,
                )
                
                va_loss, va_y, va_p, va_s = eval_epoch(model, val_loader, criterion, device)

                # LR scheduling
                if adaptive_scheduler is not None:
                    grad_norm = None
                    if grad_monitor is not None:
                        _gs = grad_monitor.get_epoch_statistics()
                        grad_norm = float(_gs.get("grad_norm_mean", 0.0))
                    adaptive_scheduler.step(float(va_loss), grad_norm=grad_norm, epoch=epoch)
                elif scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(va_loss)
                    else:
                        scheduler.step()

                history["train_loss"].append(tr_loss)
                history["val_loss"].append(va_loss)

                # Track best val loss (useful for LR schedulers), but checkpoint/early-stop is handled below.
                if va_loss < best_val_loss - 1e-12:
                    best_val_loss = va_loss


                # Validation metrics
                va_f1 = float(f1_score(va_y, va_p, zero_division=0)) if len(va_y) else float("nan")
                if grad_monitor is not None and int(getattr(args, "gradient_log_freq", 25)) > 0 and (epoch % int(getattr(args, "gradient_log_freq", 25)) == 0):
                    gstats = grad_monitor.get_epoch_statistics()
                    probs = grad_monitor.get_problematic_layers()

                    # Defensive handling: older/newer implementations may return dict or list
                    exploding = False
                    vanishing = False
                    if isinstance(probs, dict):
                        exploding = bool(probs.get("exploding")) or (len(probs.get("exploding_layers", [])) > 0)
                        vanishing = bool(probs.get("vanishing")) or (len(probs.get("vanishing_layers", [])) > 0)
                    elif isinstance(probs, list):
                        _s = " ".join([str(x).lower() for x in probs])
                        exploding = ("explod" in _s)
                        vanishing = ("vanish" in _s)

                    if exploding or vanishing:
                        if isinstance(probs, dict):
                            summary = {k: (len(v) if isinstance(v, (list, tuple)) else v) for k, v in probs.items()}
                        else:
                            summary = probs
                        print(f"    [GradMonitor] Epoch {epoch}: potential gradient issues -> {summary}")
                if len(np.unique(va_y)) == 2:
                    try:
                        va_auc = float(roc_auc_score(va_y, va_s))
                    except Exception:
                        va_auc = float("nan")
                else:
                    va_auc = float("nan")

                # Extra metrics for monitoring/selection
                if len(np.unique(va_y)) == 2:
                    try:
                        va_pr_auc = float(average_precision_score(va_y, va_s))
                    except Exception:
                        va_pr_auc = float("nan")
                else:
                    va_pr_auc = float("nan")
                try:
                    va_brier = float(brier_score_loss(va_y, va_s))
                except Exception:
                    va_brier = float("nan")

                # Best attainable at required precision (independent of threshold_strategy)
                _, va_p_at_p, va_r_at_p, va_f1_at_p, va_feasible_at_p = find_best_threshold_and_metrics(
                    va_y, va_s, strategy="f1_at_precision", min_recall=0.0, min_precision=args.precision_target
                )
                if not va_feasible_at_p:
                    va_p_at_p, va_r_at_p, va_f1_at_p = 0.0, 0.0, 0.0

                # Threshold selection (for validation reporting clarity)
                if args.threshold_strategy == "none":
                    thr = 0.5
                    va_p_sel = va_p
                    sel_tag = "0.5"
                else:
                    thr = find_best_threshold(va_y, va_s, args.threshold_strategy, min_recall=args.min_recall, min_precision=float(getattr(args, "precision_target", 0.0)))
                    va_p_sel = (va_s >= float(thr)).astype(np.int64)
                    sel_tag = "val_thr"

                va_metrics_sel, _ = compute_metrics(va_y, va_p_sel, va_s)

                # Single, unambiguous metric line (F1/Acc/Prec/Rec all computed at the SAME selected threshold)
                if args.threshold_strategy == "none":
                    thr_note = ""
                else:
                    thr_note = f" thr={thr:.3f}({sel_tag})"

                print(
                    f"    Ep {epoch}: loss={tr_loss:.4f}/{va_loss:.4f} "
                    f"acc={va_metrics_sel['acc']:.4f} "
                    f"prec={va_metrics_sel['precision']:.4f} "
                    f"rec={va_metrics_sel['recall']:.4f} "
                    f"f1={va_metrics_sel['f1']:.4f} "
                    f"auc={va_metrics_sel['auc']:.4f}"
                    f"{thr_note}"
                )

                # Optional tracking
                if wandb_run is not None:
                    try:
                        wandb_run.log({
                            'seed': int(seed),
                            'model': mname,
                            'epoch': int(epoch),
                            'train_loss': float(tr_loss),
                            'val_loss': float(va_loss),
                            'val_acc': float(va_metrics_sel['acc']),
                            'val_prec': float(va_metrics_sel['precision']),
                            'val_rec': float(va_metrics_sel['recall']),
                            'val_f1': float(va_metrics_sel['f1']),
                            'val_auc': float(va_metrics_sel['auc']),
                             'val_pr_auc': float(va_pr_auc) if np.isfinite(va_pr_auc) else None,
                             'val_brier': float(va_brier) if np.isfinite(va_brier) else None,
                             'val_f1_at_precision': float(va_f1_at_p),
                             'val_recall_at_precision': float(va_r_at_p),
                            'val_thr': float(thr),
                        })
                    except Exception:
                        pass

                if mlflow_mod is not None:
                    try:
                        mlflow_mod.log_metric(f'{mname}_val_loss_seed{seed}', float(va_loss), step=int(epoch))
                        mlflow_mod.log_metric(f'{mname}_val_f1_seed{seed}', float(va_metrics_sel['f1']), step=int(epoch))
                        mlflow_mod.log_metric(f'{mname}_val_auc_seed{seed}', float(va_metrics_sel['auc']), step=int(epoch))
                    except Exception:
                        pass


                # Metric-based checkpointing / early-stopping
                if monitor_metric == "val_loss":
                    monitor_val = float(va_loss)
                elif monitor_metric == "auc":
                    monitor_val = float(va_metrics_sel.get("auc", va_auc))
                elif monitor_metric == "pr_auc":
                    monitor_val = float(va_pr_auc)
                elif monitor_metric == "brier":
                    monitor_val = float(va_brier)
                elif monitor_metric == "f1_at_precision":
                    monitor_val = float(va_f1_at_p)
                elif monitor_metric == "recall_at_precision":
                    monitor_val = float(va_r_at_p)
                else:
                    monitor_val = float(va_loss)

                improved = (monitor_val < best_monitor - min_delta) if monitor_mode == "min" else (monitor_val > best_monitor + min_delta)
                if improved:
                    best_monitor = monitor_val
                    best_epoch = int(epoch)
                    no_improve = 0
                    torch.save(model.state_dict(), best_path)
                else:
                    no_improve += 1

                if early_stop_enabled and no_improve >= patience:
                    print(f"    Early stopping at epoch {epoch}.")
                    break

            # Load best model and evaluate on test set
            try:
                state = torch.load(best_path, map_location=device, weights_only=True)
            except:
                state = torch.load(best_path, map_location=device)
            model.load_state_dict(state)


            # Optional: short fine-tune on ORIGINAL-only training data (addresses augmented->original domain shift)
            if bool(getattr(args, 'finetune_on_original', False)) and int(getattr(args, '_orig_n', 0)) > 0 and (not bool(getattr(args, 'hpo', False))):
                ft_epochs = int(getattr(args, 'finetune_epochs', 5))
                ft_lr = float(getattr(args, 'finetune_lr', 1e-5))
                if ft_epochs > 0:
                    print(f"    [FineTune] ORIGINAL-only for {ft_epochs} epochs @ lr={ft_lr:g}")
                    ft_train_loader, _ft_val_loader, _ft_test_loader = build_loaders(
                        full_samples, splits, img_size, args.batch_size,
                        args.num_workers, False,  # no online augmentation in fine-tune by default
                        args.use_weighted_sampler and args.balance_strategy == 'none',
                        randaugment=False,
                        autoaugment=False,
                        randaugment_n=0,
                        randaugment_m=0,
                        aug_strength=str(getattr(args, "aug_strength", "light")),
                        use_smart_cache=bool(getattr(args, "use_smart_cache", False)),
                        cache_dir=(str(getattr(args, "cache_dir", "")).strip() or None),
                        cache_memory_gb=float(getattr(args, "cache_memory_gb", 4.0)),
                        train_mix="original",
                        train_orig_ratio=1.0,
                        max_aug_per_group=0,
                        orig_n=int(getattr(args, '_orig_n', 0)),
                        group_ids=getattr(args, '_full_group_ids', None),
                        # keep Quanv wrapping during fine-tune if enabled
                        use_quanv=bool(getattr(args, "use_quanv", False)),
                        quanv_input_size=int(getattr(args, "quanv_input_size", 32)),
                        quanv_patch=int(getattr(args, "quanv_patch", 2)),
                        quanv_qubits=int(getattr(args, "quanv_qubits", 4)),
                        quanv_depth=int(getattr(args, "quanv_depth", 1)),
                        quanv_seed=int(getattr(args, "quanv_seed", 42)),
                        quanv_cache_dir=(str(getattr(args, "quanv_cache_dir", "")).strip() or None),
                        rng_seed=int(seed),
                    )

                    # Override a few training hyperparams for stability
                    ft_overrides = {
                        'lr': ft_lr,
                        'head_lr': ft_lr,
                        'scheduler': 'none',
                        'warmup_epochs': 0.0,
                        'mixup_alpha': 0.0,
                        'cutmix_alpha': 0.0,
                    }
                    ft_args = override_namespace(hargs, ft_overrides)

                    optimizer_ft = make_optimizer(model.parameters(), name=str(getattr(ft_args, "optimizer", getattr(hargs, "optimizer", "adamw"))),
                                                  lr=float(getattr(ft_args, "lr", ft_lr)),
                                                  weight_decay=float(getattr(ft_args, "weight_decay", getattr(hargs, "weight_decay", args.weight_decay))),
                                                  args=ft_args)

                    best_ft_loss = float("inf")
                    best_ft_path = m_out / "best_finetune.pt"

                    for ft_ep in range(1, ft_epochs + 1):
                        tr_loss_ft, _, _, _ = train_one_epoch(
                            model, ft_train_loader, optimizer_ft, criterion, device,
                            grad_clip_norm=float(getattr(ft_args, "grad_clip_norm", args.grad_clip_norm)),
                            mixup_alpha=0.0,
                            cutmix_alpha=0.0,
                            label_smoothing=float(getattr(ft_args, "label_smoothing", getattr(hargs, "label_smoothing", 0.0))),
                            class_weights=weight,
                            loss_name=str(getattr(ft_args, "loss", "ce")),
                            grad_accum_steps=int(getattr(ft_args, "grad_accum_steps", getattr(args, "grad_accum_steps", 1))),
                            amp=bool(getattr(args, "amp", False)),
                        )
                        va_loss_ft, va_y_ft, _, va_s_ft = eval_epoch(model, val_loader, criterion, device, tta=0, threshold=None)
                        print(f"    [FineTune] Ep {ft_ep}: loss={tr_loss_ft:.4f}/{va_loss_ft:.4f}")

                        if va_loss_ft < best_ft_loss - 1e-12:
                            best_ft_loss = float(va_loss_ft)
                            try:
                                torch.save(model.state_dict(), best_ft_path)
                            except Exception:
                                pass

                    # Load best finetuned checkpoint if saved
                    if best_ft_path.exists():
                        try:
                            st = torch.load(best_ft_path, map_location=device, weights_only=True)
                        except Exception:
                            st = torch.load(best_ft_path, map_location=device)
                        model.load_state_dict(st)

            # Final validation with TTA
            # Final validation with TTA
            if args.calibrate == "temperature":
                va_loss2, va_y2, va_s2_uncal, va_ld2 = eval_epoch_with_logits(model, val_loader, criterion, device, tta=args.tta)
                T = fit_temperature_scaling(va_ld2, va_y2)
                va_s2 = _sigmoid_np(va_ld2 / float(T))
                print(f"    Temperature scaling enabled: T={T:.4f} (fit on validation)")
            else:
                va_loss2, va_y2, _, va_s2 = eval_epoch(model, val_loader, criterion, device, tta=args.tta, threshold=None)
                T = 1.0

            thr = find_best_threshold(va_y2, va_s2, args.threshold_strategy, min_recall=args.min_recall, min_precision=float(getattr(args, "precision_target", 0.0)))
            thr_for_test = None if args.threshold_strategy == "none" else thr

            # Store post-calibration validation predictions for --export_val_bundle
            try:
                val_pred_store.append({
                    "model": str(mname),
                    "y_true": np.asarray(va_y2),
                    "y_pred": (np.asarray(va_s2) >= 0.5).astype(int),
                    "y_score": np.asarray(va_s2),
                })
            except Exception as e:
                print(f"WARNING: failed to store validation predictions for {mname}: {e}")


            if thr_for_test is not None:
                try:
                    thr2, p2, r2, f12, feas2 = find_best_threshold_and_metrics(
                        va_y2,
                        va_s2,
                        args.threshold_strategy,
                        min_recall=args.min_recall,
                        min_precision=float(getattr(args, "precision_target", 0.0)),
                    )
                    thr_for_test = thr2 if args.threshold_strategy != "none" else thr_for_test
                    if args.threshold_strategy in ("f1_at_precision", "recall_at_precision") and not feas2:
                        print(
                            f"    [WARN] No validation threshold achieved precision≥{float(getattr(args, 'precision_target', 0.0)):.2f}; using max-precision threshold instead."
                        )
                    print(
                        f"    Using validation-tuned threshold {thr_for_test:.3f} (val_prec={p2:.3f}, val_rec={r2:.3f}, val_f1={f12:.3f}) for test eval"
                    )
                except Exception:
                    print(f"    Using validation-tuned threshold {thr_for_test:.3f} for test eval")

            # Test evaluation
            # We compute BOTH:
            #   - metrics at the default 0.5 threshold (te_*_05)
            #   - metrics at the validation-tuned threshold (te_*_tuned) if enabled
            # Then we pick a *selected* set based on --threshold_strategy.
            if args.calibrate == "temperature":
                te_loss, te_y, te_s_uncal, te_ld = eval_epoch_with_logits(model, test_loader, criterion, device, tta=args.tta)
                te_s = _sigmoid_np(te_ld / float(T))
                te_p_05 = (te_s >= 0.5).astype(np.int64)
            else:
                te_loss, te_y, te_p_05, te_s = eval_epoch(
                    model, test_loader, criterion, device, tta=args.tta, threshold=None
                )

            te_metrics_05, _ = compute_metrics(te_y, te_p_05, te_s)

            if args.threshold_strategy == "none":
                te_p_tuned = te_p_05
                te_metrics_tuned = te_metrics_05
                selected_tag = "0.5"
                selected_thr = 0.5
            else:
                te_p_tuned = (te_s >= float(thr)).astype(np.int64)
                te_metrics_tuned, _ = compute_metrics(te_y, te_p_tuned, te_s)
                selected_tag = "val_thr"
                selected_thr = float(thr)

            # Selected metrics/preds used for plots + "final" reporting
            te_p = te_p_tuned if args.threshold_strategy != "none" else te_p_05
            te_metrics = te_metrics_tuned if args.threshold_strategy != "none" else te_metrics_05

            print(
                f"    FINAL ({mname}, {run_label}={run_id}) "
                f"[thr={selected_thr:.3f} ({selected_tag})] "
                f"acc={te_metrics['acc']:.4f} "
                f"prec={te_metrics['precision']:.4f} "
                f"rec={te_metrics['recall']:.4f} "
                f"f1={te_metrics['f1']:.4f} "
                f"auc={te_metrics['auc']:.4f}"
            )

            # Save a detailed classification report (selected threshold) for reproducibility
            try:
                rep_path = m_out / "classification_report_test.txt"
                save_classification_report_txt(
                    te_y, te_p, rep_path,
                    neg_name=str(getattr(args, "neg_class", "neg")),
                    pos_name=str(getattr(args, "pos_class", "pos")),
                    title=f"TEST Classification Report (thr={selected_thr:.3f}, {selected_tag})",
                )
            except Exception as _e:
                print(f"    [Warn] Could not write classification report: {_e}")
            # Error analysis export: save misclassified filenames for this run/model
            try:
                export_failures_csv(m_out / 'failures.csv', run_samples, te_y, te_s, selected_thr, positive_label=1, indices=splits.test)
            except Exception as e:
                print(f"WARNING: failed to export failures.csv: {e}")

            
            # Store for optional statistical testing / model selection
            try:
                _test_files = [str(full_samples[i][0]) for i in splits.test]
            except Exception:
                _test_files = []
            model_pred_store.append({
                "model": str(mname),
                "y_true": np.asarray(te_y),
                "y_pred": np.asarray(te_p),
                "y_score": np.asarray(te_s),
                "files": _test_files,
            })

            # Optional: Bayesian uncertainty (MC-dropout / ensembles)
            if bool(getattr(args, "bayesian_ensemble", False)):
                try:
                    _method = str(getattr(args, "ensemble_method", "mc_dropout"))
                    _ns = int(getattr(args, "uncertainty_samples", 30))
                    be = BayesianEnsemble(method=_method, num_samples=_ns, temperature_scaling=False)
                    be.add_model(model, enable_dropout=(_method == "mc_dropout"))
                    be_pred, be_unc = be.predict_with_uncertainty(test_loader, device=device)
                    be_pred = np.asarray(be_pred)
                    # Positive-class probability
                    if be_pred.ndim == 2 and be_pred.shape[1] >= 2:
                        prob_pos = be_pred[:, 1]
                    else:
                        prob_pos = be_pred.reshape(-1)
                    # Uncertainty scalar (prefer 'total' if dict)
                    if isinstance(be_unc, dict):
                        unc_total = np.asarray(be_unc.get("total", be_unc.get("epistemic", list(be_unc.values())[0])))
                    else:
                        unc_total = np.asarray(be_unc)
                    out_unc = m_out / "bayesian_uncertainty.csv"
                    with open(out_unc, "w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow(["filename", "y_true", "prob_pos_mean", "uncertainty"])
                        for i in range(len(prob_pos)):
                            fn = Path(_test_files[i]).name if i < len(_test_files) else str(i)
                            yt = int(te_y[i]) if i < len(te_y) else ""
                            u = float(unc_total[i]) if i < len(unc_total) else ""
                            w.writerow([fn, yt, float(prob_pos[i]), u])
                except Exception as e:
                    print(f"WARNING: Bayesian ensemble failed: {e}")

            # Optional: model compression (pruning/quantization)
            if bool(getattr(args, "compress_model", False)):
                try:
                    method = str(getattr(args, "compression_method", "pruning"))
                    amount = float(getattr(args, "pruning_amount", 0.2))
                    compressor = ModelCompressor(method=method, pruning_amount=amount)
                    cmodel = compressor.compress(model)
                    torch.save(cmodel.state_dict(), m_out / "compressed.pt")
                    # Try ONNX export (best-effort)
                    try:
                        compressor.export_to_onnx(cmodel, m_out / "compressed.onnx", input_shape=(1, 3, img_size, img_size))
                    except Exception:
                        pass
                except Exception as e:
                    print(f"WARNING: Compression failed: {e}")

# Store for ensemble if requested
            if args.weighted_ensemble:
                if ens_val_y is None:
                    ens_val_y = va_y2.copy()
                else:
                    if not np.array_equal(ens_val_y, va_y2):
                        print("WARNING: ensemble val label order mismatch across models; disabling ensemble for this seed.")
                        args.weighted_ensemble = False
                
                if ens_test_y is None:
                    ens_test_y = te_y.copy()
                else:
                    if not np.array_equal(ens_test_y, te_y):
                        print("WARNING: ensemble test label order mismatch across models; disabling ensemble for this seed.")
                        args.weighted_ensemble = False
                
                if args.weighted_ensemble:
                    ens_val_scores[mname] = va_s2.copy()
                    ens_test_scores[mname] = te_s.copy()
            
            # Plotting
            cm = confusion_matrix(te_y, te_p, labels=[0, 1])
            plot_loss_curves(history, m_out / "loss", f"Loss ({mname})", args.plot_dpi)
            plot_confusion_matrix(cm, m_out / "confusion", f"CM ({mname})", args.plot_dpi)
            plot_roc_pr(te_y, te_s, m_out, mname, args.plot_dpi)

            # Store results
            row = {
                "seed": seed, 
                "model": mname, 
                "img_size": img_size, 
                "test_loss": te_loss, 
                "threshold": thr, 
                "tta": args.tta, 
                "threshold_strategy": args.threshold_strategy,
                "balance_strategy": args.balance_strategy,
                "balance_level": args.balance_level
            }
            # Store both metric sets + the selected set
            row.update({f"test05_{k}": v for k, v in te_metrics_05.items()})
            row.update({f"testtuned_{k}": v for k, v in te_metrics_tuned.items()})
            row.update({f"testsel_{k}": v for k, v in te_metrics.items()})
            row["thr_val"] = float(thr)
            row["thr_selected"] = float(selected_thr)
            row["thr_selected_tag"] = str(selected_tag)

            seed_rows.append(row)

            # Update metrics summary (we summarize the SELECTED set for easy reading)
            mb = metrics_by_model.setdefault(mname, {})
            for k, v in te_metrics.items():
                mb.setdefault(f"testsel_{k}", []).append(v)
        # Ensemble (if requested)
        if args.weighted_ensemble and len(ens_val_scores) >= 2 and ens_val_y is not None and ens_test_y is not None:
            try:
                # Choose weights on validation
                auc_target = args.moo_constraint_auc_target if getattr(args, "moo_constraint_auc_target", 0.0) else None
                f1_target = args.moo_constraint_f1_target if getattr(args, "moo_constraint_f1_target", 0.0) else None
                weights = optimize_ensemble_weights(
                    ens_val_scores,
                    ens_val_y,
                    metric=str(getattr(args, "ensemble_weight_metric", "f1_at_precision")),
                    precision_target=float(getattr(args, "precision_target", 0.90)),
                    trials=int(getattr(args, "ensemble_weight_trials", 2048)),
                    strategy=str(getattr(args, "ensemble_weight_opt", "random_search")),
                    seed=int(getattr(args, "ensemble_weight_seed", 123)),
                    auc_target=auc_target,
                    f1_target=f1_target,
                )


                # Weighted ensemble predictions (base)
                ens_val_base = np.zeros_like(next(iter(ens_val_scores.values())), dtype=float)
                ens_test_base = np.zeros_like(next(iter(ens_test_scores.values())), dtype=float)
                for mn in weights:
                    ens_val_base += weights[mn] * ens_val_scores[mn]
                    ens_test_base += weights[mn] * ens_test_scores[mn]

                # Precompute per-model AUCs (validation) for reporting weights
                aucs = {mn: float(roc_auc_score(ens_val_y, ens_val_scores[mn])) for mn in weights}

                def _eval_and_save_variant(tag: str, val_scores: np.ndarray, test_scores: np.ndarray, out_subdir: str,
                                           extra_row: Optional[Dict[str, Any]] = None) -> None:
                    # Threshold chosen on validation *scores*
                    ens_thr = find_best_threshold(
                        ens_val_y,
                        val_scores,
                        args.threshold_strategy,
                        min_recall=args.min_recall,
                        min_precision=float(getattr(args, "precision_target", 0.0)),
                    )

                    # Metrics at default 0.5 threshold
                    ens_pred_05 = (test_scores >= 0.5).astype(np.int64)
                    ens_metrics_05, _ = compute_metrics(ens_test_y, ens_pred_05, test_scores)

                    # Metrics at tuned threshold (unless disabled)
                    if args.threshold_strategy == "none":
                        ens_pred_tuned = ens_pred_05
                        ens_metrics_tuned = ens_metrics_05
                        ens_selected_thr = 0.5
                        ens_selected_tag = "0.5"
                    else:
                        ens_pred_tuned = (test_scores >= float(ens_thr)).astype(np.int64)
                        ens_metrics_tuned, _ = compute_metrics(ens_test_y, ens_pred_tuned, test_scores)
                        ens_selected_thr = float(ens_thr)
                        ens_selected_tag = "val_thr"

                    ens_pred = ens_pred_tuned if args.threshold_strategy != "none" else ens_pred_05
                    ens_metrics = ens_metrics_tuned if args.threshold_strategy != "none" else ens_metrics_05

                    # Save plots
                    ens_out = seed_dir / out_subdir
                    ens_out.mkdir(parents=True, exist_ok=True)
                    cm_ens = confusion_matrix(ens_test_y, ens_pred, labels=[0, 1])
                    plot_confusion_matrix(cm_ens, ens_out / "confusion", f"CM ({tag})", args.plot_dpi)
                    plot_roc_pr(ens_test_y, test_scores, ens_out, tag, args.plot_dpi)

                    # Save row
                    row = {
                        "seed": seed,
                        "model": tag,
                        "img_size": "mixed",
                        "test_loss": float("nan"),
                        "threshold": float(ens_thr),
                        "tta": args.tta,
                        "threshold_strategy": args.threshold_strategy,
                        "balance_strategy": args.balance_strategy,
                        "balance_level": args.balance_level,
                    }

                    # Add weights + per-model val AUC
                    for mn2 in sorted(weights):
                        row[f"w_{mn2}"] = float(weights[mn2])
                        row[f"val_auc_{mn2}"] = float(aucs.get(mn2, float("nan")))

                    if extra_row:
                        row.update(extra_row)

                    row.update({f"test05_{k}": v for k, v in ens_metrics_05.items()})
                    row.update({f"testtuned_{k}": v for k, v in ens_metrics_tuned.items()})
                    row.update({f"testsel_{k}": v for k, v in ens_metrics.items()})
                    row["thr_val"] = float(ens_thr)
                    row["thr_selected"] = float(ens_selected_thr)
                    row["thr_selected_tag"] = str(ens_selected_tag)

                    seed_rows.append(row)

                    # Update metrics summary (selected)
                    mb = metrics_by_model.setdefault(tag, {})
                    for k, v in ens_metrics.items():
                        mb.setdefault(f"testsel_{k}", []).append(v)

                    # Store for optional statistical testing
                    try:
                        _test_files = [str(full_samples[i][0]) for i in splits.test]
                    except Exception:
                        _test_files = []
                    try:
                        model_pred_store.append({
                            "model": str(tag),
                            "y_true": np.asarray(ens_test_y),
                            "y_pred": np.asarray(ens_pred),
                            "y_score": np.asarray(test_scores),
                            "files": _test_files,
                        })
                    except Exception:
                        pass

                # Baseline ensemble (no retrieval)
                _eval_and_save_variant("weighted_ensemble", ens_val_base, ens_test_base, "ensemble_weighted")

                # Optional: Retrieval-augmented inference (Option A)
                if bool(getattr(args, "rag_enable", False)):
                    try:
                        ens_val_rag, ens_test_rag, p_knn_val, p_knn_test, rag_meta = rag_blend_ensemble_probs(
                            base_val=ens_val_base,
                            base_test=ens_test_base,
                            args=args,
                            device=device,
                            samples=full_samples,
                            splits=splits,
                            seed_dir=seed_dir,
                            models_to_run=models_to_run,
                        )

                        extra = {
                            "rag_enable": 1,
                            "rag_model": rag_meta.get("rag_model", ""),
                            "rag_index_source": rag_meta.get("index_source", ""),
                            "rag_k": rag_meta.get("k", 0),
                            "rag_alpha": rag_meta.get("alpha", 0.0),
                            "rag_metric": rag_meta.get("metric", ""),
                            "rag_weighted": int(bool(rag_meta.get("weighted", False))),
                            "rag_temp": rag_meta.get("temp", 0.0),
                            "rag_tta": rag_meta.get("rag_tta", 0),
                            "rag_chunk": rag_meta.get("rag_chunk", 0),
                            "rag_train_index_size": rag_meta.get("train_index_size", 0),
                        }

                        _eval_and_save_variant("weighted_ensemble_rag", ens_val_rag, ens_test_rag, "ensemble_weighted_rag", extra_row=extra)

                        # Save kNN probabilities + metadata for inspection
                        try:
                            outp = seed_dir / "ensemble_weighted_rag"
                            np.savez_compressed(
                                outp / "rag_knn.npz",
                                p_knn_val=p_knn_val,
                                p_knn_test=p_knn_test,
                                base_val=ens_val_base,
                                base_test=ens_test_base,
                                blended_val=ens_val_rag,
                                blended_test=ens_test_rag,
                            )
                            with open(outp / "rag_meta.json", "w", encoding="utf-8") as f:
                                json.dump(rag_meta, f, indent=2)
                        except Exception:
                            pass

                        print(
                            "  Ensemble(RAG): "
                            + f"backbone={rag_meta.get('rag_model')} k={rag_meta.get('k')} alpha={rag_meta.get('alpha'):.3f} "
                            + f"src={rag_meta.get('index_source')} metric={rag_meta.get('metric')} weighted={bool(rag_meta.get('weighted'))}"
                        )
                    except Exception as e:
                        print(f"WARNING: RAG ensemble blend failed: {e}")

                print("  Ensemble: weighted_ensemble " + " ".join([f"{mn}={weights[mn]:.2f}" for mn in sorted(weights)]))

            except Exception as e:
                print(f"WARNING: Ensemble failed: {e}")


        # Stacking (meta-learning) ensemble
        if args.stacking_ensemble and len(ens_val_scores) >= 2:
            try:
                val_y = ens_val_y
                test_y = ens_test_y

                base_models = list(ens_val_scores.keys())

                # Optional: keep only top-K base models by a validation ranking metric
                if int(args.stacking_top_k) > 0 and len(base_models) > int(args.stacking_top_k):
                    rank_metric = str(getattr(args, "stacking_rank_metric", "f1_at_precision"))

                    def _val_rank_value(mn: str) -> float:
                        # Higher is better.
                        for r in seed_rows:
                            if r.get("model") == mn:
                                if rank_metric == "auc":
                                    return float(r.get("val_auc", -1e9))
                                if rank_metric == "pr_auc":
                                    return float(r.get("val_pr_auc", -1e9))
                                if rank_metric == "f1_at_precision":
                                    return float(r.get("val_f1_at_precision", -1e9))
                                if rank_metric == "recall_at_precision":
                                    return float(r.get("val_recall_at_precision", -1e9))
                                if rank_metric == "neg_logloss":
                                    return -float(r.get("val_logloss", 1e9))
                                if rank_metric == "neg_brier":
                                    return -float(r.get("val_brier", 1e9))
                        # Fallbacks
                        if rank_metric == "auc":
                            return float(roc_auc_score(val_y, ens_val_scores[mn]))
                        if rank_metric == "pr_auc":
                            return float(average_precision_score(val_y, ens_val_scores[mn]))
                        if rank_metric == "neg_logloss":
                            return -float(log_loss(val_y, np.clip(ens_val_scores[mn], 1e-6, 1-1e-6)))
                        if rank_metric == "neg_brier":
                            return -float(brier_score_loss(val_y, ens_val_scores[mn]))
                        return float(roc_auc_score(val_y, ens_val_scores[mn]))

                    base_models = sorted(base_models, key=_val_rank_value, reverse=True)[: int(args.stacking_top_k)]

                if len(base_models) < 2:
                    raise RuntimeError("Stacking needs at least 2 base models after selection.")

                X_val = np.column_stack([ens_val_scores[mn] for mn in base_models]).astype(np.float32)
                X_test = np.column_stack([ens_test_scores[mn] for mn in base_models]).astype(np.float32)

                feat_mode = str(getattr(args, "stacking_feature_mode", "logit"))
                X_val_f = _stacking_transform(X_val, mode=feat_mode)
                X_test_f = _stacking_transform(X_test, mode=feat_mode)

                meta_model = str(getattr(args, "stacking_meta_model", "logreg"))
                C = float(getattr(args, "stacking_C", 1.0))
                cv_folds = int(getattr(args, "stacking_cv_folds", 0))
                standardize = not bool(getattr(args, "stacking_no_standardize", False))

                meta, meta_info = fit_stacking_meta_learner(
                    X_val_f,
                    val_y,
                    meta_model=meta_model,
                    C=C,
                    seed=int(seed),
                    cv_folds=cv_folds,
                    standardize=standardize,
                )

                val_s = meta.predict_proba(X_val_f)[:, 1]
                test_s = meta.predict_proba(X_test_f)[:, 1]

                # Metrics at 0.5
                st_p05 = (test_s >= 0.5).astype(np.int64)
                st_metrics05, _ = compute_metrics(test_y, st_p05, test_s)

                # Tune threshold on val using the same strategy as base models
                st_thr = 0.5
                if str(args.threshold_strategy).lower() != "none":
                    st_thr, _, _, _, _ = find_best_threshold(
                        val_y,
                        val_s,
                        strategy=str(args.threshold_strategy),
                        min_recall=float(getattr(args, "min_recall", 0.0)),
                        min_precision=float(getattr(args, "precision_target", 0.0)),
                    )

                st_pt = (test_s >= st_thr).astype(np.int64)
                st_metricst, _ = compute_metrics(test_y, st_pt, test_s)

                # Selected = tuned
                st_psel = st_pt
                st_metricssel = st_metricst

                # Report val f1/recall at precision target
                _, va_f1_at_p, va_p_at_p, va_r_at_p, _ = find_best_threshold(
                    val_y,
                    val_s,
                    strategy="f1_at_precision",
                    min_recall=0.0,
                    min_precision=float(getattr(args, "precision_target", 0.0)),
                )

                print(f"    [STACKING] models={len(base_models)} thr={st_thr:.4f} | testsel_f1={st_metricssel.get('f1', float('nan')):.4f} auc={st_metricssel.get('auc', float('nan')):.4f}")

                row = {
                    "model": "stacking",
                    "seed": seed,
                    "base_models": ",".join(base_models),
                    "meta": json.dumps(meta_info, ensure_ascii=False),
                    "thr_val": float(st_thr),
                    "thr_selected_tag": str(args.threshold_strategy),
                    "thr_selected": float(st_thr),
                    "val_f1_at_precision": float(va_f1_at_p),
                    "val_recall_at_precision": float(va_r_at_p),
                    "val_precision_at_precision": float(va_p_at_p),
                }

                # Transparency: per-base-model val AUC
                for mn in base_models:
                    try:
                        row[f"val_auc_{mn}"] = float(roc_auc_score(val_y, ens_val_scores[mn]))
                    except Exception:
                        row[f"val_auc_{mn}"] = ""

                # Meta coefficients (if available)
                if isinstance(meta_info, dict) and isinstance(meta_info.get("coef"), list):
                    for i, mn in enumerate(base_models):
                        if i < len(meta_info["coef"]):
                            row[f"meta_coef_{mn}"] = float(meta_info["coef"][i])

                for k, v in st_metrics05.items():
                    row[f"test05_{k}"] = float(v) if isinstance(v, (int, float)) else v
                for k, v in st_metricst.items():
                    row[f"testtuned_{k}"] = float(v) if isinstance(v, (int, float)) else v
                for k, v in st_metricssel.items():
                    row[f"testsel_{k}"] = float(v) if isinstance(v, (int, float)) else v

                seed_rows.append(row)

                mb = metrics_by_model.setdefault("stacking", {})
                for k, v in st_metricssel.items():
                    if isinstance(v, (int, float)):
                        mb.setdefault(f"testsel_{k}", []).append(float(v))

                model_pred_store.append({"model": "stacking", "y_true": test_y, "y_pred": st_psel})

                if bool(getattr(args, "save_models", False)):
                    with open(seed_dir / "stacking_meta.json", "w", encoding="utf-8") as f:
                        json.dump({
                            "base_models": base_models,
                            "meta_info": meta_info,
                            "threshold": float(st_thr),
                        }, f, indent=2)

            except Exception as e:
                print(f"WARNING: Stacking failed: {e}")

        
        # Optional: statistical testing between trained models (paired on same test set)
        if bool(getattr(args, "statistical_testing", False)) and len(model_pred_store) >= 2:
            try:
                # Ensure same y_true across models
                y0 = model_pred_store[0]["y_true"]
                ok = all(np.array_equal(y0, m["y_true"]) for m in model_pred_store[1:])
                if not ok:
                    print("WARNING: Statistical testing skipped (y_true differs across models).")
                else:
                    preds_dict = {m["model"]: m["y_pred"] for m in model_pred_store}
                    selector = StatisticalModelSelector(
                        alpha=float(getattr(args, "test_alpha", 0.05)),
                        correction=str(getattr(args, "test_correction", "holm"))
                    )
                    stat_res = selector.compare_models(y0, preds_dict, metric="f1")
                    out_path = seed_dir / "statistical_tests.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(stat_res, f, indent=2)
                    print(f"    [Stats] Saved statistical test results -> {out_path}")
            except Exception as e:
                print(f"WARNING: Statistical testing failed (need scipy installed for some tests): {e}")

        # Additional exports, independent of StatisticalModelSelector above:
        # prediction bundle (for scripts/extract_prediction_bundles.py), the
        # four-way paired significance report, and the validation bundle
        # (for threshold-strategy comparison). All opt-in via existing/new flags.
        if bool(getattr(args, "statistical_testing", False)) and len(model_pred_store) >= 2:
            try:
                bundle_path = export_prediction_bundle(seed_dir, model_pred_store)
                print(f"    [Bundle] Saved prediction bundle -> {bundle_path}")
            except Exception as e:
                print(f"WARNING: failed to export prediction_bundle.npz: {e}")
            try:
                four_way = build_four_way_significance_report(
                    model_pred_store,
                    alpha=float(getattr(args, "test_alpha", 0.05)),
                    correction=str(getattr(args, "test_correction", "bonferroni")),
                )
                four_way_path = seed_dir / "significance_report_fourway.json"
                with open(four_way_path, "w", encoding="utf-8") as f:
                    json.dump(four_way, f, indent=2)
                print(f"    [Stats] Saved four-way significance report -> {four_way_path}")
            except Exception as e:
                print(f"WARNING: four-way significance report failed: {e}")

        if bool(getattr(args, "export_val_bundle", True)) and len(val_pred_store) >= 1:
            try:
                val_bundle_path = export_prediction_bundle(seed_dir, val_pred_store, filename="validation_bundle.npz")
                print(f"    [Bundle] Saved validation bundle -> {val_bundle_path}")
            except Exception as e:
                print(f"WARNING: failed to export validation_bundle.npz: {e}")

# Save per-seed results
        if seed_rows:
            csv_path = seed_dir / "results.csv"
            keys = sorted({k for r in seed_rows for k in r.keys()})
            norm_rows = [{k: r.get(k, "") for k in keys} for r in seed_rows]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(norm_rows)

    # Save summary statistics
    summary_rows = []
    for mname, md in metrics_by_model.items():
        row = {"model": mname, f"n_{run_label}s": len(run_ids)}
        for k, v in md.items():
            m, s = mean_std(v)
            row[f"{k}_mean"] = m
            row[f"{k}_std"] = s
        summary_rows.append(row)

    if summary_rows:
        summary_path = out_root / "summary_mean_std.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        
        # Also save as JSON for easier reading
        summary_json = out_root / "summary_statistics.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump({row["model"]: {k: v for k, v in row.items() if k != "model"} for row in summary_rows}, f, indent=2)
        
        print(f"\nSaved summary to: {summary_path}")
        print(f"Saved statistics to: {summary_json}")

        # Console summary (SELECTED metrics)
        # - If --threshold_strategy != none, selected metrics use the validation-tuned threshold.
        # - Else selected metrics use the default 0.5 threshold.
        print("\n" + "=" * 60)
        print("SELECTED TEST METRICS (mean±std across runs)")
        print("=" * 60)
        hdr = f"{'Model':22s}  {'Acc':>12s}  {'Prec':>12s}  {'Rec':>12s}  {'F1':>12s}  {'AUC':>12s}"
        print(hdr)
        print("-" * len(hdr))

        best_model = None
        best_f1 = -1.0

        for row in sorted(summary_rows, key=lambda r: str(r.get("model", ""))):
            def fmt(mk: str) -> str:
                mu = row.get(f"{mk}_mean", float("nan"))
                sd = row.get(f"{mk}_std", float("nan"))
                if mu != mu:  # nan
                    return "   nan±nan "
                return f"{mu:7.4f}±{sd:4.4f}"

            acc_s = fmt("testsel_acc")
            pr_s  = fmt("testsel_precision")
            rc_s  = fmt("testsel_recall")
            f1_s  = fmt("testsel_f1")
            auc_s = fmt("testsel_auc")

            model_name = str(row.get("model", ""))[:22]
            print(f"{model_name:22s}  {acc_s:>12s}  {pr_s:>12s}  {rc_s:>12s}  {f1_s:>12s}  {auc_s:>12s}")

            f1_mu = row.get("testsel_f1_mean", float("nan"))
            if f1_mu == f1_mu and f1_mu > best_f1:
                best_f1 = float(f1_mu)
                best_model = str(row.get("model", ""))

        if best_model is not None:
            print("-" * len(hdr))
            print(f"Best model by mean selected F1: {best_model} (F1={best_f1:.4f})")
        print("=" * 60 + "\n")
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {out_root}")
    print(f"Configuration saved to: {out_root / 'run_config.json'}")
    print(f"Dataset balancing: {args.balance_strategy} ({args.balance_level})")


    # Close experiment trackers (if any)
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass
    if mlflow_mod is not None:
        try:
            mlflow_mod.end_run()
        except Exception:
            pass


if __name__ == "__main__":
    main()