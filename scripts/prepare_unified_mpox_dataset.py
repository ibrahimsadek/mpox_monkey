"""Utility script to build a unified, leakage-aware mpox dataset layout.

This script merges a base dataset with MSLD v1.0 and MSLD v2.0, applies
augmentation controls, optionally de-duplicates exact files, materializes a
curated folder structure, and writes manifest files for downstream training.
"""

import os
import re
import shutil
import hashlib
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def sha1_file(p: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def infer_group_id(path: Path, group_regex: str) -> str:
    """
    Extract a group id from filename (stem). Default regex groups everything before last _<digits>.
    If it doesn't match, fallback to full stem.
    """
    stem = path.stem
    m = re.search(group_regex, stem)
    if m and m.group(1):
        return m.group(1)
    return stem

def list_images_recursive(root: Path) -> List[Path]:
    if not root.exists():
        return []
    out = []
    for p in root.rglob("*"):
        if p.is_file() and is_image(p):
            out.append(p)
    return out

def safe_copy(src: Path, dst: Path, mode: str = "copy") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(str(src), str(dst))
    elif mode == "symlink":
        os.symlink(str(src), str(dst))
    else:
        raise ValueError(f"Unknown mode: {mode}")

def cap_aug_per_group(paths: List[Path], group_regex: str, k: int, seed: int) -> List[Path]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Path]] = {}
    for p in paths:
        gid = infer_group_id(p, group_regex)
        buckets.setdefault(gid, []).append(p)
    out = []
    for gid, items in buckets.items():
        rng.shuffle(items)
        out.extend(items[:k] if k > 0 else items)
    return out

def enforce_orig_ratio(
    orig: List[Path],
    aug: List[Path],
    orig_ratio: float,
    seed: int
) -> Tuple[List[Path], List[Path]]:
    """
    Keep all original; downsample augmented so that:
      orig / (orig + aug) ~= orig_ratio
    """
    if orig_ratio <= 0.0 or orig_ratio >= 1.0:
        return orig, aug
    n_orig = len(orig)
    if n_orig == 0:
        return orig, aug
    target_aug = int(round((1.0 - orig_ratio) / orig_ratio * n_orig))
    rng = random.Random(seed)
    aug2 = list(aug)
    rng.shuffle(aug2)
    if target_aug < len(aug2):
        aug2 = aug2[:target_aug]
    return orig, aug2

def dedup_exact(paths: List[Path], seed: int) -> List[Path]:
    """
    Exact dedup by SHA1; keeps first occurrence.
    """
    rng = random.Random(seed)
    paths2 = list(paths)
    rng.shuffle(paths2)
    seen: Dict[str, Path] = {}
    out = []
    for p in paths2:
        try:
            h = sha1_file(p)
        except Exception:
            continue
        if h not in seen:
            seen[h] = p
            out.append(p)
    # Restore deterministic ordering (path string) for stability
    out.sort(key=lambda x: str(x).lower())
    return out

def write_listfile(paths: List[Path], out_path: Path, mode: str = "name", base_dir: Optional[Path] = None) -> None:
    """Write a manifest file.

    Parameters
    ----------
    paths:
        Materialized file paths to record.
    out_path:
        Destination manifest file.
    mode:
        One of ``name`` (basename only), ``relative`` (relative to ``base_dir``),
        or ``absolute``. Basename-only manifests are the safest default for
        public repositories because they do not expose local paths and are
        compatible with ``read_name_list`` in ``main.py``.
    base_dir:
        Root used when ``mode='relative'``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if mode not in {"name", "relative", "absolute"}:
        raise ValueError(f"Unknown list mode: {mode}")
    with out_path.open("w", encoding="utf-8") as f:
        for p in paths:
            if mode == "name":
                value = p.name
            elif mode == "relative":
                if base_dir is None:
                    raise ValueError("base_dir is required when list mode is 'relative'")
                value = str(p.relative_to(base_dir))
            else:
                value = str(p.resolve())
            f.write(value + "\n")

def main():
    ap = argparse.ArgumentParser("Prepare unified Mpox dataset from your data + MSLD v1 + MSLD v2")

    # Your base dataset (the one you already used)
    ap.add_argument("--base_orig_root", required=True, help="Your original images root (has Monkey Pox / Others)")
    ap.add_argument("--base_aug_root", required=True, help="Your augmented images root (has Monkeypox_augmented / Others_augmented)")
    ap.add_argument("--base_orig_pos", default="Monkey Pox")
    ap.add_argument("--base_orig_neg", default="Others")
    ap.add_argument("--base_aug_pos", default="Monkeypox_augmented")
    ap.add_argument("--base_aug_neg", default="Others_augmented")

    # MSLD v1.0
    ap.add_argument("--msld_v1_root", required=True, help="MSLD v1.0 folder (contains Original Images + Augmented Images)")
    ap.add_argument("--msld_v1_orig_pos", default="Monkey Pox")
    ap.add_argument("--msld_v1_orig_neg", default="Others")
    ap.add_argument("--msld_v1_aug_pos", default="Monkeypox_augmented")
    ap.add_argument("--msld_v1_aug_neg", default="Others_augmented")

    # MSLD v2.0
    ap.add_argument("--msld_v2_root", required=True, help="MSLD v2.0 folder (contains Original Images/FOLDS + Augmented Images/FOLDS_AUG)")
    ap.add_argument("--include_v2_test", action="store_true", help="Include v2 'Test' images in selection (default OFF for safety)")
    ap.add_argument("--v2_folds", default="fold1,fold2,fold3,fold4,fold5", help="Comma list of folds to use")

    # Selection controls
    ap.add_argument("--train_orig_ratio", type=float, default=0.7, help="Target ratio of originals vs (orig+aug). Aug will be downsampled.")
    ap.add_argument("--max_aug_per_group", type=int, default=3, help="Max augmented variants per group-id (per class pool).")
    ap.add_argument("--group_regex", default=r"(.+?)_[0-9]+$", help="Regex (with capture group 1) to infer group ID from filename stem.")
    ap.add_argument("--seed", type=int, default=42)

    # Dedup + output
    ap.add_argument("--dedup_exact", action="store_true", help="Exact dedup by SHA1 across each pool (slower, but safer).")
    ap.add_argument("--out_root", required=True, help="Output folder to write unified curated dataset")
    ap.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy", help="How to materialize files (copy is safest).")
    ap.add_argument("--list_mode", choices=["name", "relative", "absolute"], default="name",
                    help="Manifest path style. 'name' is the safest default for reproducible public repos.")

    args = ap.parse_args()

    rng = random.Random(args.seed)

    base_orig_root = Path(args.base_orig_root)
    base_aug_root = Path(args.base_aug_root)
    v1_root = Path(args.msld_v1_root)
    v2_root = Path(args.msld_v2_root)
    out_root = Path(args.out_root)

    # Output structure (matches your pipeline naming)
    out_orig_pos = out_root / "Original Images" / "Monkey Pox"
    out_orig_neg = out_root / "Original Images" / "Others"
    out_aug_pos = out_root / "Augmented Images" / "Monkeypox_augmented"
    out_aug_neg = out_root / "Augmented Images" / "Others_augmented"

    # ---- Collect BASE (your dataset) ----
    base_orig_pos_paths = list_images_recursive(base_orig_root / args.base_orig_pos)
    base_orig_neg_paths = list_images_recursive(base_orig_root / args.base_orig_neg)
    base_aug_pos_paths  = list_images_recursive(base_aug_root / args.base_aug_pos)
    base_aug_neg_paths  = list_images_recursive(base_aug_root / args.base_aug_neg)

    # ---- Collect MSLD v1 ----
    v1_orig_pos_paths = list_images_recursive(v1_root / "Original Images" / args.msld_v1_orig_pos)
    v1_orig_neg_paths = list_images_recursive(v1_root / "Original Images" / args.msld_v1_orig_neg)
    v1_aug_pos_paths  = list_images_recursive(v1_root / "Augmented Images" / args.msld_v1_aug_pos)
    v1_aug_neg_paths  = list_images_recursive(v1_root / "Augmented Images" / args.msld_v1_aug_neg)

    # ---- Collect MSLD v2 ----
    folds = [x.strip() for x in args.v2_folds.split(",") if x.strip()]
    v2_orig_pos_paths: List[Path] = []
    v2_orig_neg_paths: List[Path] = []
    v2_aug_pos_paths: List[Path] = []
    v2_aug_neg_paths: List[Path] = []

    # v2 originals: ...\Original Images\FOLDS\foldX\{Train,Valid,Test}\{classes...}
    v2_orig_base = v2_root / "Original Images" / "FOLDS"
    v2_aug_base  = v2_root / "Augmented Images" / "FOLDS_AUG"

    splits = ["Train", "Valid"] + (["Test"] if args.include_v2_test else [])
    # In v2, "Monkeypox" is positive, all other folders are negative
    for fold in folds:
        for split in splits:
            split_dir = v2_orig_base / fold / split
            if split_dir.exists():
                # positive
                v2_orig_pos_paths += list_images_recursive(split_dir / "Monkeypox")
                # negatives: all other class folders under split
                for cls in ["Chickenpox", "Cowpox", "Healthy", "HFMD", "Measles"]:
                    v2_orig_neg_paths += list_images_recursive(split_dir / cls)

        # v2 augmented: ...\Augmented Images\FOLDS_AUG\foldX_AUG\{Train,Valid,Test}\{classes...}
        fold_aug = f"{fold}_AUG"
        for split in splits:
            split_dir = v2_aug_base / fold_aug / split
            if split_dir.exists():
                v2_aug_pos_paths += list_images_recursive(split_dir / "Monkeypox")
                for cls in ["Chickenpox", "Cowpox", "Healthy", "HFMD", "Measles"]:
                    v2_aug_neg_paths += list_images_recursive(split_dir / cls)

    # ---- Merge pools ----
    orig_pos = base_orig_pos_paths + v1_orig_pos_paths + v2_orig_pos_paths
    orig_neg = base_orig_neg_paths + v1_orig_neg_paths + v2_orig_neg_paths
    aug_pos  = base_aug_pos_paths  + v1_aug_pos_paths  + v2_aug_pos_paths
    aug_neg  = base_aug_neg_paths  + v1_aug_neg_paths  + v2_aug_neg_paths

    # ---- Cap augmented per group (prevents augmented dominance) ----
    aug_pos = cap_aug_per_group(aug_pos, args.group_regex, args.max_aug_per_group, args.seed)
    aug_neg = cap_aug_per_group(aug_neg, args.group_regex, args.max_aug_per_group, args.seed)

    # ---- Enforce orig ratio (downsample aug to match) ----
    orig_pos, aug_pos = enforce_orig_ratio(orig_pos, aug_pos, args.train_orig_ratio, args.seed + 1)
    orig_neg, aug_neg = enforce_orig_ratio(orig_neg, aug_neg, args.train_orig_ratio, args.seed + 2)

    # ---- Optional exact dedup ----
    if args.dedup_exact:
        orig_pos = dedup_exact(orig_pos, args.seed + 10)
        orig_neg = dedup_exact(orig_neg, args.seed + 11)
        aug_pos  = dedup_exact(aug_pos,  args.seed + 12)
        aug_neg  = dedup_exact(aug_neg,  args.seed + 13)

    # ---- Materialize curated dataset ----
    def materialize(pool: List[Path], out_dir: Path, prefix: str) -> List[Path]:
        out_paths: List[Path] = []
        for src in pool:
            # Keep source uniqueness by prefixing with dataset hint + original filename
            dst = out_dir / f"{prefix}__{src.name}"
            try:
                safe_copy(src, dst, mode=args.mode)
                out_paths.append(dst)
            except Exception:
                # skip unreadable files
                continue
        return out_paths

    # Prefix hints to avoid name collisions
    # Base
    # v1
    # v2
    # (We detect by substring, not perfect but practical)
    def prefix_for(p: Path) -> str:
        parts_lower = {part.lower() for part in p.parts}
        if "msld v1.0" in parts_lower:
            return "v1"
        if "msld v2.0" in parts_lower:
            return "v2"
        return "base"

    # Materialize with prefixes
    orig_pos_out = []
    for p in orig_pos:
        orig_pos_out.append((p, prefix_for(p)))
    orig_neg_out = []
    for p in orig_neg:
        orig_neg_out.append((p, prefix_for(p)))
    aug_pos_out = []
    for p in aug_pos:
        aug_pos_out.append((p, prefix_for(p)))
    aug_neg_out = []
    for p in aug_neg:
        aug_neg_out.append((p, prefix_for(p)))

    orig_pos_paths_out = []
    for src, pre in orig_pos_out:
        dst = out_orig_pos / f"{pre}__{src.name}"
        try:
            safe_copy(src, dst, mode=args.mode)
            orig_pos_paths_out.append(dst)
        except Exception:
            pass

    orig_neg_paths_out = []
    for src, pre in orig_neg_out:
        dst = out_orig_neg / f"{pre}__{src.name}"
        try:
            safe_copy(src, dst, mode=args.mode)
            orig_neg_paths_out.append(dst)
        except Exception:
            pass

    aug_pos_paths_out = []
    for src, pre in aug_pos_out:
        dst = out_aug_pos / f"{pre}__{src.name}"
        try:
            safe_copy(src, dst, mode=args.mode)
            aug_pos_paths_out.append(dst)
        except Exception:
            pass

    aug_neg_paths_out = []
    for src, pre in aug_neg_out:
        dst = out_aug_neg / f"{pre}__{src.name}"
        try:
            safe_copy(src, dst, mode=args.mode)
            aug_neg_paths_out.append(dst)
        except Exception:
            pass

    # ---- Write list files ----
    lists_dir = out_root / "lists"
    write_listfile(sorted(orig_pos_paths_out, key=lambda x: str(x).lower()), lists_dir / "Unified_Monkey_Original.txt", mode=args.list_mode, base_dir=out_root)
    write_listfile(sorted(orig_neg_paths_out, key=lambda x: str(x).lower()), lists_dir / "Unified_NonMonkey_Original.txt", mode=args.list_mode, base_dir=out_root)
    write_listfile(sorted(aug_pos_paths_out,  key=lambda x: str(x).lower()), lists_dir / "Unified_Monkey_Aug.txt", mode=args.list_mode, base_dir=out_root)
    write_listfile(sorted(aug_neg_paths_out,  key=lambda x: str(x).lower()), lists_dir / "Unified_NonMonkey_Aug.txt", mode=args.list_mode, base_dir=out_root)

    # ---- Print summary ----
    print("\n=== Curated dataset created ===")
    print(f"Output root: {out_root}")
    print(f"Original Monkey: {len(orig_pos_paths_out)}")
    print(f"Original Others: {len(orig_neg_paths_out)}")
    print(f"Aug Monkey:      {len(aug_pos_paths_out)}")
    print(f"Aug Others:      {len(aug_neg_paths_out)}")
    print(f"List files in:   {lists_dir}")
    print(" - Unified_Monkey_Original.txt")
    print(" - Unified_NonMonkey_Original.txt")
    print(" - Unified_Monkey_Aug.txt")
    print(" - Unified_NonMonkey_Aug.txt")

if __name__ == "__main__":
    main()