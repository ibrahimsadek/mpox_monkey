"""
Cross-dataset overlap audit between the unified training pool (MSLD v1.0 + v2.0)
and the external validation set (MSID).

Computes 64-bit perceptual hashes (pHash) for all images in both pools and reports
any pairs with Hamming distance <= 10 (the same threshold used for the intra-dataset
near-duplicate audit in Section 3.3 of the manuscript).

Usage:
    python check_overlap_msld_msid.py \
        --training_root "D:\Ibrahim\mpoxResearch\Analysis\Data\augmented_images" \
        --training_orig_root "D:\Ibrahim\mpoxResearch\Analysis\Data\original_images" \
        --msid_root "D:\Ibrahim\mpoxResearch\Analysis\Data\MSLD" \
        --hamming_threshold 10 \
        --output "overlap_audit_msld_msid.json"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

try:
    import imagehash
except ImportError:
    print("ERROR: imagehash not installed. Run: pip install imagehash")
    sys.exit(1)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(root: Path, label: str = "") -> List[Tuple[Path, str]]:
    images = []
    if not root.is_dir():
        print(f"WARNING: {root} not found, skipping.")
        return images
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            rel = str(f.relative_to(root))
            images.append((f, f"{label}/{rel}" if label else rel))
    return images


def compute_phashes(images: List[Tuple[Path, str]], hash_size: int = 8) -> Dict[str, str]:
    hashes = {}
    for i, (path, name) in enumerate(images):
        try:
            img = Image.open(path).convert("RGB")
            h = imagehash.phash(img, hash_size=hash_size)
            hashes[name] = str(h)
        except Exception as e:
            print(f"  SKIP {name}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  Hashed {i+1}/{len(images)}...")
    return hashes


def hamming_distance(h1: str, h2: str) -> int:
    import imagehash as ih
    return ih.hex_to_hash(h1) - ih.hex_to_hash(h2)


def main():
    p = argparse.ArgumentParser(description="Cross-dataset overlap audit (MSLD training vs MSID external)")
    p.add_argument("--training_root", type=str, required=True, help="Root of augmented training images")
    p.add_argument("--training_orig_root", type=str, required=True, help="Root of original training images")
    p.add_argument("--msid_root", type=str, required=True, help="Root of MSID (4-class external dataset)")
    p.add_argument("--hamming_threshold", type=int, default=10)
    p.add_argument("--hash_size", type=int, default=8)
    p.add_argument("--output", type=str, default="overlap_audit_msld_msid.json")
    args = p.parse_args()

    training_orig = Path(args.training_orig_root)
    msid_root = Path(args.msid_root)

    print("=" * 60)
    print("CROSS-DATASET OVERLAP AUDIT: MSLD Training vs. MSID External")
    print("=" * 60)

    print(f"\n[1/4] Collecting training original images from: {training_orig}")
    train_images = collect_images(training_orig, label="TRAIN")
    print(f"  Found {len(train_images)} training original images")

    print(f"\n[2/4] Collecting MSID images from: {msid_root}")
    msid_images = collect_images(msid_root, label="MSID")
    print(f"  Found {len(msid_images)} MSID images")

    print(f"\n[3/4] Computing perceptual hashes (hash_size={args.hash_size})...")
    print("  Training set:")
    train_hashes = compute_phashes(train_images, args.hash_size)
    print(f"  Done: {len(train_hashes)} hashes")

    print("  MSID set:")
    msid_hashes = compute_phashes(msid_images, args.hash_size)
    print(f"  Done: {len(msid_hashes)} hashes")

    print(f"\n[4/4] Cross-checking all pairs (threshold={args.hamming_threshold})...")
    overlaps = []
    train_names = list(train_hashes.keys())
    msid_names = list(msid_hashes.keys())

    total_comparisons = len(train_names) * len(msid_names)
    checked = 0

    for t_name in train_names:
        t_hash = train_hashes[t_name]
        for m_name in msid_names:
            m_hash = msid_hashes[m_name]
            dist = hamming_distance(t_hash, m_hash)
            if dist <= args.hamming_threshold:
                overlaps.append({
                    "training_image": t_name,
                    "msid_image": m_name,
                    "hamming_distance": dist,
                })
            checked += 1
        if checked % (len(msid_names) * 50) == 0:
            pct = 100.0 * checked / total_comparisons
            print(f"  {pct:.1f}% done ({checked}/{total_comparisons} pairs)...")

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Training images:    {len(train_hashes)}")
    print(f"MSID images:        {len(msid_hashes)}")
    print(f"Total comparisons:  {total_comparisons:,}")
    print(f"Hamming threshold:  {args.hamming_threshold}")
    print(f"Overlapping pairs:  {len(overlaps)}")

    if overlaps:
        print(f"\nOVERLAPPING PAIRS (Hamming distance <= {args.hamming_threshold}):")
        print(f"{'Training Image':50s}  {'MSID Image':50s}  Dist")
        print("-" * 110)
        for ov in sorted(overlaps, key=lambda x: x["hamming_distance"]):
            print(f"{ov['training_image']:50s}  {ov['msid_image']:50s}  {ov['hamming_distance']}")

        dist_zero = sum(1 for o in overlaps if o["hamming_distance"] == 0)
        dist_low = sum(1 for o in overlaps if 0 < o["hamming_distance"] <= 5)
        dist_mid = sum(1 for o in overlaps if 5 < o["hamming_distance"] <= 10)
        print(f"\nDistance distribution:")
        print(f"  Exact match (d=0):   {dist_zero}")
        print(f"  Near match (1-5):    {dist_low}")
        print(f"  Weak match (6-10):   {dist_mid}")

        unique_train = set(o["training_image"] for o in overlaps)
        unique_msid = set(o["msid_image"] for o in overlaps)
        print(f"\nUnique training images involved: {len(unique_train)}")
        print(f"Unique MSID images involved:    {len(unique_msid)}")
    else:
        print("\nNO OVERLAP DETECTED. The training pool and MSID are fully disjoint.")

    report = {
        "config": {
            "training_root": str(training_orig),
            "msid_root": str(msid_root),
            "hamming_threshold": args.hamming_threshold,
            "hash_size": args.hash_size,
        },
        "counts": {
            "training_images": len(train_hashes),
            "msid_images": len(msid_hashes),
            "total_comparisons": total_comparisons,
            "overlapping_pairs": len(overlaps),
        },
        "overlaps": overlaps,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
