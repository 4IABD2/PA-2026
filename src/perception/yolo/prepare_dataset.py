"""Prépare la structure train/val pour l'entraînement YOLO.

Usage:
    uv run -m src.perception.yolo.prepare_dataset --runs data/runs/2026-05-20_town01_clearnoon --output data/yolo_dataset --split 0.8
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="Chemins vers les runs")
    parser.add_argument("--output", default="data/yolo_dataset", help="Dossier de sortie")
    parser.add_argument("--split", type=float, default=0.8, help="Ratio train (défaut 0.8)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    for sub in ("train/images", "train/labels", "val/images", "val/labels"):
        (output / sub).mkdir(parents=True, exist_ok=True)

    all_pairs: list[tuple[Path, Path]] = []
    for run_path in args.runs:
        run = Path(run_path)
        images_dir = run / "images"
        labels_dir = run / "labels_yolo_enriched"
        if not images_dir.exists() or not labels_dir.exists():
            print(f"SKIP {run} — images/ ou labels_yolo_enriched/ manquant")
            continue
        for img in sorted(images_dir.glob("*.jpg")):
            label = labels_dir / img.with_suffix(".txt").name
            if label.exists():
                all_pairs.append((img, label))

    random.seed(args.seed)
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * args.split)
    splits = {"train": all_pairs[:split_idx], "val": all_pairs[split_idx:]}

    for split_name, pairs in splits.items():
        for img, label in pairs:
            img_dst = output / split_name / "images" / img.name
            lbl_dst = output / split_name / "labels" / label.name
            if not img_dst.exists():
                img_dst.symlink_to(img.resolve())
            if not lbl_dst.exists():
                lbl_dst.symlink_to(label.resolve())
        print(f"{split_name}: {len(pairs)} frames")

    print(f"Dataset prêt dans {output}")


if __name__ == "__main__":
    main()
