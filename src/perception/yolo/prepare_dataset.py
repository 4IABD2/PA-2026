"""Prépare la structure train/val + le data.yaml pour l'entraînement YOLO.

Lit une ou plusieurs runs (images/ + labels_yolo_enriched/), mélange, split
train/val, et matérialise les fichiers dans ``--output`` (symlink si possible,
sinon copie — Windows sans droits admin). Génère aussi ``<output>/data.yaml``
avec le chemin absolu correct et les 11 classes finales (importées de
``enrich_labels.FINAL_CLASSES`` pour éviter toute désync).

Usage:
    uv run -m src.perception.yolo.prepare_dataset --runs data/runs/<session>/* --output data/yolo_dataset
    uv run -m src.perception.yolo.prepare_dataset --runs data/runs/<sessionA>/* data/runs/<sessionB>/* --split 0.85
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from src.dataset.labeling.enrich_labels import FINAL_CLASSES


def _expand_runs(paths: list[str]) -> list[Path]:
    """Étend chaque argument en liste de runs (dossiers avec images/).

    Gère trois cas (et le ``*`` littéral non expansé par PowerShell) :
    - chemin contenant ``*`` → glob manuel ;
    - dossier de session (sans images/) → ses sous-dossiers de runs ;
    - run directe (avec images/) → elle-même.
    """
    runs: list[Path] = []
    for raw in paths:
        if "*" in raw or "?" in raw:
            parent = Path(raw).parent
            pattern = Path(raw).name
            candidates = sorted(parent.glob(pattern)) if parent.exists() else []
        else:
            candidates = [Path(raw)]
        for c in candidates:
            if not c.is_dir():
                continue
            if (c / "images").is_dir():
                runs.append(c)
            else:  # dossier de session → prendre les runs dedans
                runs.extend(s for s in sorted(c.iterdir()) if (s / "images").is_dir())
    return runs


def _materialize(src: Path, dst: Path) -> None:
    """Symlink src -> dst, fallback copie si les symlinks sont interdits (Windows)."""
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _write_data_yaml(output: Path) -> None:
    """Écrit <output>/data.yaml avec chemin absolu + classes finales."""
    lines = [
        f"path: {output.resolve().as_posix()}",
        "train: train/images",
        "val: val/images",
        "",
        "names:",
        *[f"  {i}: {name}" for i, name in enumerate(FINAL_CLASSES)],
        "",
    ]
    (output / "data.yaml").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", nargs="+", required=True, help="Chemins vers les runs"
    )
    parser.add_argument(
        "--output", default="data/yolo_dataset", help="Dossier de sortie"
    )
    parser.add_argument(
        "--split", type=float, default=0.8, help="Ratio train (défaut 0.8)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    for sub in ("train/images", "train/labels", "val/images", "val/labels"):
        (output / sub).mkdir(parents=True, exist_ok=True)

    # (img, label, stem) — le stem est préfixé par le nom de la run pour éviter
    # les collisions : toutes les runs numérotent leurs frames 000000.. → sans
    # préfixe, un seul dossier plat n'en garderait qu'une seule par numéro.
    all_pairs: list[tuple[Path, Path, str]] = []
    for run in _expand_runs(args.runs):
        images_dir = run / "images"
        labels_dir = run / "labels_yolo_enriched"
        if not images_dir.exists() or not labels_dir.exists():
            print(f"SKIP {run} — images/ ou labels_yolo_enriched/ manquant")
            continue
        n_before = len(all_pairs)
        for img in sorted(images_dir.glob("*.jpg")):
            label = labels_dir / img.with_suffix(".txt").name
            if label.exists():
                all_pairs.append((img, label, f"{run.name}__{img.stem}"))
        print(f"  {run.name}: {len(all_pairs) - n_before} paires")

    if not all_pairs:
        raise SystemExit("Aucune paire image/label trouvée. As-tu lancé enrich ?")

    random.seed(args.seed)
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * args.split)
    splits = {"train": all_pairs[:split_idx], "val": all_pairs[split_idx:]}

    for split_name, pairs in splits.items():
        for img, label, stem in pairs:
            _materialize(img, output / split_name / "images" / f"{stem}.jpg")
            _materialize(label, output / split_name / "labels" / f"{stem}.txt")
        print(f"{split_name}: {len(pairs)} frames")

    _write_data_yaml(output)
    print(f"Dataset prêt dans {output}")
    print(f"Config écrite : {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
