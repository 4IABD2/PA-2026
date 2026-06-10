"""Diagnostic d'une run dataset.

Sans argument supplémentaire : compteurs par classe (vehicle/walker/feux/
panneaux), taux de frames non-vides, taille disque par sous-dossier.

Avec ``--frame N`` : inspecte le mask d'instance de la frame N — quelles
classes sont présentes, combien d'instances par classe, taille pixel des
clusters. Utile pour debugger pourquoi un objet visible n'est pas labellisé.

Usage :
    uv run -m src.dataset.inspect_run --run data/runs/<run>
    uv run -m src.dataset.inspect_run --run data/runs/<run> --frame 17
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.dataset.enrich_labels import FINAL_CLASSES
from src.dataset.yolo_labels import (
    YOLO_CLASS_MAPPING,
    _MIN_BBOX_SIDE_PX,
    _MIN_SIGN_BBOX_SIDE_PX,
    _MIN_SIGN_PIXELS,
    _MIN_TL_BBOX_SIDE_PX,
    _MIN_TL_PIXELS,
)

_RAW_CLASS_NAMES = {v: k for k, v in YOLO_CLASS_MAPPING.items()}

# Mapping CityScape class_id → nom (pour --frame)
_CITYSCAPE_NAMES = {
    0: "Unlabeled",
    1: "Roads",
    2: "SideWalks",
    3: "Building",
    4: "Wall",
    5: "Fence",
    6: "Pole",
    7: "TrafficLight",
    8: "TrafficSign",
    9: "Vegetation",
    10: "Terrain",
    11: "Sky",
    12: "Pedestrian",
    13: "Rider",
    14: "Car",
    15: "Truck",
    16: "Bus",
    17: "Train",
    18: "Motorcycle",
    19: "Bicycle",
}


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _format_size(n_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(n_bytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.1f} {units[idx]}"


def _summarize_run(run_dir: Path) -> None:
    labels_raw = run_dir / "labels_yolo"
    labels_color = run_dir / "labels_yolo_color"
    images = run_dir / "images"

    if not images.is_dir():
        raise SystemExit(f"Pas de dossier images/ dans {run_dir}")

    n_frames = len(list(images.glob("*.jpg")))
    n_labels_raw = len(list(labels_raw.glob("*.txt"))) if labels_raw.is_dir() else 0
    non_empty_raw = (
        sum(1 for p in labels_raw.glob("*.txt") if p.stat().st_size > 0)
        if labels_raw.is_dir()
        else 0
    )

    print(f"Run: {run_dir}")
    print(f"  Frames:                {n_frames}")
    print(
        f"  Labels raw generated:  {n_labels_raw} ({100 * n_labels_raw / max(1, n_frames):.0f}%)"
    )
    print(
        f"  Labels raw non-empty:  {non_empty_raw} "
        f"({100 * non_empty_raw / max(1, n_frames):.0f}%)"
    )
    print()

    # Compteurs par classe — préfère labels_yolo_color s'il existe (12 classes),
    # sinon labels_yolo (4 classes raw).
    if labels_color.is_dir():
        names = FINAL_CLASSES
        source = labels_color
        print(f"  Compteurs par classe ({labels_color.name}/) :")
    elif labels_raw.is_dir():
        names = [
            _RAW_CLASS_NAMES.get(i, f"cls_{i}")
            for i in range(max(_RAW_CLASS_NAMES) + 1)
        ]
        source = labels_raw
        print(f"  Compteurs par classe ({labels_raw.name}/, raw 4 classes) :")
    else:
        names = []
        source = None
        print("  Aucun dossier labels_yolo* trouvé.")

    if source is not None:
        counter: Counter[int] = Counter()
        for p in source.glob("*.txt"):
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    cls = int(line.split()[0])
                    counter[cls] += 1
                except ValueError:
                    continue

        max_name_len = max((len(n) for n in names), default=10)
        for i, name in enumerate(names):
            count = counter.get(i, 0)
            warn = ""
            if name == "walker" and count == 0:
                warn = "  ⚠ collector ne spawne pas encore de walkers"
            print(f"    {name:<{max_name_len}}  {count:>6}{warn}")
    print()

    print("  Disque :")
    subdirs = [
        "images",
        "depth",
        "instance",
        "semantic",
        "viz",
        "labels_yolo",
        "labels_yolo_color",
        "debug_dropped_tl",
        "debug_dropped_signs",
    ]
    total = 0
    for sub in subdirs:
        size = _dir_size_bytes(run_dir / sub)
        if size > 0:
            total += size
            print(f"    {sub:<22} {_format_size(size):>10}")
    print(f"    {'TOTAL':<22} {_format_size(total):>10}")


def _inspect_frame(run_dir: Path, frame_id: int) -> None:
    npy_path = run_dir / "instance" / f"{frame_id:06d}.npy"
    if not npy_path.exists():
        raise SystemExit(f"Frame inexistante : {npy_path}")

    packed = np.load(npy_path)
    class_map = (packed >> 16) & 0xFF
    instance_map = packed & 0xFFFF

    print(f"Frame {frame_id:06d}  ({class_map.shape[1]}x{class_map.shape[0]})")
    print()
    print("  Classes presentes (id: nom: pixels) :")
    for cid in sorted(np.unique(class_map).tolist()):
        n_px = int((class_map == cid).sum())
        name = _CITYSCAPE_NAMES.get(int(cid), "?")
        print(f"    {cid:>2}  {name:<15} {n_px:>8}")
    print()

    for target_id, target_name in (
        (7, "TrafficLight"),
        (8, "TrafficSign"),
        (14, "Car"),
        (12, "Pedestrian"),
    ):
        mask = class_map == target_id
        if not mask.any():
            continue
        instances = np.unique(instance_map[mask])
        instances = [int(i) for i in instances if i != 0]
        print(f"  Instances {target_name} (id: pixels) :")
        if not instances:
            print("    (toutes en instance_id=0)")
        else:
            for iid in sorted(instances):
                n_px = int((instance_map == iid).sum())
                print(f"    {iid:>6}  {n_px:>6}")
        print()

    # Composants connexes pour les classes statiques (TL + Sign).
    # Reproduit la logique de yolo_labels.py pour montrer ce qui passe/échoue.
    for target_id, target_name, min_pixels, min_side in (
        (7, "TrafficLight", _MIN_TL_PIXELS, _MIN_TL_BBOX_SIDE_PX),
        (8, "TrafficSign", _MIN_SIGN_PIXELS, _MIN_SIGN_BBOX_SIDE_PX),
    ):
        mask = (class_map == target_id).astype(np.uint8)
        if not mask.any():
            continue
        n_comp, comp_labels = cv2.connectedComponents(mask, connectivity=8)
        print(
            f"  Blobs CC {target_name}  (filtres : min_pixels={min_pixels}, "
            f"min_side={min_side})"
        )
        print(f"    {'idx':>4}  {'pixels':>7}  {'bbox (x1,y1,x2,y2)':>22}  status")
        kept = 0
        for comp_id in range(1, n_comp):
            blob = comp_labels == comp_id
            pixels = int(blob.sum())
            ys, xs = np.where(blob)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            w_px = x2 - x1
            h_px = y2 - y1
            reasons = []
            if pixels < min_pixels:
                reasons.append(f"pixels<{min_pixels}")
            if w_px < min_side or h_px < min_side:
                reasons.append(f"side<{min_side}")
            status = "KEEP" if not reasons else "drop:" + ",".join(reasons)
            if not reasons:
                kept += 1
            print(
                f"    {comp_id:>4}  {pixels:>7}  ({x1:>4},{y1:>4},{x2:>4},{y2:>4})"
                f"  {status}"
            )
        print(f"    -> {kept} blob(s) gardé(s), {n_comp - 1 - kept} droppé(s)")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic d'une run dataset (compteurs / inspection frame)."
    )
    parser.add_argument("--run", type=Path, required=True, help="Dossier d'une run")
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Inspecte une frame précise (mask instance) au lieu du résumé global.",
    )
    args = parser.parse_args()

    if not args.run.exists():
        raise SystemExit(f"Run introuvable : {args.run}")

    if args.frame is not None:
        _inspect_frame(args.run, args.frame)
    else:
        _summarize_run(args.run)


if __name__ == "__main__":
    main()
