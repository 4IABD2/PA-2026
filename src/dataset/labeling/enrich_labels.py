"""Post-process : résout la COULEUR des feux dans les labels YOLO.

Pipeline :
- ``labels_yolo/`` (collector) → ``labels_yolo_enriched/`` (entraînement, 11 classes)

Le collector écrit les feux en classe générique ``2`` (traffic_light). Ce script
lit le crop RGB du bbox et classe la couleur dominante (rouge/jaune/vert) en
classes finales 2/3/4. Les autres classes (vehicle, walker, et les panneaux
5..10 déjà labellisés en ground-truth par ``yolo_labels``) sont laissées telles
quelles. Non-destructif : ne lit que ``labels_yolo/``, n'écrit que ``<out_name>/``.

Usage :
    uv run -m src.dataset enrich --run data/runs/<session>/<run>
    uv run -m src.dataset enrich --run data/runs/<session>/<run> --debug-drops
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Schéma de classes
# ----------------------------------------------------------------------------

RAW_VEHICLE = 0
RAW_WALKER = 1
RAW_TRAFFIC_LIGHT = 2  # générique, couleur résolue ici

FINAL_CLASSES = [
    "vehicle",  # 0 (passe-through depuis raw 0)
    "walker",  # 1 (passe-through depuis raw 1)
    "red_light",  # 2
    "yellow_light",  # 3
    "green_light",  # 4
    "speed_30",  # 5  (ground-truth direct depuis yolo_labels)
    "speed_40",  # 6
    "speed_60",  # 7
    "speed_90",  # 8
    "stop",  # 9
    "yield",  # 10
]

# ----------------------------------------------------------------------------
# Couleur des feux — seuils HSV (OpenCV : H ∈ [0, 180])
# ----------------------------------------------------------------------------

_RED_LOWER1 = np.array([0, 100, 80])
_RED_UPPER1 = np.array([10, 255, 255])
_RED_LOWER2 = np.array([170, 100, 80])
_RED_UPPER2 = np.array([180, 255, 255])
_YELLOW_LOWER = np.array([18, 100, 80])
_YELLOW_UPPER = np.array([35, 255, 255])
_GREEN_LOWER = np.array([40, 80, 80])
_GREEN_UPPER = np.array([90, 255, 255])

_TL_RED = 2
_TL_YELLOW = 3
_TL_GREEN = 4
_MIN_TL_COLOR_PIXELS = 8


def classify_tl_color(bgr_crop: np.ndarray) -> tuple[int | None, dict[int, int]]:
    """Retourne (classe finale, compteurs HSV). None si aucun canal couleur
    n'atteint le seuil — l'appelant décide quoi faire."""
    if bgr_crop.size == 0:
        return None, {_TL_RED: 0, _TL_YELLOW: 0, _TL_GREEN: 0}
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, _RED_LOWER1, _RED_UPPER1) + cv2.inRange(
        hsv, _RED_LOWER2, _RED_UPPER2
    )
    yellow = cv2.inRange(hsv, _YELLOW_LOWER, _YELLOW_UPPER)
    green = cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)
    counts = {
        _TL_RED: int((red > 0).sum()),
        _TL_YELLOW: int((yellow > 0).sum()),
        _TL_GREEN: int((green > 0).sum()),
    }
    best = max(counts, key=counts.get)
    if counts[best] < _MIN_TL_COLOR_PIXELS:
        return None, counts
    return best, counts


def _crop_bbox(image: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    H, W = image.shape[:2]
    x1 = max(0, int((x - w / 2) * W))
    y1 = max(0, int((y - h / 2) * H))
    x2 = min(W, int((x + w / 2) * W))
    y2 = min(H, int((y + h / 2) * H))
    return image[y1:y2, x1:x2]


def process_run(
    run_dir: Path,
    debug_drops: bool = False,
    out_name: str = "labels_yolo_enriched",
) -> dict[str, int]:
    """Traverse les labels d'une run, écrit ``<out_name>/`` (couleur des feux).

    Non-destructif : ne lit que ``labels_yolo/`` (source de vérité écrite à la
    collecte) et n'écrit que dans ``<out_name>/``. ``out_name`` permet de viser
    un dossier de test pour comparer des réglages de seuils sans écraser le
    résultat canonique ``labels_yolo_enriched/``.

    Si ``debug_drops``, sauvegarde les feux droppés dans ``debug_dropped_tl/``
    (nom encodant les compteurs HSV) pour inspection visuelle.
    """
    labels_dir = run_dir / "labels_yolo"
    images_dir = run_dir / "images"
    out_dir = run_dir / out_name

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Pas de dossier labels_yolo dans {run_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Pas de dossier images dans {run_dir}")
    out_dir.mkdir(exist_ok=True)

    debug_tl_dir = run_dir / "debug_dropped_tl"
    if debug_drops:
        debug_tl_dir.mkdir(exist_ok=True)

    stats: dict[str, int] = {
        "frames": 0,
        "tl_in": 0,
        "tl_red": 0,
        "tl_yellow": 0,
        "tl_green": 0,
        "tl_dropped": 0,
    }

    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = images_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        stats["frames"] += 1

        out_lines: list[str] = []
        bbox_idx = 0
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            x, y, w, h = (float(p) for p in parts[1:5])

            # Feux : on résout la couleur via HSV. Tout le reste passe tel quel
            # (vehicle/walker + panneaux 5..10 déjà labellisés en ground-truth).
            if cls != RAW_TRAFFIC_LIGHT:
                out_lines.append(line)
                bbox_idx += 1
                continue

            stats["tl_in"] += 1
            crop = _crop_bbox(image, x, y, w, h)
            new_cls, counts = classify_tl_color(crop)
            if new_cls is None:
                stats["tl_dropped"] += 1
                if debug_drops and crop.size > 0:
                    name = (
                        f"{label_path.stem}_{bbox_idx}"
                        f"_r{counts[_TL_RED]}"
                        f"_y{counts[_TL_YELLOW]}"
                        f"_g{counts[_TL_GREEN]}.png"
                    )
                    cv2.imwrite(str(debug_tl_dir / name), crop)
                bbox_idx += 1
                continue
            if new_cls == _TL_RED:
                stats["tl_red"] += 1
            elif new_cls == _TL_YELLOW:
                stats["tl_yellow"] += 1
            elif new_cls == _TL_GREEN:
                stats["tl_green"] += 1
            out_lines.append(f"{new_cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            bbox_idx += 1

        (out_dir / label_path.name).write_text(
            "\n".join(out_lines) + ("\n" if out_lines else "")
        )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Résout la couleur des feux (raw class 2 → red/yellow/green)."
    )
    parser.add_argument(
        "--run", type=Path, required=True, help="Dossier d'une run dataset"
    )
    parser.add_argument(
        "--debug-drops",
        action="store_true",
        help="Sauvegarde les feux droppés dans debug_dropped_tl/",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="labels_yolo_enriched",
        help="Sous-dossier de sortie (defaut: labels_yolo_enriched). Utiliser un "
        "nom de test pour comparer des seuils sans écraser le résultat canonique.",
    )
    args = parser.parse_args()

    if not args.run.exists():
        raise SystemExit(f"Run introuvable : {args.run}")

    stats = process_run(args.run, debug_drops=args.debug_drops, out_name=args.out)
    print(f"Run: {args.run}")
    print(f"  Frames lues:     {stats['frames']}")
    print(f"  Feux en entree:  {stats['tl_in']}")
    print(f"    -> rouges:     {stats['tl_red']}")
    print(f"    -> jaunes:     {stats['tl_yellow']}")
    print(f"    -> verts:      {stats['tl_green']}")
    print(f"    -> dropped:    {stats['tl_dropped']}")
    print(f"Output: {args.run / args.out}")
    if args.debug_drops:
        print(f"Crops droppes TL: {args.run / 'debug_dropped_tl'}")


if __name__ == "__main__":
    main()
