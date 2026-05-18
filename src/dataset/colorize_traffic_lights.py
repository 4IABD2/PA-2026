"""Post-process : étend les labels YOLO ``traffic_light`` (classe 2) en
``red_light`` / ``yellow_light`` / ``green_light`` via analyse HSV de la zone
du bbox dans l'image RGB.

Architecture : le collector écrit des labels 3-classes (vehicle=0, walker=1,
traffic_light=2) dans ``labels_yolo/``. Ce script lit ces labels et les
images correspondantes, classe chaque feu par couleur dominante dans son bbox,
et écrit un dossier ``labels_yolo_color/`` avec le mapping 5-classes
(vehicle=0, walker=1, red_light=2, yellow_light=3, green_light=4) que YOLO
consomme à l'entraînement.

Usage:
    uv run -m src.dataset.colorize_traffic_lights --run data/runs/<run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Seuils HSV (OpenCV : H ∈ [0, 180], S ∈ [0, 255], V ∈ [0, 255]).
# Rouge "wrap-around" : nécessite deux plages aux extrémités du cercle.
_RED_LOWER1 = np.array([0, 100, 80])
_RED_UPPER1 = np.array([10, 255, 255])
_RED_LOWER2 = np.array([170, 100, 80])
_RED_UPPER2 = np.array([180, 255, 255])
_YELLOW_LOWER = np.array([18, 100, 80])
_YELLOW_UPPER = np.array([35, 255, 255])
_GREEN_LOWER = np.array([40, 80, 80])
_GREEN_UPPER = np.array([90, 255, 255])

_INPUT_TL_CLASS = 2  # classe générique "traffic_light" dans labels_yolo/
_OUT_RED = 2
_OUT_YELLOW = 3
_OUT_GREEN = 4
_MIN_COLOR_PIXELS = 8  # en dessous, on considère qu'aucune couleur n'est claire


def classify_tl_color(
    bgr_crop: np.ndarray,
) -> tuple[int | None, dict[int, int]]:
    """Classe une vignette BGR de feu en rouge/jaune/vert.

    Retourne (classe, compteurs HSV par couleur). Classe = None si aucun
    canal n'atteint le seuil minimum — laisse l'appelant décider quoi faire
    (ignorer, logger, sauvegarder le crop pour inspection).
    """
    if bgr_crop.size == 0:
        return None, {_OUT_RED: 0, _OUT_YELLOW: 0, _OUT_GREEN: 0}
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, _RED_LOWER1, _RED_UPPER1) + cv2.inRange(
        hsv, _RED_LOWER2, _RED_UPPER2
    )
    yellow = cv2.inRange(hsv, _YELLOW_LOWER, _YELLOW_UPPER)
    green = cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)

    counts = {
        _OUT_RED: int((red > 0).sum()),
        _OUT_YELLOW: int((yellow > 0).sum()),
        _OUT_GREEN: int((green > 0).sum()),
    }
    best_class = max(counts, key=counts.get)
    if counts[best_class] < _MIN_COLOR_PIXELS:
        return None, counts
    return best_class, counts


def _crop_bbox(image: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    """Découpe un bbox YOLO normalisé (centre + taille) dans l'image."""
    H, W = image.shape[:2]
    x1 = max(0, int((x - w / 2) * W))
    y1 = max(0, int((y - h / 2) * H))
    x2 = min(W, int((x + w / 2) * W))
    y2 = min(H, int((y + h / 2) * H))
    return image[y1:y2, x1:x2]


def process_run(run_dir: Path, debug_drops: bool = False) -> dict[str, int]:
    """Traite tous les labels d'une run. Retourne les compteurs par catégorie.

    Si ``debug_drops`` est vrai, chaque feu droppé est sauvegardé dans
    ``debug_dropped_tl/`` avec un nom de fichier qui encode les compteurs
    HSV : ``<frame>_<idx>_r<R>_y<Y>_g<G>.png``. Pratique pour ouvrir le
    dossier et vérifier à l'œil si les drops sont légitimes.
    """
    labels_dir = run_dir / "labels_yolo"
    images_dir = run_dir / "images"
    out_dir = run_dir / "labels_yolo_color"

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Pas de dossier labels_yolo dans {run_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Pas de dossier images dans {run_dir}")

    out_dir.mkdir(exist_ok=True)
    debug_dir = run_dir / "debug_dropped_tl"
    if debug_drops:
        debug_dir.mkdir(exist_ok=True)

    stats = {"frames": 0, "tl_in": 0, "red": 0, "yellow": 0, "green": 0, "dropped": 0}

    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = images_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        stats["frames"] += 1

        out_lines: list[str] = []
        bbox_idx_in_frame = 0
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            x, y, w, h = (float(p) for p in parts[1:5])

            if cls != _INPUT_TL_CLASS:
                out_lines.append(line)
                continue

            stats["tl_in"] += 1
            crop = _crop_bbox(image, x, y, w, h)
            new_cls, counts = classify_tl_color(crop)
            if new_cls is None:
                stats["dropped"] += 1
                if debug_drops and crop.size > 0:
                    name = (
                        f"{label_path.stem}_{bbox_idx_in_frame}"
                        f"_r{counts[_OUT_RED]}"
                        f"_y{counts[_OUT_YELLOW]}"
                        f"_g{counts[_OUT_GREEN]}.png"
                    )
                    cv2.imwrite(str(debug_dir / name), crop)
                bbox_idx_in_frame += 1
                continue
            if new_cls == _OUT_RED:
                stats["red"] += 1
            elif new_cls == _OUT_YELLOW:
                stats["yellow"] += 1
            elif new_cls == _OUT_GREEN:
                stats["green"] += 1
            out_lines.append(f"{new_cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            bbox_idx_in_frame += 1

        (out_dir / label_path.name).write_text(
            "\n".join(out_lines) + ("\n" if out_lines else "")
        )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Colorize traffic_light bboxes (red/yellow/green) via HSV"
    )
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Chemin d'une run (ex: data/runs/2026-05-18_town01_clearnoon)",
    )
    parser.add_argument(
        "--debug-drops",
        action="store_true",
        help="Sauvegarde chaque crop droppe dans <run>/debug_dropped_tl/ pour inspection",
    )
    args = parser.parse_args()

    if not args.run.exists():
        raise SystemExit(f"Run introuvable : {args.run}")

    stats = process_run(args.run, debug_drops=args.debug_drops)
    print(f"Run: {args.run}")
    print(f"  Frames lues:        {stats['frames']}")
    print(f"  Feux en entree:     {stats['tl_in']}")
    print(f"  -> rouges:          {stats['red']}")
    print(f"  -> jaunes:          {stats['yellow']}")
    print(f"  -> verts:           {stats['green']}")
    print(f"  -> dropped (flou):  {stats['dropped']}")
    print(f"Output: {args.run / 'labels_yolo_color'}")
    if args.debug_drops:
        print(f"Crops droppes:      {args.run / 'debug_dropped_tl'}")


if __name__ == "__main__":
    main()
