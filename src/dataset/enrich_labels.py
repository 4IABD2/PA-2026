"""Post-process : transforme les labels YOLO raw 4-classes en labels finaux
12-classes.

Pipeline :
- ``labels_yolo/`` (collector, 4 classes : vehicle, walker, traffic_light, traffic_sign)
  → ``labels_yolo_enriched/`` (entraînement, 12 classes)

Deux passes pendant la même traversée d'un .txt :

1. **Couleur des feux** (raw class 2 → final 2/3/4)
   Analyse HSV du crop bbox. Drop si aucun canal couleur ne dépasse le seuil.

2. **Valeur des panneaux de vitesse** (raw class 3 → final 5..11)
   Template matching avec 7 templates synthétiques (cercle rouge + chiffre).
   Drop si le meilleur score est sous le seuil.

Usage :
    uv run -m src.dataset.enrich_labels --run data/runs/<run_id>
    uv run -m src.dataset.enrich_labels --run data/runs/<run_id> --debug-drops
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------------------
# Mapping raw → final
# ----------------------------------------------------------------------------

RAW_VEHICLE = 0
RAW_WALKER = 1
RAW_TRAFFIC_LIGHT = 2
RAW_TRAFFIC_SIGN = 3

FINAL_CLASSES = [
    "vehicle",       # 0 (passe-through depuis raw 0)
    "walker",        # 1 (passe-through depuis raw 1)
    "red_light",     # 2
    "yellow_light",  # 3
    "green_light",   # 4
    "speed_30",      # 5
    "speed_40",      # 6
    "speed_50",      # 7
    "speed_60",      # 8
    "speed_70",      # 9
    "speed_80",      # 10
    "speed_90",      # 11
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

# ----------------------------------------------------------------------------
# Panneaux de vitesse — template matching
# ----------------------------------------------------------------------------

_SPEED_VALUES = (30, 40, 50, 60, 70, 80, 90)
_SPEED_TO_FINAL = {v: 5 + i for i, v in enumerate(_SPEED_VALUES)}
_TEMPLATE_SIZE = 64
# Score de matching minimum (TM_CCOEFF_NORMED). Templates synthétiques (PIL)
# ne matchent jamais parfaitement le style CARLA → seuil bas. Si tu vois trop
# de faux positifs, monte à 0.40. Si tu vois des vrais panneaux droppés,
# baisse à 0.25.
_MIN_TEMPLATE_SCORE = 0.45

_ocr_reader = None


def _classify_sign_ocr(bgr_crop: np.ndarray, debug: bool = False) -> int | None:
    """Try reading a speed number via EasyOCR. Returns final class ID (5-11)
    or None."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    h, w = bgr_crop.shape[:2]
    if max(h, w) < 150:
        scale = 150 / max(h, w)
        bgr_crop = cv2.resize(bgr_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    results = _ocr_reader.readtext(bgr_crop, detail=1, allowlist="0123456789")
    if debug and results:
        for bbox, text, conf in results:
            print(f"  [OCR] text={text!r} conf={conf:.2f} crop={bgr_crop.shape[:2]}")
    for bbox, text, conf in results:
        text = text.strip()
        if text.isdigit():
            value = int(text)
            if value in _SPEED_TO_FINAL:
                return _SPEED_TO_FINAL[value]
    return None


def _build_speed_template(value: int) -> np.ndarray:
    """Synthétise un template carré (BGR) de la signalétique vitesse CARLA :
    cercle blanc, anneau rouge, chiffre noir centré. Pas de fichier sur disque.
    """
    size = _TEMPLATE_SIZE
    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Anneau rouge épais
    draw.ellipse((1, 1, size - 2, size - 2), outline=(220, 0, 0), width=6)
    # Chiffre noir centré
    try:
        font = ImageFont.truetype("arialbd.ttf", int(size * 0.5))
    except OSError:
        font = ImageFont.load_default()
    text = str(value)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1]),
        text,
        fill=(0, 0, 0),
        font=font,
    )
    rgb = np.array(img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


_SPEED_TEMPLATES: dict[int, list[np.ndarray]] | None = None
_TEMPLATES_DIR = Path(__file__).parent / "speed_templates"


def _get_speed_templates() -> dict[int, list[np.ndarray]]:
    """Charge N templates par valeur. Pour chaque valeur, cherche :
    - ``<value>.png``  (template unique, ex: ``30.png``)
    - ``<value>_*.png``  (templates multiples, ex: ``30_a.png``, ``30_b.png``)

    Tous ces fichiers sont chargés et utilisés en parallèle au matching :
    score final = max sur tous les templates de la valeur. Permet de couvrir
    différents angles/distances/luminosités pour la même valeur.

    Si aucun fichier n'existe pour une valeur → fallback template synthétique.
    """
    global _SPEED_TEMPLATES
    if _SPEED_TEMPLATES is None:
        _SPEED_TEMPLATES = {}
        for value in _SPEED_VALUES:
            templates: list[np.ndarray] = []
            single = _TEMPLATES_DIR / f"{value}.png"
            if single.exists():
                tpl = cv2.imread(str(single))
                if tpl is not None:
                    templates.append(tpl)
            for path in sorted(_TEMPLATES_DIR.glob(f"{value}_*.png")):
                tpl = cv2.imread(str(path))
                if tpl is not None:
                    templates.append(tpl)
            if not templates:
                templates.append(_build_speed_template(value))
            _SPEED_TEMPLATES[value] = templates
    return _SPEED_TEMPLATES


# ----------------------------------------------------------------------------
# Classifieurs
# ----------------------------------------------------------------------------


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


def classify_speed_sign(bgr_crop: np.ndarray) -> tuple[int | None, dict[int, float]]:
    """Retourne (classe finale, scores par valeur). Essaie d'abord EasyOCR,
    puis fallback template matching. None si les deux échouent."""
    if bgr_crop.size == 0 or min(bgr_crop.shape[:2]) < 25:
        return None, {v: 0.0 for v in _SPEED_VALUES}

    ocr_result = _classify_sign_ocr(bgr_crop, debug=True)
    if ocr_result is not None:
        return ocr_result, {v: 0.0 for v in _SPEED_VALUES}

    crop_h, crop_w = bgr_crop.shape[:2]
    scores: dict[int, float] = {}
    for value, templates in _get_speed_templates().items():
        best_for_value = 0.0
        for template in templates:
            if template.shape[:2] != (crop_h, crop_w):
                tpl = cv2.resize(
                    template, (crop_w, crop_h), interpolation=cv2.INTER_AREA
                )
            else:
                tpl = template
            result = cv2.matchTemplate(bgr_crop, tpl, cv2.TM_CCOEFF_NORMED)
            best_for_value = max(best_for_value, float(result.max()))
        scores[value] = best_for_value

    best_value = max(scores, key=scores.get)
    if scores[best_value] < _MIN_TEMPLATE_SCORE:
        return None, scores
    return _SPEED_TO_FINAL[best_value], scores


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------


def _crop_bbox(image: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    H, W = image.shape[:2]
    x1 = max(0, int((x - w / 2) * W))
    y1 = max(0, int((y - h / 2) * H))
    x2 = min(W, int((x + w / 2) * W))
    y2 = min(H, int((y + h / 2) * H))
    return image[y1:y2, x1:x2]


def process_run(run_dir: Path, debug_drops: bool = False) -> dict[str, int]:
    """Traverse les labels d'une run, écrit ``labels_yolo_enriched/``.

    Si ``debug_drops``, sauvegarde les crops droppés dans :
    - ``debug_dropped_tl/`` avec nom encodant les counts HSV
    - ``debug_dropped_signs/`` avec nom encodant les scores templates
    """
    labels_dir = run_dir / "labels_yolo"
    images_dir = run_dir / "images"
    out_dir = run_dir / "labels_yolo_enriched"

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Pas de dossier labels_yolo dans {run_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Pas de dossier images dans {run_dir}")
    out_dir.mkdir(exist_ok=True)

    debug_tl_dir = run_dir / "debug_dropped_tl"
    debug_sign_dir = run_dir / "debug_dropped_signs"
    if debug_drops:
        debug_tl_dir.mkdir(exist_ok=True)
        debug_sign_dir.mkdir(exist_ok=True)

    stats: dict[str, int] = {
        "frames": 0,
        "tl_in": 0,
        "tl_red": 0,
        "tl_yellow": 0,
        "tl_green": 0,
        "tl_dropped": 0,
        "sign_in": 0,
        "sign_classified": 0,
        "sign_dropped": 0,
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

            if cls == RAW_VEHICLE or cls == RAW_WALKER:
                out_lines.append(line)
                bbox_idx += 1
                continue

            crop = _crop_bbox(image, x, y, w, h)

            if cls == RAW_TRAFFIC_LIGHT:
                stats["tl_in"] += 1
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
                continue

            if cls == RAW_TRAFFIC_SIGN:
                stats["sign_in"] += 1
                print(f"  [{label_path.stem}] bbox#{bbox_idx} sign crop={crop.shape[:2]}")
                new_cls, scores = classify_speed_sign(crop)
                if new_cls is None:
                    stats["sign_dropped"] += 1
                    if debug_drops and crop.size > 0:
                        best_v = max(scores, key=scores.get)
                        name = (
                            f"{label_path.stem}_{bbox_idx}"
                            f"_best{best_v}@{scores[best_v]:.2f}.png"
                        )
                        cv2.imwrite(str(debug_sign_dir / name), crop)
                    bbox_idx += 1
                    continue
                stats["sign_classified"] += 1
                out_lines.append(f"{new_cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                bbox_idx += 1
                continue

            # Classe inconnue (raw > 3) : on garde tel quel.
            out_lines.append(line)
            bbox_idx += 1

        (out_dir / label_path.name).write_text(
            "\n".join(out_lines) + ("\n" if out_lines else "")
        )

    return stats


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrichit les labels YOLO raw 4-classes en 12-classes (couleur TL + valeur panneaux vitesse)."
    )
    parser.add_argument(
        "--run", type=Path, required=True, help="Dossier d'une run dataset"
    )
    parser.add_argument(
        "--debug-drops",
        action="store_true",
        help="Sauvegarde les crops droppés dans debug_dropped_{tl,signs}/",
    )
    args = parser.parse_args()

    if not args.run.exists():
        raise SystemExit(f"Run introuvable : {args.run}")

    stats = process_run(args.run, debug_drops=args.debug_drops)
    print(f"Run: {args.run}")
    print(f"  Frames lues:           {stats['frames']}")
    print()
    print(f"  Feux en entree:        {stats['tl_in']}")
    print(f"    -> rouges:           {stats['tl_red']}")
    print(f"    -> jaunes:           {stats['tl_yellow']}")
    print(f"    -> verts:            {stats['tl_green']}")
    print(f"    -> dropped:          {stats['tl_dropped']}")
    print()
    print(f"  Panneaux en entree:    {stats['sign_in']}")
    print(f"    -> classifies:       {stats['sign_classified']}")
    print(f"    -> dropped:          {stats['sign_dropped']}")
    print()
    print(f"Output: {args.run / 'labels_yolo_enriched'}")
    if args.debug_drops:
        print(f"Crops droppes TL:     {args.run / 'debug_dropped_tl'}")
        print(f"Crops droppes signs:  {args.run / 'debug_dropped_signs'}")


if __name__ == "__main__":
    main()
