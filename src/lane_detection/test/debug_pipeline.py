"""Rejoue le pipeline de detection sur de VRAIES frames et dump un panneau
multi-etapes pour valider visuellement (independant de CARLA).

Usage :
    python debug_pipeline.py <image_ou_dossier> [--out debug_out]

Exemples :
    # une image
    python debug_pipeline.py ../dataset/lane_dataset/rgb/frame_000120.png
    # tout un dossier de frames
    python debug_pipeline.py ../dataset/lane_dataset/rgb --out debug_out

Pour chaque image, produit <nom>_stages.png avec 4 panneaux :
    [ original | masque binaire + ROI | fenetres glissantes | overlay final ]
"""

import argparse
import glob
import os

import cv2
import numpy as np

from utils import LaneDetector, draw_overlay, ROI_VERTICES


def _label(img, text):
    cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _mask_panel(detector, masked):
    """Masque binaire en BGR + contour de la ROI + bases d'histogramme."""
    panel = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    cv2.polylines(panel, [detector.roi], True, (0, 255, 255), 2)
    return _label(panel, "mask + ROI")


def _windows_panel(detector, masked, debug):
    """Pixels de voie colores + rectangles des fenetres glissantes."""
    panel = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    if debug is not None:
        if debug.get("left_px") is not None:
            xs, ys = debug["left_px"]
            panel[ys, xs] = (0, 0, 255)          # gauche = rouge
        if debug.get("right_px") is not None:
            xs, ys = debug["right_px"]
            panel[ys, xs] = (255, 0, 0)          # droite = bleu
        for (x_low, y_low, x_high, y_high, side) in debug.get("windows", []):
            col = (0, 255, 0)
            cv2.rectangle(panel, (x_low, y_low), (x_high, y_high), col, 1)
    return _label(panel, "sliding windows")


def process_image(detector, path, out_dir, reset_each=True):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[WARN] illisible : {path}")
        return

    # frame independante -> on oublie l'historique. En mode --sequence on le
    # garde, pour voir la persistance des pointilles comme en live.
    if reset_each:
        detector.reset()
    result = detector.detect(frame, skip=False, collect_debug=True)

    masked = result["mask"]
    original = _label(frame.copy(), "original")
    mask_panel = _mask_panel(detector, masked)
    win_panel = _windows_panel(detector, masked, result["debug"])
    overlay = _label(draw_overlay(frame, result, labels=None),
                     f"overlay [{result['status']}]")

    panel = np.hstack([original, mask_panel, win_panel, overlay])

    name = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{name}_stages.png")
    cv2.imwrite(out_path, panel)
    print(f"[OK] {result['status']:12s} -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="image .png ou dossier de frames")
    parser.add_argument("--out", default="debug_out", help="dossier de sortie")
    parser.add_argument("--limit", type=int, default=0,
                        help="si dossier : nombre max d'images (0 = toutes)")
    parser.add_argument("--sequence", action="store_true",
                        help="traite le dossier comme une video (garde l'historique entre frames)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if os.path.isdir(args.path):
        paths = sorted(glob.glob(os.path.join(args.path, "*.png")))
        if args.limit > 0:
            paths = paths[:args.limit]
    else:
        paths = [args.path]

    if not paths:
        print(f"[WARN] aucune image trouvee dans : {args.path}")
        return

    detector = LaneDetector()
    for p in paths:
        process_image(detector, p, args.out, reset_each=not args.sequence)

    print(f"[DONE] {len(paths)} image(s) -> {args.out}/")


if __name__ == "__main__":
    main()
