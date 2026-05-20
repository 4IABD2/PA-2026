"""Visualise une run CARLA dans FiftyOne (images + labels YOLO superposes).

Usage:
    uv run -m src.tools.visualize_fiftyone --run data/runs/<run_id>
    uv run -m src.tools.visualize_fiftyone --run data/runs/<run_id> --labels labels_yolo

Par defaut affiche les labels post-process (12 classes, dossier
``labels_yolo_color/`` produit par ``enrich_labels``). Avec
``--labels labels_yolo`` on visualise les labels bruts 4 classes sortis du
collector.

Au lancement : ouvre un onglet navigateur sur http://localhost:5151 avec une
grille des frames et les bboxes superposes (filtrables par classe).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fiftyone as fo

_CLASSES_RAW = ["vehicle", "walker", "traffic_light", "traffic_sign"]
_CLASSES_COLOR = [
    "vehicle",
    "walker",
    "red_light",
    "yellow_light",
    "green_light",
    "speed_30",
    "speed_40",
    "speed_50",
    "speed_60",
    "speed_70",
    "speed_80",
    "speed_90",
]


def _line_to_detection(line: str, classes: list[str]) -> fo.Detection | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls_id = int(parts[0])
    if cls_id < 0 or cls_id >= len(classes):
        return None
    cx, cy, w, h = (float(p) for p in parts[1:])
    return fo.Detection(
        label=classes[cls_id],
        bounding_box=[cx - w / 2.0, cy - h / 2.0, w, h],
    )


def build_dataset(run_dir: Path, labels_subdir: str) -> fo.Dataset:
    images_dir = run_dir / "images"
    labels_dir = run_dir / labels_subdir
    if not images_dir.is_dir():
        raise SystemExit(f"Dossier images introuvable : {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"Dossier labels introuvable : {labels_dir}")

    classes = _CLASSES_COLOR if labels_subdir == "labels_yolo_color" else _CLASSES_RAW
    name = f"{run_dir.name}__{labels_subdir}"

    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    dataset = fo.Dataset(name)
    dataset.default_classes = classes

    samples = []
    for image_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{image_path.stem}.txt"
        detections: list[fo.Detection] = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if line.strip():
                    det = _line_to_detection(line, classes)
                    if det is not None:
                        detections.append(det)
        sample = fo.Sample(filepath=str(image_path.resolve()))
        sample["ground_truth"] = fo.Detections(detections=detections)
        samples.append(sample)

    dataset.add_samples(samples)
    dataset.persistent = False
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualiser une run CARLA dans FiftyOne"
    )
    parser.add_argument("--run", type=Path, required=True, help="Dossier de la run")
    parser.add_argument(
        "--labels",
        default="labels_yolo_color",
        choices=["labels_yolo", "labels_yolo_color"],
        help="Sous-dossier de labels (defaut : labels_yolo_color)",
    )
    parser.add_argument(
        "--port", type=int, default=5151, help="Port FiftyOne (defaut 5151)"
    )
    args = parser.parse_args()

    if not args.run.exists():
        raise SystemExit(f"Run introuvable : {args.run}")

    dataset = build_dataset(args.run, args.labels)
    print(f"Dataset cree : {dataset.name} ({len(dataset)} frames)")
    print(f"Ouverture de FiftyOne sur http://localhost:{args.port}")
    session = fo.launch_app(dataset, port=args.port)
    session.wait()


if __name__ == "__main__":
    main()
