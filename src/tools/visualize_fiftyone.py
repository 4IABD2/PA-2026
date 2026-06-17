"""FiftyOne visualizer for CARLA dataset runs.

Accepte une ou plusieurs runs, ou un dossier de session (tous ses sous-dossiers
de runs sont chargés dans un même dataset, avec un champ ``run`` pour filtrer).

Usage:
    # Une run
    uv run -m src.tools.visualize_fiftyone --run data/runs/<SESSION>/<town_weather>
    # Toute une session de collecte (charge toutes les runs d'un coup)
    uv run -m src.tools.visualize_fiftyone --run data/runs/<SESSION>
    # Plusieurs chemins explicites
    uv run -m src.tools.visualize_fiftyone --run data/runs/<A> data/runs/<B>
    # Labels bruts au lieu des enrichis, port custom
    uv run -m src.tools.visualize_fiftyone --run data/runs/<SESSION> --labels labels_yolo --port 5152
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fiftyone as fo

FINAL_CLASSES = [
    "vehicle",
    "walker",
    "red_light",
    "yellow_light",
    "green_light",
    "speed_30",
    "speed_40",
    "speed_60",
    "speed_90",
    "stop",
    "yield",
]

# Labels bruts (labels_yolo/) : 0/1/2 = vehicle/walker/traffic_light écrits par
# le collector ; 5..10 = panneaux labellisés en ground-truth direct.
# Les index 3/4 ne sont plus émis — placeholders pour garder l'alignement.
RAW_CLASSES = [
    "vehicle",
    "walker",
    "traffic_light",
    "traffic_sign",
    "(raw4)",
    "speed_30",
    "speed_40",
    "speed_60",
    "speed_90",
    "stop",
    "yield",
]


def _parse_yolo_line(
    line: str, img_w: int, img_h: int, class_names: list[str]
) -> dict | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = int(parts[0])
    if cls_id >= len(class_names):
        return None
    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    return fo.Detection(
        label=class_names[cls_id],
        bounding_box=[cx - w / 2, cy - h / 2, w, h],
    )


def load_run(
    run_dir: Path, labels_dir_name: str = "labels_yolo_enriched"
) -> fo.Dataset:
    images_dir = run_dir / "images"
    labels_dir = run_dir / labels_dir_name
    if not labels_dir.is_dir():
        print(f"  ⚠ {run_dir.name}: pas de {labels_dir_name}/, labels ignorés")

    image_paths = sorted(images_dir.glob("*.jpg")) or sorted(images_dir.glob("*.png"))
    if not image_paths:
        print(f"  ⚠ {run_dir.name}: aucune image, run ignorée")
        return []

    samples = []
    for img_path in image_paths:
        sample = fo.Sample(filepath=str(img_path))
        sample["run"] = run_dir.name

        detections = []
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists() and label_path.stat().st_size > 0:
            for line in label_path.read_text().strip().splitlines():
                det = _parse_yolo_line(line, class_names)
                if det is not None:
                    detections.append(det)
        sample["ground_truth"] = fo.Detections(detections=detections)
        samples.append(sample)
    return samples


def build_dataset(
    run_dirs: list[Path], labels_dir_name: str = "labels_yolo_enriched"
) -> fo.Dataset:
    class_names = FINAL_CLASSES if "enriched" in labels_dir_name else RAW_CLASSES

    if len(run_dirs) == 1:
        dataset_name = f"carla_{run_dirs[0].name}_{labels_dir_name}"
    else:
        dataset_name = f"carla_{run_dirs[0].parent.name}_{labels_dir_name}"
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(name=dataset_name)
    dataset.persistent = False

    for run_dir in run_dirs:
        print(f"Chargement {run_dir.name} ({labels_dir_name})...")
        samples = _samples_for_run(run_dir, labels_dir_name, class_names)
        if samples:
            dataset.add_samples(samples)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a CARLA dataset run with FiftyOne"
    )
    parser.add_argument(
        "--run",
        required=True,
        help="Path to a run directory (e.g. data/runs/2026-05-20_town01_clearnoon)",
    )
    parser.add_argument(
        "--labels",
        default="labels_yolo_enriched",
        help="Labels subdirectory to use (default: labels_yolo_enriched)",
    )
    parser.add_argument(
        "--port", type=int, default=5151, help="FiftyOne app port (default: 5151)"
    )
    args = parser.parse_args()

    run_dirs = discover_runs([Path(p) for p in args.run])
    print(f"{len(run_dirs)} run(s) à charger.")

    print(f"Loading {run_dir.name} with labels from {args.labels}...")
    dataset = load_run(run_dir, args.labels)
    print(
        f"Loaded {len(dataset)} samples ({dataset.count('ground_truth.detections')} detections)"
    )

    session = fo.launch_app(dataset, port=args.port)
    print(f"FiftyOne app: http://localhost:{args.port}")
    print("Astuce : filtre par le champ 'run' pour isoler une map/météo.")
    session.wait()


if __name__ == "__main__":
    main()
