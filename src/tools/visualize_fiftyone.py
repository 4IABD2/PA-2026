from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fiftyone as fo

from src.dataset.labeling.enrich_labels import FINAL_CLASSES

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


def _parse_yolo_line(line: str, class_names: list[str]) -> fo.Detection | None:
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


def discover_runs(paths: list[Path]) -> list[Path]:
    runs: list[Path] = []
    for path in paths:
        if not path.is_dir():
            sys.exit(f"Run directory not found: {path}")
        if (path / "images").is_dir():
            runs.append(path)
            continue
        children = [c for c in sorted(path.iterdir()) if (c / "images").is_dir()]
        if not children:
            sys.exit(f"Aucune run (dossier avec images/) trouvée sous {path}")
        runs.extend(children)
    return runs


def _samples_for_run(run_dir: Path, labels_dir_name: str, class_names: list[str]):
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
        description="Visualize one or more CARLA dataset runs with FiftyOne"
    )
    parser.add_argument(
        "--run",
        required=True,
        nargs="+",
        help="Une ou plusieurs runs, ou un dossier de session (toutes ses runs)",
    )
    parser.add_argument(
        "--labels",
        default="labels_yolo_enriched",
        help="Sous-dossier de labels (defaut: labels_yolo_enriched)",
    )
    parser.add_argument(
        "--port", type=int, default=5151, help="Port de l'app FiftyOne (defaut: 5151)"
    )
    args = parser.parse_args()

    run_dirs = discover_runs([Path(p) for p in args.run])
    print(f"{len(run_dirs)} run(s) à charger.")

    dataset = build_dataset(run_dirs, args.labels)
    print(
        f"Chargé {len(dataset)} samples "
        f"({dataset.count('ground_truth.detections')} détections) "
        f"depuis {len(run_dirs)} run(s)."
    )

    session = fo.launch_app(dataset, port=args.port)
    print(f"FiftyOne app: http://localhost:{args.port}")
    print("Astuce : filtre par le champ 'run' pour isoler une map/météo.")
    session.wait()


if __name__ == "__main__":
    main()
