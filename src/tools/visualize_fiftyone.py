"""FiftyOne visualizer for CARLA dataset runs.

Usage:
    uv run -m src.tools.visualize_fiftyone --run data/runs/<RUN>
    uv run -m src.tools.visualize_fiftyone --run data/runs/<RUN> --labels labels_yolo
    uv run -m src.tools.visualize_fiftyone --run data/runs/<RUN> --port 5152
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
    "speed_50",
    "speed_60",
    "speed_70",
    "speed_80",
    "speed_90",
]

RAW_CLASSES = [
    "vehicle",
    "walker",
    "traffic_light",
    "traffic_sign",
]


def _parse_yolo_line(line: str, img_w: int, img_h: int, class_names: list[str]) -> dict | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = int(parts[0])
    if cls_id >= len(class_names):
        return None
    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    x = cx - w / 2
    y = cy - h / 2
    return fo.Detection(
        label=class_names[cls_id],
        bounding_box=[x, y, w, h],
    )


def load_run(run_dir: Path, labels_dir_name: str = "labels_yolo_enriched") -> fo.Dataset:
    images_dir = run_dir / "images"
    labels_dir = run_dir / labels_dir_name

    if not images_dir.is_dir():
        sys.exit(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        sys.exit(f"Labels directory not found: {labels_dir}")

    class_names = FINAL_CLASSES if "enriched" in labels_dir_name else RAW_CLASSES

    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        sys.exit(f"No images found in {images_dir}")

    dataset_name = f"carla_{run_dir.name}_{labels_dir_name}"
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(name=dataset_name)
    dataset.persistent = False

    samples = []
    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        sample = fo.Sample(filepath=str(img_path))

        detections = []
        if label_path.exists() and label_path.stat().st_size > 0:
            from PIL import Image

            img = Image.open(img_path)
            img_w, img_h = img.size
            for line in label_path.read_text().strip().splitlines():
                det = _parse_yolo_line(line, img_w, img_h, class_names)
                if det is not None:
                    detections.append(det)

        sample["ground_truth"] = fo.Detections(detections=detections)
        samples.append(sample)

    dataset.add_samples(samples)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a CARLA dataset run with FiftyOne")
    parser.add_argument("--run", required=True, help="Path to a run directory (e.g. data/runs/2026-05-20_town01_clearnoon)")
    parser.add_argument("--labels", default="labels_yolo_enriched", help="Labels subdirectory to use (default: labels_yolo_enriched)")
    parser.add_argument("--port", type=int, default=5151, help="FiftyOne app port (default: 5151)")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        sys.exit(f"Run directory not found: {run_dir}")

    print(f"Loading {run_dir.name} with labels from {args.labels}...")
    dataset = load_run(run_dir, args.labels)
    print(f"Loaded {len(dataset)} samples ({dataset.count('ground_truth.detections')} detections)")

    session = fo.launch_app(dataset, port=args.port)
    print(f"FiftyOne app running at http://localhost:{args.port}")
    session.wait()


if __name__ == "__main__":
    main()
