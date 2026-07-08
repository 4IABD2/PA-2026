"""Smoke tests for src/ai/ V1 — sans CARLA, sans GPU."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def test_pilotnet_output_shapes_and_ranges():
    """Model produces 3 named heads with correct shapes and activation ranges."""
    from src.ai.phase0.config import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS
    from src.ai.phase0.models.v1_pilotnet_speed import build_pilotnet_speed

    model = build_pilotnet_speed()

    rng = np.random.default_rng(0)
    image = rng.random((2, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.float32)
    speed = rng.random((2, 1), dtype=np.float32)

    out = model({"image": image, "speed": speed}, training=False)

    assert set(out.keys()) == {"steer", "throttle", "brake"}
    for head in ("steer", "throttle", "brake"):
        assert out[head].shape == (2, 1), f"head {head} shape {out[head].shape}"

    steer = out["steer"].numpy()
    throttle = out["throttle"].numpy()
    brake = out["brake"].numpy()
    assert np.all(steer >= -1.0) and np.all(steer <= 1.0)
    assert np.all(throttle >= 0.0) and np.all(throttle <= 1.0)
    assert np.all(brake >= 0.0) and np.all(brake <= 1.0)


def test_pilotnet_trainable():
    """Sanity: the model can fit a tiny synthetic dataset (loss decreases)."""
    from src.ai.phase0.config import (
        IMAGE_CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        LOSS_WEIGHTS,
    )
    from src.ai.phase0.models.v1_pilotnet_speed import build_pilotnet_speed

    tf.keras.utils.set_random_seed(0)
    model = build_pilotnet_speed()
    model.compile(
        optimizer="adam",
        loss={"steer": "mse", "throttle": "mse", "brake": "mse"},
        loss_weights=LOSS_WEIGHTS,
    )

    rng = np.random.default_rng(0)
    n = 8
    image = rng.random((n, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.float32)
    speed = rng.random((n, 1), dtype=np.float32)
    targets = {
        "steer": rng.uniform(-0.5, 0.5, (n, 1)).astype(np.float32),
        "throttle": rng.uniform(0.2, 0.8, (n, 1)).astype(np.float32),
        "brake": rng.uniform(0.0, 0.3, (n, 1)).astype(np.float32),
    }

    history = model.fit(
        {"image": image, "speed": speed},
        targets,
        batch_size=4,
        epochs=20,
        verbose=0,
    )

    losses = history.history["loss"]
    assert losses[-1] < losses[0] * 0.9, f"loss did not decrease meaningfully: {losses}"


def _make_synthetic_run(run_dir, frames):
    """Create a fake run directory with manifest.csv + JPEG noise images.

    `frames` is a list of dicts with keys:
        frame_id, speed_kmh, steer, throttle, brake, is_collision
    """
    import pandas as pd
    from PIL import Image

    (run_dir / "images").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    rows = []
    for f in frames:
        img_rel = f"images/{f['frame_id']:06d}.jpg"
        img_path = run_dir / img_rel
        noise = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        Image.fromarray(noise).save(img_path, quality=80)
        rows.append(
            {
                "frame_id": f["frame_id"],
                "image_path": img_rel,
                "timestamp": f["frame_id"] * 2.0,
                "command": "lane_follow",
                "speed_kmh": f["speed_kmh"],
                "steer": f["steer"],
                "throttle": f["throttle"],
                "brake": f["brake"],
                "is_collision": f["is_collision"],
                "town": "Town01",
                "weather": "ClearNoon",
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / "manifest.csv", index=False)


def test_data_loader_synthetic(tmp_path):
    """load_dataset reads a synthetic run, drops collision frames, produces correct shapes."""
    from src.ai.phase0.config import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS
    from src.ai.phase0.training.data_loader import load_dataset

    run = tmp_path / "run01"
    frames = [
        {
            "frame_id": 0,
            "speed_kmh": 10.0,
            "steer": 0.0,
            "throttle": 0.5,
            "brake": 0.0,
            "is_collision": 0,
        },
        {
            "frame_id": 1,
            "speed_kmh": 12.0,
            "steer": 0.1,
            "throttle": 0.5,
            "brake": 0.0,
            "is_collision": 0,
        },
        {
            "frame_id": 2,
            "speed_kmh": 15.0,
            "steer": -0.05,
            "throttle": 0.4,
            "brake": 0.0,
            "is_collision": 0,
        },
        {
            "frame_id": 3,
            "speed_kmh": 0.0,
            "steer": 0.3,
            "throttle": 0.0,
            "brake": 1.0,
            "is_collision": 1,
        },  # dropped
        {
            "frame_id": 4,
            "speed_kmh": 8.0,
            "steer": -0.2,
            "throttle": 0.6,
            "brake": 0.0,
            "is_collision": 0,
        },
    ]
    _make_synthetic_run(run, frames)

    train_ds, _, info = load_dataset([run], batch_size=2, seed=0)

    assert info["n_total_kept"] == 4
    assert info["n_dropped_collisions"] == 1
    assert info["n_train"] + info["n_val"] == 4

    inputs, targets = next(iter(train_ds))
    assert inputs["image"].shape[1:] == (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    assert inputs["speed"].shape[1:] == (1,)
    assert set(targets.keys()) == {"steer", "throttle", "brake"}
    assert float(inputs["image"].numpy().max()) <= 1.0
    assert float(inputs["image"].numpy().min()) >= 0.0


def test_data_loader_split_deterministic(tmp_path):
    """Same seed → same train/val split (set of (run_dir, frame_id) tuples)."""
    from src.ai.phase0.training.data_loader import load_dataset

    run = tmp_path / "run01"
    frames = [
        {
            "frame_id": i,
            "speed_kmh": 10.0 + i,
            "steer": 0.0,
            "throttle": 0.5,
            "brake": 0.0,
            "is_collision": 0,
        }
        for i in range(20)
    ]
    _make_synthetic_run(run, frames)

    _, _, info_a = load_dataset([run], batch_size=4, seed=42)
    _, _, info_b = load_dataset([run], batch_size=4, seed=42)
    _, _, info_c = load_dataset([run], batch_size=4, seed=999)

    set_a = {tuple(s) for s in info_a["splits"]["train"]}
    set_b = {tuple(s) for s in info_b["splits"]["train"]}
    set_c = {tuple(s) for s in info_c["splits"]["train"]}

    assert set_a == set_b, "Same seed should produce same split"
    assert set_a != set_c, "Different seed should produce different split"
