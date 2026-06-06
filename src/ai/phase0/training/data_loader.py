"""Manifest reading + tf.data pipeline for the V1 PilotNet training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from src.ai.phase0.config import (
    CROP_BOTTOM_PX,
    CROP_TOP_PX,
    IMAGE_HEIGHT,
    IMAGE_RAW_HEIGHT,
    IMAGE_WIDTH,
    SHUFFLE_BUFFER,
    SPEED_NORM_DIVISOR,
    TRAIN_VAL_RATIO,
)


def _read_manifests(run_dirs: list[Path]) -> pd.DataFrame:
    """Concat the manifest.csv of each run, prepend `run_dir` column with absolute path."""
    frames = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir).resolve()
        manifest = run_dir / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"manifest not found: {manifest}")
        df = pd.read_csv(manifest)
        df["run_dir"] = str(run_dir)
        df["abs_image_path"] = df["image_path"].apply(lambda p: str(run_dir / p))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _split_indices(
    n: int, train_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    return idx[:n_train], idx[n_train:]


def _decode_and_preprocess(
    path: tf.Tensor, speed: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """JPEG → crop → resize → /255 ; speed → /SPEED_NORM_DIVISOR."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.crop_to_bounding_box(
        img,
        offset_height=CROP_TOP_PX,
        offset_width=0,
        target_height=IMAGE_RAW_HEIGHT - CROP_TOP_PX - CROP_BOTTOM_PX,
        target_width=tf.shape(img)[1],
    )
    img = tf.image.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    img = tf.cast(img, tf.float32) / 255.0
    speed_norm = tf.cast(speed, tf.float32) / SPEED_NORM_DIVISOR
    return img, speed_norm


def _build_tf_dataset(
    paths: np.ndarray,
    speeds: np.ndarray,
    steers: np.ndarray,
    throttles: np.ndarray,
    brakes: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        {
            "abs_image_path": paths,
            "speed_kmh": speeds.astype(np.float32),
            "steer": steers.astype(np.float32).reshape(-1, 1),
            "throttle": throttles.astype(np.float32).reshape(-1, 1),
            "brake": brakes.astype(np.float32).reshape(-1, 1),
        }
    )
    if shuffle:
        ds = ds.shuffle(SHUFFLE_BUFFER, seed=seed, reshuffle_each_iteration=True)

    def _map(row):
        img, speed = _decode_and_preprocess(row["abs_image_path"], row["speed_kmh"])
        inputs = {"image": img, "speed": tf.reshape(speed, (1,))}
        targets = {
            "steer": row["steer"],
            "throttle": row["throttle"],
            "brake": row["brake"],
        }
        return inputs, targets

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def load_dataset(
    run_dirs: list[Path],
    *,
    batch_size: int,
    seed: int,
    train_ratio: float = TRAIN_VAL_RATIO,
) -> tuple[tf.data.Dataset, tf.data.Dataset, dict[str, Any]]:
    """Read manifests, drop collision rows, 80/20 split, build tf.data pipelines.

    Returns (train_ds, val_ds, info) where info contains per-split counts and the
    serializable splits dict ``{"train": [(run_dir, frame_id), ...], "val": [...]}``.
    """
    df = _read_manifests([Path(p) for p in run_dirs])
    n_raw = len(df)
    df = df[df["is_collision"] == 0].reset_index(drop=True)
    n_kept = len(df)

    train_idx, val_idx = _split_indices(n_kept, train_ratio, seed)

    def _make(idx):
        sub = df.iloc[idx]
        return _build_tf_dataset(
            paths=sub["abs_image_path"].to_numpy(),
            speeds=sub["speed_kmh"].to_numpy(),
            steers=sub["steer"].to_numpy(),
            throttles=sub["throttle"].to_numpy(),
            brakes=sub["brake"].to_numpy(),
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )

    train_ds = _make(train_idx)
    val_ds = _make(val_idx)

    splits = {
        "train": [
            [df.iloc[i]["run_dir"], int(df.iloc[i]["frame_id"])] for i in train_idx
        ],
        "val": [[df.iloc[i]["run_dir"], int(df.iloc[i]["frame_id"])] for i in val_idx],
    }
    info = {
        "n_raw": int(n_raw),
        "n_total_kept": int(n_kept),
        "n_dropped_collisions": int(n_raw - n_kept),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "splits": splits,
    }
    return train_ds, val_ds, info


def write_splits(splits: dict[str, list], path: Path) -> None:
    """Persist the splits dict to JSON for reproducibility."""
    Path(path).write_text(json.dumps(splits, indent=2))


def log_stats(
    df_or_info: dict[str, Any], steers: np.ndarray, speeds: np.ndarray
) -> None:
    """Print sanity stats: counts + 10-bin histograms for steer & speed."""
    print(
        f"[data] n_raw={df_or_info['n_raw']} kept={df_or_info['n_total_kept']} "
        f"dropped_collisions={df_or_info['n_dropped_collisions']} "
        f"train={df_or_info['n_train']} val={df_or_info['n_val']}",
        flush=True,
    )

    def _hist(name: str, arr: np.ndarray, lo: float, hi: float) -> None:
        bins = np.linspace(lo, hi, 11)
        counts, _ = np.histogram(arr, bins=bins)
        bar = "".join("█" if c > 0 else "·" for c in counts)
        print(
            f"[data] {name:7s} [{lo:+.2f}..{hi:+.2f}] {bar} (counts={counts.tolist()})"
        )

    _hist("steer", steers, -1.0, 1.0)
    _hist("speed", speeds, 0.0, 90.0)
