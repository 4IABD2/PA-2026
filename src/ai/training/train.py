"""Training CLI for V1 PilotNet + speed.

Usage:
    uv run -m src.ai.training.train \
        --runs data/runs/2026-05-08_town01_clearnoon \
               data/runs/2026-05-08_town01_cloudynoon \
               data/runs/2026-05-08_town03_clearnoon \
        --output checkpoints/pilotnet_v1/ \
        --epochs 30 \
        --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.ai.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    LOSS_WEIGHTS,
    REDUCE_LR_FACTOR,
    REDUCE_LR_MIN,
    REDUCE_LR_PATIENCE,
    SEED,
)
from src.ai.models.v1_pilotnet_speed import build_pilotnet_speed
from src.ai.training.data_loader import (
    _read_manifests,
    load_dataset,
    log_stats,
    write_splits,
)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


class _OneLineLogger(keras.callbacks.Callback):
    """One stdout line per epoch (no Keras progress bar)."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", float("nan"))
        val = logs.get("val_loss", float("nan"))
        lr = float(self.model.optimizer.learning_rate.numpy())
        total = self.params.get("epochs", "?")
        print(
            f"[train] epoch {epoch + 1}/{total} — loss={loss:.4f} val_loss={val:.4f} lr={lr:.2e}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train V1 PilotNet + speed")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        type=Path,
        help="Directories of dataset runs (each contains manifest.csv)",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output checkpoint directory"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    _set_seeds(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, info = load_dataset(
        args.runs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    df_full = _read_manifests(args.runs)
    df_kept = df_full[df_full["is_collision"] == 0]
    log_stats(info, df_kept["steer"].to_numpy(), df_kept["speed_kmh"].to_numpy())

    write_splits(info["splits"], args.output / "splits.json")

    model = build_pilotnet_speed()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={"steer": "mse", "throttle": "mse", "brake": "mse"},
        loss_weights=LOSS_WEIGHTS,
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(args.output / "best.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(args.output / "last.keras"),
            save_best_only=False,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN,
        ),
        keras.callbacks.CSVLogger(str(args.output / "training_log.csv")),
        _OneLineLogger(),
    ]

    config_dump = {
        "runs": [str(p) for p in args.runs],
        "output": str(args.output),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "learning_rate": LEARNING_RATE,
        "loss_weights": LOSS_WEIGHTS,
        "info": {k: v for k, v in info.items() if k != "splits"},
    }
    (args.output / "config.json").write_text(json.dumps(config_dump, indent=2))

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=0,
    )
    print(f"[train] done. checkpoints in {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
