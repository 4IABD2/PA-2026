"""Run directory management — timestamped folder for artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_run_dir(base: str = "runs", tag: str = "") -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    name = f"{timestamp}_{tag}" if tag else timestamp
    path = Path(base) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_params(run_dir: Path, params: dict) -> None:
    (run_dir / "params.json").write_text(json.dumps(params, indent=2, default=str))


def plot_reward_curve(monitor_csv: Path, output_png: Path) -> None:
    df = pd.read_csv(monitor_csv, comment="#")
    window = max(1, min(50, len(df) // 5))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["r"], alpha=0.35, color="steelblue", label="reward par épisode")
    ax.plot(
        df["r"].rolling(window).mean(),
        color="steelblue",
        linewidth=2,
        label=f"moyenne mobile ({window} ep.)",
    )
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Reward totale")
    ax.set_title("Reward au fil de l'entraînement — PPO Phase 1")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
