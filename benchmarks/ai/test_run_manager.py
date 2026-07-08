"""Smoke tests for run_manager — no CARLA, no GPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.ai.training.run_manager import make_run_dir, save_params, plot_reward_curve

# ---------------------------------------------------------------------------
# make_run_dir
# ---------------------------------------------------------------------------


def test_make_run_dir_creates_directory(tmp_path):
    run_dir = make_run_dir(base=str(tmp_path), tag="test")
    assert run_dir.exists()
    assert run_dir.is_dir()


def test_make_run_dir_name_contains_tag(tmp_path):
    run_dir = make_run_dir(base=str(tmp_path), tag="ppo_500k")
    assert "ppo_500k" in run_dir.name


def test_make_run_dir_name_contains_date(tmp_path):
    run_dir = make_run_dir(base=str(tmp_path))
    # name starts with YYYY-MM-DD
    import re

    assert re.match(r"\d{4}-\d{2}-\d{2}", run_dir.name)


def test_make_run_dir_no_tag(tmp_path):
    run_dir = make_run_dir(base=str(tmp_path))
    assert run_dir.exists()


# ---------------------------------------------------------------------------
# save_params
# ---------------------------------------------------------------------------


def test_save_params_creates_json(tmp_path):
    params = {"learning_rate": 3e-4, "n_steps": 2048, "timesteps": 500_000}
    save_params(tmp_path, params)
    assert (tmp_path / "params.json").exists()


def test_save_params_content_is_valid_json(tmp_path):
    params = {"learning_rate": 3e-4, "gamma": 0.99, "tag": "smoke"}
    save_params(tmp_path, params)
    loaded = json.loads((tmp_path / "params.json").read_text())
    assert loaded["gamma"] == pytest.approx(0.99)
    assert loaded["tag"] == "smoke"


def test_save_params_handles_non_serializable_values(tmp_path):
    def dummy_schedule(x):
        return x

    params = {"learning_rate": dummy_schedule, "gamma": 0.99}
    save_params(tmp_path, params)  # must not raise
    loaded = json.loads((tmp_path / "params.json").read_text())
    assert loaded["gamma"] == pytest.approx(0.99)
    assert isinstance(loaded["learning_rate"], str)


# ---------------------------------------------------------------------------
# plot_reward_curve
# ---------------------------------------------------------------------------


def _write_monitor_csv(path: Path, rewards: list[float]) -> None:
    """Write a minimal SB3 Monitor CSV for testing."""
    with open(path, "w") as f:
        f.write("#\n")  # header comment line
        f.write("r,l,t\n")
        for i, r in enumerate(rewards):
            f.write(f"{r},{100 + i},{float(i)}\n")


def test_plot_reward_curve_creates_png(tmp_path):
    csv_path = tmp_path / "monitor.csv"
    png_path = tmp_path / "reward_curve.png"
    _write_monitor_csv(csv_path, [float(i * 0.1) for i in range(60)])
    plot_reward_curve(csv_path, png_path)
    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_plot_reward_curve_works_with_few_episodes(tmp_path):
    csv_path = tmp_path / "monitor.csv"
    png_path = tmp_path / "reward_curve.png"
    _write_monitor_csv(csv_path, [0.5, -0.3, 1.2])
    plot_reward_curve(csv_path, png_path)
    assert png_path.exists()
