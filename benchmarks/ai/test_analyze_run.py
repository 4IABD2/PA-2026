"""Tests for scripts/analyze_run.py — no CARLA, no real training run needed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analyze_run import (
    analyze,
    _bin_training_curve,
    _by_scenario_benchmark,
    _load_monitor_csv,
    _summarize_benchmark,
    _totals,
)


def _write_monitor_csv(path: Path, header_extra: list[str], rows: list[list]) -> None:
    """Write a minimal SB3 Monitor CSV. header_extra are extra column names
    after r,l,t; each row is [r, l, *extra_values] (t is auto-filled)."""
    with open(path, "w") as f:
        f.write("#\n")
        f.write(",".join(["r", "l", "t", *header_extra]) + "\n")
        for i, row in enumerate(rows):
            r, l, *extra = row
            f.write(",".join(str(v) for v in [r, l, float(i), *extra]) + "\n")


def test_load_monitor_csv_computes_cumulative_steps(tmp_path):
    csv_path = tmp_path / "training_log.monitor.csv"
    _write_monitor_csv(csv_path, [], [[-5.0, 3000], [2.0, 4000], [1.0, 5000]])
    df = _load_monitor_csv(csv_path)
    assert list(df["cum_steps"]) == [3000, 7000, 12000]


def test_bin_training_curve_splits_by_cumulative_steps(tmp_path):
    csv_path = tmp_path / "training_log.monitor.csv"
    # cum_steps: 3000, 7000 (both in (0, 10000]), 12000 (in (10000, 20000])
    _write_monitor_csv(csv_path, [], [[-5.0, 3000], [2.0, 4000], [1.0, 5000]])
    df = _load_monitor_csv(csv_path)
    bins = _bin_training_curve(df, bin_size=10_000)
    assert len(bins) == 2
    assert bins[0]["n_episodes"] == 2
    assert bins[0]["reward_mean"] == pytest.approx((-5.0 + 2.0) / 2)
    assert bins[1]["n_episodes"] == 1
    assert bins[1]["reward_mean"] == pytest.approx(1.0)


def test_bin_training_curve_includes_reward_component_means(tmp_path):
    csv_path = tmp_path / "training_log.monitor.csv"
    _write_monitor_csv(
        csv_path,
        ["r_speed", "r_collision"],
        [[-5.0, 3000, 1.0, -5.0], [3.0, 4000, 3.0, 0.0]],
    )
    df = _load_monitor_csv(csv_path)
    bins = _bin_training_curve(df, bin_size=10_000)
    assert bins[0]["r_speed_mean"] == pytest.approx((1.0 + 3.0) / 2)
    assert bins[0]["r_collision_mean"] == pytest.approx((-5.0 + 0.0) / 2)


def test_bin_training_curve_handles_missing_component_columns(tmp_path):
    csv_path = tmp_path / "training_log.monitor.csv"
    _write_monitor_csv(csv_path, [], [[-5.0, 3000]])
    df = _load_monitor_csv(csv_path)
    bins = _bin_training_curve(df, bin_size=10_000)
    assert "r_speed_mean" not in bins[0]


def test_bin_training_curve_crashes_short_pct(tmp_path):
    csv_path = tmp_path / "training_log.monitor.csv"
    # lengths: 200 (short), 400 (not short), 100 (short) -> 2/3 short
    _write_monitor_csv(csv_path, [], [[-1.0, 200], [1.0, 400], [-1.0, 100]])
    df = _load_monitor_csv(csv_path)
    bins = _bin_training_curve(df, bin_size=10_000)
    assert bins[0]["crashes_short_pct"] == pytest.approx(2 / 3)


def test_totals_computes_crash_rate(tmp_path):
    csv_path = tmp_path / "training_log.monitor.csv"
    _write_monitor_csv(csv_path, [], [[-1.0, 200], [1.0, 400], [-1.0, 100]])
    df = _load_monitor_csv(csv_path)
    totals = _totals(df)
    assert totals["total_episodes"] == 3
    assert totals["total_steps"] == 700
    assert totals["crash_rate"] == pytest.approx(2 / 3)


def test_summarize_benchmark_counts_phase1_success():
    results = {
        "0010k": {
            "straight": {
                "success": True,
                "speed": {"mean": 20.0},
                "off_route_pct": 0.1,
                "throttle": {"mean": 0.5},
                "brake": {"mean": 0.0},
            },
            "turn_left": {
                "success": False,
                "speed": {"mean": 10.0},
                "off_route_pct": 0.3,
                "throttle": {"mean": 0.6},
                "brake": {"mean": 0.0},
            },
            "red_light": {
                "success": None,
                "speed": {"mean": 15.0},
                "off_route_pct": 0.2,
                "throttle": {"mean": 0.4},
                "brake": {"mean": 0.0},
            },
        },
    }
    summary = _summarize_benchmark(results)
    assert summary["0010k"]["p1_success"] == 1
    assert (
        summary["0010k"]["p1_total"] == 2
    )  # straight + turn_left have a success_fn (not None); red_light doesn't


def test_summarize_benchmark_computes_throttle_brake_means():
    results = {
        "best_model": {
            "a": {"throttle": {"mean": 1.0}, "brake": {"mean": 0.0}},
            "b": {"throttle": {"mean": 0.5}, "brake": {"mean": 0.2}},
        },
    }
    summary = _summarize_benchmark(results)
    assert summary["best_model"]["avg_throttle"] == pytest.approx(0.75)
    assert summary["best_model"]["avg_brake"] == pytest.approx(0.1)


def test_analyze_writes_no_benchmark_section_when_results_json_missing(tmp_path):
    csv_path = tmp_path / "training_log.monitor.csv"
    _write_monitor_csv(csv_path, [], [[-5.0, 3000]])
    data = analyze(tmp_path)
    assert data["benchmark"] == {}
    assert len(data["training_curve"]) == 1


def test_analyze_includes_benchmark_when_results_json_present(tmp_path):
    csv_path = tmp_path / "training_log.monitor.csv"
    _write_monitor_csv(csv_path, [], [[-5.0, 3000]])
    evals_dir = tmp_path / "evals"
    evals_dir.mkdir()
    (evals_dir / "results.json").write_text(
        json.dumps(
            {
                "best_model": {
                    "straight": {
                        "success": True,
                        "speed": {"mean": 30.0},
                        "off_route_pct": 0.05,
                        "throttle": {"mean": 0.8},
                        "brake": {"mean": 0.0},
                    }
                },
            }
        )
    )
    data = analyze(tmp_path)
    assert data["benchmark"]["best_model"]["p1_success"] == 1


def test_totals_computes_reward_length_correlation(tmp_path):
    csv_path = tmp_path / "training_log.monitor.csv"
    # length and reward move in opposite directions -> strong negative correlation
    _write_monitor_csv(
        csv_path, [], [[-100.0, 1000], [-50.0, 500], [-10.0, 100], [-1.0, 10]]
    )
    df = _load_monitor_csv(csv_path)
    totals = _totals(df)
    assert totals["reward_length_correlation"] < -0.9


def test_by_scenario_benchmark_flags_scenario_dead_in_every_checkpoint():
    results = {
        "0012k": {"straight": {"steps": 1}, "curve_left": {"steps": 187}},
        "0024k": {"straight": {"steps": 1}, "curve_left": {"steps": 210}},
        "0036k": {"straight": {"steps": 2}, "curve_left": {"steps": 5}},
    }
    by_scenario = _by_scenario_benchmark(results)
    assert by_scenario["straight"]["likely_broken_spawn"] is True
    assert by_scenario["straight"]["steps_by_checkpoint"] == {
        "0012k": 1,
        "0024k": 1,
        "0036k": 2,
    }
    assert by_scenario["curve_left"]["likely_broken_spawn"] is False


def test_by_scenario_benchmark_requires_at_least_two_checkpoints():
    results = {
        "0012k": {"straight": {"steps": 1}},
    }
    by_scenario = _by_scenario_benchmark(results)
    assert by_scenario["straight"]["likely_broken_spawn"] is False


def test_analyze_includes_benchmark_by_scenario_when_results_json_present(tmp_path):
    csv_path = tmp_path / "training_log.monitor.csv"
    _write_monitor_csv(csv_path, [], [[-5.0, 3000]])
    evals_dir = tmp_path / "evals"
    evals_dir.mkdir()
    (evals_dir / "results.json").write_text(
        json.dumps(
            {
                "0012k": {"straight": {"steps": 1}},
                "0024k": {"straight": {"steps": 1}},
            }
        )
    )
    data = analyze(tmp_path)
    assert data["benchmark_by_scenario"]["straight"]["likely_broken_spawn"] is True


def test_analyze_writes_no_benchmark_by_scenario_when_results_json_missing(tmp_path):
    csv_path = tmp_path / "training_log.monitor.csv"
    _write_monitor_csv(csv_path, [], [[-5.0, 3000]])
    data = analyze(tmp_path)
    assert data["benchmark_by_scenario"] == {}
