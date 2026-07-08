"""Analyze a Phase 1 RL training run — data only, no interpretation.

Reads a run directory's training_log.monitor.csv (and, if present,
evals/results.json) and writes analysis_data.json in that same directory,
plus prints a human-readable summary. This script computes; a human (with
the assistant) still writes the ANALYSIS.md diagnosis.

Usage:
    uv run python3 scripts/analyze_run.py runs/<run_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd


def _load_monitor_csv(csv_path: Path) -> pd.DataFrame:
    # comment="#" matches the convention already used by run_manager.py's
    # plot_reward_curve for this same Monitor CSV format.
    df = pd.read_csv(csv_path, comment="#")
    df["cum_steps"] = df["l"].cumsum()
    return df


def _bin_training_curve(df: pd.DataFrame, bin_size: int = 10_000) -> list[dict]:
    max_steps = int(df["cum_steps"].iloc[-1])
    n_bins = max(1, -(-max_steps // bin_size))  # ceil division
    edges = list(range(0, n_bins * bin_size + 1, bin_size))
    df = df.copy()
    df["bin"] = pd.cut(df["cum_steps"], bins=edges)

    component_cols = [c for c in df.columns if c.startswith("r_")]
    agg: dict = {
        "reward_mean": ("r", "mean"),
        "n_episodes": ("r", "count"),
        "n_positive": ("r", lambda x: int((x > 0).sum())),
        "mean_len": ("l", "mean"),
        "crashes_short_pct": ("l", lambda x: float((x < 300).mean())),
    }
    for col in component_cols:
        agg[f"{col}_mean"] = (col, "mean")

    grouped = df.groupby("bin", observed=True).agg(**agg)
    result = []
    for interval, row in grouped.iterrows():
        entry = {"bin_end": int(interval.right)}
        row_dict = row.to_dict()
        # Convert numeric values to appropriate types
        row_dict["n_episodes"] = int(row_dict["n_episodes"])
        row_dict["n_positive"] = int(row_dict["n_positive"])
        entry.update(row_dict)
        result.append(entry)
    return result


def _totals(df: pd.DataFrame) -> dict:
    return {
        "total_episodes": int(len(df)),
        "total_steps": int(df["cum_steps"].iloc[-1]),
        "crash_rate": float((df["l"] < 300).mean()),
    }


def _summarize_benchmark(results: dict) -> dict:
    summary = {}
    for ckpt, scenarios in results.items():
        p1 = [s for s in scenarios.values() if s.get("success") is not None]
        n_ok = sum(1 for s in p1 if s["success"] is True)
        speeds = [s["speed"]["mean"] for s in scenarios.values() if "speed" in s]
        off_routes = [
            s["off_route_pct"] for s in scenarios.values() if "off_route_pct" in s
        ]
        throttles = [
            s["throttle"]["mean"] for s in scenarios.values() if "throttle" in s
        ]
        brakes = [s["brake"]["mean"] for s in scenarios.values() if "brake" in s]
        summary[ckpt] = {
            "p1_success": n_ok,
            "p1_total": len(p1),
            "avg_speed_kmh": sum(speeds) / len(speeds) if speeds else None,
            "avg_off_route_pct": (
                sum(off_routes) / len(off_routes) if off_routes else None
            ),
            "avg_throttle": sum(throttles) / len(throttles) if throttles else None,
            "avg_brake": sum(brakes) / len(brakes) if brakes else None,
        }
    return summary


def analyze(run_dir: Path) -> dict:
    df = _load_monitor_csv(run_dir / "training_log.monitor.csv")
    data: dict = {
        "training_curve": _bin_training_curve(df),
        "totals": _totals(df),
        "benchmark": {},
    }
    results_path = run_dir / "evals" / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        data["benchmark"] = _summarize_benchmark(results)
    return data


def _print_summary(data: dict) -> None:
    print("=== Training curve (10k-step bins) ===")
    for entry in data["training_curve"]:
        print(
            f"  up to {entry['bin_end']:>7,} steps: reward_mean={entry['reward_mean']:8.2f}  "
            f"n_episodes={entry['n_episodes']:4d}  n_positive={entry['n_positive']:4d}  "
            f"mean_len={entry['mean_len']:7.1f}  crashes_short={entry['crashes_short_pct']:.0%}"
        )
    print()
    if data["benchmark"]:
        print("=== Benchmark by checkpoint ===")
        for ckpt, s in data["benchmark"].items():
            speed = (
                f"{s['avg_speed_kmh']:.1f}" if s["avg_speed_kmh"] is not None else "NA"
            )
            off_route = (
                f"{s['avg_off_route_pct']:.1%}"
                if s["avg_off_route_pct"] is not None
                else "NA"
            )
            throttle = (
                f"{s['avg_throttle']:.3f}" if s["avg_throttle"] is not None else "NA"
            )
            brake = f"{s['avg_brake']:.3f}" if s["avg_brake"] is not None else "NA"
            print(
                f"  {ckpt:12s} P1={s['p1_success']}/{s['p1_total']}  speed={speed}km/h  "
                f"off_route={off_route}  throttle={throttle}  brake={brake}"
            )
        print()
    t = data["totals"]
    print(
        f"Total: {t['total_episodes']} episodes, {t['total_steps']} steps, {t['crash_rate']:.0%} crash rate"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a Phase 1 RL training run.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    data = analyze(args.run_dir)
    out_path = args.run_dir / "analysis_data.json"
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote {out_path}\n")
    _print_summary(data)


if __name__ == "__main__":
    main()
