"""Unit tests for MatplotVisualizer -- confirms the figure-leak fix (no CARLA)."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt

from src.tools.matplot_visualizer import MatplotVisualizer


def _fake_waypoint(x: float, y: float):
    return SimpleNamespace(
        transform=SimpleNamespace(location=SimpleNamespace(x=x, y=y))
    )


def test_plot_plan_closes_its_figure(tmp_path, monkeypatch):
    """plot_plan() must not leave an open figure behind after saving."""
    monkeypatch.chdir(tmp_path)
    route = [_fake_waypoint(0.0, 0.0), _fake_waypoint(1.0, 1.0)]

    open_before = len(plt.get_fignums())
    MatplotVisualizer.plot_plan(route)
    open_after = len(plt.get_fignums())

    assert open_after == open_before


def test_plot_road_network_closes_its_figure(tmp_path, monkeypatch):
    """plot_road_network() must not leave an open figure behind after saving."""
    monkeypatch.chdir(tmp_path)
    graph = {
        "wp1": {"waypoint": _fake_waypoint(0.0, 0.0)},
        "wp2": {"waypoint": _fake_waypoint(2.0, 3.0)},
    }

    open_before = len(plt.get_fignums())
    MatplotVisualizer.plot_road_network(graph)
    open_after = len(plt.get_fignums())

    assert open_after == open_before


def test_plot_plan_called_many_times_does_not_accumulate_figures(tmp_path, monkeypatch):
    """The actual regression this fix targets: repeated calls (as happens once
    per RL episode reset) must not accumulate open figures over time."""
    monkeypatch.chdir(tmp_path)
    route = [_fake_waypoint(0.0, 0.0), _fake_waypoint(1.0, 1.0)]

    for _ in range(10):
        MatplotVisualizer.plot_plan(route)

    assert len(plt.get_fignums()) == 0
