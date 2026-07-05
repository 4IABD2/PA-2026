"""Smoke tests for rl_demo — no CARLA, no GPU."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from pathlib import Path

from src.ai.inference.rl_demo import load_model, run_episode, _add_hud, record_episode, _draw_route_map_card
from src.interfaces.navigation_types import Route, Waypoint
import cv2

_OBS = np.zeros(7, dtype=np.float32)
_ACT = np.zeros(3, dtype=np.float32)


def _model(rewards=None):
    """Mock model whose predict() always returns a zero action."""
    m = Mock()
    m.predict.return_value = (_ACT, None)
    return m


def _env(*steps):
    """Mock env with fixed step side effects — each entry is (reward, terminated, truncated)."""
    e = Mock()
    e.reset.return_value = (_OBS, {})
    e.step.side_effect = [(_OBS, r, te, tr, {}) for r, te, tr in steps]
    return e


# ---------------------------------------------------------------------------
# run_episode
# ---------------------------------------------------------------------------


def test_run_episode_returns_three_tuple():
    total_reward, steps, reason = run_episode(_model(), _env((0.1, False, False)), max_steps=1)
    assert isinstance(total_reward, float)
    assert isinstance(steps, int)
    assert reason in {"collision", "truncated", "max_steps"}


def test_run_episode_reset_called_once():
    env = _env((0.1, False, False), (0.1, False, False))
    run_episode(_model(), env, max_steps=2)
    env.reset.assert_called_once()


def test_run_episode_accumulates_reward():
    env = _env((0.3, False, False), (0.5, False, False), (0.2, False, False))
    total_reward, steps, reason = run_episode(_model(), env, max_steps=3)
    assert total_reward == pytest.approx(1.0)
    assert steps == 3
    assert reason == "max_steps"


def test_run_episode_stops_on_collision():
    env = _env((0.1, False, False), (-1.0, True, False), (0.1, False, False))
    total_reward, steps, reason = run_episode(_model(), env, max_steps=10)
    assert reason == "collision"
    assert steps == 2
    assert env.step.call_count == 2


def test_run_episode_stops_on_truncated():
    env = _env((0.1, False, True))
    _, steps, reason = run_episode(_model(), env, max_steps=10)
    assert reason == "truncated"
    assert steps == 1


def test_run_episode_stops_at_max_steps():
    # env never terminates — loop must exit at max_steps
    env = Mock()
    env.reset.return_value = (_OBS, {})
    env.step.return_value = (_OBS, 0.1, False, False, {})
    _, steps, reason = run_episode(_model(), env, max_steps=5)
    assert steps == 5
    assert reason == "max_steps"
    assert env.step.call_count == 5


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _add_hud
# ---------------------------------------------------------------------------


def _hud_info(**kwargs):
    base = {
        "episode": 1, "step": 10, "max_steps": 100,
        "reward": 0.5, "total_reward": 3.2,
        "speed_kmh": 30.0, "action": [0.0, 0.3, 0.0],
        "params": {"lr": "3e-4"},
    }
    return {**base, **kwargs}


def test_add_hud_returns_same_shape():
    frame = np.zeros((88, 200, 3), dtype=np.uint8)
    result = _add_hud(frame, _hud_info())
    assert result.shape == (88, 200, 3)
    assert result.dtype == np.uint8


def test_add_hud_modifies_frame():
    frame = np.zeros((88, 200, 3), dtype=np.uint8)
    result = _add_hud(frame, _hud_info())
    # HUD must have drawn something — result is not all-black
    assert result.sum() > 0


# ---------------------------------------------------------------------------
# record_episode
# ---------------------------------------------------------------------------


def _recording_env(n_steps=3):
    env = Mock()
    env.reset.return_value = (np.zeros(7, dtype=np.float32), {})
    env.step.return_value = (np.zeros(7, dtype=np.float32), 0.1, False, False, {})
    env.render.return_value = np.zeros((88, 200, 3), dtype=np.uint8)
    # explicit None (not an auto-vivified Mock) so record_episode's
    # `getattr(env, "route", None)` route-map-card check is a no-op by default
    env.route = None
    return env


def _route(coords: list[tuple[float, float]]) -> Route:
    """Builds a Route from a list of (x, y) tuples, z and yaw fixed at 0."""
    waypoints = [Waypoint(x=x, y=y, z=0.0, yaw_deg=0.0) for x, y in coords]
    destination = waypoints[-1] if waypoints else Waypoint(0.0, 0.0, 0.0, 0.0)
    return Route(waypoints=waypoints, destination=destination)


def test_record_episode_creates_video_file(tmp_path):
    output = str(tmp_path / "demo.mp4")
    record_episode(_model(), _recording_env(), output_path=output, fps=5, max_steps=3)
    assert Path(output).exists()
    assert Path(output).stat().st_size > 0


def test_record_episode_calls_render(tmp_path):
    env = _recording_env()
    output = str(tmp_path / "demo.mp4")
    record_episode(_model(), env, output_path=output, fps=5, max_steps=3)
    assert env.render.call_count >= 1


def test_record_episode_passes_spawn_idx_as_reset_options(tmp_path):
    # Regression test: the default (non-scenario) mode must be able to pin
    # down a fixed spawn point, otherwise CarlaEnv's random-spawn safety
    # retry can land on a different spawn each run even with a fixed
    # reset_seed, breaking demo reproducibility.
    env = _recording_env()
    output = str(tmp_path / "demo.mp4")
    record_episode(_model(), env, output_path=output, fps=5, max_steps=3,
                    reset_seed=42, spawn_idx=0)
    env.reset.assert_called_with(seed=42, options={"spawn_idx": 0})


def test_record_episode_reset_options_none_without_spawn_idx(tmp_path):
    env = _recording_env()
    output = str(tmp_path / "demo.mp4")
    record_episode(_model(), env, output_path=output, fps=5, max_steps=3,
                    reset_seed=42)
    env.reset.assert_called_with(seed=42, options=None)


def test_draw_route_map_card_returns_correct_shape_and_dtype():
    route = _route([(0.0, 0.0), (10.0, 5.0), (20.0, 0.0)])
    card = _draw_route_map_card(route, 200, 100, cv2)
    assert card.shape == (100, 200, 3)
    assert card.dtype == np.uint8


def test_draw_route_map_card_draws_something():
    route = _route([(0.0, 0.0), (10.0, 5.0), (20.0, 0.0)])
    card = _draw_route_map_card(route, 200, 100, cv2)
    background = np.full((100, 200, 3), (12, 12, 16), dtype=np.uint8)
    assert not np.array_equal(card, background)


def test_draw_route_map_card_empty_route_returns_blank_card():
    route = Route(waypoints=[], destination=Waypoint(0.0, 0.0, 0.0, 0.0))
    card = _draw_route_map_card(route, 200, 100, cv2)
    background = np.full((100, 200, 3), (12, 12, 16), dtype=np.uint8)
    np.testing.assert_array_equal(card, background)


def test_record_episode_writes_route_map_card_frames(tmp_path, monkeypatch):
    written_frames = []

    class _FakeWriter:
        def __init__(self, *args, **kwargs):
            pass

        def write(self, frame):
            written_frames.append(frame)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoWriter", _FakeWriter)

    env = _recording_env()
    env.route = _route([(0.0, 0.0), (10.0, 5.0), (20.0, 0.0)])
    output = str(tmp_path / "demo.mp4")
    record_episode(_model(), env, output_path=output, fps=10, max_steps=2,
                    route_map_seconds=1.0)

    # fps(10) * route_map_seconds(1.0) = 10 map-card frames + 2 driving frames (max_steps=2)
    assert len(written_frames) == 10 + 2


def test_record_episode_no_route_attribute_skips_map_card(tmp_path, monkeypatch):
    written_frames = []

    class _FakeWriter:
        def __init__(self, *args, **kwargs):
            pass

        def write(self, frame):
            written_frames.append(frame)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoWriter", _FakeWriter)

    env = _recording_env()
    del env.route  # simulate a gym.Env with no `.route` attribute (e.g. not a CarlaEnv)
    output = str(tmp_path / "demo.mp4")
    record_episode(_model(), env, output_path=output, fps=10, max_steps=2,
                    route_map_seconds=1.0)

    assert len(written_frames) == 2  # only the 2 driving frames, no map card


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


def test_load_model_calls_ppo_load(tmp_path, monkeypatch):
    mock_load = Mock(return_value=Mock())
    monkeypatch.setattr("src.ai.inference.rl_demo.PPO.load", mock_load)
    path = str(tmp_path / "ppo_model")
    load_model(path)
    mock_load.assert_called_once_with(path)
