"""Smoke tests for rl_demo — no CARLA, no GPU."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from pathlib import Path

from src.ai.inference.rl_demo import load_model, run_episode, _add_hud, record_episode

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
    return env


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


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


def test_load_model_calls_ppo_load(tmp_path, monkeypatch):
    mock_load = Mock(return_value=Mock())
    monkeypatch.setattr("src.ai.inference.rl_demo.PPO.load", mock_load)
    path = str(tmp_path / "ppo_model")
    load_model(path)
    mock_load.assert_called_once_with(path)
