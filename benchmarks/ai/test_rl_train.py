"""Smoke tests for rl_train — no CARLA, no GPU."""

from __future__ import annotations

from unittest.mock import Mock, call

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pytest

from src.ai.training.rl_train import make_model, train


# ---------------------------------------------------------------------------
# Minimal env — lets SB3 PPO initialize without CARLA
# ---------------------------------------------------------------------------


class _MinimalEnv(gym.Env):
    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    action_space = spaces.Box(
        low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )

    def reset(self, **kwargs):
        return np.zeros(7, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(7, dtype=np.float32), 0.0, False, False, {}


# ---------------------------------------------------------------------------
# make_model
# ---------------------------------------------------------------------------


def test_make_model_returns_ppo_instance():
    from stable_baselines3 import PPO
    model = make_model(_MinimalEnv())
    assert isinstance(model, PPO)


def test_make_model_has_mlp_policy():
    from stable_baselines3.common.policies import ActorCriticPolicy
    model = make_model(_MinimalEnv())
    assert isinstance(model.policy, ActorCriticPolicy)


def test_make_model_override_learning_rate():
    model = make_model(_MinimalEnv(), learning_rate=1e-3)
    assert model.learning_rate == pytest.approx(1e-3)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_train_calls_learn_and_save(tmp_path):
    model = Mock()
    save_path = str(tmp_path / "ppo_model")
    train(model, total_timesteps=500, save_path=save_path)
    model.learn.assert_called_once_with(total_timesteps=500)
    model.save.assert_called_once_with(save_path)
