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
    observation_space = spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)
    action_space = spaces.Box(
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
    )

    def reset(self, **kwargs):
        return np.zeros(13, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(13, dtype=np.float32), 0.0, False, False, {}


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


def test_ppo_defaults_use_larger_network():
    from src.ai.training.rl_train import _PPO_DEFAULTS

    assert _PPO_DEFAULTS["policy_kwargs"]["net_arch"] == [128, 128]


def test_ppo_defaults_have_entropy_coefficient():
    from src.ai.training.rl_train import _PPO_DEFAULTS

    assert _PPO_DEFAULTS["ent_coef"] == pytest.approx(0.05)


def test_ppo_defaults_are_seeded():
    from src.ai.training.rl_train import _PPO_DEFAULTS

    assert _PPO_DEFAULTS["seed"] == 7


def test_ppo_defaults_learning_rate_is_a_decaying_schedule():
    from src.ai.training.rl_train import _PPO_DEFAULTS

    lr_fn = _PPO_DEFAULTS["learning_rate"]
    assert callable(lr_fn)
    assert lr_fn(1.0) == pytest.approx(3e-4)
    assert lr_fn(0.0) == pytest.approx(0.0)
    assert lr_fn(0.5) == pytest.approx(1.5e-4)


def test_make_model_builds_with_new_defaults():
    model = make_model(_MinimalEnv())
    assert model.ent_coef == pytest.approx(0.05)
    assert model.seed == 7


def test_ppo_defaults_use_gsde():
    from src.ai.training.rl_train import _PPO_DEFAULTS

    assert _PPO_DEFAULTS["use_sde"] is True
    assert _PPO_DEFAULTS["sde_sample_freq"] == 4


def test_make_model_builds_with_gsde_and_predicts_within_bounds():
    model = make_model(_MinimalEnv())
    assert model.use_sde is True
    obs, _ = _MinimalEnv().reset()
    action, _ = model.predict(obs, deterministic=True)
    assert model.action_space.contains(action.astype(np.float32))


def test_ppo_defaults_squash_action_output():
    """v13: v9..v12 all collapsed deterministic steer to a saturated ±1.0 —
    the action distribution must be tanh-squashed, not merely clipped."""
    from src.ai.training.rl_train import _PPO_DEFAULTS

    assert _PPO_DEFAULTS["policy_kwargs"]["squash_output"] is True


def test_make_model_policy_squashes_output():
    model = make_model(_MinimalEnv())
    assert model.policy.squash_output is True


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_train_calls_learn_and_save(tmp_path):
    model = Mock()
    save_path = str(tmp_path / "ppo_model")
    train(model, total_timesteps=500, save_path=save_path)
    model.learn.assert_called_once_with(total_timesteps=500)
    model.save.assert_called_once_with(save_path)
