"""PPO training loop for Phase 1 RL — Stable-Baselines3."""

from __future__ import annotations

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm


def _linear_lr_schedule(initial_value: float):
    """SB3 learning-rate schedule: linear decay from initial_value to 0."""

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


_PPO_DEFAULTS: dict = dict(
    learning_rate=_linear_lr_schedule(3e-4),
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,  # high enough that exploration does not collapse to the
    # action-space extremes (steer/accel stuck at -1/+1)
    use_sde=True,  # generalized State-Dependent Exploration: temporally-correlated
    # noise instead of per-step Gaussian noise, smoother for continuous control
    sde_sample_freq=4,  # resample exploration noise every 4 steps rather than
    # once per rollout
    policy_kwargs=dict(
        # bigger obs space and task need more capacity than [64, 64]
        net_arch=[128, 128],
        squash_output=True,  # tanh-squash the action mean: with plain clipping
        # it can drift past the bounds where the gradient is zero and the
        # saturation self-sustains. Only supported by SB3 together with gSDE.
    ),
    seed=7,  # fixed for reproducibility (environment-level seeding stays at 42)
    verbose=1,
    device="cpu",  # MlpPolicy trains faster on CPU than GPU
)


def make_model(env: gym.Env, **override) -> PPO:
    return PPO("MlpPolicy", env, **{**_PPO_DEFAULTS, **override})


def train(model: BaseAlgorithm, total_timesteps: int, save_path: str) -> None:
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
