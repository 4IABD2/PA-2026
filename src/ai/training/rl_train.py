"""PPO training loop for Phase 1 RL — Stable-Baselines3."""

from __future__ import annotations

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

_PPO_DEFAULTS: dict = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    device="cpu",  # MlpPolicy trains faster on CPU than GPU
)


def make_model(env: gym.Env, **override) -> PPO:
    return PPO("MlpPolicy", env, **{**_PPO_DEFAULTS, **override})


def train(model: BaseAlgorithm, total_timesteps: int, save_path: str) -> None:
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
