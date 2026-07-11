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
    ent_coef=0.05,  # raised again from 0.02 -- v9 and v10 both showed the policy
    # collapsing to action-space extremes (steer/accel saturated at -1/+1, never
    # intermediate values) despite different reward functions; still not a derived
    # optimum, revisit if v11 shows the same collapse
    use_sde=True,  # generalized State-Dependent Exploration: temporally-correlated
    # exploration noise instead of independent-per-step Gaussian noise, SB3's
    # standard mechanism for smoother continuous-control exploration -- targets
    # the same collapse-to-extremes symptom directly (see JOURNAL.md for the
    # v9/v10 comparison that motivated this)
    sde_sample_freq=4,  # resample exploration noise every 4 steps rather than
    # once per rollout (SB3 default -1) -- a common starting value for
    # continuous-control tasks, not a derived optimum
    policy_kwargs=dict(
        net_arch=[128, 128]
    ),  # bigger obs space and task need more capacity than [64, 64]
    seed=42,
    verbose=1,
    device="cpu",  # MlpPolicy trains faster on CPU than GPU
)


def make_model(env: gym.Env, **override) -> PPO:
    return PPO("MlpPolicy", env, **{**_PPO_DEFAULTS, **override})


def train(model: BaseAlgorithm, total_timesteps: int, save_path: str) -> None:
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
