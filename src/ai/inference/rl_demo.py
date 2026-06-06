"""RL inference demo — runs a trained PPO model in CARLA."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm


def load_model(path: str) -> PPO:
    return PPO.load(path)


def run_episode(
    model: BaseAlgorithm,
    env: gym.Env,
    max_steps: int = 1000,
) -> tuple[float, int, str]:
    """Run one episode. Returns (total_reward, steps, reason).

    reason is 'collision', 'truncated', or 'max_steps'.
    """
    obs, _ = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        if terminated:
            return total_reward, step + 1, "collision"
        if truncated:
            return total_reward, step + 1, "truncated"

    return total_reward, max_steps, "max_steps"


def _add_hud(frame: np.ndarray, info: dict) -> np.ndarray:
    """Overlay training info on a camera frame. Returns RGB uint8 array."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    h, w = frame.shape[:2]

    # top banner — training params
    params = info.get("params", {})
    param_str = " | ".join(f"{k}={v}" for k, v in params.items())
    draw.rectangle([0, 0, w, 14], fill=(20, 20, 20))
    draw.text((4, 2), f"PPO  {param_str}", fill=(200, 200, 200))

    # bottom bar — episode / step / reward / action
    ep       = info.get("episode", 0)
    step     = info.get("step", 0)
    max_s    = info.get("max_steps", "?")
    reward   = info.get("reward", 0.0)
    total_r  = info.get("total_reward", 0.0)
    speed    = info.get("speed_kmh", 0.0)
    action   = info.get("action", [0.0, 0.0, 0.0])

    draw.rectangle([0, h - 20, w, h], fill=(20, 20, 20))
    left  = f"Ep {ep} | Step {step}/{max_s}   {speed:.1f} km/h"
    right = f"r={reward:+.2f} Σ={total_r:+.1f}  S={action[0]:+.2f} T={action[1]:.2f} B={action[2]:.2f}"
    draw.text((4, h - 17), left, fill=(200, 200, 200))
    draw.text((w // 2, h - 17), right, fill=(200, 200, 200))

    return np.array(img)


def record_episode(
    model: BaseAlgorithm,
    env: gym.Env,
    output_path: str,
    fps: int = 20,
    hud_params: dict | None = None,
    max_steps: int = 1000,
    n_episodes: int = 1,
) -> None:
    """Record n_episodes of inference to an MP4 with HUD overlay."""
    import cv2

    obs, _ = env.reset()
    sample_frame = env.render()
    h, w = sample_frame.shape[:2] if sample_frame is not None else (88, 200)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    episode = 0
    total_reward = 0.0
    step = 0

    try:
        while episode < n_episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            step += 1

            frame = env.render()
            if frame is not None:
                info = {
                    "episode": episode + 1,
                    "step": step,
                    "max_steps": max_steps,
                    "reward": float(reward),
                    "total_reward": total_reward,
                    "speed_kmh": float(obs[0]) * 90.0,
                    "action": [float(a) for a in action],
                    "params": hud_params or {},
                }
                frame_bgr = cv2.cvtColor(_add_hud(frame, info), cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)

            if terminated or truncated or step >= max_steps:
                episode += 1
                if episode < n_episodes:
                    obs, _ = env.reset()
                    total_reward = 0.0
                    step = 0
    finally:
        writer.release()
