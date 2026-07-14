"""RL inference demo — runs a trained PPO model in CARLA."""

from __future__ import annotations

import math
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from src.interfaces.navigation_types import Route

# ---------------------------------------------------------------------------
# Scenario + Highlight dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """A fixed situation to record during a checkpoint demo or benchmark eval.

    Phase 1 scenarios are evaluated (success_fn returns bool).
    Phase 2 scenarios are recorded as-is with an "upcoming" badge.

    spawn_idx: index into world.get_map().get_spawn_points().
    success_fn: (metrics: dict) -> bool — metrics keys:
        center_offsets, speeds, rewards, terminated, steps
    setup_fn: (world, ego) -> list[Actor] — spawn NPCs/walkers, return for cleanup.
    """

    name: str
    spawn_idx: int
    max_steps: int = 300
    description: str = ""
    phase: int = 1
    expected: str = ""
    success_fn: Callable | None = None
    setup_fn: Callable | None = None
    dest_spawn_idx: int | None = (
        None  # if set, route is replanned toward this spawn after reset
    )
    target_radius: float = 15.0  # metres — reaching dest counts as success


@dataclass
class HighlightSpec:
    """Declares when to cut a highlight clip from the demo stream.

    detect(obs, action, reward, terminated, truncated) -> bool

    Obs layout (11 scalars):
        [0]  speed_norm              = speed_kmh / 90                 ∈ [0, 1]
        [1]  cmd_left                = 1 if nav says LEFT             ∈ {0, 1}
        [2]  cmd_right               = 1 if nav says RIGHT            ∈ {0, 1}
        [3]  cmd_straight            = 1 if nav says STRAIGHT         ∈ {0, 1}
        [4]  lane_offset_norm        = lateral lane offset            ∈ [-1, 1]
        [5]  is_on_road              = 1 if lane detected             ∈ {0, 1}
        [6]  nearest_vehicle_norm    = nearest vehicle / 50m          ∈ [0, 1]
        [7]  red_light_distance_norm = nearest red light / 50m        ∈ [0, 1]
        [8]  speed_limit_norm        = current speed limit / 90       ∈ [0, 1]
        [9]  nearest_walker_norm     = nearest pedestrian / 50m       ∈ [0, 1]
        [10] nearest_stop_yield_norm = nearest stop/yield sign / 50m  ∈ [0, 1]
    """

    name: str
    detect: Callable[[np.ndarray, np.ndarray, float, bool, bool], bool]
    pre_s: float = 2.0
    post_s: float = 3.0
    cooldown_s: float = 10.0


DEFAULT_HIGHLIGHT_SPECS: list[HighlightSpec] = [
    HighlightSpec(
        name="turn_left",
        detect=lambda obs, act, r, done, trunc: bool(obs[1] > 0.5 and act[0] < -0.25),
        pre_s=2.0,
        post_s=4.0,
        cooldown_s=20.0,
    ),
    HighlightSpec(
        name="turn_right",
        detect=lambda obs, act, r, done, trunc: bool(obs[2] > 0.5 and act[0] > 0.25),
        pre_s=2.0,
        post_s=4.0,
        cooldown_s=20.0,
    ),
    HighlightSpec(
        name="near_obstacle",
        detect=lambda obs, act, r, done, trunc: bool(obs[6] < 0.15),
        pre_s=2.0,
        post_s=3.0,
        cooldown_s=10.0,
    ),
    HighlightSpec(
        name="collision",
        detect=lambda obs, act, r, done, trunc: bool(done),
        pre_s=3.0,
        post_s=1.0,
        cooldown_s=5.0,
    ),
    HighlightSpec(
        name="high_speed",
        detect=lambda obs, act, r, done, trunc: bool(obs[0] > 0.65),
        pre_s=1.0,
        post_s=3.0,
        cooldown_s=30.0,
    ),
    HighlightSpec(
        name="lane_drift",
        detect=lambda obs, act, r, done, trunc: bool(abs(obs[4]) > 0.6),
        pre_s=1.5,
        post_s=2.5,
        cooldown_s=15.0,
    ),
]


# ---------------------------------------------------------------------------
# HighlightRecorder
# ---------------------------------------------------------------------------


class HighlightRecorder:
    """Rolling frame buffer that writes MP4 clips when a HighlightSpec fires."""

    def __init__(
        self,
        specs: list[HighlightSpec],
        output_dir: Path,
        frame_size: tuple[int, int],
        fps: int = 20,
    ) -> None:
        self._specs = specs
        self._dir = output_dir
        self._fps = fps
        self._w, self._h = frame_size
        max_pre = max(int(s.pre_s * fps) + 1 for s in specs)
        self._buffer: deque[np.ndarray] = deque(maxlen=max_pre)
        self._recording: dict[str, dict] = {}
        self._cooldowns: dict[str, int] = {}
        self._counts: dict[str, int] = {}
        output_dir.mkdir(parents=True, exist_ok=True)

    def push(
        self,
        frame_bgr: np.ndarray,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        import cv2  # noqa: PLC0415

        self._buffer.append(frame_bgr)

        for name in list(self._cooldowns):
            self._cooldowns[name] -= 1
            if self._cooldowns[name] <= 0:
                del self._cooldowns[name]

        for name in list(self._recording):
            state = self._recording[name]
            state["writer"].write(frame_bgr)
            state["frames_left"] -= 1
            if state["frames_left"] <= 0:
                state["writer"].release()
                del self._recording[name]

        for spec in self._specs:
            name = spec.name
            if name in self._recording or name in self._cooldowns:
                continue
            if spec.detect(obs, action, reward, terminated, truncated):
                self._start_clip(spec, cv2)

    def _start_clip(self, spec: HighlightSpec, cv2) -> None:
        n = self._counts.get(spec.name, 0) + 1
        self._counts[spec.name] = n
        path = self._dir / f"{spec.name}_{n:03d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self._fps, (self._w, self._h))
        pre_frames = int(spec.pre_s * self._fps)
        buf = list(self._buffer)
        for f in buf[max(0, len(buf) - pre_frames) :]:
            writer.write(f)
        self._recording[spec.name] = {
            "writer": writer,
            "frames_left": int(spec.post_s * self._fps),
        }
        self._cooldowns[spec.name] = int(spec.cooldown_s * self._fps)

    def close(self) -> None:
        for state in self._recording.values():
            state["writer"].release()
        self._recording.clear()

    def summary(self) -> dict[str, int]:
        return dict(self._counts)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def load_model(path: str) -> PPO:
    return PPO.load(path)


def run_episode(
    model: BaseAlgorithm,
    env: gym.Env,
    max_steps: int = 1000,
) -> tuple[float, int, str]:
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
    from PIL import Image, ImageDraw  # noqa: PLC0415

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    h, w = frame.shape[:2]

    params = info.get("params", {})
    param_str = " | ".join(f"{k}={v}" for k, v in params.items())
    draw.rectangle([0, 0, w, 14], fill=(20, 20, 20))
    draw.text((4, 2), f"PPO  {param_str}", fill=(200, 200, 200))

    ep = info.get("episode", 0)
    step = info.get("step", 0)
    max_s = info.get("max_steps", "?")
    reward = info.get("reward", 0.0)
    total_r = info.get("total_reward", 0.0)
    speed = info.get("speed_kmh", 0.0)
    action = info.get("action", [0.0, 0.0])

    draw.rectangle([0, h - 20, w, h], fill=(20, 20, 20))
    left = f"Ep {ep} | Step {step}/{max_s}   {speed:.1f} km/h"
    throttle = max(action[1], 0.0)
    brake = max(-action[1], 0.0)
    right = f"r={reward:+.2f} S={total_r:+.1f}  S={action[0]:+.2f} T={throttle:.2f} B={brake:.2f}"
    draw.text((4, h - 17), left, fill=(200, 200, 200))
    draw.text((w // 2, h - 17), right, fill=(200, 200, 200))

    return np.array(img)


_EVAL_CLEAR_RADIUS_M = (
    20.0  # ambient NPCs closer than this to a scenario's spawn are relocated
)

_CLEAR_SPAWN_POOL_SIZE = 5  # spread relocated NPCs across this many far spawns
# instead of piling them all onto the single farthest one (avoids a physics
# pile-up when several NPCs are relocated at once)


def _clear_spawn_area(env, radius_m: float) -> None:
    """Relocate ambient NPC vehicles/pedestrians near the ego's spawn point.

    The 18 NPC vehicles + 6 pedestrians spawned once for the whole training+eval
    session (run_rl_training.py::main()) roam continuously and are still present
    during every benchmark scenario. A scenario with no setup_fn of its own can
    otherwise "collide" with one of them by pure luck, unrelated to policy
    quality. Relocation targets are the N farthest available map spawn points,
    cycled deterministically across relocated actors -- not a random draw, so
    repeated evals of the same scenario stay reproducible.
    """
    try:
        actors = env.world.get_actors()
        nearby = [
            a
            for a in list(actors.filter("vehicle.*"))
            + list(actors.filter("walker.pedestrian.*"))
            if a.id != env.ego.id
        ]
        if not nearby:
            return
        spawn_points = env.world.get_map().get_spawn_points()
        if not spawn_points:
            return
        ego_loc = env.ego.get_transform().location
        farthest_n = sorted(
            spawn_points, key=lambda sp: ego_loc.distance(sp.location), reverse=True
        )[:_CLEAR_SPAWN_POOL_SIZE]
        to_relocate = [
            a for a in nearby if ego_loc.distance(a.get_location()) < radius_m
        ]
        for i, actor in enumerate(to_relocate):
            actor.set_transform(farthest_n[i % len(farthest_n)])
        for _ in range(10):
            env.world.tick()
    except Exception as exc:
        print(f"    clear_spawn_area failed: {exc}")


# ---------------------------------------------------------------------------
# Visual cards for eval_model()
# ---------------------------------------------------------------------------


def _ascii(s: str) -> str:
    """Strip accented characters — cv2.putText only handles ASCII."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def _draw_title_card(sc: Scenario, w: int, h: int, cv2) -> np.ndarray:
    """Black intro card for a scenario (BGR)."""
    card = np.full((h, w, 3), (12, 12, 16), dtype=np.uint8)

    # phase badge — top right
    phase_color = (80, 200, 60) if sc.phase == 1 else (0, 160, 220)  # BGR
    badge = f"Phase {sc.phase}"
    (bw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.putText(
        card, badge, (w - bw - 16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, phase_color, 2
    )

    # scenario name — centered, large
    name_text = _ascii(sc.name.upper().replace("_", " "))
    (tw, _), _ = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 2)
    cv2.putText(
        card,
        name_text,
        ((w - tw) // 2, h // 2 - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (230, 230, 230),
        2,
    )

    # description
    if sc.description:
        desc = _ascii(sc.description)
        (tw, _), _ = cv2.getTextSize(desc, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        cv2.putText(
            card,
            desc,
            ((w - tw) // 2, h // 2 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (150, 150, 150),
            1,
        )

    # expected behavior
    if sc.expected:
        exp = _ascii(f"Expected: {sc.expected}")
        (tw, _), _ = cv2.getTextSize(exp, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(
            card,
            exp,
            ((w - tw) // 2, h // 2 + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (100, 150, 100),
            1,
        )

    return card  # already BGR


def _draw_result_card(success: bool | None, w: int, h: int, cv2) -> np.ndarray:
    """Result card after a scenario (BGR)."""
    if success is None:
        bg = (15, 20, 30)
        color = (0, 160, 220)
        text = "PHASE 2 - NOT EVALUATED"
    elif success:
        bg = (10, 25, 10)
        color = (60, 210, 80)
        text = "SUCCESS"
    else:
        bg = (20, 10, 10)
        color = (60, 60, 210)
        text = "FAILED"

    card = np.full((h, w, 3), bg, dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
    cv2.putText(
        card,
        text,
        ((w - tw) // 2, h // 2 + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        color,
        3,
    )
    return card


def _draw_route_map_card(route: Route, w: int, h: int, cv2) -> np.ndarray:
    """Top-down schematic of the planned route: path line + start/end markers (BGR)."""
    card = np.full((h, w, 3), (12, 12, 16), dtype=np.uint8)
    waypoints = route.waypoints
    if not waypoints:
        return card

    xs = [wp.x for wp in waypoints]
    ys = [wp.y for wp in waypoints]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    margin = 60
    avail_w, avail_h = w - 2 * margin, h - 2 * margin
    scale = min(avail_w / span_x, avail_h / span_y)

    def _to_px(x: float, y: float) -> tuple[int, int]:
        px = margin + int((x - min_x) * scale)
        py = h - margin - int((y - min_y) * scale)  # flip Y for a map-like "up" feel
        return px, py

    points = np.array([_to_px(wp.x, wp.y) for wp in waypoints], dtype=np.int32)
    cv2.polylines(card, [points], isClosed=False, color=(60, 200, 230), thickness=3)

    start_px = (int(points[0][0]), int(points[0][1]))
    end_px = (int(points[-1][0]), int(points[-1][1]))
    cv2.circle(card, start_px, 8, (80, 220, 80), -1)
    cv2.circle(card, end_px, 8, (60, 60, 220), -1)
    cv2.putText(
        card,
        "A",
        (start_px[0] + 12, start_px[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (80, 220, 80),
        2,
    )
    cv2.putText(
        card,
        "B",
        (end_px[0] + 12, end_px[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (60, 60, 220),
        2,
    )

    title = "PLANNED ROUTE"
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(
        card,
        title,
        ((w - tw) // 2, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (230, 230, 230),
        2,
    )
    return card


def _draw_obs_panel(
    frame_bgr: np.ndarray,
    obs: np.ndarray,
    action: np.ndarray,
    cv2,
) -> None:
    """Draw a compact obs-space + action panel on the top-right corner (in-place, BGR).

    Only drawn when frame width > 400 (skipped on tiny debug frames).
    """
    h, w = frame_bgr.shape[:2]
    if w <= 400:
        return

    PW, PH = 215, 183
    PAD = 6
    x0 = w - PW - 6
    y0 = 18
    x1, y1 = x0 + PW, y0 + PH

    # semi-transparent dark background (78 % opacity)
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (18, 18, 22), -1)
    frame_bgr[:] = cv2.addWeighted(overlay, 0.78, frame_bgr, 0.22, 0)

    # derive nav command from one-hot encoding
    if obs[1] > 0.5:
        nav_cmd, nav_col = "LEFT", (60, 220, 80)
    elif obs[2] > 0.5:
        nav_cmd, nav_col = "RIGHT", (60, 220, 80)
    elif obs[3] > 0.5:
        nav_cmd, nav_col = "STRAIGHT", (60, 220, 80)
    else:
        nav_cmd, nav_col = "FOLLOW", (60, 180, 220)

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    SZ, TH = 0.40, 1
    LBL = (120, 120, 120)  # label colour
    VAL = (210, 210, 210)  # value colour
    HDR = (170, 130, 60)  # section header colour

    VAL_X = x0 + PAD + 88  # x position for value column

    on_road = obs[5] > 0.5

    rows = [
        # (label, value_str, value_colour)
        ("OBS SPACE", None, HDR),
        ("speed", f"{obs[0] * 90:5.1f} km/h", VAL),
        ("nav", nav_cmd, nav_col),
        ("lane offset", f"{obs[4]:+.3f}", VAL),
        (
            "on road",
            "YES" if on_road else "NO",
            (60, 220, 80) if on_road else (60, 60, 220),
        ),
        ("vehicle", f"{obs[6] * 50:5.1f} m", VAL),
        ("red light", f"{obs[7] * 50:5.1f} m", VAL),
        ("walker", f"{obs[9] * 50:5.1f} m", VAL),
        ("stop/yield", f"{obs[10] * 50:5.1f} m", VAL),
        ("ACTION", None, HDR),
        ("steer", f"{action[0]:+.3f}", VAL),
        ("throttle", f"{max(action[1], 0.0):.3f}", VAL),
        ("brake", f"{max(-action[1], 0.0):.3f}", VAL),
    ]

    y = y0 + 13
    for label, value, color in rows:
        cv2.putText(frame_bgr, label, (x0 + PAD, y), FONT, SZ, LBL, TH)
        if value is not None:
            cv2.putText(frame_bgr, value, (VAL_X, y), FONT, SZ, color, TH)
        y += 13


_MINIMAP_SIZE = 130
_MINIMAP_MARGIN = 6


def _draw_minimap(
    frame_bgr: np.ndarray,
    route: Route,
    ego_location,
    w: int,
    h: int,
    cv2,
) -> None:
    """Persistent bottom-left minimap: route path + current ego position (in-place, BGR).

    Unlike _draw_route_map_card (a one-time full-screen intro card), this is
    redrawn every frame so the ego marker tracks real progress along the route.
    Only drawn when frame width > 400 (matches _draw_obs_panel's own guard) and
    when a route with waypoints exists.
    """
    if w <= 400 or route is None or not route.waypoints:
        return

    waypoints = route.waypoints
    size = _MINIMAP_SIZE
    x0 = _MINIMAP_MARGIN
    y0 = (
        h - 20 - _MINIMAP_MARGIN - size
    )  # stay clear of _add_hud's bottom bar (rows h-20..h)

    xs = [wp.x for wp in waypoints]
    ys = [wp.y for wp in waypoints]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    pad = 10
    avail = size - 2 * pad
    scale = min(avail / span_x, avail / span_y)

    def _to_px(x: float, y: float) -> tuple[int, int]:
        px = x0 + pad + int((x - min_x) * scale)
        py = (
            y0 + size - pad - int((y - min_y) * scale)
        )  # flip Y for a map-like "up" feel
        return px, py

    # semi-transparent dark background (78 % opacity), same technique as _draw_obs_panel
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + size, y0 + size), (18, 18, 22), -1)
    frame_bgr[:] = cv2.addWeighted(overlay, 0.78, frame_bgr, 0.22, 0)

    points = np.array([_to_px(wp.x, wp.y) for wp in waypoints], dtype=np.int32)
    cv2.polylines(
        frame_bgr, [points], isClosed=False, color=(60, 200, 230), thickness=2
    )

    start_px = _to_px(waypoints[0].x, waypoints[0].y)
    cv2.circle(frame_bgr, start_px, 4, (80, 220, 80), -1)

    if route.destination is not None:
        dest_px = _to_px(route.destination.x, route.destination.y)
        cv2.circle(frame_bgr, dest_px, 4, (60, 60, 220), -1)

    ego_px = _to_px(ego_location.x, ego_location.y)
    cv2.circle(frame_bgr, ego_px, 5, (255, 255, 255), -1)
    cv2.circle(frame_bgr, ego_px, 5, (0, 0, 0), 1)


_BBOX_COLORS: dict[str, tuple[int, int, int]] = {
    "vehicle": (0, 255, 0),
    "walker": (255, 255, 0),
    "red_light": (0, 0, 255),
    "yellow_light": (0, 200, 255),
    "green_light": (0, 255, 128),
    "stop": (0, 0, 200),
    "yield": (200, 0, 200),
}
_BBOX_SPEED_COLOR = (255, 128, 0)
_BBOX_DEFAULT_COLOR = (200, 200, 200)


def _bbox_color(label: str) -> tuple[int, int, int]:
    if label.startswith("speed_"):
        return _BBOX_SPEED_COLOR
    return _BBOX_COLORS.get(label, _BBOX_DEFAULT_COLOR)


def _draw_bboxes(frame_bgr: np.ndarray, objects: list, cv2) -> None:
    """Draw perception bounding boxes + class + distance labels (in-place, BGR).

    Reimplements demo_perception_live.py's _color()/_draw() logic (Franck's
    standalone perception demo) so eval/demo videos show the same detections
    with the same color scheme, without a second perception pass -- objects
    are whatever CarlaEnv already computed this step (see CarlaEnv.last_objects).
    """
    for o in objects:
        x1, y1, x2, y2 = o.bbox
        color = _bbox_color(o.class_name.value)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        txt = f"{o.class_name.value} {o.confidence:.0%}"
        if o.distance_m is not None:
            txt += f" {o.distance_m:.0f}m"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw, y1), color, -1)
        cv2.putText(
            frame_bgr, txt, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )


def _write_summary_card(
    results: dict,
    scenarios: list[Scenario],
    writer,
    w: int,
    h: int,
    fps: int,
    cv2,
) -> None:
    """Write a 3-second summary card at the end of eval_model()."""
    card = np.full((h, w, 3), (12, 12, 16), dtype=np.uint8)

    n_phase1 = sum(1 for sc in scenarios if sc.phase == 1)
    n_success = sum(
        1 for sc in scenarios if results.get(sc.name, {}).get("success") is True
    )
    n_phase2 = sum(1 for sc in scenarios if sc.phase == 2)

    title = "FINAL RESULTS"
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 2)
    cv2.putText(
        card,
        title,
        ((w - tw) // 2, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (230, 230, 230),
        2,
    )

    p1_color = (60, 210, 80) if n_success == n_phase1 else (60, 60, 210)
    p1_text = f"Phase 1: {n_success}/{n_phase1} passed"
    (tw, _), _ = cv2.getTextSize(p1_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(
        card, p1_text, ((w - tw) // 2, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, p1_color, 2
    )

    p2_text = f"Phase 2: {n_phase2} scenario(s) pending"
    (tw, _), _ = cv2.getTextSize(p2_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.putText(
        card,
        p2_text,
        ((w - tw) // 2, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 160, 220),
        1,
    )

    # per-scenario list
    x_left = max(w // 4 - 100, 20)
    x_right = w // 2 + 20
    y = 175
    col = 0
    for sc in scenarios:
        r = results.get(sc.name, {})
        s = r.get("success")
        if s is True:
            mark, color = "OK", (60, 210, 80)
        elif s is False:
            mark, color = "NO", (60, 60, 210)
        else:
            mark, color = "--", (0, 160, 220)

        x = x_left if col == 0 else x_right
        label = _ascii(f"[{mark}] {sc.name}")
        cv2.putText(card, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        col += 1
        if col == 2:
            col = 0
            y += 28
        if y > h - 30:
            break

    for _ in range(fps * 3):
        writer.write(card)


# ---------------------------------------------------------------------------
# record_episode — unchanged public API
# ---------------------------------------------------------------------------


def record_episode(
    model: BaseAlgorithm,
    env: gym.Env,
    output_path: str,
    fps: int = 20,
    hud_params: dict | None = None,
    max_steps: int = 1000,
    n_episodes: int = 1,
    render_size: tuple[int, int] | None = None,
    render_fn: Callable[[], np.ndarray | None] | None = None,
    highlight_specs: list[HighlightSpec] | None = None,
    highlight_dir: Path | str | None = None,
    reset_seed: int | None = None,
    spawn_idx: int | None = None,
    route_map_seconds: float = 3.0,
    scenarios: list[Scenario] | None = None,
) -> None:
    """Record inference to an MP4 with HUD overlay.

    Two modes:
    - Default (scenarios=None): runs n_episodes from the same spawn (reset_seed
      for determinism; pass spawn_idx too, otherwise the random-spawn safety
      retry in CarlaEnv can still land on a different spawn point run to run).
    - Scenario mode (scenarios=[...]): runs each Scenario in sequence, each from
      its own spawn_idx, for scenario.max_steps steps — crash → next scenario.
    """
    import cv2  # noqa: PLC0415

    _get_frame = render_fn if render_fn is not None else env.render

    obs, _ = env.reset(seed=reset_seed)
    sample_frame = _get_frame()
    src_h, src_w = sample_frame.shape[:2] if sample_frame is not None else (88, 200)
    out_w, out_h = (
        render_size if (render_size and render_fn is None) else (src_w, src_h)
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    recorder: HighlightRecorder | None = None
    if highlight_specs and highlight_dir is not None:
        recorder = HighlightRecorder(
            specs=highlight_specs,
            output_dir=Path(highlight_dir),
            frame_size=(out_w, out_h),
            fps=fps,
        )

    try:
        if scenarios is not None:
            _record_scenarios(
                model,
                env,
                writer,
                recorder,
                scenarios,
                _get_frame,
                hud_params,
                out_w,
                out_h,
                cv2,
            )
        else:
            route_map_frames = int(fps * route_map_seconds)
            pause_frames = int(fps * 1.0)
            _record_episodes(
                model,
                env,
                writer,
                recorder,
                n_episodes,
                max_steps,
                reset_seed,
                spawn_idx,
                _get_frame,
                hud_params,
                cv2,
                out_w,
                out_h,
                route_map_frames,
                pause_frames,
            )
    finally:
        writer.release()
        if recorder is not None:
            recorder.close()
            counts = recorder.summary()
            if counts:
                summary = ", ".join(f"{k}×{v}" for k, v in sorted(counts.items()))
                print(f"  Highlights: {summary}")


def _record_episodes(
    model,
    env,
    writer,
    recorder,
    n_episodes,
    max_steps,
    reset_seed,
    spawn_idx,
    _get_frame,
    hud_params,
    cv2,
    out_w,
    out_h,
    route_map_frames,
    pause_frames,
) -> None:
    reset_options = {"spawn_idx": spawn_idx} if spawn_idx is not None else None
    obs, _ = env.reset(seed=reset_seed, options=reset_options)

    route = getattr(env, "route", None)
    if route is not None and route.waypoints:
        map_card = _draw_route_map_card(route, out_w, out_h, cv2)
        for _ in range(route_map_frames):
            writer.write(map_card)

    # hold on the starting position for 1 second before driving begins
    frame = _get_frame()
    if frame is not None:
        info = {
            "episode": 1,
            "step": 0,
            "max_steps": max_steps,
            "reward": 0.0,
            "total_reward": 0.0,
            "speed_kmh": float(obs[0]) * 90.0,
            "action": [0.0, 0.0],
            "params": hud_params or {},
        }
        frame_bgr = cv2.cvtColor(_add_hud(frame, info), cv2.COLOR_RGB2BGR)
        _draw_bboxes(frame_bgr, getattr(env, "last_objects", []), cv2)
        _draw_obs_panel(frame_bgr, obs, [0.0, 0.0], cv2)
        for _ in range(pause_frames):
            writer.write(frame_bgr)

    episode, total_reward, step = 0, 0.0, 0

    while episode < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        step += 1

        frame = _get_frame()
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
            _draw_bboxes(frame_bgr, getattr(env, "last_objects", []), cv2)
            _draw_obs_panel(frame_bgr, obs, action, cv2)
            writer.write(frame_bgr)
            if recorder is not None:
                recorder.push(
                    frame_bgr, obs, action, float(reward), terminated, truncated
                )

        if terminated or truncated or step >= max_steps:
            episode += 1
            if episode < n_episodes:
                obs, _ = env.reset(seed=reset_seed, options=reset_options)
                total_reward, step = 0.0, 0


def _record_scenarios(
    model,
    env,
    writer,
    recorder,
    scenarios,
    _get_frame,
    hud_params,
    out_w,
    out_h,
    cv2,
) -> None:
    for sc_idx, scenario in enumerate(scenarios):
        obs, _ = env.reset(options={"spawn_idx": scenario.spawn_idx})
        _clear_spawn_area(env, _EVAL_CLEAR_RADIUS_M)
        total_reward = 0.0

        for step in range(scenario.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

            frame = _get_frame()
            if frame is not None:
                info = {
                    "episode": sc_idx + 1,
                    "step": step + 1,
                    "max_steps": scenario.max_steps,
                    "reward": float(reward),
                    "total_reward": total_reward,
                    "speed_kmh": float(obs[0]) * 90.0,
                    "action": [float(a) for a in action],
                    "params": hud_params or {},
                }
                frame_bgr = cv2.cvtColor(_add_hud(frame, info), cv2.COLOR_RGB2BGR)
                _draw_bboxes(frame_bgr, getattr(env, "last_objects", []), cv2)

                label = f"[{sc_idx + 1}/{len(scenarios)}] {scenario.name}"
                cv2.putText(
                    frame_bgr,
                    label,
                    (4, 11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 0, 0),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    label,
                    (4, 11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (100, 220, 100),
                    1,
                )

                writer.write(frame_bgr)
                if recorder is not None:
                    recorder.push(
                        frame_bgr, obs, action, float(reward), terminated, truncated
                    )

            if terminated:
                for _ in range(min(40, scenario.max_steps - step - 1)):
                    frame = _get_frame()
                    if frame is not None:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                break


# ---------------------------------------------------------------------------
# eval_model — full benchmark pipeline
# ---------------------------------------------------------------------------


def eval_model(
    model: BaseAlgorithm,
    env: gym.Env,
    output_path: str,
    scenarios: list[Scenario] | None = None,
    fps: int = 20,
    render_fn: Callable[[], np.ndarray | None] | None = None,
) -> dict[str, dict]:
    """Run all benchmark scenarios, record to output_path, return rich metrics.

    Returns dict[scenario_name, metrics] where each metrics dict contains:
        outcome       : success, terminated, collision_step, reached_dest, steps,
                        max_dist_from_start, total_reward
        speed         : {mean, max, min, std, pct_moving}           km/h
        center_offset : {mean_abs, max_abs, std, pct_centered}      pct_centered = |offset|<0.2
        obstacle      : {mean_m, min_m}
        steer         : {mean_abs, mean, std}                        mean signed → detect L/R bias
        throttle      : {mean, std}
        brake         : {mean, max, pct_braking}
        nav_commands  : {LANE_FOLLOW, LEFT, RIGHT, STRAIGHT}        step counts
        off_route_steps, off_route_pct

        time series (per step, for plotting):
            trajectory     : [[x, y, yaw], ...]
            rewards_series, speed_series, center_series, steer_series, nav_series
            throttle_series, brake_series, obstacle_series
            off_route_series : [0/1, ...]  1 = off-route at that step
            dist_series      : distance from spawn (m) at each step
    """
    import cv2  # noqa: PLC0415

    if scenarios is None:
        from src.ai.inference.benchmark import BENCHMARK_SCENARIOS  # noqa: PLC0415

        scenarios = BENCHMARK_SCENARIOS

    _get_frame = render_fn if render_fn is not None else env.render

    # sample one frame to get dimensions
    obs, _ = env.reset()
    sample = _get_frame()
    src_h, src_w = sample.shape[:2] if sample is not None else (88, 200)
    out_w, out_h = src_w, src_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    results: dict[str, dict] = {}

    try:
        for sc_idx, sc in enumerate(scenarios):
            print(f"[{sc_idx + 1}/{len(scenarios)}] {sc.name} …")

            # 1 — title card (1 second)
            title_card = _draw_title_card(sc, out_w, out_h, cv2)
            for _ in range(fps):
                writer.write(title_card)

            # 2 — reset to scenario spawn
            obs, _ = env.reset(options={"spawn_idx": sc.spawn_idx})

            # 2a — clear ambient NPCs from the spawn area (see _clear_spawn_area docstring)
            _clear_spawn_area(env, _EVAL_CLEAR_RADIUS_M)

            # 2b — replan route to scenario-specific destination (guarantees correct nav commands)
            dest_loc = None
            if sc.dest_spawn_idx is not None:
                try:
                    spawn_pts = env.world.get_map().get_spawn_points()
                    dest_carla = spawn_pts[sc.dest_spawn_idx % len(spawn_pts)]
                    env.route = env.nav.plan(
                        env.ego.get_transform().location, dest_carla.location
                    )
                    dest_loc = dest_carla.location
                    obs = env._get_obs()  # refresh with updated nav commands
                    print(f"    route replanned to spawn {sc.dest_spawn_idx}")
                except Exception as exc:
                    print(f"    route replan failed: {exc}")

            # 3 — optional NPC/walker setup
            spawned: list = []
            if sc.setup_fn is not None:
                try:
                    spawned = sc.setup_fn(env.world, env.ego) or []
                    for _ in range(10):
                        env.world.tick()
                    obs = env._get_obs()
                except Exception as exc:
                    print(f"    setup_fn failed: {exc}")

            # 3b — hold on the starting position for 1 second before driving begins
            frame = _get_frame()
            if frame is not None:
                pause_ego_loc = env.ego.get_transform().location
                info = {
                    "episode": sc_idx + 1,
                    "step": 0,
                    "max_steps": sc.max_steps,
                    "reward": 0.0,
                    "total_reward": 0.0,
                    "speed_kmh": float(obs[0]) * 90.0,
                    "action": [0.0, 0.0],
                    "params": {},
                }
                frame_bgr = cv2.cvtColor(_add_hud(frame, info), cv2.COLOR_RGB2BGR)
                _draw_bboxes(frame_bgr, getattr(env, "last_objects", []), cv2)

                label = f"[{sc_idx + 1}/{len(scenarios)}] {sc.name}"
                label_color = (80, 200, 60) if sc.phase == 1 else (200, 140, 0)
                cv2.putText(
                    frame_bgr,
                    label,
                    (4, 11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 0, 0),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    label,
                    (4, 11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    label_color,
                    1,
                )

                _draw_obs_panel(frame_bgr, obs, [0.0, 0.0], cv2)
                _draw_minimap(frame_bgr, env.route, pause_ego_loc, out_w, out_h, cv2)
                for _ in range(fps):
                    writer.write(frame_bgr)

            # 4 — record scenario steps
            _start = env.ego.get_transform().location

            # legacy metrics dict (required by success_fn API)
            metrics: dict = {
                "center_offsets": [],
                "speeds": [],
                "rewards": [],
                "terminated": False,
                "reached_dest": False,
                "max_dist_from_start": 0.0,
                "steps": 0,
            }
            # rich per-step collectors
            trajectory: list = []  # [[x, y, yaw], ...]
            steer_series: list = []  # action[0] per step
            throttle_series: list = []
            brake_series: list = []
            obstacle_series: list = []  # metres
            nav_series: list = []  # 0=FOLLOW 1=LEFT 2=RIGHT 3=STRAIGHT
            off_route_series: list = []  # 0/1 per step (1 = off-route)
            dist_series: list = []  # distance from spawn (m) per step
            collision_step: int | None = None
            total_reward = 0.0

            # reset env's off_route counter (set in rl_env.reset, but replan may shift it)
            if hasattr(env, "off_route_count"):
                env.off_route_count = 0
            prev_off_count = 0  # for per-step off_route detection

            for step in range(sc.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)

                speed_kmh = float(obs[0]) * 90.0
                metrics["center_offsets"].append(float(obs[4]))
                metrics["speeds"].append(speed_kmh)
                metrics["rewards"].append(float(reward))
                metrics["steps"] = step + 1

                # per-step rich data
                ego_t = env.ego.get_transform()
                ego_loc = ego_t.location

                # check goal reached -- must run before recording `terminated`
                # below: env-level terminated=True fires for EITHER a collision
                # or the destination already being reached, and a genuine
                # success must not be misrecorded as a crash just because both
                # land on the same step.
                if dest_loc is not None and not metrics["reached_dest"]:
                    dist_to_dest = math.sqrt(
                        (ego_loc.x - dest_loc.x) ** 2 + (ego_loc.y - dest_loc.y) ** 2
                    )
                    if dist_to_dest < sc.target_radius:
                        metrics["reached_dest"] = True

                if (
                    terminated
                    and collision_step is None
                    and not metrics["reached_dest"]
                ):
                    metrics["terminated"] = True
                    collision_step = step + 1

                trajectory.append(
                    [
                        round(ego_loc.x, 2),
                        round(ego_loc.y, 2),
                        round(ego_t.rotation.yaw, 1),
                    ]
                )
                steer_series.append(round(float(action[0]), 4))
                throttle_series.append(round(max(float(action[1]), 0.0), 4))
                brake_series.append(round(max(-float(action[1]), 0.0), 4))
                obstacle_series.append(round(float(obs[6]) * 50.0, 2))
                # nav command encoding
                if obs[1] > 0.5:
                    nav_series.append(1)
                elif obs[2] > 0.5:
                    nav_series.append(2)
                elif obs[3] > 0.5:
                    nav_series.append(3)
                else:
                    nav_series.append(0)

                # off-route per step
                curr_off_count = getattr(env, "off_route_count", 0)
                off_route_series.append(1 if curr_off_count > prev_off_count else 0)
                prev_off_count = curr_off_count

                # distance from start
                dist_from_start = math.sqrt(
                    (ego_loc.x - _start.x) ** 2 + (ego_loc.y - _start.y) ** 2
                )
                dist_series.append(round(dist_from_start, 2))
                if dist_from_start > metrics["max_dist_from_start"]:
                    metrics["max_dist_from_start"] = dist_from_start

                frame = _get_frame()
                if frame is not None:
                    info = {
                        "episode": sc_idx + 1,
                        "step": step + 1,
                        "max_steps": sc.max_steps,
                        "reward": float(reward),
                        "total_reward": total_reward,
                        "speed_kmh": speed_kmh,
                        "action": [float(a) for a in action],
                        "params": {},
                    }
                    frame_bgr = cv2.cvtColor(_add_hud(frame, info), cv2.COLOR_RGB2BGR)
                    _draw_bboxes(frame_bgr, getattr(env, "last_objects", []), cv2)

                    # scenario label — green = Phase 1, blue = Phase 2
                    label = f"[{sc_idx + 1}/{len(scenarios)}] {sc.name}"
                    label_color = (80, 200, 60) if sc.phase == 1 else (200, 140, 0)
                    cv2.putText(
                        frame_bgr,
                        label,
                        (4, 11),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (0, 0, 0),
                        2,
                    )
                    cv2.putText(
                        frame_bgr,
                        label,
                        (4, 11),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        label_color,
                        1,
                    )

                    # obs-space + action panel (top-right)
                    _draw_obs_panel(frame_bgr, obs, action, cv2)

                    # persistent route + position minimap (bottom-left)
                    _draw_minimap(frame_bgr, env.route, ego_loc, out_w, out_h, cv2)

                    writer.write(frame_bgr)

                if metrics["reached_dest"]:
                    break

                if terminated:
                    for _ in range(min(fps, sc.max_steps - step - 1)):
                        frame = _get_frame()
                        if frame is not None:
                            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    break

            # 5 — evaluate success (Phase 1 only)
            success: bool | None = None
            if sc.phase == 1 and sc.success_fn is not None:
                if not metrics["center_offsets"]:
                    metrics["center_offsets"] = [0.0]
                try:
                    success = bool(sc.success_fn(metrics))
                except Exception:
                    success = False

            # 6 — result card (0.5 second)
            result_card = _draw_result_card(success, out_w, out_h, cv2)
            for _ in range(max(fps // 2, 1)):
                writer.write(result_card)

            # 7 — cleanup NPC actors
            for actor in spawned:
                try:
                    if actor and actor.is_alive:
                        if "controller" in actor.type_id:
                            actor.stop()
                        actor.destroy()
                except Exception:
                    pass

            # 8 — compute rich stats from collected series
            sp_arr = np.array(metrics["speeds"], dtype=np.float32)
            co_arr = np.array(metrics["center_offsets"], dtype=np.float32)
            ob_arr = np.array(obstacle_series, dtype=np.float32)
            st_arr = np.array(steer_series, dtype=np.float32)
            th_arr = np.array(throttle_series, dtype=np.float32)
            br_arr = np.array(brake_series, dtype=np.float32)
            nav_arr = np.array(nav_series, dtype=np.int8)

            def _s(a):
                return float(np.std(a)) if len(a) else 0.0

            def _m(a):
                return float(np.mean(a)) if len(a) else 0.0

            def _x(a):
                return float(np.max(a)) if len(a) else 0.0

            def _n(a):
                return float(np.min(a)) if len(a) else 0.0

            n_steps = max(metrics["steps"], 1)
            off_route_steps = getattr(env, "off_route_count", 0)

            results[sc.name] = {
                # ── outcome ──────────────────────────────────────────────
                "success": success,
                "terminated": metrics["terminated"],
                "collision_step": collision_step,
                "reached_dest": metrics["reached_dest"],
                "steps": metrics["steps"],
                "max_dist_from_start": round(metrics["max_dist_from_start"], 2),
                "total_reward": round(total_reward, 3),
                # ── speed ─────────────────────────────────────────────────
                "speed": {
                    "mean": round(_m(sp_arr), 2),
                    "max": round(_x(sp_arr), 2),
                    "min": round(_n(sp_arr), 2),
                    "std": round(_s(sp_arr), 2),
                    "pct_moving": round(float(np.mean(sp_arr > 2.0)), 3),
                },
                # ── lane keeping ──────────────────────────────────────────
                "center_offset": {
                    "mean_abs": round(float(_m(np.abs(co_arr))), 4),
                    "max_abs": round(float(_x(np.abs(co_arr))), 4),
                    "std": round(_s(co_arr), 4),
                    "pct_centered": round(float(np.mean(np.abs(co_arr) < 0.2)), 3),
                },
                # ── obstacle ──────────────────────────────────────────────
                "obstacle": {
                    "mean_m": round(_m(ob_arr), 2),
                    "min_m": round(_n(ob_arr), 2),
                },
                # ── actions ───────────────────────────────────────────────
                "steer": {
                    "mean_abs": round(float(_m(np.abs(st_arr))), 4),
                    "mean": round(_m(st_arr), 4),  # signed → detect L/R bias
                    "std": round(_s(st_arr), 4),
                },
                "throttle": {
                    "mean": round(_m(th_arr), 4),
                    "std": round(_s(th_arr), 4),  # high std → oscillation
                },
                "brake": {
                    "mean": round(_m(br_arr), 4),
                    "max": round(_x(br_arr), 4),
                    "pct_braking": round(float(np.mean(br_arr > 0.05)), 3),
                },
                # ── navigation ────────────────────────────────────────────
                "nav_commands": {
                    "LANE_FOLLOW": int(np.sum(nav_arr == 0)),
                    "LEFT": int(np.sum(nav_arr == 1)),
                    "RIGHT": int(np.sum(nav_arr == 2)),
                    "STRAIGHT": int(np.sum(nav_arr == 3)),
                },
                "off_route_steps": off_route_steps,
                "off_route_pct": round(off_route_steps / n_steps, 3),
                # ── time series (for plotting) ─────────────────────────────
                "trajectory": trajectory,
                "rewards_series": [round(r, 4) for r in metrics["rewards"]],
                "speed_series": [round(s, 2) for s in metrics["speeds"]],
                "center_series": [round(c, 4) for c in metrics["center_offsets"]],
                "steer_series": steer_series,
                "throttle_series": throttle_series,
                "brake_series": brake_series,
                "obstacle_series": obstacle_series,
                "nav_series": nav_series,
                "off_route_series": off_route_series,
                "dist_series": dist_series,
            }

            status = "✓" if success else ("✗" if success is False else "–")
            r = results[sc.name]
            print(
                f"    {status}  steps={r['steps']:3d}  "
                f"dist={r['max_dist_from_start']:5.1f}m  "
                f"spd={r['speed']['mean']:5.1f}km/h  "
                f"offset={r['center_offset']['mean_abs']:.3f}  "
                f"off_route={r['off_route_pct']:.0%}"
            )

        # 8 — summary card (3 seconds)
        _write_summary_card(results, scenarios, writer, out_w, out_h, fps, cv2)

    finally:
        writer.release()

    return results


def pick_best_checkpoint(all_results: dict[str, dict[str, dict]]) -> str:
    """Returns the all_results key (checkpoint label) with the best real
    benchmark score: highest Phase 1 success count, tie-broken by highest
    mean max_dist_from_start, then by lowest mean off_route_pct across all
    scenarios. EvalCallback's own training-time pick (3 noisy eval episodes)
    is not trustworthy on its own -- see ppo_v6.3_300k's evals/results.json
    for a real-world case where it picked the worst checkpoint by
    off_route_pct.

    Distance ranks above off-route since ppo_v12_150k: with zero successes on
    every checkpoint, the old off_route-first tie-break crowned a parked
    policy (a car that never moves is never off-route by construction, so
    0105k at 2 km/h mean speed beat checkpoints that actually drove).
    """

    def _score(scenarios: dict[str, dict]) -> tuple[int, float, float]:
        successes = sum(1 for m in scenarios.values() if m.get("success") is True)
        dists = [m.get("max_dist_from_start", 0.0) for m in scenarios.values()]
        mean_dist = sum(dists) / len(dists) if dists else 0.0
        off_route_pcts = [m["off_route_pct"] for m in scenarios.values()]
        mean_off_route = (
            sum(off_route_pcts) / len(off_route_pcts) if off_route_pcts else 1.0
        )
        return (successes, mean_dist, -mean_off_route)

    return max(all_results, key=lambda label: _score(all_results[label]))
