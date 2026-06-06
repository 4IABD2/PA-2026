"""Smoke tests for CarlaEnv — no CARLA, no GPU."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from src.ai.training.rl_env import CarlaEnv
from src.interfaces.navigation_types import HighLevelCommand, Route, Waypoint
from src.interfaces.perception_types import LanesInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(
    cmd: HighLevelCommand = HighLevelCommand.LANE_FOLLOW,
    speed_mps: float = 0.0,
    ego_yaw: float = 0.0,
    wp_yaw: float = 0.0,
    center_offset: float = 0.0,
    depth_m: float = 25.0,
    max_episode_steps: int = 10,
) -> CarlaEnv:
    world = Mock()
    ego = Mock()
    nav = Mock()
    depth_est = Mock()
    lane_det = Mock()
    camera = Mock()
    col_sensor = Mock()

    vel = Mock()
    vel.x, vel.y, vel.z = speed_mps, 0.0, 0.0
    ego.get_velocity.return_value = vel

    loc = Mock()
    loc.x, loc.y, loc.z = 0.0, 0.0, 0.0
    transform = Mock()
    transform.location = loc
    transform.rotation.yaw = ego_yaw
    ego.get_transform.return_value = transform

    wp = Mock()
    wp.transform.rotation.yaw = wp_yaw
    world.get_map.return_value.get_waypoint.return_value = wp

    spawn = Mock()
    spawn.location.x, spawn.location.y, spawn.location.z = 0.0, 0.0, 0.0
    world.get_map.return_value.get_spawn_points.return_value = [spawn, spawn]

    nav.next_command.return_value = cmd
    lane_det.detect.return_value = LanesInfo(None, None, center_offset)
    depth_est.estimate.return_value = np.full((88, 200), depth_m, dtype=np.float32)

    route = Route(waypoints=[], destination=Waypoint(0.0, 0.0, 0.0, 0.0))

    return CarlaEnv(
        world=world,
        ego_vehicle=ego,
        nav=nav,
        route=route,
        depth_estimator=depth_est,
        lane_detector=lane_det,
        camera=camera,
        collision_sensor=col_sensor,
        max_episode_steps=max_episode_steps,
    )


_ZERO_ACTION = np.array([0.0, 0.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------


def test_observation_space_shape_and_dtype():
    env = _make_env()
    assert env.observation_space.shape == (7,)
    assert env.observation_space.dtype == np.float32


def test_action_space_shape_and_bounds():
    env = _make_env()
    assert env.action_space.shape == (3,)
    np.testing.assert_array_equal(env.action_space.low, [-1.0, 0.0, 0.0])
    np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_returns_obs_and_empty_info():
    obs, info = _make_env().reset()
    assert isinstance(obs, np.ndarray)
    assert info == {}


def test_reset_obs_shape_and_dtype():
    obs, _ = _make_env().reset()
    assert obs.shape == (7,)
    assert obs.dtype == np.float32


def test_reset_obs_within_observation_space():
    env = _make_env()
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)


def test_reset_clears_collision_flag():
    env = _make_env()
    env._collision_flag = True
    env.reset()
    assert env._collision_flag is False


def test_reset_teleports_ego():
    env = _make_env()
    env.reset()
    env.ego.set_transform.assert_called()
    env.ego.set_target_velocity.assert_called()


def test_reset_does_warmup_ticks():
    env = _make_env()
    env.reset()
    assert env.world.tick.call_count >= 5


def test_reset_clears_last_image():
    env = _make_env()
    env._last_image = np.zeros((88, 200, 3), dtype=np.uint8)
    env.reset()
    assert env._last_image is None


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


def test_step_returns_five_tuple_correct_types():
    env = _make_env()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(_ZERO_ACTION)
    assert obs.shape == (7,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_collision_terminates_with_penalty():
    env = _make_env()
    env.reset()
    env._collision_flag = True
    _, reward, terminated, _, _ = env.step(_ZERO_ACTION)
    assert terminated is True
    assert reward == pytest.approx(-1.0)


def test_max_steps_truncates_episode():
    env = _make_env(max_episode_steps=3)
    env.reset()
    for _ in range(2):
        _, _, _, truncated, _ = env.step(_ZERO_ACTION)
        assert not truncated
    _, _, _, truncated, _ = env.step(_ZERO_ACTION)
    assert truncated is True


# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------


def test_obs_nav_left_one_hot():
    env = _make_env(cmd=HighLevelCommand.LEFT)
    obs, _ = env.reset()
    # obs = [speed, cmd_left, cmd_right, cmd_straight, ...]
    assert obs[1] == pytest.approx(1.0)
    assert obs[2] == pytest.approx(0.0)
    assert obs[3] == pytest.approx(0.0)


def test_obs_nav_right_one_hot():
    env = _make_env(cmd=HighLevelCommand.RIGHT)
    obs, _ = env.reset()
    assert obs[1] == pytest.approx(0.0)
    assert obs[2] == pytest.approx(1.0)
    assert obs[3] == pytest.approx(0.0)


def test_obs_nav_straight_one_hot():
    env = _make_env(cmd=HighLevelCommand.STRAIGHT)
    obs, _ = env.reset()
    assert obs[1] == pytest.approx(0.0)
    assert obs[2] == pytest.approx(0.0)
    assert obs[3] == pytest.approx(1.0)


def test_obs_speed_normalized():
    # 10 m/s = 36.0 km/h → 36 / 90 = 0.4
    env = _make_env(speed_mps=10.0)
    obs, _ = env.reset()
    assert obs[0] == pytest.approx(36.0 / 90.0, abs=1e-4)


def test_obs_depth_normalized():
    # depth=25m → 25/50 = 0.5 at index 5
    env = _make_env(depth_m=25.0)
    obs, _ = env.reset()
    assert obs[5] == pytest.approx(0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------


def test_render_returns_none_before_camera_frame():
    env = _make_env()
    env.reset()
    result = env.render()
    assert result is None


def test_render_returns_rgb_array_after_image_set():
    env = _make_env()
    env.reset()
    env._last_image = np.zeros((88, 200, 3), dtype=np.uint8)
    result = env.render()
    assert isinstance(result, np.ndarray)
    assert result.shape == (88, 200, 3)
    assert result.dtype == np.uint8


def test_obs_heading_error_zero_when_aligned():
    # same yaw for ego and waypoint → heading_error = 0
    env = _make_env(ego_yaw=45.0, wp_yaw=45.0)
    obs, _ = env.reset()
    assert obs[6] == pytest.approx(0.0, abs=1e-4)
