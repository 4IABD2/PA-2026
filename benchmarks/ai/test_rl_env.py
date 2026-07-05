"""Smoke tests for CarlaEnv — no CARLA, no GPU."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from src.ai.training.rl_env import CarlaEnv
from src.interfaces.navigation_types import HighLevelCommand, Route, Waypoint
from src.interfaces.perception_types import DetectedObject, ObjectClass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(
    cmd: HighLevelCommand = HighLevelCommand.LANE_FOLLOW,
    speed_mps: float = 0.0,
    lane_angle: float = 0.0,
    lane_offset: float = 0.0,
    on_road: bool = True,
    nearest_vehicle_m: float = 50.0,
    red_light_distance_m: float | None = None,
    nearest_walker_m: float | None = None,
    nearest_stop_yield_m: float | None = None,
    max_episode_steps: int = 10,
) -> CarlaEnv:
    world = Mock()
    ego = Mock()
    nav = Mock()
    camera = Mock()
    col_sensor = Mock()

    vel = Mock()
    vel.x, vel.y, vel.z = speed_mps, 0.0, 0.0
    ego.get_velocity.return_value = vel

    loc = Mock()
    loc.x, loc.y, loc.z = 0.0, 0.0, 0.0
    transform = Mock()
    transform.location = loc
    transform.rotation.yaw = 0.0
    ego.get_transform.return_value = transform

    wp = Mock()
    wp.transform.rotation.yaw = 0.0
    world.get_map.return_value.get_waypoint.return_value = wp

    spawn = Mock()
    spawn.location.x, spawn.location.y, spawn.location.z = 0.0, 0.0, 0.0
    world.get_map.return_value.get_spawn_points.return_value = [spawn, spawn]
    world.get_actors.return_value.filter.return_value = []  # no nearby NPCs by default

    nav.next_command.return_value = cmd
    nav.plan.return_value = Route(waypoints=[], destination=Waypoint(0.0, 0.0, 0.0, 0.0))

    # Franck's perception mock
    perception = Mock()
    objects = [DetectedObject(class_name=ObjectClass.VEHICLE, bbox=(0, 0, 10, 10), confidence=0.9, distance_m=nearest_vehicle_m)]
    if red_light_distance_m is not None:
        objects.append(DetectedObject(class_name=ObjectClass.RED_LIGHT, bbox=(100, 0, 120, 30), confidence=0.95, distance_m=red_light_distance_m))
    if nearest_walker_m is not None:
        objects.append(DetectedObject(class_name=ObjectClass.WALKER, bbox=(200, 0, 220, 60), confidence=0.9, distance_m=nearest_walker_m))
    if nearest_stop_yield_m is not None:
        objects.append(DetectedObject(class_name=ObjectClass.STOP, bbox=(300, 0, 320, 40), confidence=0.9, distance_m=nearest_stop_yield_m))
    perception.perceive.return_value = (objects, np.zeros((720, 1280), dtype=np.float32))

    # Karim's lane estimate mock
    direction = "ALIGNE" if on_road else "NONE"
    lane_estimate_fn = Mock(return_value=(direction, lane_angle, lane_offset))

    route = Route(waypoints=[], destination=Waypoint(0.0, 0.0, 0.0, 0.0))

    return CarlaEnv(
        world=world,
        ego_vehicle=ego,
        nav=nav,
        route=route,
        perception=perception,
        lane_estimate_fn=lane_estimate_fn,
        camera=camera,
        collision_sensor=col_sensor,
        max_episode_steps=max_episode_steps,
    )


def _make_spawn(distance_to_nearest: float) -> Mock:
    """A spawn-point Mock whose location reports a fixed nearest-actor distance,
    regardless of which actor is queried."""
    sp = Mock()
    sp.location.distance = Mock(return_value=distance_to_nearest)
    return sp


_ZERO_ACTION = np.array([0.0, 0.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------


def test_observation_space_shape_and_dtype():
    env = _make_env()
    assert env.observation_space.shape == (12,)
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
    assert obs.shape == (12,)
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


def test_reset_replans_route_even_without_explicit_spawn_idx():
    # Regression test: SB3 calls reset() with no options after every training
    # episode (spawn_idx=None). The route must still be replanned from the
    # new (randomly teleported) position — otherwise nav commands are computed
    # against a stale route from a completely different location.
    env = _make_env()
    env.nav.plan.reset_mock()
    env.reset()
    env.nav.plan.assert_called_once()


def test_reset_replans_route_with_explicit_spawn_idx():
    env = _make_env()
    env.nav.plan.reset_mock()
    env.reset(options={"spawn_idx": 1})
    env.nav.plan.assert_called_once()


# ---------------------------------------------------------------------------
# Safe spawn selection
# ---------------------------------------------------------------------------


def test_safe_spawn_rejects_candidate_within_min_distance():
    env = _make_env()
    unsafe = _make_spawn(distance_to_nearest=5.0)
    safe = _make_spawn(distance_to_nearest=15.0)
    env.world.get_map.return_value.get_spawn_points.return_value = [unsafe, safe]
    nearby_actor = Mock(id=999, get_location=Mock(return_value=Mock()))
    env.world.get_actors.return_value.filter.return_value = [nearby_actor]
    env.np_random = Mock(integers=Mock(side_effect=[0, 1]))

    env.reset()

    env.ego.set_transform.assert_called_with(safe)


def test_safe_spawn_accepts_first_candidate_at_exactly_min_distance():
    env = _make_env()
    safe = _make_spawn(distance_to_nearest=10.0)  # exactly the threshold
    other = _make_spawn(distance_to_nearest=20.0)
    env.world.get_map.return_value.get_spawn_points.return_value = [safe, other]
    nearby_actor = Mock(id=999, get_location=Mock(return_value=Mock()))
    env.world.get_actors.return_value.filter.return_value = [nearby_actor]
    env.np_random = Mock(integers=Mock(side_effect=[0]))

    env.reset()

    env.ego.set_transform.assert_called_with(safe)
    env.np_random.integers.assert_called_once()  # short-circuits, doesn't draw all 10


def test_safe_spawn_falls_back_to_best_attempt_when_all_unsafe():
    env = _make_env()
    worse = _make_spawn(distance_to_nearest=2.0)
    better = _make_spawn(distance_to_nearest=6.0)
    env.world.get_map.return_value.get_spawn_points.return_value = [worse, better]
    nearby_actor = Mock(id=999, get_location=Mock(return_value=Mock()))
    env.world.get_actors.return_value.filter.return_value = [nearby_actor]
    # 10 draws (all below the 10m threshold), ending on `worse` so a
    # last-drawn-wins implementation would incorrectly return `worse`.
    env.np_random = Mock(integers=Mock(side_effect=[0, 1, 0, 1, 0, 1, 0, 1, 0, 0]))

    env.reset()

    env.ego.set_transform.assert_called_with(better)


def test_safe_spawn_accepts_immediately_when_no_nearby_actors():
    env = _make_env()
    only = _make_spawn(distance_to_nearest=0.0)  # would be unsafe if actors existed
    env.world.get_map.return_value.get_spawn_points.return_value = [only]
    env.world.get_actors.return_value.filter.return_value = []  # no vehicles, no walkers
    env.np_random = Mock(integers=Mock(side_effect=[0]))

    env.reset()

    env.ego.set_transform.assert_called_with(only)


def test_safe_spawn_excludes_ego_from_nearby_actors():
    env = _make_env()
    safe = _make_spawn(distance_to_nearest=999.0)
    env.world.get_map.return_value.get_spawn_points.return_value = [safe]
    env.ego.id = 1
    ego_as_actor = Mock(id=1, get_location=Mock(return_value=Mock()))
    env.world.get_actors.return_value.filter.return_value = [ego_as_actor]
    env.np_random = Mock(integers=Mock(side_effect=[0]))

    env.reset()

    env.ego.set_transform.assert_called_with(safe)
    safe.location.distance.assert_not_called()  # the only "actor" found was the ego itself


def test_explicit_spawn_idx_skips_safety_check():
    env = _make_env()
    spawn0 = _make_spawn(distance_to_nearest=0.0)
    env.world.get_map.return_value.get_spawn_points.return_value = [spawn0]

    env.reset(options={"spawn_idx": 0})

    env.world.get_actors.assert_not_called()


def test_reset_does_warmup_ticks():
    env = _make_env()
    env.reset()
    assert env.world.tick.call_count >= 5


def test_reset_clears_last_image():
    env = _make_env()
    env._last_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    env.reset()
    assert env._last_image is None


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


def test_step_returns_five_tuple_correct_types():
    env = _make_env()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(_ZERO_ACTION)
    assert obs.shape == (12,)
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
    assert reward == pytest.approx(-5.0)


def test_collision_captures_impact_speed():
    env = _make_env(speed_mps=20.0)  # 72 km/h
    env.reset()
    env._on_collision(Mock())
    assert env._collision_speed_kmh == pytest.approx(72.0, abs=0.1)


def test_collision_speed_resets_on_reset():
    env = _make_env(speed_mps=20.0)
    env.reset()
    env._on_collision(Mock())
    env.reset()
    assert env._collision_speed_kmh == pytest.approx(0.0)


def test_red_light_violation_penalized_once_not_twice():
    env = _make_env(speed_mps=10.0, red_light_distance_m=3.0)  # 36 km/h, close red light
    env.reset()
    _, reward1, _, _, _ = env.step(_ZERO_ACTION)
    _, reward2, _, _, _ = env.step(_ZERO_ACTION)
    assert reward2 - reward1 == pytest.approx(2.0, abs=1e-3)


def test_red_light_violation_rearms_after_moving_away():
    env = _make_env(speed_mps=10.0, red_light_distance_m=3.0)
    env.reset()
    env.step(_ZERO_ACTION)
    assert env._red_light_flagged is True
    far_objects = [DetectedObject(class_name=ObjectClass.RED_LIGHT, bbox=(0, 0, 1, 1), confidence=0.9, distance_m=40.0)]
    env.perception.perceive.return_value = (far_objects, np.zeros((720, 1280), dtype=np.float32))
    env.step(_ZERO_ACTION)
    assert env._red_light_flagged is False


def test_stop_yield_violation_penalized_once_not_twice():
    env = _make_env(speed_mps=10.0, nearest_stop_yield_m=3.0)
    env.reset()
    _, reward1, _, _, _ = env.step(_ZERO_ACTION)
    _, reward2, _, _, _ = env.step(_ZERO_ACTION)
    assert reward2 - reward1 == pytest.approx(1.0, abs=1e-3)


def test_max_steps_truncates_episode():
    env = _make_env(max_episode_steps=3)
    env.reset()
    for _ in range(2):
        _, _, _, truncated, _ = env.step(_ZERO_ACTION)
        assert not truncated
    _, _, _, truncated, _ = env.step(_ZERO_ACTION)
    assert truncated is True


def test_reward_center_term_uses_lane_offset_when_centered():
    # Large heading angle but laterally centred → r_center should be at its max (0.3).
    env = _make_env(lane_angle=80.0, lane_offset=0.0)
    env.reset()
    _, reward, _, _, _ = env.step(_ZERO_ACTION)
    # speed=0 -> r_speed=0, r_center=0.3, r_alive=0.01, r_stall=-0.20 (speed < 1 km/h)
    assert reward == pytest.approx(0.11, abs=1e-4)


def test_reward_center_term_uses_lane_offset_when_off_center():
    # Small heading angle but laterally off-centre → r_center should be penalised.
    env = _make_env(lane_angle=0.0, lane_offset=0.9)
    env.reset()
    _, reward, _, _, _ = env.step(_ZERO_ACTION)
    # speed=0 -> r_speed=0, r_center=(1-0.9)*0.3=0.03, r_alive=0.01, r_stall=-0.20
    assert reward == pytest.approx(-0.16, abs=1e-4)


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
    # 10 m/s = 36.0 km/h → 36 / 90 ≈ 0.4
    env = _make_env(speed_mps=10.0)
    obs, _ = env.reset()
    assert obs[0] == pytest.approx(36.0 / 90.0, abs=1e-4)


def test_obs_lane_angle_normalized():
    # 45° → 45/90 = 0.5
    env = _make_env(lane_angle=45.0)
    obs, _ = env.reset()
    assert obs[4] == pytest.approx(0.5, abs=1e-4)


def test_obs_lane_offset_normalized():
    env = _make_env(lane_offset=0.4)
    obs, _ = env.reset()
    assert obs[5] == pytest.approx(0.4, abs=1e-4)


def test_obs_lane_offset_clipped_to_bounds():
    env = _make_env(lane_offset=1.5)
    obs, _ = env.reset()
    assert obs[5] == pytest.approx(1.0)


def test_obs_is_on_road_when_aligned():
    env = _make_env(on_road=True)
    obs, _ = env.reset()
    assert obs[6] == pytest.approx(1.0)


def test_obs_is_off_road_when_none():
    env = _make_env(on_road=False)
    obs, _ = env.reset()
    assert obs[6] == pytest.approx(0.0)


def test_obs_nearest_vehicle_normalized():
    # 25m → 25/50 = 0.5
    env = _make_env(nearest_vehicle_m=25.0)
    obs, _ = env.reset()
    assert obs[7] == pytest.approx(0.5, abs=1e-4)


def test_obs_red_light_detected():
    env = _make_env(red_light_distance_m=10.0)
    obs, _ = env.reset()
    assert obs[8] == pytest.approx(10.0 / 50.0, abs=1e-4)


def test_obs_no_red_light():
    env = _make_env(red_light_distance_m=None)
    obs, _ = env.reset()
    assert obs[8] == pytest.approx(1.0)


def test_obs_nearest_walker_normalized():
    env = _make_env(nearest_walker_m=15.0)
    obs, _ = env.reset()
    assert obs[10] == pytest.approx(15.0 / 50.0, abs=1e-4)


def test_obs_no_walker_detected():
    env = _make_env(nearest_walker_m=None)
    obs, _ = env.reset()
    assert obs[10] == pytest.approx(1.0)


def test_obs_nearest_stop_yield_normalized():
    env = _make_env(nearest_stop_yield_m=8.0)
    obs, _ = env.reset()
    assert obs[11] == pytest.approx(8.0 / 50.0, abs=1e-4)


def test_obs_no_stop_yield_detected():
    env = _make_env(nearest_stop_yield_m=None)
    obs, _ = env.reset()
    assert obs[11] == pytest.approx(1.0)


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
    env._last_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = env.render()
    assert isinstance(result, np.ndarray)
    assert result.shape == (720, 1280, 3)
    assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Episode reward components
# ---------------------------------------------------------------------------


def test_episode_reward_components_empty_mid_episode():
    env = _make_env(speed_mps=10.0, max_episode_steps=10)
    env.reset()
    _, _, terminated, truncated, info = env.step(_ZERO_ACTION)
    assert terminated is False
    assert truncated is False
    assert info == {}


def test_episode_reward_components_reported_on_truncation():
    env = _make_env(speed_mps=10.0, max_episode_steps=3)  # 36 km/h
    env.reset()
    env.step(_ZERO_ACTION)
    env.step(_ZERO_ACTION)
    _, _, terminated, truncated, info = env.step(_ZERO_ACTION)
    assert truncated is True
    assert terminated is False
    # r_speed = (36/90)*0.3 = 0.12 per step, summed over 3 steps
    assert info["r_speed"] == pytest.approx(0.12 * 3, abs=1e-3)


def test_episode_reward_components_reported_on_collision():
    env = _make_env(speed_mps=10.0)
    env.reset()
    env._collision_flag = True
    _, _, terminated, _, info = env.step(_ZERO_ACTION)
    assert terminated is True
    assert info["r_collision"] != 0.0
    assert info["r_speed"] == pytest.approx(0.0)  # collision step zeroes every other component


def test_episode_reward_components_has_all_twelve_keys_when_reported():
    env = _make_env(speed_mps=10.0, max_episode_steps=1)
    env.reset()
    _, _, _, truncated, info = env.step(_ZERO_ACTION)
    assert truncated is True
    expected_keys = {
        "r_speed", "r_center", "r_alive", "r_offroad", "r_stall", "r_off_route",
        "r_following", "r_walker", "r_speeding", "r_red_light", "r_stop_yield", "r_collision",
    }
    assert set(info.keys()) == expected_keys


def test_episode_reward_components_reset_between_episodes():
    env = _make_env(speed_mps=10.0, max_episode_steps=2)
    env.reset()
    env.step(_ZERO_ACTION)
    _, _, _, truncated, _ = env.step(_ZERO_ACTION)
    assert truncated is True
    env.reset()
    assert env._episode_reward_components["r_speed"] == pytest.approx(0.0)
