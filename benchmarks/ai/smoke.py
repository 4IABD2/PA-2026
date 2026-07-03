"""Smoke tests Phase 1 — RL (no CARLA, no GPU)."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from src.ai.rewards.reward_fn import compute_reward
from src.interfaces.perception_types import LanesInfo
from src.interfaces.stubs import CarlaGTDepthEstimator, CarlaGTLaneDetector


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------


def test_collision_terminates_with_negative_reward():
    """A collision at 0 km/h impact speed must terminate the episode and cost -5.0."""
    reward, done = compute_reward(
        speed_kmh=30.0, center_offset=0.0, is_on_road=True, collision=True
    )
    assert done is True
    assert reward == pytest.approx(-5.0)


def test_offroad_applies_penalty():
    """Being off-road must give a lower reward than being on-road, by exactly -0.25."""
    reward_onroad, _ = compute_reward(
        speed_kmh=30.0, center_offset=0.0, is_on_road=True, collision=False
    )
    reward_offroad, _ = compute_reward(
        speed_kmh=30.0, center_offset=0.0, is_on_road=False, collision=False
    )
    assert reward_offroad < reward_onroad
    assert reward_offroad == pytest.approx(reward_onroad - 0.25)


def test_speed_reward_scales_with_speed():
    """Higher speed must give a higher reward (at a fixed offset)."""
    reward_slow, _ = compute_reward(
        speed_kmh=10.0, center_offset=0.0, is_on_road=True, collision=False
    )
    reward_fast, _ = compute_reward(
        speed_kmh=40.0, center_offset=0.0, is_on_road=True, collision=False
    )
    assert reward_fast > reward_slow


def test_centering_reward_maximal_at_center():
    """An offset of 0 must give more reward than an offset of 1."""
    reward_center, _ = compute_reward(
        speed_kmh=0.0, center_offset=0.0, is_on_road=True, collision=False
    )
    reward_edge, _ = compute_reward(
        speed_kmh=0.0, center_offset=1.0, is_on_road=True, collision=False
    )
    assert reward_center > reward_edge


def test_alive_bonus_always_present():
    """Even when stopped and centered, the reward must include the survival bonus."""
    reward, done = compute_reward(
        speed_kmh=0.0, center_offset=0.0, is_on_road=True, collision=False
    )
    assert done is False
    assert reward >= 0.01


def test_reward_components_sum_at_max():
    """At max speed (90 km/h), offset 0, on-road, no collision: r = 0.5 + 0.3 + 0.01."""
    reward, done = compute_reward(
        speed_kmh=90.0, center_offset=0.0, is_on_road=True, collision=False
    )
    assert done is False
    assert reward == pytest.approx(0.5 + 0.3 + 0.01)


def test_collision_overrides_other_components():
    """On collision, reward must be -5.0 (0 km/h impact) regardless of speed or offset."""
    reward, done = compute_reward(
        speed_kmh=50.0, center_offset=0.0, is_on_road=True, collision=True
    )
    assert reward == pytest.approx(-5.0)
    assert done is True


def test_stall_penalises_zero_speed():
    """Staying still must be rewarded less than moving forward."""
    reward_still, _ = compute_reward(
        speed_kmh=0.0, center_offset=0.0, is_on_road=True, collision=False
    )
    reward_moving, _ = compute_reward(
        speed_kmh=5.0, center_offset=0.0, is_on_road=True, collision=False
    )
    assert reward_moving > reward_still


def test_collision_scales_with_impact_speed():
    """A faster impact must cost more than a near-stationary one."""
    reward_slow, _ = compute_reward(
        speed_kmh=50.0, center_offset=0.0, is_on_road=True, collision=True,
        collision_speed_kmh=5.0,
    )
    reward_fast, _ = compute_reward(
        speed_kmh=50.0, center_offset=0.0, is_on_road=True, collision=True,
        collision_speed_kmh=60.0,
    )
    assert reward_fast < reward_slow
    assert reward_slow == pytest.approx(-5.0 + (-0.05 * 5.0))
    assert reward_fast == pytest.approx(-5.0 + (-0.05 * 60.0))


def test_following_penalty_none_when_vehicle_far():
    reward_far, _ = compute_reward(
        speed_kmh=50.0, center_offset=0.0, is_on_road=True, collision=False,
        nearest_vehicle_m=100.0,
    )
    reward_default, _ = compute_reward(
        speed_kmh=50.0, center_offset=0.0, is_on_road=True, collision=False,
    )
    assert reward_far == pytest.approx(reward_default)


def test_following_penalty_none_when_stopped():
    reward_stopped_close, _ = compute_reward(
        speed_kmh=0.0, center_offset=0.0, is_on_road=True, collision=False,
        nearest_vehicle_m=2.0,
    )
    reward_stopped_far, _ = compute_reward(
        speed_kmh=0.0, center_offset=0.0, is_on_road=True, collision=False,
        nearest_vehicle_m=100.0,
    )
    assert reward_stopped_close == pytest.approx(reward_stopped_far)


def test_following_penalty_triggers_under_two_second_headway():
    """10 m at 36 km/h (10 m/s) = 1.0 s headway, under the 2.0 s safe threshold."""
    reward_close, _ = compute_reward(
        speed_kmh=36.0, center_offset=0.0, is_on_road=True, collision=False,
        nearest_vehicle_m=10.0,
    )
    reward_clear, _ = compute_reward(
        speed_kmh=36.0, center_offset=0.0, is_on_road=True, collision=False,
        nearest_vehicle_m=100.0,
    )
    assert reward_close < reward_clear
    assert (reward_clear - reward_close) == pytest.approx(0.1, abs=1e-3)  # -(1 - 1.0/2.0) * 0.2


def test_walker_penalty_triggers_when_close():
    reward_close, _ = compute_reward(
        speed_kmh=20.0, center_offset=0.0, is_on_road=True, collision=False,
        nearest_walker_m=5.0,
    )
    reward_far, _ = compute_reward(
        speed_kmh=20.0, center_offset=0.0, is_on_road=True, collision=False,
        nearest_walker_m=100.0,
    )
    assert reward_close < reward_far
    assert (reward_far - reward_close) == pytest.approx(0.15, abs=1e-3)  # -(1 - 5/10) * 0.3


def test_walker_penalty_none_when_far():
    reward, _ = compute_reward(
        speed_kmh=20.0, center_offset=0.0, is_on_road=True, collision=False,
        nearest_walker_m=15.0,
    )
    reward_default, _ = compute_reward(
        speed_kmh=20.0, center_offset=0.0, is_on_road=True, collision=False,
    )
    assert reward == pytest.approx(reward_default)


def test_speeding_penalty_none_within_tolerance():
    """5 km/h over the limit is within the tolerance band — no penalty."""
    reward, _ = compute_reward(
        speed_kmh=55.0, center_offset=0.0, is_on_road=True, collision=False,
        speed_limit_kmh=50.0,
    )
    reward_no_limit, _ = compute_reward(
        speed_kmh=55.0, center_offset=0.0, is_on_road=True, collision=False,
    )
    assert reward == pytest.approx(reward_no_limit)


def test_speeding_penalty_triggers_over_tolerance():
    """70 km/h with a 50 km/h limit and 5 km/h tolerance = 15 km/h over."""
    reward, _ = compute_reward(
        speed_kmh=70.0, center_offset=0.0, is_on_road=True, collision=False,
        speed_limit_kmh=50.0,
    )
    reward_no_limit, _ = compute_reward(
        speed_kmh=70.0, center_offset=0.0, is_on_road=True, collision=False,
    )
    assert reward < reward_no_limit
    assert (reward_no_limit - reward) == pytest.approx((15.0 / 90.0) * 0.3, abs=1e-3)


def test_red_light_violation_applies_flat_penalty():
    reward_violation, _ = compute_reward(
        speed_kmh=30.0, center_offset=0.0, is_on_road=True, collision=False,
        red_light_violation=True,
    )
    reward_clean, _ = compute_reward(
        speed_kmh=30.0, center_offset=0.0, is_on_road=True, collision=False,
        red_light_violation=False,
    )
    assert (reward_clean - reward_violation) == pytest.approx(2.0, abs=1e-3)


def test_stop_yield_violation_applies_flat_penalty():
    reward_violation, _ = compute_reward(
        speed_kmh=30.0, center_offset=0.0, is_on_road=True, collision=False,
        stop_yield_violation=True,
    )
    reward_clean, _ = compute_reward(
        speed_kmh=30.0, center_offset=0.0, is_on_road=True, collision=False,
        stop_yield_violation=False,
    )
    assert (reward_clean - reward_violation) == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# CarlaGTDepthEstimator
# ---------------------------------------------------------------------------


def test_depth_estimator_registers_listener_on_init():
    """Le constructeur doit appeler sensor.listen() pour recevoir les frames."""
    sensor = Mock()
    CarlaGTDepthEstimator(sensor)
    sensor.listen.assert_called_once()


def test_depth_estimator_raises_before_first_frame():
    """estimate() doit lever RuntimeError si aucune frame n'a encore été reçue."""
    sensor = Mock()
    est = CarlaGTDepthEstimator(sensor)
    with pytest.raises(RuntimeError):
        est.estimate(np.zeros((88, 200, 3), dtype=np.uint8))


def test_depth_estimator_returns_float32_array_when_ready():
    """estimate() doit retourner _last_depth dès qu'une frame a été reçue."""
    sensor = Mock()
    est = CarlaGTDepthEstimator(sensor)
    est._last_depth = np.ones((88, 200), dtype=np.float32) * 10.0
    result = est.estimate(np.zeros((88, 200, 3), dtype=np.uint8))
    assert result.dtype == np.float32
    assert result.shape == (88, 200)
    assert float(result[0, 0]) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# CarlaGTLaneDetector
# ---------------------------------------------------------------------------


def _make_lane_detector(vx: float, vy: float, wx: float, wy: float, yaw: float, lane_width: float = 3.5):
    """Helper : construit un CarlaGTLaneDetector avec des mocks positionnés."""
    world, vehicle = Mock(), Mock()
    vehicle.get_transform.return_value.location.x = vx
    vehicle.get_transform.return_value.location.y = vy
    wp = Mock()
    wp.transform.location.x = wx
    wp.transform.location.y = wy
    wp.transform.rotation.yaw = yaw
    wp.lane_width = lane_width
    world.get_map.return_value.get_waypoint.return_value = wp
    return CarlaGTLaneDetector(world, vehicle)


def test_lane_detector_returns_lanes_info():
    """detect() doit retourner un LanesInfo avec center_offset dans [-1, 1]."""
    det = _make_lane_detector(0.0, 0.0, 0.0, 0.0, 0.0)
    result = det.detect(np.zeros((88, 200, 3), dtype=np.uint8))
    assert isinstance(result, LanesInfo)
    assert result.center_offset is not None
    assert -1.0 <= result.center_offset <= 1.0


def test_lane_detector_centered_when_on_waypoint():
    """Véhicule exactement sur le waypoint → center_offset ≈ 0."""
    det = _make_lane_detector(vx=0.0, vy=0.0, wx=0.0, wy=0.0, yaw=0.0)
    result = det.detect(np.zeros((88, 200, 3), dtype=np.uint8))
    assert result.center_offset == pytest.approx(0.0)


def test_lane_detector_positive_offset_when_right_of_waypoint():
    """Véhicule à droite du waypoint (heading +X, véhicule décalé en +Y) → offset > 0."""
    # lane_width=4m → half=2m, décalage=1m → offset=0.5
    det = _make_lane_detector(vx=0.0, vy=1.0, wx=0.0, wy=0.0, yaw=0.0, lane_width=4.0)
    result = det.detect(np.zeros((88, 200, 3), dtype=np.uint8))
    assert result.center_offset == pytest.approx(0.5)


def test_lane_detector_clamps_to_minus_one_plus_one():
    """Un décalage extrême doit être clampé à [-1, 1]."""
    det = _make_lane_detector(vx=0.0, vy=100.0, wx=0.0, wy=0.0, yaw=0.0)
    result = det.detect(np.zeros((88, 200, 3), dtype=np.uint8))
    assert result.center_offset == pytest.approx(1.0)
