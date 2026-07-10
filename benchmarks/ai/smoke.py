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
    reward, done, _ = compute_reward(
        speed_kmh=30.0, center_offset=0.0, is_on_road=True, collision=True
    )
    assert done is True
    assert reward == pytest.approx(-5.0)


def test_offroad_applies_penalty():
    """Being off-road must cost -0.90 total: r_speed and r_center both drop
    to 0.0 (no longer computed from a stale/default center_offset while
    off-road), plus the raised -0.5 _P_OFFROAD penalty (was -0.25)."""
    reward_onroad, _, _ = compute_reward(
        speed_kmh=30.0, center_offset=0.0, is_on_road=True, collision=False
    )
    reward_offroad, _, _ = compute_reward(
        speed_kmh=30.0, center_offset=0.0, is_on_road=False, collision=False
    )
    assert reward_offroad < reward_onroad
    assert reward_offroad == pytest.approx(reward_onroad - 0.90)


def test_r_center_zeroed_when_off_road():
    """r_center must be 0.0 when off-road, regardless of the reported center_offset —
    the gate is purely on is_on_road, since center_offset defaults to 0.0 (a
    "perfectly centered" value) whenever the lane detector loses track off-road."""
    _, _, components_centered = compute_reward(
        speed_kmh=30.0, center_offset=0.0, is_on_road=False, collision=False
    )
    _, _, components_edge = compute_reward(
        speed_kmh=30.0, center_offset=0.9, is_on_road=False, collision=False
    )
    assert components_centered["r_center"] == pytest.approx(0.0)
    assert components_edge["r_center"] == pytest.approx(0.0)


def test_r_speed_zeroed_when_off_road():
    """r_speed must be 0.0 when off-road, regardless of speed."""
    _, _, components_slow = compute_reward(
        speed_kmh=10.0, center_offset=0.0, is_on_road=False, collision=False
    )
    _, _, components_fast = compute_reward(
        speed_kmh=80.0, center_offset=0.0, is_on_road=False, collision=False
    )
    assert components_slow["r_speed"] == pytest.approx(0.0)
    assert components_fast["r_speed"] == pytest.approx(0.0)


def test_speed_reward_scales_with_speed():
    """Higher speed must give a higher reward (at a fixed offset)."""
    reward_slow, _, _ = compute_reward(
        speed_kmh=10.0, center_offset=0.0, is_on_road=True, collision=False
    )
    reward_fast, _, _ = compute_reward(
        speed_kmh=40.0, center_offset=0.0, is_on_road=True, collision=False
    )
    assert reward_fast > reward_slow


def test_centering_reward_maximal_at_center():
    """An offset of 0 must give more reward than an offset of 1."""
    reward_center, _, _ = compute_reward(
        speed_kmh=0.0, center_offset=0.0, is_on_road=True, collision=False
    )
    reward_edge, _, _ = compute_reward(
        speed_kmh=0.0, center_offset=1.0, is_on_road=True, collision=False
    )
    assert reward_center > reward_edge


def test_alive_bonus_always_present():
    """Even when stopped and centered, the reward must include the survival bonus."""
    reward, done, _ = compute_reward(
        speed_kmh=0.0, center_offset=0.0, is_on_road=True, collision=False
    )
    assert done is False
    assert reward >= 0.01


def test_reward_components_sum_at_max():
    """At max speed (90 km/h), offset 0, on-road, no collision: r = 0.3 + 0.3 + 0.01 + 0.05
    (the last term is r_safe — no vehicle/walker/speed-limit configured, so nothing
    dangerous is active and the safe-driving bonus fires)."""
    reward, done, _ = compute_reward(
        speed_kmh=90.0, center_offset=0.0, is_on_road=True, collision=False
    )
    assert done is False
    assert reward == pytest.approx(0.3 + 0.3 + 0.01 + 0.05)


def test_collision_overrides_other_components():
    """On collision, reward must be -5.0 (0 km/h impact) regardless of speed or offset."""
    reward, done, _ = compute_reward(
        speed_kmh=50.0, center_offset=0.0, is_on_road=True, collision=True
    )
    assert reward == pytest.approx(-5.0)
    assert done is True


def test_stall_penalises_zero_speed():
    """Staying still must be rewarded less than moving forward."""
    reward_still, _, _ = compute_reward(
        speed_kmh=0.0, center_offset=0.0, is_on_road=True, collision=False
    )
    reward_moving, _, _ = compute_reward(
        speed_kmh=5.0, center_offset=0.0, is_on_road=True, collision=False
    )
    assert reward_moving > reward_still


def test_stall_not_penalised_near_red_light():
    """A red light close ahead is a legitimate reason to stop — r_stall must
    not fire, otherwise stopping at a red light is punished far harder than
    running it."""
    _, _, components = compute_reward(
        speed_kmh=0.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        red_light_distance_m=5.0,
    )
    assert components["r_stall"] == 0.0


def test_stall_not_penalised_near_vehicle_danger():
    """A vehicle within the danger distance is a legitimate reason to stop."""
    _, _, components = compute_reward(
        speed_kmh=0.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        nearest_vehicle_m=5.0,
    )
    assert components["r_stall"] == 0.0


def test_stall_not_penalised_near_walker_danger():
    """A pedestrian within the danger distance is a legitimate reason to stop."""
    _, _, components = compute_reward(
        speed_kmh=0.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        nearest_walker_m=5.0,
    )
    assert components["r_stall"] == 0.0


def test_stall_still_penalised_with_no_legitimate_reason():
    """Regression: an unjustified stall (nothing dangerous/red nearby) must
    still cost the full -0.20 stall penalty."""
    _, _, components = compute_reward(
        speed_kmh=0.0, center_offset=0.0, is_on_road=True, collision=False
    )
    assert components["r_stall"] == pytest.approx(-0.20)


def test_stall_penalised_when_red_light_far_away():
    """A red light far ahead (outside the stall gate distance) must not
    exempt stalling — the gate has an actual boundary, not "any red light
    anywhere exempts stalling"."""
    _, _, components = compute_reward(
        speed_kmh=0.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        red_light_distance_m=20.0,
    )
    assert components["r_stall"] == pytest.approx(-0.20)


def test_collision_scales_with_impact_speed():
    """A faster impact must cost more than a near-stationary one."""
    reward_slow, _, _ = compute_reward(
        speed_kmh=50.0,
        center_offset=0.0,
        is_on_road=True,
        collision=True,
        collision_speed_kmh=5.0,
    )
    reward_fast, _, _ = compute_reward(
        speed_kmh=50.0,
        center_offset=0.0,
        is_on_road=True,
        collision=True,
        collision_speed_kmh=60.0,
    )
    assert reward_fast < reward_slow
    assert reward_slow == pytest.approx(-5.0 + (-0.20 * 5.0))
    assert reward_fast == pytest.approx(-5.0 + (-0.20 * 60.0))


def test_following_penalty_none_when_vehicle_far():
    reward_far, _, _ = compute_reward(
        speed_kmh=50.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        nearest_vehicle_m=100.0,
    )
    reward_default, _, _ = compute_reward(
        speed_kmh=50.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
    )
    assert reward_far == pytest.approx(reward_default)


def test_following_penalty_none_when_stopped():
    """r_following stays 0 at any distance once stopped (speed_kmh < 1.0) --
    "close" here uses 20.0 m, outside _WALKER_DANGER_M (10.0), so this isn't
    confounded by the separate is_legitimate_stop exemption on r_stall for a
    vehicle within danger distance."""
    reward_stopped_close, _, _ = compute_reward(
        speed_kmh=0.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        nearest_vehicle_m=20.0,
    )
    reward_stopped_far, _, _ = compute_reward(
        speed_kmh=0.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        nearest_vehicle_m=100.0,
    )
    assert reward_stopped_close == pytest.approx(reward_stopped_far)


def test_following_penalty_triggers_under_two_second_headway():
    """10 m at 36 km/h (10 m/s) = 1.0 s headway, under the 2.0 s safe threshold."""
    reward_close, _, _ = compute_reward(
        speed_kmh=36.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        nearest_vehicle_m=10.0,
    )
    reward_clear, _, _ = compute_reward(
        speed_kmh=36.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        nearest_vehicle_m=100.0,
    )
    assert reward_close < reward_clear
    # -(1 - 1.0/2.0) * 0.2 = -0.1 from r_following, plus reward_clear also earns the
    # +0.05 r_safe bonus that reward_close doesn't (it has an active following risk)
    assert (reward_clear - reward_close) == pytest.approx(0.15, abs=1e-3)


def test_walker_penalty_triggers_when_close():
    reward_close, _, _ = compute_reward(
        speed_kmh=20.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        nearest_walker_m=5.0,
    )
    reward_far, _, _ = compute_reward(
        speed_kmh=20.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        nearest_walker_m=100.0,
    )
    assert reward_close < reward_far
    # -(1 - 5/10) * 0.3 = -0.15 from r_walker, plus reward_far also earns the +0.05
    # r_safe bonus that reward_close doesn't (it has an active walker danger)
    assert (reward_far - reward_close) == pytest.approx(0.20, abs=1e-3)


def test_walker_penalty_none_when_far():
    reward, _, _ = compute_reward(
        speed_kmh=20.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        nearest_walker_m=15.0,
    )
    reward_default, _, _ = compute_reward(
        speed_kmh=20.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
    )
    assert reward == pytest.approx(reward_default)


def test_speeding_penalty_none_within_tolerance():
    """5 km/h over the limit is within the tolerance band — no penalty."""
    reward, _, _ = compute_reward(
        speed_kmh=55.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        speed_limit_kmh=50.0,
    )
    reward_no_limit, _, _ = compute_reward(
        speed_kmh=55.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
    )
    assert reward == pytest.approx(reward_no_limit)


def test_speeding_penalty_triggers_over_tolerance():
    """70 km/h with a 50 km/h limit and 5 km/h tolerance = 15 km/h over."""
    reward, _, _ = compute_reward(
        speed_kmh=70.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        speed_limit_kmh=50.0,
    )
    reward_no_limit, _, _ = compute_reward(
        speed_kmh=70.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
    )
    assert reward < reward_no_limit
    # (15/90)*0.3 = 0.05 from r_speeding, plus reward_no_limit also earns the +0.05
    # r_safe bonus that reward doesn't (it has an active speeding violation)
    assert (reward_no_limit - reward) == pytest.approx(
        (15.0 / 90.0) * 0.3 + 0.05, abs=1e-3
    )


def test_red_light_violation_applies_flat_penalty():
    reward_violation, _, _ = compute_reward(
        speed_kmh=30.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        red_light_violation=True,
    )
    reward_clean, _, _ = compute_reward(
        speed_kmh=30.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        red_light_violation=False,
    )
    assert (reward_clean - reward_violation) == pytest.approx(2.0, abs=1e-3)


def test_stop_yield_violation_applies_flat_penalty():
    reward_violation, _, _ = compute_reward(
        speed_kmh=30.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        stop_yield_violation=True,
    )
    reward_clean, _, _ = compute_reward(
        speed_kmh=30.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        stop_yield_violation=False,
    )
    assert (reward_clean - reward_violation) == pytest.approx(1.0, abs=1e-3)


def test_reward_components_sum_to_total_no_collision():
    """reward must always equal sum(components.values()) — no-collision branch."""
    reward, _, components = compute_reward(
        speed_kmh=45.0,
        center_offset=0.3,
        is_on_road=False,
        collision=False,
        off_route=True,
        nearest_vehicle_m=8.0,
        nearest_walker_m=4.0,
        speed_limit_kmh=30.0,
        red_light_violation=True,
        stop_yield_violation=True,
    )
    assert reward == pytest.approx(sum(components.values()))


def test_reward_components_sum_to_total_on_collision():
    """reward must always equal sum(components.values()) — collision branch."""
    reward, _, components = compute_reward(
        speed_kmh=45.0,
        center_offset=0.0,
        is_on_road=True,
        collision=True,
        collision_speed_kmh=33.0,
    )
    assert reward == pytest.approx(sum(components.values()))


def test_reward_components_always_has_all_fifteen_keys():
    expected_keys = {
        "r_speed",
        "r_center",
        "r_alive",
        "r_offroad",
        "r_stall",
        "r_off_route",
        "r_following",
        "r_walker",
        "r_speeding",
        "r_red_light",
        "r_stop_yield",
        "r_collision",
        "r_destination",
        "r_safe",
        "r_jerk",
    }
    _, _, components_no_collision = compute_reward(
        speed_kmh=10.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
    )
    _, _, components_collision = compute_reward(
        speed_kmh=10.0,
        center_offset=0.0,
        is_on_road=True,
        collision=True,
    )
    _, _, components_destination = compute_reward(
        speed_kmh=10.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        reached_destination=True,
    )
    assert set(components_no_collision.keys()) == expected_keys
    assert set(components_collision.keys()) == expected_keys
    assert set(components_destination.keys()) == expected_keys


def test_reward_components_only_collision_nonzero_on_collision():
    """On a collision step, every component except r_collision must be exactly 0.0."""
    _, _, components = compute_reward(
        speed_kmh=45.0,
        center_offset=0.5,
        is_on_road=False,
        collision=True,
        collision_speed_kmh=20.0,
        off_route=True,
    )
    for key, value in components.items():
        if key == "r_collision":
            assert value == pytest.approx(-5.0 + (-0.20 * 20.0))
        else:
            assert value == pytest.approx(0.0)


def test_destination_reached_terminates_with_bonus():
    reward, done, components = compute_reward(
        speed_kmh=30.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        reached_destination=True,
    )
    assert done is True
    assert reward == pytest.approx(10.0)
    assert components["r_destination"] == pytest.approx(10.0)


def test_destination_reached_zeroes_other_components():
    _, _, components = compute_reward(
        speed_kmh=30.0,
        center_offset=0.5,
        is_on_road=False,
        collision=False,
        reached_destination=True,
        off_route=True,
    )
    for key, value in components.items():
        if key == "r_destination":
            assert value == pytest.approx(10.0)
        else:
            assert value == pytest.approx(0.0)


def test_collision_takes_priority_over_destination():
    """If both fire the same step, collision must win (checked first)."""
    reward, done, components = compute_reward(
        speed_kmh=30.0,
        center_offset=0.0,
        is_on_road=True,
        collision=True,
        collision_speed_kmh=10.0,
        reached_destination=True,
    )
    assert done is True
    assert components["r_destination"] == pytest.approx(0.0)
    assert reward == pytest.approx(-5.0 + (-0.20 * 10.0))


def test_safe_driving_bonus_when_no_danger():
    _, _, components = compute_reward(
        speed_kmh=30.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        nearest_vehicle_m=100.0,
        nearest_walker_m=100.0,
        speed_limit_kmh=None,
    )
    assert components["r_safe"] == pytest.approx(0.05)


def test_safe_driving_bonus_absent_when_following_too_close():
    _, _, components = compute_reward(
        speed_kmh=36.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        nearest_vehicle_m=10.0,
        nearest_walker_m=100.0,
        speed_limit_kmh=None,
    )
    assert components["r_safe"] == pytest.approx(0.0)


def test_safe_driving_bonus_absent_when_walker_close():
    _, _, components = compute_reward(
        speed_kmh=30.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        nearest_vehicle_m=100.0,
        nearest_walker_m=5.0,
        speed_limit_kmh=None,
    )
    assert components["r_safe"] == pytest.approx(0.0)


def test_safe_driving_bonus_absent_when_speeding():
    _, _, components = compute_reward(
        speed_kmh=80.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        nearest_vehicle_m=100.0,
        nearest_walker_m=100.0,
        speed_limit_kmh=30.0,
    )
    assert components["r_safe"] == pytest.approx(0.0)


def test_jerk_penalty_zero_when_no_delta():
    _, _, components = compute_reward(
        speed_kmh=30.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        steer_delta=0.0,
    )
    assert components["r_jerk"] == pytest.approx(0.0)


def test_jerk_penalty_scales_with_steer_delta():
    _, _, components_small = compute_reward(
        speed_kmh=30.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        steer_delta=0.1,
    )
    _, _, components_large = compute_reward(
        speed_kmh=30.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        steer_delta=1.0,
    )
    assert components_small["r_jerk"] == pytest.approx(-0.1 * 0.1)
    assert components_large["r_jerk"] == pytest.approx(-1.0 * 0.1)
    assert components_large["r_jerk"] < components_small["r_jerk"]


def test_r_speed_uses_progress_speed_when_provided():
    _, _, components = compute_reward(
        speed_kmh=90.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
        progress_speed_kmh=45.0,
    )
    assert components["r_speed"] == pytest.approx((45.0 / 90.0) * 0.3)


def test_r_speed_falls_back_to_speed_kmh_without_progress():
    _, _, components = compute_reward(
        speed_kmh=45.0,
        center_offset=0.0,
        is_on_road=True,
        collision=False,
    )
    assert components["r_speed"] == pytest.approx((45.0 / 90.0) * 0.3)


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


def _make_lane_detector(
    vx: float, vy: float, wx: float, wy: float, yaw: float, lane_width: float = 3.5
):
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
