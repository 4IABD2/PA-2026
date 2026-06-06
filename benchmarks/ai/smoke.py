"""Smoke tests Phase 1 — RL (sans CARLA, sans GPU)."""

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
    """Une collision doit terminer l'épisode et donner -1.0."""
    reward, done = compute_reward(
        speed_kmh=30.0, center_offset=0.0, is_on_road=True, collision=True
    )
    assert done is True
    assert reward == pytest.approx(-1.0)


def test_offroad_applies_penalty():
    """Être hors-route doit donner une récompense plus faible qu'en route."""
    reward_onroad, _ = compute_reward(
        speed_kmh=30.0, center_offset=0.0, is_on_road=True, collision=False
    )
    reward_offroad, _ = compute_reward(
        speed_kmh=30.0, center_offset=0.0, is_on_road=False, collision=False
    )
    assert reward_offroad < reward_onroad
    assert reward_offroad == pytest.approx(reward_onroad - 0.5)


def test_speed_reward_scales_with_speed():
    """Plus la vitesse est élevée, plus la récompense est haute (à offset fixe)."""
    reward_slow, _ = compute_reward(
        speed_kmh=10.0, center_offset=0.0, is_on_road=True, collision=False
    )
    reward_fast, _ = compute_reward(
        speed_kmh=40.0, center_offset=0.0, is_on_road=True, collision=False
    )
    assert reward_fast > reward_slow


def test_centering_reward_maximal_at_center():
    """Un offset de 0 doit donner plus de récompense qu'un offset de 1."""
    reward_center, _ = compute_reward(
        speed_kmh=0.0, center_offset=0.0, is_on_road=True, collision=False
    )
    reward_edge, _ = compute_reward(
        speed_kmh=0.0, center_offset=1.0, is_on_road=True, collision=False
    )
    assert reward_center > reward_edge


def test_alive_bonus_always_present():
    """Même à l'arrêt centré, la récompense doit inclure le bonus de survie."""
    reward, done = compute_reward(
        speed_kmh=0.0, center_offset=0.0, is_on_road=True, collision=False
    )
    assert done is False
    assert reward >= 0.01


def test_reward_components_sum_at_max():
    """À vitesse max, offset 0, en route, sans collision : r = 0.5 + 0.3 + 0.01."""
    reward, done = compute_reward(
        speed_kmh=50.0, center_offset=0.0, is_on_road=True, collision=False
    )
    assert done is False
    assert reward == pytest.approx(0.5 + 0.3 + 0.01)


def test_collision_overrides_other_components():
    """En collision, la reward doit être -1.0 quelle que soit la vitesse ou l'offset."""
    reward, done = compute_reward(
        speed_kmh=50.0, center_offset=0.0, is_on_road=True, collision=True
    )
    assert reward == pytest.approx(-1.0)
    assert done is True


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
