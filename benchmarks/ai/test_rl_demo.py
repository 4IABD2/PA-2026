"""Smoke tests for rl_demo — no CARLA, no GPU."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from pathlib import Path

from src.ai.inference.rl_demo import (
    load_model,
    run_episode,
    _add_hud,
    record_episode,
    _draw_route_map_card,
    _clear_spawn_area,
    _EVAL_CLEAR_RADIUS_M,
    pick_best_checkpoint,
    _draw_minimap,
    _draw_bboxes,
    eval_model,
    Scenario,
)
from src.interfaces.navigation_types import Route, Waypoint
from src.interfaces.perception_types import DetectedObject, ObjectClass
import cv2

_OBS = np.zeros(7, dtype=np.float32)
_ACT = np.zeros(2, dtype=np.float32)


def _model(rewards=None):
    """Mock model whose predict() always returns a zero action."""
    m = Mock()
    m.predict.return_value = (_ACT, None)
    return m


def _env(*steps):
    """Mock env with fixed step side effects — each entry is (reward, terminated, truncated)."""
    e = Mock()
    e.reset.return_value = (_OBS, {})
    e.step.side_effect = [(_OBS, r, te, tr, {}) for r, te, tr in steps]
    return e


# ---------------------------------------------------------------------------
# run_episode
# ---------------------------------------------------------------------------


def test_run_episode_returns_three_tuple():
    total_reward, steps, reason = run_episode(
        _model(), _env((0.1, False, False)), max_steps=1
    )
    assert isinstance(total_reward, float)
    assert isinstance(steps, int)
    assert reason in {"collision", "truncated", "max_steps"}


def test_run_episode_reset_called_once():
    env = _env((0.1, False, False), (0.1, False, False))
    run_episode(_model(), env, max_steps=2)
    env.reset.assert_called_once()


def test_run_episode_accumulates_reward():
    env = _env((0.3, False, False), (0.5, False, False), (0.2, False, False))
    total_reward, steps, reason = run_episode(_model(), env, max_steps=3)
    assert total_reward == pytest.approx(1.0)
    assert steps == 3
    assert reason == "max_steps"


def test_run_episode_stops_on_collision():
    env = _env((0.1, False, False), (-1.0, True, False), (0.1, False, False))
    total_reward, steps, reason = run_episode(_model(), env, max_steps=10)
    assert reason == "collision"
    assert steps == 2
    assert env.step.call_count == 2


def test_run_episode_stops_on_truncated():
    env = _env((0.1, False, True))
    _, steps, reason = run_episode(_model(), env, max_steps=10)
    assert reason == "truncated"
    assert steps == 1


def test_run_episode_stops_at_max_steps():
    # env never terminates — loop must exit at max_steps
    env = Mock()
    env.reset.return_value = (_OBS, {})
    env.step.return_value = (_OBS, 0.1, False, False, {})
    _, steps, reason = run_episode(_model(), env, max_steps=5)
    assert steps == 5
    assert reason == "max_steps"
    assert env.step.call_count == 5


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _add_hud
# ---------------------------------------------------------------------------


def _hud_info(**kwargs):
    base = {
        "episode": 1,
        "step": 10,
        "max_steps": 100,
        "reward": 0.5,
        "total_reward": 3.2,
        "speed_kmh": 30.0,
        "action": [0.0, 0.3],
        "params": {"lr": "3e-4"},
    }
    return {**base, **kwargs}


def test_add_hud_returns_same_shape():
    frame = np.zeros((88, 200, 3), dtype=np.uint8)
    result = _add_hud(frame, _hud_info())
    assert result.shape == (88, 200, 3)
    assert result.dtype == np.uint8


def test_add_hud_modifies_frame():
    frame = np.zeros((88, 200, 3), dtype=np.uint8)
    result = _add_hud(frame, _hud_info())
    # HUD must have drawn something — result is not all-black
    assert result.sum() > 0


# ---------------------------------------------------------------------------
# record_episode
# ---------------------------------------------------------------------------


def _recording_env(n_steps=3):
    env = Mock()
    env.reset.return_value = (np.zeros(7, dtype=np.float32), {})
    env.step.return_value = (np.zeros(7, dtype=np.float32), 0.1, False, False, {})
    env.render.return_value = np.zeros((88, 200, 3), dtype=np.uint8)
    # explicit None (not an auto-vivified Mock) so record_episode's
    # `getattr(env, "route", None)` route-map-card check is a no-op by default
    env.route = None
    # explicit empty list (not an auto-vivified Mock) so getattr(env, "last_objects", []) works
    env.last_objects = []
    return env


def _route(coords: list[tuple[float, float]]) -> Route:
    """Builds a Route from a list of (x, y) tuples, z and yaw fixed at 0."""
    waypoints = [Waypoint(x=x, y=y, z=0.0, yaw_deg=0.0) for x, y in coords]
    destination = waypoints[-1] if waypoints else Waypoint(0.0, 0.0, 0.0, 0.0)
    return Route(waypoints=waypoints, destination=destination)


def test_record_episode_creates_video_file(tmp_path):
    output = str(tmp_path / "demo.mp4")
    record_episode(_model(), _recording_env(), output_path=output, fps=5, max_steps=3)
    assert Path(output).exists()
    assert Path(output).stat().st_size > 0


def test_record_episode_calls_render(tmp_path):
    env = _recording_env()
    output = str(tmp_path / "demo.mp4")
    record_episode(_model(), env, output_path=output, fps=5, max_steps=3)
    assert env.render.call_count >= 1


def test_record_episode_passes_spawn_idx_as_reset_options(tmp_path):
    # Regression test: the default (non-scenario) mode must be able to pin
    # down a fixed spawn point, otherwise CarlaEnv's random-spawn safety
    # retry can land on a different spawn each run even with a fixed
    # reset_seed, breaking demo reproducibility.
    env = _recording_env()
    output = str(tmp_path / "demo.mp4")
    record_episode(
        _model(),
        env,
        output_path=output,
        fps=5,
        max_steps=3,
        reset_seed=42,
        spawn_idx=0,
    )
    env.reset.assert_called_with(seed=42, options={"spawn_idx": 0})


def test_record_episode_reset_options_none_without_spawn_idx(tmp_path):
    env = _recording_env()
    output = str(tmp_path / "demo.mp4")
    record_episode(_model(), env, output_path=output, fps=5, max_steps=3, reset_seed=42)
    env.reset.assert_called_with(seed=42, options=None)


def test_record_episode_calls_draw_obs_panel_per_frame(tmp_path, monkeypatch):
    spy = Mock()
    monkeypatch.setattr("src.ai.inference.rl_demo._draw_obs_panel", spy)

    env = _recording_env()
    output = str(tmp_path / "demo.mp4")
    record_episode(_model(), env, output_path=output, fps=5, max_steps=2)

    # once for the 1s starting pause frame + once per recorded driving frame
    assert spy.call_count == 3
    frame_bgr_arg, obs_arg, action_arg, cv2_arg = spy.call_args[0]
    assert frame_bgr_arg.shape == (88, 200, 3)
    np.testing.assert_array_equal(obs_arg, np.zeros(7, dtype=np.float32))
    np.testing.assert_array_equal(action_arg, _ACT)
    assert cv2_arg is cv2


def test_draw_route_map_card_returns_correct_shape_and_dtype():
    route = _route([(0.0, 0.0), (10.0, 5.0), (20.0, 0.0)])
    card = _draw_route_map_card(route, 200, 100, cv2)
    assert card.shape == (100, 200, 3)
    assert card.dtype == np.uint8


def test_draw_route_map_card_draws_something():
    route = _route([(0.0, 0.0), (10.0, 5.0), (20.0, 0.0)])
    card = _draw_route_map_card(route, 200, 100, cv2)
    background = np.full((100, 200, 3), (12, 12, 16), dtype=np.uint8)
    assert not np.array_equal(card, background)


def test_draw_route_map_card_empty_route_returns_blank_card():
    route = Route(waypoints=[], destination=Waypoint(0.0, 0.0, 0.0, 0.0))
    card = _draw_route_map_card(route, 200, 100, cv2)
    background = np.full((100, 200, 3), (12, 12, 16), dtype=np.uint8)
    np.testing.assert_array_equal(card, background)


# ---------------------------------------------------------------------------
# _draw_minimap
# ---------------------------------------------------------------------------


def test_draw_minimap_draws_something_on_wide_frame():
    route = _route([(0.0, 0.0), (10.0, 5.0), (20.0, 0.0)])
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    before = frame.copy()
    ego_location = Mock(x=10.0, y=2.0)

    _draw_minimap(frame, route, ego_location, 1280, 720, cv2)

    assert not np.array_equal(frame, before)


def test_draw_minimap_empty_route_is_noop():
    route = Route(waypoints=[], destination=Waypoint(0.0, 0.0, 0.0, 0.0))
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    before = frame.copy()
    ego_location = Mock(x=0.0, y=0.0)

    _draw_minimap(frame, route, ego_location, 1280, 720, cv2)

    np.testing.assert_array_equal(frame, before)


def test_draw_minimap_narrow_frame_is_noop():
    route = _route([(0.0, 0.0), (10.0, 5.0), (20.0, 0.0)])
    frame = np.zeros((88, 200, 3), dtype=np.uint8)
    before = frame.copy()
    ego_location = Mock(x=10.0, y=2.0)

    _draw_minimap(frame, route, ego_location, 200, 88, cv2)

    np.testing.assert_array_equal(frame, before)


def test_draw_minimap_no_destination_does_not_crash():
    waypoints = [
        Waypoint(x=0.0, y=0.0, z=0.0, yaw_deg=0.0),
        Waypoint(x=10.0, y=5.0, z=0.0, yaw_deg=0.0),
    ]
    route = Route(waypoints=waypoints, destination=None)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    ego_location = Mock(x=5.0, y=2.0)

    _draw_minimap(frame, route, ego_location, 1280, 720, cv2)  # must not raise


def test_record_episode_writes_route_map_card_frames(tmp_path, monkeypatch):
    written_frames = []

    class _FakeWriter:
        def __init__(self, *args, **kwargs):
            pass

        def write(self, frame):
            written_frames.append(frame)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoWriter", _FakeWriter)

    env = _recording_env()
    env.route = _route([(0.0, 0.0), (10.0, 5.0), (20.0, 0.0)])
    output = str(tmp_path / "demo.mp4")
    record_episode(
        _model(), env, output_path=output, fps=10, max_steps=2, route_map_seconds=1.0
    )

    # fps(10) * route_map_seconds(1.0) = 10 map-card frames, fps(10) * 1s pause = 10 pause
    # frames, + 2 driving frames (max_steps=2)
    assert len(written_frames) == 10 + 10 + 2


def test_record_episode_no_route_attribute_skips_map_card(tmp_path, monkeypatch):
    written_frames = []

    class _FakeWriter:
        def __init__(self, *args, **kwargs):
            pass

        def write(self, frame):
            written_frames.append(frame)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoWriter", _FakeWriter)

    env = _recording_env()
    del env.route  # simulate a gym.Env with no `.route` attribute (e.g. not a CarlaEnv)
    output = str(tmp_path / "demo.mp4")
    record_episode(
        _model(), env, output_path=output, fps=10, max_steps=2, route_map_seconds=1.0
    )

    # fps(10) * 1s pause = 10 pause frames + 2 driving frames, no map card
    assert len(written_frames) == 10 + 2


def test_record_episode_pause_frames_precede_driving_frames(tmp_path, monkeypatch):
    written_frames = []

    class _FakeWriter:
        def __init__(self, *args, **kwargs):
            pass

        def write(self, frame):
            written_frames.append(frame.copy())

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoWriter", _FakeWriter)

    env = _recording_env()
    output = str(tmp_path / "demo.mp4")
    record_episode(_model(), env, output_path=output, fps=5, max_steps=1)

    # fps(5) * 1s pause = 5 identical pause frames, then 1 driving frame
    assert len(written_frames) == 5 + 1
    pause_frames = written_frames[:5]
    for f in pause_frames[1:]:
        np.testing.assert_array_equal(f, pause_frames[0])


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


def test_load_model_calls_ppo_load(tmp_path, monkeypatch):
    mock_load = Mock(return_value=Mock())
    monkeypatch.setattr("src.ai.inference.rl_demo.PPO.load", mock_load)
    path = str(tmp_path / "ppo_model")
    load_model(path)
    mock_load.assert_called_once_with(path)


# ---------------------------------------------------------------------------
# _clear_spawn_area
# ---------------------------------------------------------------------------


def _clear_area_env(ego_id=1):
    """Mock env for _clear_spawn_area: env.ego.get_transform().location.distance(x)
    resolves via each mock location/spawn's `.dist` attribute (see _actor_at,
    _spawn_at). Actor filtering is wired separately via _wire_actors.
    """
    env = Mock()
    env.ego = Mock(id=ego_id)
    ego_loc = Mock()
    ego_loc.distance = Mock(side_effect=lambda other: other.dist)
    env.ego.get_transform.return_value.location = ego_loc
    return env, ego_loc


def _wire_actors(env, vehicles=None, walkers=None):
    """Wires env.world.get_actors().filter(pattern) to return `vehicles` for
    a vehicle.* pattern and `walkers` for a walker.pedestrian.* pattern."""
    env.world.get_actors.return_value.filter.side_effect = lambda pattern: (
        list(walkers or []) if "walker" in pattern else list(vehicles or [])
    )


def _actor_at(actor_id, dist):
    """A vehicle/pedestrian actor whose location is `dist` metres from the ego."""
    actor = Mock(id=actor_id)
    actor.get_location.return_value = Mock(dist=dist)
    return actor


def _spawn_at(dist):
    """A map spawn point whose location is `dist` metres from the ego."""
    sp = Mock()
    sp.location = Mock(dist=dist)
    return sp


def test_clear_spawn_area_relocates_actor_within_radius():
    env, _ = _clear_area_env()
    near_vehicle = _actor_at(2, dist=5.0)
    _wire_actors(env, vehicles=[near_vehicle])
    near_spawn = _spawn_at(dist=10.0)
    far_spawn = _spawn_at(dist=200.0)
    env.world.get_map.return_value.get_spawn_points.return_value = [
        near_spawn,
        far_spawn,
    ]

    _clear_spawn_area(env, radius_m=20.0)

    near_vehicle.set_transform.assert_called_once_with(far_spawn)
    assert env.world.tick.call_count == 10


def test_clear_spawn_area_leaves_distant_actor_untouched():
    env, _ = _clear_area_env()
    far_vehicle = _actor_at(2, dist=25.0)
    _wire_actors(env, vehicles=[far_vehicle])
    env.world.get_map.return_value.get_spawn_points.return_value = [
        _spawn_at(dist=100.0)
    ]

    _clear_spawn_area(env, radius_m=20.0)

    far_vehicle.set_transform.assert_not_called()


def test_clear_spawn_area_relocates_nearby_walker():
    env, _ = _clear_area_env()
    near_walker = _actor_at(3, dist=1.0)
    _wire_actors(env, walkers=[near_walker])
    far_spawn = _spawn_at(dist=200.0)
    env.world.get_map.return_value.get_spawn_points.return_value = [
        _spawn_at(dist=10.0),
        far_spawn,
    ]

    _clear_spawn_area(env, radius_m=20.0)

    near_walker.set_transform.assert_called_once_with(far_spawn)


def test_clear_spawn_area_excludes_ego_from_nearby_actors():
    env, _ = _clear_area_env(ego_id=1)
    ego_as_actor = _actor_at(1, dist=0.0)  # would be "within radius" of itself
    _wire_actors(env, vehicles=[ego_as_actor])
    env.world.get_map.return_value.get_spawn_points.return_value = [
        _spawn_at(dist=100.0)
    ]

    _clear_spawn_area(env, radius_m=20.0)

    ego_as_actor.set_transform.assert_not_called()


def test_clear_spawn_area_no_nearby_actors_is_noop():
    env, _ = _clear_area_env()
    _wire_actors(env, vehicles=[], walkers=[])
    env.world.get_map.return_value.get_spawn_points.return_value = [
        _spawn_at(dist=100.0)
    ]

    _clear_spawn_area(env, radius_m=20.0)

    env.world.tick.assert_not_called()


def test_clear_spawn_area_swallows_get_actors_failure():
    env, _ = _clear_area_env()
    env.world.get_actors.side_effect = RuntimeError("boom")

    _clear_spawn_area(env, radius_m=20.0)  # must not raise


def test_clear_spawn_area_spreads_multiple_actors_across_distinct_spawns():
    # Regression test for the physics pile-up bug: relocating several nearby
    # NPCs must not all dump them onto the single farthest spawn point.
    env, _ = _clear_area_env()
    near_actors = [_actor_at(actor_id, dist=1.0) for actor_id in range(2, 6)]
    _wire_actors(env, vehicles=near_actors)
    spawns = [_spawn_at(dist=d) for d in (50.0, 100.0, 150.0, 200.0)]
    env.world.get_map.return_value.get_spawn_points.return_value = spawns

    _clear_spawn_area(env, radius_m=20.0)

    targets = [a.set_transform.call_args[0][0] for a in near_actors]
    assert len(set(targets)) > 1


# ---------------------------------------------------------------------------
# pick_best_checkpoint
# ---------------------------------------------------------------------------


def _candidate(successes, off_route_pcts, dists=None):
    """A fake eval_model() result: `successes` scenarios succeed (success=True),
    the rest are Phase-2-style (success=None); off_route_pcts gives each
    scenario's off_route_pct, in order. len(off_route_pcts) must be >= successes.
    dists optionally gives each scenario's max_dist_from_start — omitted
    entirely when None, like results predating the field.
    """
    result = {}
    for i, off_route_pct in enumerate(off_route_pcts):
        result[f"scenario_{i}"] = {
            "success": True if i < successes else None,
            "off_route_pct": off_route_pct,
        }
        if dists is not None:
            result[f"scenario_{i}"]["max_dist_from_start"] = dists[i]
    return result


def test_pick_best_checkpoint_prefers_more_successes():
    all_results = {
        "0060k": _candidate(successes=1, off_route_pcts=[0.5, 0.5]),
        "0120k": _candidate(
            successes=2, off_route_pcts=[0.9, 0.9]
        ),  # worse off_route, more successes
    }
    assert pick_best_checkpoint(all_results) == "0120k"


def test_pick_best_checkpoint_breaks_ties_on_lower_off_route_pct():
    all_results = {
        "0060k": _candidate(successes=1, off_route_pcts=[0.5, 0.3]),  # mean 0.4
        "0120k": _candidate(
            successes=1, off_route_pcts=[0.1, 0.1]
        ),  # mean 0.1 -- lower, wins
    }
    assert pick_best_checkpoint(all_results) == "0120k"


def test_pick_best_checkpoint_single_candidate():
    all_results = {"best_model": _candidate(successes=0, off_route_pcts=[0.2])}
    assert pick_best_checkpoint(all_results) == "best_model"


def test_pick_best_checkpoint_all_phase2_falls_back_to_off_route_pct():
    all_results = {
        "0060k": _candidate(successes=0, off_route_pcts=[0.8, 0.8]),
        "0120k": _candidate(successes=0, off_route_pcts=[0.05, 0.05]),
    }
    assert pick_best_checkpoint(all_results) == "0120k"


def test_pick_best_checkpoint_zero_successes_prefers_distance_driven():
    """Regression: with zero successes everywhere, the old
    off_route-first tie-break crowned a parked policy — a car that never
    moves is never off-route by construction. Distance driven must rank
    above off-route."""
    all_results = {
        "0105k": _candidate(  # parked: spotless off-route, goes nowhere
            successes=0, off_route_pcts=[0.0, 0.0], dists=[0.5, 0.3]
        ),
        "0015k": _candidate(  # actually drives
            successes=0, off_route_pcts=[0.0, 0.0], dists=[40.0, 55.0]
        ),
    }
    assert pick_best_checkpoint(all_results) == "0015k"


def test_pick_best_checkpoint_successes_still_beat_distance():
    """A checkpoint that completes a scenario outranks one that merely
    drives far without ever succeeding."""
    all_results = {
        "0060k": _candidate(successes=1, off_route_pcts=[0.2, 0.2], dists=[5.0, 5.0]),
        "0120k": _candidate(successes=0, off_route_pcts=[0.0, 0.0], dists=[80.0, 80.0]),
    }
    assert pick_best_checkpoint(all_results) == "0060k"


# ---------------------------------------------------------------------------
# _draw_bboxes
# ---------------------------------------------------------------------------


def test_draw_bboxes_draws_something():
    objects = [
        DetectedObject(
            class_name=ObjectClass.VEHICLE,
            bbox=(10, 10, 50, 50),
            confidence=0.9,
            distance_m=12.0,
        )
    ]
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    before = frame.copy()

    _draw_bboxes(frame, objects, cv2)

    assert not np.array_equal(frame, before)


def test_draw_bboxes_empty_list_is_noop():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    before = frame.copy()

    _draw_bboxes(frame, [], cv2)

    np.testing.assert_array_equal(frame, before)


def test_draw_bboxes_handles_every_color_class_without_crashing():
    classes = [
        ObjectClass.VEHICLE,
        ObjectClass.WALKER,
        ObjectClass.RED_LIGHT,
        ObjectClass.YELLOW_LIGHT,
        ObjectClass.GREEN_LIGHT,
        ObjectClass.STOP,
        ObjectClass.YIELD,
        ObjectClass.SPEED_30,
        ObjectClass.UNKNOWN,
    ]
    objects = [
        DetectedObject(
            class_name=c, bbox=(5, 5, 40, 40), confidence=0.8, distance_m=10.0
        )
        for c in classes
    ]
    frame = np.zeros((200, 300, 3), dtype=np.uint8)

    _draw_bboxes(frame, objects, cv2)  # must not raise


def test_draw_bboxes_no_distance_omits_suffix():
    objects = [
        DetectedObject(
            class_name=ObjectClass.VEHICLE,
            bbox=(10, 10, 50, 50),
            confidence=0.9,
            distance_m=None,
        )
    ]
    frame = np.zeros((200, 300, 3), dtype=np.uint8)

    _draw_bboxes(frame, objects, cv2)  # must not raise formatting None as a float


def test_record_episode_calls_draw_bboxes_with_last_objects(tmp_path, monkeypatch):
    spy = Mock()
    monkeypatch.setattr("src.ai.inference.rl_demo._draw_bboxes", spy)

    env = _recording_env()
    env.last_objects = [
        DetectedObject(
            class_name=ObjectClass.VEHICLE,
            bbox=(0, 0, 5, 5),
            confidence=0.5,
            distance_m=None,
        )
    ]
    output = str(tmp_path / "demo.mp4")
    record_episode(_model(), env, output_path=output, fps=5, max_steps=2)

    # once for the 1s starting pause frame + once per recorded driving frame
    assert spy.call_count == 3
    frame_bgr_arg, objects_arg, cv2_arg = spy.call_args[0]
    assert objects_arg == env.last_objects
    assert cv2_arg is cv2


# ---------------------------------------------------------------------------
# eval_model
# ---------------------------------------------------------------------------


def _carla_loc(x, y, z=0.0):
    """A minimal carla.Location stand-in with real numeric x/y/z (arithmetic
    in eval_model's distance checks needs real floats, not auto-Mocks)."""
    return Mock(x=x, y=y, z=z)


def _carla_transform(x, y, z=0.0, yaw=0.0):
    return Mock(location=_carla_loc(x, y, z), rotation=Mock(yaw=yaw))


def _eval_env(ego_transforms, step_results, dest_xy):
    """Mock env for eval_model()'s per-scenario loop.

    ego_transforms: replayed in order by env.ego.get_transform() -- one call
        for the dest_spawn_idx route-replan lookup, one for `_start`, then one
        per env.step() call.
    step_results: (reward, terminated, truncated) tuples, one per env.step()
        call, replayed by env.step().
    dest_xy: (x, y) of the scenario's replanned destination -- wired through
        env.world.get_map().get_spawn_points()[dest_spawn_idx].location so
        the eval loop's own dest-reached distance check has something to
        compare against.
    """
    env = Mock()
    env.reset.return_value = (_OBS, {})
    env.step.side_effect = [(_OBS, r, te, tr, {}) for r, te, tr in step_results]
    env.ego.get_transform.side_effect = ego_transforms
    env._get_obs.return_value = _OBS
    dest_spawn = Mock(location=_carla_loc(*dest_xy))
    env.world.get_map.return_value.get_spawn_points.return_value = [dest_spawn]
    return env


def _dest_scenario(**overrides):
    """A Phase 1 scenario with a replanned destination (dest_spawn_idx=0),
    mirroring turn_left/turn_right/junction_straight in BENCHMARK_SCENARIOS."""
    base = dict(
        name="dest_test",
        spawn_idx=0,
        max_steps=5,
        phase=1,
        dest_spawn_idx=0,
        target_radius=15.0,
        success_fn=lambda m: bool(m.get("reached_dest")) and not m["terminated"],
    )
    base.update(overrides)
    return Scenario(**base)


def test_eval_model_reaching_dest_and_terminated_same_step_is_recorded_as_success(
    tmp_path,
):
    # Regression test for a real success getting misrecorded as a crash: the
    # env's own terminated=True fires for EITHER a collision OR the ego
    # reaching its internal destination check -- both produce the identical
    # boolean. Here, step 2 has the ego already within target_radius of
    # dest_loc *and* env.step() reports terminated=True on that same call,
    # simulating the internal destination-reached termination landing on the
    # exact step the eval loop's own distance check would also fire.
    ego_transforms = [
        _carla_transform(0.0, 0.0),  # dest_spawn_idx route-replan lookup
        _carla_transform(0.0, 0.0),  # `_start`
        _carla_transform(50.0, 50.0),  # step 1 -- still far from dest
        _carla_transform(100.0, 100.0),  # step 2 -- at dest, terminated=True too
    ]
    step_results = [
        (0.1, False, False),
        (-1.0, True, False),
    ]
    env = _eval_env(ego_transforms, step_results, dest_xy=(100.0, 100.0))
    sc = _dest_scenario()
    output = str(tmp_path / "eval.mp4")

    results = eval_model(
        _model(), env, output, scenarios=[sc], fps=2, render_fn=lambda: None
    )

    r = results["dest_test"]
    assert r["reached_dest"] is True
    assert r["terminated"] is False
    assert r["collision_step"] is None
    assert r["success"] is True


def test_eval_model_genuine_collision_far_from_dest_still_recorded_as_crash(
    tmp_path,
):
    # Complementary test: the fix must not suppress real crash detection when
    # the ego is nowhere near the destination.
    ego_transforms = [
        _carla_transform(0.0, 0.0),  # dest_spawn_idx route-replan lookup
        _carla_transform(0.0, 0.0),  # `_start`
        _carla_transform(5.0, 5.0),  # step 1 -- nowhere near dest, collision here
    ]
    step_results = [
        (-1.0, True, False),
    ]
    env = _eval_env(ego_transforms, step_results, dest_xy=(100.0, 100.0))
    sc = _dest_scenario()
    output = str(tmp_path / "eval.mp4")

    results = eval_model(
        _model(), env, output, scenarios=[sc], fps=2, render_fn=lambda: None
    )

    r = results["dest_test"]
    assert r["terminated"] is True
    assert r["collision_step"] == 1
    assert r["reached_dest"] is False
    assert r["success"] is False


def test_eval_model_cleanup_stops_pedestrian_controller_before_destroy(tmp_path):
    # Regression test: destroying a controller.ai.walker actor without first
    # calling .stop() on it is a known CARLA stability issue. Non-controller
    # actors (plain vehicles/walkers) must not get .stop() called on them.
    ego_transforms = [
        _carla_transform(0.0, 0.0),  # dest_spawn_idx route-replan lookup
        _carla_transform(0.0, 0.0),  # `_start`
        _carla_transform(5.0, 5.0),  # step 1 -- terminates immediately
    ]
    step_results = [
        (-1.0, True, False),
    ]
    env = _eval_env(ego_transforms, step_results, dest_xy=(100.0, 100.0))

    controller = Mock(type_id="controller.ai.walker", is_alive=True)
    walker = Mock(type_id="walker.pedestrian.0001", is_alive=True)
    sc = _dest_scenario(setup_fn=lambda world, ego: [controller, walker])
    output = str(tmp_path / "eval.mp4")

    eval_model(_model(), env, output, scenarios=[sc], fps=2, render_fn=lambda: None)

    assert [c[0] for c in controller.mock_calls] == ["stop", "destroy"]
    walker.stop.assert_not_called()
    walker.destroy.assert_called_once()
