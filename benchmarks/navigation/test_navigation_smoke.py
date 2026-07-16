from unittest.mock import Mock
from src.navigation.navigation import Navigation
from src.interfaces.navigation_types import HighLevelCommand, Route, Waypoint
import random


def test_navigation():
    # RANDOM GENERATION DATA
    num_waypoints = random.randint(3, 8)
    waypoint_ids = [f"wp_{i}" for i in range(num_waypoints)]

    waypoints = {}
    for wp_id in waypoint_ids:
        wp = Mock()
        wp.id = wp_id
        wp.transform.location.distance = Mock(return_value=random.uniform(0.0, 3.5))
        wp.next = Mock(return_value=[])
        waypoints[wp_id] = wp

    start_wp = waypoints[waypoint_ids[0]]
    end_wp = waypoints[waypoint_ids[-1]]

    end_wp.transform.location.distance = Mock(return_value=0.0)

    graph = {}
    for i, wp_id in enumerate(waypoint_ids):
        if i < len(waypoint_ids) - 1:
            neighbors = [waypoints[waypoint_ids[i + 1]]]
            if random.random() > 0.6:
                random_neighbor_idx = random.randint(i + 1, len(waypoint_ids) - 1)
                neighbors.append(waypoints[waypoint_ids[random_neighbor_idx]])
        else:
            neighbors = []

        graph[wp_id] = {"waypoint": waypoints[wp_id], "neighbors": neighbors}

    # TESTING
    result = Navigation.a_star(graph, start_wp, end_wp)

    assert isinstance(result, list), "a_star should return a list"
    assert len(result) > 0, "a_star should return a valid path"
    assert result[0] == start_wp, "Path should start with start waypoint"
    assert (
        result[-1].transform.location.distance(end_wp.transform.location) < 4.0
    ), "Path should end near the end waypoint"


# ---------------------------------------------------------------------------
# next_command: high-level command from the route's heading change (fixed to
# actually emit LEFT/RIGHT at turns, instead of always STRAIGHT).
# ---------------------------------------------------------------------------


def _turning_route(yaw_of):
    """Route straight in position (+x, 2 m spacing) but whose waypoint yaws
    follow yaw_of(i) — drives next_command's heading-change logic directly."""
    wps = [Waypoint(x=2.0 * i, y=0.0, z=0.0, yaw_deg=yaw_of(i)) for i in range(12)]
    return Route(waypoints=wps, destination=wps[-1])


def _command_for(route) -> HighLevelCommand:
    veh = Mock()
    veh.get_velocity.return_value = Mock(x=0.0, y=0.0, z=0.0)
    nav = Navigation(vehicle=veh, carla_map=Mock())
    nav.index_way = 0
    ego = Waypoint(x=2.0, y=0.0, z=0.0, yaw_deg=0.0)
    return nav.next_command(ego, route)


def test_next_command_straight_route_is_straight():
    assert _command_for(_turning_route(lambda i: 0.0)) == HighLevelCommand.STRAIGHT


def test_next_command_right_turn_emits_right():
    # route heading swings 0 -> +90 (CARLA yaw grows clockwise = right)
    route = _turning_route(lambda i: min(90.0, max(0.0, (i - 3) * 30.0)))
    assert _command_for(route) == HighLevelCommand.RIGHT


def test_next_command_left_turn_emits_left():
    route = _turning_route(lambda i: max(-90.0, min(0.0, -(i - 3) * 30.0)))
    assert _command_for(route) == HighLevelCommand.LEFT


def test_next_command_empty_route_is_lane_follow():
    assert _command_for(Route(waypoints=[], destination=None)) == (
        HighLevelCommand.LANE_FOLLOW
    )
