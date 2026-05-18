from unittest.mock import Mock
from src.navigation.navigation import Navigation
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
