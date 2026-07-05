import carla
import heapq
import math

from src.tools.matplot_visualizer import MatplotVisualizer
from src.interfaces.navigation_types import HighLevelCommand, Waypoint, Route


class Navigation:

    def __init__(self, vehicle, carla_map):
        self.vehicle = vehicle
        self.carla_map = carla_map
        self.index_way = 0
        self._graph = None

    @staticmethod
    def extract_road_network(carla_map, resolution=2.0):
        waypoints = carla_map.generate_waypoints(resolution)
        graph = {
            wp.id: {"waypoint": wp, "neighbors": wp.next(resolution)}
            for wp in waypoints
        }
        MatplotVisualizer.plot_road_network(graph)
        return graph

    @staticmethod
    def heuristic(wp1, wp2):
        return wp1.transform.location.distance(wp2.transform.location)

    @staticmethod
    def a_star(graph, start_wp, end_wp):
        open_set = []
        heapq.heappush(open_set, (0, start_wp.id, start_wp))
        came_from = {}
        g_score = {start_wp.id: 0}

        while open_set:
            _, current_id, current_wp = heapq.heappop(open_set)

            if current_wp.transform.location.distance(end_wp.transform.location) < 4.0:
                path = [current_wp]
                while current_id in came_from:
                    current_id, parent_wp = came_from[current_id]
                    path.append(parent_wp)
                path.reverse()
                return path

            neighbors = (
                graph[current_id]["neighbors"]
                if current_id in graph
                else current_wp.next(2.0)
            )
            for neighbor in neighbors:
                tentative_g = g_score[
                    current_id
                ] + current_wp.transform.location.distance(neighbor.transform.location)
                if neighbor.id not in g_score or tentative_g < g_score[neighbor.id]:
                    came_from[neighbor.id] = (current_id, current_wp)
                    g_score[neighbor.id] = tentative_g
                    f_cost = tentative_g + Navigation.heuristic(neighbor, end_wp)
                    heapq.heappush(open_set, (f_cost, neighbor.id, neighbor))
        return []

    def manual_a_star(self, graph, start_location, end_location):
        start_wp = self.carla_map.get_waypoint(start_location)
        end_wp = self.carla_map.get_waypoint(end_location)
        return self.a_star(graph, start_wp, end_wp)

    def get_control(self, target_waypoint: Waypoint):
        v_transform = self.vehicle.get_transform()
        v_loc = v_transform.location
        v_rot = v_transform.rotation.yaw

        target_loc = carla.Location(
            x=target_waypoint.x,
            y=target_waypoint.y,
            z=target_waypoint.z,
        )

        dy = target_loc.y - v_loc.y
        dx = target_loc.x - v_loc.x

        target_yaw = math.degrees(math.atan2(dy, dx))
        delta_yaw = target_yaw - v_rot

        while delta_yaw > 180:
            delta_yaw -= 360
        while delta_yaw < -180:
            delta_yaw += 360

        control = carla.VehicleControl()
        control.steer = max(-1.0, min(1.0, delta_yaw / 90.0))
        control.throttle = 0.5 if abs(delta_yaw) < 20 else 0.2
        control.brake = 0.0
        control.hand_brake = False
        return control

    @staticmethod
    def control_to_only_direction(control) -> HighLevelCommand:
        if control.steer < -0.1:
            return "left"
        elif control.steer > 0.1:
            return "right"
        else:
            return "straight"

    def plan(self, start, destination) -> Route:
        # The road network never changes for a given map — building it is
        # expensive (generates waypoints for the whole town + plots them),
        # so it's cached after the first plan() call instead of rebuilt on
        # every reset (CarlaEnv.reset() now replans on every episode).
        if self._graph is None:
            self._graph = self.extract_road_network(self.carla_map)
        path = self.manual_a_star(self._graph, start, destination)
        MatplotVisualizer.plot_plan(path)
        waypoints = [
            Waypoint(
                x=wp.transform.location.x,
                y=wp.transform.location.y,
                z=wp.transform.location.z,
                yaw_deg=wp.transform.rotation.yaw,
            )
            for wp in path
        ]
        return Route(waypoints=waypoints, destination=destination)

    def next_command(
        self, vehicle_position: Waypoint, route: Route
    ) -> HighLevelCommand:
        print(f"newt command: {self.index_way}")
        control = self.get_control(route.waypoints[self.index_way])
        self.index_way += 1
        return self.control_to_only_direction(control)
