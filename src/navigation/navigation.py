import carla
import heapq
import math

from src.tools.matplot_visualizer import MatplotVisualizer
from src.interfaces.navigation_types import HighLevelCommand, Waypoint, Route


class Navigation:

    def __init__(self, vehicle, carla_map):
        self.end_wp = None
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
        self.end_wp = self.carla_map.get_waypoint(end_location)
        return self.a_star(graph, start_wp, self.end_wp)

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

    def plan(self, start, destination) -> Route:
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

    @staticmethod
    def control_to_only_direction(control) -> HighLevelCommand:
        if control.steer < -0.4:
            return HighLevelCommand.LEFT
        elif control.steer > 0.4:
            return HighLevelCommand.RIGHT
        else:
            return HighLevelCommand.STRAIGHT

    def next_command(
        self, vehicle_position: Waypoint, route: Route
    ) -> HighLevelCommand:
        if not route.waypoints:
            return HighLevelCommand.LANE_FOLLOW

        def read_vehicle_pos():
            x = getattr(vehicle_position, "x", None)
            y = getattr(vehicle_position, "y", None)
            z = getattr(vehicle_position, "z", None)
            yaw = getattr(vehicle_position, "yaw_deg", None)

            if (
                yaw is None
                and hasattr(vehicle_position, "rotation")
                and hasattr(vehicle_position.rotation, "yaw")
            ):
                yaw = vehicle_position.rotation.yaw

            if x is None or y is None or z is None or yaw is None:
                try:
                    t = self.vehicle.get_transform()
                    loc = t.location
                    rot = t.rotation
                    x = x if x is not None else loc.x
                    y = y if y is not None else loc.y
                    z = z if z is not None else loc.z
                    yaw = yaw if yaw is not None else rot.yaw
                except Exception:
                    # last-resort zeroes
                    x = 0.0 if x is None else x
                    y = 0.0 if y is None else y
                    z = 0.0 if z is None else z
                    yaw = 0.0 if yaw is None else yaw

            return float(x), float(y), float(z), float(yaw)

        vx, vy, vz, yaw_deg = read_vehicle_pos()

        if getattr(self, "end_wp", None) is not None:
            try:
                end_loc = self.end_wp.transform.location
                dist_end = math.hypot(end_loc.x - vx, end_loc.y - vy)
                if dist_end < 3.0:
                    return HighLevelCommand.LANE_FOLLOW
            except Exception:
                # ignore if end_wp is malformed
                pass

        def dist2(a_x, a_y, a_z, b: Waypoint) -> float:
            dx = a_x - b.x
            dy = a_y - b.y
            dz = a_z - b.z
            return dx * dx + dy * dy + dz * dz

        yaw_rad = math.radians(yaw_deg)
        fwd_x = math.cos(yaw_rad)
        fwd_y = math.sin(yaw_rad)

        lookahead_m = 8.0
        try:
            vel = self.vehicle.get_velocity()
            speed = math.hypot(vel.x, vel.y, vel.z)
            lookahead_m = max(6.0, min(25.0, speed * 2.0))
        except Exception:
            speed = 0.0
        lookahead2 = lookahead_m * lookahead_m

        start_idx = min(max(0, self.index_way), len(route.waypoints) - 1)

        best_idx = start_idx
        best_dist2 = dist2(vx, vy, vz, route.waypoints[start_idx])

        ahead_candidates = []
        for i in range(start_idx, len(route.waypoints)):
            wp = route.waypoints[i]
            dx = wp.x - vx
            dy = wp.y - vy
            dot = fwd_x * dx + fwd_y * dy
            d2 = dx * dx + dy * dy
            if dot > 0:
                ahead_candidates.append((i, d2))
                if d2 <= lookahead2:
                    best_idx = i
                    best_dist2 = d2
                    break
            if d2 < best_dist2:
                best_dist2 = d2
                best_idx = i

        if ahead_candidates and best_idx == start_idx:
            best_ahead = min(ahead_candidates, key=lambda it: it[1])
            best_idx = best_ahead[0]

        if best_idx >= self.index_way:
            self.index_way = best_idx

        target_idx = min(self.index_way + 1, len(route.waypoints) - 1)
        target_wp = route.waypoints[target_idx]

        dist_to_target = math.sqrt(dist2(vx, vy, vz, target_wp))
        if dist_to_target < 1.5 and self.index_way < len(route.waypoints) - 1:
            self.index_way = min(self.index_way + 1, len(route.waypoints) - 1)
            target_idx = min(self.index_way + 1, len(route.waypoints) - 1)
            target_wp = route.waypoints[target_idx]

        control = self.get_control(target_wp)
        return self.control_to_only_direction(control)
