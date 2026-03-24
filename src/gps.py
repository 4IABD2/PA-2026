import carla
import heapq
import math

from src.matplot_visualizer import MatplotVisualizer


class GPS:

    def __init__(self, vehicle):
        self.vehicle = vehicle

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

    def manual_a_star(self, graph, start_location, end_location, carla_map):
        start_wp = carla_map.get_waypoint(start_location)
        end_wp = carla_map.get_waypoint(end_location)

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
                    f_cost = tentative_g + self.heuristic(neighbor, end_wp)
                    heapq.heappush(open_set, (f_cost, neighbor.id, neighbor))
        return []

    def get_control(self, target_waypoint):
        v_transform = self.vehicle.get_transform()
        v_loc = v_transform.location
        v_rot = v_transform.rotation.yaw

        target_loc = target_waypoint.transform.location
        dy = target_loc.y - v_loc.y
        dx = target_loc.x - v_loc.x

        target_yaw = math.degrees(math.atan2(dy, dx))
        delta_yaw = target_yaw - v_rot

        while delta_yaw > 180:
            delta_yaw -= 360
        while delta_yaw < -180:
            delta_yaw += 360

        control = carla.VehicleControl()

        control.steer = delta_yaw / 90.0
        control.steer = max(-1.0, min(1.0, control.steer))

        control.throttle = 0.5 if abs(delta_yaw) < 20 else 0.2
        control.brake = 0.0
        control.hand_brake = False

        return control

    @staticmethod
    def control_to_only_direction(control):
        # return 0 for straight, -1 for left, 1 for right
        if control.steer < -0.1:
            return -1
        elif control.steer > 0.1:
            return 1
        else:
            return 0
