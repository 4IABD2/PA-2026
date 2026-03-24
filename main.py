import carla
import random
import time

from const import *
from src.deep_reinforcement_model import DeepReinforcementModel
from src.gps import GPS
from src.matplot_visualizer import MatplotVisualizer


class Main:

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        world_to_get = random.choice(
            [x for x in self.client.get_available_maps() if "Town" in x]
        )
        print(f"Loading map: {world_to_get}")
        self.world = self.client.load_world(world_to_get)
        self.last_line = None

    def gps_navigation(self, vehicle, carla_map, spawn_points, world, start_point):
        route = None
        while route is None:
            gps = GPS(vehicle)
            dest_point = random.choice(spawn_points)
            road_graph = gps.extract_road_network(carla_map)
            route = gps.manual_a_star(
                road_graph, start_point.location, dest_point.location, carla_map
            )
        MatplotVisualizer.plot_plan(route)

        if ENABLE_GPS_DEBUG_LINE:
            for i in range(len(route) - 1):
                if self.last_line is not None:
                    world.debug.remove(self.last_line)
                self.last_line = world.debug.draw_line(
                    route[i].transform.location + carla.Location(z=1),
                    route[i + 1].transform.location + carla.Location(z=1),
                    thickness=0.2,
                    color=carla.Color(0, 255, 0),
                    life_time=500.0,
                )

        return route, gps

    def run(self):

        carla_map = self.world.get_map()

        vehicle = None
        other_vehicles = []

        ai_vehicle = DeepReinforcementModel()

        try:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
            spawn_points = carla_map.get_spawn_points()

            # traffic lights
            traffic_lights = self.world.get_actors().filter("traffic_light*")
            for light in traffic_lights:
                light.set_green_time(
                    random.uniform(5, MAX_TIME_TO_UPDATE_TRAFFIC_LIGHT_IN_SECONDE)
                )
                light.set_yellow_time(
                    random.uniform(2, MAX_TIME_TO_UPDATE_TRAFFIC_LIGHT_IN_SECONDE / 2)
                )
                light.set_red_time(
                    random.uniform(5, MAX_TIME_TO_UPDATE_TRAFFIC_LIGHT_IN_SECONDE)
                )

            # autonomous vehicle
            start_point = random.choice(spawn_points)
            vehicle = self.world.spawn_actor(vehicle_bp, start_point)

            # other vehicles
            for _ in range(NB_OTHER_VEHICLES):
                vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))
                spawn_point = random.choice(spawn_points)
                actor = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if actor is not None:
                    actor.set_autopilot(True)
                other_vehicles.append(actor)

            # gps navigation
            route, gps = self.gps_navigation(
                vehicle, carla_map, spawn_points, self.world, start_point
            )

            spectator = self.world.get_spectator()
            target_idx = 0

            start_time = time.time()

            while target_idx < len(route):
                target_wp = route[target_idx]

                if vehicle.get_location().distance(target_wp.transform.location) < 3.0:
                    target_idx += 1
                    if target_idx >= len(route):
                        break
                    target_wp = route[target_idx]

                control = gps.get_control(target_wp)
                # vehicle.apply_control(control)  # TODO UPDATE WITH DEEP REINFORCEMENT MODEL CONTROL
                input_ai = {"gps": gps.control_to_only_direction(control)}
                output_to_compute_error = {
                    "gps": gps.control_to_only_direction(control),
                    "vehicle_location": vehicle.get_location(),
                    "road_distance_to_vehicle": vehicle.get_location().distance(
                        target_wp.transform.location
                    ),
                    "is_blocked": any(
                        vehicle.get_location().distance(other.get_location()) < 3.0
                        for other in other_vehicles
                        if other is not None
                    ),
                }
                vehicle.apply_control(ai_vehicle.predict(input_ai))
                if TRAINING:
                    ai_vehicle.train(output_to_compute_error)
                    if (
                        time.time() - start_time
                        > MAX_TIME_TO_RESET_DURING_TRAINING_IN_SECONDE
                    ):
                        print("Resetting road...")
                        start_point = random.choice(spawn_points)
                        vehicle = self.world.spawn_actor(vehicle_bp, start_point)
                        route, gps = self.gps_navigation(
                            vehicle, carla_map, spawn_points, self.world, start_point
                        )
                        print("Road reset !")
                        start_time = time.time()

                v_trans = vehicle.get_transform()
                spectator.set_transform(
                    carla.Transform(
                        v_trans.location + carla.Location(z=4, x=-2), v_trans.rotation
                    )
                )

                time.sleep(0.05)

        finally:
            if vehicle:
                vehicle.destroy()
            for other in other_vehicles:
                if other is not None:
                    other.destroy()


if __name__ == "__main__":
    program = Main()
    program.run()
