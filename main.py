import carla
import random
import time

from const import *
from src.model.deep_reinforcement_model import DeepReinforcementModel
from src.navigation.navigation import Navigation
from src.interfaces.navigation_types import HighLevelCommand, Route


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

    def init_navigation(self, vehicle, carla_map, spawn_points, world, start_point) -> tuple[Route, Navigation]:
        route = None
        while route is None:
            nav = Navigation(vehicle, carla_map)
            dest_point = random.choice(spawn_points)
            route: Route = nav.plan(
                start_point.location, dest_point.location
            )

        if ENABLE_GPS_DEBUG_LINE:
            for i in range(len(route) - 1):
                if self.last_line is not None:
                    world.debug.remove(self.last_line)
                carla_location_i0 = carla.Location(x=route.waypoints[i].x, y=route.waypoints[i].y,
                                                   z=route.waypoints[i].z)

                carla_location_i1 = carla.Location(x=route.waypoints[i + 1].x, y=route.waypoints[i + 1].y,
                                                   z=route.waypoints[i + 1].z)

                self.last_line = world.debug.draw_line(
                    carla_location_i0 + carla.Location(z=1),
                    carla_location_i1 + carla.Location(z=1),
                    thickness=0.2,
                    color=carla.Color(0, 255, 0),
                    life_time=500.0,
                )

            return route, nav

    def run(self):

        carla_map = self.world.get_map()
        number_reset = 0

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

            # init gps navigation
            route, navigation = self.init_navigation(
                vehicle, carla_map, spawn_points, self.world, start_point
            )

            spectator = self.world.get_spectator()
            target_idx = 0

            start_time = time.time()

            while target_idx < len(route):
                target_idx += 1
                navigation_command: HighLevelCommand = navigation.next_command(vehicle.get_location(), route)
                print(navigation_command)
                input_ai = {
                    "gps": 0 if navigation_command == "straight" else 1 if navigation_command == "right" else -1,
                    "center_left": 0,
                    "center_right": 0,
                    "distance_vehicle_in_front": -1,  # -1 if no vehicle in front
                    "distance_fire_light": -1,  # -1 if no fire or fire is green
                }
                output_to_compute_error = {
                    "control": navigation.get_control(route.waypoints[target_idx - 1]),
                    "is_blocked": any(
                        vehicle.get_location().distance(other.get_location()) < 3.0
                        for other in other_vehicles
                        if other is not None
                    ),
                }
                output_to_compute_error = output_to_compute_error | input_ai
                vehicle.apply_control(ai_vehicle.predict(input_ai))
                if TRAINING:
                    ai_vehicle.train(output_to_compute_error)
                    if (
                            time.time() - start_time
                            > MAX_TIME_TO_RESET_DURING_TRAINING_IN_SECONDE
                            + (number_reset * 2)
                    ):
                        print("Resetting road...")
                        number_reset += 1
                        start_point = random.choice(spawn_points)
                        vehicle = self.world.spawn_actor(vehicle_bp, start_point)
                        route, navigation = self.init_navigation(
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
        except Exception as error:
            print(error)
        finally:
            if vehicle:
                vehicle.destroy()
            for other in other_vehicles:
                if other is not None:
                    other.destroy()


if __name__ == "__main__":
    program = Main()
    program.run()
