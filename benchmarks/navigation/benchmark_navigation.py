import datetime
import json
import os

import carla
import random
import time

from src.navigation.navigation import Navigation
from src.interfaces.navigation_types import Route


class Main:

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)

        self.last_line = None

    @staticmethod
    def init_navigation(
        vehicle, carla_map, spawn_points, world, start_point
    ) -> tuple[Route, Navigation]:
        route = None
        while route is None:
            nav = Navigation(vehicle, carla_map)
            dest_point = random.choice(spawn_points)
            route: Route = nav.plan(start_point.location, dest_point.location)

        return route, nav

    @staticmethod
    def save_result(total_time: list[float]):
        name = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_navigation"
        data = {
            "run_id": name,
            "n_samples": len(total_time),
            "metrics": {
                "mean_time": sum(total_time) / len(total_time),
                "min_time": min(total_time),
                "max_time": max(total_time),
                "all_time": sum(total_time),
                "times": total_time,
            },
        }
        print(data)
        os.makedirs("results", exist_ok=True)
        with open(f"results/{name}.json", "w") as f:
            json.dump(data, f)

    def run(self, nb_sample: int = 10):

        # autonomous vehicle

        # init gps navigation
        all_time = []
        for i in range(nb_sample):
            world_to_get = random.choice(
                [x for x in self.client.get_available_maps() if "Town" in x]
            )
            print(f"Loading map: {world_to_get}")
            world = self.client.load_world(world_to_get)
            blueprint_library = world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
            carla_map = world.get_map()
            spawn_points = carla_map.get_spawn_points()
            start_point = random.choice(spawn_points)
            vehicle = world.spawn_actor(vehicle_bp, start_point)

            start_time = time.time()
            route, navigation = self.init_navigation(
                vehicle, carla_map, spawn_points, world, start_point
            )
            total_time = time.time() - start_time
            all_time.append(total_time)
        self.save_result(all_time)


if __name__ == "__main__":
    benchmark = Main()
    benchmark.run(nb_sample=10)
