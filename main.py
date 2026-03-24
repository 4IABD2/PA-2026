import carla
import random
import time

from const import *
from src.gps import GPS
from src.matplot_visualizer import MatplotVisualizer


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world_to_get = random.choice(
        [x for x in client.get_available_maps() if "Town" in x]
    )
    print(f"Loading map: {world_to_get}")
    world = client.load_world(world_to_get)
    carla_map = world.get_map()

    vehicle = None
    other_vehicles = []

    try:
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_points = carla_map.get_spawn_points()

        # traffic lights
        traffic_lights = world.get_actors().filter("traffic_light*")
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
        vehicle = world.spawn_actor(vehicle_bp, start_point)
        gps = GPS(vehicle)

        # other vehicles
        for _ in range(NB_OTHER_VEHICLES):
            vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))
            spawn_point = random.choice(spawn_points)
            actor = world.try_spawn_actor(vehicle_bp, spawn_point)
            if actor is not None:
                actor.set_autopilot(True)
            other_vehicles.append(actor)

        # gps navigation
        dest_point = random.choice(spawn_points)
        road_graph = gps.extract_road_network(carla_map)
        route = gps.manual_a_star(
            road_graph, start_point.location, dest_point.location, carla_map
        )

        MatplotVisualizer.plot_plan(route)

        if not route:
            print("Error: GPS")
            return

        if ENABLE_GPS_DEBUG_LINE:
            for i in range(len(route) - 1):
                world.debug.draw_line(
                    route[i].transform.location + carla.Location(z=1),
                    route[i + 1].transform.location + carla.Location(z=1),
                    thickness=0.2,
                    color=carla.Color(0, 255, 0),
                    life_time=500.0,
                )

        spectator = world.get_spectator()
        target_idx = 0

        while target_idx < len(route):
            target_wp = route[target_idx]

            if vehicle.get_location().distance(target_wp.transform.location) < 3.0:
                target_idx += 1
                if target_idx >= len(route):
                    break
                target_wp = route[target_idx]

            control = gps.get_control(target_wp)
            print(gps.control_to_only_direction(control))
            vehicle.apply_control(control)

            v_trans = vehicle.get_transform()
            spectator.set_transform(
                carla.Transform(
                    v_trans.location + carla.Location(z=4, x=-2), v_trans.rotation
                )
            )

            time.sleep(0.05)

        vehicle.apply_control(carla.VehicleControl(hand_brake=True))

    finally:
        if vehicle:
            vehicle.destroy()


if __name__ == "__main__":
    main()
