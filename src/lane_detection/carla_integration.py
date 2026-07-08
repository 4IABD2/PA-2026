import carla
import random
from utils import FPS, IMAGE_WIDTH, IMAGE_HEIGHT


def setup_world(client):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    return world, traffic_manager


def spawn_vehicle(world):
    blueprints = world.get_blueprint_library().filter("vehicle.*")
    vehicle_bp = random.choice(blueprints)

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is not None:
            return vehicle

    raise RuntimeError("Impossible de spawn le véhicule.")


def spawn_camera(world, vehicle):
    camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
    camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    camera_bp.set_attribute("fov", "90")
    camera_bp.set_attribute("sensor_tick", str(1.0 / FPS))

    transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=0.0))
    return world.spawn_actor(camera_bp, transform, attach_to=vehicle)


def cleanup_actors(world, camera, vehicle, original_settings):
    if camera is not None:
        camera.stop()
        camera.destroy()

    if vehicle is not None:
        vehicle.destroy()

    if world is not None and original_settings is not None:
        world.apply_settings(original_settings)
