import os
import queue

import carla
import cv2

from utils import carla_to_bgr
from carla_integration import (
    setup_world,
    spawn_vehicle,
    spawn_camera,
    cleanup_actors,
)
from lane_perception import LaneDetector, draw_overlay

DEBUG_DIR = "debug_frames"
WINDOW = "Lane Highlight (RGB)"


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = vehicle = camera = original_settings = None
    image_queue = queue.Queue()

    try:
        world = client.get_world()
        original_settings = world.get_settings()

        world, traffic_manager = setup_world(client)

        vehicle = spawn_vehicle(world)
        vehicle.set_autopilot(True, traffic_manager.get_port())

        camera = spawn_camera(world, vehicle)
        camera.listen(lambda image: image_queue.put(image))

        for _ in range(30):
            world.tick()

        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        print("Echap = quitter, s = sauver une frame.")

        detector = LaneDetector()

        saved = 0
        while True:
            world.tick()

            try:
                image = image_queue.get(timeout=2.0)
            except queue.Empty:
                print("[WARNING] Image non recue")
                continue

            frame = carla_to_bgr(image)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect(rgb)
            overlay = draw_overlay(frame, result)

            cv2.imshow(WINDOW, overlay)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = os.path.join(DEBUG_DIR, f"raw_{saved:04d}.png")
                cv2.imwrite(path, frame)
                saved += 1
                print(f"[OK] frame sauvee : {path}")

    finally:
        cleanup_actors(world, camera, vehicle, original_settings)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
