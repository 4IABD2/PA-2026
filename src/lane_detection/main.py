import carla
import cv2
import pandas as pd
import os
import queue
import argparse
import threading

from utils import (
    carla_to_bgr,
    make_canny_and_lines,
    get_labels,
    draw_overlay
)
from carla_integration import (
    setup_world,
    spawn_vehicle,
    spawn_camera,
    cleanup_actors
)

DEFAULT_OUTPUT_DIR = ("../dataset/lane_dataset")
DEFAULT_NUM_FRAMES = 1000

# Queue pour afficher les overlays en temps réel
overlay_display_queue = queue.Queue()


def display_overlays_thread():
    window_name = "Lane Detection - Overlay Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        try:
            overlay = overlay_display_queue.get(timeout=1.0)
            if overlay is None:  # Signal d'arrêt
                break
            cv2.imshow(window_name, overlay)
            cv2.waitKey(1)  # Petit délai pour rafraîchir l'affichage
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[ERROR] Erreur affichage overlay: {e}")
            break

    cv2.destroyAllWindows()


def start_overlay_display():
    display_thread = threading.Thread(target=display_overlays_thread, daemon=True)
    display_thread.start()
    return display_thread


def stop_overlay_display():
    overlay_display_queue.put(None)



def setup_directories(output_dir):
    rgb_dir = os.path.join(output_dir, "rgb")
    canny_dir = os.path.join(output_dir, "canny")
    overlay_dir = os.path.join(output_dir, "overlay")

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(canny_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    return rgb_dir, canny_dir, overlay_dir


def save_frame_data(frame_id, rgb_dir, canny_dir, overlay_dir, frame, canny, left_line, right_line, labels):
    filename = f"frame_{frame_id:06d}.png"

    rgb_path = os.path.join(rgb_dir, filename)
    canny_path = os.path.join(canny_dir, filename)
    overlay_path = os.path.join(overlay_dir, filename)

    overlay = draw_overlay(frame, left_line, right_line, labels)

    cv2.imwrite(rgb_path, frame)
    cv2.imwrite(canny_path, canny)
    cv2.imwrite(overlay_path, overlay)

    try:
        overlay_display_queue.put_nowait(overlay)
    except queue.Full:
        pass

    return {
        "image": rgb_path.replace("\\", "/"),
        "canny": canny_path.replace("\\", "/"),
        "left": round(labels["left"], 4),
        "right": round(labels["right"], 4),
        "signed_offset": round(labels["signed_offset"], 4),
        "aligner": bool(labels["aligner"]),
        "target_angle_deg": round(labels["target_angle_deg"], 4),
        "steer_label": round(labels["steer_label"], 4)
    }


def log_frame_info(frame_id, num_frames, labels):
    if frame_id % 50 == 0:
        print(
            f"[INFO] frame={frame_id}/{num_frames} "
            f"left={labels['left']:.3f} "
            f"right={labels['right']:.3f} "
            f"offset={labels['signed_offset']:.3f} "
            f"aligner={labels['aligner']} "
            f"angle={labels['target_angle_deg']:.2f} "
            f"steer={labels['steer_label']:.3f}"
        )


def generate_dataset(output_dir, num_frames):
    rgb_dir, canny_dir, overlay_dir = setup_directories(output_dir)
    csv_path = os.path.join(output_dir, "labels.csv")

    display_thread = start_overlay_display()

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = None
    vehicle = None
    camera = None
    original_settings = None

    image_queue = queue.Queue()
    rows = []

    try:
        world = client.get_world()
        original_settings = world.get_settings()

        world, traffic_manager = setup_world(client)

        vehicle = spawn_vehicle(world)
        vehicle.set_autopilot(True, traffic_manager.get_port())

        camera = spawn_camera(world, vehicle)
        camera.listen(lambda image: image_queue.put(image))

        # Laisser le monde se stabiliser
        for _ in range(30):
            world.tick()

        print("[INFO] Dataset generation started")
        print("[INFO] Fenêtre overlay ouverte (Lane Detection - Overlay Preview)")

        frame_id = 0
        while frame_id < num_frames:
            world.tick()

            try:
                image = image_queue.get(timeout=2.0)
            except queue.Empty:
                print("[WARNING] Image non reçue")
                continue

            frame = carla_to_bgr(image)
            canny, left_line, right_line = make_canny_and_lines(frame)

            labels = get_labels(world, vehicle)
            if labels is None:
                continue

            row = save_frame_data(frame_id, rgb_dir, canny_dir, overlay_dir, frame, canny, left_line, right_line, labels)
            rows.append(row)

            log_frame_info(frame_id, num_frames, labels)
            frame_id += 1

        # Sauvegarder le CSV
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        print("[OK] Dataset généré")
        print(f"[OK] CSV : {csv_path}")
        print(f"[OK] RGB : {rgb_dir}")
        print(f"[OK] Canny : {canny_dir}")
        print(f"[OK] Overlay : {overlay_dir}")

    finally:
        print("[INFO] Nettoyage")
        cleanup_actors(world, camera, vehicle, original_settings)
        stop_overlay_display()
        print("[INFO] Fin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", type=int, default=DEFAULT_NUM_FRAMES)

    args = parser.parse_args()

    generate_dataset(args.output, args.frames)