"""Phase 1 du fine-tuning : genere un dataset CARLA labellise pour la
segmentation de voie.

Pour chaque frame on enregistre, parfaitement alignes (meme point de vue) :
  - rgb/      : l'image camera RGB (entree du modele)
  - lane/     : masque des LIGNES de voie   (tag semantique RoadLines = 24)
  - drivable/ : masque de la ZONE ROULABLE  (tag semantique Roads = 1)

Les labels viennent de la camera de SEGMENTATION SEMANTIQUE de CARLA -> labels
parfaits, gratuits. C'est de la vision (RGB -> masque) : aucune triche waypoint.

Lancer :  python generate_seg_dataset.py [--frames 2000] [--output DIR]
Astuce : relance plusieurs fois sur des Town/heures differentes pour varier.
"""

import argparse
import os
import queue
import sys

# ce script est dans test/ : on ajoute le dossier parent au path pour importer
# utils / carla_integration (qui restent au niveau production)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla
import cv2
import numpy as np

from utils import IMAGE_WIDTH, IMAGE_HEIGHT, FPS, carla_to_bgr
from carla_integration import setup_world, spawn_vehicle, cleanup_actors

ROADLINE_TAG = 24  # tag semantique des lignes de voie
ROAD_TAG = 1  # tag semantique de la route (zone roulable)

DEFAULT_OUTPUT = "../../dataset/lane_seg_dataset"
DEFAULT_FRAMES = 2000

CAM_TRANSFORM = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=0.0))


def _spawn_camera(world, vehicle, blueprint_id):
    bp = world.get_blueprint_library().find(blueprint_id)
    bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
    bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    bp.set_attribute("fov", "90")
    bp.set_attribute("sensor_tick", str(1.0 / FPS))
    return world.spawn_actor(bp, CAM_TRANSFORM, attach_to=vehicle)


def _semantic_tags(image):
    """Le canal R de l'image semantique brute contient l'ID du tag."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        (image.height, image.width, 4)
    )
    return arr[:, :, 2]


def generate(output, num_frames):
    rgb_dir = os.path.join(output, "rgb")
    lane_dir = os.path.join(output, "lane")
    drivable_dir = os.path.join(output, "drivable")
    for d in (rgb_dir, lane_dir, drivable_dir):
        os.makedirs(d, exist_ok=True)

    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)

    world = vehicle = rgb_cam = seg_cam = original = None
    rgb_q, seg_q = queue.Queue(), queue.Queue()

    # on demarre la numerotation apres les frames deja presentes (cumul des runs)
    start = len([f for f in os.listdir(rgb_dir) if f.endswith(".png")])

    try:
        world = client.get_world()
        original = world.get_settings()
        world, tm = setup_world(client)

        vehicle = spawn_vehicle(world)
        vehicle.set_autopilot(True, tm.get_port())

        rgb_cam = _spawn_camera(world, vehicle, "sensor.camera.rgb")
        seg_cam = _spawn_camera(world, vehicle, "sensor.camera.semantic_segmentation")
        rgb_cam.listen(rgb_q.put)
        seg_cam.listen(seg_q.put)

        for _ in range(30):
            world.tick()
            try:
                rgb_q.get(timeout=2.0)
                seg_q.get(timeout=2.0)
            except queue.Empty:
                pass

        print(f"[INFO] generation -> {output} (a partir de #{start})")
        saved = 0
        while saved < num_frames:
            world.tick()
            try:
                rgb_img = rgb_q.get(timeout=2.0)
                seg_img = seg_q.get(timeout=2.0)
            except queue.Empty:
                print("[WARN] image manquante")
                continue

            frame = carla_to_bgr(rgb_img)
            tags = _semantic_tags(seg_img)
            lane = (tags == ROADLINE_TAG).astype(np.uint8) * 255
            drivable = (tags == ROAD_TAG).astype(np.uint8) * 255

            idx = start + saved
            name = f"frame_{idx:06d}.png"
            cv2.imwrite(os.path.join(rgb_dir, name), frame)
            cv2.imwrite(os.path.join(lane_dir, name), lane)
            cv2.imwrite(os.path.join(drivable_dir, name), drivable)

            saved += 1
            if saved % 100 == 0:
                print(f"[INFO] {saved}/{num_frames}")

        print(f"[OK] {saved} frames -> {output}")

    finally:
        print("[INFO] nettoyage")
        for cam in (rgb_cam, seg_cam):
            if cam is not None:
                cam.stop()
                cam.destroy()
        cleanup_actors(world, None, vehicle, original)
        print("[INFO] fin")


def run_towns(output, towns, frames_per_town):
    """Charge chaque Town puis genere frames_per_town images (diversite max)."""
    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    for i, town in enumerate(towns):
        print(f"\n===== TOWN {i + 1}/{len(towns)} : {town} =====")
        client.load_world(town)  # bloquant jusqu'au chargement
        generate(output, frames_per_town)  # gere sync + stabilisation + cleanup


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help="frames par Town si --towns, sinon total",
    )
    p.add_argument(
        "--towns",
        default="",
        help="liste de Town separees par des virgules (ex: Town01,Town03,Town05)",
    )
    args = p.parse_args()
    if args.towns:
        run_towns(args.output, [t.strip() for t in args.towns.split(",")], args.frames)
    else:
        generate(args.output, args.frames)
