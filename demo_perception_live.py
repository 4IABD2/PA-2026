"""Démo live : spawn aléatoire dans CARLA + YOLO + Depth en temps réel.

Spawn un véhicule à un point aléatoire (Town01 par défaut), l'autopilot le
conduit, et la PerceptionPipeline (YOLO + Depth Anything v2) tourne sur chaque
frame. Deux fenêtres OpenCV : détections (bboxes + classe + distance) et depth
colorisée.

Quitter : touche 'q' dans une fenêtre, ou Ctrl-C.

Prérequis :
    - CARLA lancé sur localhost:2000
    - poids YOLO dans src/perception/yolo/weights/best.pt
    - calibration depth dans src/perception/depth/calibration.json

Usage :
    uv run python demo_perception_live.py
    uv run python demo_perception_live.py --town Town04 --npcs 40 --device cuda
"""

from __future__ import annotations

import argparse
import random
import time

import carla
import cv2
import numpy as np
import pygame

from src.dataset.collection.sensors import CameraSensor, write_rgb
from src.perception.pipeline import PerceptionPipeline

# Couleurs BGR par classe (pour cv2).
_COLORS: dict[str, tuple[int, int, int]] = {
    "vehicle": (0, 255, 0),
    "walker": (255, 255, 0),
    "red_light": (0, 0, 255),
    "yellow_light": (0, 200, 255),
    "green_light": (0, 255, 128),
    "stop": (0, 0, 200),
    "yield": (200, 0, 200),
}
_SPEED_COLOR = (255, 128, 0)  # tous les panneaux de vitesse
_DEFAULT_COLOR = (200, 200, 200)


def _color(label: str) -> tuple[int, int, int]:
    if label.startswith("speed_"):
        return _SPEED_COLOR
    return _COLORS.get(label, _DEFAULT_COLOR)


def _draw(rgb: np.ndarray, objects: list) -> np.ndarray:
    """Dessine bboxes + classe + distance sur une copie RGB."""
    img = rgb.copy()
    for o in objects:
        x1, y1, x2, y2 = o.bbox
        c = _color(o.class_name.value)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        txt = f"{o.class_name.value} {o.confidence:.0%}"
        if o.distance_m is not None:
            txt += f" {o.distance_m:.0f}m"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw, y1), c, -1)
        cv2.putText(img, txt, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def _spawn_npcs(world, tm, n: int) -> list:
    bps = world.get_blueprint_library().filter("vehicle.*")
    spawns = world.get_map().get_spawn_points()
    random.shuffle(spawns)
    npcs = []
    for sp in spawns[:n]:
        v = world.try_spawn_actor(random.choice(bps), sp)
        if v is not None:
            v.set_autopilot(True, tm.get_port())
            npcs.append(v)
    return npcs


def main() -> None:
    parser = argparse.ArgumentParser(description="Démo live YOLO + Depth dans CARLA.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town01")
    parser.add_argument("--npcs", type=int, default=30)
    parser.add_argument("--device", default=None, help="cuda / cpu (defaut: auto)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    print(f"Chargement de {args.town}...")
    world = client.load_world(args.town)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    ego = None
    cam = None
    npcs: list = []
    try:
        bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        spawn = random.choice(world.get_map().get_spawn_points())
        ego = world.spawn_actor(bp, spawn)
        ego.set_autopilot(True, tm.get_port())
        npcs = _spawn_npcs(world, tm, args.npcs)
        print(f"Ego spawné + {len(npcs)} NPCs.")

        cam = CameraSensor(
            world,
            ego,
            "sensor.camera.rgb",
            write_rgb,
            width=args.width,
            height=args.height,
        )
        cam.attach()

        print("Chargement des modèles (Depth se télécharge au 1er run)...")
        pipe = PerceptionPipeline(device=args.device)

        # Fenêtre pygame : détections (bboxes + classe + distance) plein format.
        pygame.init()
        screen = pygame.display.set_mode((args.width, args.height))
        pygame.display.set_caption("CARLA — Detections  (q/Echap = quitter)")
        print(f"Device: {pipe.device}. Démo lancée — 'q' pour quitter.")

        for _ in range(10):  # warmup capteurs
            world.tick()

        fps_t, fps_n = time.time(), 0
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (
                    ev.type == pygame.KEYDOWN
                    and ev.key in (pygame.K_q, pygame.K_ESCAPE)
                ):
                    running = False
            if not running:
                break

            frame = world.tick()
            t0 = time.time()
            while not cam.has_frame(frame) and time.time() - t0 < 1.0:
                time.sleep(0.001)
            rgb = cam.last_rgb
            if rgb is None:
                continue

            objects, _ = pipe.perceive(rgb)

            det = _draw(rgb, objects)  # RGB, distances incluses dans les labels
            surf = pygame.surfarray.make_surface(det.swapaxes(0, 1))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            fps_n += 1
            if time.time() - fps_t >= 2.0:
                print(
                    f"  {fps_n / (time.time() - fps_t):.1f} FPS, {len(objects)} objets"
                )
                fps_t, fps_n = time.time(), 0
    finally:
        print("Nettoyage...")
        try:
            pygame.quit()
        except Exception:
            pass
        if cam is not None and cam._sensor is not None:
            cam._sensor.stop()
            cam._sensor.destroy()
        for actor in npcs + ([ego] if ego is not None else []):
            try:
                actor.destroy()
            except Exception:
                pass
        world.apply_settings(original)
        tm.set_synchronous_mode(False)
        print("Terminé.")


if __name__ == "__main__":
    main()
