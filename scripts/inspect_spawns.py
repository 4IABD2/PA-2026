"""Interactive spawn point inspector — teleport to any spawn point and inspect it visually.

Lets you move a static vehicle between the map's spawn points one at a time,
so a candidate spawn_idx can be visually confirmed (no wall/obstacle right at
spawn, actually on the road) before it's used in BENCHMARK_SCENARIOS.

Usage:
    uv run python3 scripts/inspect_spawns.py --host <carla-ip>

Controls:
    Right arrow / n   : next spawn point
    Left arrow / p    : previous spawn point
    0-9 then Enter    : jump directly to that spawn index
    Backspace         : remove last typed digit
    q / Escape        : quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import carla
import pygame

from src.dataset.collection.sensors import CameraSensor, write_rgb
from scripts.explore_spawns import SpawnInfo, _analyse_spawns


def _format_info(info: SpawnInfo | None) -> list[str]:
    if info is None:
        return ["no heuristic data for this spawn point"]
    junction = (
        f"{info.junction_dist_m:.0f} m" if info.junction_dist_m < float("inf") else "-"
    )
    traffic_light = f"{info.tl_dist_m:.0f} m" if info.tl_dist_m < float("inf") else "-"
    speed_limit = str(info.speed_limit) if info.speed_limit else "-"
    return [
        f"curve: {info.curve_deg:.0f} deg {info.curve_dir}",
        f"junction dist: {junction}",
        f"traffic light dist: {traffic_light}",
        f"speed limit: {speed_limit}",
        f"tags: {', '.join(info.tags) or '-'}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive spawn point inspector.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()

    original = world.get_settings()

    ego = None
    cam = None
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        print(f"Map: {world.get_map().name}")
        print("Analysing spawn points …")
        infos = _analyse_spawns(world)
        info_by_idx = {info.idx: info for info in infos}
        spawn_points = world.get_map().get_spawn_points()
        print(f"  {len(spawn_points)} spawn points on this map.")

        bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        ego = world.spawn_actor(bp, spawn_points[0])

        cam = CameraSensor(
            world,
            ego,
            "sensor.camera.rgb",
            write_rgb,
            width=args.width,
            height=args.height,
        )
        cam.attach()

        pygame.init()
        pygame.font.init()
        font = pygame.font.SysFont("monospace", 18)
        screen = pygame.display.set_mode((args.width, args.height))
        pygame.display.set_caption(
            "Spawn inspector — arrows/n-p to move, digits+Enter to jump, q to quit"
        )

        for _ in range(10):
            world.tick()

        def _goto(idx: int) -> None:
            ego.set_transform(spawn_points[idx])
            ego.set_target_velocity(carla.Vector3D(0, 0, 0))

        current_idx = 0
        input_buffer = ""
        running = True

        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif ev.key in (pygame.K_RIGHT, pygame.K_n):
                        current_idx = (current_idx + 1) % len(spawn_points)
                        _goto(current_idx)
                    elif ev.key in (pygame.K_LEFT, pygame.K_p):
                        current_idx = (current_idx - 1) % len(spawn_points)
                        _goto(current_idx)
                    elif pygame.K_0 <= ev.key <= pygame.K_9:
                        input_buffer += str(ev.key - pygame.K_0)
                    elif ev.key == pygame.K_BACKSPACE:
                        input_buffer = input_buffer[:-1]
                    elif ev.key == pygame.K_RETURN and input_buffer:
                        current_idx = int(input_buffer) % len(spawn_points)
                        input_buffer = ""
                        _goto(current_idx)
            if not running:
                break

            frame = world.tick()
            t0 = time.time()
            while not cam.has_frame(frame) and time.time() - t0 < 1.0:
                time.sleep(0.001)
            rgb = cam.last_rgb
            if rgb is None:
                continue

            surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            screen.blit(surf, (0, 0))

            lines = [
                f"spawn_idx = {current_idx}",
                *_format_info(info_by_idx.get(current_idx)),
            ]
            if input_buffer:
                lines.append(f"jump to: {input_buffer}_")
            for i, line in enumerate(lines):
                text = font.render(line, True, (255, 255, 0))
                screen.blit(text, (8, 8 + i * 20))

            pygame.display.flip()
    finally:
        print("Cleaning up…")
        try:
            pygame.quit()
        except Exception:
            pass
        if cam is not None and cam._sensor is not None:
            try:
                cam._sensor.stop()
                cam._sensor.destroy()
            except Exception:
                pass
        if ego is not None:
            try:
                ego.destroy()
            except Exception:
                pass
        world.apply_settings(original)
        print("Done.")


if __name__ == "__main__":
    main()
