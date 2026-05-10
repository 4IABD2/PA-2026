"""CARLA inference demo for V1 PilotNet + speed.

Usage (basic):
    uv run -m src.ai.inference.carla_demo \
        --weights checkpoints/pilotnet_v1/best.keras \
        --town Town01 --weather ClearNoon --duration 120

Usage (with HUD frame recording for MP4 assembly):
    uv run -m src.ai.inference.carla_demo \
        --weights checkpoints/pilotnet_v1/best.keras \
        --duration 120 --record logs/demo_v1_<date>

Connects CARLA in synchronous mode, spawns ego Tesla Model 3, attaches the
team-shared RGB camera POV, runs the model each tick and applies controls
(with mutex throttle/brake + low-speed kickstart). On collision, the ego is
despawned, then respawned at another spawn point so the demo runs the full
duration. See src/ai/README.md for full args and post-processing rationale.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf
from tensorflow import keras

from PIL import Image, ImageDraw, ImageFont

from src.ai.config import (
    CROP_BOTTOM_PX,
    CROP_TOP_PX,
    IMAGE_HEIGHT,
    IMAGE_RAW_HEIGHT,
    IMAGE_WIDTH,
    SPEED_NORM_DIVISOR,
)
from src.dataset.camera_capture import CameraCapture

if TYPE_CHECKING:
    import carla  # noqa: F401

CARLA_FPS = 20
FIXED_DELTA_SECONDS = 1.0 / CARLA_FPS

KICKSTART_SPEED_KMH = 3.0
KICKSTART_THROTTLE = 0.6

RESPAWN_DELAY_S = 2.0


def _log(msg: str) -> None:
    print(f"[demo] {msg}", flush=True)


def _preprocess_image(rgb: np.ndarray) -> np.ndarray:
    """Apply the same crop + resize + normalize as the training data loader."""
    h = IMAGE_RAW_HEIGHT
    cropped = rgb[CROP_TOP_PX : h - CROP_BOTTOM_PX, :, :]
    resized = tf.image.resize(cropped, (IMAGE_HEIGHT, IMAGE_WIDTH)).numpy()
    return (resized.astype(np.float32) / 255.0)[None, ...]


def _speed_kmh(ego: "carla.Vehicle") -> float:
    v = ego.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z) * 3.6


_HUD_FONT: ImageFont.ImageFont | ImageFont.FreeTypeFont | None = None


def _hud_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    global _HUD_FONT
    if _HUD_FONT is None:
        try:
            _HUD_FONT = ImageFont.truetype("arial.ttf", 18)
        except (IOError, OSError):
            _HUD_FONT = ImageFont.load_default()
    return _HUD_FONT


def _render_hud(rgb: np.ndarray, hud: dict) -> Image.Image:
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[..., :3]
    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([(0, 0), (520, 90)], fill=(0, 0, 0, 170))
    font = _hud_font()
    line1 = (
        f"t={hud['t']:6.1f}s  v={hud['speed']:5.1f} km/h  resp={hud['resp']}"
    )
    draw.text((10, 4), line1, fill=(255, 255, 255), font=font)
    if hud.get("state") == "wait":
        line2 = f"WAIT {hud['wait_left']:.1f}s before respawn"
        draw.text((10, 32), line2, fill=(255, 200, 100), font=font)
    else:
        line2 = (
            f"raw  S={hud['raw_steer']:+.2f}  "
            f"T={hud['raw_throttle']:.2f}  B={hud['raw_brake']:.2f}"
        )
        line3 = (
            f"app  S={hud['steer']:+.2f}  "
            f"T={hud['throttle']:.2f}  B={hud['brake']:.2f}"
            + ("  [kick]" if hud.get("kickstart") else "")
        )
        draw.text((10, 32), line2, fill=(220, 220, 220), font=font)
        draw.text((10, 60), line3, fill=(180, 255, 180), font=font)
    return img


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CARLA demo for V1 PilotNet")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--town", default="Town01")
    parser.add_argument("--weather", default="ClearNoon")
    parser.add_argument("--duration", type=int, default=120, help="Seconds to run")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="If set, dump HUD-overlaid JPEG frames + console log to this dir",
    )
    args = parser.parse_args(argv)

    import carla

    record_dir = args.record
    if record_dir is not None:
        (record_dir / "frames").mkdir(parents=True, exist_ok=True)
        record_log = (record_dir / "demo.log").open("w", encoding="utf-8")
    else:
        record_log = None

    def _log(msg: str) -> None:  # noqa: F811 -- shadows module-level on purpose
        line = f"[demo] {msg}"
        print(line, flush=True)
        if record_log is not None:
            record_log.write(line + "\n")
            record_log.flush()

    _log(f"loading model {args.weights}")
    model = keras.models.load_model(args.weights)

    _log(f"connecting CARLA {args.host}:{args.port}")
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    world = client.load_world(args.town)
    weather = getattr(carla.WeatherParameters, args.weather)
    world.set_weather(weather)

    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    world.apply_settings(settings)

    bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    collision_bp = world.get_blueprint_library().find("sensor.other.collision")
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError(f"No spawn points on {args.town}")
    random.seed(0)

    def spawn_episode():
        e = world.spawn_actor(bp, random.choice(spawn_points))
        c = CameraCapture(world, e)
        c.attach()
        cs = world.spawn_actor(collision_bp, carla.Transform(), attach_to=e)
        flag = {"hit": False}
        cs.listen(lambda _ev: flag.__setitem__("hit", True))
        return e, c, cs, flag

    def destroy_episode(e, c, cs):
        cmds = []
        for sensor in (c._sensor if c is not None else None, cs):
            if sensor is None:
                continue
            try:
                sensor.stop()
            except Exception:
                pass
            cmds.append(carla.command.DestroyActor(sensor))
        if e is not None:
            cmds.append(carla.command.DestroyActor(e))
        if cmds:
            client.apply_batch_sync(cmds, True)

    ego = None
    camera = None
    collision_sensor = None
    try:
        ego, camera, collision_sensor, collision_flag = spawn_episode()
        _log("ego spawned")

        spectator = world.get_spectator()

        for _ in range(10):
            world.tick()
        _log("warm-up done, entering inference loop")

        target_ticks = int(args.duration * CARLA_FPS)
        log_every = CARLA_FPS  # 1×/sec
        respawn_delay_ticks = int(RESPAWN_DELAY_S * CARLA_FPS)
        wait_until_tick = -1  # -1 = not in collision wait
        respawn_count = 0
        frame_idx = 0
        wall_start = time.time()
        for tick in range(target_ticks):
            world.tick()
            t_now = tick / CARLA_FPS

            if collision_flag["hit"] and wait_until_tick < 0:
                wait_until_tick = tick + respawn_delay_ticks
                respawn_count += 1
                _log(
                    f"collision #{respawn_count} at t={t_now:.1f}s, "
                    f"respawn in {RESPAWN_DELAY_S:.0f}s"
                )

            if wait_until_tick >= 0:
                if tick >= wait_until_tick:
                    destroy_episode(ego, camera, collision_sensor)
                    ego, camera, collision_sensor, collision_flag = spawn_episode()
                    for _ in range(10):
                        world.tick()
                    wait_until_tick = -1
                    _log(f"respawned (#{respawn_count})")
                else:
                    ego.apply_control(
                        carla.VehicleControl(brake=1.0, hand_brake=True)
                    )
                    if record_dir is not None and camera._last_frame is not None:
                        hud = {
                            "t": t_now,
                            "speed": _speed_kmh(ego),
                            "resp": respawn_count,
                            "state": "wait",
                            "wait_left": (wait_until_tick - tick) / CARLA_FPS,
                        }
                        _render_hud(camera._last_frame, hud).save(
                            record_dir / "frames" / f"{frame_idx:06d}.jpg",
                            quality=85,
                        )
                        frame_idx += 1
                continue

            if camera._last_frame is None:
                continue

            image_in = _preprocess_image(camera._last_frame)
            speed_kmh = _speed_kmh(ego)
            speed_in = np.array([[speed_kmh / SPEED_NORM_DIVISOR]], dtype=np.float32)

            out = model({"image": image_in, "speed": speed_in}, training=False)
            raw_steer = float(out["steer"].numpy()[0, 0])
            raw_throttle = float(out["throttle"].numpy()[0, 0])
            raw_brake = float(out["brake"].numpy()[0, 0])
            steer, throttle, brake = raw_steer, raw_throttle, raw_brake

            # Mutex throttle/brake: the V1 model emits both heads simultaneously
            # because the autopilot demonstrations did. CARLA inhibits torque when
            # any brake is applied, so without this the car never moves.
            if throttle > brake:
                brake = 0.0
            else:
                throttle = 0.0

            # Kickstart override: 51% of training frames are at low speed where
            # the autopilot brakes, so the model learned a stable "stopped → brake"
            # attractor it can never escape on its own. Force a creep below
            # KICKSTART_SPEED_KMH to break the fixed point.
            kickstart_fired = False
            if speed_kmh < KICKSTART_SPEED_KMH:
                kickstart_fired = True
                throttle = KICKSTART_THROTTLE
                brake = 0.0

            ego.apply_control(
                carla.VehicleControl(
                    steer=steer,
                    throttle=throttle,
                    brake=brake,
                )
            )

            ego_tf = ego.get_transform()
            fwd = ego_tf.get_forward_vector()
            spectator.set_transform(
                carla.Transform(
                    ego_tf.location
                    + carla.Location(x=-6.0 * fwd.x, y=-6.0 * fwd.y, z=3.0),
                    carla.Rotation(pitch=-15.0, yaw=ego_tf.rotation.yaw),
                )
            )

            if record_dir is not None:
                hud = {
                    "t": t_now,
                    "speed": speed_kmh,
                    "resp": respawn_count,
                    "raw_steer": raw_steer,
                    "raw_throttle": raw_throttle,
                    "raw_brake": raw_brake,
                    "steer": steer,
                    "throttle": throttle,
                    "brake": brake,
                    "kickstart": kickstart_fired,
                }
                _render_hud(camera._last_frame, hud).save(
                    record_dir / "frames" / f"{frame_idx:06d}.jpg", quality=85
                )
                frame_idx += 1

            if tick % log_every == 0:
                _log(
                    f"t={t_now:5.1f}s speed={speed_kmh:5.1f} "
                    f"steer={steer:+.2f} throttle={throttle:.2f} brake={brake:.2f}"
                )

        _log(
            f"done in {time.time() - wall_start:.0f}s, respawns={respawn_count}"
        )
        if record_dir is not None:
            _log(f"recorded {frame_idx} frames in {record_dir / 'frames'}")
            _log(
                "to assemble: ffmpeg -framerate 20 "
                f"-i {record_dir / 'frames'}/%06d.jpg "
                f"-c:v libx264 -pix_fmt yuv420p {record_dir / 'demo.mp4'}"
            )
    finally:
        try:
            destroy_episode(ego, camera, collision_sensor)
        except Exception:
            pass
        try:
            world.apply_settings(original)
        except Exception:
            pass
        _log("cleanup done")
        if record_log is not None:
            record_log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
