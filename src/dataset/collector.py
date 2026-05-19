"""DatasetCollector — orchestrateur de collecte CARLA.

Single-threaded, CARLA mode synchrone à 20 FPS. Capture une frame toutes les
``capture_every_n_ticks`` ticks (40 par défaut = 2s).

Architecture : la classe orchestre 4 sensors via la spec déclarative
``sensors.SENSOR_SPECS``. Toute la logique de capture/écriture est déléguée
aux writers, le collector ne fait que la séquence setup → tick loop → cleanup.
"""

from __future__ import annotations

import random
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from src.dataset.command_planner import CommandPlanner
from src.dataset.encodings import CAMERA_LOCATION, CAMERA_ROTATION_PITCH
from src.dataset.expert_driver import ExpertDriver
from src.dataset.manifest_writer import ManifestWriter
from src.dataset.sensors import SENSOR_SPECS, CameraSensor
from src.dataset.yolo_labels import YoloLabeler

if TYPE_CHECKING:
    import carla  # noqa: F401

CARLA_FPS = 20
FIXED_DELTA_SECONDS = 1.0 / CARLA_FPS
COLLISION_LOOKBACK_FRAMES = 5
# Timeout pour attendre que tous les sensors aient reçu leur frame du tick
# courant. Les callbacks Python tournent sur des threads séparés, on doit
# attendre activement la synchro avant de sauver.
SENSOR_SYNC_TIMEOUT_S = 2.0
SENSOR_SYNC_POLL_S = 0.001


def _log(msg: str) -> None:
    print(f"[collector] {msg}", flush=True)


class DatasetCollector:
    """Orchestrateur de collecte CARLA (une instance par run)."""

    def __init__(
        self,
        output_dir: str | Path,
        town: str = "Town01",
        weather: str = "ClearNoon",
        n_npc_vehicles: int = 40,
        n_npc_walkers: int = 30,
        duration_sec: int = 1800,
        capture_every_n_ticks: int = 40,
        host: str = "localhost",
        port: int = 2000,
        seed: int | None = None,
        image_width: int = 1280,
        image_height: int = 720,
        fov: int = 90,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.town = town
        self.weather = weather
        self.n_npc_vehicles = n_npc_vehicles
        self.n_npc_walkers = n_npc_walkers
        self.duration_sec = duration_sec
        self.capture_every_n_ticks = capture_every_n_ticks
        self.host = host
        self.port = port
        self.seed = seed
        self.image_width = image_width
        self.image_height = image_height
        self.fov = fov

        self._validate_output_dir()

        # État interne, initialisé dans _setup()
        self._client: "carla.Client | None" = None
        self._world: "carla.World | None" = None
        self._traffic_manager: "carla.TrafficManager | None" = None
        self._ego: "carla.Vehicle | None" = None
        self._npc_actors: list = []
        self._sensors: dict[str, CameraSensor] = {}
        self._yolo_labeler: YoloLabeler | None = None
        self._command_planner: CommandPlanner | None = None
        self._expert: ExpertDriver | None = None
        self._collision_sensor = None
        self._collision_events_recent: deque[int] = deque(
            maxlen=COLLISION_LOOKBACK_FRAMES
        )
        self._manifest: ManifestWriter | None = None
        self._original_settings = None

    def _validate_output_dir(self) -> None:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise ValueError(
                f"output_dir is not empty: {self.output_dir}. "
                f"Choose a new path or empty it manually."
            )

    # ------------------------------------------------------------------------

    def _setup(self) -> None:
        import carla

        if self.seed is not None:
            random.seed(self.seed)

        _log(f"connecting to {self.host}:{self.port}")
        self._client = carla.Client(self.host, self.port)
        self._client.set_timeout(10.0)
        _log(f"loading {self.town}")
        self._world = self._client.load_world(self.town)

        weather_preset = getattr(carla.WeatherParameters, self.weather)
        self._world.set_weather(weather_preset)

        self._original_settings = self._world.get_settings()
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self._world.apply_settings(settings)

        self._traffic_manager = self._client.get_trafficmanager()
        self._traffic_manager.set_synchronous_mode(True)
        if self.seed is not None:
            self._traffic_manager.set_random_device_seed(self.seed)

        self._world.tick()

        self._ego = self._spawn_ego()
        self._npc_actors = self._spawn_npcs()
        _log(f"spawned ego + {len(self._npc_actors)}/{self.n_npc_vehicles} NPCs")

        # Sensors via spec déclarative
        for key, blueprint, writer in SENSOR_SPECS:
            sensor = CameraSensor(
                self._world,
                self._ego,
                blueprint=blueprint,
                writer=writer,
                width=self.image_width,
                height=self.image_height,
                fov=self.fov,
            )
            sensor.attach()
            self._sensors[key] = sensor

        # Collision sensor (séparé : pas dans SENSOR_SPECS car logique d'écriture
        # différente — il pousse un événement, pas un frame complet).
        bp = self._world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            bp, carla.Transform(), attach_to=self._ego
        )
        self._collision_events_recent.clear()
        self._collision_sensor.listen(self._on_collision)

        _log("sensors attached, warming up")

        self._yolo_labeler = YoloLabeler(
            self._world,
            self._ego,
            instance_sensor=self._sensors["instance"],
            image_w=self.image_width,
            image_h=self.image_height,
        )
        self._command_planner = CommandPlanner(self._world, self._ego)
        self._expert = ExpertDriver(self._ego, self._traffic_manager)

        for _ in range(10):
            self._world.tick()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = ManifestWriter(self.output_dir, self.town, self.weather)
        _log(f"ready, writing to {self.output_dir}")

    def _spawn_ego(self) -> "carla.Vehicle":
        bp = self._world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        spawn_points = self._world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError(f"No spawn points on map {self.town}")
        spawn = random.choice(spawn_points)
        return self._world.spawn_actor(bp, spawn)

    def _spawn_npcs(self) -> list:
        """Spawn N NPC vehicles (best effort). Walkers : TODO."""
        npcs = []
        bp_lib = self._world.get_blueprint_library()
        vehicle_bps = bp_lib.filter("vehicle.*")
        spawn_points = self._world.get_map().get_spawn_points()

        for i in range(min(self.n_npc_vehicles, len(spawn_points) - 1)):
            bp = random.choice(vehicle_bps)
            try:
                npc = self._world.try_spawn_actor(bp, spawn_points[i + 1])
                if npc is not None:
                    npc.set_autopilot(True, self._traffic_manager.get_port())
                    npcs.append(npc)
            except Exception:
                continue
        return npcs

    def _on_collision(self, event) -> None:
        if self._collision_events_recent:
            self._collision_events_recent[-1] = 1

    def _wait_sensors_sync(self, expected_frame: int) -> None:
        """Bloque jusqu'à ce que tous les sensors aient leur buffer à la
        frame ``expected_frame``. Warn et continue si timeout.
        """
        deadline = time.time() + SENSOR_SYNC_TIMEOUT_S
        while True:
            if all(s.has_frame(expected_frame) for s in self._sensors.values()):
                return
            if time.time() > deadline:
                missing = [
                    k
                    for k, s in self._sensors.items()
                    if not s.has_frame(expected_frame)
                ]
                _log(
                    f"WARN: sensors not synced at frame {expected_frame} after "
                    f"{SENSOR_SYNC_TIMEOUT_S}s (missing: {missing})"
                )
                return
            time.sleep(SENSOR_SYNC_POLL_S)

    # ------------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Libère les actors CARLA et restaure les settings.

        Ordre critique : stop listeners → batch-destroy synchrone → null refs.
        Sequential destroy crashe le runtime C++ si une callback est en vol ;
        null les refs Python avant confirmation du serveur déclenche des
        warnings "sensor went out of scope".
        """
        import carla

        if self._client is not None:
            _log("cleanup: stopping listeners")
            for sensor in self._sensors.values():
                if sensor._sensor is not None:
                    try:
                        sensor._sensor.stop()
                    except Exception:
                        pass
            if self._collision_sensor is not None:
                try:
                    self._collision_sensor.stop()
                except Exception:
                    pass

            destroy_cmds = []
            if self._collision_sensor is not None:
                destroy_cmds.append(carla.command.DestroyActor(self._collision_sensor))
            for sensor in self._sensors.values():
                if sensor._sensor is not None:
                    destroy_cmds.append(carla.command.DestroyActor(sensor._sensor))
            for npc in self._npc_actors:
                destroy_cmds.append(carla.command.DestroyActor(npc))
            if self._ego is not None:
                destroy_cmds.append(carla.command.DestroyActor(self._ego))
            if destroy_cmds:
                _log(f"cleanup: destroying {len(destroy_cmds)} actors")
                self._client.apply_batch_sync(destroy_cmds, True)

            for sensor in self._sensors.values():
                sensor._sensor = None
            self._collision_sensor = None

        if self._world is not None and self._original_settings is not None:
            try:
                self._world.apply_settings(self._original_settings)
            except Exception:
                pass
        if self._traffic_manager is not None:
            try:
                self._traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass
        _log("cleanup done")

    # ------------------------------------------------------------------------

    def run(self) -> None:
        """Setup → tick loop → cleanup. Bloquant.

        Capture un frame complet (image + depth + semantic + instance + labels
        + manifest) tous les ``capture_every_n_ticks`` ticks. Cleanup garanti
        via try/finally.
        """
        try:
            self._setup()
            assert self._world is not None
            assert self._manifest is not None

            target_ticks = int(self.duration_sec * CARLA_FPS)
            target_frames = target_ticks // self.capture_every_n_ticks
            tick_count = 0
            frame_id = 0
            run_start = self._world.get_snapshot().timestamp.elapsed_seconds
            wall_start = time.time()
            _log(
                f"capture loop: target {target_frames} frames ({target_ticks} ticks)"
            )

            while tick_count < target_ticks:
                self._world.tick()
                tick_count += 1

                if tick_count % 200 == 0:
                    pct = 100 * tick_count / target_ticks
                    elapsed = time.time() - wall_start
                    _log(
                        f"tick {tick_count}/{target_ticks} ({pct:.0f}%, {elapsed:.0f}s)"
                    )

                if tick_count % self.capture_every_n_ticks != 0:
                    continue

                # Avance la fenêtre de collision (0 par défaut, callback écrit 1).
                self._collision_events_recent.append(0)

                # Synchro : attendre que tous les sensors aient reçu la frame
                # du tick courant. Sans ça, les callbacks Python (threads
                # séparés) peuvent encore avoir le buffer du tick précédent.
                expected_frame = self._world.get_snapshot().frame
                self._wait_sensors_sync(expected_frame)

                # Sauvegarde toutes les modalités via writers.
                for sensor in self._sensors.values():
                    sensor.save(self.output_dir, frame_id)

                # Labels YOLO (peuvent ne rien écrire si la frame n'a pas d'objet).
                self._yolo_labeler.save(
                    self.output_dir / "labels_yolo" / f"{frame_id:06d}.txt"
                )

                expert_controls = self._expert.read_controls()
                command = self._command_planner.current_command()
                ts = self._world.get_snapshot().timestamp.elapsed_seconds - run_start
                is_collision = sum(self._collision_events_recent) > 0

                self._manifest.append_row(
                    frame_id=frame_id,
                    image_path=f"images/{frame_id:06d}.jpg",
                    timestamp=ts,
                    command=command,
                    expert=expert_controls,
                    is_collision=is_collision,
                )
                frame_id += 1
                _log(
                    f"frame {frame_id}/{target_frames} "
                    f"({expert_controls.speed_kmh:.1f} km/h, {command.value})"
                )

            run_end = self._world.get_snapshot().timestamp.elapsed_seconds
            _log(
                f"capture done: {frame_id} frames in {time.time() - wall_start:.0f}s"
            )
            self._manifest.flush_csv()
            self._manifest.write_metadata(
                run_id=self.output_dir.name,
                carla_version=str(self._client.get_server_version()),
                fps=CARLA_FPS,
                capture_every_n_ticks=self.capture_every_n_ticks,
                n_frames=frame_id,
                n_npc_vehicles=len(self._npc_actors),
                n_npc_walkers=0,  # TODO walkers
                duration_sec_target=self.duration_sec,
                duration_sec_actual=round(run_end - run_start, 2),
                camera_pov={
                    "location": list(CAMERA_LOCATION),
                    "rotation": [0, 0, CAMERA_ROTATION_PITCH],
                },
                image_resolution=[self.image_width, self.image_height],
                fov=self.fov,
                seed=self.seed,
                available_modalities=["rgb", "depth", "semantic", "instance"],
            )
        finally:
            self._cleanup()
