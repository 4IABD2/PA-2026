from __future__ import annotations

import random
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from src.dataset.collection.command_planner import CommandPlanner
from src.dataset.collection.expert_driver import ExpertDriver
from src.dataset.collection.manifest_writer import ManifestWriter
from src.dataset.collection.sensors import CameraSensor, SENSOR_SPECS
from src.dataset.encodings import CAMERA_LOCATION, CAMERA_ROTATION_PITCH
from src.dataset.labeling.yolo_labels import YoloLabeler

if TYPE_CHECKING:
    import carla  # noqa: F401

CARLA_FPS = 20
FIXED_DELTA_SECONDS = 1.0 / CARLA_FPS
COLLISION_LOOKBACK_FRAMES = 5


def _log(msg: str) -> None:
    print(f"[collector] {msg}", flush=True)


class DatasetCollector:
    """Collection orchestrator. Single instance per run."""

    def __init__(
        self,
        output_dir: str | Path,
        town: str = "Town01",
        weather: str = "ClearNoon",
        n_npc_vehicles: int = 40,
        duration_sec: int = 1800,
        capture_every_n_ticks: int = 40,
        host: str = "localhost",
        port: int = 2000,
        seed: int | None = None,
        image_width: int = 1280,
        image_height: int = 720,
        fov: int = 90,
        timeout_sec: float = 60.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.town = town
        self.weather = weather
        self.n_npc_vehicles = n_npc_vehicles
        self.duration_sec = duration_sec
        self.capture_every_n_ticks = capture_every_n_ticks
        self.host = host
        self.port = port
        self.seed = seed
        self.image_width = image_width
        self.image_height = image_height
        self.fov = fov
        self.timeout_sec = timeout_sec

        self._validate_output_dir()

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
        """Refuse a non-empty output_dir to avoid overwriting a previous run."""
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise ValueError(
                f"output_dir is not empty: {self.output_dir}. "
                f"Choose a new path or empty it manually."
            )

    def _setup(self) -> None:
        """Connect to CARLA, set sync mode, spawn ego + NPCs + sensors + writers."""
        import carla

        if self.seed is not None:
            random.seed(self.seed)

        _log(f"connecting to {self.host}:{self.port}")
        self._client = carla.Client(self.host, self.port)
        self._client.set_timeout(self.timeout_sec)
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

        self._sensors_attach()
        _log("sensors attached, warming up")

        self._yolo_labeler = YoloLabeler(
            self._world,
            self._ego,
            self._sensors["rgb"]._sensor,
            self._sensors["instance"],
            image_w=self.image_width,
            image_h=self.image_height,
            fov=self.fov,
        )
        self._command_planner = CommandPlanner(self._world, self._ego)
        self._expert = ExpertDriver(self._ego, self._traffic_manager)

        for _ in range(10):
            self._world.tick()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = ManifestWriter(self.output_dir, self.town, self.weather)
        _log(f"ready, writing to {self.output_dir}")

    def _spawn_ego(self) -> "carla.Vehicle":
        """Spawn an ego vehicle at a random spawn point of the map."""
        bp = self._world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        spawn_points = self._world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError(f"No spawn points on map {self.town}")
        spawn = random.choice(spawn_points)
        ego = self._world.spawn_actor(bp, spawn)
        return ego

    def _spawn_npcs(self) -> list:
        """Spawn N NPC vehicles (best effort, may spawn fewer if no room).

        Vehicles only : les walkers demandent un WalkerAIController par piéton,
        non implémenté. ``metadata.json`` reporte donc toujours
        ``n_npc_walkers: 0``.
        """
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

    def _sensors_attach(self) -> None:
        """Create and attach all camera sensors + collision sensor."""
        import carla

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

        bp = self._world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            bp, carla.Transform(), attach_to=self._ego
        )
        self._collision_events_recent.clear()
        self._collision_sensor.listen(self._on_collision)

    def _on_collision(self, event) -> None:
        """Mark the current frame as having a collision."""
        if self._collision_events_recent:
            self._collision_events_recent[-1] = 1

    def _cleanup(self) -> None:
        """Release CARLA actors and restore settings.

        Order: stop listeners → batch-destroy synchronously → null Python refs.
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

    def _wait_for_sensors(self, frame: int, timeout_s: float = 2.0) -> bool:
        """Bloque jusqu'à ce que tous les capteurs aient bufferisé ``frame``.

        Les callbacks ``listen`` des capteurs sont asynchrones : après
        ``world.tick()`` les données arrivent avec un léger délai. On attend que
        chaque capteur ait livré la frame attendue pour garantir que toutes les
        modalités (rgb / depth / semantic / instance) sont parfaitement alignées.
        ``time.sleep`` relâche le GIL pour laisser les callbacks s'exécuter.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if all(s.has_frame(frame) for s in self._sensors.values()):
                return True
            time.sleep(0.001)
        return False

    def run(self) -> None:
        """Setup -> tick loop -> cleanup. Blocking.

        Captures a full frame (image, depth, labels, expert controls)
        every `capture_every_n_ticks` ticks. Stops when `duration_sec` is
        reached. Cleanup is guaranteed via try/finally.
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
            _log(f"capture loop: target {target_frames} frames ({target_ticks} ticks)")

            while tick_count < target_ticks:
                world_frame = self._world.tick()
                tick_count += 1

                if tick_count % 200 == 0:
                    pct = 100 * tick_count / target_ticks
                    elapsed = time.time() - wall_start
                    _log(
                        f"tick {tick_count}/{target_ticks} ({pct:.0f}%, {elapsed:.0f}s)"
                    )

                if tick_count % self.capture_every_n_ticks != 0:
                    continue

                self._collision_events_recent.append(0)

                if not self._wait_for_sensors(world_frame):
                    _log(f"WARN frame {frame_id}: capteurs désynchronisés, skip")
                    continue

                for sensor in self._sensors.values():
                    sensor.save(self.output_dir, frame_id)

                label_path = self.output_dir / "labels_yolo" / f"{frame_id:06d}.txt"
                try:
                    self._yolo_labeler.save(label_path)
                except NotImplementedError:
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.write_text("")

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
                    f"frame {frame_id}/{target_frames} ({expert_controls.speed_kmh:.1f} km/h, {command.value})"
                )

            run_end = self._world.get_snapshot().timestamp.elapsed_seconds
            _log(f"capture done: {frame_id} frames in {time.time() - wall_start:.0f}s")
            self._manifest.flush_csv()
            self._manifest.write_metadata(
                run_id=f"{self.output_dir.name}",
                carla_version=str(self._client.get_server_version()),
                fps=CARLA_FPS,
                capture_every_n_ticks=self.capture_every_n_ticks,
                n_frames=frame_id,
                n_npc_vehicles=len(self._npc_actors),
                n_npc_walkers=0,
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
