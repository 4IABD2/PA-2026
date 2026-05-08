"""DatasetCollector — orchestrator for CARLA dataset collection.

Single-threaded, CARLA synchronous mode at 20 FPS. Captures every
`capture_every_n_ticks` ticks (40 by default = 2s at 20 FPS).
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from src.dataset.camera_capture import (
    CAMERA_LOCATION,
    CAMERA_ROTATION_PITCH,
    CameraCapture,
)
from src.dataset.command_planner import CommandPlanner, HighLevelCommand
from src.dataset.depth_capture import DepthCapture
from src.dataset.expert_driver import ExpertDriver
from src.dataset.instance_capture import InstanceCapture
from src.dataset.manifest_writer import ManifestWriter
from src.dataset.semantic_capture import SemanticCapture
from src.dataset.yolo_labels import YoloLabeler

if TYPE_CHECKING:
    import carla  # noqa: F401

CARLA_FPS = 20
FIXED_DELTA_SECONDS = 1.0 / CARLA_FPS  # 0.05
COLLISION_LOOKBACK_FRAMES = 5

# Subdirectories created under output_dir for every run.
OUTPUT_SUBDIRS = [
    "images",
    "depth",
    "labels_yolo",
    "semantic",
    "semantic_viz",
    "instance",
    "instance_viz",
]


class DatasetCollector:
    """Collection orchestrator. Single instance per run."""

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

        # Internal state set up in setup()
        self._client: "carla.Client | None" = None
        self._world: "carla.World | None" = None
        self._traffic_manager: "carla.TrafficManager | None" = None
        self._ego: "carla.Vehicle | None" = None
        self._npc_actors: list = []
        self._camera: CameraCapture | None = None
        self._depth: DepthCapture | None = None
        self._semantic: SemanticCapture | None = None
        self._instance: InstanceCapture | None = None
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

        # Connect
        self._client = carla.Client(self.host, self.port)
        self._client.set_timeout(10.0)
        self._world = self._client.load_world(self.town)

        # Weather
        weather_preset = getattr(carla.WeatherParameters, self.weather)
        self._world.set_weather(weather_preset)

        # Synchronous mode 20 FPS
        self._original_settings = self._world.get_settings()
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self._world.apply_settings(settings)

        # Synchronous Traffic Manager
        self._traffic_manager = self._client.get_trafficmanager()
        self._traffic_manager.set_synchronous_mode(True)
        if self.seed is not None:
            self._traffic_manager.set_random_device_seed(self.seed)

        # Tick to let the world stabilize after load_world
        self._world.tick()

        # Spawn ego
        self._ego = self._spawn_ego()

        # Spawn NPCs
        self._npc_actors = self._spawn_npcs()

        # Sensors
        self._camera = CameraCapture(
            self._world,
            self._ego,
            width=self.image_width,
            height=self.image_height,
            fov=self.fov,
        )
        self._sensors_attach()

        self._yolo_labeler = YoloLabeler(
            self._world,
            self._ego,
            self._camera._sensor,
            image_w=self.image_width,
            image_h=self.image_height,
        )
        self._command_planner = CommandPlanner(self._world, self._ego)
        self._expert = ExpertDriver(self._ego, self._traffic_manager)

        # Warm-up: let sensors produce their first frames
        for _ in range(10):
            self._world.tick()

        # Output dir + manifest writer
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for subdir in OUTPUT_SUBDIRS:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        self._manifest = ManifestWriter(self.output_dir, self.town, self.weather)

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
        """Spawn N NPC vehicles + walkers (best effort, may spawn fewer if no room).

        TODO: full walker implementation requires WalkerAIController which is
        more complex. For the MVP skeleton, we spawn only NPC vehicles.
        Walkers will be added later.
        """
        npcs = []
        bp_lib = self._world.get_blueprint_library()
        vehicle_bps = bp_lib.filter("vehicle.*")
        spawn_points = self._world.get_map().get_spawn_points()

        # NPC vehicles
        for i in range(min(self.n_npc_vehicles, len(spawn_points) - 1)):
            bp = random.choice(vehicle_bps)
            try:
                npc = self._world.try_spawn_actor(bp, spawn_points[i + 1])
                if npc is not None:
                    npc.set_autopilot(True, self._traffic_manager.get_port())
                    npcs.append(npc)
            except Exception:
                continue

        # TODO walkers (Franck) — not in MVP skeleton
        return npcs

    def _sensors_attach(self) -> None:
        """Attach camera + depth + collision sensors."""
        import carla

        self._camera.attach()
        self._depth = DepthCapture(
            self._world,
            self._ego,
            width=self.image_width,
            height=self.image_height,
            fov=self.fov,
        )
        self._depth.attach()

        self._semantic = SemanticCapture(
            self._world,
            self._ego,
            width=self.image_width,
            height=self.image_height,
            fov=self.fov,
        )
        self._semantic.attach()

        self._instance = InstanceCapture(
            self._world,
            self._ego,
            width=self.image_width,
            height=self.image_height,
            fov=self.fov,
        )
        self._instance.attach()

        # Collision sensor on ego (for is_collision in manifest)
        bp = self._world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            bp, carla.Transform(), attach_to=self._ego
        )
        self._collision_events_recent.clear()
        self._collision_sensor.listen(self._on_collision)

    def _on_collision(self, event) -> None:
        """Mark the current frame as having a collision."""
        # Push 1 into the recent-frames window
        # (the main loop pushes 0 on each captured frame to advance the window;
        # the 1 will overwrite the latest 0)
        if self._collision_events_recent:
            self._collision_events_recent[-1] = 1

    def _cleanup(self) -> None:
        """Release CARLA actors and restore settings.

        Order: stop listeners → batch-destroy synchronously → null Python refs.
        Sequential destroy crashes the C++ runtime if a callback is in flight; nulling
        Python refs before the server confirms destruction triggers
        "sensor object went out of the scope" warnings.
        """
        import carla

        if self._client is not None:
            sensor_wrappers = [
                self._camera,
                self._depth,
                self._semantic,
                self._instance,
            ]

            for w in sensor_wrappers:
                if w is not None and w._sensor is not None:
                    try:
                        w._sensor.stop()
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
            for w in sensor_wrappers:
                if w is not None and w._sensor is not None:
                    destroy_cmds.append(carla.command.DestroyActor(w._sensor))
            for npc in self._npc_actors:
                destroy_cmds.append(carla.command.DestroyActor(npc))
            if self._ego is not None:
                destroy_cmds.append(carla.command.DestroyActor(self._ego))
            if destroy_cmds:
                self._client.apply_batch_sync(destroy_cmds, True)

            for w in sensor_wrappers:
                if w is not None:
                    w._sensor = None
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
            tick_count = 0
            frame_id = 0
            run_start = self._world.get_snapshot().timestamp.elapsed_seconds

            while tick_count < target_ticks:
                self._world.tick()
                tick_count += 1

                if tick_count % self.capture_every_n_ticks != 0:
                    continue

                # Advance the collision window (push 0 by default; the
                # collision callback will overwrite to 1 if event on this frame)
                self._collision_events_recent.append(0)

                # Capture
                img_rel = f"images/{frame_id:06d}.jpg"
                depth_rel = f"depth/{frame_id:06d}.npy"
                label_rel = f"labels_yolo/{frame_id:06d}.txt"
                sem_rel = f"semantic/{frame_id:06d}.npy"
                sem_viz_rel = f"semantic_viz/{frame_id:06d}.png"
                inst_rel = f"instance/{frame_id:06d}.npy"
                inst_viz_rel = f"instance_viz/{frame_id:06d}.png"

                self._camera.save_last_frame(self.output_dir / img_rel)
                self._depth.save_last_frame(self.output_dir / depth_rel)
                self._semantic.save_last_frame(
                    self.output_dir / sem_rel, self.output_dir / sem_viz_rel
                )
                self._instance.save_last_frame(
                    self.output_dir / inst_rel, self.output_dir / inst_viz_rel
                )

                # YOLO labels: best-effort, NotImplementedError expected in MVP
                try:
                    self._yolo_labeler.save(self.output_dir / label_rel)
                except NotImplementedError:
                    # Write an empty file to stay consistent with the rest
                    (self.output_dir / label_rel).write_text("")

                expert_controls = self._expert.read_controls()
                command = self._command_planner.current_command()
                ts = self._world.get_snapshot().timestamp.elapsed_seconds - run_start
                is_collision = sum(self._collision_events_recent) > 0

                self._manifest.append_row(
                    frame_id=frame_id,
                    image_path=img_rel,
                    timestamp=ts,
                    command=command,
                    expert=expert_controls,
                    is_collision=is_collision,
                )
                frame_id += 1

            # Flush
            run_end = self._world.get_snapshot().timestamp.elapsed_seconds
            self._manifest.flush_csv()
            self._manifest.write_metadata(
                run_id=f"{self.output_dir.name}",
                carla_version=str(self._client.get_server_version()),
                fps=CARLA_FPS,
                capture_every_n_ticks=self.capture_every_n_ticks,
                n_frames=frame_id,
                n_npc_vehicles=len(self._npc_actors),
                n_npc_walkers=0,  # walkers not in MVP
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
