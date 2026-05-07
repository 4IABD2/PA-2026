"""DatasetCollector — orchestrateur de la collecte CARLA pour le dataset commun.

Single-threaded, mode synchrone CARLA 20 FPS. Capture toutes les
`capture_every_n_ticks` ticks (40 par défaut = 2s à 20 FPS).
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from src.dataset.camera_capture import CameraCapture
from src.dataset.command_planner import CommandPlanner, HighLevelCommand
from src.dataset.depth_capture import DepthCapture
from src.dataset.expert_driver import ExpertDriver
from src.dataset.manifest_writer import ManifestWriter
from src.dataset.yolo_labels import YoloLabeler

if TYPE_CHECKING:
    import carla  # noqa: F401

CARLA_FPS = 20
FIXED_DELTA_SECONDS = 1.0 / CARLA_FPS  # 0.05
COLLISION_LOOKBACK_FRAMES = 5


class DatasetCollector:
    """Orchestrateur de la collecte. Single instance par run."""

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
        """Refuse un output_dir non vide pour éviter d'écraser un run précédent."""
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise ValueError(
                f"output_dir is not empty: {self.output_dir}. "
                f"Choisir un nouveau path ou vider manuellement."
            )

    def _setup(self) -> None:
        """Connexion CARLA + setup synchrone + spawn ego + NPC + sensors + writers."""
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

        # Mode synchrone 20 FPS
        self._original_settings = self._world.get_settings()
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self._world.apply_settings(settings)

        # Traffic Manager synchrone
        self._traffic_manager = self._client.get_trafficmanager()
        self._traffic_manager.set_synchronous_mode(True)
        if self.seed is not None:
            self._traffic_manager.set_random_device_seed(self.seed)

        # Tick pour laisser le monde se stabiliser après load_world
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

        # Warm-up : laisser les sensors produire leurs premières données
        for _ in range(10):
            self._world.tick()

        # Output dir + manifest writer
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "depth").mkdir(exist_ok=True)
        (self.output_dir / "labels_yolo").mkdir(exist_ok=True)
        self._manifest = ManifestWriter(self.output_dir, self.town, self.weather)

    def _spawn_ego(self) -> "carla.Vehicle":
        """Spawn un véhicule ego sur un spawn point random de la map."""
        bp = self._world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        spawn_points = self._world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError(f"Pas de spawn points sur la map {self.town}")
        spawn = random.choice(spawn_points)
        ego = self._world.spawn_actor(bp, spawn)
        return ego

    def _spawn_npcs(self) -> list:
        """Spawn N NPC vehicles + walkers (best effort, peut spawn moins si pas de place).

        TODO: implémentation full des walkers nécessite WalkerAIController qui est
        plus complexe. Pour le squelette MVP, on spawn juste les vehicles NPC.
        Les walkers seront ajoutés plus tard.
        """
        npcs = []
        bp_lib = self._world.get_blueprint_library()
        vehicle_bps = bp_lib.filter("vehicle.*")
        spawn_points = self._world.get_map().get_spawn_points()

        # Vehicle NPCs
        for i in range(min(self.n_npc_vehicles, len(spawn_points) - 1)):
            bp = random.choice(vehicle_bps)
            try:
                npc = self._world.try_spawn_actor(bp, spawn_points[i + 1])
                if npc is not None:
                    npc.set_autopilot(True, self._traffic_manager.get_port())
                    npcs.append(npc)
            except Exception:
                continue

        # TODO walkers (Franck) — pas dans le squelette MVP
        return npcs

    def _sensors_attach(self) -> None:
        """Attache camera + depth + collision."""
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

        # Collision sensor sur ego (pour is_collision dans manifest)
        bp = self._world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            bp, carla.Transform(), attach_to=self._ego
        )
        self._collision_events_recent.clear()
        self._collision_sensor.listen(self._on_collision)

    def _on_collision(self, event) -> None:
        """Marque la frame courante comme ayant eu une collision."""
        # Pousse 1 dans la fenêtre des dernières frames
        # (on pousse 0 dans la boucle principale à chaque frame capturée
        # pour avancer la fenêtre — le 1 écrasera le dernier 0)
        if self._collision_events_recent:
            self._collision_events_recent[-1] = 1

    def _cleanup(self) -> None:
        """Libère les actors CARLA et restaure les settings."""
        import carla

        # Batch destroy : évite le crash C++ causé par destroy séquentiel
        # des sensors attachés pendant que CARLA traite encore des callbacks
        if self._client is not None:
            destroy_cmds = []
            if self._collision_sensor is not None:
                destroy_cmds.append(carla.command.DestroyActor(self._collision_sensor))
            if self._camera is not None and self._camera._sensor is not None:
                destroy_cmds.append(carla.command.DestroyActor(self._camera._sensor))
                self._camera._sensor = None
            if self._depth is not None and self._depth._sensor is not None:
                destroy_cmds.append(carla.command.DestroyActor(self._depth._sensor))
                self._depth._sensor = None
            for npc in self._npc_actors:
                destroy_cmds.append(carla.command.DestroyActor(npc))
            if self._ego is not None:
                destroy_cmds.append(carla.command.DestroyActor(self._ego))
            if destroy_cmds:
                self._client.apply_batch(destroy_cmds)

        # Restore settings (libère le serveur)
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
        """Setup -> boucle ticks -> cleanup. Bloquant.

        Capture une frame complète (image, depth, labels, expert controls)
        toutes les `capture_every_n_ticks` ticks. Termine quand `duration_sec`
        est atteint. Cleanup garanti via try/finally.
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

                # Avance la fenêtre collision (pousse 0 par défaut, le callback
                # collision écrasera en 1 si event sur cette frame)
                self._collision_events_recent.append(0)

                # Capture
                img_rel = f"images/{frame_id:06d}.jpg"
                depth_rel = f"depth/{frame_id:06d}.npy"
                label_rel = f"labels_yolo/{frame_id:06d}.txt"

                self._camera.save_last_frame(self.output_dir / img_rel)
                self._depth.save_last_frame(self.output_dir / depth_rel)

                # Labels YOLO : best-effort, NotImplementedError attendu en MVP
                try:
                    self._yolo_labeler.save(self.output_dir / label_rel)
                except NotImplementedError:
                    # Écrit un fichier vide pour rester cohérent avec le reste
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
                n_npc_walkers=0,  # walkers pas dans MVP
                duration_sec_target=self.duration_sec,
                duration_sec_actual=round(run_end - run_start, 2),
                camera_pov={
                    "location": [0.5, -0.3, 1.2],
                    "rotation": [0, 0, -5],
                },
                image_resolution=[self.image_width, self.image_height],
                fov=self.fov,
                seed=self.seed,
            )
        finally:
            self._cleanup()
