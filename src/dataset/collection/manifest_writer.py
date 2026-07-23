from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.dataset.collection.command_planner import HighLevelCommand
from src.dataset.collection.expert_driver import ExpertControls

CSV_COLUMNS = [
    "frame_id",
    "image_path",
    "timestamp",
    "command",
    "speed_kmh",
    "steer",
    "throttle",
    "brake",
    "is_collision",
    "town",
    "weather",
]


class ManifestWriter:
    """Buffer manifest rows in memory, flush to CSV at the end of the run."""

    def __init__(self, output_dir: Path, town: str, weather: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.town = town
        self.weather = weather
        self._rows: list[dict[str, Any]] = []
        self._started_at = datetime.now(timezone.utc)

    def append_row(
        self,
        *,
        frame_id: int,
        image_path: str,
        timestamp: float,
        command: HighLevelCommand,
        expert: ExpertControls,
        is_collision: bool,
    ) -> None:
        self._rows.append(
            {
                "frame_id": frame_id,
                "image_path": image_path,
                "timestamp": round(timestamp, 3),
                "command": command.value,
                "speed_kmh": round(expert.speed_kmh, 1),
                "steer": round(expert.steer, 3),
                "throttle": round(expert.throttle, 3),
                "brake": round(expert.brake, 3),
                "is_collision": int(bool(is_collision)),
                "town": self.town,
                "weather": self.weather,
            }
        )

    def flush_csv(self) -> None:
        df = pd.DataFrame(self._rows, columns=CSV_COLUMNS)
        df.to_csv(self.output_dir / "manifest.csv", index=False)

    def write_metadata(self, **kwargs: Any) -> None:
        """Write metadata.json. Run-level fields (town, weather, timestamps) are injected automatically."""
        ended_at = datetime.now(timezone.utc)
        metadata = {
            "town": self.town,
            "weather": self.weather,
            "started_at": self._started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            **kwargs,
        }
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
