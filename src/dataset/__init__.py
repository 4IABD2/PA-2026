"""Common dataset generation module from CARLA.

Public API:
- DatasetCollector: main orchestrator
- HighLevelCommand: high-level command enum (manifest)
- ExpertControls: expert controls dataclass (manifest)

See src/dataset/README.md for output format and usage.
"""

from src.dataset.collector import DatasetCollector
from src.dataset.command_planner import HighLevelCommand
from src.dataset.expert_driver import ExpertControls

__all__ = ["DatasetCollector", "HighLevelCommand", "ExpertControls"]
