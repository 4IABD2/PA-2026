"""Common dataset generation module from CARLA.

Public API:
- DatasetCollector: main orchestrator (requires carla)
- HighLevelCommand: high-level command enum (manifest)
- ExpertControls: expert controls dataclass (manifest)

See src/dataset/README.md for output format and usage.
"""


def __getattr__(name: str):
    if name == "DatasetCollector":
        from src.dataset.collector import DatasetCollector
        return DatasetCollector
    if name == "HighLevelCommand":
        from src.dataset.command_planner import HighLevelCommand
        return HighLevelCommand
    if name == "ExpertControls":
        from src.dataset.expert_driver import ExpertControls
        return ExpertControls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DatasetCollector", "HighLevelCommand", "ExpertControls"]
