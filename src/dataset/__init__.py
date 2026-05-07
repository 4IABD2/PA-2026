"""Module de génération du dataset commun depuis CARLA.

API publique :
- DatasetCollector : orchestrateur principal
- HighLevelCommand : enum des commandes haut niveau (manifest)
- ExpertControls : dataclass des contrôles expert (manifest)

Voir src/dataset/README.md pour le format de sortie et l'usage.
"""

from src.dataset.collector import DatasetCollector
from src.dataset.command_planner import HighLevelCommand
from src.dataset.expert_driver import ExpertControls

__all__ = ["DatasetCollector", "HighLevelCommand", "ExpertControls"]
