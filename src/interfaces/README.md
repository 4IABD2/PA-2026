# `src/interfaces/` — Contrats partagés entre modules

> **Owner** : collectif (toute modification se discute en équipe)
> **Statut** : socle de l'architecture — toutes les autres parties du code en dépendent

---

## Pourquoi ce module existe

Les modules du projet (`perception/yolo`, `perception/depth`, `lane_detection`, `navigation`, `ai`) doivent dialoguer entre eux. Sans contrat précis, chaque module fait ses propres conventions et l'intégration finale devient un cauchemar.

Ici on définit **une seule fois** la forme exacte des données échangées. Tous les modules importent depuis `src.interfaces` et s'engagent à respecter ces structures.

C'est l'application du **pattern d'inversion de dépendances** : chaque module dépend d'un contrat (interface), pas d'une implémentation concrète.

## Conséquences pratiques

- Chaque module peut être développé et testé **indépendamment** des autres.
- Une implémentation peut être remplacée par une autre (capteur GT de développement ↔ modèle réel) **sans toucher** aux modules consommateurs — c'est exactement ce qui s'est passé à la v21 quand la perception réelle a remplacé la vérité terrain, sans changer une ligne de l'IA centrale.
- L'intégration finale est triviale : si chaque module respecte son contrat, les pièces s'emboîtent.

## Contenu

| Fichier | Rôle |
|---|---|
| `perception_types.py` | Types et protocoles pour YOLO, Depth, lignes |
| `navigation_types.py` | Types et protocoles pour GPS, route, commande haut niveau |
| `ai_types.py` | Types pour la scène fusionnée et la sortie de l'IA centrale |

## Les types principaux

### Perception

```python
from src.interfaces.perception_types import (
    DetectedObject, ObjectClass,           # YOLO
    LanesInfo, Line,                        # Lignes
    ObjectDetector, DepthEstimator, LaneDetector,  # Protocols
)
```

### Navigation

```python
from src.interfaces.navigation_types import (
    HighLevelCommand,    # LEFT / RIGHT / STRAIGHT / LANE_FOLLOW
    Waypoint, Route,
    RoutePlanner, CommandPlanner,
)
```

### IA centrale

```python
from src.interfaces.ai_types import (
    SceneState,          # ce que l'IA consomme
    ControlOutput,       # ce que l'IA produit
    VehicleState,
)
```

## Comment implémenter un contrat

Chaque protocole se respecte par **structural typing** (Python "duck typing" formalisé via `typing.Protocol`). Pas besoin d'hériter explicitement : il suffit d'avoir les bonnes méthodes avec les bonnes signatures.

Exemple — implémenter un détecteur d'objets :

```python
# src/perception/yolo/detector.py
import numpy as np
from src.interfaces.perception_types import DetectedObject, ObjectClass, ObjectDetector

class YoloDetector:
    def __init__(self, weights_path: str) -> None:
        # charge le modèle YOLO ici
        ...

    def detect(self, image: np.ndarray) -> list[DetectedObject]:
        # inférence YOLO, conversion des résultats en list[DetectedObject]
        return [
            DetectedObject(
                class_name=ObjectClass.VEHICLE,
                bbox=(x1, y1, x2, y2),
                confidence=score,
            )
            ...
        ]
```

Cette classe est automatiquement compatible avec le protocole `ObjectDetector` parce qu'elle a la bonne méthode `detect`. Elle peut être utilisée partout où un `ObjectDetector` est attendu.

> **Note historique** : pendant le développement, un fichier `stubs.py` fournissait des implémentations « triche » lisant les ground truths CARLA, pour bosser sur son module sans attendre les autres. Devenu inutile une fois les vrais modèles branchés (v21), il a été retiré au nettoyage final.

## Modifier un contrat

Toute modification d'un type ou d'un protocole **casse potentiellement plusieurs modules**. Procédure :

1. En discuter avec les owners des modules impactés
2. Proposer la modification dans une PR séparée
3. Mettre à jour les implémentations dans la même PR ou immédiatement après
4. Faire passer la CI complète

Ne pas modifier `src/interfaces/` dans une PR qui ajoute aussi de la logique métier.

## Validation

Les contrats (`@dataclass` + `Protocol`) se vérifient par **structural typing** : si un module a les bonnes méthodes avec les bonnes signatures, il est compatible partout où le protocole est attendu. La validation finale, c'est l'intégration réelle — les 4 modules branchés ensemble dans la boucle RL et la démo.
