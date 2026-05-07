# `src/interfaces/` — Contrats partagés entre modules

> **Owner** : collectif (toute modification se discute en équipe)
> **Statut** : socle de l'architecture — toutes les autres parties du code en dépendent

---

## Pourquoi ce module existe

Les 6 modules du projet (`perception/yolo`, `perception/depth`, `perception/lanes`, `navigation`, `ai`, `orchestration`) doivent dialoguer entre eux. Sans contrat précis, chaque module fait ses propres conventions et l'intégration finale devient un cauchemar.

Ici on définit **une seule fois** la forme exacte des données échangées. Tous les modules importent depuis `src.interfaces` et s'engagent à respecter ces structures.

C'est l'application du **pattern d'inversion de dépendances** : chaque module dépend d'un contrat (interface), pas d'une implémentation concrète.

## Conséquences pratiques

- Chaque module peut être développé et testé **indépendamment** des autres.
- Une implémentation peut être remplacée par une autre (modèle réel ↔ stub) **sans toucher** aux modules consommateurs.
- L'intégration finale est triviale : si chaque module respecte son contrat, les pièces s'emboîtent.

## Contenu

| Fichier | Rôle |
|---|---|
| `perception_types.py` | Types et protocoles pour YOLO, MIDAS/Depth, lignes |
| `navigation_types.py` | Types et protocoles pour GPS, route, commande haut niveau |
| `ai_types.py` | Types pour la scène fusionnée et la sortie de l'IA centrale |
| `stubs.py` | Implémentations "triche" basées sur les ground truths CARLA (dev only) |

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

## Les stubs (`stubs.py`)

Pour développer un module sans attendre les autres, on utilise des **implémentations triche** qui lisent directement les ground truths CARLA. Elles respectent les mêmes protocoles que les vraies implémentations, donc le code consommateur n'a rien à changer pour passer de l'un à l'autre.

```python
# Pendant le développement
detector: ObjectDetector = CarlaGTObjectDetector(world, ego_vehicle)

# En production
detector: ObjectDetector = YoloDetector("weights/best.pt")

# Le code qui utilise `detector` ne change pas
objects = detector.detect(image)
```

⚠️ Les stubs sont des outils de **développement uniquement**. Ils ne doivent jamais arriver dans le pipeline de production. Voir le [README racine](../../README.md) section "Conventions architecturales".

## Modifier un contrat

Toute modification d'un type ou d'un protocole **casse potentiellement plusieurs modules**. Procédure :

1. En discuter avec les owners des modules impactés
2. Proposer la modification dans une PR séparée
3. Mettre à jour les implémentations dans la même PR ou immédiatement après
4. Faire passer la CI complète

Ne pas modifier `src/interfaces/` dans une PR qui ajoute aussi de la logique métier.

## Validation

Les contrats (`@dataclass` + `Protocol`) définis ici sont vérifiés indirectement par les `smoke.py` de chaque module : si un module respecte son protocole, son smoke test passe.

Pour valider un nouveau type ou protocole ajouté ici, ajouter un check dans le `smoke.py` du module qui le consomme (ex : `benchmarks/perception/yolo/smoke.py` pour `ObjectDetector`).
