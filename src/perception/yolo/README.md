# `src/perception/yolo/` — Détection d'objets

> **Owner** : Franck Zhuang
> **Contrat** : [`ObjectDetector`](../../interfaces/perception_types.py)

---

## Responsabilité

Détecter dans une image RGB les objets pertinents pour la conduite : véhicules, piétons, feux de signalisation, panneaux. Pour chaque objet : sa classe, sa boîte englobante en pixels, son score de confiance.

La distance à l'objet est laissée vide ici (`distance_m=None`) — elle sera renseignée plus tard par fusion avec le module `depth/`.

## API publique

Doit exposer une classe (typiquement `YoloDetector`) qui implémente le protocole `ObjectDetector` :

```python
from src.interfaces.perception_types import DetectedObject, ObjectClass

class YoloDetector:
    def __init__(self, weights_path: str, device: str = "cpu") -> None: ...

    def detect(self, image: np.ndarray) -> list[DetectedObject]:
        """
        Args:
            image: RGB (H, W, 3), dtype uint8.
        Returns:
            Liste d'objets détectés (peut être vide).
        """
```

## Comment démarrer

1. Choisir un modèle de base : YOLOv8n (rapide) ou YOLOv8s (un peu plus précis).
2. Pré-entraîné COCO ou fine-tuné sur dataset CARLA — voir section "Entraînement".
3. Stocker les poids dans `src/perception/yolo/weights/` (gitignored).
4. Implémenter `detector.py` qui charge les poids et expose `YoloDetector.detect(image)`.

Squelette suggéré :

```
src/perception/yolo/
├── __init__.py
├── detector.py        ← classe YoloDetector
├── train.py           ← script de fine-tuning sur dataset CARLA
├── weights/           ← gitignored, modèles .pt / .onnx
└── README.md
```

## Mapping des classes

Convertir les classes du modèle vers les classes du projet définies dans `ObjectClass` :

| YOLO COCO | `ObjectClass` |
|---|---|
| `car`, `truck`, `bus`, `motorcycle` | `VEHICLE` |
| `person` | `WALKER` |
| `traffic light` | `TRAFFIC_LIGHT` |
| `stop sign` | `TRAFFIC_SIGN` |
| autres | `UNKNOWN` (ou ignorer) |

## Entraînement

Si on fine-tune sur du dataset CARLA (généré par `src/ai/collector/`) :

- Format YOLO : un fichier `.txt` par image avec `<class> <x_center> <y_center> <w> <h>` normalisés
- Configurer un `data.yaml` pointant vers les dossiers `images/train`, `labels/train`, etc.
- Lancement type : `yolo detect train data=data.yaml model=yolov8n.pt epochs=50`

Documenter dans ce README la procédure exacte une fois le pipeline en place.

## Performance attendue

- Inference < 50 ms par image sur GPU, < 200 ms sur CPU pour YOLOv8n
- Précision : > 0.7 mAP@50 attendu sur les classes véhicule + piéton après fine-tuning sur dataset CARLA

## Validation et benchmarks

Dans [benchmarks/perception/yolo/](../../../benchmarks/perception/yolo/) :

- **`smoke.py`** : vérifier qu'une instance respecte le protocole `ObjectDetector`, qu'une image vide produit une liste vide, que les bbox retournées sont dans les bornes de l'image
- **`benchmark.py`** : mesurer mAP@50, mAP@50-95 sur le dataset CARLA, latence p50/p95 par image

Voir [benchmarks/README.md](../../../benchmarks/README.md) pour la convention.

## Liens

- [Documentation YOLOv8 (Ultralytics)](https://docs.ultralytics.com/)
- [Format de labels YOLO](https://docs.ultralytics.com/datasets/detect/)
