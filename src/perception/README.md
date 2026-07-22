# Perception — détection d'objets + profondeur

Module de perception (Franck). Produit, à partir d'une image RGB, les **objets
détectés avec leur distance** + la **depth map** — ce que l'IA centrale consomme
dans son [`SceneState`](../interfaces/ai_types.py).

```
image RGB ──┬──> YoloDetector  ──> objets (classe, bbox, confiance)
            └──> DepthEstimator ──> depth map (mètres)
                       │
                       ▼
              PerceptionPipeline : distance par objet (depth au centre du bbox)
                       │
                       ▼
            list[DetectedObject] (avec distance_m) + depth_map
```

Sous-modules :
- [`yolo/`](yolo/) — détection d'objets, 11 classes (voir [yolo/README.md](yolo/README.md))
- [`depth/`](depth/) — profondeur monoculaire (voir [depth/README.md](depth/README.md))
- [`pipeline.py`](pipeline.py) — **fusion** des deux (le point d'entrée)

## Utilisation (IA centrale)

C'est une **fonction locale**, pas une API. On charge les modèles une fois, puis
on appelle `perceive()` par image.

```python
from src.perception.pipeline import PerceptionPipeline

# 1. UNE fois au démarrage (charge YOLO + Depth en mémoire)
perception = PerceptionPipeline(device="cuda")  # ou "cpu"

# 2. À chaque tick de la boucle temps réel
objects, depth_map = perception.perceive(image_rgb)
#   objects   : list[DetectedObject]  (class_name, bbox, confidence, distance_m)
#   depth_map : np.ndarray (H, W) float32, en mètres
```

Branchement direct dans le contrat :

```python
from src.interfaces.ai_types import SceneState

scene = SceneState(
    image_rgb=image_rgb,
    objects=objects,        # ← rempli ici (avec distance_m)
    depth_map=depth_map,    # ← rempli ici
    # lanes=..., vehicle_state=..., high_level_command=...  ← autres modules
)
```

Variante list-of-dicts (sérialisable) si besoin :

```python
perception.perceive_dict(image_rgb)
# -> [{"class": "vehicle", "bbox": [x1,y1,x2,y2], "confidence": 0.95, "distance_m": 12.3}, ...]
```

## Setup (pour utiliser les modèles entraînés)

1. **Code + calibration** (suivis git) :
   ```bash
   git pull
   uv sync          # torch CUDA, ultralytics, transformers...
   ```
2. **Poids YOLO** (`best.pt`, gitignoré par convention équipe) : à récupérer
   auprès de Franck et à placer dans `src/perception/yolo/weights/best.pt`.
3. **Premier lancement** : Depth Anything v2 (~100 MB) se télécharge depuis
   HuggingFace automatiquement (internet requis une fois).

| Point | Détail |
|---|---|
| Besoin de CARLA ? | **Non** — la perception est pure, testable sans simulateur |
| GPU ? | Optionnel. `device="cpu"` marche ; pour NVIDIA, le lock cu128 est déjà en place |
| Calibration depth | `depth/calibration.json` (versionné), chargée automatiquement |

## Artefacts

| Fichier | Suivi git ? | Rôle |
|---|---|---|
| `pipeline.py`, `*/detector.py`, `*/estimator.py` | ✅ | code |
| `depth/calibration.json` | ✅ | scale/shift depth (petit JSON) |
| `yolo/weights/best.pt` | ❌ (gitignoré) | poids YOLO entraînés — à partager à la main |

## Tester

```bash
# Test rapide sur une image (affiche le JSON des détections + distances)
uv run -m src.perception.pipeline --image data/runs/<session>/<run>/images/000100.jpg

# Grilles visuelles (bboxes + distances + depth colorisée)
uv run -m src.perception.visualize_batch --images data/runs/<session>/<run>/images \
    --yolo-weights src/perception/yolo/weights/best.pt \
    --depth-gt data/runs/<session>/<run>/depth --device cuda --n 9 --every 20

# Détection chiffrée (mAP par classe sur le val set)
uv run yolo detect val model=src/perception/yolo/weights/best.pt data=data/yolo_dataset/data.yaml device=0
```

## (Ré)entraîner / recalibrer

- **Dataset YOLO** : `uv run -m src.perception.yolo.prepare_dataset --runs data/runs/<session> --output data/yolo_dataset`
- **Entraînement** : `uv run yolo detect train data=data/yolo_dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=16 device=0`
- **Calibration depth** : `uv run -m src.perception.depth.calibrate --run data/runs/<session>/<run> --n 20`
