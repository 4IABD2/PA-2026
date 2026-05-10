# `src/dataset/` — Génération du dataset commun depuis CARLA

> **Owner principal** : Frédéric (script de collecte)
> **Contributeurs** : Franck (colonnes commande HN + actions expert), Karim (annotations lignes si nécessaire)

---

## Pourquoi un module partagé ?

Tous les modules d'apprentissage du projet ont besoin de données issues de CARLA :

| Module | Données nécessaires |
|---|---|
| `src/perception/yolo/` (Franck) | image RGB + masks d'instance (dérivation bboxes) |
| `src/perception/depth/` (Franck) | image RGB + depth GT |
| `src/perception/lanes/` (Karim) | image RGB + masks sémantiques (classe RoadLine) |
| `src/ai/` (Frédéric) | image RGB + commande HN + vitesse + actions expert |

Toutes ces données peuvent être collectées **en une seule passe** dans CARLA. CARLA fournit "gratuitement" :

- Les bounding boxes via la liste des actors (projection 3D→2D)
- Les depth maps via le capteur `sensor.camera.depth`
- La position du véhicule par rapport aux waypoints (utile pour les lignes)
- La vitesse et les actions de l'autopilot (l'expert pour Frédéric)

**Conséquences** :
- 1 collecte de ~30 min = données pour 4 personnes
- Cohérence garantie : même map, même météo, même caméra, même véhicule
- L'intégration finale est plus simple : les détections de Franck correspondent exactement aux images vues par l'IA centrale

## Format du dataset

```
data/runs/<YYYY-MM-DD>_<town>_<weather>/
├── images/                ← RGB JPEG, 1 frame toutes les 2s
│   ├── 000000.jpg
│   └── ...
├── depth/                 ← depth maps GT en mètres (.npy float32)
│   ├── 000000.npy
│   └── ...
├── semantic/              ← masks sémantique CARLA (uint8 class IDs 0-28)
│   ├── 000000.npy
│   └── ...
├── semantic_viz/          ← visualisation palette CityScape (PNG)
│   ├── 000000.png
│   └── ...
├── instance/              ← masks instance CARLA (uint32 packed)
│   ├── 000000.npy
│   └── ...
├── instance_viz/          ← visualisation couleur déterministe par instance (PNG)
│   ├── 000000.png
│   └── ...
├── labels_yolo/           ← labels YOLO (1 .txt par image)
│   ├── 000000.txt         ← format: <class> <x_center> <y_center> <w> <h>
│   └── ...
├── lanes_gt/              ← annotations lignes (.json par image, optionnel)
│   ├── 000000.json
│   └── ...
├── manifest.csv           ← 1 ligne par frame, métadonnées par-image
└── metadata.json          ← métadonnées globales de la run
```

### `manifest.csv`

| frame_id | image_path | timestamp | command | speed_kmh | steer | throttle | brake | is_collision | town | weather |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | images/000000.jpg | 0.000 | straight | 0.0 | 0.0 | 0.5 | 0.0 | 0 | Town01 | ClearNoon |
| 1 | images/000001.jpg | 2.000 | straight | 12.5 | -0.02 | 0.5 | 0.0 | 0 | Town01 | ClearNoon |

### `metadata.json`

```json
{
  "run_id": "2026-05-09_town01_clear",
  "town": "Town01",
  "weather": "ClearNoon",
  "carla_version": "0.9.16",
  "fps": 20,
  "capture_every_n_ticks": 40,
  "n_frames": 1500,
  "n_npc_vehicles": 40,
  "n_npc_walkers": 30,
  "camera_pov": {
    "location": [0.30, 0.0, 1.50],
    "rotation": [0, 0, -5]
  },
  "duration_sec": 3000
}
```

### Segmentation : encodage des `.npy`

**`semantic/<frame_id>.npy`** — `numpy.ndarray` shape `(H, W)`, dtype `uint8`. Chaque valeur = class ID CARLA (mapping CityScape officiel CARLA 0.9.13+, 0-28 — `1`=Roads, `14`=Car, `24`=RoadLine, etc.). Décodage trivial : c'est le canal R brut renvoyé par `sensor.camera.semantic_segmentation`.

**`instance/<frame_id>.npy`** — `numpy.ndarray` shape `(H, W)`, dtype `uint32`. Bits :

```
[23..16] = class_id   (R channel)
[15..8]  = G channel  (instance_id high byte)
[7..0]   = B channel  (instance_id low byte)
```

Unpack côté consommateur :

```python
class_id    = (packed >> 16) & 0xFF
instance_id = packed & 0xFFFF
```

`instance_id == 0` = pixel non trackable (fond / classe non comptée).

**Visualisations PNG** (`semantic_viz/`, `instance_viz/`) — colorisations directement ouvrables dans VSCode/feh, régénérables via `colorize_semantic` et `colorize_instance`. Sémantique : palette CityScape officielle CARLA (route grise, voitures bleues, etc.). Instance : couleur HSV golden-ratio par `instance_id`, déterministe à travers les frames d'une même run.

## API publique

```python
from src.dataset.collector import DatasetCollector

collector = DatasetCollector(
    output_dir="data/runs/2026-05-09_town01_clear",
    town="Town01",
    weather="ClearNoon",
    n_npc_vehicles=40,
    n_npc_walkers=30,
    duration_sec=1800,
    capture_every_n_ticks=40,  # 1 frame toutes les 2s à 20 FPS
)
collector.run()
```

## Squelette suggéré

```
src/dataset/
├── __init__.py
├── collector.py          ← classe DatasetCollector (orchestration)
├── camera_capture.py     ← setup caméra RGB + sauvegarde JPEG
├── depth_capture.py      ← setup capteur depth GT + sauvegarde .npy
├── yolo_labels.py        ← projection 3D→2D des actors → format YOLO
├── command_planner.py    ← calcul de la commande HN (G/D/TT/Suivre)
├── expert_driver.py      ← wrapper autopilot CARLA
└── README.md
```

Chacun peut contribuer à son fichier sans bloquer les autres :

- **Frédéric** : `collector.py`, `camera_capture.py`, `depth_capture.py`, `yolo_labels.py` (il a déjà commencé un `carla_dataset_generator.py` dans son contexte)
- **Franck** : `command_planner.py`, `expert_driver.py`, et l'enrichissement du `manifest.csv`
- **Karim** : `lanes_gt.py` si on décide d'annoter les lignes via les waypoints CARLA

## Conventions importantes

- **Mode synchrone CARLA obligatoire** : `world.tick()` à 20 FPS (`fixed_delta_seconds = 0.05`)
- **Capture toutes les 2s** (40 ticks) : à 20 FPS deux frames consécutives sont quasi identiques. Toutes les 2s, on a une vraie diversité.
- **Caméra POV partagée** : `Location(x=0.30, y=0.0, z=1.50), Rotation(pitch=-5)` — caméra centrée longitudinalement, juste au-dessus du toit Tesla (z=1.50, toit à ~1.44m). Voir le [README racine](../../README.md) section "Conventions CARLA" pour la justification (positionnement hors body imposé par les capteurs `semantic_segmentation` / `instance_segmentation` qui ne respectent pas la transparence des matériaux).
- **Naming des runs** : `<YYYY-MM-DD>_<town>_<weather>` (ex : `2026-05-09_town01_clear`)
- **Pas commit** : `data/` est gitignored. Pour partager → voir le [README racine](../../README.md) section "Stockage et partage".

## Stockage et partage

Le dossier `data/` à la racine est **gitignored**. Pour partager un dataset entre membres de l'équipe, voir le [README racine](../../README.md) section "Stockage et partage" (procédure cloud / scp / DVC).

## Validation

Pas de benchmark de performance pour ce module (ce n'est pas un modèle ML). Juste de la validation de format à mettre dans `benchmarks/dataset/smoke.py` (à créer si besoin) :

- Vérifier que les sorties produites respectent le format documenté ci-dessus (sans avoir besoin de CARLA, en mockant les capteurs)
- Vérifier que le manifest.csv a toutes les colonnes attendues
- Vérifier que les labels YOLO sont correctement normalisés `[0, 1]`

## Liens

- [CARLA — Génération de données](https://carla.readthedocs.io/en/latest/tuto_G_generate_pedestrian_navigation/)
- [Format de labels YOLO](https://docs.ultralytics.com/datasets/detect/)
- [CARLA — Capteur depth](https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera)
