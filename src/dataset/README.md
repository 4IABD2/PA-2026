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
| `src/lane_detection/` (Karim) | image RGB + masks sémantiques (classe RoadLine) |
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
├── depth/                 ← float32 (.npy), distance en mètres, plafonné 100m
├── semantic/              ← uint8 (.npy), class_id CityScape 0-28
├── instance/              ← uint32 (.npy), (class_id << 16) | instance_id
├── viz/                   ← visualisations PNG (debug uniquement)
│   ├── semantic/          ← palette CityScape
│   └── instance/          ← couleurs HSV par instance_id
├── labels_yolo/           ← raw 4-classes (collector) : vehicle/walker/traffic_light/traffic_sign
├── labels_yolo_color/     ← final 12-classes (après enrich_labels) : red/yellow/green + speed_30..90
├── debug_dropped_tl/      ← optionnel (enrich --debug-drops) : crops feux non classés
├── debug_dropped_signs/   ← optionnel : crops panneaux non classés
├── manifest.csv           ← 1 ligne par frame (timestamp, command, controles, collision)
└── metadata.json          ← métadonnées globales (run_id, seed, fps, npc count…)
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

**Visualisations PNG** (`viz/semantic/`, `viz/instance/`) — colorisations directement ouvrables dans VSCode/feh, régénérables via `colorize_semantic` et `colorize_instance` (dans `encodings.py`). Sémantique : palette CityScape officielle CARLA (route grise, voitures bleues, etc.). Instance : couleur HSV golden-ratio par `instance_id`, déterministe à travers les frames d'une même run.

## Workflow d'usage

### 1 — Lancer une collecte (CARLA tournant sur localhost:2000)

```powershell
# Une run simple (defaut Town01/ClearNoon, 30s)
uv run -m src.dataset.run_collection --town Town01 --duration 600 --npcs 30

# Multi-runs avec enrichissement automatique
uv run -m src.dataset.collect_multi --maps Town01,Town03 --weathers ClearNoon,CloudyNoon --frames-per-run 500 --npcs 30 --enrich
```

### 2 — Enrichir les labels (raw 4-classes → final 12-classes)

```powershell
uv run -m src.dataset.enrich_labels --run data/runs/<run>
uv run -m src.dataset.enrich_labels --run data/runs/<run> --debug-drops   # pour debug les drops
```

### 3 — Inspecter

```powershell
uv run -m src.dataset.inspect_run --run data/runs/<run>             # compteurs + disque
uv run -m src.dataset.inspect_run --run data/runs/<run> --frame 17  # mask d'une frame
uv run -m src.tools.visualize_fiftyone --run data/runs/<run>        # UI bboxes
```

### API programmatique

```python
from src.dataset.collector import DatasetCollector

collector = DatasetCollector(
    output_dir="data/runs/2026-05-09_town01_clear",
    town="Town01",
    weather="ClearNoon",
    n_npc_vehicles=40,
    duration_sec=1800,
    capture_every_n_ticks=40,  # 1 frame toutes les 2s à 20 FPS
)
collector.run()
```

Pour la doc narrative complète (étapes internes du collector, post-process,
gotchas), voir [COLLECTION_GUIDE.md](COLLECTION_GUIDE.md).

## Structure du module

```
src/dataset/
├── __init__.py
├── collector.py             ← orchestrateur DatasetCollector
├── sensors.py               ← CameraSensor unifié + 4 writers (rgb/depth/semantic/instance)
├── encodings.py             ← helpers numpy purs (decode/pack/colorize) + constantes POV
├── yolo_labels.py           ← YoloLabeler : bboxes via instance mask + composants connexes
├── command_planner.py       ← commande haut niveau pour le manifest
├── expert_driver.py         ← wrapper autopilot CARLA (lecture controles)
├── manifest_writer.py       ← manifest.csv + metadata.json
├── run_collection.py        ← CLI 1 run
├── collect_multi.py         ← CLI multi-runs (maps × weathers)
├── enrich_labels.py         ← post-process : HSV feux + template panneaux → 12 classes
├── inspect_run.py           ← diagnostic d'une run (compteurs / inspection frame)
├── README.md                ← ce fichier (reference)
├── COLLECTION_GUIDE.md      ← guide narratif end-to-end
└── PROBLEMS_AND_FIXES.md    ← bugs rencontrés + solutions
```

Le module produit deux jeux de labels :

- **`labels_yolo/`** — raw, 4 classes (`vehicle, walker, traffic_light, traffic_sign`) écrit par le collector en live.
- **`labels_yolo_color/`** — final, 12 classes (`vehicle, walker, red_light, yellow_light, green_light, speed_30..90`) produit hors-ligne par `enrich_labels.py`. C'est ce qu'on donne à YOLO en entraînement.

## Conventions importantes

- **Mode synchrone CARLA obligatoire** : `world.tick()` à 20 FPS (`fixed_delta_seconds = 0.05`)
- **Capture toutes les 2s** (40 ticks) : à 20 FPS deux frames consécutives sont quasi identiques. Toutes les 2s, on a une vraie diversité.
- **Caméra POV partagée** : `Location(x=0.30, y=0.0, z=1.50), Rotation(pitch=-5)` — caméra centrée longitudinalement, juste au-dessus du toit Tesla (z=1.50, toit à ~1.44m). Voir le [README racine](../../README.md) section "Conventions CARLA" pour la justification (positionnement hors body imposé par les capteurs `semantic_segmentation` / `instance_segmentation` qui ne respectent pas la transparence des matériaux).
- **Naming des runs** : `<YYYY-MM-DD>_<town>_<weather>` (ex : `2026-05-09_town01_clear`)
- **Pas commit** : `data/` est gitignored. Pour partager → voir le [README racine](../../README.md) section "Stockage et partage".

## Stockage et partage

Le dossier `data/` à la racine est **gitignored**. Pour partager un dataset entre membres de l'équipe, voir le [README racine](../../README.md) section "Stockage et partage" (procédure cloud / scp / DVC).

## Validation

Pas de benchmark de performance pour ce module (ce n'est pas un modèle ML). La validation se fait sur le format des sorties : vérifier après une collecte que les dossiers produits respectent le format documenté ci-dessus et que `manifest.csv` a toutes les colonnes attendues (`src/dataset/diagnostics/inspect_run.py`).
- Vérifier que les labels YOLO sont correctement normalisés `[0, 1]`

## Liens

- [CARLA — Génération de données](https://carla.readthedocs.io/en/latest/tuto_G_generate_pedestrian_navigation/)
- [Format de labels YOLO](https://docs.ultralytics.com/datasets/detect/)
- [CARLA — Capteur depth](https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera)
