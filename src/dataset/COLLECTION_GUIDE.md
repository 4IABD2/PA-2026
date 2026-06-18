# Collecte de dataset CARLA — guide complet

> Doc narrative qui explique **comment fonctionne la collecte** end-to-end :
> du lancement de CARLA jusqu'aux labels YOLO 12-classes prêts à entraîner.
> À lire **avant** de mettre les mains dans le code.
> Pour la doc de référence par fichier, voir [README.md](README.md).
> Pour les pièges connus et leurs solutions, voir [PROBLEMS_AND_FIXES.md](PROBLEMS_AND_FIXES.md).

---

## Pipeline en 1 schéma

```
   ┌─────────────────────┐
   │  CARLA server       │  Town01 / ClearNoon / 36 feux / 40 NPCs
   │  (synchrone 20 FPS) │
   └──────────┬──────────┘
              │
              │ world.tick()   (toutes les 50 ms)
              ▼
   ┌─────────────────────────┐
   │ DatasetCollector        │  src/dataset/collector.py
   │  - spawn ego + NPCs     │
   │  - attach 4 sensors     │  rgb / depth / semantic / instance
   │  - capture toutes 2 s   │  (= 1 frame / 40 ticks)
   └──────────┬──────────────┘
              │
              ▼  (par frame)
   ┌─────────────────────────┐
   │  Sauvegarde modalités   │  src/dataset/sensors.py (writers)
   │   images/*.jpg          │
   │   depth/*.npy           │
   │   semantic/*.npy        │  + viz/semantic/*.png
   │   instance/*.npy        │  + viz/instance/*.png
   └──────────┬──────────────┘
              │
              ▼  (lit le buffer instance live)
   ┌─────────────────────────┐
   │  YoloLabeler.compute    │  src/dataset/yolo_labels.py
   │  - bboxes via masks     │  → labels_yolo/*.txt (4 classes raw)
   └─────────────────────────┘

  ----------- fin du temps réel CARLA -----------

   ┌─────────────────────────┐
   │  enrich_labels (offline)│  src/dataset/enrich_labels.py
   │  - HSV pour feux        │  raw 2 → final 2/3/4
   │  - template pour signs  │  raw 3 → final 5..11
   └──────────┬──────────────┘
              │
              ▼
       labels_yolo_color/*.txt   (12 classes finales, prêt YOLO)
```

## Étape par étape

### 1. Prérequis CARLA

- **CARLA 0.9.16** tournant sur `localhost:2000` (port par défaut)
- Vérification rapide :
  ```powershell
  uv run python -c "import carla; print(carla.Client('localhost', 2000).get_server_version())"
  ```
- Si tu collectes longtemps, garde CARLA up : redémarrer le serveur invalide tous les `actor.id` du monde.

### 2. Configuration d'une run

Paramètres principaux (cf. [run_collection.py](run_collection.py)) :

| Param | Défaut | Effet |
|---|---|---|
| `--town` | `Town01` | Carte CARLA. Town01 simple, Town03 rond-point, Town04 highway, Town05 urbain dense |
| `--weather` | `ClearNoon` | Preset météo CARLA (`ClearNoon`, `CloudyNoon`, `WetNoon`, `ClearSunset`, …) |
| `--duration` | `30` | Secondes de simulation. `n_frames = duration / 2` (cf. `--every-n-ticks`) |
| `--npcs` | `10` | Voitures NPC en autopilot. Plus = trafic plus dense |
| `--every-n-ticks` | `40` | Capture toutes les N ticks @ 20 FPS. `40` = 2 s. Baisser pour plus de frames mais redondants |
| `--seed` | `42` | Reproductibilité (spawn, traffic manager) |

### 3. Setup interne du collector

Dans `DatasetCollector._setup()` ([collector.py](collector.py)) :

1. **Connect + load_world** : recharge la map, invalide tous les actors précédents
2. **Set weather** + **mode synchrone** (20 FPS exact, sinon labels désalignés)
3. **Traffic Manager synchrone** (avec seed si fourni)
4. **Spawn ego** : Tesla Model 3, point de spawn aléatoire de la map
5. **Spawn N NPCs** : véhicules variés, autopilot via Traffic Manager. *Walkers : TODO non spawnés actuellement.*
6. **Attach sensors** : pour chaque ligne de `SENSOR_SPECS` (rgb/depth/semantic/instance) → spawn un `CameraSensor` au POV partagé (`CAMERA_LOCATION = (0.30, 0.0, 1.50)`, `pitch=-5`)
7. **Collision sensor** séparé, sert à marquer `is_collision` dans le manifest
8. **YoloLabeler** instancié avec ref vers le sensor instance (pour lire le buffer live)
9. **CommandPlanner + ExpertDriver** pour le manifest (commande HN + steer/throttle/brake)
10. **Warmup 10 ticks** pour que les buffers se remplissent avant la première capture

### 4. Boucle de capture

Dans `DatasetCollector.run()` :

```
for tick in range(target_ticks):
    world.tick()                                 # CARLA avance d'1 frame
    if tick % capture_every_n_ticks != 0:
        continue                                 # skip jusqu'au prochain "vrai" frame
    
    # Avance la fenêtre de collision (5 dernières frames)
    collision_events_recent.append(0)
    
    # Sauvegarde 4 modalités via writers
    for sensor in self._sensors.values():
        sensor.save(output_dir, frame_id)
    
    # Bboxes YOLO depuis le buffer live de instance_sensor
    self._yolo_labeler.save(output_dir / "labels_yolo" / f"{frame_id:06d}.txt")
    
    # Ligne manifest (command, vitesse, contrôles, collision flag)
    self._manifest.append_row(...)
```

### 5. Comment compute_labels produit les bboxes

Cf. [yolo_labels.py](yolo_labels.py) — résumé :

**Classes "dynamiques"** (vehicle, walker) : via le **mask d'instance**
- `np.unique(instance_map)` donne tous les objets de la scène
- Pour chacun : récupère sa classe via le canal R, calcule la bbox via `np.where`, applique les filtres
- Filtres : taille min 6 px, fill_ratio ≥ 0.25 (anti-fuite grilles), exclu si bbox touche le bord bas (capot ego), exclu si `instance_id == ego.id & 0xFFFF`, exclu si distance Carla > 80 m

**Classes "statiques"** (traffic_light, traffic_sign) : via **composants connexes sur le mask sémantique**
- Pourquoi pas l'instance mask : les feux et panneaux statiques de la map ont des `instance_id` qui ne correspondent à aucun actor Python-accessible. Solution validée : `cv2.connectedComponents` sur les pixels `class_id == 7` (TL) ou `class_id == 8` (Sign) — un blob = un label.
- Filtres min pixels : 25 pour TL, 30 pour signs. Drop les feux fragmentés derrière la végétation.

Détail des bugs et fix dans [PROBLEMS_AND_FIXES.md](PROBLEMS_AND_FIXES.md).

### 6. Post-process : enrich_labels

Cf. [enrich_labels.py](enrich_labels.py). Lit `labels_yolo/` (raw 4 classes) et écrit `labels_yolo_color/` (final 12 classes).

**Pour chaque ligne** :

| Cls raw | Action | Cls final |
|---|---|---|
| 0 vehicle | passe-through | 0 |
| 1 walker | passe-through | 1 |
| 2 traffic_light | crop image + analyse HSV (rouge/jaune/vert dans la lentille) | 2/3/4 ou drop |
| 3 traffic_sign | crop + template matching contre 7 templates synthétiques (30..90) | 5..11 ou drop |

**Templates panneaux** : générés au runtime par PIL (cercle rouge + chiffre noir centré, 64×64). Aucun fichier sur disque.

**Drops** : si aucun canal HSV n'atteint le seuil (`_MIN_TL_COLOR_PIXELS = 8`) ou si aucun template n'a un score ≥ `_MIN_TEMPLATE_SCORE = 0.45`, le label est **supprimé** des labels finaux. Mode `--debug-drops` sauvegarde les crops droppés dans `debug_dropped_tl/` et `debug_dropped_signs/` pour inspection visuelle.

### 7. Cleanup

`_cleanup()` est dans un `try/finally` autour de `run()`. Ordre critique appris à la dure :

1. **Stop listeners** (`sensor.stop()`) avant tout
2. **Batch destroy synchrone** (`apply_batch_sync`) en un seul appel — sequential destroy crashe le runtime C++
3. **Null des refs Python** seulement après confirmation du serveur
4. **Restore CARLA settings** (sortir du mode synchrone)

Ne touche pas ce code sauf nécessité (voir [PROBLEMS_AND_FIXES.md](PROBLEMS_AND_FIXES.md) si tu dois).

## Structure de sortie d'une run

```
data/runs/<YYYY-MM-DD>_<town>_<weather>/
├── images/            ← RGB JPEG, 1 frame / 2 s
├── depth/             ← float32, mètres, plafonné 100 m
├── semantic/          ← uint8, class_id CityScape 0-28
├── instance/          ← uint32, (class<<16) | inst_id
├── viz/
│   ├── semantic/      ← PNG palette CityScape (debug visuel)
│   └── instance/      ← PNG couleur HSV par instance (debug visuel)
├── labels_yolo/       ← raw 4 classes (collector)
├── labels_yolo_color/ ← final 12 classes (après enrich_labels)
├── debug_dropped_tl/        ← optionnel (--debug-drops d'enrich_labels)
├── debug_dropped_signs/     ← optionnel
├── manifest.csv       ← une ligne par frame (vitesse, command, controles, collision)
└── metadata.json      ← métadonnées globales (run_id, seed, fps, npcs…)
```

## Commandes courantes

### Une run simple

```powershell
uv run -m src.dataset.run_collection --town Town01 --weather ClearNoon --duration 600 --npcs 30
uv run -m src.dataset.enrich_labels --run data/runs/<run>
uv run -m src.tools.visualize_fiftyone --run data/runs/<run>
```

### Multi-runs sur plusieurs maps/météos

```powershell
uv run -m src.dataset.collect_multi `
    --maps Town01,Town03 `
    --weathers ClearNoon,CloudyNoon `
    --frames-per-run 500 `
    --npcs 30 `
    --enrich
```

→ 4 runs (2 × 2), ~1000 s chacune, enrich_labels lancé après chaque.

### Inspection d'une run

```powershell
# Compteurs par classe + disque utilisé
uv run -m src.dataset.inspect_run --run data/runs/<run>

# Diagnostic d'une frame précise (mask instance)
uv run -m src.dataset.inspect_run --run data/runs/<run> --frame 17
```

### Diagnostic des drops d'enrichissement

```powershell
uv run -m src.dataset.enrich_labels --run data/runs/<run> --debug-drops
```

Puis ouvre `data/runs/<run>/debug_dropped_tl/` et `debug_dropped_signs/` dans
l'explorateur Windows en mode vignettes. Chaque PNG a un nom encodant les
compteurs HSV (TL) ou scores templates (signs) qui te dit pourquoi le label
a été droppé.

## Pièges à connaître

Tous les détails dans [PROBLEMS_AND_FIXES.md](PROBLEMS_AND_FIXES.md). En bref :

- **Si tu modifies `yolo_labels.py`** : il faut **refaire une collecte** pour voir l'effet. Les `.txt` ne sont pas régénérés sur les anciennes runs.
- **Si tu inspectes une vieille run avec un actor.id** (genre `world.get_actor()` sur un instance_id du mask) : si CARLA a été rechargé entre-temps, l'actor n'existe plus. C'est par design.
- **Feux orange rares** : sur un cycle de 30 s, le Yellow dure ~3 s. Sous-représenté → augmentation ou pondération à l'entraînement.
- **Walkers absents** : `_spawn_npcs` ne spawne pas encore de piétons. Classe `walker` reste vide en attendant.

## Prochaines étapes (TODOs côté dataset)

- Walkers + WalkerAIController pour avoir des piétons dans le dataset
- Augmenter les seuils HSV / templates si les drops d'enrich_labels grimpent au-dessus de 15-20 %
- Ajouter une colonne `speed_limit_kmh` au manifest (via `ego.get_speed_limit()`) — utile à l'IA centrale même si YOLO loupe le panneau
