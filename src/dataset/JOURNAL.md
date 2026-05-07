# Journal — `src/dataset/`

> **Owner principal** : Frédéric (script de collecte). Contributeurs : Franck, Karim.
> **Format** : entrée datée à chaque session significative (avancée, difficulté résolue, décision, format).
> **À la fin du projet** : un agent synthétisera tous les `JOURNAL.md` pour produire le bilan global.

---

## Template d'entrée

```
## YYYY-MM-DD
**Avancement** :
**Difficultés** :
**Décisions** :
**Datasets générés** :
**Prochaine étape** :
```

---

## 2026-05-07

**Avancement** :
- Squelette complet du module `src/dataset/` implémenté : `collector.py` (orchestrateur), `camera_capture.py`, `depth_capture.py`, `yolo_labels.py`, `command_planner.py`, `expert_driver.py`, `manifest_writer.py`.
- Le collector gère le cycle complet : connexion CARLA en mode synchrone 20 FPS, spawn ego + NPCs, capture périodique (image RGB JPEG + depth `.npy` + labels YOLO + contrôles expert), écriture manifest CSV + metadata JSON, cleanup garanti en `try/finally`.
- 7 smoke tests écrits dans `benchmarks/dataset/smoke.py`, tous passent (manifest CSV, metadata JSON, format labels YOLO, décodage depth CARLA, validation du répertoire de sortie, enum commandes).
- Ajout de `pandas`, `Pillow`, `numpy` explicite et `pytest` dans le `pyproject.toml`.

**Difficultés** :
- `StrEnum` non disponible en Python 3.10 (introduit en 3.11) → contourné avec `class HighLevelCommand(str, Enum)`.

**Décisions** :
- `YoloLabeler.compute_labels()` lève `NotImplementedError` pour l'instant — la projection 3D→2D est la partie la plus complexe, Franck la branchera avec sa logique de bounding boxes. Le collector écrit un fichier label vide quand le labeler n'est pas encore implémenté.
- `CommandPlanner` tente d'instancier le `LocalPlanner` CARLA au `__init__` ; si ça échoue (import ou API), il retourne `LANE_FOLLOW` par défaut. Permet de ne pas bloquer la collecte en attendant le branchement réel.
- Spawn de walkers simplifié (véhicules uniquement pour le MVP) — les walkers CARLA nécessitent un controller AI séparé avec `start()`/`go_to_location()`, à ajouter dans un second temps.

**Datasets générés** : aucun (pas encore testé bout-en-bout contre CARLA distant).

**Prochaine étape** :
1. Test end-to-end sur le PC fixe : lancer CARLA, exécuter un mini-run (~10-20 frames) et valider la sortie (images, depth, manifest, metadata).
2. Brancher le `LocalPlanner` réel dans `CommandPlanner` pour avoir les vraies commandes haut-niveau.
3. Coordination avec Franck pour la fusion avec son `carla_dataset_generator.py` et l'implémentation de la projection 3D→2D dans `YoloLabeler`.

---

## 2026-05-08

**Avancement** :
- Premier test end-to-end du collector réussi contre CARLA distant (PC fixe via TailScale). Mini-run de 20 frames sur Town01 ClearNoon, durée ~20s.
- Résultats validés : 20 images JPEG 1280×720 (120-136 KB), 20 depth maps `.npy` float32 (valeurs cohérentes 0.12-4.25m), manifest CSV 20 lignes avec contrôles autopilot réalistes (vitesse 0-29 km/h, steer, throttle, brake), metadata JSON complète.
- Labels YOLO vides (attendu — `YoloLabeler.compute_labels()` lève `NotImplementedError`, fichiers vides écrits à la place).

**Difficultés** :
- **Crash C++ CARLA au cleanup** : `terminate called after throwing an instance of 'std::runtime_error' — trying to operate on a destroyed actor`. Le `destroy()` séquentiel des sensors (stop → destroy un par un) provoquait un abort du runtime C++ impossible à capturer en Python. Résolu en passant à `client.apply_batch([DestroyActor(...)])` qui détruit tous les acteurs en une seule commande côté serveur.
- **Callbacks sensor et références C++** : les callbacks CARLA (`sensor.listen()`) reçoivent des objets `carla.Image` dont le buffer interne peut être libéré côté C++ entre deux ticks. Corrigé en copiant les pixels en numpy directement dans le callback (via `np.frombuffer` + `.copy()`) au lieu de garder une référence au `carla.Image`.
- **Warm-up nécessaire** : les sensors ne produisent pas de données immédiatement après `attach()`. Ajout de 10 ticks de warm-up après le spawn de tous les acteurs, plus 1 tick après `load_world()` pour laisser le monde se stabiliser.

**Décisions** :
- Cleanup via `apply_batch` plutôt que `destroy()` séquentiel → plus robuste et plus rapide.
- Copie numpy immédiate dans les callbacks (camera RGB et depth) → élimine les dépendances aux objets C++ CARLA.

**Datasets générés** :
- `data/runs/2026-05-08_smoke_test/` — 20 frames, Town01, ClearNoon, 5 NPC vehicles, 0 walkers, seed=None.

**Prochaine étape** :
1. Première vraie collecte dataset (~500-1000 frames) sur Town01 pour commencer le training CIL.
2. Brancher le `LocalPlanner` dans `CommandPlanner`.
3. Coordination avec Franck pour `YoloLabeler` (projection 3D→2D).
