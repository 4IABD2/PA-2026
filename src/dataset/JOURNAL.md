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
- Tout le code, docstrings et commentaires sont passés à l'anglais ; seul le JOURNAL reste en français. Convention pour la suite du projet : code en anglais, doc projet (README, JOURNAL) en français.
- **Caméra POV — exploration et décision finale** : sur la première sortie e2e, j'ai trouvé que la POV d'origine `(0.5, -0.3, 1.2)` pitch -5 cadrait trop sur le tableau de bord et le volant (~40% de l'image occupée par l'intérieur de la cabine). J'ai donc fait un travail de comparaison systématique :
  - Création d'un script de preview qui spawn une voiture statique et capture une image par profil (script supprimé après usage, plus besoin).
  - **Round 1** — 9 profils testés : 4 variantes conducteur (origine, compromise, forward, centered), 2 hood-mounted (high, low), 2 roof-mounted (top, forward), 1 bumper/dashcam.
  - **Round 2** — focus sur la variante conducteur "plus haute et reculée" : 5 profils du moins au plus agressif (`v1_slight (0.3, -0.3, 1.3)` → `v4_max (-0.3, -0.3, 1.6)`).
  - **Round 3** — micro-tuning autour de `v1_slight` : 4 sous-variantes (`v1a (0.35, -0.3, 1.33)` → `v1c (0.45, -0.3, 1.38)`).
  - Présélection : `v1a (0.35, -0.3, 1.33)` pitch -5 — un peu plus haut et reculé que l'origine, garde une partie du volant comme repère humain mais libère la route.
  - **Décision finale après délibération avec Franck** : on revient à la POV d'origine `(0.5, -0.3, 1.2)` pitch -5. Raison : c'est la convention déjà actée dans le README racine et dans le journal `src/ai/`, garder la cohérence évite de devoir re-collecter le peu de données qu'on aurait pu déjà produire et simplifie la coordination équipe. La présence du volant/tableau de bord est jugée acceptable comme repère visuel humain.
- **Centralisation des constantes POV** : suite à ce travail, le `camera_pov` du `metadata.json` lit désormais les constantes `CAMERA_LOCATION` / `CAMERA_ROTATION_PITCH` de `camera_capture.py` au lieu d'avoir des valeurs hardcodées. Évite la dérive entre le sensor réel et la metadata.

**Datasets générés** :
- `data/runs/2026-05-08_smoke_test/` — 20 frames, Town01, ClearNoon, 5 NPC vehicles, 0 walkers, seed=None.

**Prochaine étape** :
1. **Étendre le collector aux caméras GT supplémentaires de CARLA** : ajouter `sensor.camera.semantic_segmentation` et `sensor.camera.instance_segmentation` au pipeline. Bénéfices :
   - **Karim** récupère un GT propre pour la détection de lignes (les marquages au sol sont une classe distincte dans la sémantique CARLA), plus besoin de labels manuels.
   - **Franck** peut dériver les bounding boxes YOLO directement depuis les masks d'instance (`find_contours` par instance ID), ce qui résout proprement le `NotImplementedError` actuel de `YoloLabeler.compute_labels()` sans avoir à coder la projection 3D→2D matricielle.
   - Format de sortie envisagé : `.npy` brut (uint8 pour la sémantique = class IDs CARLA 0-22, uint32 pour l'instance = `class_id<<16 | instance_id`). Pas de PNG visualisable, palette régénérable au besoin.
   - Coût disque estimé : +~6.7 GB pour 1500 frames (acceptable). Design discuté et acté en session, à implémenter au prochain pass.
2. Première vraie collecte dataset (~500-1000 frames) sur Town01 pour commencer le training CIL.
3. Brancher le `LocalPlanner` dans `CommandPlanner`.
