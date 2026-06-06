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
- Squelette complet `src/dataset/` : `collector.py` (orchestrateur), `camera_capture.py`, `depth_capture.py`, `yolo_labels.py`, `command_planner.py`, `expert_driver.py`, `manifest_writer.py`.
- Cycle complet : connexion CARLA sync 20 FPS, spawn ego + NPCs, capture périodique (RGB + depth + labels YOLO + contrôles expert), manifest CSV + metadata JSON, cleanup garanti via `try/finally`.
- 7 smoke tests offline.

**Décisions** :
- `YoloLabeler.compute_labels()` lève `NotImplementedError` — projection 3D→2D à brancher par Franck plus tard. Le collector écrit un fichier label vide dans l'intervalle.
- `CommandPlanner` : fallback `LANE_FOLLOW` si le `LocalPlanner` CARLA ne s'instancie pas (import / API).
- Walkers reportés : nécessitent un `WalkerAIController` séparé.
- `StrEnum` indispo en Python 3.10 → `class HighLevelCommand(str, Enum)`.

**Prochaine étape** :
1. Test e2e contre CARLA (PC fixe).
2. Brancher le `LocalPlanner` réel.
3. Coordination avec Franck pour la projection 3D→2D dans `YoloLabeler`.

---

## 2026-05-08

**Avancement** :
- Premier e2e collector réussi sur PC fixe (Tailscale). Mini-run de 20 frames Town01 ClearNoon.
- Sortie validée : 20 JPEG 1280×720, 20 depth `.npy` float32 (0.12-4.25m), manifest CSV avec contrôles autopilot réalistes (vitesse 0-29 km/h, steer/throttle/brake), metadata complète.

**Difficultés** :
- **Crash C++ au cleanup** : `destroy()` séquentiel des sensors provoquait un abort C++ ininterceptible en Python. Résolu en passant à `client.apply_batch([DestroyActor(...)])`.
- **Buffers C++ libérés entre ticks** : les callbacks `sensor.listen()` reçoivent des `carla.Image` dont le buffer interne disparaît. Corrigé en copiant les pixels en numpy directement dans le callback.
- **Warm-up nécessaire** : 10 ticks après spawn pour que les sensors produisent leurs premiers frames.

**Décisions** :
- Cleanup via `apply_batch` (vs `destroy()` séquentiel).
- Copie numpy immédiate dans les callbacks (élimine les refs C++).
- Convention projet : code en anglais, doc (README, JOURNAL) en français.
- POV caméra explorée sur 18 profils (3 rounds : variantes conducteur, hood-mounted, roof-mounted, micro-tuning), retenu provisoirement `(0.5, -0.3, 1.2)` pitch -5° (POV conducteur historique du README racine, présence du tableau de bord acceptée comme repère humain).
- `metadata.camera_pov` lu depuis les constantes `camera_capture.py` plutôt qu'hardcodé (évite la dérive).

**Datasets générés** : `data/runs/2026-05-08_smoke_test/` (20 frames, jetable).

**Prochaine étape** :
1. Étendre le collector aux capteurs `semantic_segmentation` + `instance_segmentation` (débloque GT lignes pour Karim et bboxes via masks d'instance pour Franck).
2. Première vraie collecte Town01 (~500-1000 frames).
3. Brancher le `LocalPlanner`.

---

## 2026-05-08 (suite — segmentation cameras + POV change)

**Avancement** :
- Capteurs `semantic_segmentation` + `instance_segmentation` ajoutés au collector via `semantic_capture.py` + `instance_capture.py` (parallèles à `depth_capture.py`).
- 4 nouveaux dossiers de sortie : `semantic/` (uint8 class IDs), `semantic_viz/` (PNG palette CityScape), `instance/` (uint32 packed `class<<16|G<<8|B`), `instance_viz/` (PNG couleur HSV déterministe par instance).
- 5 nouveaux smoke tests (12 au total).
- POV équipe changée : `(0.5, -0.3, 1.2)` → `(0.30, 0.0, 1.50)`.
- READMEs racine + dataset à jour (structure de sortie, encodage des `.npy`, justification POV).
- Pass de cleanup global du module : commentaires verbeux supprimés, docstrings `Args/Returns` redondantes virées, code resserré.
- Logs de progression ajoutés au collector (helper `_log` + prints à chaque étape setup, par frame, par 200 ticks, cleanup) — débloque le diagnostic en temps réel des stalls.

**Difficultés** :

1. **Palette CityScape obsolète** — Première viz uniformément violette. Cause : ma palette hardcodée correspondait à la mapping CARLA pré-0.9.13 (23 classes), alors que CARLA 0.9.16 utilise 29 classes réorganisées (class 1 = Roads et non Building, class 14 = Car et non Ground). Refait à partir de la doc 0.9.16, smoke tests ajustés.

2. **Caméra à l'intérieur du body Tesla** — Avec la palette corrigée, 99.98% des pixels reportés en classe Car. Les capteurs semantic / instance ne respectent pas la transparence des matériaux (alors que RGB la respecte) : depuis la POV `(0.5, -0.3, 1.2)`, ils voient le mesh solide de la carrosserie ego partout. Aucun mécanisme natif CARLA pour exclure un actor d'un sensor (vérifié docs 0.9.16 + 0.8.4). Solution : déplacer la caméra hors du body. POV finale `(0.30, 0.0, 1.50)` choisie après comparaison de 4 z-levels (1.45/1.50/1.55/1.60) — z=1.50 est le minimum qui clear proprement le toit (1.44m) tout en gardant une perspective naturelle.

3. **Warnings `sensor went out of the scope` au cleanup** — Race entre destroy serveur asynchrone (`apply_batch`) et GC Python qui finalisait les wrappers Sensor. Résolu en passant à un cleanup en 3 phases : `stop()` explicite des listeners → `apply_batch_sync(cmds, True)` (synchrone, attend le serveur) → null des refs Python.

4. **Stall réseau Tailscale → WSL → PC fixe sur les runs longs** — Tentatives de première vraie collecte (40 NPCs / 900s, puis 15 NPCs / 600s, puis 10 NPCs / 180s) chokent toutes après un nombre variable de frames (70, 7, 21). `world.tick()` en sync mode bloqué silencieusement, sans timeout effectif côté Python. CARLA serveur reste joignable (port répond), mais la connexion CARLA persistante se gèle. Pas de fix code clean trouvé pendant la session. **Décision** : la vraie collecte sera lancée localement depuis le PC fixe (host=`localhost`, plus de Tailscale), le WSL reste pour le développement code/tests/smoke. À setuper hors session.

**Décisions** :
- Stockage `.npy` brut (cohérent avec `depth/.npy`, ~7 GB/1500 frames acceptable) + viz PNG colorisée séparée pour trace visuelle directement archivée.
- Palette CityScape hardcodée en numpy (vs `carla.ColorConverter.CityScapesPalette`) → 100% testable hors CARLA.
- Couleur instance déterministe via golden ratio HSV : un véhicule garde la même couleur à travers les frames d'un run.
- `YoloLabeler.compute_labels()` reste `NotImplementedError` (refactor via masks d'instance piloté par Franck, hors scope).

**Datasets générés** : `data/runs/2026-05-08_seg_test/` (10 frames, validation e2e finale, jetable).

**Prochaine étape** :
1. Communiquer le changement de POV à l'équipe (Franck, Karim, Victor). MAJ `src/ai/JOURNAL.md` à la prochaine session sur le module IA.
2. Refactor `YoloLabeler.compute_labels()` (Franck) : dériver bboxes depuis `instance/*.npy` via `np.unique` + `np.where`.
3. Brancher le `LocalPlanner` réel.
4. Première vraie collecte 500-1000 frames Town01 — **à lancer en local depuis le PC fixe** (setup `uv` + clone repo + `host="localhost"` dans le script de collecte) pour éviter les stalls Tailscale.

---

## 2026-05-09

**Avancement** :
- Setup PC fixe en local fait : `uv` installé sur Windows, repo cloné, `data/runs/.collect_baseline.py` recréé avec `host="localhost"`.
- Première vraie collecte lancée : 3 conditions (Town01 ClearNoon, Town01 CloudyNoon, Town03 ClearNoon), 30 NPC vehicles / 900s par run, 450 frames target par run = **1350 frames total**. Estimation ~2h15 wall-clock en local. Stalls Tailscale éliminés (la connexion CARLA tient sur localhost).
- Validation que tout le manifest n'a que `command=lane_follow` : confirmation que `CommandPlanner` retombe en fallback parce que le package `agents.*` (qui contient `LocalPlanner`) n'est pas dans le wheel `carla==0.9.16` de PyPI — il vit dans `C:\Carla\PythonAPI\carla\agents\` à côté de l'install CARLA. Pas bloquant pour la V1 du modèle (signal de conditionnement absent ne dégrade pas l'imitation pure).

**Prochaine étape** :
1. **Rapatrier les données du PC fixe** vers le WSL via Tailscale + rsync, en gardant une copie sur le PC fixe (qui a la GPU, pourra servir au training plus tard) : `rsync -avz fhuang5@<pc-fixe>:/c/Users/Frédéric/Developer/PA-2026/data/runs/2026-05-08_*/ data/runs/`.
2. **V1 du modèle CIL** dans `src/ai/` : brainstorm archi (CNN backbone + tête de régression steer/throttle/brake), `Dataset` PyTorch qui lit le format actuel, training loop minimal, démo inférence boucle CARLA. La V1 s'entraînera sur le dataset actuel "lane_follow only" pour valider toute la chaîne.
3. **Brancher `LocalPlanner` réel** avant le vrai CIL : vendor le dossier `agents/` de l'install CARLA dans le repo (option propre, repo autonome) ou `sys.path.insert` du chemin local (quick & dirty, pas portable). Une fois fait, recollecter une fraction du dataset avec vraies commandes diverses pour le conditionnement CIL.

---

## 2026-06-06 — Pivot IA centrale : le dataset offline n'alimente plus le training RL

**Avancement** :
- Aucun code nouveau dans ce module. Entrée de clarification suite au pivot architectural de l'IA centrale (voir [src/ai/JOURNAL.md](../ai/JOURNAL.md)).

**Décisions** :

- **L'IA centrale ne consomme plus ce dataset pour s'entraîner.** Elle passe en RL online (boucle CARLA directe). Le pipeline "collecte → manifest → data_loader → train.py" est archivé en Phase 0.

- **Ce module reste central pour les modules de perception** : Franck a besoin des images + instance masks pour fine-tuner YOLO et entraîner le modèle de depth. Karim a besoin des images + semantic masks pour entraîner le détecteur de lignes. Le `collector.py` reste donc la source de données principale de l'équipe — juste plus pour Frédéric.

- **`labels_yolo_enriched/`** (sortie de `enrich_labels.py`) est le format final utilisé par Franck pour le training YOLO custom. Le pipeline enrich reste valide.

- **`collect_multi.py`** devient l'outil principal pour générer des datasets variés (maps × weathers) pour Franck et Karim. Plus besoin de collecter spécifiquement pour l'IA centrale.

**Prochaine étape** :
1. Coordonner avec Franck pour une nouvelle collecte multi-maps si son fine-tuning YOLO en a besoin.
2. Éventuellement : collecter des runs avec plus de virages (Town02, Town04) si Karim a besoin de diversité pour les lignes.
3. Aucun changement de code prévu dans ce module pour l'instant.
