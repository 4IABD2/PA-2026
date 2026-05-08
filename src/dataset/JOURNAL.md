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

---

## 2026-05-08 (suite — segmentation cameras + POV change)

**Avancement** :
- Ajout des deux capteurs CARLA `sensor.camera.semantic_segmentation` et `sensor.camera.instance_segmentation` au pipeline du `DatasetCollector`. Deux nouveaux modules `semantic_capture.py` + `instance_capture.py` strictement parallèles à `depth_capture.py`.
- Sortie étendue : 4 nouveaux dossiers par run (`semantic/`, `semantic_viz/`, `instance/`, `instance_viz/`). Stockage double : `.npy` machine-readable + `.png` colorisé pour trace visuelle directement archivée.
- Refactor mineur du collector : la liste des sous-dossiers à créer est centralisée dans la constante module `OUTPUT_SUBDIRS`, plus de `mkdir` hardcodés disséminés.
- 5 nouveaux smoke tests (passe de 7 à 12 au total) : décodage sémantique, palette CityScape, packing instance uint32, viz instance déterministe, contenu d'`OUTPUT_SUBDIRS`. Tous offline, pas de connexion CARLA requise.
- Documentation à jour : `src/dataset/README.md` + `README.md` racine étendus (structure de sortie, encodage des nouveaux `.npy`, nouvelle convention POV avec justification).
- **Changement de la POV caméra équipe** : `(0.5, -0.3, 1.2)` → `(0.30, 0.0, 1.50)` (cf. Difficultés ci-dessous).
- **Robustification du cleanup CARLA** : passage de `client.apply_batch` à `apply_batch_sync(cmds, True)` avec stop explicite des sensors avant et nullification des refs Python après la confirmation serveur. Élimine les warnings `sensor object went out of the scope but the sensor is still alive in the simulation` qui apparaissaient sur les 3 capteurs avec callback (depth + semantic + instance) à cause d'une race entre le GC Python et le destroy serveur asynchrone.

**Difficultés** :

1. **Palette CityScape obsolète** — Premier e2e du capteur sémantique : visualisations toutes uniformes en violet, distribution rapportée `class IDs [1, 14]`. J'ai d'abord cru à un bug du capteur, en réalité ma palette hardcodée correspondait à l'ancienne mapping CARLA (≤ 0.9.12, 23 classes) alors que CARLA 0.9.13+ utilise une mapping réorganisée à 29 classes. Concrètement : class 1 est passée de "Building" (gris) à "Roads" (rose-violet), class 14 de "Ground" (violet) à "Car" (bleu foncé), nouvelles classes 24=RoadLine et 25=Ground apparaissent en fin de liste. Palette refaite from scratch (29 entrées indices 0-28) à partir de la doc officielle CARLA 0.9.16 + tests smoke ajustés (asserts sur Roads=128,64,128 et Car=0,0,142 au lieu de Road à l'index 7 dans l'ancien layout).

2. **Caméra à l'intérieur du body Tesla — la grosse découverte** — Avec la palette corrigée, deuxième e2e : 99.98% des pixels en classe Car (le mesh de la carrosserie ego), 0.02% Roads par une fente géométrique. Les capteurs `semantic_segmentation` et `instance_segmentation` de CARLA **ne respectent pas la transparence des matériaux** alors que `sensor.camera.rgb` la respecte. Notre POV historique `(0.5, -0.3, 1.2)` plaçait la caméra physiquement à l'intérieur de la cabine de la Tesla Model 3 : RGB voyait à travers le pare-brise (verre transparent en RGB), mais semantic et instance voyaient le mesh solide du body partout autour. Données semantic/instance totalement inutilisables pour Karim (lignes) et Franck (bboxes via masks d'instance).

   **Pistes natives explorées et écartées** : pas de mécanisme dans l'API CARLA 0.9.16 pour exclure un actor d'une caméra spécifique (vérifié dans la doc officielle 0.9.16 + 0.8.4) ; pas de `set_visible()` sur les actors ; re-tagger le mesh ne change pas le fait que le rayon s'arrête sur le body ; rendre le body transparent en alpha n'est respecté que par RGB.

   **Décision** : déplacer la caméra physiquement hors du body. Itérations testées :
   - `(1.2, -0.3, 1.2)` : juste après le pare-brise côté conducteur. Distribution réaliste mais cadrage hood-mount, pas le feeling POV conducteur souhaité.
   - `(1.5, 0.0, 1.4)` : version centrée hood-forward. Distribution riche (37% Roads, 21% Sky, 19% Vegetation, 0% Car). Mais perspective trop "plongée capot" pour le besoin équipe.
   - **Comparaison systématique de 4 variantes z** (0.30, 0, [1.45 / 1.50 / 1.55 / 1.60]) via un script preview qui spawn la Tesla statique et capture RGB + semantic_viz + instance_viz pour chaque z. Le toit Tesla Model 3 est à ~1.44m, donc z=1.45 grazing (semantic montre encore traces de body), z=1.50 propre, z=1.55 et z=1.60 propres aussi mais plus drone-like.
   - **POV finale adoptée : `(0.30, 0.0, 1.50)`, pitch=-5°** : centrée longitudinalement (y=0), légèrement en avant du centre véhicule (x=0.30, au-dessus de la position conducteur), juste au-dessus du toit (z=1.50, +6 cm de marge sur le mesh). Validation e2e : ~22% Roads, 21% Sky, 21% Vegetation, 14% SideWalks, 12% Car (NPCs devant, plus la voiture ego), distribution riche et lisibilité humaine confirmée. Script preview supprimé après usage.

**Décisions** :
- **Format machine-readable des masks** : `.npy` plutôt que PNG (cohérent avec `depth/.npy`, plus rapide à charger pour le training, ~7 GB additionnels pour 1500 frames jugé acceptable).
- **Palette sémantique** : table CARLA 0.9.13+ hardcodée (29 entrées) côté `colorize_semantic` plutôt que `carla.ColorConverter.CityScapesPalette`. Avantage : 100% testable hors CARLA, palette stable depuis 0.9.13.
- **Couleur instance déterministe** : HSV golden-ratio par `instance_id` (`hue = (iid * 0.618) % 1`, s=0.6, v=0.95). Une voiture suivie visuellement garde sa couleur à travers les frames du run → utile pour débugger une trajectoire.
- **Viz dans le dataset (pas à la demande)** : les `*_viz/*.png` sont écrits à chaque frame, archivés avec le dataset. Trade-off : +300 MB pour 1500 frames vs viz toujours dispo sans script utilitaire.
- **`labels_yolo/` reste inchangé** : `YoloLabeler.compute_labels()` continue de lever `NotImplementedError`, le collector écrit des fichiers vides via `try/except`. Le refactor (utiliser les masks d'instance pour dériver les bboxes via `find_contours`) est hors scope de ce commit, idéalement piloté par Franck.
- **Nouvelle POV `(0.30, 0.0, 1.50)`** : changement de la convention équipe (ancienne POV `(0.5, -0.3, 1.2)` documentée dans `src/ai/JOURNAL.md` 2026-05-07, à mentionner à l'équipe). Justification + comparaison des alternatives documentée dans le README racine. Aucune donnée n'a été produite avec l'ancienne POV (le `data/runs/2026-05-08_smoke_test` était une validation e2e jetable), donc pas de re-collecte à organiser.
- **Cleanup CARLA en 3 phases (stop → apply_batch_sync → null refs)** : pattern à propager si on ajoute d'autres sensors dans le futur. Évite la race entre destroy serveur asynchrone et GC Python.

**Datasets générés** :
- `data/runs/2026-05-08_seg_test/` — mini-run e2e de validation final, 10 frames Town01 ClearNoon avec les 4 modalités complètes et la POV `(1.5, 0.0, 1.4)`. Distribution semantic riche, viz directement lisibles dans VSCode.

**Prochaine étape** :
1. **Communiquer le changement de POV à l'équipe** : annoncer à Franck, Karim, Victor que la convention caméra est maintenant `(0.30, 0.0, 1.50)`. Mettre à jour `src/ai/JOURNAL.md` la prochaine fois que je touche au module IA centrale (l'ancienne entrée mentionne encore l'ex-POV).
2. **Refactor `YoloLabeler.compute_labels()`** : dériver les bboxes 2D depuis les masks d'instance maintenant disponibles dans le dataset. Algorithme : `np.unique` sur `instance_id`, `np.where` pour la bbox englobante de chaque instance, filtrer par `class_id` (ne garder que Car=14, Truck=15, Bus=16, Motorcycle=18, Bicycle=19, Pedestrian=12), normaliser au format YOLO. À piloter avec Franck.
3. Brancher le `LocalPlanner` réel dans `CommandPlanner`.
4. Première vraie collecte 500-1000 frames sur Town01 pour démarrer le training CIL.
