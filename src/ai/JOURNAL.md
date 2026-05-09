# Journal — `src/ai/`

> **Owner** : Frédéric Huang
> **Format** : entrée datée à chaque session significative (avancée, difficulté résolue, décision, benchmark).
> **À la fin du projet** : un agent synthétisera tous les `JOURNAL.md` pour produire le bilan global.

---

## Template d'entrée

```
## YYYY-MM-DD
**Avancement** :
**Difficultés** :
**Décisions** :
**Benchmarks** :
**Prochaine étape** :
```

---

## 2026-05-07

**Avancement** :
- Architecture du repo entièrement scaffolded (READMEs par module, contrats `src/interfaces/`, structure `src/ai/{models,training,inference}/`, `JOURNAL.md` initiés, `.gitignore` configuré pour les docs perso).
- Spec design du pipeline d'apprentissage consigné dans une note de design interne (non commitée).
- Setup CARLA sur le serveur Linux multi-GPU démarré : Docker 28.1.1 OK, 2x RTX A6000 disponibles (GPU 1 libre), ports 2000-2002 libres. NVIDIA Container Toolkit pas encore installé sur le serveur (à installer via sudo).

**Difficultés** :
- NVIDIA Container Toolkit absent du serveur → blocage pour lancer le conteneur CARLA en GPU. Installation à venir, nécessite sudo et coordination avec les autres users (restart Docker).

**Décisions** :
- **Approche d'apprentissage révisée : RL hybride** (CIL pre-training puis RL fine-tuning) au lieu de CIL pur. Phase 1 = PilotNet entraîné par imitation sur dataset autopilot CARLA (échéance courte). Phase 2 = fine-tuning du modèle pré-entraîné via PPO ou SAC en boucle CARLA (échéance fin de projet). Justification : on veut une vraie composante RL pour la défense (sujet originel parlait de "deep reinforcement learning") sans le risque d'un RL pur from scratch qui peut ne jamais converger en 3 mois sur la conduite multi-objectifs.
- **Stack** : TensorFlow / Keras (cohérence avec le code existant). Stable-Baselines3 envisagé pour la Phase 2, à confirmer.
- **Caméra POV partagée équipe** : `Location(x=0.5, y=-0.3, z=1.2), Rotation(pitch=-5)`, mode synchrone CARLA 20 FPS, capture 1 frame toutes les 2s.
- **Hardware** : tout tourne sur le serveur Linux multi-GPU (CARLA + collecte + training + démo). Le laptop sert uniquement à coder.

**Benchmarks** : néant (aucun modèle entraîné).

**Prochaine étape** :
1. Installer NVIDIA Container Toolkit sur le serveur, lancer le conteneur `carla-fhuang` en headless sur GPU 1.
2. Tester la connexion Python (`carla.Client('localhost', 2000)`).
3. Implémenter `src/dataset/` pour la collecte (en collaboration avec Franck).
4. Implémenter `src/ai/models/pilotnet.py` + `src/ai/training/train.py`.
5. Première collecte dataset (~10-15k images sur Town01) puis premier entraînement.

---

## 2026-05-07 (suite) — Pivot infra serveur → PC fixe + migration uv

**Avancement** :
- Migration de la gestion de dépendances vers **uv** (`pyproject.toml` + `uv.lock`, suppression du `requirements.txt`). Workflow `uv sync` / `uv run` documenté dans le README racine.
- Bootstrap du PC fixe perso comme machine CARLA : simulateur installé, accessible à distance depuis le laptop via tunnel SSH chiffré pour pilotage headless.
- Black appliqué sur les contrats `src/interfaces/`.

**Difficultés** :
- **Pas d'accès sudo sur le serveur du taff** : impossible d'installer NVIDIA Container Toolkit ni les libs système requises pour faire tourner CARLA. C'est ce qui a forcé le pivot d'infra (voir Décisions). Le serveur reste utile uniquement pour le training GPU.

**Décisions** :
- **Topologie 3 environnements actée** : laptop (édition code + pilotage à distance) / serveur de taf (training GPU only, pas de CARLA possible) / PC fixe perso (CARLA pour collecte de dataset + démo finale). Workflow distribué : collecte sur PC fixe → dataset transféré au serveur → training A6000 → poids retournent au PC fixe pour la démo.
- **Python 3.10 strict** dans le `pyproject.toml` (`carla==0.9.16` n'a de wheels que pour cp37/cp38/cp310). uv télécharge Python 3.10 tout seul si pas présent système → cohérent partout.
- **`carla` en dépendance principale** (pas extra optionnelle) : la wheel client n'a pas besoin des libs système, donc `uv sync` fonctionne sur les 3 envs, même là où le simulateur ne peut pas s'exécuter.

**Benchmarks** : néant.

**Prochaine étape** :
1. Squelette `src/dataset/collector.py` (avec Franck) : connexion CARLA mode synchrone 20 FPS, autopilot expert, capture toutes les 2s — format défini dans la section "Datasets" du README racine.
2. Test bout-en-bout sur Town01 (~10-20 frames) en pilotant CARLA distant depuis le laptop pour valider le pipeline.
3. En parallèle Phase 1 CIL : commencer `src/ai/models/pilotnet.py` + loader `manifest.csv`.

---

## 2026-05-09

**Avancement** :
- V1 du modèle de décision implémentée bout-en-bout : `models/pilotnet.py` (PilotNet NVIDIA + concat speed scalaire + 3 têtes nommées steer/throttle/brake), `training/data_loader.py` (lecture manifest concat 3 runs, drop collisions, split 80/20 seedé), `training/train.py` (CLI Keras avec EarlyStopping + ReduceLROnPlateau + CSVLogger + checkpoints best/last), `inference/carla_demo.py` (boucle CARLA sync 20 FPS, prédiction et apply VehicleControl).
- 4 smoke tests `benchmarks/ai/smoke.py` — output shapes/ranges du modèle, modèle entraînable sur dataset synthétique, data loader sur run synthétique avec drop collisions, déterminisme du split sur même seed.
- `config.py` centralise les hyperparams (image size 200×88, batch 64, lr 1e-4, loss weights `steer:1.0 throttle:0.5 brake:0.5`).

**Difficultés** :
- `test_pilotnet_trainable` initialement à 3 epochs sur 8 samples random : flake (Adam ne converge pas reliably en 3 steps sur du bruit, loss montait dans ~30% des runs). Passé à 20 epochs + `tf.keras.utils.set_random_seed(0)` + assertion `losses[-1] < losses[0] * 0.9` (≥10% de baisse). Test maintenant déterministe (5/5 runs consécutifs PASS).
- La preprocessing image en inférence (`carla_demo._preprocess_image`) doit être bit-pour-bit identique à celle du training (`data_loader._decode_and_preprocess`) sinon le modèle plante. J'ai factorisé en important les mêmes constantes `CROP_TOP_PX, CROP_BOTTOM_PX, IMAGE_HEIGHT, IMAGE_WIDTH` depuis `src.ai.config` partout. Petite drift résiduelle JPEG vs raw camera buffer (~3% en valeurs pixels) — acceptable pour le smoke V1 mais à monitorer.

**Décisions** :
- **Pas de branche commande HN pour la V1** — dataset actuel = `lane_follow` only (LocalPlanner pas branché). Archi `PilotNet + speed` = bon compromis : conditionnée sur la vitesse (signal pertinent immédiat), pas sur la commande (signal absent). On passera à la vraie CIL multi-têtes quand `LocalPlanner` sera vendoré et qu'on aura recollecté avec commandes diverses.
- **Pas d'augmentation V1** — on veut d'abord valider la chaîne. Brightness jitter + flip+steer en V2.
- **POV inférence importée des constantes `src.dataset.camera_capture`** (CAMERA_LOCATION, CAMERA_ROTATION_PITCH) pour garantir cohérence training / runtime — tout shift de POV invalide le modèle.
- **3 têtes nommées séparées** (steer/throttle/brake) plutôt qu'une sortie 3D unique → permet `loss_weights` natif Keras et lecture stdout plus claire des 3 contributions.
- **Crop image** : 100 px haut + 60 px bas → 1280×560 → resize 200×88. Vire le ciel et la zone basse peu utile vu la POV au-dessus du toit.
- **Filtre `is_collision`** : on drop ces frames du training (on n'imite pas l'expert pendant un crash).

**Prochaine étape** :
1. Rapatrier le dataset du PC fixe (rsync vers WSL puis vers noyse).
2. Lancer le training sur noyse (GPU A6000) — quelques secondes wall-clock attendues sur 1350 frames.
3. Rapatrier `best.keras` vers le PC fixe et lancer `carla_demo` Town01 ClearNoon. Critère de succès : la voiture avance et garde la route en environnement vide.
4. Selon résultat : passer à la V2 (augmentations, plus de data, NPCs en démo) ou aller vendor `agents/` de l'install CARLA pour avoir les vraies commandes HN et faire la vraie CIL multi-têtes.
