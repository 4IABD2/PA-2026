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

---

## 2026-05-09 (suite — training noyse)

**Avancement** :
- Première run du training V1 sur noyse (Linux, 2× RTX A6000, sans sudo). 30 epochs complets sur les 3 runs du 2026-05-08 (Town01 ClearNoon + CloudyNoon, Town03 ClearNoon), 1350 frames brutes → 1080 train / 270 val, 0 collision filtrée.
- Loss qui descend du début à la fin sans plateau ni divergence, EarlyStopping (patience=5) jamais déclenché.
- Sortie OK dans `checkpoints/pilotnet_v1/` : `best.keras` 7.0 M, `last.keras` 7.0 M, `splits.json` 147 K, `config.json`, `training_log.csv` 5.6 K (aucune NaN).
- Wall-clock 77 s sur GPU 1 (`CUDA_VISIBLE_DEVICES=1`), GPU 0 laissée libre pour les autres.

**Difficultés** :
- `pyproject.toml` déclare `tensorflow` sans extra `[and-cuda]`, donc `uv sync` n'installe pas les libs CUDA bundled (cuDNN, cuBLAS, etc.) → TF tombait en CPU silencieusement avec `GPUs: []`. Fix : `uv pip install 'tensorflow[and-cuda]==2.21.0'` qui ajoute les 12 paquets `nvidia-*-cu12` dans le venv sans toucher à `pyproject.toml` ni à `uv.lock`.
- Même après install, TF ne trouvait pas les `.so` (chargement dynamique). Il faut exporter `LD_LIBRARY_PATH` vers `.venv/lib/python3.10/site-packages/nvidia/*/lib` avant chaque appel. Solution : pattern inline dans la commande de lancement, plutôt qu'un script setup persistant qui serait spécifique à noyse.
- Driver système en CUDA 13.0 (très récent) avec runtime 12.x dans les wheels Python — pas de souci grâce à la forward-compatibility des drivers NVIDIA.

**Décisions** :
- `pyproject.toml` non modifié : la fix CUDA reste locale au venv de noyse, n'impacte pas le PC fixe Windows (CARLA + démo) où la stack est différente. Si on veut industrialiser plus tard, on rajoutera l'extra côté pyproject avec un marker plateforme.
- Pas créé de helper script `scripts/setup_gpu_env.sh` cette fois pour ne pas polluer le repo avec un fichier machine-specific. À reconsidérer si on relance régulièrement.

**Benchmarks** :
- `val_loss` final : **0.0735** (epoch 30) — départ 0.1966 (epoch 1), descente quasi-monotone, deux stagnations courtes (e5–6 et e14–15) résorbées sans intervention.
- Détail par head sur la val : `val_throttle_loss` 0.142 → 0.045 (descente nette), `val_brake_loss` 0.241 → 0.083 (descente nette), `val_steer_loss` 0.0046 → **0.0067** (légère dégradation).
- Le head steer dégrade un peu pendant que throttle/brake convergent → c'est le mode collapse "predict 0 partout" attendu vu la distribution observée par le data loader : `[data] steer counts=[0, 0, 0, 25, 281, 1025, 11, 7, 1, 0]`. Le bin central concentre 76 % des frames, 304/1350 (~22.5 %) sont dans des bins légèrement décalés, et seulement 19 frames (~1.4 %) ont `|steer| > 0.4`. Le réseau apprend à imiter le prior trivial sur la majorité et perd un peu de précision quand il s'aventure ailleurs.
- Distribution speed : `counts=[695, 93, 95, 402, 10, 9, 12, 8, 8, 18]` sur `[0..90 km/h]`, soit deux modes (arrêt/lent ~50 % et croisière ~30 km/h ~30 %), max ~88 km/h conforme aux notes.
- Wall-clock training 77 s. Setup env (uv sync + install CUDA libs) ~3 min. Smoke tests 34 s, 4 passed.

**Prochaine étape** :
1. Rapatrier `checkpoints/pilotnet_v1/best.keras` vers le PC fixe pour lancer `carla_demo` Town01 ClearNoon.
2. Critère réel à observer en sim : est-ce que la voiture braque effectivement dans les virages, ou reste-t-elle "tout droit" à cause du steer collapse ? Le `val_steer_loss` bas est trompeur (il bénéficie du prior trivial), c'est le comportement visuel qui tranchera.
3. Si la voiture ne tourne pas : avant de passer à V2 (augmentations, plus de data), tester un rééquilibrage steer (oversampling des frames `|steer|>0.05`) dans le data loader, ou une loss `huber`/`weighted_mse` sur le head steer. Si elle tourne raisonnablement : V1 considérée comme baseline acquise, on enchaîne sur la collecte de plus de data variée (Town02/04/05, dynamic_weather, NPCs).

---

## 2026-05-10 (démo CARLA, recording vidéo, analyse V1)

**Avancement** :
- Démo CARLA branchée sur le checkpoint V1 du PC fixe : 3 patches successifs ajoutés à `src/ai/inference/carla_demo.py` au fil des problèmes runtime :
  1. **Mutex throttle/brake** — sans ça, le modèle commande les deux pédales en même temps (autopilot du dataset le faisait) et CARLA inhibe la traction → voiture reste collée.
  2. **Kickstart override** à `speed < 3 km/h` (`KICKSTART_THROTTLE = 0.6`) — sinon, après le premier crash, le modèle voit "scène stationnaire" → output `brake = 0.37` → reste à 0 km/h → re-confirme brake → point fixe stable infini.
  3. **Boucle respawn** sur sensor `sensor.other.collision` (2 s wait + despawn + spawn au prochain `random.choice(spawn_points)`) — pour que la démo tienne 120 s sans intervention manuelle quel que soit le nombre de crashs.
- Plus chase cam spectator (suivi auto à -6m, +3m, pitch -15°) et flag `--record DIR` qui dump des JPEG quality 85 avec HUD overlay (raw vs applied controls + speed + respawn count) → assemblage MP4 via ffmpeg, commande imprimée à la fin du run.
- Run de démo 120 s sur Town01 ClearNoon récupéré sur noyse : `logs/demo_v1_2026-05-10/demo.log` + `demo.mp4` (35 MB).
- Notebook `benchmarks/ai/v1_analysis.ipynb` (29 cells, 5 sections : training curves, dataset distrib, val predictions, bench inférence, demo timeline) → 7 PNG dans `benchmarks/ai/figures/v1/` + `checkpoints/pilotnet_v1/metrics.json` consolidé.
- README `src/ai/README.md` réécrit : sections training et inference avec les commandes réelles, args expliqués, bloc Linux+CUDA pour noyse, bloc `--record` + ffmpeg, et récap des 3 post-process avec les constantes pour retrouver direct où les tweaker.

**Difficultés** :
- 3 itérations sur la démo avant qu'elle tourne convenablement, chaque problème étant le symptôme d'un bias du dataset (pédales conflictuelles, bias "stationary→brake", crashs fréquents).
- **Mon diagnostic initial du biais "tourne à gauche" était grossièrement faux**. J'avais lu les 10 bins de 0.2 width du data loader et fait `bins négatifs / bins positifs = 306/19 = 16×`. En réalité ces bins capturent énormément de micro-jitter autour de 0 (le data loader bin `[0.0, +0.2)` contient 1025 frames, presque toutes entre 0 et 0.05). Avec un seuil propre `|steer| > 0.05`, le ratio L/R réel est de **1.28×** (55 vs 43 sur 1350 frames). Le dataset est à **92.7 %** "tout droit", pas biaisé gauche.
- Conséquence : tout le narratif "le modèle a appris le biais gauche du dataset" était mal calibré. Le vrai problème est ailleurs (cf. benchmarks).
- Sur noyse, `LD_LIBRARY_PATH` relatif au venv s'évapore après un `cd benchmarks/ai`, donc le notebook a tourné en CPU au lieu de GPU. Pas grave pour 270 frames de val (19 ms/frame CPU c'est large), mais à savoir si on relance ailleurs.

**Décisions** :
- **Mutex / kickstart / respawn** = patches runtime explicitement non figés. Le mutex en particulier sacrifie le head qui apprend le mieux (brake, R² 0.64) au profit du moins fiable (throttle, R² 0.33). À refondre en V2 — peut-être un blending continu plutôt qu'un winner-takes-all.
- **Recording activable** via flag opt-in pour pas alourdir un run normal. JPEG 85 plutôt que PNG : 360 MB pour 120 s, jetable une fois le MP4 généré.
- **Notebook séparé** des smoke tests dans `benchmarks/ai/` : smoke = unit/CI rapide, notebook = analyse interactive lente. Cohabitent OK.
- **`metrics.json` à côté du checkpoint** plutôt que dans `benchmarks/` : c'est une métadonnée du modèle, pas du benchmark.
- `jupyterlab` + `nbformat` installés en venv-local (`uv pip install`) sans toucher pyproject.toml — même pattern temporaire que le fix CUDA. À industrialiser dans le pyproject si on en fait régulièrement.

**Benchmarks** : analyse complète + figures dans [`benchmarks/ai/v1_analysis.ipynb`](../../benchmarks/ai/v1_analysis.ipynb), chiffres consolidés dans [`checkpoints/pilotnet_v1/metrics.json`](../../checkpoints/pilotnet_v1/metrics.json). Synthèse :
- **Démo Town01 ClearNoon 120 s** : 9 respawns (1 crash / ~13 s), top 22.6 km/h, épisode médian ~10 s.
- **Inférence** : 19 ms/frame CPU (noyse) → 51 FPS, marge OK vs cible 20 FPS = 50 ms.
- **Val 270 frames** : `R² steer = -0.67` (pire que la moyenne, **échec net**), `R² throttle = +0.33`, `R² brake = +0.64` (head le plus fiable, ironiquement celui que le mutex sacrifie le plus).
- **Sur-activation** : modèle prédit `|steer| > 0.05` 8× plus que la réalité, et `brake > 0.1` 93 % du temps vs 55 % truth.
- **Dataset 1350 frames** : 92.7 % `|steer| ≤ 0.05`, ratio L/R réel = **1.28×** (pas le 16× lu naïvement sur les bins de 0.2), seulement 19 frames `|steer| > 0.4`.

**Prochaine étape** :
1. **Vraie priorité V2 = comprendre pourquoi steer R² est négatif**. Pistes (probablement plusieurs causes combinées) :
   - signal steer trop faible vs bruit modèle
   - MSE inadaptée pour cibles à 92 % nulles (passer en Huber, ou en classification soft sur des bins, ou en MSE pondérée)
   - capacity insuffisante du PilotNet baseline pour cette tâche
   - pas assez de signal informatif côté caméra (une intersection vide vue de la POV haute donne ~aucune info sur la direction à prendre)
2. **Collecter de la data avec virages serrés** : 19 frames à `|steer| > 0.4` sur 1350 ne suffisent pas. Cible : ≥100 frames `|steer| > 0.4` pour donner quelque chose à apprendre au head.
3. Reconsidérer le mutex throttle/brake — blending continu (`brake -= throttle` ou similaire) pour pas perdre le head brake qui marche.
4. Si on garde le kickstart, le déclencher sur **temps stuck** (ex : `speed < 1 km/h pendant >2 s`) plutôt qu'à chaque tick à basse vitesse, pour casser le cycle "pousse-mur".
5. Flip horizontal aug désormais bonus (dataset ~équilibré L/R), pas prioritaire.
6. Side-quest perso : nettoyage du code, repasser toute la session V1 en revue à tête reposée. Peut-être adapter le `pyproject.toml` pour formaliser jupyter et CUDA si on tient à la reproductibilité totale.

---

## 2026-06-06 — Pivot Phase 0 → Phase 1 RL

**Avancement** :
- Revue complète du travail équipe depuis le début du projet.
- Décision architecturale actée : **abandon de l'approche CIL, pivot vers RL par renforcement** (PPO, Stable-Baselines3).
- README `src/ai/` et README racine mis à jour pour refléter Phase 0 (archivé) vs Phase 1 (en cours).

**Décisions** :

- **Phase 0 CIL = archivé, pas supprimé.** Tout le code `v1_*` reste intact. C'était une introduction valide à la stack CARLA + un baseline utile pour la défense.

- **Pourquoi on abandonne le CIL** : R² steer = -0.67 l'illustre. L'imitation copie un comportement — elle ne peut pas dépasser l'expert et s'effondre sur les cas rares (virages). Ajouter de la data aurait pu aider, mais le plafond structurel de l'imitation learning sur ce type de tâche est trop bas.

- **Pourquoi le RL pur sur pixels est écarté** : espace d'état trop grand (200×88×3 pixels), millions de steps nécessaires, pas faisable dans les délais du PA.

- **Architecture retenue** : PPO (Stable-Baselines3) sur observations structurées — 7 scalaires : `speed_norm`, `cmd_one_hot (×3)`, `center_offset`, `nearest_obstacle_m`, `heading_error`. Métriques de perception depuis stubs GT CARLA d'abord, vrais modèles ensuite (swap transparent via `src/interfaces/`).

- **Navigation Victor branchée** : `nav.plan()` au reset de chaque épisode, `nav.next_command()` à chaque intersection → HighLevelCommand encodée en one-hot dans l'observation.

- **Reward function** :
  - `r_speed = (speed_kmh / MAX_SPEED) × 0.5`
  - `r_center = (1 − |center_offset|) × 0.3`
  - `r_alive = +0.01 / step`
  - `r_offroad = −0.5` si hors route
  - `r_collision = −1.0 + done=True`

- **Pas de pretraining CIL → RL** : avec des observations compactes, PPO from scratch devrait converger en quelques heures. Le transfert de poids TF → SB3 aurait ajouté de la complexité sans gain prouvé.

**Benchmarks** : néant (session de design, pas de code).

---

## 2026-06-06 (suite) — Skeleton Phase 1 RL complet, TDD

**Avancement** :
- Skeleton Phase 1 entièrement implémenté via TDD, 40 tests, 0 CARLA requis offline :
  - `src/ai/rewards/reward_fn.py` — fonction pure `compute_reward()`, 7 tests.
  - `src/interfaces/stubs.py` — `CarlaGTDepthEstimator` (listener depth CARLA, décodage BGRA→mètres) + `CarlaGTLaneDetector` (offset via produit vectoriel waypoint), 7 tests.
  - `src/ai/training/rl_env.py` — `CarlaEnv(gym.Env)` complet : spaces, reset, step, observation 7D, guard `ModuleNotFoundError` pour tests hors CARLA, 15 tests.
  - `src/ai/training/rl_train.py` — `make_model()` + `train()` PPO SB3, device CPU forcé, 4 tests.
  - `src/ai/inference/rl_demo.py` — `load_model()` + `run_episode()` avec accumulation reward et raisons d'arrêt, 7 tests.
- `gymnasium>=0.29` et `stable-baselines3>=2.3` ajoutés à `pyproject.toml` et installés.
- `scripts/run_rl_training.py` : script de lancement complet qui branche CARLA → stubs GT → nav Victor → CarlaEnv → PPO.

**Difficultés** :
- `math` pas importé au niveau module → `NameError` dans `_speed_kmh()`. Fixé.
- `step()` importait `carla.VehicleControl` directement → `ModuleNotFoundError` en tests. Isolé dans `_apply_control()` avec `try/except ModuleNotFoundError`.
- `CarlaGTDepthEstimator` : les tests ne déclenchent pas le callback `listen()`. Géré via `_last_image is None` → fallback array de zéros.
- Navigation Victor : `next_command()` incrémente `index_way` sans guard → `IndexError` en fin de route. Géré dans le script de lancement par un `NavAdapter` qui reset l'index et retombe sur `LANE_FOLLOW`.
- `plan()` de Victor appelle `MatplotVisualizer.plot_road_network()` → bloquant en headless. Contourné en passant `headless=True` dans le script (ou en désactivant l'affichage Matplotlib via `matplotlib.use('Agg')` avant import).

**Décisions** :
- `is_on_road = True` hardcodé dans `rl_env.step()` — remplacé quand Karim fournira son modèle de segmentation sémantique.
- `device='cpu'` dans les defaults PPO — MlpPolicy entraîne plus vite sur CPU que GPU (accès mémoire fréquents, batch petits).
- Chaque stub porte un commentaire `# replaced by: src/perception/...` avec le chemin attendu du vrai module.

**Benchmarks** : 40 tests offline, 0 CARLA. Wall-clock test suite : ~4 s.

**Prochaine étape** :
1. **Lancer `scripts/run_rl_training.py`** sur le PC fixe avec CARLA actif — valider que la boucle de training tourne sans crash (même 1000 steps suffit comme smoke e2e).
2. **Vérifier que la reward monte** sur les 50k premiers steps — si elle stagne à 0, investiguer l'observation (vitesse nulle ? nav commande toujours LANE_FOLLOW ?).
3. Lancer un training long (≥500k steps) sur noyse si le smoke e2e passe.
4. Swap `is_on_road` : brancher le modèle de segmentation de Karim quand disponible.
5. Swap depth estimator : brancher le modèle de Franck quand disponible.

---

## 2026-06-06 (suite) — Correction bug critique CarlaEnv.reset()

**Avancement** :
- Identification et correction du bug critique de `CarlaEnv.reset()` qui ne réinitialisait pas le monde CARLA.
- 3 nouveaux tests TDD ajoutés pour couvrir le comportement de reset (20 tests au total sur `rl_env`).
- 43 tests totaux, 43 passés, suite complète offline.

**Difficultés** :
- Le bug : après un épisode terminé par collision, PPO appelle `reset()` mais la voiture restait à la position du crash avec la même vitesse, les mêmes données capteurs périmées. L'agent repartait de la collision comme état initial → training cassé dès le 2e épisode.
- `carla.Vector3D` ne peut pas être importé hors contexte CARLA → même pattern `try/except ModuleNotFoundError` que pour `VehicleControl`, avec `set_target_velocity(None)` en fallback test.
- `self.np_random` (le RNG gymnasium) n'est disponible qu'après `super().reset(seed=seed)` — l'appel à `_teleport_to_spawn()` doit venir après.

**Décisions** :
- **Spawn aléatoire à chaque reset** : on tire un point de spawn au hasard dans `world.get_map().get_spawn_points()` via `self.np_random.integers()` (RNG gymnasium, seedable). Permet une vraie diversité des épisodes dès le départ.
- **`_WARMUP_TICKS = 5` après téléport** : on tique 5 fois pour que la physique se stabilise et que les capteurs produisent leur premier frame à la nouvelle position. Sans ça, le 1er step du nouvel épisode s'appuie sur des données de l'emplacement du crash.
- **Reset `nav.index_way` via `hasattr`** : `_NavAdapter` expose désormais `index_way` comme property qui délègue à `Navigation._nav.index_way`. L'env ne connaît pas l'implémentation concrète — si nav est un Mock (tests), `hasattr` retourne True et l'assignation passe sans effet.
- **`_last_image = None` dans reset** : évite que le 1er `_get_obs()` du nouvel épisode consomme une image périmée de l'épisode précédent. Le fallback `np.zeros((88, 200, 3))` dans `_get_obs()` gère ce cas.

**Benchmarks** : 43 tests, 43 passed, ~5 s wall-clock.

**Prochaine étape** :
1. Smoke e2e sur PC fixe : `python scripts/run_rl_training.py --timesteps 1000`. Vérifier que les épisodes 2 et suivants partent d'une nouvelle position (pas du crash précédent).
2. Si OK → training long 500k steps.
3. Ajout image dans l'observation (CombinedExtractor SB3) une fois que la boucle scalaire est validée.

---

## 2026-06-06 (suite) — Smoke e2e + correctifs post-run

**Avancement** :
- Smoke e2e validé sur le PC fixe (WSL + CARLA Windows) : pipeline complet tourne, artefacts générés (`params.json`, `model_final.zip`, `training_log.monitor.csv`, `reward_curve.png`, `demo.mp4`).
- Correctifs appliqués suite à l'analyse du premier run.

**Difficultés** :
- `uv run` sans `PYTHONPATH=.` → `ModuleNotFoundError: No module named 'src'`. Fix : `sys.path.insert(0, ...)` dans le script + `pythonpath = ["."]` dans `pyproject.toml`.
- `carla` installée en optional-dep, absente après `uv sync`. Fix : déplacée en dep principale.
- `tensorboard_log` passé à SB3 sans tensorboard installé → `ImportError`. Fix : supprimé du script.
- Monitor SB3 ajoute `.monitor.csv` au nom de fichier → `training_log.csv.monitor.csv` au lieu de `training_log.csv`. Fix : base path sans extension → `Path(str(monitor_base) + ".monitor.csv")` passé à `plot_reward_curve`.
- `newt command: X` spammé dans le terminal à chaque step → `print()` de debug dans le code de Victor (`navigation.py:125`). Fix : filtre stdout `_StdoutFilter` dans le script, bloque les lignes contenant `"newt command:"` sans toucher au code de Victor.
- **Lazy policy** : l'agent apprenait à rester immobile (reward = 0.3 r_center + 0.01 r_alive = 0.31/step × 1000 = 310 par épisode). Confirmé par le CSV du smoke (r=311, r=308). Le risque de collision dissuadait toute action. Fix : pénalité stall `−0.05` si `speed_kmh < 1.0` → rester immobile coûte plus que tenter d'avancer.
- WSL : `localhost` ne pointe pas vers Windows, CARLA inaccessible. Solution : utiliser l'IP du host Windows (`cat /etc/resolv.conf | grep nameserver`).

**Décisions** :
- `_StdoutFilter` dans `run_rl_training.py` : filtre propre, ne touche pas au code de Victor, extensible (liste `_BLOCKED`).
- `r_stall = −0.05 si speed < 1 km/h` : valeur choisie pour que avancer à 5 km/h soit toujours mieux que rester immobile, sans rendre la politique trop agressive (pas de panique = accélérer à fond pour fuir la pénalité).
- 1000 steps = smoke test (1 seul rollout PPO, 1 seule mise à jour). Résultats non significatifs en termes de conduite. Minimum pour observer quelque chose : **50 000 steps**.

**Benchmarks** :
- Smoke 1000 steps : r=311/308 (lazy policy confirmée avant fix).
- FPS CARLA-to-SB3 : ~15 FPS sur PC fixe (WSL + CARLA Windows, CPU only). Pour un training 500k : ~9h.

**Prochaine étape** :
1. Lancer training 50k–200k steps avec la pénalité stall → vérifier que la reward monte.
2. Analyser la courbe reward : si stagnation, investiguer l'observation (vitesse toujours nulle ? commande nav toujours LANE_FOLLOW ?).
3. Training long 500k steps si 50k converge.

---

## 2026-06-06 (suite) — Connexion Tailscale, correctifs vidéo, revue de cohérence

**Avancement** :
- Connexion CARLA via Tailscale validée : PC fixe Windows (100.97.91.60) joignable depuis le WSL du taff (100.70.57.54), ports 2000/2001/2002 ouverts, ~12 ms de latence, ~19 FPS en training.
- Vidéo de démo corrigée sur deux points : canaux couleurs inversés (BGRA → `arr[:,:,[2,1,0]]` → RGB correct dans `_on_camera`) et caméra d'entraînement trop petite (200×88) utilisée pour le record. Solution retenue : caméra d'entraînement reste à 200×88 (pour la bande passante Tailscale), une caméra séparée 1280×720 est spawnée uniquement pendant `record_episode` via `render_fn`. La démo se joue maintenant dans la POV équipe correcte.
- Correction de la POV caméra dans `run_rl_training.py` : j'avais mis `Location(x=0.5, y=-0.3, z=1.2)` (une valeur draft du journal jamais retenue) au lieu de la POV canonique définie dans `src/dataset/encodings.py` (`CAMERA_LOCATION = (0.30, 0.0, 1.50)`, `CAMERA_ROTATION_PITCH = -5.0`). Le script importe maintenant directement ces constantes.
- Correction d'une incohérence sur `MAX_SPEED_KMH` : `reward_fn.py` avait 50.0 alors que `rl_env.py`, `rl_demo.py` et le README utilisaient 90.0. Unifié à **90.0** partout (valeur documentée dans le README depuis le début). Le test `test_reward_components_sum_at_max` mis à jour pour appeler avec `speed_kmh=90.0`.
- README racine mis à jour : note WSL précise maintenant que Tailscale est la méthode préférée (IP stable, pas de recalcul à chaque boot) avec le nameserver comme fallback.
- README `src/ai/` corrigé : le fichier `config.py` listé dans l'arborescence n'existait pas (c'est `phase0/config.py`). Arborescence corrigée.

**Décisions** :
- **Observation 7 scalaires maintenue pour l'instant.** Le plan initial prévoyait d'ajouter l'image une fois la boucle scalaire validée. On y reviendra dès que la reward converge sur 50k steps. L'approche retenue sera `MultiInputPolicy` SB3 avec un `Dict` observation space (`"image"`: (H, W, 3) cropé/resizé, `"scalars"`: (7,)) — SB3 instancie automatiquement un CNN + MLP concaténés.
- **RL image-only techniquement possible mais écarté** : transmettre 1280×720 frames à chaque step over Tailscale ramène le FPS de ~19 à ~3 (53× plus de données réseau). Même en local, le RL end-to-end sur pixels nécessite des millions de steps. Non retenu pour ce PA.
- **Encodage one-hot des commandes nav (3 floats, pas 1 entier)** : `[cmd_left, cmd_right, cmd_straight]` plutôt qu'une valeur scalaire 0/1/2/3. Un entier imposerait une relation ordinale fictive (LEFT=1 « proche » de RIGHT=2) que le réseau devrait corriger par lui-même en gradient. Avec le one-hot, chaque commande a son neurone d'entrée dédié — les directions sont indépendantes dans l'espace d'entrée. La valeur `[0, 0, 0]` représente naturellement `LANE_FOLLOW` sans valeur spéciale à réserver.
- **`next_command()` appelé à chaque step (pas seulement aux intersections)** : la fonction de Victor incrémente `index_way` à chaque appel — elle est conçue pour avancer d'un waypoint à la fois. Dans notre env à 20 FPS, les waypoints sont consommés rapidement (une route de 200 wp ≈ 10 s). Le `_NavAdapter` gère l'épuisement de la route (`IndexError`) en retournant `LANE_FOLLOW` et remet `index_way` à 0 au `reset()`. Conséquence pratique : la commande nav change fréquemment (à chaque waypoint), ce qui donne à l'agent plus d'information directionnelle qu'un appel par intersection. Contrepartie : l'agent peut voir des transitions LEFT→STRAIGHT→RIGHT rapides même sur une ligne droite (bruit inhérent à la granularité des waypoints).

**Benchmarks** : smoke 1000 steps, ~19 FPS, vidéo 1280×720 @ 20fps générée proprement.

**Prochaine étape** :
1. Training 50k steps → vérifier que la reward progresse au-delà de la lazy policy (r ~260 à 1k steps).
2. Si converge → 500k steps sur noyse (GPU A6000, pas de Tailscale overhead).
3. Swap observation → ajouter l'image (MultiInputPolicy SB3) une fois la baseline scalaire établie.

---

## 2026-06-06 (suite) — Premier training 50k steps : convergence confirmée

**Avancement** :
- Run `ppo_v1_50k` 50 000 steps exécutée complète (~17 min à 50 FPS). Artefacts dans `runs/2026-06-06_20-15_ppo_v1_50k_50k/`.

**Benchmarks** :

| steps eval | mean_reward | ep_len_mean |
|-----------|-------------|-------------|
| 27 500 | 68 | 165 |
| 32 500 | 40 | 155 |
| 40 000 | 139 | 358 |
| 45 000 | 217 | 550 |
| 47 500 | 288 | 717 |
| **50 000** | **338** | **731** |

- Reward eval : 68 → **338** sur 50k steps. Convergence claire dans la deuxième moitié de la run.
- `ep_len_mean` eval : 165 → **731 steps** (≈36 s) — la voiture survit de plus en plus longtemps sans collision.
- Estimation conduite à 50k : ~20-30 km/h de moyenne, relativement centré. Pas encore performant mais l'agent a appris à avancer sans crasher.
- FPS : 50 FPS (meilleur qu'attendu — probablement run locale ou chemin réseau plus rapide que Tailscale).

**Points de vigilance** :
- `rollout/ep_rew_mean` descend (173 → 130) pendant que eval monte. Artefact normal : rollout = actions échantillonnées (std≈1, très bruitées), eval = actions déterministes (moyenne de la Gaussienne). Seul le `eval/mean_reward` est représentatif de la vraie policy.
- `std` reste autour de 0.97-1.0 jusqu'à la fin : la policy est encore très exploratoire, pas encore convergée. 50k steps c'est le début. La convergence réelle se joue à 500k+.
- `value_loss` augmente (0.14 → 4.46) en deuxième moitié : la value function peine à suivre les nouvelles situations explorées. Normal en phase d'expansion des comportements.

**Prochaine étape** :
1. **Training 500k steps sur noyse** (GPU A6000) : `python scripts/run_rl_training.py --timesteps 500000 --tag ppo_v1_500k --host 100.97.91.60`. Durée estimée : ~9h à 15 FPS (Tailscale noyse→PC fixe), beaucoup moins si le FPS reste à 50.
2. Observer la courbe reward autour de 200k steps — si stagnation, investiguer les hyperparams (lr, n_steps) ou la reward function.
3. Une fois la baseline 500k solide : ajouter l'image dans l'observation (`MultiInputPolicy`, `Dict` obs space).

---

## 2026-06-06 (suite) — Système de highlights + progression vidéo

**Avancement** :
- Ajout d'un système de highlights automatiques dans `src/ai/inference/rl_demo.py` : `HighlightSpec` (dataclass déclarative), `HighlightRecorder` (buffer tournant + déclenchement sur événement), `DEFAULT_HIGHLIGHT_SPECS` (6 specs par défaut).
- `record_episode()` accepte maintenant `highlight_specs` + `highlight_dir` — optionnels, zéro impact sur les tests existants (62/62 passés).
- Ajout du `CheckpointCallback` SB3 dans `run_rl_training.py` : sauvegarde un modèle tous les `timesteps/10` steps dans `checkpoints/`.
- Après training, le script enregistre une vidéo courte (200 steps) pour chaque checkpoint (`progress_XXXXk.mp4`), puis la démo finale avec highlights.

**Structure du dossier de run :**
```
runs/YYYY-MM-DD_HH-MM_<tag>/
  model_best.zip / model_final.zip
  checkpoints/rl_model_*_steps.zip   ← ~10 snapshots
  progress_0050k.mp4 … progress_500k.mp4  ← progression learning
  demo.mp4                            ← best model, épisode complet, HUD
  highlights/
    turn_left_001.mp4    (2s pre + 4s post, cooldown 20s)
    turn_right_001.mp4
    near_obstacle_001.mp4  (obstacle < 7.5 m)
    collision_001.mp4    (3s pre + 1s post)
    high_speed_001.mp4   (> 58 km/h)
    lane_drift_001.mp4   (dérive > 60% voie)
```

**Décisions** :
- **Highlights déclaratifs** : ajouter un use-case = une ligne dans `DEFAULT_HIGHLIGHT_SPECS`. Ex futur : `red_light_stop` (obs[7] = is_red_light quand les feux seront branchés), `npc_near_miss` (obs[5] < 0.1 avec NPCs).
- **Buffer rolling** : le recorder garde toujours les `pre_s` dernières secondes en mémoire, sans overhead d'écriture — le clip n'est écrit que si l'événement se déclenche.
- **Progress clips sans highlights** : les clips des checkpoints intermédiaires (200 steps) n'ont pas de highlights — trop courts et l'agent est encore en exploration, les événements n'auraient pas de sens.
- **Pourquoi les résultats à 50k semblent bons** : les observations sont Ground Truth pur (API CARLA directe). `heading_error` = angle entre la voiture et la direction de la route voisine → signal quasi-GPS. `center_offset` = géométrie CARLA exacte. Avec ces signaux parfaits, le modèle n'a pas besoin de "voir" la route, il doit juste apprendre à maintenir ces deux erreurs à zéro. C'est le comportement attendu pour Phase 1 (validation de la boucle RL). La difficulté réelle viendra quand on branchera les vrais modèles de perception (Franck/Karim) et les NPCs.

**Prochaine étape** :
1. Lancer un training intermédiaire pour valider le système de highlights et les progress clips.
2. Ajouter des NPCs dans la scène (Traffic Manager CARLA) — change radicalement la difficulté.
3. Brancher l'état des feux de signalisation comme 8ème scalaire dans l'observation.

---

## 2026-06-12 — Pipeline d'évaluation définitive + corrections benchmark

**Avancement** :
- **Fix NPC spawn dans le décor** : `_spawn_vehicle` dans `benchmark.py` spawnait au Z du waypoint surface route, le NPC se retrouvait à moitié enfoncé dans l'asphalte (visible dans le scénario `npc_follow`). Fix : `carla.Location(z=loc.z + 0.5)` dans `_spawn_vehicle`.
- **GPS route par scénario (`dest_spawn_idx`)** : pour les scénarios de jonction, la nav Victor donnait la mauvaise commande directionnelle parce que la route planifiée vers `spawn_pts[-1]` passait par le mauvais bras de carrefour. Ajout du champ `dest_spawn_idx` dans la dataclass `Scenario` — à chaque reset, la route est replanifiée vers ce spawn spécifique, ce qui force une commande nav correcte et reproductible. C'est l'équivalent d'un GPS prédéfini par scénario de test.
- **Script `find_dest_spawns.py`** : outil pour identifier les `dest_spawn_idx` corrects. Utilise l'API waypoint CARLA directement (sans `nav.plan()`) pour suivre les branches de carrefour à 40m et trouver le spawn le plus proche dans chaque direction. Résultats sur Town10HD_Opt :
  - `turn_left` (spawn 0) → `dest_spawn_idx=140` (snap 0.9m, 69.7m de distance)
  - `turn_right` (spawn 70) → `dest_spawn_idx=68` (snap 9.7m, 58.3m)
  - `junction_straight` (spawn 31) → `dest_spawn_idx=119` (snap 3.1m, 71.6m)
- **Pénalité off-route** : `_P_OFF_ROUTE = -0.5` dans `reward_fn.py`. La détection se fait dans `_is_off_route()` de `rl_env.py` via une fenêtre glissante ±60 waypoints autour de `_route_idx` (O(1) amorti). Seuil : 15m du waypoint le plus proche.
- **Critère de succès renforcé** : `_success_no_crash` et `_success_straight` requièrent désormais `max_dist_from_start >= 25m` en plus de `not terminated`. Sans ça, un modèle immobile "réussissait" tous les scénarios de phase 1.
- **Métriques riches dans `eval_model`** : refonte complète de la boucle d'évaluation. Collecte par step : trajectoire `[[x, y, yaw]]`, séries temporelles (steer, throttle, brake, heading, obstacle, nav_cmd), stats numpy (mean, std, max, pct_moving, pct_braking, off_route_pct), tracking `max_dist_from_start`, `reached_dest`, `collision_step`. Le JSON résultant permet d'analyser le comportement du modèle sans avoir à rejouer les épisodes.
- **Panel OBS sur la vidéo** : `_draw_obs_panel` dans `rl_demo.py` affiche les 7 scalaires d'entrée de la policy (speed, nav cmd, center_offset, obstacle, heading) + action courante (steer, throttle, brake) en overlay translucide en haut à droite de chaque frame de démo.
- **Sauvegarde JSON systématique** : `run_rl_training.py` sauvegarde `evals/results.json` (tous les checkpoints + best model). `run_eval.py` sauvegarde `eval_<model_stem>.json` en plus de la vidéo.
- **Corrections bugs pipeline** :
  - NameError `all_results` : variable définie à l'intérieur du bloc `if ckpt_files:` mais référencée après → déplacée avant le bloc.
  - KeyError dans `run_eval.py` : prints utilisaient `r['avg_center_offset']` et `r['avg_speed_kmh']` (anciens champs pré-refonte) → corrigés en `r['center_offset']['mean_abs']` et `r['speed']['mean']`.
- Training 300K lancé avec `--tag ppo_v5_gps` sur le PC fixe (100.97.91.60).

**Difficultés** :
- **`find_dest_spawns.py` v1 : freeze complet**. La première version appelait `nav.plan()` pour chaque point de spawn (155 appels). La fonction de Victor fait un calcul graphique complet (BFS sur le graphe de routes) à chaque invocation et bloquait la boucle pendant 20+ min. Version V2 : utilise uniquement l'API waypoint CARLA (`wp.next()`, `wp.is_junction`, `wp.road_id`) sans aucun appel à `nav.plan()`. Temps d'exécution : ~2 min.
- **Mauvaise commande nav pour turn_left** : l'overlay vidéo affichait RIGHT alors que le scénario demande un virage à gauche. Root cause : `dest_spawn_idx=None` → route vers `spawn_pts[-1]` → passage par le bras droit du carrefour → commande RIGHT dans l'obs → le modèle tournait à droite systématiquement. Fix : `dest_spawn_idx=140` force un plan vers un spawn uniquement atteignable par la gauche.
- **Faux positifs de succès** : un modèle immobile toute la durée du scénario passait `_success_no_crash` (pas de collision = succès). Constaté sur les logs de validation 100K où tous les scénarios "passaient" alors que la vidéo montrait la voiture statique. Corrigé avec le critère `max_dist_from_start >= 25m`.
- **Confusion `carla.Waypoint` vs notre dataclass `Waypoint`** : les deux coexistent avec le même nom court. `carla.Waypoint` (objet API CARLA) a `.transform`, `.is_junction`, `.next()` ; notre `Waypoint` de `navigation_types.py` a `.x`, `.y`, `.z`, `.yaw_deg`. Les erreurs de type sont silencieuses (duck typing Python) et difficiles à tracer dans la pile.

**Décisions** :
- **`dest_spawn_idx` dans `Scenario`** : on ne change pas le training (reset aléatoire sur toute la map), mais pour le benchmark on replanne vers un spawn cible spécifique après le reset. C'est un GPS prédéfini pour les tests, pas une contrainte de training. Le modèle ne sait pas à l'avance où il va — il suit juste les commandes nav que ça génère.
- **Seuil off-route à 15m** : tolère les écarts normaux dans les virages (rayon de braquage Tesla ~5m, waypoints ~2m d'espacement) sans déclencher de faux positifs, mais assez strict pour détecter une vraie sortie de route.
- **Fenêtre glissante ±60 waypoints** : couvre ~120m de route. Assez large pour les jonctions où la voiture peut "sauter" plusieurs waypoints en un seul step.
- **Métriques riches sans overhead** : toutes les collections sont des `append` Python → une seule passe numpy à la fin du scénario. Overhead mesuré : négligeable (<2ms) vs les 300 steps à 50ms chacun.

**Benchmarks** :
- Training 300K `ppo_v5_gps` en cours sur PC fixe. Résultats attendus à l'analyse du `evals/results.json`.

**Prochaine étape** :
1. Analyser `evals/results.json` du training 300K : identifier les scénarios qui échouent (off_route_pct élevé, nav_commands incorrects, trajectoire déviante).
2. Remplir `dest_spawn_idx` pour `npc_follow` et `npc_crossing` si l'analyse montre des commandes nav incorrectes.
3. Préparer la démo finale : best model sur les 13 scénarios, vidéo annotée + JSON d'analyse complet.

---

## 2026-06-12 (suite) — Enrichissement métriques eval_model

**Avancement** :
- Ajout de **6 nouvelles séries temporelles** dans le JSON de `eval_model` (`rl_demo.py`) :
  - `throttle_series`, `brake_series` — déjà collectés en mémoire mais non exportés. Permettent de tracer les patterns d'accélération/freinage step-by-step et de détecter les oscillations.
  - `heading_series` (degrés) — erreur d'orientation par step. Permet de voir si le modèle converge vers le cap correct ou oscille.
  - `obstacle_series` (mètres) — distance obstacle par step. Corrélable avec `brake_series` pour vérifier que le modèle freine bien quand un obstacle approche.
  - `off_route_series` — liste binaire `[0/1]` : 1 si la voiture était off-route à ce step. Permet de localiser exactement les points de la trajectoire où la déviation se produit.
  - `dist_series` — distance au spawn d'origine en mètres. Permet de voir si le modèle progresse vers sa destination ou tourne en rond.
- Ajout de **5 stats supplémentaires** :
  - `center_offset.pct_centered` — % du temps à |offset| < 0.2 (bien centré dans la voie).
  - `heading.std_deg` — écart-type du heading error (élevé = oscillation, faible = convergence propre).
  - `steer.mean` signé — détecte un biais systématique gauche ou droite dans le steering.
  - `throttle.std` — élevé = throttle instable / oscillant.
  - `brake.max` — valeur de freinage de pointe, utile pour `emergency_stop`.
- Ajout du tracking `off_route_series` par step via delta de `env.off_route_count` entre chaque step (pas d'appel supplémentaire à `_is_off_route`, coût nul).
- Docstring de `compute_reward` dans `reward_fn.py` complétée : description de chaque composant de reward, plages de valeurs, justification du `r_stall` et `r_off_route`.
- `src/ai/README.md` mis à jour : diagramme de la reward function avec `r_off_route`, structure des fichiers avec `benchmark.py` et les scripts, section complète "Pipeline d'évaluation benchmark" avec table des champs JSON et critères de succès par scénario, commandes utilitaires spawn.

**Difficultés** :
- Aucune — les 6 séries étaient déjà collectées en mémoire pendant la boucle. Il suffisait de les inclure dans le dict de retour. Overhead nul.

**Décisions** :
- **Séries brutes exportées** : on les inclut toutes car on ne sait pas encore lesquelles seront utiles pour l'analyse post-training 300K. Le JSON reste <1MB par run — acceptable.
- **`off_route_series` via delta de `off_route_count`** : plus simple que de ré-appeler `_is_off_route()` directement depuis `eval_model` (qui ne devrait pas connaître les internals de `CarlaEnv`). Le delta donne exactement la même information sans couplage additionnel.
- **`steer.mean` signé conservé en plus de `steer.mean_abs`** : un biais de `mean = -0.05` (légèrement gauche systématiquement) ne serait pas visible avec seulement `mean_abs = 0.05`.

**Benchmarks** :
- Training 300K `ppo_v5_gps` en cours. Le JSON enrichi sera disponible dès la fin du run.

**Prochaine étape** :
1. Analyser `evals/results.json` du training 300K avec les nouvelles séries : tracer trajectoire 2D, corrélation obstacle/brake, off_route_series superposé à la trajectoire.
2. Identifier les scénarios qui échouent et les causes (biais steering, off_route concentré à un endroit précis, heading qui n'oscille pas assez vite dans les virages).

---

## 2026-06-12 (suite) — Analyse v1, fixes reward, préparation v2

**Avancement** :
- Analyse complète de la run `ppo_v1_100k` (100k steps, baseline) :
  - Tendance d'apprentissage confirmée : reward moyen −159→−20 sur 100k steps (7× amélioration).
  - Best model sélectionné à 80k par EvalCallback (+24 sur 3 épisodes eval).
  - Succès benchmark : **0/8** scénarios Phase 1 réussis — attendu pour une baseline 100k.
- Deux bugs structurels identifiés à l'analyse :
  1. **Stall attractor** : `p_stall = −0.05` insuffisant. À 10k, la policy converge vers l'immobilisme car `r_center + r_alive − p_stall = +0.26/step > 0`. Confirmé par le benchmark (speed=0.3–0.6km/h, reward ~+78 à 10k sur les scénarios sans off-route).
  2. **Off-route immédiat au reset** : 3 scénarios sur 8 (straight, npc_follow, npc_crossing) sont à 100% off-route dès le step 1. Cause : `nav.plan(spawn, spawn_pts[-1])` retourne des waypoints dont le 1er est à >15m du spawn pour ces spawn_idx.
- Mise en place de `runs/EXPERIMENTS.md` (index transversal par run) et `ANALYSIS.md` par run.
- Corrections apportées pour v2 :
  - `reward_fn.py` : `_P_STALL` −0.05 → **−0.20** — le stall devient moins rentable que bouger (`+0.11/step` vs `+0.50/step` à pleine vitesse).
  - `rl_env.py` : ajout `_ROUTE_GRACE_STEPS = 20` — les 20 premiers steps après reset ne déclenchent pas la pénalité off-route, le temps que la voiture rejoigne la route.
  - `run_rl_training.py` : `_ROUTE_GRACE_STEPS` ajouté dans `params.json` pour traçabilité.

**Difficultés** :
- Le bug off-route sur `straight` avait été classé à tort comme "stall" dans l'analyse initiale (speed=4.1km/h ≠ stall). Corrigé après revérification des données brutes.
- `p_offroad = −0.5` n'a jamais été actif (is_on_road hardcodé True, Karim non branché) — non visible dans les métriques avant l'audit.

**Décisions** :
- **`_ROUTE_GRACE_STEPS = 20`** plutôt qu'augmenter `_OFF_ROUTE_M` : plus propre, n'affecte pas la détection pendant l'épisode, facilement ajustable. 20 steps = 1 seconde à 20 FPS — suffisant pour rejoindre la route sans ouvrir une fenêtre trop large.
- **Un seul changement à la fois** : v2 = p_stall + grace period seulement. Tous les autres hyperparamètres identiques pour isoler l'effet.
- `p_offroad` reste inactif en v2 (Karim non branché) — décision documentée dans ANALYSIS.md.

**Benchmarks** :
- `ppo_v1_100k` : 0/8 succès P1, reward −20 moy, best model 80k. Voir `runs/2026-06-12_09-20_ppo_v1_100k/ANALYSIS.md`.
- 62 tests passent après les modifications (`uv run pytest benchmarks/ai/ -q`).

**Prochaine étape** :
- Lancer `ppo_v2_300k` avec les deux fixes, analyser l'impact sur la stall phase et sur les 3 scénarios off-route.
- Commande : `uv run python3 scripts/run_rl_training.py --timesteps 300000 --tag ppo_v2 --host 100.97.91.60`

---

## 2026-06-12 (suite) — Run v2, analyse, identification speed attractor

**Avancement** :
- Run `ppo_v2_300k` (300k steps, ~2h CPU) lancée et analysée.
- **Fix stall confirmé** : à 30k, le modèle roule à 8–11 km/h (vs 0.3 km/h en v1 au même stade). p_stall=−0.20 a effectivement cassé l'attracteur immobilisme.
- **Fix grace period confirmé** : les 3 scénarios off-route dès le step 1 (straight, npc_follow, npc_crossing) fonctionnent. La distance parcourue passe de ~1m à 8–75m.
- **3/8 succès dès 30k** — première fois dans le projet qu'on passe des scénarios Phase 1.
- Reward moyen : −70 → **+17** sur 300k. 56% d'épisodes positifs (vs 22% en v1). Best model EvalCallback à 195k (+66.2, ep_len=396).
- Identification d'un **nouveau problème** : speed attractor. À partir de 120k, la vitesse monte à 30–44 km/h et le modèle crashe en 20–90 steps sur presque tout. La policy a échangé l'immobilisme contre la vitesse excessive.
- Mise en place du format d'analyse par run : `runs/EXPERIMENTS.md` (index transversal) + `ANALYSIS.md` par run.

**Difficultés** :
- La bonne policy trouvée à 195k (ep_len=396, +66.2) a été oubliée par PPO après 210k — instabilité classique avec clip_range=0.2 et lr=3e-4. La policy "survivre" est bien plus rentable (+126 vs +28 pour "fonce et crashe") mais PPO ne la maintient pas.
- Le best_model SB3 (195k, critère EvalCallback) donne 0/8 sur le benchmark — les 3 épisodes d'eval ne représentent pas bien la vraie qualité. Le checkpoint 30k (3/8) est en réalité plus fiable sur le benchmark.

**Décisions** :
- **Diagnostic speed attractor** : `p_collision = −1.0` trop faible — 50 steps à 50km/h avant crash = +28 net, ce qui est rentable. Porter à `−5.0` rendra le crash coûteux (net +24), et surtout très inférieur à survivre 300 steps (+126).
- **Ne pas toucher aux autres paramètres en v3** : stall et off-route sont réglés, on n'isole que l'effet de p_collision.
- **Enregistrement du checkpoint 30k** : à noter dans les futures analyses, la policy la plus saine est souvent dans les premiers checkpoints avant que le speed attractor s'installe.

**Benchmarks** :
- `ppo_v2_300k` : 0/8 best model, **3/8 @30k**, reward +17 moy, speed attractor ~120k. Voir `runs/2026-06-12_11-09_ppo_v2_300k/ANALYSIS.md`.

**Prochaine étape** :
- v3 : `p_collision` −1.0 → **−5.0**, 300k steps.

---

## 2026-07-01 (suite) — Branchement perception réelle (Franck + Karim)

**Avancement** :
- Remplacement complet des GT stubs (`CarlaGTDepthEstimator`, `CarlaGTLaneDetector`) par les vrais modèles.
- `rl_env.py` : espace d'observation étendu de 7 → 9 scalaires. Nouveau contrat :
  - `lane_angle_norm` (Karim / YOLOPv2) : angle de déviation de voie normalisé, remplace `center_offset` et `heading_norm` (CARLA GT supprimé).
  - `is_on_road` (Karim) : 1.0 si voie détectée, 0.0 si `NONE` — `p_offroad` (−0.5/step) actif pour la première fois.
  - `nearest_vehicle_norm` (Franck / YOLO11s + Depth Anything v2) : distance normalisée du véhicule le plus proche.
  - `has_red_light` (Franck) : booléen 0/1, feu rouge détecté dans le frame courant.
  - `speed_limit_norm` (Franck, mémorisé) : dernière limite de vitesse détectée (SPEED\_30/40/60/90), 50 km/h par défaut. Mémorisée entre frames car le panneau n'est pas toujours visible.
- Caméra RL : 200×88 → **1280×720** pour respecter la résolution d'entraînement YOLO de Franck. Karim (YOLOPv2) resize en interne, pas de contrainte.
- `src/lane_detection/` promu en package Python (`__init__.py` créé, import relatif corrigé dans `lane_perception.py`) pour permettre l'import depuis le repo root.
- `run_rl_training.py`, `run_eval.py`, `demo_mockup.py` : suppression du depth sensor CARLA, init `PerceptionPipeline` + `lane_estimate` au démarrage.
- 80 tests verts (24 tests rl_env réécrits pour la nouvelle interface, 56 autres inchangés).

**Difficultés** :
- `lane_perception.py` utilisait `from lane_geometry import lane_geometry` (import local, cassé hors du dossier `src/lane_detection/`). Résolu avec un try/except qui tente d'abord `src.lane_detection.lane_geometry` puis tombe sur l'import local — aucun script standalone de Karim cassé.
- Mode synchrone CARLA : l'env s'arrête le temps de traiter chaque tick. Avec YOLO + Depth Anything + YOLOPv2 par step, le throughput dépend directement de la vitesse GPU. Pas un problème de cohérence (le serveur attend `world.tick()`), uniquement de steps/heure.

**Décisions** :
- `nearest_walker_norm` non ajouté : les walkers ne sont pas dans les scénarios Phase 1, et le collision sensor les couvre dans tous les cas. Peut être ajouté en 10e scalaire pour Phase 2.
- `heading_norm` (dérivé du yaw CARLA GT) supprimé : `lane_angle_norm` de Karim encode la même information depuis la vision réelle, sans tricher sur la carte CARLA.
- `step()` passe `center_offset=obs[4]` (= `lane_angle_norm`) à `compute_reward` — même formule `r_center = (1 − |angle_norm|) × 0.3`, sémantique identique (0 = centré, ±1 = bord de voie).

**Prochaine étape** :
- Récupérer `best.pt` de Franck sur le serveur d'entraînement.
- Lancer run v3 : `uv run python3 scripts/run_rl_training.py --timesteps 300000 --tag ppo_v3 --host 100.97.91.60 --yolo-weights <path/best.pt>`.
- Observer si `p_offroad` et `has_red_light` génèrent un signal d'apprentissage visible dès 30k steps.

---

## 2026-07-01 — Reprise session, intégration travail équipe

**Avancement** :
- Reprise après interruption. Commits v2 (fixes stall + grace period) finalisés et pushés.
- Franck a livré son module perception complet : YOLO11s (détection objets, 11 classes) + Depth Anything v2 (profondeur monoculaire calibrée) + `PerceptionPipeline` qui fusionne les deux. Poids YOLO entraînés sur le dataset CARLA maison. Calibration depth versionnée dans `calibration.json`.
- Karim a livré son module lane detection : YOLOPv2 (segmentation zone roulable + lignes), post-traitement géométrique, sortie `estimate(rgb) → (direction, angle)`. Fonctionne sans CARLA à l'inférence.
- `feat/rl-phase1` intègre maintenant le travail des deux — les vrais modèles pourront remplacer les GT stubs dès la prochaine run.

**Difficultés** :
- Les journals de Franck (`src/perception/yolo/JOURNAL.md`, `src/perception/depth/JOURNAL.md`) et Karim (`src/perception/lanes/JOURNAL.md`) sont vides — aucune trace écrite de leur travail malgré le code livré. À signaler à l'équipe.

**Décisions** :
- **Format expérimentation acté** : `runs/EXPERIMENTS.md` + `ANALYSIS.md` par run (gitignorés). Référence pour toutes les analyses futures.
- **Prochaine run v3** : `p_collision` −1.0 → **−5.0** pour casser le speed attractor. On branche les vrais modèles de Franck et Karim dès que v3 confirme la convergence.

**Benchmarks** :
- `ppo_v1_100k` : 0/8 P1, reward −20 moy.
- `ppo_v2_300k` : 0/8 best (3/8 @30k), reward +17 moy. Stall et off-route corrigés. Speed attractor ~120k.

**Prochaine étape** :
1. Ouvrir PR `feat/rl-phase1` → `dev`.
2. Coder fix v3 : `p_collision` −1.0 → **−5.0**.
3. Lancer `ppo_v3_300k`.
- Commande : `uv run python3 scripts/run_rl_training.py --timesteps 300000 --tag ppo_v3 --host 100.97.91.60`

---

## 2026-07-02 — Revue du module lane_detection, correction du reward de centrage

**Avancement** :
- Revue complète du module `src/lane_detection/` (code de Karim) pour vérifier la cohérence avec l'intégration faite la veille.
- **Bug trouvé** : `lane_geometry.py` calcule bien deux grandeurs distinctes — `offset` (position latérale dans la voie, normalisée [-1, 1]) et `angle` (cap vers le point de fuite) — mais `lane_perception.estimate()` ne renvoyait que `(direction, angle)`, jetant l'`offset`. `rl_env.py` utilisait donc l'angle comme proxy de centrage dans le reward, alors que ce sont deux grandeurs physiques différentes (cap vs position) : une voiture peut rouler droite (angle≈0) tout en étant collée à la ligne blanche (offset≈±1), et le reward la récompensait à tort comme "bien centrée".
- Fix : `estimate()` renvoie maintenant `(direction, angle, offset)`. `rl_env.py` garde `angle` pour l'observation (`lane_angle_norm`, inchangé) mais utilise `offset` pour le terme `r_center` du reward.
- `reward_fn.py` : docstring corrigée (référençait encore l'ancien stub `CarlaGTLaneDetector`).
- 2 tests ajoutés (`test_reward_center_term_uses_lane_offset_*`) qui prouvent que le reward suit l'offset et plus l'angle. 82 tests passent.

**Difficultés** :
- La décision du 2026-07-01 ("même formule, sémantique identique") était fausse — angle et offset ne sont pas interchangeables même s'ils sont corrélés. Repéré en relisant `lane_geometry.py` ligne par ligne.

**Décisions** :
- Périmètre volontairement limité à ce bug précis. Le reste des observations faites sur `src/lane_detection/` (imports cassés dans `test/`, duplication de code entre `lane_geometry.py` et `test/yolop_lane.py`, vocabulaire français `GAUCHE/DROITE/ALIGNE`, scripts `test/main.py` et `test/debug_pipeline.py` non fonctionnels) concerne le code de Karim et lui est laissé — pas de modification de son module au-delà de l'ajout d'`offset` dans `estimate()`.

**Benchmarks** : 82 tests passent (`uv run pytest benchmarks/ -q`).

**Prochaine étape** :
- Suivre le training `ppo_v3_300k` en cours avec ce fix.
- Signaler à Karim les imports cassés dans `src/lane_detection/test/` et la duplication de logique avec `lane_geometry.py`.

---

## 2026-07-02 (suite) — Observation 9 → 10 scalaires, fix HUD/eval désynchronisé

**Avancement** :
- **Élargissement de l'observation à 10 scalaires** : ajout de `lane_offset_norm` (offset latéral réel de Karim) en plus de `lane_angle_norm` déjà présent. Raison : le reward est maintenant jugé sur `offset`, mais la policy ne le voyait jamais directement dans son observation — elle devait déduire indirectement le lien angle→reward. Avec `offset` en observation, la policy dispose à la fois du signal de position (offset) et de cap (angle), complémentaires comme un P + D en régulation.
  - Nouveau layout : `[speed, cmd_left, cmd_right, cmd_straight, lane_angle_norm, lane_offset_norm, is_on_road, nearest_vehicle_norm, has_red_light, speed_limit_norm]`.
- **Bug trouvé en cascade** : `src/ai/inference/rl_demo.py` (HUD vidéo, highlights, collecte de métriques `eval_model`) lisait encore les index de l'ancien layout à **7 scalaires** (`obs[4]`=center_offset, `obs[5]`=obstacle, `obs[6]`=heading) — jamais mis à jour depuis le passage à 9 scalaires la veille. Conséquence concrète : `benchmark.py`'s `_success_straight` vérifiait `mean(abs(center_offsets)) < 0.3` sur des valeurs qui étaient en fait `lane_angle_norm`, pas un vrai offset latéral — le critère de succès du scénario `straight` jugeait la mauvaise grandeur depuis la refonte 9-scalaires.
- Fix complet des index dans `rl_demo.py` (highlights `near_obstacle`/`lane_drift`, panneau HUD `_draw_obs_panel` enrichi avec `on_road`/`red_light`, collecte `eval_model`) + `scripts/demo_mockup.py` (`RouteFollowPolicy`) + `scripts/run_rl_training.py` (docstring, `params["obs"]`).
- 84 tests passent (2 nouveaux tests sur `lane_offset_norm`).

**Difficultés** :
- Ce bug HUD/eval ne datait pas d'hier : il existait déjà silencieusement depuis le passage 7→9 scalaires (jamais remarqué car rien ne crashait, juste des métriques et un affichage faux). Repéré en cherchant tous les usages d'`obs[N]` du repo avant de décaler les index pour le 10e scalaire.

**Décisions** :
- `_success_straight` (`benchmark.py`) n'a pas été touché : il vérifiait déjà la bonne formule (`mean(abs(center_offsets)) < 0.3`), c'est la donnée en amont (`obs[4]` au lieu d'`obs[5]`) qui était fausse. Corrigée à la source dans `rl_demo.py`.
- README `src/ai/` pas encore mis à jour (toujours à l'ancien schéma 7 scalaires + stubs GT) — à faire, mais hors du chemin critique pour lancer le prochain training.

**Benchmarks** : 84 tests passent (`uv run pytest benchmarks/ -q`).

**Prochaine étape** :
- Lancer un nouveau training propre avec le reward + l'observation corrigés (le `ppo_v3` en cours utilisait encore l'ancien reward, à ne pas garder comme référence).
- Mettre à jour `src/ai/README.md` (schéma d'observation, reward constants, perception réelle vs stubs GT).

---

## 2026-07-02 (suite) — README à jour, script de lancement guidé pour handoff Franck

**Avancement** :
- `src/ai/README.md` entièrement resynchronisé avec le code réel : diagramme perception (Franck/Karim au lieu des capteurs GT), observation 10 scalaires, `r_stall` corrigé −0.05 → −0.20 dans le doc (valeur réelle depuis la v2, jamais mise à jour), section "Contrats" en-tête corrigée (`SceneState`/`ControlOutput` ne sont utilisés nulle part dans `src/ai/` — le vrai contrat est `CarlaEnv.observation_space`/`action_space`), table de tests recomptée (28 tests sur `test_rl_env.py`, pas 20).
- **`launch_training.py`** (racine du repo) : script de preflight + lancement guidé pour le prochain training, pensé pour un handoff à distance (Franck lance en local sur son PC, résultats renvoyés après coup, pas de débogage interactif possible). Vérifie dans l'ordre : CARLA joignable (`get_server_version()`, message clair si le serveur n'est pas lancé), map active = Town10HD_Opt (sinon propose de la charger automatiquement — sinon les scénarios du benchmark pointent vers de mauvais endroits), GPU détecté (`torch.cuda.is_available()`, sinon avertit du ralentissement 10-50× et demande confirmation), poids YOLO présents sur disque. Puis menu interactif 100k/300k/500k/custom steps + tag de run, et lance `scripts/run_rl_training.py` avec les bons arguments.

**Difficultés** :
- Repéré en marge du README : les scénarios de benchmark (`src/ai/inference/benchmark.py`) ont leurs `spawn_idx`/`dest_spawn_idx` calibrés spécifiquement sur Town10HD_Opt (155 spawn points). Aucun script d'entraînement ne charge cette map automatiquement (`run_rl_training.py` utilise `client.get_world()`, la map déjà active) — un training lancé sur une autre map tournerait sans crash mais avec une évaluation finale dénuée de sens. D'où le check de map dans `launch_training.py`.

**Décisions** :
- Le script de lancement invoque `scripts/run_rl_training.py` en sous-processus (`subprocess.run`) plutôt que d'importer et ré-orchestrer sa logique — plus simple, pas de duplication, et le `try/finally` de nettoyage CARLA de `run_rl_training.py` reste intact.
- Pas de rechargement complet des modèles de perception en preflight (juste vérification de présence du fichier `best.pt`) — chargement réel déjà couvert par `run_rl_training.py` juste après ; dupliquer le chargement (YOLO + Depth Anything + YOLOPv2, ~10-30s) dans le script de check aurait été redondant.

**Benchmarks** : `uv run python3 -c "import ast; ast.parse(open('launch_training.py').read())"` (syntaxe), test manuel du chemin d'échec CARLA-injoignable (message clair confirmé). Pas de test automatisé (script interactif dépendant de CARLA/GPU réels, hors du périmètre `benchmarks/`).

**Prochaine étape** :
- Franck : `git pull` + `uv sync` + `uv run python3 launch_training.py` sur son PC, choisir 100k pour le premier run.
- Analyser sa courbe reward/eval avant de décider d'un 300k complet.

---

## 2026-07-02 (suite) — Run ppo_v3_100k analysée, conception du prochain gros passage

**Avancement** :
- Franck a lancé `ppo_v3_100k` sur son PC (5070 Ti) via `launch_training.py` — premier run avec perception réelle + reward corrigé. Résultats récupérés et analysés dans `runs/2026-07-02_15-19_ppo_v3_100k/ANALYSIS.md` (dossier renommé de `ppo_v1_100k` pour respecter la numérotation d'expérience v1→v2→v3).
- **Diagnostic v3** : reward jamais positif (−737 → −13 sur 100k, contre −70 → +17 pour v2), benchmark jamais mieux que 1/8 (contre 3/8 pour v2 à son pic). Pas une comparaison directe avec v2 : trois facteurs changent en même temps (perception bruitée au lieu de GT parfait, budget 3× plus faible, et surtout `p_offroad` actif pour la première fois — il était câblé à `is_on_road=True` en dur jusqu'à v2, donc jamais déclenché).
- **Mécanisme identifié** : avec `p_offroad` actif, rester hors-route coûte −0.5/step ; un crash ne coûte qu'une fois −1.0 puis termine l'épisode. Survivre hors-route sur un épisode complet (1000 steps, ~−500) est donc structurellement bien pire que crasher tôt (~−31 au step 60). La longueur d'épisode s'effondre progressivement (1000 → 62 steps) pendant que la vitesse grimpe (4.6 → 40.5 km/h) — cohérent avec **une policy qui apprend à "écourter vite" plutôt qu'à "bien conduire"**, plutôt qu'un vrai apprentissage de la conduite.
- Ajout d'un log de durée de training dans `run_rl_training.py` (`Training time: Xh Ym Zs` en dernier log significatif, chronométré autour de `model.learn()`).
- **Conception du prochain passage (v4)**, en session de brainstorming structurée avant tout code :
  - **Environnement** : `run_rl_training.py` ne spawnait jusqu'ici *aucun NPC* — training dans un monde vide depuis le début du projet. Ajout d'un pool fixe de véhicules + piétons en autopilot dispersés sur la map (pattern repris de `demo_mockup.py`), pour donner un vrai signal aux scalaires `nearest_vehicle_norm`/pietons qui n'avaient jusque-là presque jamais rien à détecter.
  - **Nouvelles composantes de reward** (Approche B — continu quand c'est naturel, événement ponctuel quand c'est binaire) :
    - Véhicule devant / piéton proche : pénalité continue proportionnelle à la proximité (comme `r_center`), pas un seuil dur.
    - Feu rouge / stop-yield grillé : détecté comme événement ponctuel (comme la collision), flag "déjà pénalisé pour ce feu" remis à zéro à chaque `reset()` — évite qu'un freinage propre near un feu soit puni à chaque step tant que la voiture reste proche.
    - Limite de vitesse : pénalité continue proportionnelle au dépassement, pas un seuil binaire.
  - **Observation étendue** (10 → ~13 scalaires) : `nearest_walker_norm`, `red_light_distance_norm`, `nearest_stop_yield_norm`. Décision cohérente avec la leçon du bug angle/offset du matin même : exposer à la policy la grandeur exacte sur laquelle elle est jugée, plutôt que la laisser déduire indirectement une règle depuis des signaux détournés.
  - **`p_collision` −1.0 → −5.0** : décidé depuis la v2, jamais appliqué jusqu'ici — devient encore plus pertinent avec `p_offroad` actif et de nouvelles pénalités par step (le raccourci "crasher vite" ne fait que devenir plus tentant si on ne le corrige pas).
  - **Recalibrage de `p_offroad`** (−0.5 envisagé → plus bas, ex. −0.25) : jamais réellement calibré puisqu'il n'a jamais été actif avant aujourd'hui — le laisser tel quel en empilant encore plus de pénalités par step risque d'aggraver le raccourci diagnostiqué plutôt que de le résoudre.
  - **PPO** : réseau plus grand (`net_arch=[128, 128]` au lieu du défaut `[64, 64]`) vu l'observation qui grossit et la tâche qui se complexifie ; `ent_coef` pour maintenir de l'exploration plus longtemps et éviter une convergence prématurée vers un comportement dégénéré (type speed attractor).
- Classement d'idées supplémentaires par priorité — `VecNormalize` (normalisation obs/reward, échelles très hétérogènes désormais : angles /90, distances /50, booléens 0/1), schedule de learning rate décroissant (pertinent vu l'instabilité PPO tardive observée en v2 à 210k), info de direction (gauche/droite/devant) en plus de la distance pour les dangers proches (donnée déjà disponible via les bbox de Franck, jamais exploitée), seed explicite pour la reproductibilité des expériences. Politique récurrente (LSTM), domain randomization météo et curriculum NPC progressif jugés trop de variables en plus pour ce passage — candidats pour une itération suivante.

**Difficultés** :
- Le diagnostic v2→v3 aurait été plus difficile à interpréter si on avait changé le reward, l'observation ET le budget de steps sans les isoler consciemment — noté explicitement avant de se lancer dans ce nouveau lot de changements simultanés, pour ne pas reproduire la même confusion à plus grande échelle.

**Décisions** :
- Choix délibéré de faire un "gros bond" avec plusieurs changements simultanés (environnement + reward + observation + PPO) plutôt qu'itérer un paramètre à la fois — accepté en connaissance du compromis : plus rapide en nombre d'itérations, mais un résultat décevant sera plus difficile à attribuer à une cause précise qu'un changement isolé.
- Session de conception faite via un processus de brainstorming structuré (questions one-by-one puis proposition d'approches) avant tout code, pour trancher les points ambigus (densité NPC, quelles règles de trafic inclure, observation vs reward-only) avant d'écrire quoi que ce soit.

**Benchmarks** : `ppo_v3_100k` — reward −737→−13 (jamais positif), benchmark max 1/8, 89% de crashs. Voir `runs/2026-07-02_15-19_ppo_v3_100k/ANALYSIS.md`.

**Prochaine étape** :
- Finaliser le design (spec) du passage v4, l'implémenter, lancer un nouveau training complet.

---

## 2026-07-03 — Implémentation v4 : trafic NPC, reward de sécurité, tuning PPO

**Avancement** :
- **Trafic NPC** : `scripts/run_rl_training.py` spawne maintenant **18 véhicules NPC en autopilot** (Traffic Manager CARLA, mode synchrone) et **6 piétons NPC** (`controller.ai.walker`, destination aléatoire sur la navmesh piétonne, vitesse plafonnée à 1.4 m/s). Jusqu'ici le monde d'entraînement était complètement vide depuis le tout début du projet — `nearest_vehicle_norm` et les nouveaux scalaires piéton n'avaient donc jamais rien eu de réel à détecter pendant un training. Densité configurable via `--npcs`/`--pedestrians` (défauts 18/6) ; le point de spawn de l'ego est exclu du tirage pour ne pas faire apparaître un NPC dessus.
- **Observation 10 → 12 scalaires** : ajout de `nearest_walker_norm` et `nearest_stop_yield_norm`, et remplacement de l'ancien booléen `has_red_light` par `red_light_distance_norm` (même emplacement, obs[8]). Nouveau layout : `[speed, cmd_left, cmd_right, cmd_straight, lane_angle_norm, lane_offset_norm, is_on_road, nearest_vehicle_norm, red_light_distance_norm, speed_limit_norm, nearest_walker_norm, nearest_stop_yield_norm]`. Au final 12 scalaires et non ~13 comme envisagé la veille : une fois la distance réelle au feu disponible, garder le booléen en plus n'apportait rien. Les classes `WALKER`/`STOP`/`YIELD` existaient déjà dans la taxonomie YOLO de Franck (`ObjectClass`) — rien à ajouter côté perception, juste à les consommer dans `rl_env.py`.
- **Nouveaux termes de reward** (`src/ai/rewards/reward_fn.py`) :
  - `r_following` — pénalité continue fondée sur la règle des 2 secondes (temps-avant-impact = distance / vitesse), pas un seuil de distance fixe : 15m à 10 km/h et 15m à 80 km/h ne représentent pas le même danger.
  - `r_walker` — pénalité continue sur la distance brute au piéton le plus proche ; pas de time-headway ici, un piéton peut changer de direction à tout moment indépendamment de la vitesse du véhicule.
  - `r_speeding` — pénalité continue au-delà de la limite détectée, avec 5 km/h de tolérance avant de déclencher (évite de punir un léger dépassement / bruit de mesure).
  - `r_red_light` / `r_stop_yield` — pénalités ponctuelles (−2.0 / −1.0), une seule fois par infraction : un flag retombe dès que la distance repasse au-dessus du seuil (5m), pas seulement au `reset()` comme envisagé la veille — un deuxième feu plus loin dans le même épisode peut donc être sanctionné indépendamment du premier. Une infraction exige aussi que la vitesse dépasse 5 km/h : rester à l'arrêt collé au feu n'est jamais pénalisé, seul le franchir en roulant l'est.
- **`p_offroad`** : −0.5 → **−0.25**. Il n'a été actif pour la première fois que sur `ppo_v3_100k` et n'avait donc jamais été réellement calibré pour ce régime — le laisser à −0.5 en empilant en plus les nouvelles pénalités par step aurait aggravé le raccourci diagnostiqué la veille plutôt que de le corriger.
- **`p_collision`** : flat −1.0 → base **−5.0** scalée par la vitesse d'impact (**−0.05/km-h**, capturée dans `_on_collision` via `_speed_kmh()` au moment exact du contact, pas relue plus tard au moment du calcul du reward). Ex. : choc à 72 km/h → −5.0 − 3.6 = −8.6. Fix en attente depuis l'analyse v2/v3, appliqué maintenant — d'autant plus nécessaire que les nouvelles pénalités par step rendent le raccourci « crasher vite » encore plus tentant si le collision reward restait plat.
- **PPO** (`src/ai/training/rl_train.py`) : réseau `net_arch=[128, 128]` (au lieu du défaut SB3 `[64, 64]`), `ent_coef=0.01` pour maintenir l'exploration plus longtemps, `seed=42` pour la reproductibilité des expériences, et `learning_rate` en décroissance linéaire (3e-4 → 0) au lieu d'une valeur fixe — vise directement l'instabilité de fin de training observée sur v2 autour de 210k steps.
- `src/ai/README.md` resynchronisé : observation 12 scalaires, formules de reward complètes (les 5 nouveaux termes), nouvelle section « Trafic NPC », table de tests recomptée.

**Difficultés** :
- Spawn des piétons CARLA : le `controller.ai.walker` doit être attaché après que l'acteur piéton a été effectivement enregistré dans le monde — un `world.tick()` est nécessaire entre le `try_spawn_actor` du piéton et le `spawn_actor` de son controller, sinon ce dernier peut s'attacher à un acteur pas encore prêt côté serveur.
- En remettant à plat tous les tests de collision pour la nouvelle formule base + vitesse, l'un d'eux est passé entre les mailles au premier passage : `test_collision_terminates_with_penalty` positionne `_collision_flag` directement sans passer par `_on_collision` (donc vitesse d'impact à 0) et vérifiait encore l'ancien −1.0 flat. Repéré en relisant la liste complète des tests de collision avant de considérer le fichier fini ; corrigé à −5.0.

**Décisions** :
- Le diagramme ASCII du README (section « PPO Policy ») mentionnait encore « 2 couches Dense 64 » — repéré en relisant l'ensemble juste avant de clore ce lot, corrigé à 128.
- Pas de `VecNormalize` ni de politique récurrente sur ce passage, comme tranché la veille — on s'en tient strictement aux points actés en conception plutôt que d'ajouter des idées en cours de route.

**Benchmarks** : 94 tests sur `benchmarks/ai/` (`uv run pytest benchmarks/ai/ -q`), contre 70 avant ce lot de changements (+10 dans `smoke.py`, +9 dans `test_rl_env.py`, +5 dans `test_rl_train.py`). Tout `benchmarks/` confondu (modules Franck/Karim compris) : 108 tests, contre 84 la dernière fois. Tout vert.

**Prochaine étape** :
- Lancer un training complet avec ce lot de changements (tag `ppo_v4`, 300k comme pour v2) et comparer à `ppo_v3_100k` : reward moyen, score benchmark, taux de crash, temps passé hors-route — et surtout si le raccourci « écourter vite » identifié en v3 a disparu maintenant que `p_collision` est proportionnel à la vitesse d'impact.
- Vérifier si le signal des nouveaux scalaires piéton / feu / stop-yield est déjà visible dans les métriques dès les 30-50k premiers steps, comme `is_on_road` avait commencé à l'être en v3.

---

## 2026-07-04 — Analyse ppo_v4.2_100k : "ne freine jamais" et diagnostic du virage systématique

**Avancement** :
- Franck a lancé `ppo_v4.2_100k` (trafic NPC + reward sécurité + tuning PPO du lot v4). Analyse complète dans `runs/2026-07-04_11-27_ppo_v4.2_100k/ANALYSIS.md`.
- `params.json` confirme que tout le lot v4 est bien actif de bout en bout (12 scalaires, trafic 18 véhicules/6 piétons, tous les nouveaux termes de reward, `net_arch=[128,128]`) — le fix `save_params` (learning rate schedule sérialisé en `str`) tient depuis le premier vrai run qui l'utilise.
- Point de départ moins négatif que v3 (−458 vs −737, cohérent avec `p_offroad` −0.5→−0.25) mais **1275 épisodes en 100k steps contre 650 en v3** et **97% de crashs courts** : les épisodes s'effondrent en durée bien plus qu'en v3, malgré le fix `p_collision`.
- Benchmark 13 scénarios : jamais mieux que 2/8 (à 10k, quand la vitesse est quasi nulle), 0/8 partout ensuite. Le `best_model` a le meilleur taux hors-route de tout le projet (6%) mais **termine ses 13 scénarios par une collision**, entre 12 et 45 km/h.
- **Diagnostic principal, trouvé en creusant les séries brutes (`brake_series`) plutôt que les seuls agrégats** : le frein n'est utilisé nulle part, sur aucun scénario, à aucun checkpoint. `throttle_mean` monte progressivement de 0.099 à 1.000 sur les 100k steps pendant que `brake_mean` reste à 0.000 partout — seul l'axe accélérateur/frein s'est effondré, le steering reste actif et modulé.
- Second symptôme observé sur la vidéo de démo : la voiture tourne systématiquement à gauche dès le début de chaque test, quelle que soit la commande de navigation réelle affichée à l'écran.

**Difficultés** :
- Calcul de pourquoi le fix `p_collision` (base −5.0, −0.05/km-h) ne suffit pas : à 40 km/h, `r_speed` accumulé sur une survie moyenne de 35-50 steps avant crash cumule à +7.7/+11, contre un coût de collision de −7.0 au même impact. Les deux s'annulent quasiment — le fix rend le crash plus cher qu'avant (bonne direction) mais pas assez pour dominer le gain de vitesse accumulé sur la durée de vie d'un épisode entier. Autre facteur probable : toutes les nouvelles pénalités (`r_following`, `r_walker`, `r_speeding`, `r_red_light`, `r_stop_yield`) sont punitives, aucune ne récompense positivement un évitement réussi — freiner n'apporte jamais de bonus direct, seulement l'évitement d'une pénalité future incertaine, signal bien plus faible à apprendre par descente de gradient qu'un `r_speed` immédiat et garanti à chaque step.
- Le symptôme "tourne à gauche" ne s'expliquait par aucun terme de reward — a nécessité d'aller lire le code de navigation plutôt que la fonction de reward (voir entrée suivante).

**Décisions** :
- Avant de retoucher le reward (`_W_SPEED` vs `_P_COLLISION_SPEED_SCALE`, récompense positive de maintien de distance de sécurité), identification de **4 sous-projets indépendants** à traiter en amont, chacun avec son propre design avant code :
  1. **Outillage de diagnostic** : automatiser ce qui vient d'être fait à la main (courbe binée, tableau benchmark, décomposition du reward par composante) pour ne plus dépendre d'un one-liner Python ad-hoc à chaque analyse.
  2. **Sécurité au spawn** : vérifier que l'ego ne spawn pas dans une situation dangereuse (NPC adjacent) et creuser le "tourne à gauche".
  3. **CARLA en parallèle** : évaluer si `SubprocVecEnv` (plusieurs instances CARLA en parallèle) vaut le coup pour réduire le temps d'entraînement (~5-6h/100k steps actuellement).
  4. **Overlay carte + trajet dans la vidéo de démo** : afficher le trajet A→B prévu en intro de vidéo pour vérifier visuellement que la voiture le suit.
- Le reward design proprement dit (rééquilibrage `r_speed`/collision, récompense positive de sécurité) est **volontairement mis de côté** tant que ces 4 sous-projets n'ont pas amélioré la précision du diagnostic et corrigé les bugs indépendants du reward.

**Benchmarks** : `ppo_v4.2_100k` — jamais mieux que 2/8 P1, 97% crashs courts, frein jamais utilisé (0.000 à tous les checkpoints). Voir `runs/2026-07-04_11-27_ppo_v4.2_100k/ANALYSIS.md`.

**Prochaine étape** :
- Sous-projet 1 (outillage diagnostic) en premier — condition pour analyser rapidement et précisément le prochain training.

---

## 2026-07-04 (suite) — Outillage diagnostic : reward par composante, log persistant, `analyze_run.py`

**Avancement** :
- **`compute_reward()` retourne un triplet** `(reward, terminated, components: dict[str, float])` au lieu de `(reward, terminated)`. `components` contient toujours les 12 clés (`r_speed`, `r_center`, `r_alive`, `r_offroad`, `r_stall`, `r_off_route`, `r_following`, `r_walker`, `r_speeding`, `r_red_light`, `r_stop_yield`, `r_collision`), même à zéro — sur une collision, toutes les clés sont à 0.0 sauf `r_collision`. Invariant conservé par construction : `reward = sum(components.values())`, jamais recalculé séparément.
- **`CarlaEnv`** accumule ces composantes sur toute la durée d'un épisode (`self._episode_reward_components`) et les expose dans le dict `info` retourné par `step()`, uniquement à la fin de l'épisode (`terminated` ou `truncated`) — vide sur les steps intermédiaires.
- **`Monitor(env, info_keywords=REWARD_COMPONENT_KEYS)`** dans `run_rl_training.py` : Stable-Baselines3 extrait automatiquement ces 12 clés du dict `info` à chaque fin d'épisode et les ajoute en colonnes du CSV existant (`training_log.monitor.csv`) — aucun nouveau fichier, le format existant grossit de 12 colonnes.
- **Log persistant `run.log`** : la classe `_StdoutFilter` (qui ne filtrait que stdout, seulement autour de `model.learn()`) est remplacée par `_Tee`, qui duplique chaque ligne vers le flux réel ET un fichier, avec flush immédiat après chaque ligne. Appliquée sur `sys.stdout` ET `sys.stderr`, sur l'intégralité de `main()` (connexion CARLA, training, eval, démo) — en cas de crash, `run.log` contient tout jusqu'à la dernière ligne, pas seulement ce qui s'est passé pendant l'entraînement. Le filtrage des lignes de debug de Victor (`_BLOCKED`) reste actif sur stdout uniquement — jamais sur stderr, pour ne jamais perdre une trace de crash.
- **Nouveau script `scripts/analyze_run.py`** : lit `training_log.monitor.csv` et `evals/results.json` d'un dossier de run, calcule une courbe d'apprentissage binée par tranches de 10k steps (reward moyen, % d'épisodes positifs, longueur moyenne, % de crashs courts, et la moyenne de chaque composante de reward présente), un résumé benchmark par checkpoint (succès Phase 1, vitesse/hors-route/accélérateur/frein moyens), et des totaux (épisodes, steps, taux de crash). Écrit `analysis_data.json` dans le dossier de run et affiche un résumé lisible en tables, directement copiable dans une future `ANALYSIS.md`. Volontairement **données brutes uniquement** : pas de détection automatique d'anomalie, pas de comparaison inter-run, pas d'interprétation — l'humain garde la main sur le diagnostic.
- Testé manuellement sur `runs/2026-07-04_11-27_ppo_v4.2_100k` (run analysée à la main la veille) : les chiffres produits correspondent à ceux déjà écrits dans `ANALYSIS.md`.
- 129 tests sur `benchmarks/ai/` (+21 vs les 108 précédents : +11 dans `smoke.py`/`test_rl_env.py` pour le triplet de reward et l'accumulateur, +10 dans le nouveau `test_analyze_run.py`).

**Difficultés** :
- `analyze_run.py` : agréger une ligne de `DataFrame.groupby(...).agg(...)` mélangeant colonnes de comptage (`n_episodes`, `n_positive`) et colonnes de moyenne fait remonter les comptages en `float64` au lieu d'`int64` (upcast pandas sur ligne mixte) — invisible dans les tests unitaires (qui ne comparent que la valeur, `2 == 2.0`) mais fait planter l'affichage du résumé (`f"{n:4d}"` sur un float lève `ValueError`). Repéré uniquement en testant sur la vraie run, pas dans les tests synthétiques — cast explicite en `int()` ajouté avant l'affichage.
- Décision de granularité prise en amont : reward par **somme sur l'épisode complet**, pas par step. Suffisant pour la courbe binée visée et beaucoup plus simple à brancher sur `Monitor.info_keywords` (qui ne lit `info` qu'à la fin d'un épisode de toute façon).

**Décisions** :
- Le triplet `(reward, terminated, components)` construit `reward` en sommant le dict plutôt que par un calcul séparé — élimine structurellement tout risque de dérive entre les deux valeurs, plutôt que de compter sur une convention à respecter à la main dans chaque nouveau terme de reward futur.
- Pas de détection automatique d'anomalie dans `analyze_run.py` (ex. "alerte si brake_mean < 0.01") — décision explicite : les seuils d'alerte pertinents changent avec chaque itération du reward, un système à seuils fixes serait vite obsolète ou trompeur. Le script calcule, le diagnostic reste manuel.

**Benchmarks** : 129 tests sur `benchmarks/ai/`, 143 tous modules confondus (`uv run pytest benchmarks/ -q`).

**Prochaine étape** :
- Sous-projet 2 : sécurité au spawn + investigation du "tourne à gauche systématique".

---

## 2026-07-04 (suite) — Bug critique de replanification de route, spawn sûr face aux NPC

**Avancement** :
- **Bug trouvé en creusant le "tourne à gauche systématique"** : `CarlaEnv.reset()` ne replanifiait la route (`self.route`) que si un `spawn_idx` explicite était fourni (chemin eval/demo). Or pendant l'entraînement normal, SB3 appelle `reset()` sans aucune option après chaque épisode — l'ego est téléporté à un point de spawn **aléatoire**, mais la route reste celle calculée une seule fois à la création de l'environnement, depuis le tout premier spawn. Résultat : à partir du 2e épisode de chaque run, `Navigation.next_command()` calcule la commande `LEFT/RIGHT/STRAIGHT` en pointant vers un waypoint d'une route sans aucun rapport avec la position réelle du véhicule — un signal de navigation qui n'était que du bruit pendant la quasi-totalité de chaque entraînement depuis l'introduction du GPS de Victor. Fix : le bloc de replanification dans `reset()` tourne maintenant systématiquement, plus seulement quand `spawn_idx` est explicite.
- **Cache du graphe réseau routier dans `Navigation`** : nécessaire pour que le fix ci-dessus ne ralentisse pas l'entraînement. `extract_road_network()` régénère tous les waypoints de la ville à 2m de résolution et trace un plot matplotlib à chaque appel — en faire un par épisode (toutes les ~50-90 steps) aurait sérieusement dégradé le débit. Le graphe est maintenant construit une seule fois par instance `Navigation` (statique pour une map donnée) et réutilisé pour tous les `plan()` suivants.
- **Spawn sûr par re-tirage** : `CarlaEnv._teleport_to_spawn()` interroge maintenant `world.get_actors()` (véhicules + piétons, hors ego) au moment du reset et tire jusqu'à 10 points de spawn aléatoires, acceptant le premier à au moins 10m de tout acteur dynamique. Si les 10 tentatives échouent, conserve celui qui avait la plus grande distance observée (jamais pire que le tirage aléatoire d'avant ce fix). S'applique uniquement au chemin d'entraînement (spawn aléatoire) — les scénarios d'eval/démo avec `spawn_idx` explicite ne sont pas concernés.
- **Fix de reproductibilité de la démo libre découvert en réintégrant les deux fixes ensemble** : la vidéo `demo.mp4` (mode libre, seed fixe `_DEMO_RESET_SEED=42`) perdait sa reproductibilité — le nouveau re-tirage de spawn consomme un nombre variable de tirages aléatoires selon la position des NPC au moment du reset, donc un spawn différent d'un run à l'autre malgré le seed fixe. Fix : `record_episode()` accepte maintenant un paramètre `spawn_idx` explicite (comme les scénarios de benchmark), passé à `env.reset(options={"spawn_idx": ...})` — `run_rl_training.py` fixe `spawn_idx=0` pour la démo, ce qui court-circuite entièrement le re-tirage aléatoire et retrouve un spawn garanti identique à chaque run.
- Idée de démarrage en pilote automatique avant de rendre la main à la policy (évoquée initialement pour éviter un mauvais spawn) abandonnée : le vrai risque identifié était la proximité NPC, déjà réglé par le re-tirage ; les points de spawn CARLA sont déjà bien orientés sur la route, et un warmup de ticks physiques existe déjà dans `reset()`.
- 136 tests sur `benchmarks/ai/` avant l'ajout de l'overlay démo (+7 vs 129 : 2 tests de non-régression sur la replanification de route, 5 sur le spawn sûr), 138 après le fix de reproductibilité démo (+2).

**Difficultés** :
- Aucune régression détectée, mais fix en deux temps nécessaire : corriger le spawn sûr seul aurait laissé la démo libre non-déterministe sans que rien ne le signale (aucun test existant ne couvrait ce chemin) — repéré uniquement en relisant tous les appels à `record_episode()` du repo une fois le spawn sûr en place.

**Décisions** :
- Cache du graphe routier scopé à l'instance `Navigation`, pas de cache statique/global partagé entre instances — une seule instance vit par run d'entraînement de toute façon, donc pas de bénéfice à un cache plus large, et ça évite tout risque de fuite d'état entre deux runs qui chargeraient des maps différentes dans le même process.
- Seuil de sécurité 10m / 10 tentatives max choisis empiriquement : suffisant pour laisser une vraie marge de réaction sans exclure trop de points de spawn avec seulement 18 véhicules + 6 piétons dispersés sur toute la map.

**Benchmarks** : 138 tests sur `benchmarks/ai/` après ce lot, tous verts.

**Prochaine étape** :
- Sous-projet 3 (CARLA parallèle) et sous-projet 4 (overlay démo).

---

## 2026-07-04 (suite) — Abandon du CARLA parallèle, overlay carte + trajet en démo

**Avancement** :
- **CARLA en parallèle (`SubprocVecEnv`) étudié puis abandonné.** PPO supporte nativement plusieurs environnements en parallèle (SB3 collecte `n_steps` par environnement, cumule dans un seul buffer, fait une mise à jour, relance la collecte) — la question n'était donc pas de faisabilité algorithmique mais d'infrastructure : seule la machine avec CARLA (celle de Franck) peut faire tourner le simulateur, donc "plusieurs CARLA" signifierait plusieurs instances sur le **même** poste, en concurrence sur le même GPU et son unique pipeline de perception (YOLO + Depth Anything + YOLOPv2, répliqué autant de fois que d'instances). Analyse coût/bénéfice : ça n'aurait accéléré que le temps mur par run, pas corrigé le vrai problème de fond (déséquilibre du reward, cf. entrée du 2026-07-04 matin) — seulement permis d'itérer plus vite une fois le reward design fait. Jugé pas rentable au vu de la complexité (gestion de N process CARLA, VRAM partagée) pour ce projet. Contrainte matérielle actée comme telle pour le rendu.
- **Overlay carte + trajet dans la démo libre** : nouvelle fonction `_draw_route_map_card()` dans `rl_demo.py`, dans le même style visuel que les cartons titre/résultat déjà en place (fond sombre, dessin cv2, pas de dépendance matplotlib supplémentaire). Calcule la boîte englobante des waypoints de la route planifiée, trace le trajet à l'échelle avec marge (proportions préservées), marque le départ "A" et l'arrivée "B". `record_episode()` gagne un paramètre `route_map_seconds` (défaut 3.0), affiché une seule fois en tout début de la vidéo de démo libre — pas répété entre les épisodes si `n_episodes > 1`. Ne concerne que la démo libre : les scénarios de benchmark ont déjà leurs propres cartons et n'ont pas de vrai trajet point A→B à vérifier.
- 143 tests sur `benchmarks/ai/` au global du projet (`uv run pytest benchmarks/ -q`), tout vert.

**Difficultés** :
- Aucune notable — le pattern de cartons cv2 réutilisables (déjà en place pour les scénarios de benchmark) a rendu l'ajout direct.

**Décisions** :
- Overlay en cv2 pur plutôt qu'en réutilisant `MatplotVisualizer.plot_plan()` (déjà utilisé ailleurs pour tracer la route) : cohérence visuelle avec les cartons existants, et `plot_plan()` ne nettoie jamais sa figure matplotlib globale entre deux appels (accumulation silencieuse d'un appel à l'autre) — un problème préexistant, pas dans le périmètre de ce lot, qu'il valait mieux ne pas hériter dans le nouveau code.
- Pas de fond réseau routier complet derrière le trajet (juste la ligne + les deux points A/B) : suffisant pour vérifier visuellement que la voiture suit le bon chemin, et évite de coupler `rl_demo.py` au graphe interne de `Navigation`.

**Benchmarks** : 143 tests, tout vert.

**Prochaine étape** :
- Les 4 sous-projets de ce lot sont clos (diagnostic outillé, bug de route + spawn sûr corrigés, CARLA parallèle tranché, overlay démo ajouté). Reste le reward design proprement dit (rééquilibrage vitesse/collision, récompense positive de sécurité) — à traiter ensuite, avec l'outillage de diagnostic maintenant disponible pour juger rapidement de son effet sur le prochain training.

---

## 2026-07-05 — Rééquilibrage vitesse/collision (v5)

**Avancement** :
- **`_W_SPEED` : 0.5 → 0.3** et **`_P_COLLISION_SPEED_SCALE` : −0.05 → −0.20/km-h** dans `reward_fn.py`. Objectif : casser l'incitatif "foncer sans jamais freiner" diagnostiqué sur `ppo_v4.2_100k`.
- Chiffrage complet (pas seulement `r_speed` vs `r_collision` comme dans l'analyse initiale, mais en réintégrant aussi `r_center` et `r_alive` qui continuent de s'accumuler pendant une conduite dangereuse) sur la trajectoire type observée (40 km/h, survie ~40 steps) :
  - Avant : `r_speed` +8.9, `r_center` +8.0, `r_alive` +0.4, `r_collision` −7.0 → **net +10.3**. Le déséquilibre réel était donc plus fort que ce que l'analyse `r_speed`-vs-`r_collision` seule laissait penser.
  - Après : `r_speed` +5.3 (poids réduit), `r_center` +8.0 (inchangé), `r_alive` +0.4 (inchangé), `r_collision` −13.0 (pénalité ×4 par km/h d'impact) → **net +0.7**, quasi neutre.
- Tests mis à jour en conséquence : `test_reward_components_sum_at_max` (0.5+0.3+0.01 → 0.3+0.3+0.01), `test_collision_scales_with_impact_speed` et `test_reward_components_only_collision_nonzero_on_collision` (formule −0.05 → −0.20), `test_episode_reward_components_reported_on_truncation` dans `test_rl_env.py` (r_speed attendu à 36 km/h : 0.2/step → 0.12/step). `src/ai/README.md` resynchronisé sur les deux nouvelles valeurs.
- 143 tests, tout vert.

**Difficultés** :
- Aucune sur le code — mais reconnu explicitement que ce chiffrage n'est qu'une hypothèse de travail, pas une solution fermée : le calcul à la main sur UNE trajectoire type ne capture pas toute la dynamique d'exploration de PPO. Viser un net strictement négatif aurait été possible en poussant `_P_COLLISION_SPEED_SCALE` plus loin, mais risque de recréer une paralysie ("peur de rouler"), le même type de régression déjà rencontré et corrigé via `r_stall` en v2 — préféré un rapprochement vers la neutralité, à valider empiriquement sur le prochain training plutôt qu'une correction agressive non testée.

**Décisions** :
- Changement volontairement limité à ces deux constantes — aucun des 5 termes de sécurité (`r_following`/`r_walker`/`r_speeding`/`r_red_light`/`r_stop_yield`) n'a été touché : ils n'ont jamais eu l'occasion de vraiment s'exprimer tant que foncer restait rentable, donc pas encore de données pour juger s'ils sont eux-mêmes mal calibrés. Le candidat "récompense positive de maintien de distance de sécurité" (plutôt que seulement punitive) évoqué dans l'analyse `ppo_v4.2_100k` est explicitement repoussé à une itération suivante si `ppo_v5` montre encore un déficit de freinage après ce rééquilibrage.

**Benchmarks** : 143 tests (`uv run pytest benchmarks/ -q`).

**Prochaine étape** :
- Envoyer ce lot à Franck pour un training `ppo_v5` (avec en plus le fix de replanification de route, le spawn sûr et l'outillage de diagnostic du lot précédent).
- Analyser via `scripts/analyze_run.py` : vérifier en particulier `brake_mean` par checkpoint (jamais utilisé sur v4.2) et si le "tourne à gauche systématique" a disparu.
