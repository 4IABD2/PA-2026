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
- Commande : `uv run python3 scripts/run_rl_training.py --timesteps 300000 --tag ppo_v3 --host 100.97.91.60`
