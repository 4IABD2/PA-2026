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
