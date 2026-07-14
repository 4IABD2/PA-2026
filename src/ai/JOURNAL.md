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

---

## 2026-07-05 — Revue approfondie du reward, lot v6 : progression, destination, sécurité positive, fluidité

**Avancement** :
- Pendant que `ppo_v5` tournait chez Franck, relecture complète de `reward_fn.py`/`rl_env.py` terme par terme pour juger si le rééquilibrage v5 suffisait et identifier d'autres angles morts. Deux problèmes structurels trouvés au-delà du déséquilibre déjà connu :
  1. **`r_speed` récompensait la vitesse brute (norme du vecteur vitesse), pas la progression** — une voiture qui roule vite en marche arrière, de travers ou en boucle recevait le même `r_speed` qu'une voiture qui avance correctement.
  2. **Aucun signal d'entraînement ne récompensait l'arrivée effective à destination** — `reached_dest` existe dans le code d'évaluation (`benchmark.py`) mais n'était jamais calculé ni renvoyé pendant l'entraînement ; le critère de succès mesuré en éval n'était donc jamais ce que la policy apprenait à optimiser.
  3. Confirmation que tous les termes de sécurité (`r_following`, `r_walker`, `r_speeding`, `r_red_light`, `r_stop_yield`) sont purement punitifs, sans aucun signal positif pour une conduite prudente réussie.
- **`r_speed` devient une vitesse orientée-route** : nouvelle méthode `CarlaEnv._progress_speed_kmh()` — projette le vecteur vitesse sur le cap du waypoint de route le plus proche (`_route_idx`), plancher à 0 (rouler à contresens de la route ne punit pas en plus, les pénalités hors-route/hors-voie couvrent déjà ce cas), retombe sur la vitesse brute si la route n'a pas de waypoints.
- **Bonus d'arrivée à destination** : nouvelle méthode `CarlaEnv._reached_destination()` — déclenche seulement si la voiture est à moins de 15m de `route.destination` **et** a parcouru au moins 25m depuis le début de l'épisode (garde-fou contre un spawn qui atterrirait par chance près de la destination fixe d'entraînement sans avoir vraiment roulé). Nouveau terme `r_destination = +10.0`, termine l'épisode comme une collision mais en positif. Limite assumée : la destination d'entraînement reste un point fixe unique (`spawn_pts[-1]`) pour toute la durée d'un run, donc ce bonus restera rare tant que la survie/navigation ne s'améliore pas — corriger ça (destination plus proche/mobile) est un chantier séparé, pas fait ici.
- **Premier signal positif de sécurité** : nouveau terme `r_safe = +0.05`, actif quand aucun danger n'est en cours (`_following_penalty`/`_walker_penalty`/`_speeding_penalty` tous à zéro — réutilise directement ces fonctions, pas de seuils dupliqués).
- **Pénalité de fluidité de pilotage** : nouveau terme `r_jerk = -|steer_t - steer_t-1| × 0.1`, nouvel état `CarlaEnv._prev_steer` réinitialisé à chaque `reset()`.
- **Commentaire corrigé sans changer de valeur** : `_P_OFF_ROUTE` affirmait être "aussi sévère que `_P_OFFROAD`", ce qui n'est plus vrai depuis que `_P_OFFROAD` a été volontairement abaissé en v4 (−0.5→−0.25) pour ne pas aggraver le raccourci "crasher vite" diagnostiqué à l'époque — décision qui reste valable, donc pas de ré-équilibrage, juste la documentation corrigée.
- `REWARD_COMPONENT_KEYS` passe de 12 à 15 clés. Grâce à la conception générique de l'outillage de diagnostic (accumulateur, `Monitor.info_keywords`, détection de colonnes `r_*` dans `analyze_run.py`), aucun de ces trois consommateurs n'a eu besoin d'être modifié — seuls les tests qui énuméraient les clés en dur ont dû être mis à jour.
- 163 tests sur `benchmarks/` (145 en Phase 1), tout vert.

**Difficultés** :
- **Bug de conception trouvé pendant la relecture du plan, avant tout code** : le nouveau `r_safe` n'est pas protégé par un paramètre optionnel (contrairement aux trois autres nouvelles entrées) — il se calcule à partir des arguments existants (`nearest_vehicle_m`/`nearest_walker_m`/`speed_limit_kmh`), dont les valeurs par défaut ("rien à proximité") le font se déclencher automatiquement dans la quasi-totalité des tests existants qui ne configurent pas explicitement un danger. Repéré à la relecture du plan avant dispatch, mais sous-estimé : 5 tests corrigés dans le plan initial, 2 de plus découverts seulement en lançant la suite complète après la première implémentation (un dans `test_rl_env.py` sur le terme de centrage). Sept tests au total ont dû voir leur valeur exacte attendue augmentée de `+0.05`.
- **Vrai bug de production trouvé en implémentant `_reached_destination()`** : le garde-fou initial (`if not (self.route and self.route.destination)`) s'appuyait sur la "vérité" (truthiness) de `Route`, mais `Route.__len__` délègue à `len(waypoints)` — une route avec zéro waypoint mais une destination valide est donc jugée fausse, peu importe la destination. Ce n'est pas qu'un artefact de test : `Navigation.a_star` renvoie une liste vide quand la recherche de chemin échoue, et `plan()` renvoie quand même une `Route` avec la vraie destination — dans ce cas précis, le bonus de destination aurait été silencieusement et définitivement désactivé pour tout l'épisode. Corrigé avec des vérifications explicites `is None` plutôt que la troncature.
- Un des deux agents d'implémentation a été interrompu en cours de tâche par une erreur de connexion transitoire ; repris via reprise de session, terminé proprement sans perte de travail.

**Décisions** :
- `r_safe`, volontairement non filtré par un nouveau paramètre — le déclenchement "par défaut" est le comportement voulu (le cas commun "rien de détecté" doit compter comme sûr), pas un bug à corriger en amont ; la correction porte sur les tests, pas sur la conception.
- Détection de franchissement de feu rouge/stop contournable (ralentir sous le seuil de vitesse pour passer sans pénalité), repérée à la relecture mais **volontairement exclue** de ce lot — corriger ça proprement demande un vrai signal de franchissement de ligne d'arrêt côté perception, qui n'existe pas encore (domaine de Franck, pas un changement de `reward_fn.py`/`rl_env.py`).
- `params.json` (`run_rl_training.py`) mis à jour pour inclure les 5 nouvelles constantes — trouvé manquant à la revue finale de branche, corrigé avant de considérer le lot terminé (sinon la traçabilité des futures runs aurait été incomplète).

**Benchmarks** : 163 tests (`uv run pytest benchmarks/ -q`), 145 en Phase 1 seule.

**Prochaine étape** :
- Lancer `ppo_v6` avec ce lot (branche `feat/reward-v6`) et analyser via `scripts/analyze_run.py` : le frein s'active-t-il enfin ? `r_destination` se déclenche-t-il ne serait-ce qu'une fois ? `r_safe` a-t-il un effet visible sur la fréquence des `r_following`/`r_walker`/`r_speeding` ?
- Si le déficit de freinage persiste malgré tout : reste la piste du contournement feu rouge/stop (nécessite Franck) et un éventuel rééquilibrage supplémentaire de `_W_SPEED`/`_P_COLLISION_SPEED_SCALE`.

---

## 2026-07-06 — Analyse `ppo_v5_300k`, bug critique du log persistant trouvé et corrigé

**Avancement** :
- Récupéré et analysé `ppo_v5_300k` (300k steps, lancé par Franck avec le rééquilibrage v5). Analyse complète dans `runs/2026-07-05_17-29_ppo_v5_300k/.../ANALYSIS.md`.
- **Bug critique trouvé sur le log persistant** : `run.log` s'arrêtait net juste après `Training done.`/`Cleanup done.`, sans trace d'éval ni de démo, sans traceback. Diagnostiqué en inspectant les octets bruts du fichier : `open(run_dir / "run.log", "w")` (dans `run_rl_training.py`) n'imposait pas d'encodage — sur Windows ça retombe sur cp1252, qui ne sait pas écrire le caractère `→` utilisé dans plusieurs `print()` (`"Reward curve → ..."` et suivants). Le tout premier print concerné plante le script (`UnicodeEncodeError`) juste après l'entraînement ; le `finally` de nettoyage s'exécute encore (d'où `Cleanup done.` visible), mais le traceback part sur un flux déjà restauré au moment où l'exception remonte jusqu'au niveau racine — invisible dans le fichier. Fix : `encoding="utf-8"` ajouté à l'ouverture du fichier. Ce bug aurait touché tout run Windows sans exception depuis l'introduction du log persistant, `ppo_v6` y compris si non corrigé avant.
- **Conséquence directe pour l'analyse** : aucune donnée d'éval (13 scénarios, vitesse/frein/hors-route par checkpoint) disponible pour v5 — seule la courbe d'entraînement a pu être exploitée.
- **Diagnostic principal, en creusant les composantes de reward par step plutôt que les seuls agrégats** : `r_speed` reste bas et stable tout du long (0.001 → 0.07/step, jamais plus, ~15-22 km/h de moyenne) — contrairement à `ppo_v4.2` où le throttle grimpait progressivement jusqu'à 1.000. Le rééquilibrage v5 a bien cassé le raccourci "foncer à fond sans jamais freiner" qu'il ciblait.
- **Mais un problème plus large domine, resté invisible avant cet outillage** : `r_off_route` (jusqu'à −0.44/step sur un max de −0.5) et `r_offroad` (−0.19 à −0.23/step sur un max de −0.25) cumulent jusqu'à −0.6 à −0.8 par step — très largement supérieur à toutes les autres composantes. La voiture est hors-route et/ou hors-voie la majorité du temps, sans amélioration nette sur les 300k steps (`r_offroad` en particulier reste quasi identique −0.19/−0.22 sur toute la seconde moitié du run). 3704 épisodes, taux de crash global 97%, pas de tendance d'amélioration claire dans la seconde moitié.
- **`r_walker`, `r_red_light`, `r_stop_yield` à zéro exact sur la totalité des 3704 épisodes** ; `r_speeding` quasi jamais actif. Avec des épisodes qui durent 50-100 steps en moyenne, la voiture ne va jamais assez loin pour croiser un piéton, un feu rouge ou un stop — ces mécanismes (et le nouveau `r_safe` de v6, qui dépend des mêmes conditions) n'auront probablement aucune chance de s'exprimer tant que le problème plus basique de maintien sur la route n'est pas réglé.

**Difficultés** :
- Diagnostic du bug d'encodage non trivial : aucune erreur explicite dans le log (par construction, puisque c'est justement l'écriture du log qui plante). Repéré en inspectant les octets bruts du fichier (pas juste le texte décodé) pour confirmer l'absence de troncature/corruption, puis en croisant la position exacte de l'arrêt avec la liste des `print()` du script contenant le caractère `→` — le tout premier de ces prints après l'entraînement correspond exactement au point d'arrêt.

**Décisions** :
- Fix appliqué directement (une ligne, `encoding="utf-8"`) sans passer par tout le cycle conception/plan — bug de correctness non ambigu, même traitement que le bug de replanification de route trouvé plus tôt dans le projet.
- Pas de test automatisé ajouté pour `_Tee`/le fichier de log : `run_rl_training.py` importe `carla` au niveau module, donc rien n'est testable depuis ce fichier sans CARLA installé — limite déjà actée et documentée pour ce fichier spécifique, non résolue ici (aurait demandé d'extraire `_Tee` dans un module séparé, hors périmètre d'un fix urgent).

**Benchmarks** : 163 tests (`uv run pytest benchmarks/ -q`), tous verts après le fix.

**Prochaine étape** :
- Avant d'envoyer `ppo_v6` : le lot v6 (progression, destination, sécurité positive, fluidité) reste pertinent mais son effet risque d'être marginal tant que le problème hors-route/hors-voie n'est pas investigué. Pistes à explorer : qualité du trajet planifié par Victor (A* résolution 2m), fiabilité de la détection de voie de Karim en trafic dense, seuil `_OFF_ROUTE_M=15m` peut-être trop strict vu le bruit de perception réel.
- Relancer un run avec le fix d'encodage pour enfin récupérer les données d'éval/démo (vitesse, frein, hors-route par scénario) — actuellement aveugle sur ce point pour toute la lignée v4.2→v5.

---

## 2026-07-06 (suite) — Second crash après le fix d'encodage : division par zéro sur petit run

**Avancement** :
- Franck a retenté un run après le fix du log (`--timesteps 10000`, smoke test `ppo_v6.2_10k`) — le fix d'encodage a fonctionné (`"Reward curve → ..."` s'affiche bien cette fois), mais un second bug jusque-là invisible a fait planter le script juste après, à la sélection des checkpoints pour l'éval : `ZeroDivisionError` sur `step = (len(ckpt_files) - 1) / (n_select - 1)` (`run_rl_training.py`).
- **Cause** : `n_select = min(10, max(1, args.timesteps // 10_000))` vaut **1** dès que `--timesteps <= 10_000`. Le garde-fou existant (`if n_select >= len(ckpt_files): selected = ckpt_files`) ne couvre que le cas où il y a peu de checkpoints — avec un `save_freq` plancher à 2048 steps, un run de 10k steps sauvegarde quand même 4-5 checkpoints, donc `n_select(1) >= len(ckpt_files)(4)` est faux, et le calcul du pas divise par `n_select - 1 = 0`.
- Jamais rencontré avant car tous les runs analysés jusqu'ici (100k-300k steps) donnaient `n_select=10`, jamais 1 — bug latent depuis l'introduction de cette logique de sélection, révélé seulement par un petit smoke test.
- **Fix** : cas `n_select <= 1` traité à part, sélectionne directement le dernier checkpoint (le plus représentatif du modèle entraîné) sans passer par le calcul de pas. Vérifié en isolant exactement le scénario du crash (4 checkpoints, `n_select=1`) avant de toucher au vrai script — plus de division par zéro, sélection correcte.
- Un message `ERROR: failed to destroy actor ... not found` apparaît aussi dans les logs juste avant la traceback — vient de CARLA lui-même (pas une exception Python), bénin, l'exécution continue normalement. Pas d'action prise dessus.

**Difficultés** :
- Aucune — cause identifiée directement depuis la traceback complète cette fois (contrairement au bug d'encodage qui n'en laissait aucune trace).

**Décisions** :
- Fix appliqué directement, même traitement que les bugs précédents (correctness non ambiguë, pas de décision de design). Pas de test automatisé possible pour la même raison que le bug précédent (`run_rl_training.py` importe `carla` au niveau module).

**Benchmarks** : 163 tests (`uv run pytest benchmarks/ -q`), tous verts après le fix.

**Prochaine étape** :
- Les deux bugs de crash (encodage + division par zéro) sont réglés — un run devrait maintenant aller jusqu'au bout, éval et démo comprises.
- Le point de fond reste entier : le problème hors-route/hors-voie identifié sur `ppo_v5` n'est traité par aucun de ces fixes ni par le lot v6. Décision pour l'instant : lancer `ppo_v6` quand même pour voir, plutôt que d'investiguer en amont.

---

## 2026-07-06 (suite) — Troisième blocage, cette fois sur mon propre PC : chemin Windows accentué

**Avancement** :
- En testant le chargement du modèle YOLOPv2 de Karim (`src/lane_detection/lane_perception.py`) sur mon PC fixe, `LaneDetector()` plantait avec `RuntimeError: ... No such file or directory` sur `weights/yolopv2.pt` — alors que le fichier existait bien, avec la bonne taille (156 Mo, vérifié identique à la référence).
- **Cause** : bug connu de PyTorch sous Windows — `torch.jit.load()` reçoit le chemin comme une chaîne et le passe à son chargeur C++, qui ne gère pas correctement les caractères non-ASCII dans le chemin. Mon nom d'utilisateur Windows contient un accent (`Frédéric`), ce qui suffit à faire échouer l'ouverture côté C++ même si le fichier est bien là.
- **Fix** : `LaneDetector.__init__` ouvre maintenant le fichier lui-même en Python (`open(..., "rb")`, qui gère l'Unicode correctement sur Windows) et passe l'objet fichier déjà ouvert à `torch.jit.load()` plutôt que le chemin brut — contourne entièrement le chargeur C++ pour la résolution du chemin.
- Touche le module de Karim (`lane_detection/`), mais corrigé directement comme le bug offset/angle du 2026-07-02 : périmètre limité à cette ligne précise, aucun test existant ne couvre ce fichier (nécessite les vrais poids YOLOPv2 + torch, hors périmètre des smoke tests CARLA-free).

**Difficultés** :
- Piste trouvée rapidement car le seul élément distinctif de mon environnement par rapport aux autres PC de l'équipe était l'accent dans le chemin utilisateur — pas de temps perdu sur de fausses pistes (taille de fichier, permissions, etc.), déjà écartées avant de creuser le chargeur PyTorch lui-même.

**Décisions** :
- Fix appliqué directement, non ambigu, même traitement que les bugs précédents de la journée.

**Benchmarks** : 163 tests (`uv run pytest benchmarks/ -q`), tous verts — aucun test ne couvre `LaneDetector` directement, donc pas de test cassé ni de test à ajouter pour ce fix précis.

**Prochaine étape** :
- Signaler à Karim ce bug potentiel sur son module, au cas où d'autres membres de l'équipe auraient un nom Windows accentué.
- Sinon, rien de plus à faire avant de lancer `ppo_v6`.

---

## 2026-07-07 (suite) — Confusion identifiée sur les collisions en 1 step de `curve_left`/`lane_change`

**Avancement** :
- Investigation du todo en attente sur les collisions suspectes en 1 step des scénarios `curve_left` et `lane_change` du benchmark. Aucun des deux n'a de `setup_fn` (pas de PNJ dédié). En creusant `eval_model()`/`_record_scenarios()` (`src/ai/inference/rl_demo.py`), confirmation qu'aucun PNJ n'est jamais nettoyé ou vérifié avant l'enregistrement d'un scénario.
- Les 18 véhicules + 6 piétons PNJ sont spawnés une seule fois pour toute la session (training + éval) et errent en continu en pilote automatique pendant les 13 scénarios de chaque évaluation de checkpoint. Une collision en 1 step sur un scénario sans `setup_fn` est donc très probablement un PNJ ambiant qui se trouvait par hasard à cet endroit précis à ce moment précis — un problème de chance, pas de qualité de la policy ni de mauvais choix de `spawn_idx`.
- **Fix** : nouvelle fonction `_clear_spawn_area(env, radius_m)` dans `rl_demo.py`, appelée juste après le reset de chaque scénario (dans `eval_model()` et `_record_scenarios()`) — repère tout PNJ à moins de 20m du point de spawn et le téléporte au point de spawn de la carte le plus éloigné de l'ego. Sélection déterministe (pas aléatoire) pour garder les évaluations reproductibles d'un run à l'autre. Aucun PNJ n'est restauré ensuite — les scénarios qui ont vraiment besoin de trafic proche spawnent déjà le leur via `setup_fn`.
- 169 tests (`uv run pytest benchmarks/ -q`), tout vert (6 nouveaux tests pour `_clear_spawn_area`).

**Difficultés** :
- Aucune — bug de conception plutôt qu'un vrai bug : la fonctionnalité de nettoyage n'avait simplement jamais été prévue lors de l'écriture initiale du pipeline d'éval.

**Décisions** :
- Rayon de nettoyage fixé à 20m (plus large que les 10m utilisés pour la sécurité de spawn à l'entraînement, car ici il faut couvrir tout l'enregistrement du scénario, pas juste l'instant du spawn).
- Appliqué uniformément à tous les scénarios (pas de champ optionnel par scénario) — la malchance de position PNJ peut toucher n'importe quel spawn fixe, pas seulement les deux repérés initialement.

**Benchmarks** : 169 tests, tout vert.

**Prochaine étape** :
- Reste dans la liste : décider quoi faire du choix de `best_model`, HUD démo, mini-carte GPS + overlay bounding-box sur les vidéos, passage à Town2 (avec, à cette occasion, une réflexion sur un éventuel frame-skip pour accélérer l'entraînement).

---

## 2026-07-07 (suite) — `best_model.zip` remplacé par une vraie sélection sur le benchmark complet

**Avancement** :
- Fix du problème identifié lors de l'analyse `ppo_v6.3_300k` : `best_model.zip` était choisi par `EvalCallback` (Stable-Baselines3) sur seulement 3 épisodes bruités pendant l'entraînement, puis servait tel quel pour la vidéo démo finale — c'est ce choix bruité qui avait donné un `off_route_pct` de 31.4% (pire que le checkpoint 120k à 8.4%).
- `scripts/run_rl_training.py` évalue déjà tous les checkpoints + `best_model` sur les 13 scénarios du vrai benchmark juste après l'entraînement (`evals/results.json`) — aucune éval CARLA supplémentaire nécessaire, juste exploiter ce qui est déjà calculé.
- **Fix** : nouvelle fonction pure `pick_best_checkpoint(all_results)` dans `rl_demo.py` — classe les candidats par nombre de scénarios Phase 1 réussis, puis départage par le plus petit `off_route_pct` moyen (la métrique qui a servi à démontrer le problème). Le script recharge le vrai gagnant, écrase `best_model.zip` sur le disque avec ses poids, et l'utilise pour la vidéo démo finale — la convention "`best_model.zip` = le meilleur du run" est maintenant vraie pour de bon.
- 173 tests (`uv run pytest benchmarks/ -q`), tout vert (4 nouveaux tests pour `pick_best_checkpoint`).

**Difficultés** :
- Aucune — la donnée nécessaire (`all_results`) existait déjà, il ne restait qu'à l'exploiter correctement plutôt que de faire confiance au choix de `EvalCallback`.

**Décisions** :
- Le mécanisme de sélection interne de `EvalCallback` reste actif pendant l'entraînement (il produit toujours un `best_model.zip` intermédiaire) — seul le fichier final sur le disque est écrasé après coup par le vrai gagnant, pas de changement côté SB3 lui-même.
- Règle de score fixe (pas configurable) : succès Phase 1 d'abord, `off_route_pct` en départage — reflète exactement le raisonnement déjà utilisé pour diagnostiquer le problème sur `ppo_v6.3`, pas besoin de plus compliqué pour l'instant.

**Benchmarks** : 173 tests, tout vert.

**Prochaine étape** :
- Reste dans la liste : HUD démo, mini-carte GPS + overlay bounding-box sur les vidéos, passage à Town2 (avec réflexion sur un éventuel frame-skip).

---

## 2026-07-07 (suite) — Panneau obs-space ajouté à la démo libre

**Avancement** :
- La démo libre (`demo.mp4`, chemin `_record_episodes()` de `rl_demo.py`) n'affichait que les barres `_add_hud` (paramètres, épisode/step, reward, vitesse, action) — contrairement aux vidéos d'éval qui affichent en plus le panneau obs-space en haut à droite (commande nav, angle/offset de voie, on-road, distances véhicule/feu/piéton/stop, actions brutes).
- **Fix** : ajout de l'appel à `_draw_obs_panel()` (déjà existant, déjà utilisé par `eval_model()`) dans `_record_episodes()`, à l'identique — même position relative à `_add_hud`, aucun nouveau paramètre.
- 174 tests (`uv run pytest benchmarks/ -q`), tout vert (1 nouveau test).

**Difficultés** :
- Aucune — ajout d'un seul appel à une fonction déjà existante et déjà testée en usage (via `eval_model`).

**Décisions** :
- Test par vérification d'appel (spy sur `_draw_obs_panel`) plutôt que par inspection de pixels — le comportement du panneau lui-même (garde largeur > 400, contenu affiché) est déjà la responsabilité de `_draw_obs_panel`, inchangée ici ; seule l'ajout du point d'appel est dans le périmètre de ce fix.

**Benchmarks** : 174 tests, tout vert.

**Prochaine étape** :
- Reste dans la liste : mini-carte GPS + overlay bounding-box sur les vidéos eval/démo, passage à Town2 (avec réflexion sur un éventuel frame-skip).

---

## 2026-07-07 (suite) — Mini-carte GPS persistante ajoutée aux vidéos d'éval

**Avancement** :
- Les vidéos d'éval (`eval_model()`, celles utilisées pour `evals/results.json`) n'avaient aucune vue carte pendant la conduite — seule la démo libre avait une carte plein écran, affichée une seule fois avant le départ (`_draw_route_map_card`).
- **Fix** : nouvelle fonction `_draw_minimap()` dans `rl_demo.py` — mini-carte 130×130px en bas à gauche, redessinée à chaque step (contrairement à la carte plein écran, statique) : trajectoire planifiée, marqueur de départ, marqueur de destination, et un point qui suit la position réelle du véhicule. Positionnée pour ne chevaucher ni le panneau obs-space (haut droite) ni la barre HUD du bas.
- Portée volontairement limitée à `eval_model()` — pas ajoutée à la compilation démo des 13 scénarios (`_record_scenarios()`), qui reste inchangée pour l'instant.
- 178 tests (`uv run pytest benchmarks/ -q`), tout vert (4 nouveaux tests pour `_draw_minimap`).

**Difficultés** :
- Aucune — réutilisation directe des techniques déjà en place (fond semi-transparent de `_draw_obs_panel`, mise à l'échelle + inversion Y de `_draw_route_map_card`).

**Décisions** :
- Position et taille fixes, pas de paramètre configurable — même convention que `_draw_obs_panel`, un widget de taille fixe quelle que soit la résolution de la vidéo.

**Benchmarks** : 178 tests, tout vert.

**Prochaine étape** :
- Reste dans la liste : overlay bounding-box sur les vidéos eval/démo, passage à Town2 (avec réflexion sur un éventuel frame-skip).

---

## 2026-07-07 (suite) — Overlay bounding-box ajouté aux vidéos eval/démo

**Avancement** :
- Aucune des vidéos enregistrées (éval, démo libre, compilation démo) n'affichait les détections de la pipeline de perception (Franck) — seul `demo_perception_live.py`, son script de démo autonome, dessine des bounding boxes.
- Vérifié avant de coder : la caméra d'entraînement (perception) et la caméra d'enregistrement (vidéo) sont toutes les deux spawnées avec la même transform/résolution/FOV et tickent en synchro — les objets détectés sur l'une s'alignent pixel pour pixel sur les frames enregistrées par l'autre, sans repasser par la perception une deuxième fois.
- **Fix** : `CarlaEnv` expose maintenant `last_objects` (les détections du step courant, déjà calculées dans `_get_obs()` mais jusqu'ici jetées). `rl_demo.py` gagne `_draw_bboxes()`, qui reprend telles quelles les couleurs et le format de labels de `demo_perception_live.py`. Appelée dans les trois chemins d'enregistrement : `eval_model()`, `_record_episodes()` (démo libre) et `_record_scenarios()` (compilation démo).
- 185 tests (`uv run pytest benchmarks/ -q`), tout vert (7 nouveaux tests).

**Difficultés** :
- Aucune — la seule vraie question (est-ce que les deux caméras voient bien la même chose) a été vérifiée par lecture du code de spawn des capteurs avant d'écrire quoi que ce soit, plutôt que supposé.

**Décisions** :
- Pas de deuxième appel à la perception pour la vidéo — réutilisation de `last_objects`, calculé une seule fois par step.
- Couleurs reprises telles quelles de `demo_perception_live.py` (tuples "BGR" au sens du commentaire de ce fichier) — appliquées ici sur de vraies frames BGR (contrairement à ce script qui les applique sur du RGB), donc le rendu est correct dans nos vidéos même si ce n'est peut-être pas le cas dans la démo live de Franck (pas dans le périmètre de ce fix).

**Benchmarks** : 185 tests, tout vert.

**Prochaine étape** :
- Reste dans la liste : passage à Town2 (avec réflexion sur un éventuel frame-skip pour accélérer l'entraînement).

---

## 2026-07-07 (suite) — Observation réduite à 11 paramètres, retrait de l'angle de voie

**Avancement** :
- `lane_angle_norm` retiré de l'observation donnée au modèle — passage de 12 à 11 paramètres. Tout ce qui suivait dans le vecteur (offset, on_road, véhicule, feu, limite vitesse, piéton, stop/yield) décale d'un cran.
- Répercuté partout où l'observation est lue par position (`rl_env.py`, le panneau obs-space et les détecteurs de highlights de `rl_demo.py`, le contrôleur mock `demo_mockup.py`, la doc de `run_rl_training.py` et le README).
- Deux conséquences assumées : le contrôleur `RouteFollowPolicy` (démo mock, pas le modèle entraîné) ne pilote plus que sur l'offset latéral (`steer = -offset * 0.25`, terme d'angle retiré) ; les stats "heading" (angle moyen/max/std) de `eval_model()` ont disparu des résultats, plus de source pour les calculer.
- 184 tests (`uv run pytest benchmarks/ -q`), tout vert.

**Difficultés** :
- Aucune — changement mécanique une fois la liste complète des points de lecture positionnelle établie (`obs[N]` apparaît dans 6 fichiers).

**Décisions** :
- Pas de tentative de recalculer le cap ailleurs pour préserver les stats "heading" — cohérent avec le fait qu'on ne le donne plus au modèle du tout.

**Benchmarks** : 184 tests, tout vert.

**Prochaine étape** :
- Reste dans la liste : passage à Town02 (Stage A : `_EXPECTED_MAP` dans `launch_training.py`, en attente de l'explo de spawn côté utilisateur).

---

## 2026-07-08 — Pause d'1 seconde avant le départ sur les vidéos démo et éval

**Avancement** :
- Les vidéos (démo libre et les 13 scénarios d'éval) démarraient la conduite dès la toute première frame — aucun moment pour voir la position de départ avant que la voiture bouge.
- **Fix** : `_record_episodes()` et `eval_model()` gardent maintenant la première frame (juste après le reset, avant toute action) à l'écran pendant 1 seconde, avec le même empilement d'overlays que chaque step normal (HUD, bounding boxes, panneau obs-space, label de scénario et mini-carte pour l'éval).
- 185 tests (`uv run pytest benchmarks/ -q`), tout vert (1 nouveau test).

**Difficultés** :
- Deux tests existants (`test_record_episode_writes_route_map_card_frames`, `test_record_episode_no_route_attribute_skips_map_card`) faisaient une assertion exacte sur le nombre de frames écrites — mis à jour pour compter la nouvelle pause.
- Deux autres tests, non prévus au départ, se sont cassés en cours de route (`test_record_episode_calls_draw_obs_panel_per_frame`, `test_record_episode_calls_draw_bboxes_with_last_objects`) : ils comptaient les appels à `_draw_obs_panel`/`_draw_bboxes` en espionnant ces fonctions, et la frame de pause déclenche elle aussi ces deux appels (même empilement d'overlays qu'un step normal) — mis à jour de 2 à 3 appels attendus.

**Décisions** :
- Pause non configurable, fixée à 1 seconde — correspond à la demande telle quelle, pas besoin de plus pour l'instant.
- Pas de nouvelle fonction partagée entre les deux call sites malgré la duplication partielle — les deux empilements d'overlays diffèrent déjà (l'éval a le label de scénario et la mini-carte, pas la démo libre), extraire un helper commun aurait ajouté de l'indirection pour deux call sites seulement.

**Benchmarks** : 185 tests, tout vert.

**Prochaine étape** :
- Outil interactif d'inspection des spawn points (en cours).
- Reste dans la liste : passage à Town2 (Stage A fait, calibration Town02 faite — reste à traiter les points de spawn identifiés comme mauvais, ex. spawn_idx=13).

---

## 2026-07-08 (suite) — Outil interactif d'inspection des spawn points

**Avancement** :
- Depuis la découverte que `spawn_idx=13` sur Town02 tue l'agent en 1 step (confirmé sur 3 scénarios différents dans `ppo_v7_120k`), besoin d'un moyen de vérifier visuellement un point de spawn avant de le committer dans `BENCHMARK_SCENARIOS` — jusqu'ici on faisait confiance aveuglément aux heuristiques géométriques de `explore_spawns.py`.
- **Nouveau script** `scripts/inspect_spawns.py` : téléporte un véhicule statique (pas d'autopilot, pas de PNJ) sur un `spawn_idx` donné, affiche la caméra live dans une fenêtre pygame avec les heuristiques déjà calculées par `explore_spawns.py` (angle de virage, distance jonction/feu, limite de vitesse, tags) superposées à l'écran. Navigation au clavier : flèches/n-p pour suivant/précédent, chiffres + Entrée pour sauter directement à un index.
- Réutilise entièrement l'infra existante : `CameraSensor`/`write_rgb` (collecte dataset), la transform caméra partagée de l'équipe, et `_analyse_spawns()`/`SpawnInfo` de `explore_spawns.py` — aucune nouvelle logique de géométrie.

**Difficultés** :
- `_analyse_spawns()` peut sauter des indices (si `get_waypoint()` échoue pour ce point) — le mapping index→heuristiques doit se faire par dictionnaire (`{info.idx: info}`), pas par position dans la liste, sinon les heuristiques affichées auraient correspondu au mauvais spawn_idx.

**Décisions** :
- Aucun test automatisé — outil interactif dépendant de CARLA, sans surface testable hors-ligne, cohérent avec `demo_perception_live.py` et `explore_spawns.py` (ni l'un ni l'autre n'a de test).
- Pas de mécanisme "marquer bon/mauvais" intégré — l'utilisateur inspecte visuellement et remonte les index à la main, plus simple que de construire un flux d'annotation pour un usage ponctuel.

**Benchmarks** : sans objet (script sans test).

**Prochaine étape** :
- Utiliser l'outil pour identifier tous les spawn points problématiques de Town02 (à commencer par spawn_idx=13) et corriger `BENCHMARK_SCENARIOS` en conséquence.

---

## 2026-07-08 (suite) — Détection automatique des spawns cassés et du reward-hacking dans analyze_run.py

**Avancement** :
- Deux patterns trouvés à la main en diagnostiquant `ppo_v7_120k` (spawn cassé = mort en ≤3 steps sur tous les checkpoints d'un même scénario ; reward qui monte pendant que les épisodes s'effondrent en durée = signal de reward-hacking) sont maintenant calculés automatiquement par `scripts/analyze_run.py`, dans le même esprit "données brutes, pas d'interprétation" que le reste du fichier.
- Nouvelle fonction `_by_scenario_benchmark()` : vue par scénario à travers tous les checkpoints (l'axe inverse de `_summarize_benchmark`), avec un flag `likely_broken_spawn` (steps ≤3 sur au moins 2 checkpoints, tous systématiquement). `_totals()` gagne `reward_length_correlation` (corrélation de Pearson reward/durée d'épisode sur tout le run).
- Corrigé au passage : `params.json` affichait un label `"obs"` figé en dur (`"12-scalars-traffic"`), jamais mis à jour après le retrait de l'angle de voie — maintenant dérivé dynamiquement de `len(_OBS_LOW)`.
- 190 tests (`uv run pytest benchmarks/ -q`), tout vert (5 nouveaux tests).

**Difficultés** :
- Aucune — patterns déjà bien définis depuis le diagnostic manuel de `ppo_v7_120k`, juste à les rendre automatiques.

**Décisions** :
- Le flag `likely_broken_spawn` reste une donnée calculée, pas une action automatique — cohérent avec la philosophie déjà en place de `analyze_run.py` ("computes; a human still writes the diagnosis").
- Seuil de ≤3 steps et minimum de 2 checkpoints fixés en dur, pas configurables — mêmes constantes que celles déjà utilisées à la main pendant le diagnostic, pas besoin de plus pour l'instant.

**Benchmarks** : 190 tests, tout vert.

**Prochaine étape** :
- Reste dans la liste : merger `fix/nav-reward-speed`, investiguer le faux bonus +10 sur spawn_idx=13, utiliser `inspect_spawns.py` pour identifier tous les spawns cassés de Town02.

---

## 2026-07-09 — Lot de correctifs root-cause : reset() désynchronisé du teleport, reward hors-route, EvalCallback, scoring benchmark, crop YOLO, fuite matplotlib

**Avancement** :
- Bug root-cause trouvé dans `reward_fn.py` : `r_center` et `r_speed` étaient calculés sans condition à partir de `center_offset` et de la vitesse, y compris quand `is_on_road=False`. Or `center_offset` retombe à `0.0` (valeur "parfaitement centré") dès que le détecteur de ligne de Karim perd la voie — ce qui arrive justement quand la voiture sort de la route. Résultat : au moment précis où la voiture est hors-piste, elle touchait quasiment son reward par-step maximal (r_center proche du plafond + r_speed plein si elle roulait encore), à peine compensé par la pénalité `r_offroad` de -0.25. Quasiment aucune incitation à revenir sur la route une fois sortie.
- `compute_reward()` gate maintenant `r_center` et `r_speed` sur `is_on_road` : les deux tombent à `0.0` dès que `is_on_road=False`, quel que soit `center_offset` ou la vitesse rapportée.
- `_P_OFFROAD` remonté de `-0.25` à `-0.5`, réaligné avec `_P_OFF_ROUTE` (les deux étaient volontairement désynchronisés depuis le batch v4, ce qui n'a plus lieu d'être une fois le vrai problème (le faux reward positif) corrigé à la source).
- `_DEFAULT_SPEED_LIMIT_KMH` abaissé de `50.0` à `30.0` dans `rl_env.py` : réactive de facto le `r_speeding` proportionnel (déjà implémenté et testé, mais quasi jamais déclenché tant qu'aucun panneau n'a été vu) comme plafond de vitesse "soft" par défaut en début d'épisode.
- 2 nouveaux tests smoke (`test_r_center_zeroed_when_off_road`, `test_r_speed_zeroed_when_off_road`), plus mise à jour de `test_offroad_applies_penalty` (delta hors-route désormais -0.90 au lieu de -0.25 : 0.0 + 0.0 + 0.01 + (-0.5) + 0.05 = -0.44 contre 0.46 on-road). `README.md` mis à jour (tableau reward + commentaires sur le gating).
- **`reset()` désynchronisé du teleport (`rl_env.py`)** : `set_transform()` ne prend effet côté client qu'au `world.tick()` suivant — `reset()` lisait la position de l'ego (pour `nav.plan()` et `_episode_start_location`) et remettait `_collision_flag` à `False` *avant* la boucle de warmup ticks, donc à partir de la position de fin de l'épisode *précédent*. Réordonné : teleport → warmup ticks → puis nav.plan/capture de position/reset des compteurs par épisode. Explique le faux bonus +10 "destination atteinte" vu sur spawn_idx=13 (0.38m parcourus mesurés depuis le mauvais point de départ) et plusieurs morts en 1 step (collision pendant les ticks de warmup, attribuée au nouvel épisode avant même une action de la policy).
- **Contrôle non réinitialisé au teleport (`rl_env.py`)** : `_teleport_to_spawn()` remettait position et vitesse à zéro mais jamais le contrôle (throttle/steer/brake) — le dernier contrôle de l'épisode précédent (souvent plein gaz au moment du crash) restait actif pendant les ticks de warmup et le tout début du nouvel épisode. Ajout de `apply_control(VehicleControl())` après le teleport.
- **`is_on_road` toujours faux aux intersections (`rl_env.py`)** : sans marquage au sol, le détecteur de ligne ne peut jamais dire "sur la route" à une intersection. Comme le gating ci-dessus punit maintenant tout ce qui n'est pas `is_on_road`, une traversée d'intersection légitime aurait commencé à être pénalisée. Ajout d'un second critère local (pas de changement côté détection de ligne de Karim) : `is_on_road` devient vrai aussi si le waypoint CARLA sous l'ego est dans une jonction (`get_waypoint(...).is_junction`).
- **`EvalCallback` retiré du training (`run_rl_training.py`)** : il partageait le même `CarlaEnv` que la collecte de rollout PPO — toutes les `eval_freq` steps il réinitialisait cet env en plein rollout pour 3 épisodes d'éval, corrompant l'état interne de PPO et gaspillant du budget de steps jamais compté. Son propre choix de `best_model.zip` n'était de toute façon déjà plus utilisé (voir plus bas, entrée `ppo_v6.3_300k`) — seul reste `CheckpointCallback`. Le bloc qui chargeait ensuite ce fichier comme candidat de benchmark supplémentaire a été supprimé (il n'existe plus sans EvalCallback) ; garde-fou ajouté si aucun checkpoint n'a été sauvegardé (run plus courte que `save_freq`).
- **Scoring benchmark pouvait compter un succès comme un crash (`rl_demo.py`)** : dans `eval_model()`, `terminated=True` (renvoyé par l'env aussi bien pour une collision que pour la destination atteinte) était enregistré comme un crash *avant* que le test de distance à la destination ne soit vérifié — si les deux tombaient sur le même step, un vrai succès finissait classé comme collision. Réordonné : le test de destination tourne en premier, et l'enregistrement de collision est désormais gaté sur `not reached_dest`.
- **Crop YOLO pouvait planter sur une bbox dégénérée (`detector.py`)** : une bbox de feu tricolore à aire nulle (`x1==x2` ou `y1==y2`) faisait planter `cv2.cvtColor` avant même d'atteindre le garde-fou existant de `classify_tl_color`. Garde-fou ajouté une ligne plus tôt.
- **Fuite de figure matplotlib (`matplot_visualizer.py`)** : `plot_plan()`/`plot_road_network()` n'appelaient jamais `plt.close()` après `plt.savefig()` — fuite à chaque appel, donc à chaque `nav.plan()`, donc à chaque reset d'épisode sur toute la durée d'un training. `plt.close()` ajouté aux deux.

**Difficultés** :
- Aucune sur l'implémentation du fix reward lui-même — le calcul du nouveau delta (-0.90) demandait de resommer les 5 composantes actives à la main pour ne pas se tromper de signe, vérifié par le test avant de l'écrire en dur dans la docstring.
- Le lot touche `rl_env.py` à trois endroits différents (reset/teleport/obs) répartis sur deux tâches distinctes — vérifié qu'aucune des deux ne se marchait dessus avant de considérer le lot terminé.

**Décisions** :
- Gater aussi `r_speed`, pas seulement `r_center` : une version antérieure plus étroite du fix (r_center seul) avait été commencée mais jamais mergée — en la reprenant j'ai réalisé qu'elle laissait le même trou côté vitesse (rouler vite hors-piste restait récompensé), donc les deux sont traités ensemble ici.
- `_P_OFFROAD` remonté exactement au niveau de `_P_OFF_ROUTE` (-0.5) plutôt qu'à une valeur intermédiaire : les deux pénalisent des sorties de la trajectoire prévue, pas de raison de les garder désynchronisées maintenant que le "crash fast shortcut" qui avait motivé le -0.25 n'est plus alimenté par un faux positif.
- Pour `is_on_road` aux intersections : plutôt que de faire remonter le masque `drivable` déjà calculé (mais jamais utilisé) côté YOLOPv2 de Karim — ce qui aurait changé la signature publique de son module — utilisation du waypoint CARLA (déjà disponible via `world.get_map()` dans ce fichier). Solution locale à `rl_env.py`, un peu moins précise (une voiture arrêtée hors-piste juste à côté d'une jonction pourrait être classée à tort sur la route) mais qui ne touche à aucun module d'un autre membre de l'équipe.
- Toutes les erreurs de `nav.plan()` étaient avalées silencieusement (`except Exception: pass`) — remplacé par un `print()`, seul mécanisme de log déjà utilisé ailleurs dans ce projet (pas de module `logging` configuré nulle part dans `src/`).

**Benchmarks** : 208 tests (`uv run pytest benchmarks/ -q`), tout vert (190 avant ce lot).

**Prochaine étape** :
- Lancer un run de training court (smoke test) avec ce lot complet avant de relancer un vrai run long, pour vérifier que rien ne casse en conditions réelles CARLA (tout ce qui précède n'a été vérifié qu'en tests unitaires, CARLA non disponible en local).
- Remplacer spawn_idx=13 (confirmé cassé sur `ppo_v7_120k`) une fois `inspect_spawns.py` utilisé pour trouver un remplaçant correct dans Town02.
- Vérifier sur un run d'entraînement réel que le temps passé hors-piste baisse effectivement avec le fix reward, pas seulement au niveau unitaire.

---

## 2026-07-11 — v10 : `r_speed` remplacé par `r_progress` (shaping potential-based), mémoire d'action dans l'observation

**Avancement** :
- Trois runs consécutifs (v7 → v8 → v9, tous entraînés avec les fixes des lots précédents) montrent le même schéma structurel, de pire en pire : les pénalités par step pour une conduite imparfaite (hors-route, stall, following, etc.) finissent par coûter plus cher sur un épisode qui dure que le coût one-shot d'une collision. Sur v9, ça s'est complètement effondré : 99,3 % de taux de collision, action quasi-constante braquage-à-fond/plein-gaz sur les 13 scénarios du benchmark quel que soit ce que chacun demande, corrélation reward/longueur d'épisode à -0,774. `AUDIT.md` §2.1 avait déjà posé ce diagnostic à l'avance et proposé trois pistes classées par simplicité d'implémentation (pas par efficacité attendue) : monter `w_alive`, une pénalité de collision proportionnelle au reward restant de l'épisode, ou un shaping potential-based sur la distance à la destination — la seule des trois avec une garantie théorique contre les raccourcis, et qui réglait au passage le problème voisin de `r_speed` (récompenser la vitesse brute, pas le progrès utile).
- Décidé de ne plus corriger en périphérie une troisième fois (le risque étant de simplement déplacer l'exploit ailleurs) et d'attaquer la cause racine : `r_speed` est retiré de `reward_fn.py`, remplacé par `r_progress`, un reward shaping potential-based classique : `Φ(s) = −distance_à_la_destination(s) normalisée`, `r_progress = (γ·Φ(s') − Φ(s)) × 0.3` avec `γ = 0.99` (doit rester égal au gamma de PPO pour que la garantie d'invariance de policy du shaming potential-based tienne). `r_progress` n'est volontairement pas gaté sur `is_on_road`, contrairement à l'ancien `r_speed` — `r_offroad` reste seul juge de la sortie de route, ça garde la fonction de potentiel simple.
- `CarlaEnv` calcule maintenant `progress_delta` à chaque step à partir d'une nouvelle méthode `_dist_to_destination()` (extraite de `_reached_destination()`, qui l'utilise désormais aussi plutôt que dupliquer le calcul), et le passe à `compute_reward()`. La distance initiale de l'épisode (`_dist_to_dest_initial`, calculée au `reset()` juste après la replanification de route, plancher à 1.0) sert à normaliser le signal en `[-1, 0]` indépendamment de si la destination aléatoire de l'épisode est à 50 m ou 300 m.
- Deuxième volet du lot, indépendant : l'observation gagne deux scalaires, `prev_steer_norm`/`prev_accel_norm` (le dernier steer/accel appliqué, déjà dans `[-1, 1]`, pas de renormalisation) — passage de 11 à 13 scalaires. Aujourd'hui l'observation était purement instantanée, sans aucune mémoire de ce que la policy venait de faire ; cette mémoire minimale (pas un vrai frame-stack) permet à la policy de conditionner sur son action récente, utile pour lisser le contrôle en complément de `r_jerk`.
- Point délicat dans `step()` : `self._prev_steer`/`self._prev_accel` doivent être mis à jour **avant** l'appel à `_get_obs()`, pas après comme c'était le cas pour `_prev_steer` jusqu'ici — sinon l'observation renvoyée à la policy contient l'action d'il y a deux steps, pas celle qui vient d'être appliquée. Réordonné en conséquence (le calcul de `steer_delta` remonte avec).
- `_progress_speed_kmh()` (projection de la vitesse sur le yaw du waypoint courant, qui n'alimentait plus que `r_speed`) est supprimée avec son unique appelant. `benchmarks/ai/smoke.py` et `benchmarks/ai/test_rl_env.py` mis à jour en conséquence (nouveaux tests `r_progress` positif/négatif/nul par défaut/nul en terminal, nouveaux tests obs 13 scalaires + `prev_steer`/`prev_accel`, suppression des tests qui ne testaient que le mécanisme `r_speed`/`progress_speed_kmh` retiré). README et docstring de `run_rl_training.py` mis à jour (tableau reward, layout obs).
- 243 tests (`uv run pytest benchmarks/ -q`), tout vert (238 avant ce lot, net +5 après suppressions/ajouts).

**Difficultés** :
- Recalcul à la main de plusieurs valeurs de reward attendues dans les tests existants après le retrait de `r_speed` (ex. `test_reward_components_sum_at_max` perdait son `+0.3` de vitesse, pas repéré du premier coup — repassé sur tout `smoke.py` pour vérifier qu'aucune autre assertion numérique ne dépendait silencieusement de `r_speed`).
- Effet de bord plus subtil sur `test_rl_env.py` : plusieurs tests utilisant la fixture par défaut (`_make_env()`, où la destination mockée coïncide exactement avec la position de spawn de l'ego, distance nulle) récupèrent un `r_progress` de `+0.3` au tout premier step après `reset()` — artefact du plancher à 1.0 sur `_dist_to_dest_initial` combiné à l'initialisation de `_prev_dist_to_dest_norm` à 1.0 : quand la distance réelle de départ est sous 1 m, la normalisation de départ ne colle plus exactement à l'hypothèse "on part à distance normalisée 1.0". Sans impact en conditions réelles (la sélection de destination garantit au moins 30 m entre spawn et destination, donc le plancher ne s'active jamais en pratique) mais ça a cassé 4 tests qui ne s'y attendaient pas (`test_red_light_violation_penalized_once_not_twice`, `test_stop_yield_violation_penalized_once_not_twice`, les deux `test_reward_center_term_uses_lane_offset_*`) — valeurs attendues recalculées et commentées plutôt que de complexifier la formule de shaping pour un cas qui ne se produit jamais en dehors de ce fixture de test.

**Décisions** :
- Weight de `r_progress` repris identique à l'ancien `_W_SPEED` (0.3) : ce composant reprend le même budget dans le reward total, pas de raison de le retuner en même temps que le mécanisme change.
- Pas de tentative de préserver la compatibilité avec les checkpoints v7/v8/v9 — l'espace d'observation passe de 11 à 13 scalaires (en plus du passage à une action 2D déjà fait en v9), aucun ancien checkpoint n'est chargeable contre ce nouvel env. Le prochain run repart de zéro.
- `prev_steer`/`prev_accel` ajoutés en fin de vecteur d'observation plutôt qu'insérés au milieu — évite de décaler tous les indices `obs[N]` déjà utilisés ailleurs dans `rl_env.py`/`rl_demo.py`.

**Benchmarks** : 243 tests, tout vert.

**Prochaine étape** :
- Reste dans le lot v10 : retune `ent_coef` (0.01 → 0.02), `VecNormalize(norm_reward=True)`, seeding `random`/NumPy/TrafficManager dans `run_rl_training.py`.
- Une fois le lot v10 complet : lancer un run 300k et comparer à v9 via `analyze_run.py`, en particulier sur la corrélation reward/longueur d'épisode qui s'était effondrée.

---

## 2026-07-11 (suite) — v11 : le redesign reward de v10 ne change rien au comportement mesuré, gSDE/ent_coef + rééquilibrage `w_progress`/`w_alive`

**Avancement** :
- Run `ppo_v10_150k` analysée (interrompue à ~135-150k steps côté éval vidéo, mais training complet jusqu'à 150k steps, `params.json` confirme le bon code déployé : `obs: "13-scalars-traffic"`, `w_progress: 0.3`, `ent_coef: 0.02`) et comparée directement à `ppo_v9_300k` sur `training_log.monitor.csv` : reward moyen -37.39 → -37.97, longueur moyenne d'épisode 79.7 → 80.4 steps, taux de crash 99.3% → 98.2%, `r_offroad` moyen/épisode -23.95 → -24.15, ~60% de temps hors-piste inchangé. **Chiffres quasiment identiques malgré un vrai changement de reward** (`r_speed` retiré, `r_progress` potential-based ajouté, mémoire d'action dans l'observation, `ent_coef` monté à 0.02, `VecNormalize`, seeding). La corrélation reward/longueur reste fortement négative (-0.653), signe que "finir vite plutôt que bien conduire" reste l'attracteur dominant.
- Cause identifiée par calcul direct : la contribution réelle mesurée de `r_progress` est de ~+0.003/step en moyenne — négligeable face à `r_offroad` (-0.5/step) ou `r_stall` (-0.20/step). Le shaping potential-based garantit qu'on ne peut pas le tricher, mais rien ne garantissait qu'il soit assez **fort** ; avec un poids repris tel quel de l'ancien `_W_SPEED` (0.3), il ne l'était pas.
- Inspection vidéo de `checkpoint_0120k.mp4` (4 instants pris à des scénarios différents, hors frame de pause initiale) : `steer` vaut systématiquement -1 ou +1, jamais une valeur intermédiaire ; `throttle`/`brake` valent 0 ou 1, jamais entre les deux. Pas exactement le réflexe fixe unique de v9 (ici le steer alterne selon le scénario, et `turn_left` montre un vrai freinage complet), mais la même famille de pathologie : la policy ne fait plus de contrôle continu, elle sort uniquement des commandes tout-ou-rien, et 3 des 4 instants avec une vraie décision sont hors-piste au moment de l'extrême.
- Deux pistes complémentaires retenues plutôt qu'une seule, vu que deux tentatives consécutives (v9 puis v10) ont déjà montré qu'un correctif isolé ne suffit pas :
  1. **Exploration** (`src/ai/training/rl_train.py`) : `ent_coef` remonté 0.02 → 0.05, et gSDE (`use_sde=True`, `sde_sample_freq=4`) activé — le bruit d'exploration devient temporellement corrélé au lieu d'un tirage gaussien indépendant à chaque step, ce qui cible directement le symptôme "collapse vers les extrêmes de l'espace d'action" observé sur la vidéo, indépendamment de la structure du reward.
  2. **Reward** (`src/ai/rewards/reward_fn.py`) : `_W_PROGRESS` remonté 0.3 → 1.0 (le poids seul ne corrige pas le mécanisme mais lui donne enfin un poids comparable à `r_offroad`/`r_stall` plutôt qu'un ordre de grandeur en dessous) et `_W_ALIVE` remonté 0.01 → 0.05 (option (a) de `runs/AUDIT.md` §2.1, jamais essayée jusqu'ici — rend la simple survie mécaniquement plus rentable, en complément direct du shaping plutôt qu'à sa place).
- `benchmarks/ai/smoke.py` : nouveau test `test_reward_weights_updated_for_v11` qui verrouille `_W_ALIVE`/`_W_PROGRESS` sur leurs valeurs numériques exactes plutôt que de ne les vérifier qu'indirectement via une somme de composantes — un futur changement de poids sera immédiatement détecté. Docstrings de `test_offroad_applies_penalty` et `test_reward_components_sum_at_max` recalculées (le delta -0.80 entre on-road/off-road ne bouge pas car `r_alive` contribue également des deux côtés, mais les totaux absolus si). Deux tests de `test_rl_env.py` (`test_reward_center_term_uses_lane_offset_when_centered`/`_off_center`) recalculés pour la même raison — pas prévus dans le plan initial, repérés en faisant tourner la suite complète après le changement de poids.
- 246 tests (`uv run pytest benchmarks/ -q`), tout vert (243 avant ce lot).

**Difficultés** :
- Aucune sur l'implémentation — les deux changements (exploration et reward) sont indépendants et ne touchent pas les mêmes fichiers. La seule vigilance a été de recalculer à la main chaque assertion numérique de `smoke.py`/`test_rl_env.py` qui embarquait silencieusement l'ancien `_W_ALIVE=0.01`, plutôt que de les ajuster au pif jusqu'à ce que ça passe.

**Décisions** :
- Ni `ent_coef=0.05` ni `_W_PROGRESS=1.0`/`_W_ALIVE=0.05` ne sont des optima dérivés — ce sont des ajustements motivés par le diagnostic (poids 100x trop faible pour `r_progress`, collapse d'exploration visible sur vidéo) mais non testés en isolation. Assumé volontairement : après deux runs consécutifs (v9, v10) montrant qu'un seul levier ne suffit pas, on préfère combiner deux pistes indépendantes et regarder le résultat agrégé plutôt que d'isoler chaque variable au prix d'un troisième run de 150k+ steps rien que pour ça.
- Pas touché aux options (b)/(c) restantes de l'audit (pénalité de collision proportionnelle au reward restant de l'épisode) ni à la piste "seed figé depuis v1" évoquée dans l'analyse v10 — gSDE/ent_coef couvre déjà l'hypothèse exploration/entropie, le seed lui-même reste inchangé pour ne pas perdre la reproductibilité d'une run à l'autre au même moment qu'on change autre chose.

**Benchmarks** : 246 tests, tout vert. Comparaison chiffrée v9 vs v10 documentée ci-dessus (source : `runs/2026-07-10_19-19_ppo_v9_300k/ANALYSIS.md` et `runs/2026-07-11_13-31_ppo_v10_150k/ANALYSIS.md`).

**Prochaine étape** :
- Lancer un run v11 (≥150k steps) avec ce lot complet (gSDE + ent_coef 0.05 + `w_progress`=1.0 + `w_alive`=0.05) et comparer à v9/v10 via `analyze_run.py` : reward moyen, taux de crash, longueur d'épisode, corrélation reward/longueur, et vérifier sur la vidéo si `steer`/`throttle`/`brake` sortent enfin des valeurs intermédiaires.
- Si le collapse persiste : reconsidérer les options (b)/(c) de `runs/AUDIT.md` §2.1 non encore essayées (pénalité de collision proportionnelle au reward restant) et l'hypothèse du seed figé depuis v1.

---

## 2026-07-13 — v12 : fix de l'aliasing offset/no-lane, économie de collision, changement de seed

**Avancement** :
- Premier volet du lot : `ppo_v11_150k` analysée (`runs/2026-07-12_00-45_ppo_v11_150k/ANALYSIS.md`) : le collapse vers les extrêmes persiste (100% de crash), mais l'analyse a mis au jour un second bug, indépendant du collapse lui-même et jamais isolé jusqu'ici — sur les 5867 steps loggés dans `evals/results.json`, `center_offset` vaut exactement `0.0` sur **88.3%** des steps, y compris pendant des trajectoires où la voiture part clairement en sortie de route (ex. `straight`/checkpoint 0150k : spirale de braquage plein droite sur 40 steps, `center_offset` reste à `0.0` sur 38 d'entre eux). Le bug est présent depuis au moins le Lot 2, pas spécifique à v11.
- Cause : `lane_geometry.py:59-70` (module de Karim) retourne son dict par défaut `{"offset": 0.0, "direction": "NONE"}` quand elle ne trouve pas assez de pixels pour suivre les deux bords de voie. Ce `0.0` veut dire *« pas de mesure »*, mais rien ne le distingue de *« parfaitement centré »* une fois recopié tel quel dans `lane_offset_norm` (`rl_env.py::_get_obs`) — contrairement à `r_center` dans le reward, déjà gaté sur `is_on_road` depuis le tout premier lot. La policy recevait donc un faux signal « tu es centré » dès qu'elle perdait la détection de ligne, potentiellement jusqu'à la fin de l'épisode.
- Fix implémenté (périmètre : uniquement la consommation dans `rl_env.py`, `lane_geometry.py` non touché — module de Karim, hors scope) : nouvel attribut d'instance `self._last_lane_offset_norm` (initialisé à `0.0` dans `__init__`, remis à `0.0` dans `reset()`). Dans `_get_obs()`, `lane_offset_norm` n'est mis à jour depuis la lecture fraîche `offset` que si `direction != "NONE"` ; sinon on réutilise `self._last_lane_offset_norm` (dernière lecture valide tenue en mémoire) plutôt que le `0.0` par défaut de `lane_geometry()`. Avant toute détection réussie (dès le spawn), la valeur tenue est `0.0` par convention (hypothèse "centré au spawn"), comportement inchangé par rapport à avant. `is_on_road` (calculé juste après à partir du même `direction`) n'est pas touché.
- TDD : 2 nouveaux tests dans `benchmarks/ai/test_rl_env.py` — `test_obs_lane_offset_holds_last_valid_reading_when_lane_lost` (perte de détection après une lecture valide → l'ancienne valeur est tenue) et `test_obs_lane_offset_defaults_to_zero_before_any_detection` (aucune détection valide encore → `0.0` par défaut, même si `lane_geometry()` renvoyait un offset non-nul à tort aux côtés de `direction="NONE"`, cas qui n'arrive jamais en production mais qui aurait pu fuiter à travers une implémentation naïve). Tous les tests `test_obs_lane_offset_*`/`test_obs_is_on_road_*` existants repassent sans modification (ils utilisent `on_road=True` par défaut, donc `direction != "NONE"` et la valeur se met à jour exactement comme avant).
- Deuxième volet du lot, indépendant du fix offset ci-dessus : reprise directe de l'option (b) de `runs/AUDIT.md` §2.1 (pénalité de collision proportionnelle au reward restant de l'épisode), jamais essayée jusqu'ici alors que v9/v10/v11 ont chacune contourné le même diagnostic par un autre levier (retrait de `r_speed`, potential-based shaping, `w_alive`/`w_progress`, gSDE) sans jamais s'attaquer à l'asymétrie elle-même. Le calcul est simple et n'avait encore jamais été posé noir sur blanc : `_P_OFFROAD` coûte -0.5/step, et un collision typique coûte environ -9.5 (base -5.0 plus la composante vitesse d'impact) — errer hors-piste plus de 19 steps avant de finir par percuter quelque chose (19 × -0.5 = -9.5, le point d'équilibre exact) coûte donc déjà plus cher que de crasher tout de suite. Sur `checkpoint_0150k` de `ppo_v11_150k` (`runs/2026-07-12_00-45_ppo_v11_150k/ANALYSIS.md`), le `steer` mesuré est littéralement constant (std=0.0) sur toute la durée de l'épisode quel que soit le scénario de benchmark testé (13 scénarios couvrant des observations très différentes) — la policy n'a donc pas appris à conduire, elle a trouvé et exploité ce raccourci arithmétique : crasher tout de suite, quelle que soit l'observation, bat systématiquement le fait d'essayer de récupérer une sortie de route.
- Fix (`reward_fn.py::compute_reward`, `rl_env.py::step()`) : nouveau paramètre `remaining_frac` (défaut `0.0`, donc aucun changement pour un appelant existant qui ne le passe pas), qui multiplie `r_collision` par un facteur `1.0 + clamp(remaining_frac, 0, 1)` — de 1x (crash pile à la destination) à 2x (crash juste après le spawn). `CarlaEnv.step()` calcule ce `remaining_frac` en réutilisant `dist_norm` (`dist_to_destination(s) / dist_to_destination(s0)`), déjà calculé pour le shaping `r_progress` — pas de nouveau calcul de distance, juste un branchement du chiffre déjà disponible. Un crash qui abandonne la quasi-totalité du trajet coûte maintenant sensiblement plus cher qu'un crash à l'arrivée, sans toucher aux pénalités par-step (`_P_OFFROAD`, `_P_STALL`) ni aux poids déjà retunés en v11 (`w_alive`, `w_progress`).
- TDD : 5 nouveaux tests dans `benchmarks/ai/smoke.py` (défaut inchangé, x2 à `remaining_frac=1.0`, x1.5 à `0.5`, clamp au-dessus de 1.0, clamp en-dessous de 0.0) et 1 test d'intégration dans `benchmarks/ai/test_rl_env.py` (`test_collision_reward_scales_with_remaining_route_fraction`, vérifie via le vrai `reset()`/`step()` qu'un crash loin de la destination coûte plus qu'un crash juste à côté).
- Troisième volet du lot, indépendant des deux fixes ci-dessus : changement du seed PPO (`_PPO_DEFAULTS["seed"]` dans `rl_train.py`, 42 → 7) — diagnostic isolé déjà évoqué dans `runs/AUDIT.md` §2.7 (aucun seeding hors PPO avant Lot 1/v10) et repris explicitement dans `runs/2026-07-12_00-45_ppo_v11_150k/ANALYSIS.md` ("Pistes pour la suite" #2) : v10 (sans gSDE) avait collapsé plein-gauche, v11 (avec gSDE) plein-droite, sur le même seed=42 figé depuis v1 — ce volet teste si le collapse lui-même (ou seulement sa direction) dépend de ce seed précis. Seul le seed PPO change ; le seeding environnement (`scripts/run_rl_training.py` : `random.seed(42)`, `np.random.seed(42)`, `tm.set_random_device_seed(42)`) reste à 42, pour isoler cette seule variable entre v11 et v12. 2 tests mis à jour dans `benchmarks/ai/test_rl_train.py` pour vérifier `seed == 7`.
- 254 tests (`uv run pytest benchmarks/ -q`), tout vert (248 avant ce fix).

**Décisions** :
- **Tenir la dernière valeur valide plutôt qu'une valeur hors-plage dédiée** (l'autre piste évoquée dans `ANALYSIS.md`, ex. `lane_offset_norm = ±1.0` ou un flag séparé quand `direction == "NONE"`) : une valeur hors-plage artificielle aurait quand même fallu être apprise comme "signal spécial" par la policy, alors que la dernière position connue reste une estimation raisonnable à très court terme (la voiture ne téléporte pas) et ne casse pas la sémantique `[-1, 1]` = position latérale réelle dans la voie.
- Ce fix ne touche que la consommation de `lane_geometry()` côté `rl_env.py` — le module de Karim (signature, dict de retour, default `NO_LANE`) reste inchangé, cohérent avec le fait que ce module est hors périmètre pour moi.
- **Facteur de collision borné à 2x, pas plus** : même une voiture qui resterait totalement immobile hors-piste (`_P_STALL`=-0.20/step) sur un épisode complet de 1000 steps accumulerait environ -200, très au-dessus des -19 (`-9.5 × 2`) que peut coûter au maximum un crash désormais doublé — aucun risque de recréer, à l'envers, la pathologie de l'immobilisme rentable déjà corrigée avant le Lot 2.
- Facteur appliqué à `r_collision` en entier (base + composante vitesse d'impact), pas seulement à la base `_P_COLLISION_BASE` : les deux causes de coût d'un crash (l'occurrence elle-même, sa violence) restent proportionnellement liées à la distance abandonnée, pas de raison de séparer les deux.

**Benchmarks** : 254 tests, tout vert (248 avant ce fix).

**Prochaine étape** :
- Une fois le lot complet : lancer un run v12 (≥150k steps) et comparer à v9/v10/v11 via `analyze_run.py`, en particulier sur `center_offset` (ne devrait plus jamais être `0.0` en dehors des tout premiers steps post-reset), sur la corrélation reward/longueur d'épisode, et sur le taux de crash — pour vérifier si le fait de doubler `r_collision` en début de trajet a effectivement réduit l'attractivité du crash rapide, et si changer le seed PPO affecte le collapse ou simplement sa direction.
