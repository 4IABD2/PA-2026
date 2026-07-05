# `src/ai/` — IA centrale (décision)

> **Owner** : Frédéric Huang
> **Contrats (Phase 1 réel)** : `CarlaEnv` (`gym.Env`) consomme directement `PerceptionPipeline.perceive()` (Franck), `lane_perception.estimate()` (Karim) et `Navigation.next_command()` (Victor) → `HighLevelCommand` ; produit une observation `Box(12,)` et consomme une action `Box(3,)` (steer, throttle, brake). Les dataclasses `SceneState`/`ControlOutput` de [`ai_types.py`](../interfaces/ai_types.py) faisaient partie du design initial mais ne sont plus utilisées : PPO/Stable-Baselines3 impose un espace d'observation/action `numpy` plat, pas des objets structurés.

---

## Responsabilité

Module de **décision** : à partir de l'état courant de l'environnement et de l'intention de navigation, produire les contrôles à appliquer au véhicule (steering, accélération, freinage).

## Phases

### Phase 0 — CIL / PilotNet (archivé)

> Code préfixé `v1_*`. Ne pas modifier. Gardé comme référence et comme introduction à la stack CARLA.

L'IA imitait l'autopilot CARLA (Conditional Imitation Learning) : le modèle apprenait à copier les contrôles `(steer, throttle, brake)` de l'expert à partir des images RGB. Résultats Phase 0 documentés dans le [JOURNAL](JOURNAL.md).

**Limite principale** : le head `steer` avait un R² négatif (-0.67) — le modèle prédisait "tout droit" 92 % du temps car le dataset était à 92.7 % `|steer| ≤ 0.05`. L'imitation ne donne pas assez de signal sur les virages.

### Phase 1 — RL par renforcement (en cours)

L'IA apprend **seule** via PPO (Stable-Baselines3). Elle n'imite plus un expert — elle explore CARLA, reçoit un reward à chaque step, et optimise sa policy. Elle a accès à des **observations structurées** issues des modules de perception plutôt qu'aux pixels bruts.

---

## Architecture Phase 1

### Vue d'ensemble

```
CARLA World (sync mode, 20 FPS)
         │
         ├─ RGB camera (1280×720 — résolution d'entraînement YOLO de Franck)
         └─ Collision sensor ──────────► collision (bool, épisode terminé)

[PerceptionPipeline — Franck]   src/perception/pipeline.py
         │   YOLO11s (détection, 11 classes) + Depth Anything v2 (profondeur monoculaire calibrée)
         ▼
         liste d'objets détectés + distance_m ──► nearest_vehicle_norm, red_light_distance_norm,
         speed_limit_norm, nearest_walker_norm, nearest_stop_yield_norm

[lane_perception.estimate() — Karim]   src/lane_detection/lane_perception.py
         │   YOLOPv2 (segmentation zone roulable + lignes) + géométrie (lane_geometry.py)
         ▼
         (direction, angle, offset) ──► lane_angle_norm, lane_offset_norm, is_on_road

[Navigation — Victor]
         ├─ plan(start, dest) → Route          ← au reset de chaque épisode
         └─ next_command(pos, route) → HighLevelCommand  ← à chaque step

[Observation vector] — 12 scalaires normalisés  ∈ [-1, 1] ou [0, 1]
         ├─ [0] speed_norm              = speed_kmh / 90.0                    ∈ [0, 1]
         ├─ [1] cmd_left                = 1.0 si LEFT else 0.0
         ├─ [2] cmd_right               = 1.0 si RIGHT else 0.0
         ├─ [3] cmd_straight            = 1.0 si STRAIGHT else 0.0
         │       [0,0,0] = LANE_FOLLOW (pas d'intersection)
         ├─ [4] lane_angle_norm         = angle de cap vers la voie / 90.0    ∈ [-1, 1]  (Karim)
         ├─ [5] lane_offset_norm        = déviation latérale voie             ∈ [-1, 1]  (Karim)
         ├─ [6] is_on_road              = 1.0 si voie détectée else 0.0                  (Karim)
         ├─ [7] nearest_vehicle_norm    = min(dist_véhicule, 50m) / 50m       ∈ [0, 1]  (Franck)
         ├─ [8] red_light_distance_norm = min(dist_feu_rouge, 50m) / 50m      ∈ [0, 1]  (Franck)
         ├─ [9] speed_limit_norm        = limite de vitesse mémorisée / 90.0  ∈ [0, 1]  (Franck)
         ├─ [10] nearest_walker_norm    = min(dist_piéton, 50m) / 50m         ∈ [0, 1]  (Franck)
         └─ [11] nearest_stop_yield_norm = min(dist_stop_ou_yield, 50m) / 50m ∈ [0, 1]  (Franck)

[PPO Policy] — Stable-Baselines3 MlpPolicy
         │   2 couches Dense 128, activation tanh
         ▼
[Action] — espace continu Box(3,)
         ├─ steer    ∈ [-1, 1]
         ├─ throttle ∈ [0, 1]
         └─ brake    ∈ [0, 1]

[Reward function — par step]   src/ai/rewards/reward_fn.py
         ├─ r_speed      = (progress_speed_kmh / 90.0) × 0.3                → encourage la progression le long de la route (pas la vitesse brute)
         ├─ r_center     = (1 − |lane_offset_norm|) × 0.3                   → encourage le centrage
         ├─ r_alive      = +0.01                                            → survie (anti-crash passif)
         ├─ r_stall      = −0.20 si speed < 1 km/h                          → pénalise l'immobilisme
         ├─ r_offroad    = −0.25 si is_on_road=False                        → pénalité hors route (Karim)
         ├─ r_off_route  = −0.5  si > 15m de la route GPS                   → pénalise la déviation GPS
         ├─ r_following  = −(1 − headway_s/2.0) × 0.2 si headway < 2s       → distance de sécurité, règle des 2s (Franck)
         ├─ r_walker     = −(1 − dist_piéton/10m) × 0.3 si dist < 10m       → priorité sécurité piéton (Franck)
         ├─ r_speeding   = −((speed − limite − 5) / 90.0) × 0.3 si dépassement → pénalise l'excès de vitesse (Franck)
         ├─ r_red_light  = −2.0 si franchissement de feu rouge              → sanctionne le "grillage" de feu (Franck)
         ├─ r_stop_yield = −1.0 si franchissement de stop/yield             → sanctionne le "grillage" de panneau (Franck)
         ├─ r_collision  = (−5.0 − 0.20 × vitesse_impact_kmh) + done=True   → épisode terminé, pénalité ∝ vitesse d'impact
         ├─ r_destination = +10.0 si destination atteinte (≥25m parcourus, <15m de la cible) → épisode terminé, succès
         ├─ r_safe       = +0.05 si aucun danger actif (following/walker/speeding tous OK)   → renforcement positif de la prudence
         └─ r_jerk       = −|steer_t − steer_t-1| × 0.1                     → pénalise le pilotage saccadé
```

### Espace d'observation — pourquoi des scalaires et pas des pixels

Le RL pur sur pixels (end-to-end) nécessite des dizaines de millions de steps et des semaines de compute. Les observations compactes convergent en quelques heures car :
- L'espace d'état est petit (12 floats) — même si la caméra tourne à 1280×720 (résolution imposée par le modèle YOLO de Franck), la policy PPO ne voit **jamais** l'image brute, seulement les scalaires extraits par les modules de perception
- Chaque feature est directement exploitable (le réseau n'a pas à apprendre à extraire la distance depuis les pixels)
- La variance de l'estimation de gradient (PPO) est beaucoup plus faible

### Perception réelle (Franck + Karim)

Les scalaires de perception viennent des vrais modèles d'IA, pas de capteurs GT CARLA :

- **Franck** — `PerceptionPipeline` (`src/perception/pipeline.py`) : YOLO11s (détection d'objets, 1280×720) + Depth Anything v2 (profondeur monoculaire, calibrée via `calibration.json`) fusionnés en une liste d'objets avec `distance_m`. Alimente `nearest_vehicle_norm`, `red_light_distance_norm`, `speed_limit_norm` (mémorisé entre frames car le panneau n'est pas toujours visible dans le champ de vision), `nearest_walker_norm` et `nearest_stop_yield_norm`.
- **Karim** — `lane_perception.estimate()` (`src/lane_detection/lane_perception.py`) : YOLOPv2 (segmentation zone roulable + lignes, resize interne 640×480) + géométrie (`lane_geometry.py`, clustering des lignes + suivi + polyfit) → `(direction, angle, offset)`. Alimente `lane_angle_norm` (cap vers la voie), `lane_offset_norm` (position latérale), `is_on_road` (`direction != "NONE"`).

Les anciens stubs GT CARLA (`src/interfaces/stubs.py`, `CarlaGTDepthEstimator`/`CarlaGTLaneDetector`) ne sont plus utilisés dans le pipeline d'entraînement — le code de l'IA centrale n'a pas eu besoin de changer au moment du swap, tout passait déjà par les protocoles de `src/interfaces/`.

### Intégration navigation Victor

- Au `reset()` de chaque épisode : `nav.plan(spawn_point, destination)` → Route
- À chaque step : `nav.next_command(vehicle_pos, route)` → `HighLevelCommand`
- La commande est encodée en one-hot dans l'observation (3 floats : left/right/straight)
- `LANE_FOLLOW` (hors intersection) = `[0, 0, 0]`

**Route replanifiée à chaque reset, sans exception** : `CarlaEnv.reset()` replanifie
systématiquement `self.route` depuis la position réelle de l'ego après téléportation, que le
spawn soit explicite (eval/démo) ou aléatoire (entraînement normal). Le graphe du réseau
routier est mis en cache par instance `Navigation` (statique pour une map donnée) pour que
cette replanification systématique ne coûte pas un recalcul complet à chaque épisode.

**Benchmark — GPS route par scénario (`dest_spawn_idx`)** : pour les scénarios de jonction,
la route est replanifiée vers un spawn cible spécifique après le reset, ce qui garantit que
la nav donne la bonne commande directionnelle (LEFT / RIGHT / STRAIGHT). Les `dest_spawn_idx`
ont été identifiés via `scripts/find_dest_spawns.py` (API waypoint CARLA pure, sans nav.plan).

**Pénalité off-route** : `CarlaEnv._is_off_route()` détecte si l'ego est à plus de 15m du
waypoint de route le plus proche via une fenêtre glissante (O(1) amorti). La pénalité
`−0.5/step` est appliquée dans `compute_reward(off_route=True)`.

### Trafic NPC (véhicules + piétons)

Depuis `scripts/run_rl_training.py`, chaque run de training fait aussi spawner du trafic
autour de l'ego : des véhicules NPC en pilote automatique (Traffic Manager CARLA) et des
piétons NPC pilotés par un `controller.ai.walker` qui marchent vers des destinations
aléatoires sur la navmesh piétonne. Les quantités sont configurables via `--npcs` (défaut
18) et `--pedestrians` (défaut 6).

Sans ce trafic, la map serait vide et `nearest_vehicle_norm` / `nearest_walker_norm`
resteraient quasiment toujours à 1.0 (rien à détecter) : les nouveaux termes de reward
`r_following` et `r_walker` n'auraient jamais l'occasion de s'activer pendant
l'entraînement. `r_speeding`, `r_red_light` et `r_stop_yield` réagissent eux à
l'infrastructure statique de la map (panneaux, feux) et ne dépendent donc pas du trafic NPC.

### Spawn sûr face au trafic

Pendant l'entraînement (spawn aléatoire, pas les scénarios eval/démo à `spawn_idx` fixe),
`CarlaEnv._teleport_to_spawn()` tire jusqu'à 10 points de spawn candidats et accepte le
premier situé à au moins 10m de tout véhicule ou piéton NPC actuellement dans le monde
(`world.get_actors()`, hors ego). Si aucun des 10 essais ne passe le seuil, le candidat avec
la plus grande distance observée est conservé — jamais pire qu'un tirage aléatoire simple.
Évite de téléporter l'ego au contact d'un NPC en mouvement, ce qui produirait une collision
artificielle sans rapport avec la conduite de la policy.

---

## Structure des fichiers

```
src/ai/
├── phase0/                    ← Phase 0 archivée (CIL/PilotNet), ne pas modifier
│   ├── config.py
│   ├── models/
│   ├── training/
│   └── inference/
├── training/
│   ├── rl_env.py              ← CarlaEnv (gym.Env) : spaces, reset, step, _is_off_route
│   ├── rl_train.py            ← make_model() + train() PPO SB3
│   └── run_manager.py         ← dossier de run horodaté, CSV, courbe reward
├── inference/
│   ├── rl_demo.py             ← run_episode, record_episode, eval_model (benchmark complet)
│   └── benchmark.py           ← BENCHMARK_SCENARIOS (13 scénarios fixes) + success_fn
└── rewards/
    └── reward_fn.py           ← compute_reward() — fonction pure, sans CARLA

scripts/
├── run_rl_training.py         ← Pipeline complète : training + eval checkpoints + vidéos
├── run_eval.py                ← Évaluation standalone d'un modèle → vidéo + JSON
├── analyze_run.py             ← Analyse automatisée d'une run (courbe binée, benchmark, totaux)
├── explore_spawns.py          ← Explore et classe les spawn points par catégorie
└── find_dest_spawns.py        ← Trouve les dest_spawn_idx par direction de carrefour
```

---

## Phase 0 — Documentation technique (archivée)

### Training CIL (offline)

```bash
uv pip install 'tensorflow[and-cuda]==2.21.0'
export LD_LIBRARY_PATH="$(ls -d .venv/lib/python3.10/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
CUDA_VISIBLE_DEVICES=1 uv run -m src.ai.training.train \
  --runs data/runs/<date>_town01_clearnoon \
  --output checkpoints/pilotnet_v1/
```

### Démo CIL (archivée)

```bash
uv run -m src.ai.inference.carla_demo \
  --weights checkpoints/pilotnet_v1/best.keras \
  --town Town01 --weather ClearNoon --duration 120
```

**Résultats Phase 0** (2026-05-10) :
- 9 respawns sur 120s (1 crash / ~13s)
- R² steer = -0.67 (échec — collapse vers 0)
- R² throttle = +0.33, R² brake = +0.64
- val_loss = 0.0735 sur 1350 frames

---

## Phase 1 — Lancer le training RL

```bash
# Training complet (génère aussi les evals et les vidéos automatiquement) :
uv run python3 scripts/run_rl_training.py \
  --timesteps 300000 \
  --tag ppo_v1 \
  --host <ip-carla>

# Smoke test (vérifie que le pipeline tourne, ~2 min) :
uv run python3 scripts/run_rl_training.py --timesteps 1000 --tag smoke --host <ip-carla>
```

Les artefacts sont générés dans `runs/YYYY-MM-DD_HH-MM_<tag>/` :

| Fichier / Dossier | Contenu |
|---|---|
| `params.json` | hyperparamètres + config de la run |
| `run.log` | copie intégrale de stdout+stderr sur toute la durée du script — survit à un crash |
| `model_best.zip` | meilleur checkpoint (EvalCallback SB3) |
| `model_final.zip` | poids à la fin du training |
| `training_log.monitor.csv` | reward / longueur par épisode + moyenne des 15 composantes de reward (Monitor SB3, `info_keywords`) |
| `reward_curve.png` | courbe reward brute + moyenne mobile |
| `demo.mp4` | vidéo d'inférence avec HUD (best model, spawn fixe) — ouvre sur 3s de carte du trajet prévu (départ "A" / arrivée "B") |
| `evals/checkpoint_XXXXk.mp4` | vidéo 13 scénarios par checkpoint |
| `evals/best_model.mp4` | vidéo 13 scénarios du best model |
| `evals/results.json` | métriques complètes (tous checkpoints + best model) |
| `analysis_data.json` | généré par `scripts/analyze_run.py` (voir section suivante) |

## Phase 1 — Analyser une run

```bash
uv run python3 scripts/analyze_run.py runs/<dossier_de_run>
```

Lit `training_log.monitor.csv` et `evals/results.json`, calcule une courbe d'apprentissage
binée par tranches de 10k steps (reward moyen, % d'épisodes positifs, longueur moyenne, %
de crashs courts, moyenne de chaque composante de reward présente dans le CSV), un résumé
benchmark par checkpoint (succès Phase 1, vitesse/hors-route/accélérateur/frein moyens) et
des totaux (épisodes, steps, taux de crash global). Écrit `analysis_data.json` dans le
dossier de run et affiche un résumé en tables directement réutilisable dans une `ANALYSIS.md`.

Fonctionne aussi sur une run antérieure au branchement des 15 colonnes de composantes de
reward dans le CSV — les moyennes par composante sont simplement absentes du résultat plutôt
que de faire planter le script. Données brutes uniquement : pas de détection automatique
d'anomalie ni de comparaison inter-run, le diagnostic reste manuel.

## Phase 1 — Évaluation d'un modèle existant

```bash
# Évalue un modèle sur les 13 scénarios → vidéo annotée + JSON métriques
uv run python3 scripts/run_eval.py \
  --model runs/<dir>/best_model.zip \
  --host <ip-carla>
# Sortie : eval_best_model.mp4 + eval_best_model.json dans le même dossier que le modèle
```

### Structure du JSON de résultats

Chaque scénario dans `results.json` contient :

```
{
  "straight": {
    # Résultat
    "success": true,            # bool (Phase 1) ou null (Phase 2)
    "terminated": false,        # collision détectée
    "collision_step": null,     # step de la collision (ou null)
    "reached_dest": false,      # a atteint dest_spawn_idx (scénarios GPS)
    "steps": 287,
    "max_dist_from_start": 54.3,
    "total_reward": 43.2,

    # Stats vitesse (km/h)
    "speed": {"mean": 28.4, "max": 51.2, "min": 0.0, "std": 12.1, "pct_moving": 0.94},

    # Stats centrage voie
    "center_offset": {"mean_abs": 0.08, "max_abs": 0.31, "std": 0.06, "pct_centered": 0.87},

    # Stats orientation
    "heading": {"mean_abs_deg": 4.2, "max_abs_deg": 18.7, "std_deg": 3.1},

    # Stats obstacle (metres)
    "obstacle": {"mean_m": 38.1, "min_m": 7.4},

    # Stats actions
    "steer":    {"mean_abs": 0.04, "mean": -0.003, "std": 0.06},  # mean signé → biais L/R
    "throttle": {"mean": 0.41, "std": 0.12},                       # std élevé → oscillation
    "brake":    {"mean": 0.03, "max": 0.88, "pct_braking": 0.08},

    # Commandes nav reçues (step counts)
    "nav_commands": {"LANE_FOLLOW": 250, "LEFT": 0, "RIGHT": 0, "STRAIGHT": 37},

    # Off-route
    "off_route_steps": 2,
    "off_route_pct": 0.007,

    # Séries temporelles (une valeur par step, pour tracer des courbes)
    "trajectory":      [[x, y, yaw], ...],   # position GPS
    "rewards_series":  [...],
    "speed_series":    [...],
    "center_series":   [...],
    "steer_series":    [...],
    "throttle_series": [...],
    "brake_series":    [...],
    "heading_series":  [...],                 # degrés
    "obstacle_series": [...],                 # mètres
    "nav_series":      [...],                 # 0=FOLLOW 1=LEFT 2=RIGHT 3=STRAIGHT
    "off_route_series": [...],               # 0/1 par step
    "dist_series":     [...]                 # distance au spawn (m)
  }
}
```

## Phase 1 — Utilitaires spawn

```bash
# Explorer et classifier les spawn points de la map (utile pour choisir les spawn_idx)
uv run python3 scripts/explore_spawns.py --host <ip-carla>

# Identifier les dest_spawn_idx pour les scénarios de jonction
# (utilise l'API waypoint CARLA directement — aucun appel à nav.plan())
uv run python3 scripts/find_dest_spawns.py --host <ip-carla>
# → affiche les candidats left/right/straight + la valeur à coller dans benchmark.py
```

---

## Validation et benchmarks

```bash
uv run pytest benchmarks/ai/ -v
```

| Fichier | Couverture |
|---|---|
| `smoke.py` | reward_fn — triplet `(reward, terminated, components)`, invariant somme des 15 composantes, les 5 termes de sécurité `r_following`/`r_walker`/`r_speeding`/`r_red_light`/`r_stop_yield` + `r_destination`/`r_safe`/`r_jerk` (33 tests) + stubs GT — `src/interfaces/stubs.py`, non utilisés en prod depuis le branchement Franck/Karim (7 tests) |
| `test_rl_env.py` | CarlaEnv — spaces, reset (dont replanification de route systématique + spawn sûr face aux NPC), step, observation (12 scalaires), reward (dont vitesse orientée-route, destination atteinte, jerk), accumulateur de composantes par épisode, violations feu rouge / stop-yield, render, off_route (59 tests) |
| `test_rl_train.py` | make_model (dont config PPO : réseau, entropy, seed, LR schedule), train (9 tests) |
| `test_rl_demo.py` | run_episode, _add_hud, record_episode (dont overlay carte du trajet, spawn_idx pour la reproductibilité démo), load_model (18 tests) |
| `test_run_manager.py` | make_run_dir, save_params, plot_reward_curve (9 tests) |
| `test_analyze_run.py` | binning de la courbe d'apprentissage, agrégation benchmark, compatibilité avec un CSV sans les colonnes de composantes de reward (10 tests) |

Tous les tests tournent **sans CARLA** (Mocks). Les tests Phase 0 sont dans `benchmarks/ai/phase0/`.
Total Phase 1 : 145 tests (`uv run pytest benchmarks/ai/ -v --ignore=benchmarks/ai/phase0`).

### Benchmark 13 scénarios — critères de succès Phase 1

| Scénario | spawn_idx | dest_spawn_idx | Critère de succès |
|---|---|---|---|
| `straight` | 6 | — | pas de collision + ≥ 25m parcourus + offset moyen < 0.3 |
| `curve_left` | 8 | — | pas de collision + ≥ 25m parcourus |
| `curve_right` | 32 | — | pas de collision + ≥ 25m parcourus |
| `turn_left` | 0 | 140 | atteindre dest (rayon 15m) sans collision |
| `turn_right` | 70 | 68 | atteindre dest (rayon 15m) sans collision |
| `junction_straight` | 31 | 119 | atteindre dest (rayon 15m) sans collision |
| `npc_follow` | 6 | — | pas de collision + ≥ 25m parcourus |
| `npc_crossing` | 0 | — | pas de collision + ≥ 25m parcourus |

Phase 2 (enregistré sans critère) : `red_light`, `speed_zone`, `pedestrian`, `emergency_stop`, `lane_change`.

---

## Liens

- [Schulman et al. 2017, PPO](https://arxiv.org/abs/1707.06347) — algorithme Phase 1
- [ROACH (Zhang et al. 2021)](https://arxiv.org/abs/2108.08265) — référence RL sur CARLA avec reward structuré
- [Stable-Baselines3 docs](https://stable-baselines3.readthedocs.io/) — implémentation PPO utilisée
- [Gymnasium docs](https://gymnasium.farama.org/) — interface `gym.Env` pour `CarlaEnv`
- [Bojarski et al. 2016, PilotNet](https://arxiv.org/abs/1604.07316) — Phase 0 (archivé)
- [Codevilla et al. 2018, CIL](https://arxiv.org/abs/1710.02410) — Phase 0 (archivé)
