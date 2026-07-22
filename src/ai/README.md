# `src/ai/` — IA centrale (décision)

> **Owner** : Frédéric Huang
> **Contrats (réel)** : `CarlaEnv` (`gym.Env`) consomme directement `PerceptionPipeline.perceive()` (Franck), `lane_fusion.estimate_with_drivable()` (détecteur de Karim + fallback zone roulable) et `Navigation.next_command()` (Victor) → `HighLevelCommand` ; produit une observation `Box(14,)` et consomme une action `Box(2,)` (steer, accel — throttle/brake dérivés du signe de accel). Les dataclasses `SceneState`/`ControlOutput` de [`ai_types.py`](../interfaces/ai_types.py) faisaient partie du design initial mais ne sont plus utilisées : PPO/Stable-Baselines3 impose un espace d'observation/action `numpy` plat, pas des objets structurés.

---

## Responsabilité

Module de **décision** : à partir de l'état courant de l'environnement et de l'intention de navigation, produire les contrôles à appliquer au véhicule (steering, accélération, freinage).

## Phases

**Phase 0 — CIL / PilotNet (retirée du repo)** : l'IA imitait l'autopilot CARLA (Conditional Imitation Learning) à partir des images RGB. Échec instructif — R² steer = −0.67, le modèle prédisait « tout droit » 92 % du temps (dataset à 92.7 % `|steer| ≤ 0.05`). Détail dans le [JOURNAL](JOURNAL.md) ; code retiré au nettoyage final, disponible dans l'historique git.

**Phase 1 — RL par renforcement (livrée)** : l'IA apprend **seule** via PPO (Stable-Baselines3). Elle n'imite plus un expert — elle explore CARLA, reçoit un reward à chaque step, et optimise sa policy sur des **observations structurées** issues des modules de perception plutôt que sur les pixels bruts. Modèle final : **v23 checkpoint 88k — 7/8 au benchmark, 0 collision, 0 vérité terrain** ([docs/LEADERBOARD.md](../../docs/LEADERBOARD.md), analyse complète dans [ANALYSE_FINALE.md](ANALYSE_FINALE.md)).

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

[lane_fusion.estimate_with_drivable() — Karim + fallback]   src/ai/inference/lane_fusion.py
         │   YOLOPv2 (segmentation zone roulable + lignes) + géométrie (lane_geometry.py)
         │   Quand les lignes renvoient NONE (≈94 % des frames sur Town02), l'offset
         │   vient du centroïde du masque zone roulable — signal frais 98 % du temps.
         ▼
         (direction, angle, offset, drivable_offset) ──► lane_offset_norm, is_on_road

[Navigation — Victor]
         ├─ plan(start, dest) → Route          ← au reset de chaque épisode
         ├─ next_command(pos, route) → HighLevelCommand  ← à chaque step
         └─ cap vers la route (lookahead 12 waypoints ≈ 24 m) → goal_bearing_norm

[Observation vector] — 14 scalaires normalisés  ∈ [-1, 1] ou [0, 1]
         ├─ [0] speed_norm              = speed_kmh / 90.0                    ∈ [0, 1]
         ├─ [1] cmd_left                = 1.0 si LEFT else 0.0
         ├─ [2] cmd_right               = 1.0 si RIGHT else 0.0
         ├─ [3] cmd_straight            = 1.0 si STRAIGHT else 0.0
         │       [0,0,0] = LANE_FOLLOW (pas d'intersection)
         ├─ [4] lane_offset_norm        = déviation latérale voie             ∈ [-1, 1]  (Karim)
         ├─ [5] is_on_road              = 1.0 si voie détectée else 0.0                  (Karim)
         ├─ [6] nearest_vehicle_norm    = min(dist_véhicule, 50m) / 50m       ∈ [0, 1]  (Franck)
         ├─ [7] red_light_distance_norm = min(dist_feu_rouge, 50m) / 50m      ∈ [0, 1]  (Franck)
         ├─ [8] speed_limit_norm        = limite de vitesse mémorisée / 90.0  ∈ [0, 1]  (Franck)
         ├─ [9] nearest_walker_norm     = min(dist_piéton, 50m) / 50m         ∈ [0, 1]  (Franck)
         ├─ [10] nearest_stop_yield_norm = min(dist_stop_ou_yield, 50m) / 50m ∈ [0, 1]
         ├─ [11] prev_steer_norm        = steer appliqué au step précédent    ∈ [-1, 1]
         ├─ [12] prev_accel_norm        = accel appliqué au step précédent    ∈ [-1, 1]
         └─ [13] goal_bearing_norm      = erreur de cap signée vers la route  ∈ [-1, 1]  (Victor)
                 (waypoint à ~24 m devant — direction générale, pas le braquage)

[PPO Policy] — Stable-Baselines3 MlpPolicy
         │   2 couches Dense 128, activation tanh, squash_output (tanh)
         ▼
[Action] — espace continu Box(2,)
         ├─ steer ∈ [-1, 1]
         └─ accel ∈ [-1, 1]   → throttle = max(accel, 0), brake = max(−accel, 0)
                                (jamais les deux pédales en même temps)

[Reward function — par step]   src/ai/rewards/reward_fn.py
         ├─ r_progress   = (γ·Φ(s') − Φ(s)) × 1.0, Φ(s) = −distance_to_destination(s) normalisée → shaping potential-based vers la destination, garantie théorique contre les raccourcis (remplace r_speed, qui ne récompensait que la vitesse brute) ; pas gaté sur is_on_road (r_offroad reste seul juge de la sortie de route)
         ├─ r_center     = (1 − |lane_offset_norm|) × 0.3 si is_on_road ET speed ≥ 1 km/h → encourage le centrage ; nul hors route ou à l'arrêt (gate anti-passivité v16 : rester immobile centré ne rapporte rien)
         ├─ r_alive      = +0.05                                            → survie (anti-crash passif)
         ├─ r_stall      = −0.20 × min(1 + n_stall_consécutifs/200, 2) si speed < 1 km/h → pénalise l'immobilisme, tarif ×2 après 10 s de parking ininterrompu
         ├─ r_offroad    = −0.5  si is_on_road=False                        → pénalité hors route (Karim)
         ├─ r_off_route  = −0.5  si > 15m de la route GPS                   → pénalise la déviation GPS
         ├─ r_following  = −(1 − headway_s/2.0) × 0.2 si headway < 2s       → distance de sécurité, règle des 2s (Franck)
         ├─ r_walker     = −(1 − dist_piéton/10m) × 0.3 si dist < 10m       → priorité sécurité piéton (Franck)
         ├─ r_speeding   = −((speed − limite − 5) / 90.0) × 0.3 si dépassement → pénalise l'excès de vitesse (Franck)
         ├─ r_red_light  = −2.0 si franchissement de feu rouge              → sanctionne le "grillage" de feu (Franck)
         ├─ r_stop_yield = −1.0 si franchissement de stop/yield             → sanctionne le "grillage" de panneau (Franck)
         ├─ r_collision  = (−5.0 − 0.20 × vitesse_impact_kmh) × (1.0 à 2.0 selon distance restante) + done=True   → épisode terminé, pénalité ∝ vitesse d'impact et à la distance qu'il restait à parcourir
         ├─ r_destination = +10.0 si destination atteinte (≥25m parcourus, <15m de la cible) → épisode terminé, succès
         ├─ r_safe       = +0.05 si aucun danger actif (following/walker/speeding tous OK)   → renforcement positif de la prudence
         └─ r_jerk       = −|steer_t − steer_t-1| × 0.1                     → pénalise le pilotage saccadé
```

### Espace d'observation — pourquoi des scalaires et pas des pixels

Le RL pur sur pixels (end-to-end) nécessite des dizaines de millions de steps et des semaines de compute. Les observations compactes convergent en quelques heures car :
- L'espace d'état est petit (14 floats) — même si la caméra tourne à 1280×720 (résolution imposée par le modèle YOLO de Franck), la policy PPO ne voit **jamais** l'image brute, seulement les scalaires extraits par les modules de perception
- Chaque feature est directement exploitable (le réseau n'a pas à apprendre à extraire la distance depuis les pixels)
- La variance de l'estimation de gradient (PPO) est beaucoup plus faible

### Perception réelle (Franck + Karim)

Les scalaires de perception viennent des vrais modèles d'IA, pas de capteurs GT CARLA :

- **Franck** — `PerceptionPipeline` (`src/perception/pipeline.py`) : YOLO11s (détection d'objets, 1280×720) + Depth Anything v2 (profondeur monoculaire, calibrée via `calibration.json`) fusionnés en une liste d'objets avec `distance_m`. Alimente `nearest_vehicle_norm`, `red_light_distance_norm`, `speed_limit_norm` (mémorisé entre frames car le panneau n'est pas toujours visible dans le champ de vision), `nearest_walker_norm` et `nearest_stop_yield_norm`.
- **Karim** — `lane_perception.estimate()` (`src/lane_detection/lane_perception.py`) : YOLOPv2 (segmentation zone roulable + lignes, resize interne 640×480) + géométrie (`lane_geometry.py`, clustering des lignes + suivi + polyfit) → `(direction, angle, offset)`. Consommé via le wrapper [`lane_fusion.estimate_with_drivable()`](inference/lane_fusion.py) : quand les lignes renvoient `NONE` (≈94 % des frames sur Town02, marquages rares), l'offset latéral est recalculé depuis le **masque zone roulable** (centroïde de la bande basse de l'image) — signal disponible 98 % du temps, sans modifier le code de Karim. Alimente `lane_offset_norm` et `is_on_road` (l'angle n'est pas fourni au modèle).

Le passage des capteurs GT CARLA du début de projet aux vrais modèles (v21) n'a demandé aucun changement dans le code de l'IA centrale — tout passait déjà par les mêmes signatures d'appel.

### Intégration navigation Victor

- Au `reset()` de chaque épisode : `nav.plan(spawn_point, destination)` → Route
- À chaque step : `nav.next_command(vehicle_pos, route)` → `HighLevelCommand`
- La commande est encodée en one-hot dans l'observation (3 floats : left/right/straight)
- `LANE_FOLLOW` (hors intersection) = `[0, 0, 0]`
- En complément, `obs[13]` porte un **cap continu vers la route** (`--goal-bearing`, waypoint à `--goal-bearing-lookahead 12` ≈ 24 m devant) : une direction générale du but, volontairement trop lointaine pour servir de consigne de braquage

**Route replanifiée à chaque reset, sans exception** : `CarlaEnv.reset()` replanifie
systématiquement `self.route` depuis la position réelle de l'ego après téléportation, que le
spawn soit explicite (eval/démo) ou aléatoire (entraînement normal). Le graphe du réseau
routier est mis en cache par instance `Navigation` (statique pour une map donnée) pour que
cette replanification systématique ne coûte pas un recalcul complet à chaque épisode.

**Benchmark — GPS route par scénario (`dest_spawn_idx`)** : pour les scénarios de jonction,
la route est replanifiée vers un spawn cible spécifique après le reset, ce qui garantit que
la nav donne la bonne commande directionnelle (LEFT / RIGHT / STRAIGHT). Les `dest_spawn_idx`
ont été identifiés via l'API waypoint CARLA (outil retiré au nettoyage final) et sont figés
dans [inference/benchmark.py](inference/benchmark.py).

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
├── training/
│   ├── rl_env.py              ← CarlaEnv (gym.Env) : spaces, reset, step, obs 14D, curriculum
│   ├── rl_train.py            ← make_model() + train() PPO SB3
│   └── run_manager.py         ← dossier de run horodaté, CSV, courbe reward
├── inference/
│   ├── rl_demo.py             ← run_episode, record_episode, eval_model (benchmark complet)
│   ├── benchmark.py           ← BENCHMARK_SCENARIOS (13 scénarios fixes) + success_fn
│   └── lane_fusion.py         ← Fallback offset « zone roulable » (fix décisif v23)
└── rewards/
    └── reward_fn.py           ← compute_reward() — fonction pure, sans CARLA

scripts/
├── run_rl_training.py         ← Pipeline complète : training + auto-éval checkpoints + vidéos
├── carla_helpers.py           ← Helpers CARLA partagés par la démo intégrée (main.py)
└── export_model_to_onnx.py    ← Export du modèle vers ONNX (serveur web/)
```

---

## Lancer le training RL

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
| `best_model.zip` | meilleur checkpoint réel, sélectionné par `pick_best_checkpoint()` sur le benchmark complet (plus d'EvalCallback SB3 — voir plus bas) |
| `model_final.zip` | poids à la fin du training |
| `training_log.monitor.csv` | reward / longueur par épisode + moyenne des 15 composantes de reward (Monitor SB3, `info_keywords`) |
| `reward_curve.png` | courbe reward brute + moyenne mobile |
| `demo.mp4` | vidéo d'inférence avec HUD (best model, spawn fixe) — ouvre sur 3s de carte du trajet prévu (départ "A" / arrivée "B") |
| `evals/checkpoint_XXXXk.mp4` | vidéo 13 scénarios par checkpoint |
| `evals/results.json` | métriques complètes de tous les checkpoints périodiques évalués |

L'évaluation 13 scénarios est **intégrée au training** : chaque checkpoint (toutes les 11k
steps) est évalué en fin de run, vidéo + JSON à l'appui. C'est sur cette base qu'est
sélectionné `best_model.zip` — jamais sur le seul `model_final` (systématiquement
sur-entraîné, constaté sur 4 runs).

### Structure du JSON de résultats (`evals/results.json`)

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

## Benchmark 13 scénarios — critères de succès Phase 1

Scénarios définis dans [inference/benchmark.py](inference/benchmark.py) (Town02) :

| Scénario | spawn_idx | dest_spawn_idx | Critère de succès |
|---|---|---|---|
| `straight` | 13 | — | pas de collision + ≥ 25 m parcourus |
| `curve_left` | 35 | — | pas de collision + ≥ 25 m parcourus |
| `curve_right` | 31 | — | pas de collision + ≥ 25 m parcourus |
| `turn_left` | 79 | 63 | atteindre la destination (rayon 15 m) |
| `turn_right` | 81 | 49 | atteindre la destination (rayon 15 m) |
| `junction_straight` | 47 | 49 | atteindre la destination (rayon 15 m) |
| `npc_follow` | 13 | — | pas de collision + ≥ 25 m parcourus |
| `npc_crossing` | 79 | — | pas de collision + ≥ 25 m parcourus |

Phase 2 (enregistré sans critère) : `red_light`, `speed_zone`, `pedestrian`, `emergency_stop`, `lane_change`.

> La suite de tests offline qui accompagnait le module (~270 tests, mocks sans CARLA) a été
> retirée au nettoyage final du repo — historique git.

---

## Liens

- [Schulman et al. 2017, PPO](https://arxiv.org/abs/1707.06347) — algorithme Phase 1
- [ROACH (Zhang et al. 2021)](https://arxiv.org/abs/2108.08265) — référence RL sur CARLA avec reward structuré
- [Stable-Baselines3 docs](https://stable-baselines3.readthedocs.io/) — implémentation PPO utilisée
- [Gymnasium docs](https://gymnasium.farama.org/) — interface `gym.Env` pour `CarlaEnv`
- [Bojarski et al. 2016, PilotNet](https://arxiv.org/abs/1604.07316) — Phase 0 (retirée)
- [Codevilla et al. 2018, CIL](https://arxiv.org/abs/1710.02410) — Phase 0 (retirée)
