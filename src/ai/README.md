# `src/ai/` — IA centrale (décision)

> **Owner** : Frédéric Huang
> **Contrats** : produit [`ControlOutput`](../interfaces/ai_types.py) à partir de [`SceneState`](../interfaces/ai_types.py) + [`HighLevelCommand`](../interfaces/navigation_types.py)

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
         ├─ RGB camera
         ├─ GT depth sensor ──────────► nearest_obstacle_m (float)
         ├─ GT semantic segmentation ──► is_on_road (bool)
         └─ Collision sensor ──────────► collision (bool, épisode terminé)

[Navigation — Victor]
         ├─ plan(start, dest) → Route          ← au départ de chaque épisode
         └─ next_command(pos, route) → HighLevelCommand  ← toutes les ~2s

[Observation vector] — 7 scalaires
         ├─ speed_norm          = speed_kmh / 90.0               ∈ [0, 1]
         ├─ cmd_left            = 1.0 si LEFT else 0.0
         ├─ cmd_right           = 1.0 si RIGHT else 0.0
         ├─ cmd_straight        = 1.0 si STRAIGHT else 0.0
         ├─ center_offset       = déviation voie normalisée       ∈ [-1, 1]
         ├─ nearest_obstacle_m  = min(dist, 50m) / 50.0          ∈ [0, 1]
         └─ heading_error       = delta_yaw vers prochain wp / 180.0 ∈ [-1, 1]

[PPO Policy] — Stable-Baselines3 MlpPolicy
         │   2 couches Dense 128, activation tanh
         ▼
[Action] — espace continu Box(3,)
         ├─ steer    ∈ [-1, 1]
         ├─ throttle ∈ [0, 1]
         └─ brake    ∈ [0, 1]

[Reward function — par step]
         ├─ r_speed    = (speed_kmh / MAX_SPEED) × 0.5       → avance
         ├─ r_center   = (1 − |center_offset|) × 0.3         → reste centré
         ├─ r_alive    = +0.01                                → survie
         ├─ r_offroad  = −0.5 si hors route                  → pénalité
         └─ r_collision = −1.0 + done=True                   → épisode terminé
```

### Espace d'observation — pourquoi des scalaires et pas des pixels

Le RL pur sur pixels (end-to-end) nécessite des dizaines de millions de steps et des semaines de compute. Les observations compactes convergent en quelques heures car :
- L'espace d'état est petit (7 floats vs 200×88×3 pixels)
- Chaque feature est directement exploitable (le réseau n'a pas à apprendre à extraire la distance depuis les pixels)
- La variance de l'estimation de gradient (PPO) est beaucoup plus faible

### Stratégie GT → vrais modèles

Pendant l'entraînement, les métriques de perception viennent des capteurs GT CARLA (via `src/interfaces/stubs.py`). Une fois les modèles de Franck (depth, YOLO) et Karim (lanes) prêts, on substitue les stubs — le code de l'IA centrale **ne change pas** car tout passe par les protocoles de `src/interfaces/`.

### Intégration navigation Victor

- Au `reset()` de chaque épisode : `nav.plan(spawn_point, destination)` → Route
- À chaque step ou toutes les N secondes : `nav.next_command(vehicle_pos, route)` → `HighLevelCommand`
- La commande est encodée en one-hot dans le vecteur d'observation (3 floats : left/right/straight)
- `LANE_FOLLOW` (pas d'intersection à venir) = `[0, 0, 0]`

---

## Structure des fichiers

```
src/ai/
├── config.py                  ← constantes Phase 0 + Phase 1 (MAX_SPEED, reward weights, etc.)
├── models/
│   └── v1_pilotnet_speed.py   ← Phase 0 — archivé, ne pas toucher
├── training/
│   ├── data_loader.py         ← Phase 0 — archivé
│   ├── train.py               ← Phase 0 — archivé
│   ├── rl_env.py              ← Phase 1 — CarlaEnv (gym.Env) à implémenter
│   └── rl_train.py            ← Phase 1 — PPO training loop SB3 à implémenter
├── inference/
│   ├── carla_demo.py          ← Phase 0 — archivé
│   └── rl_demo.py             ← Phase 1 — démo RL à implémenter
└── rewards/
    └── reward_fn.py           ← Phase 1 — reward function à implémenter
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

## Phase 1 — Lancer le training RL (à venir)

```bash
# Une fois rl_env.py et rl_train.py implémentés :
uv run -m src.ai.training.rl_train \
  --town Town01 --weather ClearNoon \
  --timesteps 500000 \
  --output checkpoints/ppo_v1/
```

---

## Validation et benchmarks

Dans [benchmarks/ai/](../../benchmarks/ai/) :

- **`smoke.py`** — 4 tests pytest Phase 0 (shapes PilotNet, trainability, data loader, split determinism). Lancer : `uv run pytest benchmarks/ai/smoke.py -v`
- **Benchmark Phase 1** (à définir) : reward moyen par épisode, distance parcourue sans collision, taux de succès de suivi de route sur N épisodes.

---

## Liens

- [Schulman et al. 2017, PPO](https://arxiv.org/abs/1707.06347) — algorithme Phase 1
- [ROACH (Zhang et al. 2021)](https://arxiv.org/abs/2108.08265) — référence RL sur CARLA avec reward structuré
- [Stable-Baselines3 docs](https://stable-baselines3.readthedocs.io/) — implémentation PPO utilisée
- [Gymnasium docs](https://gymnasium.farama.org/) — interface `gym.Env` pour `CarlaEnv`
- [Bojarski et al. 2016, PilotNet](https://arxiv.org/abs/1604.07316) — Phase 0 (archivé)
- [Codevilla et al. 2018, CIL](https://arxiv.org/abs/1710.02410) — Phase 0 (archivé)
