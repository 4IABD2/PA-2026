# `src/ai/` — IA centrale (décision)

> **Owner** : Frédéric Huang
> **Contrats** : produit [`ControlOutput`](../interfaces/ai_types.py) à partir de [`SceneState`](../interfaces/ai_types.py) + [`HighLevelCommand`](../interfaces/navigation_types.py)

---

## Responsabilité

Module de **décision** : à partir de l'état courant de l'environnement et de l'intention de navigation, produire les contrôles à appliquer au véhicule (steering, accélération, freinage).

Approche retenue : **RL hybride en deux phases** :

1. **Phase 1 — CIL pre-training** : un modèle de réseau de neurones apprend à imiter un expert (l'autopilot CARLA pendant la collecte) en mappant `(image | scene_state, commande_HN, vitesse) → (steer, throttle, brake)`. Architecture initiale : PilotNet baseline. Permet d'avoir une démo fonctionnelle rapidement et de bootstrap le vrai entraînement RL.
2. **Phase 2 — RL fine-tuning** : le modèle pré-entraîné est ensuite affiné en boucle avec CARLA via un algorithme RL (PPO ou SAC, à confirmer) pour optimiser une fonction de récompense (collisions, sortie de route, distance parcourue, respect des feux). Approche moderne style AlphaStar / OpenAI Five.

**Pourquoi ce choix** : du RL pur from scratch sur CARLA est très long à converger sur une tâche multi-objectifs (conduite + feux + obstacles). Pré-initialiser le modèle par imitation learning donne un agent déjà fonctionnel, que le RL peut ensuite peaufiner. On garde une vraie composante RL (cf. sujet original du projet) sans le risque "rien à montrer".

## Architecture

Le module est divisé en 3 sous-dossiers indépendants :

```
src/ai/
├── config.py              ← Hyperparamètres centralisés
├── models/                ← Architectures de réseaux
├── training/              ← Scripts d'entraînement (+ data loader spécifique IA)
└── inference/             ← Boucle de démo dans CARLA
```

> ℹ️ **La collecte du dataset n'est pas dans ce module.** Elle est partagée entre tous les modules d'apprentissage et vit dans [src/dataset/](../dataset/). Voir aussi le [README racine](../../README.md) section "Datasets".

> ℹ️ **La lecture du dataset** (parsing manifest, conversion en `tf.data.Dataset`, augmentations) vit dans `training/data_loader.py` parce qu'elle est utilisée uniquement par le training. Pas de sous-dossier `dataset/` séparé.

### `models/`

Définit les architectures de réseaux. Plusieurs variantes :

- **`pilotnet.py`** — CNN style NVIDIA PilotNet, baseline simple `image → contrôles`
- **`cil.py`** — Conditional Imitation Learning, consomme aussi commande HN + vitesse

### `training/`

Scripts CLI pour entraîner les modèles. Sauvegarde des poids dans `checkpoints/<model_name>/`.

```bash
uv run -m src.ai.training.train --model pilotnet --dataset data/runs/<date>/
```

### `inference/`

Boucle de démo : lance CARLA, charge un modèle entraîné, prédit et applique les contrôles à chaque tick. Inclut un overlay HUD pour visualiser ce que fait le modèle.

```bash
uv run -m src.ai.inference.carla_demo --weights checkpoints/<model_name>/best.h5
```

## Pipeline complet

### Phase 1 — CIL pre-training (offline, sans CARLA en boucle)

```
[src/dataset/] ──> data/runs/<date>/
                          │
                          ▼
                   training/data_loader  (lecture, aug)
                          │
                          ▼
                   training/train.py ──> checkpoints/<model>/best.h5
                          │
                          ▼
                   inference/ ──> [CARLA] (démo)
```

La collecte du dataset commun est portée par [src/dataset/](../dataset/) (Franck principalement). Une fois le dataset sur disque, l'IA centrale s'entraîne offline, sans CARLA, sur n'importe quelle machine avec un GPU.

### Phase 2 — RL fine-tuning (en boucle CARLA)

```
checkpoints/<cil_model>/best.h5
          │
          ▼ (initialisation)
training/rl_finetune.py ←─────┐
          │                   │
          ▼                   │ (action)
       [CARLA]                │
          │ (state, reward)   │
          └───────────────────┘
                              │
                              ▼
              checkpoints/<rl_model>/best.h5
```

Le modèle pré-entraîné par CIL sert de point de départ à un algorithme RL (PPO/SAC) qui interagit avec CARLA en continu pour optimiser une fonction de récompense. Cette phase nécessite que CARLA tourne pendant des jours/semaines. Spec détaillé à venir une fois la Phase 1 validée.

## Inputs et outputs

### Mode "image only" (PilotNet baseline)

- Input : `image (H, W, 3)` uint8
- Output : `(steer, throttle, brake)` floats

### Mode "scene_state" (CIL et au-delà)

- Input : `SceneState` (image + objets détectés + depth + lignes + état véhicule + commande HN)
- Output : `ControlOutput`

Les deux modes coexistent et le choix se fait au niveau du modèle (`models/pilotnet.py` vs `models/cil.py`).

## Stratégie face aux modules pas encore prêts

Tant que YOLO / MIDAS / Lignes ne sont pas opérationnels, l'IA centrale s'entraîne en mode "image only" (baseline PilotNet). Le dataset collecté contient toutefois aussi les ground truths CARLA (objets, depth) pour permettre une évolution vers le mode "scene_state" sans recollecter.

Lorsque les modules de perception seront prêts, la transition consiste à :

1. Brancher leurs implémentations dans `inference/carla_demo.py` (via les protocoles de `src/interfaces/`)
2. Entraîner un nouveau modèle `cil.py` qui consomme la `SceneState` complète
3. Comparer les performances à la baseline

## Configuration

Hyperparamètres centralisés dans `config.py` :

- Image size, batch size, learning rate, epochs
- Dataset paths, train/val split
- Augmentation params

## Validation et benchmarks

Dans [benchmarks/ai/](../../benchmarks/ai/) :

- **`smoke.py`** : vérifier que les modèles produisent des sorties dans les bons intervalles (`steer ∈ [-1, 1]`, `throttle, brake ∈ [0, 1]`), que le data loader fonctionne sur un mini-dataset synthétique, que `ControlOutput` est bien produit à partir d'une `SceneState` factice
- **`benchmark.py`** : loss val sur jeu de validation, taux de succès trajet sans collision en démo CARLA, FPS d'inférence

Voir [benchmarks/README.md](../../benchmarks/README.md) pour la convention.

## Liens

- [Codevilla et al. 2018, Conditional Imitation Learning sur CARLA](https://arxiv.org/abs/1710.02410) — Phase 1
- [Bojarski et al. 2016 (NVIDIA), End-to-End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) — Phase 1 (PilotNet)
- [Schulman et al. 2017, Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) — candidat Phase 2
- [Haarnoja et al. 2018, Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) — candidat Phase 2
- [Kendall et al. 2018, Learning to Drive in a Day (Wayve)](https://arxiv.org/abs/1807.00412) — RL sur conduite réelle, référence pour Phase 2
