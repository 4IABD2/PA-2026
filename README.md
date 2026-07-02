# PA-2026 — Voiture autonome dans CARLA

Projet annuel **4IABD2** : faire circuler de manière autonome un véhicule dans le simulateur **CARLA**, par computer vision uniquement (caméra RGB frontale).

> 📖 **Ce README est le point d'entrée unique du projet.** Tout est ici : équipe, architecture, conventions, datasets, installation. Lis-le en entier avant de commencer à coder. Pour ton module précisément, le `README.md` du sous-dossier concerné contient les détails techniques.

---

## Table des matières

- [Objectif](#objectif)
- [Équipe et responsabilités](#équipe-et-responsabilités)
- [Pipeline temps réel](#pipeline-temps-réel)
- [Pipeline d'entraînement](#pipeline-dentraînement)
- [Structure du repo](#structure-du-repo)
- [Le dossier clé : `src/interfaces/`](#le-dossier-clé--srcinterfaces)
- [Datasets](#datasets)
  - [Format du dataset](#format-du-dataset)
  - [Stockage et partage](#stockage-et-partage)
- [Installation](#installation)
- [Démarrage](#démarrage)
- [Conventions de code](#conventions-de-code)
- [Conventions Git](#conventions-git)
- [Validation et benchmarks](#validation-et-benchmarks)
- [Suivi de progression](#suivi-de-progression)
- [Conventions architecturales](#conventions-architecturales)
- [Conventions CARLA](#conventions-carla)
- [Philosophie technique](#philosophie-technique)

---

## Objectif

Le véhicule doit, dans CARLA :

- Respecter la route et les marquages au sol
- Maintenir des distances de sécurité avec les autres véhicules
- Adapter sa cinématique (vitesse, accélération) à son environnement
- Réagir aux feux de signalisation

Le pilotage se fait à partir d'une **caméra frontale RGB unique**, enrichie par les modules de perception (YOLO, depth, lignes). Aucun capteur natif de CARLA n'est utilisé en production — ils servent uniquement à entraîner et valider les modèles.

## Phases du projet

| Phase | Nom | État | Description |
|---|---|---|---|
| **Phase 0** | CIL — introduction à CARLA | ✅ Archivé | PilotNet (imitation learning sur l'autopilot CARLA). Sert de référence et d'apprentissage de la stack. Ne pas modifier. |
| **Phase 1** | RL — apprentissage par renforcement | 🔄 En cours | L'IA apprend seule via PPO (Stable-Baselines3). Observations structurées : vitesse, commande nav Victor, centrage voie, distance obstacles. Reward function à la place des labels copiés. |

> **Phase 0 est gardée telle quelle** dans `src/ai/` (préfixe `v1_`). C'était notre introduction à CARLA et à la chaîne dataset → training → démo. Elle ne sera jamais supprimée — elle est juste archivée.

## Équipe et responsabilités

| Personne | Module(s) | Dossier(s) | Lis ce README |
|---|---|---|---|
| **Frédéric Huang** | IA centrale (décision) + intégration | [src/ai/](src/ai/), [src/interfaces/](src/interfaces/) | [src/ai/README.md](src/ai/README.md) |
| **Franck Zhuang** | Détection d'objets (YOLO) + Estimation de profondeur (MIDAS / Depth Anything) | [src/perception/yolo/](src/perception/yolo/), [src/perception/depth/](src/perception/depth/) | [src/perception/yolo/README.md](src/perception/yolo/README.md), [src/perception/depth/README.md](src/perception/depth/README.md) |
| **Karim Arfaoui** | Détection de lignes (alignement) | [src/perception/lanes/](src/perception/lanes/) | [src/perception/lanes/README.md](src/perception/lanes/README.md) |
| **Victor Dalet** | Navigation (GPS, planification de route) | [src/navigation/](src/navigation/) | [src/navigation/README.md](src/navigation/README.md) |

L'orchestration générale (boucle temps réel CARLA) est portée collectivement dans [src/orchestration/](src/orchestration/). Le générateur de dataset commun est dans [src/dataset/](src/dataset/) (Franck principalement, contributions de Frédéric et Karim).

## Pipeline temps réel

```
                        ┌─────────────────────┐
                        │      CARLA          │
                        │  (simulateur)       │
                        └──┬───────────────┬──┘
                           │               ▲
                  ┌────────┘               │
                  │ Caméra RGB             │ Contrôles
                  ▼                        │ (steer, throttle, brake)
        ┌─────────────────┐                │
        │   PERCEPTION    │                │
        │                 │                │
        │ • YOLO          │ ── Franck      │
        │ • Depth (MIDAS) │ ── Franck      │
        │ • Lignes        │ ── Karim       │
        └────────┬────────┘                │
                 │                         │
                 │ scene_state             │
                 ▼                         │
        ┌─────────────────┐                │
        │  IA CENTRALE    │ ── Frédéric    │
        │  (décision)     │                │
        └────────┬────────┘                │
                 │                         │
                 └─────────────────────────┘
                           ▲
                           │ commande haut niveau
                  ┌────────┴────────┐
                  │  NAVIGATION     │ ── Victor
                  │  GPS / route    │
                  └─────────────────┘
```

À chaque tick (mode synchrone CARLA, 20 FPS) :

1. CARLA capture une frame RGB de la caméra du véhicule
2. Cette frame est traitée par les **modules de perception** (YOLO, Depth, Lignes) qui produisent des métriques structurées (distance aux obstacles, centrage sur la voie, objets détectés)
3. Le module de **navigation** fournit la prochaine commande haut niveau (gauche / droite / tout droit / suivre la voie) à chaque intersection
4. L'**IA centrale** consomme un vecteur d'observation compact `[speed_norm, cmd_one_hot, center_offset, nearest_obstacle_m, heading_error]` et produit `(steer, throttle, brake)` via sa policy PPO
5. Ces contrôles sont appliqués au véhicule via `vehicle.apply_control(...)`

> **Stratégie de substitution perception** : pendant l'entraînement RL, les métriques de perception proviennent des GT CARLA (via `src/interfaces/stubs.py`). Une fois les modèles de Franck et Karim prêts, on substitue les stubs par les vraies implémentations — le code de l'IA centrale ne change pas (protocoles `src/interfaces/`).

## Pipeline d'entraînement

### Modules de perception (Franck + Karim) — offline sur dataset

```
[CARLA + autopilot] ──> [src/dataset/] ──> data/runs/<date>/
                                                 │
                                                 │ (transfert serveur)
                                                 ▼
                              ┌──────────────────┼──────────────────┐
                              ▼                  ▼                  ▼
                       [perception/yolo/  [perception/depth/  [perception/lanes/
                        train.py]          train.py]            train.py]
                              │                  │                  │
                              ▼                  ▼                  ▼
                          weights/           weights/            weights/
```

### IA centrale (Frédéric) — online en boucle CARLA

```
                     CARLA (20 FPS, sync)
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
     GT depth        GT lanes         Navigation Victor
     (→ distance     (→ center       (→ HighLevelCommand)
      obstacles)      offset)
           │               │               │
           └───────────────┴───────────────┘
                           │
                    [Observation vector]
                    speed + cmd + offset + dist + heading
                           │
                           ▼
                    [PPO Policy] (Stable-Baselines3)
                           │
                    action (steer/throttle/brake)
                           │
                           ▼
                    [Reward Function]
                    r_speed + r_center + r_alive − r_stall − r_offroute − r_collision
                           │
                      mise à jour policy
```

Une seule collecte commune produit les données pour les modules perception (YOLO, depth, lanes). L'IA centrale s'entraîne directement en boucle CARLA — plus de dataset offline pour elle. Voir [Datasets](#datasets) pour le format et [src/dataset/README.md](src/dataset/README.md) pour le générateur.

## Structure du repo

```
PA-2026/
├── src/
│   ├── perception/              ← Vision (Franck + Karim)
│   │   ├── yolo/                ← Détection d'objets (Franck)
│   │   ├── depth/               ← MIDAS / Depth Anything (Franck)
│   │   └── lanes/               ← Détection de lignes (Karim)
│   ├── navigation/              ← GPS / planification (Victor)
│   ├── ai/                      ← IA centrale, décision (Frédéric)
│   │   ├── models/              ← Architectures
│   │   ├── training/            ← Scripts d'entraînement (+ data loader)
│   │   └── inference/           ← Démo en boucle CARLA
│   ├── dataset/                 ← Génération du dataset commun depuis CARLA (partagé)
│   ├── interfaces/              ← ★ Contrats partagés entre modules
│   ├── orchestration/           ← Boucle temps réel CARLA
│   └── tools/                   ← Utilitaires partagés
├── scripts/
│   ├── run_rl_training.py       ← Lance le training Phase 1 (CARLA + PPO + artifacts + eval)
│   ├── run_eval.py              ← Évalue un modèle sur les 13 scénarios benchmark → vidéo + JSON
│   ├── explore_spawns.py        ← Explore et classe les spawn points Town10HD_Opt
│   └── find_dest_spawns.py      ← Trouve les dest_spawn_idx par direction de carrefour
├── runs/                        ← Artifacts d'entraînement horodatés (gitignored)
│   └── YYYY-MM-DD_HH-MM_tag/
│       ├── params.json, model_best.zip, model_final.zip, reward_curve.png, demo.mp4
│       └── evals/               ← Résultats benchmark par checkpoint (JSON + vidéo)
├── data/                        ← Datasets (gitignored)
├── checkpoints/                 ← Modèles Phase 0 (gitignored)
├── benchmarks/                  ← Validation contrats (smoke) + mesures de perf (mAP, RMSE, FPS)
├── docs/
│   ├── Description_Sujet.md     ← Sujet original du projet
│   ├── Architecture_PA2026.png
│   └── Canva_PA2026.pdf
├── const.py                     ← Constantes globales
├── main.py                      ← Point d'entrée
├── pyproject.toml               ← Déclaration des dépendances (uv)
├── uv.lock                      ← Versions figées (généré, commité)
└── README.md                    ← Ce fichier

Chaque dossier de module dans `src/` contient en plus :
- Un `README.md` (doc technique du module)
- Un `JOURNAL.md` (suivi de progression au fil de l'eau, voir [Suivi de progression](#suivi-de-progression))
```

## Le dossier clé : `src/interfaces/`

C'est **le dossier le plus important du repo**. Il contient les contrats Python (`@dataclass` + `Protocol`) qui définissent les structures d'échange entre modules.

**Pourquoi c'est central** : tant qu'un module respecte ces contrats, il est interchangeable. Cela permet à 4 personnes de bosser **en parallèle, indépendamment, sans se marcher dessus**. L'intégration finale est triviale parce que les pièces s'emboîtent automatiquement.

Exemple — implémenter un détecteur d'objets :

```python
import numpy as np
from src.interfaces.perception_types import DetectedObject, ObjectClass

class YoloDetector:
    def __init__(self, weights_path: str) -> None: ...

    def detect(self, image: np.ndarray) -> list[DetectedObject]:
        return [
            DetectedObject(
                class_name=ObjectClass.VEHICLE,
                bbox=(x1, y1, x2, y2),
                confidence=score,
            )
            ...
        ]
```

Cette classe est automatiquement compatible avec le protocole `ObjectDetector` parce qu'elle a la bonne méthode `detect`. Elle peut être utilisée partout où un `ObjectDetector` est attendu.

[src/interfaces/stubs.py](src/interfaces/stubs.py) fournit aussi des **implémentations "triche"** qui lisent directement les ground truths CARLA. Pratique pour bosser sur son module sans attendre que les autres soient prêts. ⚠️ Ces stubs ne doivent **jamais** arriver en production — voir [Conventions architecturales](#conventions-architecturales).

Lis [src/interfaces/README.md](src/interfaces/README.md) avant de commencer à implémenter ton module.

## Datasets

**Un seul dataset commun** alimente tous les modules d'apprentissage. Il est généré depuis CARLA par le module [src/dataset/](src/dataset/) et contient toutes les modalités en parallèle :

| Module | Données utilisées |
|---|---|
| `src/perception/yolo/` (Franck) | `images/*.jpg` + `instance/*.npy` (dérivation bboxes) |
| `src/perception/depth/` (Franck) | `images/*.jpg` + `depth/*.npy` |
| `src/perception/lanes/` (Karim) | `images/*.jpg` + `semantic/*.npy` (classe RoadLine) |
| `src/ai/` (Frédéric) | `images/*.jpg` + `manifest.csv` (commande, vitesse, actions expert) |

Une seule collecte de ~30 min = données pour 4 personnes, avec cohérence garantie (même map, même météo, même caméra POV).

### Format du dataset

```
data/runs/<YYYY-MM-DD>_<town>_<weather>/
├── images/                ← RGB JPEG, 1 frame toutes les 2s
│   ├── 000000.jpg
│   └── ...
├── depth/                 ← depth maps GT en mètres (.npy float32)
│   ├── 000000.npy
│   └── ...
├── semantic/              ← masks sémantique CARLA (uint8 class IDs 0-28)
│   ├── 000000.npy
│   └── ...
├── semantic_viz/          ← visualisation palette CityScape (PNG)
│   └── ...
├── instance/              ← masks instance CARLA (uint32 packed: class_id<<16|G<<8|B)
│   ├── 000000.npy
│   └── ...
├── instance_viz/          ← visualisation couleur déterministe par instance (PNG)
│   └── ...
├── labels_yolo/           ← labels YOLO (.txt par image)
│   ├── 000000.txt         ← format: <class> <x_center> <y_center> <w> <h>
│   └── ...
├── lanes_gt/              ← (optionnel) annotations lignes (.json)
│   ├── 000000.json
│   └── ...
├── manifest.csv           ← 1 ligne par frame (voir ci-dessous)
└── metadata.json          ← métadonnées globales de la run
```

**`manifest.csv`** — colonnes :

| Colonne | Type | Description |
|---|---|---|
| `frame_id` | int | Identifiant unique de la frame (0, 1, 2...) |
| `image_path` | str | Chemin relatif depuis la racine de la run |
| `timestamp` | float | Temps écoulé depuis le début de la run, en secondes |
| `command` | str | Commande haut niveau : `left`, `right`, `straight`, `lane_follow` |
| `speed_kmh` | float | Vitesse instantanée du véhicule |
| `steer` | float | Steering appliqué par l'expert (autopilot CARLA), [-1, 1] |
| `throttle` | float | Throttle appliqué, [0, 1] |
| `brake` | float | Brake appliqué, [0, 1] |
| `is_collision` | int | 1 si collision durant les 5 dernières frames, 0 sinon (à filtrer) |
| `town` | str | Nom de la map CARLA |
| `weather` | str | Preset météo CARLA |

**`metadata.json`** — exemple :

```json
{
  "run_id": "2026-05-09_town01_clear",
  "town": "Town01",
  "weather": "ClearNoon",
  "carla_version": "0.9.16",
  "fps": 20,
  "capture_every_n_ticks": 40,
  "n_frames": 1500,
  "n_npc_vehicles": 40,
  "n_npc_walkers": 30,
  "camera_pov": {"location": [0.30, 0.0, 1.50], "rotation": [0, 0, -5]},
  "duration_sec": 3000,
  "available_modalities": ["rgb", "depth", "semantic", "instance"]
}
```

**`labels_yolo/<frame_id>.txt`** — un objet par ligne, valeurs normalisées dans `[0, 1]` :
```
<class_id> <x_center> <y_center> <width> <height>
```

Mapping des classes : 0=vehicle, 1=walker, 2=traffic_light (extensible).

**`depth/<frame_id>.npy`** — `numpy.ndarray` de forme `(H, W)`, dtype `float32`, valeurs en mètres, plafonnées à 100m.

**`semantic/<frame_id>.npy`** — `numpy.ndarray` de forme `(H, W)`, dtype `uint8`, valeurs = class ID CARLA (mapping CityScape CARLA 0.9.13+, 0-28 — `1`=Roads, `14`=Car, `24`=RoadLine, etc.).

**`instance/<frame_id>.npy`** — `numpy.ndarray` de forme `(H, W)`, dtype `uint32`. Layout : `(class_id << 16) | (G_byte << 8) | B_byte`. Unpack : `class_id = (p >> 16) & 0xFF`, `instance_id = p & 0xFFFF`. `instance_id == 0` = fond.

### Stockage et partage

Le dossier `data/` à la racine est **gitignored**. Aucun fichier de dataset n'arrive sur Git.

**Local** : chacun garde ses runs dans `data/runs/`. Naming recommandé : `<YYYY-MM-DD>_<town>_<weather>`.

**Partagé entre membres de l'équipe** (procédure manuelle) :

```bash
# Compresser
tar -czf 2026-MM-DD_town01_clear.tar.gz data/runs/2026-MM-DD_town01_clear

# Partager via cloud (Drive, scp serveur perso, GitHub Releases...)
# Documenter dans un message équipe : nom, lien, taille, particularités

# Récupérer
wget <url>
tar -xzf 2026-MM-DD_town01_clear.tar.gz -C data/runs/
```

**Bonnes pratiques** :
- **Diversité** : varier les maps (Town01 → Town03 → Town05), les météos, les densités de NPC
- **Filtrage** : exclure les frames `is_collision=1` du training
- **Équilibrage** : la commande `lane_follow` représente >80% des frames, sous-échantillonner ou pondérer
- **Reproductibilité** : tous les paramètres dans `metadata.json` (seed, version, params NPC)

## Installation

### Prérequis

- **Python 3.10+**
- **CARLA Simulator 0.9.16** (installé séparément, voir [docs CARLA](https://carla.readthedocs.io/en/0.9.16/start_quickstart/))
- **uv** pour la gestion des dépendances ([install uv](https://docs.astral.sh/uv/))
- **GPU NVIDIA recommandé** pour faire tourner CARLA confortablement

### Stack principale

| Composant | Bibliothèque | Usage |
|---|---|---|
| Simulateur | CARLA 0.9.16 | Environnement de simulation |
| Vision | OpenCV, Ultralytics (YOLOv8) | Perception |
| Depth | MIDAS / Depth Anything v2 | Estimation profondeur |
| IA Phase 0 | TensorFlow / Keras | CIL archivé |
| IA Phase 1 | Stable-Baselines3 + Gymnasium | RL PPO en boucle CARLA |
| Gestion deps | uv | `pyproject.toml` + `uv.lock` |
| Format | black | Linter imposé |

### Dépendances Python

Le projet utilise [`uv`](https://docs.astral.sh/uv/) pour la gestion des dépendances. Les sources de vérité sont :

- `pyproject.toml` — déclare les dépendances
- `uv.lock` — fige les versions exactes (commité, **ne pas éditer à la main**)

À la première installation :

```bash
uv sync                 # crée .venv/ + installe toutes les deps (incl. dev group)
uv sync --no-dev        # variante sans black (utile en CI/prod)
```

Au quotidien :

```bash
uv run main.py          # exécute dans le venv (pas besoin d'activate)
uv run -m pytest benchmarks/
uv run -m black .

uv add <package>            # ajouter une dep runtime
uv add --group dev <package>  # ajouter une dep dev
uv lock --upgrade           # mettre à jour le lockfile
```

Python 3.10 est requis (CARLA 0.9.16 n'a pas de wheels pour 3.11+). Si tu n'as pas Python 3.10 sur ta machine, uv le télécharge tout seul au premier `uv sync`.

## Démarrage

### Lancer CARLA

Selon ton environnement, choisir une des options ci-dessous. Par défaut le serveur CARLA écoute sur `localhost:2000`.

#### Option A — Serveur Linux partagé (recommandé) via Docker

C'est l'approche recommandée pour un serveur Linux multi-GPU partagé entre plusieurs utilisateurs.

**1. Vérifier les prérequis sur le serveur :**

```bash
# Vérifier Docker installé
docker --version

# Vérifier les GPU NVIDIA disponibles et leur charge actuelle
nvidia-smi

# Vérifier que NVIDIA Container Toolkit est installé (pour Docker --gpus)
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

Si la dernière commande échoue, installer le NVIDIA Container Toolkit :
[github.com/NVIDIA/nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit/blob/main/install-guide.md).

**2. Choisir un GPU libre et un port disponible :**

```bash
# Identifier les GPU peu chargés
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# Vérifier qu'un port n'est pas déjà utilisé par un autre user
ss -ltn | grep 2000  # vide = libre. Sinon, essayer 2010, 2020, 3000...
```

**3. Pull et lancer l'image officielle CARLA 0.9.16 :**

```bash
# Pull (~7 GB, à faire une seule fois)
docker pull carlasim/carla:0.9.16

# Lancer en headless, sur GPU 0, ports 2000-2002
docker run -d --gpus device=0 \
    --name carla-<ton-prenom> \
    -p 2000-2002:2000-2002 \
    carlasim/carla:0.9.16 \
    /bin/bash -c "./CarlaUE4.sh -RenderOffScreen -nosound -world-port=2000"
```

`-d` = détaché (en arrière-plan). Vérifier que ça tourne :

```bash
docker ps | grep carla-<ton-prenom>
docker logs carla-<ton-prenom> --tail 20
```

**4. Tester la connexion Python :**

```python
# test_carla.py
import carla
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
print("CARLA version:", client.get_server_version())
print("Available maps:", client.get_available_maps()[:3], "...")
```

```bash
uv run test_carla.py
```

Si ça affiche la version et quelques maps, c'est OK.

**5. Arrêter / nettoyer :**

```bash
docker stop carla-<ton-prenom>
docker rm carla-<ton-prenom>
```

#### Option B — Installation native sur Linux (sans Docker)

```bash
# Télécharger CARLA 0.9.16 (~7 GB)
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.16.tar.gz
tar -xzf CARLA_0.9.16.tar.gz -C ~/carla-0.9.16
cd ~/carla-0.9.16

# Lancer en headless, GPU 0
./CarlaUE4.sh -RenderOffScreen -graphicsadapter=0 -world-port=2000
```

#### Option C — Local sur Windows / Mac (avec écran)

Télécharger l'archive depuis [github.com/carla-simulator/carla/releases](https://github.com/carla-simulator/carla/releases) (chercher `CARLA_0.9.16.zip` ou `.tar.gz` selon l'OS), décompresser, lancer :

- Windows : `CarlaUE4.exe`
- Linux/Mac avec écran : `./CarlaUE4.sh`

#### Astuces multi-utilisateurs sur serveur partagé

- Préfixer le nom du conteneur Docker avec ton prénom (`--name carla-fhuang`) pour éviter les collisions
- Utiliser des ports différents si plusieurs CARLA tournent simultanément (`-world-port=2000`, `2010`, `2020`...)
- Spécifier le GPU : `--gpus device=N` (Docker) ou `-graphicsadapter=N` (natif). Vérifier avec `nvidia-smi` quel GPU est libre
- Pour tunneler le port CARLA depuis ton laptop vers le serveur via SSH : `ssh -L 2000:localhost:2000 <user>@<serveur>` puis ton client Python local se connecte à `localhost:2000` comme si CARLA était local

### Lancer le projet

**Phase 0 (archivé) :**
```bash
uv run main.py
```

**Phase 1 — training RL (CARLA requis) :**
```bash
# Smoke test (vérif pipeline, ~2 min)
uv run python3 scripts/run_rl_training.py --timesteps 1000 --tag smoke --host <ip-carla>

# Training réel (résultats visibles à partir de 50k steps)
uv run python3 scripts/run_rl_training.py --timesteps 300000 --tag ppo_v1 --host <ip-carla>
```

**Phase 1 — évaluation benchmark (13 scénarios) :**
```bash
# Évaluer un modèle sur les 13 scénarios → vidéo annotée + JSON métriques
uv run python3 scripts/run_eval.py --model runs/<dir>/best_model.zip --host <ip-carla>
# Sortie : eval_best_model.mp4 + eval_best_model.json (trajectoires, stats, off_route_pct...)
```

**Utilitaires spawn (Town10HD_Opt) :**
```bash
# Identifier et classer les spawn points
uv run python3 scripts/explore_spawns.py --host <ip-carla>

# Trouver les dest_spawn_idx pour les scénarios de jonction (sans appeler nav.plan)
uv run python3 scripts/find_dest_spawns.py --host <ip-carla> --spawn-idx 0
```

> **WSL** : CARLA tourne sur Windows. Si Tailscale est installé, utiliser directement l'IP Tailscale de la machine Windows (`tailscale status` pour la voir). Sinon, l'IP du host Windows se trouve avec `cat /etc/resolv.conf | grep nameserver`.

Les artefacts sont générés dans `runs/YYYY-MM-DD_HH-MM_<tag>/` (voir [src/ai/README.md](src/ai/README.md) pour le détail).

### Validation et benchmarks

```bash
uv run -m pytest benchmarks/        # smoke tests rapides
uv run -m benchmarks.<module>.benchmark   # benchmarks de perf (lent)
```

**Benchmark IA centrale — 13 scénarios fixes :**

Le benchmark Phase 1 couvre 8 scénarios évalués avec critère de succès strict + 5 scénarios Phase 2 enregistrés. Pour chaque modèle, `run_eval.py` génère :
- Une vidéo annotée (13 scénarios en séquence, overlay OBS space + action)
- Un JSON structuré avec : trajectoire `[[x,y,yaw]]`, séries temporelles, stats (speed, offset, off_route_pct, nav_commands), `reached_dest`, `collision_step`

Les `dest_spawn_idx` des scénarios de jonction garantissent que la nav Victor donne les bonnes commandes directionnelles (route replanifiée vers un spawn cible spécifique à chaque reset).

Critères de succès Phase 1 :
- `straight`, `curve_*`, `npc_*` : pas de collision + avoir parcouru ≥ 25m
- `turn_left`, `turn_right`, `junction_straight` : atteindre la destination (dans un rayon de 15m du spawn cible)

### Linter

```bash
uv run -m black .
```

Le linter est aussi vérifié en CI via [.github/workflows/linter.yaml](.github/workflows/linter.yaml).

## Conventions de code

### Nommage

- `snake_case` pour les fichiers, fonctions, variables, attributs
- `PascalCase` pour les classes
- `SCREAMING_SNAKE_CASE` pour les constantes (`MAX_SPEED_KMH`)
- Préfixer par `_` les attributs/méthodes privés (`_internal_state`)

### Imports

**Toujours absolus** depuis `src.` :

```python
# Bon
from src.interfaces.perception_types import DetectedObject
from src.tools.matplot_visualizer import MatplotVisualizer

# Mauvais
from ..interfaces.perception_types import DetectedObject
```

Trier dans cet ordre, séparés par une ligne vide :
1. Standard library
2. Tiers (carla, numpy, tensorflow, cv2)
3. Projet (`src.*`)

### Type hints

Obligatoires sur toutes les **signatures publiques** (fonctions/méthodes non préfixées par `_`).

```python
def detect(self, image: np.ndarray) -> list[DetectedObject]:
    ...
```

Utiliser la syntaxe Python 3.10+ (`list[...]`, `int | None`) plutôt que `typing.List`, `typing.Optional`.

### Format

**`black`** est imposé. Aucun PR ne passe la CI sans `black .` clean.

```bash
uv run -m black .
```

Pas de configuration custom : on utilise les défauts (88 colonnes).

### Docstrings et commentaires

- Une docstring courte (1 ligne) suffit pour les fonctions évidentes.
- Pour les fonctions publiques d'un module, expliquer rapidement les arguments importants et la valeur de retour si elle n'est pas triviale.
- Pas de format imposé (Google, NumPy, reST) tant que c'est lisible.
- Un commentaire qui paraphrase le code est à supprimer. Un commentaire qui explique **pourquoi** quelque chose est fait (contrainte non évidente, choix technique surprenant, lien vers un papier) est utile.

## Conventions Git

### Branches

Format : `feat/<scope>_<short_desc>`

Exemples (convention déjà utilisée dans le repo) :
- `feat/add_first_idea_of_input_model`
- `feat/improve_gps_output_for_training`
- `feat/add_linter_black_ga`

Une branche = une feature ou un fix logique. Pas de "branche fourre-tout".

### Commits

Format : `feat(scope): message court à l'impératif`

Préfixes courants :
- `feat(scope):` — nouvelle fonctionnalité
- `fix(scope):` — correction de bug
- `refactor(scope):` — réorganisation sans changement de comportement
- `docs(scope):` — documentation
- `test(scope):` — tests
- `chore(scope):` — outillage, dépendances, CI

`scope` désigne le module concerné (ex : `gps`, `model`, `ai`, `perception`).

Exemples (depuis l'historique) : `feat(gps): Add gps algo`, `feat(deep_renforcement): Add first base model`, `feat(lint): ga for black`.

### Pull requests

- Toujours par PR (pas de push direct sur `main`).
- Au moins une review par un membre de l'équipe avant merge.
- Le titre suit le format des commits.
- Le merge se fait sur `main` après validation de la CI.

## Validation et benchmarks

Le projet est en grande partie composé de modules ML / CV. Les tests unitaires classiques apportent peu — ce qui compte, c'est (1) que les contrats soient respectés et (2) la performance réelle (mAP, RMSE, FPS, taux de succès en démo).

**Pas de dossier `tests/` séparé.** Tout est regroupé dans `benchmarks/`, par module :

```
benchmarks/
├── README.md
├── perception/
│   ├── yolo/
│   │   ├── smoke.py            ← validation contrat (rapide)
│   │   ├── benchmark.py        ← mesure de perf (lent)
│   │   └── results/            ← gitignored
│   ├── depth/{smoke.py, benchmark.py, results/}
│   └── lanes/{smoke.py, benchmark.py}
├── navigation/{smoke.py, benchmark.py}
├── ai/{smoke.py, benchmark.py}
└── pipeline/benchmark.py       ← FPS et latence end-to-end
```

### `smoke.py` — validation rapide du contrat

Vérifie que l'implémentation respecte le protocole défini dans [src/interfaces/](src/interfaces/). Pas de dataset, pas de CARLA, lance-le souvent (idéal en CI).

```bash
uv run -m pytest benchmarks/
```

Pour tester un module qui dépend d'autres modules, utiliser les stubs de [src/interfaces/stubs.py](src/interfaces/stubs.py). Ne **jamais** instancier un vrai modèle dans un smoke test.

### `benchmark.py` — mesure de performance

Exécution sur un vrai dataset, mesure latence + métriques métier. Les résultats vont dans `benchmarks/<module>/results/` (gitignored).

```bash
uv run -m benchmarks.perception.yolo.benchmark \
    --weights checkpoints/yolo_v1/best.pt \
    --dataset data/runs/2026-05-09_town01_clear
```

Voir [benchmarks/README.md](benchmarks/README.md) pour la convention complète, le format JSON des résultats, et le tableau des métriques attendues par module.

Le résumé d'un benchmark significatif est à reporter dans le `JOURNAL.md` du module concerné.

## Suivi de progression

Pour éviter de devoir reconstituer 3 mois de travail au moment du rendu, **chaque module avec un owner a un `JOURNAL.md`** que l'owner met à jour à chaque session significative.

| Fichier | Owner |
|---|---|
| [src/perception/yolo/JOURNAL.md](src/perception/yolo/JOURNAL.md) | Franck |
| [src/perception/depth/JOURNAL.md](src/perception/depth/JOURNAL.md) | Franck |
| [src/perception/lanes/JOURNAL.md](src/perception/lanes/JOURNAL.md) | Karim |
| [src/navigation/JOURNAL.md](src/navigation/JOURNAL.md) | Victor |
| [src/ai/JOURNAL.md](src/ai/JOURNAL.md) | Frédéric |
| [src/dataset/JOURNAL.md](src/dataset/JOURNAL.md) | Franck (principal), Frédéric et Karim contribuent |

**Format d'une entrée** (libre tant que c'est régulier) :

```markdown
## YYYY-MM-DD
**Avancement** : ce qui a été fait ce jour
**Difficultés** : blocages rencontrés et comment ils ont été (ou pas) résolus
**Décisions** : choix techniques pris (et pourquoi)
**Benchmarks** : résultats chiffrés si pertinent
**Prochaine étape** : ce qui est prévu après
```

**Notebooks** : si tu fais des notebooks d'analyse / graphes, mets-les **dans le dossier de ton module** (`src/<module>/notebooks/`). Pas de dossier central. Tu peux aussi sauvegarder les images générées à côté pour qu'on les retrouve.

À la fin du projet, on demandera à un agent de **synthétiser tous les `JOURNAL.md`** pour produire un bilan complet — donc l'écriture peut rester libre et naturelle, ce qui compte c'est qu'elle soit régulière.

## Conventions architecturales

Quatre règles transverses qui structurent le projet :

1. **Aucun module ne dépend directement d'un autre module concret**. Tout passe par les interfaces ([src/interfaces/](src/interfaces/)).
2. **Aucun import de `carla` en dehors de [src/orchestration/](src/orchestration/), [src/dataset/](src/dataset/), [src/ai/inference/](src/ai/inference/), [src/interfaces/stubs.py](src/interfaces/stubs.py)**. Les modèles purs (perception, ai/models) doivent être testables sans CARLA.
3. **Pas de triche en production** : les capteurs de ground truth de CARLA (`sensor.camera.semantic_segmentation`, `sensor.camera.depth`) ne sont consommés que pendant la collecte/entraînement, jamais en input de la boucle production. Si un module en a besoin pour bosser sans dépendance, passer par `src/interfaces/stubs.py` (qui marque explicitement la triche temporaire).
4. **Datasets et modèles entraînés ne sont jamais commités** (`.gitignore` couvre `data/`, `checkpoints/`, `**/weights/`, `**.h5`, `**.png`).

## Conventions CARLA

### Mode synchrone obligatoire pour la collecte et la démo

```python
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 20 FPS
world.apply_settings(settings)
```

Sans mode synchrone, les frames arrivent à un rythme imprévisible et le dataset est inutilisable.

### Caméra POV "rooftop driver" (convention partagée)

```python
cam_transform = carla.Transform(
    carla.Location(x=0.30, y=0.0, z=1.50),
    carla.Rotation(pitch=-5),
)
```

Caméra **centrée longitudinalement** (axe y=0), légèrement en avant du centre véhicule (x=0.30), placée **juste au-dessus du toit Tesla Model 3** (z=1.50, le toit étant à ~1.44m), avec un léger tilt vers le bas (pitch=-5°). Tous les datasets et démos doivent utiliser cette caméra pour assurer la compatibilité entre les modules entraînés indépendamment.

**Pourquoi pas une caméra à l'intérieur de la cabine ?** Les capteurs `sensor.camera.semantic_segmentation` et `sensor.camera.instance_segmentation` de CARLA **ne respectent pas la transparence des matériaux** (alors que `sensor.camera.rgb` le fait). Une caméra placée derrière le pare-brise génère donc des masks où ~100% des pixels sont la classe `Car` (le mesh de la carrosserie ego), ce qui rend la donnée inutilisable pour la détection de lignes (Karim) et la dérivation des bboxes YOLO via masks d'instance (Franck). La position « rooftop driver » (z=1.50, +6 cm au-dessus du toit) est le minimum qui clear proprement le body tout en gardant une perspective naturelle "tête au-dessus du conducteur" — testée systématiquement contre plusieurs alternatives (z=1.45 grazing, z=1.55, z=1.60 hood-forward) pour le meilleur compromis lisibilité humaine / cadrage utile pour les modèles.

### Capture toutes les 2s (1 frame / 40 ticks à 20 FPS)

Deux frames consécutives à 20 FPS sont quasi identiques. Capturer toutes les 2s donne un dataset plus diversifié sans doublons.

### Maps utiles

- **Town01** : petite, simple, pour débuter
- **Town03** : plus grande, avec rond-point
- **Town04** : autoroute + petite ville
- **Town05** : urbain dense, plusieurs voies

## Philosophie technique

- **Computer vision pure** : on ne pilote qu'à partir de la caméra RGB. Les capteurs ground truth de CARLA sont des outils d'entraînement, pas des inputs de production.
- **Modules découplés via contrats** : chaque module respecte une interface définie dans `src/interfaces/`. On peut développer, tester et remplacer chaque module indépendamment.
- **Mode synchrone CARLA** : `world.tick()` à 20 FPS pour une collecte et un comportement reproductibles, sans surprise temporelle.
- **Datasets et modèles non commités** : pour partager un dataset, voir la procédure dans la section [Stockage et partage](#stockage-et-partage).

---

**Questions ?** Lis d'abord ce README en entier, puis le README de ton module, puis [src/interfaces/README.md](src/interfaces/README.md). Si ce n'est toujours pas clair, demande sur le canal de l'équipe.
