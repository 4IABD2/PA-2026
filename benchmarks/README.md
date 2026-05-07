# Benchmarks — Validation et mesure de performance par module

> Ce qui compte vraiment pour le projet : que les contrats soient respectés (sinon rien ne s'intègre) et combien de FPS / mAP / RMSE / taux de succès on obtient. Les deux se font ici.

---

## Pourquoi pas un dossier `tests/` séparé ?

Les modules de ce projet sont en majorité des modèles ML / CV. Les tests unitaires classiques apportent peu sur ce genre de code. Ce qui compte :

1. **Validation du contrat** (smoke) : ton détecteur retourne-t-il bien `list[DetectedObject]` avec des bbox dans `[0, image_size]` ?
2. **Mesure de performance** (benchmark) : quelle est la mAP / RMSE / latence / FPS ?

On regroupe les deux ici, par module. Pas de dossier `tests/` séparé.

## Structure

```
benchmarks/
├── README.md                       ← ce fichier
├── perception/
│   ├── yolo/
│   │   ├── smoke.py                ← validation contrat (rapide, lance-le souvent)
│   │   ├── benchmark.py            ← mesure de perf (lent, lance-le quand tu changes le modèle)
│   │   └── results/                ← gitignored, .json/.csv des runs
│   ├── depth/
│   │   ├── smoke.py
│   │   ├── benchmark.py
│   │   └── results/
│   └── lanes/
│       ├── smoke.py
│       └── benchmark.py
├── navigation/
│   ├── smoke.py
│   └── benchmark.py
├── ai/
│   ├── smoke.py
│   └── benchmark.py
└── pipeline/
    └── benchmark.py                ← FPS et latence end-to-end
```

## Convention pour `smoke.py`

Validation rapide qu'on peut lancer en CI (sans dataset, sans CARLA). Vérifie que l'implémentation respecte le contrat de [src/interfaces/](../src/interfaces/).

Exemple :

```python
# benchmarks/perception/yolo/smoke.py
import numpy as np
from src.perception.yolo.detector import YoloDetector
from src.interfaces.perception_types import DetectedObject

def test_detector_respects_contract():
    detector = YoloDetector(weights_path="benchmarks/perception/yolo/fixtures/tiny.pt")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(image)
    assert isinstance(result, list)
    for obj in result:
        assert isinstance(obj, DetectedObject)
        x1, y1, x2, y2 = obj.bbox
        assert 0 <= x1 < x2 <= image.shape[1]
        assert 0 <= y1 < y2 <= image.shape[0]
        assert 0.0 <= obj.confidence <= 1.0
```

Lancement : `uv run -m pytest benchmarks/` (pytest découvre tous les `smoke.py` automatiquement si on les nomme `test_*.py` ou si on les liste).

Tu peux aussi structurer en `pytest` : nommer les fichiers `test_smoke.py`, ou utiliser un seul `smoke.py` qui définit des fonctions `test_*`.

## Convention pour `benchmark.py`

Un script CLI qui mesure les performances réelles :

1. Charge le module à benchmarker (vrai modèle, pas stub)
2. Charge un dataset de test (depuis `data/runs/...`)
3. Exécute N inférences, mesure latence + métriques
4. Écrit les résultats dans `results/<YYYY-MM-DD>_<commit_sha>.json` (gitignored)
5. Affiche un résumé en console

Exemple d'invocation :

```bash
uv run -m benchmarks.perception.yolo.benchmark \
    --weights checkpoints/yolo_v1/best.pt \
    --dataset data/runs/2026-05-09_town01_clear \
    --n_samples 1000
```

## Format des résultats (`results/<run>.json`)

```json
{
  "run_id": "2026-05-09_yolo_v1_eval",
  "commit_sha": "abc1234",
  "dataset": "data/runs/2026-05-09_town01_clear",
  "n_samples": 1000,
  "metrics": {
    "mAP_50": 0.78,
    "mAP_50_95": 0.52,
    "latency_p50_ms": 42,
    "latency_p95_ms": 67
  },
  "hardware": {
    "device": "cuda:0",
    "gpu_name": "NVIDIA RTX 3090"
  },
  "timestamp": "2026-05-09T14:32:18"
}
```

## Comment publier ses résultats

1. Lancer le benchmark
2. Vérifier le `.json` généré dans `results/`
3. Copier le résumé dans le `JOURNAL.md` du module (entrée datée)
4. Si c'est un nouveau record / changement significatif, mentionner dans le canal équipe

## Ne pas commiter les résultats

`benchmarks/*/results/` est gitignored. Les résultats bruts restent locaux. Ce qui doit arriver sur Git, c'est :

- Les scripts `smoke.py` et `benchmark.py` (forcément)
- Un résumé dans le `JOURNAL.md` du module concerné

## Métriques par module

| Module | Métriques smoke | Métriques benchmark |
|---|---|---|
| `perception/yolo/` | Forme des sorties (`list[DetectedObject]`, bbox normalisées) | mAP@50, mAP@50-95, latence p50/p95 |
| `perception/depth/` | Shape `(H, W)`, dtype float32, valeurs bornées | RMSE / MAE / AbsRel vs GT CARLA, latence |
| `perception/lanes/` | Type `LanesInfo`, offset dans `[-1, 1]` ou None | Précision détection ligne, latence |
| `navigation/` | Type `Route`, waypoints non vides | Temps de calcul A*, longueur vs optimale |
| `ai/` | `ControlOutput` valide (steer ∈ [-1,1], throttle/brake ∈ [0,1]) | Loss val, taux de succès trajet sans collision, FPS |
| `pipeline/` | — | FPS global, latence end-to-end |
