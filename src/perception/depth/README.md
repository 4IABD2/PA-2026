# `src/perception/depth/` — Estimation de profondeur

> **Owner** : Franck Zhuang
> **Contrat** : [`DepthEstimator`](../../interfaces/perception_types.py)

---

## Responsabilité

À partir d'une image RGB, estimer la profondeur (distance en mètres) de chaque pixel. La sortie est une depth map dense `(H, W)` qui pourra être combinée avec les bounding boxes de YOLO pour obtenir la distance de chaque objet détecté.

## API publique

Doit exposer une classe (typiquement `DepthEstimator`) qui implémente le protocole homonyme :

```python
class DepthEstimator:
    def __init__(self, weights_path: str, device: str = "cpu") -> None: ...

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: RGB (H, W, 3), dtype uint8.
        Returns:
            Depth map (H, W) float32, distances en mètres.
        """
```

## Choix du modèle

Deux options principales, à arbitrer par l'owner :

- **MIDAS** (v3 small/large) — historique, robuste, bien documenté
- **Depth Anything v2 (small)** — plus récent (2024), meilleurs résultats sur scènes routières

Les deux fonctionnent en **monoculaire** (une seule caméra RGB suffit, pas besoin de stéréo).

## Squelette suggéré

```
src/perception/depth/
├── __init__.py
├── estimator.py       ← classe DepthEstimator
├── train.py           ← script d'entraînement / fine-tuning si nécessaire
├── weights/           ← gitignored, modèles
└── README.md
```

## Entraînement / validation

Le ground truth de CARLA (capteur `sensor.camera.depth`) sert pour :

- **Valider** la qualité du modèle pré-entraîné sur les scènes CARLA
- **Fine-tuner** si la performance n'est pas suffisante en zero-shot

Procédure de validation suggérée :

1. Générer N paires `(image_RGB, depth_GT)` via le module collector
2. Lancer le modèle sur les images RGB
3. Comparer aux depth GT via une métrique standard (RMSE, MAE, AbsRel)
4. Documenter les résultats ici

⚠️ Les capteurs depth de CARLA ne sont **jamais** utilisés en input en production. Voir le [README racine](../../../README.md) section "Conventions architecturales".

## Sortie attendue

- `(H, W)` float32, mêmes dimensions que l'image d'entrée
- Valeurs en mètres (pas en disparité, pas en pixels normalisés)
- Distance maximale plafonnée à une valeur raisonnable (ex : 100m)

Le code consommateur (fusion avec YOLO) suppose ce format.

## Performance attendue

- Inference < 100 ms par image sur GPU, < 500 ms sur CPU
- Erreur RMSE < 5m sur scènes CARLA (à valider)

## Validation

La calibration (`calibrate.py`) affiche le RMSE contre la depth GT CARLA au moment du fit. La validation fonctionnelle passe par l'intégration réelle (distances affichées dans les démos, cohérentes avec la scène).

## Liens

- [Depth Anything v2](https://depth-anything-v2.github.io/) — papier et code
- [MIDAS](https://github.com/isl-org/MiDaS) — modèle historique
- [CARLA depth sensor docs](https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera)
