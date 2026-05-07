# `src/orchestration/` — Boucle temps réel et intégration CARLA

> **Owner** : collectif (à porter ensemble)
> **Statut** : remplacera progressivement le `main.py` actuel

---

## Responsabilité

Orchestrer la boucle principale du véhicule autonome dans CARLA :

1. Initialiser CARLA (connexion, monde, mode synchrone)
2. Spawner le véhicule ego, configurer la caméra frontale (et les capteurs GT pour la collecte)
3. Instancier les modules de perception, navigation et IA centrale
4. À chaque tick :
   - Récupérer la frame courante
   - Demander à chaque module de perception sa sortie
   - Construire la `SceneState`
   - Demander à l'IA centrale le `ControlOutput`
   - Appliquer au véhicule via `apply_control`
   - `world.tick()`
5. Gérer la fermeture propre (destruction des actors, restauration des settings CARLA)

## Squelette suggéré

```
src/orchestration/
├── __init__.py
├── carla_runtime.py    ← Setup CARLA + capteurs (réutilisable par collector et inference)
├── pipeline.py         ← Classe Pipeline, boucle principale paramétrable
└── README.md
```

## API publique

```python
from src.orchestration.pipeline import Pipeline
from src.interfaces.perception_types import ObjectDetector, DepthEstimator, LaneDetector
from src.interfaces.ai_types import SceneState, ControlOutput

class Pipeline:
    def __init__(
        self,
        object_detector: ObjectDetector,
        depth_estimator: DepthEstimator,
        lane_detector: LaneDetector,
        ai_decision,  # callable: (SceneState) -> ControlOutput
        host: str = "localhost",
        port: int = 2000,
    ) -> None: ...

    def run(self, max_ticks: int | None = None) -> None: ...
```

Cette API permet de **passer n'importe quelle implémentation** des protocoles : modèles réels, stubs CARLA GT, mocks pour les tests. Le code de la boucle ne change pas.

## Pourquoi un module dédié ?

Le `main.py` actuel mélange initialisation CARLA, gestion des routes, application des contrôles, gestion du training. C'est lisible pour une démo mais difficile à réutiliser. En séparant l'orchestration dans son propre module :

- On peut écrire plusieurs scripts (`main.py`, `collect.py`, `demo.py`) qui réutilisent la même classe `Pipeline`
- On peut tester l'orchestration en mockant CARLA
- On peut paramétrer la boucle (avec/sans collecte, avec/sans HUD) sans dupliquer du code

## Migration depuis `main.py`

`main.py` actuel devra à terme se résumer à :

```python
from src.orchestration.pipeline import Pipeline
from src.perception.yolo.detector import YoloDetector
# ...

def main():
    pipeline = Pipeline(
        object_detector=YoloDetector("weights/best.pt"),
        depth_estimator=...,
        lane_detector=...,
        ai_decision=...,
    )
    pipeline.run()

if __name__ == "__main__":
    main()
```

À faire progressivement à mesure que les modules deviennent disponibles. Pas urgent, mais à garder en tête.

## Validation et benchmarks

Dans [benchmarks/pipeline/](../../benchmarks/pipeline/) :

- **`benchmark.py`** : FPS global et latence end-to-end du pipeline complet (capture → perception → ai → control), mesurés en CARLA réel

Pour valider la logique de fusion (construction de `SceneState`) sans CARLA, on utilise des mocks de modules dans le smoke test correspondant.

L'orchestration en boucle CARLA réelle se valide manuellement en lançant `main.py` localement.

## Liens

- [CARLA Synchronous mode](https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/)
- [CARLA Sensors](https://carla.readthedocs.io/en/latest/core_sensors/)
