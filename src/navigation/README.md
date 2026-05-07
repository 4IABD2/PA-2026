# `src/navigation/` — GPS et planification de route

> **Owner** : Victor Dalet
> **Contrats** : [`RoutePlanner`, `CommandPlanner`](../interfaces/navigation_types.py)

---

## Responsabilité

Deux fonctions principales :

1. **Planification de route globale** : étant donné un point de départ et une destination, calculer un chemin sur le graphe de waypoints CARLA (algorithme A*)
2. **Commande haut niveau** : à chaque tick, fournir à l'IA centrale la prochaine intention de navigation (`LEFT`, `RIGHT`, `STRAIGHT`, `LANE_FOLLOW`)

L'IA centrale n'a pas accès à la route complète : elle n'utilise que la commande haut niveau courante.

## API publique

```python
from src.interfaces.navigation_types import (
    Waypoint, Route, HighLevelCommand,
    RoutePlanner, CommandPlanner,
)

class GpsRoutePlanner:
    def plan(self, start: Waypoint, destination: Waypoint) -> Route: ...

class GpsCommandPlanner:
    def next_command(self, vehicle_position: Waypoint, route: Route) -> HighLevelCommand: ...
```

## Squelette suggéré

```
src/navigation/
├── __init__.py
├── gps.py             ← code A* existant (à migrer depuis src/gps/gps.py)
├── route_planner.py   ← classe GpsRoutePlanner
├── command_planner.py ← classe GpsCommandPlanner (G/D/TT/Suivre)
└── README.md
```

## Migration depuis `src/gps/`

Le code actuel dans `src/gps/gps.py` est la base de ce module. Il contient déjà :

- L'extraction du graphe de waypoints CARLA
- Un A* manuel
- Un `get_control()` de bas niveau (à terme, à isoler dans un module de contrôle séparé ou à supprimer puisque l'IA centrale produira les contrôles)

À faire (par l'owner du module navigation) :

1. Déplacer `src/gps/gps.py` → `src/navigation/gps.py`
2. Refactorer en deux classes distinctes : `GpsRoutePlanner` et `GpsCommandPlanner`
3. Adapter les imports dans `main.py`
4. Mettre à jour `MatplotVisualizer.plot_road_network` qui consomme le graphe

Attention à ne pas casser l'`A*` existant qui marche : refactorer en gardant la logique intacte, juste en l'enveloppant dans les classes définies par les contrats.

## Calcul de la commande haut niveau

À chaque tick, on regarde le segment de route à venir (les N prochains waypoints) et on déduit la commande :

- Si la route à venir tourne significativement à gauche dans les ~30 m → `LEFT`
- Idem à droite → `RIGHT`
- Si on approche d'une intersection sans virage → `STRAIGHT`
- Sinon (route droite sans intersection imminente) → `LANE_FOLLOW`

Les seuils exacts (angle minimum, distance d'anticipation) sont à calibrer empiriquement.

## Validation et benchmarks

Dans [benchmarks/navigation/](../../benchmarks/navigation/) :

- **`smoke.py`** : vérifier que `GpsRoutePlanner.plan(start, dest)` retourne une route non vide entre deux waypoints connectés, que `GpsCommandPlanner.next_command` retourne `LEFT/RIGHT` aux bonnes positions, que les protocoles sont respectés. Exécutable sans CARLA en mockant les waypoints.
- **`benchmark.py`** : temps de calcul A* sur cartes de référence, longueur de route vs route optimale

Voir [benchmarks/README.md](../../benchmarks/README.md) pour la convention.

## Liens

- Code existant : `src/gps/gps.py`
- [CARLA Map and Waypoint API](https://carla.readthedocs.io/en/latest/core_map/)
- [Algorithme A* (Wikipedia)](https://en.wikipedia.org/wiki/A*_search_algorithm)
