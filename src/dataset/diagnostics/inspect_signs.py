"""Compte les acteurs de signalisation présents dans les maps CARLA.

Sert à : (1) choisir la map la plus dense en panneaux de vitesse pour tester,
(2) découvrir tous les types de panneaux disponibles (stop, yield, etc.) qu'on
pourrait labelliser en plus.

Les panneaux CARLA sont des acteurs ``traffic.*`` :
    traffic.speed_limit.30 / .40 / ... (limitations)
    traffic.stop                       (stop)
    traffic.yield                      (cédez-le-passage)
    traffic.traffic_light              (feux)
    traffic.unknown                    (autres)

Usage :
    # Map courante (déjà chargée sur le serveur)
    uv run -m src.dataset inspect-signs
    # Maps précises
    uv run -m src.dataset inspect-signs --maps Town01,Town04,Town05,Town10HD
    # Toutes les maps disponibles (lent : charge chaque map)
    uv run -m src.dataset inspect-signs --maps all
"""

from __future__ import annotations

import argparse
from collections import Counter

import carla


def _count_for_world(world: "carla.World") -> Counter:
    actors = world.get_actors()
    return Counter(a.type_id for a in actors if a.type_id.startswith("traffic."))


def _print_counts(town: str, counts: Counter) -> None:
    speed = {t: n for t, n in counts.items() if t.startswith("traffic.speed_limit")}
    n_speed = sum(speed.values())
    print(f"\n=== {town} : {n_speed} panneaux de vitesse ===")
    for tid, n in sorted(counts.items()):
        print(f"  {tid:<28} {n}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compte les acteurs de signalisation par map CARLA."
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--maps",
        type=str,
        default=None,
        help="Maps séparées par virgules, 'all', ou rien (map courante).",
    )
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    if args.maps is None:
        world = client.get_world()
        town = world.get_map().name.split("/")[-1]
        _print_counts(town, _count_for_world(world))
        return

    if args.maps.strip().lower() == "all":
        maps = [m.split("/")[-1] for m in client.get_available_maps()]
        # dédoublonne en gardant l'ordre
        seen = set()
        maps = [m for m in maps if not (m in seen or seen.add(m))]
    else:
        maps = [m.strip() for m in args.maps.split(",") if m.strip()]

    totals: list[tuple[str, int]] = []
    for town in maps:
        try:
            world = client.load_world(town)
        except Exception as exc:  # noqa: BLE001
            print(f"!! {town}: {exc}")
            continue
        counts = _count_for_world(world)
        _print_counts(town, counts)
        totals.append(
            (
                town,
                sum(
                    n for t, n in counts.items() if t.startswith("traffic.speed_limit")
                ),
            )
        )

    if totals:
        print("\n=== Classement panneaux de vitesse ===")
        for town, n in sorted(totals, key=lambda x: -x[1]):
            print(f"  {town:<14} {n}")


if __name__ == "__main__":
    main()
