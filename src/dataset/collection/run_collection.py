from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.dataset.collection.collector import DatasetCollector


def _default_output_dir(town: str, weather: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return Path("data/runs") / stamp / f"{town.lower()}_{weather.lower()}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CARLA dataset collection (run unique)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dir (default: data/runs/<datetime>/<town>_<weather>)",
    )
    parser.add_argument("--town", default="Town01", help="CARLA map (default: Town01)")
    parser.add_argument(
        "--weather",
        default="ClearNoon",
        help="CARLA weather preset (default: ClearNoon)",
    )
    parser.add_argument(
        "--duration", type=int, default=30, help="Duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--npcs", type=int, default=10, help="Number of NPC vehicles (default: 10)"
    )
    parser.add_argument(
        "--every-n-ticks",
        type=int,
        default=40,
        help="Capture every N ticks at 20 FPS (default: 40 = one frame every 2s)",
    )
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Par défaut None = spawn + trajectoire aléatoires "
        "(seed effectif enregistré dans metadata.json).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout client CARLA en s (defaut: 60 ; monter si grosses maps lentes)",
    )
    args = parser.parse_args()

    output_dir = args.output or _default_output_dir(args.town, args.weather)

    DatasetCollector(
        output_dir=output_dir,
        town=args.town,
        weather=args.weather,
        n_npc_vehicles=args.npcs,
        duration_sec=args.duration,
        capture_every_n_ticks=args.every_n_ticks,
        host=args.host,
        port=args.port,
        seed=args.seed,
        timeout_sec=args.timeout,
    ).run()


if __name__ == "__main__":
    main()
