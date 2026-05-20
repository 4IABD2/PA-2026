"""Lance plusieurs collectes CARLA en boucle sur (map × météo).

Calcule la durée d'une run depuis le nombre de frames souhaité (en supposant
le défaut ``capture_every_n_ticks=40`` du collector → 1 frame toutes les 2 s).

Usage :
    uv run -m src.dataset.collect_multi \\
        --maps Town01,Town03 \\
        --weathers ClearNoon,CloudyNoon \\
        --frames-per-run 500 \\
        --npcs 30 \\
        --enrich
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.dataset.collector import DatasetCollector
from src.dataset.enrich_labels import process_run as enrich_run


_DEFAULT_CAPTURE_EVERY_N_TICKS = 40
_CARLA_FPS = 20


def _output_dir(town: str, weather: str) -> Path:
    date = datetime.now().strftime("%Y-%m-%d")
    return Path("data/runs") / f"{date}_{town.lower()}_{weather.lower()}"


def _duration_for_frames(frames: int, every_n_ticks: int) -> int:
    return int(frames * every_n_ticks / _CARLA_FPS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lance plusieurs collectes CARLA (loop maps × weathers)."
    )
    parser.add_argument(
        "--maps",
        type=str,
        required=True,
        help="Liste de maps séparées par virgules (ex: Town01,Town03,Town05)",
    )
    parser.add_argument(
        "--weathers",
        type=str,
        default="ClearNoon",
        help="Liste de météos séparées par virgules (defaut: ClearNoon)",
    )
    parser.add_argument(
        "--frames-per-run",
        type=int,
        default=300,
        help="Nombre de frames cibles par run (defaut: 300 = ~10 min)",
    )
    parser.add_argument(
        "--every-n-ticks",
        type=int,
        default=_DEFAULT_CAPTURE_EVERY_N_TICKS,
        help=f"Capture toutes les N ticks à 20 FPS (defaut: {_DEFAULT_CAPTURE_EVERY_N_TICKS} = 2s)",
    )
    parser.add_argument(
        "--npcs", type=int, default=30, help="Nombre de NPC vehicles (defaut: 30)"
    )
    parser.add_argument("--host", default="localhost", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Lance enrich_labels après chaque collecte (TL color + speed values)",
    )
    args = parser.parse_args()

    maps = [m.strip() for m in args.maps.split(",") if m.strip()]
    weathers = [w.strip() for w in args.weathers.split(",") if w.strip()]
    if not maps or not weathers:
        raise SystemExit("Au moins une map et une weather requises.")

    duration_sec = _duration_for_frames(args.frames_per_run, args.every_n_ticks)
    print(
        f"Plan : {len(maps)} maps x {len(weathers)} weathers = {len(maps) * len(weathers)} runs, "
        f"~{duration_sec}s chacune ({args.frames_per_run} frames cibles)."
    )

    failures: list[tuple[str, str, str]] = []
    for town in maps:
        for weather in weathers:
            out_dir = _output_dir(town, weather)
            print()
            print(f"=== {town} / {weather} -> {out_dir} ===")
            try:
                DatasetCollector(
                    output_dir=out_dir,
                    town=town,
                    weather=weather,
                    n_npc_vehicles=args.npcs,
                    duration_sec=duration_sec,
                    capture_every_n_ticks=args.every_n_ticks,
                    host=args.host,
                    port=args.port,
                    seed=args.seed,
                ).run()
            except Exception as exc:  # noqa: BLE001
                print(f"!! Erreur collecte {town}/{weather} : {exc}", file=sys.stderr)
                failures.append((town, weather, str(exc)))
                continue

            if args.enrich:
                print(f"=== Enrichissement de {out_dir.name} ===")
                try:
                    stats = enrich_run(out_dir)
                    print(
                        f"   TL classifies: {stats['tl_red'] + stats['tl_yellow'] + stats['tl_green']}/"
                        f"{stats['tl_in']}, "
                        f"Signs classifies: {stats['sign_classified']}/{stats['sign_in']}"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"!! Erreur enrichissement {out_dir.name} : {exc}",
                        file=sys.stderr,
                    )

    print()
    print(f"Termine : {len(maps) * len(weathers) - len(failures)} runs OK"
          f"{', ' + str(len(failures)) + ' echecs' if failures else ''}.")
    for t, w, err in failures:
        print(f"  echec {t}/{w}: {err}")


if __name__ == "__main__":
    main()
