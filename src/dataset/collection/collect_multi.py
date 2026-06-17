"""CLI : collecte CARLA de plusieurs runs en boucle (maps × météos).

Calcule la durée d'une run depuis le nombre de frames souhaité (en supposant
le défaut ``capture_every_n_ticks=40`` du collector → 1 frame toutes les 2 s).

Sortie : un dossier de session horodaté, un sous-dossier par run.
    data/runs/<datetime>/<town>_<weather>/

Usage :
    uv run -m src.dataset collect-multi \\
        --maps Town01,Town03,Town04,Town05,Town10HD \\
        --weathers ClearNoon,CloudyNoon,WetNoon \\
        --frames-per-run 500 --npcs 30 --enrich
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

from src.dataset.collector import DatasetCollector
from src.dataset.enrich_labels import process_run as enrich_run

_DEFAULT_CAPTURE_EVERY_N_TICKS = 40
_CARLA_FPS = 20


def _session_dir() -> Path:
    """Dossier principal d'une session de collecte, horodaté."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return Path("data/runs") / stamp


def _run_dir(session_dir: Path, town: str, weather: str) -> Path:
    """Sous-dossier d'une run (map × météo) sous la session."""
    return session_dir / f"{town.lower()}_{weather.lower()}"


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
        help="Liste de maps séparées par virgules (ex: Town01,Town04,Town05)",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed de base. Par défaut None = seed aléatoire DIFFÉRENT par run "
        "(spawn + trajectoire variés). Si fourni, run i utilise seed+i "
        "(reproductible, distinct par run). Le seed effectif est dans metadata.json.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout client CARLA en s (defaut: 60 ; monter si grosses maps lentes)",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Lance enrich_labels après chaque collecte (couleur des feux)",
    )
    args = parser.parse_args()

    maps = [m.strip() for m in args.maps.split(",") if m.strip()]
    weathers = [w.strip() for w in args.weathers.split(",") if w.strip()]
    if not maps or not weathers:
        raise SystemExit("Au moins une map et une weather requises.")

    duration_sec = _duration_for_frames(args.frames_per_run, args.every_n_ticks)
    session_dir = _session_dir()
    print(
        f"Plan : {len(maps)} maps x {len(weathers)} weathers = {len(maps) * len(weathers)} runs, "
        f"~{duration_sec}s chacune ({args.frames_per_run} frames cibles)."
    )
    print(f"Session : {session_dir}")

    failures: list[tuple[str, str, str]] = []
    run_index = 0
    for town in maps:
        for weather in weathers:
            out_dir = _run_dir(session_dir, town, weather)
            # Seed distinct par run pour varier spawn + trajectoire (sinon même
            # map/seed → mêmes images). Aléatoire si --seed absent, sinon
            # déterministe (base + index) pour rester reproductible.
            if args.seed is None:
                run_seed = random.randrange(2**31)
            else:
                run_seed = args.seed + run_index
            run_index += 1
            print()
            print(f"=== {town} / {weather} -> {out_dir} (seed={run_seed}) ===")
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
                    seed=run_seed,
                    timeout_sec=args.timeout,
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
                        f"   Feux classifies: "
                        f"{stats['tl_red'] + stats['tl_yellow'] + stats['tl_green']}/"
                        f"{stats['tl_in']}"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"!! Erreur enrichissement {out_dir.name} : {exc}",
                        file=sys.stderr,
                    )

    print()
    print(
        f"Termine : {len(maps) * len(weathers) - len(failures)} runs OK"
        f"{', ' + str(len(failures)) + ' echecs' if failures else ''}."
    )
    for t, w, err in failures:
        print(f"  echec {t}/{w}: {err}")


if __name__ == "__main__":
    main()
