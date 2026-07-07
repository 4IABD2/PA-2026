"""Preflight checks + guided launcher for Phase 1 RL training.

Checks CARLA connectivity, active map, GPU availability and YOLO weights
before handing off to scripts/run_rl_training.py. Meant for teammates running
a training session for the first time on their own machine: clone, `uv sync`,
run this script, follow the prompts.

Usage:
    uv run python3 launch_training.py [--host HOST] [--port PORT]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_EXPECTED_MAP = "Town02"
_YOLO_WEIGHTS = Path("src/perception/yolo/weights/best.pt")

_STEP_PRESETS: dict[str, int] = {
    "1": 100_000,
    "2": 300_000,
    "3": 500_000,
}


def _fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def _ask_yes_no(prompt: str, default_yes: bool) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    answer = input(f"{prompt} {suffix} ").strip().lower()
    if not answer:
        return default_yes
    return answer in ("y", "yes")


def check_carla(host: str, port: int) -> "carla.Client":
    import carla

    client = carla.Client(host, port)
    client.set_timeout(5.0)
    try:
        version = client.get_server_version()
    except RuntimeError:
        _fail(
            f"Cannot reach CARLA at {host}:{port}.\n"
            f"       Start the server first, e.g.:\n"
            f"       ./CarlaUE4.sh -RenderOffScreen -nosound -world-port={port}"
        )
    print(f"[OK]   CARLA server reachable (v{version}) at {host}:{port}")
    return client


def check_map(client: "carla.Client") -> None:
    world = client.get_world()
    current_map = world.get_map().name  # e.g. "Carla/Maps/Town02"
    if _EXPECTED_MAP in current_map:
        print(f"[OK]   Map loaded: {current_map}")
        return

    print(f"[WARN] Current map is '{current_map}', expected '{_EXPECTED_MAP}'.")
    print("       The 13-scenario benchmark's spawn points only make sense on this map.")
    if _ask_yes_no(f"       Load {_EXPECTED_MAP} now?", default_yes=True):
        print(f"       Loading {_EXPECTED_MAP} (this can take a moment)...")
        client.load_world(_EXPECTED_MAP)
        print(f"[OK]   Map loaded: {_EXPECTED_MAP}")
    else:
        print("[WARN] Continuing with the current map — benchmark results will be meaningless.")


def check_gpu() -> None:
    import torch

    if torch.cuda.is_available():
        print(f"[OK]   GPU detected: {torch.cuda.get_device_name(0)}")
        return

    print("[WARN] No GPU detected — YOLO, Depth Anything v2 and YOLOPv2 will all run on CPU.")
    print("       This can make training 10-50x slower. Check your NVIDIA drivers / CUDA install.")
    if not _ask_yes_no("       Continue anyway?", default_yes=False):
        sys.exit(1)


def check_yolo_weights() -> None:
    if not _YOLO_WEIGHTS.exists():
        _fail(
            f"YOLO weights not found at {_YOLO_WEIGHTS}.\n"
            f"       This file should be committed in the repo — check your clone/pull."
        )
    size_mb = _YOLO_WEIGHTS.stat().st_size / 1e6
    print(f"[OK]   YOLO weights found ({size_mb:.1f} MB)")


def ask_timesteps() -> int:
    print()
    print("How many training steps?")
    print("  1) 100 000  — quick sanity check (~1-1.5h on a modern GPU)")
    print("  2) 300 000  — full run (~3-4h on a modern GPU)")
    print("  3) 500 000  — extended run (~5-7h on a modern GPU)")
    print("  4) custom")
    choice = input("Choice [1-4]: ").strip()
    if choice in _STEP_PRESETS:
        return _STEP_PRESETS[choice]
    if choice == "4":
        while True:
            raw = input("Enter number of timesteps: ").strip()
            if raw.isdigit() and int(raw) > 0:
                return int(raw)
            print("Please enter a positive integer.")
    _fail(f"Invalid choice: '{choice}'")


def ask_tag() -> str:
    tag = input("Run tag (used in the run folder name, e.g. 'ppo_v4') [ppo]: ").strip()
    return tag or "ppo"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()

    print("=== Preflight checks ===")
    client = check_carla(args.host, args.port)
    check_map(client)
    check_gpu()
    check_yolo_weights()
    print()

    timesteps = ask_timesteps()
    tag = ask_tag()

    cmd = [
        sys.executable, "scripts/run_rl_training.py",
        "--timesteps", str(timesteps),
        "--tag", tag,
        "--host", args.host,
        "--port", str(args.port),
    ]
    print(f"\nLaunching: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
