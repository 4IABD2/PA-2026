"""Load a specific CARLA map on the server.

Usage:
    uv run python3 scripts/load_town.py --host <carla-ip> --town Town02
"""

from __future__ import annotations

import argparse

import carla


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a specific CARLA map on the server."
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", required=True, help="e.g. Town02")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    print(f"Loading {args.town} on {args.host}:{args.port} (this can take a moment)...")
    client.load_world(args.town)
    print(f"Map loaded: {args.town}")


if __name__ == "__main__":
    main()
