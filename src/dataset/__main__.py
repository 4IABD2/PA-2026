"""Point d'entrée unifié du module dataset.

Usage :
    uv run -m src.dataset <command> [options]

Commands :
    collect         Collecte une run unique (1 map × 1 météo)
    collect-multi   Collecte plusieurs runs (maps × météos)
    enrich          Résout la couleur des feux d'une run (labels enrichis)
    inspect-run     Diagnostic d'une run (compteurs par classe, frame)
    inspect-signs   Compte les panneaux de signalisation par map CARLA

Aide d'une sous-commande :
    uv run -m src.dataset collect-multi --help
"""

from __future__ import annotations

import importlib
import sys

_COMMANDS = {
    "collect": "src.dataset.collection.run_collection",
    "collect-multi": "src.dataset.collection.collect_multi",
    "enrich": "src.dataset.labeling.enrich_labels",
    "inspect-run": "src.dataset.diagnostics.inspect_run",
    "inspect-signs": "src.dataset.diagnostics.inspect_signs",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        raise SystemExit(0 if len(sys.argv) >= 2 else 2)

    cmd = sys.argv[1]
    module_name = _COMMANDS.get(cmd)
    if module_name is None:
        print(f"Commande inconnue : {cmd!r}\n")
        print(__doc__)
        raise SystemExit(2)

    module = importlib.import_module(module_name)
    # Délègue : la sous-commande devient argv[0] pour l'argparse du module visé.
    sys.argv = [f"{sys.argv[0]} {cmd}", *sys.argv[2:]]
    module.main()


if __name__ == "__main__":
    main()
