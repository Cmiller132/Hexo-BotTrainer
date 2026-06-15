"""Thin launcher for the hexfield BC prefit (path-shimmed; no venv installs).

Run (WSL): /root/.venvs/hexgt-build/bin/python scripts/_hexfield_prefit.py \
    --data data/hexfield_bootstrap --out runs/hexfield_bc_1 --epochs 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "hexfield" / "python"))

from hexfield.prefit import main

if __name__ == "__main__":
    main()
