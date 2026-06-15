import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "hexfield" / "python"))
import hexfield.evaluation  # noqa: F401  (import-checks the edited module)
from hexfield.config import parse_hexfield_config
c = parse_hexfield_config({"evaluation": {"eval_every": 5}})
print("eval_every =", c.evaluation.eval_every, "-> evaluation.py imports clean")
