import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
for p in (_root, _root / "botify"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
