"""Bridge to the harness tree (VERA), which the suite imports but does not modify.

The tree is ``harness/`` at the root of this repository. ``AD_TREE`` overrides it, for example to run
against a copy. The scripted controls refuse a tree without the per-variant overrides in
``vera/pilot/instances.py`` (see ``harness/README.md``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_HARNESS = Path(__file__).resolve().parents[4] / "harness"


def tree_path() -> Path:
    value = os.environ.get("AD_TREE")
    if value:
        return Path(value)
    if (REPO_HARNESS / "vera" / "__init__.py").is_file():
        return REPO_HARNESS
    raise SystemExit("harness/ not found next to src/; set AD_TREE to the harness tree")


def use_tree() -> Path:
    tree = tree_path()
    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    return tree


def is_patched() -> bool:
    use_tree()
    from vera.pilot import instances

    return hasattr(instances, "_specialise")
