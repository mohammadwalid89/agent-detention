"""Bridge to the harness tree (VERA), which is imported, never copied.

``AD_TREE`` must point at the harness tree with ``suite/harness_patch/harness_v14b.patch`` applied
(see ``harness_patch/make_patched_copy.py``). The scripted controls refuse to run on an unpatched tree.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def tree_path() -> Path:
    value = os.environ.get("AD_TREE")
    if not value:
        raise SystemExit("set AD_TREE to the patched harness tree")
    return Path(value)


def use_tree() -> Path:
    tree = tree_path()
    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    return tree


def is_patched() -> bool:
    use_tree()
    from vera.pilot import instances

    return hasattr(instances, "_specialise")
