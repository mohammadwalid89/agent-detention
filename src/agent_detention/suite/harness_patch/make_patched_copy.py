"""Make the patched harness tree the suite runs against (the source tree is only read).

    python -m agent_detention.suite.harness_patch.make_patched_copy <harness_tree> <dest_dir>   # dest must not exist
    AD_TREE=<dest_dir> ...                                                                       # then point the suite at it

Copies everything except ``artifacts/`` and caches, then applies ``harness_v14b.patch`` (four files: per-variant
overrides in ``vera/pilot/instances.py``, the case-text overlay hooks in ``vera/delegation/case_text.py`` and
``adapters.py``, and the provider-refusal handling in ``runner_v13.py``). Needs the ``patch`` program.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PATCH = Path(__file__).resolve().parent / "harness_v14b.patch"


def make(source: str | Path, dest: str | Path) -> Path:
    dest = Path(dest)
    if dest.exists():
        raise SystemExit(f"{dest} exists; refusing to overwrite")
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns("artifacts", "__pycache__", ".pytest_cache"))
    subprocess.run(["patch", "-p1", "--quiet", "-i", str(PATCH)], cwd=dest, check=True)
    return dest


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    make(sys.argv[1], sys.argv[2])
