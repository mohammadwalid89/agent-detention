"""Import tau2's own domain tool code (read-only clone) with shims for absent third-party packages.

tau2 is MIT-licensed code that is only imported; the clone is never written to. ``tau2/__init__``
eagerly imports its whole runner, none of which is needed to replay a domain tool, so a bare ``tau2``
package object pointing at the clone's directory is registered instead and only the submodules asked for
are imported. Shims in ``_shims`` replace the small third-party packages those submodules import.
The clone location comes from ``$TAU2_ROOT``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

from agent_detention import paths

HERE = Path(__file__).resolve().parent


def domain_dir() -> Path:
    return paths.tau2_root() / "data" / "tau2" / "domains"


def ensure_path() -> None:
    sys.dont_write_bytecode = True  # never write __pycache__ into the clone
    shims = str(HERE / "_shims")
    if shims not in sys.path:
        sys.path.insert(0, shims)
    if "tau2" not in sys.modules:
        pkg = types.ModuleType("tau2")
        pkg.__path__ = [str(paths.tau2_root() / "src" / "tau2")]
        sys.modules["tau2"] = pkg
