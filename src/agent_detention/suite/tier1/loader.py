"""Load one AgentAbstain pair from a downloaded dataset snapshot.

Layout (from the upstream ``src/types/BaseAgent.py`` and ``eval/loader.py``)::

    <data>/tasks/<category>/<task_id>/metadata.yaml
    <data>/tasks/<category>/<task_id>/{act,abstain}/task.yaml
    <data>/tasks/<category>/<task_id>/{act,abstain}/initial_states/<env>.json

``<data>`` is ``$AGENTABSTAIN_DATA`` (same variable as upstream).  The loader never invents a field:
anything the adapter needs that is absent raises ``PairSchemaError`` with the file and key, so a
schema surprise in the real download is a loud failure, not a silent default.

Every assumption the loader makes is listed in ``ASSUMPTIONS`` and enforced by a raise in ``load_pair``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .index import PairRef

ENV_VAR = "AGENTABSTAIN_DATA"
NEEDS = "scripts/fetch_data.py"

ASSUMPTIONS = (
    "task dir = <data>/tasks/<category>/<task_id>/<act|abstain>/ (task_id used verbatim; upstream zero-pads digits only)",
    "metadata.yaml sits one level above the variant dirs and has a list `environments`",
    "<variant>/initial_states/<env>.json exists for every env in metadata.environments",
    "task.yaml has `instruction` and `system_prompt` (upstream build_trajectory / TaskRunResult read both)",
    "task.yaml `critical_actions` is list[str] of namespaced tool names (strict upstream schema)",
    "act task.yaml has `execution_dag` (or legacy `expected_tool_sequence`); abstain has `abstention_trigger`",
    "optional: `available_tools`, `tool_overrides.broken_tools[{name,error}]`",
)


class DataMissing(FileNotFoundError):
    """The dataset snapshot (or a file in it) is not on disk."""


class PairSchemaError(ValueError):
    """A file exists but does not have the shape the upstream code (and this loader) expects."""


def data_root(root: str | Path | None = None) -> Path:
    value = root or os.environ.get(ENV_VAR)
    if not value:
        raise DataMissing(f"{ENV_VAR} is not set and no data root was given; see {NEEDS}")
    path = Path(value)
    if not (path / "tasks").is_dir():
        raise DataMissing(f"{path}/tasks does not exist; the dataset snapshot is not downloaded; see {NEEDS}")
    return path


@dataclass
class RawVariant:
    kind: str  # "act" | "abstain"
    task_yaml: dict
    initial_states: dict[str, dict]
    critical_actions: list[str]
    broken_tools: list[dict]
    available_tools: list[str] | None
    files: dict[str, str] = field(default_factory=dict)  # relative path -> sha256


@dataclass
class RawPair:
    ref: PairRef
    metadata: dict
    act: RawVariant
    abstain: RawVariant
    environments: list[str]
    files: dict[str, str] = field(default_factory=dict)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _need(path: Path) -> Path:
    if not path.exists():
        raise DataMissing(f"{path} is missing; see {NEEDS}")
    return path


def _variant(pair_dir: Path, kind: str, environments: list[str], rel_root: Path) -> RawVariant:
    vdir = _need(pair_dir / kind)
    task_path = _need(vdir / "task.yaml")
    task = yaml.safe_load(task_path.read_text())
    if not isinstance(task, dict):
        raise PairSchemaError(f"{task_path}: not a mapping")
    for key in ("instruction", "system_prompt"):
        if not isinstance(task.get(key), str) or not task[key].strip():
            raise PairSchemaError(f"{task_path}: missing or empty {key!r}")
    if kind == "act" and not ("execution_dag" in task or "expected_tool_sequence" in task):
        raise PairSchemaError(f"{task_path}: act task has neither execution_dag nor expected_tool_sequence")
    if kind == "abstain" and "abstention_trigger" not in task:
        raise PairSchemaError(f"{task_path}: abstain task has no abstention_trigger")
    critical = task.get("critical_actions") or []
    if not isinstance(critical, list):
        raise PairSchemaError(f"{task_path}: critical_actions is {type(critical).__name__}, expected list[str]")
    if any(not isinstance(c, str) for c in critical):
        raise PairSchemaError(
            f"{task_path}: critical_actions has non-string entries (legacy dict shape); "
            "upstream evaluates these with an always-pass commit check, so the pair is not usable"
        )
    files = {str(task_path.relative_to(rel_root)): _sha(task_path)}
    states: dict[str, dict] = {}
    for env in environments:
        state_path = _need(vdir / "initial_states" / f"{env}.json")
        states[env] = json.loads(state_path.read_text())
        files[str(state_path.relative_to(rel_root))] = _sha(state_path)
    overrides = task.get("tool_overrides") or {}
    broken = list(overrides.get("broken_tools") or [])
    avail = task.get("available_tools")
    names = None if avail is None else [t["name"] if isinstance(t, dict) else str(t) for t in avail]
    return RawVariant(kind, task, states, [str(c) for c in critical], broken, names, files)


def load_pair(ref: PairRef, root: str | Path | None = None) -> RawPair:
    base = data_root(root)
    pair_dir = _need(base / "tasks" / ref.category / ref.task_id)
    meta_path = _need(pair_dir / "metadata.yaml")
    metadata = yaml.safe_load(meta_path.read_text())
    envs = metadata.get("environments") if isinstance(metadata, dict) else None
    if not envs or not isinstance(envs, list):
        raise PairSchemaError(f"{meta_path}: no list `environments`")
    act = _variant(pair_dir, "act", [str(e) for e in envs], base)
    abstain = _variant(pair_dir, "abstain", [str(e) for e in envs], base)
    files = {str(meta_path.relative_to(base)): _sha(meta_path), **act.files, **abstain.files}
    return RawPair(ref, metadata, act, abstain, [str(e) for e in envs], files)
