"""The AgentAbstain pair index and the pre-registered compatibility sample.

The index is ``src/configs/tasks_preview.yaml`` in the clone: 526 entries = 263 act/abstain pairs
over 8 scenarios.  (``tasks.yaml``, which the README calls "the full benchmark", lists only the 10
``missing_critical_parameter`` pairs ``001``-``010``; the preview list is the 263-pair index.)
Nothing here needs the dataset download.
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import SCENARIOS

INDEX_REL = "src/configs/tasks_preview.yaml"
SAMPLE_SEED = 20261118  # fixed before any data is read
SAMPLE_N = 20


@dataclass(frozen=True)
class PairRef:
    category: str
    task_id: str

    @property
    def key(self) -> str:
        return f"{self.category}/{self.task_id}"


class IndexError_(ValueError):
    pass


def load_index(clone: str | Path | None = None) -> list[PairRef]:
    """All pairs, in index order; fails loudly on a duplicate, a half pair or an unknown scenario.

    ``clone`` defaults to ``$AGENTABSTAIN_REPO`` (the AgentAbstain code repository)."""
    path = Path(clone or os.environ.get("AGENTABSTAIN_REPO") or "") / INDEX_REL
    if not path.is_file():
        raise SystemExit(f"{path} not found; set AGENTABSTAIN_REPO (see scripts/fetch_data.py)")
    tasks = yaml.safe_load(path.read_text())["tasks"]
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    order: list[tuple[str, str]] = []
    for entry in tasks:
        key = (str(entry["category"]), str(entry["task_id"]))
        kind = str(entry["task_type"])
        if kind not in ("act", "abstain"):
            raise IndexError_(f"{key}: task_type {kind!r}")
        if kind in seen[key]:
            raise IndexError_(f"{key}: duplicate {kind} entry")
        if not seen[key]:
            order.append(key)
        seen[key].add(kind)
    for key, kinds in seen.items():
        if kinds != {"act", "abstain"}:
            raise IndexError_(f"{key}: not a complete pair ({sorted(kinds)})")
        if key[0] not in SCENARIOS:
            raise IndexError_(f"{key}: unknown scenario {key[0]!r}")
    return [PairRef(c, t) for c, t in order]


def counts(pairs: list[PairRef]) -> dict[str, int]:
    return dict(sorted(Counter(p.category for p in pairs).items()))


def quotas(pairs: list[PairRef], n: int = SAMPLE_N) -> dict[str, int]:
    """Equal share per scenario; the remainder goes to the largest scenarios (ties: alphabetical)."""
    size = counts(pairs)
    base, extra = divmod(n, len(size))
    ranked = sorted(size, key=lambda c: (-size[c], c))
    return {c: base + (1 if c in ranked[:extra] else 0) for c in sorted(size)}


def _rank(seed: int, p: PairRef) -> str:
    return hashlib.sha256(f"{seed}|{p.category}|{p.task_id}".encode()).hexdigest()


def eligible_sample(pairs: list[PairRef], eligible: set[str], n: int = SAMPLE_N, seed: int = SAMPLE_SEED) -> list[PairRef]:
    """The pre-registered sample rule restricted to pairs that can be adapted (``eligible`` = set of ``PairRef.key``).

    Same seed, ranking and per-scenario quotas as ``sample_pairs``; a pre-registered pair that is eligible stays in
    (filtering keeps relative order), an ineligible one is replaced by the next-ranked eligible pair of its scenario.
    Defined after the data showed 13 of 20 pre-registered pairs have no commit action (informational); applied before
    any control or live run."""
    quota = quotas(pairs, n)
    out: list[PairRef] = []
    for category in sorted(quota):
        pool = sorted((p for p in pairs if p.category == category and p.key in eligible), key=lambda p: _rank(seed, p))
        out.extend(pool[: quota[category]])
    return out


def sample_pairs(pairs: list[PairRef], n: int = SAMPLE_N, seed: int = SAMPLE_SEED) -> list[PairRef]:
    """Seeded, stratified over the 8 scenarios (rule fixed before the data was read)."""
    quota = quotas(pairs, n)
    out: list[PairRef] = []
    for category in sorted(quota):
        pool = sorted((p for p in pairs if p.category == category), key=lambda p: _rank(seed, p))
        out.extend(pool[: quota[category]])
    return out
