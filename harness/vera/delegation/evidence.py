"""The case's evidence-read action (M4, PILOT_READINESS section 2).

Before M4 a DODGE cell whose graph had no read edge (39 of the 51 DODGE cells) gave the peer the
inert class ``observe``.  ``observe`` is not in any case vocabulary, so a read-only consult
naming it was always rejected, and the task text's "gather the evidence you need" had no action
behind it for the orchestrator either.  In protocol ``v14`` every case has one real, executable
evidence-read action:

1. a read edge the graph already has and that is not part of the required repair;
2. else a read-prefixed action the case already declares in its vocabulary but has no edge for
   (the world executes it as a no-effect read: it is authority-checked and logged like any
   step, and it changes no state);
3. else (the F7 transaction cases declare no read at all) the synthetic ``read_state``, added to
   that case's vocabulary.

``protocol="v13"`` is left byte-for-byte as it was: the preregistered condition digest, the
frozen assignment table and every v13 cell hash depend on it.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Iterable

PROTOCOL_V13 = "v13"
PROTOCOL_V14 = "v14"
PROTOCOLS = (PROTOCOL_V13, PROTOCOL_V14)

READ_PREFIXES = ("read_", "list_", "get_", "inspect_", "query_", "view_", "check_")
SYNTHETIC_EVIDENCE_READ = "read_state"


def choose_evidence_read(instance, required: Iterable[str] = ()) -> tuple[str, str]:
    """``(action, source)``; source is ``edge``, ``declared`` or ``synthetic``."""
    edge_actions = {edge["action"] for edge in instance.graph.get("edges", [])}
    required_set = set(required)
    for action in sorted(edge_actions):
        if action.startswith(READ_PREFIXES) and action not in required_set:
            return action, "edge"
    for action in instance.permitted_actions:
        if action.startswith(READ_PREFIXES) and action not in edge_actions:
            return action, "declared"
    return SYNTHETIC_EVIDENCE_READ, "synthetic"


def apply_evidence_read(instance, action: str):
    """A copy of ``instance`` that records its evidence-read action in its vocabulary."""
    vocabulary = list(instance.permitted_actions)
    if action not in vocabulary:
        vocabulary.append(action)
    return dataclasses.replace(
        instance,
        evidence_read=action,
        permitted_actions=vocabulary,
        graph=copy.deepcopy(instance.graph),
    )


__all__ = [
    "PROTOCOLS",
    "PROTOCOL_V13",
    "PROTOCOL_V14",
    "READ_PREFIXES",
    "SYNTHETIC_EVIDENCE_READ",
    "apply_evidence_read",
    "choose_evidence_read",
]
