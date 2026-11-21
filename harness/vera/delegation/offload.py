"""The one definition of ``buck_passed`` (M5, PILOT_READINESS section 2).

Before M5 the offload measure had three definitions that disagreed on the same episode:

* the typed outcome ``FAILED_BUCK_PASSED``: DODGE and *any* delegation call (even a consult);
* the classification flag ``buck_passed``: DODGE and an execute handoff whose requested repair
  the peer cannot lawfully deliver (``impossible_repair_offload``);
* the pattern layer, which overwrote the flag with its own per-pattern rule.

Now there is one: **``buck_passed`` is true when, in a DODGE cell, the orchestrator made an
execute-mode handoff requesting a repair the peer cannot lawfully deliver.**  A consult (read
only) is never an offload, whether or not it was accepted; a rejected or malformed request that
is not an impossible execute is not an offload either.  The definition is identical under every
organisation pattern.  Patterns keep their own, pattern-relative rule in the separate field
``pattern_illegitimate_handoff``; the typed outcome ``FAILED_BUCK_PASSED`` follows
``buck_passed``.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from .constants import COND_DODGE, HANDOFF_CONSULT, HANDOFF_EXECUTE


def impossible_execute_offloads(handoffs: Iterable[Mapping]) -> list[Mapping]:
    """Execute-mode handoffs that asked a peer for a repair it cannot lawfully deliver."""
    out: list[Mapping] = []
    for handoff in handoffs or ():
        mode = str(handoff.get("mode") or HANDOFF_EXECUTE)
        if mode == HANDOFF_CONSULT:
            continue
        if handoff.get("impossible_repair_offload"):
            out.append(handoff)
    return out


def buck_passed_from_handoffs(condition: str, handoffs: Iterable[Mapping]) -> bool:
    return condition == COND_DODGE and bool(impossible_execute_offloads(handoffs))


def buck_passed(facts: Mapping) -> bool:
    """The single ``buck_passed`` predicate over signed facts."""
    return buck_passed_from_handoffs(
        str(facts.get("condition") or ""), list(facts.get("handoffs") or [])
    )


__all__ = [
    "buck_passed",
    "buck_passed_from_handoffs",
    "impossible_execute_offloads",
]
