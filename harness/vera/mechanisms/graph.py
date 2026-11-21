"""Actionable state-transition graphs: canonical hash and reference-policy simulation.

A mechanism's `stg` must be actionable, not prose: nodes carry predicate sets, edges
carry an action from the permitted list plus required predicates. Two distinctness tests
use this module:
  * structural  - canonical hash (blocks isomorphic copies)
  * behavioral  - reference-policy outcome signature (blocks cosmetic rewordings)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..canonical import canonical_json, sha256_text
from ..errors import ManifestError

REFERENCE_POLICIES = ("HEURISTIC", "ALWAYS_ACT", "ABSTAIN_FIRST")
_POLICY_ORDER = {"HEURISTIC": 0, "ALWAYS_ACT": 1, "ABSTAIN_FIRST": 2}
ABSTAIN_ACTIONS = ("abstain", "escalate", "request_approval")


@dataclass(frozen=True)
class Node:
    id: str
    predicates: frozenset[str]
    postconditions_met: frozenset[str]


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    action: str
    requires: frozenset[str]
    preferred: bool = False


def _nodes(stg: dict) -> dict[str, Node]:
    out: dict[str, Node] = {}
    for n in stg["nodes"]:
        if n["id"] in out:
            raise ManifestError(f"duplicate STG node id {n['id']!r}")
        out[n["id"]] = Node(
            id=n["id"],
            predicates=frozenset(n.get("predicates", [])),
            postconditions_met=frozenset(n.get("postconditions_met", [])),
        )
    if not out:
        raise ManifestError("STG has no nodes")
    return out


def _edges(stg: dict) -> list[Edge]:
    nodes = _nodes(stg)
    out = []
    for e in stg["edges"]:
        if e["from"] not in nodes or e["to"] not in nodes:
            raise ManifestError(f"edge references unknown node: {e}")
        out.append(
            Edge(
                source=e["from"],
                target=e["to"],
                action=e["action"],
                requires=frozenset(e.get("requires", [])),
                preferred=bool(e.get("preferred", False)),
            )
        )
    return out


def canonical_stg_hash(stg: dict) -> str:
    """Isomorphism-sensitive canonical hash (node ids normalised by sorted predicates)."""
    nodes = _nodes(stg)
    edges = _edges(stg)
    node_key = {
        n.id: canonical_json({"p": sorted(n.predicates), "po": sorted(n.postconditions_met)})
        for n in nodes.values()
    }
    payload = {
        "nodes": sorted(node_key.values()),
        "edges": sorted(
            canonical_json(
                {
                    "from": node_key[e.source],
                    "to": node_key[e.target],
                    "a": e.action,
                    "r": sorted(e.requires),
                    "p": e.preferred,
                }
            )
            for e in edges
        ),
    }
    return sha256_text(canonical_json(payload))


def _applicable(edges: list[Edge], node_id: str, active: set[str], policy: str) -> Edge | None:
    options = [e for e in edges if e.source == node_id and e.requires <= active]
    if not options:
        return None
    if policy == "HEURISTIC":
        preferred = [e for e in options if e.preferred]
        return sorted(preferred or options, key=lambda e: e.action)[0]
    if policy == "ALWAYS_ACT":
        return sorted(options, key=lambda e: (e.action, e.target))[0]
    if policy == "ABSTAIN_FIRST":
        abstain = [e for e in options if e.action in ABSTAIN_ACTIONS]
        return sorted(abstain, key=lambda e: e.action)[0] if abstain else None
    raise ManifestError(f"unknown reference policy {policy!r}")


def simulate(stg: dict, *, extra_predicates: tuple[str, ...] = (), policy: str = "HEURISTIC",
             required_postconditions: tuple[str, ...] = (), max_steps: int = 8) -> str:
    """Return PASS | PARTIAL | FAIL | INVALID for one reference policy on one variant."""
    nodes = _nodes(stg)
    edges = _edges(stg)
    entry = stg.get("entry", "s0")
    if entry not in nodes:
        raise ManifestError(f"entry node {entry!r} does not exist")
    active = set(nodes[entry].predicates) | set(extra_predicates)
    met: set[str] = set(nodes[entry].postconditions_met)
    current = entry
    steps = 0
    while steps < max_steps:
        edge = _applicable(edges, current, active, policy)
        if edge is None:
            break
        current = edge.target
        active |= set(nodes[current].predicates)
        met |= set(nodes[current].postconditions_met)
        steps += 1
    required = set(required_postconditions) or {
        p for n in nodes.values() for p in n.postconditions_met
    }
    if required and required <= met:
        return "PASS"
    if met:
        return "PARTIAL"
    return "FAIL" if steps else "INVALID"


def behavioral_signature(mechanism: dict) -> tuple[str, ...]:
    """One outcome per variant for each reference policy; the order is fixed."""
    required = tuple(mechanism.get("required_postconditions", []))
    sig: list[str] = []
    for variant in mechanism["variants"]:
        extra = tuple(variant.get("active_predicates", []))
        for policy in REFERENCE_POLICIES:
            sig.append(
                f"{policy}:"
                + simulate(
                    mechanism["stg"],
                    extra_predicates=extra,
                    policy=policy,
                    required_postconditions=required,
                )
            )
    return tuple(sig)


def signature_distance(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    """Number of differing outcome entries (0 => behaviorally indistinguishable).

    Entries are `POLICY:OUTCOME` per variant in a fixed order, so mechanism ids and
    cosmetic wording cannot make two identical behaviours look different.
    """
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for x, y in zip(a, b) if x != y)


def policy_order(policy: str) -> int:
    return _POLICY_ORDER[policy]
