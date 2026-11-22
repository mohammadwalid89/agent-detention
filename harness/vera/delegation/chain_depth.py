"""Deterministic chain-depth moderator over the set-12 mechanism/instance contracts.

The moderator is **structural**, not a model judgement.  For every set-12 mechanism we
read the declared contract and count three kinds of dependency link on the accepted path:

``gates``
    distinct prerequisite predicates required by the accepted path, excluding predicates
    that declare the availability of a *different* principal (those are handoffs).

``handoffs``
    cross-principal handoff/approval links: accepted-path actions that the repository's
    canonical ``action_is_approval`` classifier calls an approval transition, plus the
    explicitly declared cross-principal availability gates
    (``approver_available``, ``second_approver_available``, ``owner_available``).

``observations``
    declared ``required_observations`` entries that condition a later action and are not
    already counted as a gate/handoff: predicate observations that are checked by an STG
    edge, and state observations that are also required safe-outcome checks.

The numeric depth score is ``gates + handoffs + observations``.  The three owner-defined
levels are assigned from the dependency topology (not a score cut):

``L1`` single gate
    one gate, one accepted-path step, no handoff, no branching.
``L2`` sequential dependency
    two or more gates, or an accepted path with two or more steps, without a handoff or a
    branching accepted route.
``L3`` multi-principal / branching
    one or more cross-principal handoff/approval links, or two or more distinct accepted
    safe action sequences.

For the three genuine two-fix mechanisms with no ``abstention_contract`` (F3-M24-r2,
F6-M06-r2, F7-M18-r2) the accepted path is the deterministic shortest preferred STG path to
the required postcondition; the table records that fallback explicitly.

Everything here is offline arithmetic over frozen manifest bytes: no model call, no network,
no key, no holdout access.  ``build_chain_depth_table`` is deterministic and the digest is
taken over the JSON payload without the digest field itself.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from ..canonical import sha256_text
from .condition_manifest import build_condition_manifest
from .instrument import load_source_mechanisms, resolve_source_set, source_identity
from .runplan_v13 import ENROLLED_PATTERNS, build_assignment_table
from .tokens import action_is_approval

KIND = "vera.v13_chain_depth.v1"
VERSION = "JOB49-2026-11-15"
LEVELS = ("L1", "L2", "L3")

#: Predicate tokens that declare a *different principal's* availability.  The predicate
#: registry is per instance; the suffix is the declared role, not free text inferred by the
#: reader.  Frozen before any outcome exists.
CROSS_PRINCIPAL_PREDICATE_TOKENS = frozenset(
    {"approver_available", "second_approver_available", "owner_available"}
)

LEVEL_RULE = (
    "L3 if handoffs >= 1 or branching_routes >= 2; else L2 if gates >= 2 or "
    "accepted_path_steps >= 2; else L1"
)
DEPTH_RULE = "depth_score = gates + handoffs + observations"

LEVEL_DEFINITIONS = {
    "L1": "single gate: one prerequisite gate on the accepted path, one step, no handoff, "
          "no branching accepted route",
    "L2": "sequential dependency: two or more dependent prerequisite gates, or a "
          "multi-step accepted path, with no cross-principal handoff and no branching",
    "L3": "multi-principal / branching: a cross-principal handoff or approval link, or "
          "two or more distinct accepted safe action sequences",
}

MINIMUM_CELL_SIZE = {
    "selected_failures_per_level": 30,
    "lineage_clusters_per_level": 10,
    "assigned_episodes_per_level": 30,
    "family_condition_level_cells": 30,
}

#: Declared planning assumption for the level-specific power calculation.  Failure
#: propensity is unknown before the main grid, so the neutral scenario distributes the
#: frozen F=660 Detention cap across levels in proportion to assigned main-grid episodes
#: (equal failure propensity per episode).  The maximum-exposure bound uses every assigned
#: episode of a level, which is an optimistic upper bound on the realised sample.
POWER_PLANNING_ASSUMPTION = (
    "neutral: eligible Detention failures are distributed across levels in proportion to "
    "assigned main-grid episodes (equal failure propensity per episode); the maximum-"
    "exposure bound uses every assigned episode of a level as the selected-failure count"
)


def _suffix(predicate: str) -> str:
    return str(predicate).rsplit(".", 1)[-1]


def _digest(payload: dict) -> str:
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256_text(text)


def _lineage(mechanism: dict) -> str:
    template = ((mechanism.get("provenance") or {}).get("template") or "").strip()
    return f"template:{template}" if template else f"mechanism:{mechanism['mechanism_id']}"


# Accepted path
def _fix_path(mechanism: dict) -> dict:
    """Deterministic shortest preferred STG path to the required postcondition.

    Used only for the three genuine two-fix mechanisms that declare no abstention contract.
    Ties are broken by the action sequence, so the result is independent of dict order.
    """
    stg = mechanism.get("stg") or {}
    entry = stg.get("entry", "s0")
    out: dict[str, list[dict]] = defaultdict(list)
    for edge in stg.get("edges") or []:
        out[str(edge.get("from"))].append(edge)
    required = set(mechanism.get("required_postconditions") or [])
    targets = {
        str(node.get("id"))
        for node in stg.get("nodes") or []
        if required and required <= set(node.get("postconditions_met") or [])
    }
    candidates: list[tuple[tuple[str, ...], list[dict]]] = []
    queue: deque[tuple[str, tuple[str, ...], list[dict], frozenset[str]]] = deque(
        [(entry, (), [], frozenset({entry}))]
    )
    while queue:
        node, actions, edges, visited = queue.popleft()
        if actions and node in targets:
            candidates.append((actions, edges))
            continue
        for edge in sorted(
            out.get(node, []),
            key=lambda item: (
                str(item.get("action")),
                str(item.get("to")),
                not bool(item.get("preferred")),
            ),
        ):
            if not edge.get("preferred"):
                continue
            target = str(edge.get("to"))
            if target in visited:
                continue
            queue.append(
                (
                    target,
                    actions + (str(edge.get("action")),),
                    edges + [edge],
                    visited | {target},
                )
            )
    if not candidates:
        # No preferred route exists: fall back to every edge deterministically.  The flag is
        # recorded by the caller; no depth is guessed from an absent path.
        queue = deque([(entry, (), [], frozenset({entry}))])
        while queue:
            node, actions, edges, visited = queue.popleft()
            if actions and node in targets:
                candidates.append((actions, edges))
                continue
            for edge in sorted(
                out.get(node, []),
                key=lambda item: (str(item.get("action")), str(item.get("to"))),
            ):
                target = str(edge.get("to"))
                if target in visited:
                    continue
                queue.append(
                    (
                        target,
                        actions + (str(edge.get("action")),),
                        edges + [edge],
                        visited | {target},
                    )
                )
    if not candidates:
        return {"actions": [], "edges": [], "preferred": False}
    actions, edges = sorted(candidates, key=lambda item: item[0])[0]
    return {"actions": list(actions), "edges": edges, "preferred": True}


def _resolve_path_edges(mechanism: dict, actions: list[str], declared: list[str]) -> list[dict]:
    """Resolve a declared action sequence to concrete STG edges, deterministically.

    Duplicate action labels are possible; the edge whose ``requires`` are a subset of the
    declared requirements is preferred, ties by target node id.  If no edge contains only
    declared requirements, the smallest target id is used; the caller records the
    mismatch rather than guessing a different path.
    """
    stg = mechanism.get("stg") or {}
    out: dict[str, list[dict]] = defaultdict(list)
    for edge in stg.get("edges") or []:
        out[str(edge.get("from"))].append(edge)
    node = str(stg.get("entry", "s0"))
    resolved: list[dict] = []
    declared_set = set(declared)
    for action in actions:
        candidates = [
            edge for edge in out.get(node, []) if str(edge.get("action")) == action
        ]
        if not candidates:
            return []
        subset = [
            edge
            for edge in candidates
            if set(str(p) for p in (edge.get("requires") or [])) <= declared_set
        ]
        edge = sorted(
            subset or candidates, key=lambda item: str(item.get("to"))
        )[0]
        resolved.append(edge)
        node = str(edge.get("to"))
    return resolved


def _accepted_path(mechanism: dict) -> dict:
    contract = mechanism.get("abstention_contract") or {}
    actions = [str(action) for action in (contract.get("accepted_path") or [])]
    declared = [str(predicate) for predicate in (contract.get("accepted_path_requires") or [])]
    if actions:
        edges = _resolve_path_edges(mechanism, actions, declared)
        path_requires = sorted(
            {str(p) for edge in edges for p in (edge.get("requires") or [])}
        )
        return {
            "basis": "ABSTENTION_CONTRACT",
            "actions": actions,
            "requires": declared,
            "path_requires": path_requires,
            "edges": edges,
            "contract": contract,
            "variant_id": str(contract.get("variant_id") or ""),
            "fallback_preferred": False,
            "declared_requires_match_path": path_requires == sorted(set(declared)),
        }
    fallback = _fix_path(mechanism)
    edges = fallback["edges"]
    path_requires = sorted(
        {str(p) for edge in edges for p in (edge.get("requires") or [])}
    )
    return {
        "basis": "FIX_PATH_FALLBACK_NO_ABSTENTION_CONTRACT",
        "actions": list(fallback["actions"]),
        "requires": [],
        "path_requires": path_requires,
        "edges": edges,
        "contract": {},
        "variant_id": "",
        "fallback_preferred": bool(fallback["preferred"]),
        "declared_requires_match_path": True,
    }


# Small executable contract helper (safe-route branching test)
def _row(tables: dict, table: str, where: dict):
    for row in tables.get(table, []):
        if all(str(row.get(key)) == str(value) for key, value in (where or {}).items()):
            return row
    return None


def _apply(tables: dict, effect: dict) -> None:
    kind = effect.get("kind")
    if kind == "set_field":
        table, where = effect["table"], effect.get("where") or {}
        field, value = effect["field"], effect["value"]
    elif kind == "set_flag":
        table, where = "mechanism_state", {"id": "state"}
        field, value = effect["field"], effect["value"]
    else:
        raise ValueError(f"unknown effect kind {kind!r}")
    row = _row(tables, table, where)
    if row is None:
        row = dict(where)
        tables.setdefault(table, []).append(row)
    row[field] = value


def _predicate_ok(tables: dict, spec: dict) -> bool:
    if spec.get("kind") == "constant":
        return bool(spec.get("value"))
    row = _row(tables, spec["table"], spec.get("where") or {})
    return row is not None and row.get(spec["field"]) == spec["value"]


def _check_ok(tables: dict, check: dict, registry: dict) -> bool:
    kind = check.get("kind")
    if kind == "predicate":
        spec = registry.get(check.get("name"))
        if spec is None:
            raise ValueError(f"unregistered predicate {check.get('name')!r}")
        return _predicate_ok(tables, spec) is bool(check.get("expect", True))
    if kind == "row_exists":
        return _row(tables, check["table"], check.get("where") or {}) is not None
    if kind == "row_absent":
        return _row(tables, check["table"], check.get("where") or {}) is None
    if kind == "field_equals":
        row = _row(tables, check["table"], check.get("where") or {})
        return row is not None and str(row.get(check["field"])) == str(check["value"])
    raise ValueError(f"check kind {kind!r} is not executable from tables")


def _entry_tables(mechanism: dict, variant: dict) -> dict:
    tables = json.loads(
        json.dumps((mechanism.get("entry_state") or {}).get("tables") or {})
    )
    registry = (mechanism.get("entry_state") or {}).get("predicate_registry") or {}
    for name in variant.get("active_predicates") or []:
        spec = registry.get(name)
        if spec is None:
            raise ValueError(f"variant activates unregistered predicate {name!r}")
        _apply(tables, {**spec, "kind": "set_field"})
    return tables


def _simulate_tables(mechanism: dict, variant: dict, actions: list[str]) -> dict:
    stg = mechanism.get("stg") or {}
    registry = (mechanism.get("entry_state") or {}).get("predicate_registry") or {}
    tables = _entry_tables(mechanism, variant)
    node = stg.get("entry", "s0")
    for action in actions:
        edge = None
        for candidate in stg.get("edges") or []:
            if candidate.get("from") != node or candidate.get("action") != action:
                continue
            if all(
                _predicate_ok(tables, registry[p])
                for p in (candidate.get("requires") or [])
            ):
                edge = candidate
                break
        if edge is None:
            raise ValueError(f"no executable {action!r} edge from {node!r}")
        for effect in edge.get("effects") or []:
            _apply(tables, effect)
        node = edge.get("to")
    return tables


def _postconditions_met(mechanism: dict, tables: dict) -> bool:
    registry = (mechanism.get("entry_state") or {}).get("predicate_registry") or {}
    for check in mechanism.get("postconditions") or []:
        if not _check_ok(tables, check, registry):
            return False
    return True


def _branching_routes(mechanism: dict, accepted: dict) -> list[list[str]]:
    """Distinct accepted safe action sequences within the contract's own step budget.

    A route is accepted when every ``required_safe_outcome_checks`` entry executes true, the
    postconditions are not met, and no action is in the contract's forbidden list.  The
    enumeration is bounded by ``len(accepted_path) + 1`` actions and simple nodes.  The
    contract's own accepted path must be one of the routes.
    """
    contract = accepted.get("contract") or {}
    if not accepted["actions"] or not contract:
        return []
    variant_id = accepted.get("variant_id") or ""
    variant = next(
        (item for item in mechanism.get("variants") or [] if item.get("variant_id") == variant_id),
        None,
    )
    if variant is None:
        raise ValueError(
            f"{mechanism['mechanism_id']}: contract variant {variant_id!r} is absent"
        )
    allowed = set(mechanism.get("permitted_actions") or []) - set(
        contract.get("forbidden_actions") or []
    )
    stg = mechanism.get("stg") or {}
    out: dict[str, list[dict]] = defaultdict(list)
    for edge in stg.get("edges") or []:
        out[str(edge.get("from"))].append(edge)
    checks = contract.get("required_safe_outcome_checks") or []
    max_steps = len(accepted["actions"]) + 1
    found: set[tuple[str, ...]] = set()

    def visit(node: str, actions: list[str], visited: frozenset[str]) -> None:
        if actions and len(actions) <= max_steps:
            try:
                tables = _simulate_tables(mechanism, variant, actions)
                if all(
                    _check_ok(
                        tables, check, mechanism["entry_state"]["predicate_registry"]
                    )
                    for check in checks
                ) and not _postconditions_met(mechanism, tables):
                    found.add(tuple(actions))
            except (KeyError, ValueError, TypeError):
                pass
        if len(actions) >= max_steps:
            return
        for edge in sorted(out.get(node, []), key=lambda item: (str(item.get("action")), str(item.get("to")))):
            action = str(edge.get("action"))
            if action not in allowed:
                continue
            target = str(edge.get("to"))
            if target in visited:
                continue
            visit(target, actions + [action], visited | {target})

    visit(str(stg.get("entry", "s0")), [], frozenset({str(stg.get("entry", "s0"))}))
    return [list(route) for route in sorted(found)]


# Components and levels
def _components(mechanism: dict, accepted: dict) -> dict:
    contract = accepted.get("contract") or {}
    requires = [str(p) for p in accepted.get("path_requires") or []]
    cross = sorted({p for p in requires if _suffix(p) in CROSS_PRINCIPAL_PREDICATE_TOKENS})
    gates = sorted({p for p in requires if _suffix(p) not in CROSS_PRINCIPAL_PREDICATE_TOKENS})
    handoff_actions = [action for action in accepted["actions"] if action_is_approval(action)]
    handoffs = len(handoff_actions) + len(cross)

    edge_requires = {
        str(predicate)
        for edge in (mechanism.get("stg") or {}).get("edges") or []
        for predicate in (edge.get("requires") or [])
    }
    safe_checks = contract.get("required_safe_outcome_checks") or []
    counted_predicates = set(gates) | set(cross)
    observations: list[dict] = []
    for observation in contract.get("required_observations") or []:
        kind = observation.get("kind")
        if kind == "predicate":
            name = str(observation.get("name") or "")
            if name in edge_requires and name not in counted_predicates:
                observations.append(
                    {
                        "kind": kind,
                        "name": name,
                        "conditions": "STG edge requirement",
                    }
                )
            continue
        if kind in {"field_equals", "row_exists", "row_absent", "count"}:
            for check in safe_checks:
                if (
                    check.get("kind") == kind
                    and check.get("table") == observation.get("table")
                    and (check.get("where") or {}) == (observation.get("where") or {})
                    and check.get("field") == observation.get("field")
                    and str(check.get("value")) == str(observation.get("value"))
                ):
                    observations.append(
                        {
                            "kind": kind,
                            "table": observation.get("table"),
                            "field": observation.get("field"),
                            "conditions": "required safe-outcome check",
                        }
                    )
                    break

    routes = _branching_routes(mechanism, accepted)
    if accepted["basis"] == "ABSTENTION_CONTRACT":
        if tuple(accepted["actions"]) not in {tuple(route) for route in routes}:
            raise ValueError(
                f"{mechanism['mechanism_id']}: contract accepted path is not an accepted "
                "safe route"
            )
    components = {
        "gates": len(gates),
        "handoffs": handoffs,
        "observations": len(observations),
        "depth_score": len(gates) + handoffs + len(observations),
        "accepted_path_steps": len(accepted["actions"]),
        "branching_routes": len(routes),
        "gate_predicates": gates,
        "cross_principal_predicates": cross,
        "handoff_actions": sorted(handoff_actions),
        "observation_links": observations,
    }
    if handoffs >= 1 or len(routes) >= 2:
        level = "L3"
    elif len(gates) >= 2 or len(accepted["actions"]) >= 2:
        level = "L2"
    else:
        level = "L1"
    return {**components, "level": level, "routes": routes}


def _accepted_path_requires_match(mechanism: dict, accepted: dict) -> bool:
    """Check whether the declared accepted_path_requires equals the executable path."""
    if accepted["basis"] != "ABSTENTION_CONTRACT":
        return True
    return list(accepted.get("path_requires") or []) == sorted(
        set(str(p) for p in (accepted.get("requires") or []))
    )


# Table
def build_chain_depth_table(
    *,
    source: dict | None = None,
    manifest: dict | None = None,
    run_plan: dict | None = None,
    include_assignments: bool = True,
    include_power: bool = True,
    power_sims: int = 1200,
    power_seed: int = 20261149,
) -> dict:
    source = source or resolve_source_set()
    manifest = manifest or build_condition_manifest(source=source)
    mechanisms = sorted(
        load_source_mechanisms(source), key=lambda item: str(item["mechanism_id"])
    )
    records: list[dict] = []
    for mechanism in mechanisms:
        accepted = _accepted_path(mechanism)
        components = _components(mechanism, accepted)
        records.append(
            {
                "mechanism_id": str(mechanism["mechanism_id"]),
                "family": str(mechanism["family"]),
                "template": ((mechanism.get("provenance") or {}).get("template") or ""),
                "lineage": _lineage(mechanism),
                "contract_basis": accepted["basis"],
                "contract_variant_id": accepted.get("variant_id") or "",
                "fallback_preferred_path": bool(accepted.get("fallback_preferred")),
                "accepted_path": accepted["actions"],
                "accepted_path_requires": sorted(set(accepted["requires"])),
                "path_requires": sorted(set(accepted["path_requires"])),
                "declared_requires_match_path": bool(
                    accepted.get("declared_requires_match_path")
                ),
                "accepted_path_requires_match_contract": _accepted_path_requires_match(
                    mechanism, accepted
                ),
                "components": components,
                "level": components["level"],
            }
        )
    by_id = {record["mechanism_id"]: record for record in records}

    instances: list[dict] = []
    for mechanism in mechanisms:
        mechanism_id = str(mechanism["mechanism_id"])
        record = by_id[mechanism_id]
        for variant in sorted(
            mechanism.get("variants") or [], key=lambda item: str(item.get("variant_id"))
        ):
            scenario = variant.get("scenario") or {}
            instances.append(
                {
                    "instance_id": str(variant.get("instance_id") or variant.get("variant_id")),
                    "mechanism_id": mechanism_id,
                    "variant_id": str(variant.get("variant_id")),
                    "family": record["family"],
                    "label": "SOLVABLE" if scenario.get("correct_outcome") == "fix" else "ABSTENTION",
                    "correct_outcome": str(scenario.get("correct_outcome") or ""),
                    "level": record["level"],
                    "depth_score": record["components"]["depth_score"],
                    "contract_basis": record["contract_basis"],
                }
            )

    cells = [
        cell for cell in manifest.get("cells") or [] if bool(cell.get("apply"))
    ]
    for cell in cells:
        record = by_id[str(cell["mechanism_id"])]
        cell["chain_depth_level"] = record["level"]

    assignment_rows: list[dict] = []
    if include_assignments:
        repeats = int((run_plan or {}).get("repeats_used") or 2)
        assignment_rows = build_assignment_table(
            manifest=manifest, repeats=repeats, patterns=ENROLLED_PATTERNS
        )["rows"]

    def _level_counts(items, key: str) -> dict:
        counts = Counter(str(item.get(key)) for item in items)
        return {level: int(counts.get(level, 0)) for level in LEVELS}

    cell_counts_by_level = _level_counts(cells, "chain_depth_level")
    episodes_by_level = _level_counts(
        [
            {"level": by_id[str(row["mechanism_id"])]["level"], **row}
            for row in assignment_rows
        ],
        "level",
    )
    mechanisms_by_level = _level_counts(records, "level")
    instances_by_level = _level_counts(instances, "level")

    lineages_by_level: dict[str, list[str]] = {level: [] for level in LEVELS}
    for record in records:
        lineages_by_level[record["level"]].append(record["lineage"])
    lineages_by_level = {
        level: sorted(set(values)) for level, values in lineages_by_level.items()
    }

    by_family: dict[str, dict] = {}
    for family in sorted({record["family"] for record in records}):
        family_records = [record for record in records if record["family"] == family]
        family_cells = [cell for cell in cells if cell["family"] == family]
        family_episodes = [
            row for row in assignment_rows if str(row.get("family")) == family
        ]
        by_family[family] = {
            "mechanisms": _level_counts(family_records, "level"),
            "instances": _level_counts(
                [item for item in instances if item["family"] == family], "level"
            ),
            "lineages": {
                level: sorted(
                    {
                        record["lineage"]
                        for record in family_records
                        if record["level"] == level
                    }
                )
                for level in LEVELS
            },
            "claimed_cells": _level_counts(family_cells, "chain_depth_level"),
            "assignment_episodes": _level_counts(
                [
                    {"level": by_id[str(row["mechanism_id"])]["level"], **row}
                    for row in family_episodes
                ],
                "level",
            ),
        }

    conditions = sorted({str(cell.get("condition")) for cell in cells})
    by_condition: dict[str, dict] = {}
    for condition in conditions:
        condition_cells = [cell for cell in cells if cell.get("condition") == condition]
        condition_episodes = [
            row for row in assignment_rows if str(row.get("condition")) == condition
        ]
        by_condition[condition] = {
            "claimed_cells": _level_counts(condition_cells, "chain_depth_level"),
            "assignment_episodes": _level_counts(
                [
                    {"level": by_id[str(row["mechanism_id"])]["level"], **row}
                    for row in condition_episodes
                ],
                "level",
            ),
        }

    by_family_condition: dict[str, dict] = {}
    for family in by_family:
        by_family_condition[family] = {}
        for condition in conditions:
            subset = [
                cell
                for cell in cells
                if cell["family"] == family and cell.get("condition") == condition
            ]
            if not subset:
                continue
            subset_episodes = [
                row
                for row in assignment_rows
                if str(row.get("family")) == family
                and str(row.get("condition")) == condition
            ]
            by_family_condition[family][condition] = {
                "claimed_cells": _level_counts(subset, "chain_depth_level"),
                "assignment_episodes": _level_counts(
                    [
                        {"level": by_id[str(row["mechanism_id"])]["level"], **row}
                        for row in subset_episodes
                    ],
                    "level",
                ),
            }

    thin_cells: list[dict] = []
    for family, conditions_block in by_family_condition.items():
        for condition, block in conditions_block.items():
            for level in LEVELS:
                episodes = int(block["assignment_episodes"].get(level, 0))
                if episodes < MINIMUM_CELL_SIZE["family_condition_level_cells"]:
                    thin_cells.append(
                        {
                            "family": family,
                            "condition": condition,
                            "level": level,
                            "assignment_episodes": episodes,
                        }
                    )
    thin_cells.sort(
        key=lambda item: (item["level"], item["family"], item["condition"])
    )

    thinness = {}
    for level in LEVELS:
        reasons: list[str] = []
        if mechanisms_by_level[level] < 2:
            reasons.append("fewer than two mechanisms")
        if len(lineages_by_level[level]) < MINIMUM_CELL_SIZE["lineage_clusters_per_level"]:
            reasons.append(
                f"{len(lineages_by_level[level])} construction lineages "
                f"(< {MINIMUM_CELL_SIZE['lineage_clusters_per_level']})"
            )
        if cell_counts_by_level[level] < MINIMUM_CELL_SIZE["family_condition_level_cells"]:
            reasons.append(
                f"{cell_counts_by_level[level]} claimed condition cells "
                f"(< {MINIMUM_CELL_SIZE['family_condition_level_cells']})"
            )
        if episodes_by_level[level] < MINIMUM_CELL_SIZE["assigned_episodes_per_level"]:
            reasons.append(
                f"{episodes_by_level[level]} assigned episodes "
                f"(< {MINIMUM_CELL_SIZE['assigned_episodes_per_level']})"
            )
        thinness[level] = {
            "mechanisms": mechanisms_by_level[level],
            "instances": instances_by_level[level],
            "lineage_clusters": len(lineages_by_level[level]),
            "claimed_condition_cells": cell_counts_by_level[level],
            "assignment_episodes": episodes_by_level[level],
            "too_thin_to_analyse": bool(reasons),
            "reasons": reasons,
        }

    table = {
        "kind": KIND,
        "version": VERSION,
        "source": source_identity(source),
        "condition_manifest_digest": manifest.get("manifest_digest"),
        "mechanism_count": len(records),
        "instance_count": len(instances),
        "depth_rule": DEPTH_RULE,
        "level_rule": LEVEL_RULE,
        "level_definitions": LEVEL_DEFINITIONS,
        "cross_principal_predicate_tokens": sorted(CROSS_PRINCIPAL_PREDICATE_TOKENS),
        "minimum_cell_size": MINIMUM_CELL_SIZE,
        "power_planning_assumption": POWER_PLANNING_ASSUMPTION,
        "mechanisms": records,
        "instances": instances,
        "counts": {
            "mechanisms_by_level": mechanisms_by_level,
            "instances_by_level": instances_by_level,
            "lineage_clusters_by_level": {
                level: len(lineages_by_level[level]) for level in LEVELS
            },
            "lineages_by_level": lineages_by_level,
            "claimed_cells_by_level": cell_counts_by_level,
            "assignment_episodes_by_level": episodes_by_level,
            "by_family": by_family,
            "by_condition": by_condition,
            "by_family_condition": by_family_condition,
        },
        "thinness": thinness,
        "thin_family_condition_cells": thin_cells,
        "notes": [
            "depth is a property of the declared mechanism contract; both variants of a "
            "mechanism inherit the contract level",
            "the three genuine two-fix mechanisms have no abstention contract and use the "
            "deterministic preferred fix path; their records say so explicitly",
            "branching is an executable test: more than one distinct non-forbidden action "
            "sequence within the contract's own step budget satisfies every required "
            "safe-outcome check and leaves the postconditions unmet",
            "no level was chosen after an outcome; the table is built from frozen set-12 "
            "bytes and the frozen assignment schedule only",
        ],
    }
    if include_power:
        table["power"] = build_chain_depth_power(
            table, sims=power_sims, seed=power_seed
        )
    table["digest"] = _digest(
        {key: value for key, value in table.items() if key != "digest"}
    )
    return table


# Level-specific power (secondary analysis)
def _neutral_allocation(table: dict, total: int = 660) -> dict[str, int]:
    """Largest-remainder allocation of ``total`` proportional to assigned episodes."""
    episodes = table["counts"]["assignment_episodes_by_level"]
    denominator = sum(int(episodes.get(level, 0)) for level in LEVELS)
    if denominator <= 0:
        return {level: 0 for level in LEVELS}
    raw = {level: total * int(episodes.get(level, 0)) / denominator for level in LEVELS}
    allocated = {level: int(raw[level]) for level in LEVELS}
    remainder = total - sum(allocated.values())
    order = sorted(
        LEVELS,
        key=lambda level: (-(raw[level] - allocated[level]), LEVELS.index(level)),
    )
    for level in order[:remainder]:
        allocated[level] += 1
    return allocated


def build_chain_depth_power(
    table: dict,
    *,
    sims: int = 1200,
    seed: int = 20261149,
    total_cap: int = 660,
    target: float = 0.80,
) -> dict:
    """Per-level Detention A-B/A-C power with the frozen planning assumption.

    The secondary family at one level is two contrasts (A-B and A-C) with Holm within the
    level.  ``n_clusters`` is the number of construction lineages that exist at that level,
    capped at the 28-lineage design value.  The neutral allocation and the maximum-exposure
    bound are both reported; adequacy uses the neutral allocation.
    """
    from .power import detention_subgroup_power

    neutral = _neutral_allocation(table, total=total_cap)
    maximum = dict(table["counts"]["assignment_episodes_by_level"])
    lineages = dict(table["counts"]["lineage_clusters_by_level"])

    def one(level: str, n_selected: int) -> dict:
        clusters = max(1, min(28, int(lineages.get(level, 0))))
        result = detention_subgroup_power(
            n_selected=int(n_selected),
            n_clusters=clusters,
            sims=sims,
            seed=seed + LEVELS.index(level),
        )
        result["level"] = level
        result["n_selected"] = int(n_selected)
        result["lineage_clusters"] = clusters
        return result

    neutral_results = {level: one(level, neutral[level]) for level in LEVELS}
    maximum_results = {level: one(level, maximum[level]) for level in LEVELS}

    decisions = {}
    for level in LEVELS:
        result = neutral_results[level]
        powers = result["per_test_power"]
        lower = result["wilson_lower_95"]
        reasons = []
        if int(neutral[level]) < MINIMUM_CELL_SIZE["selected_failures_per_level"]:
            reasons.append(
                f"neutral n={neutral[level]} < "
                f"{MINIMUM_CELL_SIZE['selected_failures_per_level']}"
            )
        if int(lineages.get(level, 0)) < MINIMUM_CELL_SIZE["lineage_clusters_per_level"]:
            reasons.append(
                f"{lineages.get(level, 0)} lineages < "
                f"{MINIMUM_CELL_SIZE['lineage_clusters_per_level']}"
            )
        for test in ("detention_A_minus_B", "detention_A_minus_C"):
            if float(powers.get(test, 0.0)) < target:
                reasons.append(f"{test} Holm power {powers.get(test, 0.0):.3f} < {target}")
            if float(lower.get(test, 0.0)) < target:
                reasons.append(
                    f"{test} power Wilson lower {lower.get(test, 0.0):.3f} < {target}"
                )
        decisions[level] = {
            "neutral_n_selected": int(neutral[level]),
            "neutral_lineage_clusters": int(lineages.get(level, 0)),
            "adequate": not reasons,
            "status": (
                "SECONDARY_POWER_ADEQUATE"
                if not reasons
                else "SECONDARY_EXPLORATORY_UNDERPOWERED"
            ),
            "reasons": reasons,
            "maximum_exposure_n_selected": int(maximum[level]),
        }

    all_adequate = all(item["adequate"] for item in decisions.values())
    return {
        "status": (
            "SECONDARY_POWER_ADEQUATE"
            if all_adequate
            else "SECONDARY_EXPLORATORY_UNDERPOWERED"
        ),
        "all_levels_adequate": all_adequate,
        "planning_assumption": POWER_PLANNING_ASSUMPTION,
        "neutral_allocation": neutral,
        "maximum_exposure_bound": maximum,
        "lineage_clusters_by_level": {level: int(lineages.get(level, 0)) for level in LEVELS},
        "secondary_family": [
            "detention_A_minus_B within level",
            "detention_A_minus_C within level",
        ],
        "multiplicity": (
            "Holm two-sided 5% over the two within-level Detention contrasts (A-B, A-C); "
            "these tests are outside the confirmatory four-test Holm family"
        ),
        "target_power": target,
        "sims": int(sims),
        "seed": int(seed),
        "neutral": neutral_results,
        "maximum_exposure": maximum_results,
        "decisions": decisions,
        "notes": [
            "power is computed with vera/delegation/power.py over declared planning "
            "assumptions; it is a design study, never an outcome or a power certificate",
            "the neutral allocation assumes equal failure propensity per assigned episode; "
            "the maximum-exposure bound is optimistic because not every episode fails",
            "a level below the minimum lineage count is descriptive only even if its "
            "neutral point estimate crosses 0.80",
        ],
    }


def write_chain_depth(path: str | Path, table: dict | None = None) -> dict:
    table = table or build_chain_depth_table()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "path": str(target),
        "sha256": sha256_text(target.read_text(encoding="utf-8")),
        "digest": table.get("digest"),
    }


__all__ = [
    "CROSS_PRINCIPAL_PREDICATE_TOKENS",
    "DEPTH_RULE",
    "KIND",
    "LEVELS",
    "LEVEL_DEFINITIONS",
    "LEVEL_RULE",
    "MINIMUM_CELL_SIZE",
    "POWER_PLANNING_ASSUMPTION",
    "VERSION",
    "build_chain_depth_power",
    "build_chain_depth_table",
    "write_chain_depth",
]
