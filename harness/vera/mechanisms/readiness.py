"""Benchmark-readiness checks for mechanism records (design §6.5).

A unique behavioural signature is not enough.  Before a mechanism can be counted as
benchmark-ready, this module checks that it has an explicit entry state, that every
symbolic predicate is backed by a DB predicate over that state, that every edge has an
executable effect, and that at least one reference-policy path reaches the required
postconditions and satisfies the declared postcondition DSL on the resulting tables.

The helper ``attach_readiness_fields`` is deliberately data-only: it adds the missing
declarations to a copy of a mechanism.  Historical manifests are never rewritten.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from ..canonical import sha256_text
from ..verifier import evaluate_check

_REQUIRED_ENTRY_KEYS = ("tables", "predicate_registry", "postcondition_bindings")


def _safe_field(prefix: str, name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]", "_", name)[:32]
    return f"{prefix}_{slug}_{sha256_text(name)[:8]}"


def _initial_value(target: Any) -> Any:
    if isinstance(target, bool):
        return not target
    if isinstance(target, int):
        return 0 if target != 0 else -1
    if isinstance(target, float):
        return 0.0 if target != 0.0 else -1.0
    if isinstance(target, str):
        return "open" if target != "open" else "closed"
    return None


def _snapshot(tables: dict) -> dict:
    return {"tables": tables, "invariants": {}, "tables_digest": {}}


def _row(tables: dict, table: str, where: dict) -> dict | None:
    for row in tables.get(table, []):
        if all(str(row.get(k)) == str(v) for k, v in where.items()):
            return row
    return None


def _set_field(tables: dict, table: str, where: dict, field: str, value: Any) -> None:
    row = _row(tables, table, where)
    if row is None:
        row = dict(where)
        tables.setdefault(table, []).append(row)
    row[field] = value


def _effect(effect: dict) -> dict:
    kind = effect.get("kind")
    if kind == "set_field":
        for key in ("table", "where", "field", "value"):
            if key not in effect:
                raise ValueError(f"set_field effect missing {key!r}")
        return effect
    if kind == "set_flag":
        if "field" not in effect or "value" not in effect:
            raise ValueError("set_flag effect needs field and value")
        return effect
    raise ValueError(f"unknown effect kind {kind!r}")


def _apply_effect(tables: dict, effect: dict) -> None:
    effect = _effect(effect)
    if effect["kind"] == "set_field":
        _set_field(tables, effect["table"], effect["where"], effect["field"], effect["value"])
    else:
        _set_field(tables, "mechanism_state", {"id": "state"}, effect["field"], effect["value"])


def _predicate_ok(tables: dict, spec: dict) -> bool:
    kind = spec.get("kind")
    if kind == "constant":
        return bool(spec.get("value"))
    if kind == "db_flag":
        row = _row(tables, spec["table"], spec.get("where", {}))
        return row is not None and row.get(spec["field"]) == spec["value"]
    raise ValueError(f"unknown predicate kind {kind!r}")


def _binding_ok(tables: dict, binding: dict) -> bool:
    return bool(evaluate_check(_snapshot(tables), binding)[0])


def attach_readiness_fields(mechanism: dict) -> dict:
    """Return a copy with entry_state, predicate registry, bindings and edge effects.

    The generated fields are explicit data.  A mechanism is not counted as ready until
    ``validate_ready_mechanism`` has executed them against the DB tables.
    """
    m = copy.deepcopy(mechanism)
    stg = m.get("stg", {})
    nodes = {n["id"]: n for n in stg.get("nodes", [])}
    entry_id = stg.get("entry", "s0")
    if entry_id not in nodes:
        raise ValueError(f"{m.get('mechanism_id')}: entry node {entry_id!r} missing")
    predicates = sorted(
        {p for n in nodes.values() for p in n.get("predicates", [])}
        | {p for e in stg.get("edges", []) for p in e.get("requires", [])}
    )
    post_strings = sorted(
        {p for n in nodes.values() for p in n.get("postconditions_met", [])}
    )
    required = sorted(set(m.get("required_postconditions", [])))
    postconditions = list(m.get("postconditions") or [])
    if not postconditions:
        raise ValueError(f"{m.get('mechanism_id')}: no postcondition DSL")
    if required:
        if len(postconditions) == len(required):
            req_bindings = {post: copy.deepcopy(pc) for post, pc in zip(required, postconditions)}
        else:
            req_bindings = {post: copy.deepcopy(postconditions[0]) for post in required}
    else:
        req_bindings = {}

    pred_fields = {pred: _safe_field("p", pred) for pred in predicates}
    state_row: dict[str, Any] = {"id": "state"}
    registry: dict[str, dict] = {}
    for pred in predicates:
        field = pred_fields[pred]
        state_row[field] = 0
        registry[pred] = {
            "kind": "db_flag",
            "table": "mechanism_state",
            "where": {"id": "state"},
            "field": field,
            "value": 1,
        }

    bindings: dict[str, dict] = {post: copy.deepcopy(pc) for post, pc in req_bindings.items()}
    for post in post_strings:
        if post in bindings:
            continue
        field = _safe_field("po", post)
        state_row[field] = 0
        bindings[post] = {
            "kind": "field_equals",
            "table": "mechanism_state",
            "where": {"id": "state"},
            "field": field,
            "value": 1,
        }

    final_pc = req_bindings.get(required[0], postconditions[0]) if required else postconditions[0]
    tables: dict[str, list] = {
        final_pc["table"]: [{**final_pc["where"], final_pc["field"]: _initial_value(final_pc["value"])}],
        "mechanism_state": [state_row],
    }
    for pred in nodes[entry_id].get("predicates", []):
        state_row[pred_fields[pred]] = 1

    for edge in stg.get("edges", []):
        target = nodes[edge["to"]]
        effects = []
        for pred in sorted(target.get("predicates", [])):
            effects.append({"kind": "set_flag", "field": pred_fields[pred], "value": 1})
        for post in sorted(target.get("postconditions_met", [])):
            # Some mechanisms also use a postcondition name as an edge predicate.
            if post in pred_fields:
                effects.append({"kind": "set_flag", "field": pred_fields[post], "value": 1})
            binding = bindings[post]
            if binding["table"] == "mechanism_state":
                effects.append({"kind": "set_flag", "field": binding["field"], "value": 1})
            else:
                effects.append({
                    "kind": "set_field",
                    "table": binding["table"],
                    "where": binding["where"],
                    "field": binding["field"],
                    "value": binding["value"],
                })
        edge["effects"] = effects

    m["entry_state"] = {
        "tables": tables,
        "predicate_registry": registry,
        "postcondition_bindings": bindings,
        "active_predicates": sorted(nodes[entry_id].get("predicates", [])),
    }
    return m


def _set_variant_predicates(tables: dict, registry: dict, predicates: list[str]) -> None:
    for pred in predicates:
        spec = registry.get(pred)
        if spec is None:
            raise ValueError(f"variant predicate {pred!r} is not registered")
        _set_field(tables, spec["table"], spec.get("where", {}), spec["field"], spec["value"])


def _find_path(m: dict, variant: dict) -> dict:
    edges = m["stg"]["edges"]
    entry_state = m["entry_state"]
    registry = entry_state["predicate_registry"]
    bindings = entry_state["postcondition_bindings"]
    required = list(m.get("required_postconditions", []))
    start = m["stg"].get("entry", "s0")
    initial_tables = copy.deepcopy(entry_state["tables"])
    _set_variant_predicates(initial_tables, registry, list(variant.get("active_predicates", [])))

    best = {"ok": False, "reasons": [], "path": [], "tables": None}
    seen: set[tuple] = set()

    def walk(node_id: str, tables: dict, path: list[dict], steps: int) -> None:
        if best["ok"] or steps > 8:
            return
        key = (node_id, steps, tuple((e["action"], e["to"]) for e in path))
        if key in seen:
            return
        seen.add(key)
        if required and all(_binding_ok(tables, bindings[p]) for p in required):
            best.update({"ok": True, "path": list(path), "tables": copy.deepcopy(tables)})
            return
        for edge in [e for e in edges if e["from"] == node_id]:
            if not all(_predicate_ok(tables, registry[p]) for p in edge.get("requires", [])):
                continue
            next_tables = copy.deepcopy(tables)
            for effect in edge.get("effects", []):
                _apply_effect(next_tables, effect)
            path.append({"from": node_id, "to": edge["to"], "action": edge["action"]})
            walk(edge["to"], next_tables, path, steps + 1)
            path.pop()

    walk(start, initial_tables, [], 0)
    if not best["ok"]:
        if not required:
            best["reasons"].append("no required_postconditions declared")
        else:
            best["reasons"].append("no executable path satisfies all required postconditions")
    return best


def validate_ready_mechanism(mechanism: dict) -> dict:
    """Return a machine-readable readiness record; it never guesses a pass."""
    mid = mechanism.get("mechanism_id", "<unknown>")
    reasons: list[str] = []
    entry_state = mechanism.get("entry_state")
    if not isinstance(entry_state, dict):
        return {"mechanism_id": mid, "benchmark_ready": False,
                "reasons": ["entry_state is missing"], "variant_results": []}
    for key in _REQUIRED_ENTRY_KEYS:
        if key not in entry_state:
            reasons.append(f"entry_state missing {key!r}")
    if reasons:
        return {"mechanism_id": mid, "benchmark_ready": False,
                "reasons": reasons, "variant_results": []}
    if not isinstance(entry_state["tables"], dict) or not entry_state["tables"]:
        reasons.append("entry_state.tables must be a non-empty mapping")
    registry = entry_state["predicate_registry"]
    bindings = entry_state["postcondition_bindings"]
    if not isinstance(registry, dict) or not isinstance(bindings, dict):
        reasons.append("predicate_registry and postcondition_bindings must be mappings")
    if reasons:
        return {"mechanism_id": mid, "benchmark_ready": False,
                "reasons": reasons, "variant_results": []}

    # DB predicate validation: every registered predicate must evaluate on entry tables.
    predicate_checks = {}
    for pred, spec in sorted(registry.items()):
        try:
            predicate_checks[pred] = bool(_predicate_ok(entry_state["tables"], spec))
        except (KeyError, ValueError, TypeError) as exc:
            reasons.append(f"predicate {pred!r} is not executable: {exc}")
    node_preds = {p for n in mechanism["stg"]["nodes"] for p in n.get("predicates", [])}
    edge_preds = {p for e in mechanism["stg"]["edges"] for p in e.get("requires", [])}
    missing_preds = sorted((node_preds | edge_preds) - set(registry))
    if missing_preds:
        reasons.append(f"predicates missing DB definitions: {missing_preds}")

    node_posts = {p for n in mechanism["stg"]["nodes"] for p in n.get("postconditions_met", [])}
    required = set(mechanism.get("required_postconditions", []))
    missing_posts = sorted((node_posts | required) - set(bindings))
    if missing_posts:
        reasons.append(f"postconditions missing DSL bindings: {missing_posts}")

    # Executable transitions: every edge must declare effects and reach a known node.
    node_ids = {n["id"] for n in mechanism["stg"]["nodes"]}
    for edge in mechanism["stg"]["edges"]:
        if edge.get("from") not in node_ids or edge.get("to") not in node_ids:
            reasons.append(f"edge references unknown node: {edge}")
        if not isinstance(edge.get("effects"), list):
            reasons.append(f"edge {edge.get('from')}->{edge.get('to')} has no executable effects")
            continue
        try:
            for effect in edge["effects"]:
                _effect(effect)
        except (KeyError, ValueError, TypeError) as exc:
            reasons.append(f"edge {edge.get('from')}->{edge.get('to')} effect invalid: {exc}")

    variant_results = []
    for variant in mechanism.get("variants", []):
        try:
            result = _find_path(mechanism, variant)
        except (KeyError, ValueError, TypeError) as exc:
            result = {"ok": False, "reasons": [str(exc)], "path": [], "tables": None}
        variant_results.append({
            "variant_id": variant.get("variant_id"),
            "ready": bool(result["ok"]),
            "path": result["path"],
            "reasons": result["reasons"],
        })
    if not any(v["ready"] for v in variant_results):
        reasons.append("no variant has an executable path to the required postconditions")
    ready = not reasons and any(v["ready"] for v in variant_results)
    return {
        "mechanism_id": mid,
        "benchmark_ready": bool(ready),
        "reasons": reasons,
        "entry_state_ok": not any("entry_state" in r for r in reasons),
        "predicates_ok": not missing_preds and not any("predicate" in r for r in reasons),
        "postconditions_ok": not missing_posts,
        "transitions_ok": not any("edge" in r for r in reasons),
        "variant_results": variant_results,
        "predicate_values_at_entry": predicate_checks,
    }
