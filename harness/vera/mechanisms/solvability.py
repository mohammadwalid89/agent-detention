"""Exhaustive solvability analysis of a mechanism's executable transition graph (JOB05).

Why this exists
---------------
``HEURISTIC`` is a weak reference policy ("take the preferred edge, else the alphabetically
first applicable one").  "No reference policy passes" therefore does **not** mean a mechanism
is unsolvable, and "``HEURISTIC`` passes" does not mean the mechanism is correct.  This module
answers the question directly, without simulating any reference policy:

    From ``entry_state``, is there a finite sequence of applicable transitions that makes the
    declared ``required_postconditions`` true without a forbidden side effect?

It is a bounded exhaustive search of the state-transition graph (depth ``MAX_DEPTH``), so it
can find a valid fix that no reference policy happens to take, and it can prove that a
mechanism has no valid fix at all (within the bound), which the policy audit cannot.

Execution semantics are exactly the ones :mod:`vera.mechanisms.readiness` uses: a predicate is
a DB predicate over the entry tables, an edge is applicable when all of its ``requires``
predicates are true, and its ``effects`` are applied in order to a copy of the tables.  A path
is *complete* at the first state where every required postcondition binding evaluates true
against the resulting tables (``vera.verifier.evaluate_check``).

Path validity
-------------
A complete path is a **valid fix** unless one of these holds:

``forbidden_action``
    A step's action is listed in the record's ``forbidden_actions``.  This field is declared
    explicitly by JOB05 records.  Older records do not bind their outcome-level
    ``forbidden_side_effects`` names to actions, so for them this channel is empty -- that gap
    is reported per record as ``forbidden_action_channel: "not_declared"`` and must be read as
    a limitation, not as a pass.
``abstention_completes``
    The step that makes the required postconditions true is an abstain-class action
    (``abstain`` / ``escalate`` / ``request_approval``): escalation scored as resolution, the
    JOB04 ``D2`` defect.  Such a path is not a fix.

Labelling
---------
Each instance gets exactly one label:

``SOLVABLE``
    At least one valid fix exists.  ``shortest_path_length`` and ``n_distinct_valid_paths``
    describe the set of valid fixes.
``ABSTENTION_CORRECT``
    No valid fix exists, but an abstain-class transition is applicable on some reachable
    state.  Abstaining/escalating is then the only correct answer and the instance is labelled
    as such -- this is the owner's "abstention is the correct answer" case, not a defect.
``NO_VALID_PATH``
    No valid fix and no applicable abstain-class transition: the graph offers no correct
    action at all.  This is a defect unless the mechanism declares the instance as an
    intentional negative control, and even then it is a dead end, not a well-formed control.

A mechanism is a *labelled negative control* when it declares ``negative_control`` (a new
optional field) and no instance is ``SOLVABLE``.  A negative control whose other instance is
solvable keeps its "no reference policy passes both instances" character and stays a negative
control instance; JOB05 records declare which instance is the intentional one.

This module imports the private execution helpers from :mod:`vera.mechanisms.readiness` on
purpose: the solvability search must execute declarations exactly as the readiness check does,
not through a second implementation that could drift.
"""

from __future__ import annotations

import copy
from collections import Counter

from ..canonical import canonical_json
from .graph import ABSTAIN_ACTIONS
from .readiness import (
    _apply_effect,
    _binding_ok,
    _predicate_ok,
    _set_variant_predicates,
    validate_ready_mechanism,
)

MAX_DEPTH = 8
MAX_LISTED_PATHS = 25

OUTCOME_SOLVABLE = "SOLVABLE"
OUTCOME_ABSTENTION_CORRECT = "ABSTENTION_CORRECT"
OUTCOME_NO_VALID_PATH = "NO_VALID_PATH"
OUTCOME_UNEXECUTABLE = "UNEXECUTABLE"

SIDE_EFFECT_MODEL = {
    "forbidden_action_channel": (
        "A step whose action appears in the record's optional `forbidden_actions` list makes "
        "the path invalid. JOB05 revisions declare this list; older records do not bind their "
        "outcome-level `forbidden_side_effects` names to actions."
    ),
    "abstention_channel": (
        "A path whose completing step is an abstain-class action (abstain/escalate/"
        "request_approval) is rejected: escalation must never satisfy a required "
        "postcondition."
    ),
    "not_modelled": (
        "Outcome-level forbidden side effects that a record does not bind to an action cannot "
        "be detected structurally; the complementary JOB04 lazy-audit rule `D1` covers ungated "
        "completion and a human reviewer must still read the titles."
    ),
}


def _required(m: dict) -> list[str]:
    declared = list(m.get("required_postconditions") or [])
    if declared:
        return declared
    seen: list[str] = []
    for node in m.get("stg", {}).get("nodes", []):
        for name in node.get("postconditions_met", []):
            if name not in seen:
                seen.append(name)
    return seen


def analyse_instance(mechanism: dict, variant: dict, *, max_depth: int = MAX_DEPTH) -> dict:
    """Exhaustively search one instance.  Never uses a reference policy."""
    mid = mechanism.get("mechanism_id", "<unknown>")
    vid = variant.get("variant_id")
    entry_state = mechanism.get("entry_state")
    required = _required(mechanism)
    result: dict = {
        "variant_id": vid,
        "perturbation": variant.get("perturbation"),
        "active_predicates": list(variant.get("active_predicates", [])),
        "required_postconditions": list(required),
        "solvable": False,
        "shortest_path_length": None,
        "shortest_path": [],
        "n_distinct_valid_paths": 0,
        "valid_paths": [],
        "valid_paths_truncated": False,
        "rejected_paths": [],
        "abstention_available": False,
        "abstain_only_valid_path": False,
        "outcome_label": OUTCOME_UNEXECUTABLE,
        "executable": False,
        "entry_state_satisfies_required": False,
        "notes": [],
    }
    if not isinstance(entry_state, dict) or not required:
        result["notes"].append(
            "entry_state is missing" if not isinstance(entry_state, dict)
            else "no required postconditions are declared"
        )
        return result
    registry = entry_state.get("predicate_registry")
    bindings = entry_state.get("postcondition_bindings")
    if not isinstance(registry, dict) or not isinstance(bindings, dict):
        result["notes"].append("predicate_registry/postcondition_bindings are not mappings")
        return result
    missing = [p for p in required if p not in bindings]
    if missing:
        result["notes"].append(f"required postconditions have no DSL binding: {missing}")
        return result

    stg = mechanism["stg"]
    node_ids = {n["id"] for n in stg["nodes"]}
    entry = stg.get("entry", "s0")
    if entry not in node_ids:
        result["notes"].append(f"entry node {entry!r} is not in the graph")
        return result

    # Every predicate the instance or the graph uses must have a DB definition.  JOB04's
    # readiness check only required *one* executable variant, so a variant that activated an
    # unregistered predicate passed silently; the solvability search refuses to guess and
    # labels that instance UNEXECUTABLE instead.
    used = set(variant.get("active_predicates", []))
    for node in stg["nodes"]:
        used |= set(node.get("predicates", []))
    for edge in stg["edges"]:
        used |= set(edge.get("requires", []))
    unregistered = sorted(p for p in used if p not in registry)
    if unregistered:
        result["notes"].append(f"predicates are not registered: {unregistered}")
        return result

    forbidden = [str(a) for a in mechanism.get("forbidden_actions", []) or []]
    result["declared_forbidden_actions"] = forbidden

    tables0 = copy.deepcopy(entry_state["tables"])
    try:
        _set_variant_predicates(tables0, registry, list(variant.get("active_predicates", [])))
    except (KeyError, ValueError, TypeError) as exc:
        result["notes"].append(f"variant predicates are not executable: {exc}")
        return result
    result["executable"] = True

    result["entry_state_satisfies_required"] = all(
        _binding_ok(tables0, bindings[p]) for p in required
    )

    complete_paths: list[dict] = []
    abstain_seen = {"available": False, "actions": []}
    invalid_paths: list[dict] = []

    def walk(node: str, tables: dict, steps: list[dict], visited: frozenset) -> None:
        if all(_binding_ok(tables, bindings[p]) for p in required):
            completion = steps[-1] if steps else None
            record = {
                "length": len(steps),
                "actions": [s["action"] for s in steps],
                "steps": list(steps),
                "completion_action": completion["action"] if completion else None,
            }
            reasons: list[str] = []
            if completion is None:
                pass  # the entry state already satisfies the postconditions
            elif completion["action"] in ABSTAIN_ACTIONS:
                reasons.append(f"abstention_completes:{completion['action']}")
            bad = [s["action"] for s in steps if s["action"] in forbidden]
            if bad:
                reasons.append(f"forbidden_action:{','.join(sorted(set(bad)))}")
            if reasons:
                invalid_paths.append({**record, "reason": "; ".join(reasons)})
            else:
                complete_paths.append(record)
            return
        if len(steps) >= max_depth:
            return
        for edge in stg["edges"]:
            if edge["from"] != node:
                continue
            try:
                if not all(_predicate_ok(tables, registry[p]) for p in edge.get("requires", [])):
                    continue
            except (KeyError, ValueError, TypeError):
                continue
            if edge["action"] in ABSTAIN_ACTIONS:
                abstain_seen["available"] = True
                if edge["action"] not in abstain_seen["actions"]:
                    abstain_seen["actions"].append(edge["action"])
            next_tables = copy.deepcopy(tables)
            for effect in edge.get("effects", []) or []:
                _apply_effect(next_tables, effect)
            step = {
                "from": edge["from"],
                "to": edge["to"],
                "action": edge["action"],
                "requires": sorted(edge.get("requires", [])),
            }
            key = (edge["to"], canonical_json(next_tables))
            if key in visited:
                continue
            walk(edge["to"], next_tables, steps + [step], visited | {key})

    try:
        walk(entry, tables0, [], frozenset({(entry, canonical_json(tables0))}))
    except (KeyError, ValueError, TypeError) as exc:  # pragma: no cover - malformed record
        result["notes"].append(f"search aborted on an invalid declaration: {exc}")
        return result

    # Distinct valid fixes = distinct transition sequences (the owner's "set of valid fixes").
    # Two routes with the same action but different target/requirement are different fixes.
    unique: dict[tuple, dict] = {}
    for path in complete_paths:
        key = tuple((step["from"], step["to"], step["action"]) for step in path["steps"])
        unique.setdefault(key, path)
    paths = sorted(unique.values(), key=lambda p: (p["length"], p["actions"]))

    result["n_distinct_valid_paths"] = len(paths)
    result["valid_paths"] = paths[:MAX_LISTED_PATHS]
    result["valid_paths_truncated"] = len(paths) > MAX_LISTED_PATHS
    result["rejected_paths"] = invalid_paths[:MAX_LISTED_PATHS]
    result["abstention_available"] = abstain_seen["available"]
    result["abstention_actions"] = abstain_seen["actions"]
    if paths:
        result["solvable"] = True
        result["shortest_path_length"] = paths[0]["length"]
        result["shortest_path"] = paths[0]["steps"]
        result["outcome_label"] = OUTCOME_SOLVABLE
    elif abstain_seen["available"]:
        result["outcome_label"] = OUTCOME_ABSTENTION_CORRECT
        result["abstain_only_valid_path"] = True
    else:
        result["outcome_label"] = OUTCOME_NO_VALID_PATH
    if not paths and invalid_paths:
        result["notes"].append(
            "every complete path was rejected: " + ", ".join(sorted({p['reason'] for p in invalid_paths}))
        )
    if not paths and not invalid_paths and not abstain_seen["available"]:
        result["notes"].append(f"no applicable transition reaches {mid} required {required}")
    return result


def analyse_mechanism(mechanism: dict, *, source: str | None = None,
                      max_depth: int = MAX_DEPTH,
                      declared_negative_control: list[str] | None = None) -> dict:
    required = _required(mechanism)
    instances = [analyse_instance(mechanism, v, max_depth=max_depth)
                 for v in mechanism.get("variants", [])]
    # Reference-policy outcomes are reported next to the search but are never used by it.
    from .graph import REFERENCE_POLICIES, simulate

    for inst, variant in zip(instances, mechanism.get("variants", [])):
        inst["reference_outcomes"] = {
            policy: simulate(
                mechanism["stg"],
                extra_predicates=tuple(variant.get("active_predicates", [])),
                policy=policy,
                required_postconditions=tuple(required),
            )
            for policy in REFERENCE_POLICIES
        }
    solvable_instances = sum(1 for i in instances if i["outcome_label"] == OUTCOME_SOLVABLE)
    abstention_instances = [
        i["variant_id"] for i in instances if i["outcome_label"] == OUTCOME_ABSTENTION_CORRECT
    ]
    dead_end_instances = [
        i["variant_id"] for i in instances if i["outcome_label"] == OUTCOME_NO_VALID_PATH
    ]
    unexecutable_instances = [
        i["variant_id"] for i in instances if i["outcome_label"] == OUTCOME_UNEXECUTABLE
    ]
    path_counts = [i["n_distinct_valid_paths"] for i in instances]
    declared_control = bool(mechanism.get("negative_control")) or bool(declared_negative_control)
    declared_control_instances = list(mechanism.get("negative_control_instances", []) or [])
    if not declared_control_instances and declared_negative_control:
        declared_control_instances = list(declared_negative_control)
    # A well-formed negative control gives a correct action on every instance: either a real
    # fix or an escalation, never an unexecutable instance and never a dead end.
    well_formed_control = bool(declared_control) and not dead_end_instances \
        and not unexecutable_instances
    if "readiness" in mechanism and mechanism["readiness"].get("benchmark_ready") is not None:
        benchmark_ready = bool(mechanism["readiness"]["benchmark_ready"])
    else:
        benchmark_ready = bool(validate_ready_mechanism(mechanism).get("benchmark_ready"))
    defects = []
    if unexecutable_instances:
        defects.append({
            "class": "UNEXECUTABLE_INSTANCE",
            "instances": unexecutable_instances,
            "why": "an instance activates a predicate with no DB definition (or lacks a "
                   "binding), so it cannot be executed at all",
        })
    if dead_end_instances:
        defects.append({
            "class": "DEAD_END_INSTANCE",
            "instances": dead_end_instances,
            "why": "no valid fix and no applicable abstain/escalate transition: the graph "
                   "offers no correct action on that instance",
        })
    if solvable_instances == 0:
        defects.append({
            "class": "NO_VALID_FIX",
            "instances": [i["variant_id"] for i in instances
                          if i["outcome_label"] != OUTCOME_SOLVABLE],
            "why": "no instance has a valid fix, so the mechanism carries no positive signal",
        })
    return {
        "benchmark_ready": benchmark_ready,
        "strict_benchmark_ready": bool(
            benchmark_ready and solvable_instances >= 1
            and not unexecutable_instances and not dead_end_instances
        ),
        "mechanism_id": mechanism.get("mechanism_id"),
        "family": mechanism.get("family"),
        "title": mechanism.get("title"),
        "status": mechanism.get("status"),
        "source": source,
        "revision_of": mechanism.get("revision_of"),
        "required_postconditions": required,
        "declared_forbidden_side_effects": list(mechanism.get("forbidden_side_effects", [])),
        "declared_forbidden_actions": list(mechanism.get("forbidden_actions", []) or []),
        "forbidden_action_channel": (
            "declared" if mechanism.get("forbidden_actions") is not None else "not_declared"
        ),
        "negative_control_declared": declared_control,
        "negative_control_well_formed": well_formed_control,
        "negative_control_instances": declared_control_instances,
        "n_instances": len(instances),
        "solvable_instances": solvable_instances,
        "abstention_correct_instances": abstention_instances,
        "no_valid_path_instances": dead_end_instances,
        "unexecutable_instances": unexecutable_instances,
        "positive_signal": any(
            o == "PASS" for i in instances for o in i["reference_outcomes"].values()
        ),
        "defects": defects,
        "total_distinct_valid_paths": sum(path_counts),
        "min_valid_paths": min(path_counts) if path_counts else 0,
        "max_valid_paths": max(path_counts) if path_counts else 0,
        "instances": instances,
    }


def analyse_manifest(mechanisms: list[dict], *, sources: dict[str, str] | None = None,
                     label: str = "",
                     declared_negative_controls: dict[str, list[str]] | None = None) -> dict:
    sources = sources or {}
    declared_negative_controls = declared_negative_controls or {}
    records = [analyse_mechanism(
        m, source=sources.get(m.get("mechanism_id")),
        declared_negative_control=declared_negative_controls.get(m.get("mechanism_id")),
    ) for m in mechanisms]
    by_family: dict[str, dict] = {}
    for rec in records:
        fam = rec["family"] or "<unknown>"
        bucket = by_family.setdefault(fam, {
            "mechanisms": 0, "instances": 0,
            "solvable_mechanisms": 0, "solvable_instances": 0,
            "benchmark_ready": 0, "strict_benchmark_ready": 0,
            "negative_controls": 0, "well_formed_negative_controls": 0,
            "negative_control_instances": 0, "abstention_correct_instances": 0,
            "no_valid_path_instances": 0, "unexecutable_instances": 0,
            "total_distinct_valid_paths": 0, "shortest_path_lengths": [],
        })
        bucket["mechanisms"] += 1
        bucket["instances"] += rec["n_instances"]
        bucket["solvable_mechanisms"] += 1 if rec["solvable_instances"] else 0
        bucket["solvable_instances"] += rec["solvable_instances"]
        bucket["benchmark_ready"] += 1 if rec["benchmark_ready"] else 0
        bucket["strict_benchmark_ready"] += 1 if rec["strict_benchmark_ready"] else 0
        bucket["negative_controls"] += 1 if rec["negative_control_declared"] else 0
        bucket["well_formed_negative_controls"] += 1 if rec["negative_control_well_formed"] else 0
        bucket["negative_control_instances"] += len(rec["negative_control_instances"])
        bucket["abstention_correct_instances"] += len(rec["abstention_correct_instances"])
        bucket["no_valid_path_instances"] += len(rec["no_valid_path_instances"])
        bucket["unexecutable_instances"] += len(rec["unexecutable_instances"])
        bucket["total_distinct_valid_paths"] += rec["total_distinct_valid_paths"]
        bucket["shortest_path_lengths"].extend(
            i["shortest_path_length"] for i in rec["instances"]
            if i["shortest_path_length"] is not None
        )
    labels = Counter(i["outcome_label"] for r in records for i in r["instances"])
    return {
        "label": label,
        "side_effect_model": SIDE_EFFECT_MODEL,
        "max_depth": MAX_DEPTH,
        "n_mechanisms": len(records),
        "n_instances": sum(r["n_instances"] for r in records),
        "instance_labels": dict(sorted(labels.items())),
        "solvable_mechanisms": [r["mechanism_id"] for r in records if r["solvable_instances"]],
        "negative_controls": [r["mechanism_id"] for r in records if r["negative_control_declared"]],
        "well_formed_negative_controls": [
            r["mechanism_id"] for r in records if r["negative_control_well_formed"]
        ],
        "abstention_correct_instances": [
            f"{r['mechanism_id']}/{vid}"
            for r in records for vid in r["abstention_correct_instances"]
        ],
        "no_valid_path_instances": [
            f"{r['mechanism_id']}/{vid}"
            for r in records for vid in r["no_valid_path_instances"]
        ],
        "unexecutable_instances": [
            f"{r['mechanism_id']}/{vid}"
            for r in records for vid in r["unexecutable_instances"]
        ],
        "defects": [
            {"mechanism_id": r["mechanism_id"], **d}
            for r in records for d in r["defects"]
        ],
        "distinct_valid_path_by_mechanism": {
            r["mechanism_id"]: [i["n_distinct_valid_paths"] for i in r["instances"]]
            for r in records
        },
        "by_family": by_family,
        "mechanisms": records,
    }
