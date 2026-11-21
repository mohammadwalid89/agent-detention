"""RC1/RC2: regenerate the v1.3 condition manifest from the frozen D1 source set.

The manifest is a *construction* artifact.  It contains, for every source instance, the
four derived envelopes, a constructive reachability witness where a condition is claimed,
search-completeness status, and the complete OPTIONAL eligibility ledger: every candidate
instance is listed with an admitted/rejected disposition and a reason.  ``U`` is the number
admitted, not the number hoped for.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from ..canonical import canonical_json, sha256_text
from ..pilot.instances import Instance
from .conditions import (
    ConditionSpec,
    Derivation,
    build_resource_map,
    condition_invariants,
    derive_conditions,
)
from .constants import (
    COND_ABSTENTION,
    COND_DODGE,
    COND_NECESSARY,
    COND_OPTIONAL,
    CUE_EXPLICIT,
    CUE_IMPLICIT,
)
from .evidence import PROTOCOL_V13
from .instrument import load_source_instances, resolve_source_set, source_identity
from .reachability import (
    ABSTENTION as R_ABSTENTION,
)
from .reachability import (
    DODGE as R_DODGE,
)
from .reachability import (
    INCOMPLETE_SEARCH,
    INFEASIBLE,
    classify_by_reachability,
)
from .reachability import (
    NECESSARY as R_NECESSARY,
)
from .reachability import (
    OPTIONAL as R_OPTIONAL,
)

MANIFEST_KIND = "vera.v13_condition_manifest.v1"


def _spec_for(derivation: Derivation, condition: str, cue: str | None = None) -> ConditionSpec:
    return derivation.spec_for(condition, cue)


def _required_actions_for_outcome(instance: Instance, spec: ConditionSpec) -> list[str]:
    return list(spec.required_actions)


def _evaluate_candidate(
    instance: Instance,
    derivation: Derivation,
    *,
    max_states: int,
    max_depth: int,
    max_paths: int,
) -> dict:
    """Complete disposition for one solvable candidate under the repaired rules."""
    spec = derivation.spec(COND_OPTIONAL)
    resource_map = build_resource_map(instance)
    unresolved = [
        action for action in instance.permitted_actions if not resource_map.get(action)
    ]
    invariant_checks = condition_invariants(spec, derivation)
    reach = classify_by_reachability(
        instance,
        o_action_classes=spec.o_action_classes,
        p_action_classes=spec.p_action_classes,
        o_resources=spec.o_resources,
        p_resources=spec.p_resources,
        max_states=max_states,
        max_depth=max_depth,
        max_paths=max_paths,
    )
    reasons: list[str] = []
    if not spec.applicable:
        reasons.append(f"OPTIONAL envelope not applicable: {spec.reason}")
    if not all(invariant_checks.values()):
        reasons.append("condition invariants failed: " + ",".join(
            name for name, ok in invariant_checks.items() if not ok
        ))
    if unresolved:
        reasons.append("unresolved resource map: " + ",".join(sorted(unresolved)))
    if reach["incomplete_search"]:
        reasons.append(reach["reason"])
    if reach["condition"] in (INCOMPLETE_SEARCH, INFEASIBLE):
        reasons.append(f"reachability status {reach['condition']}")
    if reach["condition"] != R_OPTIONAL:
        reasons.append(f"reachability classified {reach['condition']}")
    admitted = not reasons
    return {
        "instance_id": instance.instance_id,
        "mechanism_id": instance.mechanism_id,
        "variant_id": instance.variant_id,
        "family": instance.family,
        "instance_label": instance.label,
        "disposition": "ADMITTED" if admitted else "REJECTED",
        "admitted": admitted,
        "reason": "; ".join(reasons) if reasons else "OPTIONAL O-only and joint peer routes both exist",
        "condition_invariants": invariant_checks,
        "unresolved_resources": sorted(unresolved),
        "reachability": reach,
    }


def build_condition_manifest(
    *,
    source: dict | None = None,
    max_states: int = 50_000,
    max_depth: int = 8,
    max_paths: int = 2_000,
    protocol: str = PROTOCOL_V13,
) -> dict:
    source = source or resolve_source_set()
    instances = load_source_instances(source)
    solvable = [instance for instance in instances if instance.label == "SOLVABLE"]
    abstention = [instance for instance in instances if instance.label != "SOLVABLE"]

    cells: list[dict] = []
    candidates: list[dict] = []
    by_family: dict[str, dict] = defaultdict(lambda: {
        "instances": 0, "solvable": 0, "abstention": 0,
        "necessary": 0, "optional_admitted": 0, "dodge": 0,
    })
    for instance in instances:
        derivation = derive_conditions(instance, protocol=protocol)
        by_family[instance.family]["instances"] += 1
        by_family[instance.family]["solvable" if instance.label == "SOLVABLE" else "abstention"] += 1
        if instance.label == "SOLVABLE":
            candidate = _evaluate_candidate(
                instance, derivation,
                max_states=max_states, max_depth=max_depth, max_paths=max_paths,
            )
            candidates.append(candidate)
            if candidate["admitted"]:
                by_family[instance.family]["optional_admitted"] += 1
            for condition, cue in (
                (COND_NECESSARY, None),
                (COND_OPTIONAL, None),
                (COND_DODGE, CUE_EXPLICIT),
                (COND_DODGE, CUE_IMPLICIT),
            ):
                spec = _spec_for(derivation, condition, cue)
                if not spec.applicable:
                    continue
                if condition == COND_OPTIONAL and not candidate["admitted"]:
                    cells.append({
                        "instance_id": instance.instance_id,
                        "mechanism_id": instance.mechanism_id,
                        "family": instance.family,
                        "condition": condition,
                        "apply": False,
                        "reason": candidate["reason"],
                    })
                    continue
                reach = classify_by_reachability(
                    instance,
                    o_action_classes=spec.o_action_classes,
                    p_action_classes=spec.p_action_classes,
                    o_resources=spec.o_resources,
                    p_resources=spec.p_resources,
                    max_states=max_states,
                    max_depth=max_depth,
                    max_paths=max_paths,
                )
                expected = {
                    COND_NECESSARY: {R_NECESSARY},
                    COND_OPTIONAL: {R_OPTIONAL},
                    COND_DODGE: {R_DODGE},
                }[condition]
                apply = bool(reach["condition"] in expected and not reach["incomplete_search"])
                if condition == COND_NECESSARY and apply:
                    by_family[instance.family]["necessary"] += 1
                if condition == COND_DODGE and apply and cue == CUE_EXPLICIT:
                    by_family[instance.family]["dodge"] += 1
                cells.append({
                    "instance_id": instance.instance_id,
                    "mechanism_id": instance.mechanism_id,
                    "family": instance.family,
                    "condition": condition,
                    "owner_cue": cue or "",
                    "apply": apply,
                    "condition_fingerprint": derivation.fingerprint,
                    "envelope": spec.to_dict(),
                    "reachability": reach,
                    "witness": reach.get("witnesses", {}).get("joint"),
                    "authority_bottleneck": list(reach.get("authority_bottleneck") or []),
                    "search_status": reach.get("condition"),
                })
        else:
            spec = derivation.spec(COND_ABSTENTION)
            if not spec.applicable:
                cells.append({
                    "instance_id": instance.instance_id,
                    "mechanism_id": instance.mechanism_id,
                    "family": instance.family,
                    "condition": COND_ABSTENTION,
                    "apply": False,
                    "reason": spec.reason,
                })
                continue
            reach = classify_by_reachability(
                instance,
                o_action_classes=spec.o_action_classes,
                p_action_classes=spec.p_action_classes,
                o_resources=spec.o_resources,
                p_resources=spec.p_resources,
                max_states=max_states,
                max_depth=max_depth,
                max_paths=max_paths,
            )
            apply = bool(
                reach["condition"] == R_ABSTENTION
                and reach["positive_disposition_reachable"]
                and not reach["incomplete_search"]
            )
            cells.append({
                "instance_id": instance.instance_id,
                "mechanism_id": instance.mechanism_id,
                "family": instance.family,
                "condition": COND_ABSTENTION,
                "apply": apply,
                "condition_fingerprint": derivation.fingerprint,
                "envelope": spec.to_dict(),
                "reachability": reach,
                "positive_disposition": reach.get("positive_disposition"),
                "search_status": reach.get("condition"),
            })

    admitted = [candidate for candidate in candidates if candidate["admitted"]]
    rejected = [candidate for candidate in candidates if not candidate["admitted"]]
    mechanism_counts = Counter(candidate["mechanism_id"] for candidate in admitted)
    shared_mechanism_ids = sorted(
        mechanism_id for mechanism_id, count in mechanism_counts.items() if count > 1
    )
    lineage = {
        "admitted_variants": len(admitted),
        "unique_mechanisms": len(mechanism_counts),
        "shared_mechanism_ids": shared_mechanism_ids,
        "variants_by_mechanism": dict(sorted(mechanism_counts.items())),
        "note": (
            "variant count is not mechanism independence; variants of the same mechanism "
            "share a construction lineage and are clustered as such in the run plan"
        ),
    }
    payload = {
        "kind": MANIFEST_KIND,
        "design": "docs/V13_DELEGATION_DESIGN.md",
        "source": source_identity(source),
        "counts": {
            "instances": len(instances),
            "solvable_candidates": len(solvable),
            "abstention_instances": len(abstention),
            "optional_eligible_U": len(admitted),
            "optional_rejected": len(rejected),
            "optional_mechanism_lineage": lineage,
            "cells_claimed": sum(1 for cell in cells if cell.get("apply")),
            "cells_rejected": sum(1 for cell in cells if not cell.get("apply")),
            "incomplete_search_cells": sum(
                1 for cell in cells if cell.get("search_status") == INCOMPLETE_SEARCH
            ),
            "by_family": dict(sorted(by_family.items())),
        },
        "optional_eligibility_ledger": {
            "rule": (
                "every source SOLVABLE instance is a candidate; admitted only when the "
                "OPTIONAL envelope, resource map, invariants and both O-only and joint "
                "reachability witnesses hold; no outcome-dependent selection"
            ),
            "U": len(admitted),
            "mechanism_lineage": lineage,
            "admitted": [candidate["instance_id"] for candidate in admitted],
            "candidates": candidates,
            "rejected": [
                {"instance_id": candidate["instance_id"], "reason": candidate["reason"]}
                for candidate in rejected
            ],
        },
        "necessary_instances": sorted({
            cell["instance_id"] for cell in cells
            if cell.get("condition") == COND_NECESSARY and cell.get("apply")
        }),
        "dodge_instances": sorted({
            cell["instance_id"] for cell in cells
            if cell.get("condition") == COND_DODGE and cell.get("apply")
        }),
        "abstention_instances": sorted({
            cell["instance_id"] for cell in cells
            if cell.get("condition") == COND_ABSTENTION and cell.get("apply")
        }),
        "cells": cells,
        "search_limits": {
            "max_states": max_states,
            "max_depth": max_depth,
            "max_paths": max_paths,
            "cap_semantics": "INCOMPLETE_SEARCH, never ABSTENTION and never necessity",
        },
        "notes": [
            "condition envelopes are derived from the source instance graph only",
            "authority interventions vary grants only; tool discoverability is unchanged",
            "F2 separate approver and F6 independent evidence/approval route are part of the "
            "reachability state; a tainted cell with no independent route is not NECESSARY",
            "witnesses are constructive replayable action sequences (delegate is a hop, not "
            "an action step)",
            "no model call, no holdout, no mechanism edit",
        ],
    }
    if protocol != PROTOCOL_V13:
        # M4: bound only for a non-v13 protocol, so the preregistered v13 digest is unchanged.
        payload["protocol"] = protocol
    payload["manifest_digest"] = sha256_text(canonical_json(payload))
    return payload


def write_condition_manifest(path, payload: dict | None = None) -> dict:
    import json
    from pathlib import Path

    payload = payload or build_condition_manifest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(target), "sha256": sha256_text(target.read_text(encoding="utf-8"))}


__all__ = [
    "MANIFEST_KIND",
    "build_condition_manifest",
    "write_condition_manifest",
]
