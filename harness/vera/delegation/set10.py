"""Read-only access to active set 10.

Set 10 is self-contained: every mechanism record is stored in full in
``manifests/reliability/dev-active-set-10.json``.  Nothing here edits that file, the
abstention declarations, or any readiness/mechanism module.  The loader mirrors the pilot's
instance construction (``_instance_from_record``) so the delegation study sees exactly the
same tables, graph, predicates and frozen solvability labels as v1 does for its own set.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..canonical import sha256_file
from ..mechanisms.solvability import (
    OUTCOME_ABSTENTION_CORRECT,
    OUTCOME_SOLVABLE,
    analyse_instance,
)
from ..pilot.instances import DEV_FAMILIES, Instance, _instance_from_record

ROOT = Path(__file__).resolve().parents[2]
ACTIVE_SET_10 = ROOT / "manifests" / "reliability" / "dev-active-set-10.json"


def active_set_10_sha256() -> str:
    return sha256_file(ACTIVE_SET_10)


def load_mechanisms() -> list[dict]:
    data = json.loads(ACTIVE_SET_10.read_text())
    mechanisms = data.get("mechanisms")
    if not isinstance(mechanisms, list) or not mechanisms:
        raise ValueError(f"{ACTIVE_SET_10} has no mechanisms list")
    return mechanisms


def load_instances() -> list[Instance]:
    instances: list[Instance] = []
    for mechanism in load_mechanisms():
        for variant in mechanism.get("variants", []):
            instances.append(_instance_from_record(mechanism, variant))
    return instances


def analysis_of(instance: Instance) -> dict:
    return analyse_instance(instance.mechanism, instance.variant)


def mechanism_variants() -> list[dict]:
    """One row per mechanism with its frozen variant labels (read-only)."""
    rows: list[dict] = []
    for mechanism in load_mechanisms():
        labels: list[dict] = []
        for variant in mechanism.get("variants", []):
            analysis = analyse_instance(mechanism, variant)
            labels.append(
                {
                    "variant_id": variant["variant_id"],
                    "outcome_label": analysis["outcome_label"],
                    "n_distinct_valid_paths": analysis["n_distinct_valid_paths"],
                    "shortest_path": [step["action"] for step in analysis["shortest_path"]],
                }
            )
        rows.append(
            {
                "mechanism_id": mechanism["mechanism_id"],
                "family": mechanism["family"],
                "variants": labels,
            }
        )
    return rows


def select_both_label_mechanisms(limit: int = 3) -> list[dict]:
    """Deterministically select one mechanism per family that has a fix and an abstention.

    Family order is the frozen ``DEV_FAMILIES`` order; within a family the first
    mechanism id (sorted) that has a SOLVABLE variant with a non-empty action intersection
    and an ABSTENTION_CORRECT variant.  Nothing is chosen by outcome.
    """
    mechanisms = load_mechanisms()
    by_family: dict[str, list[dict]] = {}
    for mechanism in mechanisms:
        by_family.setdefault(mechanism["family"], []).append(mechanism)
    selected: list[dict] = []
    for family in DEV_FAMILIES:
        candidates = sorted(by_family.get(family, []), key=lambda m: m["mechanism_id"])
        for mechanism in candidates:
            solvable = None
            abstention = None
            for variant in mechanism.get("variants", []):
                analysis = analyse_instance(mechanism, variant)
                if analysis["outcome_label"] == OUTCOME_SOLVABLE and analysis["valid_paths"]:
                    intersection = set(analysis["valid_paths"][0]["actions"])
                    for path in analysis["valid_paths"]:
                        intersection &= set(path["actions"])
                    if intersection and solvable is None:
                        solvable = {
                            "variant": variant,
                            "analysis": analysis,
                            "required_actions": sorted(intersection),
                        }
                elif analysis["outcome_label"] == OUTCOME_ABSTENTION_CORRECT:
                    abstention = {"variant": variant, "analysis": analysis}
            if solvable and abstention:
                selected.append(
                    {
                        "mechanism_id": mechanism["mechanism_id"],
                        "family": family,
                        "mechanism": mechanism,
                        "solvable": solvable,
                        "abstention": abstention,
                    }
                )
                break
        if len(selected) >= limit:
            break
    return selected[:limit]
