"""JOB50: freeze the L3 multi-agent extension block before any main-grid outcome.

The main set-12 grid has only four L3 construction lineages and twenty claimed L3
condition cells (JOB49).  The owner therefore declared a prospective **L3 multi-agent
extension** that is authored after the public breadth research, reviewed by the owner and
by Astra, sealed, and run after the main grid this sprint.  This module freezes that plan
now, while no main-grid outcome exists.

What is frozen here (never chosen after an outcome):

* the target: **>= 10 L3 lineage clusters and >= 30 L3 claimed condition cells** for the
  main grid plus the extension combined, with the exact additional requirement derived
  from the frozen JOB49 counts;
* the authoring/review/run order: breadth research -> authoring -> owner review -> Astra
  review -> review sheets -> seal -> outcomes opened -> extension run after the main grid;
* blindness: the extension is selected without opening any main-grid outcome.  If the
  writing, sheets and seal cannot be completed before the first main outcome is opened,
  the combined estimate is **not** computed; the extension is analysed separately and
  labelled ``POST_MAIN``;
* the re-estimation rule: the PREREG-5 chain-depth moderator is re-estimated on the main
  grid alone (unchanged), on main + extension combined, and on the extension alone, with
  the same estimator, adequacy rule and exploratory label;
* isolation: extension cases are disjoint from the main case set and can never enter a
  main-grid denominator or analysis;
* the unchanged invariants: primary hypotheses, the four-test Holm family at 5%, the
  Detention cap F=660 and the frozen main assignment.

The extension cases themselves do not exist yet.  They are bound at the named stage
``L3_EXTENSION_AUTHORING_AND_SEAL_BEFORE_MAIN_OUTCOMES`` and no placeholder is called
final.  Everything here is offline arithmetic over frozen bytes: no model call, no network,
no key and no holdout access.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..canonical import sha256_text
from .chain_depth import MINIMUM_CELL_SIZE, build_chain_depth_table
from .runplan_v13 import ENROLLED_PATTERNS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "manifests" / "delegation" / "V13_L3_EXTENSION.json"
CHAIN_DEPTH_PATH = ROOT / "manifests" / "delegation" / "V13_CHAIN_DEPTH.json"
RUN_PLAN_PATH = ROOT / "manifests" / "delegation" / "V13_RUN_PLAN.json"

KIND = "vera.v13_l3_multi_agent_extension.v1"
VERSION = "JOB50-2026-11-15"
LEVEL = "L3"
PREREG_VERSION_ID = "V13-PREREG-6"
RESEARCH_REQUIRED = "vera-research-breadth-20261114"
AUTHORING_STAGE = "L3_EXTENSION_AUTHORING_AND_SEAL_BEFORE_MAIN_OUTCOMES"

#: Both targets are the already-frozen JOB49 minimum cell sizes; JOB50 binds them
#: explicitly to the L3 extension rather than inventing a new threshold.
MINIMUM_LINEAGE_CLUSTERS = int(MINIMUM_CELL_SIZE["lineage_clusters_per_level"])
MINIMUM_CLAIMED_CONDITION_CELLS = int(MINIMUM_CELL_SIZE["family_condition_level_cells"])

CASE_SETS = ("main", "extension", "main_plus_extension")
BLIND_LABEL = "PRE_MAIN_BLIND"
POST_MAIN_LABEL = "POST_MAIN"
PRIMARY_FAMILY = (
    "within_backend_label_effect",
    "detention_A_minus_B",
    "detention_A_minus_C",
    "detention_A_minus_D",
)
HOLM_ALPHA = 0.05
DETENTION_CAP = 660
RETRY_REPEATS_PER_FAILURE = 1


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(payload: dict) -> str:
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sha256_text(text)


def _load_run_plan() -> dict:
    return _load_json(RUN_PLAN_PATH)


def build_l3_extension_declaration(
    *,
    chain_depth_table: dict | None = None,
    run_plan: dict | None = None,
) -> dict:
    """Return the frozen L3 extension declaration.

    Deterministic in the frozen run plan and the JOB49 counts: rebuilding on the same
    bytes returns the same digest.  The extension case ids are deliberately absent (they
    are bound at ``AUTHORING_STAGE`` after the breadth research).
    """
    run_plan = run_plan or _load_run_plan()
    chain_depth_table = chain_depth_table or build_chain_depth_table(
        run_plan=run_plan
    )
    l3 = chain_depth_table["thinness"][LEVEL]
    baseline_lineages = int(l3["lineage_clusters"])
    baseline_cells = int(l3["claimed_condition_cells"])
    baseline_gap_lineages = max(0, MINIMUM_LINEAGE_CLUSTERS - baseline_lineages)
    baseline_gap_cells = max(0, MINIMUM_CLAIMED_CONDITION_CELLS - baseline_cells)
    episode_counts = run_plan.get("episode_counts") or {}
    detention = run_plan.get("detention") or {}

    declaration = {
        "kind": KIND,
        "version": VERSION,
        "prereg_version": PREREG_VERSION_ID,
        "level": LEVEL,
        "status": "DECLARED_FROZEN_EXTENSION_NOT_YET_AUTHORED",
        "frozen_before_any_main_grid_outcome": True,
        "required_research": RESEARCH_REQUIRED,
        "authoring_stage": AUTHORING_STAGE,
        "authoring": {
            "authored_after_research": RESEARCH_REQUIRED,
            "research_must_complete_first": True,
            "authored_from_public_breadth_research": True,
            "owner_reviewed": True,
            "astra_reviewed": True,
            "review_sheets_required": True,
            "run_order": "AFTER_MAIN_GRID_THIS_SPRINT",
            "sequence": [
                f"{RESEARCH_REQUIRED} completed (public breadth research; no main-grid "
                "outcome opened)",
                "extension cases authored from the breadth research "
                "(no main-grid outcome opened)",
                "owner review bound to the exact extension bytes",
                "Astra review bound to the exact extension bytes",
                "review sheets completed for every extension case",
                "extension sealed (hash-bound) before the first main-grid outcome is "
                "opened",
                "main grid produces outcomes; outcomes stay sealed until the extension "
                "seal exists",
                "main outcomes opened; main analyses run with extension cases excluded",
                "extension runs after the main grid this sprint with the same runner, "
                "routes, conditions, identity/label design and analysis code",
            ],
            "authored_cases": (
                "MECHANICAL_LATER_STAGE at "
                f"{AUTHORING_STAGE}: exact extension mechanism/instance ids, lineages, "
                "assignment table, sheets and seal are appended; no placeholder is final"
            ),
            "no_placeholder_case_ids": True,
        },
        "targets": {
            "level": LEVEL,
            "minimum_lineage_clusters": MINIMUM_LINEAGE_CLUSTERS,
            "minimum_claimed_condition_cells": MINIMUM_CLAIMED_CONDITION_CELLS,
            # The target applies to the extension's own L3 cases, so the extension-alone
            # re-estimate is not thinner than the level it is meant to strengthen.  The
            # main + extension combined count is automatically above it.
            "scope": "extension",
            "extension_minimum": {
                "lineage_clusters": MINIMUM_LINEAGE_CLUSTERS,
                "claimed_condition_cells": MINIMUM_CLAIMED_CONDITION_CELLS,
            },
            "baseline_main_grid": {
                "lineage_clusters": baseline_lineages,
                "claimed_condition_cells": baseline_cells,
            },
            "baseline_gap": {
                "lineage_clusters": baseline_gap_lineages,
                "claimed_condition_cells": baseline_gap_cells,
            },
            "main_plus_extension_minimum": {
                "lineage_clusters": baseline_lineages + MINIMUM_LINEAGE_CLUSTERS,
                "claimed_condition_cells": (
                    baseline_cells + MINIMUM_CLAIMED_CONDITION_CELLS
                ),
            },
            "additional_required_from_extension": {
                "lineage_clusters": MINIMUM_LINEAGE_CLUSTERS,
                "claimed_condition_cells": MINIMUM_CLAIMED_CONDITION_CELLS,
            },
            "frozen_rule_source": (
                "manifests/delegation/V13_CHAIN_DEPTH.json#/minimum_cell_size"
            ),
            "reach_rule": (
                "the extension's own L3 cases must reach >= 10 lineage clusters and >= 30 "
                "claimed condition cells; main + extension combined is then >= 14 "
                "lineages and >= 50 cells.  Extension cases must add new construction "
                "lineages, not relabel existing ones"
            ),
            "shortage_rule": (
                "if the extension closes below target, report the achieved lineages/cells "
                "and the reason; the moderator stays exploratory, the extension-only "
                "estimate is still reported, and no case is added to reach significance "
                "after an outcome"
            ),
            "thin_until_reached": True,
        },
        "chain_depth_reestimation": {
            "moderator": "PREREG-5 chain_depth_moderator",
            "table_path": "manifests/delegation/V13_CHAIN_DEPTH.json",
            "table_digest": str(chain_depth_table.get("digest") or ""),
            "estimator_and_test_source": (
                "V13-PREREG-6 secondary_analyses.chain_depth_moderator "
                "(identical to the V13-PREREG-5 estimator and test)"
            ),
            "same_estimator_and_test": True,
            "estimates": [
                {
                    "id": "main_grid_only",
                    "case_set": "main",
                    "role": (
                        "the unchanged PREREG-5 secondary moderator; extension cases are "
                        "never added to this estimate"
                    ),
                },
                {
                    "id": "main_plus_extension_combined",
                    "case_set": "main_plus_extension",
                    "role": (
                        "re-estimate the moderator on the combined L3 cases; computed only "
                        f"when the extension label is {BLIND_LABEL}"
                    ),
                },
                {
                    "id": "extension_only",
                    "case_set": "extension",
                    "role": (
                        "re-estimate the moderator on the extension cases alone; always "
                        "reported separately and labelled exploratory"
                    ),
                },
            ],
            "minimum_cell_size": dict(MINIMUM_CELL_SIZE),
            "label": "EXPLORATORY_UNLESS_FROZEN_ADEQUACY_RULE_MET",
            "never_confirmatory": True,
            "no_pooling_to_manufacture_precision": True,
            "no_level_dropping": True,
            "combined_estimate_requires_blind_label": True,
        },
        "isolation": {
            "extension_cases_cannot_influence_any_main_grid_analysis": True,
            "main_grid_case_set_frozen": True,
            "extension_cases_excluded_from_main_denominators": True,
            "extension_mechanism_and_instance_ids_disjoint_from_main": True,
            "disjointness_check": (
                "vera.delegation.l3_extension.assert_extension_disjoint"
            ),
            "separate_assignment_table": True,
            "separate_resume_namespace": True,
            "separate_seal": True,
            "separate_review_sheets": True,
            "analysis_case_sets": list(CASE_SETS),
            "main_analyses_never_read_extension_rows": True,
            "extension_never_changes_main_multiplicity": True,
            "main_assignment_unchanged": True,
        },
        "blindness": {
            "selection_blind_to_main_grid_outcomes": True,
            "preconditions_before_any_main_outcome_opened": [
                f"{RESEARCH_REQUIRED} complete",
                "extension authored",
                "owner review complete",
                "Astra review complete",
                "review sheets complete",
                "extension sealed",
            ],
            "seal_must_precede_outcome_opening": True,
            "verification": (
                "a hash-bound extension seal plus owner attestation is recorded before the "
                "first main-grid outcome is opened; outcomes stay sealed until then"
            ),
            "if_not_possible": {
                "analysis": "extension_only",
                "label": POST_MAIN_LABEL,
                "combined_estimate": "NOT_COMPUTED",
                "claim": (
                    "no blinding claim; the extension is analysed separately and labelled "
                    "post-main"
                ),
            },
            "labels": {"blind": BLIND_LABEL, "fallback": POST_MAIN_LABEL},
            "combined_allowed_labels": [BLIND_LABEL],
            "no_retroactive_blinding_claim": True,
        },
        "design_binding": {
            "same_runner": {
                "module": "vera/delegation/runner_v13.py",
                "class": "GridRunner",
                "unchanged": True,
            },
            "same_routes": {
                "registry": "vera/delegation/routes/registry.json",
                "route_protocol": (
                    "manifests/delegation/V13_ROUTE_PROTOCOL.json"
                ),
                "no_route_substitution": True,
            },
            "same_conditions": ["ABSTENTION", "DODGE", "NECESSARY", "OPTIONAL"],
            "same_organization_patterns": list(ENROLLED_PATTERNS),
            "same_identity_and_label_design": {
                "module": "vera/delegation/identity_assignment.py",
                "randomised_label_permutations": True,
                "same_seed_family": True,
                "unchanged": True,
            },
            "same_measurement_contract": "vera/delegation/measurement.py",
            "same_analysis_code": [
                "vera/delegation/power.py",
                "vera/delegation/chain_depth.py",
                "vera/delegation/preregistration.py",
            ],
        },
        "unchanged": {
            "run_plan_version_id": str(run_plan.get("version_id") or ""),
            "run_plan_digest": sha256_text(json.dumps(run_plan, sort_keys=True)),
            "primary_hypotheses": list(run_plan.get("primary_family") or []),
            "holm_family": list(run_plan.get("primary_family") or []),
            "holm_alpha": HOLM_ALPHA,
            "multiplicity": run_plan.get("multiplicity"),
            "detention_cap": int(detention.get("cap") or DETENTION_CAP),
            "detention_retry_repeats_per_failure": int(
                detention.get("retry_repeats_per_failure")
                or RETRY_REPEATS_PER_FAILURE
            ),
            "main_assignment": {
                "main_enrolled": int(episode_counts.get("main_enrolled") or 0),
                "main_plus_detention_enrolled": int(
                    episode_counts.get("main_plus_detention_enrolled") or 0
                ),
                "detention_retry_episodes_upper": int(
                    episode_counts.get("detention_retry_episodes_upper") or 0
                ),
            },
            "extension_adds_primary_hypotheses": False,
            "extension_enters_confirmatory_holm_family": False,
            "extension_changes_main_assignment": False,
            "extension_changes_detention_cap": False,
        },
        "no_holdout": True,
        "no_outcome_dependent_case_selection": True,
        "no_extension_primary_hypothesis": True,
        "notes": [
            "the extension block is declared and frozen now; the cases are authored later "
            "from the public breadth research",
            "the main case set, assignment, multiplicity, cap and primary hypotheses are "
            "unchanged by this declaration",
            "a post-main extension is never relabelled as blind and never enters the "
            "combined estimate",
        ],
    }
    declaration["digest"] = _digest(declaration)
    return declaration


def write_l3_extension(
    path: str | Path | None = None, declaration: dict | None = None
) -> dict:
    target = Path(path) if path is not None else DEFAULT_PATH
    declaration = declaration or build_l3_extension_declaration()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(declaration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "path": str(target),
        "sha256": sha256_text(target.read_text(encoding="utf-8")),
        "digest": declaration.get("digest"),
    }


def load_l3_extension(path: str | Path | None = None) -> dict:
    target = Path(path) if path is not None else DEFAULT_PATH
    return _load_json(target)


def extension_label(*, sealed_before_any_main_outcome: bool) -> str:
    """The only allowed labels: blind pre-main, or the post-main fallback."""
    return BLIND_LABEL if sealed_before_any_main_outcome else POST_MAIN_LABEL


def combined_estimate_allowed(label: str) -> bool:
    """The combined estimate is allowed only for the blind pre-main label."""
    return label == BLIND_LABEL


def assert_extension_disjoint(
    main_case_ids: list[str] | tuple[str, ...] | set[str],
    extension_case_ids: list[str] | tuple[str, ...] | set[str],
) -> None:
    """Fail closed if any extension case id also appears in the main case set."""
    overlap = sorted(set(map(str, main_case_ids)) & set(map(str, extension_case_ids)))
    if overlap:
        raise ValueError(
            "extension case ids overlap the main grid: " + ", ".join(overlap)
        )


def verify_l3_extension(
    declaration: dict | None = None,
    *,
    run_plan: dict | None = None,
    chain_depth_table: dict | None = None,
) -> dict:
    """Check the frozen invariants; return ``{"valid", "problems"}``."""
    declaration = declaration or load_l3_extension()
    run_plan = run_plan or _load_run_plan()
    chain_depth_table = chain_depth_table or build_chain_depth_table(
        run_plan=run_plan
    )
    problems: list[str] = []
    payload = {key: value for key, value in declaration.items() if key != "digest"}
    if declaration.get("digest") != _digest(payload):
        problems.append("declaration digest mismatch")
    if declaration.get("kind") != KIND:
        problems.append(f"unexpected kind {declaration.get('kind')!r}")
    if declaration.get("prereg_version") != PREREG_VERSION_ID:
        problems.append("declaration is not bound to V13-PREREG-6")
    if declaration.get("frozen_before_any_main_grid_outcome") is not True:
        problems.append("declaration is not frozen before main outcomes")
    if declaration.get("status") != "DECLARED_FROZEN_EXTENSION_NOT_YET_AUTHORED":
        problems.append("declaration status changed")

    targets = declaration.get("targets") or {}
    if int(targets.get("minimum_lineage_clusters") or 0) != MINIMUM_LINEAGE_CLUSTERS:
        problems.append("L3 lineage-cluster target changed")
    if (
        int(targets.get("minimum_claimed_condition_cells") or 0)
        != MINIMUM_CLAIMED_CONDITION_CELLS
    ):
        problems.append("L3 claimed-cell target changed")
    if targets.get("scope") != "extension":
        problems.append("L3 target is not bound to the extension cases")
    extension_minimum = targets.get("extension_minimum") or {}
    if int(extension_minimum.get("lineage_clusters") or 0) != MINIMUM_LINEAGE_CLUSTERS:
        problems.append("extension-only lineage target changed")
    if (
        int(extension_minimum.get("claimed_condition_cells") or 0)
        != MINIMUM_CLAIMED_CONDITION_CELLS
    ):
        problems.append("extension-only claimed-cell target changed")
    l3 = chain_depth_table["thinness"][LEVEL]
    baseline = targets.get("baseline_main_grid") or {}
    if int(baseline.get("lineage_clusters", -1)) != int(l3["lineage_clusters"]):
        problems.append("baseline L3 lineage count does not match the frozen table")
    if int(baseline.get("claimed_condition_cells", -1)) != int(
        l3["claimed_condition_cells"]
    ):
        problems.append("baseline L3 claimed-cell count does not match the frozen table")

    unchanged = declaration.get("unchanged") or {}
    family = list(run_plan.get("primary_family") or [])
    if list(unchanged.get("primary_hypotheses") or []) != family:
        problems.append("primary hypotheses changed")
    if list(unchanged.get("holm_family") or []) != family:
        problems.append("Holm family changed")
    if family != list(PRIMARY_FAMILY):
        problems.append("run-plan primary family is not the frozen family")
    if int(unchanged.get("detention_cap") or 0) != DETENTION_CAP:
        problems.append("Detention cap changed")
    if int((run_plan.get("detention") or {}).get("cap") or 0) != DETENTION_CAP:
        problems.append("run-plan Detention cap changed")
    if int(unchanged.get("detention_retry_repeats_per_failure") or 0) != (
        RETRY_REPEATS_PER_FAILURE
    ):
        problems.append("Detention retry repeats changed")
    if unchanged.get("extension_enters_confirmatory_holm_family") is not False:
        problems.append("extension enters the confirmatory family")
    if unchanged.get("extension_adds_primary_hypotheses") is not False:
        problems.append("extension adds a primary hypothesis")
    if unchanged.get("extension_changes_main_assignment") is not False:
        problems.append("extension changes the main assignment")

    blindness = declaration.get("blindness") or {}
    fallback = blindness.get("if_not_possible") or {}
    if fallback.get("label") != POST_MAIN_LABEL:
        problems.append("post-main fallback label changed")
    if fallback.get("combined_estimate") != "NOT_COMPUTED":
        problems.append("post-main fallback would still compute the combined estimate")

    isolation = declaration.get("isolation") or {}
    if isolation.get("extension_cases_cannot_influence_any_main_grid_analysis") is not True:
        problems.append("extension isolation is not frozen")
    if isolation.get("main_assignment_unchanged") is not True:
        problems.append("main assignment is not declared unchanged")
    if sorted(isolation.get("analysis_case_sets") or []) != sorted(CASE_SETS):
        problems.append("analysis case sets changed")
    return {"valid": not problems, "problems": problems}


__all__ = [
    "AUTHORING_STAGE",
    "BLIND_LABEL",
    "CASE_SETS",
    "DEFAULT_PATH",
    "DETENTION_CAP",
    "KIND",
    "LEVEL",
    "MINIMUM_CLAIMED_CONDITION_CELLS",
    "MINIMUM_LINEAGE_CLUSTERS",
    "POST_MAIN_LABEL",
    "PREREG_VERSION_ID",
    "RESEARCH_REQUIRED",
    "VERSION",
    "assert_extension_disjoint",
    "build_l3_extension_declaration",
    "combined_estimate_allowed",
    "extension_label",
    "load_l3_extension",
    "verify_l3_extension",
    "write_l3_extension",
]
