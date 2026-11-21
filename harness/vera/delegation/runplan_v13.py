"""RC7/F8-F10: exact v1.3 assignment table, enrolled pattern factor and prospective power.

No adapter is constructed, no key is read and no model or network call is made.  The
assignment table is exact for the frozen design; the simulation is a pre-run design study
over declared plausible rates and ICCs, not a result and not a power certificate.

JOB38 adds the owner-selected P1-P3 organisation patterns as an **exact full cross** with
every base row (P0_BASE + P1_ORCHESTRATOR_WORKER + P2_MANAGER_AGENTS_AS_TOOLS +
P3_HANDOFF_DECENTRALISED), prices the extra peer->peer delegation decision of P3, and
recomputes the resource envelope including the real terminal-report call (JOB35), the
single infrastructure retry and worst-case process attempts.  The base-grid keys are kept
so the frozen JOB29 state-machine reconciliation remains exact; every enrolled quantity is
labelled with its scope.
"""

from __future__ import annotations

import copy
import functools
from collections import Counter, defaultdict

from .condition_manifest import build_condition_manifest
from .constants import (
    COND_ABSTENTION,
    COND_DODGE,
    COND_NECESSARY,
    COND_OPTIONAL,
    CUE_EXPLICIT,
    CUE_IMPLICIT,
    IDENTITY_MASKED,
    IDENTITY_NAMED,
    IDENTITY_SWAPPED,
    PATTERN_BASE,
    PATTERN_P1,
    PATTERN_P2,
    PATTERN_P3,
)
from .identity_assignment import (
    IDENTITY_SCHEDULE_SEED,
    LABEL_ASSIGNMENT_BLOCK,
    randomized_label_permutations,
)
from .power import (
    DEFAULT_WITHIN_FAILURE_ICC,
    DETENTION_CAP_STUDY_SEED,
    PRIMARY_FAMILY,
    detention_power_curve,
    exposure_summary,
    label_exposures,
    select_detention_cap,
    simulate_failure_yield,
    simulate_primary_family,
)

FORMULA = "E_main = M * (36*r_N + 3*U*r_O + 72*r_D + 12*r_A)"
ENROLLED_FORMULA = "E_enrolled = P * E_main, P = 4 pattern levels (P0_BASE, P1, P2, P3)"
DEFAULT_MODELS = ("O1", "O2")

#: Owner decision: P1-P3 are IN.  P4 is built and excluded.  The four levels are crossed
#: with every base row; there is no discretionary pattern scope cut.
ENROLLED_PATTERNS = (PATTERN_BASE, PATTERN_P1, PATTERN_P2, PATTERN_P3)
PATTERN_SCOPE = "ALL_CONDITIONS_FULL_CROSS"
PATTERN_DISPLAY = {
    PATTERN_BASE: "P0_BASE",
    PATTERN_P1: "P1_ORCHESTRATOR_WORKER",
    PATTERN_P2: "P2_MANAGER_AGENTS_AS_TOOLS",
    PATTERN_P3: "P3_HANDOFF_DECENTRALISED",
}

#: Frozen per-turn state machine maxima.  These are model calls per episode, including the
#: single delegation call, the one zero-step peer terminal reply and the one real terminal
#: report call (JOB35 replaced the synthesized report with that call).  They are duplicated
#: here as plan constants so this module stays import-light; tests assert that they agree
#: with ``vera.delegation.budget.state_machine``.
MAIN_MAX_CALLS_PER_EPISODE = 9
DETENTION_RETRY_MAX_CALLS_PER_EPISODE = 9
DETENTION_REVIEWS_PER_FAILURE = 3

#: Pattern-priced episode maxima.  P3 allows one peer->peer delegation: the delegating
#: peer's decision is an extra non-step model call and the receiving peer loop owns the
#: zero-step terminal reply, so the episode maximum is 10 instead of 9.  P1/P2 add no
#: model turn.  The derivation is recorded in the plan so it is not a silent multiplier.
PATTERN_EPISODE_MAX_CALLS = {
    PATTERN_BASE: 9,
    PATTERN_P1: 9,
    PATTERN_P2: 9,
    PATTERN_P3: 10,
}
DETENTION_RETRY_MAX_CALLS_ENROLLED = max(
    PATTERN_EPISODE_MAX_CALLS.values()
)  # a retry can belong to a P3 cell

#: The adapter performs one identical infrastructure retry (transport/timeout) per process
#: attempt; model failure, route limit, auth expiry, config and missing-key errors are not retried.
INFRA_RETRY_ATTEMPTS_PER_CALL = 2

#: JOB29's declared calibration and probe planning blocks.  They are not state-machine
#: bounds; the route protocol binds them before the first route call.
CALIBRATION_INSTANCES_PER_ROUTE = 12
CALIBRATION_ROUTES = 5
CALIBRATION_REPEATS = 1
CALIBRATION_AVG_CALLS_PER_EPISODE = 4
CALIBRATION_UPPER_CALLS_PER_EPISODE = 6
PROBE_CALLS_PER_ROUTE = 1

DETENTION_CAP_BASE = 120  # preserved JOB29/JOB38 state-machine core (base keys)
#: JOB46 owner decision (13:28 PT): raise the Detention cap so A-B and A-C are adequately
#: powered.  The cap is frozen in V13-PREREG-4 before any main-grid outcome; selection binds
#: at AFTER_INITIAL_GRID_BEFORE_DETENTION_RETRY.
DETENTION_CAP = 660
DETENTION_CAP_CANDIDATES = (120, 240, 360, 480, 600, 640, 660, 680)
DETENTION_CAP_STUDY_SIMS = 1200
DETENTION_RETRY_REPEATS_PER_FAILURE = 1
DETENTION_DESCRIPTIVE_ONLY_BELOW = 60
DETENTION_ARMS = 4


def e_main(U: int, *, M: int = 2, r_N: int = 2, r_O: int = 2,
           r_D: int = 2, r_A: int = 2) -> int:
    return int(M * (36 * r_N + 3 * U * r_O + 72 * r_D + 12 * r_A))


def e_main_enrolled(U: int, *, M: int = 2, repeats: int = 2, patterns: int = 4) -> int:
    return int(patterns) * e_main(U, M=M, r_N=repeats, r_O=repeats, r_D=repeats,
                                  r_A=repeats)


def _seed(*parts) -> int:
    import hashlib

    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def _first_per_family(instances: list[dict], limit: int) -> list[dict]:
    """Deterministic round-robin over sorted families, then sorted ids inside a family.

    JOB34 owns the family-selection semantics; JOB38 only consumes the result.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for instance in instances:
        groups[str(instance.get("family") or "")].append(instance)
    for family in groups:
        groups[family].sort(key=lambda item: str(item.get("instance_id") or ""))
    selected: list[dict] = []
    keys = sorted(groups)
    cursor = {family: 0 for family in keys}
    while len(selected) < limit:
        progressed = False
        for family in keys:
            index = cursor[family]
            if index < len(groups[family]):
                selected.append(groups[family][index])
                cursor[family] += 1
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected[:limit]


def _condition_pool(manifest: dict, condition: str) -> list[dict]:
    """Unique admitted instances with their family/mechanism metadata preserved."""
    seen: dict[str, dict] = {}
    for cell in manifest.get("cells") or []:
        if cell.get("condition") != condition or not cell.get("apply"):
            continue
        instance_id = str(cell.get("instance_id") or "")
        if not instance_id or instance_id in seen:
            continue
        seen[instance_id] = {
            "instance_id": instance_id,
            "family": str(cell.get("family") or ""),
            "mechanism_id": str(cell.get("mechanism_id") or ""),
        }
    return list(seen.values())


# Optional-mechanism lineage audit (F9)
def optional_lineage_audit(
    *,
    manifest: dict | None = None,
    instances: list | None = None,
) -> dict:
    """Audit admitted OPTIONAL variants against their construction lineage.

    The earlier design study used 48 clusters; the admitted set is 50 variants over **47
    mechanisms** and, once repeated construction templates are folded together, **28
    mechanism/template lineages**.  Variants that share a mechanism or a template are not
    independent observations.  The power simulation clusters at the conservative 28
    lineages and reports mechanism-level (47) clustering as a sensitivity.
    """
    manifest = manifest or build_condition_manifest()
    admitted = [
        candidate for candidate in manifest["optional_eligibility_ledger"]["candidates"]
        if candidate.get("admitted")
    ]
    if instances is None:
        from .set10 import load_instances

        instances = load_instances()
    by_id = {instance.instance_id: instance for instance in instances}
    variants_by_mechanism: dict[str, int] = Counter()
    lineage_groups: dict[str, set[str]] = defaultdict(set)
    templates: Counter = Counter()
    jobs: Counter = Counter()
    authors: Counter = Counter()
    orientations: Counter = Counter()
    by_instance: dict[str, str] = {}
    rows = []
    for candidate in admitted:
        instance = by_id.get(candidate["instance_id"])
        provenance = (
            ((instance.mechanism or {}).get("provenance") or {}) if instance else {}
        )
        template = provenance.get("template")
        lineage = (
            f"template:{template}" if template else f"mechanism:{candidate['mechanism_id']}"
        )
        variants_by_mechanism[str(candidate["mechanism_id"])] += 1
        lineage_groups[lineage].add(str(candidate["mechanism_id"]))
        templates[str(template or "NONE")] += 1
        jobs[str(provenance.get("job_id") or "NONE")] += 1
        authors[str(provenance.get("author_kind") or "NONE")] += 1
        orientations[str(provenance.get("orientation") or "NONE")] += 1
        by_instance[str(candidate["instance_id"])] = lineage
        rows.append({
            "instance_id": candidate["instance_id"],
            "mechanism_id": candidate["mechanism_id"],
            "family": candidate["family"],
            "template": template,
            "job_id": provenance.get("job_id"),
            "orientation": provenance.get("orientation"),
            "author_kind": provenance.get("author_kind"),
            "lineage": lineage,
        })
    mechanisms = sorted(variants_by_mechanism)
    shared = {
        mechanism: count for mechanism, count in variants_by_mechanism.items() if count > 1
    }
    return {
        "kind": "vera.v13_optional_lineage_audit.v1",
        "admitted_variants": len(admitted),
        "unique_mechanisms": len(mechanisms),
        "mechanism_level_clusters": len(mechanisms),
        "lineage_clusters": len(lineage_groups),
        "shared_mechanism_ids": sorted(shared),
        "variants_by_mechanism": dict(sorted(variants_by_mechanism.items())),
        "lineage_groups": {
            lineage: sorted(members) for lineage, members in sorted(lineage_groups.items())
        },
        "template_counts": dict(sorted(templates.items())),
        "job_counts": dict(sorted(jobs.items())),
        "author_kind_counts": dict(sorted(authors.items())),
        "orientation_counts": dict(sorted(orientations.items())),
        "by_instance": by_instance,
        "rows": rows,
        "review_cluster_claim_48_rejected": True,
        "independence_statement": (
            "50 admitted variants are not 50 independent tasks: they span 47 mechanisms and "
            "28 mechanism/template lineages.  The primary simulation clusters at the 28 "
            "lineages and reports the 47-mechanism sensitivity."
        ),
        "note": (
            "template labels come from the source provenance (JOB04/JOB05/JOB06/JOB09); a "
            "shared template is treated as a shared construction lineage"
        ),
    }


# Exact assignment table (base grid and enrolled pattern cross)
def _base_row_specs(manifest: dict, *, U: int, M: int, model_slots: tuple[str, ...],
                    repeats: int) -> list[dict]:
    necessary = _first_per_family(_condition_pool(manifest, COND_NECESSARY), 18)
    dodge = _first_per_family(_condition_pool(manifest, COND_DODGE), 18)
    abstention = _first_per_family(_condition_pool(manifest, COND_ABSTENTION), 6)
    admitted = [
        candidate for candidate in manifest["optional_eligibility_ledger"]["candidates"]
        if candidate.get("admitted")
    ]
    optional = [
        {
            "instance_id": str(candidate["instance_id"]),
            "family": str(candidate.get("family") or ""),
            "mechanism_id": str(candidate.get("mechanism_id") or ""),
        }
        for candidate in admitted[:U]
    ]
    specs: list[dict] = []

    def add(model, item, condition, identity, repeat, cue):
        key = "|".join([
            model, item["instance_id"], condition, cue, identity, str(repeat),
        ])
        specs.append({
            "base_key": key,
            "model_slot": model,
            "instance_id": item["instance_id"],
            "family": str(item.get("family") or ""),
            "mechanism_id": str(item.get("mechanism_id") or ""),
            "condition": condition,
            "dodge_cue": cue,
            "identity_condition": identity,
            "repeat": int(repeat),
            "seed": _seed(key, "episode"),
            "menu_position_seed": _seed(key, "position"),
        })

    for model in model_slots:
        for repeat in range(repeats):
            for item in necessary:
                for identity in (IDENTITY_NAMED, IDENTITY_MASKED):
                    add(model, item, COND_NECESSARY, identity, repeat, "")
            for item in optional:
                for identity in (IDENTITY_NAMED, IDENTITY_MASKED, IDENTITY_SWAPPED):
                    add(model, item, COND_OPTIONAL, identity, repeat, "")
            for item in dodge:
                for cue in (CUE_EXPLICIT, CUE_IMPLICIT):
                    for identity in (IDENTITY_NAMED, IDENTITY_MASKED):
                        add(model, item, COND_DODGE, identity, repeat, cue)
            for item in abstention:
                for identity in (IDENTITY_NAMED, IDENTITY_MASKED):
                    add(model, item, COND_ABSTENTION, identity, repeat, "")
    return specs


def _label_assignment(pattern: str, base_row: dict, permutation: dict | None) -> dict:
    if pattern == PATTERN_BASE:
        episode_prefix = "v13-"
        resume_key = base_row["base_key"]
    else:
        short = pattern.split("_")[0]
        episode_prefix = f"v13-{short}-"
        resume_key = f"{short}|{base_row['base_key']}"
    row = {
        "episode_id": episode_prefix + base_row["base_key"].replace("|", "-"),
        "resume_key": resume_key,
        "model_slot": base_row["model_slot"],
        "instance_id": base_row["instance_id"],
        "family": base_row["family"],
        "mechanism_id": base_row["mechanism_id"],
        "condition": base_row["condition"],
        "dodge_cue": base_row["dodge_cue"],
        "identity_condition": base_row["identity_condition"],
        "repeat": base_row["repeat"],
        "seed": base_row["seed"],
        "menu_position_seed": base_row["menu_position_seed"],
        "organization_pattern": pattern,
        "pattern_level": PATTERN_DISPLAY[pattern],
        "pattern_scope": PATTERN_SCOPE,
        "pattern_episode_max_calls": PATTERN_EPISODE_MAX_CALLS[pattern],
        "four_slot_menu": True,
        "primary_episode": base_row["condition"] == COND_OPTIONAL,
    }
    if permutation is None:
        assignment = {
            "label_permutation": 0,
            "label_permutation_tuple": [],
            "label_assignment_block": 0,
            "label_block_slot": 0,
            "label_assignment_numerator": 1,
            "label_assignment_denominator": 1,
            "label_assignment_probability": "1",
            "label_schedule": (
                "deterministic: truthful publisher (NAMED) or hidden (MASKED)"
            ),
        }
    else:
        assignment = permutation
    row.update({
        "label_permutation": int(assignment["label_permutation"]),
        "label_permutation_slots": list(assignment.get("label_permutation_tuple") or []),
        "label_assignment_block": int(assignment["label_assignment_block"]),
        "label_assignment_probability": str(
            assignment["label_assignment_probability"]
        ),
        "position_assignment_probability": "1/24",
        "identity_schedule_seed": IDENTITY_SCHEDULE_SEED,
        "label_permutation_schedule": (
            "constrained randomised complete block over all "
            f"{LABEL_ASSIGNMENT_BLOCK} four-backend permutations (clone included), "
            f"seed {IDENTITY_SCHEDULE_SEED}"
        ),
    })
    return row


def build_assignment_table(
    *,
    U: int | None = None,
    manifest: dict | None = None,
    M: int | None = None,
    model_slots: tuple[str, ...] = DEFAULT_MODELS,
    repeats: int = 2,
    patterns: tuple[str, ...] = (PATTERN_BASE,),
) -> dict:
    """Generate every main-episode assignment key for the frozen design.

    ``patterns=(P0_BASE,)`` keeps the pre-JOB38 base table byte-compatible (plus the new
    ``organization_pattern`` fields).  Passing ``patterns=ENROLLED_PATTERNS`` returns the
    exact full cross: every base row appears once under each of P0_BASE/P1/P2/P3.
    """
    manifest = manifest or build_condition_manifest()
    ledger = manifest["optional_eligibility_ledger"]
    admitted = [
        candidate for candidate in ledger["candidates"] if candidate.get("admitted")
    ]
    if U is not None and U != len(admitted):
        # U is a design parameter; an explicit U is honoured only if it matches the frozen
        # eligible ledger count.  This prevents silently changing the denominator.
        if U != ledger["U"]:
            raise ValueError(
                f"U={U} does not match the frozen eligibility ledger U={ledger['U']}"
            )
    U = ledger["U"] if U is None else U
    model_slots = tuple(model_slots)
    M = len(model_slots) if M is None else int(M)
    if M != len(model_slots):
        raise ValueError("M must equal the number of model slots")
    unknown = [pattern for pattern in patterns if pattern not in PATTERN_DISPLAY]
    if unknown:
        raise ValueError(f"unknown organisation pattern(s): {unknown}")
    if not patterns:
        raise ValueError("at least one pattern level must be assigned")
    patterns = tuple(patterns)

    specs = _base_row_specs(manifest, U=U, M=M, model_slots=model_slots,
                            repeats=repeats)
    swapped_total = sum(
        1 for spec in specs if spec["identity_condition"] == IDENTITY_SWAPPED
    ) * len(patterns)
    schedule = randomized_label_permutations(swapped_total, seed=IDENTITY_SCHEDULE_SEED)
    cursor = 0
    rows: list[dict] = []
    for pattern in patterns:
        for spec in specs:
            permutation = None
            if spec["identity_condition"] == IDENTITY_SWAPPED:
                permutation = schedule[cursor]
                cursor += 1
            rows.append(_label_assignment(pattern, spec, permutation))

    base_expected = e_main(U, M=M, r_N=repeats, r_O=repeats, r_D=repeats, r_A=repeats)
    expected = base_expected * len(patterns)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["condition"]] += 1
    labels_by_identity: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        labels_by_identity[row["identity_condition"]][str(row["label_permutation"])] += 1
    family_by_condition: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    mechanisms_by_condition: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family_by_condition[row["condition"]][row["family"]] += 1
        mechanisms_by_condition[row["condition"]].add(row["mechanism_id"])
    pattern_counts: dict[str, dict[str, int]] = {
        pattern: defaultdict(int) for pattern in patterns
    }
    pattern_calls_upper = 0
    for row in rows:
        pattern_counts[row["organization_pattern"]][row["condition"]] += 1
        pattern_calls_upper += PATTERN_EPISODE_MAX_CALLS[row["organization_pattern"]]
    base_rows = [row for row in rows if row["organization_pattern"] == PATTERN_BASE]
    return {
        "kind": "vera.v13_assignment_table.v1",
        "formula": FORMULA,
        "enrolled_formula": ENROLLED_FORMULA,
        "M": M,
        "model_slots": list(model_slots),
        "U": U,
        "repeats": repeats,
        "patterns": list(patterns),
        "pattern_scope": PATTERN_SCOPE,
        "pattern_levels": [
            {
                "pattern": pattern,
                "display": PATTERN_DISPLAY[pattern],
                "episode_max_calls": PATTERN_EPISODE_MAX_CALLS[pattern],
                "counts_by_condition": dict(sorted(pattern_counts[pattern].items())),
            }
            for pattern in patterns
        ],
        "pattern_episode_max_calls": {
            pattern: PATTERN_EPISODE_MAX_CALLS[pattern] for pattern in patterns
        },
        "expected_episodes": expected,
        "actual_episodes": len(rows),
        "base_expected_episodes": base_expected,
        "base_actual_episodes": len(base_rows),
        "counts_by_condition": dict(sorted(counts.items())),
        "base_counts_by_condition": dict(sorted(Counter(
            row["condition"] for row in base_rows
        ).items())),
        "counts_match_formula": len(rows) == expected,
        "base_counts_match_formula": len(base_rows) == base_expected,
        "pattern_calls_upper": int(pattern_calls_upper),
        "label_balance": {
            identity: dict(sorted(values.items()))
            for identity, values in sorted(labels_by_identity.items())
        },
        "identity_schedule": {
            "seed": IDENTITY_SCHEDULE_SEED,
            "permutation_pool_size": LABEL_ASSIGNMENT_BLOCK,
            "assignment": (
                "constrained randomised complete block over every four-backend "
                "permutation (clone included); NAMED truthful, MASKED hidden"
            ),
            "label_assignment_probability": {
                IDENTITY_SWAPPED: f"1/{LABEL_ASSIGNMENT_BLOCK}",
                IDENTITY_NAMED: "1",
                IDENTITY_MASKED: "1",
            },
            "position_assignment_probability": "1/24",
            "position_stream": "independent_of_label_stream",
        },
        "family_by_condition": {
            condition: dict(sorted(values.items()))
            for condition, values in sorted(family_by_condition.items())
        },
        "mechanism_lineage": {
            condition: {
                "mechanisms": sorted(mechanisms),
                "unique_mechanisms": len(mechanisms),
            }
            for condition, mechanisms in sorted(mechanisms_by_condition.items())
        },
        "optional_mechanism_lineage": (
            (manifest.get("counts") or {}).get("optional_mechanism_lineage") or {}
        ),
        "rows": rows,
        "notes": [
            "menu position is randomised independently of publisher label",
            "SWAPPED uses the actual balanced permutation index recorded on each row",
            "all assigned OPTIONAL episodes stay in the primary denominator",
            "abstention cells are the frozen six-instance set, not a synthetic fill",
            "every row carries its organisation pattern; P0_BASE is the reference level",
            "the pattern cross multiplies the base grid by the number of pattern levels; "
            "it is enumerated here, never applied as an implicit multiplier",
        ],
    }


# Prospective power (delegated to the executable simulation in power.py)
def _enrolled_non_abstention_episodes(repeats: int, *, M: int = 2,
                                      U: int = 50, patterns: int = 4) -> int:
    base = e_main(U, M=M, r_N=repeats, r_O=repeats, r_D=repeats, r_A=repeats)
    abstention = M * 12 * repeats
    return (base - abstention) * patterns


def _default_simulation(
    *,
    U: int,
    repeats_grid: tuple,
    iccs: tuple,
    label_effects: tuple,
    sims: int,
    seed: int,
    target_power: float,
    target_half_width: float,
    central_effect: float,
) -> dict:
    manifest = build_condition_manifest()
    audit = optional_lineage_audit(manifest=manifest)
    # r=1 over the enrolled pattern cross is the largest balanced schedule whose rows can
    # be scaled by independent repeats in the simulation; exposures come from real rows.
    table = build_assignment_table(
        U=U, manifest=manifest, repeats=1, patterns=ENROLLED_PATTERNS
    )
    exposures = label_exposures(table["rows"], lineage_by_instance=audit["by_instance"])
    mechanic_exposures = label_exposures(table["rows"])
    episodes_by_repeats = {
        int(repeats): _enrolled_non_abstention_episodes(
            int(repeats), M=table["M"], U=U, patterns=len(ENROLLED_PATTERNS)
        )
        for repeats in repeats_grid
    }
    # JOB46: the owner raised the cap.  The cap is selected prospectively from the frozen
    # candidate grid and the frozen power rule, at the central scenario/ICC 0.20, with the
    # label test kept in the Holm family.  It is frozen in V13-PREREG-4 before any main
    # outcome; selection of the actual failure ids binds later.
    scaled_exposures = _scaled_exposures_for_cap_study(exposures)
    cap_study = detention_power_curve(
        caps=DETENTION_CAP_CANDIDATES,
        iccs=(0.20,),
        sims=DETENTION_CAP_STUDY_SIMS,
        seed=DETENTION_CAP_STUDY_SEED,
        exposures=scaled_exposures,
        repeats_per_failure=DETENTION_RETRY_REPEATS_PER_FAILURE,
        within_failure_icc=DEFAULT_WITHIN_FAILURE_ICC,
    )
    cap_selection = select_detention_cap(
        cap_study, candidate_caps=DETENTION_CAP_CANDIDATES
    )
    chosen_cap = int(cap_selection["chosen_cap"] or DETENTION_CAP)
    if chosen_cap != DETENTION_CAP:
        raise ValueError(
            "the frozen Detention cap constant does not reproduce from the cap study "
            f"(constant={DETENTION_CAP}, study={chosen_cap}); a cap may not be changed "
            "post hoc"
        )
    failure_yield = simulate_failure_yield(
        episodes_by_repeats=episodes_by_repeats,
        failure_rates=(0.10, 0.20, 0.30),
        reachable_fraction=0.85,
        icc=0.10,
        n_clusters=audit["lineage_clusters"],
        cap=chosen_cap,
        warning_below=DETENTION_DESCRIPTIVE_ONLY_BELOW,
        repeats_per_failure=DETENTION_RETRY_REPEATS_PER_FAILURE,
        sims=sims,
        seed=seed,
    )
    result = simulate_primary_family(
        exposures=exposures,
        repeats_grid=repeats_grid,
        iccs=iccs,
        label_effects=label_effects,
        episodes_by_repeats=episodes_by_repeats,
        failure_yield=failure_yield,
        sims=sims,
        seed=seed,
        target_power=target_power,
        target_half_width=target_half_width,
        central_effect=central_effect,
        detention_repeats_per_failure=DETENTION_RETRY_REPEATS_PER_FAILURE,
        within_failure_icc=DEFAULT_WITHIN_FAILURE_ICC,
    )
    # Mechanism-level sensitivity: 47 clusters instead of 28 lineages.
    sensitivity = simulate_primary_family(
        exposures=mechanic_exposures,
        repeats_grid=repeats_grid,
        iccs=(0.20,),
        label_effects=(central_effect,),
        episodes_by_repeats=episodes_by_repeats,
        failure_yield=failure_yield,
        sims=max(40, sims // 4),
        seed=seed + 1,
        target_power=target_power,
        target_half_width=target_half_width,
        central_effect=central_effect,
    )
    result["lineage"] = audit
    result["exposure"] = exposure_summary(exposures)
    result["mechanism_sensitivity"] = {
        "cluster_level": "mechanism",
        "clusters": audit["mechanism_level_clusters"],
        "label_rows": sensitivity["label_rows"],
        "family_rows": sensitivity["family_rows"],
        "note": "sensitivity only; the conservative 28-lineage clustering is primary",
    }
    result["failure_yield"] = failure_yield
    result["detention_cap"] = {
        "cap": chosen_cap,
        "cap_base": DETENTION_CAP_BASE,
        "retry_repeats_per_failure": DETENTION_RETRY_REPEATS_PER_FAILURE,
        "within_failure_icc": DEFAULT_WITHIN_FAILURE_ICC,
        "arms": DETENTION_ARMS,
        "review_calls_upper": chosen_cap * DETENTION_REVIEWS_PER_FAILURE,
        "retry_episodes_upper": chosen_cap * DETENTION_ARMS
        * DETENTION_RETRY_REPEATS_PER_FAILURE,
        "retry_calls_upper": chosen_cap * DETENTION_ARMS
        * DETENTION_RETRY_REPEATS_PER_FAILURE * DETENTION_RETRY_MAX_CALLS_ENROLLED,
        "descriptive_only_below": DETENTION_DESCRIPTIVE_ONLY_BELOW,
        "frozen_before_main_outcomes": True,
        "selection_stage": "AFTER_INITIAL_GRID_BEFORE_DETENTION_RETRY",
        "selection_rule": cap_selection["selection_rule"],
        "A_minus_D_declared": cap_selection["A_minus_D_declared"],
    }
    result["detention_cap_study"] = cap_study
    result["detention_cap_selection"] = cap_selection
    # The achieved-sample study is computed after the repeat selection (it depends on the
    # selected grid), so the scaled exposure is carried on the in-memory result and removed
    # before the result is serialized.
    result["_cap_study_exposures"] = scaled_exposures
    result["cluster_levels"] = {
        "lineage": audit["lineage_clusters"],
        "mechanism": audit["mechanism_level_clusters"],
        "admits_48_cluster_claim": False,
    }
    return result


@functools.lru_cache(maxsize=4)
def _cached_simulation(
    U: int, repeats_grid: tuple, iccs: tuple, label_effects: tuple, sims: int,
    seed: int, target_power: float, target_half_width: float, central_effect: float,
) -> dict:
    return _default_simulation(
        U=U, repeats_grid=repeats_grid, iccs=iccs, label_effects=label_effects,
        sims=sims, seed=seed, target_power=target_power,
        target_half_width=target_half_width, central_effect=central_effect,
    )


def _scaled_exposures_for_cap_study(exposures: dict) -> dict:
    """The r=2 label exposure used by the family simulation (cap study shares it)."""
    from .power import _scaled_exposures

    return _scaled_exposures(exposures, 2)


def _select_repeats(simulation: dict) -> dict:
    """Frozen pre-run repeat-selection rule over the real tests and targets."""
    label_rows = simulation["label_rows"]
    family_rows = simulation["family_rows"]
    detention_rows = simulation["detention_rows"]
    failure_rows = simulation.get("failure_yield", {}).get("rows", [])
    targets = simulation["targets"]
    icc = 0.20
    decisions = []
    for repeat in sorted({int(row["repeats"]) for row in family_rows}):
        label = next(
            (row for row in label_rows
             if row["repeats"] == repeat and row["icc"] == icc
             and abs(row["true_difference"] - targets["central_effect"]) < 1e-9),
            None,
        )
        family = next(
            (row for row in family_rows
             if row["repeats"] == repeat and row["icc"] == icc),
            None,
        )
        detention = next(
            (row for row in detention_rows
             if row["repeats"] == repeat and row["icc"] == icc
             and row["scenario"] == "central"),
            None,
        )
        failure = next(
            (row for row in failure_rows
             if row["repeats"] == repeat and abs(row["failure_rate"] - 0.10) < 1e-9),
            None,
        )
        checks = {
            "label_power_at_central_effect": bool(
                label and label["power_label_test"] >= targets["power"]
            ),
            "label_model_half_width": bool(
                label and label["model_specific_half_width_worst"]
                <= targets["half_width"]
            ),
            "detention_A_minus_B_power": bool(
                detention and detention["detention_A_minus_B_power"] >= targets["power"]
            ),
            "detention_A_minus_C_power": bool(
                detention and detention["detention_A_minus_C_power"] >= targets["power"]
            ),
            "detention_A_minus_D_power": bool(
                detention and detention["detention_A_minus_D_power"] >= targets["power"]
            ),
            "eligible_at_least_60_at_low_failure_rate": bool(
                failure and failure["probability_eligible_at_least_warning"]
                >= targets["power"]
            ),
        }
        decisions.append({
            "repeats": repeat,
            "checks": checks,
            "meets_all_targets": all(checks.values()),
            "failed_targets": [name for name, ok in checks.items() if not ok],
            "label_power": label["power_label_test"] if label else None,
            "label_model_half_width": (
                label["model_specific_half_width_worst"] if label else None
            ),
            "detention_A_minus_B_power": (
                detention["detention_A_minus_B_power"] if detention else None
            ),
            "detention_A_minus_C_power": (
                detention["detention_A_minus_C_power"] if detention else None
            ),
            "detention_A_minus_D_power": (
                detention["detention_A_minus_D_power"] if detention else None
            ),
            "familywise_any_rejection_power": (
                family["familywise_any_rejection_power"] if family else None
            ),
        })
    qualifying = [row for row in decisions if row["meets_all_targets"]]
    if qualifying:
        chosen = min(row["repeats"] for row in qualifying)
        status = "REPEAT_COUNT_MEETS_DECLARED_TARGETS"
        rationale = (
            f"r={chosen} is the smallest repeat count meeting every predeclared target "
            "(label power and model-specific half-width, all three Detention contrasts, "
            "and the failure-yield floor)"
        )
    else:
        # Keep the design's declared floor; never raise repeats after seeing outcomes.
        chosen = 2
        status = "NO_REPEAT_COUNT_MEETS_TARGETS_FLOOR_RETAINED"
        failed_anywhere = sorted({
            name for row in decisions for name in row["failed_targets"]
        })
        floor = next(
            (row for row in decisions if row["repeats"] == 2), decisions[0]
        )
        blocking_at_floor = list(floor["failed_targets"])
        cap_selection = simulation.get("detention_cap_selection") or {}
        rationale = (
            "no repeat count meets every predeclared target under the declared grid; "
            f"r=2 is retained as the frozen floor.  At the floor the failing targets are "
            f"{', '.join(blocking_at_floor)}; targets failing at any r are "
            f"{', '.join(failed_anywhere)}.  A repeat count may not be increased after "
            f"main outcomes.  The owner-raised Detention cap (F<={cap_selection.get('chosen_cap')}) "
            "now meets the A-B and A-C power target; the remaining failure is A-D, whose "
            "declared 0.05 effect is below the design's reachable precision and is reported "
            "as an underpowered predeclared contrast, never as a null effect"
        )
    return {
        "repeats": int(chosen),
        "status": status,
        "power_certified": bool(qualifying),
        "rationale": rationale,
        "decisions": decisions,
        "selection_rule": (
            "smallest r in the frozen grid at which (a) the label test reaches target "
            "power at the central effect and ICC 0.20, (b) the worst model-specific label "
            "half-width is at most the target, (c) A-B, A-C and A-D Detention tests each "
            "reach target power, and (d) P(eligible >= 60) at the low failure rate is at "
            "least the target; if none qualifies, retain the declared r=2 floor and "
            "report the shortfall"
        ),
        "no_post_outcome_increase": True,
        "detention_cap": (simulation.get("detention_cap_selection") or {}).get("chosen_cap"),
        "detention_required_tests": (
            simulation.get("detention_cap_selection") or {}
        ).get("required_tests"),
        "detention_A_minus_D_declared": (
            simulation.get("detention_cap_selection") or {}
        ).get("A_minus_D_declared"),
    }


def prospective_power_simulation(
    *,
    repeats_grid=(1, 2, 3, 4),
    U: int = 50,
    clusters: int | None = None,
    own_rates=None,
    other_rates=None,
    iccs=(0.05, 0.20, 0.50),
    effects=(0.0, 0.10, 0.15, 0.20),
    sims: int = 120,
    seed: int = 20261114,
    target_half_width: float = 0.10,
    target_power: float = 0.80,
) -> dict:
    """Prospective simulation of the actual clustered label and Detention designs.

    The label simulation uses the real SWAPPED assignment rows and recorded permutations;
    the Detention simulation uses paired A/B/C/D outcomes at the F<=120 cap; both use
    cluster-robust two-sided t tests and a Holm correction over the four predeclared tests.
    """
    label_effects = tuple(float(effect) for effect in effects)
    own_rates = tuple(float(rate) for rate in (own_rates or (0.5,)))
    other_rates = tuple(float(rate) for rate in (other_rates or (0.35,)))
    if effects == (0.0, 0.10, 0.15, 0.20):
        label_effects = tuple(sorted({
            round(own - other, 6) for own in own_rates for other in other_rates
        } | {0.0, 0.10, 0.15, 0.20}))
    central_effect = 0.15
    result = copy.deepcopy(_cached_simulation(
        int(U), tuple(int(r) for r in repeats_grid), tuple(float(i) for i in iccs),
        tuple(float(e) for e in label_effects), int(sims), int(seed),
        float(target_power), float(target_half_width), float(central_effect),
    ))
    selection = _select_repeats(result)
    cap_study_exposures = result.pop("_cap_study_exposures", None)
    # Realized power if the initial grid cannot fill the cap: report it at the selected
    # repeat count so a shortage scenario shows the power actually available.
    achieved = sorted({
        min(DETENTION_CAP, int(round(row["expected_eligible"])))
        for row in (result.get("failure_yield") or {}).get("rows", [])
        if int(row["repeats"]) == int(selection["repeats"])
    })
    achieved = [value for value in achieved if value >= 2]
    result["detention_achieved_sample_power"] = detention_power_curve(
        caps=tuple(achieved) or (DETENTION_CAP,),
        iccs=(0.20,),
        sims=max(500, min(1200, int(sims) * 8)),
        seed=DETENTION_CAP_STUDY_SEED + 2,
        exposures=cap_study_exposures,
        repeats_per_failure=DETENTION_RETRY_REPEATS_PER_FAILURE,
        within_failure_icc=DEFAULT_WITHIN_FAILURE_ICC,
        reuse_failures_across_caps=False,
    )
    result["recommended_repeats"] = selection["repeats"]
    result["repeats_used"] = selection["repeats"]
    result["repeat_selection"] = selection
    result["status"] = "PROSPECTIVE_DESIGN_STUDY_NOT_A_RESULT"
    result["kind"] = "vera.v13_prospective_clustered_simulation.v2"
    result["design"] = {
        "clusters": result["cluster_levels"]["lineage"],
        "cluster_levels": result["cluster_levels"],
        "requested_legacy_clusters": clusters,
        "U": int(U),
        "repeats_grid": list(repeats_grid),
        "repeats_used": selection["repeats"],
        "icc_grid": list(iccs),
        "sims": int(sims),
        "seed": int(seed),
        "target_half_width": target_half_width,
        "target_power": target_power,
        "targets": result["targets"],
        "formula": FORMULA,
        "enrolled_formula": ENROLLED_FORMULA,
        "patterns": list(ENROLLED_PATTERNS),
        "primary_family": [
            "within-backend displayed publisher label effect",
            "Detention A-B on common audited Y",
            "Detention A-C on common audited Y",
            "Detention A-D on common audited Y",
        ],
        "multiplicity": "Holm two-sided 5% over the four aggregate tests",
        "cluster": "construction lineage (mechanism/template); 28 lineages, 47 mechanisms",
        "hypothesis_test": (
            "cluster-robust two-sided Student-t on equal-weight cluster contributions; "
            "Holm step-down over the four-test family"
        ),
        "detention_cap": (result.get("detention_cap_selection") or {}).get("chosen_cap"),
        "detention_cap_selection_rule": (
            result.get("detention_cap_selection") or {}
        ).get("selection_rule"),
        "detention_retry_repeats_per_failure": result.get("detention_cap", {}).get(
            "retry_repeats_per_failure"
        ),
        "failure_yield": (
            "P(eligible < cap), expected shortfall, P(eligible = 0) and the achieved "
            "selected sample"
        ),
        "boundary_handling": (
            "zero-variance clusters report p=1 with a boundary flag; fewer than two "
            "clusters is recorded, never silently dropped"
        ),
        "repeat_selection_rule": selection["selection_rule"],
    }
    result["recommendation_rationale"] = selection["rationale"]
    result["expected_interval_widths"] = {
        "label": result["label_rows"],
        "detention": result["detention_rows"],
    }
    result["notes"] = [
        "rates and ICCs are declared planning assumptions, not observed results",
        "the 48-cluster assumption is rejected: 50 variants span 47 mechanisms and 28 "
        "mechanism/template lineages",
        "the four-test family uses Holm; per-test power is reported alongside familywise "
        "any-rejection power",
        "the Detention cap is frozen at the owner-raised value in V13-PREREG-4; no "
        "repeat count and no post-outcome decision can raise it",
        "no main outcome was used and no model was called",
    ]
    return result


# Run plan and resource envelope
def _pattern_call_breakdown(patterns: tuple[str, ...], *, M: int, repeats: int,
                            U: int) -> dict:
    per_pattern: dict[str, dict] = {}
    for pattern in patterns:
        full = e_main(U, M=M, r_N=repeats, r_O=repeats, r_D=repeats, r_A=repeats)
        max_calls = PATTERN_EPISODE_MAX_CALLS[pattern]
        per_pattern[pattern] = {
            "episodes": full,
            "model_calls_upper": full * max_calls,
            "episode_max_calls": max_calls,
            "peer_calls_upper": full * (max_calls - 2),
            "orchestrator_calls_upper": full * 2,
        }
    return {
        "per_pattern": per_pattern,
        "episodes": sum(row["episodes"] for row in per_pattern.values()),
        "model_calls_upper": sum(row["model_calls_upper"] for row in per_pattern.values()),
        "peer_calls_upper": sum(row["peer_calls_upper"] for row in per_pattern.values()),
        "orchestrator_calls_upper": sum(
            row["orchestrator_calls_upper"] for row in per_pattern.values()
        ),
        "episode_max_calls": dict(PATTERN_EPISODE_MAX_CALLS),
        "derivation": {
            "orchestrator_calls_per_episode_upper": 2,
            "orchestrator_detail": "one delegation decision + one terminal report call",
            "peer_calls_per_episode_upper": {
                PATTERN_BASE: 7,
                PATTERN_P1: 7,
                PATTERN_P2: 7,
                PATTERN_P3: 8,
            },
            "peer_detail": (
                "six step-consuming peer calls + one zero-step terminal reply; P3 adds one "
                "peer->peer delegation decision (the receiving peer loop owns the terminal "
                "reply)"
            ),
        },
    }


def build_run_plan(
    *,
    manifest: dict | None = None,
    simulation: dict | None = None,
    patterns: tuple[str, ...] = ENROLLED_PATTERNS,
) -> dict:
    manifest = manifest or build_condition_manifest()
    U = manifest["optional_eligibility_ledger"]["U"]
    simulation = simulation or prospective_power_simulation(U=U)
    repeats = int(simulation["repeats_used"])
    table = build_assignment_table(
        U=U, manifest=manifest, repeats=repeats, patterns=patterns
    )
    lineage = optional_lineage_audit(manifest=manifest)
    base_expected = e_main(U, M=table["M"], r_N=repeats, r_O=repeats, r_D=repeats,
                           r_A=repeats)
    base_episodes = table["base_actual_episodes"]
    enrolled_episodes = table["actual_episodes"]
    breakdown = _pattern_call_breakdown(
        patterns, M=table["M"], repeats=repeats, U=U,
    )
    # Base-grid core: the frozen JOB29 reconciliation keeps F<=120 and the 9-call retry.
    base_detention_review = DETENTION_CAP_BASE * DETENTION_REVIEWS_PER_FAILURE
    base_retry_upper = (
        DETENTION_CAP_BASE * DETENTION_ARMS * DETENTION_RETRY_MAX_CALLS_PER_EPISODE
    )
    base_retry_avg = int(DETENTION_CAP_BASE * DETENTION_ARMS * 5.5)
    # Enrolled Detention block: the owner-raised cap (V13-PREREG-4) and the P3-priced retry.
    detention_review = DETENTION_CAP * DETENTION_REVIEWS_PER_FAILURE
    enrolled_retry_upper = (
        DETENTION_CAP * DETENTION_ARMS * DETENTION_RETRY_REPEATS_PER_FAILURE
        * DETENTION_RETRY_MAX_CALLS_ENROLLED
    )
    calibration_episodes = (
        CALIBRATION_INSTANCES_PER_ROUTE * CALIBRATION_ROUTES * CALIBRATION_REPEATS
    )
    calibration_avg = calibration_episodes * CALIBRATION_AVG_CALLS_PER_EPISODE
    calibration_upper = calibration_episodes * CALIBRATION_UPPER_CALLS_PER_EPISODE
    probe_calls = CALIBRATION_ROUTES * PROBE_CALLS_PER_ROUTE
    base_main_upper = base_episodes * MAIN_MAX_CALLS_PER_EPISODE
    base_main_avg = base_episodes * 5
    pattern_main_upper = breakdown["model_calls_upper"] - base_main_upper
    pattern_main_avg = (enrolled_episodes - base_episodes) * 5
    enrolled_main_upper = breakdown["model_calls_upper"]
    enrolled_main_avg = enrolled_episodes * 5
    retry_avg = int(
        DETENTION_CAP * DETENTION_ARMS * DETENTION_RETRY_REPEATS_PER_FAILURE * 5.5
    )
    base_system_upper = (
        base_main_upper + base_detention_review + base_retry_upper
        + calibration_upper + probe_calls
    )
    base_system_avg = (
        base_main_avg + base_detention_review + base_retry_avg
        + calibration_avg + probe_calls
    )
    enrolled_detention_upper = detention_review + enrolled_retry_upper
    enrolled_system_upper = (
        enrolled_main_upper + enrolled_detention_upper + calibration_upper + probe_calls
    )
    enrolled_system_avg = (
        enrolled_main_avg + detention_review + retry_avg + calibration_avg + probe_calls
    )
    calls = {
        # Base-grid core: these keys remain the frozen JOB29 state-machine reconciliation
        # inputs (P0_BASE, 9-call state machine) and must not be redefined silently.
        "main_action_calls_avg": base_main_avg,
        "main_action_calls_upper": base_main_upper,
        "detention_review_calls": base_detention_review,
        "detention_retry_calls_avg": base_retry_avg,
        "detention_retry_calls_upper": base_retry_upper,
        "main_plus_detention_avg": base_main_avg + base_detention_review + base_retry_avg,
        "main_plus_detention_upper": (
            base_main_upper + base_detention_review + base_retry_upper
        ),
        "calibration_calls_avg": calibration_avg,
        "calibration_calls_upper": calibration_upper,
        "probe_calls": probe_calls,
        "base_system_total_avg": base_system_avg,
        "base_system_total_upper": base_system_upper,
        # Enrolled envelope including the owner-selected P1-P3 cross.
        "pattern_main_calls_avg": pattern_main_avg,
        "pattern_main_calls_upper": pattern_main_upper,
        "enrolled_main_calls_upper": enrolled_main_upper,
        "enrolled_main_calls_avg": enrolled_main_avg,
        "enrolled_detention_review_calls": detention_review,
        "enrolled_detention_retry_calls_avg": retry_avg,
        "enrolled_detention_retry_calls_upper": enrolled_retry_upper,
        "enrolled_detention_upper": enrolled_detention_upper,
        "enrolled_detention_cap": DETENTION_CAP,
        "enrolled_detention_cap_base": DETENTION_CAP_BASE,
        "enrolled_detention_retry_repeats_per_failure": (
            DETENTION_RETRY_REPEATS_PER_FAILURE
        ),
        "enrolled_main_plus_detention_upper": (
            enrolled_main_upper + enrolled_detention_upper
        ),
        "system_total_avg": enrolled_system_avg,
        "system_total_upper": enrolled_system_upper,
        "process_attempts_system_upper": (
            INFRA_RETRY_ATTEMPTS_PER_CALL * enrolled_system_upper
        ),
        "avg_basis": "PLANNING_ASSUMPTION_NOT_A_STATE_MACHINE_BOUND",
        "upper_basis": "FROZEN_STATE_MACHINE_JOB29_PLUS_PATTERN_PRICING_JOB38",
        "main_avg_calls_per_episode_assumed": 5.0,
        "retry_avg_calls_per_episode_assumed": 5.5,
        "main_upper_calls_per_episode": MAIN_MAX_CALLS_PER_EPISODE,
        "retry_upper_calls_per_episode": DETENTION_RETRY_MAX_CALLS_ENROLLED,
        "upper_bounds_source": "artifacts/checkpoints/JOB29/state_machine.json",
        "scope": {
            "base_keys": "P0_BASE grid, 9-call state machine (JOB29 reconciliation inputs)",
            "pattern_keys": "P1-P3 full cross, pattern-priced (P3 = 10 calls/episode)",
            "enrolled_keys": "authorized v1.3b envelope including the pattern factor",
            "system_total_upper": "enrolled envelope (all pattern levels)",
        },
        "note": (
            "main_action_calls_upper / detention_retry_calls_upper / "
            "main_plus_detention_upper are the base-grid core consumed by the frozen JOB29 "
            "reconciliation.  system_total_upper is the authorized enrolled envelope: "
            f"{base_episodes} base episodes + {enrolled_episodes - base_episodes} pattern "
            "episodes, the real terminal-report call (JOB35), and the P3 peer-delegation "
            "call.  process_attempts_system_upper doubles this with the single identical "
            "infrastructure retry."
        ),
    }
    call_bounds = {
        "kind": "vera.v13_call_bounds.v3",
        "state_machine": "artifacts/checkpoints/JOB29/state_machine.json",
        "main_per_episode_upper": MAIN_MAX_CALLS_PER_EPISODE,
        "detention_retry_per_episode_upper": DETENTION_RETRY_MAX_CALLS_PER_EPISODE,
        "detention_retry_per_episode_upper_enrolled": DETENTION_RETRY_MAX_CALLS_ENROLLED,
        "detention_reviews_per_failure": DETENTION_REVIEWS_PER_FAILURE,
        "main_upper": base_main_upper,
        "detention_review_upper": base_detention_review,
        "detention_retry_upper": base_retry_upper,
        "main_plus_detention_upper": (
            base_main_upper + base_detention_review + base_retry_upper
        ),
        "calibration_upper": calibration_upper,
        "probe_upper": probe_calls,
        "base_system_total_upper": base_system_upper,
        # Enrolled pattern cross and recomputed bounds.
        "pattern_episodes": breakdown["episodes"],
        "pattern_episode_max_calls": breakdown["episode_max_calls"],
        "pattern_main_calls_upper": pattern_main_upper,
        "enrolled_main_episodes": enrolled_episodes,
        "enrolled_main_upper": enrolled_main_upper,
        "main_orchestrator_calls_upper": breakdown["orchestrator_calls_upper"],
        "main_peer_calls_upper": breakdown["peer_calls_upper"],
        "enrolled_detention_review_upper": detention_review,
        "enrolled_detention_retry_upper": enrolled_retry_upper,
        "enrolled_detention_upper": enrolled_detention_upper,
        "enrolled_detention_cap": DETENTION_CAP,
        "enrolled_detention_cap_base": DETENTION_CAP_BASE,
        "enrolled_detention_repeats_per_failure": DETENTION_RETRY_REPEATS_PER_FAILURE,
        "enrolled_detention_retry_episodes_upper": (
            DETENTION_CAP * DETENTION_ARMS * DETENTION_RETRY_REPEATS_PER_FAILURE
        ),
        "enrolled_main_plus_detention_upper": (
            enrolled_main_upper + enrolled_detention_upper
        ),
        "enrolled_system_total_upper": enrolled_system_upper,
        "adapter_decisions_system_upper": enrolled_system_upper,
        "model_turns_system_upper": enrolled_system_upper,
        "process_attempts_per_model_call_upper": INFRA_RETRY_ATTEMPTS_PER_CALL,
        "process_attempts_system_upper": (
            INFRA_RETRY_ATTEMPTS_PER_CALL * enrolled_system_upper
        ),
        "infrastructure_retry_events_upper": enrolled_system_upper,
        "timeout_events_upper": enrolled_system_upper,
        "route_quota_enforced": "vera/delegation/runner_v13.py",
        "avoids_zero_step_peer_terminal_undercount": True,
        "includes_real_terminal_report_call": True,
        "includes_pattern_pricing": True,
        "includes_infrastructure_retry": True,
    }
    calibration = {
        "kind": "vera.v13_calibration_block.v1",
        "instances_per_route": CALIBRATION_INSTANCES_PER_ROUTE,
        "routes": CALIBRATION_ROUTES,
        "repeats": CALIBRATION_REPEATS,
        "episodes": calibration_episodes,
        "solo_capability_card_no_delegation": True,
        "avg_calls_per_episode": CALIBRATION_AVG_CALLS_PER_EPISODE,
        "upper_calls_per_episode": CALIBRATION_UPPER_CALLS_PER_EPISODE,
        "avg_calls": calibration_avg,
        "upper_calls": calibration_upper,
        "status": "DECLARED_PLANNING_BLOCK_PENDING_ROUTE_PREFLIGHT_BINDING",
        "source": "JOB29 prompt-budget audit; not present in the JOB26 run-plan draft",
        "admission_rule": (
            "calibration tasks are separate frozen tasks with public expected outcomes; "
            "they may not be main episodes, holdout content or outcome-dependent"
        ),
        "task_ids": "MECHANICAL_LATER_STAGE: bound by the frozen admission rule at "
                    "ROUTE_PREFLIGHT_BEFORE_FIRST_CALL",
    }
    probes = {
        "kind": "vera.v13_probe_block.v1",
        "calls_per_route": PROBE_CALLS_PER_ROUTE,
        "routes": CALIBRATION_ROUTES,
        "calls": probe_calls,
        "probe": "A6 self-identity probe, one call per route",
        "avg_calls": probe_calls,
        "upper_calls": probe_calls,
    }
    token_calibration = {
        "anthropic_estimate_rule": "ceil(chars / 3.6)",
        "anthropic_chars_per_token_band": [3.2, 4.0],
        "calibration_rule": (
            "On the first real route call, record the route's usage.input_tokens, compare "
            "it with the recorded ceil(chars/3.6) estimate (Anthropic) or the recorded "
            "tokenizer count (OpenAI-compatible), and log the measured/estimated ratio "
            "with the run.  Keep the planning estimate and the measured value side by "
            "side; do not replace prior estimates retrospectively."
        ),
        "applies_to": [
            "o2_claude_sonnet_5_5",
            "anthropic_claude_haiku_5_5",
            "o1_gpt_6_sol",
        ],
        "status": "PENDING_FIRST_REAL_ROUTE_CALL",
    }
    return {
        "kind": "vera.v13_run_plan.v1",
        "version_id": "V13-RUN-PLAN-4",
        "supersedes_version_id": "V13-RUN-PLAN-3",
        "status": "FROZEN_DRAFT_FOR_ASTRA_REVIEW_NOT_A_RESULT",
        "source": manifest["source"],
        "condition_manifest_digest": manifest["manifest_digest"],
        "U": U,
        "M": table["M"],
        "formula": FORMULA,
        "enrolled_formula": ENROLLED_FORMULA,
        "organization_patterns": {
            "levels": list(patterns),
            "display": {pattern: PATTERN_DISPLAY[pattern] for pattern in patterns},
            "scope": PATTERN_SCOPE,
            "scope_rule": (
                "the owner selected P1-P3 IN; every base row is crossed with every level, "
                "so the factor is enumerated rather than applied as an implicit multiplier"
            ),
            "episode_max_calls": {
                pattern: PATTERN_EPISODE_MAX_CALLS[pattern] for pattern in patterns
            },
            "p3_derivation": (
                "P3 allows one peer->peer delegation: +1 peer decision call; the receiving "
                "peer loop owns the zero-step terminal reply, so 9+1=10"
            ),
            "p4": "built and excluded by owner decision",
        },
        "recommended_repeats": repeats,
        "repeats_used": repeats,
        "repeat_selection": simulation.get("repeat_selection"),
        "power_certified": bool(simulation.get("repeat_selection", {}).get(
            "power_certified", False
        )),
        "episode_counts": {
            "main": base_episodes,
            "main_enrolled": enrolled_episodes,
            "pattern_block_episodes": enrolled_episodes - base_episodes,
            "pattern_episodes_by_level": {
                pattern: e_main(U, M=table["M"], r_N=repeats, r_O=repeats,
                                r_D=repeats, r_A=repeats)
                for pattern in patterns
            },
            # detention_cap is the frozen JOB29 reconciliation input (base P0 core, 120).
            # The current owner-raised cap is detention_cap_enrolled and the detention block.
            "detention_cap": DETENTION_CAP_BASE,
            "detention_cap_base": DETENTION_CAP_BASE,
            "detention_cap_enrolled": DETENTION_CAP,
            "detention_arms": DETENTION_ARMS,
            "detention_retry_repeats_per_failure": DETENTION_RETRY_REPEATS_PER_FAILURE,
            "detention_retry_episodes_upper": (
                DETENTION_CAP * DETENTION_ARMS * DETENTION_RETRY_REPEATS_PER_FAILURE
            ),
            "main_plus_detention_upper": (
                base_episodes + DETENTION_CAP_BASE * DETENTION_ARMS
            ),
            "main_plus_detention_enrolled": (
                enrolled_episodes + DETENTION_CAP * DETENTION_ARMS
                * DETENTION_RETRY_REPEATS_PER_FAILURE
            ),
            "calibration_episodes": calibration_episodes,
            "probe_calls": probe_calls,
        },
        "assignment_table": {
            "expected_episodes": base_expected,
            "actual_episodes": base_episodes,
            "counts_by_condition": table["base_counts_by_condition"],
            "counts_match_formula": table["base_counts_match_formula"],
            "label_balance": table["label_balance"],
            "identity_schedule": table["identity_schedule"],
            "patterns": list(patterns),
            "pattern_scope": PATTERN_SCOPE,
            "pattern_levels": table["pattern_levels"],
            "enrolled_episodes": enrolled_episodes,
            "enrolled_counts_by_condition": table["counts_by_condition"],
            "pattern_call_breakdown": breakdown,
            "assignment_row_fields": [
                "organization_pattern",
                "pattern_level",
                "pattern_episode_max_calls",
                "label_permutation",
                "label_permutation_slots",
                "label_assignment_block",
                "label_assignment_probability",
                "position_assignment_probability",
                "identity_schedule_seed",
                "menu_position_seed",
            ],
            "scope_note": (
                "actual_episodes is the P0_BASE grid used by the frozen JOB29 "
                "reconciliation; enrolled_episodes is the exact P0-P3 cross"
            ),
        },
        "power_simulation": simulation,
        "lineage_audit": lineage,
        "calls": calls,
        "call_bounds": call_bounds,
        "calibration": calibration,
        "probes": probes,
        "token_calibration": token_calibration,
        "capacity_plan": "manifests/delegation/V13_CAPACITY_PLAN.json",
        "route_protocol": (
            "manifests/delegation/V13_ROUTE_PROTOCOL.json"
        ),
        "primary_family": PRIMARY_FAMILY,
        "multiplicity": "Holm two-sided 5% over the four aggregate tests",
        "cluster": "construction lineage (mechanism/template): 28 lineages, 47 mechanisms",
        "detention": {
            "eligibility": (
                "typed genuine failures with a complete ledger and reachable recovery, "
                f"seeded stratified F<={DETENTION_CAP}"
            ),
            "cap": DETENTION_CAP,
            "cap_base": DETENTION_CAP_BASE,
            "cap_frozen_before_main_outcomes": True,
            "cap_selection_stage": "AFTER_INITIAL_GRID_BEFORE_DETENTION_RETRY",
            "cap_selection_rule": (
                simulation.get("detention_cap_selection") or {}
            ).get("selection_rule"),
            "retry_repeats_per_failure": DETENTION_RETRY_REPEATS_PER_FAILURE,
            "descriptive_only_below": DETENTION_DESCRIPTIVE_ONLY_BELOW,
            "shortage_rule": (
                "select min(cap, eligible); when the initial grid yields fewer eligible "
                "failures, every eligible failure is selected and the achieved n, the "
                "shortfall, the realized precision and the realized power are reported.  "
                "No post-hoc cap change and no extra acquisition."
            ),
            "arms": ["A_DETENTION", "B_PLAIN_RETRY", "C_COMPUTE_MATCHED_REVIEW",
                     "D_SCHEMA_UNCHECKED"],
            "primary_endpoint": "common independently audited next-attempt repair Y",
            "required_power_tests": (
                simulation.get("detention_cap_selection") or {}
            ).get("required_tests"),
            "A_minus_D_declared": (
                simulation.get("detention_cap_selection") or {}
            ).get("A_minus_D_declared"),
            "no_parse_gate": True,
            "retry_calls_upper": DETENTION_RETRY_MAX_CALLS_ENROLLED,
        },
        "no_model_calls_in_this_job": True,
        "no_keys": True,
        "no_holdout": True,
    }


simulate_power = prospective_power_simulation
power_simulation = prospective_power_simulation
exact_assignment_table = build_assignment_table


__all__ = [
    "DETENTION_CAP",
    "DETENTION_CAP_BASE",
    "DETENTION_CAP_CANDIDATES",
    "DETENTION_CAP_STUDY_SIMS",
    "DETENTION_DESCRIPTIVE_ONLY_BELOW",
    "DETENTION_RETRY_MAX_CALLS_ENROLLED",
    "DETENTION_RETRY_REPEATS_PER_FAILURE",
    "ENROLLED_PATTERNS",
    "FORMULA",
    "INFRA_RETRY_ATTEMPTS_PER_CALL",
    "PATTERN_EPISODE_MAX_CALLS",
    "PATTERN_SCOPE",
    "build_assignment_table",
    "build_run_plan",
    "e_main",
    "e_main_enrolled",
    "exact_assignment_table",
    "optional_lineage_audit",
    "power_simulation",
    "prospective_power_simulation",
    "simulate_power",
]
