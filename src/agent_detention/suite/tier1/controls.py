"""Scripted controls run through the REAL harness world (``run_episode``), no model call.

always_act   commits every critical tool once (never stops)
coin         decision-level random: a seeded fair coin picks always_act or always_stop for the episode
                 (this is the "random near 50 %" control of the compatibility gate)
always_stop  escalates at once
noop         finishes without any action (a "did nothing" control, to show the stop clause matters)
random       uniform random permitted action each step, ``seeds`` seeds (mean); descriptive only

Cells are built exactly as ``tools/delegation/floors.py`` builds them (solo, delegation budget 0, protocol
v13: v14 needs a signed case-text overlay per mechanism, which AgentAbstain cases do not have yet).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

from . import check as checks
from . import harness

POLICIES = ("always_act", "always_stop", "noop", "coin", "random")
HARNESS_KEY = b"suite-harness"
VERIFIER_KEY = b"suite-verifier"


def _imports():
    harness.use_tree()
    if not harness.is_patched():
        raise RuntimeError("the harness tree lacks the per-variant overrides in vera/pilot/instances.py; "
                           "use harness/ from this repository. Refusing: without them both variants would see A's tables.")
    from tools.delegation import floors  # noqa: PLC0415
    from vera.delegation.constants import COND_ABSTENTION, COND_OPTIONAL  # noqa: PLC0415
    from vera.delegation.plan import build_cell  # noqa: PLC0415
    from vera.pilot.instances import _instance_from_record  # noqa: PLC0415

    return floors, COND_ABSTENTION, COND_OPTIONAL, build_cell, _instance_from_record


def always_act_policy(floors, tools):
    taken: list[str] = []

    def body(observation):
        offered = observation["observations"]["permitted_actions"]
        for tool in tools:
            if tool in offered and tool not in taken:
                taken.append(tool)
                return {"action": tool, "reason": "always act"}
        return {"done": True}

    return floors._with_report(body)


def noop_policy(floors):
    return floors._with_report(lambda observation: {"done": True})


def seed_for(instance_id: str, seed: int) -> int:
    return int(hashlib.sha256(f"{instance_id}|{seed}".encode()).hexdigest()[:12], 16)


def build(record: dict, variant_index: int):
    floors, cond_abs, cond_opt, build_cell, from_record = _imports()
    instance = from_record(record, record["variants"][variant_index])
    condition = cond_opt if instance.label == "SOLVABLE" else cond_abs
    cell = build_cell(instance, condition, four_slot=True, protocol="v13")
    return instance, dataclasses.replace(cell, delegation_budget=0)


def run_policy(record: dict, variant_index: int, policy: str, seed: int = 0) -> dict:
    floors = _imports()[0]
    instance, cell = build(record, variant_index)
    tools = [a for a in record["permitted_actions"] if a not in ("escalate", "abstain")]
    if policy == "always_act":
        pol = always_act_policy(floors, tools)
    elif policy == "always_stop":
        pol = floors.always_escalate_policy()
    elif policy == "noop":
        pol = noop_policy(floors)
    elif policy == "coin":
        heads = seed_for(instance.instance_id, seed) % 2 == 0
        pol = always_act_policy(floors, tools) if heads else floors.always_escalate_policy()
    elif policy == "random":
        pol = floors.uniform_random_policy(seed_for(instance.instance_id, seed))
    else:
        raise ValueError(policy)
    result = floors._episode(cell, pol, f"{policy}-{instance.instance_id}-{seed}")
    kind = "A" if variant_index == 0 else "B"
    return {"instance_id": instance.instance_id, "label": instance.label, "policy": policy, "seed": seed,
            "check": checks.check_episode(kind, result, record),
            "termination": result["termination_reason"], "typed_outcome": result["facts"].get("typed_outcome")}


def run_pair(record: dict, seeds: int = 20) -> dict:
    """All controls on both variants of one pair."""
    out: dict = {"mechanism_id": record["mechanism_id"], "scenario": record["provenance"]["scenario"],
                 "A": {}, "B": {}}
    for index, key in ((0, "A"), (1, "B")):
        for policy in ("always_act", "always_stop", "noop"):
            out[key][policy] = run_policy(record, index, policy)
        for policy in ("coin", "random"):
            runs = [run_policy(record, index, policy, seed) for seed in range(seeds)]
            out[key][policy] = {
                "pass_rate": sum(r["check"]["pass"] for r in runs) / seeds,
                "disagreements": sum(not r["check"]["agrees_with_harness"] for r in runs),
                "attempt_then_stop": sum(r["check"]["disagreement"] == "ATTEMPT_THEN_STOP" for r in runs),
                "unexplained": sum(r["check"]["disagreement"] == "UNEXPLAINED" for r in runs),
                "seeds": seeds,
            }
    return out


def dump(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, default=str)
