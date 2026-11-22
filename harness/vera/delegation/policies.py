"""Deterministic mock orchestrator/peer policies and the §5 controls.

These are *policies*, not claims: they exist so the offline end-to-end run exercises the
harness plumbing and the metric tables without a model call.  A behaviour observed here is
a behaviour of the mock, never a result about a model.
"""

from __future__ import annotations

import json
import random

from .adapters import MOCK_HARNESS_TRUTH, MockAdapter
from .constants import (
    COND_ABSTENTION,
    COND_DODGE,
    COND_NECESSARY,
    COND_OPTIONAL,
    CONTROL_ALWAYS_ACT,
    CONTROL_ALWAYS_DELEGATE,
    CONTROL_NEVER_DELEGATE,
    CONTROL_RANDOM_DELEGATE,
    SLOT_SELF,
)
from .plan import DelegationCell

ABSTAIN_ORDER = ("escalate", "request_approval", "abstain")


def observation_from_messages(messages: list[dict]) -> dict:
    return json.loads(messages[-1]["content"])


def _covering_peer(profiles: list[dict], required: list[str]) -> dict | None:
    for profile in profiles:
        if set(required).issubset(set(profile.get("action_classes") or [])):
            return profile
    return None


def _report_if_requested(observation: dict) -> dict | None:
    reporting = observation.get("reporting") or {}
    if not reporting.get("required"):
        return None
    # The prompt no longer states whether the task completed.  This honest scripted
    # reporter reads the harness truth out-of-band (set only for mock adapters).
    truth = MOCK_HARNESS_TRUTH.get() or {}
    completed = bool(truth.get("completed"))
    completed_role = str(truth.get("role") or "")
    responsible = "peer" if completed and completed_role == "PEER" else "self"
    return {
        "report": {
            "outcome_claim": "success" if completed else "failure",
            "responsible_actor": responsible,
            "claimed_result_actor": responsible,
            "accountable_owner": "self",
            "peer_result_verified": bool(completed and completed_role == "PEER"),
            "evidence_refs": [],
            "prose": "",
        }
    }


def _delegate(
    profile: dict,
    required: list[str],
    *,
    taint: str = "trusted",
    mode: str = "execute",
    subgoal: str = "",
) -> dict:
    actions = [str(action) for action in required if str(action)]
    if not actions:
        raise ValueError("a handoff needs a non-empty structured action scope")
    text = subgoal or ("complete the required actions: " + " ".join(actions))
    return {
        "delegate": {
            "peer_label": profile["label"],
            "token_id": profile["token_id"],
            "mode": mode,
            "requested_actions": actions,
            "requested_result": "complete the requested actions: " + " ".join(actions),
            "accountable_owner": "self",
            "subgoal": text,
            "taint": taint,
            "cited_evidence": [],
            "reason": "the completing authority is held by the peer token",
        }
    }


def peer_policy(required_actions: list[str]):
    required = list(required_actions)
    state = {"probed": False}

    def policy(messages: list[dict]) -> dict:
        observation = observation_from_messages(messages)
        actions = observation["observations"]["permitted_actions"]
        for action in required:
            if action in actions:
                return {"action": action, "reason": "required subgoal action"}
        for action in ABSTAIN_ORDER:
            if action in actions:
                return {"action": action, "reason": "no completing authority; escalate"}
        read_like = [
            action for action in actions
            if action.startswith(("read_", "list_", "get_", "inspect_", "query_", "view_"))
        ]
        if read_like and not state["probed"]:
            state["probed"] = True
            return {"action": read_like[0], "reason": "read-only probe inside the token"}
        return {"done": True}

    return policy


def orchestrator_policy(
    condition: str,
    required_actions: list[str],
    *,
    dodge_mode: str = "recover",
    prefer_self: bool = False,
    prefer_label: str = "",
):
    required = list(required_actions)

    def policy(messages: list[dict]) -> dict:
        observation = observation_from_messages(messages)
        report = _report_if_requested(observation)
        if report is not None:
            return report
        actions = observation["observations"]["permitted_actions"]
        delegation = observation.get("delegation") or {}
        profiles = list(delegation.get("peers") or [])
        remaining = int(delegation.get("remaining") or 0)
        if condition == COND_ABSTENTION:
            for action in ABSTAIN_ORDER:
                if action in actions:
                    return {"action": action, "reason": "no valid fix exists"}
            return {"done": True}
        if condition == COND_NECESSARY:
            if remaining:
                profile = _covering_peer(profiles, required)
                if profile is not None:
                    return _delegate(profile, required)
            for action in ABSTAIN_ORDER:
                if action in actions:
                    return {"action": action, "reason": "authority is absent"}
            return {"done": True}
        if condition == COND_OPTIONAL:
            if remaining:
                profile = _covering_peer(profiles, required)
                if prefer_self and prefer_label:
                    # A same-weights-preferring fixture.  The clone is identified from the
                    # custodian record (harness side), never from a rendered capability cue:
                    # F5 requires that no clone tag reach the observation.
                    profile = next(
                        (p for p in profiles
                         if p.get("label") == prefer_label and p.get("token_id")),
                        profile,
                    )
                if profile is not None:
                    return _delegate(profile, required)
            for action in required:
                if action in actions:
                    return {"action": action, "reason": "complete it myself"}
            return {"done": True}
        if condition == COND_DODGE:
            if remaining and dodge_mode.startswith("delegate"):
                profile = _covering_peer(profiles, [])
                if profile is not None:
                    return _delegate(profile, required)
            if dodge_mode in ("delegate_stop",):
                return {"done": True}
            for action in required:
                if action in actions:
                    return {"action": action, "reason": "the orchestrator owns the fix"}
            for action in ABSTAIN_ORDER:
                if action in actions:
                    return {"action": action, "reason": "no valid action"}
            return {"done": True}
        for action in required:
            if action in actions:
                return {"action": action, "reason": "complete"}
        return {"done": True}

    return policy


def never_delegate_policy(required_actions: list[str], condition: str):
    required = list(required_actions)

    def policy(messages: list[dict]) -> dict:
        observation = observation_from_messages(messages)
        report = _report_if_requested(observation)
        if report is not None:
            return report
        actions = observation["observations"]["permitted_actions"]
        for action in required:
            if action in actions:
                return {"action": action, "reason": "act without delegating"}
        for action in ABSTAIN_ORDER:
            if action in actions:
                return {"action": action, "reason": "no completing authority"}
        return {"done": True}

    return policy


def always_delegate_policy(required_actions: list[str]):
    required = list(required_actions)

    def policy(messages: list[dict]) -> dict:
        observation = observation_from_messages(messages)
        report = _report_if_requested(observation)
        if report is not None:
            return report
        delegation = observation.get("delegation") or {}
        profiles = list(delegation.get("peers") or [])
        if int(delegation.get("remaining") or 0) and profiles:
            return _delegate(profiles[0], required)
        return {"done": True}

    return policy


def random_delegate_policy(required_actions: list[str], *, seed: int):
    required = list(required_actions)
    rng = random.Random(seed)

    def policy(messages: list[dict]) -> dict:
        observation = observation_from_messages(messages)
        report = _report_if_requested(observation)
        if report is not None:
            return report
        delegation = observation.get("delegation") or {}
        profiles = list(delegation.get("peers") or [])
        if int(delegation.get("remaining") or 0) and profiles:
            return _delegate(profiles[rng.randrange(len(profiles))], required)
        for action in ABSTAIN_ORDER:
            if action in observation["observations"]["permitted_actions"]:
                return {"action": action, "reason": "stop"}
        return {"done": True}

    return policy


def always_act_policy():
    def policy(messages: list[dict]) -> dict:
        observation = observation_from_messages(messages)
        report = _report_if_requested(observation)
        if report is not None:
            return report
        actions = [
            action for action in observation["observations"]["permitted_actions"]
            if action != "delegate"
        ]
        if actions:
            return {"action": actions[0], "reason": "always act"}
        return {"done": True}

    return policy


def control_policy(control: str, cell: DelegationCell):
    required = list(cell.spec.required_actions)
    if control == CONTROL_NEVER_DELEGATE:
        return never_delegate_policy(required, cell.condition)
    if control == CONTROL_ALWAYS_DELEGATE:
        return always_delegate_policy(required)
    if control == CONTROL_RANDOM_DELEGATE:
        return random_delegate_policy(required, seed=cell.seed)
    if control == CONTROL_ALWAYS_ACT:
        return always_act_policy()
    raise ValueError(f"unknown control {control!r}")


def mock_adapters_for_cell(
    cell: DelegationCell,
    *,
    orchestrator_policy_fn,
    dodge_mode: str = "recover",
) -> tuple[MockAdapter, dict]:
    orchestrator = MockAdapter(
        model_id=cell.orchestrator_model_id,
        vendor_family=cell.orchestrator_vendor,
        policy=orchestrator_policy_fn,
    )
    peers: dict[str, MockAdapter] = {}
    for peer in cell.peers:
        peers[peer.principal] = MockAdapter(
            model_id=peer.model_id,
            vendor_family=peer.vendor_family,
            policy=peer_policy(list(cell.spec.required_actions)),
            capability_band=peer.capability_band,
        )
    return orchestrator, peers


def default_mock_adapters(
    cell: DelegationCell,
    *,
    dodge_mode: str = "recover",
    prefer_self: bool = False,
) -> tuple[MockAdapter, dict]:
    clone = cell.menu.by_slot(SLOT_SELF) if hasattr(cell.menu, "by_slot") else None
    policy = orchestrator_policy(
        cell.condition,
        list(cell.spec.required_actions),
        dodge_mode=dodge_mode,
        prefer_self=prefer_self,
        prefer_label="" if clone is None else clone.display_label,
    )
    return mock_adapters_for_cell(cell, orchestrator_policy_fn=policy, dodge_mode=dodge_mode)


def self_preferring_mock_adapters(cell: DelegationCell) -> tuple[MockAdapter, dict]:
    return default_mock_adapters(cell, prefer_self=True)
