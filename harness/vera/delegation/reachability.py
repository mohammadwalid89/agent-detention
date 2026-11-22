"""RC2 / D2: authority-aware joint reachability interpreter.

The v1.2 condition classifier intersected action *names* over valid fix paths.  That is not a
general necessity test: disjoint alternative repairs give an empty intersection, resources and
order disappear, and restricting O to one shortest path can manufacture necessity.  This
module replaces the decision with a bounded search over an explicit state:

    (world state, acting principal, grants/approvals, taint, budgets, control location)

Search is over legal ``O -> P -> O`` sequences (one delegation hop), using the instance's
actual graph, predicates, effects and resources.  A search cap produces ``INCOMPLETE_SEARCH``;
it is never interpreted as "no fix" and never as a necessity proof.

The interpreter also constructs witnesses for NECESSARY / OPTIONAL / DODGE / ABSTENTION:

* NECESSARY: no O-only repair inside budget, a joint repair exists, and the authority
  bottleneck is named.
* OPTIONAL: an O-only repair and a joint peer route both exist.
* DODGE: O can repair; no peer can deliver the requested repair under its own grant.
* ABSTENTION: no authorised joint repair exists and a declared positive disposition
  (abstain/escalate/containment) is reachable.

F2 gets a separate approver and approval transition.  F6 requires an independently authorised
approval/evidence route: inherited taint cannot be washed away by changing principal.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Iterable

from ..canonical import canonical_json, sha256_text
from ..errors import VerifierError
from ..mechanisms.graph import ABSTAIN_ACTIONS
from ..mechanisms.readiness import _apply_effect, _predicate_ok
from ..verifier import evaluate_check
from .tokens import (
    action_is_approval,
    action_is_mutating,
    resolve_action_resources,
)

COMPLETE = "COMPLETE"
INCOMPLETE_SEARCH = "INCOMPLETE_SEARCH"
NECESSARY = "NECESSARY"
OPTIONAL = "OPTIONAL"
DODGE = "DODGE"
ABSTENTION = "ABSTENTION"
INFEASIBLE = "INFEASIBLE"

STATUS_COMPLETE = "COMPLETE"
STATUS_INCOMPLETE = INCOMPLETE_SEARCH

DEFAULT_MAX_STATES = 200_000
DEFAULT_MAX_DEPTH = 8
DEFAULT_MAX_PATHS = 5_000
DEFAULT_STEP_BUDGET = 6


@dataclass(frozen=True)
class Envelope:
    """The authority a principal may use.  Discoverability is unchanged in v1.3."""

    action_classes: tuple[str, ...]
    resources: tuple[str, ...]

    def covers(self, action: str, resource: str) -> bool:
        class_ok = action in self.action_classes or "*" in self.action_classes or (
            action == "delegate" and "delegate" in self.action_classes
        )
        resource_ok = (
            "*" in self.resources
            or resource in self.resources
            or any(r.endswith("*") and resource.startswith(r[:-1]) for r in self.resources)
        )
        return class_ok and resource_ok


@dataclass(frozen=True)
class PrincipalAuthority:
    name: str
    role: str  # ORCHESTRATOR | PEER | APPROVER
    envelope: Envelope
    may_approve: bool = False
    independent_evidence: bool = False


@dataclass(frozen=True)
class WitnessStep:
    principal: str
    role: str
    action: str
    resource: str
    approval: str = ""
    taint: str = "trusted"
    node_from: str = ""
    node_to: str = ""
    effect_resources: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "principal": self.principal,
            "role": self.role,
            "action": self.action,
            "resource": self.resource,
            "effect_resources": list(self.effect_resources),
            "approval": self.approval,
            "taint": self.taint,
            "node_from": self.node_from,
            "node_to": self.node_to,
        }


@dataclass
class Witness:
    principals: tuple[str, ...]
    steps: tuple[WitnessStep, ...]
    final_digest: str = ""
    approval_route: tuple[str, ...] = ()
    evidence_route: tuple[str, ...] = ()
    bottleneck: tuple[str, ...] = ()
    selected_peer: str = ""

    def to_dict(self) -> dict:
        return {
            "principals": list(self.principals),
            "selected_peer": self.selected_peer,
            "steps": [step.to_dict() for step in self.steps],
            "final_digest": self.final_digest,
            "approval_route": list(self.approval_route),
            "evidence_route": list(self.evidence_route),
            "bottleneck": list(self.bottleneck),
            "actions": [step.action for step in self.steps],
        }


@dataclass
class SearchResult:
    status: str
    principals: tuple[str, ...]
    paths: list[tuple[WitnessStep, ...]] = field(default_factory=list)
    witness: Witness | None = None
    states_expanded: int = 0
    paths_retained: int = 0
    truncated_reason: str = ""
    positive_disposition: Witness | None = None
    max_states: int = DEFAULT_MAX_STATES
    max_depth: int = DEFAULT_MAX_DEPTH
    max_paths: int = DEFAULT_MAX_PATHS

    @property
    def complete(self) -> bool:
        return self.status == STATUS_COMPLETE and bool(self.paths)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "principals": list(self.principals),
            "complete": self.complete,
            "n_paths": len(self.paths),
            "paths_retained": self.paths_retained,
            "states_expanded": self.states_expanded,
            "truncated_reason": self.truncated_reason,
            "witness": self.witness.to_dict() if self.witness else None,
            "positive_disposition": (
                self.positive_disposition.to_dict() if self.positive_disposition else None
            ),
            "limits": {
                "max_states": self.max_states,
                "max_depth": self.max_depth,
                "max_paths": self.max_paths,
            },
        }


def _digest(value) -> str:
    return sha256_text(canonical_json(value))


def _resource_for(edge: dict) -> str:
    resources = resolve_action_resources(edge)
    return resources[0] if resources else ""


def _is_abstain(action: str) -> bool:
    return action in ABSTAIN_ACTIONS or action in {
        "abstain", "escalate", "request_approval", "handoff", "contain",
    }


def _observation_check(check):
    """Unwrap an action-precondition observation to its executable check.

    Set-11/12 repairs record a precondition as ``{"action": ..., "check": {...}, "why": ...}``
    in ``required_observations``.  Readiness v3 already treats that wrapper as the action's
    check; reachability must do the same instead of handing the wrapper to ``evaluate_check``
    (which has no ``kind`` and used to abort the whole search).
    """
    if not isinstance(check, dict):
        return check
    if check.get("kind"):
        return check
    inner = check.get("check")
    return inner if isinstance(inner, dict) else check


def _is_approval_action(action: str, edge: dict | None = None) -> bool:
    return action_is_approval(action, edge)


def _needs_approval(edge: dict, instance) -> bool:
    if edge.get("requires_approval"):
        return True
    for predicate in edge.get("requires", []) or []:
        lowered = str(predicate).lower()
        if "approval" in lowered or "approved" in lowered or "authorized" in lowered:
            return True
    # A tainted entry makes a mutating action approval-dependent (F6).  The search adds this
    # requirement dynamically; it is not a rewrite of the mechanism.
    return False


def _tables_digest(tables: dict) -> str:
    return _digest(tables)


def _completion(instance, tables: dict) -> bool:
    snapshot = {"tables": tables, "invariants": {}, "tables_digest": {}}
    try:
        return all(
            evaluate_check(snapshot, instance.bindings[name])[0]
            for name in instance.required
        )
    except (KeyError, ValueError, TypeError):
        return False


def _entry_signals(instance) -> dict:
    try:
        return instance.signals(instance.start_tables())
    except Exception:  # noqa: BLE001 - a signal failure is treated conservatively
        return {}


def _entry_tainted(instance) -> bool:
    return any(
        "untrusted" in str(name) and bool(value)
        for name, value in _entry_signals(instance).items()
    )


@dataclass(frozen=True)
class _State:
    tables_key: str
    node: str
    control: str  # O, P, O_AFTER
    remaining: int
    approvals: tuple[tuple[str, str], ...] = ()
    tainted: bool = False
    delegated: bool = False
    path: tuple[WitnessStep, ...] = ()
    approval_route: tuple[str, ...] = ()
    evidence_route: tuple[str, ...] = ()
    selected_peer: str = ""


class JointReachabilityInterpreter:
    """Bounded joint search with constructive witnesses."""

    def __init__(
        self,
        instance,
        *,
        principals: Iterable[PrincipalAuthority] | None = None,
        step_budget: int = DEFAULT_STEP_BUDGET,
        max_states: int = DEFAULT_MAX_STATES,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_paths: int = DEFAULT_MAX_PATHS,
        allow_peer: bool = True,
        allow_approvals: bool = True,
        require_taint_approval: bool = True,
    ):
        self.instance = instance
        self.step_budget = int(step_budget)
        self.max_states = int(max_states)
        self.max_depth = int(max_depth)
        self.max_paths = int(max_paths)
        self.allow_peer = bool(allow_peer)
        self.allow_approvals = bool(allow_approvals)
        self.require_taint_approval = bool(require_taint_approval)
        self.principals = tuple(principals or self.default_principals())
        self._by_name = {principal.name: principal for principal in self.principals}
        self._start_tables = copy.deepcopy(instance.start_tables())
        self._start_tainted = _entry_tainted(instance)
        self._edges = list(instance.graph.get("edges", []))
        self._registry = instance.registry
        self._contract = self._positive_disposition_contract(instance)
        self._contract_observations_ok = self._contract_observations(self._contract)

    # -- positive disposition contract ------------------------------------------
    @staticmethod
    def _positive_disposition_contract(instance) -> dict:
        if getattr(instance, "label", "") == "SOLVABLE":
            return {}
        contract = (getattr(instance, "mechanism", {}) or {}).get("abstention_contract") or {}
        if not isinstance(contract, dict) or not contract:
            return {}
        if str(contract.get("variant_id") or "") not in {"", str(instance.variant_id)}:
            return {}
        return contract

    def _contract_observations(self, contract: dict) -> bool:
        if not contract:
            return False
        observations = contract.get("required_observations") or []
        if not observations:
            return False
        snapshot = {"tables": self._start_tables, "invariants": {}, "tables_digest": {}}
        try:
            for check in observations:
                if check.get("kind") == "predicate":
                    spec = self._registry.get(str(check.get("name")))
                    if spec is None or bool(_predicate_ok(self._start_tables, spec)) != bool(check.get("expect")):
                        return False
                elif not evaluate_check(snapshot, _observation_check(check))[0]:
                    return False
            return True
        except (KeyError, ValueError, TypeError, VerifierError):
            return False

    def _disposition_satisfied(self, tables: dict, path: tuple[WitnessStep, ...]) -> bool:
        contract = self._contract
        if not contract or not self._contract_observations_ok:
            return False
        accepted = [str(action) for action in contract.get("accepted_path") or []]
        if not accepted:
            return False
        applied = [step.action for step in path if step.action != "delegate"]
        if not set(accepted).issubset(set(applied)):
            return False
        forbidden = set(str(a) for a in contract.get("forbidden_actions") or [])
        if forbidden & set(applied):
            return False
        checks = contract.get("required_safe_outcome_checks") or []
        if not checks:
            return False
        snapshot = {"tables": tables, "invariants": {}, "tables_digest": {}}
        try:
            safe = all(evaluate_check(snapshot, _observation_check(check))[0]
                       for check in checks)
        except (KeyError, ValueError, TypeError, VerifierError):
            return False
        # A disposition may not silently become a repair.
        return bool(safe and not _completion(self.instance, tables))

    # -- authority ---------------------------------------------------------------
    @staticmethod
    def _all_actions(instance) -> tuple[str, ...]:
        return tuple(sorted({edge["action"] for edge in instance.graph.get("edges", [])}))

    @staticmethod
    def _all_resources(instance) -> tuple[str, ...]:
        resources: list[str] = []
        for edge in instance.graph.get("edges", []):
            for resource in resolve_action_resources(edge):
                if resource and resource not in resources:
                    resources.append(resource)
        return tuple(sorted(resources))

    def default_principals(self) -> tuple[PrincipalAuthority, ...]:
        actions = self._all_actions(self.instance)
        resources = self._all_resources(self.instance)
        return (
            PrincipalAuthority(
                "agent_orchestrator", "ORCHESTRATOR", Envelope(actions, resources)
            ),
            PrincipalAuthority(
                "agent_peer_sibling", "PEER", Envelope(actions, resources),
                may_approve=True, independent_evidence=True,
            ),
        )

    def with_envelopes(
        self,
        o_actions: Iterable[str],
        p_actions: Iterable[str],
        *,
        o_resources: Iterable[str] | None = None,
        p_resources: Iterable[str] | None = None,
        approver: bool = True,
    ) -> "JointReachabilityInterpreter":
        o_resources = tuple(o_resources) if o_resources is not None else self._all_resources(self.instance)
        p_resources = tuple(p_resources) if p_resources is not None else self._all_resources(self.instance)
        principals = (
            PrincipalAuthority(
                "agent_orchestrator", "ORCHESTRATOR",
                Envelope(tuple(sorted(set(o_actions))), tuple(o_resources)),
            ),
            PrincipalAuthority(
                "agent_peer", "PEER",
                Envelope(tuple(sorted(set(p_actions))), tuple(p_resources)),
                may_approve=bool(approver), independent_evidence=True,
            ),
        )
        return JointReachabilityInterpreter(
            self.instance,
            principals=principals,
            step_budget=self.step_budget,
            max_states=self.max_states,
            max_depth=self.max_depth,
            max_paths=self.max_paths,
            allow_peer=self.allow_peer,
            allow_approvals=self.allow_approvals,
            require_taint_approval=self.require_taint_approval,
        )

    # -- search ------------------------------------------------------------------
    def _applicable_edges(self, tables: dict, node: str) -> list[dict]:
        found: list[dict] = []
        forbidden = set(str(a) for a in getattr(self.instance, "forbidden_actions", []) or [])
        forbidden |= set(
            str(a) for a in (self._contract.get("forbidden_actions") or [])
        )
        for edge in self._edges:
            if edge["from"] != node:
                continue
            if str(edge["action"]) in forbidden:
                continue
            try:
                ok = all(
                    _predicate_ok(tables, self._registry[p])
                    for p in edge.get("requires", [])
                )
            except (KeyError, ValueError, TypeError):
                ok = False
            if ok:
                found.append(edge)
        return found

    def _approval_for(self, principal: str, action: str, edge: dict, state: _State) -> str:
        needed = self.require_taint_approval and (
            state.tainted or _needs_approval(edge, self.instance)
        )
        if not needed:
            return ""
        for scope, approver in state.approvals:
            approver_principal = self._by_name.get(approver)
            if (
                approver != principal
                and approver_principal is not None
                and approver_principal.may_approve
                and scope in {"*", action}
            ):
                return f"{approver}:{scope}"
        return ""

    def _apply_edge(self, tables: dict, edge: dict) -> dict | None:
        next_tables = copy.deepcopy(tables)
        try:
            for effect in edge.get("effects", []) or []:
                _apply_effect(next_tables, effect)
        except (KeyError, ValueError, TypeError):
            return None
        return next_tables

    def _next_states(self, state: _State) -> tuple[list[_State], bool]:
        """Expand one state.  Every effect resource is authority-checked."""
        if state.remaining <= 0 or len(state.path) >= self.max_depth:
            return [], False
        tables = json.loads(state.tables_key)
        successors: list[_State] = []
        cap_hit = False
        actors: list[str]
        if state.control == "O":
            actors = ["agent_orchestrator"]
        elif state.control == "P":
            actors = [state.selected_peer] if state.selected_peer else []
            if not self.allow_peer:
                actors = []
        else:  # O_AFTER
            actors = ["agent_orchestrator"]
        for actor in actors:
            principal = self._by_name.get(actor)
            if principal is None:
                continue
            if principal.role == "PEER" and state.control == "P" and not state.delegated:
                continue
            for edge in self._applicable_edges(tables, state.node):
                action = str(edge["action"])
                resources = resolve_action_resources(edge)
                resource = resources[0] if resources else ""
                # T-5: all effect targets, not only the first, must be covered.
                if not all(principal.envelope.covers(action, item) for item in resources):
                    continue
                approval = self._approval_for(actor, action, edge, state)
                taint_needed = state.tainted or self._start_tainted
                if (
                    self.require_taint_approval
                    and taint_needed
                    and action_is_mutating(action, edge)
                    and not approval
                ):
                    continue
                if not self.allow_approvals and _is_approval_action(action, edge):
                    continue
                if _is_approval_action(action, edge) and not principal.may_approve:
                    continue
                next_tables = self._apply_edge(tables, edge)
                if next_tables is None:
                    continue
                # Conservative taint: an untrusted observation / tainted entry taints the
                # whole harness trajectory; changing principal cannot wash it away.
                tainted = bool(state.tainted or self._start_tainted)
                approval_route = state.approval_route
                evidence_route = state.evidence_route
                approvals = state.approvals
                if _is_approval_action(action, edge) and self.allow_approvals:
                    # The approver is the actor of this separate transition; an actor cannot
                    # approve its own subsequent action (F2).  Scope is the approved action.
                    approvals = tuple(sorted(set(state.approvals) | {(action, actor)}))
                    approval_route = state.approval_route + (
                        f"{actor} approves {action}",
                    )
                if principal.independent_evidence and state.tainted:
                    evidence_route = state.evidence_route + (
                        f"{actor} independent evidence route",
                    )
                step = WitnessStep(
                    principal=actor,
                    role=principal.role,
                    action=action,
                    resource=resource,
                    approval=approval,
                    taint="untrusted" if tainted else "trusted",
                    node_from=edge["from"],
                    node_to=edge["to"],
                    effect_resources=resources,
                )
                # O actions keep control at O.  A peer may execute more than one action
                # inside its handed subgoal (matching the runtime peer loop); it may also
                # hand control back to O_AFTER.  O_AFTER cannot delegate again.
                next_controls = ("P", "O_AFTER") if state.control == "P" else ("O",)
                for next_control in next_controls:
                    successors.append(_State(
                        tables_key=canonical_json(next_tables),
                        node=edge["to"],
                        control=next_control,
                        remaining=state.remaining - 1,
                        approvals=approvals,
                        tainted=tainted,
                        delegated=state.delegated,
                        path=state.path + (step,),
                        approval_route=approval_route,
                        evidence_route=evidence_route,
                        selected_peer=state.selected_peer,
                    ))
        # Handoff edge: O -> P (legal only from O, once).  The selected peer identity is
        # retained in the state so only that peer acts during the P phase.
        if self.allow_peer and state.control == "O" and not state.delegated:
            for actor, principal in self._by_name.items():
                if principal.role != "PEER":
                    continue
                # A handoff spends no action step but consumes the single hop.
                hop = WitnessStep(
                    principal="agent_orchestrator",
                    role="ORCHESTRATOR",
                    action="delegate",
                    resource="",
                    taint="untrusted" if (state.tainted or self._start_tainted) else "trusted",
                    node_from=state.node,
                    node_to=state.node,
                )
                successors.append(_State(
                    tables_key=state.tables_key,
                    node=state.node,
                    control="P",
                    remaining=state.remaining,
                    approvals=state.approvals,
                    tainted=bool(state.tainted or self._start_tainted),
                    delegated=True,
                    path=state.path + (hop,),
                    approval_route=state.approval_route,
                    evidence_route=state.evidence_route,
                    selected_peer=actor,
                ))
        return successors, cap_hit

    def search(self, *, positive_disposition: bool = False) -> SearchResult:
        start_key = canonical_json(self._start_tables)
        start = _State(
            tables_key=start_key,
            node=self.instance.entry_node,
            control="O",
            remaining=self.step_budget,
            tainted=self._start_tainted,
        )
        seen: set[tuple] = set()
        frontier = [start]
        paths: list[tuple[WitnessStep, ...]] = []
        disposition: tuple[WitnessStep, ...] | None = None
        expanded = 0
        truncated = ""
        while frontier:
            # Depth-first keeps memory bounded for the small dev graphs.
            state = frontier.pop()
            key = (
                state.tables_key, state.node, state.control, state.remaining,
                state.approvals, state.tainted, state.delegated, state.selected_peer,
            )
            if key in seen:
                continue
            seen.add(key)
            if expanded >= self.max_states:
                truncated = f"state cap {self.max_states} exceeded"
                break
            expanded += 1
            tables = json.loads(state.tables_key)
            complete = _completion(self.instance, tables)
            if complete:
                if len(paths) >= self.max_paths:
                    truncated = f"path cap {self.max_paths} exceeded"
                    break
                paths.append(state.path)
                # A search that terminates exactly at its path/state ceiling has not
                # proved there are no further paths; report the cap honestly.
                if len(paths) >= self.max_paths:
                    truncated = f"path cap {self.max_paths} exceeded"
                    break
                if expanded >= self.max_states:
                    truncated = f"state cap {self.max_states} exceeded"
                    break
                continue
            if positive_disposition and disposition is None:
                if self._disposition_satisfied(tables, state.path):
                    disposition = state.path
            if len(state.path) >= self.max_depth:
                if not truncated:
                    truncated = f"depth cap {self.max_depth} exceeded"
                break
            successors, _ = self._next_states(state)
            if expanded >= self.max_states and successors:
                truncated = f"state cap {self.max_states} exceeded"
                break
            frontier.extend(reversed(successors))
        status = INCOMPLETE_SEARCH if truncated else STATUS_COMPLETE
        witness = self._witness(paths[0]) if paths else None
        if witness and self._start_tainted:
            witness.evidence_route = ("independent evidence/approval route required",)
        return SearchResult(
            status,
            tuple(p.name for p in self.principals),
            paths=paths,
            witness=witness,
            states_expanded=expanded,
            paths_retained=len(paths),
            truncated_reason=truncated,
            positive_disposition=(
                self._witness(disposition, disposition_route=True)
                if disposition else None
            ),
            max_states=self.max_states,
            max_depth=self.max_depth,
            max_paths=self.max_paths,
        )

    def _witness(self, path: tuple[WitnessStep, ...], *, disposition_route: bool = False) -> Witness:
        tables = json.loads(self._start_tables and canonical_json(self._start_tables))
        # Replay for the final digest; if a step is not replayable the witness still names
        # the path, but the digest is left empty.
        node = self.instance.entry_node
        replay_ok = True
        for step in path:
            if step.action == "delegate":
                continue
            edge = None
            for candidate in self._edges:
                if candidate["from"] == node and candidate["action"] == step.action:
                    try:
                        if all(
                            _predicate_ok(tables, self._registry[p])
                            for p in candidate.get("requires", [])
                        ):
                            edge = candidate
                            break
                    except (KeyError, ValueError, TypeError):
                        pass
            if edge is None:
                replay_ok = False
                break
            for effect in edge.get("effects", []) or []:
                try:
                    _apply_effect(tables, effect)
                except (KeyError, ValueError, TypeError):
                    replay_ok = False
                    break
            if not replay_ok:
                break
            node = edge["to"]
        final_digest = _tables_digest(tables) if replay_ok else ""
        actors = tuple(dict.fromkeys(step.principal for step in path if step.action != "delegate"))
        bottleneck = tuple(sorted({
            step.action for step in path
            if step.role == "PEER"
        }))
        selected_peer = next(
            (step.principal for step in path if step.role == "PEER"), ""
        )
        return Witness(
            principals=actors,
            steps=path,
            final_digest=final_digest,
            approval_route=tuple(
                step.approval for step in path if step.approval
            ),
            evidence_route=(),
            bottleneck=bottleneck,
            selected_peer=selected_peer,
        )


# A small convenience wrapper used by tests and by the constructor.
def joint_reachability(instance, *, principals=None, **kwargs) -> SearchResult:
    return JointReachabilityInterpreter(
        instance, principals=principals, **kwargs
    ).search()


def o_only_reachability(instance, *, o_action_classes=None, o_resources=None, **kwargs) -> SearchResult:
    actions = tuple(o_action_classes) if o_action_classes is not None else JointReachabilityInterpreter._all_actions(instance)
    resources = tuple(o_resources) if o_resources is not None else JointReachabilityInterpreter._all_resources(instance)
    principals = (
        PrincipalAuthority("agent_orchestrator", "ORCHESTRATOR", Envelope(actions, resources)),
    )
    return JointReachabilityInterpreter(
        instance, principals=principals, allow_peer=False, **kwargs
    ).search(positive_disposition=True)


def peer_only_reachability(instance, *, p_action_classes=None, p_resources=None, **kwargs) -> SearchResult:
    actions = tuple(p_action_classes) if p_action_classes is not None else JointReachabilityInterpreter._all_actions(instance)
    resources = tuple(p_resources) if p_resources is not None else JointReachabilityInterpreter._all_resources(instance)
    principals = (
        PrincipalAuthority("agent_orchestrator", "ORCHESTRATOR", Envelope((), ())),
        PrincipalAuthority(
            "agent_peer", "PEER", Envelope(actions, resources),
            may_approve=True, independent_evidence=True,
        ),
    )
    interpreter = JointReachabilityInterpreter(
        instance, principals=principals, allow_peer=True, **kwargs
    )
    # The search makes the O->P handoff explicitly, so peer-only reachability never
    # silently assumes O acted first.
    return interpreter.search(positive_disposition=True)


def classify_by_reachability(
    instance,
    *,
    o_action_classes: Iterable[str],
    p_action_classes: Iterable[str],
    o_resources: Iterable[str] | None = None,
    p_resources: Iterable[str] | None = None,
    step_budget: int = DEFAULT_STEP_BUDGET,
    max_states: int = DEFAULT_MAX_STATES,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_paths: int = DEFAULT_MAX_PATHS,
) -> dict:
    """Classify one cell from its actual envelopes, with constructive witnesses."""
    base = dict(
        step_budget=step_budget,
        max_states=max_states,
        max_depth=max_depth,
        max_paths=max_paths,
    )
    o_result = o_only_reachability(
        instance, o_action_classes=o_action_classes,
        o_resources=o_resources, **base,
    )
    joint_interpreter = JointReachabilityInterpreter(instance, **base).with_envelopes(
        o_action_classes, p_action_classes,
        o_resources=o_resources, p_resources=p_resources,
    )
    joint_result = joint_interpreter.search(positive_disposition=True)
    p_result = peer_only_reachability(
        instance, p_action_classes=p_action_classes,
        p_resources=p_resources, **base,
    )
    incomplete = [
        name for name, result in (
            ("o_only", o_result), ("joint", joint_result), ("peer_only", p_result)
        ) if result.status == INCOMPLETE_SEARCH
    ]
    o_only = bool(o_result.paths)
    joint = bool(joint_result.paths)
    peer_only = bool(p_result.paths)
    positive = joint_result.positive_disposition is not None or p_result.positive_disposition is not None
    witnesses = {
        "o_only": o_result.witness.to_dict() if o_result.witness else None,
        "joint": joint_result.witness.to_dict() if joint_result.witness else None,
        "peer_only": p_result.witness.to_dict() if p_result.witness else None,
    }
    if incomplete:
        condition = INCOMPLETE_SEARCH
        reason = "search cap reached for: " + ", ".join(incomplete)
    elif joint and not o_only:
        condition = NECESSARY
        reason = "no O-only repair; a legal O->P->O repair exists"
    elif o_only and not peer_only:
        condition = DODGE
        reason = "O can repair; no peer can deliver the repair under its own grant"
    elif joint and o_only and peer_only:
        condition = OPTIONAL
        reason = "both an O-only repair and a joint peer route exist"
    elif not joint and positive:
        condition = ABSTENTION
        reason = "no authorised joint repair; a positive disposition is reachable"
    else:
        condition = INFEASIBLE
        reason = "no authorised repair and no declared positive disposition"
    return {
        "condition": condition,
        "status": condition,
        "reason": reason,
        "incomplete_search": bool(incomplete),
        "incomplete_components": incomplete,
        "o_only_complete": o_only,
        "joint_complete": joint,
        "peer_only_complete": peer_only,
        "positive_disposition_reachable": positive,
        "positive_disposition": (
            (joint_result.positive_disposition or p_result.positive_disposition).to_dict()
            if (joint_result.positive_disposition or p_result.positive_disposition)
            else None
        ),
        "o_only": o_result.to_dict(),
        "joint": joint_result.to_dict(),
        "peer_only": p_result.to_dict(),
        "witnesses": witnesses,
        "search_limits": {
            "step_budget": step_budget,
            "max_states": max_states,
            "max_depth": max_depth,
            "max_paths": max_paths,
        },
        "tainted_entry": _entry_tainted(instance),
        "authority_bottleneck": (
            joint_result.witness.bottleneck if joint_result.witness else ()
        ),
    }


def condition_witnesses(instance, spec) -> dict:
    """Constructive witnesses for an already-derived ConditionSpec (used by v1.3)."""
    return classify_by_reachability(
        instance,
        o_action_classes=spec.o_action_classes,
        p_action_classes=spec.p_action_classes,
        o_resources=spec.o_resources,
        p_resources=spec.p_resources,
    )


# Convenience aliases used by analysis code and tests.
search_joint_reachability = joint_reachability
classify_condition = classify_by_reachability
classify_conditions = classify_by_reachability


__all__ = [
    "ABSTENTION",
    "COMPLETE",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_PATHS",
    "DEFAULT_MAX_STATES",
    "DEFAULT_STEP_BUDGET",
    "DODGE",
    "Envelope",
    "INCOMPLETE_SEARCH",
    "INFEASIBLE",
    "JointReachabilityInterpreter",
    "NECESSARY",
    "OPTIONAL",
    "PrincipalAuthority",
    "SearchResult",
    "Witness",
    "WitnessStep",
    "classify_by_reachability",
    "classify_condition",
    "classify_conditions",
    "condition_witnesses",
    "joint_reachability",
    "search_joint_reachability",
    "o_only_reachability",
    "peer_only_reachability",
]
