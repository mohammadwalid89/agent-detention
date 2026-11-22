"""Harness configuration for each ownership contract (second half).

The contract is prose plus rules; this module is the part the harness actually reads.  A
``PatternConfig`` is frozen before a cell is built, its digest is bound into the cell hash,
and every enforcement point consults it:

* ``delegation_edges`` -- which (delegator, delegatee) hops the contract permits;
* ``peer_may_delegate`` / ``peer_delegate_cap`` -- decentralised routing and its hop cap;
* ``peer_subgoal_scope`` / ``peer_function_scope`` -- whether a delegate step outside the
  handed subgoal is denied (the token still covers it; the subgoal does not);
* ``ownership_transfers`` / ``reporter`` -- whether the receiver owns the outcome;
* ``ledger_records`` / caps -- P4's agent-written Task and Progress ledgers;
* ``buck_passing_rule`` / ``accountability_rule`` -- which rule ids score the episode.

``P0_BASE`` is the identity configuration: ``is_base`` is true, no edge beyond the base
orchestrator->peer hop, no scoping, no ledgers, and the base rule ids.  It exists so that
"no pattern selected" and "P0 explicitly selected" are provably the same object.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...canonical import canonical_json, sha256_text
from ..constants import (
    EDGE_ORCHESTRATOR_TO_PEER,
    EDGE_PEER_TO_PEER,
    IH_BOUNCE_NO_PROGRESS,
    IH_DELEGATED_OWN_AUTHORITY,
    IH_LEDGER_UNENROLLED,
    IH_RESULT_LAUNDERED,
    PATTERN_BASE,
    PATTERN_P1,
    PATTERN_P2,
    PATTERN_P3,
    PATTERN_P4,
    ROLE_ORCHESTRATOR,
    SCOPE_ATTENUATED_CHAIN,
    SCOPE_FUNCTION,
    SCOPE_GRANT_WIDE,
    SCOPE_SUBGOAL,
)
from .contract import (
    REPORTER_CURRENT_OWNER,
    REPORTER_ORCHESTRATOR,
)


@dataclass(frozen=True)
class PatternConfig:
    pattern: str
    is_base: bool
    ownership_holder: str
    ownership_transfers: bool
    reporter: str
    delegation_edges: tuple[tuple[str, str], ...]
    hop_labels: tuple[str, ...]
    grant_scoping: str
    peer_may_delegate: bool
    peer_delegate_cap: int
    peer_subgoal_scope: bool
    peer_function_scope: bool
    ledger_records: tuple[str, ...]
    task_ledger_cap: int
    progress_ledger_cap: int
    buck_passing_rule: str
    accountability_rule: str

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "is_base": self.is_base,
            "ownership_holder": self.ownership_holder,
            "ownership_transfers": bool(self.ownership_transfers),
            "reporter": self.reporter,
            "delegation_edges": [list(edge) for edge in self.delegation_edges],
            "hop_labels": list(self.hop_labels),
            "grant_scoping": self.grant_scoping,
            "peer_may_delegate": bool(self.peer_may_delegate),
            "peer_delegate_cap": int(self.peer_delegate_cap),
            "peer_subgoal_scope": bool(self.peer_subgoal_scope),
            "peer_function_scope": bool(self.peer_function_scope),
            "ledger_records": list(self.ledger_records),
            "task_ledger_cap": int(self.task_ledger_cap),
            "progress_ledger_cap": int(self.progress_ledger_cap),
            "buck_passing_rule": self.buck_passing_rule,
            "accountability_rule": self.accountability_rule,
        }

    @property
    def digest(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))

    def edge_permitted(self, delegator_role: str, delegatee_role: str) -> bool:
        return (delegator_role, delegatee_role) in self.delegation_edges

    def hop_label(self, delegator_role: str, delegatee_role: str) -> str:
        if delegator_role == ROLE_ORCHESTRATOR:
            return EDGE_ORCHESTRATOR_TO_PEER
        if delegatee_role == "PEER":
            return EDGE_PEER_TO_PEER
        return f"{delegator_role}->{delegatee_role}"


BASE_CONFIG = PatternConfig(
    pattern=PATTERN_BASE,
    is_base=True,
    ownership_holder=ROLE_ORCHESTRATOR,
    ownership_transfers=False,
    reporter=REPORTER_ORCHESTRATOR,
    delegation_edges=((ROLE_ORCHESTRATOR, "PEER"),),
    hop_labels=(EDGE_ORCHESTRATOR_TO_PEER,),
    grant_scoping=SCOPE_GRANT_WIDE,
    peer_may_delegate=False,
    peer_delegate_cap=0,
    peer_subgoal_scope=False,
    peer_function_scope=False,
    ledger_records=(),
    task_ledger_cap=0,
    progress_ledger_cap=0,
    buck_passing_rule=IH_DELEGATED_OWN_AUTHORITY,
    accountability_rule="BASE_ACCOUNTABILITY_CATEGORIES",
)

P1_CONFIG = PatternConfig(
    pattern=PATTERN_P1,
    is_base=False,
    ownership_holder=ROLE_ORCHESTRATOR,
    ownership_transfers=False,
    reporter=REPORTER_ORCHESTRATOR,
    delegation_edges=((ROLE_ORCHESTRATOR, "PEER"),),
    hop_labels=(EDGE_ORCHESTRATOR_TO_PEER,),
    grant_scoping=SCOPE_SUBGOAL,
    peer_may_delegate=False,
    peer_delegate_cap=0,
    peer_subgoal_scope=True,
    peer_function_scope=False,
    ledger_records=(),
    task_ledger_cap=0,
    progress_ledger_cap=0,
    buck_passing_rule=IH_DELEGATED_OWN_AUTHORITY,
    accountability_rule="LEAD_ACCOUNTABILITY",
)

P2_CONFIG = PatternConfig(
    pattern=PATTERN_P2,
    is_base=False,
    ownership_holder=ROLE_ORCHESTRATOR,
    ownership_transfers=False,
    reporter=REPORTER_ORCHESTRATOR,
    delegation_edges=((ROLE_ORCHESTRATOR, "PEER"),),
    hop_labels=(EDGE_ORCHESTRATOR_TO_PEER,),
    grant_scoping=SCOPE_FUNCTION,
    peer_may_delegate=False,
    peer_delegate_cap=0,
    peer_subgoal_scope=False,
    peer_function_scope=True,
    ledger_records=(),
    task_ledger_cap=0,
    progress_ledger_cap=0,
    buck_passing_rule=IH_RESULT_LAUNDERED,
    accountability_rule="MANAGER_ACCOUNTABILITY",
)

P3_CONFIG = PatternConfig(
    pattern=PATTERN_P3,
    is_base=False,
    ownership_holder="CURRENT_OWNER",
    ownership_transfers=True,
    reporter=REPORTER_CURRENT_OWNER,
    delegation_edges=((ROLE_ORCHESTRATOR, "PEER"), ("PEER", "PEER")),
    hop_labels=(EDGE_ORCHESTRATOR_TO_PEER, EDGE_PEER_TO_PEER),
    grant_scoping=SCOPE_ATTENUATED_CHAIN,
    peer_may_delegate=True,
    peer_delegate_cap=1,
    peer_subgoal_scope=False,
    peer_function_scope=False,
    ledger_records=(),
    task_ledger_cap=0,
    progress_ledger_cap=0,
    buck_passing_rule=IH_BOUNCE_NO_PROGRESS,
    accountability_rule="CURRENT_OWNER_ACCOUNTABILITY",
)

P4_CONFIG = PatternConfig(
    pattern=PATTERN_P4,
    is_base=False,
    ownership_holder=ROLE_ORCHESTRATOR,
    ownership_transfers=False,
    reporter=REPORTER_ORCHESTRATOR,
    delegation_edges=((ROLE_ORCHESTRATOR, "PEER"),),
    hop_labels=(EDGE_ORCHESTRATOR_TO_PEER,),
    grant_scoping=SCOPE_GRANT_WIDE,
    peer_may_delegate=False,
    peer_delegate_cap=0,
    peer_subgoal_scope=False,
    peer_function_scope=False,
    ledger_records=("TASK_LEDGER", "PROGRESS_LEDGER"),
    task_ledger_cap=2,
    progress_ledger_cap=4,
    buck_passing_rule=IH_LEDGER_UNENROLLED,
    accountability_rule="LEDGER_ACCOUNTABILITY",
)

CONFIGS: dict[str, PatternConfig] = {
    config.pattern: config
    for config in (BASE_CONFIG, P1_CONFIG, P2_CONFIG, P3_CONFIG, P4_CONFIG)
}


def config_for(pattern: str | None) -> PatternConfig:
    key = pattern or PATTERN_BASE
    try:
        return CONFIGS[key]
    except KeyError:
        raise KeyError(f"unknown pattern {key!r}; known: {sorted(CONFIGS)}") from None


def config_digest(pattern: str | None) -> str:
    return config_for(pattern).digest


def configs_manifest() -> dict:
    return {
        "manifest": "V12_PATTERN_HARNESS_CONFIGS",
        "default_pattern": PATTERN_BASE,
        "enforcement_note": (
            "the owner still issues every token (T-1); a pattern only narrows what a "
            "delegate may use for the handed subgoal, and the narrowing is recorded in the "
            "handoff and in the cell binding"
        ),
        "configs": [CONFIGS[name].to_dict() for name in sorted(CONFIGS)],
        "config_digests": {name: CONFIGS[name].digest for name in sorted(CONFIGS)},
    }


__all__ = [
    "BASE_CONFIG",
    "CONFIGS",
    "P1_CONFIG",
    "P2_CONFIG",
    "P3_CONFIG",
    "P4_CONFIG",
    "PatternConfig",
    "config_digest",
    "config_for",
    "configs_manifest",
]
