"""Authority, delegation and taint policy (design §5.3–5.4, owner direction A–D).

The model may propose; only a signed task envelope or a valid delegation grant may
authorize. Every check is evaluated at execution time against stored timestamps.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Iterable

from .contracts import (
    CONSEQUENTIAL_ACTION_CLASSES,
    ApprovalDecision,
    DelegationGrant,
    TaskSpec,
)
from .errors import AuthorityError

ALLOW = "ALLOW"
NEEDS_APPROVAL = "NEEDS_APPROVAL"
DENY = "DENY"
BLOCK = "BLOCK_NO_GRANT"

STEP_UP_TTL_SECONDS = 300

# Predicate names are stable strings so gate artifacts can name the failing check.
P_GRANT_EXISTS = "grant_exists"
P_GRANT_ISSUED = "grant_issued"
P_GRANT_NOT_EXPIRED = "grant_not_expired"
P_GRANT_NOT_REVOKED = "grant_not_revoked"
P_SCOPE_COVERS = "scope_covers"
P_RESOURCE_COVERS = "resource_covers"
P_NO_SELF_APPROVAL = "no_self_approval"
P_STEP_UP_FRESH = "step_up_fresh"
P_TRUSTED_AUTHORITY = "trusted_authority"
P_ATTRIBUTION = "attribution_correct"


@dataclass(frozen=True)
class GrantCheck:
    valid: bool
    predicate: str
    detail: str
    grant: DelegationGrant | None = None


@dataclass(frozen=True)
class ActionIntent:
    """What the executor wants to do, before any call is made."""

    run_id: str
    action_class: str
    resource: str
    authority_source: str  # signed_task_envelope | delegation_grant | untrusted
    taint: str  # trusted | untrusted
    at_ts: str
    grant_id: str = ""
    on_behalf_of: str = ""


@dataclass(frozen=True)
class AuthorityDecision:
    outcome: str
    predicate: str
    reason: str
    attribution_ok: bool
    grant_id: str = ""

    @property
    def allowed(self) -> bool:
        return self.outcome == ALLOW


class GrantStore:
    """Reads grants/revocations from the append-only domain tables."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def add(self, grant: DelegationGrant) -> None:
        # Rows are appended through db.insert_record so the hash chain covers them.
        from .db import insert_record

        insert_record(self.conn, "delegation_grant", grant, ts=grant.issued_at)

    def revoke(self, *, revocation_id: str, grant_id: str, ts: str, actor: str, reason: str) -> None:
        self.conn.execute(
            "INSERT INTO grant_revocation (revocation_id, grant_id, ts, actor, reason) VALUES (?,?,?,?,?)",
            (revocation_id, grant_id, ts, actor, reason),
        )

    def get(self, grant_id: str) -> DelegationGrant | None:
        row = self.conn.execute(
            "SELECT record_json FROM delegation_grant WHERE json_extract(record_json,'$.id')=? "
            "ORDER BY seq DESC LIMIT 1",
            (grant_id,),
        ).fetchone()
        if row is None:
            return None
        import json

        from .contracts import DelegationGrant as DG

        d = json.loads(row["record_json"])
        return DG(
            **{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k != "_schema"}
        )

    def revoked_at(self, grant_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT ts FROM grant_revocation WHERE grant_id=? ORDER BY ts ASC LIMIT 1", (grant_id,)
        ).fetchone()
        return None if row is None else str(row["ts"])

    def check(self, grant_id: str, action_class: str, resource: str, at_ts: str) -> GrantCheck:
        grant = self.get(grant_id)
        if grant is None:
            return GrantCheck(False, P_GRANT_EXISTS, f"no grant {grant_id!r}")
        if at_ts < grant.issued_at:
            return GrantCheck(False, P_GRANT_ISSUED, "action predates the grant", grant)
        if at_ts >= grant.expires_at:
            return GrantCheck(False, P_GRANT_NOT_EXPIRED, f"grant expired at {grant.expires_at}", grant)
        revoked = self.revoked_at(grant.id)
        if revoked is not None and at_ts >= revoked:
            return GrantCheck(False, P_GRANT_NOT_REVOKED, f"grant revoked at {revoked}", grant)
        if action_class not in grant.action_classes and "*" not in grant.action_classes:
            return GrantCheck(False, P_SCOPE_COVERS, f"{action_class!r} outside {grant.action_classes}", grant)
        if resource not in grant.resources and "*" not in grant.resources:
            # prefix scopes are allowed as "res:prefix/*"
            ok = any(r.endswith("*") and resource.startswith(r[:-1]) for r in grant.resources)
            if not ok:
                return GrantCheck(False, P_RESOURCE_COVERS, f"{resource!r} outside {grant.resources}", grant)
        return GrantCheck(True, "ok", "grant valid", grant)


def _seconds_between(a: str, b: str) -> int:
    """ISO-8601 UTC timestamps with second precision; returns b - a in seconds."""
    from datetime import datetime

    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return int((datetime.strptime(b, fmt) - datetime.strptime(a, fmt)).total_seconds())


def attribution_ok(task: TaskSpec, grant: DelegationGrant | None) -> bool:
    if task.attribution_mode == "AS_SELF":
        return not task.on_behalf_of
    if task.attribution_mode == "ON_BEHALF_OF":
        if grant is None:
            return False
        return (
            grant.grantee_principal == task.principal_ref
            and grant.grantor_principal == task.on_behalf_of
            and task.on_behalf_of != task.principal_ref
        )
    return False


def validate_action(
    *,
    task: TaskSpec,
    intent: ActionIntent,
    store: GrantStore,
    approval: ApprovalDecision | None = None,
    step_up_ts: str | None = None,
) -> AuthorityDecision:
    """Deterministic pre-execution authority decision. No model output can override it."""
    if intent.taint == "untrusted" or intent.authority_source == "untrusted":
        return AuthorityDecision(
            NEEDS_APPROVAL, P_TRUSTED_AUTHORITY,
            "authority derived from untrusted content; approval required",
            attribution_ok=False,
        )

    grant = None
    if task.delegation_required or intent.grant_id or task.delegation_id:
        # The write-intent must cite the grant it relies on; inheriting it from the task
        # would let an executor act without naming its authority.
        grant_id = intent.grant_id
        if not grant_id:
            return AuthorityDecision(BLOCK, P_GRANT_EXISTS,
                                     "delegation required but the intent cites no grant", False)
        chk = store.check(grant_id, intent.action_class, intent.resource, intent.at_ts)
        if not chk.valid:
            outcome = DENY if chk.predicate in (P_SCOPE_COVERS, P_RESOURCE_COVERS) else DENY
            return AuthorityDecision(outcome, chk.predicate, chk.detail, False, grant_id)
        grant = chk.grant
        if grant is not None and grant.grantor_principal != task.on_behalf_of and task.attribution_mode == "ON_BEHALF_OF":
            return AuthorityDecision(DENY, P_ATTRIBUTION, "grantor is not the on-behalf-of principal", False, grant_id)

    consequential = intent.action_class in CONSEQUENTIAL_ACTION_CLASSES or task.risk_tier == "HIGH"
    if consequential:
        if approval is None or approval.decision != "APPROVE":
            return AuthorityDecision(
                NEEDS_APPROVAL, P_STEP_UP_FRESH,
                "consequential action requires a fresh step-up approval",
                attribution_ok(task, grant), intent.grant_id,
            )
        if approval.proposer_principal == approval.approver_principal:
            return AuthorityDecision(DENY, P_NO_SELF_APPROVAL, "proposer may not approve their own action", False, intent.grant_id)
        if step_up_ts is None:
            return AuthorityDecision(NEEDS_APPROVAL, P_STEP_UP_FRESH, "no step-up timestamp", attribution_ok(task, grant), intent.grant_id)
        age = _seconds_between(approval.ts, intent.at_ts)
        if age < 0:
            return AuthorityDecision(NEEDS_APPROVAL, P_STEP_UP_FRESH, "approval timestamp is in the future", False, intent.grant_id)
        if age > STEP_UP_TTL_SECONDS:
            return AuthorityDecision(
                NEEDS_APPROVAL, P_STEP_UP_FRESH,
                f"step-up is stale ({age}s > {STEP_UP_TTL_SECONDS}s)",
                attribution_ok(task, grant), intent.grant_id,
            )
        if approval.grant_id and grant is not None and approval.grant_id != grant.id:
            return AuthorityDecision(DENY, P_STEP_UP_FRESH, "approval references a different grant", False, intent.grant_id)

    return AuthorityDecision(
        ALLOW, "ok", "authority valid", attribution_ok(task, grant), intent.grant_id
    )


def require_valid_grant(store: GrantStore, grant_id: str, action_class: str, resource: str, at_ts: str) -> DelegationGrant:
    chk = store.check(grant_id, action_class, resource, at_ts)
    if not chk.valid or chk.grant is None:
        raise AuthorityError(f"{chk.predicate}: {chk.detail}")
    return chk.grant


# Metrics (owner direction D), conditioned per dissent D6/O-2


def escalation_metrics(rows: Iterable[dict]) -> dict:
    """rows: {escalated: bool, grant_discoverable_at_decision: bool, authorized_at_decision: bool}

    Over-escalation is only counted where the grant was discoverable and unambiguous at
    decision time. The ambiguous subset is reported separately as conservative escalation.
    """
    rows = list(rows)
    discoverable = [r for r in rows if r.get("grant_discoverable_at_decision")]
    ambiguous = [r for r in rows if not r.get("grant_discoverable_at_decision")]
    over = [
        r for r in discoverable
        if r.get("escalated") and r.get("authorized_at_decision")
    ]
    correct = [r for r in discoverable if r.get("escalated") and not r.get("authorized_at_decision")]
    return {
        "n_total": len(rows),
        "n_grant_discoverable": len(discoverable),
        "n_ambiguous": len(ambiguous),
        "over_escalation_n": len(over),
        "over_escalation_denominator": len(discoverable),
        "conservative_escalation_n": len(
            [r for r in ambiguous if r.get("escalated")]
        ),
        "correct_escalation_n": len(correct),
        "correct_escalation_denominator": len([r for r in discoverable if not r.get("authorized_at_decision")]),
        "escalation_metrics_version": "v3.3-conditioned",
    }
