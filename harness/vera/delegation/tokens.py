"""Authority-scoped, non-transferable, attenuating delegation tokens (design §2.2).

The v1 authority module is *not* modified.  This module implements the three v1.2 rules
that v1 lacks, using the same decision vocabulary (``ALLOW`` / ``NEEDS_APPROVAL`` /
``DENY``) and the same predicate strings where they exist:

* **T-1** owner-only issuance -- a model principal has no issuance path at all;
* **T-2** attenuation -- a grant may not amplify the issuer's action classes, resources
  or lifetime; :func:`validate_grant_issuance` fails closed on amplification;
* **T-3** actor binding -- ``grant.grantee_principal`` must be the acting principal;
* **T-5** resource resolution -- a step's resource is the *effect target*
  (``<table>:<key>``) resolved from the graph edge, not the episode-wide
  ``instance:<ref>`` used by v1.

The store is in-memory and append-only for the episode.  Grants are still ordinary
``vera.contracts.DelegationGrant`` objects, so an offline custodian can replay a log
without this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts import DelegationGrant
from ..errors import VeraError
from .constants import (
    AUTH_ALLOW,
    AUTH_BLOCK_NO_GRANT,
    AUTH_DENY,
    P_ACTOR_BINDING,
    P_TOKEN_EXISTS,
    P_TOKEN_EXPIRED,
    P_TOKEN_NOT_ISSUED,
    P_TOKEN_RESOURCE,
    P_TOKEN_REVOKED,
    P_TOKEN_SCOPE,
)


class TokenError(VeraError):
    """A token could not be issued or used under T-1/T-2/T-3."""


@dataclass(frozen=True)
class AuthorityEnvelope:
    """The authority a principal actually holds for one episode."""

    action_classes: tuple[str, ...]
    resources: tuple[str, ...]
    expires_at: str

    def covers(self, action_class: str, resource: str) -> bool:
        if action_class not in self.action_classes and "*" not in self.action_classes:
            return False
        return _resource_covers(self.resources, resource)


@dataclass(frozen=True)
class TokenCheck:
    valid: bool
    predicate: str
    detail: str
    grant: DelegationGrant | None = None

    @property
    def outcome(self) -> str:
        if self.valid:
            return AUTH_ALLOW
        if self.predicate in (P_TOKEN_SCOPE, P_TOKEN_RESOURCE, P_ACTOR_BINDING):
            return AUTH_DENY
        return AUTH_BLOCK_NO_GRANT


def _resource_covers(scopes: tuple[str, ...] | list[str], resource: str) -> bool:
    if "*" in scopes:
        return True
    if resource in scopes:
        return True
    return any(scope.endswith("*") and resource.startswith(scope[:-1]) for scope in scopes)


def _subset(child: tuple[str, ...] | list[str], parent: tuple[str, ...] | list[str]) -> bool:
    return all(item in parent or "*" in parent for item in child)


def attenuation_errors(
    grant: DelegationGrant,
    issuer_envelope: AuthorityEnvelope,
) -> list[str]:
    """Return every T-2 amplification.  Empty means the grant attenuates."""
    errors: list[str] = []
    if not _subset(grant.action_classes, issuer_envelope.action_classes):
        errors.append(
            "action_classes amplify issuer authority: "
            f"{sorted(set(grant.action_classes) - set(issuer_envelope.action_classes))}"
        )
    for resource in grant.resources:
        if not _resource_covers(issuer_envelope.resources, resource):
            errors.append(f"resource amplifies issuer authority: {resource!r}")
    if grant.expires_at > issuer_envelope.expires_at:
        errors.append(
            f"expires_at {grant.expires_at} outlives issuer {issuer_envelope.expires_at}"
        )
    return errors


def validate_grant_issuance(
    grant: DelegationGrant,
    *,
    issuer_principal: str,
    issuer_envelope: AuthorityEnvelope,
    owner_principal: str,
) -> list[str]:
    """T-1 + T-2, fail closed.  Model principals cannot issue at all."""
    errors: list[str] = []
    if issuer_principal != owner_principal:
        errors.append(
            f"T-1 owner-only issuance: {issuer_principal!r} may not issue a grant "
            f"(owner is {owner_principal!r})"
        )
    if grant.grantor_principal != issuer_principal:
        errors.append(
            f"grantor {grant.grantor_principal!r} is not the issuing principal "
            f"{issuer_principal!r}"
        )
    errors.extend(attenuation_errors(grant, issuer_envelope))
    return errors


class TokenStore:
    """Episode-local token store with the same check shape as ``GrantStore``."""

    def __init__(self, grants: tuple[DelegationGrant, ...] | list[DelegationGrant] = ()):  # noqa: B006
        self._grants: dict[str, DelegationGrant] = {}
        self._revoked: dict[str, str] = {}
        for grant in grants:
            self.add(grant)

    def add(self, grant: DelegationGrant) -> None:
        if grant.id in self._grants:
            raise TokenError(f"grant {grant.id!r} already exists; tokens are append-only")
        self._grants[grant.id] = grant

    def revoke(self, grant_id: str, *, at_ts: str) -> None:
        if grant_id not in self._grants:
            raise TokenError(f"cannot revoke unknown grant {grant_id!r}")
        self._revoked[grant_id] = at_ts

    def get(self, grant_id: str) -> DelegationGrant | None:
        return self._grants.get(grant_id)

    def grants(self) -> list[DelegationGrant]:
        return list(self._grants.values())

    def check(
        self,
        grant_id: str,
        actor_principal: str,
        action_class: str,
        resource: str,
        at_ts: str,
    ) -> TokenCheck:
        grant = self._grants.get(grant_id)
        if grant is None:
            return TokenCheck(False, P_TOKEN_EXISTS, f"no token {grant_id!r}")
        if grant.grantee_principal != actor_principal:
            return TokenCheck(
                False,
                P_ACTOR_BINDING,
                f"token {grant_id!r} was issued to {grant.grantee_principal!r}, "
                f"not to the acting principal {actor_principal!r} (T-3 non-transferable)",
                grant,
            )
        if at_ts < grant.issued_at:
            return TokenCheck(False, P_TOKEN_NOT_ISSUED, "action predates the token", grant)
        if at_ts >= grant.expires_at:
            return TokenCheck(
                False, P_TOKEN_EXPIRED, f"token expired at {grant.expires_at}", grant
            )
        revoked = self._revoked.get(grant.id)
        if revoked is not None and at_ts >= revoked:
            return TokenCheck(False, P_TOKEN_REVOKED, f"token revoked at {revoked}", grant)
        if action_class not in grant.action_classes and "*" not in grant.action_classes:
            return TokenCheck(
                False,
                P_TOKEN_SCOPE,
                f"{action_class!r} outside token classes {grant.action_classes}",
                grant,
            )
        if not _resource_covers(grant.resources, resource):
            return TokenCheck(
                False,
                P_TOKEN_RESOURCE,
                f"{resource!r} outside token resources {grant.resources}",
                grant,
            )
        return TokenCheck(True, "ok", "token valid", grant)


def token_covers(
    grant: DelegationGrant, *, action_classes: tuple[str, ...] | list[str],
    resources: tuple[str, ...] | list[str],
) -> bool:
    return _subset(action_classes, grant.action_classes) and all(
        _resource_covers(grant.resources, resource) for resource in resources
    )


@dataclass(frozen=True)
class ResolvedAction:
    action: str
    resources: tuple[str, ...]

    @property
    def primary(self) -> str:
        return self.resources[0] if self.resources else ""


def resolve_effect_resource(effect: dict) -> str:
    """``<table>:<key>`` for one declared effect (T-5).

    ``set_field`` names its row explicitly.  ``set_flag`` writes the single
    ``mechanism_state`` row, so its resource is ``mechanism_state:state``.
    """
    kind = effect.get("kind")
    if kind == "set_field":
        table = str(effect.get("table", ""))
        where = effect.get("where") or {}
        key = ""
        if isinstance(where, dict) and where:
            key = str(where[sorted(where)[0]])
        return f"{table}:{key}" if table else ""
    if kind == "set_flag":
        return "mechanism_state:state"
    return ""


def resolve_action_resources(edge: dict) -> tuple[str, ...]:
    resources: list[str] = []
    for effect in edge.get("effects", []) or []:
        resource = resolve_effect_resource(effect)
        if resource and resource not in resources:
            resources.append(resource)
    return tuple(resources)


def resolve_action_resource(edge: dict) -> str:
    """Primary effect target for one graph edge; empty when the edge changes nothing."""
    resources = resolve_action_resources(edge)
    return resources[0] if resources else ""


READ_ACTION_PREFIXES = (
    "read_", "list_", "get_", "inspect_", "query_", "view_", "check_",
)
ABSTAIN_ACTION_NAMES = frozenset({"abstain", "escalate", "request_approval"})
APPROVAL_ACTION_TOKENS = ("approve", "approval", "authorize", "authorise")


def action_is_abstain(action: str) -> bool:
    return str(action) in ABSTAIN_ACTION_NAMES


def action_is_approval(action: str, edge: dict | None = None) -> bool:
    lowered = str(action).lower()
    if any(token in lowered for token in APPROVAL_ACTION_TOKENS):
        return True
    for effect in (edge or {}).get("effects", []) or []:
        table = str(effect.get("table", "")).lower()
        field = str(effect.get("field", "")).lower()
        if "approval" in table and field in {"granted", "approved", "state", "status"}:
            return True
    return False


def action_is_mutating(action: str, edge: dict | None = None) -> bool:
    """A repair/write transition, excluding reads, dispositions and approval transitions."""
    lowered = str(action).lower()
    if action_is_abstain(action) or action_is_approval(action, edge):
        return False
    if lowered.startswith(READ_ACTION_PREFIXES):
        return False
    return bool((edge or {}).get("effects"))


def action_is_read_only(action: str, edge: dict | None = None) -> bool:
    """True for an advisory/observation action that cannot repair the task state.

    Consult mode is a read-only execution scope.  Read-prefixed actions may record an
    observation flag (for example ``read_record`` setting an inspection marker), but they do
    not perform a repair.  Approval transitions and dispositions are not consultation reads.
    """
    lowered = str(action).lower()
    if not lowered.startswith(READ_ACTION_PREFIXES):
        return False
    return not action_is_approval(action, edge)


@dataclass(frozen=True)
class ApprovalCheck:
    valid: bool
    problems: tuple[str, ...] = ()
    approver_principal: str = ""
    approval_id: str = ""
    scope_actions: tuple[str, ...] = ()
    scope_resources: tuple[str, ...] = ()


def validate_approval(
    approval: dict | None,
    *,
    actor_principal: str,
    action: str,
    resources: tuple[str, ...] | list[str],
    at_ts: str,
) -> ApprovalCheck:
    """Validate an independent, scoped, live approval for one action and all its effects.

    An actor can never approve its own action.  The approver must carry ``may_approve``
    authority, the approval must name the action/all effect resources, and its TTL must be
    live at the execution timestamp.  Revoked approvals fail closed.
    """
    if not isinstance(approval, dict):
        return ApprovalCheck(False, ("no approval record",))
    problems: list[str] = []
    approver = str(
        approval.get("approver_principal")
        or approval.get("approver")
        or approval.get("reviewer_principal")
        or approval.get("reviewer")
        or ""
    )
    approval_id = str(approval.get("approval_id") or approval.get("id") or "")
    if not approver:
        problems.append("approval has no independent approver principal")
    if approver and approver == actor_principal:
        problems.append("actor may not approve its own action")
    if not bool(approval.get("may_approve", approval.get("approver_may_approve", False))):
        problems.append("approver does not hold may_approve authority")
    if approval_id and approval_id == actor_principal:
        problems.append("approval id launders the actor principal")
    issued = str(approval.get("issued_at") or approval.get("ts") or "")
    expires = str(approval.get("expires_at") or approval.get("expiry") or "")
    if not issued or not expires:
        problems.append("approval has no live issued_at/expires_at TTL")
    else:
        if at_ts < issued:
            problems.append("approval is not yet live")
        if at_ts >= expires:
            problems.append("approval expired")
    revoked = str(approval.get("revoked_at") or "")
    if revoked and at_ts >= revoked:
        problems.append("approval was revoked")
    scope_actions = tuple(
        str(item)
        for item in (
            approval.get("action_classes")
            or approval.get("actions")
            or approval.get("scope_actions")
            or ([] if not approval.get("scope") else [approval.get("scope")])
        )
    )
    if not scope_actions:
        problems.append("approval has no action scope")
    elif action not in scope_actions and "*" not in scope_actions:
        problems.append(f"approval scope does not cover action {action!r}")
    scope_resources = tuple(
        str(item)
        for item in (
            approval.get("resources")
            or approval.get("resource_scope")
            or ([] if not approval.get("resource") else [approval.get("resource")])
        )
    )
    if not scope_resources:
        problems.append("approval has no resource scope")
    else:
        for resource in resources:
            if not _resource_covers(scope_resources, resource):
                problems.append(
                    f"approval scope does not cover effect resource {resource!r}"
                )
    return ApprovalCheck(
        not problems,
        tuple(problems),
        approver_principal=approver,
        approval_id=approval_id,
        scope_actions=scope_actions,
        scope_resources=scope_resources,
    )


@dataclass
class GrantIssuer:
    """Owner-side issuance helper: every grant is validated before it is stored."""

    store: TokenStore
    owner_principal: str
    owner_envelope: AuthorityEnvelope
    issued: list[str] = field(default_factory=list)

    def issue(self, grant: DelegationGrant, *, issuer_principal: str | None = None) -> DelegationGrant:
        issuer = issuer_principal or self.owner_principal
        errors = validate_grant_issuance(
            grant,
            issuer_principal=issuer,
            issuer_envelope=self.owner_envelope,
            owner_principal=self.owner_principal,
        )
        if errors:
            raise TokenError("; ".join(errors))
        self.store.add(grant)
        self.issued.append(grant.id)
        return grant
