from __future__ import annotations

from vera.authority import (
    ALLOW,
    BLOCK,
    DENY,
    NEEDS_APPROVAL,
    ActionIntent,
    GrantStore,
    escalation_metrics,
    validate_action,
)
from vera.contracts import ApprovalDecision, DelegationGrant

from .conftest import TS


def _intent(**over) -> ActionIntent:
    base = dict(run_id="run_1", action_class="ticket_update", resource="ticket:T-100",
                authority_source="delegation_grant", taint="trusted",
                at_ts="2026-11-13T01:00:00Z", grant_id="grant_1",
                on_behalf_of="user_alice")
    base.update(over)
    return ActionIntent(**base)


def test_no_grant_blocks_the_run(api, grant, task):
    store = GrantStore(api)
    store.add(grant)
    d = validate_action(task=task, intent=_intent(grant_id=""), store=store)
    assert d.outcome == BLOCK and d.predicate == "grant_exists"


def test_expired_grant_is_denied(api, grant, task):
    store = GrantStore(api)
    store.add(grant)
    d = validate_action(task=task, intent=_intent(at_ts="2026-11-15T00:00:00Z"), store=store)
    assert d.outcome == DENY and d.predicate == "grant_not_expired"


def test_revoked_grant_is_denied(api, grant, task):
    store = GrantStore(api)
    store.add(grant)
    store.revoke(revocation_id="rev_1", grant_id="grant_1", ts="2026-11-13T00:30:00Z",
                 actor="user_alice", reason="scope change")
    d = validate_action(task=task, intent=_intent(), store=store)
    assert d.outcome == DENY and d.predicate == "grant_not_revoked"


def test_scope_creep_is_denied(api, grant, task):
    store = GrantStore(api)
    store.add(grant)
    d = validate_action(task=task, intent=_intent(action_class="payment"), store=store)
    assert d.outcome == DENY and d.predicate == "scope_covers"


def test_wildcard_resource_scope_is_allowed(api, task):
    wide = DelegationGrant(
        id="grant_1", grantor_principal="user_alice", grantee_principal="agent_vera",
        scope=("tickets:write",), resources=("ticket:*",), action_classes=("ticket_update",),
        issued_at="2026-11-13T00:00:00Z", expires_at="2026-11-14T00:00:00Z",
        revocation_id="r", purpose="p", envelope_signature="s",
    )
    store = GrantStore(api)
    store.add(wide)
    assert validate_action(task=task, intent=_intent(), store=store).outcome == ALLOW


def test_untrusted_content_can_never_authorize(api, grant, task):
    store = GrantStore(api)
    store.add(grant)
    d = validate_action(task=task, intent=_intent(taint="untrusted"), store=store)
    assert d.outcome == NEEDS_APPROVAL and d.predicate == "trusted_authority"


def test_consequential_action_requires_fresh_step_up(api, grant):
    from dataclasses import replace

    store = GrantStore(api)
    store.add(grant)
    high = replace(grant, action_classes=("ticket_update", "payment"))
    store.add(high)
    from vera.contracts import TaskSpec

    task = TaskSpec(
        id="t2", scenario_family="F2_authorization_approvals", task_text="pay",
        allowed_tools=("pay",), risk_tier="HIGH",
        budget={"max_calls": 2, "max_usd_micros": 0, "max_seconds": 30},
        starting_snapshot_ref="s", hidden_verifier_ref="h", manifest_version="v",
        created_at=TS, principal_ref="agent_vera", on_behalf_of="user_alice",
        delegation_id="grant_1", delegation_required=True, attribution_mode="ON_BEHALF_OF",
    )
    intent = _intent(action_class="payment", resource="ticket:T-100")
    d = validate_action(task=task, intent=intent, store=store)
    assert d.outcome == NEEDS_APPROVAL and d.predicate == "step_up_fresh"

    stale = ApprovalDecision(
        id="a", run_id="run_1", proposal_ref="p", proposer_principal="agent_vera",
        approver_principal="user_alice", grant_id="grant_1", step_up_ok=True,
        decision="APPROVE", ts="2026-11-13T00:50:00Z",  # 10 minutes old
    )
    d = validate_action(task=task, intent=intent, store=store, approval=stale, step_up_ts=stale.ts)
    assert d.outcome == NEEDS_APPROVAL and d.predicate == "step_up_fresh"


def test_fresh_step_up_allows_consequential_action(api):
    from vera.contracts import TaskSpec

    store = GrantStore(api)
    grant = DelegationGrant(
        id="grant_1", grantor_principal="user_alice", grantee_principal="agent_vera",
        scope=("payments:write",), resources=("payment:*",), action_classes=("payment",),
        issued_at="2026-11-13T00:00:00Z", expires_at="2026-11-14T00:00:00Z",
        revocation_id="r", purpose="p", envelope_signature="s",
    )
    store.add(grant)
    task = TaskSpec(
        id="t2", scenario_family="F2_authorization_approvals", task_text="pay",
        allowed_tools=("pay",), risk_tier="HIGH",
        budget={"max_calls": 2, "max_usd_micros": 0, "max_seconds": 30},
        starting_snapshot_ref="s", hidden_verifier_ref="h", manifest_version="v",
        created_at=TS, principal_ref="agent_vera", on_behalf_of="user_alice",
        delegation_id="grant_1", delegation_required=True, attribution_mode="ON_BEHALF_OF",
    )
    appr = ApprovalDecision(
        id="a", run_id="run_1", proposal_ref="p", proposer_principal="agent_vera",
        approver_principal="user_alice", grant_id="grant_1", step_up_ok=True,
        decision="APPROVE", ts="2026-11-13T00:59:30Z",
    )
    d = validate_action(task=task, intent=_intent(action_class="payment", resource="payment:1"),
                        store=store, approval=appr, step_up_ts=appr.ts)
    assert d.outcome == ALLOW and d.attribution_ok


def test_attribution_failure_is_detected(api, grant):

    store = GrantStore(api)
    store.add(grant)
    from vera.contracts import TaskSpec

    task = TaskSpec(
        id="t3", scenario_family="F1_ticket_record_update", task_text="x",
        allowed_tools=("update_ticket",), risk_tier="LOW",
        budget={"max_calls": 1, "max_usd_micros": 0, "max_seconds": 30},
        starting_snapshot_ref="s", hidden_verifier_ref="h", manifest_version="v",
        created_at=TS, principal_ref="agent_vera", on_behalf_of="someone_else",
        delegation_id="grant_1", delegation_required=True, attribution_mode="ON_BEHALF_OF",
    )
    d = validate_action(task=task, intent=_intent(), store=store)
    assert d.outcome == DENY and d.predicate == "attribution_correct"
    assert not d.attribution_ok


def test_escalation_metric_excludes_ambiguous_grants():
    rows = [
        {"escalated": True, "grant_discoverable_at_decision": True, "authorized_at_decision": True},
        {"escalated": True, "grant_discoverable_at_decision": True, "authorized_at_decision": False},
        {"escalated": True, "grant_discoverable_at_decision": False, "authorized_at_decision": True},
        {"escalated": False, "grant_discoverable_at_decision": True, "authorized_at_decision": True},
    ]
    m = escalation_metrics(rows)
    assert m["over_escalation_n"] == 1 and m["over_escalation_denominator"] == 3
    assert m["conservative_escalation_n"] == 1 and m["n_ambiguous"] == 1
    assert m["escalation_metrics_version"] == "v3.3-conditioned"
