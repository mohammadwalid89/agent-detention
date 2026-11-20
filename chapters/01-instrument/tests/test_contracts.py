from __future__ import annotations

import json

import pytest
from vera.canonical import HashChain, canonical_json, chain_hash, dstr
from vera.contracts import (
    ApprovalDecision,
    Attribution,
    DelegationGrant,
    OperationalVerdict,
    PolicyVersion,
    RcaReport,
    RoutingDecision,
    TaskSpec,
    WspExperiment,
)
from vera.errors import ContractError

from .conftest import TS


def test_canonical_json_is_stable_and_sorted():
    a = canonical_json({"b": 1, "a": [1, 2, {"z": "x"}]})
    b = canonical_json({"a": [1, 2, {"z": "x"}], "b": 1})
    assert a == b == '{"a":[1,2,{"z":"x"}],"b":1}'


def test_float_in_hash_relevant_field_is_rejected():
    with pytest.raises(ContractError):
        canonical_json({"x": 1.5})
    with pytest.raises(ContractError):
        canonical_json({"nested": [{"y": 0.1}]})


def test_float_allowed_when_explicitly_not_hash_relevant():
    assert json.loads(canonical_json({"x": 1.5}, hash_safe=False))["x"] == 1.5


def test_dstr_rejects_non_finite():
    with pytest.raises(ContractError):
        dstr(float("nan"))
    assert dstr(0.5) == "0.500000"


def test_hash_chain_detects_tampering():
    chain = HashChain()
    rows = [{"a": 1}, {"a": 2}, {"a": 3}]
    for r in rows:
        chain.append(r)
    assert chain.verify(rows)
    rows[1] = {"a": 99}
    assert not chain.verify(rows)
    assert chain_hash(HashChain.GENESIS, {"a": 1}) == chain._rows[0][2]


def test_task_spec_requires_int_micro_usd_and_delegation(grant):
    with pytest.raises(ContractError):
        TaskSpec(
            id="t", scenario_family="f", task_text="x", allowed_tools=(), risk_tier="LOW",
            budget={"max_calls": 1, "max_usd_micros": 0.5, "max_seconds": 1},
            starting_snapshot_ref="s", hidden_verifier_ref="h", manifest_version="v",
            created_at=TS,
        )
    with pytest.raises(ContractError):
        TaskSpec(
            id="t", scenario_family="f", task_text="x", allowed_tools=(), risk_tier="LOW",
            budget={"max_calls": 1, "max_usd_micros": 0, "max_seconds": 1},
            starting_snapshot_ref="s", hidden_verifier_ref="h", manifest_version="v",
            created_at=TS, delegation_required=True,
        )


def test_on_behalf_of_requires_principal():
    with pytest.raises(ContractError):
        TaskSpec(
            id="t", scenario_family="f", task_text="x", allowed_tools=(), risk_tier="LOW",
            budget={"max_calls": 1, "max_usd_micros": 0, "max_seconds": 1},
            starting_snapshot_ref="s", hidden_verifier_ref="h", manifest_version="v",
            created_at=TS, attribution_mode="ON_BEHALF_OF",
        )


def test_grant_cannot_be_self_granted():
    with pytest.raises(ContractError):
        DelegationGrant(
            id="g", grantor_principal="agent_a", grantee_principal="agent_a",
            scope=("x",), resources=("r",), action_classes=("ticket_update",),
            issued_at=TS, expires_at="2026-11-14T00:00:00Z", revocation_id="r",
            purpose="p", envelope_signature="s",
        )


def test_approval_rejects_self_approval():
    with pytest.raises(ContractError):
        ApprovalDecision(
            id="a", run_id="r", proposal_ref="p", proposer_principal="agent_a",
            approver_principal="agent_a", grant_id="g", step_up_ok=True,
            decision="APPROVE", ts=TS,
        )


def test_router_feature_whitelist_forbids_oracle_keys():
    with pytest.raises(ContractError):
        RoutingDecision(
            id="rd", run_id="r", decision_ts=TS, policy_id="p", catalog_version="c",
            feature_snapshot={"scenario_id": "x"}, feature_schema_version="v",
            eligible_actions=("a",), selected_action="a", selection_probability="1.0",
            selection_reason="x", budget={},
        )


def test_learned_policy_cannot_be_active():
    with pytest.raises(ContractError):
        PolicyVersion(
            id="pv", type="learned", artifact_sha256="a", dataset_sha256="d",
            feature_schema_version="v", code_commit="c", eval_report_sha256="e",
            status="ACTIVE", approved_by="owner", activated_at=TS, rollback_to="",
            claim_scope="NONE",
        )


def test_policy_active_requires_approver():
    with pytest.raises(ContractError):
        PolicyVersion(
            id="pv", type="rule", artifact_sha256="a", dataset_sha256="d",
            feature_schema_version="v", code_commit="c", eval_report_sha256="e",
            status="ACTIVE", approved_by="", activated_at=TS, rollback_to="",
            claim_scope="NONE",
        )


def test_rca_report_is_read_only_and_needs_citations():
    with pytest.raises(ContractError):
        RcaReport(
            id="r", subject_ref="i", evidence_snapshot_hash="h", symptoms=(),
            hypotheses=({"cause": "c", "component": "x", "mechanism": "m",
                         "support_refs": (), "refute_refs": (), "confidence": "low"},),
            chosen_cause="c", confidence_bucket="low", limitations=(), proposed_action="a",
            read_only=True, submitted_at=TS,
        )
    with pytest.raises(ContractError):
        RcaReport(
            id="r", subject_ref="i", evidence_snapshot_hash="h", symptoms=(),
            hypotheses=({"cause": "c", "component": "x", "mechanism": "m",
                         "support_refs": ("s",), "refute_refs": (), "confidence": "low"},),
            chosen_cause="c", confidence_bucket="low", limitations=(), proposed_action="a",
            read_only=False, submitted_at=TS,
        )


def test_wsp_experiment_requires_all_five_controls():
    with pytest.raises(ContractError):
        WspExperiment(
            id="e", pair_id="p", alpha_grid=(), controls=("noop_B",), seed=1,
            eval_manifest_sha256="m", preregistration_sha256="p", status="REGISTERED",
            conclusion="NULL", intervention_class="INCONCLUSIVE", artifact_path="a",
            created_at=TS,
        )


def test_verdict_signing_payload_excludes_the_signature():
    attribution = Attribution(mode="AS_SELF", grant_id="", on_behalf_of="", attribution_ok=True)
    v = OperationalVerdict(
        id="v", run_id="r", checks=(), authorization_ok=True, monitor_integrity_ok=True,
        attribution=attribution, verdict="VERIFIED_SUCCESS", ground_truth_version="v1",
        ts=TS, signature_hmac="deadbeef", verifier_version="w1",
    )
    assert "deadbeef" not in v.signing_payload()
    v2 = OperationalVerdict(**{**v.__dict__, "signature_hmac": "other"})
    assert v2.signing_payload() == v.signing_payload()
    assert v2.sha256() != v.sha256()
