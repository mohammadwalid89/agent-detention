"""Immutable data contracts (design §4) including the additive v3.3 delegation schema.

Every record is append-only at the application layer. `to_dict()` output is what the
hash chain signs, so no float may appear in it (enforced by canonical_json).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar

from . import SCHEMA_VERSION
from .canonical import canonical_json, sha256_text
from .errors import ContractError

KIND_USER = "USER"
KIND_AGENT = "AGENT"
KIND_EXTERNAL = "EXTERNAL_PARTY"
KIND_TOOL = "TOOL"
KIND_DOCUMENT = "DOCUMENT"
PRINCIPAL_KINDS = (KIND_USER, KIND_AGENT, KIND_EXTERNAL, KIND_TOOL, KIND_DOCUMENT)

RISK_TIERS = ("LOW", "MEDIUM", "HIGH")
CONSEQUENTIAL_ACTION_CLASSES = (
    "payment",
    "public_post",
    "external_send",
    "delete",
    "privilege_change",
)
VERDICTS = ("VERIFIED_SUCCESS", "VERIFIED_FAILURE")
RUN_STATES = (
    "CREATED",
    "POLICY_CHECKED",
    "ROUTED",
    "EXECUTING",
    "VERIFYING",
    "VERIFIED_SUCCESS",
    "VERIFIED_FAILURE",
    "UNCERTAIN_EFFECT",
    "NEEDS_APPROVAL",
    "BLOCKED",
)
INTENT_STATES = ("PENDING", "APPLIED", "ABANDONED")
ATTRIBUTION_MODES = ("AS_SELF", "ON_BEHALF_OF")
CONFIDENCE_BUCKETS = ("high", "medium", "low", "insufficient")


class Record:
    """Base record: canonical serialization, stable hash, enum/type validation."""

    SCHEMA: ClassVar[str] = SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = asdict(self)  # type: ignore[arg-type]
        d["_schema"] = self.SCHEMA
        return d

    def canonical(self) -> str:
        return canonical_json(self.to_dict())

    def sha256(self) -> str:
        return sha256_text(self.canonical())

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    def _require(self, name: str, value: Any) -> None:
        if value is None or value == "":
            raise ContractError(f"{type(self).__name__}.{name} is required")

    def _enum(self, name: str, value: Any, allowed: tuple[str, ...]) -> None:
        if value not in allowed:
            raise ContractError(f"{type(self).__name__}.{name}={value!r} not in {allowed}")


# --------------------------------------------------------------------------------------
# Principals, delegation, authority
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Principal(Record):
    id: str
    kind: str
    display_name: str
    credential_ref: str = ""  # pointer to a secret store entry; never a secret

    def __post_init__(self) -> None:
        self._require("id", self.id)
        self._enum("kind", self.kind, PRINCIPAL_KINDS)
        if self.credential_ref and self.credential_ref.startswith(("sk-", "ghp_", "AKIA")):
            raise ContractError("credential_ref looks like a secret; store a pointer only")


@dataclass(frozen=True)
class DelegationGrant(Record):
    id: str
    grantor_principal: str
    grantee_principal: str
    scope: tuple[str, ...]
    resources: tuple[str, ...]
    action_classes: tuple[str, ...]
    issued_at: str
    expires_at: str
    revocation_id: str
    purpose: str
    envelope_signature: str

    def __post_init__(self) -> None:
        for name in ("id", "grantor_principal", "grantee_principal", "issued_at",
                     "expires_at", "revocation_id", "purpose", "envelope_signature"):
            self._require(name, getattr(self, name))
        if not self.scope or not self.resources or not self.action_classes:
            raise ContractError("DelegationGrant requires non-empty scope/resources/action_classes")
        if self.grantor_principal == self.grantee_principal:
            raise ContractError("a principal cannot grant authority to itself")
        if self.issued_at >= self.expires_at:
            raise ContractError("grant expires_at must be after issued_at")

    def consequential(self) -> bool:
        return any(a in CONSEQUENTIAL_ACTION_CLASSES for a in self.action_classes)


@dataclass(frozen=True)
class ApprovalDecision(Record):
    id: str
    run_id: str
    proposal_ref: str
    proposer_principal: str
    approver_principal: str
    grant_id: str
    step_up_ok: bool
    decision: str  # APPROVE | REJECT
    ts: str

    def __post_init__(self) -> None:
        self._enum("decision", self.decision, ("APPROVE", "REJECT"))
        if self.proposer_principal == self.approver_principal:
            raise ContractError("self-approval is structurally forbidden (proposer == approver)")


# --------------------------------------------------------------------------------------
# Task / run / execution
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec(Record):
    id: str
    scenario_family: str
    task_text: str
    allowed_tools: tuple[str, ...]
    risk_tier: str
    budget: dict
    starting_snapshot_ref: str
    hidden_verifier_ref: str
    manifest_version: str
    created_at: str
    principal_ref: str = ""
    on_behalf_of: str = ""
    delegation_id: str = ""
    delegation_required: bool = False
    attribution_mode: str = "AS_SELF"

    def __post_init__(self) -> None:
        self._require("id", self.id)
        self._enum("risk_tier", self.risk_tier, RISK_TIERS)
        self._enum("attribution_mode", self.attribution_mode, ATTRIBUTION_MODES)
        for k in ("max_calls", "max_usd_micros", "max_seconds"):
            if k not in self.budget:
                raise ContractError(f"TaskSpec.budget missing {k!r}")
        if isinstance(self.budget.get("max_usd_micros"), float):
            raise ContractError("budget.max_usd_micros must be an int (micro-USD)")
        if self.delegation_required and not self.delegation_id:
            raise ContractError("delegation_required=true requires delegation_id")
        if self.attribution_mode == "ON_BEHALF_OF" and not self.on_behalf_of:
            raise ContractError("ON_BEHALF_OF requires on_behalf_of principal")


@dataclass(frozen=True)
class RunTransition(Record):
    run_id: str
    seq: int
    from_state: str
    to_state: str
    reason: str
    ts: str
    actor: str

    def __post_init__(self) -> None:
        self._enum("from_state", self.from_state, RUN_STATES)
        self._enum("to_state", self.to_state, RUN_STATES)
        if self.seq < 0:
            raise ContractError("seq must be >= 0")


@dataclass(frozen=True)
class RoutingDecision(Record):
    id: str
    run_id: str
    decision_ts: str
    policy_id: str
    catalog_version: str
    feature_snapshot: dict
    feature_schema_version: str
    eligible_actions: tuple[str, ...]
    selected_action: str
    selection_probability: str  # decimal string
    selection_reason: str
    budget: dict

    FORBIDDEN_FEATURES: ClassVar[tuple[str, ...]] = (
        "starting_snapshot_hash",
        "scenario_id",
        "seed",
        "fault_id",
        "verdict",
        "delegation_id",
        "grant_id",
    )

    def __post_init__(self) -> None:
        leaked = set(self.feature_snapshot) & set(self.FORBIDDEN_FEATURES)
        if leaked:
            raise ContractError(f"router feature snapshot contains forbidden keys: {sorted(leaked)}")
        if self.selected_action not in self.eligible_actions:
            raise ContractError("selected_action must be one of eligible_actions")


@dataclass(frozen=True)
class Authority(Record):
    source: str  # signed_task_envelope | delegation_grant | untrusted
    scope: tuple[str, ...]
    grant_id: str = ""
    on_behalf_of: str = ""

    def __post_init__(self) -> None:
        if self.source == "untrusted" and self.scope:
            raise ContractError("untrusted content cannot carry a scope")


@dataclass(frozen=True)
class ToolEvent(Record):
    id: str
    run_id: str
    trace_id: str
    span_id: str
    tool_name: str
    side_effect_class: str  # READ | WRITE | EXTERNAL
    idempotency_key: str
    request_hash: str
    response_status: int
    error_type: str
    duration_ms: int
    outcome_uncertain: bool
    authority: Authority
    taint: str  # trusted | untrusted
    redaction_version: str
    ts: str

    def __post_init__(self) -> None:
        if self.side_effect_class == "WRITE" and not self.idempotency_key:
            raise ContractError("a WRITE tool event requires an idempotency_key")
        if self.taint not in ("trusted", "untrusted"):
            raise ContractError("taint must be trusted|untrusted")


@dataclass(frozen=True)
class WriteIntent(Record):
    id: str
    run_id: str
    idempotency_key: str
    payload_hash: str
    state: str
    ts: str
    grant_id: str = ""
    on_behalf_of: str = ""

    def __post_init__(self) -> None:
        self._enum("state", self.state, INTENT_STATES)
        self._require("idempotency_key", self.idempotency_key)


@dataclass(frozen=True)
class ObservedFinalState(Record):
    id: str
    run_id: str
    snapshot_hash: str
    read_at: str
    reader_version: str
    tables_digest: dict
    row_counts: dict
    invariants_ok: bool

    def __post_init__(self) -> None:
        self._require("snapshot_hash", self.snapshot_hash)
        for k, v in self.tables_digest.items():
            if not isinstance(v, str):
                raise ContractError(f"tables_digest[{k!r}] must be a hex string")


@dataclass(frozen=True)
class Attribution(Record):
    mode: str
    grant_id: str
    on_behalf_of: str
    attribution_ok: bool

    def __post_init__(self) -> None:
        self._enum("mode", self.mode, ATTRIBUTION_MODES)


@dataclass(frozen=True)
class OperationalVerdict(Record):
    id: str
    run_id: str
    checks: tuple[dict, ...]
    authorization_ok: bool
    monitor_integrity_ok: bool
    attribution: Attribution
    verdict: str
    ground_truth_version: str
    ts: str
    signature_hmac: str
    verifier_version: str

    def __post_init__(self) -> None:
        self._enum("verdict", self.verdict, VERDICTS)
        for c in self.checks:
            for k in ("name", "expected", "actual", "pass"):
                if k not in c:
                    raise ContractError(f"verdict check missing {k!r}")
            if not isinstance(c["pass"], bool):
                raise ContractError("check.pass must be a bool")

    def signing_payload(self) -> str:
        return canonical_json(
            {
                "run_id": self.run_id,
                "checks": [dict(c) for c in self.checks],
                "authorization_ok": self.authorization_ok,
                "monitor_integrity_ok": self.monitor_integrity_ok,
                "attribution": asdict(self.attribution),
                "verdict": self.verdict,
                "ground_truth_version": self.ground_truth_version,
                "ts": self.ts,
                "verifier_version": self.verifier_version,
            }
        )


# --------------------------------------------------------------------------------------
# RCA / audit / policy / WSP
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RcaReport(Record):
    id: str
    subject_ref: str
    evidence_snapshot_hash: str
    symptoms: tuple[str, ...]
    hypotheses: tuple[dict, ...]
    chosen_cause: str
    confidence_bucket: str
    limitations: tuple[str, ...]
    proposed_action: str
    read_only: bool
    submitted_at: str

    def __post_init__(self) -> None:
        self._enum("confidence_bucket", self.confidence_bucket, CONFIDENCE_BUCKETS)
        if not self.read_only:
            raise ContractError("release 1 RCA is read-only")
        # An abstention must not name a cause; a confident report must state at least one
        # hypothesis. This keeps "insufficient evidence" from being a cosmetic label on a
        # report that actually asserts something, and stops empty non-abstaining reports.
        if self.confidence_bucket == "insufficient":
            if self.chosen_cause:
                raise ContractError("an abstention (confidence_bucket=insufficient) must not name a cause")
        elif not self.hypotheses:
            raise ContractError("a non-abstaining RCA report must state at least one hypothesis")
        for h in self.hypotheses:
            for k in ("cause", "component", "mechanism", "support_refs", "refute_refs", "confidence"):
                if k not in h:
                    raise ContractError(f"hypothesis missing {k!r}")
            if not h["support_refs"] and not h["refute_refs"]:
                raise ContractError("hypothesis must cite at least one support or refute reference")


@dataclass(frozen=True)
class AuditVerdict(Record):
    id: str
    subject_ref: str
    rubric_version: str
    component_score: str
    mechanism_score: str
    evidence_score: str
    safety_score: str
    abstention_correct: bool
    verdict: str
    grader_version: str
    ts: str

    def __post_init__(self) -> None:
        self._enum("verdict", self.verdict, ("PASS", "PARTIAL", "FAIL"))


@dataclass(frozen=True)
class PolicyVersion(Record):
    id: str
    type: str  # static | rule | learned | wsp
    artifact_sha256: str
    dataset_sha256: str
    feature_schema_version: str
    code_commit: str
    eval_report_sha256: str
    status: str  # DRAFT | SHADOW_TESTED | EVALUATED | APPROVED | ACTIVE | REJECTED | INCONCLUSIVE | ROLLED_BACK
    approved_by: str
    activated_at: str
    rollback_to: str
    claim_scope: str

    def __post_init__(self) -> None:
        self._enum("type", self.type, ("static", "rule", "learned", "wsp"))
        self._enum(
            "status",
            self.status,
            ("DRAFT", "SHADOW_TESTED", "EVALUATED", "APPROVED", "ACTIVE",
             "REJECTED", "INCONCLUSIVE", "ROLLED_BACK"),
        )
        self._enum("claim_scope", self.claim_scope, ("NONE", "TRANSFER_PROBE"))
        if self.status == "ACTIVE" and not self.approved_by:
            raise ContractError("an ACTIVE policy requires approved_by (no auto-promotion)")
        if self.type == "learned" and self.status == "ACTIVE":
            raise ContractError(
                "learned routing policy cannot be ACTIVE in v1 (8 families < 10-family bar; SHADOW only)"
            )


@dataclass(frozen=True)
class WspCheckpointPair(Record):
    id: str
    family: str
    base_model_id: str
    checkpoint_a_sha256: str
    checkpoint_b_sha256: str
    state_dict_keys_hash: str
    param_count: int
    dtype: str
    init_seed: int
    train_data_sha256: str
    objective_a: dict
    objective_b: dict
    feature_schema_version: str
    created_at: str

    def __post_init__(self) -> None:
        self._enum("family", self.family, ("ROUTER", "ADAPTER"))
        if self.family == "ROUTER" and self.param_count < 1000:
            raise ContractError("router checkpoint pair param_count looks wrong (<1000)")
        if self.objective_a == self.objective_b:
            raise ContractError("the two checkpoints must differ in objective")


@dataclass(frozen=True)
class WspExperiment(Record):
    id: str
    pair_id: str
    alpha_grid: tuple[str, ...]
    controls: tuple[str, ...]
    seed: int
    eval_manifest_sha256: str
    preregistration_sha256: str
    status: str
    conclusion: str
    intervention_class: str
    artifact_path: str
    created_at: str

    def __post_init__(self) -> None:
        self._enum(
            "conclusion",
            self.conclusion,
            ("POSITIVE", "NULL", "NEGATIVE", "INCONCLUSIVE", "NOT_RUN_RESOURCE_GATE"),
        )
        self._enum(
            "intervention_class",
            self.intervention_class,
            ("PARAMETRIC", "THRESHOLD_EQUIVALENT", "INCONCLUSIVE", "NOT_APPLICABLE"),
        )
        required = {"noop_B", "random_norm_matched", "shuffled_delta", "threshold_only", "direct_tuning"}
        missing = required - set(self.controls)
        if missing:
            raise ContractError(f"WSP experiment missing controls: {sorted(missing)}")


@dataclass(frozen=True)
class WspCandidateEvaluation(Record):
    id: str
    experiment_id: str
    candidate_kind: str
    alpha: str
    metrics: dict
    delta_norms: dict
    changed_param_pct: str
    mean_logit_shift: str
    n_cases: int
    n_repeats: int
    verdict: str

    def __post_init__(self) -> None:
        if self.n_cases <= 0 or self.n_repeats <= 0:
            raise ContractError("n_cases and n_repeats must be positive")


@dataclass(frozen=True)
class WspdEnvironment(Record):
    id: str
    env_family: str
    env_version: str
    generator_sha256: str
    oracle_sha256: str
    parser_sha256: str
    dev_manifest_sha256: str
    val_manifest_sha256: str
    sealed_manifest_sha256: str
    feature_list: tuple[str, ...]
    param_layout_sha256: str
    leak_audit_verdict: str
    discrimination_margin: str
    created_at: str

    def __post_init__(self) -> None:
        self._enum("env_family", self.env_family, ("GRID_KEY", "QUEUE_TWIN", "AUTH_TRIAGE"))
        self._enum(
            "leak_audit_verdict",
            self.leak_audit_verdict,
            ("PASS", "TRIVIAL_FEATURE_LEAK", "LEAK_UNRESOLVED", "NOT_RUN"),
        )


RECORD_TYPES = {
    cls.__name__: cls
    for cls in (
        Principal,
        DelegationGrant,
        ApprovalDecision,
        TaskSpec,
        RunTransition,
        RoutingDecision,
        Authority,
        ToolEvent,
        WriteIntent,
        ObservedFinalState,
        Attribution,
        OperationalVerdict,
        RcaReport,
        AuditVerdict,
        PolicyVersion,
        WspCheckpointPair,
        WspExperiment,
        WspCandidateEvaluation,
        WspdEnvironment,
    )
}
