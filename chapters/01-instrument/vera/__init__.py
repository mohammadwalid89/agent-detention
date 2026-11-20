"""VERA — verified execution, recovery and adaptation (W1 deterministic core)."""

__version__ = "0.1.0-w1"
SCHEMA_VERSION = "3.3-delegation"
FEATURE_SCHEMA_VERSION = "router-features-v3.3"

# Benchmark population constants (v3.3, binding). No code may hard-code a v3.1 count.
FAMILIES = (
    "F1_ticket_record_update",
    "F2_authorization_approvals",
    "F3_reconciliation_conflict",
    "F4_schema_data_quality",
    "F5_dependency_timeout",
    "F6_prompt_injection_taint",
    "F7_multistep_transaction",
    "F8_incident_triage_escalation",
)
MECHANISMS_PER_FAMILY = 8
VARIANTS_PER_MECHANISM = 2
PILOT_MECHANISMS = 24
# Sealed holdout families are authored by a separate author; the builder may not
# author, read, or count them as dev mechanisms.
HOLDOUT_FAMILIES = ("F5_dependency_timeout", "F8_incident_triage_escalation")
DEV_FAMILIES = tuple(f for f in FAMILIES if f not in HOLDOUT_FAMILIES)
ADVERSARIAL_OVERLAYS = (
    "SYNTHETIC_DEPENDENCY",
    "AMBIGUOUS_AFTER_COMMIT",
    "METRIC_GAMING",
    "TAMPERED_EVIDENCE",
    "INCOMPLETE_TASK",
    "LEAKAGE_SENTINEL",
    "COMPLY_01",
)
WSPD_ENVS = ("GRID_KEY", "QUEUE_TWIN", "AUTH_TRIAGE")
WSPD_POLICIES_PER_ENV = 8
ROUTER_FEATURES = (
    "task_family",
    "n_required_tools",
    "has_write",
    "schema_complexity",
    "input_tokens_bucket",
    "risk_tier",
    "budget_tier",
    "model1_available",
    "model2_available",
    "historical_success_rate_model1",
    "historical_success_rate_model2",
)
PROMOTION_MIN_FAMILIES = 10
PROMOTION_MIN_CONTEXTS = 60
