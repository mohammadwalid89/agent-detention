"""Tier 1: AgentAbstain (MIT code; dataset CC-BY-4.0).

index    the 263 act/abstain pair index (read from the clone's own task list)
loader   one pair from the downloaded dataset snapshot (needs ``AGENTABSTAIN_DATA``)
adapter  one pair -> our sibling record (mechanism style; variant A act-correct, B stop-correct)
check    the end-state check (commit / no commit) that mirrors AgentAbstain's own commit check
controls scripted controls (always-act / always-stop / random) run through the real harness
"""

SCENARIOS = (
    "ambiguous_action_specification",
    "conflicting_constraints",
    "conflicting_evidence",
    "critical_tool_failure",
    "emergent_risk_discovery",
    "high_stakes_action",
    "insufficient_tool_capability",
    "missing_critical_parameter",
)
