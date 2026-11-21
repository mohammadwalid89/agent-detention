"""VERA v1.2 delegation core (JOB19 core, JOB20 build part 2).

A new package: nothing here changes v1 pilot behaviour, readiness v2/v3, the mechanism
manifests or the pilot runner.  Entry points:

``vera.delegation.conditions``
    pure task-condition derivation from active set 10;
``vera.delegation.plan``
    cell construction, owner-issued attenuating tokens, peer menu;
``vera.delegation.world``
    one shared world, the ``delegate`` tool and signed episode records;
``vera.delegation.metrics``
    §4 metrics from signed logs;
``vera.delegation.readiness``
    ``delegation_readiness_v1`` D1-D6;
``vera.delegation.adapters``
    the mock, NVIDIA-hosted and Anthropic API adapters (the gated paid-adapter
    interface and the command-line adapters were removed before publication);
``vera.delegation.detention`` / ``heldback``
    Detention arm plumbing and the offline HB1-HB4 auditor;
``vera.delegation.e2e``
    the offline 3-mechanism x 4-condition run;
``vera.delegation.telemetry``
    per-episode JSONL and dashboard summary tables;
``vera.delegation.runplan``
    the unfrozen v1.2 run-plan draft, call counts and design sensitivity;
``vera.delegation.identity_probe``
    the A6 self-identity probe;
``vera.delegation.sidecar``
    the private Detention side-car draft and its independent-review gate;
``vera.delegation.providers``
    JOB23 paid-provider adapters (Anthropic Messages, OpenAI Responses/Chat, Gemini),
    a price-table-driven cost accounting layer, the hard paid gate and a running meter.
    Every adapter refuses without the full gate and no adapter dials a network implicitly.

In the published tree ``readiness``, ``e2e``, ``telemetry``, ``runplan``, ``identity_probe``,
``sidecar`` and ``providers`` are absent; the list above is the package as it was built
(see ``harness/README.md``).
"""

from __future__ import annotations

from .constants import (  # noqa: F401
    ABSTAINED_INSUFFICIENT,
    COND_ABSTENTION,
    COND_DODGE,
    COND_NECESSARY,
    COND_OPTIONAL,
    CONDITIONS,
    CONTAINED_ONLY,
    CUE_EXPLICIT,
    CUE_IMPLICIT,
    DELEGATED_REPAIRED_VERIFIED,
    DELEGATION_HARNESS_VERSION,
    DELEGATION_RUBRIC_VERSION,
    DODGE_CUES,
    FAILED_BUCK_PASSED,
    FAILED_INCORRECT,
    FAILED_UNSAFE,
    HANDED_OFF_VALID,
    IDENTITY_CONDITIONS,
    INVALID_HARNESS,
    ORCHESTRATOR_REPAIRED_VERIFIED,
    PRIMARY_DODGE_CUE,
    RCA_DETENTION_VERSION,
)

__all__ = [
    "ABSTAINED_INSUFFICIENT",
    "COND_ABSTENTION",
    "COND_DODGE",
    "COND_NECESSARY",
    "COND_OPTIONAL",
    "CONDITIONS",
    "CONTAINED_ONLY",
    "CUE_EXPLICIT",
    "CUE_IMPLICIT",
    "DODGE_CUES",
    "PRIMARY_DODGE_CUE",
    "DELEGATED_REPAIRED_VERIFIED",
    "DELEGATION_HARNESS_VERSION",
    "DELEGATION_RUBRIC_VERSION",
    "FAILED_BUCK_PASSED",
    "FAILED_INCORRECT",
    "FAILED_UNSAFE",
    "HANDED_OFF_VALID",
    "IDENTITY_CONDITIONS",
    "INVALID_HARNESS",
    "ORCHESTRATOR_REPAIRED_VERIFIED",
    "RCA_DETENTION_VERSION",
]

__version__ = "v1.2-build3-job23"
