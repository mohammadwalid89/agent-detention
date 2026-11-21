"""Organisation-pattern factor for the v1.2 delegation harness (JOB21, OPTIONAL).

Selecting a pattern is an explicit opt-in: ``pattern=None`` (or ``P0_BASE``) is the base
v1.2 design, byte for byte.  The package holds

``contract``
    the written ownership contract per pattern, with the source whose *structure* is
    replicated (never its numbers);
``config``
    the frozen harness configuration the world and the cell hash read;
``rules``
    pattern-relative illegitimate-handoff / buck-passing and accountability predicates over
    signed facts;
``metrics``
    the per-pattern metric table with the base-rule yardstick kept separate;
``mocks``
    deterministic offline policies that exercise each contract's plumbing.

Nothing here is a result about any model.  No module in this package calls a model, reads a
key or touches the holdout.
"""

from __future__ import annotations

from ..constants import (  # noqa: F401
    DEFAULT_PATTERN,
    IH_BOUNCE_NO_PROGRESS,
    IH_DELEGATED_OWN_AUTHORITY,
    IH_HOP_CAP_EXCEEDED,
    IH_HOP_NOT_PERMITTED,
    IH_LEDGER_CONTRADICTS_VERIFIER,
    IH_LEDGER_STALL_UNRECORDED,
    IH_LEDGER_UNENROLLED,
    IH_OWNERSHIP_DISCLAIMED,
    IH_RECEIVER_LACKS_AUTHORITY,
    IH_RESULT_LAUNDERED,
    IH_SUBGOAL_OUTSIDE_GRANT,
    IH_SUBGOAL_UNBOUNDED,
    ILLEGITIMATE_HANDOFF_RULES,
    LEDGER_PROGRESS,
    LEDGER_TASK,
    PATTERN_BASE,
    PATTERN_LEVELS,
    PATTERN_OPTIONAL_LEVELS,
    PATTERN_P1,
    PATTERN_P2,
    PATTERN_P3,
    PATTERN_P4,
)
from .config import (  # noqa: F401
    BASE_CONFIG,
    CONFIGS,
    P1_CONFIG,
    P2_CONFIG,
    P3_CONFIG,
    P4_CONFIG,
    PatternConfig,
    config_digest,
    config_for,
    configs_manifest,
)
from .contract import (  # noqa: F401
    BASE_CONTRACT,
    CONTRACT_DOC,
    CONTRACTS,
    P1_CONTRACT,
    P2_CONTRACT,
    P3_CONTRACT,
    P4_CONTRACT,
    P5_NOT_PROPOSED,
    OwnershipContract,
    contract_for,
    contracts_manifest,
)
from .metrics import format_pattern_table, pattern_section  # noqa: F401
from .rules import (  # noqa: F401
    accountability,
    base_rule_buck_passed,
    classify_with_pattern,
    illegitimate_handoff,
)

__all__ = [
    "BASE_CONFIG",
    "BASE_CONTRACT",
    "CONFIGS",
    "CONTRACTS",
    "CONTRACT_DOC",
    "DEFAULT_PATTERN",
    "ILLEGITIMATE_HANDOFF_RULES",
    "OwnershipContract",
    "P1_CONFIG",
    "P1_CONTRACT",
    "P2_CONFIG",
    "P2_CONTRACT",
    "P3_CONFIG",
    "P3_CONTRACT",
    "P4_CONFIG",
    "P4_CONTRACT",
    "P5_NOT_PROPOSED",
    "PATTERN_BASE",
    "PATTERN_LEVELS",
    "PATTERN_OPTIONAL_LEVELS",
    "PATTERN_P1",
    "PATTERN_P2",
    "PATTERN_P3",
    "PATTERN_P4",
    "PatternConfig",
    "accountability",
    "base_rule_buck_passed",
    "classify_with_pattern",
    "config_digest",
    "config_for",
    "configs_manifest",
    "contract_for",
    "contracts_manifest",
    "format_pattern_table",
    "illegitimate_handoff",
    "pattern_section",
]

PATTERN_PACKAGE_VERSION = "v1.2-pattern-1"
