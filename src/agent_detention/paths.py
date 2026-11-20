"""Where inputs and generated files live. Everything comes from environment variables.

AD_WORK              directory for generated records, overlays and reports (default ./work)
AD_TREE              the harness tree (the VERA code the runner executes); required by anything that runs episodes
AGENTABSTAIN_DATA    clone of the AgentAbstain dataset (CC-BY-4.0)
AGENTABSTAIN_REPO    clone of the AgentAbstain code (MIT), used for the pair index
TAU2_ROOT            clone of tau2-bench (MIT)
CRM_BENCH_ROOT       clone of crm-agent-bench (Apache-2.0)
"""

from __future__ import annotations

import os
from pathlib import Path


def env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"set {name} (see scripts/fetch_data.py)")
    return Path(value)


def work() -> Path:
    return Path(os.environ.get("AD_WORK", "work"))


def tier1_records() -> Path:
    return work() / "tier1" / "tier1_set.json"


def tier2_out() -> Path:
    return work() / "tier2"


def reports() -> Path:
    return work() / "reports"


def tau2_root() -> Path:
    return env_path("TAU2_ROOT")


def crm_root() -> Path:
    return env_path("CRM_BENCH_ROOT")
