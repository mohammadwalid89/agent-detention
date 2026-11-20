from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vera.contracts import (  # noqa: E402
    ApprovalDecision,
    DelegationGrant,
    Principal,
    TaskSpec,
)
from vera.db import connect, create_schema  # noqa: E402
from vera.sandbox.service import SandboxService  # noqa: E402

TS = "2026-11-13T01:00:00Z"


def pytest_addoption(parser) -> None:
    # order-dependence probes that do not need pytest-randomly.
    parser.addoption(
        "--reverse",
        action="store_true",
        default=False,
        help="run collected tests in reverse order",
    )
    parser.addoption(
        "--shuffle-seed",
        action="store",
        default=None,
        metavar="SEED",
        help="run collected tests in a deterministic shuffled order",
    )


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--reverse"):
        items.reverse()
    seed = config.getoption("--shuffle-seed")
    if seed is not None:
        random.Random(str(seed)).shuffle(items)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "vera.db"
    create_schema(path)
    return path


@pytest.fixture()
def api(db_path: Path):
    conn = connect(db_path, "api_rw")
    yield conn
    conn.close()


@pytest.fixture()
def verifier_conn(db_path: Path):
    conn = connect(db_path, "verifier_ro")
    yield conn
    conn.close()


@pytest.fixture()
def grader_conn(db_path: Path):
    conn = connect(db_path, "grader_ro")
    yield conn
    conn.close()


@pytest.fixture()
def service(db_path: Path) -> SandboxService:
    svc = SandboxService(str(db_path))
    svc.seed()
    return svc


@pytest.fixture()
def case_principal() -> Principal:
    return Principal(id="user_alice", kind="USER", display_name="Alice")


@pytest.fixture()
def agent_principal() -> Principal:
    return Principal(id="agent_vera", kind="AGENT", display_name="VERA")


@pytest.fixture()
def grant() -> DelegationGrant:
    return DelegationGrant(
        id="grant_1",
        grantor_principal="user_alice",
        grantee_principal="agent_vera",
        scope=("tickets:write",),
        resources=("ticket:T-*",),
        action_classes=("ticket_update",),
        issued_at="2026-11-13T00:00:00Z",
        expires_at="2026-11-14T00:00:00Z",
        revocation_id="rev_1",
        purpose="close resolved tickets",
        envelope_signature="sig_test",
    )


@pytest.fixture()
def task(grant) -> TaskSpec:
    return TaskSpec(
        id="task_1",
        scenario_family="F1_ticket_record_update",
        task_text="close T-100 after verifying the fix",
        allowed_tools=("read_ticket", "update_ticket"),
        risk_tier="LOW",
        budget={"max_calls": 4, "max_usd_micros": 0, "max_seconds": 60},
        starting_snapshot_ref="snap_start",
        hidden_verifier_ref="verifier://f1-m01",
        manifest_version="seed-dev-v1",
        created_at=TS,
        principal_ref="agent_vera",
        on_behalf_of="user_alice",
        delegation_id=grant.id,
        delegation_required=True,
        attribution_mode="ON_BEHALF_OF",
    )


@pytest.fixture()
def approval() -> ApprovalDecision:
    return ApprovalDecision(
        id="appr_1", run_id="run_1", proposal_ref="prop_1",
        proposer_principal="agent_vera", approver_principal="user_alice",
        grant_id="grant_1", step_up_ok=True, decision="APPROVE",
        ts="2026-11-13T00:59:30Z",
    )
