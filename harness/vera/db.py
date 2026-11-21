"""SQLite persistence with per-identity authorizer, insert-only tables and hash chains.

W1 boundary note (design §5.2): role separation is enforced by a DB authorizer on a
separate connection per identity, plus insert-only triggers. That is a
PROCESS_BOUNDARY. It is NOT the W2 container/uid boundary and must never be
described as one.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .canonical import HashChain, canonical_json, chain_hash
from .contracts import (
    ApprovalDecision,
    AuditVerdict,
    DelegationGrant,
    ObservedFinalState,
    OperationalVerdict,
    PolicyVersion,
    Principal,
    RcaReport,
    RoutingDecision,
    RunTransition,
    TaskSpec,
    ToolEvent,
    WriteIntent,
)
from .errors import RoleViolation, VeraError

RECORD_TABLES: dict[str, type] = {
    "principal": Principal,
    "delegation_grant": DelegationGrant,
    "approval_decision": ApprovalDecision,
    "task": TaskSpec,
    "run_transition": RunTransition,
    "routing_decision": RoutingDecision,
    "tool_event": ToolEvent,
    "write_intent": WriteIntent,
    "observed_final_state": ObservedFinalState,
    "verification_event": OperationalVerdict,
    "rca_report": RcaReport,
    "audit_verdict": AuditVerdict,
    "policy_version": PolicyVersion,
}

DOMAIN_TABLES = frozenset(
    {
        "run", "ticket", "alert", "job", "queue_job", "service_config",
        "change_log", "cache", "effects", "grant_revocation", "span", "log", "metric",
    }
)

# Tables no role may UPDATE or DELETE (append-only by construction).
# write_intent is append-then-transition: its payload is frozen by a trigger and only
# `state` may change, so it is not in the blanket-immutable set.
IMMUTABLE_TABLES = frozenset(
    (set(RECORD_TABLES) - {"write_intent"})
    | {"effects", "chain", "change_log", "span", "log", "metric"}
)

# Tables each role may INSERT into. Anything else is denied by the authorizer.
WRITE_TABLES: dict[str, frozenset[str]] = {
    "api_rw": frozenset(
        {
            "principal", "delegation_grant", "approval_decision", "task",
            "run_transition", "routing_decision", "tool_event", "write_intent",
            "observed_final_state", "rca_report", "effects", "run", "ticket",
            "alert", "job", "queue_job", "service_config", "change_log", "cache",
            "grant_revocation", "span", "log", "metric",
        }
    ),
    "verifier_ro": frozenset({"verification_event"}),
    "grader_ro": frozenset({"audit_verdict"}),
    "wsp_ro": frozenset(),
    "admin": frozenset(RECORD_TABLES)
    | frozenset(DOMAIN_TABLES)
    | frozenset({"service_config"}),
}

# sqlite3.Connection does not allow attribute assignment or weak references, so the role
# is tracked by connection identity for clear pre-check errors (the authorizer remains the
# actual enforcement point).
_CONN_ROLE: dict[sqlite3.Connection, str] = {}

_TABLE_WRITERS: dict[str, frozenset[str]] = {}
for _role, _tables in WRITE_TABLES.items():
    for _t in _tables:
        _TABLE_WRITERS[_t] = _TABLE_WRITERS.get(_t, frozenset()) | {_role}

# Read restrictions (the learner/router identities must not read outcomes).
READ_DENIED: dict[str, frozenset[str]] = {
    "wsp_ro": frozenset({"verification_event", "audit_verdict", "delegation_grant", "approval_decision"}),
    "api_rw": frozenset({"audit_verdict"}),
    "verifier_ro": frozenset({"audit_verdict"}),
}

DOMAIN_DDL = """
CREATE TABLE IF NOT EXISTS run (
    run_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, state TEXT NOT NULL,
    policy_id TEXT NOT NULL DEFAULT '', grant_id TEXT NOT NULL DEFAULT '',
    on_behalf_of TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ticket (
    ticket_id TEXT PRIMARY KEY, status TEXT NOT NULL, severity TEXT NOT NULL DEFAULT 'low',
    assignee TEXT NOT NULL DEFAULT '', resolution TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS alert (
    alert_id TEXT PRIMARY KEY, state TEXT NOT NULL, rule_hash TEXT NOT NULL,
    silenced_by TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS job (
    job_id TEXT PRIMARY KEY, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS queue_job (
    job_id TEXT PRIMARY KEY, queue TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0,
    deadline_step INTEGER NOT NULL DEFAULT 0, served_at INTEGER NOT NULL DEFAULT -1,
    arrived_at INTEGER NOT NULL DEFAULT 0, dropped INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS service_config (
    key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL,
    updated_by TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS change_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, change_id TEXT NOT NULL, key TEXT NOT NULL,
    old_value TEXT NOT NULL, new_value TEXT NOT NULL, ts TEXT NOT NULL, actor TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS span (
    span_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL, run_id TEXT NOT NULL,
    tool TEXT NOT NULL, status INTEGER NOT NULL, ts TEXT NOT NULL, duration_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS log (
    log_id TEXT PRIMARY KEY, span_id TEXT NOT NULL, level TEXT NOT NULL,
    message TEXT NOT NULL, ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metric (
    metric_id TEXT PRIMARY KEY, name TEXT NOT NULL, value_pm INTEGER NOT NULL, ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS grant_revocation (
    revocation_id TEXT PRIMARY KEY, grant_id TEXT NOT NULL, ts TEXT NOT NULL,
    actor TEXT NOT NULL, reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cache (
    cache_key TEXT PRIMARY KEY, value TEXT NOT NULL, written_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS effects (
    id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL UNIQUE, run_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL, effect_ref TEXT NOT NULL, ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chain (
    table_name TEXT NOT NULL, seq INTEGER NOT NULL, prev_hash TEXT NOT NULL,
    row_hash TEXT NOT NULL, ts TEXT NOT NULL, PRIMARY KEY (table_name, seq)
);
"""


def table_ddl(name: str) -> str:
    """Every record table stores the canonical payload; columns are derived, not authoritative."""
    return (
        f"CREATE TABLE IF NOT EXISTS {name} ("
        "seq INTEGER PRIMARY KEY, prev_hash TEXT NOT NULL, row_hash TEXT NOT NULL, "
        "record_json TEXT NOT NULL)"
    )


def _authorizer(role: str):
    def hook(action: int, arg1: str | None, arg2: str | None, dbname: str | None, source: str | None):
        table = (arg1 or "").lower()
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ):
            if table in READ_DENIED.get(role, frozenset()):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_INSERT:
            if table == "chain":
                return sqlite3.SQLITE_OK if role in WRITE_TABLES and role != "wsp_ro" else sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK if role in _TABLE_WRITERS.get(table, frozenset()) else sqlite3.SQLITE_DENY
        if action in (sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE):
            if table in IMMUTABLE_TABLES:
                return sqlite3.SQLITE_DENY
            if table == "write_intent":
                return sqlite3.SQLITE_OK if role in ("api_rw", "admin") else sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK if role in ("api_rw", "admin") else sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    return hook


def connect(path: str | Path, role: str, *, setup: bool = False) -> sqlite3.Connection:
    if role not in WRITE_TABLES:
        raise RoleViolation(f"unknown role {role!r}")
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    if not setup:
        conn.set_authorizer(_authorizer(role))
    _CONN_ROLE[conn] = role
    return conn


def create_schema(path: str | Path) -> None:
    conn = connect(path, "admin", setup=True)
    try:
        for name in RECORD_TABLES:
            conn.execute(table_ddl(name))
        conn.executescript(DOMAIN_DDL)
        for name in sorted(set(RECORD_TABLES) - {"write_intent"}):
            conn.execute(
                f"CREATE TRIGGER IF NOT EXISTS {name}_no_update BEFORE UPDATE ON {name} "
                f"BEGIN SELECT RAISE(ABORT, '{name} is insert-only'); END"
            )
        for name in sorted(set(RECORD_TABLES) - {"write_intent"}):
            conn.execute(
                f"CREATE TRIGGER IF NOT EXISTS {name}_no_delete BEFORE DELETE ON {name} "
                f"BEGIN SELECT RAISE(ABORT, '{name} is insert-only'); END"
            )
        conn.execute(
            "CREATE TRIGGER IF NOT EXISTS write_intent_state_only BEFORE UPDATE ON write_intent "
            "BEGIN SELECT CASE WHEN "
            "json_extract(NEW.record_json,'$.id') <> json_extract(OLD.record_json,'$.id') "
            "OR json_extract(NEW.record_json,'$.run_id') <> json_extract(OLD.record_json,'$.run_id') "
            "OR json_extract(NEW.record_json,'$.idempotency_key') <> json_extract(OLD.record_json,'$.idempotency_key') "
            "OR json_extract(NEW.record_json,'$.payload_hash') <> json_extract(OLD.record_json,'$.payload_hash') "
            "OR json_extract(NEW.record_json,'$.grant_id') <> json_extract(OLD.record_json,'$.grant_id') "
            "OR json_extract(NEW.record_json,'$.on_behalf_of') <> json_extract(OLD.record_json,'$.on_behalf_of') "
            "THEN RAISE(ABORT, 'write_intent: only state may change') END; END"
        )
        conn.execute(
            "CREATE TRIGGER IF NOT EXISTS write_intent_no_delete BEFORE DELETE ON write_intent "
            "BEGIN SELECT RAISE(ABORT, 'write_intent is never deleted'); END"
        )
        for name in ("effects", "change_log", "span", "log", "metric"):
            conn.execute(
                f"CREATE TRIGGER IF NOT EXISTS {name}_no_update BEFORE UPDATE ON {name} "
                f"BEGIN SELECT RAISE(ABORT, '{name} is insert-only'); END"
            )
            conn.execute(
                f"CREATE TRIGGER IF NOT EXISTS {name}_no_delete BEFORE DELETE ON {name} "
                f"BEGIN SELECT RAISE(ABORT, '{name} is insert-only'); END"
            )
        conn.execute(
            "CREATE TRIGGER IF NOT EXISTS chain_no_update BEFORE UPDATE ON chain "
            "BEGIN SELECT RAISE(ABORT, 'chain is insert-only'); END"
        )
        conn.execute(
            "CREATE TRIGGER IF NOT EXISTS chain_no_delete BEFORE DELETE ON chain "
            "BEGIN SELECT RAISE(ABORT, 'chain is insert-only'); END"
        )
    finally:
        conn.close()


def _chain_head(conn: sqlite3.Connection, table: str) -> tuple[int, str]:
    row = conn.execute(
        "SELECT seq, row_hash FROM chain WHERE table_name=? ORDER BY seq DESC LIMIT 1", (table,)
    ).fetchone()
    if row is None:
        return 0, HashChain.GENESIS
    return int(row["seq"]), str(row["row_hash"])


def insert_record(conn: sqlite3.Connection, table: str, record: Any, *, ts: str = "") -> str:
    """Append a record and extend its hash chain. Raises RoleViolation if the identity may not write."""
    if table not in RECORD_TABLES:
        raise VeraError(f"unknown record table {table!r}")
    role = _CONN_ROLE.get(conn, "admin")
    if role not in _TABLE_WRITERS.get(table, frozenset()):
        raise RoleViolation(f"{table}: identity {role!r} may not insert")
    payload = record.to_dict()
    payload_json = canonical_json(payload)
    seq, prev = _chain_head(conn, table)
    row_hash = chain_hash(prev, payload)
    try:
        conn.execute(
            f"INSERT INTO {table} (seq, prev_hash, row_hash, record_json) VALUES (?,?,?,?)",
            (seq + 1, prev, row_hash, payload_json),
        )
        conn.execute(
            "INSERT INTO chain (table_name, seq, prev_hash, row_hash, ts) VALUES (?,?,?,?,?)",
            (table, seq + 1, prev, row_hash, ts or str(payload.get("ts", ""))),
        )
    except sqlite3.DatabaseError as exc:
        raise RoleViolation(f"{table}: insert denied for this identity: {exc}") from exc
    return row_hash


def _check_table(table: str) -> None:
    if table not in RECORD_TABLES:
        raise VeraError(f"unknown record table {table!r}")


def load_rows(conn: sqlite3.Connection, table: str) -> list[dict]:
    _check_table(table)
    return [
        json.loads(row["record_json"])
        for row in conn.execute(f"SELECT record_json FROM {table} ORDER BY seq")
    ]


def verify_chain(conn: sqlite3.Connection, table: str) -> tuple[bool, int | None]:
    _check_table(table)
    head = HashChain.GENESIS
    expected_seq = 1
    for row in conn.execute(
        "SELECT seq, prev_hash, row_hash, record_json FROM {} ORDER BY seq".format(table)
    ):
        payload = json.loads(row["record_json"])
        if row["prev_hash"] != head:
            return False, int(row["seq"])
        head = chain_hash(row["prev_hash"], payload)
        if head != row["row_hash"]:
            return False, int(row["seq"])
        if int(row["seq"]) != expected_seq:
            return False, int(row["seq"])
        expected_seq += 1
    return True, None


def log_change(
    conn: sqlite3.Connection, *, change_id: str, key: str, old: str, new: str, ts: str, actor: str
) -> None:
    conn.execute(
        "INSERT INTO change_log (change_id, key, old_value, new_value, ts, actor) VALUES (?,?,?,?,?,?)",
        (change_id, key, old, new, ts, actor),
    )
