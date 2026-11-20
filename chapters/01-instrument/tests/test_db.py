from __future__ import annotations

import json
import sqlite3

import pytest
from vera.contracts import Principal, WriteIntent
from vera.db import connect, insert_record, load_rows, verify_chain
from vera.errors import RoleViolation
from vera.sandbox.service import SandboxService

from .conftest import TS


def _principal() -> Principal:
    return Principal(id="p1", kind="USER", display_name="P")


def test_verdict_table_is_insert_only(api, verifier_conn):
    from vera.contracts import Attribution, OperationalVerdict

    v = OperationalVerdict(
        id="v1", run_id="r1", checks=(), authorization_ok=True, monitor_integrity_ok=True,
        attribution=Attribution(mode="AS_SELF", grant_id="", on_behalf_of="", attribution_ok=True),
        verdict="VERIFIED_SUCCESS", ground_truth_version="v1", ts=TS,
        signature_hmac="sig", verifier_version="w1",
    )
    insert_record(verifier_conn, "verification_event", v, ts=TS)
    # Denied by the authorizer before the trigger is even reached.
    with pytest.raises(sqlite3.DatabaseError):
        verifier_conn.execute("UPDATE verification_event SET record_json='{}'")
    with pytest.raises(sqlite3.DatabaseError):
        verifier_conn.execute("DELETE FROM verification_event")
    # The api identity may not insert a verdict at all.
    with pytest.raises(RoleViolation):
        insert_record(api, "verification_event", v, ts=TS)


def test_api_cannot_read_audit_verdicts(db_path):
    conn = connect(db_path, "api_rw")
    try:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("SELECT COUNT(*) FROM audit_verdict").fetchone()
    finally:
        conn.close()


def test_grader_cannot_write_domain_tables(grader_conn):
    with pytest.raises(RoleViolation):
        insert_record(grader_conn, "task", object(), ts=TS)


def test_wsp_identity_cannot_read_verdicts(db_path):
    conn = connect(db_path, "wsp_ro")
    try:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("SELECT COUNT(*) FROM verification_event").fetchone()
    finally:
        conn.close()


def test_chain_verifies_and_detects_tamper(api):
    for i in range(3):
        insert_record(api, "principal", Principal(id=f"p{i}", kind="USER", display_name="P"), ts=TS)
    ok, bad = verify_chain(api, "principal")
    assert ok and bad is None
    # Bypass the authorizer with a raw connection to simulate an out-of-band edit.
    raw = sqlite3.connect(api.execute("PRAGMA database_list").fetchone()["file"])
    raw.execute("DROP TRIGGER principal_no_update")
    raw.execute(
        "UPDATE principal SET record_json=? WHERE seq=2",
        (json.dumps({"id": "hacked", "kind": "USER", "display_name": "P", "_schema": "3.3"}),),
    )
    raw.commit()
    raw.close()
    ok, bad = verify_chain(api, "principal")
    assert not ok and bad == 2


def test_write_intent_payload_is_frozen_but_state_may_change(api):
    intent = WriteIntent(id="wi1", run_id="r1", idempotency_key="k1", payload_hash="h",
                         state="PENDING", ts=TS)
    insert_record(api, "write_intent", intent, ts=TS)
    api.execute("UPDATE write_intent SET record_json=json_set(record_json,'$.state','APPLIED')")
    rows = load_rows(api, "write_intent")
    assert rows[0]["state"] == "APPLIED"
    with pytest.raises(sqlite3.IntegrityError):
        api.execute("UPDATE write_intent SET record_json=json_set(record_json,'$.payload_hash','zzz')")


def test_effects_unique_key(db_path):
    svc = SandboxService(str(db_path))
    svc.seed()
    svc.update_ticket(run_id="r", ticket_id="T-100", fields={"status": "in_progress"},
                      idempotency_key="dup-key", ts=TS)
    with pytest.raises(sqlite3.IntegrityError):
        svc.conn.execute(
            "INSERT INTO effects VALUES ('e2','dup-key','r','h','ref','ts')"
        )


def test_change_log_is_insert_only(api):
    api.execute("INSERT INTO change_log (change_id,key,old_value,new_value,ts,actor) "
                "VALUES ('c1','k','a','b',?,'t')", (TS,))
    with pytest.raises(sqlite3.DatabaseError):
        api.execute("UPDATE change_log SET new_value='z'")
    with pytest.raises(sqlite3.DatabaseError):
        api.execute("DELETE FROM change_log")
