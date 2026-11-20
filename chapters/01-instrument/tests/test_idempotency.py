from __future__ import annotations

import pytest
from vera.errors import VeraError
from vera.idempotency import IdempotencyLedger

from .conftest import TS


def _set(service, key, value):
    service.set_config(key=key, value=value, ts=TS, actor="test")


def test_commit_then_timeout_produces_one_effect(service):
    _set(service, "fail_after_commit", "1")
    attempt = service.update_ticket(
        run_id="r1", ticket_id="T-100", fields={"status": "in_progress"},
        idempotency_key="k-timeout", ts=TS,
    )
    assert attempt.state == "UNCERTAIN" and attempt.retried is False
    assert service.ledger.effects_for_key("k-timeout") == 1
    rec = service.ledger.reconcile(idempotency_key="k-timeout")
    assert rec.state == "APPLIED"
    assert service.ledger.effects_for_key("k-timeout") == 1
    status = service.conn.execute("SELECT status FROM ticket WHERE ticket_id='T-100'").fetchone()[0]
    assert status == "in_progress"


def test_replay_five_times_produces_one_effect(service):
    states = [
        service.update_ticket(run_id="r1", ticket_id="T-100",
                              fields={"status": "closed"}, idempotency_key="k-replay", ts=TS).state
        for _ in range(5)
    ]
    assert states[0] == "APPLIED"
    assert states[1:] == ["DUPLICATE_SUPPRESSED"] * 4
    assert service.ledger.effects_for_key("k-replay") == 1


def test_intent_is_written_before_the_call(service):
    service.update_ticket(run_id="r1", ticket_id="T-100", fields={"status": "closed"},
                          idempotency_key="k-order", ts=TS)
    intent = service.ledger.intent("k-order")
    span = service.conn.execute(
        "SELECT ts FROM span WHERE tool='update_ticket' ORDER BY ts LIMIT 1"
    ).fetchone()
    assert intent is not None
    assert intent["ts"] <= span["ts"]


def test_crash_without_effect_abandons_the_intent(service):
    ledger = IdempotencyLedger(service.conn)

    def boom():
        raise RuntimeError("worker killed before commit")

    attempt = ledger.apply(run_id="r1", idempotency_key="k-crash", payload_hash="h",
                           ts=TS, apply_fn=boom)
    assert attempt.state == "FAILED"
    assert ledger.effects_for_key("k-crash") == 0
    assert ledger.intent("k-crash")["state"] == "ABANDONED"


def test_reconcile_with_state_probe_reports_uncertain(service):
    ledger = IdempotencyLedger(service.conn)

    def lost_response():
        raise RuntimeError("response lost")

    ledger.apply(run_id="r1", idempotency_key="k-lost", payload_hash="h", ts=TS,
                 apply_fn=lost_response)
    # Pretend the state probe shows the effect happened even though the ledger has no row.
    rec = ledger.reconcile(idempotency_key="k-lost", probe_fn=lambda: True)
    assert rec.state == "UNCERTAIN"
    assert "manual reconciliation" in rec.detail


def test_duplicate_intent_is_rejected(service):
    service.update_ticket(run_id="r1", ticket_id="T-100", fields={"status": "closed"},
                          idempotency_key="k-dup-intent", ts=TS)
    with pytest.raises(VeraError):
        service.ledger.begin(run_id="r1", idempotency_key="k-dup-intent",
                             payload_hash="h", ts=TS)
