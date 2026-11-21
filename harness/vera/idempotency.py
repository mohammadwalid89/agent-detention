"""Write-intent ledger and idempotent effect application (design §4, F2).

Invariant: the intent row is committed BEFORE the call, and `effects.idempotency_key`
is UNIQUE. A timeout after commit leaves the intent PENDING and is reconciled by key
lookup — never by a blind retry.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass

from .contracts import WriteIntent
from .db import insert_record, load_rows
from .errors import VeraError

PENDING = "PENDING"
APPLIED = "APPLIED"
ABANDONED = "ABANDONED"


@dataclass(frozen=True)
class WriteAttempt:
    idempotency_key: str
    state: str  # APPLIED | UNCERTAIN | FAILED | DUPLICATE_SUPPRESSED
    effect_ref: str
    retried: bool
    detail: str


class IdempotencyLedger:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # -- intents -----------------------------------------------------------------
    def intent(self, key: str) -> dict | None:
        for row in load_rows(self.conn, "write_intent"):
            if row["idempotency_key"] == key:
                return row
        return None

    def begin(self, *, run_id: str, idempotency_key: str, payload_hash: str, ts: str,
              grant_id: str = "", on_behalf_of: str = "") -> WriteIntent:
        existing = self.intent(idempotency_key)
        if existing is not None:
            raise VeraError(f"write intent already exists for key {idempotency_key!r}")
        intent = WriteIntent(
            id=f"wi_{uuid.uuid4().hex[:16]}",
            run_id=run_id,
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
            state=PENDING,
            ts=ts,
            grant_id=grant_id,
            on_behalf_of=on_behalf_of,
        )
        insert_record(self.conn, "write_intent", intent, ts=ts)
        return intent

    def set_state(self, idempotency_key: str, state: str) -> None:
        if state not in (PENDING, APPLIED, ABANDONED):
            raise VeraError(f"bad intent state {state!r}")
        self.conn.execute(
            "UPDATE write_intent SET record_json = json_set(record_json, '$.state', ?) "
            "WHERE json_extract(record_json,'$.idempotency_key') = ?",
            (state, idempotency_key),
        )

    # -- effects -----------------------------------------------------------------
    def effect_for(self, idempotency_key: str) -> dict | None:
        row = self.conn.execute(
            "SELECT id, idempotency_key, run_id, payload_hash, effect_ref, ts "
            "FROM effects WHERE idempotency_key=?",
            (idempotency_key,),
        ).fetchone()
        return None if row is None else dict(row)

    def record_effect(self, *, key: str, run_id: str, payload_hash: str,
                      effect_ref: str, ts: str) -> dict:
        """Record an effect exactly once. Safe to call from inside the write path."""
        self.conn.execute(
            "INSERT INTO effects (id, idempotency_key, run_id, payload_hash, effect_ref, ts) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(idempotency_key) DO NOTHING",
            (f"eff_{uuid.uuid4().hex[:16]}", key, run_id, payload_hash, effect_ref, ts),
        )
        found = self.effect_for(key)
        if found is None:
            raise VeraError("effect insert vanished; ledger is inconsistent")
        return found

    def apply(
        self,
        *,
        run_id: str,
        idempotency_key: str,
        payload_hash: str,
        ts: str,
        apply_fn,
        grant_id: str = "",
        on_behalf_of: str = "",
    ) -> WriteAttempt:
        """Apply a mutating action exactly once.

        `apply_fn()` performs the real effect and returns an `effect_ref` string. If it
        raises TimeoutAfterCommit, the attempt is UNCERTAIN and must be reconciled.
        """
        prior = self.effect_for(idempotency_key)
        if prior is not None:
            return WriteAttempt(
                idempotency_key, "DUPLICATE_SUPPRESSED", str(prior["effect_ref"]), False,
                "effect already recorded for this key; no second call made",
            )
        existing_intent = self.intent(idempotency_key)
        if existing_intent is None:
            self.begin(
                run_id=run_id, idempotency_key=idempotency_key, payload_hash=payload_hash,
                ts=ts, grant_id=grant_id, on_behalf_of=on_behalf_of,
            )
        try:
            effect_ref = apply_fn()
        except Exception as exc:  # noqa: BLE001 - classified below, never swallowed silently
            if getattr(exc, "applied", False):
                return WriteAttempt(
                    idempotency_key, "UNCERTAIN", "", False,
                    f"effect may have committed ({type(exc).__name__}); reconcile by key",
                )
            self.set_state(idempotency_key, ABANDONED)
            return WriteAttempt(idempotency_key, "FAILED", "", False, f"{type(exc).__name__}: {exc}")
        effect = self.record_effect(
            key=idempotency_key, run_id=run_id, payload_hash=payload_hash,
            effect_ref=effect_ref, ts=ts,
        )
        self.set_state(idempotency_key, APPLIED)
        return WriteAttempt(idempotency_key, "APPLIED", str(effect["effect_ref"]), False, "applied once")

    def reconcile(self, *, idempotency_key: str, probe_fn=None) -> WriteAttempt:
        """Look up the effect by key. A blind retry is never performed here."""
        effect = self.effect_for(idempotency_key)
        if effect is not None:
            self.set_state(idempotency_key, APPLIED)
            return WriteAttempt(
                idempotency_key, "APPLIED", str(effect["effect_ref"]), False,
                "reconciled: effect exists for key",
            )
        if probe_fn is not None and probe_fn():
            return WriteAttempt(
                idempotency_key, "UNCERTAIN", "", False,
                "state probe says the effect happened but no ledger row exists; manual reconciliation required",
            )
        self.set_state(idempotency_key, ABANDONED)
        return WriteAttempt(idempotency_key, "FAILED", "", False, "reconciled: no effect, intent abandoned")

    def effects_for_key(self, idempotency_key: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM effects WHERE idempotency_key=?", (idempotency_key,)
        ).fetchone()
        return int(row["n"])
