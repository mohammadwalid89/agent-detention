"""Content-addressed, read-only state snapshots (design §4, §6.1).

A snapshot is what the verifier reads. It never contains the tool response, and it is
written once with mode 0444 so a later mutation is detectable by hash.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from ..canonical import canonical_json, sha256_text
from ..contracts import ObservedFinalState
from ..errors import VeraError

SNAPSHOT_TABLES = ("ticket", "alert", "job", "queue_job", "effects", "cache")

INVARIANTS = {
    "ticket_status_valid": "SELECT COUNT(*) FROM ticket WHERE status NOT IN ('open','in_progress','closed')",
    "alert_state_valid": "SELECT COUNT(*) FROM alert WHERE state NOT IN ('firing','resolved','silenced')",
    "job_status_valid": "SELECT COUNT(*) FROM job WHERE status NOT IN ('queued','running','resolved','failed')",
    "queue_non_negative": "SELECT COUNT(*) FROM queue_job WHERE arrived_at < 0 OR deadline_step < 0",
}


def _rows(conn: sqlite3.Connection, table: str) -> list[dict]:
    return [dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]  # type: ignore[arg-type]


def take_snapshot(
    db_path: str | Path,
    *,
    run_id: str,
    out_dir: str | Path,
    read_at: str,
    reader_version: str = "w1-snapshot-v1",
    tables: tuple[str, ...] = SNAPSHOT_TABLES,
) -> ObservedFinalState:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        payload: dict = {"tables": {}, "row_counts": {}, "invariants": {}}
        digest: dict[str, str] = {}
        for t in tables:
            rows = _rows(conn, t)
            payload["tables"][t] = rows
            payload["row_counts"][t] = len(rows)
            digest[t] = sha256_text(canonical_json(rows))
        payload["tables_digest"] = digest
        for name, sql in INVARIANTS.items():
            payload["invariants"][name] = int(conn.execute(sql).fetchone()[0])
    finally:
        conn.close()

    invariants_ok = all(v == 0 for v in payload["invariants"].values())
    body = canonical_json(payload)
    snapshot_hash = sha256_text(body)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{snapshot_hash}.json"
    if path.exists():  # content-addressed: identical state is the same file
        existing = path.read_text()
        if sha256_text(existing) != snapshot_hash:
            raise VeraError("snapshot hash collision or corrupted snapshot file")
    else:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(body)
        os.chmod(tmp, 0o444)
        os.replace(tmp, path)

    return ObservedFinalState(
        id=f"snap_{snapshot_hash[:16]}",
        run_id=run_id,
        snapshot_hash=snapshot_hash,
        read_at=read_at,
        reader_version=reader_version,
        tables_digest=digest,
        row_counts={str(k): int(v) for k, v in payload["row_counts"].items()},
        invariants_ok=invariants_ok,
    )


def load_snapshot(snapshot_dir: str | Path, snapshot_hash: str) -> dict:
    path = Path(snapshot_dir) / f"{snapshot_hash}.json"
    if not path.exists():
        raise VeraError(f"snapshot {snapshot_hash} not found")
    raw = path.read_text()
    if sha256_text(raw) != snapshot_hash:
        raise VeraError("snapshot content does not match its hash (tampered evidence)")
    return json.loads(raw)
