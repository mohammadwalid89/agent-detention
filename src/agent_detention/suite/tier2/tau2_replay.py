"""Replay a tau2 task's gold *assistant* actions on the real tau2 domain tools (airline, retail).

Produces the before/after database (as plain dicts) and a row-level diff.  This is the same
mechanism tau2's own evaluator uses to build the golden end state (``get_db_hash`` after the
gold actions), so the act-correct end-state check in the adapter is tau2's DB state, not ours.
"""
from __future__ import annotations

import json

from agent_detention.suite import _tau2_import as _t

_t.ensure_path()

WRITE_TOOLS = {
    "airline": {
        "book_reservation", "cancel_reservation", "update_reservation_baggages",
        "update_reservation_flights", "update_reservation_passengers", "send_certificate",
    },
    "retail": {
        "cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
        "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
        "return_delivered_order_items",
    },
}


def load_tasks(domain: str) -> list[dict]:
    return json.loads((_t.domain_dir() / domain / "tasks.json").read_text())


def load_split(domain: str) -> dict:
    return json.loads((_t.domain_dir() / domain / "split_tasks.json").read_text())


def make_env(domain: str):
    """A fresh (db, toolkit) pair built from tau2's own data model and tools."""
    if domain == "retail":
        from tau2.domains.retail.data_model import RetailDB
        from tau2.domains.retail.tools import RetailTools
        db = RetailDB.load(str(_t.domain_dir() / "retail" / "db.json"))
        return db, RetailTools(db)
    if domain == "airline":
        from tau2.domains.airline.data_model import FlightDB
        from tau2.domains.airline.tools import AirlineTools
        db = FlightDB.load(str(_t.domain_dir() / "airline" / "db.json"))
        return db, AirlineTools(db)
    raise ValueError(domain)


def _check_no_init_state(task: dict) -> None:
    init = task.get("initial_state") or {}
    if init.get("initialization_data") or init.get("initialization_actions"):
        raise NotImplementedError("task carries initial_state; not used by airline/retail tasks")


def replay(domain: str, task: dict, db=None, tools=None) -> dict:
    """Run the gold actions; return before/after DB dicts, per-call outcomes and the db hash."""
    _check_no_init_state(task)
    if db is None:
        db, tools = make_env(domain)
    before = db.model_dump(mode="json")
    calls, errors = [], []
    for action in task["evaluation_criteria"].get("actions") or []:
        if action.get("requestor", "assistant") != "assistant":
            continue
        name, args = action["name"], action.get("arguments") or {}
        is_write = name in WRITE_TOOLS[domain]
        try:
            tools.use_tool(name, **args)
            calls.append({"name": name, "arguments": args, "ok": True, "write": is_write})
        except Exception as exc:  # tau2 itself records these as golden-action errors
            errors.append({"name": name, "arguments": args, "error": f"{type(exc).__name__}: {exc}"})
            calls.append({"name": name, "arguments": args, "ok": False, "write": is_write})
    after = db.model_dump(mode="json")
    return {"before": before, "after": after, "calls": calls, "errors": errors,
            "db_hash": tools.get_db_hash(), "changed": before != after}


def collections_of(db: dict) -> dict:
    """Collection name -> mapping key -> row, for the dict-valued top-level collections."""
    return {coll: rows for coll, rows in db.items() if isinstance(rows, dict)}


def row_diff(before: dict, after: dict) -> list[dict]:
    """Row-level diff: one entry per added / removed / changed row, with old/new per field."""
    diffs = []
    tb, ta = collections_of(before), collections_of(after)
    for coll in sorted(set(tb) | set(ta)):
        b, a = tb.get(coll, {}), ta.get(coll, {})
        for key in sorted(set(b) | set(a), key=str):
            if key not in a:
                diffs.append({"collection": coll, "key": key, "kind": "removed", "fields": {}})
            elif key not in b:
                diffs.append({"collection": coll, "key": key, "kind": "added", "fields": {}})
            elif b[key] != a[key]:
                fields = {k: [b[key].get(k), a[key].get(k)]
                          for k in set(b[key]) | set(a[key]) if b[key].get(k) != a[key].get(k)}
                diffs.append({"collection": coll, "key": key, "kind": "changed", "fields": fields})
    return diffs
