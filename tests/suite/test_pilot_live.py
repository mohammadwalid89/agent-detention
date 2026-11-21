"""A live (pilot-scope) run through ``run_suite --pilot`` with ``AnthropicApiAdapter``, offline.

The adapters' network transport is replaced by an in-process fake, so no request leaves the process and no model is
called.  The test takes the path a live run takes: the probed preflight by the command ``harness/README.md`` gives, a
pilot gate built from the current case-text overlay digest, a runner config with a call cap, one suite pair (two rows)
and the Detention phase.  It then checks that every request carried the key from the environment, went to the Messages
API address, and that the key is in no file the run wrote.  Needs the suite inputs (``build_inputs``), like the other
suite tests.
"""

from __future__ import annotations

import contextlib
import io
import json
import time

import pytest

from agent_detention.suite.tier1 import harness

TREE = harness.tree_path()
pytestmark = pytest.mark.skipif(not (TREE / "manifests/suite/SUITE_SOURCE_V14B.json").is_file(), reason="suite inputs not built")

FAKE_KEY = "fake-key-for-the-pilot-test-" + "7" * 24
MAX_CALLS = 20


def _first_pair_id() -> str:
    source = json.loads((TREE / "manifests/suite/SUITE_SOURCE_V14B.json").read_text())
    return next(m["mechanism_id"] for m in source["mechanisms"] if m["mechanism_id"].startswith("AA-"))


def test_pilot_run_through_the_api_adapter_keeps_the_key_out_of_every_file(tmp_path, monkeypatch, patched_tree):
    from vera.delegation import adapters, case_text
    from vera.delegation import runner_v13 as r
    from vera.delegation.routes import preflight
    from vera.delegation.routes.registry import load_registry

    from agent_detention.suite.integrate import rows as rows_mod
    from agent_detention.suite.integrate import run_suite
    from agent_detention.suite.integrate.binding import resolve_suite_binding

    sent: list[dict] = []

    def fake_transport(url, headers, body, timeout):
        payload = json.loads(body)
        sent.append({"url": url, "key": headers.get("x-api-key"), "model": payload["model"], "fields": sorted(payload)})
        text = "OK" if "preflight" in payload["messages"][0]["content"].lower() else (
            '{"action": "escalate", "reason": "fake reply"}'
        )
        reply = {"id": "msg_fake", "type": "message", "role": "assistant", "model": payload["model"],
                 "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                 "usage": {"input_tokens": 100, "output_tokens": 10}}
        return 200, json.dumps(reply)

    # adapters take the module's default transport when they are built
    monkeypatch.setattr(adapters, "_urllib_transport", fake_transport)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)

    out = tmp_path / "pilot"
    out.mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        rc = preflight.cmd_preflight(["--profile", "claude_only", "--probe", "--require-measured",
                                      "--out", str(out / "preflight.json")])
    assert rc == 0, (out / "preflight.json").read_text()
    registry = load_registry(profile="claude_only")
    assert len(sent) == len(registry["routes"])  # one probe per route

    resolve_suite_binding()  # selects the suite's case-text overlay for the process; undone at the end
    try:
        _run_pilot(out, registry, r, case_text.load_case_text, rows_mod, run_suite)
    finally:
        case_text.reset_overlay()

    assert len(sent) > len(registry["routes"])  # the run itself made calls
    assert all(s["key"] == FAKE_KEY for s in sent)
    assert {s["url"] for s in sent} == {adapters.ANTHROPIC_API_URL}
    assert {s["model"] for s in sent} <= {route["model_id"] for route in registry["routes"]}
    assert all(set(s["fields"]) <= {"model", "max_tokens", "messages", "system"} for s in sent)

    records = [json.loads(line) for path in sorted((out / "run").rglob("episode_calls/*.jsonl"))
               for line in path.read_text().splitlines() if line.strip()]
    assert records
    assert {rec.get("adapter") for rec in records} == {"AnthropicApiAdapter"}
    assert {rec.get("served_model_check") for rec in records} == {"MATCH"}

    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert files
    leaked = [str(p.relative_to(tmp_path)) for p in files if FAKE_KEY in p.read_text(errors="replace")]
    assert leaked == []


def _run_pilot(out, registry, r, load_case_text, rows_mod, run_suite):
    gate = {
        "kind": r.PILOT_GATE_KIND, "valid": True, "not_main_grid": True, "scope": "pilot",
        "owner": "test fixture", "job": "pilot-fake-transport-test", "model_calls_approved": True,
        "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3600)),
        "case_text_overlay_digest": str(load_case_text().get("digest") or ""),
    }
    (out / "gate.json").write_text(json.dumps(gate))
    config = {
        "kind": r.CONFIG_KIND, "episode_workers": 1, "infra_retries": 0, "max_model_calls": MAX_CALLS,
        "routes": {route["route_id"]: {"concurrency": 1, "min_interval_seconds": 0.0, "measured_calls_per_min": 30.0}
                   for route in registry["routes"]},
    }
    (out / "config.json").write_text(json.dumps(config))
    (out / "signing.key").write_text("test-signing-key-not-a-secret\n")
    rows = rows_mod.make_rows([_first_pair_id()], [rows_mod.DEFAULT_SLOT], 1)
    assert len(rows) == 2
    rows_mod.write_table(out / "rows.json", rows, "pilot fake-transport test")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = run_suite.main([
            "--assignment", str(out / "rows.json"), "--out", str(out / "run"), "--pilot",
            "--pilot-gate", str(out / "gate.json"), "--preflight-acceptance", str(out / "preflight.json"),
            "--config", str(out / "config.json"), "--signing-key-file", str(out / "signing.key"), "--detention",
        ])
    summary = json.loads(buf.getvalue())
    assert rc == 0, summary
    assert summary["run_scope"] == "pilot" and summary["rows_completed"] == 2 and summary["rows_remaining"] == 0
    assert summary["detention"] is not None
