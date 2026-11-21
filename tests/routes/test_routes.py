"""The Anthropic API adapter, the pilot route registry and the route preflight, offline.

No test leaves the machine: every adapter gets a fake transport, the structural preflight
gets one that fails the test if it is called, and the redirect test talks to two servers on
the loopback address.  The pilot run through the runner is in ``tests/suite/test_pilot_live.py``.
"""

from __future__ import annotations

import http.server
import json
import threading
import urllib.request

import pytest

from agent_detention.suite.tier1 import harness

harness.use_tree()

from vera.delegation.adapters import (  # noqa: E402
    ERROR_AUTH_EXPIRED,
    ERROR_KEY_MISSING,
    ERROR_MODEL_FAILURE,
    ERROR_PROVIDER_REFUSAL,
    ERROR_ROUTE_LIMIT,
    ERROR_TRANSPORT,
    AnthropicApiAdapter,
    _NoRedirectHandler,
    _urllib_transport,
)
from vera.delegation.routes import preflight  # noqa: E402
from vera.delegation.routes.registry import (  # noqa: E402
    REGISTRY_CLAUDE_ONLY_PATH,
    adapter_for_route,
    load_registry,
    validate_registry,
)

KEY = "test-key-not-a-secret"
MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]


class FakeTransport:
    def __init__(self, *replies: tuple[int, dict | str]):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, url, headers, body, timeout):
        self.requests.append({"url": url, "headers": dict(headers), "body": json.loads(body)})
        status, payload = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return status, payload if isinstance(payload, str) else json.dumps(payload)


def ok_reply(text="OK", model="claude-sonnet-5-5", stop_reason="end_turn"):
    return 200, {
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }


def adapter(transport, env=None, **kwargs):
    return AnthropicApiAdapter(
        "claude-sonnet-5-5",
        transport=transport,
        env={"ANTHROPIC_API_KEY": KEY} if env is None else env,
        **kwargs,
    )


def test_missing_key_makes_no_request():
    transport = FakeTransport(ok_reply())
    result = adapter(transport, env={}).complete(MESSAGES)
    assert result.error_kind == ERROR_KEY_MISSING
    assert transport.requests == []


def test_request_shape_and_result():
    transport = FakeTransport(ok_reply("fine"))
    result = adapter(transport).complete(MESSAGES, max_tokens=5, temperature=0.7)
    request = transport.requests[0]
    assert request["headers"]["x-api-key"] == KEY
    assert request["body"]["system"] == "sys"
    assert request["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert "temperature" not in request["body"] and "tools" not in request["body"]
    assert request["body"]["max_tokens"] == 8192  # per-call hints are not sent
    assert result.ok and result.text == "fine"
    assert result.served_model_id == "claude-sonnet-5-5"
    assert (result.prompt_tokens, result.completion_tokens) == (12, 3)
    assert KEY not in repr(result)


def test_key_read_only_from_named_variable():
    transport = FakeTransport(ok_reply())
    env = {"OTHER_KEY": KEY}
    assert adapter(transport, env=env).complete(MESSAGES).error_kind == ERROR_KEY_MISSING
    assert adapter(transport, env=env, api_key_env="OTHER_KEY").complete(MESSAGES).ok


@pytest.mark.parametrize(
    "status, error_type, kind",
    [
        (429, "rate_limit_error", ERROR_ROUTE_LIMIT),
        (401, "authentication_error", ERROR_AUTH_EXPIRED),
        (403, "permission_error", ERROR_AUTH_EXPIRED),
    ],
)
def test_error_kinds(status, error_type, kind):
    transport = FakeTransport((status, {"type": "error", "error": {"type": error_type, "message": "x"}}))
    result = adapter(transport).complete(MESSAGES)
    assert result.error_kind == kind
    assert len(transport.requests) == 1  # only transport errors are retried here


def test_transport_error_retried_once():
    transport = FakeTransport((529, {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}))
    a = adapter(transport)
    result = a.complete(MESSAGES)
    assert result.error_kind == ERROR_TRANSPORT
    assert a.calls == 2 and result.attempts == 2


def test_refusal_and_empty_content():
    assert adapter(FakeTransport(ok_reply("", stop_reason="refusal"))).complete(MESSAGES).error_kind == (
        ERROR_PROVIDER_REFUSAL
    )
    assert adapter(FakeTransport(ok_reply(""))).complete(MESSAGES).error_kind == ERROR_MODEL_FAILURE


def test_error_text_redacts_keys():
    leaked = "sk-ant-" + "a" * 20
    transport = FakeTransport((400, {"type": "error", "error": {"type": "invalid_request_error", "message": leaked}}))
    result = adapter(transport).complete(MESSAGES)
    assert leaked not in result.error


def test_error_text_redacts_a_key_of_any_shape():
    # A key with no known prefix: the shape match cannot see it, the literal match must.
    key = "plainlookingkeywithoutprefix0123456789"
    env = {"ANTHROPIC_API_KEY": key}
    echoed = {"type": "error", "error": {"type": "authentication_error", "message": f"invalid x-api-key: {key}"}}
    result = adapter(FakeTransport((401, echoed)), env=env).complete(MESSAGES)
    assert result.error_kind == ERROR_AUTH_EXPIRED
    assert key not in result.error and "<redacted>" in result.error

    def raises(url, headers, body, timeout):
        raise OSError(f"connect failed with headers {dict(headers)}")

    result = adapter(raises, env=env).complete(MESSAGES)
    assert result.error_kind == ERROR_TRANSPORT
    assert key not in result.error and "<redacted>" in result.error


class _Server:
    """A loopback HTTP server in a thread; records the headers of every request it gets."""

    def __init__(self, status: int, headers: dict | None = None, body: bytes = b""):
        seen = self.seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append({name.lower(): value for name, value in self.headers.items()})
                self.rfile.read(int(self.headers.get("content-length") or 0))
                self.send_response(status)
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/messages"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_redirect_handler_refuses_every_redirect():
    request = urllib.request.Request("https://api.example.invalid/v1/messages", data=b"{}", method="POST")
    for code in (301, 302, 303, 307, 308):
        assert _NoRedirectHandler().redirect_request(request, None, code, "", {}, "http://elsewhere/") is None


def test_default_transport_does_not_send_the_key_to_a_redirect_target(monkeypatch):
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("NO_PROXY", "*")
    target = _Server(200, {"content-type": "application/json"}, json.dumps(ok_reply()[1]).encode())
    origin = _Server(302, {"location": target.url})
    try:
        status, _ = _urllib_transport(origin.url, {"x-api-key": KEY, "content-type": "application/json"}, b"{}", 5.0)
        assert status == 302
        result = AnthropicApiAdapter("claude-sonnet-5-5", url=origin.url, env={"ANTHROPIC_API_KEY": KEY}).complete(MESSAGES)
        assert result.error_kind == ERROR_TRANSPORT and not result.ok
        assert len(origin.seen) == 3  # the direct call, then the adapter's call and its one retry
        assert all(seen.get("x-api-key") == KEY for seen in origin.seen)
        assert target.seen == []  # the key never reached the redirect target
    finally:
        origin.close()
        target.close()


def test_pilot_registry_routes():
    registry = load_registry(profile="claude_only")
    assert [r["route_id"] for r in registry["routes"]] == ["o2_claude_sonnet_5_5", "anthropic_claude_haiku_5_5"]
    raw = REGISTRY_CLAUDE_ONLY_PATH.read_text()
    for word in ("credential", "cli_binary", "run_as_user", "~"):
        assert word not in raw
    for route in registry["routes"]:
        built = adapter_for_route(route)
        assert isinstance(built, AnthropicApiAdapter)
        assert built.model_id == route["model_id"] and built.api_key_env == "ANTHROPIC_API_KEY"


def test_registry_refuses_a_key_in_place_of_a_variable_name():
    data = json.loads(REGISTRY_CLAUDE_ONLY_PATH.read_text())
    data["routes"][0]["api_key_env"] = "sk-ant-literal-key"
    report = validate_registry(data)
    assert not report["valid"] and any("api_key_env" in e for e in report["errors"])


def _pilot_adapters(transport):
    routes = load_registry(profile="claude_only")["routes"]
    adapters = {
        r["route_id"]: AnthropicApiAdapter(r["model_id"], transport=transport, env={"ANTHROPIC_API_KEY": KEY})
        for r in routes
    }
    return routes, adapters


def _live(routes):
    return {
        r["route_id"]: {"route_id": r["route_id"], "model_id": r["model_id"], "requires_cli_isolation": False}
        for r in routes
    }


def test_structural_preflight_makes_no_call():
    def no_call(*args):
        raise AssertionError("the structural preflight must not call the model")

    routes, adapters = _pilot_adapters(no_call)
    report = preflight.run_preflight(routes, adapters=adapters)
    assert report["ok"] and report["probes"] == []
    with pytest.raises(preflight.IsolationRefused):
        preflight.validate_preflight_acceptance(report, route_ids=list(adapters), live_routes=_live(routes))


def test_probed_preflight_is_accepted():
    def reply(url, headers, body, timeout):
        status, payload = ok_reply(model=json.loads(body)["model"])
        return status, json.dumps(payload)

    routes, adapters = _pilot_adapters(reply)
    report = preflight.run_preflight(routes, adapters=adapters, probe=True, require_measured_routes=True)
    assert report["ok"], report["refusals"]
    accepted = preflight.validate_preflight_acceptance(report, route_ids=list(adapters), live_routes=_live(routes))
    assert accepted["probe_route_ids"] == sorted(adapters)


def test_probe_refuses_a_substituted_model():
    routes, adapters = _pilot_adapters(FakeTransport(ok_reply(model="some-other-model")))
    report = preflight.run_preflight(routes, adapters=adapters, probe=True, require_measured_routes=True)
    assert not report["ok"]
    assert any("probe_served_model" in refusal for refusal in report["refusals"])


def test_undeclared_transport_is_refused():
    routes, adapters = _pilot_adapters(FakeTransport(ok_reply()))
    route = dict(routes[0], route="some_cli")
    route.pop("_transport_registry", None)
    report = preflight.run_preflight([route], adapters=adapters)
    assert not report["ok"]
