"""Model adapters: the mock backend, the free NVIDIA route and the Anthropic API.

``MockAdapter``
    deterministic scripted/policy backend for tests and the offline end-to-end run;
``NvidiaAdapter``
    the free NVIDIA NIM route, reusing ``vera.pilot.nvidia`` request settings, throttle,
    timeout and model-failure classification.  It reads ``NVIDIA_API_KEY`` only through the
    existing client and never copies the key into a result;
``AnthropicApiAdapter``
    a single-turn Anthropic Messages API call.  It reads the key from an environment
    variable (``ANTHROPIC_API_KEY`` by default) and from nowhere else.

Every adapter returns an ``AdapterResult``.  ``RecordingAdapter`` wraps any of them and
writes the per-episode call log (messages in, raw reply out) that the leak audit reads.
"""

from __future__ import annotations

import contextvars
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..pilot.nvidia import (
    NVIDIA_PROVIDER_FAMILY,
    NVIDIA_REQUEST_MAX_TOKENS,
    NVIDIA_REQUEST_TIMEOUT_SECONDS,
    NvidiaCallError,
    NvidiaClient,
    NvidiaKeyMissing,
    NvidiaModelFailure,
)


class AdapterError(RuntimeError):
    """Base adapter failure."""


@dataclass(frozen=True)
class AdapterResult:
    text: str = ""
    model_id: str = ""
    vendor_family: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    paid: bool = False
    error: str = ""
    error_kind: str = ""  # MODEL_FAILURE | TRANSPORT_ERROR | ROUTE_LIMIT | AUTH_EXPIRED | ...
    provider: str = ""
    attempts: int = 0
    finish_reason: str = ""
    cost_micros: int = 0
    resume_key: str = ""
    route: str = ""
    # The model the *provider* says served the call (parsed from the reply payload).
    # ``model_id`` is the id we configured/expect.  Empty = not reported.
    served_model_id: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def usage(self) -> dict:
        micros = int(self.cost_micros)
        if not micros and self.cost_usd:
            micros = int(round(self.cost_usd * 1_000_000))
        return {
            "model_id": self.model_id,
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "cost_micros": micros,
            "paid": bool(self.paid),
            "error_kind": self.error_kind,
        }


SERVED_MODEL_MATCH = "MATCH"
SERVED_MODEL_MISMATCH = "MISMATCH"
SERVED_MODEL_NOT_REPORTED = "NOT_REPORTED"

_DATE_SUFFIX = re.compile(r"^-?\d{6,8}$")
_CONTEXT_SUFFIX = re.compile(r"\[[0-9a-z]+\]$", re.IGNORECASE)


def served_model_matches(expected: str, served: str) -> bool:
    """Is the provider-reported model the expected one?

    Equal after dropping a context-window suffix such as ``[1m]``, or the served id is the
    expected id plus a dated-snapshot suffix (``claude-x-5-5`` vs ``claude-x-5-5-20261001``).
    Anything else is a different model.  An alias that the provider resolves to a *different*
    family or size is exactly what this check exists to catch, so there is no fuzzy matching.
    """
    expected = _CONTEXT_SUFFIX.sub("", str(expected or "").strip())
    served = _CONTEXT_SUFFIX.sub("", str(served or "").strip())
    if not expected or not served:
        return False
    if served == expected:
        return True
    if served.startswith(expected) and _DATE_SUFFIX.match(served[len(expected):]):
        return True
    return False


def served_model_check(expected: str, served: str) -> str:
    """MATCH / MISMATCH / NOT_REPORTED for one call."""
    if not str(served or "").strip():
        return SERVED_MODEL_NOT_REPORTED
    return (
        SERVED_MODEL_MATCH if served_model_matches(expected, served) else SERVED_MODEL_MISMATCH
    )


#: Out-of-band harness truth for *mock* fixtures only.  The terminal report prompt no
#: longer prints whether the task completed, so an honest scripted reporter has to learn it
#: some other way.  The episode sets this around a terminal-report call **only when the
#: orchestrator adapter is a MockAdapter** (see :func:`mock_truth_scope`); a real model never
#: sees it and nothing in a prompt carries it.  Value: ``{"completed": bool, "role": str}``.
MOCK_HARNESS_TRUTH: "contextvars.ContextVar[dict | None]" = contextvars.ContextVar(
    "vera_mock_harness_truth", default=None
)


def _unwrap_adapter(adapter: object) -> object:
    seen = 0
    while hasattr(adapter, "_inner") and seen < 8:
        adapter = adapter._inner  # RecordingAdapter and similar wrappers
        seen += 1
    return adapter


def is_mock_adapter(adapter: object) -> bool:
    return isinstance(_unwrap_adapter(adapter), MockAdapter)


class mock_truth_scope:
    """Context manager: expose harness truth to a mock policy, and to nothing else."""

    def __init__(self, adapter: object, *, completed: bool, role: str):
        self._active = is_mock_adapter(adapter)
        self._value = {"completed": bool(completed), "role": str(role or "")}
        self._token = None

    def __enter__(self):
        if self._active:
            self._token = MOCK_HARNESS_TRUTH.set(self._value)
        return self

    def __exit__(self, *exc):
        if self._token is not None:
            MOCK_HARNESS_TRUTH.reset(self._token)
        return False


class MockAdapter:
    """Scripted or policy-driven backend.  Never contacts a network."""

    paid = False

    def __init__(
        self,
        *,
        model_id: str,
        vendor_family: str,
        responses: list[dict | str] | None = None,
        policy: Callable[[list[dict]], dict | str] | None = None,
        capability_band: str = "band_test",
        served_model_id: str = "",
    ):
        self.model_id = model_id
        # What this mock claims the provider served (default: its own model id).  A test
        # sets a different value to simulate silent model substitution.
        self.served_model_id = str(served_model_id or "")
        self.vendor_family = vendor_family
        self.capability_band = capability_band
        self._responses = list(responses or [])
        self._policy = policy
        self.calls = 0

    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> AdapterResult:
        self.calls += 1
        if self._policy is not None:
            value = self._policy(messages)
        elif self._responses:
            value = self._responses.pop(0)
        else:
            return AdapterResult(
                model_id=self.model_id,
                vendor_family=self.vendor_family,
                error="mock adapter has no scripted response",
                error_kind="MODEL_FAILURE",
            )
        if isinstance(value, AdapterResult):
            return value
        if isinstance(value, dict):
            if value.get("__error__"):
                return AdapterResult(
                    model_id=self.model_id,
                    vendor_family=self.vendor_family,
                    error=str(value["__error__"]),
                    error_kind=str(value.get("__error_kind__", "MODEL_FAILURE")),
                )
            return AdapterResult(
                text=json.dumps(value, sort_keys=True),
                model_id=self.model_id,
                # A mock "serves" its own model, so the runner's check has a value.
                served_model_id=self.served_model_id or self.model_id,
                vendor_family=self.vendor_family,
                prompt_tokens=max(1, len(json.dumps(messages)) // 4),
                completion_tokens=max(1, len(json.dumps(value)) // 4),
            )
        return AdapterResult(
            text=str(value),
            served_model_id=self.served_model_id or self.model_id,
            model_id=self.model_id,
            vendor_family=self.vendor_family,
            prompt_tokens=max(1, len(json.dumps(messages)) // 4),
            completion_tokens=max(1, len(str(value)) // 4),
        )


class NvidiaAdapter:
    """Free NVIDIA NIM backend; reuses the v1 client settings unchanged."""

    paid = False
    #: A free HTTP transport declares its own capability instead of being
    #: recognised by the literal route label ``nvidia_nim``.
    requires_cli_isolation = False

    def __init__(
        self,
        model_id: str,
        *,
        client: NvidiaClient | None = None,
        vendor_family: str = NVIDIA_PROVIDER_FAMILY,
    ):
        self.model_id = model_id
        self.vendor_family = vendor_family
        self.client = client or NvidiaClient()
        self.max_tokens = NVIDIA_REQUEST_MAX_TOKENS
        self.timeout_seconds = NVIDIA_REQUEST_TIMEOUT_SECONDS

    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> AdapterResult:
        try:
            result = self.client.run(
                self.model_id,
                messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature,
            )
        except NvidiaKeyMissing as exc:
            return AdapterResult(
                model_id=self.model_id,
                vendor_family=self.vendor_family,
                error=str(exc),
                error_kind="KEY_MISSING",
            )
        except NvidiaModelFailure as exc:
            return AdapterResult(
                model_id=self.model_id,
                vendor_family=self.vendor_family,
                error=str(exc),
                error_kind="MODEL_FAILURE",
            )
        except NvidiaCallError as exc:
            return AdapterResult(
                model_id=self.model_id,
                vendor_family=self.vendor_family,
                error=str(exc),
                error_kind="TRANSPORT_ERROR",
            )
        return AdapterResult(
            text=result.text,
            model_id=self.model_id,
            vendor_family=self.vendor_family,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=result.latency_ms,
            cost_usd=0.0,
            paid=False,
        )


# Error kinds shared by every live route.

ERROR_ROUTE_LIMIT = "ROUTE_LIMIT"
ERROR_AUTH_EXPIRED = "AUTH_EXPIRED"
ERROR_TRANSPORT = "TRANSPORT_ERROR"
ERROR_MODEL_FAILURE = "MODEL_FAILURE"
ERROR_CONFIG = "CONFIG_ERROR"
ERROR_KEY_MISSING = "KEY_MISSING"
# v14b: the provider's own safeguard declined THIS message.  It is a property of the
# prompt, not of the route, transport or auth: neither a model failure nor a route error.
ERROR_PROVIDER_REFUSAL = "PROVIDER_REFUSAL"
PROVIDER_REFUSAL_MESSAGE_TABLE = ("safeguards flagged this message",)

# Fallback for error text that carries no structured type.  Case-insensitive substring
# matching over error output only, never over model content.
ROUTE_LIMIT_MESSAGE_TABLE = (
    "usage limit reached",
    "usage_limit_reached",
    "rate limit",
    "rate_limit_error",
    "rate-limited",
    "too many requests",
    "quota exceeded",
    "exceeded your current quota",
    "http 429",
)
AUTH_EXPIRED_MESSAGE_TABLE = (
    "invalid api key",
    "incorrect api key",
    "invalid x-api-key",
    "authentication_error",
    "authentication failed",
    "could not authenticate",
    "not authenticated",
    "unauthenticated",
    "401 unauthorized",
    "403 forbidden",
    "http 401",
    "http 403",
    "account is not authorized",
    "unauthorized",
)

_SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_\-]{8,}|nvapi-[A-Za-z0-9_\-]{8,}|"
    r"AIza[A-Za-z0-9_\-]{20,}|Bearer\s+[A-Za-z0-9._~+/\-]{16,})",
    re.IGNORECASE,
)


def classify_error_text(text: str) -> tuple[str, str]:
    """Map error output to ``AUTH_EXPIRED``/``ROUTE_LIMIT``/``PROVIDER_REFUSAL`` (or empty).

    Returns ``(error_kind, matched_message)``.  Authentication is checked first: an auth
    failure must stop the run, while a usage limit can resume after the reset window.
    """
    lowered = str(text or "").lower()
    for message in AUTH_EXPIRED_MESSAGE_TABLE:
        if message in lowered:
            return ERROR_AUTH_EXPIRED, message
    for message in ROUTE_LIMIT_MESSAGE_TABLE:
        if message in lowered:
            return ERROR_ROUTE_LIMIT, message
    # Last, so a real auth/limit error is never reclassified as a refusal.
    for message in PROVIDER_REFUSAL_MESSAGE_TABLE:
        if message in lowered:
            return ERROR_PROVIDER_REFUSAL, message
    return "", ""


def _redact(text: Any, *, limit: int = 700, secrets: Sequence[str] = ()) -> str:
    """Remove key-shaped text, and every literal value in ``secrets``, from ``text``.

    The shape match alone misses a key of an unknown form, so a caller that holds the key
    passes it in ``secrets`` as well.
    """
    cleaned = str(text or "")
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "<redacted>")
    cleaned = _SECRET_RE.sub("<redacted>", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned[:limit]


ADAPTER_CALL_KIND = "vera.job39.adapter_call.v1"


def verify_call_record(record: Mapping) -> list[str]:
    """Problems with one call record's transcript fields (empty list = verifies).

    ``messages`` must be a non-empty list of ``{role, content}`` objects whose contents are
    strings, with at least one non-empty ``user`` turn.  ``reply_text`` must be present and a
    string; it must also be non-empty when the call succeeded.  A call that returned an error
    has no reply to store, so an empty ``reply_text`` is accepted there, but the key must
    still be present and the record must say why (``error``).
    """
    problems: list[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        problems.append("messages missing or empty")
    else:
        saw_user = False
        for index, message in enumerate(messages):
            if (
                not isinstance(message, Mapping)
                or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
            ):
                problems.append(f"messages[{index}] is not a {{role, content}} string pair")
                continue
            if message["role"] == "user" and message["content"].strip():
                saw_user = True
        if not saw_user and not any(p.startswith("messages[") for p in problems):
            problems.append("messages has no non-empty user turn")
    if "reply_text" not in record:
        problems.append("reply_text missing")
    elif not isinstance(record["reply_text"], str):
        problems.append("reply_text is not a string")
    else:
        failed = bool(record.get("error")) or record.get("ok") is False
        if not failed and not record["reply_text"].strip():
            problems.append("reply_text empty on a successful call")
        if failed and not str(record.get("error") or "").strip():
            problems.append("failed call has no error text")
    return problems


def verify_call_log(path: str | Path) -> dict:
    """Verify every record of one episode call log for transcript completeness.

    Returns ``{"ok", "records", "problems"}``.  An unreadable line, a record of another
    kind or a record that fails :func:`verify_call_record` is a problem.  A log with no
    records at all does not verify: a call that left no transcript is the failure this
    check exists to catch.
    """
    path = Path(path)
    problems: list[str] = []
    count = 0
    if not path.is_file():
        return {"ok": False, "records": 0, "problems": [f"call log missing: {path}"]}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {number}: not JSON ({exc.msg})")
            continue
        if not isinstance(record, dict) or record.get("kind") != ADAPTER_CALL_KIND:
            problems.append(f"line {number}: not an adapter call record")
            continue
        count += 1
        problems.extend(f"line {number}: {p}" for p in verify_call_record(record))
    if count == 0 and not problems:
        problems.append("call log holds no call records")
    return {"ok": not problems, "records": count, "problems": problems}


class RecordingAdapter:
    """Transcript-recording wrapper: the runner wraps every adapter in one per episode.

    Each call appends one record: the exact messages in and the raw reply out.  It forwards
    every attribute it does not define, so it can stand in for the wrapped adapter anywhere
    the runner expects one.
    """

    def __init__(self, inner: Any, log: "EpisodeCallLog", *, route: str = "mock"):
        self._inner = inner
        self._log = log
        self.route = getattr(inner, "route", "") or route

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def complete(self, messages: list[dict], **kwargs: Any) -> "AdapterResult":
        result = self._inner.complete(messages, **kwargs)
        sent = [
            {
                "role": str(m.get("role", "")),
                "content": (
                    m.get("content")
                    if isinstance(m.get("content"), str)
                    else json.dumps(m.get("content", ""), sort_keys=True)
                ),
            }
            for m in (messages or ())
        ]
        self._log.append({
            "kind": ADAPTER_CALL_KIND,
            "adapter": type(self._inner).__name__,
            "route": self.route,
            "model_id": str(getattr(self._inner, "model_id", "") or ""),
            "messages": sent,
            "reply_text": str(result.text or ""),
            "prompt_tokens": int(result.prompt_tokens),
            "completion_tokens": int(result.completion_tokens),
            "ok": bool(result.ok),
            "error_kind": result.error_kind,
            "error": result.error,
            "expected_model_id": str(getattr(self._inner, "model_id", "") or ""),
            "served_model_id": result.served_model_id,
            "served_model_check": served_model_check(
                str(getattr(self._inner, "model_id", "") or ""), result.served_model_id
            ),
        })
        return result


class EpisodeCallLog:
    """Append-only per-episode adapter-call log.

    Each record carries the route, model, token counts, the served-model check and the
    transcript: the messages delivered and the raw reply.  The runner binds one log per
    episode; the file is created 0600 because it holds prompts and replies.
    """

    def __init__(self, path: str | Path, *, episode_id: str = ""):
        self.path = Path(path)
        self.episode_id = str(episode_id or "")
        self._lock = threading.Lock()

    def append(self, record: Mapping) -> dict:
        payload = dict(record)
        payload.setdefault("kind", ADAPTER_CALL_KIND)
        payload.setdefault("episode_id", self.episode_id)
        payload.setdefault("recorded_at_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        line = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, (line + "\n").encode("utf-8"))
            finally:
                os.close(fd)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        return payload

    def records(self) -> list[dict]:
        if not self.path.is_file():
            return []
        out: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


def _split_messages(messages: Sequence[Mapping]) -> tuple[str, str]:
    system_parts: list[str] = []
    user_parts: list[str] = []
    for message in messages or ():
        role = str(message.get("role", ""))
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, sort_keys=True)
        if role == "system":
            system_parts.append(content)
        else:
            user_parts.append(content)
    return "\n\n".join(part for part in system_parts if part), "\n\n".join(
        part for part in user_parts if part
    )


# Anthropic Messages API, key from an environment variable

ROUTE_ANTHROPIC_API = "anthropic_api"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_MAX_OUTPUT_TOKENS = 8192

#: ``transport(url, headers, body, timeout_seconds) -> (http_status, response_text)``.
Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, str]]


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow no redirect.

    ``urlopen`` would answer a 301, 302 or 303 by sending the same headers, the key among
    them, to the new address.  Returning ``None`` here makes a 3xx come back as an HTTP error
    instead, so the key only ever goes to ``url``.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _urllib_transport(
    url: str, headers: Mapping[str, str], body: bytes, timeout_seconds: float
) -> tuple[int, str]:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    opener = urllib.request.build_opener(_NoRedirectHandler())
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            return int(response.status), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")


def _api_error_kind(status: int, error_type: str, message: str) -> str:
    if status == 429 or error_type == "rate_limit_error":
        return ERROR_ROUTE_LIMIT
    if status in (401, 403) or error_type in ("authentication_error", "permission_error"):
        return ERROR_AUTH_EXPIRED
    if status >= 500 or error_type in ("overloaded_error", "api_error"):
        return ERROR_TRANSPORT
    kind, _ = classify_error_text(message)
    return kind or ERROR_CONFIG


class AnthropicApiAdapter:
    """Single-turn call to the Anthropic Messages API.

    The key is read from the environment variable named by ``api_key_env`` at call time and
    is not stored on the adapter.  Error text is cleaned of the key's literal value and of
    key-shaped text before it goes into a result (and so into a call log), and the default
    transport follows no redirect, so the key is sent only to ``url``.

    The adapter does not compute cost: ``paid`` is true but ``cost_usd`` and ``cost_micros``
    stay 0.  The pilot gate's ``model_calls_approved`` only allows calls; cap them with the
    runner config's ``max_model_calls``.

    System messages become the ``system`` field and the rest one user turn; no tools are
    offered.  The per-call ``max_tokens`` and ``temperature`` the harness passes are not
    sent, because the published runs set neither per call: ``max_output_tokens`` is the
    only output limit and the provider's default temperature applies.  A transport error is
    retried ``max_retries`` times; limits, authentication failures and refusals are returned
    to the runner, which has its own pause and retry rules for them.
    """

    paid = True
    requires_cli_isolation = False
    enforces_max_tokens = False
    route = ROUTE_ANTHROPIC_API

    def __init__(
        self,
        model_id: str,
        *,
        vendor_family: str = "Anthropic",
        api_key_env: str = ANTHROPIC_KEY_ENV,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        timeout_seconds: float = 180.0,
        max_retries: int = 1,
        url: str = ANTHROPIC_API_URL,
        transport: Transport | None = None,
        env: Mapping[str, str] | None = None,
    ):
        self.model_id = str(model_id)
        self.vendor_family = str(vendor_family)
        self.api_key_env = str(api_key_env)
        self.max_output_tokens = int(max_output_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.url = str(url)
        self._transport = transport or _urllib_transport
        self._env = env
        self.calls = 0

    def _key(self) -> str:
        env = self._env if self._env is not None else os.environ
        return str(env.get(self.api_key_env, "") or "").strip()

    def _result(self, **kwargs: Any) -> AdapterResult:
        return AdapterResult(
            model_id=self.model_id,
            vendor_family=self.vendor_family,
            provider="anthropic",
            route=self.route,
            paid=True,
            **kwargs,
        )

    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> AdapterResult:
        del max_tokens, temperature  # see the class docstring
        system_text, user_text = _split_messages(messages)
        if not user_text.strip():
            return self._result(error="no user content to send", error_kind=ERROR_CONFIG)
        key = self._key()
        if not key:
            return self._result(
                error=f"{self.api_key_env} is not set; no request was made",
                error_kind=ERROR_KEY_MISSING,
            )
        body: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self.max_output_tokens,
            "messages": [{"role": "user", "content": user_text}],
        }
        if system_text:
            body["system"] = system_text
        headers = {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
        data = json.dumps(body).encode("utf-8")
        result = self._result(error="no attempt made", error_kind=ERROR_TRANSPORT)
        for attempt in range(1, self.max_retries + 2):
            self.calls += 1
            result = self._attempt(headers, data, attempt)
            if result.error_kind != ERROR_TRANSPORT:
                return result
        return result

    def _attempt(self, headers: Mapping[str, str], data: bytes, attempt: int) -> AdapterResult:
        secrets = (str(headers.get("x-api-key") or ""),)
        started = time.perf_counter()
        try:
            status, text = self._transport(self.url, headers, data, self.timeout_seconds)
        except (OSError, ValueError) as exc:
            return self._result(
                error=_redact(f"{type(exc).__name__}: {exc}", secrets=secrets),
                error_kind=ERROR_TRANSPORT,
                attempts=attempt,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if not isinstance(payload, dict):
            # A proxy or gateway may answer an auth failure or a rate limit in HTML or plain
            # text.  The status alone decides those, so they stop instead of being retried.
            if status == 200:
                kind = ERROR_MODEL_FAILURE
            elif status in (401, 403, 429):
                kind = _api_error_kind(status, "", "")
            else:
                kind = ERROR_TRANSPORT
            return self._result(
                error=f"HTTP {status}: response is not a JSON object",
                error_kind=kind,
                attempts=attempt,
                latency_ms=latency_ms,
            )
        if status != 200:
            error = payload.get("error") if isinstance(payload.get("error"), Mapping) else {}
            error_type = str(error.get("type") or "")
            message = _redact(error.get("message") or "", secrets=secrets)
            return self._result(
                error=f"HTTP {status} {error_type}: {message}".strip(),
                error_kind=_api_error_kind(status, error_type, message),
                attempts=attempt,
                latency_ms=latency_ms,
            )
        usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        stop_reason = str(payload.get("stop_reason") or "")
        reply = "".join(
            str(block.get("text") or "")
            for block in payload.get("content") or []
            if isinstance(block, Mapping) and block.get("type") == "text"
        )
        common = {
            "served_model_id": str(payload.get("model") or ""),
            "prompt_tokens": int(usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("output_tokens") or 0),
            "finish_reason": stop_reason,
            "attempts": attempt,
            "latency_ms": latency_ms,
        }
        if stop_reason == "refusal":
            return self._result(
                text=reply,
                error="the provider declined this message (stop_reason=refusal)",
                error_kind=ERROR_PROVIDER_REFUSAL,
                **common,
            )
        if not reply.strip():
            return self._result(
                error="empty model content (recorded as a model failure)",
                error_kind=ERROR_MODEL_FAILURE,
                **common,
            )
        return self._result(text=reply, **common)


@dataclass
class AdapterPool:
    """The adapters attached to one episode, keyed by principal."""

    orchestrator: Any
    peers: dict[str, Any] = field(default_factory=dict)

    def for_principal(self, principal: str) -> Any:
        if principal in self.peers:
            return self.peers[principal]
        raise AdapterError(f"no adapter for principal {principal!r}")

    def peer_for_slot(self, slot: str, peers: list) -> Any:
        for peer in peers:
            if peer.slot == slot:
                return self.peers[peer.principal]
        raise AdapterError(f"no adapter for peer slot {slot!r}")
