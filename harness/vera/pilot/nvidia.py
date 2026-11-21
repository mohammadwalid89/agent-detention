"""Free NVIDIA NIM chat route.

NVIDIA's hosted ``integrate.api.nvidia.com`` endpoint is a free (rate-limited) route.  The
credential is read **only** from ``NVIDIA_API_KEY`` in the process environment.  This module
does not read key files, does not copy the key into a result or artifact, and redacts it from
every exception message.  It never calls a paid proxy.

The free tier is throttled to at most 30 requests per minute (the lead reports a ~40/min
limit).  HTTP 429 is retried with backoff; a caller can stop cleanly by not retrying.

JOB12 declared pre-pilot wiring amendment (decided from the lead's 2026-11-14T04:33Z
wiring probe, not from any pilot outcome):

* both NVIDIA models are requested with ``max_tokens = 2048`` and a 120 s request timeout;
* HTTP 200 with empty content and ``finish_reason == 'length'`` is a **recorded model
  failure** for that step, not a transport error, and is never retried with a larger
  token budget;
* a read timeout is retried **once** with the identical request, then recorded as a
  transport error.

JOB13 declared pre-pilot wiring amendment #2 (supersedes only the token budget):

* both NVIDIA models are requested with ``max_tokens = 4096`` (same budget for both, for
  fairness); the 120 s timeout and the single identical-body retry on a read timeout are
  unchanged;
* JOB12's 2048 value is kept as declared history in :func:`nvidia_request_amendment`.

The scoring side of JOB13 lives in ``vera.pilot.runtime``: a model that returns no usable
action is a scored FAIL, never INVALID.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_CHAT_PATH = "/chat/completions"
NVIDIA_KEY_ENV = "NVIDIA_API_KEY"
NVIDIA_PROVIDER_FAMILY = "nvidia_nim"
NVIDIA_MAX_REQUESTS_PER_MINUTE = 30
NVIDIA_MIN_INTERVAL_SECONDS = 60.0 / NVIDIA_MAX_REQUESTS_PER_MINUTE
# JOB14: optional one-worker-per-model parallelism.  Each worker gets its own client and
# its own throttle; the per-worker interval is widened so the JOINT ceiling stays at the
# declared 30 requests/min (2 workers x 15 rpm = 30 rpm, i.e. one request every 4 s each).
NVIDIA_PARALLEL_WORKERS = 2
NVIDIA_PARALLEL_MIN_INTERVAL_SECONDS = (
    NVIDIA_MIN_INTERVAL_SECONDS * NVIDIA_PARALLEL_WORKERS
)
# JOB08's declared floor, kept unchanged for the preregistered plan and route records.
NVIDIA_MIN_MAX_TOKENS = 512
# JOB12's declared request settings, kept as declared history.
NVIDIA_JOB12_REQUEST_MAX_TOKENS = 2048
# JOB13 declared request settings for BOTH models (pre-pilot wiring amendment #2).
NVIDIA_REQUEST_MAX_TOKENS = 4096
NVIDIA_REQUEST_TIMEOUT_SECONDS = 120
NVIDIA_TIMEOUT_RETRIES = 1
DEFAULT_TIMEOUT_SECONDS = NVIDIA_REQUEST_TIMEOUT_SECONDS
DEFAULT_MAX_RETRIES = 2

# Internal ids keep the provider prefix so the inventory and plan are unambiguous.  The API
# model id is the part after ``nvidia/``.
NVIDIA_MODEL_IDS = (
    "nvidia/openai/gpt-oss-20b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
)
NVIDIA_SLOT_MODELS = {
    "FREE_M1": NVIDIA_MODEL_IDS[0],
    "FREE_M2": NVIDIA_MODEL_IDS[1],
}

# The plan's internal ids keep the provider prefix so the route is unambiguous.  The NIM
# API id is *usually* the internal id minus that prefix, but not always: the lead's
# 2026-11-13 probe lists the second model as ``nvidia/nemotron-3.5-lightning-30b-a3b``, so
# its API id keeps the ``nvidia/`` organisation prefix.  Stripping blindly sent the JOB08
# smoke to ``nemotron-3.5-lightning-30b-a3b`` and the endpoint answered HTTP 404.  The
# mapping is explicit and total for the preregistered ids; unknown ids keep the old rule.
NVIDIA_API_MODEL_IDS = {
    NVIDIA_MODEL_IDS[0]: "openai/gpt-oss-20b",
    NVIDIA_MODEL_IDS[1]: "nvidia/nemotron-3.5-lightning-30b-a3b",
}
NVIDIA_MODEL_NOTES = {
    NVIDIA_MODEL_IDS[0]: (
        "internal id -> API id openai/gpt-oss-20b; lead probe 2026-11-13 16:58 PT returned 200"
    ),
    NVIDIA_MODEL_IDS[1]: (
        "internal id -> API id nvidia/nemotron-3.5-lightning-30b-a3b; lead probe "
        "2026-11-13 16:58 PT returned 200 (the API id keeps the nvidia/ prefix)"
    ),
}

_MODEL_RE = re.compile(
    r"^nvidia/[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*$"
)
_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)


class NvidiaError(Exception):
    """Base class for NVIDIA route failures; messages contain no credential."""


class NvidiaKeyMissing(NvidiaError):
    """NVIDIA_API_KEY is not in the environment; no request was made."""


class NvidiaCallError(NvidiaError):
    """The NVIDIA endpoint failed or returned an unusable response."""

    error_kind = "CALL_ERROR"


class NvidiaModelFailure(NvidiaCallError):
    """HTTP 200 with no usable answer: a model failure recorded against the step.

    The JOB12 declaration separates this from a transport error: the request and the
    connection worked, the model spent its whole token budget on reasoning and returned
    empty content with ``finish_reason == 'length'``.  It is counted as a recorded model
    failure for that step, never retried with a larger budget.
    """

    error_kind = "MODEL_FAILURE"


class NvidiaResponseError(NvidiaCallError):
    """HTTP 200 whose payload contains no usable action (JOB13).

    JOB12 kept this as an ordinary call error; JOB13 (scoring integrity) classifies any
    HTTP 200 response that yields no action text as a *model* failure, because the
    request and connection worked and the model produced nothing usable.  It is never an
    infrastructure fault, so it can never make an episode INVALID.  It remains a
    ``NvidiaCallError`` subclass so JOB12's declared type checks keep holding.
    """

    error_kind = "MODEL_FAILURE"


class NvidiaTransportError(NvidiaCallError):
    """The request did not complete (read timeout, unreachable host, reset connection)."""

    error_kind = "TRANSPORT_ERROR"

    def __init__(self, message: str, *, timed_out: bool = False):
        super().__init__(message)
        self.timed_out = bool(timed_out)


@dataclass(frozen=True)
class NvidiaResult:
    text: str
    requested_model: str
    served_model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    raw_usage: dict = field(default_factory=dict)
    finish_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "requested_model": self.requested_model,
            "served_model": self.served_model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "finish_reason": self.finish_reason,
            "cost_usd": 0.0,
            "paid_call": False,
        }


def nvidia_request_amendment() -> dict:
    """The JOB12 declared pre-pilot wiring amendment, reused by both manifests.

    Historical record only: JOB13's :func:`nvidia_budget_amendment` supersedes the token
    budget.  The stored value (2048) is intentionally not updated here so the JOB12
    declaration stays auditable.
    """
    return {
        "amendment_id": "JOB12_NVIDIA_REQUEST_SETTINGS",
        "job_id": "JOB12",
        "declared_at_utc": "2026-11-14T04:10:00Z",
        "status": "DECLARED_PRE_PILOT_WIRING_AMENDMENT",
        "decision_basis": (
            "the lead's 2026-11-14T04:33Z wiring probe on the same key, not any pilot "
            "outcome; appended as an amendment, earlier JOB08 fields are preserved"
        ),
        "applies_to": list(NVIDIA_MODEL_IDS),
        "api_model_ids": {
            model_id: api_model_id(model_id) for model_id in NVIDIA_MODEL_IDS
        },
        "request": {
            "max_tokens": NVIDIA_JOB12_REQUEST_MAX_TOKENS,
            "timeout_seconds": NVIDIA_REQUEST_TIMEOUT_SECONDS,
            "timeout_retries": NVIDIA_TIMEOUT_RETRIES,
            "timeout_retry_uses_identical_request": True,
            "max_tokens_increased_on_retry": False,
        },
        "error_handling": {
            "empty_content_finish_reason_length": {
                "condition": (
                    "HTTP 200, empty message content and finish_reason == 'length'"
                ),
                "classification": "MODEL_FAILURE",
                "counted_as": "a recorded model failure for that step",
                "counted_as_transport_error": False,
                "retry": "none",
                "budget_changed_on_retry": False,
            },
            "read_timeout": {
                "condition": "the request times out while reading the response",
                "retry": "one retry with the identical request body",
                "after_retry": "recorded as a transport error",
                "classification": "TRANSPORT_ERROR",
            },
        },
        "probe_evidence": {
            "by": "Claude (VERA lead)",
            "at_utc": "2026-11-14T04:33:00Z",
            "prompt": "Pick one action from [read_ticket, update_ticket]...",
            "nemotron_max_tokens_512": "timed out",
            "nemotron_max_tokens_2048": {
                "latency_s": 2.2,
                "finish_reason": "stop",
                "content": "read_ticket",
                "completion_tokens": 441,
                "note": "reasoning plus answer; 512 is too small for this model",
            },
            "system_no_think_with_512": {
                "latency_s": 1.7,
                "content": "read_ticket",
                "completion_tokens": 359,
                "note": "still reasoned; not used as the fix",
            },
            "chat_template_kwargs_enable_thinking_false": "timed out",
        },
        "history_preserved": True,
        "note": (
            "Request settings are wiring, not design: no plan, instance, label, gate, "
            "threshold or scoring change.  The stored JOB08 fields remain unedited."
        ),
    }


def nvidia_budget_amendment() -> dict:
    """JOB13 declared pre-pilot wiring amendment #2: both models get 4096 tokens.

    Decided from the `$0` JOB12 smoke evidence (both models called, 0 transport errors,
    nemotron hit the 2048-token ceiling with empty content at `finish_reason='length'`),
    not from any pilot outcome.  Same budget for both models keeps the comparison fair;
    timeout and retry policy are unchanged from JOB12.
    """
    return {
        "amendment_id": "JOB13_NVIDIA_TOKEN_BUDGET",
        "amendment_number": 2,
        "job_id": "JOB13",
        "declared_at_utc": "2026-11-14T04:45:00Z",
        "status": "DECLARED_PRE_PILOT_WIRING_AMENDMENT",
        "supersedes": {
            "amendment_id": "JOB12_NVIDIA_REQUEST_SETTINGS",
            "field": "request.max_tokens",
            "value": NVIDIA_JOB12_REQUEST_MAX_TOKENS,
        },
        "evidence": {
            "path": "artifacts/checkpoints/JOB12/nvidia-smoke/nvidia_smoke.json",
            "job_id": "JOB12",
            "route_proven": True,
            "transport_errors": 0,
            "paid_calls": 0,
            "cost_usd": 0.0,
            "observed": (
                "nemotron F1-M02-r2-V1 returned empty content with "
                "finish_reason='length' at max_tokens=2048 (NvidiaModelFailure, step 1)"
            ),
        },
        "decision_basis": (
            "scoring integrity + fairness: a 2048-token budget was consumed entirely by "
            "reasoning on the free route, so both NVIDIA models get the same declared "
            "4096-token budget.  No plan, instance, label, gate, threshold or scoring "
            "rule is changed.  Earlier JOB08/JOB12 fields are preserved unedited."
        ),
        "applies_to": list(NVIDIA_MODEL_IDS),
        "request": {
            "max_tokens": NVIDIA_REQUEST_MAX_TOKENS,
            "same_budget_for_both_models": True,
            "timeout_seconds": NVIDIA_REQUEST_TIMEOUT_SECONDS,
            "timeout_retries": NVIDIA_TIMEOUT_RETRIES,
            "timeout_retry_uses_identical_request": True,
            "max_tokens_increased_on_retry": False,
        },
        "error_handling": {
            "empty_content_finish_reason_length": {
                "classification": "MODEL_FAILURE",
                "counted_as": "a recorded model failure for that step",
                "scored_as": "FAIL/PARTIAL episode that stays in the denominator (JOB13)",
                "counted_as_transport_error": False,
                "retry": "none",
            },
            "no_usable_action_at_http_200": {
                "classification": "MODEL_FAILURE",
                "counted_as": "a recorded model failure for that step",
                "scored_as": "FAIL/PARTIAL episode that stays in the denominator (JOB13)",
                "counted_as_transport_error": False,
                "retry": "none",
            },
            "read_timeout": {
                "classification": "TRANSPORT_ERROR",
                "retry": "one retry with the identical request body",
                "after_retry": "recorded as a transport error; episode INVALID (harness)",
            },
        },
        "history_preserved": True,
        "probe_evidence": {
            "source": "artifacts/checkpoints/JOB12/nvidia-smoke/nvidia_smoke.json",
            "completion_tokens_by_model": {
                "nvidia/openai/gpt-oss-20b": 4354,
                "nvidia/nemotron-3.5-lightning-30b-a3b": 2938,
            },
            "note": (
                "one nemotron reply spent the full 2048-token budget on reasoning and "
                "returned empty content at finish_reason='length'; a single 441-token "
                "reply was measured in the lead's 04:33Z probe"
            ),
        },
    }


def validate_nvidia_model_id(model_id: str) -> str:
    """Accept only ``nvidia/<api-model>``; reject path/query tricks."""
    if (
        not isinstance(model_id, str)
        or not _MODEL_RE.match(model_id)
        or ".." in model_id
    ):
        raise NvidiaCallError(
            "nvidia model id must look like nvidia/<model>; no request was made"
        )
    return model_id


def api_model_id(model_id: str) -> str:
    """Map an internal ``nvidia/<...>`` id to the model id the NIM API expects."""
    validate_nvidia_model_id(model_id)
    mapped = NVIDIA_API_MODEL_IDS.get(model_id)
    if mapped is not None:
        return mapped
    return model_id[len("nvidia/"):]


def _estimate_tokens(messages: list[dict]) -> int:
    chars = sum(len(str(message.get("content", ""))) for message in messages)
    return max(1, chars // 4 + 8)


class _UrllibNvidiaTransport:
    """Default transport.  The credential is attached to the request only."""

    def __init__(self, base_url: str = NVIDIA_BASE_URL, timeout: int = DEFAULT_TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post(self, path: str, body: dict, key: str) -> tuple[int, dict]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                status = int(response.status)
        except urllib.error.HTTPError as exc:
            raw = ""
            try:
                raw = exc.read().decode("utf-8", errors="replace")
            except Exception:  # pragma: no cover - body may be consumed already
                raw = ""
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {"error": raw[:300]}
            return int(exc.code), payload if isinstance(payload, dict) else {"error": raw[:300]}
        except urllib.error.URLError as exc:
            timed_out = isinstance(exc.reason, (TimeoutError, socket.timeout))
            raise NvidiaTransportError(
                f"NVIDIA unreachable: {type(exc.reason).__name__}",
                timed_out=timed_out,
            ) from None
        except (TimeoutError, socket.timeout):
            raise NvidiaTransportError(
                f"NVIDIA read timeout after {self.timeout} s", timed_out=True
            ) from None
        except OSError as exc:
            raise NvidiaTransportError(f"NVIDIA unreachable: {type(exc).__name__}") from None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise NvidiaCallError("NVIDIA returned non-JSON") from None
        if not isinstance(payload, dict):
            raise NvidiaCallError("NVIDIA returned a non-object JSON payload")
        return status, payload


class NvidiaClient:
    """Key-safe, throttled client for the free NVIDIA NIM route."""

    provider_family = NVIDIA_PROVIDER_FAMILY
    base_url = NVIDIA_BASE_URL

    def __init__(
        self,
        *,
        base_url: str = NVIDIA_BASE_URL,
        transport: Any = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        min_interval_seconds: float = NVIDIA_MIN_INTERVAL_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        sleep: Any = time.sleep,
        clock: Any = time.monotonic,
    ):
        self.base_url = base_url.rstrip("/")
        self.transport = transport or _UrllibNvidiaTransport(self.base_url, timeout)
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self.max_retries = max(0, int(max_retries))
        self._sleep = sleep
        self._clock = clock
        self._last_call: float | None = None
        self.calls = 0

    def _key(self) -> str:
        key = os.environ.get(NVIDIA_KEY_ENV, "")
        if not key:
            raise NvidiaKeyMissing(
                f"{NVIDIA_KEY_ENV} is not set; the NVIDIA route refuses to guess a key"
            )
        return key

    def _redact(self, text: Any, key: str) -> str:
        safe = _BEARER_RE.sub("Bearer <redacted>", str(text))
        if key:
            safe = safe.replace(key, "<redacted>")
        return safe

    def _throttle(self) -> None:
        if self._last_call is not None and self.min_interval_seconds > 0.0:
            wait = self.min_interval_seconds - (self._clock() - self._last_call)
            if wait > 0.0:
                self._sleep(wait)
        self._last_call = self._clock()

    @staticmethod
    def _retry_after_seconds(payload: dict, attempt: int) -> float:
        for header in ("retry_after", "retry-after"):
            value = payload.get(header)
            if value is not None:
                try:
                    return max(0.0, float(value))
                except (TypeError, ValueError):
                    pass
        return float(min(30, 2 ** (attempt + 1)))

    def run(
        self,
        model_id: str,
        messages: list[dict],
        *,
        max_tokens: int = NVIDIA_REQUEST_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> NvidiaResult:
        key = self._key()
        api_id = api_model_id(model_id)
        # JOB13: both NVIDIA models get the declared 4096-token request budget (same
        # budget for both).  A caller asking for less (the plan asks for 384/512 output
        # tokens) is raised to the declared budget; a retry never raises it further.
        token_cap = max(int(max_tokens), NVIDIA_REQUEST_MAX_TOKENS)
        body = {
            "model": api_id,
            "messages": messages,
            "max_tokens": token_cap,
            "temperature": float(temperature),
            "stream": False,
        }
        last_error = ""
        retries_left = self.max_retries
        timeout_retries_left = NVIDIA_TIMEOUT_RETRIES
        retry_index = 0
        while True:
            self._throttle()
            started = time.perf_counter()
            try:
                status, payload = self.transport.post(NVIDIA_CHAT_PATH, body, key)
            except NvidiaTransportError as exc:
                # JOB12: a read timeout gets exactly one retry with the identical body;
                # after that it is a recorded transport error.  Other reachability
                # failures keep the ordinary retry budget.
                if exc.timed_out and timeout_retries_left > 0:
                    timeout_retries_left -= 1
                    continue
                raise NvidiaTransportError(
                    self._redact(str(exc), key), timed_out=exc.timed_out
                ) from None
            except NvidiaCallError as exc:
                last_error = self._redact(str(exc), key)
                if retries_left > 0:
                    retries_left -= 1
                    self._sleep(float(min(30, 2 ** (retry_index + 1))))
                    retry_index += 1
                    continue
                raise NvidiaCallError(last_error) from None
            latency_ms = (time.perf_counter() - started) * 1000.0
            if status == 429:
                last_error = f"NVIDIA HTTP 429 rate limited (model {api_id!r})"
                if retries_left > 0:
                    self._sleep(self._retry_after_seconds(payload, retry_index))
                    retries_left -= 1
                    retry_index += 1
                    continue
                raise NvidiaCallError(
                    f"{last_error} after {retry_index + 1} attempts"
                )
            if status >= 500 and retries_left > 0:
                last_error = f"NVIDIA HTTP {status} (model {api_id!r})"
                self._sleep(float(min(30, 2 ** (retry_index + 1))))
                retries_left -= 1
                retry_index += 1
                continue
            if status != 200:
                detail = payload.get("error") or payload.get("detail") or payload.get("errors") or ""
                if isinstance(detail, (dict, list)):
                    detail = json.dumps(detail, sort_keys=True)
                # Name the requested API model: without it a 404 cannot be told apart from
                # a bad model id, a bad route or a retired model.
                raise NvidiaCallError(
                    f"NVIDIA HTTP {status} (model {api_id!r}): "
                    f"{self._redact(detail, key)[:300]}"
                )
            self.calls += 1
            try:
                choice = (payload.get("choices") or [])[0]
                text = choice["message"]["content"]
            except (KeyError, IndexError, TypeError):
                raise NvidiaResponseError(
                    "NVIDIA response has no choices[0].message.content"
                ) from None
            finish_reason = ""
            if isinstance(choice, dict):
                finish_reason = str(choice.get("finish_reason") or "")
            text_value = "" if text is None else str(text)
            if not text_value.strip():
                # JOB12: an HTTP 200 with empty content because the token budget ran out
                # is a model failure for this step.  It is not retried with a larger
                # budget and it is not reported as a transport error.
                if finish_reason == "length":
                    raise NvidiaModelFailure(
                        f"NVIDIA model {api_id!r} returned empty content with "
                        f"finish_reason='length' at max_tokens={token_cap}; recorded as a "
                        "model failure for this step (no larger-budget retry)"
                    )
                # JOB13: any other HTTP 200 without usable action text is a model failure
                # too (the request worked; the model produced nothing usable), so it is
                # scored FAIL/PARTIAL and never makes an episode INVALID.
                raise NvidiaResponseError(
                    "NVIDIA response has no message content "
                    f"(finish_reason={finish_reason!r}); recorded as a model failure for "
                    "this step"
                )
            usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            return NvidiaResult(
                text=text_value,
                requested_model=model_id,
                served_model=str(payload.get("model") or api_id),
                prompt_tokens=(
                    int(prompt_tokens) if prompt_tokens is not None
                    else _estimate_tokens(messages)
                ),
                completion_tokens=(
                    int(completion_tokens) if completion_tokens is not None
                    else max(1, len(text_value) // 4)
                ),
                latency_ms=latency_ms,
                raw_usage=usage,
                finish_reason=finish_reason,
            )
        raise NvidiaCallError(last_error or "NVIDIA call failed")
