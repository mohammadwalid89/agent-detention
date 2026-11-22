"""v1.3 grid runner for the model routes in the route registry.

The runner executes the *frozen* assignment table produced by :mod:`runplan_v13`.  It does
not choose tasks, conditions, models or labels.  It adds the operational layer the design
requires around the already-built v1.3 mechanism:

* one paced worker pool per route (concurrency and min interval are configuration data from
  the measured route smoke, never constants in this module);
* an append-only episode ledger keyed by every assignment factor, so a restart is idempotent
  and a completed episode is never executed twice;
* ``ROUTE_LIMIT`` pauses only that route, persists the pause, auto-resumes at the configured
  time or on command, and leaves other routes running; ``AUTH_EXPIRED`` stops the whole run;
* the frozen exception rule: a model failure is a scored episode, while an infrastructure
  failure is recorded separately and is never silently converted into a model result;
* a stage gate that rejects a live main-grid start unless a valid preregistration root hash,
  a reviewer acceptance file and an owner approval file are supplied;
* a live progress file and a capacity planner over measured calls/minute and limit windows.

No command in this module makes a model call by default.  The live path constructs the
route adapters; ``--dry-run`` uses ``MockAdapter`` and needs no stage gate.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import hmac
import json
import math
import os
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..canonical import canonical_json, sha256_text
from .adapters import (
    ERROR_AUTH_EXPIRED,
    ERROR_CONFIG,
    ERROR_KEY_MISSING,
    ERROR_PROVIDER_REFUSAL,
    ERROR_ROUTE_LIMIT,
    ERROR_TRANSPORT,
    SERVED_MODEL_MISMATCH,
    AdapterResult,
    EpisodeCallLog,
    RecordingAdapter,
    served_model_check,
    verify_call_log,
)
from .constants import (
    COND_DODGE,
    CONDITIONS,
    CUE_EXPLICIT,
    CUE_IMPLICIT,
    IDENTITY_CONDITIONS,
    SLOT_SELF,
)
from .identity_assignment import first_choice_outcome
from .policies import default_mock_adapters
from .preregistration import verify_bundle
from .principals import PeerSpec
from .routes.registry import (
    adapter_for_route,
    load_registry,
    route_by_id,
    route_requires_cli_isolation,
    sha256_file,
)
from .runplan_v13 import (
    ENROLLED_PATTERNS,
    PATTERN_BASE,
    PATTERN_DISPLAY,
    build_assignment_table,
)
from .source_binding import (
    COMPLETE_STATUSES,
    PENDING_STATUSES,
    PRE_MAIN_STAGES,
    SourceBindingError,
    resolve_source_binding,
    verify_main_assignment,
)

RUNNER_KIND = "vera.v13_grid_runner.v1"
CONFIG_KIND = "vera.v13_runner_config.v1"
LEDGER_KIND = "vera.v13_episode_ledger.v1"
PROGRESS_KIND = "vera.v13_runner_progress.v1"
RESULT_KIND = "vera.v13_episode_result.v1"
SUMMARY_KIND = "vera.v13_runner_summary.v1"
GATE_KIND = "vera.v13_stage_gate.v1"
SHAKEDOWN_GATE_KIND = "vera.v13_shakedown_gate.v1"
PREFLIGHT_RECORD_KIND = "vera.v13_runner_preflight.v1"
SIGNATURE_ALGORITHM = "HMAC-SHA256"
DETENTION_SUMMARY_KIND = "vera.v13_detention_summary.v1"
DETENTION_RESULT_KIND = "vera.v13_detention_result.v1"

RUN_SCOPE_MAIN_GRID = "main_grid"
RUN_SCOPE_SHAKEDOWN = "shakedown"
RUN_SCOPE_DRY_RUN = "dry_run"
#: The Detention pilot.  Not the main grid, never pooled with it; requires transcripts,
#: measured routes, the Claude-only profile and an owner-signed case text; labels every record.
RUN_SCOPE_PILOT = "pilot"
PILOT_LABEL = "PILOT"
PILOT_GATE_KIND = "vera.v14_pilot_gate.v1"
RUN_SCOPES = (RUN_SCOPE_MAIN_GRID, RUN_SCOPE_SHAKEDOWN, RUN_SCOPE_DRY_RUN, RUN_SCOPE_PILOT)
MAIN_GRID_SCOPES = (RUN_SCOPE_MAIN_GRID,)

DEFAULT_LEDGER = "episode_ledger.jsonl"
DEFAULT_STATE = "runner_state.json"
DEFAULT_PROGRESS = "progress.json"
DEFAULT_ASSIGNMENT = "assignment_table.json"
DEFAULT_SUMMARY = "run_summary.json"
DEFAULT_EPISODE_DIR = "episodes"
DEFAULT_EPISODE_CALL_DIR = "episode_calls"
#: How many times one Detention row is re-queued after a route limit before the
#: phase stops incomplete (the main phase re-queues without a cap; a mock that never recovers
#: must not spin forever).
DETENTION_MAX_REQUEUES = 20
DEFAULT_PREFLIGHT = "preflight.json"
DEFAULT_LOCK = ".runner.lock"
DEFAULT_DETENTION_DIR = "detention"

# The only hard operational cap in this module: the owner's free NVIDIA route is limited to
# 30 requests/minute.  Measured rates below this value are used as given.
NVIDIA_MAX_REQUESTS_PER_MINUTE = 30.0

# The frozen exception rule.  These are infrastructure outcomes, not model
# failures.  ROUTE_LIMIT and AUTH_EXPIRED have their own run-level behaviour.
INFRASTRUCTURE_KINDS = (
    ERROR_TRANSPORT,
    ERROR_CONFIG,
    ERROR_KEY_MISSING,
)
FATAL_INFRASTRUCTURE_KINDS = (ERROR_CONFIG, ERROR_KEY_MISSING)
MODEL_SUBSTITUTION = "MODEL_SUBSTITUTION"
RUN_STOPPED = "RUN_STOPPED"
HARNESS_FAULT = "HARNESS_FAULT"

# Every independent assignment factor participates in the resume key.  The frozen table's
# shorter ``resume_key`` is kept alongside it for traceability.
RESUME_KEY_FACTORS = (
    "model_slot",
    "instance_id",
    "condition",
    "organization_pattern",
    "dodge_cue",
    "identity_condition",
    "repeat",
    "seed",
    "menu_position_seed",
    "label_permutation",
)

_CURRENT_CONTEXT = threading.local()


class RunnerConfigError(ValueError):
    """The configuration or the frozen table is internally inconsistent."""


class LedgerIntegrityError(RunnerConfigError):
    """An existing signed ledger does not verify; refusing to append to it."""


class RunDirectoryLocked(RuntimeError):
    """Another live runner owns this run directory."""


class StageGateError(RuntimeError):
    """A live main-grid run was attempted without valid stage-gate evidence."""


class RunStopped(RuntimeError):
    """The episode called a route after the whole run had been stopped."""


# Small filesystem / time helpers


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path: str | Path, payload: Mapping | Sequence) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)


def append_jsonl(path: str | Path, payload: Mapping) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | Path) -> tuple[list[dict], list[str]]:
    """Read an append-only ledger, tolerating one torn final line after a crash."""
    target = Path(path)
    if not target.is_file():
        return [], []
    records: list[dict] = []
    problems: list[str] = []
    for index, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {index}: unreadable JSON ({exc})")
            continue
        if not isinstance(record, dict):
            problems.append(f"line {index}: record is not an object")
            continue
        records.append(record)
    return records, problems


def safe_component(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(value)
    )
    return cleaned[:120] or "episode"


def json_digest(payload: Any) -> str:
    """A runner-local digest that tolerates operational floats.

    The scientific containers keep using ``canonical_json`` (no floats).  Runner metadata
    contains measured seconds/rates, so it uses a stable JSON encoding instead.
    """
    return sha256_text(json.dumps(payload, sort_keys=True, default=str))


def percentile_ms(samples: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile over millisecond samples (deterministic, no interpolation)."""
    values = sorted(float(value) for value in samples)
    if not values:
        return None
    rank = max(1, int(math.ceil(fraction * len(values))))
    return round(values[min(len(values), rank) - 1], 3)


def summarise_latencies(samples: Sequence[float]) -> dict:
    """Per-call latency summary reported in progress and signed summaries."""
    values = [float(value) for value in samples]
    if not values:
        return {
            "samples": 0,
            "p50_ms": None,
            "p95_ms": None,
            "max_ms": None,
            "mean_ms": None,
        }
    return {
        "samples": len(values),
        "p50_ms": percentile_ms(values, 0.50),
        "p95_ms": percentile_ms(values, 0.95),
        "max_ms": round(max(values), 3),
        "mean_ms": round(sum(values) / len(values), 3),
    }


def _signing_bytes(payload: Mapping) -> bytes:
    """Stable bytes signed by the runner (no signature fields, no float ambiguity)."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


SIGNATURE_FIELDS = (
    "signature_hmac",
    "signature_alg",
    "signing_key_id",
    "record_sha256",
    "signature",
)


def derive_dry_run_key(run_id: str, assignment_digest: str) -> bytes:
    """A reproducible integrity key for offline/mock runs.

    Dry runs make no model call and carry no secret.  The key is still derived from the
    run identity so an auditor can re-verify the signed ledger from ``run_summary.json``
    without a key file.  Live runs must supply an explicit key (see ``GridRunner``).
    """
    material = f"vera.v13.runner.dry_run|{run_id}|{assignment_digest}"
    return hashlib.sha256(material.encode("utf-8")).digest()


def load_signing_key_file(path: str | Path) -> bytes:
    """Read a runner signing key: hex text or raw bytes from a 0600 file."""
    raw = Path(path).read_bytes()
    text = raw.strip()
    if text and len(text) % 2 == 0:
        try:
            decoded = bytes.fromhex(text.decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            decoded = b""
        if decoded:
            return decoded
    if not raw:
        raise RunnerConfigError("the signing key file is empty")
    return raw


class RunnerSigner:
    """HMAC signer for the append-only ledger and the result/summary payloads."""

    def __init__(self, key: bytes, *, mode: str):
        if not isinstance(key, (bytes, bytearray)) or not bytes(key):
            raise RunnerConfigError("a non-empty runner signing key is required")
        self.key = bytes(key)
        self.mode = str(mode)
        self.key_id = hashlib.sha256(self.key).hexdigest()[:16]

    def info(self) -> dict:
        return {
            "algorithm": SIGNATURE_ALGORITHM,
            "mode": self.mode,
            "key_id": self.key_id,
            "key_bytes": len(self.key),
        }

    def sign_digest(self, digest: str) -> str:
        return hmac.new(
            self.key, digest.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def sign_mapping(self, payload: Mapping, *, field_name: str = "signature") -> dict:
        body = {
            key: value
            for key, value in payload.items()
            if key not in SIGNATURE_FIELDS and key != field_name
        }
        digest = sha256_text(_signing_bytes(body).decode("utf-8"))
        signed = dict(payload)
        signed[field_name] = {
            "algorithm": SIGNATURE_ALGORITHM,
            "key_id": self.key_id,
            "mode": self.mode,
            "payload_sha256": digest,
            "hmac": self.sign_digest(digest),
        }
        return signed


def verify_signed_mapping(
    payload: Mapping, key: bytes, *, field_name: str = "signature"
) -> bool:
    signature = payload.get(field_name)
    if not isinstance(signature, Mapping):
        return False
    if signature.get("algorithm") != SIGNATURE_ALGORITHM:
        return False
    body = {
        name: value
        for name, value in payload.items()
        if name not in SIGNATURE_FIELDS and name != field_name
    }
    digest = sha256_text(_signing_bytes(body).decode("utf-8"))
    expected = hmac.new(
        bytes(key), digest.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return (
        hmac.compare_digest(str(signature.get("payload_sha256", "")), digest)
        and hmac.compare_digest(str(signature.get("hmac", "")), expected)
    )


def ledger_record_hash(record: Mapping) -> str:
    """The hash of a ledger record body; legacy unsigned records still chain."""
    recorded = str(record.get("record_sha256") or "")
    if recorded:
        return recorded
    body = {
        name: value
        for name, value in record.items()
        if name not in SIGNATURE_FIELDS
    }
    return sha256_text(_signing_bytes(body).decode("utf-8"))


def verify_ledger_records(records: Sequence[Mapping], key: bytes) -> dict:
    """Verify sequence, hash chain and HMACs for a signed ledger."""
    problems: list[str] = []
    previous = ""
    signed = 0
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            problems.append(f"record {index}: not an object")
            continue
        seq = record.get("seq")
        if seq is not None and int(seq) != index:
            problems.append(f"record {index}: seq {seq!r} is not {index}")
        if str(record.get("prev_sha256") or "") != previous:
            problems.append(f"record {index}: prev_sha256 does not chain")
        recorded = str(record.get("record_sha256") or "")
        signature = str(record.get("signature_hmac") or "")
        if recorded or signature:
            body = {
                name: value
                for name, value in record.items()
                if name not in SIGNATURE_FIELDS
            }
            digest = sha256_text(_signing_bytes(body).decode("utf-8"))
            if not hmac.compare_digest(recorded, digest):
                problems.append(f"record {index}: record_sha256 mismatch")
            expected = hmac.new(
                bytes(key), digest.encode("utf-8"), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(signature, expected):
                problems.append(f"record {index}: signature mismatch")
            signed += 1
        previous = ledger_record_hash(record)
    return {
        "ok": not problems,
        "records": len(records),
        "signed_records": signed,
        "problems": problems,
        "head_sha256": previous,
    }


def verify_ledger_file(path: str | Path, key: bytes) -> dict:
    records, problems = read_jsonl(path)
    report = verify_ledger_records(records, key)
    if problems:
        report = dict(report)
        report["ok"] = False
        report["problems"] = list(report.get("problems") or []) + problems
    report["path"] = str(path)
    return report



def assignment_table_digest(rows: Sequence[Mapping]) -> str:
    return json_digest([dict(row) for row in rows])


def assignment_resume_key(row: Mapping) -> str:
    """The idempotency key: every assignment factor, in a fixed order."""
    return "|".join(f"{name}={row.get(name, '')}" for name in RESUME_KEY_FACTORS)


# Configuration


@dataclass(frozen=True)
class RouteSettings:
    route_id: str
    concurrency: int
    min_interval_seconds: float
    measured_calls_per_min: float | None = None
    max_requests_per_minute: float | None = None
    limit_requests: int | None = None
    limit_window_seconds: float | None = None
    route_limit_retry_seconds: float | None = None
    raw: Mapping = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        route_id: str,
        data: Mapping,
        *,
        defaults: Mapping | None = None,
    ) -> "RouteSettings":
        defaults = dict(defaults or {})
        merged = {**defaults, **dict(data)}
        if "concurrency" not in merged:
            raise RunnerConfigError(
                f"route {route_id!r}: config must provide concurrency (no hidden default)"
            )
        if "min_interval_seconds" not in merged:
            raise RunnerConfigError(
                f"route {route_id!r}: config must provide min_interval_seconds "
                "(values come from the measured smoke)"
            )
        try:
            concurrency = int(merged["concurrency"])
            min_interval = float(merged["min_interval_seconds"])
        except (TypeError, ValueError):
            raise RunnerConfigError(
                f"route {route_id!r}: concurrency/min_interval_seconds must be numeric"
            ) from None
        if concurrency < 1:
            raise RunnerConfigError(f"route {route_id!r}: concurrency must be >= 1")
        if min_interval < 0:
            raise RunnerConfigError(
                f"route {route_id!r}: min_interval_seconds must be >= 0"
            )

        def optional_float(name: str) -> float | None:
            value = merged.get(name)
            if value is None or value == "":
                return None
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise RunnerConfigError(f"route {route_id!r}: {name} must be numeric") from None
            if number <= 0:
                raise RunnerConfigError(f"route {route_id!r}: {name} must be > 0")
            return number

        limit_requests = merged.get("limit_requests")
        if limit_requests in (None, ""):
            limit_requests = None
        else:
            try:
                limit_requests = int(limit_requests)
            except (TypeError, ValueError):
                raise RunnerConfigError(
                    f"route {route_id!r}: limit_requests must be an integer"
                ) from None
            if limit_requests <= 0:
                raise RunnerConfigError(f"route {route_id!r}: limit_requests must be > 0")
        limit_window = optional_float("limit_window_seconds")
        if (limit_requests is None) != (limit_window is None):
            raise RunnerConfigError(
                f"route {route_id!r}: limit_requests and limit_window_seconds are both "
                "required when a limit window is declared"
            )
        retry_after = optional_float("route_limit_retry_seconds")
        return cls(
            route_id=route_id,
            concurrency=concurrency,
            min_interval_seconds=min_interval,
            measured_calls_per_min=optional_float("measured_calls_per_min"),
            max_requests_per_minute=optional_float("max_requests_per_minute"),
            limit_requests=limit_requests,
            limit_window_seconds=limit_window,
            route_limit_retry_seconds=retry_after,
            raw=dict(merged),
        )

    def rate_cap(self, *, nvidia_route: bool = False) -> float | None:
        """The smallest declared requests/minute cap, if any cap is declared."""
        caps: list[float] = []
        if self.measured_calls_per_min is not None:
            caps.append(self.measured_calls_per_min)
        if self.max_requests_per_minute is not None:
            caps.append(self.max_requests_per_minute)
        if self.limit_requests is not None and self.limit_window_seconds:
            caps.append(self.limit_requests * 60.0 / self.limit_window_seconds)
        if self.min_interval_seconds > 0:
            caps.append(60.0 / self.min_interval_seconds)
        if nvidia_route:
            caps.append(NVIDIA_MAX_REQUESTS_PER_MINUTE)
        return min(caps) if caps else None

    def start_interval(self, *, nvidia_route: bool = False) -> float:
        rate = self.rate_cap(nvidia_route=nvidia_route)
        interval = 0.0 if rate is None else 60.0 / rate
        return max(self.min_interval_seconds, interval)


@dataclass(frozen=True)
class RunnerConfig:
    kind: str
    episode_workers: int
    infra_retries: int
    route_limit_retry_seconds: float | None
    routes: Mapping[str, RouteSettings]
    max_model_calls: int | None = None
    raw: Mapping = field(default_factory=dict)

    def route(self, route_id: str) -> RouteSettings:
        try:
            return self.routes[route_id]
        except KeyError:
            raise RunnerConfigError(
                f"route {route_id!r} is used by the assignment table but has no runner config"
            ) from None


def load_runner_config(
    config: Mapping | str | Path | None,
    *,
    registry: Mapping,
    dry_run: bool = False,
) -> RunnerConfig:
    if config is None:
        if not dry_run:
            raise RunnerConfigError(
                "a route config file is required for a live run "
                "(--config measured_router_config.json)"
            )
        # Dry-run still uses an explicit generated config, not measured values.  It never
        # contacts a provider, so a zero interval is a local mock pacing value only.
        raw: dict = {
            "kind": CONFIG_KIND,
            "episode_workers": 2,
            "infra_retries": 0,
            "routes": {
                route["route_id"]: {
                    "concurrency": 1,
                    "min_interval_seconds": 0.0,
                }
                for route in registry.get("routes", [])
            },
        }
    elif isinstance(config, (str, Path)):
        path = Path(config)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RunnerConfigError(f"runner config is unreadable: {exc}") from None
        except json.JSONDecodeError as exc:
            raise RunnerConfigError(f"runner config is not JSON: {exc}") from None
    else:
        raw = dict(config)
    if not isinstance(raw, dict):
        raise RunnerConfigError("runner config must be a JSON object")
    if raw.get("kind") not in (None, CONFIG_KIND):
        raise RunnerConfigError(f"runner config kind must be {CONFIG_KIND!r}")

    route_rows = raw.get("routes")
    if isinstance(route_rows, list):
        route_rows = {
            str(item.get("route_id")): item
            for item in route_rows
            if isinstance(item, dict) and item.get("route_id")
        }
    if not isinstance(route_rows, Mapping) or not route_rows:
        raise RunnerConfigError("runner config needs a non-empty routes object")
    defaults = dict(raw.get("defaults") or {})
    if not isinstance(defaults, Mapping):
        raise RunnerConfigError("runner config defaults must be an object")
    # A top-level retry time is the default auto-resume policy for every route; a per-route
    # value still wins.  Without either, a pause waits for an explicit resume command.
    if "route_limit_retry_seconds" not in defaults and raw.get("route_limit_retry_seconds") not in (None, ""):
        defaults["route_limit_retry_seconds"] = raw["route_limit_retry_seconds"]

    settings: dict[str, RouteSettings] = {}
    registry_routes = {route["route_id"]: route for route in registry.get("routes", [])}
    for route_id, item in route_rows.items():
        if not isinstance(item, Mapping):
            raise RunnerConfigError(f"route {route_id!r}: settings must be an object")
        route = registry_routes.get(str(route_id))
        if route is None:
            raise RunnerConfigError(
                f"route {route_id!r}: not present in the read-only D4 registry"
            )
        if (
            route.get("route") == "nvidia_nim"
            and item.get("max_requests_per_minute") is not None
            and float(item["max_requests_per_minute"]) > NVIDIA_MAX_REQUESTS_PER_MINUTE
        ):
            raise RunnerConfigError(
                f"route {route_id!r}: NVIDIA free routes are capped at "
                f"{NVIDIA_MAX_REQUESTS_PER_MINUTE:g} requests/minute"
            )
        settings[str(route_id)] = RouteSettings.from_mapping(
            str(route_id), item, defaults=defaults
        )

    try:
        episode_workers = int(raw.get("episode_workers", 4))
        infra_retries = int(raw.get("infra_retries", 0))
    except (TypeError, ValueError):
        raise RunnerConfigError("episode_workers/infra_retries must be integers") from None
    if episode_workers < 1:
        raise RunnerConfigError("episode_workers must be >= 1")
    if infra_retries < 0:
        raise RunnerConfigError("infra_retries must be >= 0")
    retry = raw.get("route_limit_retry_seconds")
    if retry in (None, ""):
        retry_value = None
    else:
        try:
            retry_value = float(retry)
        except (TypeError, ValueError):
            raise RunnerConfigError("route_limit_retry_seconds must be numeric") from None
        if retry_value <= 0:
            raise RunnerConfigError("route_limit_retry_seconds must be > 0")
    # The owner's call cap is a runner option, charged before every adapter call.
    max_calls = raw.get("max_model_calls")
    if max_calls in (None, ""):
        max_calls_value = None
    else:
        try:
            max_calls_value = int(max_calls)
        except (TypeError, ValueError):
            raise RunnerConfigError("max_model_calls must be an integer") from None
        if max_calls_value < 1:
            raise RunnerConfigError("max_model_calls must be >= 1")
    return RunnerConfig(
        kind=str(raw.get("kind") or CONFIG_KIND),
        episode_workers=episode_workers,
        infra_retries=infra_retries,
        route_limit_retry_seconds=retry_value,
        routes=settings,
        max_model_calls=max_calls_value,
        raw=raw,
    )


# Stage gate


def _load_json_object(path: str | Path, label: str) -> dict:
    target = Path(path)
    if not target.is_file():
        raise StageGateError(f"{label} file does not exist: {target}")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageGateError(f"{label} is not readable JSON: {exc}") from None
    if not isinstance(data, dict):
        raise StageGateError(f"{label} must be a JSON object")
    return data


def _affirmative(data: Mapping, label: str) -> None:
    accepted = (
        data.get("accepted") is True
        or data.get("approved") is True
        or data.get("status") in {"ACCEPTED", "PASS", "APPROVED", "ACCEPT"}
        or data.get("verdict") in {"ACCEPT", "ACCEPTED", "PASS", "APPROVED"}
    )
    if not accepted:
        raise StageGateError(f"{label} is not an affirmative acceptance/approval")


def _bound_hash(data: Mapping, expected: str, label: str) -> str:
    supplied = next(
        (
            str(data.get(field_name))
            for field_name in (
                "prereg_root_sha256",
                "root_sha256",
                "preregistration_root_sha256",
            )
            if data.get(field_name)
        ),
        "",
    )
    if not supplied:
        raise StageGateError(f"{label} does not bind the preregistration root hash")
    if supplied != expected:
        raise StageGateError(
            f"{label} binds preregistration root {supplied}, expected {expected}"
        )
    return supplied


def validate_pre_main_stage_completion(
    *, prereg_root: str | Path, root: Mapping | None = None
) -> dict:
    """Review finding 6: the main gate needs every pre-main row completed.

    The preregistration deliberately defers ``FREEZE_BEFORE_MAIN_RUN``,
    ``ROUTE_PREFLIGHT_BEFORE_FIRST_CALL`` and ``CALIBRATION_BEFORE_FIRST_MAIN_EPISODE``.
    A main-grid run is refused while any of those rows is still
    ``MECHANICAL_LATER_STAGE``, and the exact assignment hash must be present.  There is no
    flag that turns this check off.
    """
    root_path = Path(prereg_root)
    data = dict(root) if root is not None else _load_json_object(root_path, "preregistration root")
    rows = data.get("rows") or {}
    problems: list[str] = []
    completed: dict[str, str] = {}
    pending: dict[str, str] = {}
    for name, record in sorted(rows.items()):
        if not isinstance(record, Mapping):
            problems.append(f"row {name!r} is not an object")
            continue
        status = str(record.get("status") or "")
        stage = str(record.get("later_stage") or "")
        if stage not in PRE_MAIN_STAGES:
            continue
        if status in COMPLETE_STATUSES:
            completed[name] = status
        elif status in PENDING_STATUSES or status not in COMPLETE_STATUSES:
            pending[name] = f"{status or 'NO_STATUS'} at {stage}"
            problems.append(
                f"preregistration row {name!r} is not complete before the main grid: "
                f"{pending[name]}"
            )
    from .source_binding import preregistered_assignment_hash

    digest, digest_problem = preregistered_assignment_hash(root_path)
    if not digest:
        problems.append(f"preregistered assignment hash unavailable: {digest_problem}")
    if problems:
        raise StageGateError(
            "main-grid run refused: " + "; ".join(problems)
        )
    return {
        "kind": "vera.v13.pre_main_stage_completion.v1",
        "completed_rows": completed,
        "preregistered_assignment_hash": digest,
    }


def validate_stage_gate(
    *,
    prereg_root: str | Path | None,
    prereg_hash: str | None,
    reviewer_acceptance: str | Path | None,
    owner_approval: str | Path | None,
    now_utc: str | None = None,
) -> dict:
    """Validate the three live-run gate inputs.  Any defect refuses the run."""
    missing = [
        name
        for name, value in (
            ("preregistration ROOT_MANIFEST path", prereg_root),
            ("preregistration ROOT_MANIFEST hash", prereg_hash),
            ("reviewer acceptance file path", reviewer_acceptance),
            ("owner approval file path", owner_approval),
        )
        if not value
    ]
    if missing:
        raise StageGateError("live main-grid run refused; missing " + ", ".join(missing))

    root_path = Path(prereg_root)
    root = _load_json_object(root_path, "preregistration root manifest")
    root_for_hash = {key: value for key, value in root.items() if key != "root_sha256"}
    calculated = sha256_text(canonical_json(root_for_hash))
    if calculated != root.get("root_sha256"):
        raise StageGateError("preregistration root manifest root_sha256 does not verify")
    expected = str(prereg_hash).strip().lower()
    if calculated.lower() != expected:
        raise StageGateError(
            f"preregistration root hash {calculated} does not match supplied {expected}"
        )
    bundle_report = verify_bundle(root_path.parent)
    if not bundle_report.get("valid"):
        raise StageGateError(
            "preregistration bundle does not verify: "
            + "; ".join(bundle_report.get("problems") or ["unknown problem"])
        )

    acceptance_path = Path(reviewer_acceptance)
    acceptance = _load_json_object(acceptance_path, "reviewer acceptance")
    _affirmative(acceptance, "reviewer acceptance")
    reviewer = str(
        acceptance.get("reviewer") or acceptance.get("reviewed_by") or acceptance.get("accepted_by") or ""
    ).strip()
    if not reviewer:
        raise StageGateError("reviewer acceptance needs a named reviewer")
    _bound_hash(acceptance, calculated, "reviewer acceptance")
    if acceptance.get("kind") not in (None, "vera.reviewer_acceptance.v1"):
        raise StageGateError("reviewer acceptance kind is not vera.reviewer_acceptance.v1")

    owner_path = Path(owner_approval)
    owner = _load_json_object(owner_path, "owner approval")
    _affirmative(owner, "owner approval")
    owner_name = str(
        owner.get("owner") or owner.get("approved_by") or owner.get("reviewer") or ""
    ).strip()
    if not owner_name:
        raise StageGateError("owner approval needs a named owner")
    _bound_hash(owner, calculated, "owner approval")
    if owner.get("kind") not in (None, "vera.v13_owner_approval.v1"):
        raise StageGateError("owner approval kind is not vera.v13_owner_approval.v1")
    if owner.get("scope") not in (None, "", "main_grid", ["main_grid"]):
        raise StageGateError("owner approval scope does not include main_grid")
    if owner.get("main_grid") is False:
        raise StageGateError("owner approval explicitly excludes the main grid")
    expires_at = owner.get("expires_at")
    if expires_at:
        now = now_utc or utc_now()
        if str(expires_at) < now:
            raise StageGateError(
                f"owner approval expired at {expires_at} (now {now})"
            )

    # Review findings 1+6: the main gate is not just a hash-bound approval.  It also
    # requires the active source to BE the preregistered set-12 bytes, every pre-main
    # prerequisite row to be complete and the exact assignment hash to exist.  All three are
    # hard refusals; no optional flag can bypass them.
    try:
        source_binding = resolve_source_binding(prereg_path=root_path)
    except SourceBindingError as exc:
        raise StageGateError(str(exc)) from None
    completion = validate_pre_main_stage_completion(prereg_root=root_path, root=root)

    return {
        "kind": GATE_KIND,
        "valid": True,
        "validated_at_utc": utc_now(),
        "prereg_root_manifest": str(root_path),
        "prereg_root_sha256": calculated,
        "prereg_root_file_sha256": sha256_file(root_path),
        "reviewer_acceptance": str(acceptance_path),
        "reviewer_acceptance_sha256": sha256_file(acceptance_path),
        "acceptance_reviewer": reviewer,
        "owner_approval": str(owner_path),
        "owner_approval_sha256": sha256_file(owner_path),
        "owner": owner_name,
        "run_scope": RUN_SCOPE_MAIN_GRID,
        "source_binding": {
            "source_set": source_binding["source"]["source_set"],
            "source_sha256": source_binding["source"]["sha256"],
            "condition_manifest_digest": source_binding["condition_manifest_digest"],
            "prereg_version_id": source_binding["preregistration"]["version_id"],
        },
        "pre_main_completion": completion,
    }


def _validate_shakedown_gate_mapping(
    data: Mapping, *, now_utc: str | None = None
) -> dict:
    """Validate an honest non-main-grid shakedown gate.

    A shakedown gate does not claim a preregistration, a reviewer acceptance or a main-grid
    owner approval.  It must instead carry the owner's job reference, an explicit
    ``not_main_grid`` marker, a scope other than ``main_grid`` and an expiry.  The run
    records the scope, and :func:`assert_main_grid_summary` refuses to read it as
    main-grid data.
    """
    if not isinstance(data, Mapping):
        raise StageGateError("shakedown gate must be a JSON object")
    if data.get("kind") != SHAKEDOWN_GATE_KIND:
        raise StageGateError(f"shakedown gate kind must be {SHAKEDOWN_GATE_KIND!r}")
    if data.get("valid") is not True:
        raise StageGateError("shakedown gate is not marked valid: true")
    if data.get("not_main_grid") is not True:
        raise StageGateError("shakedown gate must set not_main_grid: true")
    scope = data.get("scope")
    if isinstance(scope, (list, tuple)):
        scope_values = [str(item) for item in scope]
        explicit_main = any(item == RUN_SCOPE_MAIN_GRID for item in scope_values)
    elif scope is None or str(scope).strip() == "":
        scope_values = []
        explicit_main = False
    else:
        scope_values = [str(scope)]
        explicit_main = str(scope) == RUN_SCOPE_MAIN_GRID
    if not scope_values:
        raise StageGateError("shakedown gate needs a scope")
    if explicit_main:
        raise StageGateError("a shakedown gate may not declare main_grid scope")
    job_reference = str(
        data.get("job")
        or data.get("job_reference")
        or data.get("owner_reference")
        or ""
    ).strip()
    if not job_reference:
        raise StageGateError("shakedown gate needs the owner's job reference")
    expires_at = str(data.get("expires_at") or "").strip()
    if not expires_at:
        raise StageGateError("shakedown gate needs expires_at")
    now = now_utc or utc_now()
    if expires_at < now:
        raise StageGateError(f"shakedown gate expired at {expires_at} (now {now})")
    return {
        "kind": SHAKEDOWN_GATE_KIND,
        "valid": True,
        "validated_at_utc": utc_now(),
        "scope": scope,
        "not_main_grid": True,
        "job_reference": job_reference,
        "expires_at": expires_at,
        "run_scope": RUN_SCOPE_SHAKEDOWN,
    }


def validate_shakedown_gate(
    path: str | Path, *, now_utc: str | None = None
) -> dict:
    """Load and validate a shakedown gate document (fail-closed)."""
    data = _load_json_object(path, "shakedown gate")
    gate = _validate_shakedown_gate_mapping(data, now_utc=now_utc)
    gate["gate_file"] = str(path)
    gate["gate_sha256"] = sha256_file(path)
    return gate


def _validate_pilot_gate_mapping(data: Mapping, *, now_utc: str | None = None) -> dict:
    """Validate the owner's pilot gate.

    The pilot is not the main grid, so it needs no preregistration; it needs the owner's own
    statement that this pilot may spend model calls on THIS case text:
    ``case_text_overlay_digest`` must equal the digest of the overlay on disk, so an overlay
    edited after the owner signed is refused.
    """
    from .case_text import load_case_text

    if not isinstance(data, Mapping):
        raise StageGateError("pilot gate must be a JSON object")
    if data.get("kind") != PILOT_GATE_KIND:
        raise StageGateError(f"pilot gate kind must be {PILOT_GATE_KIND!r}")
    if data.get("valid") is not True:
        raise StageGateError("pilot gate is not marked valid: true")
    if data.get("not_main_grid") is not True:
        raise StageGateError("pilot gate must set not_main_grid: true")
    if str(data.get("scope") or "") != RUN_SCOPE_PILOT:
        raise StageGateError("pilot gate scope must be 'pilot'")
    if not str(data.get("owner") or "").strip():
        raise StageGateError("pilot gate needs the owner's name")
    job = str(data.get("job") or data.get("job_reference") or "").strip()
    if not job:
        raise StageGateError("pilot gate needs the owner's job reference")
    if data.get("model_calls_approved") is not True:
        raise StageGateError("pilot gate must record model_calls_approved: true")
    expires_at = str(data.get("expires_at") or "").strip()
    if not expires_at:
        raise StageGateError("pilot gate needs expires_at")
    now = now_utc or utc_now()
    if expires_at < now:
        raise StageGateError(f"pilot gate expired at {expires_at} (now {now})")
    signed = str(data.get("case_text_overlay_digest") or "").strip()
    try:
        current = str(load_case_text().get("digest") or "")
    except Exception as exc:  # the overlay missing or tampered is a refusal
        raise StageGateError(f"pilot gate: the case-text overlay does not load: {exc}") from None
    if not signed:
        raise StageGateError(
            "pilot gate has no case_text_overlay_digest: the case text is not signed off"
        )
    if signed != current:
        raise StageGateError(
            "pilot gate: the case text changed since the owner signed it "
            f"(signed {signed[:12]}, current {current[:12]})"
        )
    return {
        "kind": PILOT_GATE_KIND,
        "valid": True,
        "validated_at_utc": utc_now(),
        "scope": RUN_SCOPE_PILOT,
        "not_main_grid": True,
        "owner": str(data["owner"]),
        "job_reference": job,
        "expires_at": expires_at,
        "case_text_overlay_digest": signed,
        "model_calls_approved": True,   # carried through: the runner re-validates this dict (fix 2026-11-18)
        "run_scope": RUN_SCOPE_PILOT,
    }


def validate_pilot_gate(path: str | Path, *, now_utc: str | None = None) -> dict:
    data = _load_json_object(path, "pilot gate")
    gate = _validate_pilot_gate_mapping(data, now_utc=now_utc)
    gate["gate_file"] = str(path)
    gate["gate_sha256"] = sha256_file(path)
    return gate


def validate_gate_document(
    gate: Mapping | None, *, now_utc: str | None = None
) -> dict:
    """Validate a gate *inside* the runner instead of trusting ``gate['valid']``.

    ``main_grid`` is re-validated against its preregistration root file, reviewer acceptance
    and owner approval.  A shakedown gate is validated for kind, scope, job reference and
    expiry.  Anything else is refused.
    """
    if not isinstance(gate, Mapping) or not gate:
        raise StageGateError("a validated stage gate is required")
    kind = gate.get("kind")
    if kind == GATE_KIND:
        if gate.get("valid") is not True:
            raise StageGateError("stage gate is not marked valid: true")
        validated = validate_stage_gate(
            prereg_root=gate.get("prereg_root_manifest"),
            prereg_hash=gate.get("prereg_root_sha256"),
            reviewer_acceptance=gate.get("reviewer_acceptance"),
            owner_approval=gate.get("owner_approval"),
            now_utc=now_utc,
        )
        if str(validated.get("prereg_root_sha256")) != str(
            gate.get("prereg_root_sha256")
        ):
            raise StageGateError("stage gate preregistration hash changed on re-check")
        return validated
    if kind == SHAKEDOWN_GATE_KIND:
        return _validate_shakedown_gate_mapping(gate, now_utc=now_utc)
    if kind == PILOT_GATE_KIND:
        return _validate_pilot_gate_mapping(gate, now_utc=now_utc)
    raise StageGateError(
        "stage gate kind must be "
        f"{GATE_KIND!r}, {SHAKEDOWN_GATE_KIND!r} or {PILOT_GATE_KIND!r}; got {kind!r}"
    )


def assert_pilot_summary(summary: Mapping) -> dict:
    """The pilot analysis reads only pilot data, and only data that says so."""
    scope = str(summary.get("run_scope") or "")
    if scope != RUN_SCOPE_PILOT or summary.get("label") != PILOT_LABEL:
        raise RunnerConfigError(
            f"refusing to read run_scope={scope!r} label={summary.get('label')!r} as pilot "
            "data; a pilot summary carries run_scope=pilot and label=PILOT"
        )
    return dict(summary)


def assert_main_grid_summary(summary: Mapping) -> dict:
    """The analysis layer refuses to read a shakedown/dry run as main-grid data."""
    scope = str(summary.get("run_scope") or "")
    if scope not in MAIN_GRID_SCOPES:
        raise RunnerConfigError(
            f"refusing to read run_scope={scope!r} as main-grid data; only "
            f"{list(MAIN_GRID_SCOPES)} is analysable"
        )
    return dict(summary)


# Episode context and route adapters / dispatchers


@dataclass
class EpisodeContext:
    resume_key: str
    row: Mapping
    route_calls: Counter = field(default_factory=Counter)
    route_errors: Counter = field(default_factory=Counter)
    # v14b: provider safeguard refusals are their own outcome: counted here, NOT in route_errors / error_kinds, so they can
    # never classify the unit as an infrastructure or route failure (``_episode_verdict`` only reads error_kinds).
    route_refusals: Counter = field(default_factory=Counter)
    error_kinds: list[str] = field(default_factory=list)
    call_latencies: list[dict] = field(default_factory=list)
    served_model_checks: Counter = field(default_factory=Counter)
    fatal_kind: str = ""

    def record_call(
        self,
        route_id: str,
        result: AdapterResult,
        latency_ms: float = 0.0,
        expected_model_id: str = "",
    ) -> None:
        self.route_calls[route_id] += 1
        latency = round(float(latency_ms or 0.0), 3)
        # The expected and the provider-reported model, and the verdict, on every call.
        served_verdict = served_model_check(expected_model_id, result.served_model_id)
        self.served_model_checks[served_verdict] += 1
        # Every call's latency is recorded, not only an aggregate afterwards.
        self.call_latencies.append({
            "route_id": route_id,
            "latency_ms": latency,
            "ok": bool(result.ok),
            "error_kind": result.error_kind,
            "expected_model_id": expected_model_id,
            "served_model_id": result.served_model_id,
            "served_model_check": served_verdict,
        })
        if result.error_kind == ERROR_PROVIDER_REFUSAL:
            self.route_refusals[route_id] += 1
        elif result.error_kind:
            self.route_errors[route_id] += 1
            self.error_kinds.append(result.error_kind)
            if result.error_kind == MODEL_SUBSTITUTION:
                self.fatal_kind = MODEL_SUBSTITUTION

    def latency_summary(self) -> dict:
        by_route: dict[str, list[float]] = defaultdict(list)
        for record in self.call_latencies:
            by_route[str(record.get("route_id", ""))].append(
                float(record.get("latency_ms") or 0.0)
            )
        return {
            route_id: summarise_latencies(values)
            for route_id, values in sorted(by_route.items())
        }


def current_episode_context() -> EpisodeContext | None:
    return getattr(_CURRENT_CONTEXT, "context", None)


def _set_episode_context(context: EpisodeContext | None) -> None:
    _CURRENT_CONTEXT.context = context


class RouteAdapter:
    """Adapter facade that sends every call through one route's paced pool."""

    def __init__(
        self,
        dispatcher: "RouteDispatcher",
        inner: object,
        *,
        model_id: str,
        vendor_family: str,
    ):
        self.dispatcher = dispatcher
        self.inner = inner
        self.model_id = model_id
        self.vendor_family = vendor_family
        self.route = getattr(inner, "route", "")

    def complete(self, messages: list[dict], *, max_tokens: int = 512,
                 temperature: float = 0.0) -> AdapterResult:
        context = current_episode_context()
        return self.dispatcher.complete(
            self.inner,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            context=context,
        )


class RouteDispatcher:
    """A paced worker pool for one route.

    Concurrency is the number of calls this route may have in flight; the start interval is
    derived from the configured min interval and the measured/declared requests/minute cap.
    A route pause gates *new* calls only.  In-flight calls finish under the adapter timeout.
    """

    def __init__(
        self,
        *,
        route: Mapping,
        settings: RouteSettings,
        adapter: object,
        stop_event: threading.Event,
        on_pause: Callable[[str, dict], None],
        on_auth_expired: Callable[[str, dict], None],
        on_auto_resume: Callable[[str, str], None],
        on_model_substitution: Callable[[str, dict], None],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        on_call_recorded: Callable[[str, AdapterResult, float], None] | None = None,
        charge_call: Callable[[str], None] | None = None,
    ):
        self.route = dict(route)
        self.route_id = settings.route_id
        self.settings = settings
        self.adapter = adapter
        self.expected_model_id = str(route.get("model_id") or "")
        self.stop_event = stop_event
        self._clock = clock
        self._sleep = sleep
        self._on_pause = on_pause
        self._on_auth_expired = on_auth_expired
        self._on_auto_resume = on_auto_resume
        self._on_model_substitution = on_model_substitution
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._paused = False
        self._pause_until: float | None = None
        self._pause_info: dict = {}
        self._last_start = float("-inf")
        self._start_interval = settings.start_interval(
            nvidia_route=route.get("route") == "nvidia_nim"
        )
        self._calls = 0
        self._successes = 0
        self._errors: Counter = Counter()
        self._refusals = 0  # v14b: PROVIDER_REFUSAL results (not in ``_errors``)
        self._last_error = ""
        self._latencies: deque[float] = deque(maxlen=5000)
        self._success_latencies: deque[float] = deque(maxlen=5000)
        self._segment_calls = 0
        self._on_call_recorded = on_call_recorded
        self._charge_call = charge_call
        self._executor = ThreadPoolExecutor(
            max_workers=settings.concurrency,
            thread_name_prefix=f"vera-route-{safe_component(settings.route_id)}",
        )
        self._closed = False

    # -- pause state ---------------------------------------------------------
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def pause_snapshot(self) -> dict:
        with self._lock:
            return {
                "paused": self._paused,
                "pause_until_monotonic": self._pause_until,
                "pause": dict(self._pause_info) if self._paused else {},
            }

    def restore_pause(self, info: Mapping, *, pause_until: float | None) -> None:
        with self._condition:
            self._paused = True
            self._pause_until = pause_until
            self._pause_info = dict(info)
            self._condition.notify_all()

    def maybe_auto_resume(self) -> bool:
        """Resume a paused route whose configured retry time has passed.

        The scheduler calls this even when no call is waiting, so a route that was paused
        just before its queue emptied still auto-resumes on time.
        """
        with self._condition:
            if not self._paused:
                return False
            if self._pause_until is None or self._clock() < self._pause_until:
                return False
            self._paused = False
            self._pause_until = None
            self._condition.notify_all()
        self._on_auto_resume(self.route_id, "configured retry time reached")
        return True

    def resume(self, *, reason: str = "command") -> dict:
        with self._condition:
            was_paused = self._paused
            info = dict(self._pause_info)
            self._paused = False
            self._pause_until = None
            self._pause_info = {}
            self._condition.notify_all()
        return {"was_paused": was_paused, "previous": info, "reason": reason}

    def _enter_pause(self, result: AdapterResult) -> dict:
        retry_after = (
            self.settings.route_limit_retry_seconds
            if self.settings.route_limit_retry_seconds is not None
            else None
        )
        pause_until = None if retry_after is None else self._clock() + retry_after
        info = {
            "route_id": self.route_id,
            "reason": ERROR_ROUTE_LIMIT,
            "message": result.error,
            "paused_at_utc": utc_now(),
            "retry_after_seconds": retry_after,
            "auto_resume": pause_until is not None,
        }
        with self._condition:
            self._paused = True
            self._pause_until = pause_until
            self._pause_info = info
            self._condition.notify_all()
        return info

    # -- rate state ----------------------------------------------------------
    def seed_counters(
        self,
        *,
        calls: int = 0,
        successes: int = 0,
        errors: Mapping[str, int] | None = None,
    ) -> None:
        """Seed cumulative route counters from persisted runner state."""
        with self._lock:
            self._calls = max(self._calls, int(calls))
            self._successes = max(self._successes, int(successes))
            for kind, count in (errors or {}).items():
                self._errors[str(kind)] += int(count)

    def counters(self) -> dict:
        with self._lock:
            latency = summarise_latencies(self._latencies)
            success_latency = summarise_latencies(self._success_latencies)
            return {
                "route_id": self.route_id,
                "calls": self._calls,
                "successes": self._successes,
                "errors": dict(sorted(self._errors.items())),
                "refusals": self._refusals,
                "last_error": self._last_error,
                "start_interval_seconds": round(self._start_interval, 6),
                "concurrency": self.settings.concurrency,
                "paused": self._paused,
                "pause": dict(self._pause_info) if self._paused else {},
                "latency_ms_p50": latency["p50_ms"],
                "latency_ms_p95": latency["p95_ms"],
                "latency_ms_max": latency["max_ms"],
                "latency_samples": latency["samples"],
                "latency": latency,
                "latency_success": success_latency,
                "calls_this_segment": self._segment_calls,
            }

    # -- execution -----------------------------------------------------------
    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise RunStopped("run stopped before the route call started")

    def _wait_turn(self) -> bool:
        """Block until a start is allowed.  Returns True if this woke an auto-resume."""
        auto_resumed = False
        while True:
            self._check_stop()
            with self._condition:
                now = self._clock()
                if self._paused:
                    if self._pause_until is not None and now >= self._pause_until:
                        self._paused = False
                        self._pause_until = None
                        self._condition.notify_all()
                        auto_resumed = True
                        continue
                    timeout = 0.25
                    if self._pause_until is not None:
                        timeout = max(0.01, min(timeout, self._pause_until - now))
                    self._condition.wait(timeout=timeout)
                    continue
                wait_for = self._last_start + self._start_interval - now
                if wait_for > 0:
                    self._condition.wait(timeout=min(wait_for, 0.25))
                    continue
                self._last_start = now if now != float("-inf") else self._clock()
                return auto_resumed

    def _run_task(
        self,
        inner: object,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        context: EpisodeContext | None,
    ) -> AdapterResult:
        """One logical model call.  v14b: a provider safeguard refusal (``PROVIDER_REFUSAL``) is retried AT MOST ONCE, with the
        identical messages.  Both attempts are real calls: each is paced, charged against ``max_model_calls``, written to the
        episode transcript and ledger as a refusal, and counted under refusals (never as a route/infrastructure error).  If
        the retry is refused too, that refusal is returned and the caller records the outcome."""
        result = self._run_task_once(
            inner, messages, max_tokens=max_tokens, temperature=temperature, context=context
        )
        if result.error_kind == ERROR_PROVIDER_REFUSAL:
            result = self._run_task_once(
                inner, messages, max_tokens=max_tokens, temperature=temperature, context=context
            )
        return result

    def _run_task_once(
        self,
        inner: object,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        context: EpisodeContext | None,
    ) -> AdapterResult:
        auto_resumed = self._wait_turn()
        if auto_resumed:
            self._on_auto_resume(self.route_id, "configured retry time reached")
        if self._charge_call is not None:
            # The cap is charged before the adapter call, inside the runner.
            self._charge_call(self.route_id)
        started = self._clock()
        try:
            result = inner.complete(  # type: ignore[attr-defined]
                messages, max_tokens=max_tokens, temperature=temperature
            )
        except RunStopped:
            raise
        except Exception as exc:  # an adapter exception is infrastructure, not model output
            result = AdapterResult(
                model_id=str(getattr(inner, "model_id", self.expected_model_id)),
                vendor_family=str(getattr(inner, "vendor_family", "")),
                route=str(getattr(inner, "route", "")),
                error=f"{type(exc).__name__}: {exc}",
                error_kind=ERROR_TRANSPORT,
            )
        latency_ms = max(0.0, (self._clock() - started) * 1000.0)
        if not isinstance(result, AdapterResult):
            result = AdapterResult(
                model_id=self.expected_model_id,
                route=str(self.route.get("route", "")),
                error="adapter returned a non-AdapterResult value",
                error_kind=ERROR_TRANSPORT,
            )
        # Compare the provider-reported (served) model, not only the configured id:
        # ``model_id`` is filled from the adapter's own configuration, so comparing it alone
        # could never fire for a real route.
        served_mismatch = (
            served_model_check(self.expected_model_id, result.served_model_id)
            == SERVED_MODEL_MISMATCH
        )
        legacy_mismatch = bool(
            result.model_id
            and self.expected_model_id
            and result.model_id != self.expected_model_id
        )
        if served_mismatch or legacy_mismatch:
            original = result
            reported_id = original.served_model_id if served_mismatch else original.model_id
            result = replace(
                result,
                error=(
                    f"route {self.route_id} expected {self.expected_model_id!r}, provider "
                    f"reported {reported_id!r}"
                ),
                error_kind=MODEL_SUBSTITUTION,
            )
            with self._lock:
                self._errors[MODEL_SUBSTITUTION] += 1
                self._last_error = result.error
            self._on_model_substitution(
                self.route_id,
                {
                    "route_id": self.route_id,
                    "expected_model_id": self.expected_model_id,
                    "reported_model_id": reported_id,
                    "message": result.error,
                },
            )
        with self._lock:
            self._calls += 1
            self._segment_calls += 1
            self._latencies.append(round(latency_ms, 3))
            if result.ok:
                self._successes += 1
                self._success_latencies.append(round(latency_ms, 3))
            elif result.error_kind == ERROR_PROVIDER_REFUSAL:
                self._refusals += 1  # v14b: its own outcome, never a route error
            elif result.error_kind:
                self._errors[result.error_kind] += 1
                self._last_error = result.error
        if context is not None:
            context.record_call(
                self.route_id, result, latency_ms, expected_model_id=self.expected_model_id
            )
        if self._on_call_recorded is not None:
            self._on_call_recorded(self.route_id, result, latency_ms)
        if result.error_kind == ERROR_ROUTE_LIMIT:
            info = self._enter_pause(result)
            self._on_pause(self.route_id, info)
        elif result.error_kind == ERROR_AUTH_EXPIRED:
            self._on_auth_expired(
                self.route_id,
                {
                    "route_id": self.route_id,
                    "message": result.error,
                    "at_utc": utc_now(),
                },
            )
        return result

    def complete(
        self,
        inner: object,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        context: EpisodeContext | None,
    ) -> AdapterResult:
        self._check_stop()
        future: Future = self._executor.submit(
            self._run_task,
            inner,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            context=context,
        )
        return future.result()

    def shutdown(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._executor.shutdown(wait=False, cancel_futures=True)


# Assignment loading, validation and the frozen interleaving schedule


def normalise_assignment_rows(rows: Sequence[Mapping]) -> list[dict]:
    """Validate assignment factors and attach the runner-only resume key."""
    cleaned: list[dict] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RunnerConfigError(f"assignment row {index} is not an object")
        item = dict(row)
        missing = [
            name
            for name in ("model_slot", "instance_id", "condition", "identity_condition", "repeat")
            if item.get(name) in (None, "")
        ]
        if missing:
            raise RunnerConfigError(
                f"assignment row {index} is missing {', '.join(missing)}"
            )
        if item["condition"] not in CONDITIONS:
            raise RunnerConfigError(
                f"assignment row {index}: unknown condition {item['condition']!r}"
            )
        if item["identity_condition"] not in IDENTITY_CONDITIONS:
            raise RunnerConfigError(
                f"assignment row {index}: unknown identity {item['identity_condition']!r}"
            )
        if item["condition"] == COND_DODGE and item.get("dodge_cue") not in (
            CUE_EXPLICIT, CUE_IMPLICIT
        ):
            raise RunnerConfigError(
                f"assignment row {index}: DODGE needs an explicit or implicit cue"
            )
        item.setdefault("seed", 0)
        item.setdefault("menu_position_seed", item.get("seed", 0))
        item.setdefault("label_permutation", 0)
        item.setdefault("dodge_cue", "")
        item.setdefault("family", "")
        item.setdefault("organization_pattern", PATTERN_BASE)
        if item["organization_pattern"] not in PATTERN_DISPLAY:
            raise RunnerConfigError(
                f"assignment row {index}: unknown organization pattern "
                f"{item['organization_pattern']!r}"
            )
        try:
            item["repeat"] = int(item["repeat"])
            item["seed"] = int(item["seed"])
            item["menu_position_seed"] = int(item["menu_position_seed"])
            item["label_permutation"] = int(item["label_permutation"])
        except (TypeError, ValueError):
            raise RunnerConfigError(
                f"assignment row {index}: repeat/seed/menu/label fields must be integers"
            ) from None
        key = assignment_resume_key(item)
        if key in seen:
            raise RunnerConfigError(
                f"assignment row {index}: duplicate resume key {key!r}; refusing to run "
                "two episodes with the same assignment factors"
            )
        seen.add(key)
        item["_runner_resume_key"] = key
        item["_row_index"] = index
        cleaned.append(item)
    return cleaned


def load_assignment_rows(
    path: str | Path | None = None,
    *,
    manifest: Mapping | None = None,
    repeats: int | None = None,
    patterns: Sequence[str] | None = None,
    scope: str = "",
) -> tuple[list[dict], dict]:
    """Load the exact frozen rows, or regenerate them with the frozen generator.

    ``patterns`` defaults to the base P0 block for API compatibility.  Pass
    ``ENROLLED_PATTERNS`` (or use the CLI ``--full-plan`` flag) to generate the
    full-cross plan: every base row under P0_BASE/P1/P2/P3.

    Review finding 1: in ``main_grid`` scope an omitted pattern level, a short table or
    a reduced repeat count is a hard refusal.  There is no partial main assignment.
    """
    meta: dict = {}
    if path is not None:
        target = Path(path)
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunnerConfigError(f"assignment table is unreadable: {exc}") from None
        if isinstance(data, list):
            rows = data
            meta = {"source": str(target), "source_kind": "row_list"}
        elif isinstance(data, dict) and isinstance(data.get("rows"), list):
            rows = data["rows"]
            meta = {
                "source": str(target),
                "source_kind": str(data.get("kind") or "assignment_table"),
                "source_formula": data.get("formula", ""),
                "source_expected_episodes": data.get("expected_episodes"),
            }
        elif isinstance(data, dict) and isinstance(data.get("assignment_table"), dict):
            raise RunnerConfigError(
                "the run-plan summary does not contain rows; supply the assignment table "
                "or omit --assignment to regenerate the frozen rows"
            )
        else:
            raise RunnerConfigError(
                "assignment table must be a row list or an object with a rows list"
            )
    else:
        if manifest is None:
            raise RunnerConfigError("manifest or assignment path is required")
        generated = build_assignment_table(
            manifest=dict(manifest),
            repeats=2 if repeats is None else int(repeats),
            patterns=tuple(patterns or (PATTERN_BASE,)),
        )
        rows = generated["rows"]
        meta = {
            "source": "generated:runplan_v13.build_assignment_table",
            "source_kind": generated["kind"],
            "source_formula": generated["formula"],
            "source_expected_episodes": generated["expected_episodes"],
            "source_patterns": list(patterns or (PATTERN_BASE,)),
            "generator_counts_by_condition": generated["counts_by_condition"],
        }
    if not rows:
        raise RunnerConfigError("assignment table has no rows")
    if path is None:
        effective_patterns = tuple(patterns or (PATTERN_BASE,))
        effective_repeats = 2 if repeats is None else int(repeats)
        meta["repeats"] = effective_repeats
        meta["patterns"] = list(effective_patterns)
        if scope == RUN_SCOPE_MAIN_GRID:
            if sorted(effective_patterns) != sorted(ENROLLED_PATTERNS):
                missing = sorted(set(ENROLLED_PATTERNS) - set(effective_patterns))
                raise RunnerConfigError(
                    "main_grid requires the complete frozen P0-P3 pattern cross; omitted "
                    f"pattern(s): {missing or 'none'} (got {list(effective_patterns)})"
                )
            if effective_repeats != 2:
                raise RunnerConfigError(
                    "main_grid requires the frozen repeat count 2, got "
                    f"{effective_repeats}"
                )
    elif scope == RUN_SCOPE_MAIN_GRID:
        # A file-supplied main table is checked by digest/row-count in the runner gate.
        meta.setdefault("source_kind", "assignment_table")
    cleaned = normalise_assignment_rows(rows)
    meta["row_count"] = len(cleaned)
    meta["assignment_digest"] = assignment_table_digest(cleaned)
    return cleaned, meta


def interleaved_schedule(rows: Sequence[Mapping]) -> list[dict]:
    """Deterministically interleave model x condition groups from the frozen rows.

    Rows are not reordered by any outcome.  The algorithm is fixed: group by
    ``(model_slot, condition)``, keep original order inside a group, then round-robin over
    group keys sorted by ``(condition, model_slot)``.  This prevents the generator's
    model-major loop from running an entire model or condition before another has started.
    """
    groups: dict[tuple[str, str], deque] = defaultdict(deque)
    for row in rows:
        groups[(str(row["model_slot"]), str(row["condition"]))].append(dict(row))
    # Order groups by condition then model so the round-robin alternates both dimensions
    # as evenly as the frozen table permits.
    keys = sorted(groups, key=lambda key: (key[1], key[0]))
    schedule: list[dict] = []
    while any(groups[key] for key in keys):
        for key in keys:
            if groups[key]:
                item = groups[key].popleft()
                item.setdefault("_runner_resume_key", assignment_resume_key(item))
                item["_schedule_index"] = len(schedule)
                schedule.append(item)
    return schedule


# Progress / state


class ProgressTracker:
    def __init__(
        self,
        *,
        workdir: Path,
        run_id: str,
        mode: str,
        rows: Sequence[Mapping],
        registry: Mapping,
        config: RunnerConfig,
        started_monotonic: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.workdir = Path(workdir)
        self.run_id = run_id
        self.mode = mode
        self.rows = [dict(row) for row in rows]
        self.registry = registry
        self.config = config
        self.clock = clock
        self.started_monotonic = (
            clock() if started_monotonic is None else started_monotonic
        )
        self.started_utc = utc_now()
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self.status = "initialising"
        self.done = 0
        self.interrupted = 0
        self.infrastructure_failed = 0
        self.model_calls_total = 0
        self.model_calls_segment = 0
        self.max_model_calls: int | None = None
        self.completed_keys: set[str] = set()
        self.infra_by_route: Counter = Counter()
        self.infra_by_condition: Counter = Counter()
        self.infra_by_pattern: Counter = Counter()
        self.infra_by_route_condition: Counter = Counter()
        self.done_by_route: Counter = Counter()
        self.done_by_condition: Counter = Counter()
        self.done_by_pattern: Counter = Counter()
        self.done_by_route_condition: Counter = Counter()
        self.assigned_by_route: Counter = Counter()
        self.assigned_by_condition: Counter = Counter()
        self.assigned_by_pattern: Counter = Counter()
        self.assigned_by_route_condition: Counter = Counter()
        self.route_calls: Counter = Counter()
        self.route_errors: Counter = Counter()
        self.pauses: dict[str, dict] = {}
        self._dispatchers: dict[str, RouteDispatcher] = {}
        self._primary_route_cache: dict[str, str] = {}
        for row in self.rows:
            route_id = self.primary_route(row)
            condition = str(row["condition"])
            pattern = str(row.get("organization_pattern") or PATTERN_BASE)
            self.assigned_by_route[route_id] += 1
            self.assigned_by_condition[condition] += 1
            self.assigned_by_pattern[pattern] += 1
            self.assigned_by_route_condition[(route_id, condition)] += 1

    def primary_route(self, row: Mapping) -> str:
        model_slot = str(row["model_slot"])
        if model_slot in self._primary_route_cache:
            return self._primary_route_cache[model_slot]
        route_id = str(row.get("route_id") or row.get("orchestrator_route_id") or "")
        if not route_id:
            try:
                route_id = str(_registry_slots_for(model_slot, self.registry)[SLOT_SELF]["route_id"])
            except ValueError as exc:
                raise RunnerConfigError(str(exc)) from None
        if not route_id:
            raise RunnerConfigError(
                f"assignment row {row.get('_row_index')}: cannot map model slot "
                f"{model_slot!r} to a registry route"
            )
        self._primary_route_cache[model_slot] = route_id
        return route_id

    def attach_dispatchers(self, dispatchers: Mapping[str, RouteDispatcher]) -> None:
        with self._lock:
            self._dispatchers = dict(dispatchers)

    def mark_status(self, status: str) -> None:
        with self._lock:
            self.status = status

    def record_completed(
        self,
        row: Mapping,
        *,
        resume_key: str,
        route_calls: Mapping[str, int] | None = None,
        route_errors: Mapping[str, int] | None = None,
    ) -> None:
        with self._lock:
            if resume_key in self.completed_keys:
                return
            self.completed_keys.add(resume_key)
            self.done += 1
            route_id = self.primary_route(row)
            condition = str(row["condition"])
            pattern = str(row.get("organization_pattern") or PATTERN_BASE)
            self.done_by_route[route_id] += 1
            self.done_by_condition[condition] += 1
            self.done_by_pattern[pattern] += 1
            self.done_by_route_condition[(route_id, condition)] += 1
            for route, count in (route_calls or {}).items():
                self.route_calls[str(route)] += int(count)
            for route, count in (route_errors or {}).items():
                self.route_errors[str(route)] += int(count)

    def record_interrupted(self, *, completed: bool = False) -> None:
        with self._lock:
            if not completed:
                self.interrupted += 1

    def record_infrastructure_failed(self, row: Mapping | None = None) -> None:
        with self._lock:
            self.infrastructure_failed += 1
            if row is not None:
                route_id = self.primary_route(row)
                condition = str(row["condition"])
                pattern = str(row.get("organization_pattern") or PATTERN_BASE)
                self.infra_by_route[route_id] += 1
                self.infra_by_condition[condition] += 1
                self.infra_by_pattern[pattern] += 1
                self.infra_by_route_condition[(route_id, condition)] += 1

    def set_pause(self, route_id: str, info: Mapping) -> None:
        with self._lock:
            self.pauses[str(route_id)] = dict(info)

    def clear_pause(self, route_id: str) -> None:
        with self._lock:
            self.pauses.pop(str(route_id), None)

    def snapshot(self) -> dict:
        with self._lock:
            now = self.clock()
            elapsed = max(0.0, now - self.started_monotonic)
            rate = self.done / elapsed if elapsed > 0 and self.done else None
            remaining = max(0, len(self.rows) - self.done - self.infrastructure_failed)
            routes: dict[str, dict] = {}
            pause_delay = 0.0
            for route_id, assigned in sorted(self.assigned_by_route.items()):
                dispatcher = self._dispatchers.get(route_id)
                counters = dispatcher.counters() if dispatcher else {}
                pause_info = dict(self.pauses.get(route_id) or {})
                resume_at = pause_info.get("resume_at_utc")
                routes[route_id] = {
                    "assigned_episodes": assigned,
                    "done_episodes": self.done_by_route.get(route_id, 0),
                    "remaining_episodes": max(
                        0,
                        assigned - self.done_by_route.get(route_id, 0)
                        - self.infra_by_route.get(route_id, 0),
                    ),
                    "paused": bool(dispatcher.is_paused()) if dispatcher else False,
                    "pause": pause_info,
                    "calls": self.route_calls.get(route_id, 0),
                    "call_errors": self.route_errors.get(route_id, 0),
                    "observed_calls_per_min": (
                        round(self.route_calls.get(route_id, 0) / elapsed * 60.0, 6)
                        if elapsed > 0 else None
                    ),
                    "dispatcher": counters,
                    "latency_ms_p50": counters.get("latency_ms_p50"),
                    "latency_ms_p95": counters.get("latency_ms_p95"),
                    "latency_samples": counters.get("latency_samples", 0),
                    "latency": dict(counters.get("latency") or {
                        "samples": 0,
                        "p50_ms": None,
                        "p95_ms": None,
                        "max_ms": None,
                        "mean_ms": None,
                    }),
                    "calls_this_segment": counters.get("calls_this_segment", 0),
                    "configured_concurrency": self.config.route(route_id).concurrency
                    if route_id in self.config.routes else None,
                    "configured_min_interval_seconds": (
                        self.config.route(route_id).min_interval_seconds
                        if route_id in self.config.routes else None
                    ),
                    "resume_at_utc": resume_at,
                }
                if pause_info and pause_info.get("retry_after_seconds") is not None:
                    pause_delay = max(
                        pause_delay,
                        float(pause_info["retry_after_seconds"]),
                    )
            eta = None
            eta_basis = ""
            if remaining == 0:
                eta = 0.0
                eta_basis = "no episodes remaining"
            elif rate:
                eta = remaining / rate + pause_delay
                eta_basis = (
                    "remaining / observed completed-episodes-per-second"
                    + (" + configured pause delay" if pause_delay else "")
                )
            by_route_condition: dict[str, dict] = {}
            for (route_id, condition), assigned in sorted(
                self.assigned_by_route_condition.items()
            ):
                done = self.done_by_route_condition.get((route_id, condition), 0)
                failed = self.infra_by_route_condition.get((route_id, condition), 0)
                by_route_condition.setdefault(route_id, {})[condition] = {
                    "assigned": assigned,
                    "done": done,
                    "infrastructure_failed": failed,
                    "remaining": max(0, assigned - done - failed),
                }
            return {
                "kind": PROGRESS_KIND,
                "run_id": self.run_id,
                "mode": self.mode,
                "status": self.status,
                "updated_at_utc": utc_now(),
                "started_at_utc": self.started_utc,
                "elapsed_seconds": round(elapsed, 3),
                "totals": {
                    "assigned": len(self.rows),
                    "done": self.done,
                    "remaining": remaining,
                    "interrupted": self.interrupted,
                    "infrastructure_failed": self.infrastructure_failed,
                },
                "by_condition": {
                    condition: {
                        "assigned": assigned,
                        "done": self.done_by_condition.get(condition, 0),
                        "infrastructure_failed": self.infra_by_condition.get(condition, 0),
                        "remaining": max(
                            0,
                            assigned
                            - self.done_by_condition.get(condition, 0)
                            - self.infra_by_condition.get(condition, 0),
                        ),
                    }
                    for condition, assigned in sorted(self.assigned_by_condition.items())
                },
                "by_pattern": {
                    pattern: {
                        "assigned": assigned,
                        "done": self.done_by_pattern.get(pattern, 0),
                        "infrastructure_failed": self.infra_by_pattern.get(pattern, 0),
                        "remaining": max(
                            0,
                            assigned
                            - self.done_by_pattern.get(pattern, 0)
                            - self.infra_by_pattern.get(pattern, 0),
                        ),
                    }
                    for pattern, assigned in sorted(self.assigned_by_pattern.items())
                },
                "by_route": {
                    route_id: {
                        "assigned_episodes": assigned,
                        "done_episodes": self.done_by_route.get(route_id, 0),
                        "infrastructure_failed_episodes": self.infra_by_route.get(route_id, 0),
                        "remaining_episodes": max(
                            0,
                            assigned
                            - self.done_by_route.get(route_id, 0)
                            - self.infra_by_route.get(route_id, 0),
                        ),
                    }
                    for route_id, assigned in sorted(self.assigned_by_route.items())
                },
                "by_route_condition": by_route_condition,
                "routes": routes,
                "route_latency": {
                    route_id: dict(data.get("latency") or {})
                    for route_id, data in routes.items()
                },
                "model_calls_total": self.model_calls_total,
                "model_calls_this_segment": self.model_calls_segment,
                "max_model_calls": self.max_model_calls,
                "pauses": dict(sorted(self.pauses.items())),
                "observed_completed_episodes_per_second": (
                    round(rate, 6) if rate is not None else None
                ),
                "observed_calls_per_min": (
                    round(sum(self.route_calls.values()) / elapsed * 60.0, 6)
                    if elapsed > 0 else None
                ),
                "eta_seconds": None if eta is None else round(eta, 3),
                "eta_minutes": None if eta is None else round(eta / 60.0, 3),
                "eta_basis": eta_basis,
            }

    def write(self, extra: Mapping | None = None) -> dict:
        snapshot = self.snapshot()
        if extra:
            snapshot.update(extra)
        with self._write_lock:
            atomic_write_json(self.workdir / DEFAULT_PROGRESS, snapshot)
        return snapshot


# Runner


@dataclass
class _EpisodeRun:
    row: Mapping
    attempt: int
    cell: object
    started_monotonic: float


class GridRunner:
    """Execute one frozen v1.3 assignment table with resumable, paced route pools."""

    def __init__(
        self,
        rows: Sequence[Mapping],
        *,
        config: RunnerConfig,
        workdir: str | Path,
        registry: Mapping | None = None,
        dry_run: bool = False,
        gate: Mapping | None = None,
        adapter_factory: Callable[[str, Mapping], object] | None = None,
        instances: Mapping[str, object] | None = None,
        event_hook: Callable[[str, Mapping], None] | None = None,
        control_file: str | Path | None = None,
        resume_routes: Sequence[str] = (),
        max_episodes: int | None = None,
        signing_key: bytes | str | None = None,
        preflight_runner: Callable[..., Mapping] | None = None,
        require_measured_routes: bool = True,
        preflight_acceptance: str | Path | None = None,
        source_binding: Mapping | None = None,
        assignment_meta: Mapping | None = None,
        episode_call_log_dir: str | Path | None = None,
        record_transcripts: bool = False,
        detention_protocol: str = "v13",
        solo: bool = False,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        # Live runs always write transcripts.  ``record_transcripts`` extends the same
        # record to the mocks used by dry runs and tests, so a dry run can be read the way a
        # live run can.
        self.record_transcripts = bool(record_transcripts)
        # "v13" is the frozen four-arm retry-only Detention; "v14" is the pilot design
        # (arms A, B, C, E; paired-mechanism eligibility; sibling transfer is the outcome).
        if detention_protocol not in ("v13", "v14"):
            raise RunnerConfigError(f"unknown detention protocol {detention_protocol!r}")
        self.detention_protocol = detention_protocol
        # Solo mode.  No peer menu, delegation budget 0, a prompt that never mentions
        # delegating.  A row may also ask for it (``solo``).
        self.solo = bool(solo)
        self.rows = [dict(row) for row in rows]
        if any("_runner_resume_key" not in row for row in self.rows):
            self.rows = normalise_assignment_rows(self.rows)
        self.config = config
        self.workdir = Path(workdir)
        self.registry = dict(registry or load_registry())
        self.dry_run = bool(dry_run)
        self.gate = dict(gate or {})
        self.adapter_factory = adapter_factory
        self.event_hook = event_hook
        self.control_file = Path(control_file) if control_file else None
        self.resume_routes = list(resume_routes)
        self.max_episodes = max_episodes
        self.clock = clock
        self.sleep = sleep
        self.preflight_runner = preflight_runner
        self.require_measured_routes = bool(require_measured_routes)
        self.preflight_acceptance = (
            dict(preflight_acceptance)
            if isinstance(preflight_acceptance, Mapping)
            else (Path(preflight_acceptance) if preflight_acceptance else None)
        )
        self.assignment_meta: dict = dict(assignment_meta or {})
        self._main_binding: dict = {}
        self._explicit_signing_key = (
            signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
        )
        self._signer: RunnerSigner | None = None
        self._signing_info: dict = {}
        self._ledger_seq = 0
        self._ledger_last_hash = ""
        self.run_id = str(
            self.config.raw.get("run_id")
            or f"v13-run-{utc_now().replace(':', '').replace('-', '')}"
        )
        self._source_binding = (
            dict(source_binding)
            if source_binding is not None
            else resolve_source_binding()
        )
        self._instances = (
            dict(instances)
            if instances is not None
            else self._load_instances()
        )
        self._verify_instances_against_source()
        self._ledger_path = self.workdir / DEFAULT_LEDGER
        self._state_path = self.workdir / DEFAULT_STATE
        self._result_dir = self.workdir / DEFAULT_EPISODE_DIR
        self._episode_call_dir = (
            Path(episode_call_log_dir)
            if episode_call_log_dir
            else self.workdir / DEFAULT_EPISODE_CALL_DIR
        )
        self._preflight_report: dict = {}
        self._route_defs: dict[str, dict] = {}
        self._ledger_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._fatal_reason = ""
        self._auth_expired = False
        self._dispatchers: dict[str, RouteDispatcher] = {}
        self._completed_keys: set[str] = set()
        self._terminal_failed_keys: set[str] = set()
        self._attempts: Counter = Counter()
        self._state: dict = {
            "pauses": {},
            "auth_expired": False,
            "stopped": "",
            "model_calls_total": 0,
            "route_calls": {},
            "route_successes": {},
            "route_errors": {},
            "route_refusals": {},
        }
        self._model_calls_total = 0
        self._segment_model_calls = 0
        self._run_scope = RUN_SCOPE_DRY_RUN if self.dry_run else ""
        self._lock_fd: int | None = None
        self._lock_path = self.workdir / DEFAULT_LOCK
        self._last_run_summary: dict = {}
        self._last_detention_summary: dict = {}
        self.progress: ProgressTracker | None = None
        self._started_monotonic = self.clock()
        self._started_utc = utc_now()

    # -- setup ---------------------------------------------------------------
    def _source_binding_record(self) -> dict:
        """The signed provenance of every instance and row this runner executes."""
        binding = self._source_binding
        return {
            "kind": "vera.v13.source_binding.record.v1",
            "source_set": binding["source"]["source_set"],
            "source_path": binding["source"]["path"],
            "source_sha256": binding["source"]["sha256"],
            "condition_manifest_digest": binding["condition_manifest_digest"],
            "preregistration": dict(binding["preregistration"]),
            "instances": len(self._instances),
        }

    def _load_instances(self) -> dict[str, object]:
        """Load instances from the validated frozen active source (set 12).

        Review finding 1: this used the set-10 ``load_instances()`` helper regardless of
        the D1 resolver, so the runner executed set 10 while claiming set 12.  There is now
        exactly one source: :func:`source_binding.resolve_source_binding`, which proves the
        attributable owner-review marker, the preregistered source hash and the
        preregistered condition-manifest digest before any instance is built.
        """
        from .instrument import load_source_instances

        binding = getattr(self, "_source_binding", None) or resolve_source_binding()
        by_id = {
            instance.instance_id: instance
            for instance in load_source_instances(binding["source"])
        }
        if not by_id:
            raise RunnerConfigError("the validated active source returned no instances")
        return by_id

    def _verify_instances_against_source(self) -> None:
        """Refuse any caller-supplied instance that is not the validated source record."""
        binding = self._source_binding
        source = binding["source"]
        from .instrument import load_source_instances

        bound = {instance.instance_id: instance for instance in load_source_instances(source)}
        problems: list[str] = []
        for instance_id, instance in self._instances.items():
            expected = bound.get(instance_id)
            if expected is None:
                problems.append(f"{instance_id}: not in the validated source")
                continue
            if canonical_json(instance.mechanism) != canonical_json(expected.mechanism):
                problems.append(f"{instance_id}: mechanism bytes differ from the source")
            if canonical_json(instance.graph) != canonical_json(expected.graph):
                problems.append(f"{instance_id}: graph bytes differ from the source")
        if problems:
            raise SourceBindingError(
                "runner refused: supplied instances are not the validated frozen source: "
                + "; ".join(problems[:5])
            )
        if len(self._instances) != len(bound):
            raise SourceBindingError(
                "runner refused: the instance table is incomplete: "
                f"{len(self._instances)} of {len(bound)} validated instances"
            )

    # -- exclusive run-directory lock -----------------------------
    def _acquire_run_lock(self) -> None:
        """Take an exclusive, non-blocking lock on this run directory.

        Two live runners on one directory interleaved signed ledger records in an earlier run.  The
        lock is an ``flock`` so a killed shell cannot orphan it: the kernel releases the
        lock when the last descriptor closes.
        """
        self.workdir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = ""
            try:
                holder = os.read(fd, 4096).decode("utf-8", "replace").strip()
            except OSError:
                holder = ""
            finally:
                os.close(fd)
            raise RunDirectoryLocked(
                "run directory is locked by another live runner: "
                f"{self._lock_path}"
                + (f" holder={holder}" if holder else "")
            ) from None
        info = json.dumps(
            {
                "run_id": self.run_id,
                "pid": os.getpid(),
                "locked_at_utc": utc_now(),
            },
            sort_keys=True,
        ) + "\n"
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, info.encode("utf-8"))
        os.fsync(fd)
        self._lock_fd = fd

    def _release_run_lock(self) -> None:
        fd = self._lock_fd
        self._lock_fd = None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    # -- gate validation inside the runner ------------------------
    def _resolve_run_id(self) -> None:
        """Keep one run id for one run directory across crash/resume segments.

        A dry-run signing key is derived from (run_id, assignment digest); adopting the
        persisted run id makes a resumed segment verify the same signed ledger instead of
        deriving a new key from a new wall-clock timestamp.
        """
        if self.config.raw.get("run_id"):
            return
        for path in (self.workdir / DEFAULT_ASSIGNMENT, self._state_path):
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            run_id = str((data or {}).get("run_id") or "")
            if run_id:
                self.run_id = run_id
                return

    def _validate_run_gate(self) -> str:
        if self.dry_run:
            if self.gate:
                self.gate = {**self.gate, "dry_run_bypass": True}
            return RUN_SCOPE_DRY_RUN
        validated = validate_gate_document(self.gate)
        scope = str(validated.get("run_scope") or "")
        if scope not in (RUN_SCOPE_MAIN_GRID, RUN_SCOPE_SHAKEDOWN, RUN_SCOPE_PILOT):
            raise StageGateError(f"validated gate has an unknown run scope {scope!r}")
        if scope == RUN_SCOPE_MAIN_GRID:
            self._validate_main_scope(validated)
        elif scope == RUN_SCOPE_PILOT:
            self._validate_pilot_scope(validated)
        # The re-validated document wins over any fields the caller injected.
        self.gate = {**self.gate, **validated, "run_scope": scope}
        return scope

    def _declared_non_cli_routes(self) -> dict:
        """Live-route facts for the routes whose declared transport is direct.

        The preflight acceptance check compares each route the run uses with these facts.
        A route whose transport is undeclared is left out, so the check refuses it.
        """
        live: dict[str, dict] = {}
        for route in (self.registry or {}).get("routes") or []:
            route_id = str(route.get("route_id") or "")
            if route_id and not route_requires_cli_isolation(route):
                live[route_id] = {
                    "route_id": route_id,
                    "route": str(route.get("route") or ""),
                    "model_id": str(route.get("model_id") or ""),
                    "requires_cli_isolation": False,
                }
        return live

    def _validate_pilot_scope(self, validated: Mapping) -> dict:
        """A pilot run cannot be weakened by flags either."""
        from .routes.preflight import validate_preflight_acceptance

        problems: list[str] = []
        if not self.record_transcripts:
            problems.append(
                "pilot scope requires transcripts (record_transcripts); nothing may be "
                "interpreted that cannot be read"
            )
        if self.require_measured_routes is not True:
            problems.append("pilot scope requires measured routes")
        if (self.registry or {}).get("pilot_profile") != "claude_only":
            problems.append("pilot scope requires the claude_only registry profile")
        bad_rows = [
            str(row.get("resume_key") or row.get("episode_id") or "?")
            for row in self.rows
            if str(row.get("protocol") or "v13") != "v14"
        ]
        if bad_rows:
            problems.append(
                "pilot scope runs protocol v14 only; v13 rows: " + ", ".join(bad_rows[:3])
            )
        route_ids = sorted(str(route_id) for route_id in self.config.routes)
        report: dict = {}
        try:
            report = validate_preflight_acceptance(
                self.preflight_acceptance,
                route_ids=route_ids,
                live_routes=self._declared_non_cli_routes(),
            )
        except Exception as exc:
            problems.append(f"bounded preflight acceptance: {exc}")
        if problems:
            raise StageGateError("pilot run refused: " + "; ".join(problems))
        self.require_measured_routes = True
        self._main_binding = {
            "kind": "vera.v14.pilot_scope_binding.v1",
            "run_scope": RUN_SCOPE_PILOT,
            "label": PILOT_LABEL,
            "preflight_acceptance": {
                "path": str(self.preflight_acceptance or ""),
                "report_sha256": report.get("report_sha256", ""),
                "route_ids": list(report.get("route_ids") or []),
            },
            "case_text_overlay_digest": str(validated.get("case_text_overlay_digest") or ""),
        }
        return self._main_binding

    def _label_fields(self) -> dict:
        """``{"label": "PILOT"}`` on every record of a pilot run, else nothing."""
        return {"label": PILOT_LABEL} if self._run_scope == RUN_SCOPE_PILOT else {}

    def _validate_main_scope(self, validated: Mapping) -> dict:
        """Review finding 6: a main-grid run cannot be weakened by flags.

        The scope requires the complete frozen assignment, measured routes and a separately
        authorized bounded-preflight artifact.  ``--allow-unmeasured-routes``,
        ``--pattern``/``--limit`` and ``--assignment`` are refusals here, not overrides.
        """
        from .routes.preflight import validate_preflight_acceptance

        problems: list[str] = []
        if self.require_measured_routes is not True:
            problems.append(
                "main_grid requires measured routes; --allow-unmeasured-routes cannot "
                "weaken this scope"
            )
        meta = self.assignment_meta
        try:
            assignment_binding = verify_main_assignment(
                self.rows,
                self._source_binding,
                patterns=meta.get("source_patterns"),
                repeats=meta.get("repeats"),
            )
        except SourceBindingError as exc:
            problems.append(str(exc))
            assignment_binding = {}
        route_ids = sorted(str(route_id) for route_id in self.config.routes)
        preflight_report: dict = {}
        try:
            preflight_report = validate_preflight_acceptance(
                self.preflight_acceptance,
                route_ids=route_ids,
                live_routes=self._declared_non_cli_routes(),
            )
        except Exception as exc:  # IsolationRefused or a file error is a refusal
            problems.append(f"bounded preflight acceptance: {exc}")
        if problems:
            raise SourceBindingError(
                "main-grid run refused: " + "; ".join(problems)
            )
        self.require_measured_routes = True
        self._main_binding = {
            "kind": "vera.v13.main_scope_binding.v1",
            "run_scope": RUN_SCOPE_MAIN_GRID,
            "assignment": assignment_binding,
            "preflight_acceptance": {
                "path": str(self.preflight_acceptance or ""),
                "report_sha256": preflight_report.get("report_sha256", ""),
                "stage": preflight_report.get("stage", ""),
                "route_ids": list(preflight_report.get("route_ids") or []),
            },
            "prereg_root_sha256": str(validated.get("prereg_root_sha256") or ""),
            "prereg_version_id": str(
                ((validated.get("source_binding") or {}).get("prereg_version_id")) or ""
            ),
            "source_sha256": self._source_binding["source"]["sha256"],
            "condition_manifest_digest": self._source_binding[
                "condition_manifest_digest"
            ],
        }
        return self._main_binding

    # -- call cap and cumulative counters -------------------------
    def _charge_model_call(self, route_id: str) -> None:
        limit = self.config.max_model_calls
        blocked = False
        with self._state_lock:
            if limit is not None and self._model_calls_total >= limit:
                blocked = True
            else:
                self._model_calls_total += 1
                self._segment_model_calls += 1
                self._state["model_calls_total"] = self._model_calls_total
        if self.progress is not None:
            self.progress.model_calls_total = self._model_calls_total
            self.progress.model_calls_segment = self._segment_model_calls
        if blocked:
            if not self._stop_event.is_set():
                self._fatal_reason = "MAX_MODEL_CALLS"
                self._state["stopped"] = "MAX_MODEL_CALLS"
                self._persist_state()
                self._stop_event.set()
                self._append_ledger({
                    "kind": "run_stopped",
                    "reason": "MAX_MODEL_CALLS",
                    "max_model_calls": limit,
                    "model_calls_total": self._model_calls_total,
                })
                self._emit("run_stopped", {
                    "reason": "MAX_MODEL_CALLS",
                    "max_model_calls": limit,
                })
            raise RunStopped(
                f"max_model_calls={limit} reached before route {route_id} call"
            )
        self._persist_state()

    def _on_call_recorded(
        self, route_id: str, result: AdapterResult, latency_ms: float
    ) -> None:
        with self._state_lock:
            calls = self._state.setdefault("route_calls", {})
            calls[route_id] = int(calls.get(route_id, 0)) + 1
            if result.ok:
                successes = self._state.setdefault("route_successes", {})
                successes[route_id] = int(successes.get(route_id, 0)) + 1
            elif result.error_kind == ERROR_PROVIDER_REFUSAL:
                # v14b: counted apart from route errors, and written to the signed ledger call by call
                refusals = self._state.setdefault("route_refusals", {})
                refusals[route_id] = int(refusals.get(route_id, 0)) + 1
                self._append_ledger({
                    "kind": "provider_refusal",
                    "route_id": route_id,
                    "error_kind": ERROR_PROVIDER_REFUSAL,
                    "message": str(result.error or "")[:300],
                    "latency_ms": round(float(latency_ms or 0.0), 3),
                    "route_refusals_total": int(refusals[route_id]),
                })
            elif result.error_kind:
                errors = self._state.setdefault("route_errors", {})
                per_route = errors.setdefault(route_id, {})
                per_route[result.error_kind] = int(per_route.get(result.error_kind, 0)) + 1
            self._state["model_calls_total"] = self._model_calls_total
        self._persist_state()

    def _slots_for_row(self, row: Mapping) -> dict:
        """The four registry slots a row may touch.

        v14b (2026-11-18): a solo row shows no peer menu and has delegation budget 0, so only its own (SELF)
        route is ever called.  Every slot of a solo row therefore resolves to that SELF route, which lets a
        single-route (Sonnet-only) run have one measured configuration, one dispatcher and no unmeasured peer
        route (under ``claude_only`` every slot map includes Haiku).  Non-solo rows are unchanged: all four
        registry slots.  Cell construction (``_cell_for_row``) deliberately still uses the real registry slots,
        so cell identity does not move.
        """
        slots = _registry_slots_for(str(row["model_slot"]), self.registry)
        if self.solo or row.get("solo"):
            return {name: slots[SLOT_SELF] for name in slots}
        return slots

    def _validate_rows_against_registry(self) -> None:
        for row in self.rows:
            slot = str(row["model_slot"])
            try:
                slots = _registry_slots_for(slot, self.registry)
            except ValueError as exc:
                raise RunnerConfigError(str(exc)) from None
            expected_primary = str(slots[SLOT_SELF]["route_id"])
            declared = row.get("route_id") or row.get("orchestrator_route_id")
            if declared and str(declared) != expected_primary:
                raise RunnerConfigError(
                    f"row {row.get('_row_index')}: route {declared!r} contradicts the "
                    f"registry mapping {expected_primary!r} for {slot}"
                )
            for slot_name, route in self._slots_for_row(row).items():
                if route["route_id"] not in self.config.routes:
                    raise RunnerConfigError(
                        f"row {row.get('_row_index')}: route {route['route_id']!r} "
                        f"({slot_name}) has no measured runner configuration"
                    )

    def _load_state(self) -> None:
        if not self._state_path.is_file():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._append_ledger({
                "kind": "runner_warning",
                "at_utc": utc_now(),
                "warning": f"runner_state.json unreadable: {exc}",
            })
            return
        if isinstance(data, dict):
            self._state["pauses"] = dict(data.get("pauses") or {})
            self._state["auth_expired"] = bool(data.get("auth_expired"))
            self._state["stopped"] = str(data.get("stopped") or "")
            self._state["model_calls_total"] = int(data.get("model_calls_total") or 0)
            self._state["route_calls"] = {
                str(key): int(value)
                for key, value in dict(data.get("route_calls") or {}).items()
            }
            self._state["route_successes"] = {
                str(key): int(value)
                for key, value in dict(data.get("route_successes") or {}).items()
            }
            self._state["route_errors"] = {
                str(key): {str(kind): int(count) for kind, count in dict(value or {}).items()}
                for key, value in dict(data.get("route_errors") or {}).items()
            }
            self._state["route_refusals"] = {
                str(key): int(value) for key, value in dict(data.get("route_refusals") or {}).items()
            }
            self._model_calls_total = int(self._state["model_calls_total"])

    def _persist_state(self) -> None:
        with self._state_lock:
            payload = {
                "kind": "vera.v13_runner_state.v1",
                **self._label_fields(),
                "run_id": self.run_id,
                "updated_at_utc": utc_now(),
                "pauses": dict(sorted(self._state.get("pauses", {}).items())),
                "auth_expired": bool(self._state.get("auth_expired")),
                "stopped": str(self._state.get("stopped") or ""),
                "model_calls_total": int(self._model_calls_total),
                "max_model_calls": self.config.max_model_calls,
                "route_calls": dict(
                    sorted(dict(self._state.get("route_calls") or {}).items())
                ),
                "route_successes": dict(
                    sorted(dict(self._state.get("route_successes") or {}).items())
                ),
                "route_errors": {
                    key: dict(sorted(value.items()))
                    for key, value in sorted(
                        dict(self._state.get("route_errors") or {}).items()
                    )
                },
                "route_refusals": dict(
                    sorted(dict(self._state.get("route_refusals") or {}).items())
                ),
            }
            atomic_write_json(self._state_path, payload)

    def _ensure_signer(self) -> RunnerSigner:
        if self._signer is not None:
            return self._signer
        if self._explicit_signing_key is not None:
            self._signer = RunnerSigner(self._explicit_signing_key, mode="explicit")
        elif self.dry_run:
            key = derive_dry_run_key(
                self.run_id, assignment_table_digest(self.rows)
            )
            self._signer = RunnerSigner(key, mode="dry_run_derived")
        else:
            raise RunnerConfigError(
                "a live run requires an explicit runner signing key "
                "(--signing-key-file or VERA_RUNNER_SIGNING_KEY); refusing to write "
                "unsigned episode logs"
            )
        self._signing_info = self._signer.info()
        return self._signer

    def _append_ledger(self, record: Mapping) -> dict:
        signer = self._ensure_signer()
        body = {
            "kind": LEDGER_KIND,
            **self._label_fields(),
            "run_id": self.run_id,
            "at_utc": utc_now(),
            **dict(record),
        }
        with self._ledger_lock:
            self._ledger_seq += 1
            body["seq"] = self._ledger_seq
            body["prev_sha256"] = self._ledger_last_hash
            digest = sha256_text(_signing_bytes(body).decode("utf-8"))
            payload = {
                **body,
                "record_sha256": digest,
                "signature_hmac": signer.sign_digest(digest),
                "signature_alg": SIGNATURE_ALGORITHM,
                "signing_key_id": signer.key_id,
            }
            append_jsonl(self._ledger_path, payload)
            self._ledger_last_hash = digest
        return payload

    def _emit(self, event: str, payload: Mapping) -> None:
        if self.event_hook is not None:
            self.event_hook(event, dict(payload))

    def _load_ledger(self) -> list[str]:
        records, problems = read_jsonl(self._ledger_path)
        if records:
            report = verify_ledger_records(records, self._ensure_signer().key)
            failures = list(report.get("problems") or [])
            if int(report.get("signed_records") or 0) != len(records):
                failures.append(
                    f"{len(records) - int(report.get('signed_records') or 0)} "
                    "record(s) are not HMAC-signed"
                )
            if failures:
                raise LedgerIntegrityError(
                    "refusing to append to an existing episode ledger that does not "
                    "verify: "
                    + "; ".join(failures[:5])
                )
        for record in records:
            try:
                self._ledger_seq = max(
                    self._ledger_seq, int(record.get("seq") or 0)
                )
            except (TypeError, ValueError):
                pass
            resume_key = str(record.get("resume_key") or "")
            if not resume_key:
                continue
            kind = str(record.get("kind") or "")
            if kind in ("episode_started", "episode_interrupted"):
                self._attempts[resume_key] = max(
                    self._attempts[resume_key], int(record.get("attempt") or 0)
                )
            elif kind in ("episode_completed", "episode_result_adopted"):
                self._completed_keys.add(resume_key)
                self._attempts[resume_key] = max(
                    self._attempts[resume_key], int(record.get("attempt") or 0)
                )
            elif kind == "episode_infrastructure_failed":
                self._terminal_failed_keys.add(resume_key)
                self._attempts[resume_key] = max(
                    self._attempts[resume_key], int(record.get("attempt") or 0)
                )
        if records:
            self._ledger_last_hash = ledger_record_hash(records[-1])
        for problem in problems:
            self._append_ledger({
                "kind": "runner_warning",
                "warning": f"ledger read problem: {problem}",
            })
        return problems

    def _main_result_inventory(self, signer: "RunnerSigner") -> dict:
        """Signed completed-main-result inventory, independent of the ledger's claims."""
        inventory: dict[str, dict] = {}
        for path in sorted(self._result_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise LedgerIntegrityError(
                    f"main inventory result {path.name} is unreadable: {exc}"
                ) from None
            if not isinstance(payload, dict) or not verify_signed_mapping(
                payload, signer.key
            ):
                raise LedgerIntegrityError(
                    f"main inventory result {path.name} does not verify"
                )
            if str(payload.get("status")) != "completed":
                continue
            key = str(payload.get("resume_key") or "")
            result = payload.get("result")
            result_digest = str(payload.get("result_digest") or "")
            if not key or not isinstance(result, dict) or not result_digest:
                raise LedgerIntegrityError(
                    f"main inventory result {path.name} has no bound resume key/result"
                )
            if json_digest(result) != result_digest:
                raise LedgerIntegrityError(
                    f"main inventory result {path.name} result_digest does not bind the "
                    "inner transcript/state"
                )
            inventory[key] = {
                "result_file": path.name,
                "result_sha256": sha256_file(path),
                "result_digest": result_digest,
                "episode_id": str(result.get("episode_id") or ""),
                "cell_hash": str(result.get("cell_hash") or ""),
            }
        return inventory

    def _verify_integrity_gate(
        self,
        main_summary: Mapping | None = None,
        *,
        require_completed_main: bool = False,
    ) -> dict:
        """Verify the CURRENT signed ledger chain and its completed-result inventory.

        Review finding 3: Detention previously trusted the historical
        ``main_summary.ledger.verification.ok`` flag.  This gate is shared by the main and
        Detention entry points, runs under the already acquired run lock, reads the current
        ledger, verifies every HMAC/chain link, verifies every signed completed-result file,
        and binds the two in both directions.  It never appends a record.
        """
        signer = self._ensure_signer()
        records, read_problems = read_jsonl(self._ledger_path)
        report = verify_ledger_records(records, signer.key)
        problems = list(report.get("problems") or []) + list(read_problems or [])
        if int(report.get("signed_records") or 0) != len(records):
            problems.append(
                f"{len(records) - int(report.get('signed_records') or 0)} ledger "
                "record(s) are not HMAC-signed"
            )
        if problems:
            raise LedgerIntegrityError(
                "current episode ledger does not verify; refusing the integrity gate: "
                + "; ".join(problems[:8])
            )

        inventory = self._main_result_inventory(signer)
        completed_records: dict[str, Mapping] = {}
        for record in records:
            kind = str(record.get("kind") or "")
            if kind not in ("episode_completed", "episode_result_adopted"):
                continue
            key = str(record.get("resume_key") or "")
            if not key:
                problems.append(
                    f"ledger record {record.get('seq')} has no completed resume key"
                )
                continue
            completed_records[key] = record

        if require_completed_main:
            if main_summary is None:
                problems.append("completed-main integrity gate needs the main summary")
            elif not verify_signed_mapping(main_summary, signer.key):
                problems.append("main summary signature does not verify")
        if main_summary is not None:
            ledger_block = main_summary.get("ledger") or {}
            verification = ledger_block.get("verification") or {}
            if verification.get("ok") is not True:
                problems.append("main summary historical ledger verification is not ok")
            try:
                main_count = int(ledger_block.get("records"))
            except (TypeError, ValueError):
                main_count = -1
            main_head = str(ledger_block.get("head_sha256") or "")
            if main_count < 0 or main_count > len(records):
                problems.append(
                    "main summary ledger length is not a prefix of the current ledger"
                )
                prefix = []
            else:
                prefix = records[:main_count]
                if not main_head or not prefix or ledger_record_hash(prefix[-1]) != main_head:
                    problems.append(
                        "main summary ledger head is not a prefix of the current ledger"
                    )
            prefix_completed = {
                str(record.get("resume_key") or "")
                for record in prefix
                if str(record.get("kind") or "")
                in ("episode_completed", "episode_result_adopted")
                and record.get("resume_key")
            }
            if require_completed_main:
                expected = int(main_summary.get("rows_completed") or 0)
                if len(inventory) != expected:
                    problems.append(
                        f"completed main inventory has {len(inventory)} signed results, "
                        f"summary declares {expected}"
                    )
                if prefix_completed != set(inventory):
                    problems.append(
                        "main ledger completed keys do not equal the signed result inventory"
                    )
                computed_inventory = json_digest([
                    inventory[key] for key in sorted(inventory)
                ])
                declared_inventory = str(
                    main_summary.get("main_inventory_digest") or ""
                )
                if declared_inventory and declared_inventory != computed_inventory:
                    problems.append(
                        "main summary inventory digest does not bind the signed results"
                    )
        # For both entry points every signed completed result must have a ledger record and
        # every completed ledger key must have a signed result.  (The main prefix check
        # above is stricter for the Detention entry point.)
        missing_results = sorted(set(completed_records) - set(inventory))
        if missing_results:
            problems.append(
                f"{len(missing_results)} completed ledger key(s) have no signed result"
            )
        if main_summary is not None:
            missing_records = sorted(set(inventory) - set(completed_records))
            if missing_records:
                problems.append(
                    f"{len(missing_records)} signed completed result(s) have no ledger key"
                )
        for key, item in inventory.items():
            record = completed_records.get(key)
            if record is None:
                continue
            if record.get("result_digest") and str(record["result_digest"]) != item[
                "result_digest"
            ]:
                problems.append(
                    f"ledger result_digest does not bind signed result for {key}"
                )
        if problems:
            raise LedgerIntegrityError(
                "current ledger/inventory integrity gate failed; refusing Detention: "
                + "; ".join(problems[:8])
            )
        return {
            "ok": True,
            "ledger_records": len(records),
            "ledger_head_sha256": report.get("head_sha256", ""),
            "completed_results": len(inventory),
            "main_inventory_digest": json_digest([
                inventory[key] for key in sorted(inventory)
            ]),
        }

    def _result_path(self, resume_key: str) -> Path:
        return self._result_dir / (safe_component(sha256_text(resume_key)[:24]) + ".json")

    def _save_result(self, resume_key: str, payload: Mapping) -> Path:
        path = self._result_path(resume_key)
        atomic_write_json(path, payload)
        return path

    def _load_result(self, resume_key: str) -> dict | None:
        path = self._result_path(resume_key)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or data.get("resume_key") != resume_key:
            return None
        return data

    def _build_dispatchers(self) -> None:
        if self.dry_run:
            return
        used_route_ids: list[str] = []
        for row in self.rows:
            slots = self._slots_for_row(row)
            for route in slots.values():
                if route["route_id"] not in used_route_ids:
                    used_route_ids.append(route["route_id"])
        for route_id in used_route_ids:
            route = route_by_id(self.registry, route_id)
            self._route_defs[route_id] = dict(route)
            settings = self.config.route(route_id)
            inner = (
                self.adapter_factory(route_id, route)
                if self.adapter_factory is not None
                else adapter_for_route(route)
            )
            # Every request an adapter makes must have a finite hard timeout.
            for attribute in ("timeout_seconds",):
                if hasattr(inner, attribute):
                    try:
                        value = float(getattr(inner, attribute))
                    except (TypeError, ValueError):
                        raise RunnerConfigError(
                            f"route {route_id}: {attribute} is not numeric"
                        ) from None
                    if not math.isfinite(value) or value <= 0:
                        raise RunnerConfigError(
                            f"route {route_id}: {attribute} must be a finite positive "
                            "number; every request must run under a timeout"
                        )
            reported_model = str(getattr(inner, "model_id", "") or "")
            if reported_model and reported_model != str(route["model_id"]):
                raise RunnerConfigError(
                    f"route {route_id}: adapter model {reported_model!r} does not match "
                    f"registry model {route['model_id']!r}; refusing silent substitution"
                )
            dispatcher = RouteDispatcher(
                route=route,
                settings=settings,
                adapter=inner,
                stop_event=self._stop_event,
                on_pause=self._on_route_pause,
                on_auth_expired=self._on_auth_expired,
                on_auto_resume=self._on_auto_resume,
                on_model_substitution=self._on_model_substitution,
                clock=self.clock,
                sleep=self.sleep,
                on_call_recorded=self._on_call_recorded,
                charge_call=self._charge_model_call,
            )
            # A resumed segment reports cumulative, persisted route counters.
            dispatcher.seed_counters(
                calls=int(dict(self._state.get("route_calls") or {}).get(route_id, 0)),
                successes=int(
                    dict(self._state.get("route_successes") or {}).get(route_id, 0)
                ),
                errors=dict(
                    dict(self._state.get("route_errors") or {}).get(route_id, {})
                ),
            )
            self._dispatchers[route_id] = dispatcher
        # Restore persisted pauses only when their retry time has not already passed.
        for route_id, info in sorted(self._state.get("pauses", {}).items()):
            dispatcher = self._dispatchers.get(route_id)
            if dispatcher is None:
                continue
            retry_after = info.get("retry_after_seconds")
            if retry_after is not None:
                paused_at = float(info.get("_paused_monotonic") or self.clock())
                remaining = max(0.0, paused_at + float(retry_after) - self.clock())
                if remaining <= 0 and route_id not in self.resume_routes:
                    self._append_ledger({
                        "kind": "route_resumed",
                        "route_id": route_id,
                        "reason": "persisted retry time already passed",
                    })
                    self._state["pauses"].pop(route_id, None)
                    self._persist_state()
                    continue
                pause_until = self.clock() + remaining if remaining > 0 else None
            else:
                pause_until = None
            if route_id in self.resume_routes:
                # Restore the persisted pause first, then resume it, so the
                # signed ledger always receives an attributable ``route_resumed`` record.
                dispatcher.restore_pause(info, pause_until=pause_until)
                self.resume_route(route_id, reason="startup command")
                continue
            dispatcher.restore_pause(info, pause_until=pause_until)

    def _on_route_pause(self, route_id: str, info: Mapping) -> None:
        persisted = dict(info)
        persisted["_paused_monotonic"] = self.clock()
        self._state.setdefault("pauses", {})[route_id] = persisted
        self._persist_state()
        self._append_ledger({
            "kind": "route_paused",
            "route_id": route_id,
            "reason": info.get("reason", ERROR_ROUTE_LIMIT),
            "message": info.get("message", ""),
            "retry_after_seconds": info.get("retry_after_seconds"),
            "auto_resume": info.get("auto_resume"),
        })
        if self.progress is not None:
            self.progress.set_pause(route_id, info)
            self.progress.write()
        self._emit("route_paused", {"route_id": route_id, **dict(info)})

    def _on_auth_expired(self, route_id: str, info: Mapping) -> None:
        self._auth_expired = True
        self._fatal_reason = ERROR_AUTH_EXPIRED
        self._state["auth_expired"] = True
        self._state["stopped"] = ERROR_AUTH_EXPIRED
        self._persist_state()
        self._stop_event.set()
        self._append_ledger({
            "kind": "route_auth_expired",
            "route_id": route_id,
            "message": info.get("message", ""),
        })
        if self.progress is not None:
            self.progress.mark_status("stopped_auth_expired")
            self.progress.write()
        self._emit("auth_expired", {"route_id": route_id, **dict(info)})

    def _on_auto_resume(self, route_id: str, reason: str) -> None:
        self._state.get("pauses", {}).pop(route_id, None)
        self._persist_state()
        self._append_ledger({
            "kind": "route_resumed",
            "route_id": route_id,
            "reason": reason,
        })
        if self.progress is not None:
            self.progress.clear_pause(route_id)
            self.progress.write()
        self._emit("route_resumed", {"route_id": route_id, "reason": reason})

    def _on_model_substitution(self, route_id: str, info: Mapping) -> None:
        self._fatal_reason = MODEL_SUBSTITUTION
        self._state["stopped"] = MODEL_SUBSTITUTION
        self._persist_state()
        self._stop_event.set()
        self._append_ledger({
            "kind": "model_substitution_detected",
            "route_id": route_id,
            **dict(info),
        })
        if self.progress is not None:
            self.progress.mark_status("stopped_model_substitution")
            self.progress.write()
        self._emit("model_substitution", {"route_id": route_id, **dict(info)})

    def resume_route(self, route_id: str, *, reason: str = "command") -> dict:
        dispatcher = self._dispatchers.get(route_id)
        if dispatcher is None:
            raise RunnerConfigError(f"route {route_id!r} has no dispatcher")
        persisted = route_id in (self._state.get("pauses") or {})
        outcome = dispatcher.resume(reason=reason)
        self._state.get("pauses", {}).pop(route_id, None)
        self._persist_state()
        # A startup resume removes a *persisted* pause, so the signed record must
        # be written even when the fresh dispatcher reports was_paused=False.
        if outcome.get("was_paused") or persisted:
            self._append_ledger({
                "kind": "route_resumed",
                "route_id": route_id,
                "reason": reason,
                "resumed_from": "persisted" if persisted else "live",
                "was_paused_live": bool(outcome.get("was_paused")),
            })
            if self.progress is not None:
                self.progress.clear_pause(route_id)
                self.progress.write()
            self._emit("route_resumed", {"route_id": route_id, "reason": reason})
        return outcome

    def _process_control_file(self) -> None:
        if self.control_file is None or not self.control_file.is_file():
            return
        try:
            data = json.loads(self.control_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        if data.get("stop") is True and not self._stop_event.is_set():
            self._fatal_reason = "CONTROL_FILE_STOP"
            self._state["stopped"] = "CONTROL_FILE_STOP"
            self._persist_state()
            self._stop_event.set()
            self._append_ledger({
                "kind": "run_stopped",
                "reason": "CONTROL_FILE_STOP",
            })
        for route_id in data.get("resume_routes") or []:
            if str(route_id) in self._dispatchers:
                self.resume_route(str(route_id), reason="command file")

    # -- preflight and signed episode call logs ------------------------------
    def _run_isolation_preflight(self) -> dict:
        """Protocol stage (now the route preflight), executed before any live episode."""
        if self.dry_run:
            report = {
                "kind": PREFLIGHT_RECORD_KIND,
                **self._label_fields(),
                "stage": "isolation_preflight",
                "ok": True,
                "start_allowed": True,
                "skipped": True,
                "reason": "dry run uses mock adapters",
            }
            self._preflight_report = report
            return report
        routes = [self._route_defs[route_id] for route_id in sorted(self._dispatchers)]
        adapters = {
            route_id: dispatcher.adapter
            for route_id, dispatcher in self._dispatchers.items()
        }
        if self.preflight_runner is not None:
            report = dict(
                self.preflight_runner(
                    routes,
                    adapters=adapters,
                    require_measured_routes=self.require_measured_routes,
                )
                or {}
            )
            if report.get("ok") is not True:
                from .routes.preflight import IsolationRefused

                reasons = report.get("refusals") or [
                    report.get("reason") or "preflight_runner returned not-ok"
                ]
                raise IsolationRefused(
                    "run refused: route preflight failed: "
                    + "; ".join(str(item) for item in reasons)
                )
            acceptance = self._bounded_preflight_acceptance(adapters)
            self._preflight_report = {
                "kind": PREFLIGHT_RECORD_KIND,
                **self._label_fields(),
                "stage": "isolation_preflight",
                "start_allowed": True,
                "injected": True,
                **({"bounded_preflight_acceptance": acceptance} if acceptance else {}),
                **report,
            }
            return self._preflight_report
        from .routes.preflight import run_preflight_or_refuse

        # The structural stage makes no model call.  In main-grid and pilot scope the probe
        # is proven by the separately run acceptance report, not by this structural check.
        report = run_preflight_or_refuse(
            routes,
            adapters=adapters,
            require_measured_routes=bool(self.require_measured_routes),
        )
        acceptance = self._bounded_preflight_acceptance(adapters)
        self._preflight_report = {
            "kind": PREFLIGHT_RECORD_KIND,
            **self._label_fields(),
            "stage": "isolation_preflight",
            "start_allowed": True,
            **({"bounded_preflight_acceptance": acceptance} if acceptance else {}),
            **dict(report),
        }
        return self._preflight_report

    def _bounded_preflight_acceptance(self, adapters: Mapping) -> dict:
        """Require and cross-check the probed preflight report for a main-grid or pilot run."""
        if self._run_scope not in (RUN_SCOPE_MAIN_GRID, RUN_SCOPE_PILOT):
            return {}
        from .routes.preflight import IsolationRefused, validate_preflight_acceptance

        problems: list[str] = []
        live_routes: dict[str, dict] = {}
        for route_id, adapter in sorted(adapters.items()):
            route_id = str(route_id)
            route_def = dict(self._route_defs.get(route_id) or {})
            adapter_capability = bool(getattr(adapter, "requires_cli_isolation", True))
            declared_capability = (
                route_requires_cli_isolation(route_def) if route_def else adapter_capability
            )
            if declared_capability != adapter_capability:
                problems.append(
                    f"route {route_id}: declared transport capability "
                    f"{declared_capability!r} does not match the live adapter "
                    f"{adapter_capability!r}"
                )
            live_routes[route_id] = {
                "route_id": route_id,
                "route": str(getattr(adapter, "route", "")),
                "model_id": str(getattr(adapter, "model_id", "")),
                "requires_cli_isolation": bool(declared_capability or adapter_capability),
            }
        if problems:
            raise IsolationRefused(
                "run refused: preflight is not for this configuration: " + "; ".join(problems)
            )
        accepted = validate_preflight_acceptance(
            self.preflight_acceptance,
            route_ids=sorted(live_routes),
            live_routes=live_routes,
        )
        return {
            key: accepted.get(key)
            for key in (
                "kind",
                "stage",
                "report_sha256",
                "route_ids",
                "probe_route_ids",
                "non_cli_route_ids",
            )
        } | {"path": str(self.preflight_acceptance or "")}

    def _episode_call_log_path(
        self, resume_key: str, directory: str | Path | None = None
    ) -> Path:
        base = Path(directory) if directory is not None else self._episode_call_dir
        return base / (safe_component(sha256_text(resume_key)[:24]) + ".jsonl")

    def _episode_adapter(
        self,
        inner: object,
        row: Mapping,
        resume_key: str,
        directory: str | Path | None = None,
    ):
        """Give an adapter an episode-scoped call log without touching other threads.

        Live runs always record, so every live call leaves a transcript; a dry run records
        when ``record_transcripts`` is set.  The wrapper is per episode; the shared
        dispatcher/pacing state remains the original adapter and route executor.
        """
        if self.dry_run and not self.record_transcripts:
            return inner
        return RecordingAdapter(
            inner,
            EpisodeCallLog(
                self._episode_call_log_path(resume_key, directory),
                episode_id=str(row.get("episode_id") or resume_key),
            ),
        )

    def _episode_call_log_meta(
        self, resume_key: str, directory: str | Path | None = None
    ) -> dict:
        path = self._episode_call_log_path(resume_key, directory)
        if not path.is_file():
            return {}
        records, problems = read_jsonl(path)
        transcript = verify_call_log(path)
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "records": len(records),
            "read_problems": problems,
            # Whether every record carries its messages and reply_text.
            "transcripts_complete": bool(transcript["ok"]),
            "transcript_problems": list(transcript["problems"]),
        }

    def _episode_verdict(self, context: "EpisodeContext", attempt: int) -> dict:
        """Classify what happened on the route during one episode or Detention row.

        One rule for the main phase and the Detention phase.  ``interrupted``: the unit is not
        complete and must not be written as completed (model substitution and auth expiry end
        the run; a route limit is re-queued after the pause).  ``infra_retry`` / ``infra_terminal``:
        a transport-class error that is retried up to ``infra_retries`` and then recorded as an
        infrastructure failure (never as a result).  ``clean``: nothing went wrong on the route.
        """
        error_kinds = list(context.error_kinds)
        base = {
            "error_kinds": error_kinds,
            "reason": "",
            "requeue": False,
            "infrastructure": [],
            "fatal_kinds": [],
        }
        if context.fatal_kind == MODEL_SUBSTITUTION:
            return {**base, "verdict": "interrupted", "reason": MODEL_SUBSTITUTION}
        if ERROR_AUTH_EXPIRED in error_kinds:
            return {**base, "verdict": "interrupted", "reason": ERROR_AUTH_EXPIRED}
        if ERROR_ROUTE_LIMIT in error_kinds:
            return {
                **base, "verdict": "interrupted", "reason": ERROR_ROUTE_LIMIT, "requeue": True,
            }
        infrastructure = [kind for kind in error_kinds if kind in INFRASTRUCTURE_KINDS]
        if infrastructure:
            fatal = [kind for kind in infrastructure if kind in FATAL_INFRASTRUCTURE_KINDS]
            terminal = bool(fatal) or attempt > self.config.infra_retries
            return {
                **base,
                "verdict": "infra_terminal" if terminal else "infra_retry",
                "reason": "INFRASTRUCTURE_FAILURE",
                "infrastructure": infrastructure,
                "fatal_kinds": fatal,
            }
        return {**base, "verdict": "clean"}

    def _record_mocks(
        self,
        orchestrator: object,
        peers: Mapping,
        row: Mapping,
        resume_key: str,
        directory: str | Path | None = None,
    ) -> tuple[object, dict]:
        """Dry-run mocks, wrapped so their calls leave the same transcript as live ones."""
        if not self.record_transcripts:
            return orchestrator, dict(peers)
        return (
            self._episode_adapter(orchestrator, row, resume_key, directory),
            {
                name: self._episode_adapter(peer, row, resume_key, directory)
                for name, peer in peers.items()
            },
        )

    # -- cells and one episode ----------------------------------------------
    def _cell_for_row(self, row: Mapping):
        instance = self._instances.get(str(row["instance_id"]))
        if instance is None:
            raise RunnerConfigError(
                f"instance {row['instance_id']!r} is not present in the active set"
            )
        slots = _registry_slots_for(str(row["model_slot"]), self.registry)
        peers = tuple(
            PeerSpec(
                slot=slot_name,
                principal=f"peer_{slot_name}",
                model_id=str(route["model_id"]),
                vendor_family=str(route["publisher"]),
                capability_band=(
                    "band_same_weights" if slot_name == SLOT_SELF
                    else str(route.get("capability_band") or "band_within_0_10")
                ),
            )
            for slot_name, route in slots.items()
        )
        orchestrator_route = slots[SLOT_SELF]
        from .plan import build_cell  # local import keeps module import cheap

        return build_cell(
            instance,
            str(row["condition"]),
            seed=int(row["seed"]),
            repeat=int(row["repeat"]),
            identity_condition=str(row["identity_condition"]),
            owner_cue=str(row.get("dodge_cue") or CUE_EXPLICIT),
            peers=peers,
            orchestrator_model_id=str(orchestrator_route["model_id"]),
            orchestrator_vendor=str(orchestrator_route["publisher"]),
            pattern=str(row.get("organization_pattern") or PATTERN_BASE),
            four_slot=True,
            menu_position_seed=int(row["menu_position_seed"]),
            label_permutation=int(row["label_permutation"]),
            # Rows carry an optional ``protocol``; absent = the frozen v13 behaviour.
            protocol=str(row.get("protocol") or "v13"),
            solo=bool(self.solo or row.get("solo")),
        )

    def _main_route_for_row(self, row: Mapping) -> str:
        return str(_registry_slots_for(str(row["model_slot"]), self.registry)[SLOT_SELF]["route_id"])

    def _adopt_existing_result(self, row: Mapping, key: str) -> bool:
        payload = self._load_result(key)
        if payload is None:
            return False
        status = str(payload.get("status") or "")
        if status not in ("completed", "infrastructure_failed"):
            return False
        route_calls = payload.get("route_calls") or {}
        route_errors = payload.get("route_errors") or {}
        if status == "completed":
            self._completed_keys.add(key)
            self._append_ledger({
                "kind": "episode_result_adopted",
                "resume_key": key,
                "assignment_resume_key": row.get("resume_key", ""),
                "episode_id": row.get("episode_id", ""),
                "row_index": row.get("_row_index"),
                "attempt": int(payload.get("attempt") or 0),
                "result_file": str(self._result_path(key)),
                "result_digest": payload.get("result_digest", ""),
                "route_calls": dict(route_calls),
                "route_errors": dict(route_errors),
            })
            if self.progress is not None:
                self.progress.record_completed(
                    row,
                    resume_key=key,
                    route_calls=route_calls,
                    route_errors=route_errors,
                )
        else:
            self._terminal_failed_keys.add(key)
        self._emit("result_adopted", {"resume_key": key, "status": status})
        return True

    def _run_episode(self, episode: _EpisodeRun) -> dict:
        row = episode.row
        key = str(row["_runner_resume_key"])
        attempt = episode.attempt
        cell = episode.cell
        context = EpisodeContext(resume_key=key, row=row)
        self._append_ledger({
            "kind": "episode_started",
            "resume_key": key,
            "assignment_resume_key": row.get("resume_key", ""),
            "episode_id": row.get("episode_id", ""),
            "row_index": row.get("_row_index"),
            "schedule_index": row.get("_schedule_index"),
            "attempt": attempt,
        })
        self._emit("episode_started", {"resume_key": key, "row": row})
        _set_episode_context(context)
        started = self.clock()
        try:
            if self.dry_run:
                orchestrator, peers = default_mock_adapters(cell)
                orchestrator, peers = self._record_mocks(
                    orchestrator, peers, row, key
                )
            else:
                slots = self._slots_for_row(row)
                orchestrator_route = str(slots[SLOT_SELF]["route_id"])
                orchestrator = RouteAdapter(
                    self._dispatchers[orchestrator_route],
                    self._episode_adapter(
                        self._dispatchers[orchestrator_route].adapter, row, key
                    ),
                    model_id=str(slots[SLOT_SELF]["model_id"]),
                    vendor_family=str(slots[SLOT_SELF]["publisher"]),
                )
                peers = {}
                for slot_name, route in slots.items():
                    route_id = str(route["route_id"])
                    peers[f"peer_{slot_name}"] = RouteAdapter(
                        self._dispatchers[route_id],
                        self._episode_adapter(
                            self._dispatchers[route_id].adapter, row, key
                        ),
                        model_id=str(route["model_id"]),
                        vendor_family=str(route["publisher"]),
                    )
            from .world import run_episode

            result = run_episode(
                cell,
                orchestrator,
                peers,
                harness_key=b"v13-runner-harness-" + cell.cell_hash[:20].encode(),
                verifier_key=b"v13-runner-verifier-" + cell.cell_hash[:20].encode(),
                run_id=str(row.get("episode_id") or f"v13-{key}"),
                seed=int(row["seed"]),
            )
        except RunStopped as exc:
            _set_episode_context(None)
            self._append_ledger({
                "kind": "episode_interrupted",
                "resume_key": key,
                "attempt": attempt,
                "reason": RUN_STOPPED,
                "message": str(exc),
            })
            self._emit("episode_interrupted", {
                "resume_key": key,
                "reason": RUN_STOPPED,
            })
            if self.progress is not None:
                self.progress.record_interrupted()
            return {"status": "interrupted", "requeue": False, "key": key}
        except Exception as exc:
            _set_episode_context(None)
            self._append_ledger({
                "kind": "episode_infrastructure_failed",
                "resume_key": key,
                "attempt": attempt,
                "reason": HARNESS_FAULT,
                "message": f"{type(exc).__name__}: {exc}",
                "route_calls": dict(context.route_calls),
                "route_errors": dict(context.route_errors),
                "error_kinds": list(context.error_kinds),
            })
            if self.progress is not None:
                self.progress.record_infrastructure_failed(row)
            return {"status": "infrastructure_failed", "requeue": False, "key": key}
        finally:
            _set_episode_context(None)

        elapsed = max(0.0, self.clock() - started)
        facts = result.get("facts") or {}
        first_choice = first_choice_outcome(facts)
        self._append_ledger({
            "kind": "first_choice_scored",
            "resume_key": key,
            "episode_id": result.get("episode_id", ""),
            "choice": first_choice.get("choice"),
            "actual_slot": first_choice.get("actual_slot"),
            "displayed_publisher": first_choice.get("displayed_publisher"),
            "displayed_label": first_choice.get("displayed_label"),
            "menu_position": first_choice.get("menu_position"),
        })
        self._emit("first_choice_scored", {"resume_key": key, "first_choice": first_choice})

        error_kinds = list(context.error_kinds)
        verdict = self._episode_verdict(context, attempt)
        if verdict["verdict"] == "interrupted" and verdict["reason"] == MODEL_SUBSTITUTION:
            self._append_ledger({
                "kind": "episode_interrupted",
                "resume_key": key,
                "attempt": attempt,
                "reason": MODEL_SUBSTITUTION,
                "route_calls": dict(context.route_calls),
                "route_errors": dict(context.route_errors),
            })
            self._emit("episode_interrupted", {
                "resume_key": key,
                "reason": MODEL_SUBSTITUTION,
            })
            if self.progress is not None:
                self.progress.record_interrupted()
            return {"status": "interrupted", "requeue": False, "key": key}
        if verdict["verdict"] == "interrupted" and verdict["reason"] == ERROR_AUTH_EXPIRED:
            self._append_ledger({
                "kind": "episode_interrupted",
                "resume_key": key,
                "attempt": attempt,
                "reason": ERROR_AUTH_EXPIRED,
                "route_calls": dict(context.route_calls),
            })
            self._emit("episode_interrupted", {
                "resume_key": key,
                "reason": ERROR_AUTH_EXPIRED,
            })
            if self.progress is not None:
                self.progress.record_interrupted()
            return {"status": "interrupted", "requeue": False, "key": key}
        if verdict["verdict"] == "interrupted" and verdict["reason"] == ERROR_ROUTE_LIMIT:
            self._append_ledger({
                "kind": "episode_interrupted",
                "resume_key": key,
                "attempt": attempt,
                "reason": ERROR_ROUTE_LIMIT,
                "route_calls": dict(context.route_calls),
            })
            self._emit("episode_interrupted", {
                "resume_key": key,
                "reason": ERROR_ROUTE_LIMIT,
            })
            if self.progress is not None:
                self.progress.record_interrupted()
            return {"status": "interrupted", "requeue": True, "key": key}

        infrastructure = verdict["infrastructure"]
        if infrastructure:
            fatal = verdict["fatal_kinds"]
            terminal = verdict["verdict"] == "infra_terminal"
            if terminal:
                infra_payload = {
                    "kind": RESULT_KIND,
                    **self._label_fields(),
                    "run_id": self.run_id,
                    "resume_key": key,
                    "assignment_resume_key": row.get("resume_key", ""),
                    "row": {k: v for k, v in row.items() if not k.startswith("_")},
                    "attempt": attempt,
                    "status": "infrastructure_failed",
                    "completed_at_utc": utc_now(),
                    "duration_seconds": round(elapsed, 6),
                    "first_choice": first_choice,
                    "route_calls": dict(context.route_calls),
                    "route_errors": dict(context.route_errors),
                    "route_latency": context.latency_summary(),
                    "call_latencies": list(context.call_latencies),
                    "served_model_checks": dict(context.served_model_checks),
                    "error_kinds": error_kinds,
                    "termination_reason": result.get("termination_reason", ""),
                    "result_digest": json_digest(result),
                    "result": result,
                }
                call_log = (
                    {}
                    if self.dry_run and not self.record_transcripts
                    else self._episode_call_log_meta(key)
                )
                if call_log:
                    infra_payload["episode_call_log"] = call_log
                infra_payload = self._ensure_signer().sign_mapping(infra_payload)
                self._save_result(key, infra_payload)
                self._terminal_failed_keys.add(key)
                self._append_ledger({
                    "kind": "episode_infrastructure_failed",
                    "resume_key": key,
                    "assignment_resume_key": row.get("resume_key", ""),
                    "episode_id": row.get("episode_id", ""),
                    "row_index": row.get("_row_index"),
                    "attempt": attempt,
                    "reason": "INFRASTRUCTURE_FAILURE",
                    "error_kinds": error_kinds,
                    "route_calls": dict(context.route_calls),
                    "route_errors": dict(context.route_errors),
                    "fatal": bool(fatal),
                    "result_file": str(self._result_path(key)),
                })
                if self.progress is not None:
                    self.progress.record_infrastructure_failed(row)
                if fatal:
                    self._fatal_reason = "INFRASTRUCTURE_FATAL"
                    self._state["stopped"] = "INFRASTRUCTURE_FATAL"
                    self._persist_state()
                    self._stop_event.set()
                self._emit("episode_infrastructure_failed", {
                    "resume_key": key,
                    "error_kinds": error_kinds,
                    "fatal": bool(fatal),
                })
                return {
                    "status": "infrastructure_failed",
                    "requeue": False,
                    "key": key,
                }
            self._append_ledger({
                "kind": "episode_infrastructure_retry",
                "resume_key": key,
                "attempt": attempt,
                "error_kinds": error_kinds,
                "route_calls": dict(context.route_calls),
                "route_errors": dict(context.route_errors),
            })
            self._emit("episode_infrastructure_retry", {
                "resume_key": key,
                "error_kinds": error_kinds,
            })
            return {"status": "infrastructure_retry", "requeue": True, "key": key}

        payload = {
            "kind": RESULT_KIND,
            **self._label_fields(),
            "run_id": self.run_id,
            "resume_key": key,
            "assignment_resume_key": row.get("resume_key", ""),
            "row": {k: v for k, v in row.items() if not k.startswith("_")},
            "attempt": attempt,
            "status": "completed",
            "completed_at_utc": utc_now(),
            "duration_seconds": round(elapsed, 6),
            "first_choice": first_choice,
            "route_calls": dict(context.route_calls),
            "route_errors": dict(context.route_errors),
            "route_refusals": dict(context.route_refusals),  # v14b: provider safeguard refusals, not errors
            "route_latency": context.latency_summary(),
            "call_latencies": list(context.call_latencies),
            "served_model_checks": dict(context.served_model_checks),
            "error_kinds": error_kinds,
            "termination_reason": result.get("termination_reason", ""),
            "classification": result.get("classification", {}),
            "facts": facts,
            "log_verified": bool((result.get("verified") or {}).get("ok", False))
            if isinstance(result.get("verified"), dict)
            else bool(result.get("verified")),
            "result": result,
        }
        payload["result_digest"] = json_digest(result)
        call_log = (
            {}
            if self.dry_run and not self.record_transcripts
            else self._episode_call_log_meta(key)
        )
        if call_log:
            payload["episode_call_log"] = call_log
        payload = self._ensure_signer().sign_mapping(payload)
        self._save_result(key, payload)
        self._emit("result_persisted", {"resume_key": key, "path": str(self._result_path(key))})
        self._completed_keys.add(key)
        self._append_ledger({
            "kind": "episode_completed",
            "resume_key": key,
            "assignment_resume_key": row.get("resume_key", ""),
            "episode_id": row.get("episode_id", ""),
            "row_index": row.get("_row_index"),
            "attempt": attempt,
            "status": "completed",
            "termination_reason": result.get("termination_reason", ""),
            "result_file": str(self._result_path(key)),
            "result_digest": payload["result_digest"],
            "route_calls": dict(context.route_calls),
            "route_errors": dict(context.route_errors),
            "route_refusals": dict(context.route_refusals),
            "error_kinds": error_kinds,
            "first_choice": first_choice,
            "duration_seconds": round(elapsed, 6),
        })
        if self.progress is not None:
            self.progress.record_completed(
                row,
                resume_key=key,
                route_calls=context.route_calls,
                route_errors=context.route_errors,
            )
        self._emit("episode_completed", {"resume_key": key, "status": "completed"})
        return {"status": "completed", "requeue": False, "key": key}

    # -- detention phase -----------------------------------------
    def _detention_dir(self) -> Path:
        return self.workdir / DEFAULT_DETENTION_DIR

    def _detention_adapters(self, row: Mapping, resume_key: str):
        directory = self._detention_dir() / DEFAULT_EPISODE_CALL_DIR
        slots = self._slots_for_row(row)
        orchestrator_route = str(slots[SLOT_SELF]["route_id"])
        dispatcher = self._dispatchers[orchestrator_route]
        orchestrator = RouteAdapter(
            dispatcher,
            self._episode_adapter(dispatcher.adapter, row, resume_key, directory),
            model_id=str(slots[SLOT_SELF]["model_id"]),
            vendor_family=str(slots[SLOT_SELF]["publisher"]),
        )
        peers: dict[str, RouteAdapter] = {}
        for slot_name, route in slots.items():
            route_id = str(route["route_id"])
            dispatcher = self._dispatchers[route_id]
            peers[f"peer_{slot_name}"] = RouteAdapter(
                dispatcher,
                self._episode_adapter(dispatcher.adapter, row, resume_key, directory),
                model_id=str(route["model_id"]),
                vendor_family=str(route["publisher"]),
            )
        return orchestrator, peers

    def run_detention_phase(self) -> dict:
        """Execute the frozen Detention retry phase after a completed main phase.

        The runner now owns this path.  It refuses to start without a signed,
        completed main summary and a verifying ledger, uses the frozen F=660 eligibility
        cap / 4-arm table / one retry, routes every review and retry through the same
        dispatchers, ledger, signer and call cap, and writes a separate signed summary
        that is never merged into the main denominator.
        """
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._acquire_run_lock()
        try:
            return self._run_detention_phase_locked()
        finally:
            self._release_run_lock()

    def _run_detention_phase_locked(self) -> dict:
        from .constants import DETENTION_ELIGIBLE_CAP_V13
        from .detention import (
            ARM_ELIMINATION,
            DETENTION_ARMS,
            DETENTION_ARMS_V14,
            PILOT_ELIGIBLE_CAP,
            DetentionReviewError,
            arm_specs,
            arm_specs_v14,
            build_arm_schedule,
            pre_registration,
            pre_registration_v14,
            run_detention_arm,
            select_eligible,
        )

        v14 = self.detention_protocol == "v14"
        arms_table = tuple(DETENTION_ARMS_V14 if v14 else DETENTION_ARMS)

        summary_path = self.workdir / DEFAULT_SUMMARY
        if not summary_path.is_file():
            raise RunnerConfigError(
                "detention phase refused: no main-phase run_summary.json in this run "
                "directory"
            )
        try:
            main_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunnerConfigError(
                f"detention phase refused: main summary is unreadable ({exc})"
            ) from None
        if not isinstance(main_summary, dict):
            raise RunnerConfigError("detention phase refused: main summary is not an object")
        signer = self._ensure_signer()
        if not verify_signed_mapping(main_summary, signer.key):
            raise RunnerConfigError(
                "detention phase refused: main summary signature does not verify"
            )
        if str(main_summary.get("status")) not in ("completed", "completed_limited"):
            raise RunnerConfigError(
                "detention phase refused: main phase status is "
                f"{main_summary.get('status')!r}, not completed"
            )
        # Review finding 3: make the CURRENT ledger + completed main inventory the gate,
        # under the lock already acquired by run_detention_phase().  This is the same helper
        # the main entry point uses; the signed summary's historical flag is not sufficient.
        integrity = self._verify_integrity_gate(
            main_summary, require_completed_main=True
        )
        self._run_scope = str(main_summary.get("run_scope") or self._run_scope)
        # The main phase's ``finally`` sets the stop event as part of its normal wind-down.
        # ``main()`` runs this phase on the same runner object straight afterwards, so without
        # this every Detention route call raised RunStopped ("run stopped before the route call
        # started") and no live Detention row could ever complete.  A real stop (fatal reason,
        # substitution, auth expiry) is not cleared.
        if not self._fatal_reason:
            self._stop_event.clear()

        prereg = pre_registration_v14() if v14 else pre_registration()
        frozen_arms = [str(item.get("arm")) for item in prereg.get("arms") or []]
        current_arms = [
            str(item.get("arm")) for item in (arm_specs_v14() if v14 else arm_specs())
        ]
        if tuple(frozen_arms) != arms_table or tuple(current_arms) != arms_table:
            raise RunnerConfigError(
                f"detention phase refused: the {'v14 pilot' if v14 else 'frozen'} arm table "
                "does not verify"
            )
        schedule_spec = prereg.get("arm_schedule") or {}
        if tuple(schedule_spec.get("arms") or ()) != arms_table:
            raise RunnerConfigError(
                "detention phase refused: the frozen arm schedule does not verify"
            )
        cap = int(prereg.get("eligible_cap") or 0)
        expected_cap = int(PILOT_ELIGIBLE_CAP if v14 else DETENTION_ELIGIBLE_CAP_V13)
        if cap != expected_cap:
            raise RunnerConfigError(
                "detention phase refused: eligibility cap does not match the frozen "
                f"{'pilot' if v14 else 'V13'} cap {expected_cap}"
            )
        selection_seed = int((prereg.get("sample") or {}).get("seed") or 20261114)
        schedule_seed = int(schedule_spec.get("seed") or 20261114)

        detention = self._detention_dir()
        detention_result_dir = detention / DEFAULT_EPISODE_DIR
        detention_call_dir = detention / DEFAULT_EPISODE_CALL_DIR
        detention_result_dir.mkdir(parents=True, exist_ok=True)
        detention_call_dir.mkdir(parents=True, exist_ok=True)
        selection_path = detention / "detention_selection.json"
        schedule_path = detention / "detention_arm_schedule.json"
        state_path = detention / "detention_state.json"
        main_summary_sha256 = sha256_file(summary_path)

        # Load only signed main-phase results.  An unsigned or tampered result refuses the
        # whole phase; a detention row is never built on unverified evidence.
        results: dict[str, dict] = {}
        for path in sorted(self._result_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RunnerConfigError(
                    f"detention phase refused: unreadable result {path.name} ({exc})"
                ) from None
            if not isinstance(payload, dict) or not verify_signed_mapping(
                payload, signer.key
            ):
                raise RunnerConfigError(
                    f"detention phase refused: result {path.name} does not verify"
                )
            if str(payload.get("status")) != "completed":
                continue
            row = payload.get("row") or {}
            episode_id = str(
                row.get("episode_id")
                or (payload.get("result") or {}).get("episode_id")
                or ""
            )
            if not episode_id:
                continue
            results[episode_id] = {"payload": payload, "row": row}

        candidates: list[dict] = []
        cells_by_hash: dict[str, object] = {}
        for episode_id, item in results.items():
            try:
                cell = self._cell_for_row(item["row"])
            except Exception:
                continue
            cells_by_hash[str(cell.cell_hash)] = cell
            candidate = item["payload"].get("result") or {}
            if isinstance(candidate, dict) and candidate:
                candidates.append(candidate)

        # Review finding 2: selection is persisted once and resumed by binding, not
        # recomputed (resampled) on every phase invocation.
        if selection_path.is_file():
            try:
                selection_payload = json.loads(
                    selection_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise RunnerConfigError(
                    f"detention phase refused: persisted selection is unreadable ({exc})"
                ) from None
            if (
                not isinstance(selection_payload, dict)
                or not verify_signed_mapping(selection_payload, signer.key)
                or selection_payload.get("kind") != "vera.v13_detention_selection.v1"
            ):
                raise RunnerConfigError(
                    "detention phase refused: persisted selection does not verify"
                )
            if str(selection_payload.get("main_summary_sha256") or "") != main_summary_sha256:
                raise RunnerConfigError(
                    "detention phase refused: persisted selection binds a different main "
                    "summary"
                )
            selected_ids = {
                str(value) for value in selection_payload.get("selected_episode_ids") or []
            }
            selected = [
                dict(item) for item in candidates
                if str(item.get("episode_id")) in selected_ids
            ]
            if len(selected) != len(selected_ids):
                raise RunnerConfigError(
                    "detention phase refused: persisted selected episodes do not reconcile "
                    "with signed main results"
                )
            n_selected = selection_payload.get("n_selected")
            if (
                not isinstance(n_selected, int)
                or isinstance(n_selected, bool)
                or n_selected != len(selected_ids)
            ):
                raise RunnerConfigError(
                    "detention phase refused: persisted selection n_selected does not "
                    "reconcile"
                )
        else:
            selection = select_eligible(
                candidates,
                cap=cap,
                seed=selection_seed,
                cells_by_hash=cells_by_hash,
                mode="sibling" if v14 else "retry",
            )
            selected_ids = {
                str(row.get("episode_id"))
                for row in selection.get("ledger") or []
                if row.get("selected") is True
            }
            selected = [
                dict(item)
                for item in candidates
                if str(item.get("episode_id")) in selected_ids
            ]
            if len(selected) != int(selection.get("n_selected") or 0):
                raise RunnerConfigError(
                    "detention phase refused: selected facts do not reconcile with the "
                    "frozen selector's n_selected"
                )
            selection_payload = {
                "kind": "vera.v13_detention_selection.v1",
                **self._label_fields(),
                "run_id": self.run_id,
                "run_scope": self._run_scope,
                "cap": cap,
                "eligible_cap": cap,
                "n_selected": len(selected),
                "arm_table": list(arms_table),
                "protocol": self.detention_protocol,
                "two_fix_mechanisms_excluded": list(
                    selection.get("two_fix_mechanisms_excluded") or []
                ),
                "seed": selection_seed,
                "created_at_utc": utc_now(),
                "main_summary_file": str(summary_path),
                "main_summary_sha256": main_summary_sha256,
                "selected_episode_ids": [
                    str(item.get("episode_id")) for item in selected
                ],
                "selected_episodes": copy.deepcopy(selection.get("episodes") or []),
                "allocation": selection.get("allocation"),
                "eligible_ledger": selection.get("ledger"),
            }
            if not selection_payload["selected_episodes"]:
                selection_payload["selected_episodes"] = [
                    {
                        "episode_id": str(item.get("episode_id")),
                        "cell_hash": str(item.get("cell_hash")),
                        "sampling_weight": None,
                        "selection_probability": None,
                    }
                    for item in selected
                ]
            selection_payload = signer.sign_mapping(selection_payload)
            atomic_write_json(selection_path, selection_payload)
        selection_sha256 = sha256_file(selection_path)
        self._append_ledger({
            "kind": "detention_selection",
            "run_scope": self._run_scope,
            "phase": "detention",
            "cap": cap,
            "arms": list(arms_table),
            "selected_episodes": len(selected),
            "eligible_candidates": len(candidates),
            "selection_file": str(selection_path),
            "selection_sha256": selection_sha256,
            "main_summary_sha256": main_summary_sha256,
            "resumed": bool(selection_path.is_file() and selected_ids == {
                str(item.get("episode_id")) for item in selected
            }),
        })
        self._emit("detention_selection", {
            "selected_episodes": len(selected),
            "cap": cap,
        })

        # The arm schedule is built, bound and persisted BEFORE any Detention adapter call.
        schedule_episodes = copy.deepcopy(
            selection_payload.get("selected_episodes") or []
        )
        if not schedule_episodes:
            schedule_episodes = [
                {
                    "episode_id": str(item.get("episode_id")),
                    "cell_hash": str(item.get("cell_hash")),
                    "sampling_weight": None,
                    "selection_probability": None,
                }
                for item in selected
            ]
        expected_schedule = build_arm_schedule(
            schedule_episodes,
            seed=schedule_seed,
            arms=arms_table,
        )
        if schedule_path.is_file():
            try:
                schedule_payload = json.loads(schedule_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RunnerConfigError(
                    f"detention phase refused: persisted arm schedule is unreadable ({exc})"
                ) from None
            if (
                not isinstance(schedule_payload, dict)
                or not verify_signed_mapping(schedule_payload, signer.key)
                or schedule_payload.get("kind") != "vera.v13_detention_arm_schedule.v1"
            ):
                raise RunnerConfigError(
                    "detention phase refused: persisted arm schedule does not verify"
                )
            if str(schedule_payload.get("main_summary_sha256") or "") != main_summary_sha256:
                raise RunnerConfigError(
                    "detention phase refused: persisted arm schedule binds a different main "
                    "summary"
                )
            if str(schedule_payload.get("selection_sha256") or "") != selection_sha256:
                raise RunnerConfigError(
                    "detention phase refused: persisted arm schedule binds a different "
                    "selection"
                )
            if int(schedule_payload.get("seed") or -1) != schedule_seed:
                raise RunnerConfigError(
                    "detention phase refused: persisted arm schedule seed changed"
                )
            if _signing_bytes(schedule_payload.get("rows") or []) != _signing_bytes(
                expected_schedule["rows"]
            ):
                raise RunnerConfigError(
                    "detention phase refused: persisted arm schedule was resampled or "
                    "tampered"
                )
            schedule = schedule_payload
        else:
            schedule = {
                "kind": "vera.v13_detention_arm_schedule.v1",
                **self._label_fields(),
                "run_id": self.run_id,
                "run_scope": self._run_scope,
                "phase": "detention",
                "method": "seeded block randomization, one of each arm per episode",
                "seed": schedule_seed,
                "arms": list(arms_table),
                "main_summary_file": str(summary_path),
                "main_summary_sha256": main_summary_sha256,
                "selection_file": str(selection_path),
                "selection_sha256": selection_sha256,
                "created_at_utc": utc_now(),
                "rows": copy.deepcopy(expected_schedule["rows"]),
            }
            schedule = signer.sign_mapping(schedule)
            atomic_write_json(schedule_path, schedule)
        schedule_sha256 = sha256_file(schedule_path)
        schedule_rows = list(schedule.get("rows") or [])
        rows_planned = len(schedule_rows)
        self._append_ledger({
            "kind": "detention_arm_schedule",
            "phase": "detention",
            "phase_namespace": "detention",
            "schedule_file": str(schedule_path),
            "schedule_sha256": schedule_sha256,
            "schedule_digest": json_digest(schedule_rows),
            "seed": schedule_seed,
            "rows_planned": rows_planned,
            "episodes": len(selected),
        })

        # Phase-local persistent state: interruption disposition and cumulative counters.
        state: dict = {
            "kind": "vera.v13_detention_state.v1",
            **self._label_fields(),
            "run_id": self.run_id,
            "run_scope": self._run_scope,
            "phase": "detention",
            "main_summary_sha256": main_summary_sha256,
            "selection_sha256": selection_sha256,
            "schedule_sha256": schedule_sha256,
            "status": "running",
            "last_disposition": "scheduled",
            "interrupted_episode_id": "",
            "interrupted_arm": "",
            "rows_completed_total": 0,
            "rows_failed_total": 0,
            "rows_interrupted_total": 0,
            "rows_refused_total": 0,  # v14b: rows whose review call was refused by the provider safeguard (after one retry)
            "route_calls": {},
            "route_errors": {},
            "route_refusals": {},
            "latency_by_route": {},
            "updated_at_utc": utc_now(),
        }
        if state_path.is_file():
            try:
                loaded_state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RunnerConfigError(
                    f"detention phase refused: phase state is unreadable ({exc})"
                ) from None
            if (
                not isinstance(loaded_state, dict)
                or not verify_signed_mapping(loaded_state, signer.key)
                or loaded_state.get("kind") != "vera.v13_detention_state.v1"
            ):
                raise RunnerConfigError(
                    "detention phase refused: phase state does not verify"
                )
            for field_name in (
                "main_summary_sha256", "selection_sha256", "schedule_sha256"
            ):
                if str(loaded_state.get(field_name) or "") != str(state[field_name]):
                    raise RunnerConfigError(
                        "detention phase refused: phase state binds different phase bytes "
                        f"({field_name})"
                    )
            state.update({
                "status": str(loaded_state.get("status") or "running"),
                "last_disposition": str(
                    loaded_state.get("last_disposition") or "scheduled"
                ),
                "interrupted_episode_id": str(
                    loaded_state.get("interrupted_episode_id") or ""
                ),
                "interrupted_arm": str(loaded_state.get("interrupted_arm") or ""),
                "rows_completed_total": int(
                    loaded_state.get("rows_completed_total") or 0
                ),
                "rows_failed_total": int(
                    loaded_state.get("rows_failed_total") or 0
                ),
                "rows_interrupted_total": int(
                    loaded_state.get("rows_interrupted_total") or 0
                ),
                "rows_refused_total": int(loaded_state.get("rows_refused_total") or 0),
                "route_refusals": {
                    str(key): int(value)
                    for key, value in dict(loaded_state.get("route_refusals") or {}).items()
                },
                "route_calls": {
                    str(key): int(value)
                    for key, value in dict(loaded_state.get("route_calls") or {}).items()
                },
                "route_errors": {
                    str(key): int(value)
                    for key, value in dict(loaded_state.get("route_errors") or {}).items()
                },
                "latency_by_route": {
                    str(key): [float(item) for item in list(value or [])]
                    for key, value in dict(
                        loaded_state.get("latency_by_route") or {}
                    ).items()
                },
            })

        def persist_state(
            status_value: str,
            disposition: str,
            *,
            episode_id: str = "",
            arm: str = "",
        ) -> dict:
            payload = {
                **state,
                "status": str(status_value),
                "last_disposition": str(disposition),
                "interrupted_episode_id": str(
                    episode_id or state.get("interrupted_episode_id") or ""
                ),
                "interrupted_arm": str(arm or state.get("interrupted_arm") or ""),
                "updated_at_utc": utc_now(),
            }
            payload = signer.sign_mapping(payload)
            atomic_write_json(state_path, payload)
            return payload

        persist_state("running", "scheduled")

        # Resume from signed Detention result files, never from a fresh key resample.
        # Review finding 2: reconcile the signed completion ledger, the signed result
        # inventory and the persisted arm schedule in both directions before any new
        # adapter call.  A completed ledger key whose result file is missing or mismatched
        # is a refusal: the original artifact must be recovered, never reacquired.
        expected_paths: dict[str, Path] = {}
        schedule_by_key: dict[str, dict] = {}
        for row in schedule_rows:
            episode_id = str(row.get("episode_id") or "")
            arm = str(row.get("arm") or "")
            key = (
                f"phase=detention|episode_id={episode_id}|"
                f"arm={arm}|repeat=0"
            )
            if key in expected_paths:
                raise RunnerConfigError(
                    "detention phase refused: persisted arm schedule has a duplicate "
                    f"key {key!r}"
                )
            expected_paths[key] = detention_result_dir / (
                safe_component(sha256_text(key)[:24]) + ".json"
            )
            schedule_by_key[key] = dict(row)

        result_inventory: dict[str, dict] = {}
        for key, result_path in expected_paths.items():
            episode_id = str(schedule_by_key[key].get("episode_id") or "")
            arm = str(schedule_by_key[key].get("arm") or "")
            if not result_path.is_file():
                continue
            try:
                completed_payload = json.loads(
                    result_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise RunnerConfigError(
                    f"detention phase refused: completed result {result_path.name} is "
                    f"unreadable ({exc})"
                ) from None
            if (
                not isinstance(completed_payload, dict)
                or not verify_signed_mapping(completed_payload, signer.key)
                or completed_payload.get("kind") != DETENTION_RESULT_KIND
            ):
                raise RunnerConfigError(
                    f"detention phase refused: completed result {result_path.name} does "
                    "not verify"
                )
            if (
                str(completed_payload.get("status")) != "completed"
                or str(completed_payload.get("resume_key") or "") != key
                or str(completed_payload.get("episode_id") or "") != episode_id
                or str(completed_payload.get("arm") or "") != arm
            ):
                raise RunnerConfigError(
                    f"detention phase refused: completed result {result_path.name} is not "
                    "bound to its schedule key"
                )
            outcome = completed_payload.get("outcome")
            if not isinstance(outcome, dict) or json_digest(outcome) != str(
                completed_payload.get("outcome_digest") or ""
            ):
                raise RunnerConfigError(
                    f"detention phase refused: completed result {result_path.name} outcome "
                    "digest does not verify"
                )
            result_inventory[key] = completed_payload

        scheduled_names = {path.name for path in expected_paths.values()}
        for path in sorted(detention_result_dir.glob("*.json")):
            if path.name not in scheduled_names:
                raise RunnerConfigError(
                    f"detention phase refused: result file {path.name} is not bound to "
                    "the persisted arm schedule"
                )

        ledger_records, ledger_problems = read_jsonl(self._ledger_path)
        if ledger_problems:
            raise RunnerConfigError(
                "detention phase refused: completion ledger is unreadable: "
                + "; ".join(map(str, ledger_problems[:5]))
            )
        completed_ledger: dict[str, Mapping] = {}
        for record in ledger_records:
            if str(record.get("kind") or "") not in (
                "detention_row_completed",
                "detention_row_adopted",
            ):
                continue
            key = str(record.get("resume_key") or "")
            if not key or key not in expected_paths:
                raise RunnerConfigError(
                    "detention phase refused: completion ledger key "
                    f"{key!r} is not in the persisted arm schedule"
                )
            if key in completed_ledger:
                raise RunnerConfigError(
                    "detention phase refused: completion ledger has more than one "
                    f"record for {key!r}"
                )
            schedule_row = schedule_by_key[key]
            if (
                str(record.get("episode_id") or "")
                != str(schedule_row.get("episode_id") or "")
                or str(record.get("arm") or "")
                != str(schedule_row.get("arm") or "")
            ):
                raise RunnerConfigError(
                    "detention phase refused: completion ledger record for "
                    f"{key!r} is not bound to its schedule row"
                )
            completed_ledger[key] = record

        completed: dict[str, dict] = {}
        for key, record in completed_ledger.items():
            completed_payload = result_inventory.get(key)
            if completed_payload is None:
                raise RunnerConfigError(
                    "detention phase refused: completed ledger key "
                    f"{key!r} has no signed Detention result; recover the original "
                    "artifact instead of reacquiring it"
                )
            record_digest = str(record.get("result_digest") or "")
            result_digest = str(completed_payload.get("outcome_digest") or "")
            if not record_digest or record_digest != result_digest:
                raise RunnerConfigError(
                    "detention phase refused: completed ledger outcome digest for "
                    f"{key!r} does not match the signed Detention result"
                )
            recorded_file = str(record.get("result_file") or "")
            if recorded_file and Path(recorded_file).name != expected_paths[key].name:
                raise RunnerConfigError(
                    "detention phase refused: completed ledger result file for "
                    f"{key!r} is not the scheduled artifact"
                )
            completed[key] = completed_payload

        # A crash can leave an atomically written result before its ledger append.  Adopt
        # that signed result and heal the ledger, without making another adapter call.
        for key in sorted(set(result_inventory) - set(completed)):
            completed_payload = result_inventory[key]
            schedule_row = schedule_by_key[key]
            self._append_ledger({
                "kind": "detention_row_adopted",
                "phase": "detention",
                "resume_key": key,
                "episode_id": str(schedule_row.get("episode_id") or ""),
                "arm": str(schedule_row.get("arm") or ""),
                "repeat": 0,
                "status": "completed",
                "result_file": str(expected_paths[key]),
                "result_digest": str(completed_payload.get("outcome_digest") or ""),
                "route_calls": dict(completed_payload.get("route_calls") or {}),
                "route_errors": dict(completed_payload.get("route_errors") or {}),
                "rows_completed_total": int(state["rows_completed_total"]),
            })
            completed[key] = completed_payload
        rows_previously_completed = len(completed)
        state["rows_completed_total"] = max(
            int(state["rows_completed_total"]), rows_previously_completed
        )

        # Build fresh dispatchers for this phase; the main-phase pools are shut down.
        if not self.dry_run:
            for dispatcher in self._dispatchers.values():
                dispatcher.shutdown()
            self._dispatchers.clear()
            self._route_defs.clear()
            self._build_dispatchers()

        rows_completed = 0
        rows_failed = 0
        rows_interrupted = 0
        rows_refused = 0
        stopped = False
        route_calls: Counter = Counter(state.get("route_calls") or {})
        route_errors: Counter = Counter(state.get("route_errors") or {})
        route_refusals: Counter = Counter(state.get("route_refusals") or {})
        refused_keys: set[str] = set()  # rows refused in THIS invocation (a refused row is re-attempted on resume)
        latency_by_route: dict[str, list[float]] = {
            str(key): [float(item) for item in list(value or [])]
            for key, value in dict(state.get("latency_by_route") or {}).items()
        }

        def absorb_context(context) -> None:
            for route_id, count in context.route_calls.items():
                route_calls[route_id] += int(count)
            for route_id, count in context.route_errors.items():
                route_errors[route_id] += int(count)
            for route_id, count in context.route_refusals.items():
                route_refusals[route_id] += int(count)
            state["route_refusals"] = dict(sorted(route_refusals.items()))
            for record in context.call_latencies:
                latency_by_route.setdefault(
                    str(record.get("route_id")), []
                ).append(float(record.get("latency_ms") or 0.0))
            state["route_calls"] = dict(sorted(route_calls.items()))
            state["route_errors"] = dict(sorted(route_errors.items()))
            state["latency_by_route"] = {
                key: list(values) for key, values in sorted(latency_by_route.items())
            }

        try:
            for schedule_row in schedule_rows:
                episode_id = str(schedule_row.get("episode_id") or "")
                arm = str(schedule_row.get("arm") or "")
                key = (
                    f"phase=detention|episode_id={episode_id}|"
                    f"arm={arm}|repeat=0"
                )
                if key in completed:
                    continue
                item = results.get(episode_id)
                if item is None:
                    raise RunnerConfigError(
                        "detention phase refused: selected episode "
                        f"{episode_id!r} has no signed main result"
                    )
                row = dict(item["row"])
                failed_episode = dict(item["payload"].get("result") or {})
                cell = cells_by_hash.get(str(schedule_row.get("cell_hash"))) or (
                    self._cell_for_row(row)
                )
                self._append_ledger({
                    "kind": "detention_row_started",
                    "phase": "detention",
                    "resume_key": key,
                    "episode_id": episode_id,
                    "arm": arm,
                    "repeat": 0,
                    "cell_hash": str(cell.cell_hash),
                    "schedule_position": int(schedule_row.get("position") or 0),
                })
                # One row = one or more attempts.  After every attempt the route errors
                # are classified by the same helper the main phase uses; a row that hit a route
                # limit, auth expiry, substitution or transport error is NEVER written as
                # completed (it used to be: the arm swallowed the errored call and carried on).
                row_result = "completed"
                attempt = 0
                requeue_count = 0
                while True:
                    attempt += 1
                    context = EpisodeContext(resume_key=key, row=row)
                    started = self.clock()
                    if self.dry_run or arm == ARM_ELIMINATION:
                        # Arm E is scripted: no route, no model call, in a live run as well.
                        orchestrator, peers = default_mock_adapters(cell)
                        orchestrator, peers = self._record_mocks(
                            orchestrator,
                            peers,
                            row,
                            key,
                            self._detention_dir() / DEFAULT_EPISODE_CALL_DIR,
                        )
                    else:
                        orchestrator, peers = self._detention_adapters(row, key)
                    _set_episode_context(context)
                    arm_error: Exception | None = None
                    stop_error: RunStopped | None = None
                    outcome = None
                    try:
                        outcome = run_detention_arm(
                            cell,
                            failed_episode,
                            orchestrator,
                            peers,
                            arm,
                            harness_key=b"v13-runner-harness-"
                            + cell.cell_hash[:20].encode(),
                            verifier_key=b"v13-runner-verifier-"
                            + cell.cell_hash[:20].encode(),
                            sibling_transfer=v14,
                        )
                    except RunStopped as exc:
                        stop_error = exc
                    except Exception as exc:
                        arm_error = exc
                    finally:
                        _set_episode_context(None)
                    if isinstance(arm_error, DetentionReviewError) and arm_error.error_kind:
                        # The errored call is evidence even where no dispatcher recorded it
                        # (a dry run bypasses the route pools).
                        if arm_error.error_kind == ERROR_PROVIDER_REFUSAL:
                            # v14b: a refusal is never an error kind; count it only if no dispatcher did
                            if not context.route_refusals:
                                context.route_refusals[str(getattr(cell, "orchestrator_model_id", "") or "unrecorded")] += 1
                        elif arm_error.error_kind not in context.error_kinds:
                            context.error_kinds.append(arm_error.error_kind)
                    absorb_context(context)
                    if stop_error is not None:
                        rows_interrupted += 1
                        state["rows_interrupted_total"] = (
                            int(state["rows_interrupted_total"]) + 1
                        )
                        state.update({
                            "interrupted_episode_id": episode_id,
                            "interrupted_arm": arm,
                        })
                        stopped = True
                        self._append_ledger({
                            "kind": "detention_row_interrupted",
                            "phase": "detention",
                            "resume_key": key,
                            "episode_id": episode_id,
                            "arm": arm,
                            "reason": RUN_STOPPED,
                            "message": str(stop_error),
                            "rows_completed_total": int(state["rows_completed_total"]),
                        })
                        persist_state(
                            "stopped_incomplete",
                            "interrupted",
                            episode_id=episode_id,
                            arm=arm,
                        )
                        self._emit("detention_row_interrupted", {
                            "resume_key": key,
                            "reason": RUN_STOPPED,
                            "interrupted_arm": arm,
                        })
                        row_result = "stop"
                        break
                    verdict = self._episode_verdict(context, attempt)
                    if verdict["verdict"] == "interrupted":
                        rows_interrupted += 1
                        state["rows_interrupted_total"] = (
                            int(state["rows_interrupted_total"]) + 1
                        )
                        can_requeue = bool(
                            verdict["requeue"]
                            and requeue_count < DETENTION_MAX_REQUEUES
                            and not self._stop_event.is_set()
                        )
                        self._append_ledger({
                            "kind": "detention_row_interrupted",
                            "phase": "detention",
                            "resume_key": key,
                            "episode_id": episode_id,
                            "arm": arm,
                            "attempt": attempt,
                            "reason": verdict["reason"],
                            "requeue": can_requeue,
                            "error_kinds": verdict["error_kinds"],
                            "route_calls": dict(context.route_calls),
                            "route_errors": dict(context.route_errors),
                            "rows_completed_total": int(state["rows_completed_total"]),
                        })
                        self._emit("detention_row_interrupted", {
                            "resume_key": key,
                            "reason": verdict["reason"],
                            "interrupted_arm": arm,
                        })
                        if can_requeue:
                            requeue_count += 1
                            persist_state(
                                "running", "requeued", episode_id=episode_id, arm=arm
                            )
                            continue
                        state.update({
                            "interrupted_episode_id": episode_id,
                            "interrupted_arm": arm,
                        })
                        stopped = True
                        persist_state(
                            "stopped_incomplete",
                            "interrupted",
                            episode_id=episode_id,
                            arm=arm,
                        )
                        row_result = "stop"
                        break
                    if verdict["verdict"] == "infra_retry":
                        self._append_ledger({
                            "kind": "detention_row_infrastructure_retry",
                            "phase": "detention",
                            "resume_key": key,
                            "episode_id": episode_id,
                            "arm": arm,
                            "attempt": attempt,
                            "error_kinds": verdict["error_kinds"],
                            "route_calls": dict(context.route_calls),
                            "route_errors": dict(context.route_errors),
                        })
                        continue
                    if verdict["verdict"] == "infra_terminal":
                        rows_failed += 1
                        state["rows_failed_total"] = int(state["rows_failed_total"]) + 1
                        fatal_infra = bool(verdict["fatal_kinds"])
                        self._append_ledger({
                            "kind": "detention_row_failed",
                            "phase": "detention",
                            "resume_key": key,
                            "episode_id": episode_id,
                            "arm": arm,
                            "attempt": attempt,
                            "reason": "INFRASTRUCTURE_FAILURE",
                            "error_kinds": verdict["error_kinds"],
                            "fatal": fatal_infra,
                            "message": "" if arm_error is None else (
                                f"{type(arm_error).__name__}: {arm_error}"
                            ),
                            "rows_failed_total": int(state["rows_failed_total"]),
                        })
                        self._emit("detention_row_failed", {
                            "resume_key": key,
                            "error_kinds": verdict["error_kinds"],
                            "fatal": fatal_infra,
                        })
                        if fatal_infra:
                            self._fatal_reason = "INFRASTRUCTURE_FATAL"
                            self._state["stopped"] = "INFRASTRUCTURE_FATAL"
                            stopped = True
                            persist_state(
                                "stopped_incomplete",
                                "interrupted",
                                episode_id=episode_id,
                                arm=arm,
                            )
                            row_result = "stop"
                        else:
                            persist_state(
                                "running", "failed", episode_id=episode_id, arm=arm
                            )
                            row_result = "skip"
                        break
                    if (
                        isinstance(arm_error, DetentionReviewError)
                        and arm_error.error_kind == ERROR_PROVIDER_REFUSAL
                    ):
                        # v14b: the provider's safeguard declined the review prompt, again after the one retry the route
                        # dispatcher makes.  This is the arm's own recorded outcome (PROVIDER_REFUSAL): not an infrastructure
                        # or route error, not a harness fault, and it never interrupts or stops the phase.  Only real
                        # transport/auth/route errors reach the branches above.  The refused row is not a completed result
                        # (no outcome exists), so it is written to its own signed record and re-attempted on resume.
                        rows_refused += 1
                        refused_keys.add(key)
                        state["rows_refused_total"] = int(state["rows_refused_total"]) + 1
                        refusal_record = signer.sign_mapping({
                            "kind": "vera.v14b.detention_refusal.v1",
                            **self._label_fields(),
                            "run_id": self.run_id,
                            "phase": "detention",
                            "resume_key": key,
                            "episode_id": episode_id,
                            "arm": arm,
                            "outcome": ERROR_PROVIDER_REFUSAL,
                            "message": str(arm_error.error)[:400],
                            "route_calls": dict(context.route_calls),
                            "route_refusals": dict(context.route_refusals),
                            "call_latencies": list(context.call_latencies),
                            "recorded_at_utc": utc_now(),
                        })
                        refusal_dir = detention / "refusals"
                        refusal_dir.mkdir(parents=True, exist_ok=True)
                        atomic_write_json(
                            refusal_dir / (safe_component(sha256_text(key)[:24]) + ".json"), refusal_record
                        )
                        self._append_ledger({
                            "kind": "detention_row_refused",
                            "phase": "detention",
                            "resume_key": key,
                            "episode_id": episode_id,
                            "arm": arm,
                            "outcome": ERROR_PROVIDER_REFUSAL,
                            "route_calls": dict(context.route_calls),
                            "route_refusals": dict(context.route_refusals),
                            "rows_refused_total": int(state["rows_refused_total"]),
                        })
                        persist_state("running", "refused", episode_id=episode_id, arm=arm)
                        self._emit("detention_row_refused", {"resume_key": key, "arm": arm})
                        row_result = "skip"
                        break
                    if arm_error is not None:
                        rows_failed += 1
                        state["rows_failed_total"] = int(state["rows_failed_total"]) + 1
                        self._append_ledger({
                            "kind": "detention_row_failed",
                            "phase": "detention",
                            "resume_key": key,
                            "episode_id": episode_id,
                            "arm": arm,
                            "reason": HARNESS_FAULT,
                            "message": f"{type(arm_error).__name__}: {arm_error}",
                            "rows_failed_total": int(state["rows_failed_total"]),
                        })
                        persist_state(
                            "running", "failed", episode_id=episode_id, arm=arm
                        )
                        self._emit("detention_row_failed", {
                            "resume_key": key,
                            "error": f"{type(arm_error).__name__}: {arm_error}",
                        })
                        row_result = "skip"
                        break
                    break
                if row_result == "stop":
                    break
                if row_result == "skip":
                    continue
                elapsed = max(0.0, self.clock() - started)
                result_payload = {
                    "kind": DETENTION_RESULT_KIND,
                    **self._label_fields(),
                    "run_id": self.run_id,
                    "run_scope": self._run_scope,
                    "phase": "detention",
                    "resume_key": key,
                    "episode_id": episode_id,
                    "arm": arm,
                    "repeat": 0,
                    "cell_hash": str(cell.cell_hash),
                    "status": "completed",
                    "completed_at_utc": utc_now(),
                    "duration_seconds": round(elapsed, 6),
                    "route_calls": dict(context.route_calls),
                    "route_errors": dict(context.route_errors),
                    "route_refusals": dict(context.route_refusals),
                    "route_latency": context.latency_summary(),
                    "call_latencies": list(context.call_latencies),
                    "served_model_checks": dict(context.served_model_checks),
                    "outcome_digest": json_digest(outcome),
                    "outcome": outcome,
                }
                # The digest of this row's call log (transcripts included).
                row_call_log = (
                    {}
                    if self.dry_run and not self.record_transcripts
                    else self._episode_call_log_meta(
                        key, self._detention_dir() / DEFAULT_EPISODE_CALL_DIR
                    )
                )
                if row_call_log:
                    result_payload["episode_call_log"] = row_call_log
                result_path = detention_result_dir / (
                    safe_component(sha256_text(key)[:24]) + ".json"
                )
                result_payload = signer.sign_mapping(result_payload)
                atomic_write_json(result_path, result_payload)
                completed[key] = result_payload
                rows_completed += 1
                state["rows_completed_total"] = int(state["rows_completed_total"]) + 1
                self._append_ledger({
                    "kind": "detention_row_completed",
                    "phase": "detention",
                    "resume_key": key,
                    "episode_id": episode_id,
                    "arm": arm,
                    "repeat": 0,
                    "status": "completed",
                    "rca_valid": outcome.get("rca_valid"),
                    "retry_visible_success": outcome.get("retry_visible_success"),
                    "retry_owned_success": outcome.get("retry_owned_success"),
                    "route_calls": dict(context.route_calls),
                    "route_errors": dict(context.route_errors),
                    "result_file": str(result_path),
                    "result_digest": result_payload["outcome_digest"],
                    "duration_seconds": round(elapsed, 6),
                    "rows_completed_total": int(state["rows_completed_total"]),
                })
                persist_state(
                    "running", "completed", episode_id=episode_id, arm=arm
                )
                self._emit("detention_row_completed", {
                    "resume_key": key,
                    "arm": arm,
                })
        finally:
            if not self.dry_run:
                for dispatcher in self._dispatchers.values():
                    dispatcher.shutdown()

        rows_completed_total = int(state["rows_completed_total"])
        rows_failed_total = int(state["rows_failed_total"])
        rows_interrupted_total = int(state["rows_interrupted_total"])
        rows_refused_total = int(state.get("rows_refused_total") or 0)
        # v14b: a row refused by the provider safeguard is accounted for (it has its own recorded outcome), so a phase in which
        # every planned row is completed or refused is a completed phase; it is reported as such, with the refusals counted.
        all_planned_completed = len(completed) + len(refused_keys - set(completed)) >= rows_planned
        if stopped:
            status = "stopped_incomplete"
        elif all_planned_completed:
            status = "completed"
        elif rows_failed_total:
            status = "completed_with_infrastructure_failures"
        else:
            status = "stopped_incomplete"
        final_state = persist_state(
            status,
            "interrupted" if stopped else "phase_finished",
        )
        route_latency = {
            route_id: summarise_latencies(values)
            for route_id, values in sorted(latency_by_route.items())
        }
        summary = {
            "kind": DETENTION_SUMMARY_KIND,
            **self._label_fields(),
            "run_id": self.run_id,
            "run_scope": self._run_scope,
            "phase": "detention",
            "status": status,
            "created_at_utc": utc_now(),
            "cap": cap,
            "eligible_cap": cap,
            "arms": list(arms_table),
            "protocol": self.detention_protocol,
            "arm_schedule": {
                "file": str(schedule_path),
                "sha256": schedule_sha256,
                "digest": json_digest(schedule_rows),
                "seed": schedule_seed,
                "method": schedule.get("method"),
                "rows": rows_planned,
                "resumed_order": True,
            },
            "repeats_per_arm": 1,
            "selected_episodes": len(selected),
            "rows_planned": rows_planned,
            "rows_completed": rows_completed_total,
            "rows_completed_this_invocation": rows_completed,
            "rows_previously_completed": rows_previously_completed,
            "rows_failed": rows_failed_total,
            "rows_failed_this_invocation": rows_failed,
            "rows_interrupted": rows_interrupted_total,
            "rows_interrupted_this_invocation": rows_interrupted,
            "rows_refused": rows_refused_total,
            "rows_refused_this_invocation": rows_refused,
            "refused_row_keys_this_invocation": sorted(refused_keys),
            "route_refusals": dict(sorted(route_refusals.items())),
            "interruption_disposition": {
                "status": final_state.get("status"),
                "last_disposition": final_state.get("last_disposition"),
                "interrupted_episode_id": final_state.get("interrupted_episode_id"),
                "interrupted_arm": final_state.get("interrupted_arm"),
                "state_file": str(state_path),
                "state_sha256": sha256_file(state_path),
            },
            "phase_state": {
                "file": str(state_path),
                "sha256": sha256_file(state_path),
                "call_counters_persisted": True,
            },
            "route_calls": dict(sorted(route_calls.items())),
            "route_errors": dict(sorted(route_errors.items())),
            "route_latency": route_latency,
            "model_calls_total": self._model_calls_total,
            "max_model_calls": self.config.max_model_calls,
            "selection_file": str(selection_path),
            "selection_sha256": selection_sha256,
            "results_dir": str(detention_result_dir),
            "main_summary_file": str(summary_path),
            "main_summary_sha256": main_summary_sha256,
            "integrity_gate": dict(integrity),
            "denominator_note": (
                "Detention rows are reported here only; they are never merged into the "
                "main-grid denominator."
            ),
        }
        self._append_ledger({
            "kind": "detention_phase_finished",
            "phase": "detention",
            "status": status,
            "rows_planned": rows_planned,
            "rows_completed": rows_completed_total,
            "rows_completed_this_invocation": rows_completed,
            "rows_failed": rows_failed_total,
            "rows_interrupted": rows_interrupted_total,
            "rows_refused": rows_refused_total,
            "interruption_disposition": summary["interruption_disposition"],
        })
        ledger_records, ledger_problems = read_jsonl(self._ledger_path)
        ledger_verification = verify_ledger_records(ledger_records, signer.key)
        if ledger_problems:
            ledger_verification["ok"] = False
            ledger_verification["problems"] = list(
                ledger_verification.get("problems") or []
            ) + ledger_problems
        summary["ledger"] = {
            "records": len(ledger_records),
            "head_sha256": ledger_verification.get("head_sha256", ""),
            "verification": ledger_verification,
        }
        summary = signer.sign_mapping(summary)
        summary_path = detention / "detention_summary.json"
        atomic_write_json(summary_path, summary)
        self._last_detention_summary = summary
        self._emit("detention_phase_finished", {
            "status": status,
            "rows_completed": rows_completed_total,
        })
        return summary

    # -- main loop -----------------------------------------------------------
    def run(self) -> dict:
        """Execute the frozen table under an exclusive run-directory lock."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._acquire_run_lock()
        try:
            return self._run_locked()
        finally:
            self._release_run_lock()

    def _run_locked(self) -> dict:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._result_dir.mkdir(parents=True, exist_ok=True)

        # The runner validates the gate kind/scope itself (and the main-grid
        # preregistration hash) instead of trusting a caller-injected ``valid`` flag.
        self._run_scope = self._validate_run_gate()
        self._resolve_run_id()

        self._ensure_signer()
        self._validate_rows_against_registry()
        expected_digest = assignment_table_digest(self.rows)
        existing_assignment = self.workdir / DEFAULT_ASSIGNMENT
        if existing_assignment.is_file():
            try:
                existing = json.loads(existing_assignment.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            if (
                isinstance(existing, dict)
                and existing.get("assignment_digest")
                and existing["assignment_digest"] != expected_digest
            ):
                raise RunnerConfigError(
                    "this run directory contains a different frozen assignment table; "
                    "refusing to mix ledgers"
                )
        self._load_ledger()
        self._load_state()
        # Review finding 3: the main entry point uses the same integrity gate as
        # Detention.  It verifies the current ledger chain and its binding to every signed
        # completed result before any dispatcher or adapter is built.
        self._integrity_gate = self._verify_integrity_gate()
        self._build_dispatchers()
        self._episode_call_dir.mkdir(parents=True, exist_ok=True)
        preflight = self._run_isolation_preflight()
        atomic_write_json(self.workdir / DEFAULT_PREFLIGHT, preflight)
        self._append_ledger({
            "kind": "isolation_preflight",
            "ok": bool(preflight.get("ok", True)),
            "skipped": bool(preflight.get("skipped")),
            "stage": preflight.get("stage", "isolation_preflight"),
            "report_file": str(self.workdir / DEFAULT_PREFLIGHT),
            "report_sha256": json_digest(preflight),
            "require_measured_routes": self.require_measured_routes,
        })

        rows = self.rows
        if self.max_episodes is not None:
            if self.max_episodes < 1:
                raise RunnerConfigError("max_episodes must be >= 1")
            rows = rows[: int(self.max_episodes)]
        schedule = interleaved_schedule(rows)
        # Persist the *whole* frozen table, even for a --limit smoke run.  A restart with
        # the same table must be recognised as the same run namespace.
        assignment_payload = {
            "kind": "vera.v13_assignment_table.runner_copy.v1",
            **self._label_fields(),
            "run_id": self.run_id,
            "assignment_digest": assignment_table_digest(self.rows),
            "row_count": len(self.rows),
            "scheduled_rows": len(rows),
            "resume_key_factors": list(RESUME_KEY_FACTORS),
            "rows": [
                {k: v for k, v in row.items() if not k.startswith("_")}
                for row in self.rows
            ],
        }
        atomic_write_json(self.workdir / DEFAULT_ASSIGNMENT, assignment_payload)

        self.progress = ProgressTracker(
            workdir=self.workdir,
            run_id=self.run_id,
            mode="dry_run_mock" if self.dry_run else "live",
            rows=rows,
            registry=self.registry,
            config=self.config,
            started_monotonic=self._started_monotonic,
            clock=self.clock,
        )
        self.progress.attach_dispatchers(self._dispatchers)
        self.progress.model_calls_total = self._model_calls_total
        self.progress.model_calls_segment = self._segment_model_calls
        self.progress.max_model_calls = self.config.max_model_calls
        # Seed the live counters from an existing ledger so a restart reports cumulative
        # progress rather than appearing to start from zero.
        for row in rows:
            key = str(row["_runner_resume_key"])
            if key in self._completed_keys:
                self.progress.record_completed(row, resume_key=key)
            elif key in self._terminal_failed_keys:
                self.progress.record_infrastructure_failed(row)
        for route_id, info in self._state.get("pauses", {}).items():
            self.progress.set_pause(route_id, info)
        self.progress.mark_status("running")
        self.progress.write()

        schedule_digest = json_digest([row["_runner_resume_key"] for row in schedule])
        self._append_ledger({
            "kind": "run_started",
            "started_at_utc": self._started_utc,
            "mode": "dry_run_mock" if self.dry_run else "live",
            "run_scope": self._run_scope,
            "max_model_calls": self.config.max_model_calls,
            "model_calls_total_at_start": self._model_calls_total,
            "assignment_digest": assignment_payload["assignment_digest"],
            "row_count": len(self.rows),
            "scheduled_rows": len(rows),
            "schedule_digest": schedule_digest,
            "schedule_algorithm": (
                "deterministic round-robin over groups sorted by (condition, model_slot)"
            ),
            "resume_key_factors": list(RESUME_KEY_FACTORS),
            "config_digest": json_digest(self.config.raw),
            "route_registry_sha256": self.registry.get("_registry_sha256", ""),
            "signing": dict(self._signing_info),
            "preflight": {
                "ok": bool(preflight.get("ok", True)),
                "skipped": bool(preflight.get("skipped")),
                "report_sha256": json_digest(preflight),
            },
            "gate": dict(self.gate),
            "source_binding": self._source_binding_record(),
        })
        self._emit("run_started", {"run_id": self.run_id, "rows": len(rows)})

        pending: list[dict] = []
        adopted = 0
        for row in schedule:
            key = str(row["_runner_resume_key"])
            if key in self._completed_keys or key in self._terminal_failed_keys:
                continue
            if self._adopt_existing_result(row, key):
                adopted += 1
                continue
            pending.append(row)
        if adopted:
            self.progress.write()

        # Build cells once, before any model call, so a construction defect fails fast.
        prepared: deque[_EpisodeRun] = deque()
        for row in pending:
            key = str(row["_runner_resume_key"])
            attempt = int(self._attempts.get(key, 0)) + 1
            prepared.append(_EpisodeRun(
                row=row,
                attempt=attempt,
                cell=self._cell_for_row(row),
                started_monotonic=self.clock(),
            ))

        episode_executor = ThreadPoolExecutor(
            max_workers=self.config.episode_workers,
            thread_name_prefix="vera-episode",
        )
        in_flight: dict[Future, _EpisodeRun] = {}
        fatal_exception: BaseException | None = None
        try:
            while prepared or in_flight:
                self._process_control_file()
                for dispatcher in self._dispatchers.values():
                    dispatcher.maybe_auto_resume()
                if self._stop_event.is_set():
                    break
                # Submit what can start now.  Paused primary routes are skipped so other
                # routes keep making progress.
                index = 0
                while (
                    index < len(prepared)
                    and len(in_flight) < self.config.episode_workers
                    and not self._stop_event.is_set()
                ):
                    candidate = prepared[index]
                    route_id = self._main_route_for_row(candidate.row)
                    dispatcher = self._dispatchers.get(route_id)
                    if dispatcher is not None and dispatcher.is_paused():
                        index += 1
                        continue
                    prepared.remove(candidate)
                    future = episode_executor.submit(self._run_episode, candidate)
                    in_flight[future] = candidate
                    index = 0
                if not in_flight:
                    if prepared and not self._stop_event.is_set():
                        self.progress.write()
                        self._stop_event.wait(0.1)
                        continue
                    break
                done, _ = wait(
                    in_flight, timeout=0.25, return_when=FIRST_COMPLETED
                )
                for future in done:
                    episode = in_flight.pop(future)
                    try:
                        outcome = future.result()
                    except BaseException as exc:  # crash/injected fault: propagate after stop
                        fatal_exception = exc
                        self._fatal_reason = f"RUNNER_EXCEPTION: {type(exc).__name__}"
                        self._stop_event.set()
                        break
                    if outcome.get("requeue"):
                        # A route-limit interruption waits for the route to resume; an
                        # allowed infrastructure retry is bounded by ``infra_retries`` in
                        # the episode itself.  Either way the episode is not completed.
                        key = str(episode.row["_runner_resume_key"])
                        attempts = int(self._attempts.get(key, 0)) + 1
                        self._attempts[key] = attempts
                        prepared.appendleft(
                            _EpisodeRun(
                                row=episode.row,
                                attempt=attempts + 1,
                                cell=episode.cell,
                                started_monotonic=self.clock(),
                            )
                        )
                    self.progress.write()
                if fatal_exception is not None:
                    break
                if self._stop_event.is_set():
                    break
                self.progress.write()
        finally:
            self._stop_event.set()
            for dispatcher in self._dispatchers.values():
                with dispatcher._condition:  # wake waiters before shutdown
                    dispatcher._condition.notify_all()
            for dispatcher in self._dispatchers.values():
                dispatcher.shutdown()
            episode_executor.shutdown(wait=False, cancel_futures=True)

        if fatal_exception is not None:
            self.progress.mark_status("stopped_exception")
            self.progress.write()
            raise fatal_exception

        current = self.progress.snapshot()
        if self._auth_expired:
            status = "stopped_auth_expired"
        elif self._fatal_reason:
            status = "stopped"
        elif prepared or current["totals"]["remaining"] > 0:
            status = "stopped_incomplete"
        elif current["totals"]["infrastructure_failed"]:
            status = "completed_with_infrastructure_failures"
        elif self.max_episodes is not None and len(self.rows) > self.max_episodes:
            status = "completed_limited"
        else:
            status = "completed"
        self.progress.mark_status(status)
        self.progress.model_calls_total = self._model_calls_total
        self.progress.model_calls_segment = self._segment_model_calls
        self.progress.max_model_calls = self.config.max_model_calls
        progress = self.progress.write()

        route_counters = {
            route_id: dispatcher.counters()
            for route_id, dispatcher in sorted(self._dispatchers.items())
        }
        summary = {
            "kind": SUMMARY_KIND,
            **self._label_fields(),
            "run_id": self.run_id,
            "status": status,
            "mode": "dry_run_mock" if self.dry_run else "live",
            "run_scope": self._run_scope,
            "started_at_utc": self._started_utc,
            "finished_at_utc": utc_now(),
            "assignment_digest": assignment_payload["assignment_digest"],
            "schedule_digest": schedule_digest,
            "rows_assigned": len(rows),
            "rows_completed": progress["totals"]["done"],
            "rows_remaining": progress["totals"]["remaining"],
            "rows_interrupted": progress["totals"]["interrupted"],
            "rows_infrastructure_failed": progress["totals"]["infrastructure_failed"],
            "pauses": progress["pauses"],
            "auth_expired": self._auth_expired,
            "stopped_reason": self._fatal_reason,
            "gate_kind": str(self.gate.get("kind") or ""),
            "model_calls_total": self._model_calls_total,
            "model_calls_this_segment": self._segment_model_calls,
            "max_model_calls": self.config.max_model_calls,
            # v14b: provider safeguard refusals are reported on their own, never inside route/infrastructure errors
            "route_refusals": dict(sorted(dict(self._state.get("route_refusals") or {}).items())),
            "route_counters": route_counters,
            "route_latency": {
                route_id: dict(counters.get("latency") or {})
                for route_id, counters in route_counters.items()
            },
            "segment": {
                "run_id": self.run_id,
                "model_calls": self._segment_model_calls,
                "model_calls_cumulative": self._model_calls_total,
                "route_calls_cumulative": {
                    route_id: int(counters.get("calls") or 0)
                    for route_id, counters in route_counters.items()
                },
                "route_calls_this_segment": {
                    route_id: int(counters.get("calls_this_segment") or 0)
                    for route_id, counters in route_counters.items()
                },
            },
            "progress_file": str(self.workdir / DEFAULT_PROGRESS),
            "ledger_file": str(self._ledger_path),
            "episode_call_log_dir": str(self._episode_call_dir),
            "signing": dict(self._signing_info),
            "preflight": {
                "ok": bool(preflight.get("ok", True)),
                "skipped": bool(preflight.get("skipped")),
                "report_file": str(self.workdir / DEFAULT_PREFLIGHT),
                "report_sha256": json_digest(preflight),
                "bounded_preflight_acceptance": preflight.get(
                    "bounded_preflight_acceptance", {}
                ),
            },
            "source_binding": self._source_binding_record(),
            "main_scope_binding": dict(self._main_binding),
            "rows_by_condition": dict(
                sorted(Counter(str(row["condition"]) for row in self.rows).items())
            ),
            "rows_by_pattern": dict(
                sorted(
                    Counter(
                        str(row.get("organization_pattern") or PATTERN_BASE)
                        for row in self.rows
                    ).items()
                )
            ),
        }
        main_inventory = self._main_result_inventory(self._ensure_signer())
        summary["main_inventory_digest"] = json_digest([
            main_inventory[key] for key in sorted(main_inventory)
        ])
        summary["main_inventory_results"] = len(main_inventory)
        self._append_ledger({
            "kind": "run_finished",
            "status": status,
            "stopped_reason": self._fatal_reason,
            "rows_completed": summary["rows_completed"],
            "rows_remaining": summary["rows_remaining"],
            "main_inventory_digest": summary["main_inventory_digest"],
        })
        ledger_records, ledger_problems = read_jsonl(self._ledger_path)
        ledger_verification = verify_ledger_records(
            ledger_records, self._ensure_signer().key
        )
        if ledger_problems:
            ledger_verification = dict(ledger_verification)
            ledger_verification["ok"] = False
            ledger_verification["problems"] = list(
                ledger_verification.get("problems") or []
            ) + ledger_problems
        summary["ledger"] = {
            "records": len(ledger_records),
            "head_sha256": ledger_verification.get("head_sha256", ""),
            "verification": ledger_verification,
        }
        summary = self._ensure_signer().sign_mapping(summary)
        atomic_write_json(self.workdir / DEFAULT_SUMMARY, summary)
        self._last_run_summary = summary
        self._emit("run_finished", {"status": status, **summary})
        return summary


# Registry slot helper (read-only, mirrors identity_assignment.registry_slots_for)


def _registry_slots_for(orchestrator: str, registry: Mapping) -> dict:
    from .identity_assignment import registry_slots_for

    return registry_slots_for(orchestrator, registry)["slots"]


# Capacity planner


@dataclass(frozen=True)
class CapacityProfile:
    route_id: str
    measured_calls_per_min: float
    max_requests_per_minute: float | None = None
    limit_requests: int | None = None
    limit_window_seconds: float | None = None
    min_interval_seconds: float = 0.0
    concurrency: int = 1
    nvidia_route: bool = False

    @classmethod
    def from_mapping(cls, route_id: str, data: Mapping, *, nvidia_route: bool = False):
        def number(name: str, default=None):
            value = data.get(name, default)
            if value in (None, ""):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                raise RunnerConfigError(
                    f"capacity route {route_id!r}: {name} must be numeric"
                ) from None

        measured = number("measured_calls_per_min")
        if measured is None:
            measured = number("calls_per_min")
        if measured is None or measured <= 0:
            raise RunnerConfigError(
                f"capacity route {route_id!r}: measured_calls_per_min must be > 0"
            )
        limit_requests = data.get("limit_requests")
        if limit_requests not in (None, ""):
            try:
                limit_requests = int(limit_requests)
            except (TypeError, ValueError):
                raise RunnerConfigError(
                    f"capacity route {route_id!r}: limit_requests must be an integer"
                ) from None
        else:
            limit_requests = None
        return cls(
            route_id=route_id,
            measured_calls_per_min=measured,
            max_requests_per_minute=number("max_requests_per_minute"),
            limit_requests=limit_requests,
            limit_window_seconds=number("limit_window_seconds"),
            min_interval_seconds=number("min_interval_seconds", 0.0) or 0.0,
            concurrency=int(data.get("concurrency", 1) or 1),
            nvidia_route=nvidia_route or bool(data.get("nvidia_route")),
        )

    def effective_calls_per_min(self) -> tuple[float, list[dict]]:
        caps: list[dict] = [
            {"cap": "measured", "value": self.measured_calls_per_min}
        ]
        if self.max_requests_per_minute is not None:
            caps.append({
                "cap": "declared_max_requests_per_minute",
                "value": self.max_requests_per_minute,
            })
        if self.limit_requests is not None and self.limit_window_seconds:
            caps.append({
                "cap": "limit_window",
                "value": self.limit_requests * 60.0 / self.limit_window_seconds,
            })
        if self.min_interval_seconds > 0:
            caps.append({
                "cap": "min_interval",
                "value": 60.0 / self.min_interval_seconds,
            })
        if self.nvidia_route:
            caps.append({
                "cap": "nvidia_free_route",
                "value": NVIDIA_MAX_REQUESTS_PER_MINUTE,
            })
        effective = min(float(item["value"]) for item in caps)
        return effective, caps


def _normalise_demands(
    demands: Mapping[str, Any],
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Accept either route->{phase:calls} or phase->{route:calls}."""
    phases: list[str] = []
    if demands and all(
        isinstance(value, Mapping) and not any(
            isinstance(inner, Mapping) for inner in value.values()
        )
        for value in demands.values()
    ):
        # Ambiguous flat shape: treat as route -> {phase: calls} only if phases are known.
        if any("main" in value or "detention" in value for value in demands.values()):
            normalised: dict[str, dict[str, int]] = {}
            for route_id, per_phase in demands.items():
                normalised[str(route_id)] = {
                    str(phase): int(calls) for phase, calls in per_phase.items()
                }
            phases = sorted({
                phase for per_phase in normalised.values() for phase in per_phase
            })
            return normalised, phases
    normalised = {}
    for phase, per_route in demands.items():
        if not isinstance(per_route, Mapping):
            raise RunnerConfigError("capacity demands must be route->{phase:calls} or phase->{route:calls}")
        phases.append(str(phase))
        for route_id, calls in per_route.items():
            normalised.setdefault(str(route_id), {})[str(phase)] = int(calls)
    return normalised, phases


def plan_capacity(
    route_profiles: Mapping[str, Mapping | CapacityProfile],
    call_demands: Mapping[str, Any],
    *,
    phases: Sequence[str] = ("main", "detention"),
    phase_order: Sequence[str] | None = None,
) -> dict:
    """Compute wall-clock per route and in total from measured rates and limit windows.

    The per-phase wall clock is the slowest route (routes run concurrently).  The two study
    phases are sequential (main grid -> Detention), so the total is the sum of the phase
    maxima.  The table also reports the fully-serial sum for comparison.
    """
    profiles: dict[str, CapacityProfile] = {
        str(route_id): (
            profile
            if isinstance(profile, CapacityProfile)
            else CapacityProfile.from_mapping(str(route_id), profile)
        )
        for route_id, profile in route_profiles.items()
    }
    demands, discovered = _normalise_demands(call_demands)
    for route_id in demands:
        if route_id not in profiles:
            raise RunnerConfigError(
                f"capacity demand names route {route_id!r} with no measured profile"
            )
    order = list(phase_order or phases)
    for phase in discovered:
        if phase not in order:
            order.append(phase)

    table: list[dict] = []
    phase_totals: dict[str, dict] = {}
    serial_by_phase: dict[str, float] = {}
    for phase in order:
        phase_seconds = 0.0
        serial = 0.0
        slowest = ""
        for route_id in sorted(profiles):
            calls = int(demands.get(route_id, {}).get(phase, 0))
            profile = profiles[route_id]
            effective, caps = profile.effective_calls_per_min()
            seconds = calls / effective * 60.0 if calls else 0.0
            row = {
                "phase": phase,
                "route_id": route_id,
                "calls": calls,
                "measured_calls_per_min": round(profile.measured_calls_per_min, 6),
                "effective_calls_per_min": round(effective, 6),
                "limit_requests": profile.limit_requests,
                "limit_window_seconds": profile.limit_window_seconds,
                "min_interval_seconds": profile.min_interval_seconds,
                "concurrency": profile.concurrency,
                "caps": [
                    {"cap": item["cap"], "value": round(float(item["value"]), 6)}
                    for item in caps
                ],
                "wall_clock_seconds": round(seconds, 3),
                "wall_clock_minutes": round(seconds / 60.0, 3),
                "wall_clock_hours": round(seconds / 3600.0, 3),
                "critical": False,
            }
            table.append(row)
            serial += seconds
            if seconds > phase_seconds:
                phase_seconds = seconds
                slowest = route_id
        for row in table:
            if row["phase"] == phase and row["route_id"] == slowest:
                row["critical"] = True
        phase_totals[phase] = {
            "wall_clock_seconds": round(phase_seconds, 3),
            "wall_clock_minutes": round(phase_seconds / 60.0, 3),
            "wall_clock_hours": round(phase_seconds / 3600.0, 3),
            "critical_route": slowest,
        }
        serial_by_phase[phase] = serial
    total_seconds = sum(item["wall_clock_seconds"] for item in phase_totals.values())
    serial_seconds = sum(serial_by_phase.values())
    return {
        "kind": "vera.v13_capacity_plan.v1",
        "status": "PLANNING_ESTIMATE_FROM_DECLARED_RATES_NOT_A_RESULT",
        "phases": order,
        "phase_totals": phase_totals,
        "total_parallel_wall_clock_seconds": round(total_seconds, 3),
        "total_parallel_wall_clock_minutes": round(total_seconds / 60.0, 3),
        "total_parallel_wall_clock_hours": round(total_seconds / 3600.0, 3),
        "total_serial_wall_clock_seconds": round(serial_seconds, 3),
        "total_serial_wall_clock_hours": round(serial_seconds / 3600.0, 3),
        "table": table,
        "table_text": format_capacity_table(table, phase_totals, total_seconds),
        "notes": [
            "per-phase wall clock is the slowest route; routes are assumed concurrent",
            "main grid and Detention are sequential study phases",
            "effective rate is the minimum of measured, declared, limit-window and interval caps",
            "NVIDIA free routes are capped at 30 requests/minute in code",
        ],
    }


def format_capacity_table(
    table: Sequence[Mapping],
    phase_totals: Mapping[str, Mapping],
    total_seconds: float,
) -> str:
    header = (
        "phase      route                               calls  measured  effective  "
        "limit/window      wall-clock"
    )
    lines = [header, "-" * len(header)]
    for row in table:
        limit = "—"
        if row.get("limit_requests") and row.get("limit_window_seconds"):
            limit = f"{row['limit_requests']}/{row['limit_window_seconds']:g}s"
        marker = "*" if row.get("critical") else " "
        lines.append(
            f"{row['phase']:<10} {row['route_id']:<33} "
            f"{row['calls']:>6}  {row['measured_calls_per_min']:>8.2f}  "
            f"{row['effective_calls_per_min']:>9.2f}  {limit:<17} "
            f"{row['wall_clock_minutes']:>8.2f} min{marker}"
        )
    lines.append("-" * len(header))
    for phase in phase_totals:
        item = phase_totals[phase]
        lines.append(
            f"{phase:<10} critical={item['critical_route'] or '—':<33} "
            f"wall-clock {item['wall_clock_minutes']:.2f} min"
        )
    lines.append(
        f"TOTAL (main then Detention) parallel wall-clock: "
        f"{total_seconds / 3600.0:.2f} h ({total_seconds / 60.0:.1f} min)"
    )
    lines.append("* = critical route for that phase")
    return "\n".join(lines)


def call_demands_from_run_plan(
    run_plan: Mapping,
    rows: Sequence[Mapping],
    *,
    registry: Mapping,
    include_detention: bool = True,
) -> dict[str, dict[str, int]]:
    """Default per-route demand split from the frozen run plan.

    Main-grid calls use the run plan's upper action-call bound.  Calls are split over the
    four slots with the clone route sharing the orchestrator route: one half of the calls on
    the primary route (orchestrator + clone), one sixth on each non-clone slot.  Detention
    calls are the run plan's review+retry bound, split over the same four-slot shape and the
    two model deployments in proportion to their main-grid episode counts.  An explicit
    ``--demands`` file overrides this planning assumption.
    """
    slots_by_model = {
        model: _registry_slots_for(model, registry) for model in ("O1", "O2")
    }
    episodes_by_model: Counter = Counter()
    for row in rows:
        episodes_by_model[str(row["model_slot"])] += 1
    episode_total = sum(episodes_by_model.values()) or 1
    calls = run_plan.get("calls") or {}
    main_calls = int(calls.get("main_action_calls_upper") or 0)
    detention_calls = 0
    if include_detention:
        detention_calls = int(calls.get("detention_review_calls") or 0) + int(
            calls.get("detention_retry_calls_upper") or 0
        )
    demands: dict[str, dict[str, int]] = defaultdict(lambda: {"main": 0, "detention": 0})

    def add_split(model: str, total_calls: int, phase: str) -> None:
        if total_calls <= 0:
            return
        slots = slots_by_model.get(model)
        if not slots:
            return
        primary = str(slots[SLOT_SELF]["route_id"])
        non_clone = [
            str(route["route_id"])
            for slot, route in slots.items()
            if slot != SLOT_SELF
        ]
        demands[primary][phase] += int(round(total_calls * 0.5))
        if non_clone:
            share = int(round(total_calls * 0.5 / len(non_clone)))
            for route_id in non_clone:
                demands[route_id][phase] += share

    for model in ("O1", "O2"):
        share = episodes_by_model.get(model, 0) / episode_total
        add_split(model, int(round(main_calls * share)), "main")
        if include_detention:
            add_split(model, int(round(detention_calls * share)), "detention")
    return {route_id: dict(per_phase) for route_id, per_phase in demands.items()}


def capacity_plan_from_config(
    config: Mapping | RunnerConfig,
    *,
    registry: Mapping | None = None,
    run_plan: Mapping | None = None,
    rows: Sequence[Mapping] | None = None,
    demands: Mapping | None = None,
    include_detention: bool = True,
) -> dict:
    """CLI helper: derive profiles from the runner config and demands from the run plan."""
    registry = dict(registry or load_registry())
    if isinstance(config, RunnerConfig):
        runner_config = config
    else:
        runner_config = load_runner_config(config, registry=registry, dry_run=True)
    profiles: dict[str, CapacityProfile] = {}
    registry_routes = {route["route_id"]: route for route in registry.get("routes", [])}
    for route_id, settings in runner_config.routes.items():
        route = registry_routes.get(route_id, {})
        data = dict(settings.raw)
        data.setdefault("measured_calls_per_min", data.get("measured_calls_per_min"))
        if data.get("measured_calls_per_min") in (None, ""):
            # The planner needs a measured number; without it a table row is not an estimate.
            raise RunnerConfigError(
                f"capacity route {route_id!r}: config needs measured_calls_per_min "
                "from the smoke"
            )
        profiles[route_id] = CapacityProfile.from_mapping(
            route_id, data, nvidia_route=route.get("route") == "nvidia_nim"
        )
    if demands is None:
        if run_plan is None or rows is None:
            raise RunnerConfigError(
                "capacity planner needs either explicit demands or (run_plan + rows)"
            )
        demands = call_demands_from_run_plan(
            run_plan, rows, registry=registry, include_detention=include_detention
        )
    return plan_capacity(profiles, demands)


# CLI


def _load_manifest_and_plan(
    manifest_path: str | Path | None,
    run_plan_path: str | Path | None,
) -> tuple[dict | None, dict | None]:
    manifest = None
    run_plan = None
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if run_plan_path:
        run_plan = json.loads(Path(run_plan_path).read_text(encoding="utf-8"))
    return manifest, run_plan


def _default_manifest_path() -> Path:
    """The condition manifest declared by the preregistration (set 12)."""
    from .source_binding import CONDITION_MANIFEST_SET12

    return CONDITION_MANIFEST_SET12


def _default_run_plan_path() -> Path:
    return Path(__file__).resolve().parents[2] / "manifests" / "delegation" / "V13_RUN_PLAN.json"


def _signing_key_from_args(args: argparse.Namespace) -> bytes | None:
    if getattr(args, "signing_key_file", None):
        return load_signing_key_file(args.signing_key_file)
    env_key = os.environ.get("VERA_RUNNER_SIGNING_KEY", "")
    if env_key:
        text = env_key.strip()
        try:
            decoded = bytes.fromhex(text)
        except (ValueError, TypeError):
            decoded = b""
        return decoded or text.encode("utf-8")
    return None


def _patterns_from_args(args: argparse.Namespace):
    if getattr(args, "full_plan", False):
        return ENROLLED_PATTERNS
    chosen = tuple(getattr(args, "pattern", None) or ())
    return chosen or (PATTERN_BASE,)


def _run_scope_from_args(args: argparse.Namespace) -> str:
    if getattr(args, "dry_run", False):
        return RUN_SCOPE_DRY_RUN
    if getattr(args, "shakedown", False):
        return RUN_SCOPE_SHAKEDOWN
    if getattr(args, "pilot", False):
        return RUN_SCOPE_PILOT
    return RUN_SCOPE_MAIN_GRID


def _load_run_condition_manifest(
    path: str | Path | None, *, binding: Mapping, scope: str
) -> dict:
    """Load the condition manifest for this run, bound to the validated source.

    The default is the set-12 file the preregistration names.  In ``main_grid`` scope an
    explicit ``--manifest`` must hash to the same manifest; an unbound file is refused.
    """
    target = Path(path) if path is not None else _default_manifest_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerConfigError(f"condition manifest is unreadable: {target}: {exc}") from None
    digest = str(data.get("manifest_digest") or "")
    expected = str(binding.get("condition_manifest_digest") or "")
    source_sha = str((data.get("source") or {}).get("sha256") or "")
    if scope == RUN_SCOPE_MAIN_GRID:
        problems = []
        if digest != expected:
            problems.append(
                f"condition manifest digest {digest!r} is not the bound set-12 digest "
                f"{expected!r}"
            )
        if source_sha != binding["source"]["sha256"]:
            problems.append("condition manifest records a different source sha256")
        if problems:
            raise SourceBindingError(
                "main-grid run refused: " + "; ".join(problems)
            )
    return data


def cmd_runner(args: argparse.Namespace) -> int:
    dry_run = bool(args.dry_run)
    try:
        profile = getattr(args, "registry_profile", None) or None
        registry = load_registry(None if profile else (args.registry or None), profile=profile)
        scope = _run_scope_from_args(args)
        binding = resolve_source_binding()
        gate: dict = {}
        if not dry_run:
            # Gate validation happens inside the JSON-refusal path, so a missing
            # gate returns REFUSED_STAGE_GATE instead of an uncaught traceback.
            if getattr(args, "pilot", False):
                if any((args.prereg_root, args.prereg_hash, args.reviewer_acceptance, args.owner_approval)):
                    raise StageGateError(
                        "--pilot may not be combined with main-grid gate arguments"
                    )
                if not getattr(args, "pilot_gate", None):
                    raise StageGateError("--pilot requires --pilot-gate <gate JSON>")
                if not args.assignment:
                    raise StageGateError(
                        "--pilot requires --assignment <pilot row table> (protocol v14 rows)"
                    )
                gate = validate_pilot_gate(args.pilot_gate)
            elif getattr(args, "shakedown", False):
                if any(
                    (
                        args.prereg_root,
                        args.prereg_hash,
                        args.reviewer_acceptance,
                        args.owner_approval,
                    )
                ):
                    raise StageGateError(
                        "--shakedown may not be combined with main-grid gate arguments"
                    )
                if not getattr(args, "shakedown_gate", None):
                    raise StageGateError(
                        "--shakedown requires --shakedown-gate <gate JSON>"
                    )
                gate = validate_shakedown_gate(args.shakedown_gate)
            else:
                gate = validate_stage_gate(
                    prereg_root=args.prereg_root,
                    prereg_hash=args.prereg_hash,
                    reviewer_acceptance=args.reviewer_acceptance,
                    owner_approval=args.owner_approval,
                )
            print(json.dumps({"stage_gate": gate}, sort_keys=True))
        config = load_runner_config(args.config, registry=registry, dry_run=dry_run)

        manifest = None
        run_plan = None
        patterns = _patterns_from_args(args)
        if args.assignment:
            rows, meta = load_assignment_rows(args.assignment, scope=scope)
        else:
            manifest = _load_run_condition_manifest(
                args.manifest, binding=binding, scope=scope
            )
            run_plan_path = Path(args.run_plan) if args.run_plan else _default_run_plan_path()
            run_plan = (
                json.loads(run_plan_path.read_text(encoding="utf-8"))
                if run_plan_path.is_file()
                else None
            )
            rows, meta = load_assignment_rows(
                None,
                manifest=manifest,
                repeats=(run_plan or {}).get("recommended_repeats", 2),
                patterns=patterns,
                scope=scope,
            )
        runner = GridRunner(
            rows,
            config=config,
            workdir=args.out,
            registry=registry,
            dry_run=dry_run,
            gate=gate,
            control_file=args.control_file,
            resume_routes=args.resume_route or (),
            max_episodes=args.limit,
            signing_key=_signing_key_from_args(args),
            require_measured_routes=not bool(args.allow_unmeasured_routes),
            preflight_acceptance=getattr(args, "preflight_acceptance", None),
            source_binding=binding,
            assignment_meta=meta,
            episode_call_log_dir=args.episode_call_log_dir or None,
            record_transcripts=bool(getattr(args, "record_transcripts", False)),
            detention_protocol=(
                "v14" if getattr(args, "detention_protocol", "") == "v14" else "v13"
            ),
            solo=bool(getattr(args, "solo", False)),
        )
        summary = runner.run()
        detention_summary = None
        if getattr(args, "detention", False):
            detention_summary = runner.run_detention_phase()
    except LedgerIntegrityError as exc:
        print(json.dumps(
            {"status": "REFUSED_LEDGER_INTEGRITY", "reason": str(exc)}, sort_keys=True
        ))
        return 8
    except SourceBindingError as exc:
        print(json.dumps(
            {"status": "REFUSED_SOURCE_BINDING", "reason": str(exc)}, sort_keys=True
        ))
        return 9
    except StageGateError as exc:
        print(json.dumps({"status": "REFUSED_STAGE_GATE", "reason": str(exc)}, sort_keys=True))
        return 4
    except RunnerConfigError as exc:
        print(json.dumps({"status": "REFUSED_CONFIG", "reason": str(exc)}, sort_keys=True))
        return 5
    except RunDirectoryLocked as exc:
        print(json.dumps(
            {"status": "RUN_DIRECTORY_LOCKED", "reason": str(exc)}, sort_keys=True
        ))
        return 7
    except Exception as exc:  # IsolationRefused is an AdapterError; import lazily
        from .routes.preflight import IsolationRefused

        if isinstance(exc, IsolationRefused):
            print(json.dumps({"status": "REFUSED_PREFLIGHT", "reason": str(exc)}, sort_keys=True))
            return 6
        raise
    output = {
        "status": summary["status"],
        "run_id": summary["run_id"],
        "run_scope": summary.get("run_scope"),
        "rows_assigned": summary["rows_assigned"],
        "rows_completed": summary["rows_completed"],
        "rows_remaining": summary["rows_remaining"],
        "rows_by_condition": summary.get("rows_by_condition", {}),
        "rows_by_pattern": summary.get("rows_by_pattern", {}),
        "signing": summary.get("signing", {}),
        "model_calls_total": summary.get("model_calls_total"),
        "model_calls_this_segment": summary.get("model_calls_this_segment"),
        "max_model_calls": summary.get("max_model_calls"),
        "progress_file": summary["progress_file"],
        "ledger_file": summary["ledger_file"],
        "assignment_source": meta,
    }
    if detention_summary is not None:
        output["detention"] = {
            "status": detention_summary.get("status"),
            "selected_episodes": detention_summary.get("selected_episodes"),
            "rows_completed": detention_summary.get("rows_completed"),
            "summary_file": str(
                runner.workdir / DEFAULT_DETENTION_DIR / "detention_summary.json"
            ),
        }
    print(json.dumps(output, indent=2, sort_keys=True))
    ok = summary["status"] in ("completed", "completed_limited")
    if detention_summary is not None:
        ok = ok and detention_summary.get("status") == "completed"
    return 0 if ok else 3


def cmd_capacity(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry or None)
    config = load_runner_config(args.config, registry=registry, dry_run=True)
    if args.demands:
        demands = json.loads(Path(args.demands).read_text(encoding="utf-8"))
        plan = capacity_plan_from_config(
            config, registry=registry, demands=demands,
            include_detention=not args.no_detention,
        )
    else:
        run_plan_path = Path(args.run_plan) if args.run_plan else _default_run_plan_path()
        run_plan = json.loads(run_plan_path.read_text(encoding="utf-8"))
        if args.assignment:
            rows, _ = load_assignment_rows(args.assignment)
        else:
            manifest_path = Path(args.manifest) if args.manifest else _default_manifest_path()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows, _ = load_assignment_rows(
                None, manifest=manifest,
                repeats=run_plan.get("recommended_repeats", 2),
                patterns=_patterns_from_args(args),
            )
        plan = capacity_plan_from_config(
            config, registry=registry, run_plan=run_plan, rows=rows,
            include_detention=not args.no_detention,
        )
    print(plan["table_text"])
    if args.json:
        atomic_write_json(args.json, plan)
    return 0


def add_runner_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="measured runner route config JSON")
    parser.add_argument("--registry", default=None, help="D4 registry path (read-only)")
    parser.add_argument("--assignment", default=None, help="frozen assignment table JSON")
    parser.add_argument("--manifest", default=None, help="v1.3 condition manifest")
    parser.add_argument("--run-plan", default=None, help="V13_RUN_PLAN.json")
    parser.add_argument("--out", default="artifacts/run",
                        help="run directory (ledger, progress, results)")
    parser.add_argument("--dry-run", action="store_true",
                        help="use MockAdapter; no stage gate, no network")
    parser.add_argument("--pilot", action="store_true",
                        help="pilot scope (requires --pilot-gate, --assignment, "
                        "--record-transcripts, --preflight-acceptance, the claude_only profile)")
    parser.add_argument("--pilot-gate", default=None,
                        help="the owner's pilot gate JSON (names the signed case-text digest)")
    parser.add_argument("--solo", action="store_true",
                        help="solo mode: no peer menu, delegation budget 0")
    parser.add_argument("--registry-profile", default=None, choices=("claude_only",),
                        help="route registry profile (pilot: claude_only)")
    parser.add_argument("--detention-protocol", default="v13", choices=("v13", "v14"),
                        help="Detention design: frozen v13, or the v14 pilot (arms A,B,C,E; "
                        "sibling transfer)")
    parser.add_argument("--record-transcripts", action="store_true",
                        help="also record messages/reply_text for mock (dry-run) adapters; "
                        "live runs always record them")
    parser.add_argument("--full-plan", action="store_true",
                        help="execute the full cross (P0-P3, 4320 episodes)")
    parser.add_argument("--pattern", action="append", default=[],
                        choices=list(PATTERN_DISPLAY),
                        help="organization pattern to assign (repeatable; default P0_BASE)")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N scheduled assignments (smoke)")
    parser.add_argument("--signing-key-file", default=None,
                        help="0600 file with the runner HMAC key (hex or raw); live requires it")
    parser.add_argument("--episode-call-log-dir", default=None,
                        help="per-episode adapter call log directory (default <out>/episode_calls)")
    parser.add_argument("--allow-unmeasured-routes", action="store_true",
                        help="structural preflight only (still fail-closed); for a measured "
                             "smoke; refused for main_grid scope")
    parser.add_argument("--preflight-acceptance", default=None,
                        help="bounded-preflight report JSON (required for main_grid scope)")
    parser.add_argument("--control-file", default=None,
                        help='live control JSON: {"resume_routes": [...], "stop": true}')
    parser.add_argument("--resume-route", action="append", default=[],
                        help="resume a persisted paused route before/while starting")
    parser.add_argument("--prereg-root", default=None,
                        help="path to preregistration ROOT_MANIFEST.json (live only)")
    parser.add_argument("--prereg-hash", default=None,
                        help="expected preregistration root_sha256 (live only)")
    parser.add_argument("--reviewer-acceptance", default=None,
                        help="independent reviewer's acceptance JSON path (live only)")
    parser.add_argument("--owner-approval", default=None,
                        help="owner approval JSON path (live only)")
    parser.add_argument("--shakedown", action="store_true",
                        help="non-main-grid shakedown scope (requires --shakedown-gate)")
    parser.add_argument("--shakedown-gate", default=None,
                        help="vera.v13_shakedown_gate.v1 JSON (owner job reference + expiry)")
    parser.add_argument("--detention", action="store_true",
                        help="run the frozen Detention retry phase after a completed run")
    parser.set_defaults(func=cmd_runner)


def add_capacity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="measured route config JSON")
    parser.add_argument("--registry", default=None, help="D4 registry path (read-only)")
    parser.add_argument("--assignment", default=None, help="frozen assignment table JSON")
    parser.add_argument("--manifest", default=None, help="v1.3 condition manifest")
    parser.add_argument("--run-plan", default=None, help="V13_RUN_PLAN.json")
    parser.add_argument("--demands", default=None,
                        help="explicit calls per route/phase JSON (overrides the run plan)")
    parser.add_argument("--no-detention", action="store_true",
                        help="plan the main grid only")
    parser.add_argument("--full-plan", action="store_true",
                        help="plan the P0-P3 full cross")
    parser.add_argument("--pattern", action="append", default=[],
                        choices=list(PATTERN_DISPLAY),
                        help="organization pattern to plan (repeatable; default P0_BASE)")
    parser.add_argument("--json", default=None, help="also write the full plan JSON here")
    parser.set_defaults(func=cmd_capacity)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vera.delegation.runner_v13")
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run", help="execute the frozen v1.3 main-grid assignment table")
    add_runner_arguments(runner)
    capacity = sub.add_parser("capacity", help="wall-clock plan from measured route rates")
    add_capacity_arguments(capacity)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


__all__ = [
    "CONFIG_KIND",
    "DETENTION_SUMMARY_KIND",
    "CapacityProfile",
    "GridRunner",
    "LedgerIntegrityError",
    "NVIDIA_MAX_REQUESTS_PER_MINUTE",
    "ProgressTracker",
    "RESUME_KEY_FACTORS",
    "PILOT_GATE_KIND",
    "PILOT_LABEL",
    "RUN_SCOPE_DRY_RUN",
    "RUN_SCOPE_MAIN_GRID",
    "RUN_SCOPE_PILOT",
    "RUN_SCOPE_SHAKEDOWN",
    "assert_pilot_summary",
    "validate_pilot_gate",
    "RUN_SCOPES",
    "RouteDispatcher",
    "RouteSettings",
    "RunDirectoryLocked",
    "RunnerConfig",
    "RunnerConfigError",
    "SourceBindingError",
    "SHAKEDOWN_GATE_KIND",
    "StageGateError",
    "assert_main_grid_summary",
    "assignment_resume_key",
    "assignment_table_digest",
    "build_parser",
    "call_demands_from_run_plan",
    "capacity_plan_from_config",
    "cmd_capacity",
    "cmd_runner",
    "derive_dry_run_key",
    "format_capacity_table",
    "interleaved_schedule",
    "load_assignment_rows",
    "load_runner_config",
    "load_signing_key_file",
    "normalise_assignment_rows",
    "main",
    "plan_capacity",
    "RunnerSigner",
    "percentile_ms",
    "summarise_latencies",
    "validate_gate_document",
    "validate_pre_main_stage_completion",
    "validate_shakedown_gate",
    "validate_stage_gate",
    "verify_ledger_file",
    "verify_ledger_records",
    "verify_signed_mapping",
]


if __name__ == "__main__":
    raise SystemExit(main())
