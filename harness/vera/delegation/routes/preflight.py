"""Route preflight: a fail-closed stage that runs before any live episode.

Structural checks, with no call: each route's transport is declared in the closed registry
vocabulary, the adapter class and its capability match that declaration, and the adapter
is one this repository provides.  A transport that declares it needs an isolated
command-line client is refused, because no such adapter is published.

``--probe`` adds one tiny call per route and records whether it succeeded and which model
the provider says served it.  A pilot or main-grid run must present a probed report
(``validate_preflight_acceptance``); a structural report is not enough.

    report = run_preflight(routes, adapters=adapters)
    assert_start_allowed(report)          # raises IsolationRefused
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..adapters import (
    SERVED_MODEL_MATCH,
    AdapterError,
    served_model_check,
)
from .registry import adapter_for_route, load_registry, transport_spec

PREFLIGHT_KIND = "vera.route_preflight.v1"
PREFLIGHT_STAGE_ID = "isolation_preflight"
PREFLIGHT_STAGE = {
    "stage_id": PREFLIGHT_STAGE_ID,
    "kind": PREFLIGHT_KIND,
    "order": 0,
    "fail_closed": True,
    "refuses_run_on_failure": True,
    "calls_model_only_with_probe": True,
    "required_before": "main_enrollment",
}
PREFLIGHT_PROMPT_SYSTEM = "You are a VERA preflight probe. Reply with OK."
PREFLIGHT_PROMPT_USER = "VERA preflight: reply with exactly OK."
PROBE_KIND = "vera.route_preflight_probe.v1"


class IsolationRefused(AdapterError):
    """A run may not start because the route preflight failed."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _check(
    checks: list[dict],
    route_id: str,
    name: str,
    ok: bool,
    detail: str = "",
) -> None:
    checks.append(
        {
            "route_id": route_id,
            "check": name,
            "ok": bool(ok),
            "detail": str(detail),
        }
    )


def _check_route(route: Mapping, adapter: Any, *, checks: list[dict]) -> dict:
    route_id = str(route.get("route_id") or "")
    declared = transport_spec(route)
    _check(
        checks,
        route_id,
        "declared_transport_known",
        declared is not None,
        (
            f"route transport {route.get('route')!r} is not declared in the closed "
            "registry transport vocabulary"
            if declared is None
            else json.dumps(declared, sort_keys=True)
        ),
    )
    adapter_capability = bool(getattr(adapter, "requires_cli_isolation", True))
    if declared is not None:
        _check(
            checks,
            route_id,
            "declared_transport_capability",
            bool(declared.get("requires_cli_isolation", True)) == adapter_capability,
            (
                f"declared={declared.get('requires_cli_isolation')!r} "
                f"adapter={adapter_capability!r}"
            ),
        )
        _check(
            checks,
            route_id,
            "declared_adapter_class",
            str(declared.get("adapter")) == type(adapter).__name__,
            f"declared={declared.get('adapter')!r} adapter={type(adapter).__name__!r}",
        )
    _check(
        checks,
        route_id,
        "direct_transport",
        adapter_capability is False,
        (
            "the adapter calls the provider directly"
            if adapter_capability is False
            else f"{type(adapter).__name__} needs an isolated command-line client, "
            "which this repository does not provide"
        ),
    )
    _check(
        checks,
        route_id,
        "model_matches_registry",
        str(getattr(adapter, "model_id", "")) == str(route.get("model_id") or ""),
        f"registry={route.get('model_id')!r} adapter={getattr(adapter, 'model_id', '')!r}",
    )
    return {
        "route_id": route_id,
        "adapter": type(adapter).__name__,
        "transport": str(route.get("route", "")),
        "model_id": str(route.get("model_id") or ""),
        "requires_cli_isolation": adapter_capability,
        # The isolation stage does not apply to a direct transport; the acceptance check
        # reads this flag together with the live route's declared capability.
        "skipped": adapter_capability is False,
    }


def _probe(route_id: str, adapter: Any) -> dict:
    result = adapter.complete(
        [
            {"role": "system", "content": PREFLIGHT_PROMPT_SYSTEM},
            {"role": "user", "content": PREFLIGHT_PROMPT_USER},
        ],
        max_tokens=16,
    )
    expected = str(getattr(adapter, "model_id", "") or "")
    return {
        "kind": PROBE_KIND,
        "route_id": route_id,
        "model_id": expected,
        "ok": bool(result.ok),
        "error_kind": result.error_kind,
        "error": result.error,
        "served_model_id": result.served_model_id,
        "served_model_check": served_model_check(expected, result.served_model_id),
        "prompt_tokens": int(result.prompt_tokens),
        "completion_tokens": int(result.completion_tokens),
        "latency_ms": float(result.latency_ms),
        "probed_at_utc": _utc_now(),
    }


def run_preflight(
    routes: Sequence[Mapping] | None = None,
    *,
    adapters: Mapping[str, Any] | None = None,
    probe: bool = False,
    require_measured_routes: bool = False,
) -> dict:
    """Run the route preflight.  ``probe=True`` makes one tiny call per route.

    ``require_measured_routes`` is recorded in the report; the acceptance check for a pilot
    or main-grid run requires it together with a probe.
    """
    if routes is None:
        routes = list(load_registry().get("routes") or [])
    routes = [dict(route) for route in routes]
    checks: list[dict] = []
    route_reports: list[dict] = []
    probes: list[dict] = []
    for route in routes:
        route_id = str(route.get("route_id", ""))
        adapter = (adapters or {}).get(route_id)
        if adapter is None:
            try:
                adapter = adapter_for_route(route)
            except Exception as exc:
                _check(checks, route_id, "adapter_constructs", False, f"{type(exc).__name__}: {exc}")
                route_reports.append({"route_id": route_id, "adapter": ""})
                continue
        route_reports.append(_check_route(route, adapter, checks=checks))
        if not probe or any(
            not check["ok"] for check in checks if check["route_id"] == route_id
        ):
            continue
        try:
            record = _probe(route_id, adapter)
        except Exception as exc:
            record = {
                "kind": PROBE_KIND,
                "route_id": route_id,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "error_kind": "TRANSPORT_ERROR",
            }
        probes.append(record)
        _check(
            checks,
            route_id,
            "probe_call_ok",
            record.get("ok") is True,
            f"{record.get('error_kind')}: {record.get('error')}",
        )
        _check(
            checks,
            route_id,
            "probe_served_model",
            record.get("served_model_check") == SERVED_MODEL_MATCH,
            f"served={record.get('served_model_id')!r} check={record.get('served_model_check')!r}",
        )
    refusals = [
        f"{check['route_id']}:{check['check']}: {check['detail']}"
        for check in checks
        if not check["ok"]
    ]
    return {
        "kind": PREFLIGHT_KIND,
        "stage": PREFLIGHT_STAGE_ID,
        "stage_manifest": dict(PREFLIGHT_STAGE),
        "generated_at_utc": _utc_now(),
        "ok": not refusals,
        "start_allowed": not refusals,
        "probe_performed": bool(probe),
        "require_measured_routes": bool(require_measured_routes),
        "routes": route_reports,
        "checks": checks,
        "refusals": refusals,
        "probes": probes,
    }


def load_preflight_report(source: str | Path | Mapping | None) -> dict:
    """Load a preflight report from a path or a mapping (fail closed)."""
    if source is None or str(source) == "":
        raise IsolationRefused(
            "no preflight acceptance report was supplied; a pilot or main-grid run "
            "requires the separately executed probe stage"
        )
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    if not path.is_file():
        raise IsolationRefused(f"preflight acceptance report is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IsolationRefused(
            f"preflight acceptance report is unreadable: {path}: {exc}"
        ) from None
    if not isinstance(data, dict):
        raise IsolationRefused("preflight acceptance report must be a JSON object")
    return data


def validate_preflight_acceptance(
    source: str | Path | Mapping | None,
    *,
    route_ids: Sequence[str] | None = None,
    live_routes: Mapping[str, Mapping] | None = None,
) -> dict:
    """Validate the probed preflight report that a pilot or main-grid run must present.

    The report must be ok, probed and run with ``require_measured_routes``; it must cover
    every route the run uses; each route needs a successful probe whose served model
    matches; and each route must be live in this configuration as a direct transport with
    the same model.
    """
    report = load_preflight_report(source)
    problems: list[str] = []
    if report.get("kind") != PREFLIGHT_KIND:
        problems.append(f"kind is {report.get('kind')!r}, expected {PREFLIGHT_KIND!r}")
    if str(report.get("stage") or "") != PREFLIGHT_STAGE_ID:
        problems.append(f"stage is {report.get('stage')!r}, expected {PREFLIGHT_STAGE_ID!r}")
    if report.get("ok") is not True or report.get("start_allowed") is not True:
        problems.append("the preflight report is not ok/start_allowed")
    if report.get("probe_performed") is not True:
        problems.append("the preflight stage did not perform a probe")
    if report.get("require_measured_routes") is not True:
        problems.append("the preflight was not run with require_measured_routes=True")
    if report.get("refusals"):
        problems.append(
            "the preflight recorded refusals: " + "; ".join(map(str, report["refusals"]))
        )
    stage_manifest = report.get("stage_manifest") or {}
    if stage_manifest and str(stage_manifest.get("required_before") or "") != "main_enrollment":
        problems.append("the preflight stage manifest does not gate main_enrollment")
    covered = {
        str(item.get("route_id") or ""): dict(item)
        for item in report.get("routes") or []
        if isinstance(item, Mapping)
    }
    probes = {
        str(item.get("route_id") or ""): dict(item)
        for item in report.get("probes") or []
        if isinstance(item, Mapping)
    }
    wanted = [str(item) for item in (route_ids or [])] or sorted(covered)
    live = {
        str(key): dict(value)
        for key, value in (live_routes or {}).items()
        if isinstance(value, Mapping)
    }
    for route_id in wanted:
        item = covered.get(route_id)
        if item is None:
            problems.append(f"the preflight does not cover route {route_id}")
            continue
        live_item = live.get(route_id)
        if not live_item:
            problems.append(f"route {route_id} is not a declared direct transport in this run")
        else:
            if live_item.get("requires_cli_isolation") is not False:
                problems.append(f"route {route_id} is not a direct transport")
            live_model = str(live_item.get("model_id") or "")
            if not live_model or live_model != str(item.get("model_id") or ""):
                problems.append(f"route {route_id} report model does not match the live route")
        if item.get("skipped") is not True:
            problems.append(f"route {route_id} is not recorded as a direct transport")
        probe = probes.get(route_id)
        if probe is None:
            problems.append(f"route {route_id} has no probe record")
            continue
        if probe.get("ok") is not True:
            problems.append(f"route {route_id} probe failed: {probe.get('error_kind')}")
        if probe.get("served_model_check") != SERVED_MODEL_MATCH:
            problems.append(
                f"route {route_id} probe served model {probe.get('served_model_id')!r} "
                f"does not match {probe.get('model_id')!r}"
            )
    if problems:
        raise IsolationRefused("preflight acceptance refused: " + "; ".join(problems))
    payload = json.dumps(report, sort_keys=True, default=str).encode("utf-8")
    return {
        "kind": "vera.route_preflight_acceptance.v1",
        "stage": PREFLIGHT_STAGE_ID,
        "stage_manifest": dict(stage_manifest),
        "report": report,
        "report_sha256": hashlib.sha256(payload).hexdigest(),
        "route_ids": wanted,
        "probe_route_ids": sorted(route_id for route_id in wanted if route_id in probes),
        "non_cli_route_ids": sorted(wanted),
    }


def assert_start_allowed(report: Mapping) -> dict:
    """Protocol-stage gate: raise ``IsolationRefused`` unless every check passed."""
    if report.get("kind") != PREFLIGHT_KIND:
        raise IsolationRefused("preflight report has the wrong kind; refusing to start")
    if report.get("ok") is not True or report.get("start_allowed") is not True:
        refusals = list(report.get("refusals") or ["unknown preflight failure"])
        raise IsolationRefused("run refused: route preflight failed: " + "; ".join(refusals))
    return dict(report)


def run_preflight_or_refuse(routes: Sequence[Mapping] | None = None, **kwargs) -> dict:
    report = run_preflight(routes, **kwargs)
    return assert_start_allowed(report)


def cmd_preflight(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Route preflight (no model call unless --probe)."
    )
    parser.add_argument("--registry", default="", help="route registry JSON path")
    parser.add_argument("--profile", default="", help="registry profile (e.g. claude_only)")
    parser.add_argument("--route", action="append", default=[], help="route_id to check")
    parser.add_argument("--probe", action="store_true", help="one tiny call per route")
    parser.add_argument(
        "--require-measured", action="store_true",
        help="mark the report as the probe stage a pilot or main-grid run accepts",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    if args.registry:
        routes = list(load_registry(args.registry).get("routes") or [])
    else:
        routes = list(load_registry(profile=args.profile or None).get("routes") or [])
    if args.route:
        wanted = set(args.route)
        routes = [route for route in routes if route.get("route_id") in wanted]
        missing = wanted - {str(route.get("route_id")) for route in routes}
        if missing:
            print(json.dumps(
                {"kind": PREFLIGHT_KIND, "ok": False, "refusals": [f"unknown route_id: {sorted(missing)}"]},
                indent=2, sort_keys=True,
            ))
            return 2
    report = run_preflight(
        routes, probe=bool(args.probe), require_measured_routes=bool(args.require_measured)
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 5


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cmd_preflight())


__all__ = [
    "IsolationRefused",
    "PREFLIGHT_KIND",
    "PREFLIGHT_STAGE",
    "PREFLIGHT_STAGE_ID",
    "assert_start_allowed",
    "cmd_preflight",
    "load_preflight_report",
    "run_preflight",
    "run_preflight_or_refuse",
    "validate_preflight_acceptance",
]
