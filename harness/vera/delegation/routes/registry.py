"""D4 route registry and adapter factory.

``registry.json`` is the frozen D4 identity menu.  The ``publisher`` field always names the
organisation that trained the weights; ``route`` names how the call is transported
(``anthropic_api`` or ``nvidia_nim``).  Keeping those fields separate matters for the
identity experiment: gpt-oss-20b is an OpenAI-weight model reached over NVIDIA's free route,
so its publisher is OpenAI while its route is NVIDIA.

No function here reads a key or makes a call.  A route names the environment variable its
key is read from (``api_key_env``); the adapter reads it at call time.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping

from ..adapters import AnthropicApiAdapter, NvidiaAdapter

REGISTRY_PATH = Path(__file__).with_name("registry.json")
REGISTRY_KIND = "vera.route_registry.v1"
ROUTE_ANTHROPIC_API = "anthropic_api"
ROUTE_NVIDIA_NIM = "nvidia_nim"
ROUTES = (ROUTE_ANTHROPIC_API, ROUTE_NVIDIA_NIM)
PUBLISHERS = ("OpenAI", "Anthropic", "NVIDIA")
ADAPTERS = {
    "AnthropicApiAdapter": AnthropicApiAdapter,
    "NvidiaAdapter": NvidiaAdapter,
}

#: The route vocabulary is resolved from data, but it stays closed and
#: fail-closed.  ``main_grid`` is the sealed D4 profile: it is the default, it refuses any
#: override, and every unknown transport/publisher/adapter is rejected.  A non-main-grid
#: registry may declare its own vocabulary only with an explicit ``profile: declared``
#: block, ``not_main_grid: true`` and adapter capabilities that match the code.
PROFILE_MAIN_GRID = "main_grid"
PROFILE_DECLARED = "declared"
PROFILES = (PROFILE_MAIN_GRID, PROFILE_DECLARED)
MAIN_GRID_TRANSPORTS: dict[str, dict] = {
    ROUTE_ANTHROPIC_API: {
        "adapter": "AnthropicApiAdapter",
        "requires_cli_isolation": False,
    },
    ROUTE_NVIDIA_NIM: {
        "adapter": "NvidiaAdapter",
        "requires_cli_isolation": False,
    },
}
FREE_TRANSPORTS = (ROUTE_NVIDIA_NIM,)
REQUIRED_FIELDS = (
    "route_id",
    "model_id",
    "publisher",
    "route",
    "adapter",
    "identity_slot",
)
#: A key must come from the environment: the registry may name the variable, never hold
#: the value.
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


REGISTRY_CLAUDE_ONLY_PATH = Path(__file__).with_name("registry_claude_only.json")
PILOT_PROFILE_CLAUDE_ONLY = "claude_only"


def load_registry(path: str | Path | None = None, *, profile: str | None = None) -> dict:
    """Load and validate the route registry; raise ``ValueError`` on any defect.

    ``profile="claude_only"`` loads the pilot profile: the two Claude routes only, with
    slot names O1 = Sonnet 5.5 and O2 = Haiku 5.5 (see ``identity_assignment``).  It is never
    the main grid.
    """
    if profile:
        if profile != PILOT_PROFILE_CLAUDE_ONLY:
            raise ValueError(f"unknown registry profile {profile!r}")
        if path is not None:
            raise ValueError("a registry profile and an explicit path are mutually exclusive")
        path = REGISTRY_CLAUDE_ONLY_PATH
    registry_path = Path(path) if path is not None else REGISTRY_PATH
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"route registry is unreadable: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"route registry is not JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ValueError("route registry must be a JSON object")
    report = validate_registry(data)
    if not report["valid"]:
        raise ValueError("invalid route registry: " + "; ".join(report["errors"]))
    data["_registry_path"] = str(registry_path)
    data["_registry_sha256"] = sha256_file(registry_path)
    data["_profile"] = report["profile"]
    data["_transport_registry"] = {
        name: dict(spec) for name, spec in report["transport_registry"].items()
    }
    data["_publishers"] = list(report["publishers"])
    # Stamp each route with the resolved table so the preflight can verify the declared
    # transport capability without re-reading the registry or trusting a literal label.
    for route in data.get("routes") or []:
        if isinstance(route, dict):
            route["_transport_registry"] = {
                name: dict(spec)
                for name, spec in report["transport_registry"].items()
            }
    return data


def resolve_vocabulary(data: Mapping | None = None) -> dict:
    """Resolve a registry's closed transport/publisher vocabulary.

    The default (``main_grid``) profile is the frozen D4 table and refuses every override.
    A shakedown/instrument registry must say ``profile: declared`` and ``not_main_grid:
    true`` and enumerate its transports; each declared transport must name a *code-known*
    adapter class and the declared capability must match that class.  An unknown profile,
    transport, adapter or publisher is an error, never a silent acceptance.
    """
    data = dict(data or {})
    errors: list[str] = []
    profile = str(data.get("profile") or PROFILE_MAIN_GRID)
    transports: dict[str, dict] = {}
    publishers: list[str] = []
    if profile == PROFILE_MAIN_GRID:
        if data.get("transport_registry") not in (None, {}, []):
            errors.append(
                "profile 'main_grid' is sealed; a transport_registry override is refused"
            )
        if data.get("publishers") not in (None, [], ()):
            errors.append(
                "profile 'main_grid' is sealed; a publishers override is refused"
            )
        transports = {name: dict(spec) for name, spec in MAIN_GRID_TRANSPORTS.items()}
        publishers = list(PUBLISHERS)
    elif profile == PROFILE_DECLARED:
        if data.get("not_main_grid") is not True:
            errors.append("profile 'declared' requires not_main_grid: true")
        raw = data.get("transport_registry")
        if not isinstance(raw, Mapping) or not raw:
            errors.append(
                "profile 'declared' requires a non-empty transport_registry object"
            )
            raw = {}
        for name, spec in raw.items():
            transport = str(name)
            if not transport:
                errors.append("transport_registry contains an empty transport name")
                continue
            if not isinstance(spec, Mapping):
                errors.append(f"transport_registry.{transport} must be an object")
                continue
            adapter_name = str(spec.get("adapter") or "")
            capability = spec.get("requires_cli_isolation")
            if adapter_name not in ADAPTERS:
                errors.append(
                    f"transport_registry.{transport}.adapter must be one of "
                    f"{sorted(ADAPTERS)}"
                )
            if not isinstance(capability, bool):
                errors.append(
                    f"transport_registry.{transport}.requires_cli_isolation must be a "
                    "boolean capability"
                )
                capability = True
            elif adapter_name in ADAPTERS:
                actual = bool(
                    getattr(ADAPTERS[adapter_name], "requires_cli_isolation", True)
                )
                if actual != capability:
                    errors.append(
                        f"transport_registry.{transport}: declared "
                        f"requires_cli_isolation={capability} contradicts adapter "
                        f"{adapter_name} ({actual})"
                    )
            transports[transport] = {
                "adapter": adapter_name,
                "requires_cli_isolation": capability,
            }
        raw_publishers = data.get("publishers")
        if (
            not isinstance(raw_publishers, list)
            or not raw_publishers
            or any(not isinstance(item, str) or not item.strip() for item in raw_publishers)
        ):
            errors.append(
                "profile 'declared' requires a non-empty publishers list of names"
            )
        else:
            publishers = [str(item) for item in raw_publishers]
    else:
        errors.append(f"profile must be one of {list(PROFILES)}, got {profile!r}")
    return {
        "profile": profile,
        "transports": transports,
        "publishers": publishers,
        "errors": errors,
    }


def default_transport_registry() -> dict[str, dict]:
    """The sealed D4 transport table, returned as a copy."""
    return {name: dict(spec) for name, spec in MAIN_GRID_TRANSPORTS.items()}


def transport_spec(
    route: Mapping, table: Mapping[str, Mapping] | None = None
) -> dict | None:
    """Return the declared transport entry for a route row, or ``None``."""
    resolved = table if table is not None else route.get("_transport_registry")
    if not isinstance(resolved, Mapping):
        resolved = MAIN_GRID_TRANSPORTS
    spec = resolved.get(str(route.get("route", "")))
    return dict(spec) if isinstance(spec, Mapping) else None


def route_requires_cli_isolation(
    route: Mapping, table: Mapping[str, Mapping] | None = None
) -> bool:
    """Fail-closed: an undeclared transport counts as one that needs isolation, and no
    adapter in this repository provides it, so the preflight refuses the route."""
    spec = transport_spec(route, table)
    if spec is None:
        return True
    return bool(spec.get("requires_cli_isolation", True))


def validate_registry(data: Mapping) -> dict:
    errors: list[str] = []
    if data.get("kind") != REGISTRY_KIND:
        errors.append(f"kind must be {REGISTRY_KIND!r}")
    vocabulary = resolve_vocabulary(data)
    errors.extend(vocabulary["errors"])
    transports = vocabulary["transports"]
    publishers = vocabulary["publishers"]
    routes = data.get("routes")
    if not isinstance(routes, list) or not routes:
        errors.append("routes must be a non-empty list")
        routes = []
    seen_ids: set[str] = set()
    seen_models: set[str] = set()
    for index, route in enumerate(routes):
        prefix = f"routes[{index}]"
        if not isinstance(route, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for field_name in REQUIRED_FIELDS:
            if not isinstance(route.get(field_name), str) or not route[field_name].strip():
                errors.append(f"{prefix}.{field_name} must be a non-empty string")
        route_id = str(route.get("route_id", ""))
        if route_id:
            if "/" in route_id or "\\" in route_id or route_id in {".", ".."}:
                errors.append(f"{prefix}.route_id must be a safe filename component")
            if route_id in seen_ids:
                errors.append(f"{prefix}.route_id is duplicated: {route_id}")
            seen_ids.add(route_id)
        model_id = str(route.get("model_id", ""))
        if model_id in seen_models:
            errors.append(f"{prefix}.model_id is duplicated: {model_id}")
        seen_models.add(model_id)
        publisher = route.get("publisher")
        if publisher not in publishers:
            errors.append(
                f"{prefix}.publisher must be one of the closed "
                f"{vocabulary['profile']} list {publishers}"
            )
        transport = route.get("route")
        if transport not in transports:
            errors.append(
                f"{prefix}.route must be declared in the closed "
                f"{vocabulary['profile']} transport vocabulary "
                f"{sorted(transports)}; got {transport!r}"
            )
            continue
        spec = transports[str(transport)]
        adapter_name = route.get("adapter")
        if adapter_name not in ADAPTERS:
            errors.append(f"{prefix}.adapter must be one of {sorted(ADAPTERS)}")
        elif adapter_name != spec.get("adapter"):
            errors.append(
                f"{prefix}.adapter {adapter_name!r} does not match route "
                f"{transport!r} (expected {spec.get('adapter')!r})"
            )
        key_env = route.get("api_key_env")
        if key_env is not None and (
            not isinstance(key_env, str) or not _ENV_NAME_RE.match(key_env)
        ):
            errors.append(
                f"{prefix}.api_key_env must name an environment variable "
                "(upper case, digits, underscores); a key never goes in the registry"
            )
        max_output = route.get("max_output_tokens")
        if max_output is not None and (
            not isinstance(max_output, int) or isinstance(max_output, bool) or max_output < 1
        ):
            errors.append(f"{prefix}.max_output_tokens must be a positive integer")
    excluded = data.get("excluded_models", [])
    if not isinstance(excluded, list):
        errors.append("excluded_models must be a list")
    elif "reviewer-model" not in excluded:
        errors.append("reviewer-model must be explicitly excluded (reviewer independence)")
    return {
        "valid": not errors,
        "errors": errors,
        "route_count": len(routes),
        "profile": vocabulary["profile"],
        "transport_registry": transports,
        "publishers": publishers,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def route_by_id(registry: Mapping, route_id: str) -> dict:
    for route in registry.get("routes", []):
        if route.get("route_id") == route_id:
            return dict(route)
    raise KeyError(f"unknown route {route_id!r}")


def select_routes(registry: Mapping, route_ids: list[str] | None = None) -> list[dict]:
    if not route_ids:
        return [dict(route) for route in registry.get("routes", [])]
    wanted = list(dict.fromkeys(route_ids))
    return [route_by_id(registry, route_id) for route_id in wanted]


def adapter_for_route(
    route: Mapping,
    *,
    timeout_seconds: float | None = None,
) -> AnthropicApiAdapter | NvidiaAdapter:
    """Construct the adapter for a registry entry.  Does not execute anything."""
    adapter_name = route.get("adapter")
    model_id = str(route["model_id"])
    if adapter_name == "NvidiaAdapter":
        return NvidiaAdapter(model_id)
    if adapter_name == "AnthropicApiAdapter":
        options: dict = {"vendor_family": str(route.get("publisher") or "Anthropic")}
        if route.get("api_key_env"):
            options["api_key_env"] = str(route["api_key_env"])
        if route.get("max_output_tokens"):
            options["max_output_tokens"] = int(route["max_output_tokens"])
        if timeout_seconds is not None:
            options["timeout_seconds"] = float(timeout_seconds)
        return AnthropicApiAdapter(model_id, **options)
    raise ValueError(f"unknown adapter {adapter_name!r}")


__all__ = [
    "ADAPTERS",
    "FREE_TRANSPORTS",
    "MAIN_GRID_TRANSPORTS",
    "PROFILES",
    "PROFILE_DECLARED",
    "PROFILE_MAIN_GRID",
    "AnthropicApiAdapter",
    "NvidiaAdapter",
    "PUBLISHERS",
    "REGISTRY_KIND",
    "REGISTRY_PATH",
    "ROUTES",
    "ROUTE_ANTHROPIC_API",
    "ROUTE_NVIDIA_NIM",
    "adapter_for_route",
    "default_transport_registry",
    "load_registry",
    "route_by_id",
    "route_requires_cli_isolation",
    "resolve_vocabulary",
    "select_routes",
    "sha256_file",
    "transport_spec",
    "validate_registry",
]
