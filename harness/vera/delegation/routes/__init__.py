"""Route registry (D4) and the route preflight.

Importing this package does not read a key or make a network call.
"""

from __future__ import annotations

from .registry import (
    ADAPTERS,
    MAIN_GRID_TRANSPORTS,
    PROFILE_DECLARED,
    PROFILE_MAIN_GRID,
    PUBLISHERS,
    REGISTRY_KIND,
    REGISTRY_PATH,
    ROUTE_ANTHROPIC_API,
    ROUTE_NVIDIA_NIM,
    ROUTES,
    adapter_for_route,
    default_transport_registry,
    load_registry,
    resolve_vocabulary,
    route_by_id,
    route_requires_cli_isolation,
    select_routes,
    sha256_file,
    transport_spec,
    validate_registry,
)

__all__ = [
    "ADAPTERS",
    "MAIN_GRID_TRANSPORTS",
    "PROFILE_DECLARED",
    "PROFILE_MAIN_GRID",
    "PUBLISHERS",
    "REGISTRY_KIND",
    "REGISTRY_PATH",
    "ROUTES",
    "ROUTE_ANTHROPIC_API",
    "ROUTE_NVIDIA_NIM",
    "adapter_for_route",
    "default_transport_registry",
    "load_registry",
    "resolve_vocabulary",
    "route_by_id",
    "route_requires_cli_isolation",
    "select_routes",
    "sha256_file",
    "transport_spec",
    "validate_registry",
]
