"""Roles and principals for one shared world (design §2.1).

The owner issues authority and never acts.  The orchestrator and the peers are model
principals; the harness signs records and is not a model.  A principal is a string, so a
signed log can be replayed without this module; the dataclasses exist to stop construction
from silently giving a model owner authority.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    PEER_PRINCIPAL_PREFIX,
    PRINCIPAL_HARNESS,
    PRINCIPAL_ORCHESTRATOR,
    PRINCIPAL_OWNER,
    ROLE_HARNESS,
    ROLE_ORCHESTRATOR,
    ROLE_OWNER,
    ROLE_PEER,
    SLOT_OTHER,
    SLOT_SELF,
    SLOT_SIBLING,
    SLOT_THIRD,
)

DEFAULT_ORCHESTRATOR_VENDOR = "vendor_alpha"
DEFAULT_OTHER_VENDOR = "vendor_beta"


class PrincipalError(ValueError):
    """A role/principal construction would give a model authority it must not have."""


@dataclass(frozen=True)
class Role:
    name: str
    principal: str
    is_model: bool

    def __post_init__(self) -> None:
        if self.name == ROLE_OWNER and self.is_model:
            raise PrincipalError("the task owner is never a model (design §2.1)")
        if self.name == ROLE_HARNESS and self.is_model:
            raise PrincipalError("the harness is not a model and cannot be prompted")
        if self.name == ROLE_ORCHESTRATOR and not self.is_model:
            raise PrincipalError("the orchestrator is a model principal")
        if self.name == ROLE_PEER and not self.is_model:
            raise PrincipalError("a peer is a model principal")


@dataclass(frozen=True)
class PeerSpec:
    """One menu slot.  ``vendor_family`` is the publisher of the weights (§6.1)."""

    slot: str
    principal: str
    model_id: str
    vendor_family: str
    capability_band: str

    @property
    def display_name(self) -> str:
        return f"{self.vendor_family}:{self.model_id}"


def default_peers(
    *,
    orchestrator_model_id: str = "mock-orchestrator",
    orchestrator_vendor: str = DEFAULT_ORCHESTRATOR_VENDOR,
    sibling_model_id: str = "mock-sibling",
    other_model_id: str = "mock-other",
    other_vendor: str = DEFAULT_OTHER_VENDOR,
    third_model_id: str = "mock-third-publisher",
    third_vendor: str = "vendor_gamma",
    include_third: bool = False,
) -> tuple[PeerSpec, ...]:
    """Peers for one cell.

    The v1.3 D4 menu is four slots (clone, own-publisher sibling, other-frontier peer,
    third-publisher peer).  ``include_third`` defaults to False only so the frozen v1.2
    pattern/base tests keep their historical three-slot construction; v1.3 call sites pass
    True or use :func:`default_peers_v13`.
    """
    return (
        PeerSpec(
            slot=SLOT_SELF,
            principal=PEER_PRINCIPAL_PREFIX + SLOT_SELF,
            model_id=orchestrator_model_id,
            vendor_family=orchestrator_vendor,
            capability_band="band_same_weights",
        ),
        PeerSpec(
            slot=SLOT_SIBLING,
            principal=PEER_PRINCIPAL_PREFIX + SLOT_SIBLING,
            model_id=sibling_model_id,
            vendor_family=orchestrator_vendor,
            capability_band="band_within_0_10",
        ),
        PeerSpec(
            slot=SLOT_OTHER,
            principal=PEER_PRINCIPAL_PREFIX + SLOT_OTHER,
            model_id=other_model_id,
            vendor_family=other_vendor,
            capability_band="band_within_0_10",
        ),
    ) + ((PeerSpec(
            slot=SLOT_THIRD,
            principal=PEER_PRINCIPAL_PREFIX + SLOT_THIRD,
            model_id=third_model_id,
            vendor_family=third_vendor,
            capability_band="band_within_0_10",
        ),) if include_third else ())


def default_peers_v13(
    *,
    orchestrator_model_id: str = "mock-orchestrator",
    orchestrator_vendor: str = DEFAULT_ORCHESTRATOR_VENDOR,
    sibling_model_id: str = "mock-sibling",
    other_model_id: str = "mock-other",
    other_vendor: str = DEFAULT_OTHER_VENDOR,
    third_model_id: str = "mock-third-publisher",
    third_vendor: str = "vendor_gamma",
) -> tuple[PeerSpec, ...]:
    """The required D4 four-slot identity menu for v1.3 runs."""
    return default_peers(
        orchestrator_model_id=orchestrator_model_id,
        orchestrator_vendor=orchestrator_vendor,
        sibling_model_id=sibling_model_id,
        other_model_id=other_model_id,
        other_vendor=other_vendor,
        third_model_id=third_model_id,
        third_vendor=third_vendor,
        include_third=True,
    )


def principal_set(peers: tuple[PeerSpec, ...] | list[PeerSpec]) -> dict[str, str]:
    principals = {
        "owner": PRINCIPAL_OWNER,
        "orchestrator": PRINCIPAL_ORCHESTRATOR,
        "harness": PRINCIPAL_HARNESS,
    }
    for peer in peers:
        principals[f"peer_{peer.slot}"] = peer.principal
    return principals


def roles_for(peer: PeerSpec) -> Role:
    return Role(ROLE_PEER, peer.principal, True)
