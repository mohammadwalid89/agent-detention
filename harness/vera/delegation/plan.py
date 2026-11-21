"""Cell construction: frozen instance + condition + grants + peer menu (design §2.7).

A cell is fully determined before any model call.  The owner issues every token at
construction; the orchestrator and peers can only route/use authority they were already
given.  The cell hash binds the instance, the condition envelope, the grants, the identity
condition, the seed and the repeat.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..canonical import canonical_json, sha256_text
from ..contracts import DelegationGrant
from ..pilot.instances import Instance
from .case_text import apply_case_text, case_text_digest
from .conditions import (
    ConditionSpec,
    Derivation,
    build_resource_map,
    derive_conditions,
    task_text_for,
)
from .constants import (
    COND_DODGE,
    CUE_EXPLICIT,
    DEFAULT_DELEGATION_BUDGET,
    DEFAULT_PATTERN,
    DEFAULT_STEP_BUDGET,
    IDENTITY_NAMED,
    PRINCIPAL_ORCHESTRATOR,
    PRINCIPAL_OWNER,
    SLOT_OTHER,
    SLOT_SIBLING,
)
from .evidence import PROTOCOL_V13, PROTOCOL_V14, apply_evidence_read
from .masking import PeerMenu, build_peer_menu
from .patterns.config import config_digest, config_for
from .principals import (
    DEFAULT_ORCHESTRATOR_VENDOR,
    PeerSpec,
    default_peers,
    default_peers_v13,
)
from .tokens import AuthorityEnvelope, GrantIssuer, TokenStore

TS_ISSUED = "2026-11-13T00:00:00Z"
TS_EXPIRES = "2026-11-14T00:00:00Z"
OWNER_ACTION_CLASS = "*"


@dataclass(frozen=True)
class DelegationCell:
    instance: Instance
    condition: str
    spec: ConditionSpec
    derivation: Derivation
    seed: int
    repeat: int
    identity_condition: str
    peers: tuple[PeerSpec, ...]
    menu: PeerMenu
    grants: dict
    envelopes: dict
    resource_map: dict
    orchestrator_principal: str
    orchestrator_model_id: str
    orchestrator_vendor: str
    owner_principal: str
    cell_hash: str
    step_budget: int = DEFAULT_STEP_BUDGET
    delegation_budget: int = DEFAULT_DELEGATION_BUDGET
    pattern: str = DEFAULT_PATTERN
    ts_issued: str = TS_ISSUED
    ts_expires: str = TS_EXPIRES
    protocol: str = PROTOCOL_V13
    # M11: a solo cell has no peer menu and no delegation: the orchestrator is shown no
    # ``delegation`` block, its prompt does not mention delegating, and the budget is 0.
    solo: bool = False

    @property
    def pattern_config(self):
        return config_for(self.pattern)

    def token_store(self) -> TokenStore:
        return TokenStore(list(self.grants.values()))

    def grant_for(self, principal: str) -> DelegationGrant:
        if principal not in self.grants:
            raise KeyError(f"no grant issued to {principal!r}")
        return self.grants[principal]

    def o_grant(self) -> DelegationGrant:
        return self.grant_for(self.orchestrator_principal)

    def task_text(self) -> str:
        return task_text_for(self.instance, self.spec)

    @property
    def owner_cue(self) -> str:
        return self.spec.owner_cue

    def header_binding(self) -> dict:
        binding = {
            "cell_hash": self.cell_hash,
            "instance_id": self.instance.instance_id,
            "instance_hash": self.instance.instance_hash,
            "mechanism_id": self.instance.mechanism_id,
            "variant_id": self.instance.variant_id,
            "family": self.instance.family,
            "instance_label": self.instance.label,
            "condition": self.condition,
            "owner_cue": self.owner_cue,
            "orchestrator_model_id": self.orchestrator_model_id,
            "orchestrator_vendor": self.orchestrator_vendor,
            "identity_condition": self.identity_condition,
            "seed": self.seed,
            "repeat": self.repeat,
            "step_budget": self.step_budget,
            "delegation_budget": self.delegation_budget,
            "endpoint_resources": list(self.spec.endpoint_resources),
            "required_actions": list(self.spec.required_actions),
            "o_action_classes": list(self.spec.o_action_classes),
            "p_action_classes": list(self.spec.p_action_classes),
            "o_resources": list(self.spec.o_resources),
            "p_resources": list(self.spec.p_resources),
            "condition_fingerprint": self.derivation.fingerprint,
        }
        if not self.pattern_config.is_base:
            binding["pattern"] = self.pattern
            binding["pattern_config_digest"] = config_digest(self.pattern)
        if self.protocol != PROTOCOL_V13:
            binding["protocol"] = self.protocol
            binding["evidence_read"] = self.instance.evidence_read
            binding["case_text_sha256"] = case_text_digest(self.instance)
        if self.solo:
            binding["solo"] = True
        return binding


def _grant_id(public_ref: str, condition: str, slot: str, seed: int, repeat: int) -> str:
    return "g-" + sha256_text(f"{public_ref}|{condition}|{slot}|{seed}|{repeat}")[:18]


def _make_grant(
    *,
    cell_public_ref: str,
    condition: str,
    seed: int,
    repeat: int,
    slot: str,
    grantee: str,
    action_classes: tuple[str, ...],
    resources: tuple[str, ...],
) -> DelegationGrant:
    payload = {
        "grantee": grantee,
        "action_classes": list(action_classes),
        "resources": list(resources),
        "condition": condition,
        "slot": slot,
        "seed": seed,
        "repeat": repeat,
    }
    return DelegationGrant(
        id=_grant_id(cell_public_ref, condition, slot, seed, repeat),
        grantor_principal=PRINCIPAL_OWNER,
        grantee_principal=grantee,
        scope=tuple(sorted(action_classes)),
        resources=tuple(sorted(resources)),
        action_classes=tuple(sorted(action_classes)),
        issued_at=TS_ISSUED,
        expires_at=TS_EXPIRES,
        revocation_id="rev-" + sha256_text(canonical_json(payload))[:12],
        purpose=f"v1.2 {condition} {slot} authority for {cell_public_ref}",
        envelope_signature="owner-sha256:" + sha256_text(canonical_json(payload)),
    )


def build_cell(
    instance: Instance,
    condition: str,
    *,
    seed: int = 20261113,
    repeat: int = 0,
    identity_condition: str = IDENTITY_NAMED,
    owner_cue: str = CUE_EXPLICIT,
    peers: tuple[PeerSpec, ...] | None = None,
    orchestrator_model_id: str = "mock-orchestrator",
    orchestrator_vendor: str = DEFAULT_ORCHESTRATOR_VENDOR,
    derivation: Derivation | None = None,
    pattern: str = DEFAULT_PATTERN,
    four_slot: bool = False,
    menu_position_seed: int | None = None,
    label_permutation: int | None = None,
    protocol: str = PROTOCOL_V13,
    solo: bool = False,
) -> DelegationCell:
    if derivation is None:
        derivation = derive_conditions(instance, protocol=protocol)
    elif derivation.protocol != protocol:
        raise ValueError(
            f"derivation was made under protocol {derivation.protocol!r}, cell asks {protocol!r}"
        )
    if protocol == PROTOCOL_V14:
        # The cell's instance carries the evidence-read action (and its vocabulary entry).
        instance = apply_case_text(
            apply_evidence_read(instance, derivation.evidence_read)
        )
    pattern_config = config_for(pattern)
    spec = (
        derivation.spec_for(COND_DODGE, owner_cue)
        if condition == COND_DODGE
        else derivation.conditions[condition]
    )
    if not spec.applicable:
        raise ValueError(
            f"{instance.instance_id} cannot build {condition}: {spec.reason}"
        )
    peers = peers or (
        default_peers_v13(
            orchestrator_model_id=orchestrator_model_id,
            orchestrator_vendor=orchestrator_vendor,
        ) if four_slot else default_peers(
            orchestrator_model_id=orchestrator_model_id,
            orchestrator_vendor=orchestrator_vendor,
        )
    )
    resource_map = build_resource_map(instance)

    o_classes = spec.o_action_classes
    o_resources = spec.o_resources or spec.endpoint_resources
    if not o_classes or not o_resources:
        raise ValueError(
            f"{instance.instance_id} {condition}: orchestrator envelope is empty"
        )

    owner_envelope = AuthorityEnvelope(
        action_classes=(OWNER_ACTION_CLASS,),
        resources=("*",),
        expires_at=TS_EXPIRES,
    )
    store = TokenStore()
    issuer = GrantIssuer(
        store=store, owner_principal=PRINCIPAL_OWNER, owner_envelope=owner_envelope
    )

    grants: dict[str, DelegationGrant] = {}
    envelopes: dict[str, AuthorityEnvelope] = {}
    o_grant = _make_grant(
        cell_public_ref=instance.public_ref,
        condition=condition,
        seed=seed,
        repeat=repeat,
        slot="orchestrator",
        grantee=PRINCIPAL_ORCHESTRATOR,
        action_classes=tuple(o_classes),
        resources=tuple(o_resources),
    )
    issuer.issue(o_grant)
    grants[PRINCIPAL_ORCHESTRATOR] = o_grant
    envelopes[PRINCIPAL_ORCHESTRATOR] = AuthorityEnvelope(
        action_classes=tuple(o_classes), resources=tuple(o_resources), expires_at=TS_EXPIRES
    )

    for peer in peers:
        p_classes = spec.p_action_classes
        p_resources = spec.p_resources or spec.endpoint_resources
        if not p_classes or not p_resources:
            raise ValueError(
                f"{instance.instance_id} {condition}: peer {peer.slot} envelope is empty"
            )
        grant = _make_grant(
            cell_public_ref=instance.public_ref,
            condition=condition,
            seed=seed,
            repeat=repeat,
            slot=peer.slot,
            grantee=peer.principal,
            action_classes=tuple(p_classes),
            resources=tuple(p_resources),
        )
        issuer.issue(grant)
        grants[peer.principal] = grant
        envelopes[peer.principal] = AuthorityEnvelope(
            action_classes=tuple(p_classes),
            resources=tuple(p_resources),
            expires_at=TS_EXPIRES,
        )

    menu = build_peer_menu(
        peers, identity_condition=identity_condition,
        seed=seed if menu_position_seed is None else menu_position_seed,
        label_permutation=label_permutation,
    )
    binding = {
        "instance_hash": instance.instance_hash,
        "condition": condition,
        "spec": spec.to_dict(),
        "derivation_fingerprint": derivation.fingerprint,
        "seed": seed,
        "repeat": repeat,
        "identity_condition": identity_condition,
        "four_slot": bool(four_slot),
        "menu": menu.record(),
        "grants": [
            {
                "id": grants[principal].id,
                "grantee": principal,
                "action_classes": list(grants[principal].action_classes),
                "resources": list(grants[principal].resources),
                "issued_at": grants[principal].issued_at,
                "expires_at": grants[principal].expires_at,
                "envelope_signature": grants[principal].envelope_signature,
            }
            for principal in sorted(grants)
        ],
    }
    # The pattern factor is OPTIONAL: the base design's cell hash must not move, so the
    # pattern fields are bound only when a non-base contract is selected.
    if not pattern_config.is_base:
        binding["pattern"] = pattern_config.pattern
        binding["pattern_config_digest"] = pattern_config.digest
    if protocol != PROTOCOL_V13:
        binding["protocol"] = protocol
        binding["evidence_read"] = instance.evidence_read
        binding["case_text_sha256"] = case_text_digest(instance)
    if solo:
        binding["solo"] = True
    cell_hash = sha256_text(canonical_json(binding))
    return DelegationCell(
        instance=instance,
        condition=condition,
        spec=spec,
        derivation=derivation,
        seed=seed,
        repeat=repeat,
        identity_condition=identity_condition,
        peers=tuple(peers),
        menu=menu,
        grants=grants,
        envelopes=envelopes,
        resource_map=resource_map,
        orchestrator_principal=PRINCIPAL_ORCHESTRATOR,
        orchestrator_model_id=orchestrator_model_id,
        orchestrator_vendor=orchestrator_vendor,
        owner_principal=PRINCIPAL_OWNER,
        cell_hash=cell_hash,
        pattern=pattern_config.pattern,
        protocol=protocol,
        solo=bool(solo),
        **({"delegation_budget": 0} if solo else {}),
    )


def retry_cell(
    cell: DelegationCell,
    *,
    repeat: int,
    seed: int | None = None,
) -> DelegationCell:
    """A fresh cell with the same condition and a *fresh* budget for a Detention retry."""
    return build_cell(
        cell.instance,
        cell.condition,
        seed=cell.seed if seed is None else seed,
        repeat=repeat,
        identity_condition=cell.identity_condition,
        owner_cue=cell.owner_cue,
        peers=cell.peers,
        orchestrator_model_id=cell.orchestrator_model_id,
        orchestrator_vendor=cell.orchestrator_vendor,
        derivation=cell.derivation,
        pattern=cell.pattern,
        # JOB56/Astra finding 4: a retry changes only the declared repeat/treatment id.
        # The original menu/identity assignment (and its independent position seed) is
        # carried over exactly instead of being redrawn from the new repeat.
        menu_position_seed=cell.menu.seed,
        label_permutation=cell.menu.label_permutation,
        protocol=cell.protocol,
        solo=cell.solo,
    )


def peer_by_slot(cell: DelegationCell, slot: str) -> PeerSpec:
    for peer in cell.peers:
        if peer.slot == slot:
            return peer
    raise KeyError(f"no peer slot {slot!r}")


def visible_peer_profile(cell: DelegationCell) -> list[dict]:
    """Profiles the orchestrator may read.

    The ``publisher`` field is the publisher label actually displayed under the cell's
    identity condition: truthful under NAMED, a permuted label over all fixed backends
    (clone included) under SWAPPED, and ``hidden`` under MASKED.  M3: no capability card is
    rendered at all until a real, bound card exists (the placeholder said nothing and named
    itself a placeholder); no clone tag (``band_*``) can leak into an observation.
    """
    profiles: list[dict] = []
    for entry in cell.menu.entries:
        grant = cell.grant_for(entry.principal)
        profiles.append({
            "label": entry.display_label,
            "publisher": entry.displayed_publisher,
            "token_id": grant.id,
            "action_classes": list(grant.action_classes),
            "resources": list(grant.resources),
        })
    return profiles


def sibling_and_other(cell: DelegationCell) -> tuple[PeerSpec, PeerSpec]:
    return peer_by_slot(cell, SLOT_SIBLING), peer_by_slot(cell, SLOT_OTHER)
