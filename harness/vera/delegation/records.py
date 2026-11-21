"""Signed, append-only episode records (design §2.4).

Every object is canonical-JSON hashed and HMAC-signed:

* harness-key entries: header, identity assignment, handoff, peer/world step, delegation
  result, episode report, close and gaming events;
* verifier-key entries: actor-bound verification events (``pilot-verifier-v2``).

The verifier key is the v1 HMAC key and the signing primitive is exactly
``vera.verifier.Verifier.sign``.  The log is a hash chain, so the custodian can replay an
episode end to end without a model, and any mutation is detectable.

The log is **custodian material**.  The agent under test never sees it: observations are
built from the shared world only (see ``world.py``).
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass, field
from typing import Iterable

from ..canonical import canonical_json, sha256_text
from ..verifier import Verifier
from .constants import VERIFIER_VERSION_ACTOR_BOUND

LOG_VERSION = "delegation-log-v1"
GENESIS = "0" * 64

KIND_HEADER = "EPISODE_HEADER"
KIND_IDENTITY = "IDENTITY_ASSIGNMENT"
KIND_HANDOFF = "DELEGATION_HANDOFF"
KIND_STEP = "STEP"
KIND_VERDICT = "VERIFICATION_EVENT"
KIND_RESULT = "DELEGATION_RESULT"
KIND_REPORT = "EPISODE_REPORT"
KIND_REPORT_ATTEMPT = "EPISODE_REPORT_ATTEMPT"
KIND_CLOSE = "EPISODE_CLOSE"
KIND_GAMING = "GAMING_EVENT"
# P4 ledger orchestrator (JOB21): the *content* is written by the orchestrator agent; the
# harness signs the record, so the signature attests what the agent emitted, not that the
# content is true.  A ledger is therefore an agent-written self-report, not a verdict.
KIND_TASK_LEDGER = "TASK_LEDGER"
KIND_PROGRESS_LEDGER = "PROGRESS_LEDGER"

VERIFIER_KINDS = frozenset({KIND_VERDICT})


def _signer_for(kind: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    return "verifier" if kind in VERIFIER_KINDS else "harness"


@dataclass(frozen=True)
class ActorVerdict:
    """``OperationalVerdict`` extended with the acting principal (design §2.8)."""

    id: str
    run_id: str
    actor_principal: str
    agent_role: str
    verdict: str
    checks: tuple[dict, ...]
    authorization_ok: bool
    monitor_integrity_ok: bool
    snapshot_hash: str
    ground_truth_version: str
    ts: str
    verifier_version: str = VERIFIER_VERSION_ACTOR_BOUND
    signature_hmac: str = ""

    def signing_payload(self) -> str:
        return canonical_json(self.to_dict(signature=False))

    def to_dict(self, *, signature: bool = True) -> dict:
        data = {
            "id": self.id,
            "run_id": self.run_id,
            "actor_principal": self.actor_principal,
            "agent_role": self.agent_role,
            "verdict": self.verdict,
            "checks": [dict(check) for check in self.checks],
            "authorization_ok": bool(self.authorization_ok),
            "monitor_integrity_ok": bool(self.monitor_integrity_ok),
            "snapshot_hash": self.snapshot_hash,
            "ground_truth_version": self.ground_truth_version,
            "ts": self.ts,
            "verifier_version": self.verifier_version,
        }
        if signature:
            data["signature_hmac"] = self.signature_hmac
        return data


def sign_actor_verdict(verdict: ActorVerdict, verifier_key: bytes) -> ActorVerdict:
    signer = Verifier(key=verifier_key)
    signature = signer.sign(verdict.signing_payload())
    return ActorVerdict(**{**verdict.__dict__, "signature_hmac": signature})


def actor_verdict_valid(verdict: ActorVerdict, verifier_key: bytes) -> bool:
    if not verdict.signature_hmac:
        return False
    signer = Verifier(key=verifier_key)
    expected = signer.sign(verdict.signing_payload())
    return hmac.compare_digest(expected, verdict.signature_hmac)


def actor_verdict_from_dict(row: dict) -> ActorVerdict:
    data = {k: v for k, v in row.items() if k != "_schema"}
    data["checks"] = tuple(data.get("checks") or ())
    return ActorVerdict(**data)


class SignedLog:
    """Append-only hash chain with per-record HMAC signatures."""

    def __init__(self, *, harness_key: bytes, verifier_key: bytes):
        if not harness_key or not verifier_key:
            raise ValueError("both the harness key and the verifier key are required to sign")
        self.harness_key = harness_key
        self.verifier_key = verifier_key
        self._entries: list[dict] = []

    # -- writing -----------------------------------------------------------------
    def append(self, kind: str, payload: dict, *, signer: str | None = None) -> dict:
        signer_name = _signer_for(kind, signer)
        prev = self._entries[-1]["hash"] if self._entries else GENESIS
        payload_json = canonical_json(payload)
        body = prev + "|" + kind + "|" + payload_json
        entry = {
            "seq": len(self._entries) + 1,
            "kind": kind,
            "payload": payload,
            "prev_hash": prev,
            "hash": sha256_text(body),
            "signer": signer_name,
            "signature": self._sign(body, signer_name),
        }
        self._entries.append(entry)
        return entry

    def _key(self, signer_name: str) -> bytes:
        if signer_name == "verifier":
            return self.verifier_key
        return self.harness_key

    def _sign(self, body: str, signer_name: str) -> str:
        return Verifier(key=self._key(signer_name)).sign(body)

    # -- reading -----------------------------------------------------------------
    @property
    def entries(self) -> list[dict]:
        return list(self._entries)

    def of_kind(self, kind: str) -> list[dict]:
        return [entry for entry in self._entries if entry["kind"] == kind]

    def last_of_kind(self, kind: str) -> dict | None:
        found = self.of_kind(kind)
        return found[-1] if found else None

    def head(self) -> str:
        return self._entries[-1]["hash"] if self._entries else GENESIS

    def to_dict(self) -> dict:
        return {"log_version": LOG_VERSION, "entries": self.entries}

    # -- verification ------------------------------------------------------------
    def verify(self) -> dict:
        return verify_log(self.to_dict(), harness_key=self.harness_key,
                          verifier_key=self.verifier_key)


def verify_log(
    log: dict | SignedLog,
    *,
    harness_key: bytes | None = None,
    verifier_key: bytes | None = None,
) -> dict:
    """Verify the chain, every HMAC and every actor-bound verdict signature."""
    if isinstance(log, SignedLog):
        data = log.to_dict()
        harness_key = harness_key or log.harness_key
        verifier_key = verifier_key or log.verifier_key
    else:
        data = log
    problems: list[str] = []
    entries = list(data.get("entries") or [])
    if not harness_key or not verifier_key:
        problems.append("no key supplied; fail closed")
        return {"valid": False, "problems": problems, "entries": len(entries)}
    prev = GENESIS
    for expected_seq, entry in enumerate(entries, start=1):
        kind = str(entry.get("kind", ""))
        payload = entry.get("payload")
        if entry.get("seq") != expected_seq:
            problems.append(f"seq {entry.get('seq')!r} != {expected_seq}")
        if entry.get("prev_hash") != prev:
            problems.append(f"seq {expected_seq}: prev_hash breaks the chain")
        try:
            payload_json = canonical_json(payload)
        except Exception as exc:  # noqa: BLE001 - a malformed payload is simply invalid
            problems.append(f"seq {expected_seq}: payload is not canonical: {exc}")
            prev = str(entry.get("hash", ""))
            continue
        body = prev + "|" + kind + "|" + payload_json
        expected_hash = sha256_text(body)
        if entry.get("hash") != expected_hash:
            problems.append(f"seq {expected_seq}: hash mismatch")
        signer_name = str(entry.get("signer", _signer_for(kind, None)))
        key = verifier_key if signer_name == "verifier" else harness_key
        expected_sig = Verifier(key=key).sign(body)
        if not hmac.compare_digest(str(entry.get("signature", "")), expected_sig):
            problems.append(f"seq {expected_seq}: {signer_name} signature invalid")
        if kind == KIND_VERDICT:
            try:
                verdict = actor_verdict_from_dict(payload)
            except TypeError as exc:
                problems.append(f"seq {expected_seq}: malformed verdict: {exc}")
            else:
                if not actor_verdict_valid(verdict, verifier_key):
                    problems.append(f"seq {expected_seq}: verdict HMAC invalid")
        prev = str(entry.get("hash", ""))
    return {
        "valid": not problems,
        "problems": problems,
        "entries": len(entries),
        "head": prev,
    }


def entries_payload(entries: Iterable[dict], kind: str) -> list[dict]:
    return [entry["payload"] for entry in entries if entry["kind"] == kind]


@dataclass
class EpisodeRecordSet:
    """Convenience view over one verified episode log."""

    log: SignedLog
    state_bundle: dict = field(default_factory=dict)

    def facts_input(self) -> dict:
        return {"log": self.log.to_dict(), "state_bundle": self.state_bundle}
