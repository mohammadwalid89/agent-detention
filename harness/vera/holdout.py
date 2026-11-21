"""Holdout sealing and the W2 entry blocker (design §6.6, dissent D8).

The builder does not author the held-out families. This module only implements the
mechanics: seal, hash, ordering check, author separation, and a hard W2 entry refusal.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from . import HOLDOUT_FAMILIES
from .canonical import canonical_json, sha256_file, sha256_text
from .errors import HoldoutSealError

BUILDER_IDENTITIES = frozenset({"builder", "optimizer", "deepseek-builder-w1", "vera-api"})
SEAL_FILE = "SEAL.json"


@dataclass(frozen=True)
class SealStatus:
    status: str  # SEALED | AWAITING_SEPARATE_AUTHOR | INVALID
    detail: str
    author_id: str = ""
    sealed_manifest_sha256: str = ""
    sealed_at: int = 0

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "detail": self.detail,
            "author_id": self.author_id,
            "sealed_manifest_sha256": self.sealed_manifest_sha256,
            "sealed_at": self.sealed_at,
            "required_families": list(HOLDOUT_FAMILIES),
        }


def _walk_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def compute_manifest_hash(root: Path) -> str:
    files = [p for p in _walk_files(root) if p.name != SEAL_FILE]
    payload = {str(p.relative_to(root)): sha256_file(p) for p in files}
    return sha256_text(canonical_json(payload))


def seal(
    root: str | Path,
    *,
    author_id: str,
    identity: str = "",
    families: tuple[str, ...] = HOLDOUT_FAMILIES,
) -> SealStatus:
    """Seal an already-authored holdout directory. Never called by the W1 builder."""
    root = Path(root)
    ident = identity or os.environ.get("VERA_IDENTITY", "builder")
    if ident in BUILDER_IDENTITIES or author_id in BUILDER_IDENTITIES:
        raise HoldoutSealError(
            "the builder/optimizer identity may not author or seal the holdout families; "
            "a separate session/author is required"
        )
    if not root.exists():
        raise HoldoutSealError(f"holdout directory {root} does not exist")
    files = [p for p in _walk_files(root) if p.name != SEAL_FILE]
    if not files:
        raise HoldoutSealError("holdout directory is empty; nothing to seal")
    manifest = {str(p.relative_to(root)): sha256_file(p) for p in files}
    # Families must be present as directories so "two entire families" is checkable.
    present = {p.relative_to(root).parts[0] for p in files}
    missing = [f for f in families if f not in present]
    if missing:
        raise HoldoutSealError(f"sealed set missing families: {missing}")
    sealed_at = int(time.time())
    body = {
        "author_id": author_id,
        "sealed_at": sealed_at,
        "families": list(families),
        "files": manifest,
        "sealed_manifest_sha256": sha256_text(canonical_json(manifest)),
        "statement": (
            "I authored these mechanisms independently of the development set. No development "
            "manifest referenced them before this seal, and I did not consult development "
            "results before authoring them."
        ),
    }
    seal_path = root / SEAL_FILE
    seal_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    # make read-only; directories 0500, files 0444
    for p in files:
        os.chmod(p, 0o444)
    os.chmod(seal_path, 0o444)
    for d in sorted({p.parent for p in files}, key=lambda q: len(str(q)), reverse=True):
        try:
            os.chmod(d, 0o500)
        except PermissionError:
            pass
    return SealStatus(
        status="SEALED",
        detail=f"{len(files)} files sealed across {len(present)} families",
        author_id=author_id,
        sealed_manifest_sha256=body["sealed_manifest_sha256"],
        sealed_at=sealed_at,
    )


def seal_status(root: str | Path) -> SealStatus:
    root = Path(root)
    if not root.exists():
        return SealStatus("AWAITING_SEPARATE_AUTHOR", f"{root} does not exist")
    seal_path = root / SEAL_FILE
    if not seal_path.exists():
        return SealStatus("AWAITING_SEPARATE_AUTHOR", "no SEAL.json (families not authored yet)")
    body = json.loads(seal_path.read_text())
    actual = compute_manifest_hash(root)
    if actual != body.get("sealed_manifest_sha256"):
        return SealStatus("INVALID", "sealed content hash mismatch (files changed after seal)")
    if body.get("author_id") in BUILDER_IDENTITIES:
        return SealStatus("INVALID", "sealed by the builder identity")
    missing = [f for f in HOLDOUT_FAMILIES if f not in body.get("families", [])]
    if missing:
        return SealStatus("INVALID", f"sealed set does not cover families {missing}")
    return SealStatus(
        status="SEALED",
        detail=f"{len(body['files'])} files",
        author_id=body["author_id"],
        sealed_manifest_sha256=body["sealed_manifest_sha256"],
        sealed_at=int(body["sealed_at"]),
    )


def assert_can_read(root: str | Path, *, identity: str = "") -> None:
    """The builder and the learner/optimizer may not read the sealed directory."""
    ident = identity or os.environ.get("VERA_IDENTITY", "builder")
    if ident in BUILDER_IDENTITIES:
        raise HoldoutSealError(
            f"identity {ident!r} may not read {root}; sealed sets are grader-only"
        )


def verify_seal(
    holdout_root: str | Path,
    *,
    dev_manifest_path: str | Path,
    dev_author_ids: tuple[str, ...] = ("deepseek-builder-w1",),
) -> dict:
    """Full seal verification: hash, author separation, and seal-before-freeze ordering."""
    root = Path(holdout_root)
    st = seal_status(root)
    if st.status != "SEALED":
        raise HoldoutSealError(f"{st.status}: {st.detail}")
    if st.author_id in dev_author_ids:
        raise HoldoutSealError(
            f"holdout author {st.author_id!r} also authored a dev manifest (not independent)"
        )
    dev = Path(dev_manifest_path)
    if not dev.exists():
        raise HoldoutSealError(f"dev manifest {dev} missing; ordering cannot be proven")
    freeze_ts = int(dev.stat().st_mtime)
    if st.sealed_at > freeze_ts:
        raise HoldoutSealError(
            "seal timestamp is after the dev manifest freeze; holdout could have leaked into dev"
        )
    # Also record the ordering evidence so a reviewer can inspect it rather than trust it.
    return {
        "status": "SEALED",
        "author_id": st.author_id,
        "sealed_at": st.sealed_at,
        "dev_manifest_mtime": freeze_ts,
        "ordering_ok": st.sealed_at <= freeze_ts,
        "sealed_manifest_sha256": st.sealed_manifest_sha256,
    }


def assert_w2_entry(holdout_root: str | Path, *, dev_manifest_path: str | Path) -> dict:
    """Hard gate used by `make w2-entry`; raises instead of warning."""
    return verify_seal(holdout_root, dev_manifest_path=dev_manifest_path)
