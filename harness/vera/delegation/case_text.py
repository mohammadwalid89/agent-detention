"""Load and apply the v14 case-text overlay (M6).

The overlay (``manifests/delegation/V14_CASE_TEXT_SET12.json``, built by
``tools/delegation/build_v14_case_text.py``) gives every case a goal sentence and a plain label
for every signal.  It is bound to the set-12 bytes by sha256 and to itself by a digest; a
mismatch, a missing mechanism or a missing label is a hard error, never a fallback to raw ids.
"""

from __future__ import annotations

import dataclasses
import json
from functools import lru_cache
from pathlib import Path

from ..canonical import canonical_json, sha256_file, sha256_text

ROOT = Path(__file__).resolve().parents[2]
CASE_TEXT_PATH = ROOT / "manifests" / "delegation" / "V14_CASE_TEXT_SET12.json"
SOURCE_PATH = ROOT / "manifests" / "reliability" / "dev-active-set-12.json"
KIND = "vera.v14_case_text.v1"
STATUS_DRAFT = "DRAFT_PENDING_OWNER_REVIEW"
STATUS_SIGNED = "OWNER_SIGNED"


class CaseTextError(Exception):
    """The case-text overlay is missing, stale or incomplete."""


# Opt-in (v14b suite integration): which overlay and which source bytes ``load_case_text()`` binds to.
# The default is the set-12 pair above, byte for byte the old behaviour; ``select_overlay`` swaps in a
# suite overlay and its own source file for the life of the process, ``reset_overlay`` restores the default.
_ACTIVE = {"overlay": CASE_TEXT_PATH, "source": SOURCE_PATH}


def select_overlay(overlay_path: str | Path, source_path: str | Path) -> None:
    """Bind ``load_case_text()`` to ``overlay_path``, which must be bound to ``source_path``'s bytes."""
    _ACTIVE["overlay"], _ACTIVE["source"] = Path(overlay_path), Path(source_path)
    _load.cache_clear()


def reset_overlay() -> None:
    _ACTIVE["overlay"], _ACTIVE["source"] = CASE_TEXT_PATH, SOURCE_PATH
    _load.cache_clear()


@lru_cache(maxsize=4)
def _load(path_text: str, source_text: str) -> dict:
    path = Path(path_text)
    if not path.is_file():
        raise CaseTextError(f"case-text overlay is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("kind") != KIND:
        raise CaseTextError(f"unexpected case-text kind {data.get('kind')!r}")
    body = {key: value for key, value in data.items() if key != "digest"}
    if data.get("digest") != sha256_text(canonical_json(body)):
        raise CaseTextError("case-text overlay digest does not verify")
    if data["source"]["sha256"] != sha256_file(Path(source_text)):
        raise CaseTextError("case-text overlay was built for different source bytes")
    return data


def load_case_text(path: str | Path | None = None) -> dict:
    """The active overlay (default: set 12) unless an explicit ``path`` of the active source is given."""
    return _load(str(path or _ACTIVE["overlay"]), str(_ACTIVE["source"]))


def case_text_status(path: str | Path | None = None) -> str:
    """``DRAFT_PENDING_OWNER_REVIEW`` until the owner signs; the pilot gate checks this."""
    return str(load_case_text(path).get("status") or "")


def load_set12_instances() -> list:
    """The set-12 instances read from the set-12 bytes, whatever the owner-marker resolver says.

    ``instrument.resolve_source_set`` picks set 12 only when the owner-review marker verifies
    (it cannot in a sandbox that shows files under an unknown uid).  Protocol v14 is defined
    over the set-12 bytes, so it names them directly; the hash is checked against the overlay.
    """
    from .instrument import load_source_instances

    return load_source_instances({
        "source_set": "12",
        "path": str(SOURCE_PATH.relative_to(ROOT)),
        "sha256": sha256_file(SOURCE_PATH),
    })


def case_entry_for(instance, table: dict | None = None) -> dict:
    table = table or load_case_text()
    entry = (table.get("mechanisms") or {}).get(instance.mechanism_id)
    if entry is None:
        raise CaseTextError(f"no case text for mechanism {instance.mechanism_id!r}")
    recorded = str(entry.get("stg_hash") or "")
    actual = str(instance.mechanism.get("stg_hash") or "")
    if not recorded or recorded != actual:
        # The same mechanism id exists in sets 10, 11 and 12 with different content.
        raise CaseTextError(
            f"{instance.mechanism_id}: the overlay was written for a different revision "
            f"of this mechanism (graph hash {recorded[:12]!r} vs {actual[:12]!r})"
        )
    # Opt-in per-variant text (suite overlays: the two variants of a pair are different tasks with
    # different instructions).  Set-12 entries carry no ``variants`` key and are returned unchanged.
    per_variant = (entry.get("variants") or {}).get(instance.variant_id)
    if per_variant:
        entry = {**entry, **per_variant}
    missing =sorted(set(instance.registry) - set(entry.get("signal_labels") or {}))
    if missing:
        raise CaseTextError(f"{instance.mechanism_id}: no label for signal(s) {missing}")
    if not str(entry.get("goal_text") or "").strip():
        raise CaseTextError(f"{instance.mechanism_id}: empty goal text")
    return entry


def apply_case_text(instance, table: dict | None = None):
    """A copy of ``instance`` that carries its goal text and signal labels."""
    entry = case_entry_for(instance, table)
    return dataclasses.replace(
        instance,
        goal_text=str(entry["goal_text"]),
        signal_labels={str(k): str(v) for k, v in entry["signal_labels"].items()},
    )


def case_text_digest(instance) -> str:
    """A digest of exactly the text one instance was given (bound into its cell hash)."""
    return sha256_text(
        canonical_json({"goal": instance.goal_text, "labels": instance.signal_labels})
    )


__all__ = [
    "CASE_TEXT_PATH",
    "CaseTextError",
    "STATUS_DRAFT",
    "STATUS_SIGNED",
    "apply_case_text",
    "case_entry_for",
    "case_text_digest",
    "case_text_status",
    "load_case_text",
    "load_set12_instances",
    "reset_overlay",
    "select_overlay",
]
