"""JOB55: bind the runner to the one validated, frozen source and assignment.

Astra's final review finding 1 (``REVIEW_V13B_FINAL.md``) showed that ``GridRunner`` executed
set-10 instances while the D1 resolver selected the owner-reviewed set 12.  This module makes
that impossible:

* the active source is resolved only through :func:`instrument.resolve_source_set` (set 12
  requires the attributable, hash-bound owner-review marker; there is no caller boolean),
* the resolved source hash must equal the hash recorded in the passed preregistration root
  manifest (``source_set`` row and ``source_and_instrument`` row),
* the condition manifest is rebuilt from those bytes and its ``manifest_digest`` must equal
  the preregistered ``condition_manifest_digest``,
* the full main assignment table is rebuilt from that manifest and its digest must equal the
  frozen set-12 assignment record (whose digest is anchored to pre-JOB55 dry-run evidence).

Nothing here edits a mechanism, declaration, condition manifest or preregistration row.  A
fallback to set 10/11, an omitted pattern level, a partial table or a stale digest is a hard
refusal, never a warning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from ..canonical import canonical_json, sha256_file, sha256_text
from .instrument import load_source_instances, resolve_source_set

ROOT = Path(__file__).resolve().parents[2]
PREREG_ROOT = ROOT / "manifests" / "delegation" / "V13_PREREGISTRATION" / "ROOT_MANIFEST.json"
CONDITION_MANIFEST_SET12 = ROOT / "manifests" / "delegation" / "V13_CONDITIONS_SET12.json"
FROZEN_ASSIGNMENT = ROOT / "manifests" / "delegation" / "V13_ASSIGNMENT_SET12.json"

MAIN_SOURCE_SET = "12"
# The preregistration stage names that must already be complete before the first main
# episode.  ``AFTER_INITIAL_GRID_BEFORE_DETENTION_RETRY`` is deliberately not in this set.
PRE_MAIN_STAGES = (
    "FREEZE_BEFORE_MAIN_RUN",
    "ROUTE_PREFLIGHT_BEFORE_FIRST_CALL",
    "CALIBRATION_BEFORE_FIRST_MAIN_EPISODE",
)
COMPLETE_STATUSES = {"FROZEN_IN_SKELETON", "FROZEN", "COMPLETE", "COMPLETED", "SEALED"}
PENDING_STATUSES = {"MECHANICAL_LATER_STAGE", "PENDING", "DEFERRED"}
ASSIGNMENT_HASH_FIELDS = ("assignment_digest", "exact_rows_sha256", "assignment_table_sha256")
_HEX64 = set("0123456789abcdef")


class SourceBindingError(Exception):
    """The runner is not bound to the validated frozen source/assignment."""


def rows_digest(rows: Sequence[Mapping]) -> str:
    """The runner's assignment digest (same encoding as ``runner_v13.json_digest``)."""
    return sha256_text(json.dumps([dict(row) for row in rows], sort_keys=True, default=str))


def _hex64(value: object) -> str:
    text = str(value or "").strip().lower()
    if len(text) == 64 and set(text) <= _HEX64:
        return text
    return ""


def _problems_message(problems: Sequence[str]) -> str:
    return "; ".join(problems)


def load_preregistration_root(path: str | Path | None = None) -> dict:
    """Load and hash-verify a preregistration root manifest."""
    target = Path(path) if path is not None else PREREG_ROOT
    if not target.is_file():
        raise SourceBindingError(f"preregistration root manifest is missing: {target}")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceBindingError(
            f"preregistration root manifest is unreadable: {exc}"
        ) from None
    if not isinstance(data, dict):
        raise SourceBindingError("preregistration root manifest must be a JSON object")
    body = {key: value for key, value in data.items() if key != "root_sha256"}
    calculated = sha256_text(canonical_json(body))
    recorded = str(data.get("root_sha256") or "").strip().lower()
    if recorded != calculated:
        raise SourceBindingError(
            "preregistration root_sha256 does not verify against its own rows"
        )
    return {"path": str(target), "data": data, "root_sha256": calculated}


_MANIFEST_CACHE: dict[tuple[str, str], str] = {}


def build_condition_manifest_for(source: Mapping) -> dict:
    """Build the condition manifest for exactly these source bytes.

    The cache key is the source content hash and set label, so a changed source can never
    reuse a stale manifest.  ``source`` must be the complete dictionary returned by
    :func:`instrument.resolve_source_set`: the manifest records that identity verbatim, and
    the preregistered digest is over those exact bytes.
    """
    from .condition_manifest import build_condition_manifest

    path = str(source.get("path") or "")
    digest = str(source.get("sha256") or "").strip().lower()
    source_set = str(source.get("source_set") or "")
    if not path or not _hex64(digest) or not source_set:
        raise SourceBindingError("source binding has no path/sha256/source_set")
    key = (source_set, digest)
    cached = _MANIFEST_CACHE.get(key)
    if cached is None:
        manifest = build_condition_manifest(source=dict(source))
        if str(manifest.get("manifest_digest") or "") == "":
            raise SourceBindingError("condition manifest has no manifest_digest")
        cached = canonical_json(manifest)
        _MANIFEST_CACHE[key] = cached
    return json.loads(cached)


def _row_payload(prereg: Mapping, name: str) -> dict:
    """Load and hash-verify one preregistration row file."""
    root = prereg["data"]
    record = (root.get("rows") or {}).get(name)
    if not isinstance(record, Mapping):
        raise SourceBindingError(f"preregistration has no {name!r} row")
    relative = str(record.get("file") or "")
    if not relative:
        raise SourceBindingError(f"preregistration row {name!r} names no file")
    path = Path(prereg["path"]).parent / relative
    if not path.is_file():
        raise SourceBindingError(f"preregistration row file is missing: {path}")
    if sha256_file(path) != _hex64(record.get("sha256")):
        raise SourceBindingError(f"preregistration row file hash does not verify: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceBindingError(
            f"preregistration row file is unreadable: {path}: {exc}"
        ) from None
    if not isinstance(payload, dict):
        raise SourceBindingError(f"preregistration row file is not an object: {path}")
    return payload


def resolve_source_binding(
    *,
    prereg_path: str | Path | None = None,
    require_set12: bool = True,
) -> dict:
    """Resolve the active source and prove it equals the preregistered set-12 bytes."""
    source = resolve_source_set()
    prereg = load_preregistration_root(prereg_path)
    root = prereg["data"]
    problems: list[str] = []

    selected = str(source.get("source_set") or "")
    if require_set12 and selected != MAIN_SOURCE_SET:
        problems.append(
            f"active source resolved to set {selected!r} instead of the owner-reviewed "
            f"set {MAIN_SOURCE_SET}; source fallback is refused"
        )
    if require_set12 and source.get("owner_review_passed") is not True:
        problems.append("active source was not admitted by the attributable owner-review marker")
    if require_set12 and source.get("owner_review_marker_valid") is not True:
        problems.append("owner-review marker did not re-verify its complete owner chain")
    if not str(source.get("sha256") or "").strip():
        problems.append("active source has no content hash")
    if not (ROOT / str(source.get("path") or "")).is_file():
        problems.append(f"active source file is missing: {source.get('path')!r}")

    prereg_source = root.get("source_set") or {}
    if str(prereg_source.get("source_set") or "") != str(source.get("source_set") or ""):
        problems.append(
            "preregistration binds source set "
            f"{prereg_source.get('source_set')!r}, active source is {source.get('source_set')!r}"
        )
    if _hex64(prereg_source.get("sha256")) != _hex64(source.get("sha256")):
        problems.append(
            "preregistration source hash "
            f"{prereg_source.get('sha256')!r} does not equal the active source hash "
            f"{source.get('sha256')!r}"
        )
    source_row = _row_payload(prereg, "source_and_instrument").get("content") or {}
    row_source = source_row.get("source_set") or {}
    if _hex64(row_source.get("sha256")) != _hex64(source.get("sha256")):
        problems.append(
            "preregistration rows/source_and_instrument does not bind the active source hash"
        )
    if str(row_source.get("source_set") or "") != str(source.get("source_set") or ""):
        problems.append(
            "preregistration rows/source_and_instrument binds a different source set"
        )
    if _hex64(source_row.get("condition_manifest_digest")) != _hex64(
        root.get("condition_manifest_digest")
    ):
        problems.append(
            "preregistration rows/source_and_instrument condition digest disagrees with the root"
        )

    manifest: dict | None = None
    if not problems:
        manifest = build_condition_manifest_for(source)
        expected_digest = str(root.get("condition_manifest_digest") or "").strip().lower()
        actual_digest = str(manifest.get("manifest_digest") or "").strip().lower()
        if _hex64(actual_digest) != _hex64(expected_digest):
            problems.append(
                f"condition manifest digest {actual_digest!r} does not equal the "
                f"preregistered digest {expected_digest!r}"
            )
        manifest_source = manifest.get("source") or {}
        if _hex64(manifest_source.get("sha256")) != _hex64(source.get("sha256")):
            problems.append("condition manifest records a different source hash")
        counts = manifest.get("counts") or {}
        if counts.get("instances") != 96 or counts.get("solvable_candidates") != 51:
            problems.append(
                f"condition manifest instance counts are not the frozen set-12 counts: "
                f"{counts.get('instances')}/{counts.get('solvable_candidates')}"
            )

    if problems:
        raise SourceBindingError(
            "runner refused: source binding failed: " + _problems_message(problems)
        )
    assert manifest is not None
    return {
        "kind": "vera.v13.source_binding.v1",
        "source": dict(source),
        "condition_manifest": manifest,
        "condition_manifest_digest": manifest["manifest_digest"],
        "preregistration": {
            "path": prereg["path"],
            "root_sha256": prereg["root_sha256"],
            "version_id": str(root.get("version_id") or ""),
            "run_plan_version_id": str(root.get("run_plan_version_id") or ""),
        },
        "instance_ids": sorted(
            {instance.instance_id for instance in load_source_instances(source)}
        ),
    }


# Frozen main assignment


def _assignment_hash_from_prereg(prereg: Mapping) -> tuple[str, str]:
    """Return ``(digest, problem)`` for the preregistered exact-rows hash."""
    content = (_row_payload(prereg, "assignment_table").get("content")) or {}
    for field in ASSIGNMENT_HASH_FIELDS:
        value = content.get(field)
        digest = ""
        if isinstance(value, Mapping):
            digest = _hex64(value.get("sha256") or value.get("digest"))
        else:
            digest = _hex64(value)
        if digest:
            return digest, ""
    exact = content.get("exact_rows")
    if isinstance(exact, str) and exact.strip().upper().startswith("MECHANICAL_LATER_STAGE"):
        return "", "the preregistration exact assignment rows are still mechanical later stage"
    return "", "the preregistration records no exact assignment-table hash"


def preregistered_assignment_hash(
    prereg_path: str | Path | None = None,
) -> tuple[str, str]:
    """Public accessor: ``(digest, problem)`` for the preregistered exact rows."""
    return _assignment_hash_from_prereg(load_preregistration_root(prereg_path))


def load_frozen_assignment(path: str | Path | None = None) -> dict:
    """Load the frozen set-12 assignment record and verify its own bindings."""
    target = Path(path) if path is not None else FROZEN_ASSIGNMENT
    if not target.is_file():
        raise SourceBindingError(f"frozen set-12 assignment record is missing: {target}")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceBindingError(f"frozen assignment record is unreadable: {exc}") from None
    if not isinstance(data, dict):
        raise SourceBindingError("frozen assignment record must be a JSON object")
    if data.get("kind") != "vera.v13_frozen_assignment.v1":
        raise SourceBindingError(
            f"frozen assignment record has the wrong kind {data.get('kind')!r}"
        )
    digest = _hex64(data.get("assignment_digest"))
    if not digest:
        raise SourceBindingError("frozen assignment record has no assignment_digest")
    source = data.get("source") or {}
    if str(source.get("source_set") or "") != MAIN_SOURCE_SET:
        raise SourceBindingError("frozen assignment record does not bind set 12")
    if not _hex64(source.get("sha256")):
        raise SourceBindingError("frozen assignment record has no source hash")
    manifest = data.get("condition_manifest") or {}
    if _hex64(manifest.get("digest")) != _hex64(data.get("condition_manifest_digest")):
        raise SourceBindingError(
            "frozen assignment record condition-manifest digest is inconsistent"
        )
    manifest_path = ROOT / str(manifest.get("path") or "")
    if not manifest_path.is_file():
        raise SourceBindingError(
            f"frozen assignment record names a missing condition manifest: {manifest_path}"
        )
    if sha256_file(manifest_path) != _hex64(manifest.get("sha256")):
        raise SourceBindingError(
            "the condition manifest on disk does not match the frozen assignment record"
        )
    evidence = data.get("evidence") or {}
    if evidence:
        evidence_path = ROOT / str(evidence.get("path") or "")
        if not evidence_path.is_file():
            raise SourceBindingError(
                f"frozen assignment evidence is missing: {evidence_path}"
            )
        if sha256_file(evidence_path) != _hex64(evidence.get("sha256")):
            raise SourceBindingError("frozen assignment evidence hash does not verify")
        if _hex64(evidence.get("assignment_digest")) != digest:
            raise SourceBindingError(
                "frozen assignment digest disagrees with its recorded pre-JOB55 evidence"
            )
    return {**data, "path": str(target), "file_sha256": sha256_file(target)}


def main_assignment_requirements() -> dict:
    """The frozen full-plan shape every main-grid assignment must have."""
    from .runplan_v13 import ENROLLED_PATTERNS

    record = load_frozen_assignment()
    generator = record.get("generator") or {}
    return {
        "record": record,
        "patterns": list(generator.get("patterns") or ENROLLED_PATTERNS),
        "repeats": int(generator.get("repeats") or 0),
        "row_count": int(record.get("row_count") or 0),
        "counts_by_condition": dict(record.get("counts_by_condition") or {}),
        "assignment_digest": record["assignment_digest"],
    }


def verify_main_assignment(
    rows: Sequence[Mapping],
    binding: Mapping,
    *,
    patterns: Sequence[str] | None,
    repeats: int | None,
) -> dict:
    """Prove the runner's exact rows are the frozen full set-12 assignment.

    Any omitted pattern level, partial row list or changed row fails closed.
    """
    from .runplan_v13 import ENROLLED_PATTERNS

    requirements = main_assignment_requirements()
    record = requirements["record"]
    problems: list[str] = []
    actual_digest = rows_digest(rows)
    expected = requirements["assignment_digest"]

    source = binding.get("source") or {}
    if _hex64(source.get("sha256")) != _hex64(record["source"]["sha256"]):
        problems.append("frozen assignment source hash does not equal the bound source")
    if _hex64(binding.get("condition_manifest_digest")) != _hex64(
        record["condition_manifest_digest"]
    ):
        problems.append(
            "frozen assignment condition-manifest digest does not equal the bound manifest"
        )
    if patterns is not None:
        chosen = [str(item) for item in patterns]
        expected_patterns = list(
            (record.get("generator") or {}).get("patterns") or ENROLLED_PATTERNS
        )
        if sorted(chosen) != sorted(expected_patterns):
            missing = sorted(set(expected_patterns) - set(chosen))
            problems.append(
                "main_grid requires the complete frozen pattern cross "
                f"{expected_patterns}; missing {missing or 'none'}"
            )
    if repeats is not None and int(repeats) != requirements["repeats"]:
        problems.append(
            f"main_grid requires repeats={requirements['repeats']}, got {repeats!r}"
        )
    if len(rows) != requirements["row_count"]:
        problems.append(
            f"main_grid requires {requirements['row_count']} assignment rows, got {len(rows)}"
        )
    counts: dict[str, int] = {}
    for row in rows:
        condition = str(row.get("condition") or "")
        counts[condition] = counts.get(condition, 0) + 1
    expected_counts = requirements["counts_by_condition"]
    if counts != expected_counts:
        problems.append(
            "main_grid per-condition counts do not equal the frozen assignment: "
            f"{json.dumps(counts, sort_keys=True)}"
        )
    if actual_digest != expected:
        problems.append(
            f"assignment digest {actual_digest} does not equal the frozen set-12 digest "
            f"{expected}"
        )
    missing_ids = sorted(
        {
            str(row.get("instance_id"))
            for row in rows
            if str(row.get("instance_id")) not in binding.get("instance_ids", set())
        }
    )
    if binding.get("instance_ids") and missing_ids:
        problems.append(
            "assignment rows name instances absent from the bound source: "
            + ",".join(missing_ids[:5])
        )
    prereg = binding.get("preregistration") or {}
    if _hex64(prereg.get("root_sha256")) != _hex64(record.get("prereg_root_sha256")):
        problems.append(
            "frozen assignment was generated against a different preregistration root"
        )
    prereg_digest, prereg_problem = _assignment_hash_from_prereg(
        load_preregistration_root(prereg.get("path"))
    )
    if not prereg_digest:
        problems.append(f"preregistered assignment hash unavailable: {prereg_problem}")
    elif prereg_digest != expected:
        problems.append(
            f"preregistered assignment hash {prereg_digest} does not equal the frozen digest "
            f"{expected}"
        )
    if problems:
        raise SourceBindingError(
            "runner refused: main assignment binding failed: " + _problems_message(problems)
        )
    return {
        "kind": "vera.v13.assignment_binding.v1",
        "assignment_digest": expected,
        "row_count": requirements["row_count"],
        "counts_by_condition": expected_counts,
        "patterns": requirements["patterns"],
        "repeats": requirements["repeats"],
        "frozen_record": record["path"],
        "frozen_record_sha256": record["file_sha256"],
        "preregistered_assignment_hash": prereg_digest,
    }


__all__ = [
    "ASSIGNMENT_HASH_FIELDS",
    "COMPLETE_STATUSES",
    "CONDITION_MANIFEST_SET12",
    "FROZEN_ASSIGNMENT",
    "MAIN_SOURCE_SET",
    "PENDING_STATUSES",
    "PRE_MAIN_STAGES",
    "PREREG_ROOT",
    "SourceBindingError",
    "build_condition_manifest_for",
    "load_frozen_assignment",
    "load_preregistration_root",
    "main_assignment_requirements",
    "preregistered_assignment_hash",
    "resolve_source_binding",
    "rows_digest",
    "verify_main_assignment",
]
