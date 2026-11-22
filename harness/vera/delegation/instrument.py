"""RC1 / D1: hash-bound source-set selection and review-03 closure map.

The v1.3 instrument is derived only from the latest active set that has passed the independent
review.  The frozen D1 rule is: **set 11 if review 05 passed, else set 10**.  The builder
cannot manufacture that review, so the default resolution is set 10 unless a caller supplies
an affirmative review-05 pass (or writes the reviewed marker file).

Nothing in this module edits a mechanism, declaration, set manifest or readiness module.
It only reads them and records hashes so the derived condition manifest can be checked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..canonical import canonical_json, sha256_file, sha256_text
from ..errors import ManifestError
from ..pilot.instances import Instance, _instance_from_record

ROOT = Path(__file__).resolve().parents[2]
RELIABILITY = ROOT / "manifests" / "reliability"
SET_PATHS = {
    "10": RELIABILITY / "dev-active-set-10.json",
    "11": RELIABILITY / "dev-active-set-11.json",
    "12": RELIABILITY / "dev-active-set-12.json",
}
REVIEW_05_MARKER = ROOT / "artifacts" / "checkpoints" / "JOB24" / "ASTRA_W1_REVIEW_05_PASS.json"

# The attributable, hash-bound marker that admits set 12.  It is not a boolean: the
# resolver re-hashes the marker, both owner rationale files, the combined projection, every
# approval artifact and every current mechanism/sheet binding before it will select set 12.
SET12_REVIEW_MARKER = (ROOT / "artifacts" / "review" / "approvals" / "set12"
                       / "SET12_OWNER_REVIEW_PASS.json")
SET12_DECLARATIONS = RELIABILITY / "dev-abstention-declarations-12.json"
SET12_SHEETS = ROOT / "artifacts" / "review_sheets" / "set12"

# The L3 extension is a separate post-main-grid set.  Its attributable marker is not
# used by the main-grid source-set resolver, but the same marker consumer must be available
# so the extension cannot be admitted without re-validating its complete owner chain.
EXT_L3_REVIEW_MARKER = (ROOT / "artifacts" / "review" / "approvals" / "extL3_01"
                        / "EXT_L3_01_REVIEW_PASS.json")
EXT_L3_SEAL = (ROOT / "artifacts" / "review" / "approvals" / "extL3_01"
               / "EXT_L3_01_SEAL.json")
EXT_L3_MANIFEST = RELIABILITY / "dev-extension-L3-01.json"
_OWNER_MARKER_REVIEWER = "owner"
_OWNER_MARKER_TOKENS = (
    "PENDING", "UNREVIEWED", "TBD", "MODEL", "DEEPSEEK", "OPENAI", "GPT",
    "LLM", "ASSISTANT", "AI REVIEWER", "NOT HUMAN", "BUILDER",
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# The four review-03 obligations, and the bytes/artifacts that close them.  These are
# closure references, not a substitute for the independent review: the reviewer verifies them.
REVIEW03_CLOSURE = {
    "genuine_abstention": {
        "required": (
            "independent task rationale plus an executable positive disposition/safety "
            "contract; forbidden actions checked on abstention paths; contradictory "
            "declaration fails closed"
        ),
        "evidence": [
            "manifests/reliability/dev-active-set-11.json",
            "manifests/reliability/dev-abstention-declarations-11.json",
            "artifacts/checkpoints/JOB24/abstention_and_approval_probes.json",
            "artifacts/checkpoints/JOB24/semantic_audit_set11.json",
            "tests/test_job24.py::test_abstention_probe_set_all_fails_closed",
        ],
        "status": "EVIDENCE_PRESENT_PENDING_REVIEW_05",
    },
    "faithful_repairs_and_provenance": {
        "required": (
            "meaningful containment/escalation/evidence effects; no artificial dead ends; "
            "no read/wait/escalation manufacturing a prerequisite; whole-path and "
            "intermediate-invariant checks"
        ),
        "evidence": [
            "artifacts/checkpoints/JOB24/semantic_audit_set10.json",
            "artifacts/checkpoints/JOB24/semantic_audit_set11.json",
            "artifacts/checkpoints/JOB24/review03_probes.json",
            "artifacts/checkpoints/JOB24/review04_probes.json",
            "tests/test_job24.py::test_escalation_may_not_write_a_registered_gate",
            "tests/test_job24.py::test_escalation_may_not_create_approval_authority",
        ],
        "status": "EVIDENCE_PRESENT_PENDING_REVIEW_05",
    },
    "valid_freeze": {
        "required": (
            "attributable affirmative human approval bound to actual mechanism/contract "
            "bytes; non-approval, wrong-task and changed-task artifacts fail"
        ),
        "evidence": [
            "artifacts/checkpoints/JOB24/abstention_and_approval_probes.json",
            "tests/test_job24.py::test_astra_approval_probes_with_positive_and_negative_controls",
            "tests/test_job24.py::test_recorder_refuses_a_sheet_that_is_not_the_current_mechanism",
        ],
        "status": "EVIDENCE_PRESENT_PENDING_REVIEW_05",
    },
    "reproducible_acceptance": {
        "required": (
            "one preserved runnable bundle, versioned manifest and independent checks on "
            "the same bytes; a projected manifest or mixed live tree is not acceptance"
        ),
        "evidence": [
            "artifacts/checkpoints/W1_PACKET_05/PACKET_MANIFEST.sha256",
            "artifacts/checkpoints/W1_PACKET_05/run_acceptance.sh",
            "artifacts/checkpoints/W1_PACKET_05/ACCEPTANCE.log",
        ],
        "status": "EVIDENCE_PRESENT_PENDING_REVIEW_05",
    },
}


def _marker_pass() -> bool:
    """A boolean/environment assertion is not a review.

    Set 11 is admissible only when an attributable PASS artifact exists, names the reviewer
    and time, and is bound to the exact set-11 bytes by ``source_set_sha256``.  There is no
    filename-only or caller-boolean bypass.
    """
    if not REVIEW_05_MARKER.is_file() or not SET_PATHS["11"].is_file():
        return False
    try:
        data = json.loads(REVIEW_05_MARKER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        set11_sha = sha256_file(SET_PATHS["11"])
    except OSError:
        return False
    bound_hash = str(
        data.get("source_set_sha256")
        or data.get("set11_sha256")
        or data.get("source_sha256")
        or ""
    )
    return bool(
        str(data.get("review", "")).lower() in {"05", "w1_review_05", "review_05"}
        and str(data.get("verdict", "")).upper() == "PASS"
        and str(data.get("source_set", "")) == "11"
        and bound_hash == set11_sha
        and str(data.get("reviewer", "")).strip()
        and str(data.get("reviewed_at", "")).strip()
    )


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def _set12_marker_pass() -> bool:
    """Re-verify the full owner-review chain before admitting set 12.

    Everything here is a hash or digest check against the *current* bytes: no caller boolean,
    environment flag or filename-only marker can select set 12.  The marker must bind the
    set-12 manifest, both owner files, both rationale files, the combined projection, all 45
    approval artifacts and every reviewed mechanism's current contract/sheet binding.
    """
    if not SET12_REVIEW_MARKER.is_file() or not SET_PATHS["12"].is_file():
        return False
    try:
        data = json.loads(SET12_REVIEW_MARKER.read_text(encoding="utf-8"))
        set12_sha = sha256_file(SET_PATHS["12"])
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    if data.get("schema") != "vera.set_review_marker.v1":
        return False
    if data.get("kind") != "owner_page_fully_reviewed_set":
        return False
    if str(data.get("set")) != "12" or str(data.get("source_set")) != "12":
        return False
    if str(data.get("verdict", "")).upper() != "PASS":
        return False
    if str(data.get("source_set_sha256") or "") != set12_sha:
        return False
    reviewer = str(data.get("reviewer") or "").strip()
    if reviewer != _OWNER_MARKER_REVIEWER:
        return False
    if any(token in reviewer.upper() for token in _OWNER_MARKER_TOKENS):
        return False
    if not str(data.get("reviewed_at") or "").strip():
        return False

    # 1. Both owner files, with their sha256s.  A still-mounted file must hash to the record.
    owner_files = data.get("owner_files")
    if not isinstance(owner_files, list) or len(owner_files) < 2:
        return False
    owner_hashes: set[str] = set()
    owner_roles: set[str] = set()
    decisions = 0
    for item in owner_files:
        if not isinstance(item, dict):
            return False
        path = str(item.get("path") or "").strip()
        digest = str(item.get("sha256") or "").strip().lower()
        if not path or not _HEX64.fullmatch(digest):
            return False
        if Path(path).is_file() and sha256_file(Path(path)) != digest:
            return False
        role = str(item.get("role") or "").strip()
        if role not in {"owner_page_attestation", "owner_page_carryover"}:
            return False
        count = item.get("decisions")
        if not isinstance(count, int) or count <= 0:
            return False
        owner_hashes.add(digest)
        owner_roles.add(role)
        decisions += count
    if not {"owner_page_attestation", "owner_page_carryover"} <= owner_roles:
        return False
    if decisions != 45:
        return False

    # 2. Both rationale files re-bind to the current set-12 manifest and to an owner file.
    rationales = data.get("rationale_reviews")
    if not isinstance(rationales, list) or len(rationales) != 2:
        return False
    merged_reviews: dict[str, dict] = {}
    for item in rationales:
        if not isinstance(item, dict):
            return False
        path = Path(str(item.get("path") or ""))
        digest = str(item.get("sha256") or "").strip().lower()
        if not path.is_file() or not _HEX64.fullmatch(digest):
            return False
        if sha256_file(path) != digest:
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if payload.get("schema") != "vera.rationale_review.v1":
            return False
        if payload.get("set") != "set12":
            return False
        if payload.get("manifest_sha256") != set12_sha:
            return False
        if payload.get("reviewer") != _OWNER_MARKER_REVIEWER:
            return False
        if str((payload.get("source") or {}).get("sha256") or "") not in owner_hashes:
            return False
        mechanisms = payload.get("mechanisms")
        if not isinstance(mechanisms, dict) or not mechanisms:
            return False
        for mid, entries in mechanisms.items():
            if mid in merged_reviews:
                return False
            if not isinstance(entries, dict):
                return False
            merged_reviews[mid] = entries
    if len(merged_reviews) != 45:
        return False

    # 3. The combined projection's hash, counts, content failures and approvals.
    combined = data.get("combined_projection")
    if not isinstance(combined, dict):
        return False
    projection_path = Path(str(combined.get("path") or ""))
    projection_sha = str(combined.get("sha256") or "").strip().lower()
    if not projection_path.is_file() or not _HEX64.fullmatch(projection_sha):
        return False
    if sha256_file(projection_path) != projection_sha:
        return False
    try:
        projection = json.loads(projection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if projection.get("schema") != "vera.combined_owner_review_readiness_projection.v1":
        return False
    if projection.get("set") != "set12":
        return False
    if projection.get("manifest_sha256") != set12_sha:
        return False
    counts = projection.get("counts") or {}
    if counts.get("mechanisms") != 48 or counts.get("ready") != 48:
        return False
    if counts.get("pending") != 0 or counts.get("ready_without_owner_approval") != 3:
        return False
    if projection.get("matches_expected") is not True:
        return False
    evidence = projection.get("combined") or {}
    if evidence.get("duplicate_mechanism_ids") != []:
        return False
    if (evidence.get("content_failures") or {}).get("count") != 0:
        return False
    if evidence.get("reviewed_mechanisms") != 45:
        return False
    if sorted(evidence.get("reviewed_mechanism_ids") or []) != sorted(merged_reviews):
        return False
    approvals = evidence.get("approvals") or {}
    if approvals.get("count") != 45:
        return False
    artifacts = approvals.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 45:
        return False
    if approvals.get("digest") != sha256_text(canonical_json(artifacts)):
        return False
    rows = projection.get("mechanisms") or {}
    if len(rows) != 48:
        return False
    actual_files = {str(p) for p in projection_path.parent.glob("*.approval.json")}
    listed_files = {str(a.get("artifact")) for a in artifacts}
    if actual_files != listed_files:
        return False
    marker_approvals = data.get("approvals") or {}
    if marker_approvals.get("count") != approvals.get("count"):
        return False
    if marker_approvals.get("digest") != approvals.get("digest"):
        return False
    if marker_approvals.get("artifacts") != artifacts:
        return False
    reported = data.get("combined_projection") or {}
    if reported.get("counts") != counts:
        return False
    if reported.get("content_failures") != 0:
        return False
    if reported.get("duplicate_mechanism_ids") != []:
        return False
    # Review-06 finding 2: every artifact goes through the ONE complete approval-chain
    # validator (protected source decision + canonical Markdown + all prior checks).  There is
    # no hand-maintained subset of approval checks in this resolver.
    try:
        from ..mechanisms.approval import validate_approval_artifact

        manifest_records = {m["mechanism_id"]: m for m in
                            json.loads(SET_PATHS["12"].read_text(encoding="utf-8"))
                            .get("mechanisms") or []}
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False
    for artifact in artifacts:
        artifact_path = Path(str(artifact.get("artifact") or ""))
        if not artifact_path.is_file():
            return False
        if sha256_file(artifact_path) != str(artifact.get("artifact_sha256") or ""):
            return False
        mid = str(artifact.get("mechanism_id") or "")
        row = rows.get(mid) or {}
        if row.get("approval_id") != artifact.get("approval_id"):
            return False
        if row.get("contract_sha256") != artifact.get("contract_sha256"):
            return False
        if row.get("sheet_binding_sha256") != artifact.get("binding_sha256"):
            return False
        if row.get("approval_artifact_sha256") != artifact.get("artifact_sha256"):
            return False
        try:
            body = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        record = manifest_records.get(mid)
        if record is None:
            return False
        try:
            validate_approval_artifact(body, record, label=artifact_path.name, root=ROOT)
        except (ManifestError, KeyError, TypeError, ValueError):
            return False

    # 4. Re-derive every mechanism's current contract/sheet binding and readiness from the
    #    current set-12 bytes and the merged rationale.  A projection cannot outlive its set.
    try:
        from ..mechanisms.approval import contract_digest, sheet_binding_digest
        from ..mechanisms.readiness_v3 import (
            load_declarations,
            validate_ready_mechanism_v3,
        )

        manifest = json.loads(SET_PATHS["12"].read_text(encoding="utf-8"))
        records = {m["mechanism_id"]: m for m in manifest.get("mechanisms") or []}
        declarations = load_declarations(SET12_DECLARATIONS)
        if set(rows) != set(records):
            return False
        for mid, record in records.items():
            row = rows.get(mid) or {}
            if row.get("contract_sha256") != contract_digest(record):
                return False
            if row.get("sheet_binding_sha256") != sheet_binding_digest(
                    record, declarations=declarations):
                return False
            result = validate_ready_mechanism_v3(
                record, declarations=declarations, rationale_reviews=merged_reviews)
            if not result.get("benchmark_ready"):
                return False
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    return True


def _ext_l3_marker_pass() -> bool:
    """Consume the extL3_01 marker only through the complete owner-chain verifier.

    This does not select a main-grid source set; it answers whether the extension set's own
    attributable marker still binds the current extension manifest, owner file, rationale,
    combined projection and every approval artifact.
    """
    try:
        from tools.review.import_owner_attestation import verify_extension_review_marker
    except Exception:
        return False
    try:
        return bool(verify_extension_review_marker(EXT_L3_REVIEW_MARKER).get("valid"))
    except Exception:
        return False


def resolve_source_set(review05_passed: bool | None = None) -> dict:
    """Resolve D1's source set and bind its bytes.

    ``review05_passed`` is retained for callers' provenance assertions but cannot select a
    set.  Set 12 requires the attributable, hash-bound owner-review marker; set 11 requires
    the attributable review-05 PASS artifact bound to the set-11 hash; absent either, the
    rule selects set 10.
    """
    marker_ok = _marker_pass()
    set12_ok = _set12_marker_pass()
    selected = "12" if set12_ok else ("11" if marker_ok else "10")
    path = SET_PATHS[selected]
    if not path.is_file():
        raise FileNotFoundError(f"source set {selected} is missing: {path}")
    return {
        "source_set": selected,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "review_05_passed": marker_ok,
        "review_05_asserted": review05_passed,
        "review_05_marker_valid": marker_ok,
        "review_05_marker": _display_path(REVIEW_05_MARKER),
        "owner_review_passed": set12_ok,
        "owner_review_marker_valid": set12_ok,
        "owner_review_marker": _display_path(SET12_REVIEW_MARKER),
        "rule": (
            "set 12 only when an attributable owner-review marker is bound to the set-12 "
            "hash and its combined projection re-verifies (48/48, 0 content failures); "
            "otherwise set 11 only when an attributable W1 review-05 PASS artifact is bound "
            "to the set-11 hash; otherwise set 10"
        ),
        "set10_sha256": sha256_file(SET_PATHS["10"]),
        "set11_sha256": sha256_file(SET_PATHS["11"]) if SET_PATHS["11"].is_file() else "",
        "set12_sha256": sha256_file(SET_PATHS["12"]) if SET_PATHS["12"].is_file() else "",
        "closure_map": REVIEW03_CLOSURE,
        "closure_digest": sha256_text(canonical_json(REVIEW03_CLOSURE)),
    }


def load_source_mechanisms(source: dict | None = None) -> list[dict]:
    source = source or resolve_source_set()
    data = json.loads((ROOT / source["path"]).read_text(encoding="utf-8"))
    mechanisms = data.get("mechanisms")
    if not isinstance(mechanisms, list) or not mechanisms:
        raise ValueError(f"source set {source['path']} has no mechanisms")
    return mechanisms


def load_source_instances(source: dict | None = None) -> list[Instance]:
    return [
        _instance_from_record(mechanism, variant)
        for mechanism in load_source_mechanisms(source)
        for variant in mechanism.get("variants", [])
    ]


def source_identity(source: dict | None = None) -> dict:
    source = source or resolve_source_set()
    mechanisms = load_source_mechanisms(source)
    instances = [
        instance for mechanism in mechanisms for instance in [mechanism]
    ]
    n_instances = sum(len(mechanism.get("variants", [])) for mechanism in mechanisms)
    return {
        **source,
        "mechanisms": len(mechanisms),
        "instances": n_instances,
        "set10_unchanged_from_job24": (
            source["set10_sha256"]
            == "f988101054f65522dcda9cce863dd8edc08a66e9a14a7032e834d4b94b2afb2f"
        ),
        "no_set09_counts_reused": True,
        "mechanism_slots": len(instances),
    }
