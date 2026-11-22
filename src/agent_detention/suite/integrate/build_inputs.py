"""Build the runner inputs for the suite: source file, v14 conditions manifest, case-text overlay.

    python -m agent_detention.suite.integrate.build_inputs            # write into harness/manifests/suite/ (git-ignored)
    python -m agent_detention.suite.integrate.build_inputs --check    # fail if any file differs from a rebuild

Inputs: ``$AD_WORK/tier1/tier1_set.json`` (chapter 03, ``build_tier1_records.py``) and the Tier 2 records and
overlay in ``$AD_WORK/tier2``.  Output (inside the harness tree):

* ``SUITE_SOURCE_V14B.json``     the mechanism records the runner loads (``{"mechanisms": [...]}``, set-file shape)
* ``V14_CONDITIONS_SUITE.json``  the v14 condition manifest, built by the tree's own ``build_condition_manifest``
* ``V14_CASE_TEXT_SUITE.json``   the case-text overlay (goal text per variant, a plain label per signal)

No model call, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_detention import paths
from agent_detention.suite.tier1 import harness

TREE = harness.use_tree()
OUT_DIR = TREE / "manifests" / "suite"
SOURCE_REL = "manifests/suite/SUITE_SOURCE_V14B.json"
CONDITIONS_REL = "manifests/suite/V14_CONDITIONS_SUITE.json"
OVERLAY_REL = "manifests/suite/V14_CASE_TEXT_SUITE.json"

AUTHORITY = "owner waiver 2026-11-17 21:08 PT: public benchmark case text, conditional on COMPAT_PASS"
OVERLAY_STATUS = "OWNER_WAIVER_CONDITIONAL_ON_COMPAT_PASS"
SOURCE_SET = "suite-v14b"


def dump(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def tier1_label(predicate: str) -> str:
    if predicate.startswith("tool_available:"):
        return f"critical tool {predicate.split(':', 1)[1]} is working"
    raise ValueError(f"no label rule for Tier 1 predicate {predicate!r}")


def tier1_entry(record: dict) -> dict:
    registry = record["entry_state"]["predicate_registry"]
    labels = {name: tier1_label(name) for name in registry}
    variants = {}
    for variant in record["variants"]:
        variants[variant["variant_id"]] = {"goal_text": variant["goal_text"], "signal_labels": dict(labels)}
    return {
        "family": record["family"],
        "goal_text": "",  # per-variant only: the two variants are different tasks
        "signal_labels": dict(labels),
        "stg_hash": record["stg_hash"],
        "text_origin": "source_verbatim: the variant's own system_prompt + 'Request: ' + its own instruction, unedited",
        "variants": variants,
    }


def tier2_entry(entry: dict) -> dict:
    out = dict(entry)
    out["text_origin"] = ("generated_from_source_fields: the task's own request plus the translated requested actions "
                          "and the policy sentence (Tier 2 adapter template); NOT the source's verbatim instruction")
    return out


def build_source(tier1: list[dict], tier2: list[dict]) -> dict:
    ids = [m["mechanism_id"] for m in tier1 + tier2]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate mechanism ids across tiers")
    return {
        "kind": "suite.source.v14b",
        "note": ("Public-benchmark suite source for the Agent Detention runner: AgentAbstain (CC-BY-4.0 data, MIT code) "
                 "adapted pairs and tau2-bench / CRM-Agent-Bench sibling pairs.  Not a reviewed set; not set 12."),
        "counts": {"tier1_pairs": len(tier1), "tier2_pairs": len(tier2), "instances": 2 * len(ids)},
        "mechanisms": tier1 + tier2,
    }


# What the harness's resolve_source_set() returned in the run: development set 10 (no review marker present).
# The development mechanism sets are not published, so their hashes are recorded here; the generated
# manifests are byte-identical to the run's.
DEV_SOURCE_SET = {
    "source_set": "10",
    "path": "manifests/reliability/dev-active-set-10.json",
    "sha256": "f988101054f65522dcda9cce863dd8edc08a66e9a14a7032e834d4b94b2afb2f",
    "review_05_passed": False,
    "review_05_asserted": None,
    "review_05_marker_valid": False,
    "review_05_marker": "artifacts/checkpoints/JOB24/ASTRA_W1_REVIEW_05_PASS.json",
    "owner_review_passed": False,
    "owner_review_marker_valid": False,
    "owner_review_marker": "artifacts/review/approvals/set12/SET12_OWNER_REVIEW_PASS.json",
    "set10_sha256": "f988101054f65522dcda9cce863dd8edc08a66e9a14a7032e834d4b94b2afb2f",
    "set11_sha256": "291491a51809c4f0a654363f7b6fb951e86727dd3e98f784d2e7c8f4da35fe30",
    "set12_sha256": "4976acc5d8c0a19a32be455588f6280d25ded1ed8ab0e8091f8c4343f16c7517",
}
DEV_SOURCE_RULE = (
    "set 12 only when an attributable owner-review marker is bound to the set-12 "
    "hash and its combined projection re-verifies (48/48, 0 content failures); "
    "otherwise set 11 only when an attributable W1 review-05 PASS artifact is bound "
    "to the set-11 hash; otherwise set 10"
)


def source_identity(source_path: Path) -> dict:
    """The set-12-style source identity, bound to the suite file's bytes (same shape build_v14_conditions uses)."""
    from vera.canonical import canonical_json, sha256_file, sha256_text
    from vera.delegation.instrument import REVIEW03_CLOSURE

    base = {
        **DEV_SOURCE_SET,
        "rule": DEV_SOURCE_RULE,
        "closure_map": REVIEW03_CLOSURE,
        "closure_digest": sha256_text(canonical_json(REVIEW03_CLOSURE)),
    }
    return {
        **base,
        "source_set": SOURCE_SET,
        "path": str(source_path.relative_to(TREE)),
        "sha256": sha256_file(source_path),
        "review_05_passed": False,
        "review_05_marker_valid": False,
        "owner_review_passed": False,
        "owner_review_marker_valid": False,
        "owner_review_marker": "NOT APPLICABLE: public-benchmark suite source; authority = " + AUTHORITY,
    }


def build_overlay(tier1: list[dict], tier2_text: dict, source_sha: str) -> dict:
    from vera.canonical import canonical_json, sha256_text

    mechanisms = {m["mechanism_id"]: tier1_entry(m) for m in tier1}
    for mid, entry in tier2_text["mechanisms"].items():
        if mid in mechanisms:
            raise SystemExit(f"overlay id clash {mid}")
        mechanisms[mid] = tier2_entry(entry)
    body = {
        "kind": "vera.v14_case_text.v1",
        "protocol": "v14",
        "status": OVERLAY_STATUS,
        "authority": AUTHORITY,
        "authority_scope": ("Tier 1 text is the source's own instruction text, unedited.  Tier 2 text is generated from source "
                            "fields by a template (it states the governing policy sentence); whether the waiver covers it is "
                            "an open question for the owner.  The waiver is conditional on COMPAT_PASS.md existing."),
        "note": "Suite overlay for the v14b working tree.  No owner signature: the authority basis is the waiver above.",
        "owner_signature": None,
        "source": {"path": SOURCE_REL, "sha256": source_sha},
        "counts": {
            "mechanisms": len(mechanisms),
            "labels": sum(len(e["signal_labels"]) for e in mechanisms.values()),
            "variant_text_entries": sum(len(e.get("variants") or {}) for e in mechanisms.values()),
        },
        "mechanisms": mechanisms,
    }
    body["digest"] = sha256_text(canonical_json(body))
    return body


def build_all(*, check: bool = False) -> tuple[dict[str, str], str | None]:
    """Return ({relative path: file text} for every output, the source file's text before the build or None).

    The manifest build reads the source file from disk by path, so it is written first.  With ``check`` the
    tree is left as it was found, also when the build fails: the previous source file is put back, or the new
    one removed (with the output folders, if the build created them).
    """
    from vera.delegation.condition_manifest import build_condition_manifest
    from vera.delegation.evidence import PROTOCOL_V14

    tier1 = json.loads(paths.tier1_records().read_text())["mechanisms"]
    tier2 = json.loads((paths.tier2_out() / "tau2_mechanisms.json").read_text())["mechanisms"]
    tier2_text = json.loads((paths.tier2_out() / "tau2_case_text.json").read_text())
    source_text = dump(build_source(tier1, tier2))
    made_dirs = [d for d in (OUT_DIR, *OUT_DIR.parents) if not d.exists()]  # deepest first
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_path = TREE / SOURCE_REL
    previous = source_path.read_text() if source_path.is_file() else None
    source_path.write_text(source_text)
    try:
        identity = source_identity(source_path)
        conditions = build_condition_manifest(source=identity, protocol=PROTOCOL_V14)
        overlay = build_overlay(tier1, tier2_text, identity["sha256"])
    finally:
        if check:
            if previous is None:
                source_path.unlink(missing_ok=True)
                for d in made_dirs:
                    d.rmdir()
            else:
                source_path.write_text(previous)
    outputs = {SOURCE_REL: source_text, CONDITIONS_REL: dump(conditions), OVERLAY_REL: dump(overlay)}
    return outputs, previous


def stale(outputs: dict[str, str], previous_source: str | None) -> list[str]:
    """The outputs whose file is missing or differs; the source file is compared as it was before the build."""
    bad = []
    for rel, text in outputs.items():
        path = TREE / rel
        current = previous_source if rel == SOURCE_REL else (path.read_text() if path.is_file() else None)
        if current != text:
            bad.append(rel)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    outputs, previous = build_all(check=args.check)
    if args.check:
        bad = stale(outputs, previous)
        print("STALE: " + ", ".join(bad) if bad else "ok: suite inputs match a rebuild")
        return 1 if bad else 0
    for rel, text in outputs.items():
        (TREE / rel).write_text(text)
        print(f"wrote {rel} ({len(text)} bytes)")
    cond = json.loads(outputs[CONDITIONS_REL])
    ov = json.loads(outputs[OVERLAY_REL])
    print("conditions", cond["counts"]["instances"], "instances;", len(cond["necessary_instances"]), "necessary;",
          len(cond["abstention_instances"]), "abstention;", "U =", cond["counts"]["optional_eligible_U"],
          "manifest_digest", cond["manifest_digest"][:16])
    print("overlay digest", ov["digest"][:16], ov["counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
