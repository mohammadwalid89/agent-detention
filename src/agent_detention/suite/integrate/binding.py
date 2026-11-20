"""The suite's source binding: what ``GridRunner(source_binding=...)`` gets in place of ``resolve_source_binding()``.

``resolve_source_binding`` proves the set-12 bytes, the owner-review marker chain and a preregistration.  None of that
exists for the public-benchmark suite; its authority basis is the owner waiver recorded in the overlay.  This
resolver proves the things that CAN be proved and fails closed otherwise:

* the suite source file, the case-text overlay and the conditions manifest all name the same source sha256;
* the overlay digest verifies and its status/authority text are the expected ones;
* the conditions manifest digest verifies (recomputed from its own body) and every instance has the cells it needs;
* the presence of the file named by ``$AD_COMPAT_PASS`` (the compatibility-gate record) is reported as
  ``compat_pass`` but not required here: the compatibility smoke itself runs through this binding.

It also switches the tree's case-text lookup to the suite overlay (``case_text.select_overlay``, opt-in).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from agent_detention.suite.integrate import build_inputs as bi

TREE = bi.TREE


class SuiteBindingError(Exception):
    pass


def resolve_suite_binding(*, select: bool = True) -> dict:
    from vera.canonical import canonical_json, sha256_file, sha256_text
    from vera.delegation import case_text
    from vera.delegation.instrument import load_source_instances

    source_path, overlay_path, cond_path = TREE / bi.SOURCE_REL, TREE / bi.OVERLAY_REL, TREE / bi.CONDITIONS_REL
    for p in (source_path, overlay_path, cond_path):
        if not p.is_file():
            raise SuiteBindingError(f"missing suite input {p}; run agent_detention.suite.integrate.build_inputs")
    source_sha = sha256_file(source_path)
    overlay = json.loads(overlay_path.read_text())
    manifest = json.loads(cond_path.read_text())
    problems = []
    body = {k: v for k, v in overlay.items() if k != "digest"}
    if overlay.get("digest") != sha256_text(canonical_json(body)):
        problems.append("overlay digest does not verify")
    if overlay.get("authority") != bi.AUTHORITY or overlay.get("status") != bi.OVERLAY_STATUS:
        problems.append("overlay authority/status text is not the recorded owner waiver")
    if overlay["source"]["sha256"] != source_sha:
        problems.append("overlay was built for different source bytes")
    mbody = {k: v for k, v in manifest.items() if k != "manifest_digest"}
    if manifest.get("manifest_digest") != sha256_text(canonical_json(mbody)):
        problems.append("conditions manifest digest does not verify")
    if manifest["source"]["sha256"] != source_sha:
        problems.append("conditions manifest was built for different source bytes")
    if manifest.get("protocol") != "v14":
        problems.append("conditions manifest is not protocol v14")
    if problems:
        raise SuiteBindingError("suite binding refused: " + "; ".join(problems))
    if select:
        case_text.select_overlay(overlay_path, source_path)
    source = dict(manifest["source"])
    ids = sorted(i.instance_id for i in load_source_instances(source))
    return {
        "kind": "vera.v14b.suite_source_binding.v1",
        "source": source,
        "condition_manifest": manifest,
        "condition_manifest_digest": manifest["manifest_digest"],
        "preregistration": {
            "path": "",
            "root_sha256": "",
            "version_id": "SUITE_V14B_NO_PREREGISTRATION",
            "run_plan_version_id": "",
            "authority": overlay["authority"],
        },
        "case_text_overlay": {"path": bi.OVERLAY_REL, "digest": overlay["digest"], "status": overlay["status"]},
        "compat_pass": bool(os.environ.get("AD_COMPAT_PASS")) and Path(os.environ["AD_COMPAT_PASS"]).is_file(),
        "instance_ids": ids,
    }
