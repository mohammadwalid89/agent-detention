"""Tier 2 checks: run after ``build_tier2.py``. No model call, no network.

    AD_TREE=<patched harness tree> AD_WORK=work TAU2_ROOT=... CRM_BENCH_ROOT=... python -m pytest tests/tier2 -q
"""
import copy
import json
import os

import pytest

from agent_detention import paths

OUT = paths.tier2_out()
if not (os.environ.get("AD_TREE") and (OUT / "tau2_mechanisms.json").is_file()):
    pytest.skip("needs AD_TREE and the built Tier 2 files", allow_module_level=True)

from agent_detention.suite.tier2 import record_builder as B  # noqa: E402
from agent_detention.suite.tier2 import tau2_adapter as A  # noqa: E402
from agent_detention.suite.tier2 import tau2_replay as R  # noqa: E402

B._vera()

MECHS = json.loads((OUT / "tau2_mechanisms.json").read_text())["mechanisms"]
TEXTS = json.loads((OUT / "tau2_case_text.json").read_text())["mechanisms"]


def _apply(mech, variant):
    """Entry tables with the variant's active predicates applied (harness semantics)."""
    from vera.mechanisms.readiness import _set_variant_predicates
    t = copy.deepcopy(mech["entry_state"]["tables"])
    _set_variant_predicates(t, mech["entry_state"]["predicate_registry"], variant["active_predicates"])
    return t


def test_counts_are_what_the_report_says():
    assert len(MECHS) == 133
    assert len({m["mechanism_id"] for m in MECHS}) == 133
    assert set(TEXTS) == {m["mechanism_id"] for m in MECHS}


@pytest.mark.parametrize("m", MECHS, ids=lambda m: m["mechanism_id"])
def test_harness_labels_and_readiness(m):
    labels = B.check_labels(m)
    v1, v2 = (v["variant_id"] for v in m["variants"])
    assert labels[v1]["label"] == "SOLVABLE"
    assert labels[v2]["label"] == "ABSTENTION_CORRECT"
    assert m["readiness"]["benchmark_ready"] is True


@pytest.mark.parametrize("m", MECHS, ids=lambda m: m["mechanism_id"])
def test_every_variant_is_marked_native_or_derived_and_derived_edits_are_listed(m):
    origins = [v["origin"] for v in m["variants"]]
    assert sorted(origins) in (["derived", "native"], ["native", "native"])
    if "derived" in origins:
        assert m["suite"]["edit"], "a derived sibling must list the edit that derives it"
        for pc in ([m["suite"]["policy_cite"]] if isinstance(m["suite"]["policy_cite"], dict) else m["suite"]["policy_cite"]):
            assert pc["quote"] and pc["line"] >= 1


@pytest.mark.parametrize("m", [m for m in MECHS if m["mechanism_id"].startswith(("T2-AIRLINE", "T2-RETAIL", "T2-TELECOM"))
                               and any(v["origin"] == "derived" for v in m["variants"])],
                         ids=lambda m: m["mechanism_id"])
def test_derived_sibling_differs_by_the_listed_edit_only(m):
    t1, t2 = _apply(m, m["variants"][0]), _apply(m, m["variants"][1])
    cells = []
    for table in sorted(set(t1) | set(t2)):
        r1, r2 = t1.get(table, []), t2.get(table, [])
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            for k in sorted(set(a) | set(b)):
                if a.get(k) != b.get(k):
                    cells.append((table, k))
    edits = {(e["table"], e["field"]) for e in m["suite"]["edit"]}
    assert set(cells) == edits, (cells, edits)
    assert len(cells) == 1, "a derived sibling is a ONE-field edit"


def test_policy_quotes_are_really_in_the_policy_files():
    for m in MECHS:
        cites = m["suite"]["policy_cite"]
        for c in ([cites] if isinstance(cites, dict) else cites):
            base = paths.tau2_root() if "tau2" in c["file"] else paths.crm_root()
            path = base / c["file"]
            first = c["quote"].split(" ")[0:6]
            assert " ".join(first) in " ".join(path.read_text().split()), (m["mechanism_id"], c["quote"][:60])


def test_no_floats_in_hash_relevant_fields():
    def walk(o):
        if isinstance(o, float):
            raise AssertionError("float found")
        if isinstance(o, dict):
            [walk(v) for v in o.values()]
        elif isinstance(o, list):
            [walk(v) for v in o]
    for m in MECHS:
        walk(m["entry_state"])
        walk(m["stg"])
        walk(m["postconditions"])


@pytest.mark.parametrize("dom,tid", [("retail", "16"), ("retail", "0"), ("airline", "7")])
def test_end_state_is_taus_own_gold_db(dom, tid):
    """The recorded db hash is recomputed from a fresh replay of the gold actions on tau2's tools."""
    task = [t for t in R.load_tasks(dom) if str(t["id"]) == tid][0]
    rep = A.replay_steps(dom, task)
    m = next(x for x in MECHS if x["mechanism_id"] == f"T2-{dom.upper()}-{tid}")
    assert m["suite"]["tau2_gold_db_hash"] == rep["db_hash"]
    # and tau2's own evaluator hash (replay module) agrees with the step-wise replay
    assert R.replay(dom, task)["db_hash"] == rep["db_hash"]


def test_build_is_deterministic_for_a_sample():
    task = [t for t in R.load_tasks("retail") if str(t["id"]) == "16"][0]
    a = A.build_pair("retail", task)["mechanism"]
    b = A.build_pair("retail", task)["mechanism"]
    assert B.canonical(a) == B.canonical(b)
    assert B.canonical(a) == B.canonical(next(m for m in MECHS if m["mechanism_id"] == a["mechanism_id"]))
