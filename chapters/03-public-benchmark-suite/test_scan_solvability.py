"""Known-answer tests for the A2 scan. Records are edited IN MEMORY only (deep copies); no input file is written.

Run (needs AD_WORK with tier1_set.json and AGENTABSTAIN_DATA): python -m pytest -q chapters/03-public-benchmark-suite/test_scan_solvability.py
"""
import copy
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_solvability as S  # noqa: E402

if not (os.environ.get("AGENTABSTAIN_DATA") and S.SET.is_file()):
    pytest.skip("needs AGENTABSTAIN_DATA and the adapted record set", allow_module_level=True)

RECORDS = {r["mechanism_id"]: r for r in json.load(open(S.SET))["mechanisms"]}
SOLVABLE_CASE = "AA-ambiguous_action_specification-preview_008"   # SOLVABLE: the required commit is reachable
ITC_CASE = "AA-insufficient_tool_capability-preview_012"           # its task text mandates a tool that is not offered
ITC_TOOL = "store_procurement_and_inventory.verify_item_authenticity"


def _verdict(rec):
    act, _ = S.load_act_task(rec)
    return S.classify_act(rec, act, S.catalog_for(rec["provenance"]["environments"]))


def _verdict_with_text(rec, extra: str):
    act, _ = S.load_act_task(rec)
    act = copy.deepcopy(act)
    act["instruction"] = act["instruction"].rstrip() + " " + extra
    return S.classify_act(rec, act, S.catalog_for(rec["provenance"]["environments"]))


# ---- KAT 1 (the brief's test): remove a required tool from a solvable case -> must be flagged -------------------
def test_unmodified_solvable_case_is_solvable():
    assert _verdict(RECORDS[SOLVABLE_CASE])["verdict"] == "SOLVABLE"


def test_removing_the_required_commit_tool_is_flagged_unsolvable():
    rec = copy.deepcopy(RECORDS[SOLVABLE_CASE])
    required = rec["provenance"]["a_critical"][0]
    rec["permitted_actions"] = [a for a in rec["permitted_actions"] if a != required]
    v = _verdict(rec)
    assert v["verdict"] == "UNSOLVABLE"
    assert "S1_REQUIRED_STATE_UNREACHABLE" in v["codes"]
    assert v["min_steps_to_goal"] is None


def test_removing_the_required_tool_in_every_eligible_case_is_flagged():
    """Not one case can survive the loss of its own commit tool (guards against a check that cannot fail)."""
    missed = []
    for mid, rec in RECORDS.items():
        if _verdict(rec)["verdict"] != "SOLVABLE":
            continue
        rec2 = copy.deepcopy(rec)
        drop = rec2["provenance"]["a_critical"][-1]
        rec2["permitted_actions"] = [a for a in rec2["permitted_actions"] if a != drop]
        if _verdict(rec2)["verdict"] != "UNSOLVABLE":
            missed.append(mid)
    assert missed == []


# ---- KAT 2: the mandated-verify-tool check, both directions ----------------------------------------------------
def test_itc_case_is_unsolvable_as_adapted():
    v = _verdict(RECORDS[ITC_CASE])
    assert v["verdict"] == "UNSOLVABLE" and "S2_MANDATED_TOOL_NOT_PERMITTED" in v["codes"]
    assert any(r.get("tool") == ITC_TOOL for r in v["reasons"])


def test_offering_the_mandated_tool_clears_the_flag_and_removing_it_again_restores_it():
    rec = copy.deepcopy(RECORDS[ITC_CASE])
    rec["permitted_actions"] = rec["permitted_actions"] + [ITC_TOOL]
    assert "S2_MANDATED_TOOL_NOT_PERMITTED" not in _verdict(rec)["codes"]
    rec["permitted_actions"].remove(ITC_TOOL)
    assert "S2_MANDATED_TOOL_NOT_PERMITTED" in _verdict(rec)["codes"]


def test_a_planted_mandate_sentence_is_flagged_and_without_mandate_wording_is_only_unclear():
    rec = RECORDS[SOLVABLE_CASE]
    cat = S.catalog_for(rec["provenance"]["environments"])
    tool = next(t for t in sorted(cat) if t not in rec["permitted_actions"])
    short = tool.split(".", 1)[1]
    hard = _verdict_with_text(rec, f"You MUST call the `{short}` tool first - do not skip it.")
    soft = _verdict_with_text(rec, f"You could look this up with {short}.")
    assert hard["verdict"] == "UNSOLVABLE" and "S2_MANDATED_TOOL_NOT_PERMITTED" in hard["codes"]
    assert soft["verdict"] == "UNCLEAR" and soft["codes"] == ["U1_NAMED_TOOL_NOT_PERMITTED"]


def test_naming_a_permitted_tool_is_not_flagged():
    rec = RECORDS[SOLVABLE_CASE]
    short = rec["provenance"]["a_critical"][0].split(".", 1)[1]
    assert _verdict_with_text(rec, f"You MUST call {short}.")["verdict"] == "SOLVABLE"


# ---- KAT 3: the pair check ---------------------------------------------------------------------------------------
def test_pair_check_flags_identical_observations_and_passes_different_ones():
    assert S.pair_indistinguishable(RECORDS[ITC_CASE])["indistinguishable"] is True
    rec = copy.deepcopy(RECORDS[SOLVABLE_CASE])
    assert S.pair_indistinguishable(rec)["indistinguishable"] is False
    v1, v2 = rec["variants"]
    for f in S.OBS_FIELDS:
        v2[f] = copy.deepcopy(v1.get(f))
    assert S.pair_indistinguishable(rec)["indistinguishable"] is True


# ---- whole-run properties ----------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def result():
    return S.scan()


def test_scan_covers_all_131_and_counts_add_up(result):
    s = result["summary"]
    assert s["act_versions"] == 131 == len(RECORDS)
    assert sum(s["verdicts"].values()) == 131
    assert s["act_task_hash_mismatches"] == 0
    assert s["eligible_pairs"] == sum(c["verdict"] == "SOLVABLE" and not c["pair"]["indistinguishable"]
                                      for c in result["cases"])


def test_scan_is_deterministic(result):
    again = S.scan()
    assert json.dumps(again, sort_keys=True) == json.dumps(result, sort_keys=True)
