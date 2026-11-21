"""audit_run.py: M1/M2/M3 on synthetic run directories. The first version of the M2 scan attributed no call to any case
(it looked for an ``instance_id`` the call records do not carry) and so checked nothing; these tests pin that it scans,
that it catches the leaks it claims to, and that M3 is never vacuously true."""

from __future__ import annotations

import json

import pytest

from agent_detention.suite.tier1 import harness

TREE = harness.tree_path()
pytestmark = pytest.mark.skipif(not (TREE / "manifests/suite/SUITE_SOURCE_V14B.json").is_file(), reason="suite inputs not built")


def _mech():
    source = json.loads((TREE / "manifests/suite/SUITE_SOURCE_V14B.json").read_text())
    return next(m for m in source["mechanisms"] if m["mechanism_id"].startswith("AA-"))


def _run(tmp_path, *, main=None, detention=None, refusal=False, rows=True):
    """A tiny run directory: one AA case, opaque episode id, hand-made call logs."""
    from agent_detention.suite.integrate import audit_run

    mech = _mech()
    run = tmp_path / "RUN"
    (run / "episode_calls").mkdir(parents=True)
    (run / "episodes").mkdir()
    table = [{"episode_id": "ep-aaaa", "instance_id": mech["mechanism_id"] + "/" + mech["variants"][0]["variant_id"]}]
    (run / "assignment_table.json").write_text(json.dumps({"rows": table if rows else []}))

    def rec(text, **extra):
        body = {"episode_id": "ep-aaaa", "reply_text": "r", **extra,
                "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": text}]}
        return json.dumps(body)

    (run / "episode_calls" / "m.jsonl").write_text("\n".join(rec(t) for t in (main or ["a clean prompt"])) + "\n")
    if detention is not None or refusal:
        (run / "detention" / "episode_calls").mkdir(parents=True)
        lines = [rec(t) for t in (detention or [])]
        if refusal:
            lines.append(rec("review prompt", error_kind="PROVIDER_REFUSAL", error="safeguards flagged this message"))
        (run / "detention" / "episode_calls" / "d.jsonl").write_text("\n".join(lines) + "\n")
    return audit_run.audit(run)


def test_a_clean_run_is_scanned_call_by_call_and_passes(tmp_path):
    report = _run(tmp_path, detention=["retry prompt one", "retry prompt two"])
    assert report["m2_calls_checked"] == 3 and report["m2_unattributed_calls"] == 0
    assert report["ok_m1"] and report["ok_m2"] and report["ok_m3"] and report["ok_no_errors"] and report["m3_calls_checked"] == 2


def test_scenario_name_task_id_and_variant_id_in_a_prompt_are_leaks(tmp_path):
    mech = _mech()
    cases = ((mech["provenance"]["scenario"], "provenance.scenario"), (mech["provenance"]["task_id"], "provenance.task_id"),
             (mech["variants"][0]["variant_id"], "variant_id V1"))
    for leaked, label in cases:
        report = _run(tmp_path / label.replace(" ", "_"), main=["x " + leaked + " y"])
        assert not report["ok_m2"] and label in {leak["what"] for leak in report["leaks"]}, (label, report["leaks"])


def test_m2_is_not_vacuous_when_no_call_can_be_attributed(tmp_path):
    report = _run(tmp_path, rows=False)
    assert report["m2_calls_checked"] == 0 and report["m2_unattributed_calls"] == 1 and not report["ok_m2"]


@pytest.mark.parametrize("text,found", [
    ("this is arm A_DETENTION", "A_DETENTION"), ("mode: free_form notes", "free_form"),
    ("review_mode=structured_valid", "structured_valid"), ("condition ABSTENTION applies", "ABSTENTION"),
    ("condition: OPTIONAL", "OPTIONAL"),
])
def test_m3_flags_arm_names_review_modes_and_condition_labels_in_detention_prompts(tmp_path, text, found):
    report = _run(tmp_path, detention=["clean", text])
    assert report["m3_exercised"] and not report["ok_m3"]
    assert found in report["m3_hits"][0]["found"]


def test_m3_ignores_ordinary_words_and_is_never_vacuously_true(tmp_path):
    report = _run(tmp_path, detention=["an optional field; the free form of the request; necessary steps"])
    assert report["ok_m3"], report["m3_hits"]  # lower-case ordinary words are not labels
    none = _run(tmp_path / "b")
    assert none["m3_calls_checked"] == 0 and not none["m3_exercised"] and not none["ok_m3"]


def test_a_provider_refusal_is_counted_separately_and_is_not_an_error(tmp_path):
    report = _run(tmp_path, detention=["retry prompt"], refusal=True)
    assert report["refusal_calls"] == 1 and report["call_errors"] == [] and report["ok_no_errors"]


def test_opaque_row_episode_ids_carry_no_case_information():
    from agent_detention.suite.integrate import rows

    source = json.loads((TREE / "manifests/suite/SUITE_SOURCE_V14B.json").read_text())
    ids = [m["mechanism_id"] for m in source["mechanisms"] if m["mechanism_id"].startswith("AA-")][:3]
    made = rows.make_rows(ids, ["O1"], 1)
    mech_names = [m.split("-", 1)[1] for m in ids] + ["preview_", "-V1", "-V2", "insufficient", "emergent", "ambiguous"]
    assert len({r["episode_id"] for r in made}) == len(made)
    for row in made:
        assert row["episode_id"].startswith("ep-") and not any(name in row["episode_id"] for name in mech_names)
