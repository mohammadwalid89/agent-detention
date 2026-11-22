"""``build_inputs --check`` compares the source file as it was before the build and leaves the tree as found.

Runs without the datasets: empty Tier 1 and Tier 2 inputs in a temporary work folder, a temporary tree, and a
stand-in for the harness's condition-manifest builder.
"""

from __future__ import annotations

import json

import pytest

from agent_detention.suite.integrate import build_inputs as bi


@pytest.fixture
def tree(tmp_path, monkeypatch):
    work = tmp_path / "work"
    (work / "tier1").mkdir(parents=True)
    (work / "tier2").mkdir()
    (work / "tier1" / "tier1_set.json").write_text(json.dumps({"mechanisms": []}))
    (work / "tier2" / "tau2_mechanisms.json").write_text(json.dumps({"mechanisms": []}))
    (work / "tier2" / "tau2_case_text.json").write_text(json.dumps({"mechanisms": {}}))
    monkeypatch.setenv("AD_WORK", str(work))
    root = tmp_path / "tree"
    root.mkdir()  # as in harness/: no manifests/ folder until a build writes one
    monkeypatch.setattr(bi, "TREE", root)
    monkeypatch.setattr(bi, "OUT_DIR", root / "manifests" / "suite")

    import vera.delegation.condition_manifest as cm

    def fake_manifest(*, source, protocol):
        assert (root / source["path"]).is_file()  # the builder reads the source file from disk
        return {
            "counts": {"instances": 0, "optional_eligible_U": 0},
            "necessary_instances": [],
            "abstention_instances": [],
            "manifest_digest": source["sha256"],
            "protocol": protocol,
        }

    monkeypatch.setattr(cm, "build_condition_manifest", fake_manifest)
    return root


def files(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def run(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["build_inputs", *argv])
    return bi.main()


def test_check_on_an_empty_tree_reports_stale_and_writes_nothing(tree, monkeypatch, capsys):
    assert run(monkeypatch, "--check") == 1
    assert files(tree) == []
    assert bi.SOURCE_REL in capsys.readouterr().out


def test_check_after_a_build_is_ok_and_changes_nothing(tree, monkeypatch):
    assert run(monkeypatch) == 0
    before = {rel: (tree / rel).read_text() for rel in (bi.SOURCE_REL, bi.CONDITIONS_REL, bi.OVERLAY_REL)}
    assert run(monkeypatch, "--check") == 0
    assert {rel: (tree / rel).read_text() for rel in before} == before


def test_check_reports_a_stale_source_file_and_keeps_it(tree, monkeypatch, capsys):
    assert run(monkeypatch) == 0
    source = tree / bi.SOURCE_REL
    source.write_text(source.read_text().replace("suite.source.v14b", "edited"))
    edited = source.read_text()
    capsys.readouterr()
    assert run(monkeypatch, "--check") == 1
    assert bi.SOURCE_REL in capsys.readouterr().out
    assert source.read_text() == edited


def test_check_restores_the_tree_when_the_build_fails(tree, monkeypatch):
    import vera.delegation.condition_manifest as cm

    def broken(**kwargs):
        raise RuntimeError("manifest build failed")

    monkeypatch.setattr(cm, "build_condition_manifest", broken)
    with pytest.raises(RuntimeError):
        run(monkeypatch, "--check")
    assert files(tree) == []
