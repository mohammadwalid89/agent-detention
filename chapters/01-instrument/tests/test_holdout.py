from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from vera import HOLDOUT_FAMILIES
from vera.errors import HoldoutSealError
from vera.holdout import (
    assert_can_read,
    assert_w2_entry,
    seal,
    seal_status,
    verify_seal,
)


def _author_holdout(root: Path) -> None:
    """Stand-in for the separate author's output (this test does not author mechanisms)."""
    for fam in HOLDOUT_FAMILIES:
        d = root / fam
        d.mkdir(parents=True)
        (d / "mechanisms.json").write_text(json.dumps({"family": fam, "mechanisms": []}) + "\n")
        (d / "expected_states.json").write_text(json.dumps({"family": fam}) + "\n")


def _unlock(root: Path) -> None:
    for p in sorted(root.rglob("*"), key=lambda q: len(str(q)), reverse=True):
        try:
            os.chmod(p, 0o700 if p.is_dir() else 0o600)
        except PermissionError:
            pass


def test_missing_seal_is_reported_honestly(tmp_path: Path):
    st = seal_status(tmp_path / "SEALED")
    assert st.status == "AWAITING_SEPARATE_AUTHOR"
    assert st.author_id == ""


def test_builder_cannot_seal(tmp_path: Path):
    root = tmp_path / "SEALED"
    _author_holdout(root)
    with pytest.raises(HoldoutSealError, match="separate session/author"):
        seal(root, author_id="deepseek-builder-w1")
    with pytest.raises(HoldoutSealError):
        seal(root, author_id="someone", identity="builder")


def test_builder_cannot_read_sealed_set(tmp_path: Path):
    with pytest.raises(HoldoutSealError):
        assert_can_read(tmp_path, identity="builder")


def test_seal_verify_and_ordering(tmp_path: Path):
    root = tmp_path / "SEALED"
    _author_holdout(root)
    st = seal(root, author_id="independent-author", identity="holdout-author")
    assert st.status == "SEALED"
    # The dev manifest is frozen AFTER the seal (design §6.6 ordering rule).
    dev = tmp_path / "seed-dev.json"
    dev.write_text("{}")
    result = verify_seal(root, dev_manifest_path=dev, dev_author_ids=("deepseek-builder-w1",))
    assert result["ordering_ok"] is True
    # Ordering violation: dev manifest predates the seal.
    past = time.time() - 3600
    os.utime(dev, (past, past))
    with pytest.raises(HoldoutSealError, match="after the dev manifest freeze"):
        verify_seal(root, dev_manifest_path=dev)
    _unlock(root)


def test_tampering_after_seal_invalidates_it(tmp_path: Path):
    root = tmp_path / "SEALED"
    _author_holdout(root)
    seal(root, author_id="independent-author", identity="holdout-author")
    _unlock(root)
    victim = next(p for p in root.rglob("mechanisms.json"))
    victim.write_text("{}")
    st = seal_status(root)
    assert st.status == "INVALID" and "hash mismatch" in st.detail


def test_w2_entry_is_blocked_without_a_seal(tmp_path: Path):
    dev = tmp_path / "seed-dev.json"
    dev.write_text("{}")
    with pytest.raises(HoldoutSealError):
        assert_w2_entry(tmp_path / "SEALED", dev_manifest_path=dev)


def test_same_author_on_dev_and_holdout_is_rejected(tmp_path: Path):
    root = tmp_path / "SEALED"
    _author_holdout(root)
    dev = tmp_path / "seed-dev.json"
    dev.write_text("{}")
    old = time.time() - 7200
    os.utime(dev, (old, old))
    seal(root, author_id="deepseek-builder-w1-x", identity="holdout-author")
    with pytest.raises(HoldoutSealError, match="not independent"):
        verify_seal(root, dev_manifest_path=dev,
                    dev_author_ids=("deepseek-builder-w1-x",))
    _unlock(root)
