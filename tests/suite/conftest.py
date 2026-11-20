import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def patched_tree():
    """The harness tree with harness_patch/harness_v14b.patch applied, from $AD_TREE."""
    if not os.environ.get("AD_TREE"):
        pytest.skip("AD_TREE not set")
    from agent_detention.suite.tier1 import harness

    harness.use_tree()
    assert harness.is_patched()
    return Path(os.environ["AD_TREE"])


@pytest.fixture(scope="session")
def synthetic_root():
    from agent_detention.suite.tier1 import synthetic

    root = Path(tempfile.mkdtemp(prefix="aa_syn_"))
    refs = synthetic.make(root)
    return root, refs
