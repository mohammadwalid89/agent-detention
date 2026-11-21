import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def patched_tree():
    """The harness tree with the per-variant overrides: harness/ in this repository, or $AD_TREE."""
    from agent_detention.suite.tier1 import harness

    tree = harness.use_tree()
    assert harness.is_patched()
    return tree


@pytest.fixture(scope="session")
def synthetic_root():
    from agent_detention.suite.tier1 import synthetic

    root = Path(tempfile.mkdtemp(prefix="aa_syn_"))
    refs = synthetic.make(root)
    return root, refs
