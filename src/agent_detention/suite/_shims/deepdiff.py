"""Minimal stand-in for deepdiff (only used by tau2's show_dict_diff, never in replay)."""
class DeepDiff(dict):
    def __init__(self, a, b, **k):
        super().__init__()
        self["equal"] = (a == b)
