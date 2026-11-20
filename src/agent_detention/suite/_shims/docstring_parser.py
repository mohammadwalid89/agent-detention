"""Minimal stand-in for docstring_parser: tool *descriptions* are not needed to replay tools."""
class _Doc:
    short_description = ""
    long_description = ""
    params = []
    returns = None
    raises = []
    examples = []
def parse(text):
    return _Doc()
