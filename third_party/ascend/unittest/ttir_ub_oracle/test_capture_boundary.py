import importlib.util
from pathlib import Path

import pytest


TOOL = Path(__file__).parents[2] / "tools" / "ttir_ub_capture_boundary.py"
SPEC = importlib.util.spec_from_file_location("ttir_ub_capture_boundary", TOOL)
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


def test_inline_leading_aliases_preserves_generic_module():
    text = """#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
"builtin.module"() ({
  "test.op"() {maps = [#map, #map1]} : () -> ()
}) : () -> ()
"""

    result = capture.inline_leading_aliases(text)

    assert result.startswith('"builtin.module"()')
    assert "#map" not in result
    assert "affine_map<(d0, d1) -> (d0, d1)>" in result
    assert "affine_map<(d0, d1) -> (d1, d0)>" in result


def test_inline_leading_aliases_rejects_unknown_prefix():
    with pytest.raises(RuntimeError, match="prefix declaration"):
        capture.inline_leading_aliases(
            '#loc = loc("unstable")\n"builtin.module"() ({}) : () -> ()\n'
        )


def test_inline_leading_aliases_requires_module_root():
    with pytest.raises(RuntimeError, match="omitted builtin.module"):
        capture.inline_leading_aliases('"func.func"() ({}) : () -> ()\n')
