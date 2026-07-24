import importlib.util
from pathlib import Path


TOOL = (
    Path(__file__).parents[2]
    / "tools"
    / "ttir_ub_prepare_same_schema_oracle.py"
)
SPEC = importlib.util.spec_from_file_location(
    "ttir_ub_prepare_same_schema_oracle", TOOL
)
prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare)


def test_suffix_oracle_patch_matches_pinned_bishengir_schema():
    ascend_root = Path(__file__).parents[2]
    source = ascend_root / "AscendNPU-IR"
    patch = ascend_root / "tools" / "patches" / prepare.PATCH_NAME

    assert prepare.source_revision(source) == (
        prepare.EXPECTED_ASCENDNPU_IR_REVISION
    )
    assert prepare.source_file_state(source) == "ready"
    assert prepare.patch_state(source, patch) == "ready"
