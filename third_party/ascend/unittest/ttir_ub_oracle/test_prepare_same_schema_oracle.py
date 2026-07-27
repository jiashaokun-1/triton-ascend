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
    compat_patch = ascend_root / "tools" / "patches" / prepare.COMPAT_PATCH_NAME

    # The oracle supports both a Git checkout and an exported source tree.
    # In the latter case, the pinned file fingerprints and patch applicability
    # are the identity evidence; requiring `.git` would reject a valid source
    # export before those checks can run.
    revision = prepare.source_revision(source)
    assert revision in (None, prepare.EXPECTED_ASCENDNPU_IR_REVISION)
    # CI may run after the helper has already applied either patch.  Verify
    # both directions of the idempotent patch-state protocol instead of
    # falsely requiring a pristine exported source snapshot.
    primary_state = prepare.source_file_state(source)
    assert primary_state in ("ready", "applied")
    assert prepare.patch_state(source, patch) == primary_state
    compat_state = prepare.compat_patch_state(source, compat_patch)
    assert compat_state in ("ready", "applied")


def test_suffix_oracle_configure_enables_ascend_backend():
    command = prepare._configure_command(
        Path("/src"), Path("/build"), "cmake", ["-DCUSTOM=ON"]
    )

    assert "-DTRITON_CODEGEN_BACKENDS=ascend" in command
    assert "-DTRITON_ASCEND_BUILD_BISHENGIR_ORACLE_TOOLS=ON" in command
    assert "-DBISHENGIR_BUILD_PYTHON_BINDINGS=OFF" in command
    assert "-DBSPUB_DAVINCI_BISHENGIR=ON" in command
    assert command[-1] == "-DCUSTOM=ON"


def test_suffix_oracle_rejects_llvm_snapshot_without_required_extension(
    tmp_path: Path,
):
    source = tmp_path / "AscendNPU-IR"
    linalg_dir = (
        source
        / "third-party"
        / "llvm-project"
        / "mlir"
        / "lib"
        / "Dialect"
        / "Linalg"
        / "IR"
    )
    linalg_dir.mkdir(parents=True)
    (linalg_dir / "CMakeLists.txt").write_text("add_mlir_dialect_library(MLIRLinalgDialect)\n")
    patch = tmp_path / prepare.LLVM_LINALG_EXTENSION_PATCH_NAME
    patch.write_text("not reached when LinalgExtensions.cpp is absent\n")

    # The required source disappeared from a newer LLVM snapshot.  The oracle
    # must fail closed instead of faking the missing library.
    try:
        prepare.llvm_linalg_extension_patch_state(source, patch)
    except RuntimeError as error:
        assert "LinalgExtensions.cpp is absent" in str(error)
    else:
        raise AssertionError("incompatible LLVM snapshot unexpectedly accepted")
