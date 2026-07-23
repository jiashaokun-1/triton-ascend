import importlib.util
import json
from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).parents[2] / "tools"


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolves postponed annotations through the defining module.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bundle = _load_tool("ttir_ub_fixture_bundle")
oracle = _load_tool("ttir_ub_oracle")


def _dump_dir(root, allocation_count=2, allocation_elements=1024):
    root.mkdir()
    (root / "kernel.ttir.mlir").write_text("module { tt.func public @copy() }\n")
    (root / "before_cvpipelining.mlir").write_text(
        "".join(
            f'%{index} = "memref.alloc"() : () -> memref<{allocation_elements}xf32>\n'
            for index in range(allocation_count)
        )
    )
    return root


def _config(**updates):
    values = {
        "name": "binary-add-f32-65536-a2",
        "operation_family": "binary-add",
        "arch": "Ascend910B",
        "source_elements": 65536,
        "element_bit_width": 32,
        "max_tiles": 64,
        "auto_tile_and_bind_subblock_outcome": False,
    }
    values.update(updates)
    return bundle.BundleConfig(**values)


def test_bundle_packages_one_same_dump_pair_for_the_oracle(tmp_path):
    dump_dir = _dump_dir(tmp_path / "same-compilation-dump")
    output_dir = tmp_path / "fixture"

    result = bundle.create_fixture_bundle(dump_dir, output_dir, _config())

    manifest = oracle.load_manifest(Path(result["manifest"]))
    case = manifest["cases"][0]
    assert case["operation_family"] == "binary-add"
    assert case["contract_proposal"] == {
        "expected_resource_count": 2,
        "expected_source_elements": 65536,
        "expected_element_bit_width": 32,
        "expected_input_payload_bytes": 262144,
        "materialization_stage": "ttir.triton-to-linalg",
        "max_tiles": 64,
        "auto_tile_and_bind_subblock_outcome": False,
    }
    assert result["ttir_sha256"] == oracle.file_sha256(case["ttir"])
    assert result["before_cvpipelining_sha256"] == oracle.file_sha256(
        case["before_cvpipelining"]
    )
    assert result["before_cvpipelining_allocations_bytes"] == \
        oracle.parse_boundary_allocation_bytes(
            case["before_cvpipelining"].read_text(encoding="utf-8")
        )


def test_bundle_refuses_partial_or_overwritten_pairs(tmp_path):
    dump_dir = tmp_path / "incomplete"
    dump_dir.mkdir()
    (dump_dir / "kernel.ttir.mlir").write_text("module {}\n")
    with pytest.raises(bundle.BundleError, match="must contain"):
        bundle.create_fixture_bundle(dump_dir, tmp_path / "fixture", _config())

    complete = _dump_dir(tmp_path / "complete")
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text("do not replace\n")
    with pytest.raises(bundle.BundleError, match="overwrite"):
        bundle.create_fixture_bundle(complete, output_dir, _config())
    assert (output_dir / "manifest.json").read_text() == "do not replace\n"


def test_bundle_rejects_raw_ttir_to_linalg_output_as_the_cvpipeline_boundary(tmp_path):
    raw_ttir_adapter = _dump_dir(
        tmp_path / "raw-ttadapter", allocation_count=2, allocation_elements=65536
    )
    with pytest.raises(bundle.BundleError, match="allocations do not match"):
        bundle.create_fixture_bundle(
            raw_ttir_adapter, tmp_path / "fixture", _config()
        )


def test_bundle_refuses_a_broken_output_symlink(tmp_path):
    dump_dir = _dump_dir(tmp_path / "dump")
    output_dir = tmp_path / "fixture"
    output_dir.mkdir()
    outside = tmp_path / "outside.ttir"
    (output_dir / "binary-add-f32-65536-a2.ttir.mlir").symlink_to(outside)

    with pytest.raises(bundle.BundleError, match="overwrite"):
        bundle.create_fixture_bundle(dump_dir, output_dir, _config())
    assert not outside.exists()


@pytest.mark.parametrize("outside_name", ["../outside.mlir", "/tmp/outside.mlir"])
def test_bundle_dump_overrides_cannot_escape_the_same_compilation_directory(
    tmp_path, outside_name
):
    dump_dir = _dump_dir(tmp_path / "dump")
    with pytest.raises(bundle.BundleError, match="dump directory"):
        bundle.create_fixture_bundle(
            dump_dir,
            tmp_path / "fixture",
            _config(),
            ttir_dump_name=outside_name,
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"name": "../escape"},
        {"operation_family": "broadcast"},
        {"source_elements": 0},
        {"element_bit_width": 7},
        {"element_bit_width": 12},
        {"max_tiles": 0},
    ],
)
def test_bundle_rejects_unsafe_or_inconsistent_configuration(tmp_path, updates):
    with pytest.raises(bundle.BundleError):
        bundle.create_fixture_bundle(
            _dump_dir(tmp_path / "dump"), tmp_path / "fixture", _config(**updates)
        )


def test_bundle_cli_emits_machine_readable_result(tmp_path, capsys):
    dump_dir = _dump_dir(tmp_path / "dump", allocation_count=1)
    output_dir = tmp_path / "fixture"
    status = bundle.main([
        "--dump-dir", str(dump_dir),
        "--output-dir", str(output_dir),
        "--name", "reshape-copy-f32-65536-a2",
        "--operation-family", "reshape-copy",
        "--arch", "Ascend910B",
        "--source-elements", "65536",
        "--element-bit-width", "32",
        "--max-tiles", "64",
    ])
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["expected_input_payload_bytes"] == 262144
    assert payload["before_cvpipelining_allocations_bytes"] == [4096]
    case = oracle.load_manifest(output_dir / "manifest.json")["cases"][0]
    assert case["operation_family"] == "reshape-copy"
    assert case["expected_analyzer_decision"] is None
    assert case["contract_proposal"]["auto_tile_and_bind_subblock_outcome"] is None
