import importlib.util
import json
from pathlib import Path

import pytest


TOOL = Path(__file__).parents[2] / "tools" / "ttir_ub_oracle.py"
SPEC = importlib.util.spec_from_file_location("ttir_ub_oracle", TOOL)
oracle = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(oracle)


SUCCESS = """
PLANMEM_PEAK\t0\t2\t4194304
PLANMEM_PEAK\t0\t6\t1048576
PLANMEM_PEAK\t0\t6\t1572864
PLANMEM_PLAN_ATTEMPT\tcopy\t0\tsuccess
PLANMEM_UB_ORACLE_COMPLETE\t0
PLANMEM_RUN_RESULT\tsuccess
"""
UB_RESULT = """
PLANMEM_REQUIRED\t0\t6\t1700000
UB overflow, requires 1700000 bits while 1572864 bits available!
PLANMEM_RUN_RESULT\tfailure
"""
L1_RESULT = "L1 overflow, requires 5000000 bits while 4194304 bits available!"


def _semantic_result(seed, *, status="success", peak=1024, capacity=196608 * 8):
    return {
        "status": status,
        "overflow_scope": "UB" if status == "overflow" else None,
        "actual_peak_bits": peak,
        "capacity_bits": capacity,
        "selected_seed": seed if seed >= 0 else 0,
        "returncode": 2 if status == "overflow" else 0,
    }


def test_planmemory_parsers_are_scope_and_attempt_specific():
    assert oracle.parse_planmemory_peak(SUCCESS, attempt=0, scope="6") == 1572864
    assert oracle.parse_completed_attempt(SUCCESS) == 0
    with pytest.raises(oracle.OracleUnavailable):
        oracle.parse_planmemory_peak(SUCCESS, attempt=1, scope="6")
    assert oracle.parse_overflow_scope(UB_RESULT) == "UB"
    assert oracle.parse_overflow_scope(L1_RESULT) == "L1"
    assert oracle.parse_overflow_scope(
        "ub overflow, requires 1700000 bits while 1572864 bits available!"
    ) == "UB"
    assert oracle.classify_failure(
        "ub overflow, requires 1700000 bits while 1572864 bits available!"
    )["overflow_scope"] == "UB"


def test_completed_attempt_selects_last_retry_result():
    text = """
PLANMEM_PEAK\t0\t6\t100
PLANMEM_PLAN_ATTEMPT\tcopy\t0\tfailure
PLANMEM_PEAK\t4\t6\t200
PLANMEM_PLAN_ATTEMPT\tcopy\t4\tsuccess
PLANMEM_UB_ORACLE_COMPLETE\t-1
PLANMEM_RUN_RESULT\tsuccess
"""
    attempt = oracle.parse_completed_attempt(text)
    assert attempt == 4
    assert oracle.parse_planmemory_peak(text, attempt, "6") == 200


def test_auto_tile_outcome_comes_from_real_stage_snapshot():
    assert oracle.detect_auto_tile_outcome(
        "module {}", '"hivm.hir.get_sub_block_idx"() : () -> i64'
    ) is True
    assert oracle.detect_auto_tile_outcome("module {}", "module {}") is False
    with pytest.raises(oracle.OracleUnavailable):
        oracle.detect_auto_tile_outcome(
            '"hivm.hir.get_sub_block_idx"() : () -> i64',
            '"hivm.hir.get_sub_block_idx"() : () -> i64',
        )


def test_direct_copy_boundary_extracts_one_static_local_allocation():
    assert oracle.parse_direct_copy_boundary_allocation_bytes(
        '  %0 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> '
        ': () -> memref<1024xf32>\n'
    ) == 4096
    assert oracle.parse_direct_copy_boundary_allocation_bytes(
        '  %0 = "memref.alloc"() : () -> memref<17xbf16>\n'
    ) == 34


def test_binary_add_boundary_extracts_two_static_local_allocations():
    text = (
        '%0 = "memref.alloc"() : () -> memref<1024xf32>\n'
        '%1 = "memref.alloc"() : () -> memref<1024xf32>\n'
    )
    assert oracle.parse_boundary_allocation_bytes(text) == [4096, 4096]


@pytest.mark.parametrize(
    "text",
    [
        "module {}",
        '%0 = "memref.alloc"() : () -> memref<?xf32>',
        '%0 = "memref.alloc"() : () -> memref<16xf32>\n%1 = "memref.alloc"() : () -> memref<8xf32>',
    ],
)
def test_direct_copy_boundary_rejects_unknown_or_multiple_allocations(text):
    with pytest.raises(oracle.OracleUnavailable):
        oracle.parse_direct_copy_boundary_allocation_bytes(text)


def test_suffix_runner_reads_real_auto_tile_stage_snapshot(monkeypatch, tmp_path):
    compiler = tmp_path / "suffix-compiler"
    compiler.write_text("#!/bin/sh\n", encoding="utf-8")
    compiler.chmod(0o755)
    input_path = tmp_path / "before.mlir"
    input_path.write_text("module {}", encoding="utf-8")

    def run(command, **_kwargs):
        assert Path(_kwargs["cwd"]).is_dir()
        assert "--tile-mix-cube-loop=1" in command
        assert "--tile-mix-vector-loop=1" in command
        dump_option = next(item for item in command if item.startswith("--dump-stage-oracle-dir="))
        stage_dir = Path(dump_option.split("=", 1)[1])
        stage_dir.mkdir(parents=True)
        (stage_dir / "post-7-TileAndBindSubBlock.generic.mlir").write_text(
            '"hivm.hir.get_sub_block_idx"() : () -> i64', encoding="utf-8"
        )
        return oracle.subprocess.CompletedProcess(command, 0, SUCCESS, "")

    monkeypatch.setattr(oracle.subprocess, "run", run)
    result = oracle.run_suffix_compiler(
        compiler, input_path, 0,
        ["--tile-mix-cube-loop=1", "--tile-mix-vector-loop=1"],
    )
    assert result["status"] == "success"
    assert result["actual_peak_bits"] == 1572864
    assert result["auto_tile_and_bind_subblock_outcome"] is True


def test_suffix_runner_rejects_missing_auto_tile_snapshot(monkeypatch, tmp_path):
    compiler = tmp_path / "suffix-compiler"
    compiler.write_text("#!/bin/sh\n", encoding="utf-8")
    compiler.chmod(0o755)
    input_path = tmp_path / "before.mlir"
    input_path.write_text("module {}", encoding="utf-8")
    monkeypatch.setattr(
        oracle.subprocess,
        "run",
        lambda command, **_kwargs: oracle.subprocess.CompletedProcess(command, 0, SUCCESS, ""),
    )
    with pytest.raises(oracle.OracleUnavailable, match="snapshot"):
        oracle.run_suffix_compiler(compiler, input_path, 0)


def test_semantic_model_runner_parses_exact_result(monkeypatch, tmp_path):
    model = tmp_path / "semantic-model"
    model.write_text("#!/bin/sh\n", encoding="utf-8")
    model.chmod(0o755)
    input_path = tmp_path / "before.mlir"
    input_path.write_text("module {}", encoding="utf-8")
    payload = {
        "precision": "exact",
        "oracle": "cvpipelining_to_plan_memory",
        "status": "success",
        "capacity_bits": 196608 * 8,
        "ub_peak_bits": 32768,
        "functions": [{"status": "success", "selected_seed": 7, "ub_peak_bits": 32768}],
    }

    def run(command, **_kwargs):
        assert f"--before-cvpipelining-ir={input_path.resolve()}" in command
        assert "--format=json" in command
        assert "--random-seed=7" in command
        assert "--tile-mix-cube-loop=1" in command
        assert "--tile-mix-vector-loop=1" in command
        return oracle.subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(oracle.subprocess, "run", run)
    assert oracle.run_semantic_model(
        model, input_path, 7,
        ["--tile-mix-cube-loop=1", "--tile-mix-vector-loop=1"],
    ) == {
        "status": "success",
        "overflow_scope": None,
        "actual_peak_bits": 32768,
        "capacity_bits": 196608 * 8,
        "selected_seed": 7,
        "returncode": 0,
    }


def test_suffix_pipeline_arguments_bind_multibuffer_mode():
    assert oracle.suffix_pipeline_arguments({
        "multibuffer": True,
        "tile_mix_cube_loop": 1,
        "tile_mix_vector_loop": 2,
    }) == [
        "--enable-auto-multi-buffer=true",
        "--tile-mix-cube-loop=1",
        "--tile-mix-vector-loop=2",
    ]


@pytest.mark.parametrize(
    "update",
    [
        {"precision": "conservative"},
        {"oracle": "different"},
        {"functions": [{"status": "success", "selected_seed": 8, "ub_peak_bits": 32768}]},
    ],
)
def test_semantic_model_rejects_non_exact_or_wrong_seed(update):
    payload = {
        "precision": "exact",
        "oracle": "cvpipelining_to_plan_memory",
        "status": "success",
        "capacity_bits": 196608 * 8,
        "ub_peak_bits": 32768,
        "functions": [{"status": "success", "selected_seed": 7, "ub_peak_bits": 32768}],
    }
    payload.update(update)
    with pytest.raises(oracle.OracleUnavailable):
        oracle.parse_semantic_model_result(payload, 7)


def test_failure_classification_keeps_memory_scopes_distinct():
    assert oracle.classify_failure(UB_RESULT)["actual_peak_bits"] == 1700000
    assert oracle.classify_failure(L1_RESULT)["overflow_scope"] == "L1"
    with pytest.raises(oracle.OracleUnavailable):
        oracle.classify_failure("input could not be parsed")


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0-2", [0, 1, 2]), ("0,2,4", [0, 2, 4]), ("-1", [-1]), ("0-1,3", [0, 1, 3])],
)
def test_parse_seeds(text, expected):
    assert oracle.parse_seeds(text) == expected


@pytest.mark.parametrize("text", ["2-1", "1,1", "", "x"])
def test_parse_seeds_rejects_ambiguous_values(text):
    with pytest.raises(Exception):
        oracle.parse_seeds(text)


def _write_manifest(root, **updates):
    (root / "case.ttir").write_text("module {}", encoding="utf-8")
    (root / "case.mlir").write_text("module {}", encoding="utf-8")
    case = {
        "name": "case",
        "operation_family": "direct-copy",
        "ttir": "case.ttir",
        "before_cvpipelining": "case.mlir",
        "arch": "Ascend910B",
        "options": {
            "compile_mode": "simd",
            "multibuffer": False,
            "tile_mix_cube_loop": 2,
            "tile_mix_vector_loop": 2,
        },
        "expected_analyzer_decision": "defer",
        "contract_proposal": {
            "expected_resource_count": 1,
            "expected_source_elements": 65536,
            "expected_element_bit_width": 32,
            "expected_input_payload_bytes": 262144,
            "materialization_stage": "ttir.triton-to-linalg",
            "max_tiles": 64,
            "auto_tile_and_bind_subblock_outcome": False,
        },
    }
    case.update(updates)
    path = root / "manifest.json"
    path.write_text(json.dumps({"schema": "ttir-ub-oracle-v1", "cases": [case]}), encoding="utf-8")
    return path


def test_manifest_resolves_files_and_rejects_identity_override(tmp_path):
    manifest = oracle.load_manifest(_write_manifest(tmp_path))
    assert manifest["cases"][0]["ttir"] == (tmp_path / "case.ttir").resolve()
    path = _write_manifest(tmp_path, pipeline_identity="caller-value")
    with pytest.raises(oracle.ManifestError):
        oracle.load_manifest(path)


def test_manifest_requires_family_specific_resource_count(tmp_path):
    path = _write_manifest(tmp_path, operation_family="binary-add")
    with pytest.raises(oracle.ManifestError, match="resource_count"):
        oracle.load_manifest(path)


@pytest.mark.parametrize(
    "proposal_update",
    [
        {"expected_element_bit_width": 7},
        {"expected_element_bit_width": 12},
        {"expected_input_payload_bytes": 1},
        {
            "expected_source_elements": 1 << 63,
            "expected_element_bit_width": 8,
            "expected_input_payload_bytes": 1 << 63,
        },
    ],
)
def test_manifest_requires_payload_to_match_source_facts(tmp_path, proposal_update):
    path = _write_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["cases"][0]["contract_proposal"].update(proposal_update)
    path.write_text(json.dumps(manifest))
    with pytest.raises(oracle.ManifestError):
        oracle.load_manifest(path)


def test_manifest_allows_oracle_to_derive_auto_tile_outcome(tmp_path):
    path = _write_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["cases"][0]["contract_proposal"][
        "auto_tile_and_bind_subblock_outcome"
    ] = None
    path.write_text(json.dumps(manifest))
    loaded = oracle.load_manifest(path)
    assert loaded["cases"][0]["contract_proposal"][
        "auto_tile_and_bind_subblock_outcome"
    ] is None


def test_manifest_allows_analyzer_decision_without_a_golden_expectation(tmp_path):
    path = _write_manifest(tmp_path, expected_analyzer_decision=None)
    loaded = oracle.load_manifest(path)
    assert loaded["cases"][0]["expected_analyzer_decision"] is None


def test_manifest_paths_stay_with_fixtures(tmp_path):
    outside = tmp_path.parent / "outside.ttir"
    outside.write_text("module {}", encoding="utf-8")
    path = _write_manifest(tmp_path, ttir="../outside.ttir")
    with pytest.raises(oracle.ManifestError):
        oracle.load_manifest(path)


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"compile_mode": "simd", "multibuffer": True,
          "tile_mix_cube_loop": 2, "tile_mix_vector_loop": 2},
         "restricted"),
        ({"compile_mode": "simt", "multibuffer": False,
          "tile_mix_cube_loop": 2, "tile_mix_vector_loop": 2},
         "compile_mode=simd"),
        ({"compile_mode": "simd", "tile_mix_cube_loop": 2,
          "tile_mix_vector_loop": 2},
         "options fields"),
    ],
)
def test_manifest_restricts_first_direct_copy_profile(options, error, tmp_path):
    with pytest.raises(oracle.ManifestError, match=error):
        oracle.load_manifest(_write_manifest(tmp_path, options=options))


def test_manifest_accepts_exact_loop_multibuffer_slice(tmp_path):
    path = _write_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    case = manifest["cases"][0]
    case["operation_family"] = "loop-carried-add"
    case["options"]["multibuffer"] = True
    proposal = case["contract_proposal"]
    proposal["expected_resource_count"] = 2
    proposal["max_tiles"] = 1
    proposal["expected_step_input_instances"] = 2
    path.write_text(json.dumps(manifest))

    loaded = oracle.load_manifest(path)
    assert loaded["cases"][0]["contract_proposal"][
        "expected_step_input_instances"
    ] == 2


def test_proposed_contract_chain_transitions_at_materialization_stage():
    identity = {"sha256": "identity"}
    stages = [
        {"stage_name": "before", "options": {}},
        {"stage_name": "ttir.triton-to-linalg", "options": {"named_ops": "true"}},
        {"stage_name": "after", "options": {}},
    ]
    proposal = {
        "expected_resource_count": 1,
        "expected_source_elements": 65536,
        "expected_element_bit_width": 32,
        "expected_input_payload_bytes": 262144,
        "materialization_stage": "ttir.triton-to-linalg",
        "max_tiles": 64,
        "auto_tile_and_bind_subblock_outcome": False,
    }
    profile = oracle.build_proposed_contract_profile(identity, stages, proposal)
    bindings = profile["profiles"][0]["pipeline_stages"]
    assert [binding["contract_id"] for binding in bindings] == [
        "direct-copy-preserve", "direct-copy-max-tiles", "direct-copy-preserve"
    ]
    assert bindings[0]["contract_parameters"]["expected_input_payload_bytes"] == "262144"
    assert bindings[1]["contract_parameters"]["max_tiles"] == "64"
    assert bindings[2]["contract_parameters"]["expected_input_payload_bytes"] == "4096"


def test_binary_add_contract_chain_uses_binary_contracts():
    identity = {"sha256": "identity"}
    stages = [
        {"stage_name": "ttir.triton-to-linalg", "options": {}},
        {"stage_name": "after", "options": {}},
    ]
    proposal = {
        "expected_resource_count": 2,
        "expected_source_elements": 65536,
        "expected_element_bit_width": 32,
        "expected_input_payload_bytes": 262144,
        "materialization_stage": "ttir.triton-to-linalg",
        "max_tiles": 64,
        "auto_tile_and_bind_subblock_outcome": False,
    }
    profile = oracle.build_proposed_contract_profile(
        identity, stages, proposal, "binary-add"
    )
    bindings = profile["profiles"][0]["pipeline_stages"]
    assert [binding["contract_id"] for binding in bindings] == [
        "binary-add-max-tiles", "binary-add-preserve"
    ]
    assert bindings[1]["contract_parameters"]["expected_input_payload_bytes"] == "4096"


def test_loop_carried_contract_chain_keeps_full_iteration_payload():
    proposal = {
        "expected_resource_count": 2,
        "expected_source_elements": 65536,
        "expected_element_bit_width": 32,
        "expected_input_payload_bytes": 262144,
        "materialization_stage": "ttir.triton-to-linalg",
        "max_tiles": 1,
        "auto_tile_and_bind_subblock_outcome": False,
    }
    stages = [
        {"stage_name": "ttir.triton-to-linalg", "options": {}},
        {"stage_name": "bisheng.ub-affecting-suffix", "options": {}},
    ]
    profile = oracle.build_proposed_contract_profile(
        {"sha256": "identity"}, stages, proposal, "loop-carried-add"
    )
    bindings = profile["profiles"][0]["pipeline_stages"]
    assert [binding["contract_id"] for binding in bindings] == [
        "loop-carried-add-max-tiles", "loop-carried-add-preserve"
    ]
    assert bindings[0]["contract_parameters"]["max_tiles"] == "1"
    assert bindings[1]["contract_parameters"][
        "expected_input_payload_bytes"
    ] == "262144"


def test_loop_carried_multibuffer_chain_raises_step_input_instances():
    proposal = {
        "expected_resource_count": 2,
        "expected_source_elements": 65536,
        "expected_element_bit_width": 32,
        "expected_input_payload_bytes": 262144,
        "materialization_stage": "ttir.triton-to-linalg",
        "max_tiles": 1,
        "expected_step_input_instances": 2,
        "auto_tile_and_bind_subblock_outcome": False,
    }
    stages = [
        {"stage_name": "ttir.triton-to-linalg", "options": {}},
        {"stage_name": "bisheng.ub-affecting-suffix", "options": {}},
    ]
    profile = oracle.build_proposed_contract_profile(
        {"sha256": "identity"}, stages, proposal, "loop-carried-add"
    )
    bindings = profile["profiles"][0]["pipeline_stages"]
    assert [binding["contract_id"] for binding in bindings] == [
        "loop-carried-add-max-tiles",
        "loop-carried-add-multibuffer",
    ]
    assert bindings[1]["contract_parameters"][
        "expected_step_input_instances"
    ] == "2"


def test_reshape_copy_contract_chain_uses_view_contracts():
    proposal = {
        "expected_resource_count": 2,
        "expected_source_elements": 65536,
        "expected_element_bit_width": 32,
        "expected_input_payload_bytes": 262144,
        "materialization_stage": "ttir.triton-to-linalg",
        "max_tiles": 64,
        "auto_tile_and_bind_subblock_outcome": False,
    }
    profile = oracle.build_proposed_contract_profile(
        {"sha256": "identity"},
        [{"stage_name": "ttir.triton-to-linalg", "options": {}}],
        proposal,
        "reshape-copy",
    )
    assert profile["profiles"][0]["pipeline_stages"][0]["contract_id"] == \
        "reshape-copy-max-tiles"


def test_reduction_sum_contract_chain_models_suffix_extra_buffer():
    proposal = {
        "expected_resource_count": 3,
        "expected_source_elements": 65536,
        "expected_element_bit_width": 32,
        "expected_input_payload_bytes": 262144,
        "materialization_stage": "ttir.triton-to-linalg",
        "max_tiles": 1,
        "auto_tile_and_bind_subblock_outcome": False,
    }
    stages = [
        {"stage_name": "ttir.triton-to-linalg", "options": {}},
        {"stage_name": "bisheng.ub-affecting-suffix", "options": {}},
    ]
    profile = oracle.build_proposed_contract_profile(
        {"sha256": "identity"}, stages, proposal, "reduction-sum"
    )
    bindings = profile["profiles"][0]["pipeline_stages"]
    assert [binding["contract_id"] for binding in bindings] == [
        "reduction-sum-max-tiles", "reduction-sum-extra-buffer"
    ]
    assert bindings[0]["contract_parameters"]["max_tiles"] == "1"
    assert bindings[1]["contract_parameters"][
        "expected_scratch_payload_bytes"
    ] == "131072"
    assert bindings[1]["contract_parameters"][
        "expected_accumulator_payload_bytes"
    ] == "4"


def test_proposed_contract_chain_requires_unique_materialization_stage():
    with pytest.raises(oracle.OracleUnavailable, match="exactly once"):
        oracle.build_proposed_contract_profile(
            {"sha256": "identity"},
            [{"stage_name": "other", "options": {}}],
            {
                "expected_resource_count": 1,
                "expected_source_elements": 65536,
                "expected_element_bit_width": 32,
                "expected_input_payload_bytes": 262144,
                "materialization_stage": "ttir.triton-to-linalg",
                "max_tiles": 64,
                "auto_tile_and_bind_subblock_outcome": False,
            },
        )


def _analysis(
    decision="defer", lower_bound_bytes=0, operation_family="direct-copy",
    multibuffer=False,
):
    is_binary_add = operation_family == "binary-add"
    is_loop_carried = operation_family == "loop-carried-add"
    is_reshape_copy = operation_family == "reshape-copy"
    is_reduction = operation_family == "reduction-sum"
    materialization_id = (
        "reduction-sum-max-tiles" if is_reduction else
        "loop-carried-add-max-tiles" if is_loop_carried else
        "binary-add-max-tiles" if is_binary_add else
        "reshape-copy-max-tiles" if is_reshape_copy else
        "direct-copy-max-tiles"
    )
    matcher_trace = (
        "ttir-reduction-sum-v1" if is_reduction else
        "ttir-loop-carried-add-v1" if is_loop_carried else
        "ttir-binary-add-v1" if is_binary_add else
        "ttir-reshape-copy-v1" if is_reshape_copy else
        "ttir-direct-load-v1"
    )
    contract_trace = [matcher_trace, materialization_id]
    if is_reduction:
        contract_trace.append("reduction-sum-extra-buffer")
    elif is_loop_carried and multibuffer:
        contract_trace.append("loop-carried-add-multibuffer")
    certificate = {
        "kind": "witness" if (
            is_binary_add or is_loop_carried or is_reduction
        ) else "singleton",
        "bytes": lower_bound_bytes,
        "resource_ids": [0, 1, 2] if is_reduction else (
            [0, 1] if (is_binary_add or is_loop_carried) else [0]
        ),
        "contract_trace": contract_trace,
    }
    common_parameters = {
        "expected_resource_count": "3" if is_reduction else (
            "2" if (
                is_binary_add or is_loop_carried or is_reshape_copy
            ) else "1"
        ),
        "expected_source_elements": "65536",
        "expected_element_bit_width": "32",
        "expected_input_payload_bytes": "262144",
    }
    if is_reduction:
        common_parameters.update({
            "expected_scratch_payload_bytes": "131072",
            "expected_accumulator_payload_bytes": "4",
        })
    materialization_parameters = {**common_parameters, "max_tiles": (
        "1" if (is_reduction or is_loop_carried) else "64"
    )}
    stages = [{
        "stage_name": "ttir.triton-to-linalg",
        "options": {},
        "contract_id": materialization_id,
        "contract_version": "1",
        "contract_parameters": materialization_parameters,
    }]
    if is_reduction:
        stages.append({
            "stage_name": "bisheng.ub-affecting-suffix",
            "options": {},
            "contract_id": "reduction-sum-extra-buffer",
            "contract_version": "1",
            "contract_parameters": common_parameters,
        })
    elif is_loop_carried and multibuffer:
        stages.append({
            "stage_name": "bisheng.ub-affecting-suffix",
            "options": {},
            "contract_id": "loop-carried-add-multibuffer",
            "contract_version": "1",
            "contract_parameters": {
                **common_parameters,
                "expected_step_input_instances": "2",
            },
        })
    return {
        "operation_family": operation_family,
        "decision": decision,
        "lower_bound_bytes": lower_bound_bytes,
        "capacity_bytes": 196608,
        "certificates": ([] if lower_bound_bytes == 0 else [certificate]),
        "unsupported_reasons": ([] if lower_bound_bytes else ["unknown-pipeline-profile"]),
        "pipeline_identity": "identity",
        "pipeline_identity_detail": {"sha256": "identity", "target_arch": "Ascend910B"},
        "pipeline_stages_detail": stages,
        "auto_tile_and_bind_subblock_outcome": False,
        "before_cvpipelining_allocations_bytes": (
            [262144] if is_reduction else
            [262144, 262144] if is_loop_carried else
            [4096, 4096] if is_binary_add else [4096]
        ),
        "before_cvpipelining_allocation_bytes": (
            262144 if is_reduction else
            524288 if is_loop_carried else
            8192 if is_binary_add else 4096
        ),
        "ttir_fixture_sha256": "a" * 64,
        "before_cvpipelining_sha256": "b" * 64,
        "contract_version": "ttir-ub-lb-v1",
    }


def test_binary_add_certificate_requires_two_resource_witness():
    analysis = _analysis("defer", 8192, "binary-add")
    assert oracle.has_valid_certificate(analysis)
    analysis["certificates"][0]["kind"] = "singleton"
    assert not oracle.has_valid_certificate(analysis)
    analysis["certificates"][0]["kind"] = "witness"
    analysis["certificates"][0]["resource_ids"] = [0, 0]
    assert not oracle.has_valid_certificate(analysis)


def test_binary_add_materialization_bridge_requires_two_exact_allocations():
    analysis = _analysis("defer", 8192, "binary-add")
    assert oracle.has_valid_materialization_bridge(analysis)
    analysis["before_cvpipelining_allocations_bytes"] = [8192]
    assert not oracle.has_valid_materialization_bridge(analysis)


def test_reshape_copy_bridge_maps_two_logical_resources_to_one_allocation():
    analysis = _analysis("defer", 4096, "reshape-copy")
    assert oracle.has_valid_certificate(analysis)
    assert oracle.has_valid_materialization_bridge(analysis)


def test_reduction_sum_bridge_adds_validated_suffix_resources():
    analysis = _analysis("reject", 393220, "reduction-sum")
    assert oracle.has_valid_certificate(analysis)
    assert oracle.has_valid_materialization_bridge(analysis)
    analysis["lower_bound_bytes"] -= 4
    assert not oracle.has_valid_materialization_bridge(analysis)


def test_loop_multibuffer_bridge_counts_two_step_input_instances():
    analysis = _analysis(
        "reject", 786432, "loop-carried-add", multibuffer=True
    )
    assert oracle.has_valid_certificate(analysis)
    assert oracle.has_valid_materialization_bridge(analysis)
    analysis["pipeline_stages_detail"][1]["contract_parameters"][
        "expected_step_input_instances"
    ] = "3"
    assert not oracle.has_valid_materialization_bridge(analysis)


def test_evaluate_accepts_available_defer_results(tmp_path):
    manifest = oracle.load_manifest(_write_manifest(tmp_path))

    def run(_compiler, _input, seed, _pipeline_arguments):
        return {
            "seed": seed, "status": "success", "overflow_scope": None,
            "actual_peak_bits": 1024, "auto_tile_and_bind_subblock_outcome": False,
        }

    report = oracle.evaluate(manifest, Path("compiler"), [0, 1], True, lambda _case: _analysis(), run)
    assert report["summary"] == {
        "cases": 1, "seeds": [0, 1], "retry_checked": True,
        "semantic_replay_checked": False, "violations": 0, "unavailable": 0
    }
    assert [item["seed"] for item in report["cases"][0]["runs"]] == [0, 1, -1]


def test_evaluate_derives_one_consistent_auto_tile_outcome(tmp_path):
    path = _write_manifest(tmp_path)
    raw = json.loads(path.read_text())
    raw["cases"][0]["contract_proposal"]["auto_tile_and_bind_subblock_outcome"] = None
    path.write_text(json.dumps(raw))
    manifest = oracle.load_manifest(path)
    analysis = _analysis()
    analysis["auto_tile_and_bind_subblock_outcome"] = None

    def run(_compiler, _input, seed, _pipeline_arguments):
        return {
            "seed": seed, "status": "success", "overflow_scope": None,
            "actual_peak_bits": 1024, "auto_tile_and_bind_subblock_outcome": False,
        }

    report = oracle.evaluate(manifest, Path("compiler"), [0, 1], True, lambda _case: analysis, run)
    assert report["violations"] == []
    assert report["cases"][0]["analysis"][
        "auto_tile_and_bind_subblock_outcome"
    ] is False


def test_evaluate_accepts_actual_decision_when_fixture_has_no_golden_expectation(tmp_path):
    manifest = oracle.load_manifest(
        _write_manifest(tmp_path, expected_analyzer_decision=None)
    )

    def run(_compiler, _input, seed, _pipeline_arguments):
        return {
            "seed": seed,
            "status": "overflow",
            "overflow_scope": "UB",
            "actual_peak_bits": 4096,
            "auto_tile_and_bind_subblock_outcome": False,
        }

    report = oracle.evaluate(
        manifest, Path("compiler"), [0], False,
        lambda _case: _analysis("reject", 256), run,
    )
    assert report["violations"] == []


def test_evaluate_keeps_explicit_golden_decision_as_an_assertion(tmp_path):
    manifest = oracle.load_manifest(_write_manifest(tmp_path))

    def run(_compiler, _input, seed, _pipeline_arguments):
        return {
            "seed": seed,
            "status": "overflow",
            "overflow_scope": "UB",
            "actual_peak_bits": 4096,
            "auto_tile_and_bind_subblock_outcome": False,
        }

    report = oracle.evaluate(
        manifest, Path("compiler"), [0], False,
        lambda _case: _analysis("reject", 256), run,
    )
    assert {item["kind"] for item in report["violations"]} == {
        "unexpected-analyzer-decision"
    }


def test_evaluate_rejects_seed_dependent_auto_tile_outcome(tmp_path):
    path = _write_manifest(tmp_path)
    raw = json.loads(path.read_text())
    raw["cases"][0]["contract_proposal"]["auto_tile_and_bind_subblock_outcome"] = None
    path.write_text(json.dumps(raw))
    manifest = oracle.load_manifest(path)
    analysis = _analysis()
    analysis["auto_tile_and_bind_subblock_outcome"] = None

    def run(_compiler, _input, seed, _pipeline_arguments):
        return {
            "seed": seed, "status": "success", "overflow_scope": None,
            "actual_peak_bits": 1024,
            "auto_tile_and_bind_subblock_outcome": seed == 1,
        }

    report = oracle.evaluate(manifest, Path("compiler"), [0, 1], False, lambda _case: analysis, run)
    assert {item["kind"] for item in report["violations"]} == {
        "auto-tile-outcome-nondeterministic"
    }


def test_evaluate_detects_invalid_lower_bound_and_reject_result(tmp_path):
    path = _write_manifest(tmp_path, expected_analyzer_decision="reject")
    manifest = oracle.load_manifest(path)

    def run(_compiler, _input, seed, _pipeline_arguments):
        return {
            "seed": seed, "status": "success", "overflow_scope": None,
            "actual_peak_bits": 1024, "auto_tile_and_bind_subblock_outcome": False,
        }

    report = oracle.evaluate(manifest, Path("compiler"), [0], False, lambda _case: _analysis("reject", 256), run)
    assert {item["kind"] for item in report["violations"]} == {
        "lower-bound-exceeds-actual", "reject-without-ub-capacity-result"
    }


def test_evaluate_requires_semantic_replay_to_match_real_suffix(tmp_path):
    manifest = oracle.load_manifest(_write_manifest(tmp_path))

    def run(_compiler, _input, seed, _pipeline_arguments):
        return {
            "seed": seed, "status": "success", "overflow_scope": None,
            "actual_peak_bits": 1024, "auto_tile_and_bind_subblock_outcome": False,
        }

    report = oracle.evaluate(
        manifest, Path("compiler"), [0], False, lambda _case: _analysis(), run,
        Path("semantic-model"),
        lambda _model, _input, seed, _pipeline_arguments: _semantic_result(seed),
    )
    assert report["summary"]["semantic_replay_checked"] is True
    assert report["violations"] == []
    assert report["cases"][0]["runs"][0]["semantic_replay"] == _semantic_result(0)

    mismatch = oracle.evaluate(
        manifest, Path("compiler"), [0], False, lambda _case: _analysis(), run,
        Path("semantic-model"),
        lambda _model, _input, seed, _pipeline_arguments: _semantic_result(
            seed, peak=2048
        ),
    )
    assert {item["kind"] for item in mismatch["violations"]} == {
        "semantic-replay-result-mismatch"
    }


def test_evaluate_rejects_certificate_from_a_different_contract_chain(tmp_path):
    manifest = oracle.load_manifest(_write_manifest(tmp_path))
    analysis = _analysis("defer", 256)
    analysis["certificates"][0]["contract_trace"] = ["ttir-direct-load-v1", "different-contract"]

    def run(_compiler, _input, seed, _pipeline_arguments):
        return {
            "seed": seed, "status": "success", "overflow_scope": None,
            "actual_peak_bits": 4096, "auto_tile_and_bind_subblock_outcome": False,
        }

    report = oracle.evaluate(manifest, Path("compiler"), [0], False, lambda _case: analysis, run)
    assert {item["kind"] for item in report["violations"]} == {
        "invalid-certificate-contract-trace"
    }


def test_evaluate_rejects_invalid_materialization_bridge(tmp_path):
    manifest = oracle.load_manifest(_write_manifest(tmp_path))
    analysis = _analysis()
    analysis["before_cvpipelining_allocations_bytes"] = [2048]

    def run(_compiler, _input, seed, _pipeline_arguments):
        return {
            "seed": seed, "status": "success", "overflow_scope": None,
            "actual_peak_bits": 4096, "auto_tile_and_bind_subblock_outcome": False,
        }

    report = oracle.evaluate(manifest, Path("compiler"), [0], False, lambda _case: analysis, run)
    assert {item["kind"] for item in report["violations"]} == {
        "invalid-materialization-bridge"
    }


def test_profile_candidate_requires_complete_certificate(tmp_path):
    manifest = oracle.load_manifest(_write_manifest(tmp_path))
    run = lambda _compiler, _input, seed, _pipeline_arguments: {
        "seed": seed, "status": "success", "overflow_scope": None, "actual_peak_bits": 1024,
        "auto_tile_and_bind_subblock_outcome": False,
    }
    report = oracle.evaluate(manifest, Path("compiler"), [0], False, lambda _case: _analysis(), run)
    with pytest.raises(oracle.OracleUnavailable):
        oracle.build_profile_candidate(report)


def test_profile_candidate_uses_outcome_derived_from_all_real_runs(tmp_path):
    path = _write_manifest(tmp_path, expected_analyzer_decision="reject")
    raw = json.loads(path.read_text())
    raw["cases"][0]["contract_proposal"]["auto_tile_and_bind_subblock_outcome"] = None
    path.write_text(json.dumps(raw))
    manifest = oracle.load_manifest(path)
    analysis = _analysis("reject", 256)
    analysis["auto_tile_and_bind_subblock_outcome"] = None

    def run(_compiler, _input, seed, _pipeline_arguments):
        return {
            "seed": seed,
            "status": "overflow",
            "overflow_scope": "UB",
            "actual_peak_bits": 4096,
            "auto_tile_and_bind_subblock_outcome": False,
        }

    report = oracle.evaluate(
        manifest,
        Path("compiler"),
        list(range(20)),
        True,
        lambda _case: analysis,
        run,
        Path("semantic-model"),
        lambda _model, _input, seed, _pipeline_arguments: _semantic_result(
            seed, status="overflow", peak=4096
        ),
    )
    report["suffix_compiler_sha256"] = "c" * 64
    report["semantic_model_sha256"] = "d" * 64
    assert report["violations"] == []
    assert report["unavailable"] == []
    candidate = oracle.build_profile_candidate(report)
    assert candidate["profiles"][0]["auto_tile_and_bind_subblock_outcome"] is False


def test_profile_candidate_rejects_an_empty_case_set():
    report = {
        "schema": "ttir-ub-oracle-report-v1",
        "suffix_compiler_sha256": "c" * 64,
        "cases": [],
        "violations": [],
        "unavailable": [],
        "summary": {
            "cases": 0,
            "seeds": list(range(20)),
            "retry_checked": True,
            "violations": 0,
            "unavailable": 0,
        },
    }
    with pytest.raises(oracle.OracleUnavailable, match="non-empty"):
        oracle.build_profile_candidate(report)


def test_profile_candidate_records_exact_identity_and_retry(tmp_path):
    analysis_false = _analysis("reject", 256)
    analysis_true = _analysis("reject", 256)
    analysis_true["pipeline_identity_detail"] = {"sha256": "identity-true", "target_arch": "Ascend910B"}
    analysis_true["auto_tile_and_bind_subblock_outcome"] = True
    runs_false = [
        {
            "seed": seed, "status": "overflow", "overflow_scope": "UB", "actual_peak_bits": 4096,
            "auto_tile_and_bind_subblock_outcome": False,
        }
        for seed in list(range(20)) + [-1]
    ]
    for run in runs_false:
        run["semantic_replay"] = _semantic_result(run["seed"], status="overflow", peak=4096)
    runs_true = [
        {**run, "auto_tile_and_bind_subblock_outcome": True} for run in runs_false
    ]
    report = {
        "schema": "ttir-ub-oracle-report-v1",
        "suffix_compiler_sha256": "c" * 64,
        "semantic_model_sha256": "d" * 64,
        "cases": [
            {"name": "false", "analysis": analysis_false, "runs": runs_false},
            {"name": "true", "analysis": analysis_true, "runs": runs_true},
        ],
        "violations": [],
        "unavailable": [],
        "summary": {
            "cases": 2,
            "seeds": list(range(20)),
            "retry_checked": True,
            "semantic_replay_checked": True,
            "violations": 0,
            "unavailable": 0,
        },
    }
    candidate = oracle.build_profile_candidate(report)
    assert candidate["profiles"][0]["pipeline_identity"]["sha256"] == "identity"
    assert candidate["profiles"][0]["validated_seeds"] == list(range(20))
    assert candidate["profiles"][0]["retry_validated"] is True
    assert candidate["profiles"][0]["semantic_model_sha256"] == "d" * 64
    assert [profile["auto_tile_and_bind_subblock_outcome"] for profile in candidate["profiles"]] == [
        False, True
    ]

    false_only_report = {
        **report,
        "cases": report["cases"][:1],
        "summary": {**report["summary"], "cases": 1},
    }
    false_only_candidate = oracle.build_profile_candidate(false_only_report)
    assert false_only_candidate["profiles"][0]["auto_tile_and_bind_subblock_outcome"] is False


def test_profile_candidate_requires_semantic_replay():
    report = {
        "violations": [],
        "unavailable": [],
        "cases": [{}],
        "suffix_compiler_sha256": "c" * 64,
        "summary": {
            "cases": 1, "seeds": list(range(20)), "retry_checked": True,
            "semantic_replay_checked": False,
        },
    }
    with pytest.raises(oracle.OracleUnavailable, match="semantic replay"):
        oracle.build_profile_candidate(report)


@pytest.mark.parametrize(
    ("violations", "unavailable", "expected"),
    [([], [], 0), ([{"kind": "comparison"}], [], 1), ([], [{"phase": "analyzer"}], 2)],
)
def test_main_exit_status_is_machine_readable(monkeypatch, violations, unavailable, expected):
    report = {
        "schema": "ttir-ub-oracle-report-v1",
        "cases": [],
        "violations": violations,
        "unavailable": unavailable,
        "summary": {"cases": 0, "seeds": [0], "retry_checked": False,
                    "violations": len(violations), "unavailable": len(unavailable)},
    }
    monkeypatch.setattr(oracle, "load_manifest", lambda _path: {"cases": []})
    monkeypatch.setattr(oracle, "evaluate", lambda *_args, **_kwargs: report)
    assert oracle.main(["--manifest", "manifest.json", "--suffix-compiler", "compiler", "--seeds", "0"]) == expected


def test_profile_candidate_requires_saved_report():
    assert oracle.main([
        "--manifest", "manifest.json",
        "--suffix-compiler", "compiler",
        "--profile-candidate", "candidate.json",
    ]) == 2
