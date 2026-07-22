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


def test_planmemory_parsers_are_scope_and_attempt_specific():
    assert oracle.parse_planmemory_peak(SUCCESS, attempt=0, scope="6") == 1572864
    assert oracle.parse_completed_attempt(SUCCESS) == 0
    with pytest.raises(oracle.OracleUnavailable):
        oracle.parse_planmemory_peak(SUCCESS, attempt=1, scope="6")
    assert oracle.parse_overflow_scope(UB_RESULT) == "UB"
    assert oracle.parse_overflow_scope(L1_RESULT) == "L1"


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
        dump_option = next(item for item in command if item.startswith("--dump-stage-oracle-dir="))
        stage_dir = Path(dump_option.split("=", 1)[1])
        stage_dir.mkdir(parents=True)
        (stage_dir / "post-7-TileAndBindSubBlock.generic.mlir").write_text(
            '"hivm.hir.get_sub_block_idx"() : () -> i64', encoding="utf-8"
        )
        return oracle.subprocess.CompletedProcess(command, 0, SUCCESS, "")

    monkeypatch.setattr(oracle.subprocess, "run", run)
    result = oracle.run_suffix_compiler(compiler, input_path, 0)
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
        "ttir": "case.ttir",
        "before_cvpipelining": "case.mlir",
        "arch": "Ascend910B",
        "options": {"compile_mode": "simd", "multibuffer": False},
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


def test_manifest_paths_stay_with_fixtures(tmp_path):
    outside = tmp_path.parent / "outside.ttir"
    outside.write_text("module {}", encoding="utf-8")
    path = _write_manifest(tmp_path, ttir="../outside.ttir")
    with pytest.raises(oracle.ManifestError):
        oracle.load_manifest(path)


@pytest.mark.parametrize(
    "options",
    [
        {"compile_mode": "simd", "multibuffer": True},
        {"compile_mode": "simt", "multibuffer": False},
        {"compile_mode": "simd"},
    ],
)
def test_manifest_restricts_first_direct_copy_profile(options, tmp_path):
    with pytest.raises(oracle.ManifestError, match="compile_mode=simd"):
        oracle.load_manifest(_write_manifest(tmp_path, options=options))


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


def _analysis(decision="defer", lower_bound_bytes=0):
    certificate = {
        "kind": "singleton",
        "bytes": lower_bound_bytes,
        "resource_ids": [0],
        "contract_trace": ["ttir-direct-load-v1", "direct-copy-max-tiles"],
    }
    return {
        "decision": decision,
        "lower_bound_bytes": lower_bound_bytes,
        "capacity_bytes": 196608,
        "certificates": ([] if lower_bound_bytes == 0 else [certificate]),
        "unsupported_reasons": ([] if lower_bound_bytes else ["unknown-pipeline-profile"]),
        "pipeline_identity": "identity",
        "pipeline_identity_detail": {"sha256": "identity", "target_arch": "Ascend910B"},
        "pipeline_stages_detail": [{
            "stage_name": "ttir.triton-to-linalg",
            "options": {},
            "contract_id": "direct-copy-max-tiles",
            "contract_version": "1",
            "contract_parameters": {
                "expected_resource_count": "1",
                "expected_source_elements": "65536",
                "expected_element_bit_width": "32",
                "expected_input_payload_bytes": "262144",
                "max_tiles": "64",
            },
        }],
        "auto_tile_and_bind_subblock_outcome": False,
        "before_cvpipelining_allocation_bytes": 4096,
        "ttir_fixture_sha256": "a" * 64,
        "before_cvpipelining_sha256": "b" * 64,
        "contract_version": "ttir-ub-lb-v1",
    }


def test_evaluate_accepts_available_defer_results(tmp_path):
    manifest = oracle.load_manifest(_write_manifest(tmp_path))

    def run(_compiler, _input, seed):
        return {
            "seed": seed, "status": "success", "overflow_scope": None,
            "actual_peak_bits": 1024, "auto_tile_and_bind_subblock_outcome": False,
        }

    report = oracle.evaluate(manifest, Path("compiler"), [0, 1], True, lambda _case: _analysis(), run)
    assert report["summary"] == {
        "cases": 1, "seeds": [0, 1], "retry_checked": True, "violations": 0, "unavailable": 0
    }
    assert [item["seed"] for item in report["cases"][0]["runs"]] == [0, 1, -1]


def test_evaluate_detects_invalid_lower_bound_and_reject_result(tmp_path):
    path = _write_manifest(tmp_path, expected_analyzer_decision="reject")
    manifest = oracle.load_manifest(path)

    def run(_compiler, _input, seed):
        return {
            "seed": seed, "status": "success", "overflow_scope": None,
            "actual_peak_bits": 1024, "auto_tile_and_bind_subblock_outcome": False,
        }

    report = oracle.evaluate(manifest, Path("compiler"), [0], False, lambda _case: _analysis("reject", 256), run)
    assert {item["kind"] for item in report["violations"]} == {
        "lower-bound-exceeds-actual", "reject-without-ub-capacity-result"
    }


def test_evaluate_rejects_certificate_from_a_different_contract_chain(tmp_path):
    manifest = oracle.load_manifest(_write_manifest(tmp_path))
    analysis = _analysis("defer", 256)
    analysis["certificates"][0]["contract_trace"] = ["ttir-direct-load-v1", "different-contract"]

    def run(_compiler, _input, seed):
        return {
            "seed": seed, "status": "success", "overflow_scope": None,
            "actual_peak_bits": 4096, "auto_tile_and_bind_subblock_outcome": False,
        }

    report = oracle.evaluate(manifest, Path("compiler"), [0], False, lambda _case: analysis, run)
    assert {item["kind"] for item in report["violations"]} == {
        "invalid-certificate-contract-trace"
    }


def test_profile_candidate_requires_complete_certificate(tmp_path):
    manifest = oracle.load_manifest(_write_manifest(tmp_path))
    run = lambda _compiler, _input, seed: {
        "seed": seed, "status": "success", "overflow_scope": None, "actual_peak_bits": 1024,
        "auto_tile_and_bind_subblock_outcome": False,
    }
    report = oracle.evaluate(manifest, Path("compiler"), [0], False, lambda _case: _analysis(), run)
    with pytest.raises(oracle.OracleUnavailable):
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
    runs_true = [
        {**run, "auto_tile_and_bind_subblock_outcome": True} for run in runs_false
    ]
    report = {
        "schema": "ttir-ub-oracle-report-v1",
        "suffix_compiler_sha256": "c" * 64,
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
            "violations": 0,
            "unavailable": 0,
        },
    }
    candidate = oracle.build_profile_candidate(report)
    assert candidate["profiles"][0]["pipeline_identity"]["sha256"] == "identity"
    assert candidate["profiles"][0]["validated_seeds"] == list(range(20))
    assert candidate["profiles"][0]["retry_validated"] is True


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
