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
    with pytest.raises(oracle.OracleUnavailable):
        oracle.parse_planmemory_peak(SUCCESS, attempt=1, scope="6")
    assert oracle.parse_overflow_scope(UB_RESULT) == "UB"
    assert oracle.parse_overflow_scope(L1_RESULT) == "L1"


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


def _analysis(decision="defer", lower_bound_bytes=0):
    return {
        "decision": decision,
        "lower_bound_bytes": lower_bound_bytes,
        "capacity_bytes": 196608,
        "certificates": ([] if lower_bound_bytes == 0 else [{"kind": "singleton"}]),
        "unsupported_reasons": ([] if lower_bound_bytes else ["unknown-pipeline-profile"]),
        "pipeline_identity": "identity",
        "pipeline_identity_detail": {"sha256": "identity", "target_arch": "Ascend910B"},
        "contract_version": "ttir-ub-lb-v1",
    }


def test_evaluate_accepts_available_defer_results(tmp_path):
    manifest = oracle.load_manifest(_write_manifest(tmp_path))

    def run(_compiler, _input, seed):
        return {"seed": seed, "status": "success", "overflow_scope": None, "actual_peak_bits": 1024}

    report = oracle.evaluate(manifest, Path("compiler"), [0, 1], True, lambda _case: _analysis(), run)
    assert report["summary"] == {
        "cases": 1, "seeds": [0, 1], "retry_checked": True, "violations": 0, "unavailable": 0
    }
    assert [item["seed"] for item in report["cases"][0]["runs"]] == [0, 1, -1]


def test_evaluate_detects_invalid_lower_bound_and_reject_result(tmp_path):
    path = _write_manifest(tmp_path, expected_analyzer_decision="reject")
    manifest = oracle.load_manifest(path)

    def run(_compiler, _input, seed):
        return {"seed": seed, "status": "success", "overflow_scope": None, "actual_peak_bits": 1024}

    report = oracle.evaluate(manifest, Path("compiler"), [0], False, lambda _case: _analysis("reject", 256), run)
    assert {item["kind"] for item in report["violations"]} == {
        "lower-bound-exceeds-actual", "reject-without-ub-capacity-result"
    }


def test_profile_candidate_requires_complete_certificate(tmp_path):
    manifest = oracle.load_manifest(_write_manifest(tmp_path))
    run = lambda _compiler, _input, seed: {
        "seed": seed, "status": "success", "overflow_scope": None, "actual_peak_bits": 1024
    }
    report = oracle.evaluate(manifest, Path("compiler"), [0], False, lambda _case: _analysis(), run)
    with pytest.raises(oracle.OracleUnavailable):
        oracle.build_profile_candidate(report)


def test_profile_candidate_records_exact_identity_and_retry(tmp_path):
    manifest_path = _write_manifest(tmp_path, expected_analyzer_decision="reject")
    manifest = oracle.load_manifest(manifest_path)

    def run(_compiler, _input, seed):
        return {"seed": seed, "status": "overflow", "overflow_scope": "UB", "actual_peak_bits": 4096}

    report = oracle.evaluate(manifest, Path("compiler"), [0, 1], True, lambda _case: _analysis("reject", 256), run)
    candidate = oracle.build_profile_candidate(report)
    assert candidate["profiles"][0]["pipeline_identity"]["sha256"] == "identity"
    assert candidate["profiles"][0]["validated_seeds"] == [0, 1]
    assert candidate["profiles"][0]["retry_validated"] is True
