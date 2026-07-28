import json
from pathlib import Path


MATRIX = Path(__file__).with_name("coverage_matrix.json")
P4_EVIDENCE = Path(__file__).with_name("p4_defer_evidence.json")
PHASES = {"P0", "P1", "P2", "P3", "P4", "P5"}
OPERATION_STATUSES = {
    "candidate-supported",
    "deliberate-defer",
    "source-fact-only",
    "replay-shadow",
}
CONTRACT_STATUSES = {
    "implemented",
    "implemented-slice",
    "implemented-factor-2",
    "identity-bound-defer",
    "deliberate-defer",
    "replay-shadow",
    "oracle-validated",
    "candidate-primitive",
}


def load_matrix():
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def test_p0_through_p4_operation_families_have_closed_safe_statuses():
    matrix = load_matrix()
    assert matrix["schema"] == "ttir-ub-coverage-v1"
    assert matrix["completion_boundary"] == "P4"
    families = matrix["operation_families"]
    assert families
    assert {item["phase"] for item in families} <= PHASES
    assert {item["status"] for item in families} <= OPERATION_STATUSES
    assert all(item["operations"] and item["evidence"] for item in families)
    assert {"descriptor-memory", "irregular-memory", "dynamic-cv-mix"} <= {
        item["family"] for item in families
    }


def test_p0_through_p4_contract_inventory_contains_no_todo():
    matrix = load_matrix()
    contracts = matrix["contracts"]
    names = [item["name"] for item in contracts]
    assert len(names) == len(set(names))
    assert {item["phase"] for item in contracts} <= PHASES
    assert {item["status"] for item in contracts} <= CONTRACT_STATUSES
    assert {
        "PipelineApplicabilityContract",
        "DynamicCVPipelineContract",
        "SplitMixContract",
        "DescriptorMemoryContract",
        "IrregularMemoryContract",
        "MultiBufferContract",
        "PlanMemoryInterpretationContract",
    } <= set(names)


def test_p4_unsafe_generalization_is_shadow_or_deliberate_defer():
    matrix = load_matrix()
    for item in matrix["operation_families"]:
        if item["phase"] == "P4":
            assert item["status"] in {"replay-shadow", "deliberate-defer"}
    for item in matrix["contracts"]:
        if item["phase"] == "P4":
            assert item["status"] in {"replay-shadow", "deliberate-defer"}


def test_p5_is_explicitly_fail_closed_until_server_oracle_gates_pass():
    matrix = load_matrix()
    families = {
        item["family"]: item for item in matrix["operation_families"]
        if item["phase"] == "P5"
    }
    assert {"dot-general", "atomic", "custom-ops"} <= set(families)
    assert {
        item["status"] for item in families.values()
    } == {"deliberate-defer"}
    assert "unsupported-dot-requires-full-boundary" in \
        families["dot-general"]["evidence"]
    assert "unsupported-dot-scaled-requires-full-boundary" in \
        families["dot-general"]["evidence"]
    assert families["atomic"]["evidence"] == "unsupported-op-atomic"
    assert "unsupported-op-custom" in families["custom-ops"]["evidence"]
    contracts = {
        item["name"]: item for item in matrix["contracts"]
        if item["phase"] == "P5"
    }
    assert contracts["DotMaterializationContract"]["status"] == "deliberate-defer"
    assert contracts["AtomicMemoryContract"]["status"] == "deliberate-defer"
    assert contracts["SequentialResourceContract"]["status"] == "implemented"
    assert contracts["AlignmentContract"]["status"] == "candidate-primitive"
    assert set(matrix["p5_server_gates"]) == {
        "plain-dot-full-compiler-boundary",
        "alignment-planmemory-ledger-seeds-and-retry",
        "atomic-full-compiler-boundary",
    }


def test_p4_replay_shadow_evidence_is_hash_bound_and_never_promotable():
    evidence = json.loads(P4_EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["schema"] == "ttir-ub-p4-defer-evidence-v2"
    assert evidence["server_validation"] == "blocked-no-matching-consumer-toolchain"
    producer = evidence["producer_libtriton_sha256"]
    assert len(producer) == 64
    assert set(producer) <= set("0123456789abcdef")
    assert evidence["consumer_suffix_compiler_sha256"] is None
    assert evidence["consumer_semantic_model_sha256"] is None
    candidates = evidence["diagnostic_candidates"]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["name"] == "cvpipeline-ub-post-model"
    for key in ("source_revision", "vendor_llvm_revision",
                "suffix_compiler_sha256", "semantic_model_sha256"):
        assert len(candidate[key]) == 40 or len(candidate[key]) == 64
        assert set(candidate[key]) <= set("0123456789abcdef")
    assert candidate["boundary_parse"] == "accepted"
    assert candidate["compatibility_patch"] == \
        "cvpipeline_suffix_default_memory_space.patch"
    default_replay = candidate["default_alignment_replay"]
    assert default_replay["command_overrides"] == []
    assert default_replay["peak_bits_by_execution_scope"] == {
        "AIC": {"UB": 4096, "L0A": 16384, "L0C": 8192},
        "AIV": {"UB": 4096},
    }
    assert default_replay["result"] == "success"
    assert candidate["semantic_model_canonical_boundary"] == \
        "generic IR parser: malformed operand list"
    assert candidate["semantic_model_before_planmemory_generic_snapshot"] == {
        "result": "unsafe-empty-replay",
        "reported_function": "debug_aiv",
        "reported_ub_peak_bits": 0,
        "expected_aic_ub_peak_bits": 4096,
        "expected_aiv_ub_peak_bits": 4096,
    }
    assert candidate["promotion_allowed"] is False
    assert {case["name"] for case in evidence["cases"]} == {
        "dynamic-cv-mix-dot-exp",
        "irregular-indirect-add",
    }
    cases = {case["name"]: case for case in evidence["cases"]}
    for case in cases.values():
        assert case["status"] in {"replay-shadow", "deliberate-defer"}
        assert case["promotion_allowed"] is False
        assert case["reason"]
        assert len(case["ttir_sha256"]) == 64
        assert len(case["boundary_sha256"]) == 64
    dynamic = cases["dynamic-cv-mix-dot-exp"]
    assert dynamic["validated_seeds"] == [0]
    assert dynamic["analyzer_lower_bound_bits"] == 4096
    assert dynamic["analyzer_lower_bound_bits"] == max(
        dynamic["planmemory_peak_bits_by_scope"].values()
    )
    irregular = cases["irregular-indirect-add"]
    assert irregular["validated_seeds"] == []
    assert irregular["analyzer_lower_bound_bits"] == 0
    assert irregular["planmemory_peak_bits_by_scope"] is None
    assert irregular["reason"] == "unsupported-irregular-source-extent"
