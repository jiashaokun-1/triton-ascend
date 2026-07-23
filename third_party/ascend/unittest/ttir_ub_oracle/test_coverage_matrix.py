import json
from pathlib import Path


MATRIX = Path(__file__).with_name("coverage_matrix.json")
P4_EVIDENCE = Path(__file__).with_name("p4_defer_evidence.json")
PHASES = {"P0", "P1", "P2", "P3", "P4"}
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


def test_p4_defer_evidence_is_hash_bound_and_never_promotable():
    evidence = json.loads(P4_EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["schema"] == "ttir-ub-p4-defer-evidence-v1"
    for field in (
        "producer_libtriton_sha256",
        "consumer_suffix_compiler_sha256",
        "consumer_semantic_model_sha256",
    ):
        value = evidence[field]
        assert len(value) == 64
        assert set(value) <= set("0123456789abcdef")
    assert {case["name"] for case in evidence["cases"]} == {
        "dynamic-cv-mix-dot-exp",
        "irregular-indirect-add",
    }
    for case in evidence["cases"]:
        assert case["status"] == "deliberate-defer"
        assert case["promotion_allowed"] is False
        assert case["reason"]
        assert len(case["ttir_sha256"]) == 64
        assert len(case["boundary_sha256"]) == 64
