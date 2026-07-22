#!/usr/bin/env python3
"""Compare TTIR UB lower-bound certificates with real PlanMemory results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Callable, Iterable


UB_SCOPE = "6"
_MANIFEST_KEYS = frozenset({"schema", "cases"})
_CASE_KEYS = frozenset({
    "name",
    "ttir",
    "before_cvpipelining",
    "arch",
    "options",
    "expected_analyzer_decision",
    "contract_proposal",
})
_CONTRACT_PROPOSAL_KEYS = frozenset({
    "expected_resource_count",
    "expected_source_elements",
    "expected_element_bit_width",
    "expected_input_payload_bytes",
    "materialization_stage",
    "max_tiles",
    "auto_tile_and_bind_subblock_outcome",
})
_IDENTITY_CONTRACT = {
    "auto_tile_and_bind_subblock": {
        "identity_value": "module-derived-both-outcomes",
        "profile_promotion_requires": "task7-oracle-validates-enabled-and-disabled-outcomes",
    },
}
_PEAK_RE = re.compile(r"^PLANMEM_PEAK\t(-?\d+)\t(\d+)\t(\d+)$", re.MULTILINE)
_REQUIRED_RE = re.compile(r"^PLANMEM_REQUIRED\t(-?\d+)\t(\d+)\t(\d+)$", re.MULTILINE)
_COMPLETE_RE = re.compile(r"^PLANMEM_UB_ORACLE_COMPLETE\t(-?\d+)$", re.MULTILINE)
_PLAN_ATTEMPT_RE = re.compile(r"^PLANMEM_PLAN_ATTEMPT\t[^\t]+\t(-?\d+)\t(success|failure)$", re.MULTILINE)
_OVERFLOW_RE = re.compile(
    r"\b(UB|L1|L0A|L0B|L0C) overflow, requires (\d+) bits while (\d+) bits available!"
)
_SUB_BLOCK_INDEX_OP = "hivm.hir.get_sub_block_idx"
_STATIC_ALLOC_RE = re.compile(
    r'"memref\.alloc"\([^\n]*?\)\s*(?:<[^\n]*?>\s*)?:\s*\([^\n]*?\)\s*->\s*'
    r'memref<(\d+)x(bf16|f16|f32|f64|i8|i16|i32|i64)>'
)


class OracleUnavailable(RuntimeError):
    """The compiler output does not contain a usable PlanMemory result."""


class ManifestError(ValueError):
    """The fixture manifest does not satisfy the strict schema."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise OracleUnavailable(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def parse_planmemory_peak(text: str, attempt: int, scope: str = UB_SCOPE) -> int:
    values = [
        int(bits)
        for attempt_text, scope_text, bits in _PEAK_RE.findall(text)
        if int(attempt_text) == attempt and scope_text == str(scope)
    ]
    if not values:
        raise OracleUnavailable(f"missing PlanMemory peak for attempt={attempt}, scope={scope}")
    return max(values)


def parse_planmemory_required(text: str, attempt: int, scope: str = UB_SCOPE) -> int:
    values = [
        int(bits)
        for attempt_text, scope_text, bits in _REQUIRED_RE.findall(text)
        if int(attempt_text) == attempt and scope_text == str(scope)
    ]
    if not values:
        raise OracleUnavailable(f"missing PlanMemory requirement for attempt={attempt}, scope={scope}")
    return max(values)


def parse_overflow_scope(text: str) -> str | None:
    match = _OVERFLOW_RE.search(text)
    return match.group(1) if match else None


def parse_completed_attempt(text: str) -> int:
    if not _COMPLETE_RE.search(text):
        raise OracleUnavailable("missing PlanMemory completion marker")
    attempts = [int(value) for value, status in _PLAN_ATTEMPT_RE.findall(text) if status == "success"]
    if not attempts:
        raise OracleUnavailable("missing successful PlanMemory attempt")
    return attempts[-1]


def detect_auto_tile_outcome(input_ir: str, after_tile_ir: str) -> bool:
    """Use the real post-TileAndBindSubBlock snapshot as the outcome oracle."""
    if _SUB_BLOCK_INDEX_OP in input_ir:
        raise OracleUnavailable("input already contains a sub-block index operation")
    return _SUB_BLOCK_INDEX_OP in after_tile_ir


def parse_direct_copy_boundary_allocation_bytes(text: str) -> int:
    """Extract the one local static allocation admitted by the P1 fixture."""
    if text.count('"memref.alloc"') != 1:
        raise OracleUnavailable("direct-copy boundary must contain exactly one memref.alloc")
    matches = _STATIC_ALLOC_RE.findall(text)
    if len(matches) != 1:
        raise OracleUnavailable("direct-copy boundary allocation is not a supported static 1-D memref")
    elements_text, element_type = matches[0]
    bit_width = 16 if element_type == "bf16" else int(re.search(r"\d+", element_type).group())
    elements = int(elements_text)
    if elements <= 0 or bit_width < 8 or bit_width % 8 != 0:
        raise OracleUnavailable("direct-copy boundary allocation has an unsupported element type")
    allocation_bytes = elements * (bit_width // 8)
    if allocation_bytes > (1 << 63) - 1:
        raise OracleUnavailable("direct-copy boundary allocation overflows int64")
    return allocation_bytes


def classify_failure(text: str, attempt: int = 0) -> dict:
    match = _OVERFLOW_RE.search(text)
    if not match:
        raise OracleUnavailable("compiler failure is not a recognized memory-capacity result")
    scope, required, available = match.groups()
    result = {
        "status": "overflow",
        "overflow_scope": scope,
        "required_bits": int(required),
        "available_bits": int(available),
        "actual_peak_bits": None,
    }
    if scope == "UB":
        try:
            result["actual_peak_bits"] = parse_planmemory_required(text, attempt, UB_SCOPE)
        except OracleUnavailable:
            result["actual_peak_bits"] = int(required)
    return result


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise argparse.ArgumentTypeError("empty seed component")
        if re.fullmatch(r"\d+-\d+", part):
            first, last = (int(item) for item in part.split("-", 1))
            if first > last:
                raise argparse.ArgumentTypeError("seed range must be increasing")
            seeds.extend(range(first, last + 1))
        elif re.fullmatch(r"-?\d+", part):
            seeds.append(int(part))
        else:
            raise argparse.ArgumentTypeError(f"invalid seed component: {part}")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("duplicate seed")
    return seeds


def _fixture_path(root: Path, relative: object, field: str) -> Path:
    if type(relative) is not str or not relative or Path(relative).is_absolute():
        raise ManifestError(f"{field} must be a relative fixture path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ManifestError(f"{field} must stay inside the fixture directory") from error
    if not candidate.is_file():
        raise ManifestError(f"missing fixture: {relative}")
    return candidate


def load_manifest(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read manifest: {error}") from error
    if type(raw) is not dict or set(raw) != _MANIFEST_KEYS or raw.get("schema") != "ttir-ub-oracle-v1":
        raise ManifestError("unsupported manifest schema")
    if type(raw["cases"]) is not list or not raw["cases"]:
        raise ManifestError("manifest cases must be a non-empty list")
    root = path.resolve().parent
    names: set[str] = set()
    cases = []
    for item in raw["cases"]:
        if type(item) is not dict or set(item) != _CASE_KEYS:
            raise ManifestError("case fields do not match the schema")
        if type(item["name"]) is not str or not item["name"] or item["name"] in names:
            raise ManifestError("case names must be unique non-empty strings")
        names.add(item["name"])
        if type(item["arch"]) is not str or not item["arch"]:
            raise ManifestError("arch must be a non-empty string")
        if type(item["options"]) is not dict:
            raise ManifestError("options must be an object")
        if item["options"].get("compile_mode") != "simd" or item["options"].get("multibuffer") is not False:
            raise ManifestError("P1 direct-copy cases require compile_mode=simd and multibuffer=false")
        if item["expected_analyzer_decision"] not in ("defer", "reject"):
            raise ManifestError("expected_analyzer_decision must be defer or reject")
        proposal = item["contract_proposal"]
        if type(proposal) is not dict or set(proposal) != _CONTRACT_PROPOSAL_KEYS:
            raise ManifestError("contract_proposal fields do not match the schema")
        for field in (
            "expected_resource_count",
            "expected_source_elements",
            "expected_element_bit_width",
            "expected_input_payload_bytes",
            "max_tiles",
        ):
            if type(proposal[field]) is not int or proposal[field] <= 0:
                raise ManifestError(f"contract_proposal.{field} must be a positive integer")
        if type(proposal["materialization_stage"]) is not str or not proposal["materialization_stage"]:
            raise ManifestError("contract_proposal.materialization_stage must be non-empty")
        if type(proposal["auto_tile_and_bind_subblock_outcome"]) is not bool:
            raise ManifestError("contract_proposal.auto_tile_and_bind_subblock_outcome must be boolean")
        case = dict(item)
        case["ttir"] = _fixture_path(root, item["ttir"], "ttir")
        case["before_cvpipelining"] = _fixture_path(
            root, item["before_cvpipelining"], "before_cvpipelining"
        )
        cases.append(case)
    return {"schema": raw["schema"], "cases": cases}


def build_proposed_contract_profile(identity: dict, pipeline_stages: list[dict], proposal: dict) -> dict:
    """Build an uninstalled, reviewable contract chain for oracle evaluation."""
    materialization_stage = proposal["materialization_stage"]
    if sum(stage["stage_name"] == materialization_stage for stage in pipeline_stages) != 1:
        raise OracleUnavailable("materialization stage must occur exactly once in the real pipeline")
    common = {
        "expected_resource_count": str(proposal["expected_resource_count"]),
        "expected_source_elements": str(proposal["expected_source_elements"]),
        "expected_element_bit_width": str(proposal["expected_element_bit_width"]),
    }
    input_payload = proposal["expected_input_payload_bytes"]
    output_payload = (input_payload + proposal["max_tiles"] - 1) // proposal["max_tiles"]
    materialized = False
    bindings = []
    for stage in pipeline_stages:
        is_materialization = stage["stage_name"] == materialization_stage
        parameters = {
            **common,
            "expected_input_payload_bytes": str(output_payload if materialized else input_payload),
        }
        contract_id = "direct-copy-preserve"
        if is_materialization:
            contract_id = "direct-copy-max-tiles"
            parameters["max_tiles"] = str(proposal["max_tiles"])
            materialized = True
        bindings.append({
            **stage,
            "contract_id": contract_id,
            "contract_version": "1",
            "contract_parameters": parameters,
        })
    return {
        "schema": "ttir-ub-lb-profile-v1",
        "profiles": [{
            "pipeline_identity": identity,
            "pipeline_stages": bindings,
        }],
    }


def run_suffix_compiler(compiler: Path, input_path: Path, seed: int, timeout: float = 120.0) -> dict:
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        raise OracleUnavailable(f"suffix compiler is not executable: {compiler}")
    compiler = compiler.resolve()
    input_path = input_path.resolve()
    with tempfile.TemporaryDirectory(prefix="ttir-ub-oracle-") as directory:
        output_path = Path(directory) / "output.mlir"
        stage_oracle_dir = Path(directory) / "stages"
        command = [
            str(compiler),
            str(input_path),
            "-o",
            str(output_path),
            f"--plan-memory-seed={seed}",
            "--mlir-disable-threading",
            "--ub-oracle-only",
            f"--dump-stage-oracle-dir={stage_oracle_dir}",
        ]
        environment = os.environ.copy()
        environment["BISHENGIR_DUMP_PLAN_MEMORY_ATTEMPTS"] = "1"
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=directory,
                env=environment,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise OracleUnavailable(f"suffix compiler did not complete: {error}") from error
        combined = completed.stdout + "\n" + completed.stderr
        snapshots = list(stage_oracle_dir.glob("post-*-TileAndBindSubBlock.generic.mlir"))
        if len(snapshots) != 1:
            raise OracleUnavailable("missing or ambiguous post-TileAndBindSubBlock snapshot")
        try:
            input_ir = input_path.read_text(encoding="utf-8")
            after_tile_ir = snapshots[0].read_text(encoding="utf-8")
        except OSError as error:
            raise OracleUnavailable(f"cannot read auto-tile outcome snapshots: {error}") from error
        auto_tile_outcome = detect_auto_tile_outcome(input_ir, after_tile_ir)
        if completed.returncode != 0:
            result = classify_failure(combined)
        else:
            if "PLANMEM_RUN_RESULT\tsuccess" not in combined:
                raise OracleUnavailable("successful process omitted the PlanMemory completion marker")
            completed_attempt = parse_completed_attempt(combined)
            result = {
                "status": "success",
                "overflow_scope": None,
                "required_bits": None,
                "available_bits": None,
                "actual_peak_bits": parse_planmemory_peak(combined, attempt=completed_attempt, scope=UB_SCOPE),
            }
        result.update({
            "seed": seed,
            "returncode": completed.returncode,
            "auto_tile_and_bind_subblock_outcome": auto_tile_outcome,
        })
    return result


def analyze_case(case: dict) -> dict:
    """Run the installed analyzer using identity data derived by the real backend."""
    try:
        import triton
        from triton._C.libtriton import ascend, ir
        from triton.backends.ascend import compiler as ascend_compiler
        from triton.backends.compiler import GPUTarget

        context = ir.context()
        ascend.load_dialects(context)
        module = ir.parse_mlir_module(str(case["ttir"]), context)
        option_values = dict(case["options"])
        option_values.setdefault("arch", case["arch"])
        options = ascend_compiler.NPUOptions(**option_values)
        metadata = dict(options.__dict__)
        metadata.update({
            "hash": hashlib.sha256(case["ttir"].read_bytes()).hexdigest(),
            "target": GPUTarget("ascend", case["arch"], 32),
            "triton_version": triton.__version__,
        })
        # The pybind Module wrapper does not expose its context on every
        # supported build.  Pipeline construction only needs that context, so
        # use the context that parsed this exact module.
        pipeline_stages = []
        pipeline = ascend_compiler._build_ttir_to_linalg_pass_manager(
            SimpleNamespace(context=context), metadata, options, named_ops=True,
            pipeline_stages=pipeline_stages,
        )
        ascend_compiler._record_pipeline_stage(
            pipeline_stages, ascend_compiler.TTIR_UB_BISHENG_SUFFIX_STAGE
        )
        identity = ascend_compiler._ttir_ub_pipeline_identity(
            pipeline.get_pipeline_str(), metadata, str(module)
        )
        profile = build_proposed_contract_profile(identity, pipeline_stages, case["contract_proposal"])
        result = ascend.analysis.ttir_ub_lower_bound_candidate_for_oracle(
            module,
            {
                "arch": case["arch"],
                "compile_mode": "aiv" if options.compile_mode == "simd" else options.compile_mode,
                "pipeline_identity": identity,
                "pipeline_stages": pipeline_stages,
                "contract_profile": profile,
            },
        )
    except Exception as error:
        raise OracleUnavailable(f"analyzer did not complete: {error}") from error
    if type(result) is not dict or result.get("decision") not in ("defer", "reject"):
        raise OracleUnavailable("analyzer returned an unsupported result")
    result = dict(result)
    result["pipeline_identity_detail"] = identity
    result["pipeline_stages_detail"] = profile["profiles"][0]["pipeline_stages"]
    result["auto_tile_and_bind_subblock_outcome"] = \
        case["contract_proposal"]["auto_tile_and_bind_subblock_outcome"]
    try:
        boundary_ir = case["before_cvpipelining"].read_text(encoding="utf-8")
    except OSError as error:
        raise OracleUnavailable(f"cannot read before-CVPipelining fixture: {error}") from error
    result["before_cvpipelining_allocation_bytes"] = \
        parse_direct_copy_boundary_allocation_bytes(boundary_ir)
    result["ttir_fixture_sha256"] = file_sha256(case["ttir"])
    result["before_cvpipelining_sha256"] = file_sha256(case["before_cvpipelining"])
    return result


def evaluate(
    manifest: dict,
    compiler: Path,
    seeds: Iterable[int],
    check_retry: bool,
    analyzer: Callable[[dict], dict] = analyze_case,
    suffix_runner: Callable[[Path, Path, int], dict] = run_suffix_compiler,
) -> dict:
    seed_list = list(seeds)
    run_seeds = seed_list + ([-1] if check_retry and -1 not in seed_list else [])
    report = {
        "schema": "ttir-ub-oracle-report-v1",
        "suffix_compiler_sha256": file_sha256(compiler) if compiler.is_file() else None,
        "cases": [],
        "violations": [],
        "unavailable": [],
    }
    for case in manifest["cases"]:
        case_report = {"name": case["name"], "runs": []}
        try:
            analysis = analyzer(case)
            case_report["analysis"] = analysis
            if analysis["decision"] != case["expected_analyzer_decision"]:
                report["violations"].append({
                    "case": case["name"],
                    "kind": "unexpected-analyzer-decision",
                    "expected": case["expected_analyzer_decision"],
                    "actual": analysis["decision"],
                })
            boundary_bytes = analysis.get("before_cvpipelining_allocation_bytes")
            if type(boundary_bytes) is not int or boundary_bytes <= 0:
                raise OracleUnavailable("analyzer omitted the before-CVPipelining allocation")
            if analysis.get("lower_bound_bytes", 0) > boundary_bytes:
                report["violations"].append({
                    "case": case["name"], "kind": "lower-bound-exceeds-before-cvpipelining-allocation",
                    "lower_bound_bytes": analysis.get("lower_bound_bytes", 0),
                    "allocation_bytes": boundary_bytes,
                })
        except OracleUnavailable as error:
            report["unavailable"].append({"case": case["name"], "phase": "analyzer", "reason": str(error)})
            report["cases"].append(case_report)
            continue
        lower_bound_bits = int(analysis.get("lower_bound_bytes", 0)) * 8
        for seed in run_seeds:
            try:
                actual = suffix_runner(compiler, case["before_cvpipelining"], seed)
                case_report["runs"].append(actual)
            except OracleUnavailable as error:
                report["unavailable"].append({
                    "case": case["name"], "phase": "plan-memory", "seed": seed, "reason": str(error)
                })
                continue
            peak = actual.get("actual_peak_bits")
            actual_outcome = actual.get("auto_tile_and_bind_subblock_outcome")
            expected_outcome = case["contract_proposal"]["auto_tile_and_bind_subblock_outcome"]
            if type(actual_outcome) is not bool:
                report["unavailable"].append({
                    "case": case["name"], "phase": "auto-tile-outcome", "seed": seed,
                    "reason": "suffix compiler omitted the auto-tile outcome",
                })
                continue
            if actual_outcome != expected_outcome:
                report["violations"].append({
                    "case": case["name"], "kind": "auto-tile-outcome-mismatch", "seed": seed,
                    "expected": expected_outcome, "actual": actual_outcome,
                })
            if type(peak) is int and lower_bound_bits > peak:
                report["violations"].append({
                    "case": case["name"], "kind": "lower-bound-exceeds-actual", "seed": seed,
                    "lower_bound_bits": lower_bound_bits, "actual_peak_bits": peak,
                })
            if analysis["decision"] == "reject" and not (
                actual.get("status") == "overflow" and actual.get("overflow_scope") == "UB"
            ):
                report["violations"].append({
                    "case": case["name"], "kind": "reject-without-ub-capacity-result", "seed": seed,
                })
        report["cases"].append(case_report)
    report["summary"] = {
        "cases": len(report["cases"]),
        "seeds": seed_list,
        "retry_checked": check_retry,
        "violations": len(report["violations"]),
        "unavailable": len(report["unavailable"]),
    }
    return report


def build_profile_candidate(report: dict) -> dict:
    if report["violations"] or report["unavailable"]:
        raise OracleUnavailable("profile candidate requires a complete zero-violation report")
    if report.get("summary", {}).get("seeds") != list(range(20)) or not report.get("summary", {}).get(
            "retry_checked"):
        raise OracleUnavailable("profile candidate requires seeds 0..19 and retry")
    outcomes = {
        case.get("analysis", {}).get("auto_tile_and_bind_subblock_outcome")
        for case in report["cases"]
    }
    if outcomes != {False, True}:
        raise OracleUnavailable("profile candidate requires both auto-tile outcomes")
    if not _is_sha256(report.get("suffix_compiler_sha256")):
        raise OracleUnavailable("profile candidate requires the suffix compiler hash")
    report_sha256 = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    profiles = []
    fingerprints = set()
    for case in report["cases"]:
        analysis = case.get("analysis", {})
        identity = analysis.get("pipeline_identity_detail")
        stages = analysis.get("pipeline_stages_detail")
        if (analysis.get("lower_bound_bytes", 0) <= 0 or not analysis.get("certificates")
                or type(identity) is not dict or type(stages) is not list or not stages):
            raise OracleUnavailable("profile candidate requires a non-empty analyzer certificate")
        boundary_bytes = analysis.get("before_cvpipelining_allocation_bytes")
        if type(boundary_bytes) is not int or analysis["lower_bound_bytes"] > boundary_bytes:
            raise OracleUnavailable("profile candidate lacks a valid materialization bridge")
        if not _is_sha256(analysis.get("ttir_fixture_sha256")) or not _is_sha256(
                analysis.get("before_cvpipelining_sha256")):
            raise OracleUnavailable("profile candidate requires fixture hashes")
        fingerprint = identity.get("sha256")
        if type(fingerprint) is not str or not fingerprint or fingerprint in fingerprints:
            raise OracleUnavailable("profile candidate requires unique pipeline identities")
        fingerprints.add(fingerprint)
        validated_seeds = [run["seed"] for run in case["runs"] if run["seed"] >= 0]
        retry_validated = any(run["seed"] == -1 for run in case["runs"])
        if validated_seeds != list(range(20)) or not retry_validated:
            raise OracleUnavailable("profile candidate case is missing seeds or retry")
        expected_outcome = analysis.get("auto_tile_and_bind_subblock_outcome")
        if type(expected_outcome) is not bool or any(
                run.get("auto_tile_and_bind_subblock_outcome") is not expected_outcome
                for run in case["runs"]):
            raise OracleUnavailable("profile candidate case has unverified auto-tile outcomes")
        profiles.append({
            "pipeline_identity": identity,
            "pipeline_stages": stages,
            "contract_version": analysis.get("contract_version"),
            "oracle_report_sha256": report_sha256,
            "validated_seeds": validated_seeds,
            "retry_validated": retry_validated,
            "auto_tile_and_bind_subblock_outcomes": [False, True],
        })
    return {
        "schema": "ttir-ub-lb-profile-v1",
        "identity_contract": _IDENTITY_CONTRACT,
        "profiles": profiles,
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suffix-compiler", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0-19"))
    parser.add_argument("--check-retry", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--profile-candidate", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.profile_candidate and not arguments.report:
        print(json.dumps({
            "status": "unavailable",
            "reason": "--profile-candidate requires --report for auditability",
        }, sort_keys=True))
        return 2
    try:
        manifest = load_manifest(arguments.manifest)
        report = evaluate(
            manifest, arguments.suffix_compiler.resolve(), arguments.seeds, arguments.check_retry
        )
        if arguments.report:
            _write_json(arguments.report, report)
        if arguments.profile_candidate:
            _write_json(arguments.profile_candidate, build_profile_candidate(report))
    except (ManifestError, OracleUnavailable) as error:
        print(json.dumps({"status": "unavailable", "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(report["summary"], sort_keys=True))
    if report["violations"]:
        return 1
    if report["unavailable"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
