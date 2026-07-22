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
})
_PEAK_RE = re.compile(r"^PLANMEM_PEAK\t(-?\d+)\t(\d+)\t(\d+)$", re.MULTILINE)
_REQUIRED_RE = re.compile(r"^PLANMEM_REQUIRED\t(-?\d+)\t(\d+)\t(\d+)$", re.MULTILINE)
_COMPLETE_RE = re.compile(r"^PLANMEM_UB_ORACLE_COMPLETE\t(-?\d+)$", re.MULTILINE)
_PLAN_ATTEMPT_RE = re.compile(r"^PLANMEM_PLAN_ATTEMPT\t[^\t]+\t(-?\d+)\t(success|failure)$", re.MULTILINE)
_OVERFLOW_RE = re.compile(
    r"\b(UB|L1|L0A|L0B|L0C) overflow, requires (\d+) bits while (\d+) bits available!"
)


class OracleUnavailable(RuntimeError):
    """The compiler output does not contain a usable PlanMemory result."""


class ManifestError(ValueError):
    """The fixture manifest does not satisfy the strict schema."""


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
        if item["expected_analyzer_decision"] not in ("defer", "reject"):
            raise ManifestError("expected_analyzer_decision must be defer or reject")
        case = dict(item)
        case["ttir"] = _fixture_path(root, item["ttir"], "ttir")
        case["before_cvpipelining"] = _fixture_path(
            root, item["before_cvpipelining"], "before_cvpipelining"
        )
        cases.append(case)
    return {"schema": raw["schema"], "cases": cases}


def run_suffix_compiler(compiler: Path, input_path: Path, seed: int, timeout: float = 120.0) -> dict:
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        raise OracleUnavailable(f"suffix compiler is not executable: {compiler}")
    with tempfile.TemporaryDirectory(prefix="ttir-ub-oracle-") as directory:
        output_path = Path(directory) / "output.mlir"
        command = [
            str(compiler),
            str(input_path),
            "-o",
            str(output_path),
            f"--plan-memory-seed={seed}",
            "--mlir-disable-threading",
            "--ub-oracle-only",
        ]
        environment = os.environ.copy()
        environment["BISHENGIR_DUMP_PLAN_MEMORY_ATTEMPTS"] = "1"
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=environment,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise OracleUnavailable(f"suffix compiler did not complete: {error}") from error
    combined = completed.stdout + "\n" + completed.stderr
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
    result.update({"seed": seed, "returncode": completed.returncode})
    return result


def analyze_case(case: dict) -> dict:
    """Run the installed analyzer using identity data derived by the real backend."""
    try:
        import triton
        from triton._C.libtriton import ascend, ir
        from triton.backends.ascend import compiler as ascend_compiler
        from triton.backends.ascend.ub_lower_bound import load_contract_profiles
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
        identity = ascend_compiler._ttir_ub_pipeline_identity(pipeline.get_pipeline_str(), metadata)
        profile = load_contract_profiles()
        result = ascend.analysis.ttir_ub_lower_bound(
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
    report = {"schema": "ttir-ub-oracle-report-v1", "cases": [], "violations": [], "unavailable": []}
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
    profiles = []
    for case in report["cases"]:
        analysis = case.get("analysis", {})
        identity = analysis.get("pipeline_identity_detail")
        if analysis.get("lower_bound_bytes", 0) <= 0 or not analysis.get("certificates") or type(identity) is not dict:
            raise OracleUnavailable("profile candidate requires a non-empty analyzer certificate")
        profiles.append({
            "case": case["name"],
            "pipeline_identity": identity,
            "contract_version": analysis.get("contract_version"),
            "validated_seeds": [run["seed"] for run in case["runs"] if run["seed"] >= 0],
            "retry_validated": any(run["seed"] == -1 for run in case["runs"]),
        })
    return {"schema": "ttir-ub-profile-candidate-v1", "profiles": profiles}


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
