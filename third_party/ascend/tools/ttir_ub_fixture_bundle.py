#!/usr/bin/env python3
"""Package one compiler dump into a paired TTIR UB oracle fixture."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys


_OPERATION_FAMILY_RESOURCE_COUNTS = {
    "direct-copy": 1,
    "binary-add": 2,
    "reshape-copy": 2,
}
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_MAX_INT64 = (1 << 63) - 1


class BundleError(ValueError):
    """The compiler dump cannot form an auditable oracle fixture pair."""


@dataclass(frozen=True)
class BundleConfig:
    name: str
    operation_family: str
    arch: str
    source_elements: int
    element_bit_width: int
    max_tiles: int
    auto_tile_and_bind_subblock_outcome: bool
    expected_analyzer_decision: str = "defer"
    materialization_stage: str = "ttir.triton-to-linalg"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(config: BundleConfig) -> int:
    if not _SAFE_NAME.fullmatch(config.name):
        raise BundleError("name must contain only letters, digits, '.', '_' or '-'")
    if config.operation_family not in _OPERATION_FAMILY_RESOURCE_COUNTS:
        raise BundleError("unsupported operation family")
    if not config.arch:
        raise BundleError("arch must be non-empty")
    if config.expected_analyzer_decision not in ("defer", "reject"):
        raise BundleError("expected analyzer decision must be defer or reject")
    if not config.materialization_stage:
        raise BundleError("materialization stage must be non-empty")
    for name, value in (
        ("source elements", config.source_elements),
        ("element bit width", config.element_bit_width),
        ("max tiles", config.max_tiles),
    ):
        if type(value) is not int or value <= 0:
            raise BundleError(f"{name} must be a positive integer")
    if config.element_bit_width < 8 or config.element_bit_width % 8:
        raise BundleError("element bit width must be a whole positive number of bytes")
    payload_bytes = config.source_elements * (config.element_bit_width // 8)
    if payload_bytes > _MAX_INT64:
        raise BundleError("input payload overflows int64")
    if type(config.auto_tile_and_bind_subblock_outcome) is not bool:
        raise BundleError("auto-tile outcome must be boolean")
    return payload_bytes


def _dump_file(dump_dir: Path, relative_name: str, field: str) -> Path:
    if type(relative_name) is not str or not relative_name or Path(relative_name).is_absolute():
        raise BundleError(f"{field} must be a relative path inside the dump directory")
    candidate = (dump_dir / relative_name).resolve()
    try:
        candidate.relative_to(dump_dir)
    except ValueError as error:
        raise BundleError(f"{field} must stay inside the dump directory") from error
    return candidate


def create_fixture_bundle(
    dump_dir: Path,
    output_dir: Path,
    config: BundleConfig,
    *,
    ttir_dump_name: str = "kernel.ttir.mlir",
    before_dump_name: str = "kernel.ttadapter.mlir",
) -> dict:
    """Copy a same-compilation dump pair and create its strict oracle manifest."""
    payload_bytes = _validate_config(config)
    dump_dir = dump_dir.resolve()
    output_dir = output_dir.resolve()
    ttir_source = _dump_file(dump_dir, ttir_dump_name, "TTIR dump name")
    before_source = _dump_file(dump_dir, before_dump_name, "before-CVPipelining dump name")
    if not ttir_source.is_file() or not before_source.is_file():
        raise BundleError(
            "dump directory must contain kernel.ttir.mlir and kernel.ttadapter.mlir"
        )
    ttir_name = f"{config.name}.ttir.mlir"
    before_name = f"{config.name}.before_cvpipelining.mlir"
    targets = [output_dir / ttir_name, output_dir / before_name, output_dir / "manifest.json"]
    existing = [path.name for path in targets if path.exists() or path.is_symlink()]
    if existing:
        raise BundleError(f"refusing to overwrite existing fixture files: {', '.join(existing)}")

    manifest = {
        "schema": "ttir-ub-oracle-v1",
        "cases": [{
            "name": config.name,
            "operation_family": config.operation_family,
            "ttir": ttir_name,
            "before_cvpipelining": before_name,
            "arch": config.arch,
            "options": {
                "compile_mode": "simd",
                "multibuffer": False,
            },
            "expected_analyzer_decision": config.expected_analyzer_decision,
            "contract_proposal": {
                "expected_resource_count": _OPERATION_FAMILY_RESOURCE_COUNTS[
                    config.operation_family
                ],
                "expected_source_elements": config.source_elements,
                "expected_element_bit_width": config.element_bit_width,
                "expected_input_payload_bytes": payload_bytes,
                "materialization_stage": config.materialization_stage,
                "max_tiles": config.max_tiles,
                "auto_tile_and_bind_subblock_outcome": (
                    config.auto_tile_and_bind_subblock_outcome
                ),
            },
        }],
    }

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ttir_source, targets[0])
        shutil.copyfile(before_source, targets[1])
        targets[2].write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as error:
        # Do not silently leave a bundle that looks complete after a partial write.
        for path in targets:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise BundleError(f"cannot write fixture bundle: {error}") from error

    return {
        "manifest": str(targets[2]),
        "ttir_sha256": _file_sha256(targets[0]),
        "before_cvpipelining_sha256": _file_sha256(targets[1]),
        "expected_input_payload_bytes": payload_bytes,
    }


def _parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--operation-family", choices=sorted(_OPERATION_FAMILY_RESOURCE_COUNTS), required=True
    )
    parser.add_argument("--arch", required=True)
    parser.add_argument("--source-elements", type=int, required=True)
    parser.add_argument("--element-bit-width", type=int, required=True)
    parser.add_argument("--max-tiles", type=int, required=True)
    parser.add_argument("--auto-tile-outcome", type=_parse_bool, required=True)
    parser.add_argument(
        "--expected-analyzer-decision", choices=("defer", "reject"), default="defer"
    )
    parser.add_argument("--materialization-stage", default="ttir.triton-to-linalg")
    parser.add_argument("--ttir-dump-name", default="kernel.ttir.mlir")
    parser.add_argument("--before-dump-name", default="kernel.ttadapter.mlir")
    arguments = parser.parse_args(argv)
    try:
        result = create_fixture_bundle(
            arguments.dump_dir,
            arguments.output_dir,
            BundleConfig(
                name=arguments.name,
                operation_family=arguments.operation_family,
                arch=arguments.arch,
                source_elements=arguments.source_elements,
                element_bit_width=arguments.element_bit_width,
                max_tiles=arguments.max_tiles,
                auto_tile_and_bind_subblock_outcome=arguments.auto_tile_outcome,
                expected_analyzer_decision=arguments.expected_analyzer_decision,
                materialization_stage=arguments.materialization_stage,
            ),
            ttir_dump_name=arguments.ttir_dump_name,
            before_dump_name=arguments.before_dump_name,
        )
    except BundleError as error:
        print(json.dumps({"status": "unavailable", "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "ok", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
