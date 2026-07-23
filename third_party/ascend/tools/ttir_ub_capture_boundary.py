#!/usr/bin/env python3
"""Capture a real TTIR-to-Linalg before-CVPipelining oracle boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace


_DUMP_HEADER = "// -----// IR Dump Before CVPipelining (cv-pipelining) //----- //\n"
_ALIAS_RE = re.compile(r"^(#[A-Za-z_][A-Za-z0-9_]*) = (.+)$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inline_leading_aliases(text: str) -> str:
    root = text.find('"builtin.module"()')
    if root < 0:
        raise RuntimeError("generic boundary serializer omitted builtin.module")
    prefix = text[:root]
    aliases: dict[str, str] = {}
    for line in prefix.splitlines():
        if not line.strip():
            continue
        match = _ALIAS_RE.fullmatch(line)
        if (
            not match
            or match.group(1) in aliases
            or not match.group(2).startswith(("affine_map<", "affine_set<"))
        ):
            raise RuntimeError(
                f"unsupported generic boundary prefix declaration: {line!r}"
            )
        aliases[match.group(1)] = match.group(2)
    result = text[root:]
    for name in sorted(aliases, key=len, reverse=True):
        result = re.sub(
            re.escape(name) + r"(?![A-Za-z0-9_])",
            aliases[name],
            result,
        )
    if any(
        re.search(re.escape(name) + r"(?![A-Za-z0-9_])", result)
        for name in aliases
    ):
        raise RuntimeError("generic boundary alias expansion was incomplete")
    return result


def _capture(
    ttir: Path,
    arch: str,
    compile_mode: str,
    dynamic_cv: bool,
    multibuffer: bool,
    num_stages: int,
    intra_cache_num: int | None,
    inter_cache_num: int | None,
    load_cache_num: int | None,
    assembly_format: str,
) -> tuple[str, dict]:
    import triton
    import triton._C.libtriton as libtriton
    from triton._C.libtriton import ascend, buffer_ir, ir
    from triton._C.libtriton.ascend import ir as ascend_ir
    from triton.backends.ascend import compiler as ascend_compiler
    from triton.backends.compiler import GPUTarget

    context = ir.context()
    ir.load_dialects(context)
    buffer_ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ir.parse_mlir_module(str(ttir), context)
    compile_on_910_95 = arch == "Ascend950" or arch.startswith("Ascend910_95")
    options = ascend_compiler.NPUOptions(
        arch=arch,
        compile_on_910_95=compile_on_910_95,
        compile_mode=compile_mode,
        enable_dynamic_cv_pipeline=dynamic_cv,
        multibuffer=multibuffer,
        num_stages=num_stages,
    )
    metadata = dict(options.__dict__)
    metadata.update({
        "hash": _file_sha256(ttir),
        "target": GPUTarget("ascend", arch, 32),
        "triton_version": triton.__version__,
        "intra_cache_num": intra_cache_num,
        "inter_cache_num": inter_cache_num,
        "load_cache_num": load_cache_num,
    })
    pipeline_stages: list[dict] = []
    pipeline = ascend_compiler._build_ttir_to_linalg_pass_manager(
        SimpleNamespace(context=context),
        metadata,
        options,
        named_ops=True,
        pipeline_stages=pipeline_stages,
    )
    ascend_compiler._set_ttir_to_linalg_buffer_counts(metadata)
    pipeline.run(module)
    if assembly_format == "generic":
        serialized = inline_leading_aliases(
            ascend.analysis.ttir_ub_generic_module_text(module)
        )
        if (not serialized.startswith('"builtin.module"()')
                or '"func.func"' not in serialized):
            raise RuntimeError(
                "generic boundary serializer returned an invalid module: "
                f"prefix={serialized[:160]!r}, "
                f"func_count={serialized.count('func.func')}"
            )
    else:
        serialized = ascend.analysis.ttir_ub_portable_module_text(module)
        if "module" not in serialized or "func.func" not in serialized:
            raise RuntimeError(
                "portable boundary serializer returned an invalid module: "
                f"prefix={serialized[:160]!r}"
            )
    capture = _DUMP_HEADER + serialized.rstrip() + "\n"
    audit = {
        "schema": "ttir-ub-boundary-capture-v1",
        "producer_libtriton_sha256": _file_sha256(
            Path(libtriton.__file__).resolve()
        ),
        "ttir_sha256": _file_sha256(ttir),
        "boundary_sha256": hashlib.sha256(capture.encode("utf-8")).hexdigest(),
        "arch": arch,
        "compile_mode": compile_mode,
        "compile_on_910_95": compile_on_910_95,
        "enable_dynamic_cv_pipeline": dynamic_cv,
        "multibuffer": multibuffer,
        "num_stages": num_stages,
        "intra_cache_num": intra_cache_num,
        "inter_cache_num": inter_cache_num,
        "load_cache_num": load_cache_num,
        "assembly_format": assembly_format,
        "pipeline": pipeline.get_pipeline_str(),
        "pipeline_stages": pipeline_stages,
    }
    return capture, audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ttir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--arch", required=True)
    parser.add_argument(
        "--compile-mode",
        choices=("simd", "simd_simt", "simt_template", "simt_only"),
        default="simd",
    )
    parser.add_argument("--enable-dynamic-cv-pipeline", action="store_true")
    parser.add_argument("--multibuffer", action="store_true")
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--intra-cache-num", type=int)
    parser.add_argument("--inter-cache-num", type=int)
    parser.add_argument("--load-cache-num", type=int)
    parser.add_argument(
        "--assembly-format",
        choices=("portable", "generic"),
        default="generic",
        help="generic is the exact oracle format and requires an identical "
             "operation schema; portable is only for same-schema diagnostics",
    )
    arguments = parser.parse_args(argv)
    audit_output = arguments.audit_output or Path(str(arguments.output) + ".json")
    if not arguments.ttir.is_file():
        parser.error(f"missing TTIR input: {arguments.ttir}")
    if arguments.output.exists() or audit_output.exists():
        parser.error("refusing to overwrite an existing capture or audit sidecar")
    if arguments.num_stages <= 0:
        parser.error("--num-stages must be positive")
    for name in ("intra_cache_num", "inter_cache_num", "load_cache_num"):
        value = getattr(arguments, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    capture, audit = _capture(
        arguments.ttir.resolve(),
        arguments.arch,
        arguments.compile_mode,
        arguments.enable_dynamic_cv_pipeline,
        arguments.multibuffer,
        arguments.num_stages,
        arguments.intra_cache_num,
        arguments.inter_cache_num,
        arguments.load_cache_num,
        arguments.assembly_format,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(capture, encoding="utf-8")
    audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "boundary": str(arguments.output),
        "audit": str(audit_output),
        "boundary_sha256": audit["boundary_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
