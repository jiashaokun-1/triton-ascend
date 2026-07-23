#!/usr/bin/env python3
"""Generate path-stable canonical TTIR inputs for UB oracle fixtures."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


def _enable_source_tree_cann_language() -> None:
    """Make the source-tree CANN language package visible without installing."""
    import triton.language.extra as extra

    language_root = Path(__file__).resolve().parents[1] / "language"
    if (language_root / "cann").is_dir():
        root = str(language_root)
        if root not in extra.__path__:
            extra.__path__.append(root)


_enable_source_tree_cann_language()

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def dynamic_cv_mix_dot_exp(a_ptr, b_ptr, c_ptr, M: tl.constexpr,
                           N: tl.constexpr, K: tl.constexpr):
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]
    inner = tl.arange(0, K)
    a = tl.load(a_ptr + rows * K + inner[None, :])
    b = tl.load(b_ptr + inner[:, None] * N + cols)
    accumulator = tl.dot(a, b)
    tl.store(c_ptr + rows * N + cols, tl.exp(accumulator))


@triton.jit
def irregular_indirect_add(src_ptr, indices_ptr, out_ptr, add_value,
                           BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    indices = tl.load(indices_ptr + offsets)
    values = tl.load(src_ptr + indices)
    tl.store(out_ptr + offsets, values + add_value)


_FIXTURES = {
    "dynamic-cv-mix-dot-exp": {
        "kernel": dynamic_cv_mix_dot_exp,
        "signature": {
            "a_ptr": "*fp32",
            "b_ptr": "*fp32",
            "c_ptr": "*fp32",
        },
        "constants": {"M": 16, "N": 16, "K": 16},
        "compile_mode": "simd",
    },
    "irregular-indirect-add": {
        "kernel": irregular_indirect_add,
        "signature": {
            "src_ptr": "*fp32",
            "indices_ptr": "*i64",
            "out_ptr": "*fp32",
            "add_value": "fp32",
        },
        "constants": {"BLOCK": 8},
        "compile_mode": "simd_simt",
    },
}


def generate_fixture_ttir(name: str, arch: str) -> str:
    from triton._C.libtriton import ascend, buffer_ir, ir
    from triton._C.libtriton.ascend import ir as ascend_ir
    from triton.backends.ascend import compiler as ascend_compiler
    from triton.backends.compiler import GPUTarget
    from triton.compiler.code_generator import ast_to_ttir
    from triton.compiler.compiler import ASTSource

    fixture = _FIXTURES[name]
    context = ir.context()
    ir.load_dialects(context)
    buffer_ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    options = ascend_compiler.NPUOptions(
        arch=arch,
        compile_on_910_95=True,
        compile_mode=fixture["compile_mode"],
    )
    target = GPUTarget("ascend", arch, 32)
    kernel = fixture["kernel"]
    source = ASTSource(kernel, fixture["signature"], fixture["constants"])
    codegen_fns = {"min_dot_size": ascend_compiler.min_dot_size(target)}
    module = ast_to_ttir(kernel, source, context, options, codegen_fns, {})
    metadata = dict(options.__dict__)
    metadata.update({
        "hash": hashlib.sha256(str(module).encode("utf-8")).hexdigest(),
        "target": target,
        "triton_version": triton.__version__,
    })
    ascend_compiler.make_ttir(module, metadata, options)
    return ascend.analysis.ttir_ub_generic_module_text(module)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=sorted(_FIXTURES), required=True)
    parser.add_argument("--arch", default="Ascend950")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output.exists():
        parser.error(f"refusing to overwrite existing output: {arguments.output}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    text = generate_fixture_ttir(arguments.fixture, arguments.arch)
    arguments.output.write_text(text.rstrip() + "\n", encoding="utf-8")
    print(arguments.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
