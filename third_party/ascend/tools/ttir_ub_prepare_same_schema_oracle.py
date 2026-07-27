#!/usr/bin/env python3
"""Prepare and build the BiShengIR suffix oracle matching libtriton's schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


EXPECTED_ASCENDNPU_IR_REVISION = (
    "de76a453a7e22afa08a8bf00ad059d6afddc43e6"
)
PATCH_NAME = "0001-feat-port-same-schema-CVPipeline-suffix-oracle.patch"
COMPAT_PATCH_NAME = (
    "0002-fix-disable-bishengir-python-bindings-for-suffix-oracle.patch"
)
LLVM_LINALG_EXTENSION_PATCH_NAME = "0051-[Huawei]-Modify-CMakeLists.patch"
ORACLE_CODEGEN_BACKEND = "ascend"
BASE_FILE_SHA256 = {
    "bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td":
        "52935a51153ba5ee35ac1bc24836dfd559fb20273569bf605940fe39b07969e8",
    "bishengir/include/bishengir/Dialect/HIVM/Transforms/PlanMemory.h":
        "201daf3788ecc3025ec595820004b64069e00a02fbabd11ef9cb6b705e4ea61e",
    "bishengir/lib/Dialect/HIVM/Transforms/PlanMemory.cpp":
        "1ff04e9b34b64d0c1fc7ecef0defc45c81800b52131bb2ef37bb3100d8a78333",
    "bishengir/lib/Dialect/HIVM/Transforms/AlignBuffer/"
    "MarkStrideAlign.cpp":
        "6eaa79b7a45a2eb092b426f6a8ea84d291088ddb1eab9e731d4f724ff7711d2b",
    "bishengir/lib/Dialect/HIVM/IR/HIVMInterfaces.cpp":
        "de01e3a24c2511206bb5167e7d1af5a5e4b19e436708d43fa1bb5001b38388ef",
    "bishengir/tools/CMakeLists.txt":
        "47eb0304d146fb0bf24ea1ed19dc512e3895f4dc11c009860e07582c9110f13b",
    "bishengir/tools/bishengir-cvpipeline-suffix-compile/CMakeLists.txt":
        None,
    "bishengir/tools/bishengir-cvpipeline-suffix-compile/"
    "bishengir-cvpipeline-suffix-compile.cpp": None,
}
PATCHED_FILE_SHA256 = {
    "bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td":
        "69254cc73f918a6ec244bbef58b225a54ceb2e39832c4ffad276cebf4b7b9b10",
    "bishengir/include/bishengir/Dialect/HIVM/Transforms/PlanMemory.h":
        "5912365b76e33196e0c6bd562dcb3e05ce6c80d1e752cf42a2246a8090a17c6a",
    "bishengir/lib/Dialect/HIVM/Transforms/PlanMemory.cpp":
        "acf8194bd0d8c669455a80f302f2cf8dd4dffb9e0a2873160a0b0d91e7609d37",
    "bishengir/lib/Dialect/HIVM/Transforms/AlignBuffer/"
    "MarkStrideAlign.cpp":
        "a3e41ad71b179085334e7ad770cf178c556174b8de8d3fbbcf7b2efb15163400",
    "bishengir/lib/Dialect/HIVM/IR/HIVMInterfaces.cpp":
        "bd1af74fe9a83cc884a332566a7715f642c2a79a003950c1fbb9eecae8890192",
    "bishengir/tools/CMakeLists.txt":
        "2b1267a125003437c1ee94ff211e92aa62c89510d98c694d9288688b26fe0f8a",
    "bishengir/tools/bishengir-cvpipeline-suffix-compile/CMakeLists.txt":
        "455f44fee4f2d2aa7a8441f9bb01a2a44e7edc8138a9fe6b09558194e0703baa",
    "bishengir/tools/bishengir-cvpipeline-suffix-compile/"
    "bishengir-cvpipeline-suffix-compile.cpp":
        "dbb7016da94efeeba0a23a668ddb073bed2e0fcac7bc4cbec852eedb2adf4840",
}

COMPAT_BASE_FILE_SHA256 = (
    "eb29b6bc0fad418d73c35792f9992eaf3b6e98802e35b864ca34eb09f282a571"
)
COMPAT_PATCHED_FILE_SHA256 = (
    "3351c40eb65490f4959d165c63a0ff67a23b7d8fe79c07d12087801aa27c399f"
)


def _run(
    command: list[str], *, check: bool = True, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout.strip()}\n"
            f"stderr:\n{result.stderr.strip()}"
        )
    return result


def source_revision(source: Path) -> str | None:
    result = _run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_file_state(source: Path) -> str:
    actual = {
        relative: _file_sha256(source / relative)
        for relative in BASE_FILE_SHA256
    }
    if actual == BASE_FILE_SHA256:
        return "ready"
    if actual == PATCHED_FILE_SHA256:
        return "applied"
    mismatches = [
        relative for relative, digest in actual.items()
        if digest not in (
            BASE_FILE_SHA256[relative],
            PATCHED_FILE_SHA256[relative],
        )
    ]
    raise RuntimeError(
        "same-schema suffix source fingerprint mismatch: "
        + ", ".join(mismatches)
    )


def patch_state(source: Path, patch: Path) -> str:
    state = source_file_state(source)
    arguments = ["git", "apply"]
    if state == "applied":
        arguments.append("--reverse")
    result = _run(
        [*arguments, "--check", str(patch)], check=False, cwd=source
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"same-schema suffix patch state {state!r} did not pass "
            f"git apply --check:\n{result.stderr.strip()}"
        )
    return state


def compat_patch_state(source: Path, patch: Path) -> str:
    digest = _file_sha256(source / "CMakeLists.txt")
    if digest == COMPAT_BASE_FILE_SHA256:
        state = "ready"
    elif digest == COMPAT_PATCHED_FILE_SHA256:
        state = "applied"
    else:
        raise RuntimeError("same-schema suffix compatibility source fingerprint mismatch")
    arguments = ["git", "apply", "--ignore-space-change"]
    if state == "applied":
        arguments.append("--reverse")
    result = _run([*arguments, "--check", str(patch)], check=False, cwd=source)
    if result.returncode != 0:
        raise RuntimeError(
            f"same-schema suffix compatibility patch state {state!r} did not "
            f"pass git apply --check:\n{result.stderr.strip()}"
        )
    return state


def llvm_linalg_extension_patch_state(source: Path, patch: Path) -> str:
    """Return whether the required BiShengIR MLIR extension is available.

    ``BiShengIRLinalgDialectExt`` is not defined by AscendNPU-IR itself.  It is
    added to the matching LLVM/MLIR source by AscendNPU-IR patch 0051.  Do not
    configure the suffix oracle against a newer LLVM snapshot and silently
    replace that library with a dummy target: doing so would change the suffix
    pipeline that the oracle is intended to validate.
    """
    llvm_source = source / "third-party" / "llvm-project"
    linalg_dir = llvm_source / "mlir" / "lib" / "Dialect" / "Linalg" / "IR"
    cmake = linalg_dir / "CMakeLists.txt"
    extension_source = linalg_dir / "LinalgExtensions.cpp"
    if not cmake.is_file():
        raise RuntimeError(
            "same-schema suffix LLVM source is missing "
            "mlir/lib/Dialect/Linalg/IR/CMakeLists.txt"
        )
    if "BiShengIRLinalgDialectExt" in cmake.read_text():
        return "applied"
    if not extension_source.is_file():
        raise RuntimeError(
            "same-schema suffix LLVM/MLIR source is incompatible with "
            f"AscendNPU-IR patch {patch.name}: required "
            "mlir/lib/Dialect/Linalg/IR/LinalgExtensions.cpp is absent. "
            "Use an AscendNPU-IR revision whose LLVM submodule and patch "
            "series match, or supply a prebuilt same-schema BiShengIR "
            "toolchain. The oracle will not synthesize a replacement target."
        )
    result = _run(
        ["git", "apply", "--check", str(patch)],
        check=False,
        cwd=llvm_source,
    )
    if result.returncode == 0:
        return "ready"
    raise RuntimeError(
        "same-schema suffix LLVM/MLIR source does not accept required "
        f"AscendNPU-IR patch {patch.name}:\n{result.stderr.strip()}"
    )


def _require_clean_source(source: Path) -> None:
    if source_revision(source) is None:
        return
    status = _run(
        ["git", "-C", str(source), "status", "--porcelain"]
    ).stdout
    if status.strip():
        raise RuntimeError(
            "AscendNPU-IR source is dirty before applying the oracle patch"
        )


def _configure_command(
    triton_root: Path, build_dir: Path, cmake: str, cmake_args: list[str]
) -> list[str]:
    """Configure the top-level build with the Ascend plugin enabled.

    The suffix compiler lives below ``third_party/ascend``.  Configuring the
    top-level project without this codegen backend succeeds but silently omits
    the oracle target, so this argument is part of the oracle's identity.
    """
    return [
        cmake,
        "-S",
        str(triton_root),
        "-B",
        str(build_dir),
        f"-DTRITON_CODEGEN_BACKENDS={ORACLE_CODEGEN_BACKEND}",
        "-DTRITON_ASCEND_BUILD_BISHENGIR_ORACLE_TOOLS=ON",
        "-DTRITON_BUILD_UT=OFF",
        "-DTRITON_BUILD_PYTHON_MODULE=OFF",
        "-DBISHENGIR_BUILD_PYTHON_BINDINGS=OFF",
        "-DBSPUB_DAVINCI_BISHENGIR=ON",
        *cmake_args,
    ]


def main(argv: list[str] | None = None) -> int:
    tool = Path(__file__).resolve()
    triton_root = tool.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=triton_root / "third_party/ascend/AscendNPU-IR",
    )
    parser.add_argument(
        "--patch",
        type=Path,
        default=tool.parent / "patches" / PATCH_NAME,
    )
    parser.add_argument(
        "--compat-patch",
        type=Path,
        default=tool.parent / "patches" / COMPAT_PATCH_NAME,
    )
    parser.add_argument(
        "--llvm-linalg-extension-patch",
        type=Path,
        default=(
            triton_root
            / "third_party"
            / "ascend"
            / "AscendNPU-IR"
            / "build-tools"
            / "patches"
            / "llvm-project"
            / LLVM_LINALG_EXTENSION_PATCH_NAME
        ),
    )
    parser.add_argument("--apply-patch", action="store_true")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument(
        "--cmake-arg",
        action="append",
        default=[],
        help="additional CMake configure argument; may be repeated",
    )
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args(argv)

    source = arguments.source.resolve()
    patch = arguments.patch.resolve()
    compat_patch = arguments.compat_patch.resolve()
    llvm_linalg_extension_patch = arguments.llvm_linalg_extension_patch.resolve()
    if not (source / "CMakeLists.txt").is_file():
        parser.error(f"invalid AscendNPU-IR source: {source}")
    if not patch.is_file():
        parser.error(f"missing same-schema suffix patch: {patch}")
    if not compat_patch.is_file():
        parser.error(f"missing same-schema suffix compatibility patch: {compat_patch}")
    if not llvm_linalg_extension_patch.is_file():
        parser.error(
            "missing same-schema suffix LLVM Linalg extension patch: "
            f"{llvm_linalg_extension_patch}"
        )
    if arguments.jobs <= 0:
        parser.error("--jobs must be positive")

    revision = source_revision(source)
    if revision is not None and revision != EXPECTED_ASCENDNPU_IR_REVISION:
        raise RuntimeError(
            "AscendNPU-IR revision does not match the libtriton schema: "
            f"expected {EXPECTED_ASCENDNPU_IR_REVISION}, got {revision}"
        )

    state = patch_state(source, patch)
    compat_state = compat_patch_state(source, compat_patch)
    llvm_linalg_extension_state = llvm_linalg_extension_patch_state(
        source, llvm_linalg_extension_patch
    )
    needs_apply = (
        state == "ready"
        or compat_state == "ready"
        or llvm_linalg_extension_state == "ready"
    )
    if needs_apply and arguments.apply_patch:
        _require_clean_source(source)

    if state == "ready" and arguments.apply_patch:
        _run(["git", "apply", str(patch)], cwd=source)
        state = patch_state(source, patch)

    if compat_state == "ready" and arguments.apply_patch:
        _run(["git", "apply", "--ignore-space-change", str(compat_patch)],
             cwd=source)
        compat_state = compat_patch_state(source, compat_patch)

    if llvm_linalg_extension_state == "ready" and arguments.apply_patch:
        _run(
            ["git", "apply", str(llvm_linalg_extension_patch)],
            cwd=source / "third-party" / "llvm-project",
        )
        llvm_linalg_extension_state = llvm_linalg_extension_patch_state(
            source, llvm_linalg_extension_patch
        )

    binary = None
    if arguments.build_dir is not None:
        if (
            state != "applied"
            or compat_state != "applied"
            or llvm_linalg_extension_state != "applied"
        ):
            raise RuntimeError(
                "--build-dir requires applied BiShengIR and LLVM/MLIR "
                "patches; pass --apply-patch"
            )
        build_dir = arguments.build_dir.resolve()
        _run(_configure_command(
            triton_root, build_dir, arguments.cmake, arguments.cmake_arg
        ))
        _run([
            arguments.cmake,
            "--build",
            str(build_dir),
            "--target",
            "bishengir-cvpipeline-suffix-compile",
            "--",
            f"-j{arguments.jobs}",
        ])
        binary = build_dir / "bin/bishengir-cvpipeline-suffix-compile"
        if not binary.is_file():
            raise RuntimeError(
                f"same-schema suffix build omitted expected binary: {binary}"
            )

    print(json.dumps({
        "source": str(source),
        "revision": revision or f"snapshot:{EXPECTED_ASCENDNPU_IR_REVISION}",
        "patch": str(patch),
        "patch_state": state,
        "compat_patch": str(compat_patch),
        "compat_patch_state": compat_state,
        "llvm_linalg_extension_patch": str(llvm_linalg_extension_patch),
        "llvm_linalg_extension_patch_state": llvm_linalg_extension_state,
        "binary": str(binary) if binary is not None else None,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
