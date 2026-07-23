# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import shlex
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from triton._C.libtriton import ascend, ir
from triton.backends.ascend import compiler as ascend_compiler
from triton.backends.ascend.errors import UBLowerBoundOverflow
from triton.backends.ascend import ub_lower_bound
from triton.backends.ascend.ub_lower_bound import apply_ub_lower_bound_policy, load_contract_profiles
from triton.backends.ascend.runtime import utils as runtime_utils
from triton.compiler import compiler as core_compiler

DIRECT_LOAD_COPY = """
module {
  tt.func public @copy(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 65536 : i32, start = 0 : i32} : tensor<65536xi32>
    %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %srcs, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %dsts = tt.splat %dst : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dsts, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %value = tt.load %src_ptrs : tensor<65536x!tt.ptr<f32>>
    tt.store %dst_ptrs, %value : tensor<65536x!tt.ptr<f32>>
    tt.return
  }
}
"""

BINARY_ADD = """
module {
  tt.func public @add(%lhs: !tt.ptr<f32>, %rhs: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 65536 : i32, start = 0 : i32} : tensor<65536xi32>
    %lhs_splat = tt.splat %lhs : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %lhs_ptrs = tt.addptr %lhs_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %rhs_splat = tt.splat %rhs : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %rhs_ptrs = tt.addptr %rhs_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dst_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %lhs_value = tt.load %lhs_ptrs : tensor<65536x!tt.ptr<f32>>
    %rhs_value = tt.load %rhs_ptrs : tensor<65536x!tt.ptr<f32>>
    %sum = arith.addf %lhs_value, %rhs_value : tensor<65536xf32>
    tt.store %dst_ptrs, %sum : tensor<65536x!tt.ptr<f32>>
    tt.return
  }
}
"""

RESHAPE_COPY = """
module {
  tt.func public @reshape_copy(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 65536 : i32, start = 0 : i32} : tensor<65536xi32>
    %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %srcs, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %dsts = tt.splat %dst : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dsts, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %value = tt.load %src_ptrs : tensor<65536x!tt.ptr<f32>>
    %view = tt.reshape %value : tensor<65536xf32> -> tensor<256x256xf32>
    %dst_view = tt.reshape %dst_ptrs : tensor<65536x!tt.ptr<f32>> -> tensor<256x256x!tt.ptr<f32>>
    tt.store %dst_view, %view : tensor<256x256x!tt.ptr<f32>>
    tt.return
  }
}
"""

REDUCTION_SUM = """
module {
  tt.func public @reduction_sum(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 65536 : i32, start = 0 : i32} : tensor<65536xi32>
    %srcs = tt.splat %src : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %srcs, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %value = tt.load %src_ptrs : tensor<65536x!tt.ptr<f32>>
    %sum = "tt.reduce" (%value) ({
    ^bb0(%lhs: f32, %rhs: f32):
      %add = arith.addf %lhs, %rhs : f32
      tt.reduce.return %add : f32
    }) {axis = 0 : i32} : (tensor<65536xf32>) -> f32
    tt.store %dst, %sum : !tt.ptr<f32>
    tt.return
  }
}
"""

LOOP_CARRIED_ADD = """
module {
  tt.func public @loop_carried_add(%init_src: !tt.ptr<f32>, %step_src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %range = tt.make_range {end = 65536 : i32, start = 0 : i32} : tensor<65536xi32>
    %init_splat = tt.splat %init_src : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %init_ptrs = tt.addptr %init_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %step_splat = tt.splat %step_src : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %step_ptrs = tt.addptr %step_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<65536x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dst_splat, %range : tensor<65536x!tt.ptr<f32>>, tensor<65536xi32>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %init = tt.load %init_ptrs : tensor<65536x!tt.ptr<f32>>
    %result = scf.for %iv = %c0 to %c2 step %c1 iter_args(%acc = %init) -> tensor<65536xf32> {
      %step = tt.load %step_ptrs : tensor<65536x!tt.ptr<f32>>
      %next = arith.addf %acc, %step : tensor<65536xf32>
      scf.yield %next : tensor<65536xf32>
    }
    tt.store %dst_ptrs, %result : tensor<65536x!tt.ptr<f32>>
    tt.return
  }
}
"""

INT64_MAX = (1 << 63) - 1
UINT32_MAX = (1 << 32) - 1


class ExplosiveEquality:

    def __eq__(self, _other):
        raise RuntimeError("must not escape normalization")


@dataclass
class Options:
    ub_lower_bound_mode: str
    arch: str = "Ascend910B"
    compile_mode: str = "aiv"
    debug: bool = False


@pytest.fixture(scope="module", autouse=True)
def assign_npu():
    """This analysis-only suite does not need an NPU device."""


def _analysis_result(decision="reject"):
    return {
        "decision": decision,
        "lower_bound_bytes": 262144,
        "capacity_bytes": 196608,
        "certificates": [{
            "kind": "singleton",
            "bytes": 262144,
            "resource_ids": [0],
            "contract_trace": ["ttir-direct-load-v1", "direct-copy-max-tiles"],
        }],
        "unsupported_reasons": [],
        "contract_version": "ttir-ub-lb-v1",
        "pipeline_identity": "test-id",
    }


def _pickle_round_trip(value):
    return pickle.loads(pickle.dumps(value))


def _result_with(path, value):
    result = _analysis_result()
    container = result
    for key in path[:-1]:
        container = container[key]
    container[path[-1]] = value
    return result


def _compiler_metadata(**updates):
    options = ascend_compiler.NPUOptions(compile_on_910_95=False)
    metadata = dict(options.__dict__)
    metadata.update({
        "hash": "compiler-test-hash",
        "target": SimpleNamespace(arch="Ascend910B"),
        "triton_version": "test-triton-version",
    })
    metadata.update(updates)
    return metadata


EXPECTED_TTIR_TO_LINALG_PIPELINE = (
    "any(auto-blockify{auto-blockify-size=1},"
    "triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},"
    "discrete-mask-access-conversion{compile-mode=simd compile-on-910-95=false "
    "enable-sync-block-lock=false force-simt-template=false},"
    "triton-to-annotation,"
    "triton-to-unstructure{compile-mode=simd compile-on-910-95=false "
    "force-scalarize-mode=false force-simt-template=false},"
    "triton-to-hivm,triton-to-hfusion,triton-to-llvm,"
    "bubble-up-operation{enable-aggressive-mode=true},"
    "triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},"
    "triton-to-linalg{compile-mode=simd compile-on-910-95=false "
    "enable-nd2nz-on-vector=false enable-select-analysis=true force-simt-template=false "
    "global-kernel=false named-ops=false})")

EXPECTED_UB_AFFECTING_OPTIONS = (
    "add_auto_scheduling",
    "auto_blockify_size",
    "auto_tile_and_bind_subblock",
    "auto_vectorize_v2_max_fused_ops_num",
    "bisheng_options",
    "compile_mode",
    "compile_on_910_95",
    "disable_auto_inject_block_sync",
    "disable_fma",
    "disable_size_align_for_cast",
    "disable_tightly_coupled_buffer_reuse",
    "enable_auto_bind_sub_block",
    "enable_auto_blockify",
    "enable_auto_vectorize_v2",
    "enable_bishengir_simt_optimization",
    "enable_cce_vf_auto_sync",
    "enable_cce_vf_remove_membar",
    "enable_drop_unit_dims",
    "enable_dynamic_cv_pipeline",
    "enable_flatten",
    "enable_hivm_auto_cv_balance",
    "enable_mask_fallback_conversion",
    "enable_mixed_cv",
    "enable_nd2nz_on_vector",
    "enable_preload",
    "enable_select_analysis",
    "enable_simt_reorder_instruction",
    "enable_sync_block_lock",
    "enable_ubuf_saving",
    "enable_vf_fusion",
    "force_simt_only",
    "force_simt_template",
    "hfusion_enable_multiple_consumer_fusion",
    "inject_barrier_all",
    "inject_block_all",
    "inter_cache_num",
    "intra_cache_num",
    "ir_override",
    "limit_auto_multi_buffer_of_local_buffer",
    "limit_auto_multi_buffer_only_for_local_buffer",
    "load_cache_num",
    "mix_mode",
    "multibuffer",
    "num_stages",
    "num_warps",
    "optimize_dynamic_offset",
    "prevec_max_fused_ops_num",
    "set_workspace_multibuffer",
    "shared_mem_dynamic_size",
    "simt_stack_limit",
    "sync_solver",
    "tile_mix_cube_loop",
    "tile_mix_vector_loop",
    "unit_flag",
    "use_bytecode",
    "vf_merge_level",
    "warp_size",
)

EXPECTED_FORWARDED_BISHENG_SCANNED_HELPERS = (
    "_direct_simt_libdevice_compile_options",
    "get_common_bishengir_compile_options",
    "linalg_to_bin_enable_npu_compile_910_95",
    "linalg_to_bin_enable_npu_compile_A2_A3",
    "ttir_to_npubin",
)
EXPECTED_NON_BISHENG_LONG_FLAG_HELPERS = (
    "bc_to_linalg_by_bishengir_opt",
    "linalg_to_bc_by_triton_mlir_opt",
    "ttir_to_linalg",
)


def _identity_runtime(monkeypatch):
    monkeypatch.setattr(ascend_compiler, "get_cann_version_file_hash", lambda: "test-cann-hash")
    monkeypatch.setattr(ascend_compiler, "_is_auto_map_parallel_blocks_enabled", lambda: False)
    monkeypatch.setattr(ascend_compiler, "force_disable_ffts", lambda: False)
    monkeypatch.setattr(ascend_compiler, "_check_bishengir_is_regbased", lambda: False)
    monkeypatch.setattr(ascend_compiler, "_get_npucompiler_path", lambda: ("/test/bishengir-compile", {}))
    monkeypatch.setattr(
        ascend_compiler,
        "_npu_compiler_content_fingerprint",
        lambda _path: "test-compiler-content-sha256",
        raising=False,
    )
    monkeypatch.delenv("TRITON_ENABLE_VF_FUSION", raising=False)


def _forwarded_bisheng_flag_literals(source=None):
    source = source or inspect.getsource(ascend_compiler).lstrip("\ufeff")
    tree = ast.parse(source)
    flags = set()
    for function in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
        if function.name not in EXPECTED_FORWARDED_BISHENG_SCANNED_HELPERS:
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if not node.value.startswith("--"):
                continue
            flag = node.value.split("=", 1)[0]
            flags.add(flag)
    return flags


def _long_flag_helper_names(source=None):
    source = source or inspect.getsource(ascend_compiler).lstrip("\ufeff")
    tree = ast.parse(source)
    return {
        function.name
        for function in tree.body
        if isinstance(function, ast.FunctionDef) and any(
            isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("--")
            for node in ast.walk(function))
    }


def test_mode_validation():
    assert ascend_compiler.NPUOptions().ub_lower_bound_mode == "off"
    assert ascend_compiler.NPUOptions(ub_lower_bound_mode="shadow").ub_lower_bound_mode == "shadow"
    assert ascend_compiler.NPUOptions(ub_lower_bound_mode="enforce").ub_lower_bound_mode == "enforce"
    with pytest.raises(ValueError, match="ub_lower_bound_mode"):
        ascend_compiler.NPUOptions(ub_lower_bound_mode="probabilistic")


def test_ttir_to_linalg_builder_preserves_default_pipeline_byte_for_byte(monkeypatch):
    monkeypatch.setattr(ascend_compiler, "_is_auto_map_parallel_blocks_enabled", lambda: False)
    context = ir.context()
    ascend.load_dialects(context)
    module = SimpleNamespace(context=context)
    metadata = _compiler_metadata()
    options = ascend_compiler.NPUOptions(compile_on_910_95=False)
    pipeline_stages = []
    pm = ascend_compiler._build_ttir_to_linalg_pass_manager(
        module, metadata, options, named_ops=False, pipeline_stages=pipeline_stages
    )
    assert pm.get_pipeline_str() == EXPECTED_TTIR_TO_LINALG_PIPELINE
    assert [stage["stage_name"] for stage in pipeline_stages] == [
        "ttir.auto-blockify",
        "ttir.triton-to-structure",
        "ttir.discrete-mask-access-conversion",
        "ttir.triton-to-annotation",
        "ttir.triton-to-unstructure",
        "ttir.triton-to-hivm",
        "ttir.triton-to-hfusion",
        "ttir.triton-to-llvm",
        "ttir.bubble-up-operation",
        "ttir.triton-to-structure",
        "ttir.triton-to-linalg",
    ]
    assert pipeline_stages[0]["options"] == {"auto_blockify_size": "1"}
    assert pipeline_stages[-1]["options"]["compile_mode"] == '"simd"'
    assert pipeline_stages[-1]["options"]["named_ops"] == "false"


def test_pipeline_identity_is_canonical_json_and_contains_complete_payload(monkeypatch):
    _identity_runtime(monkeypatch)
    metadata = _compiler_metadata(multibuffer=False)
    identity = ascend_compiler._ttir_ub_pipeline_identity("pass-a,pass-b", metadata)
    options = json.loads(identity["relevant_options_json"])
    assert identity.keys() == {
        "open_source_pipeline",
        "canonical_ttir_sha256",
        "relevant_options_json",
        "target_arch",
        "triton_version",
        "cann_version_hash",
        "sha256",
    }
    assert identity["open_source_pipeline"] == "pass-a,pass-b"
    assert identity["canonical_ttir_sha256"] == hashlib.sha256(b"").hexdigest()
    assert identity["target_arch"] == "Ascend910B"
    assert identity["triton_version"] == "test-triton-version"
    assert identity["cann_version_hash"] == "test-cann-hash"
    assert options["multibuffer"] is False
    assert identity["relevant_options_json"] == json.dumps(options, sort_keys=True, separators=(",", ":"))
    payload = {
        "cann_version_hash": "test-cann-hash",
        "canonical_ttir_sha256": hashlib.sha256(b"").hexdigest(),
        "open_source_pipeline": "pass-a,pass-b",
        "relevant_options": options,
        "target_arch": "Ascend910B",
        "triton_version": "test-triton-version",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert identity["sha256"] == hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)


def test_pipeline_identity_changes_when_pass_order_changes(monkeypatch):
    _identity_runtime(monkeypatch)
    metadata = _compiler_metadata()
    first = ascend_compiler._ttir_ub_pipeline_identity("pass-a,pass-b", metadata)
    second = ascend_compiler._ttir_ub_pipeline_identity("pass-b,pass-a", metadata)
    assert first["sha256"] != second["sha256"]


def test_pipeline_identity_changes_when_canonical_ttir_changes(monkeypatch):
    _identity_runtime(monkeypatch)
    metadata = _compiler_metadata()
    first = ascend_compiler._ttir_ub_pipeline_identity("pipeline", metadata, "module { // a\n}")
    second = ascend_compiler._ttir_ub_pipeline_identity("pipeline", metadata, "module { // b\n}")
    assert first["canonical_ttir_sha256"] != second["canonical_ttir_sha256"]
    assert first["sha256"] != second["sha256"]


def test_pipeline_identity_binds_compiler_kind_and_content(monkeypatch, tmp_path):
    monkeypatch.setattr(ascend_compiler, "get_cann_version_file_hash", lambda: "test-cann-hash")
    monkeypatch.setattr(ascend_compiler, "_is_auto_map_parallel_blocks_enabled", lambda: False)
    monkeypatch.setattr(ascend_compiler, "force_disable_ffts", lambda: False)
    monkeypatch.setattr(ascend_compiler, "_check_bishengir_is_regbased", lambda: False)
    monkeypatch.delenv("TRITON_ENABLE_VF_FUSION", raising=False)
    metadata = _compiler_metadata()

    first_path = tmp_path / "first" / "npuc"
    second_path = tmp_path / "second" / "npuc"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    first_path.write_bytes(b"same compiler content")
    second_path.write_bytes(b"same compiler content")
    monkeypatch.setattr(ascend_compiler, "_get_npucompiler_path", lambda: (str(first_path), {}))
    first = ascend_compiler._ttir_ub_pipeline_identity("pipeline", metadata)
    monkeypatch.setattr(ascend_compiler, "_get_npucompiler_path", lambda: (str(second_path), {}))
    second = ascend_compiler._ttir_ub_pipeline_identity("pipeline", metadata)
    assert first["sha256"] == second["sha256"]

    third_path = tmp_path / "third" / "bishengir-compile"
    third_path.parent.mkdir()
    third_path.write_bytes(b"same compiler content")
    monkeypatch.setattr(ascend_compiler, "_get_npucompiler_path", lambda: (str(third_path), {}))
    third = ascend_compiler._ttir_ub_pipeline_identity("pipeline", metadata)
    assert third["sha256"] != first["sha256"]
    first_options = json.loads(first["relevant_options_json"])
    third_options = json.loads(third["relevant_options_json"])
    assert first_options["effective_npu_compiler_kind"] == "npuc"
    assert third_options["effective_npu_compiler_kind"] == "bishengir-compile"
    assert first_options["effective_npu_compiler_content_sha256"] == \
        third_options["effective_npu_compiler_content_sha256"]


def test_pipeline_identity_binds_libdevice_content_not_install_path(monkeypatch, tmp_path):
    _identity_runtime(monkeypatch)
    first_path = tmp_path / "first install" / "libdevice.10.bc"
    second_path = tmp_path / "second install" / "libdevice.10.bc"
    changed_path = tmp_path / "changed install" / "libdevice.10.bc"
    for path, content in (
        (first_path, b"same libdevice"),
        (second_path, b"same libdevice"),
        (changed_path, b"changed libdevice"),
    ):
        path.parent.mkdir()
        path.write_bytes(content)

    def identity(path):
        metadata = _compiler_metadata(
            bisheng_options=f"-cce-link-aicore-ll-module {shlex.quote(str(path))} -mllvm -test-option"
        )
        return ascend_compiler._ttir_ub_pipeline_identity("pipeline", metadata)

    first = identity(first_path)
    second = identity(second_path)
    changed = identity(changed_path)
    assert first["sha256"] == second["sha256"]
    assert first["sha256"] != changed["sha256"]
    normalized = json.loads(first["relevant_options_json"])["bisheng_options"]
    assert str(first_path) not in normalized
    assert hashlib.sha256(b"same libdevice").hexdigest() in normalized


def test_compiler_kind_changes_linalg_command_with_identical_content(monkeypatch, tmp_path):
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        output = Path(command[command.index("-o") + 1] + ".o")
        output.write_bytes(b"npubin")
        return SimpleNamespace(stdout=b"", stderr=b"", returncode=0)

    npuc = tmp_path / "npuc"
    bishengir_compile = tmp_path / "bishengir-compile"
    npuc.write_bytes(b"same compiler content")
    bishengir_compile.write_bytes(b"same compiler content")
    monkeypatch.setattr(ascend_compiler, "_parse_linalg_metadata", lambda linalg, metadata: (linalg, metadata))
    monkeypatch.setattr(ascend_compiler, "_check_bishengir_api_change", lambda: True)
    monkeypatch.setattr(ascend_compiler, "_check_bishengir_is_regbased", lambda: False)
    monkeypatch.setattr(ascend_compiler, "_is_ascend_sanitizer_enabled", lambda: False)
    monkeypatch.setattr(ascend_compiler, "_is_debug_line_info_disabled", lambda: True)
    monkeypatch.setattr(ascend_compiler, "_enable_print_ub_bits", lambda: False)
    monkeypatch.setattr(ascend_compiler, "_enable_dump_memory_info", lambda: False)
    monkeypatch.setattr(ascend_compiler, "_enable_msdebug", lambda: False)
    monkeypatch.setattr(ascend_compiler, "_is_auto_map_parallel_blocks_enabled", lambda: False)
    monkeypatch.setattr(ascend_compiler, "get_libdevice", lambda: "/test/libdevice.bc")
    monkeypatch.setattr(ascend_compiler, "NPUUtils", lambda: SimpleNamespace(get_arch=lambda: "Ascend910B"))
    monkeypatch.setattr(ascend_compiler.subprocess, "run", run)
    options = ascend_compiler.NPUOptions(compile_on_910_95=False)
    metadata = _compiler_metadata(bitcodes=[], auto_tile_and_bind_subblock=True)

    monkeypatch.setattr(ascend_compiler, "_get_npucompiler_path", lambda: (str(npuc), {}))
    ascend_compiler.linalg_to_bin_enable_npu_compile_A2_A3("module", dict(metadata), options)
    monkeypatch.setattr(ascend_compiler, "_get_npucompiler_path", lambda: (str(bishengir_compile), {}))
    ascend_compiler.linalg_to_bin_enable_npu_compile_A2_A3("module", dict(metadata), options)

    assert "--enable-triton-kernel-compile=true" not in commands[0]
    assert "--enable-triton-kernel-compile=true" in commands[1]
    assert commands[0] != commands[1]


def test_compiler_content_fingerprint_rehashes_preserved_stat_file(monkeypatch, tmp_path):
    monkeypatch.setattr(ascend_compiler, "get_cann_version_file_hash", lambda: "test-cann-hash")
    monkeypatch.setattr(ascend_compiler, "_is_auto_map_parallel_blocks_enabled", lambda: False)
    monkeypatch.setattr(ascend_compiler, "force_disable_ffts", lambda: False)
    monkeypatch.setattr(ascend_compiler, "_check_bishengir_is_regbased", lambda: False)
    monkeypatch.delenv("TRITON_ENABLE_VF_FUSION", raising=False)
    compiler = tmp_path / "npuc"
    compiler.write_bytes(b"AAAA")
    original_stat = compiler.stat()
    monkeypatch.setattr(ascend_compiler, "_get_npucompiler_path", lambda: (str(compiler), {}))
    first = ascend_compiler._ttir_ub_pipeline_identity("pipeline", _compiler_metadata())

    compiler.write_bytes(b"BBBB")
    os.utime(compiler, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    rewritten_stat = compiler.stat()
    assert rewritten_stat.st_ino == original_stat.st_ino
    assert rewritten_stat.st_size == original_stat.st_size
    assert rewritten_stat.st_mtime_ns == original_stat.st_mtime_ns

    second = ascend_compiler._ttir_ub_pipeline_identity("pipeline", _compiler_metadata())
    first_options = json.loads(first["relevant_options_json"])
    second_options = json.loads(second["relevant_options_json"])
    assert first_options["effective_npu_compiler_content_sha256"] == hashlib.sha256(b"AAAA").hexdigest()
    assert second_options["effective_npu_compiler_content_sha256"] == hashlib.sha256(b"BBBB").hexdigest()
    assert first["sha256"] != second["sha256"]


def test_selected_compiler_read_failure_makes_policy_identity_fail_open(monkeypatch, tmp_path):
    canonical_pm = MagicMock()
    future_pm = MagicMock()
    future_pm.get_pipeline_str.return_value = "pipeline"
    monkeypatch.setattr(ascend_compiler.ir, "pass_manager", lambda _context: canonical_pm)
    monkeypatch.setattr(ascend_compiler, "passes", MagicMock())
    monkeypatch.setattr(ascend_compiler, "_build_ttir_to_linalg_pass_manager", lambda *_args, **_kwargs: future_pm)
    monkeypatch.setattr(ascend_compiler, "get_cann_version_file_hash", lambda: "test-cann-hash")
    monkeypatch.setattr(ascend_compiler, "_is_auto_map_parallel_blocks_enabled", lambda: False)
    monkeypatch.setattr(ascend_compiler, "force_disable_ffts", lambda: False)
    monkeypatch.setattr(ascend_compiler, "_check_bishengir_is_regbased", lambda: False)
    missing = tmp_path / "missing" / "npuc"
    monkeypatch.setattr(ascend_compiler, "_get_npucompiler_path", lambda: (str(missing), {}))
    captured = []
    monkeypatch.setattr(
        ascend_compiler,
        "apply_ub_lower_bound_policy",
        lambda _mod, _metadata, _options, identity, stages: captured.append((identity, stages)),
    )
    options = ascend_compiler.NPUOptions(ub_lower_bound_mode="shadow", compile_on_910_95=False)
    module = SimpleNamespace(context=object(), __str__=lambda: "module")
    ascend_compiler.make_ttir(module, _compiler_metadata(), options)
    assert captured == [("", [])]


def test_pipeline_identity_changes_for_effective_environment_lowering_switches(monkeypatch):
    _identity_runtime(monkeypatch)
    a2_metadata = _compiler_metadata()
    baseline = ascend_compiler._ttir_ub_pipeline_identity("pipeline", a2_metadata)["sha256"]
    monkeypatch.setattr(ascend_compiler, "_is_auto_map_parallel_blocks_enabled", lambda: True)
    auto_blockify = ascend_compiler._ttir_ub_pipeline_identity("pipeline", a2_metadata)["sha256"]
    monkeypatch.setattr(ascend_compiler, "_is_auto_map_parallel_blocks_enabled", lambda: False)
    monkeypatch.setattr(ascend_compiler, "_check_bishengir_is_regbased", lambda: True)
    reg_based = ascend_compiler._ttir_ub_pipeline_identity("pipeline", a2_metadata)["sha256"]
    assert len({baseline, auto_blockify, reg_based}) == 3

    _identity_runtime(monkeypatch)
    a5_metadata = _compiler_metadata(compile_on_910_95=True)
    baseline = ascend_compiler._ttir_ub_pipeline_identity("pipeline", a5_metadata)["sha256"]
    monkeypatch.setattr(ascend_compiler, "force_disable_ffts", lambda: True)
    disable_ffts = ascend_compiler._ttir_ub_pipeline_identity("pipeline", a5_metadata)["sha256"]
    monkeypatch.setattr(ascend_compiler, "force_disable_ffts", lambda: False)
    monkeypatch.setenv("TRITON_ENABLE_VF_FUSION", "1")
    vf_fusion = ascend_compiler._ttir_ub_pipeline_identity("pipeline", a5_metadata)["sha256"]
    assert len({baseline, disable_ffts, vf_fusion}) == 3


@pytest.mark.parametrize(
    "option_name",
    (name for name in EXPECTED_UB_AFFECTING_OPTIONS if name != "auto_tile_and_bind_subblock"),
)
def test_pipeline_identity_changes_for_every_ub_affecting_option(monkeypatch, option_name):
    _identity_runtime(monkeypatch)
    metadata = _compiler_metadata()
    before = ascend_compiler._ttir_ub_pipeline_identity("pipeline", metadata)["sha256"]
    current = metadata.get(option_name)
    metadata[option_name] = "identity-test-value" if current is None else None
    after = ascend_compiler._ttir_ub_pipeline_identity("pipeline", metadata)["sha256"]
    assert before != after, option_name


def test_module_derived_auto_tile_placeholder_is_bound_by_exact_ttir_identity(monkeypatch):
    _identity_runtime(monkeypatch)
    enabled = _compiler_metadata(auto_tile_and_bind_subblock=True)
    disabled = _compiler_metadata(auto_tile_and_bind_subblock=False)
    enabled_identity = ascend_compiler._ttir_ub_pipeline_identity("pipeline", enabled)
    disabled_identity = ascend_compiler._ttir_ub_pipeline_identity("pipeline", disabled)
    enabled_options = json.loads(enabled_identity["relevant_options_json"])
    disabled_options = json.loads(disabled_identity["relevant_options_json"])
    assert enabled_options["auto_tile_and_bind_subblock"] == "module-derived-per-exact-ttir"
    assert disabled_options["auto_tile_and_bind_subblock"] == "module-derived-per-exact-ttir"
    assert enabled_identity["sha256"] == disabled_identity["sha256"]


def test_generated_disable_auto_tile_attr_cannot_claim_enabled_only_profile(monkeypatch):
    _identity_runtime(monkeypatch)
    linalg = '''
module attributes {mix_mode = "aiv", parallel_mode = "mix_simd_simt",
                   hivm.disable_auto_tile_and_bind_subblock} {
  func.func @kernel() { return }
}
'''
    metadata = _compiler_metadata(auto_tile_and_bind_subblock=True)
    _, parsed_metadata = ascend_compiler._parse_linalg_metadata(linalg, metadata)
    assert parsed_metadata["auto_tile_and_bind_subblock"] is False

    identity = ascend_compiler._ttir_ub_pipeline_identity("pipeline", parsed_metadata)
    identity_options = json.loads(identity["relevant_options_json"])
    contract = load_contract_profiles()["identity_contract"]["auto_tile_and_bind_subblock"]
    assert identity_options["auto_tile_and_bind_subblock"] == contract["identity_value"]
    assert contract == {
        "identity_value": "module-derived-per-exact-ttir",
        "profile_promotion_requires": "oracle-validates-exact-profile-outcome",
    }


def test_ub_affecting_options_and_forwarded_bisheng_flags_are_closed_goldens():
    assert ascend_compiler.UB_AFFECTING_OPTIONS == EXPECTED_UB_AFFECTING_OPTIONS
    assert ascend_compiler.FORWARDED_BISHENG_FLAG_SCANNED_HELPERS == EXPECTED_FORWARDED_BISHENG_SCANNED_HELPERS
    assert _long_flag_helper_names() == (set(EXPECTED_FORWARDED_BISHENG_SCANNED_HELPERS)
                                         | set(EXPECTED_NON_BISHENG_LONG_FLAG_HELPERS))
    assert _forwarded_bisheng_flag_literals() == set(ascend_compiler.FORWARDED_BISHENG_FLAG_CLASSIFICATION)
    for category, justification in ascend_compiler.FORWARDED_BISHENG_FLAG_CLASSIFICATION.values():
        assert category in ("ub-affecting", "non-ub")
        assert justification
    assert ascend_compiler.FORWARDED_BISHENG_FLAG_CLASSIFICATION["--enable-hivm-cross-core-gss"][0] == \
        "ub-affecting"


def test_direct_helper_unknown_forwarded_flag_fails_closure_gate():
    source = inspect.getsource(ascend_compiler).lstrip("\ufeff")
    marker = "def _direct_simt_libdevice_compile_options(metadata):\n"
    assert marker in source
    mutated = source.replace(marker, marker + '    future_option = "--future-direct-ub=true"\n', 1)
    unclassified = _forwarded_bisheng_flag_literals(mutated) - set(
        ascend_compiler.FORWARDED_BISHENG_FLAG_CLASSIFICATION)
    assert unclassified == {"--future-direct-ub"}

    new_helper = '\ndef _future_direct_compile_options():\n    return ["--future-helper-ub=true"]\n'
    mutated_with_helper = source + new_helper
    known_helpers = (set(EXPECTED_FORWARDED_BISHENG_SCANNED_HELPERS) | set(EXPECTED_NON_BISHENG_LONG_FLAG_HELPERS))
    assert _long_flag_helper_names(mutated_with_helper) - known_helpers == {"_future_direct_compile_options"}


def test_make_ttir_calls_policy_after_canonicalization_and_does_not_run_future_pm(monkeypatch):
    calls = []
    canonical_pm = MagicMock()
    canonical_pm.run.side_effect = lambda _mod: calls.append("canonical-pm-run")
    future_pm = MagicMock()
    future_pm.get_pipeline_str.side_effect = lambda: calls.append("future-pipeline-string") or "pipeline"
    future_pm.run.side_effect = AssertionError("future pass manager must not run during make_ttir")
    monkeypatch.setattr(ascend_compiler.ir, "pass_manager", lambda _context: canonical_pm)
    monkeypatch.setattr(ascend_compiler, "passes", MagicMock())
    monkeypatch.setattr(
        ascend_compiler,
        "_build_ttir_to_linalg_pass_manager",
        lambda *_args, **_kwargs: future_pm,
        raising=False,
    )
    _identity_runtime(monkeypatch)

    def policy(_mod, _metadata, _options, identity, stages):
        calls.append("policy")
        assert identity["open_source_pipeline"] == "pipeline"
        assert stages == [{"stage_name": ascend_compiler.TTIR_UB_BISHENG_SUFFIX_STAGE, "options": {}}]

    monkeypatch.setattr(ascend_compiler, "apply_ub_lower_bound_policy", policy, raising=False)
    options = ascend_compiler.NPUOptions(ub_lower_bound_mode="shadow", compile_on_910_95=False)
    module = SimpleNamespace(context=object(), __str__=lambda: "module")
    ascend_compiler.make_ttir(module, _compiler_metadata(), options)
    assert calls == ["canonical-pm-run", "future-pipeline-string", "policy"]


def test_make_ttir_off_preserves_old_behavior_without_identity_or_policy(monkeypatch):
    canonical_pm = MagicMock()
    monkeypatch.setattr(ascend_compiler.ir, "pass_manager", lambda _context: canonical_pm)
    monkeypatch.setattr(ascend_compiler, "passes", MagicMock())
    monkeypatch.setattr(
        ascend_compiler,
        "_build_ttir_to_linalg_pass_manager",
        lambda *_args, **_kwargs: pytest.fail("off mode built the future pipeline"),
        raising=False,
    )
    monkeypatch.setattr(
        ascend_compiler,
        "apply_ub_lower_bound_policy",
        lambda *_args, **_kwargs: pytest.fail("off mode called the policy"),
        raising=False,
    )
    options = ascend_compiler.NPUOptions(compile_on_910_95=False)
    module = SimpleNamespace(context=object(), __str__=lambda: "module")
    assert ascend_compiler.make_ttir(module, _compiler_metadata(), options) is module
    canonical_pm.run.assert_called_once_with(module)


def test_make_ttir_options_without_mode_preserve_old_off_behavior(monkeypatch):
    canonical_pm = MagicMock()
    monkeypatch.setattr(ascend_compiler.ir, "pass_manager", lambda _context: canonical_pm)
    monkeypatch.setattr(ascend_compiler, "passes", MagicMock())
    monkeypatch.setattr(
        ascend_compiler,
        "_build_ttir_to_linalg_pass_manager",
        lambda *_args, **_kwargs: pytest.fail("legacy options built the future pipeline"),
    )
    module = SimpleNamespace(context=object(), __str__=lambda: "module")
    options = SimpleNamespace(debug=False)
    assert ascend_compiler.make_ttir(module, _compiler_metadata(), options) is module


def test_make_ttir_identity_error_fails_open_to_policy(monkeypatch):
    canonical_pm = MagicMock()
    future_pm = MagicMock()
    future_pm.get_pipeline_str.return_value = "pipeline"
    monkeypatch.setattr(ascend_compiler.ir, "pass_manager", lambda _context: canonical_pm)
    monkeypatch.setattr(ascend_compiler, "passes", MagicMock())
    monkeypatch.setattr(ascend_compiler, "_build_ttir_to_linalg_pass_manager", lambda *_args, **_kwargs: future_pm)

    def identity_error(*_args):
        raise RuntimeError("identity unavailable")

    captured = []
    monkeypatch.setattr(ascend_compiler, "_ttir_ub_pipeline_identity", identity_error)
    monkeypatch.setattr(
        ascend_compiler,
        "apply_ub_lower_bound_policy",
        lambda _mod, _metadata, _options, identity, stages: captured.append((identity, stages)),
    )
    options = ascend_compiler.NPUOptions(ub_lower_bound_mode="shadow", compile_on_910_95=False)
    module = SimpleNamespace(context=object(), __str__=lambda: "module")
    ascend_compiler.make_ttir(module, _compiler_metadata(), options)
    assert captured == [("", [])]


def test_make_ttir_simt_identity_uses_real_direct_pipeline_without_future_pm(monkeypatch):
    canonical_pm = MagicMock()
    monkeypatch.setattr(ascend_compiler.ir, "pass_manager", lambda _context: canonical_pm)
    monkeypatch.setattr(ascend_compiler, "passes", MagicMock())
    monkeypatch.setattr(
        ascend_compiler,
        "_build_ttir_to_linalg_pass_manager",
        lambda *_args, **_kwargs: pytest.fail("direct SIMT path built an unused TTIR-to-Linalg pipeline"),
    )
    captured = []
    captured_stages = []
    monkeypatch.setattr(
        ascend_compiler,
        "_ttir_ub_pipeline_identity",
        lambda pipeline, _metadata, _canonical_ttir: captured.append(pipeline) or {"sha256": "direct-id"},
    )
    monkeypatch.setattr(
        ascend_compiler,
        "apply_ub_lower_bound_policy",
        lambda _mod, _metadata, _options, _identity, stages: captured_stages.extend(stages),
    )
    options = ascend_compiler.NPUOptions(
        ub_lower_bound_mode="shadow",
        compile_on_910_95=True,
        compile_mode="simt_only",
    )
    module = SimpleNamespace(context=object(), __str__=lambda: "module")
    ascend_compiler.make_ttir(module, _compiler_metadata(compile_on_910_95=True), options)
    assert captured == [ascend_compiler.TTIR_UB_DIRECT_BISHENG_PIPELINE]
    assert captured_stages == [{"stage_name": "bisheng.direct-ttir-pipeline", "options": {}}]


def test_direct_simt_libdevice_toggle_changes_command_and_identity(monkeypatch):
    _identity_runtime(monkeypatch)
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        output = Path(command[command.index("-o") + 1] + ".o")
        output.write_bytes(b"npubin")
        return SimpleNamespace(stderr=b"")

    monkeypatch.setattr(ascend_compiler.subprocess, "run", run)
    metadata = _compiler_metadata(bisheng_options="-mllvm -test-libdevice-option")
    options = ascend_compiler.NPUOptions(compile_on_910_95=True, compile_mode="simt_only")

    monkeypatch.setenv("TRITON_ENABLE_LIBDEVICE_SIMT", "0")
    disabled_identity = ascend_compiler._ttir_ub_pipeline_identity(
        ascend_compiler.TTIR_UB_DIRECT_BISHENG_PIPELINE,
        metadata,
    )
    ascend_compiler.ttir_to_npubin(DIRECT_LOAD_COPY, dict(metadata), options)

    monkeypatch.setenv("TRITON_ENABLE_LIBDEVICE_SIMT", "1")
    enabled_identity = ascend_compiler._ttir_ub_pipeline_identity(
        ascend_compiler.TTIR_UB_DIRECT_BISHENG_PIPELINE,
        metadata,
    )
    ascend_compiler.ttir_to_npubin(DIRECT_LOAD_COPY, dict(metadata), options)

    append_flag = "--append-bisheng-options=-mllvm -test-libdevice-option"
    assert append_flag not in commands[0]
    assert append_flag in commands[1]
    assert disabled_identity["sha256"] != enabled_identity["sha256"]
    disabled_options = json.loads(disabled_identity["relevant_options_json"])
    enabled_options = json.loads(enabled_identity["relevant_options_json"])
    assert disabled_options["effective_libdevice_simt"] is False
    assert disabled_options["effective_libdevice_bisheng_options_forwarded"] is False
    assert enabled_options["effective_libdevice_simt"] is True
    assert enabled_options["effective_libdevice_bisheng_options_forwarded"] is True


def test_debug_policy_dump_uses_dump_manager_and_contains_full_result(monkeypatch):
    result = _analysis_result("defer")
    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", lambda *_args: result)
    dump_manager = MagicMock()
    monkeypatch.setattr(ub_lower_bound, "get_dump_manager", lambda _hash: dump_manager, raising=False)
    metadata = {"hash": "debug-hash"}
    apply_ub_lower_bound_policy(object(), metadata, Options("shadow", debug=True), "test-id")
    dump_manager.put.assert_called_once()
    content, filename = dump_manager.put.call_args.args[:2]
    assert filename == "kernel.ttir.ub-lower-bound.json"
    assert dump_manager.put.call_args.kwargs == {"binary": False}
    assert json.loads(content) == result
    assert json.loads(content)["certificates"][0]["contract_trace"] == [
        "ttir-direct-load-v1",
        "direct-copy-max-tiles",
    ]


def test_core_compiler_preserves_ub_lower_bound_overflow(monkeypatch):
    error = UBLowerBoundOverflow(262144, 196608, {"kind": "singleton"}, "test-id")

    class FakeBackend:

        def parse_options(self, _options):
            return SimpleNamespace()

        def add_stages(self, stages, _options, _language):
            stages["ttir"] = lambda _module, _metadata: (_ for _ in ()).throw(error)

        def load_dialects(self, _context):
            pass

        def get_codegen_implementation(self, _options):
            return {}

        def get_module_map(self):
            return {}

    class FakeCacheManager:

        def get_group(self, _filename):
            return None

        def put(self, _value, filename, *args, **kwargs):
            return f"/tmp/{filename}"

    source = object.__new__(core_compiler.ASTSource)
    source.name = "ub_overflow"
    source.ext = "ttir"
    source.language = core_compiler.Language.TRITON
    source.hash = lambda: "source-hash"
    source.parse_options = lambda: {}
    source.make_ir = lambda *_args, **_kwargs: object()

    fake_ir = SimpleNamespace(context=lambda: object(), load_dialects=lambda _context: None)
    fake_dialect = SimpleNamespace(load_dialects=lambda _context: None)
    monkeypatch.setattr(core_compiler, "make_backend", lambda _target: FakeBackend())
    monkeypatch.setattr(core_compiler, "get_cache_key", lambda *_args, **_kwargs: "cache-key")
    monkeypatch.setattr(core_compiler, "get_cache_manager", lambda _hash: FakeCacheManager())
    monkeypatch.setattr(core_compiler, "ir", fake_ir)
    monkeypatch.setattr(core_compiler, "buffer_ir", fake_dialect)
    monkeypatch.setattr(core_compiler, "ascend_ir", fake_dialect)
    monkeypatch.setattr(core_compiler.knobs.compilation, "listener", None)
    monkeypatch.setattr(core_compiler.knobs.compilation, "override", False)
    monkeypatch.setattr(core_compiler.knobs.compilation, "dump_ir", False)
    monkeypatch.setattr(core_compiler.knobs.compilation, "store_binary_only", False)
    monkeypatch.setattr(core_compiler.knobs.compilation, "always_compile", True)
    monkeypatch.setattr(core_compiler.knobs.compilation, "use_ir_loc", None)
    target = core_compiler.GPUTarget("npu", "Ascend910B", 32)
    with pytest.raises(UBLowerBoundOverflow) as raised:
        core_compiler.compile(source, target=target, _env_vars={})
    assert raised.value is error
    assert raised.value.required == 262144
    assert raised.value.limit == 196608
    assert raised.value.certificate == {"kind": "singleton"}
    assert raised.value.pipeline_identity == "test-id"


def test_off_does_not_call_analyzer(monkeypatch):

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("off mode called the analyzer")

    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", unexpected_call)
    monkeypatch.setattr(ascend.analysis, "get_ub_capacity_bytes", unexpected_call)
    metadata = {"hash": "abc"}
    apply_ub_lower_bound_policy(object(), metadata, Options("off"), "test-id")
    assert metadata == {"hash": "abc"}


def test_shadow_records_but_does_not_raise(monkeypatch):
    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", lambda *_: _analysis_result())
    metadata = {"hash": "abc"}
    apply_ub_lower_bound_policy(object(), metadata, Options("shadow"), "test-id")
    assert metadata == {
        "hash": "abc",
        "ub_lower_bound_mode": "shadow",
        "ub_lower_bound_decision": "reject",
        "ub_lower_bound_bytes": 262144,
        "ub_capacity_bytes": 196608,
        "ub_lower_bound_contract_version": "ttir-ub-lb-v1",
        "ub_lower_bound_pipeline_identity": "test-id",
        "ub_lower_bound_certificate_count": 1,
        "ub_lower_bound_unsupported_reasons": [],
    }


def test_enforce_raises_only_proven_reject(monkeypatch):
    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", lambda *_: _analysis_result())
    with pytest.raises(UBLowerBoundOverflow) as error:
        apply_ub_lower_bound_policy(object(), {}, Options("enforce"), "test-id")
    assert error.value.required == 262144
    assert error.value.limit == 196608
    assert error.value.certificate == _analysis_result()["certificates"][0]
    assert error.value.pipeline_identity == "test-id"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("decision", ), None),
        (("decision", ), "unknown"),
        (("lower_bound_bytes", ), -1),
        (("lower_bound_bytes", ), True),
        (("lower_bound_bytes", ), INT64_MAX + 1),
        (("capacity_bytes", ), None),
        (("capacity_bytes", ), -1),
        (("capacity_bytes", ), False),
        (("capacity_bytes", ), INT64_MAX + 1),
        (("capacity_bytes", ), 1),
        (("certificates", ), None),
        (("certificates", ), [None]),
        (("certificates", ), [object()]),
        (("certificates", ), _analysis_result()["certificates"] * 2),
        (("certificates", 0, "kind"), None),
        (("certificates", 0, "kind"), "pairwise"),
        (("certificates", 0, "kind"), ExplosiveEquality()),
        (("certificates", 0, "bytes"), -1),
        (("certificates", 0, "bytes"), True),
        (("certificates", 0, "bytes"), INT64_MAX + 1),
        (("certificates", 0, "bytes"), 262143),
        (("certificates", 0, "resource_ids"), None),
        (("certificates", 0, "resource_ids"), []),
        (("certificates", 0, "resource_ids"), [True]),
        (("certificates", 0, "resource_ids"), [-1]),
        (("certificates", 0, "resource_ids"), [INT64_MAX + 1]),
        (("certificates", 0, "resource_ids"), [UINT32_MAX]),
        (("certificates", 0, "resource_ids"), [1 << 32]),
        (("certificates", 0, "resource_ids"), [0, 1]),
        (("certificates", 0, "resource_ids"), [object()]),
        (("certificates", 0, "contract_trace"), None),
        (("certificates", 0, "contract_trace"), []),
        (("certificates", 0, "contract_trace"), [""]),
        (("certificates", 0, "contract_trace"), [object()]),
        (("certificates", 0, "extra"), object()),
        (("unsupported_reasons", ), None),
        (("unsupported_reasons", ), [None]),
        (("unsupported_reasons", ), [object()]),
        (("contract_version", ), None),
        (("contract_version", ), object()),
        (("contract_version", ), ""),
        (("contract_version", ), "ttir-ub-lb-v2"),
        (("pipeline_identity", ), None),
        (("pipeline_identity", ), object()),
        (("pipeline_identity", ), "other-id"),
        (("extra", ), object()),
    ],
)
def test_invalid_analysis_schema_fails_open_with_json_metadata(monkeypatch, path, value):
    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", lambda *_: _result_with(path, value))
    metadata = {}
    apply_ub_lower_bound_policy(object(), metadata, Options("enforce"), "test-id")
    assert metadata["ub_lower_bound_decision"] == "defer"
    assert metadata["ub_lower_bound_bytes"] == 0
    assert metadata["ub_capacity_bytes"] is None
    assert metadata["ub_lower_bound_certificate_count"] == 0
    assert metadata["ub_lower_bound_unsupported_reasons"] == ["invalid-analysis-result"]
    assert metadata["ub_lower_bound_pipeline_identity"] == "test-id"
    json.dumps(metadata)


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_unknown_target_cannot_use_fabricated_analyzer_capacity(monkeypatch, mode):
    result = _analysis_result()
    result["capacity_bytes"] = 1
    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", lambda *_: result)
    metadata = {}
    apply_ub_lower_bound_policy(object(), metadata, Options(mode, arch="future-chip"), "test-id")
    assert metadata["ub_lower_bound_decision"] == "defer"
    assert metadata["ub_lower_bound_bytes"] == 0
    assert metadata["ub_capacity_bytes"] is None
    assert metadata["ub_lower_bound_certificate_count"] == 0
    assert metadata["ub_lower_bound_unsupported_reasons"] == ["invalid-analysis-result"]
    json.dumps(metadata)


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_capacity_lookup_exception_fails_open(monkeypatch, mode):

    def fail(_arch):
        raise RuntimeError("capacity lookup failed")

    monkeypatch.setattr(ascend.analysis, "get_ub_capacity_bytes", fail)
    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", lambda *_: _analysis_result())
    metadata = {}
    apply_ub_lower_bound_policy(object(), metadata, Options(mode), "test-id")
    assert metadata["ub_lower_bound_decision"] == "defer"
    assert metadata["ub_lower_bound_bytes"] == 0
    assert metadata["ub_capacity_bytes"] is None
    assert metadata["ub_lower_bound_certificate_count"] == 0
    assert metadata["ub_lower_bound_unsupported_reasons"] == ["invalid-analysis-result"]
    json.dumps(metadata)


def test_shadow_capacity_mismatch_records_canonical_defer(monkeypatch):
    result = _analysis_result()
    result["capacity_bytes"] = 1
    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", lambda *_: result)
    metadata = {}
    apply_ub_lower_bound_policy(object(), metadata, Options("shadow"), "test-id")
    assert metadata["ub_lower_bound_decision"] == "defer"
    assert metadata["ub_lower_bound_bytes"] == 0
    assert metadata["ub_capacity_bytes"] is None
    assert metadata["ub_lower_bound_certificate_count"] == 0
    assert metadata["ub_lower_bound_unsupported_reasons"] == ["invalid-analysis-result"]
    json.dumps(metadata)


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_known_target_defer_capacity_mismatch_is_canonical_invalid(monkeypatch, mode):
    result = _analysis_result("defer")
    result["capacity_bytes"] = 1
    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", lambda *_: result)
    metadata = {}
    apply_ub_lower_bound_policy(object(), metadata, Options(mode), "test-id")
    assert metadata["ub_lower_bound_decision"] == "defer"
    assert metadata["ub_lower_bound_bytes"] == 0
    assert metadata["ub_capacity_bytes"] is None
    assert metadata["ub_lower_bound_certificate_count"] == 0
    assert metadata["ub_lower_bound_unsupported_reasons"] == ["invalid-analysis-result"]
    json.dumps(metadata)


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_unknown_target_defer_fabricated_capacity_is_canonical_invalid(monkeypatch, mode):
    result = _analysis_result("defer")
    result["capacity_bytes"] = 1
    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", lambda *_: result)
    metadata = {}
    apply_ub_lower_bound_policy(object(), metadata, Options(mode, arch="future-chip"), "test-id")
    assert metadata["ub_lower_bound_decision"] == "defer"
    assert metadata["ub_lower_bound_bytes"] == 0
    assert metadata["ub_capacity_bytes"] is None
    assert metadata["ub_lower_bound_certificate_count"] == 0
    assert metadata["ub_lower_bound_unsupported_reasons"] == ["invalid-analysis-result"]
    json.dumps(metadata)


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
@pytest.mark.parametrize("capacity", [None, 192 * 1024])
def test_known_target_defer_accepts_none_or_trusted_capacity(monkeypatch, mode, capacity):
    result = _analysis_result("defer")
    result["capacity_bytes"] = capacity
    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", lambda *_: result)
    metadata = {}
    apply_ub_lower_bound_policy(object(), metadata, Options(mode), "test-id")
    assert metadata["ub_lower_bound_decision"] == "defer"
    assert metadata["ub_lower_bound_bytes"] == 262144
    assert metadata["ub_capacity_bytes"] == capacity
    assert metadata["ub_lower_bound_certificate_count"] == 1
    assert metadata["ub_lower_bound_unsupported_reasons"] == []
    json.dumps(metadata)


def test_mapping_pipeline_identity_uses_sha256_fingerprint(monkeypatch):
    identity = {"sha256": "mapped-id"}
    result = _analysis_result("defer")
    result["pipeline_identity"] = "mapped-id"
    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", lambda *_: result)
    metadata = {}
    apply_ub_lower_bound_policy(object(), metadata, Options("shadow"), identity)
    assert metadata["ub_lower_bound_pipeline_identity"] == "mapped-id"
    json.dumps(metadata)


@pytest.mark.parametrize("identity", [None, object(), {}, {"sha256": None}, {"sha256": object()}])
def test_bad_caller_identity_is_string_on_exception_path(monkeypatch, identity):

    def fail(*_args):
        raise RuntimeError("bad")

    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", fail)
    metadata = {}
    apply_ub_lower_bound_policy(object(), metadata, Options("enforce"), identity)
    assert metadata["ub_lower_bound_pipeline_identity"] == ""
    json.dumps(metadata)


def test_mapping_identity_exception_path_uses_sha256_string(monkeypatch):

    def fail(*_args):
        raise RuntimeError("bad")

    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", fail)
    metadata = {}
    apply_ub_lower_bound_policy(object(), metadata, Options("enforce"), {"sha256": "mapped-id"})
    assert metadata["ub_lower_bound_pipeline_identity"] == "mapped-id"
    json.dumps(metadata)


@pytest.mark.parametrize("decision", ["defer", "unknown"])
def test_enforce_defers_non_proven_results(monkeypatch, decision):
    result = _analysis_result(decision)
    result["unsupported_reasons"] = ["unsupported"]
    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", lambda *_: result)
    metadata = {}
    apply_ub_lower_bound_policy(object(), metadata, Options("enforce"), "test-id")
    assert metadata["ub_lower_bound_decision"] == "defer"


def test_analyzer_exception_fails_open(monkeypatch):

    def fail(*_args):
        raise RuntimeError("bad")

    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", fail)
    metadata = {}
    apply_ub_lower_bound_policy(object(), metadata, Options("enforce"), "test-id")
    assert metadata["ub_lower_bound_decision"] == "defer"
    assert metadata["ub_lower_bound_unsupported_reasons"] == ["internal-error: bad"]


def test_policy_passes_packaged_profile_and_real_pipeline_stages(monkeypatch):
    captured = {}

    def analyze(_mod, options):
        captured.update(options)
        return _analysis_result("defer")

    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", analyze)
    pipeline_stages = [{"stage_name": "ttir.auto-blockify", "options": {"auto_blockify_size": "1"}}]
    apply_ub_lower_bound_policy(object(), {}, Options("shadow"), "test-id", pipeline_stages)
    assert captured["contract_profile"] == {
        "schema": "ttir-ub-lb-profile-v1",
        "identity_contract": {
            "auto_tile_and_bind_subblock": {
                "identity_value": "module-derived-per-exact-ttir",
                "profile_promotion_requires": "oracle-validates-exact-profile-outcome",
            },
        },
        "profiles": [],
    }
    assert "allow_unvalidated" not in captured
    assert captured["compile_mode"] == "aiv"
    assert captured["pipeline_stages"] == pipeline_stages


def test_non_vector_compile_mode_is_not_reclassified(monkeypatch):
    captured = {}

    def analyze(_mod, options):
        captured.update(options)
        return _analysis_result("defer")

    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", analyze)
    apply_ub_lower_bound_policy(object(), {}, Options("shadow", compile_mode="simt_only"), "test-id")
    assert captured["compile_mode"] == "simt_only"


def test_simd_compile_mode_is_mapped_to_aiv_core_kind(monkeypatch):
    captured = {}

    def analyze(_mod, options):
        captured.update(options)
        return _analysis_result("defer")

    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", analyze)
    apply_ub_lower_bound_policy(object(), {}, Options("shadow", compile_mode="simd"), "test-id")
    assert captured["compile_mode"] == "aiv"


def test_capacity_boundary_is_not_rejected(monkeypatch):
    result = _analysis_result("defer")
    result["lower_bound_bytes"] = result["capacity_bytes"]
    result["certificates"][0]["bytes"] = result["capacity_bytes"]
    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", lambda *_args: result)
    metadata = {}
    apply_ub_lower_bound_policy(object(), metadata, Options("enforce"), "test-id")
    assert metadata["ub_lower_bound_decision"] == "defer"
    assert metadata["ub_lower_bound_bytes"] == metadata["ub_capacity_bytes"]


def test_witness_certificate_is_preserved_by_policy_validation(monkeypatch):
    result = _analysis_result()
    result["certificates"][0].update({
        "kind": "witness",
        "resource_ids": [0, 1],
        "contract_trace": ["ttir-binary-add-v1", "binary-add-max-tiles"],
    })
    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", lambda *_args: result)
    metadata = {}

    with pytest.raises(UBLowerBoundOverflow) as raised:
        apply_ub_lower_bound_policy(object(), metadata, Options("enforce"), "test-id")
    assert raised.value.certificate["kind"] == "witness"
    assert raised.value.certificate["resource_ids"] == [0, 1]


def test_overflow_exception_pickles_across_process_pool():
    error = UBLowerBoundOverflow(262144, 196608, {"kind": "singleton"}, "test-id")
    with ProcessPoolExecutor(max_workers=1) as executor:
        restored = executor.submit(_pickle_round_trip, error).result()
    assert type(restored) is UBLowerBoundOverflow
    assert (restored.required, restored.limit) == (262144, 196608)
    assert restored.certificate == {"kind": "singleton"}
    assert restored.pipeline_identity == "test-id"


def test_packaged_contract_profiles_start_empty():
    assert load_contract_profiles() == {
        "schema": "ttir-ub-lb-profile-v1",
        "identity_contract": {
            "auto_tile_and_bind_subblock": {
                "identity_value": "module-derived-per-exact-ttir",
                "profile_promotion_requires": "oracle-validates-exact-profile-outcome",
            },
        },
        "profiles": [],
    }


def test_profile_loader_rejects_missing_contract_version(monkeypatch, tmp_path):
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(json.dumps({
        "schema": "ttir-ub-lb-profile-v1",
        "identity_contract": ub_lower_bound._IDENTITY_CONTRACT,
        "profiles": [{
            "pipeline_identity": {
                "open_source_pipeline": "pipeline",
                "canonical_ttir_sha256": hashlib.sha256(b"ttir").hexdigest(),
                "relevant_options_json": "{}",
                "target_arch": "Ascend910B",
                "triton_version": "test",
                "cann_version_hash": "test",
                "sha256": "identity",
            },
            "pipeline_stages": [{
                "stage_name": "stage",
                "options": {},
                "contract_id": "invalidate-unmodeled-stage",
            }],
        }],
    }))
    monkeypatch.setattr(ub_lower_bound, "_PROFILE_PATH", profile_path)
    with pytest.raises(ValueError, match="invalid packaged"):
        load_contract_profiles()


def test_profile_loader_rejects_duplicate_identity(monkeypatch, tmp_path):
    profile_path = tmp_path / "profiles.json"
    identity = {
        "open_source_pipeline": "pipeline",
        "canonical_ttir_sha256": hashlib.sha256(b"ttir").hexdigest(),
        "relevant_options_json": "{}",
        "target_arch": "Ascend910B",
        "triton_version": "test",
        "cann_version_hash": "test",
        "sha256": "",
    }
    identity["sha256"] = hashlib.sha256(json.dumps({
        "cann_version_hash": "test",
        "canonical_ttir_sha256": identity["canonical_ttir_sha256"],
        "open_source_pipeline": "pipeline",
        "relevant_options": {},
        "target_arch": "Ascend910B",
        "triton_version": "test",
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    entry = {
        "pipeline_identity": identity,
        "pipeline_stages": [{
            "stage_name": "stage",
            "options": {},
            "contract_id": "invalidate-unmodeled-stage",
            "contract_version": "1",
            "contract_parameters": {},
        }],
        "contract_version": "ttir-ub-lb-v1",
        "oracle_report_sha256": "0" * 64,
        "semantic_model_sha256": "1" * 64,
        "validated_seeds": list(range(20)),
        "retry_validated": True,
        "auto_tile_and_bind_subblock_outcome": False,
    }
    profile_path.write_text(json.dumps({
        "schema": "ttir-ub-lb-profile-v1",
        "identity_contract": ub_lower_bound._IDENTITY_CONTRACT,
        "profiles": [entry, entry],
    }))
    monkeypatch.setattr(ub_lower_bound, "_PROFILE_PATH", profile_path)
    with pytest.raises(ValueError, match="duplicate packaged"):
        load_contract_profiles()


def _direct_copy_profile_entry(*, compile_mode="simd", multibuffer=False):
    relevant_options = {"compile_mode": compile_mode, "multibuffer": multibuffer}
    identity = {
        "open_source_pipeline": "pipeline",
        "canonical_ttir_sha256": "a" * 64,
        "relevant_options_json": json.dumps(relevant_options, sort_keys=True, separators=(",", ":")),
        "target_arch": "Ascend910B",
        "triton_version": "test",
        "cann_version_hash": "test",
        "sha256": "",
    }
    identity["sha256"] = hashlib.sha256(json.dumps({
        "cann_version_hash": identity["cann_version_hash"],
        "canonical_ttir_sha256": identity["canonical_ttir_sha256"],
        "open_source_pipeline": identity["open_source_pipeline"],
        "relevant_options": relevant_options,
        "target_arch": identity["target_arch"],
        "triton_version": identity["triton_version"],
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "pipeline_identity": identity,
        "pipeline_stages": [{
            "stage_name": "bisheng.ub-affecting-suffix",
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
        "contract_version": "ttir-ub-lb-v1",
        "oracle_report_sha256": "b" * 64,
        "semantic_model_sha256": "c" * 64,
        "validated_seeds": list(range(20)),
        "retry_validated": True,
        "auto_tile_and_bind_subblock_outcome": False,
    }


def test_profile_loader_accepts_certified_direct_copy_schema(monkeypatch, tmp_path):
    profile_path = tmp_path / "profiles.json"
    document = {
        "schema": "ttir-ub-lb-profile-v1",
        "identity_contract": ub_lower_bound._IDENTITY_CONTRACT,
        "profiles": [_direct_copy_profile_entry()],
    }
    profile_path.write_text(json.dumps(document))
    monkeypatch.setattr(ub_lower_bound, "_PROFILE_PATH", profile_path)
    assert load_contract_profiles() == document


def test_profile_loader_accepts_certified_binary_add_schema(monkeypatch, tmp_path):
    profile_path = tmp_path / "profiles.json"
    entry = _direct_copy_profile_entry()
    stage = entry["pipeline_stages"][0]
    stage["contract_id"] = "binary-add-max-tiles"
    stage["contract_parameters"]["expected_resource_count"] = "2"
    document = {
        "schema": "ttir-ub-lb-profile-v1",
        "identity_contract": ub_lower_bound._IDENTITY_CONTRACT,
        "profiles": [entry],
    }
    profile_path.write_text(json.dumps(document))
    monkeypatch.setattr(ub_lower_bound, "_PROFILE_PATH", profile_path)
    assert load_contract_profiles() == document


def test_profile_loader_accepts_certified_loop_carried_schema(
    monkeypatch, tmp_path
):
    profile_path = tmp_path / "profiles.json"
    entry = _direct_copy_profile_entry()
    stage = entry["pipeline_stages"][0]
    stage["contract_id"] = "loop-carried-add-max-tiles"
    stage["contract_parameters"]["expected_resource_count"] = "2"
    stage["contract_parameters"]["max_tiles"] = "1"
    document = {
        "schema": "ttir-ub-lb-profile-v1",
        "identity_contract": ub_lower_bound._IDENTITY_CONTRACT,
        "profiles": [entry],
    }
    profile_path.write_text(json.dumps(document))
    monkeypatch.setattr(ub_lower_bound, "_PROFILE_PATH", profile_path)
    assert load_contract_profiles() == document


def test_profile_loader_accepts_certified_reshape_copy_schema(monkeypatch, tmp_path):
    profile_path = tmp_path / "profiles.json"
    entry = _direct_copy_profile_entry()
    stage = entry["pipeline_stages"][0]
    stage["contract_id"] = "reshape-copy-max-tiles"
    stage["contract_parameters"]["expected_resource_count"] = "2"
    document = {
        "schema": "ttir-ub-lb-profile-v1",
        "identity_contract": ub_lower_bound._IDENTITY_CONTRACT,
        "profiles": [entry],
    }
    profile_path.write_text(json.dumps(document))
    monkeypatch.setattr(ub_lower_bound, "_PROFILE_PATH", profile_path)
    assert load_contract_profiles() == document


def test_profile_loader_accepts_certified_reduction_sum_schema(monkeypatch, tmp_path):
    profile_path = tmp_path / "profiles.json"
    entry = _direct_copy_profile_entry()
    stage = entry["pipeline_stages"][0]
    stage["contract_id"] = "reduction-sum-extra-buffer"
    stage["contract_parameters"] = {
        "expected_resource_count": "3",
        "expected_source_elements": "65536",
        "expected_element_bit_width": "32",
        "expected_input_payload_bytes": "262144",
        "expected_scratch_payload_bytes": "131072",
        "expected_accumulator_payload_bytes": "4",
    }
    document = {
        "schema": "ttir-ub-lb-profile-v1",
        "identity_contract": ub_lower_bound._IDENTITY_CONTRACT,
        "profiles": [entry],
    }
    profile_path.write_text(json.dumps(document))
    monkeypatch.setattr(ub_lower_bound, "_PROFILE_PATH", profile_path)
    assert load_contract_profiles() == document


@pytest.mark.parametrize("value", [None, "x" * 64, "A" * 64])
def test_profile_loader_rejects_invalid_semantic_model_hash(monkeypatch, tmp_path, value):
    profile_path = tmp_path / "profiles.json"
    entry = _direct_copy_profile_entry()
    if value is None:
        entry.pop("semantic_model_sha256")
    else:
        entry["semantic_model_sha256"] = value
    profile_path.write_text(json.dumps({
        "schema": "ttir-ub-lb-profile-v1",
        "identity_contract": ub_lower_bound._IDENTITY_CONTRACT,
        "profiles": [entry],
    }))
    monkeypatch.setattr(ub_lower_bound, "_PROFILE_PATH", profile_path)
    with pytest.raises(ValueError, match="invalid packaged"):
        load_contract_profiles()


def test_profile_loader_rejects_non_boolean_auto_tile_outcome(monkeypatch, tmp_path):
    profile_path = tmp_path / "profiles.json"
    entry = _direct_copy_profile_entry()
    entry["auto_tile_and_bind_subblock_outcome"] = [False, True]
    profile_path.write_text(json.dumps({
        "schema": "ttir-ub-lb-profile-v1",
        "identity_contract": ub_lower_bound._IDENTITY_CONTRACT,
        "profiles": [entry],
    }))
    monkeypatch.setattr(ub_lower_bound, "_PROFILE_PATH", profile_path)
    with pytest.raises(ValueError, match="invalid packaged"):
        load_contract_profiles()


@pytest.mark.parametrize(
    ("compile_mode", "multibuffer"),
    [("simt", False), ("simd", True)],
)
def test_profile_loader_rejects_uncertified_direct_copy_modes(
        monkeypatch, tmp_path, compile_mode, multibuffer):
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(json.dumps({
        "schema": "ttir-ub-lb-profile-v1",
        "identity_contract": ub_lower_bound._IDENTITY_CONTRACT,
        "profiles": [_direct_copy_profile_entry(
            compile_mode=compile_mode, multibuffer=multibuffer)],
    }))
    monkeypatch.setattr(ub_lower_bound, "_PROFILE_PATH", profile_path)
    with pytest.raises(ValueError, match="invalid packaged"):
        load_contract_profiles()


@pytest.mark.parametrize(
    ("arch", "expected"),
    [
        ("Ascend910B", 192 * 1024),
        ("Ascend910_93", 192 * 1024),
        ("Ascend910B1", 192 * 1024),
        ("Ascend910B2", 192 * 1024),
        ("Ascend910B3", 192 * 1024),
        ("Ascend910B4", 192 * 1024),
        ("Ascend910_9362", 192 * 1024),
        ("Ascend910_9372", 192 * 1024),
        ("Ascend910_9381", 192 * 1024),
        ("Ascend910_9382", 192 * 1024),
        ("Ascend910_9391", 192 * 1024),
        ("Ascend910_9392", 192 * 1024),
        ("Ascend310B1", 248 * 1024),
        ("Ascend310B2", 248 * 1024),
        ("Ascend310B3", 248 * 1024),
        ("Ascend310B4", 248 * 1024),
        ("Ascend910_95", 256 * 1024),
        ("Ascend910_9579", 256 * 1024),
        ("Ascend910_9581", 256 * 1024),
        ("Ascend910_9589", 256 * 1024),
        ("Ascend910_9599", 256 * 1024),
        ("Ascend950", 256 * 1024),
    ],
)
def test_binding_capacity_source(arch, expected):
    assert ascend.analysis.get_ub_capacity_bytes(arch) == expected


@pytest.mark.parametrize(
    "arch",
    [
        "future-chip",
        "Ascend910B-future",
        "Ascend910BLAH",
        "Ascend910_93future",
        "Ascend910_95future",
        "Ascend950Future",
        "Ascend310B",
        "Ascend310B5",
    ],
)
def test_binding_capacity_unknown_is_none(arch):
    assert ascend.analysis.get_ub_capacity_bytes(arch) is None


def test_binding_result_is_json_serializable():
    context = ir.context()
    module = ir.builder(context).create_module()
    result = ascend.analysis.ttir_ub_lower_bound(
        module,
        {
            "arch": "Ascend910B",
            "compile_mode": "aiv",
            "pipeline_identity": "unknown-production-profile",
            "pipeline_stages": [],
            "contract_profile": load_contract_profiles(),
        },
    )
    json.dumps(result)
    assert result["decision"] == "defer"
    assert result["capacity_bytes"] is None


def test_binding_does_not_expose_allow_unvalidated():
    context = ir.context()
    module = ir.builder(context).create_module()
    with pytest.raises((KeyError, TypeError, ValueError), match="allow_unvalidated"):
        ascend.analysis.ttir_ub_lower_bound(
            module,
            {
                "arch": "Ascend910B",
                "compile_mode": "aiv",
                "pipeline_identity": "anything",
                "pipeline_stages": [],
                "contract_profile": load_contract_profiles(),
                "allow_unvalidated": True,
            },
        )


def test_caller_supplied_contract_profile_cannot_enable_rejection(tmp_path):
    source = tmp_path / "direct-load.ttir"
    source.write_text(DIRECT_LOAD_COPY)
    context = ir.context()
    ascend.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    result = ascend.analysis.ttir_ub_lower_bound(
        module,
        {
            "arch": "Ascend910B",
            "compile_mode": "aiv",
            "pipeline_identity": {
                "open_source_pipeline": "synthetic-all-preserve",
                "canonical_ttir_sha256": hashlib.sha256(DIRECT_LOAD_COPY.encode()).hexdigest(),
                "relevant_options_json": "{}",
                "target_arch": "Ascend910B",
                "triton_version": "test",
                "cann_version_hash": "test",
                "sha256": "synthetic-all-preserve-v1",
            },
            "pipeline_stages": [{"stage_name": "preserve", "options": {}}],
            "contract_profile": {
                "schema": "ttir-ub-lb-profile-v1",
                "profiles": [{"contract": "fixed-disposition", "disposition": "preserve"}],
            },
        },
    )
    assert result["decision"] == "defer"
    assert result["unsupported_reasons"] == ["unknown-pipeline-profile"]


def test_binding_loads_exact_ordered_fail_closed_profile(tmp_path):
    source = tmp_path / "direct-load.ttir"
    source.write_text(DIRECT_LOAD_COPY)
    context = ir.context()
    ascend.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    identity = {
        "open_source_pipeline": "p0-test-pipeline",
        "canonical_ttir_sha256": hashlib.sha256(DIRECT_LOAD_COPY.encode()).hexdigest(),
        "relevant_options_json": "{}",
        "target_arch": "Ascend910B",
        "triton_version": "test",
        "cann_version_hash": "test",
        "sha256": "p0-test-identity",
    }
    stages = [{"stage_name": "ttir.auto-blockify", "options": {"auto_blockify_size": "1"}}]
    profile = {
        "schema": "ttir-ub-lb-profile-v1",
        "profiles": [{
            "pipeline_identity": identity,
            "pipeline_stages": [{
                **stages[0],
                "contract_id": "invalidate-unmodeled-stage",
                "contract_version": "1",
            }],
        }],
    }
    result = ascend.analysis.ttir_ub_lower_bound(
        module,
        {
            "arch": "Ascend910B",
            "compile_mode": "aiv",
            "pipeline_identity": identity,
            "pipeline_stages": stages,
            "contract_profile": profile,
        },
    )
    assert result["decision"] == "defer"
    assert result["unsupported_reasons"] == ["invalidate-unmodeled-stage"]
    assert result["certificates"] == []


def test_binding_runs_parameterized_direct_copy_contract_chain(tmp_path):
    source = tmp_path / "direct-load.ttir"
    source.write_text(DIRECT_LOAD_COPY)
    context = ir.context()
    ascend.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    identity = {
        "open_source_pipeline": "p1-direct-copy-test",
        "canonical_ttir_sha256": hashlib.sha256(DIRECT_LOAD_COPY.encode()).hexdigest(),
        "relevant_options_json": "{}",
        "target_arch": "Ascend910B",
        "triton_version": "test",
        "cann_version_hash": "test",
        "sha256": "p1-direct-copy-test-identity",
    }
    stages = [
        {"stage_name": "canonicalize", "options": {}},
        {"stage_name": "ttir.triton-to-linalg", "options": {"named_ops": "true"}},
        {"stage_name": "bisheng.ub-affecting-suffix", "options": {}},
    ]
    common = {
        "expected_resource_count": "1",
        "expected_source_elements": "65536",
        "expected_element_bit_width": "32",
    }
    profile = {
        "schema": "ttir-ub-lb-profile-v1",
        "profiles": [{
            "pipeline_identity": identity,
            "pipeline_stages": [
                {
                    **stages[0],
                    "contract_id": "direct-copy-preserve",
                    "contract_version": "1",
                    "contract_parameters": {
                        **common, "expected_input_payload_bytes": "262144",
                    },
                },
                {
                    **stages[1],
                    "contract_id": "direct-copy-max-tiles",
                    "contract_version": "1",
                    "contract_parameters": {
                        **common,
                        "expected_input_payload_bytes": "262144",
                        "max_tiles": "64",
                    },
                },
                {
                    **stages[2],
                    "contract_id": "direct-copy-preserve",
                    "contract_version": "1",
                    "contract_parameters": {
                        **common, "expected_input_payload_bytes": "4096",
                    },
                },
            ],
        }],
    }
    raw_options = {
        "arch": "Ascend910B",
        "compile_mode": "aiv",
        "pipeline_identity": identity,
        "pipeline_stages": stages,
        "contract_profile": profile,
    }
    production_result = ascend.analysis.ttir_ub_lower_bound(module, raw_options)
    assert production_result["decision"] == "defer"
    assert production_result["lower_bound_bytes"] == 0
    assert production_result["unsupported_reasons"] == ["unknown-pipeline-profile"]

    result = ascend.analysis.ttir_ub_lower_bound_candidate_for_oracle(
        module,
        raw_options,
    )
    assert result["decision"] == "defer"
    assert result["lower_bound_bytes"] == 4096
    assert result["unsupported_reasons"] == []
    assert result["certificates"] == [{
        "kind": "singleton",
        "bytes": 4096,
        "resource_ids": [0],
        "contract_trace": [
            "ttir-direct-load-v1",
            "direct-copy-preserve",
            "direct-copy-max-tiles",
            "direct-copy-preserve",
        ],
    }]

    profile["profiles"][0].update({
        "contract_version": "ttir-ub-lb-v1",
        "oracle_report_sha256": "a" * 64,
        "validated_seeds": list(range(20)),
        "retry_validated": True,
        "auto_tile_and_bind_subblock_outcome": False,
    })
    missing_semantic_model_result = ascend.analysis.ttir_ub_lower_bound(module, raw_options)
    assert missing_semantic_model_result["lower_bound_bytes"] == 0
    assert missing_semantic_model_result["unsupported_reasons"] == ["unknown-pipeline-profile"]

    profile["profiles"][0]["semantic_model_sha256"] = "b" * 64
    certified_result = ascend.analysis.ttir_ub_lower_bound(module, raw_options)
    assert certified_result["lower_bound_bytes"] == 4096
    assert certified_result["unsupported_reasons"] == []


def test_binding_runs_binary_add_witness_contract(tmp_path):
    source = tmp_path / "binary-add.ttir"
    source.write_text(BINARY_ADD)
    context = ir.context()
    ascend.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    identity = {
        "open_source_pipeline": "p2-binary-add-test",
        "canonical_ttir_sha256": hashlib.sha256(BINARY_ADD.encode()).hexdigest(),
        "relevant_options_json": "{}",
        "target_arch": "Ascend910B",
        "triton_version": "test",
        "cann_version_hash": "test",
        "sha256": "p2-binary-add-test-identity",
    }
    stages = [{"stage_name": "ttir.triton-to-linalg", "options": {}}]
    profile_entry = {
        "pipeline_identity": identity,
        "pipeline_stages": [{
            **stages[0],
            "contract_id": "binary-add-max-tiles",
            "contract_version": "1",
            "contract_parameters": {
                "expected_resource_count": "2",
                "expected_source_elements": "65536",
                "expected_element_bit_width": "32",
                "expected_input_payload_bytes": "262144",
                "max_tiles": "64",
            },
        }],
    }
    profile = {"schema": "ttir-ub-lb-profile-v1", "profiles": [profile_entry]}
    raw_options = {
        "arch": "Ascend910B",
        "compile_mode": "aiv",
        "pipeline_identity": identity,
        "pipeline_stages": stages,
        "contract_profile": profile,
    }

    production_result = ascend.analysis.ttir_ub_lower_bound(module, raw_options)
    assert production_result["lower_bound_bytes"] == 0
    assert production_result["unsupported_reasons"] == ["unknown-pipeline-profile"]

    candidate_result = ascend.analysis.ttir_ub_lower_bound_candidate_for_oracle(
        module, raw_options)
    assert candidate_result["lower_bound_bytes"] == 8192
    assert candidate_result["unsupported_reasons"] == []
    assert candidate_result["certificates"] == [{
        "kind": "witness",
        "bytes": 8192,
        "resource_ids": [0, 1],
        "contract_trace": ["ttir-binary-add-v1", "binary-add-max-tiles"],
    }]

    profile_entry.update({
        "contract_version": "ttir-ub-lb-v1",
        "oracle_report_sha256": "a" * 64,
        "semantic_model_sha256": "b" * 64,
        "validated_seeds": list(range(20)),
        "retry_validated": True,
        "auto_tile_and_bind_subblock_outcome": False,
    })
    certified_result = ascend.analysis.ttir_ub_lower_bound(module, raw_options)
    assert certified_result["lower_bound_bytes"] == 8192
    assert certified_result["certificates"] == candidate_result["certificates"]


def test_binding_runs_reshape_copy_alias_contract(tmp_path):
    source = tmp_path / "reshape-copy.ttir"
    source.write_text(RESHAPE_COPY)
    context = ir.context()
    ascend.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    identity = {
        "open_source_pipeline": "p2-reshape-copy-test",
        "canonical_ttir_sha256": hashlib.sha256(RESHAPE_COPY.encode()).hexdigest(),
        "relevant_options_json": "{}",
        "target_arch": "Ascend910B",
        "triton_version": "test",
        "cann_version_hash": "test",
        "sha256": "p2-reshape-copy-test-identity",
    }
    stages = [{"stage_name": "ttir.triton-to-linalg", "options": {}}]
    profile_entry = {
        "pipeline_identity": identity,
        "pipeline_stages": [{
            **stages[0],
            "contract_id": "reshape-copy-max-tiles",
            "contract_version": "1",
            "contract_parameters": {
                "expected_resource_count": "2",
                "expected_source_elements": "65536",
                "expected_element_bit_width": "32",
                "expected_input_payload_bytes": "262144",
                "max_tiles": "64",
            },
        }],
    }
    profile = {"schema": "ttir-ub-lb-profile-v1", "profiles": [profile_entry]}
    raw_options = {
        "arch": "Ascend910B",
        "compile_mode": "aiv",
        "pipeline_identity": identity,
        "pipeline_stages": stages,
        "contract_profile": profile,
    }

    candidate_result = ascend.analysis.ttir_ub_lower_bound_candidate_for_oracle(
        module, raw_options)
    assert candidate_result["lower_bound_bytes"] == 4096
    assert candidate_result["unsupported_reasons"] == []
    assert candidate_result["certificates"] == [{
        "kind": "singleton",
        "bytes": 4096,
        "resource_ids": [0],
        "contract_trace": ["ttir-reshape-copy-v1", "reshape-copy-max-tiles"],
    }]

    profile_entry.update({
        "contract_version": "ttir-ub-lb-v1",
        "oracle_report_sha256": "a" * 64,
        "semantic_model_sha256": "b" * 64,
        "validated_seeds": list(range(20)),
        "retry_validated": True,
        "auto_tile_and_bind_subblock_outcome": False,
    })
    certified_result = ascend.analysis.ttir_ub_lower_bound(module, raw_options)
    assert certified_result["certificates"] == candidate_result["certificates"]


def test_binding_runs_reduction_sum_extra_buffer_contract(tmp_path):
    source = tmp_path / "reduction-sum.ttir"
    source.write_text(REDUCTION_SUM)
    context = ir.context()
    ascend.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    identity = {
        "open_source_pipeline": "p3-reduction-sum-test",
        "canonical_ttir_sha256": hashlib.sha256(
            REDUCTION_SUM.encode()
        ).hexdigest(),
        "relevant_options_json": "{}",
        "target_arch": "Ascend910B",
        "triton_version": "test",
        "cann_version_hash": "test",
        "sha256": "p3-reduction-sum-test-identity",
    }
    stages = [
        {"stage_name": "ttir.triton-to-linalg", "options": {}},
        {"stage_name": "bisheng.ub-affecting-suffix", "options": {}},
    ]
    common = {
        "expected_resource_count": "3",
        "expected_source_elements": "65536",
        "expected_element_bit_width": "32",
        "expected_input_payload_bytes": "262144",
        "expected_scratch_payload_bytes": "131072",
        "expected_accumulator_payload_bytes": "4",
    }
    profile_entry = {
        "pipeline_identity": identity,
        "pipeline_stages": [
            {
                **stages[0],
                "contract_id": "reduction-sum-max-tiles",
                "contract_version": "1",
                "contract_parameters": {**common, "max_tiles": "1"},
            },
            {
                **stages[1],
                "contract_id": "reduction-sum-extra-buffer",
                "contract_version": "1",
                "contract_parameters": common,
            },
        ],
    }
    profile = {"schema": "ttir-ub-lb-profile-v1", "profiles": [profile_entry]}
    raw_options = {
        "arch": "Ascend910B",
        "compile_mode": "aiv",
        "pipeline_identity": identity,
        "pipeline_stages": stages,
        "contract_profile": profile,
    }

    candidate_result = ascend.analysis.ttir_ub_lower_bound_candidate_for_oracle(
        module, raw_options
    )
    assert candidate_result["decision"] == "reject"
    assert candidate_result["lower_bound_bytes"] == 393220
    assert candidate_result["unsupported_reasons"] == []
    assert candidate_result["certificates"] == [{
        "kind": "witness",
        "bytes": 393220,
        "resource_ids": [0, 1, 2],
        "contract_trace": [
            "ttir-reduction-sum-v1",
            "reduction-sum-max-tiles",
            "reduction-sum-extra-buffer",
        ],
    }]


def test_binding_runs_loop_carried_lifetime_contract(tmp_path):
    source = tmp_path / "loop-carried-add.ttir"
    source.write_text(LOOP_CARRIED_ADD)
    context = ir.context()
    ascend.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    identity = {
        "open_source_pipeline": "p3-loop-carried-add-test",
        "canonical_ttir_sha256": hashlib.sha256(
            LOOP_CARRIED_ADD.encode()
        ).hexdigest(),
        "relevant_options_json": "{}",
        "target_arch": "Ascend910B1",
        "triton_version": "test",
        "cann_version_hash": "test",
        "sha256": "p3-loop-carried-add-test-identity",
    }
    stages = [
        {"stage_name": "ttir.triton-to-linalg", "options": {}},
        {"stage_name": "bisheng.ub-affecting-suffix", "options": {}},
    ]
    common = {
        "expected_resource_count": "2",
        "expected_source_elements": "65536",
        "expected_element_bit_width": "32",
        "expected_input_payload_bytes": "262144",
    }
    profile = {
        "schema": "ttir-ub-lb-profile-v1",
        "profiles": [{
            "pipeline_identity": identity,
            "pipeline_stages": [
                {
                    **stages[0],
                    "contract_id": "loop-carried-add-max-tiles",
                    "contract_version": "1",
                    "contract_parameters": {**common, "max_tiles": "1"},
                },
                {
                    **stages[1],
                    "contract_id": "loop-carried-add-preserve",
                    "contract_version": "1",
                    "contract_parameters": common,
                },
            ],
        }],
    }
    candidate_result = ascend.analysis.ttir_ub_lower_bound_candidate_for_oracle(
        module,
        {
            "arch": "Ascend910B1",
            "compile_mode": "aiv",
            "pipeline_identity": identity,
            "pipeline_stages": stages,
            "contract_profile": profile,
        },
    )

    assert candidate_result["decision"] == "reject"
    assert candidate_result["lower_bound_bytes"] == 524288
    assert candidate_result["unsupported_reasons"] == []
    assert candidate_result["certificates"] == [{
        "kind": "witness",
        "bytes": 524288,
        "resource_ids": [0, 1],
        "contract_trace": [
            "ttir-loop-carried-add-v1",
            "loop-carried-add-max-tiles",
            "loop-carried-add-preserve",
        ],
    }]


def test_binding_direct_copy_contract_defers_on_parameter_drift(tmp_path):
    source = tmp_path / "direct-load.ttir"
    source.write_text(DIRECT_LOAD_COPY)
    context = ir.context()
    ascend.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    identity = {
        "open_source_pipeline": "p1-direct-copy-test",
        "canonical_ttir_sha256": hashlib.sha256(DIRECT_LOAD_COPY.encode()).hexdigest(),
        "relevant_options_json": "{}",
        "target_arch": "Ascend910B",
        "triton_version": "test",
        "cann_version_hash": "test",
        "sha256": "p1-direct-copy-test-identity",
    }
    stages = [{"stage_name": "materialize", "options": {}}]
    profile = {
        "schema": "ttir-ub-lb-profile-v1",
        "profiles": [{
            "pipeline_identity": identity,
            "pipeline_stages": [{
                **stages[0],
                "contract_id": "direct-copy-max-tiles",
                "contract_version": "1",
                "contract_parameters": {
                    "expected_resource_count": "1",
                    "expected_source_elements": "32768",
                    "expected_element_bit_width": "32",
                    "expected_input_payload_bytes": "131072",
                    "max_tiles": "32",
                },
            }],
        }],
    }
    result = ascend.analysis.ttir_ub_lower_bound_candidate_for_oracle(
        module,
        {
            "arch": "Ascend910B",
            "compile_mode": "aiv",
            "pipeline_identity": identity,
            "pipeline_stages": stages,
            "contract_profile": profile,
        },
    )
    assert result["decision"] == "defer"
    assert result["lower_bound_bytes"] == 0
    assert result["unsupported_reasons"] == ["direct-copy-max-tiles"]


def test_binding_rejects_stage_option_drift_from_profile(tmp_path):
    source = tmp_path / "direct-load.ttir"
    source.write_text(DIRECT_LOAD_COPY)
    context = ir.context()
    ascend.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    identity = {
        "open_source_pipeline": "p0-test-pipeline",
        "canonical_ttir_sha256": hashlib.sha256(DIRECT_LOAD_COPY.encode()).hexdigest(),
        "relevant_options_json": "{}",
        "target_arch": "Ascend910B",
        "triton_version": "test",
        "cann_version_hash": "test",
        "sha256": "p0-test-identity",
    }
    profile = {
        "schema": "ttir-ub-lb-profile-v1",
        "profiles": [{
            "pipeline_identity": identity,
            "pipeline_stages": [{
                "stage_name": "ttir.auto-blockify",
                "options": {"auto_blockify_size": "1"},
                "contract_id": "invalidate-unmodeled-stage",
                "contract_version": "1",
            }],
        }],
    }
    result = ascend.analysis.ttir_ub_lower_bound(
        module,
        {
            "arch": "Ascend910B",
            "compile_mode": "aiv",
            "pipeline_identity": identity,
            "pipeline_stages": [{
                "stage_name": "ttir.auto-blockify",
                "options": {"auto_blockify_size": "2"},
            }],
            "contract_profile": profile,
        },
    )
    assert result["decision"] == "defer"
    assert result["unsupported_reasons"] == ["unknown-pipeline-profile"]


def _fake_active_driver(arch):
    return SimpleNamespace(
        get_current_target=lambda: SimpleNamespace(arch=arch),
        get_current_device=lambda: 0,
        utils=SimpleNamespace(get_device_properties=lambda _device: {"num_aicore": 24}),
    )


@pytest.mark.parametrize(
    ("arch", "expected_ub", "expected_rf"),
    [
        ("Ascend910B1", 192, None),
        ("Ascend910_9362", 192, None),
        ("Ascend310B1", 248, None),
        ("Ascend910_9599", 256, 128),
    ],
)
def test_runtime_capacity_uses_binding_and_keeps_rf_separate(monkeypatch, arch, expected_ub, expected_rf):
    from triton.runtime.driver import driver

    monkeypatch.setattr(driver, "_active", _fake_active_driver(arch))
    monkeypatch.setattr(runtime_utils, "_cached_params", None)
    params = runtime_utils._init_npu_params()
    assert params["ub_size_in_kbytes"] == expected_ub
    assert params["rf_size_in_kbytes"] == expected_rf


def test_runtime_unknown_arch_fails_explicitly(monkeypatch):
    from triton.runtime.driver import driver

    monkeypatch.setattr(driver, "_active", _fake_active_driver("future-chip"))
    monkeypatch.setattr(runtime_utils, "_cached_params", None)
    with pytest.raises(RuntimeError, match="Unknown Ascend UB capacity for future-chip"):
        runtime_utils._init_npu_params()
