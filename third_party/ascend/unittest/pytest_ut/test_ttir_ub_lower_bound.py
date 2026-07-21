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
import pickle
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
        "certificates": [{"kind": "singleton", "bytes": 262144, "resource_ids": [0]}],
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

EXPECTED_UB_AFFECTING_BISHENG_FLAGS = {
    "--append-bisheng-options",
    "--disable-auto-inject-block-sync",
    "--disable-ffts",
    "--disable-fma",
    "--disable-hfusion-vectorize",
    "--disable-size-align-for-cast",
    "--disable-tightly-coupled-buffer-reuse",
    "--enable-auto-bind-sub-block",
    "--enable-auto-blockify-loop",
    "--enable-auto-multi-buffer",
    "--enable-auto-vectorize-v2",
    "--enable-bishengir-simt-optimization",
    "--enable-drop-unit-dims",
    "--enable-flatten",
    "--enable-hivm-auto-cv-balance",
    "--enable-hivm-graph-sync-solver",
    "--enable-hivm-inject-barrier-all-sync",
    "--enable-hivm-inject-block-all-sync",
    "--enable-hivm-unit-flag-sync",
    "--enable-hfusion-compile",
    "--enable-mixed-cv",
    "--enable-preload",
    "--enable-simd-simt-mix-compile",
    "--enable-simt-reorder-instruction",
    "--enable-ubuf-saving",
    "--enable-vf-fusion",
    "--enable-vf-merge-level",
    "--hfusion-enable-multiple-consumer-fusion",
    "--hfusion-max-fused-elementwise-ops",
    "--hfusion-max-fused-ops-in-auto-vectorize-v2",
    "--limit-auto-multi-buffer-of-local-buffer",
    "--limit-auto-multi-buffer-only-for-local-buffer",
    "--num-warps",
    "--pure-simt",
    "--reg-based",
    "--set-workspace-multibuffer",
    "--shared-mem-dynamic-size",
    "--simt-stack-limit",
    "--threads-per-warp",
    "--tile-mix-cube-loop",
    "--tile-mix-vector-loop",
}


def _identity_runtime(monkeypatch):
    monkeypatch.setattr(ascend_compiler, "get_cann_version_file_hash", lambda: "test-cann-hash")
    monkeypatch.setattr(ascend_compiler, "_is_auto_map_parallel_blocks_enabled", lambda: False)
    monkeypatch.setattr(ascend_compiler, "force_disable_ffts", lambda: False)
    monkeypatch.setattr(ascend_compiler, "_check_bishengir_is_regbased", lambda: False)
    monkeypatch.setattr(ascend_compiler, "_get_npucompiler_path", lambda: ("/test/bishengir-compile", {}))
    monkeypatch.delenv("TRITON_ENABLE_VF_FUSION", raising=False)


def _ub_affecting_bisheng_flag_literals():
    source = inspect.getsource(ascend_compiler).lstrip("\ufeff")
    tree = ast.parse(source)
    function_names = {
        "linalg_to_bin_enable_npu_compile_910_95",
        "linalg_to_bin_enable_npu_compile_A2_A3",
        "ttir_to_npubin",
    }
    keywords = (
        "blockify",
        "buffer",
        "bind-sub-block",
        "cv",
        "flatten",
        "fused",
        "fusion",
        "preload",
        "simt",
        "size-align",
        "sync",
        "tile",
        "ubuf",
        "unit-dims",
        "vectorize",
        "vf-",
        "workspace",
    )
    always_relevant = {
        "--append-bisheng-options",
        "--disable-ffts",
        "--disable-fma",
        "--num-warps",
        "--reg-based",
        "--shared-mem-dynamic-size",
        "--threads-per-warp",
    }
    flags = set()
    for function in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
        if function.name not in function_names:
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if not node.value.startswith("--"):
                continue
            flag = node.value.split("=", 1)[0]
            if flag in always_relevant or any(keyword in flag for keyword in keywords):
                flags.add(flag)
    return flags


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
    pm = ascend_compiler._build_ttir_to_linalg_pass_manager(module, metadata, options, named_ops=False)
    assert pm.get_pipeline_str() == EXPECTED_TTIR_TO_LINALG_PIPELINE


def test_pipeline_identity_is_canonical_json_and_contains_complete_payload(monkeypatch):
    _identity_runtime(monkeypatch)
    metadata = _compiler_metadata(multibuffer=False)
    identity = ascend_compiler._ttir_ub_pipeline_identity("pass-a,pass-b", metadata)
    options = json.loads(identity["relevant_options_json"])
    assert identity.keys() == {
        "open_source_pipeline",
        "relevant_options_json",
        "target_arch",
        "triton_version",
        "cann_version_hash",
        "sha256",
    }
    assert identity["open_source_pipeline"] == "pass-a,pass-b"
    assert identity["target_arch"] == "Ascend910B"
    assert identity["triton_version"] == "test-triton-version"
    assert identity["cann_version_hash"] == "test-cann-hash"
    assert options["multibuffer"] is False
    assert identity["relevant_options_json"] == json.dumps(options, sort_keys=True, separators=(",", ":"))
    payload = {
        "cann_version_hash": "test-cann-hash",
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


def test_pipeline_identity_changes_when_bisheng_compiler_kind_changes(monkeypatch):
    _identity_runtime(monkeypatch)
    metadata = _compiler_metadata()
    first = ascend_compiler._ttir_ub_pipeline_identity("pipeline", metadata)
    monkeypatch.setattr(ascend_compiler, "_get_npucompiler_path", lambda: ("/test/npu-compiler", {}))
    second = ascend_compiler._ttir_ub_pipeline_identity("pipeline", metadata)
    assert first["sha256"] != second["sha256"]


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


@pytest.mark.parametrize("option_name", EXPECTED_UB_AFFECTING_OPTIONS)
def test_pipeline_identity_changes_for_every_ub_affecting_option(monkeypatch, option_name):
    _identity_runtime(monkeypatch)
    metadata = _compiler_metadata()
    before = ascend_compiler._ttir_ub_pipeline_identity("pipeline", metadata)["sha256"]
    current = metadata.get(option_name)
    metadata[option_name] = "identity-test-value" if current is None else None
    after = ascend_compiler._ttir_ub_pipeline_identity("pipeline", metadata)["sha256"]
    assert before != after, option_name


def test_ub_affecting_options_and_forwarded_bisheng_flags_are_closed_goldens():
    assert ascend_compiler.UB_AFFECTING_OPTIONS == EXPECTED_UB_AFFECTING_OPTIONS
    assert _ub_affecting_bisheng_flag_literals() == EXPECTED_UB_AFFECTING_BISHENG_FLAGS


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

    def policy(_mod, _metadata, _options, identity):
        calls.append("policy")
        assert identity["open_source_pipeline"] == "pipeline"

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
        lambda _mod, _metadata, _options, identity: captured.append(identity),
    )
    options = ascend_compiler.NPUOptions(ub_lower_bound_mode="shadow", compile_on_910_95=False)
    module = SimpleNamespace(context=object(), __str__=lambda: "module")
    ascend_compiler.make_ttir(module, _compiler_metadata(), options)
    assert captured == [""]


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
    monkeypatch.setattr(
        ascend_compiler,
        "_ttir_ub_pipeline_identity",
        lambda pipeline, _metadata: captured.append(pipeline) or {"sha256": "direct-id"},
    )
    monkeypatch.setattr(ascend_compiler, "apply_ub_lower_bound_policy", lambda *_args: None)
    options = ascend_compiler.NPUOptions(
        ub_lower_bound_mode="shadow",
        compile_on_910_95=True,
        compile_mode="simt_only",
    )
    module = SimpleNamespace(context=object(), __str__=lambda: "module")
    ascend_compiler.make_ttir(module, _compiler_metadata(compile_on_910_95=True), options)
    assert captured == [ascend_compiler.TTIR_UB_DIRECT_BISHENG_PIPELINE]


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


def test_policy_passes_only_packaged_empty_profile(monkeypatch):
    captured = {}

    def analyze(_mod, options):
        captured.update(options)
        return _analysis_result("defer")

    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", analyze)
    apply_ub_lower_bound_policy(object(), {}, Options("shadow"), "test-id")
    assert captured["contract_profile"] == {
        "schema": "ttir-ub-lb-profile-v1",
        "profiles": [],
    }
    assert "allow_unvalidated" not in captured


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
        "profiles": [],
    }


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
