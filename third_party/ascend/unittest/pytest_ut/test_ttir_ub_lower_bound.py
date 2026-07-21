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
import json
import pickle
from types import SimpleNamespace

import pytest

from triton._C.libtriton import ascend, ir
from triton.backends.ascend.errors import UBLowerBoundOverflow
from triton.backends.ascend.ub_lower_bound import apply_ub_lower_bound_policy, load_contract_profiles
from triton.backends.ascend.runtime import utils as runtime_utils

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
