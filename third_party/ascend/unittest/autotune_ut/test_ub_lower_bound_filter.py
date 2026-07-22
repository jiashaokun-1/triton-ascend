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

from concurrent.futures import ThreadPoolExecutor
from types import MethodType, SimpleNamespace

import pytest

import triton
from triton.backends.ascend.errors import UBLowerBoundOverflow
from triton.backends.ascend.runtime.autotuner import AutoTilingTuner
from triton.backends.ascend import ub_lower_bound
from triton.runtime._async_compile import active_mode
from triton.runtime.autotuner import Config
from triton.runtime.errors import OutOfResources


def _resource_error(error_type):
    if error_type is UBLowerBoundOverflow:
        return error_type(2, 1, {"kind": "singleton"}, "pipeline")
    return error_type(2, 1, "test resource")


def _make_filter_tuner(compile_parallel, failures):
    tuner = object.__new__(AutoTilingTuner)
    tuner.compile_parallel = compile_parallel
    tuner.do_bench = lambda fn, quantiles: (fn(), quantiles)
    tuner.user_defined_do_bench = True

    def _make_kernel_call(self, *args, config, **meta):
        compiled = False

        def compile_fn():
            failure = failures.get(config.kwargs["ID"])
            if callable(failure):
                failure()
            elif failure is not None:
                raise failure
            return config.kwargs["ID"]

        def finalize_fn(_kernel):
            nonlocal compiled
            compiled = True

        def kernel_call(warmup):
            nonlocal compiled
            if compiled:
                return config.kwargs["ID"]
            mode = active_mode.get()
            if warmup and mode is not None:
                return mode.submit(config.kwargs["ID"], compile_fn, finalize_fn)
            kernel = compile_fn()
            finalize_fn(kernel)
            return kernel

        return kernel_call

    tuner._make_kernel_call = MethodType(_make_kernel_call, tuner)
    return tuner


@pytest.mark.parametrize("compile_parallel", [False, True])
@pytest.mark.parametrize("error_type", [OutOfResources, UBLowerBoundOverflow])
def test_batch_bench_discards_only_resource_failure_and_keeps_compilable_config(compile_parallel, error_type):
    rejected = Config({"ID": "rejected"})
    accepted = Config({"ID": "accepted"})
    tuner = _make_filter_tuner(compile_parallel, {"rejected": _resource_error(error_type)})

    timings = tuner._batch_bench(configs=[rejected, accepted])

    assert list(timings) == [accepted]


def test_async_compile_exit_does_not_reraise_observed_exception():

    def fail_compile():
        raise OutOfResources(2, 1, "test resource")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with triton.AsyncCompileMode(pool) as mode:
            future = mode.submit("rejected", fail_compile, lambda kernel: None)
            with pytest.raises(OutOfResources):
                future.result()
    assert active_mode.get() is None


def test_async_compile_exit_still_raises_unobserved_exception():

    def fail_compile():
        raise RuntimeError("unobserved compile failure")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="unobserved compile failure"):
            with triton.AsyncCompileMode(pool) as mode:
                mode.submit("failed", fail_compile, lambda kernel: None)
    assert active_mode.get() is None


def _options(mode="enforce"):
    return SimpleNamespace(
        arch="Ascend910B1",
        compile_mode="aot",
        debug=False,
        ub_lower_bound_mode=mode,
    )


def _analysis_result(decision, pipeline_identity):
    if decision == "reject":
        return {
            "decision": "reject",
            "lower_bound_bytes": 196609,
            "capacity_bytes": 196608,
            "certificates": [{
                "kind": "singleton",
                "bytes": 196609,
                "resource_ids": [7],
                "contract_trace": ["synthetic-autotune-contract"],
            }],
            "unsupported_reasons": [],
            "pipeline_identity": pipeline_identity,
            "contract_version": "ttir-ub-lb-v1",
        }
    return {
        "decision": "defer",
        "lower_bound_bytes": 0,
        "capacity_bytes": 196608,
        "certificates": [],
        "unsupported_reasons": ["unsupported-test-op"],
        "pipeline_identity": pipeline_identity,
        "contract_version": "ttir-ub-lb-v1",
    }


def _install_analysis(monkeypatch, decisions):
    monkeypatch.setattr(ub_lower_bound.ascend.analysis, "get_ub_capacity_bytes", lambda arch: 196608)

    def analyze(_mod, options):
        identity = options["pipeline_identity"]
        return _analysis_result(decisions[identity], identity)

    monkeypatch.setattr(ub_lower_bound.ascend.analysis, "ttir_ub_lower_bound", analyze)


def _snapshot(stats):
    with stats.lock:
        return (
            stats.analyzed,
            stats.rejected,
            stats.deferred,
            stats.passed_to_backend,
        )


def _apply_policy(identity, mode="enforce"):
    return ub_lower_bound.apply_ub_lower_bound_policy(object(), {}, _options(mode), identity)


def test_policy_telemetry_counts_each_non_off_config_once(monkeypatch):
    _install_analysis(monkeypatch, {"reject": "reject", "defer": "defer"})

    with ub_lower_bound.ub_filter_telemetry_session() as stats:
        with pytest.raises(UBLowerBoundOverflow):
            _apply_policy("reject")
        _apply_policy("defer")

    assert _snapshot(stats) == (2, 1, 1, 1)
    assert stats.analyzed == stats.rejected + stats.deferred


def test_policy_telemetry_off_does_not_analyze(monkeypatch):
    calls = 0

    def unexpected_analysis(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("off mode must not analyze")

    monkeypatch.setattr(ub_lower_bound.ascend.analysis, "ttir_ub_lower_bound", unexpected_analysis)

    with ub_lower_bound.ub_filter_telemetry_session() as stats:
        _apply_policy("off", mode="off")

    assert calls == 0
    assert _snapshot(stats) == (0, 0, 0, 0)


def test_policy_telemetry_preanalysis_defer_counts_as_analyzed(monkeypatch):

    def unexpected_analysis(*args, **kwargs):
        raise AssertionError("invalid identity must defer before the binding")

    monkeypatch.setattr(ub_lower_bound.ascend.analysis, "ttir_ub_lower_bound", unexpected_analysis)

    with ub_lower_bound.ub_filter_telemetry_session() as stats:
        _apply_policy("")

    assert _snapshot(stats) == (1, 0, 1, 1)


def test_policy_telemetry_shadow_reject_still_passes_to_backend(monkeypatch):
    _install_analysis(monkeypatch, {"reject": "reject"})

    with ub_lower_bound.ub_filter_telemetry_session() as stats:
        _apply_policy("reject", mode="shadow")

    assert _snapshot(stats) == (1, 1, 0, 1)


def test_policy_telemetry_has_no_effect_without_active_session(monkeypatch):
    _install_analysis(monkeypatch, {"defer": "defer"})

    with ub_lower_bound.ub_filter_telemetry_session() as completed_stats:
        _apply_policy("defer")
    before = _snapshot(completed_stats)

    _apply_policy("defer")

    assert _snapshot(completed_stats) == before


def test_policy_telemetry_nested_sessions_are_isolated(monkeypatch):
    _install_analysis(monkeypatch, {"defer": "defer"})

    with ub_lower_bound.ub_filter_telemetry_session() as outer:
        _apply_policy("defer")
        with ub_lower_bound.ub_filter_telemetry_session() as inner:
            _apply_policy("defer")
        _apply_policy("defer")

    assert outer is not inner
    assert _snapshot(outer) == (2, 0, 2, 2)
    assert _snapshot(inner) == (1, 0, 1, 1)


def test_policy_telemetry_concurrent_sessions_are_isolated(monkeypatch):
    _install_analysis(monkeypatch, {"defer": "defer"})

    def run_session(count):
        with ub_lower_bound.ub_filter_telemetry_session() as stats:
            for _ in range(count):
                _apply_policy("defer")
        return _snapshot(stats)

    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(run_session, 1)
        three = pool.submit(run_session, 3)

    assert one.result() == (1, 0, 1, 1)
    assert three.result() == (3, 0, 3, 3)


@pytest.mark.parametrize("compile_parallel", [False, True])
def test_batch_bench_prints_one_round_summary_and_propagates_telemetry(monkeypatch, capsys, compile_parallel):
    _install_analysis(monkeypatch, {"reject": "reject", "defer": "defer"})
    monkeypatch.setenv("TRITON_PRINT_AUTOTUNING", "1")
    rejected = Config({"ID": "rejected"})
    accepted = Config({"ID": "accepted"})
    tuner = _make_filter_tuner(
        compile_parallel,
        {
            "rejected": lambda: _apply_policy("reject"),
            "accepted": lambda: _apply_policy("defer"),
        },
    )

    timings = tuner._batch_bench(configs=[rejected, accepted])

    assert list(timings) == [accepted]
    summary = "Triton autotuning UB filter: analyzed=2, rejected=1, deferred=1, passed_to_backend=1"
    assert capsys.readouterr().out.splitlines().count(summary) == 1


def test_batch_bench_does_not_print_summary_when_disabled(monkeypatch, capsys):
    _install_analysis(monkeypatch, {"defer": "defer"})
    monkeypatch.setenv("TRITON_PRINT_AUTOTUNING", "0")
    accepted = Config({"ID": "accepted"})
    tuner = _make_filter_tuner(False, {"accepted": lambda: _apply_policy("defer")})

    tuner._batch_bench(configs=[accepted])

    assert "Triton autotuning UB filter:" not in capsys.readouterr().out


def test_parallel_batch_telemetry_counts_all_workers_under_lock(monkeypatch, capsys):
    _install_analysis(monkeypatch, {"defer": "defer"})
    monkeypatch.setenv("TRITON_PRINT_AUTOTUNING", "1")
    configs = [Config({"ID": f"accepted-{index}"}) for index in range(32)]
    actions = {config.kwargs["ID"]: lambda: _apply_policy("defer") for config in configs}
    tuner = _make_filter_tuner(True, actions)

    timings = tuner._batch_bench(configs=configs)

    assert list(timings) == configs
    summary = "Triton autotuning UB filter: analyzed=32, rejected=0, deferred=32, passed_to_backend=32"
    assert capsys.readouterr().out.splitlines().count(summary) == 1
