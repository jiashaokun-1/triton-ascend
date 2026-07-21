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

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
import threading

from triton._C.libtriton import ascend
from triton.runtime.cache import get_dump_manager

from .errors import UBLowerBoundOverflow

_PROFILE_PATH = Path(__file__).with_name("ub_contract_profiles.json")
_INT64_MAX = (1 << 63) - 1
_INVALID_RESOURCE_ID = (1 << 32) - 1
_CONTRACT_VERSION = "ttir-ub-lb-v1"
_IDENTITY_CONTRACT = {
    "auto_tile_and_bind_subblock": {
        "identity_value": "module-derived-both-outcomes",
        "profile_promotion_requires": "task7-oracle-validates-enabled-and-disabled-outcomes",
    },
}
_RESULT_KEYS = frozenset({
    "decision",
    "lower_bound_bytes",
    "capacity_bytes",
    "certificates",
    "unsupported_reasons",
    "pipeline_identity",
    "contract_version",
})
_CERTIFICATE_KEYS = frozenset({"kind", "bytes", "resource_ids"})


@dataclass
class UBFilterStats:
    """Counters for one autotune compilation round.

    ``analyzed`` counts every non-off policy attempt, including attempts that
    conservatively defer before entering the analyzer binding. With the V1
    reject/defer decision schema, ``analyzed == rejected + deferred``.
    ``passed_to_backend`` counts attempts whose mode and decision allow the
    downstream compiler to run; shadow rejects therefore count as passed.
    """

    analyzed: int = 0
    rejected: int = 0
    deferred: int = 0
    passed_to_backend: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


_active_stats = ContextVar("ascend_ub_filter_stats", default=None)


@contextmanager
def ub_filter_telemetry_session():
    """Create an isolated telemetry session for one autotune round."""
    stats = UBFilterStats()
    token = _active_stats.set(stats)
    try:
        yield stats
    finally:
        _active_stats.reset(token)


def _record_telemetry(mode, result):
    stats = _active_stats.get()
    if stats is None:
        return
    with stats.lock:
        stats.analyzed += 1
        if result["decision"] == "reject":
            stats.rejected += 1
        else:
            stats.deferred += 1
        if mode != "enforce" or result["decision"] != "reject":
            stats.passed_to_backend += 1


def load_contract_profiles():
    profile = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
    if (profile.get("schema") != "ttir-ub-lb-profile-v1" or profile.get("identity_contract") != _IDENTITY_CONTRACT
            or not isinstance(profile.get("profiles"), list)):
        raise ValueError("invalid packaged TTIR UB contract profile")
    return profile


def _pipeline_fingerprint(pipeline_identity):
    if type(pipeline_identity) is str and pipeline_identity:
        return pipeline_identity
    if type(pipeline_identity) is dict:
        fingerprint = pipeline_identity.get("sha256")
        if type(fingerprint) is str and fingerprint:
            return fingerprint
    return ""


def _analysis_compile_mode(compile_mode):
    """Map backend mode names to the core-kind vocabulary used by MURG."""
    return "aiv" if compile_mode == "simd" else compile_mode


def _defer_result(pipeline_fingerprint, reason):
    return {
        "decision": "defer",
        "lower_bound_bytes": 0,
        "capacity_bytes": None,
        "certificates": [],
        "unsupported_reasons": [reason],
        "pipeline_identity": pipeline_fingerprint,
        "contract_version": _CONTRACT_VERSION,
    }


def _is_int64(value):
    return type(value) is int and 0 <= value <= _INT64_MAX


def _normalize_certificate(certificate, lower_bound_bytes):
    if type(certificate) is not dict or set(certificate) != _CERTIFICATE_KEYS:
        return None
    if type(certificate.get("kind")) is not str or certificate["kind"] != "singleton":
        return None
    if not _is_int64(certificate.get("bytes")) or certificate["bytes"] != lower_bound_bytes:
        return None
    resource_ids = certificate.get("resource_ids")
    if type(resource_ids) is not list or len(resource_ids) != 1:
        return None
    resource_id = resource_ids[0]
    if type(resource_id) is not int or not 0 <= resource_id < _INVALID_RESOURCE_ID:
        return None
    return {
        "kind": certificate["kind"],
        "bytes": certificate["bytes"],
        "resource_ids": list(resource_ids),
    }


def _normalize_result(result, pipeline_fingerprint, trusted_capacity):
    invalid = lambda: _defer_result(pipeline_fingerprint, "invalid-analysis-result")
    if type(result) is not dict or set(result) != _RESULT_KEYS:
        return invalid()

    decision = result["decision"]
    lower_bound_bytes = result["lower_bound_bytes"]
    capacity_bytes = result["capacity_bytes"]
    certificates = result["certificates"]
    unsupported_reasons = result["unsupported_reasons"]
    contract_version = result["contract_version"]
    result_fingerprint = result["pipeline_identity"]
    if type(decision) is not str or decision not in ("defer", "reject"):
        return invalid()
    if not _is_int64(lower_bound_bytes):
        return invalid()
    if capacity_bytes is None:
        if decision != "defer":
            return invalid()
    elif not _is_int64(capacity_bytes):
        return invalid()
    elif capacity_bytes != trusted_capacity:
        return invalid()
    if type(certificates) is not list:
        return invalid()
    normalized_certificates = []
    for certificate in certificates:
        normalized_certificate = _normalize_certificate(certificate, lower_bound_bytes)
        if normalized_certificate is None:
            return invalid()
        normalized_certificates.append(normalized_certificate)
    if type(unsupported_reasons) is not list or not all(type(reason) is str for reason in unsupported_reasons):
        return invalid()
    if type(contract_version) is not str or contract_version != _CONTRACT_VERSION:
        return invalid()
    if type(result_fingerprint) is not str or result_fingerprint != pipeline_fingerprint:
        return invalid()
    if decision == "reject" and (lower_bound_bytes <= capacity_bytes or len(normalized_certificates) != 1
                                 or unsupported_reasons):
        return invalid()

    return {
        "decision": decision,
        "lower_bound_bytes": lower_bound_bytes,
        "capacity_bytes": capacity_bytes,
        "certificates": normalized_certificates,
        "unsupported_reasons": list(unsupported_reasons),
        "pipeline_identity": result_fingerprint,
        "contract_version": contract_version,
    }


def _exception_reason(error):
    try:
        message = str(error)
    except Exception:
        message = type(error).__name__
    return f"internal-error: {message}"


def _record_metadata(metadata, mode, result):
    metadata.update({
        "ub_lower_bound_mode": mode,
        "ub_lower_bound_decision": result["decision"],
        "ub_lower_bound_bytes": result["lower_bound_bytes"],
        "ub_capacity_bytes": result["capacity_bytes"],
        "ub_lower_bound_contract_version": result["contract_version"],
        "ub_lower_bound_pipeline_identity": result["pipeline_identity"],
        "ub_lower_bound_certificate_count": len(result["certificates"]),
        "ub_lower_bound_unsupported_reasons": result["unsupported_reasons"],
    })


def apply_ub_lower_bound_policy(mod, metadata, opt, pipeline_identity):
    mode = getattr(opt, "ub_lower_bound_mode", "off")
    if mode == "off":
        return

    pipeline_fingerprint = _pipeline_fingerprint(pipeline_identity)
    if not pipeline_fingerprint:
        result = _defer_result("", "invalid-analysis-result")
    else:
        try:
            trusted_capacity = ascend.analysis.get_ub_capacity_bytes(opt.arch)
        except Exception:
            trusted_capacity = None
        if not _is_int64(trusted_capacity):
            result = _defer_result(pipeline_fingerprint, "invalid-analysis-result")
        else:
            try:
                raw_result = ascend.analysis.ttir_ub_lower_bound(
                    mod,
                    {
                        "arch": opt.arch,
                        "compile_mode": _analysis_compile_mode(opt.compile_mode),
                        "pipeline_identity": pipeline_identity,
                        "pipeline_stages": [],
                        "contract_profile": load_contract_profiles(),
                    },
                )
            except Exception as error:
                result = _defer_result(pipeline_fingerprint, _exception_reason(error))
            else:
                try:
                    result = _normalize_result(raw_result, pipeline_fingerprint, trusted_capacity)
                except Exception:
                    result = _defer_result(pipeline_fingerprint, "invalid-analysis-result")

    _record_telemetry(mode, result)
    _record_metadata(metadata, mode, result)
    if getattr(opt, "debug", False):
        dump_manager = get_dump_manager(metadata["hash"])
        dump_manager.put(
            json.dumps(result, sort_keys=True, separators=(",", ":")),
            "kernel.ttir.ub-lower-bound.json",
            binary=False,
        )
    if mode == "enforce" and result["decision"] == "reject":
        raise UBLowerBoundOverflow(
            result["lower_bound_bytes"],
            result["capacity_bytes"],
            result["certificates"][0],
            result["pipeline_identity"],
        )
