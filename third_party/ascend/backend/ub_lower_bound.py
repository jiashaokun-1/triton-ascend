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
from pathlib import Path

from triton._C.libtriton import ascend

from .errors import UBLowerBoundOverflow

_PROFILE_PATH = Path(__file__).with_name("ub_contract_profiles.json")


def load_contract_profiles():
    profile = json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
    if profile.get("schema") != "ttir-ub-lb-profile-v1" or not isinstance(profile.get("profiles"), list):
        raise ValueError("invalid packaged TTIR UB contract profile")
    return profile


def _defer_result(pipeline_identity, reason):
    return {
        "decision": "defer",
        "lower_bound_bytes": 0,
        "capacity_bytes": None,
        "certificates": [],
        "unsupported_reasons": [reason],
        "pipeline_identity": pipeline_identity,
        "contract_version": "ttir-ub-lb-v1",
    }


def _normalize_result(result, pipeline_identity):
    if not isinstance(result, dict):
        return _defer_result(pipeline_identity, "invalid-analysis-result")

    normalized = {
        "decision": "defer",
        "lower_bound_bytes": result.get("lower_bound_bytes", 0),
        "capacity_bytes": result.get("capacity_bytes"),
        "certificates": result.get("certificates", []),
        "unsupported_reasons": result.get("unsupported_reasons", []),
        "pipeline_identity": result.get("pipeline_identity", pipeline_identity),
        "contract_version": result.get("contract_version", "ttir-ub-lb-v1"),
    }
    if not isinstance(normalized["lower_bound_bytes"], int) or isinstance(normalized["lower_bound_bytes"], bool):
        return _defer_result(pipeline_identity, "invalid-analysis-result")
    if normalized["capacity_bytes"] is not None and (not isinstance(normalized["capacity_bytes"], int)
                                                     or isinstance(normalized["capacity_bytes"], bool)):
        return _defer_result(pipeline_identity, "invalid-analysis-result")
    if not isinstance(normalized["certificates"], list) or not isinstance(normalized["unsupported_reasons"], list):
        return _defer_result(pipeline_identity, "invalid-analysis-result")

    proven_reject = (result.get("decision") == "reject" and normalized["capacity_bytes"] is not None
                     and normalized["lower_bound_bytes"] > normalized["capacity_bytes"]
                     and bool(normalized["certificates"]) and not normalized["unsupported_reasons"])
    if proven_reject:
        normalized["decision"] = "reject"
    return normalized


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

    try:
        result = ascend.analysis.ttir_ub_lower_bound(
            mod,
            {
                "arch": opt.arch,
                "compile_mode": opt.compile_mode,
                "pipeline_identity": pipeline_identity,
                "pipeline_stages": [],
                "contract_profile": load_contract_profiles(),
            },
        )
        result = _normalize_result(result, pipeline_identity)
    except Exception as error:
        result = _defer_result(pipeline_identity, f"internal-error: {error}")

    _record_metadata(metadata, mode, result)
    if mode == "enforce" and result["decision"] == "reject":
        raise UBLowerBoundOverflow(
            result["lower_bound_bytes"],
            result["capacity_bytes"],
            result["certificates"][0],
            result["pipeline_identity"],
        )
