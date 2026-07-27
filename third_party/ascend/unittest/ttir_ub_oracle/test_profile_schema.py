import importlib.util
from pathlib import Path
import sys
from types import ModuleType


BACKEND = Path(__file__).parents[2] / "backend" / "ub_lower_bound.py"


def load_schema_module(monkeypatch):
    modules = {
        "triton": ModuleType("triton"),
        "triton._C": ModuleType("triton._C"),
        "triton._C.libtriton": ModuleType("triton._C.libtriton"),
        "triton.runtime": ModuleType("triton.runtime"),
        "triton.runtime.cache": ModuleType("triton.runtime.cache"),
        "triton.backends": ModuleType("triton.backends"),
        "triton.backends.ascend": ModuleType("triton.backends.ascend"),
        "triton.backends.ascend.errors": ModuleType(
            "triton.backends.ascend.errors"
        ),
    }
    modules["triton._C.libtriton"].ascend = object()
    modules["triton.runtime.cache"].get_dump_manager = lambda: None
    modules["triton.backends.ascend.errors"].UBLowerBoundOverflow = type(
        "UBLowerBoundOverflow", (RuntimeError,), {}
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "triton.backends.ascend.ub_lower_bound", BACKEND
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def direct_copy_alignment_stage():
    return {
        "stage_name": "bisheng.ub-affecting-suffix",
        "options": {},
        "contract_id": "direct-copy-max-tiles+ub-alignment",
        "contract_version": "1",
        "contract_parameters": {
            "expected_resource_count": "1",
            "expected_source_elements": "65536",
            "expected_element_bit_width": "32",
            "expected_input_payload_bytes": "262144",
            "max_tiles": "64",
            "alignment_bytes": "32",
        },
    }


def test_alignment_profile_schema_is_composite_and_fail_closed(monkeypatch):
    schema = load_schema_module(monkeypatch)
    stage = direct_copy_alignment_stage()
    assert schema._is_valid_profile_stage(stage)

    standalone = dict(stage)
    standalone["contract_id"] = "ub-alignment"
    standalone["contract_parameters"] = {
        "expected_resource_count": "1",
        "alignment_bytes": "32",
    }
    assert not schema._is_valid_profile_stage(standalone)

    malformed = direct_copy_alignment_stage()
    malformed["contract_parameters"]["alignment_bytes"] = "0"
    assert not schema._is_valid_profile_stage(malformed)

    unknown = direct_copy_alignment_stage()
    unknown["contract_id"] = "unknown-family+ub-alignment"
    assert not schema._is_valid_profile_stage(unknown)
