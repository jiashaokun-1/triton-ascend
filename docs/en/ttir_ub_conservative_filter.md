# TTIR UB conservative filter

## Purpose

The TTIR UB conservative filter removes an autotune configuration only when a
checked proof shows that its mandatory Unified Buffer (UB) requirement is
greater than the target capacity. It runs after canonical TTIR is produced and
before the expensive TTIR-to-HIVM compilation path.

The filter is one-sided:

- `reject` proves that the configuration exceeds UB capacity;
- `defer` means that the current rules cannot decide;
- `defer` does not mean that the configuration fits in UB.

The default mode is `off`. The filter never changes TTIR.

## Activation and rollback

Use a compiler option on each autotune configuration:

```python
triton.Config({"BLOCK_SIZE": 65536, "ub_lower_bound_mode": "shadow"})
triton.Config({"BLOCK_SIZE": 65536, "ub_lower_bound_mode": "enforce"})
```

The modes are:

| Mode | Analysis | Compilation behavior |
| --- | --- | --- |
| `off` | Not called | Existing behavior |
| `shadow` | Called | Records the decision and always continues |
| `enforce` | Called | Stops only configurations with a `reject` proof |

Rollback is immediate: set `ub_lower_bound_mode="off"` or remove the option.

## Architecture

The implementation has four layers:

1. The strict TTIR matcher recognizes supported static direct loads.
2. The Mandatory UB Resource Graph (MURG) records mandatory resources,
   identity/alias relations, coexistence relations, minimum payload, and the
   proof trace.
3. A `UBResourceContract` describes how one exact pipeline stage preserves,
   transforms, or invalidates those resources.
4. The policy compares a certified lower bound with the target capacity. Only
   `lower_bound_bytes > capacity_bytes` becomes `reject`; equality continues.

Every proof is bound to an exact pipeline identity: ordered pass pipeline,
UB-relevant options, target, Triton version, CANN identity, selected compiler
kind, and compiler content digest. Unknown operations, targets, identities, or
contracts produce a structured `defer` reason.

Packaged production profiles start empty. A profile is added only after the
real compiler comparison described below succeeds for every required seed and
retry mode.

## Metadata and diagnostics

`shadow` and `enforce` add compact JSON-serializable fields to compiler
metadata, including the mode, decision, lower bound, capacity, pipeline
fingerprint, certificate count, and structured reasons.

With compiler debug output enabled, the complete certificate is written to:

```text
kernel.ttir.ub-lower-bound.json
```

Normal compilation does not emit the full graph or certificate.

## Extending the model

To support another TTIR pattern or lowering stage:

1. Add resources and relations to MURG without relying on temporary SSA names.
2. Add a versioned `UBResourceContract` for every affected pipeline stage.
3. Return `defer` whenever applicability or the transfer rule is incomplete.
4. Add C++ matcher/graph tests and Python policy tests.
5. Add a paired canonical TTIR and before-CVPipelining fixture.
6. Compare the lower bound with real PlanMemory results for seeds `0..19` and
   retry mode before reviewing a profile candidate.

First capture `kernel.ttir.mlir` and the BiSheng Generic IR immediately before
`createCVPipeliningPass` from the same compilation. Place the latter at
`before_cvpipelining.mlir` in one dedicated capture directory, then package the
pair:

```bash
python third_party/ascend/tools/ttir_ub_fixture_bundle.py \
  --dump-dir /path/to/paired-capture/ONE_CASE \
  --output-dir /tmp/binary-add-fixture \
  --name binary-add-f32-65536-a2 \
  --operation-family binary-add \
  --arch Ascend910B \
  --source-elements 65536 \
  --element-bit-width 32 \
  --max-tiles 64
```

The packager rejects a raw `kernel.ttadapter.mlir`: the exact number and size of
static allocations must match the proposed before-CVPipelining boundary. It
also derives the input payload, emits both file hashes, and refuses to overwrite
an existing fixture. By default, the oracle derives the auto-tile outcome from
every real seed/retry snapshot and rejects nondeterministic outcomes. Use an
explicit `--auto-tile-outcome true|false` only as an additional assertion for a
known golden fixture.

Run the comparison with:

```bash
python third_party/ascend/tools/ttir_ub_oracle.py \
  --manifest third_party/ascend/unittest/ttir_ub_oracle/fixtures/manifest.json \
  --suffix-compiler /path/to/bishengir-cvpipeline-suffix-compile \
  --semantic-model /path/to/cvpipeline_ub_model \
  --seeds 0-19 --check-retry \
  --report /tmp/ttir-ub-oracle-report.json \
  --profile-candidate /tmp/ttir-ub-profile-candidate.json
```

The tool never edits `ub_contract_profiles.json`. A candidate is produced only
from a complete run with no comparison violations and a non-empty certificate;
promotion remains an explicit review step.
