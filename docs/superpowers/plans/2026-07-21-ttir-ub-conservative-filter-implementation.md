# TTIR UB Conservative Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a zero-false-reject TTIR UB lower-bound analyzer that filters an autotune config only when a versioned proof certificate shows its mandatory UB use exceeds the target capacity.

**Architecture:** Canonicalized per-config TTIR is converted into a `MandatoryUBResourceGraph` (MURG). A versioned `UBResourceContract` chain validates or invalidates every resource across the real lowering pipeline; Python applies `off/shadow/enforce` policy and the Ascend autotuner discards only `UBLowerBoundOverflow` configs.

**Tech Stack:** C++17, LLVM/MLIR, Triton TTIR, pybind11, Python 3, pytest, GoogleTest, CMake/Ninja, BiShengIR suffix compiler and PlanMemory oracle.

## Global Constraints

- Correctness invariant: `lower_bound_bytes <= actual PlanMemory UB peak bytes` for every supported config.
- Hard rejection condition: `decision == "reject"` and `lower_bound_bytes > capacity_bytes`.
- Unknown target, unknown pass, unmatched pipeline identity, unsupported TTIR or analyzer error must return `defer`.
- The production contract profile list starts empty; an identity enters it only after the real oracle reports zero violations for seeds `0..19` and retry mode.
- `off` is the default and must preserve current compilation behavior.
- V1 enables singleton resources only; MURG coexistence data structures are tested but multi-resource summation remains disabled in production.
- V1 counts raw payload bytes with `minInstances=1`; alignment and multi-buffer increases are not added.
- The probabilistic AST model must never feed the hard-rejection decision.
- Do not modify or commit unrelated files. Every commit uses DCO: `git commit -s`.
- Before compiling, initialize this worktree's submodules with `git submodule update --init --recursive` and create a build environment in which `python -c 'import triton'` succeeds.
- Configure unit-test builds at `.build-ttir-ub` with `TRITON_BUILD_UT=ON`; all commands below run from the worktree root.

---

## File Map

New C++ analysis files:

```text
third_party/ascend/include/Analysis/TTIRUBLowerBound/
  MandatoryUBResourceGraph.h       graph data and lower-bound solver
  UBResourceContract.h             stage contract interface and pipeline identity
  TTIRUBLowerBound.h               public analyzer/result API

third_party/ascend/lib/Analysis/TTIRUBLowerBound/
  MandatoryUBResourceGraph.cpp
  UBResourceContract.cpp
  DirectTensorLoadMaterialization.cpp
  TTIRUBLowerBound.cpp
  CMakeLists.txt
```

New Python and binding files:

```text
third_party/ascend/ttir_ub_lower_bound_bindings.cc
third_party/ascend/backend/ub_lower_bound.py
third_party/ascend/backend/errors.py
third_party/ascend/backend/ub_contract_profiles.json
```

New tests and tools:

```text
third_party/ascend/unittest/TTIRUBLowerBoundTest.cpp
third_party/ascend/unittest/pytest_ut/test_ttir_ub_lower_bound.py
third_party/ascend/unittest/autotune_ut/test_ub_lower_bound_filter.py
python/test/unit/runtime/test_async_compile_context.py
third_party/ascend/tools/ttir_ub_oracle.py
third_party/ascend/unittest/ttir_ub_oracle/test_oracle.py
third_party/ascend/unittest/ttir_ub_oracle/fixtures/manifest.json
```

Existing integration files:

```text
third_party/ascend/include/CMakeLists.txt
third_party/ascend/lib/CMakeLists.txt
third_party/ascend/CMakeLists.txt
third_party/ascend/triton_ascend.cc
third_party/ascend/backend/compiler.py
third_party/ascend/backend/runtime/utils.py
third_party/ascend/backend/runtime/autotuner.py
python/triton/compiler/compiler.py
python/triton/runtime/_async_compile.py
third_party/ascend/unittest/CMakeLists.txt
```

### Task 1: Mandatory UB Resource Graph and safe solver

**Files:**
- Create: `third_party/ascend/include/Analysis/TTIRUBLowerBound/MandatoryUBResourceGraph.h`
- Create: `third_party/ascend/lib/Analysis/TTIRUBLowerBound/MandatoryUBResourceGraph.cpp`
- Create: `third_party/ascend/lib/Analysis/TTIRUBLowerBound/CMakeLists.txt`
- Modify: `third_party/ascend/lib/CMakeLists.txt`
- Create: `third_party/ascend/unittest/TTIRUBLowerBoundTest.cpp`
- Modify: `third_party/ascend/unittest/CMakeLists.txt`

**Interfaces:**
- Produces: `MandatoryUBResourceGraph::addResource`, `addMayAlias`, `addMustAlias`, `addMustDistinct`, `addWitness`, `invalidate`, and `solveSingletonLowerBound`.
- Produces: `MandatoryUBResource`, `CoexistenceWitness`, `LowerBoundCertificate`, `ValidityState`, and stable numeric IDs.

- [ ] **Step 1: Write graph tests before the implementation**

Add tests with these exact cases:

```cpp
TEST(MandatoryUBResourceGraph, SingletonUsesLargestMandatoryResource) {
  MandatoryUBResourceGraph graph;
  graph.addResource({"load0", 64 * 1024, 1});
  graph.addResource({"load1", 192 * 1024 + 4, 1});
  auto result = graph.solveSingletonLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->bytes, 192 * 1024 + 4);
  EXPECT_EQ(result->resourceIds.size(), 1u);
}

TEST(MandatoryUBResourceGraph, InvalidResourceCannotContribute) {
  MandatoryUBResourceGraph graph;
  auto id = graph.addResource({"load0", 256 * 1024, 1});
  graph.invalidate(id, "unknown-stage");
  auto result = graph.solveSingletonLowerBound();
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->bytes, 0);
}

TEST(MandatoryUBResourceGraph, ArithmeticOverflowFailsClosed) {
  MandatoryUBResourceGraph graph;
  graph.addResource({"load0", INT64_MAX, 2});
  EXPECT_TRUE(failed(graph.solveSingletonLowerBound()));
}

TEST(MandatoryUBResourceGraph, PairwiseOverlapIsNotAThreeWayWitness) {
  MandatoryUBResourceGraph graph;
  auto a = graph.addResource({"a", 32, 1});
  auto b = graph.addResource({"b", 64, 1});
  auto c = graph.addResource({"c", 128, 1});
  graph.addMustDistinct(a, b);
  graph.addMustDistinct(b, c);
  graph.addMustDistinct(a, c);
  graph.addWitness({a, b});
  graph.addWitness({b, c});
  EXPECT_EQ(graph.solveWitnessLowerBound()->bytes, 192);
}

TEST(MandatoryUBResourceGraph, PossibleAliasCannotBeSummed) {
  MandatoryUBResourceGraph graph;
  auto a = graph.addResource({"a", 64, 1});
  auto b = graph.addResource({"b", 128, 1});
  graph.addMayAlias(a, b);
  graph.addWitness({a, b});
  EXPECT_EQ(graph.solveWitnessLowerBound()->bytes, 128);
}
```

- [ ] **Step 2: Run the test target and confirm it fails to compile**

Run:

```bash
cmake --build .build-ttir-ub --target TestAscendTTIRUBLowerBound -j8
```

Expected: compilation fails because `MandatoryUBResourceGraph.h` and its types do not exist.

- [ ] **Step 3: Implement graph storage and checked arithmetic**

Use this public shape; keep MLIR operations out of this file so graph tests remain lightweight:

```cpp
namespace mlir::triton::ascend::ub {
using ResourceId = uint32_t;
using WitnessId = uint32_t;

enum class ValidityState { Valid, Invalid };
enum class UBAddressSpace { UB };
enum class MaterializationKind { GMToUBLoad };

struct ProgramPoint {
  uint64_t ordinal = 0;
};

struct MandatoryUBResource {
  std::string debugName;
  int64_t minPayloadBytes;
  int64_t minInstances;
  std::string origin;
  UBAddressSpace addressSpace = UBAddressSpace::UB;
  MaterializationKind kind = MaterializationKind::GMToUBLoad;
  ProgramPoint birth;
  ProgramPoint lastRequiredUse;
  ValidityState validity = ValidityState::Valid;
  SmallVector<std::string> contractTrace;
  std::string invalidReason;
};

struct CoexistenceWitness {
  SmallVector<ResourceId> resources;
  SmallVector<std::string> contractTrace;
};

struct LowerBoundCertificate {
  int64_t bytes = 0;
  SmallVector<ResourceId> resourceIds;
  std::string kind;
};

class MandatoryUBResourceGraph {
public:
  ResourceId addResource(MandatoryUBResource resource);
  void addMayAlias(ResourceId lhs, ResourceId rhs);
  void addMustAlias(ResourceId lhs, ResourceId rhs);
  void addMustDistinct(ResourceId lhs, ResourceId rhs);
  WitnessId addWitness(CoexistenceWitness witness);
  void invalidate(ResourceId id, StringRef reason);
  FailureOr<LowerBoundCertificate> solveSingletonLowerBound() const;
  FailureOr<LowerBoundCertificate> solveWitnessLowerBound() const;
  ArrayRef<MandatoryUBResource> resources() const;
};
}
```

Implement multiplication and addition with explicit `INT64_MAX` division/subtraction checks. Do not wrap or saturate; return `failure()` so the caller can `defer`. `must-alias` resources are collapsed to one equivalence class using the maximum member lower bound; `may-alias` resources cannot be summed; a witness sums resources only when every pair has an explicit `must-distinct` relation.

- [ ] **Step 4: Register the library and test target**

`third_party/ascend/lib/Analysis/TTIRUBLowerBound/CMakeLists.txt`:

```cmake
add_triton_library(TTIRUBLowerBound
  MandatoryUBResourceGraph.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  TritonIR
)
```

Add `add_subdirectory(Analysis/TTIRUBLowerBound)` to `third_party/ascend/lib/CMakeLists.txt` and add this target to `third_party/ascend/unittest/CMakeLists.txt`:

```cmake
add_triton_ut(
  NAME TestAscendTTIRUBLowerBound
  SRCS TTIRUBLowerBoundTest.cpp
  LIBS TTIRUBLowerBound TritonIR MLIRIR MLIRSupport
)
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
cmake --build .build-ttir-ub --target TestAscendTTIRUBLowerBound -j8
ctest --test-dir .build-ttir-ub -R TestAscendTTIRUBLowerBound --output-on-failure
git diff --check
```

Expected: all four graph tests pass and `git diff --check` prints nothing.

Commit:

```bash
git add third_party/ascend/include/Analysis/TTIRUBLowerBound/MandatoryUBResourceGraph.h \
        third_party/ascend/lib/Analysis/TTIRUBLowerBound \
        third_party/ascend/lib/CMakeLists.txt \
        third_party/ascend/unittest/TTIRUBLowerBoundTest.cpp \
        third_party/ascend/unittest/CMakeLists.txt
git commit -s -m "feat: add mandatory UB resource graph"
```

### Task 2: UBResourceContract, capacity and pipeline profile registry

**Files:**
- Create: `third_party/ascend/include/Analysis/TTIRUBLowerBound/UBResourceContract.h`
- Create: `third_party/ascend/lib/Analysis/TTIRUBLowerBound/UBResourceContract.cpp`
- Modify: `third_party/ascend/lib/Analysis/TTIRUBLowerBound/CMakeLists.txt`
- Modify: `third_party/ascend/unittest/TTIRUBLowerBoundTest.cpp`
- Create: `third_party/ascend/backend/ub_contract_profiles.json`

**Interfaces:**
- Consumes: `MandatoryUBResourceGraph` from Task 1.
- Produces: `PipelineIdentity`, `PipelineStageContext`, `ContractDisposition`, `UBResourceContract`, `PipelineContractRegistry`, and `getUBCapacityBytes`.

- [ ] **Step 1: Add failing contract and capacity tests**

```cpp
TEST(UBResourceContract, UnknownStageInvalidatesResources) {
  MandatoryUBResourceGraph graph;
  graph.addResource({"load0", 262144, 1});
  PipelineContractRegistry registry;
  registry.applyOrInvalidateAll(graph, {.stageName = "unknown-pass"});
  EXPECT_EQ(graph.solveSingletonLowerBound()->bytes, 0);
}

TEST(UBResourceContract, TransformCanOnlyLowerToProvenMinimum) {
  MandatoryUBResourceGraph graph;
  auto id = graph.addResource({"load0", 262144, 1});
  PipelineContractRegistry registry;
  registry.addForTesting(makeFixedTileContract("tile", 2));
  registry.applyOrInvalidateAll(graph, {.stageName = "tile"});
  EXPECT_EQ(graph.resources()[id].minPayloadBytes, 131072);
}

TEST(UBResourceContract, CapacityHasNoUnknownDefault) {
  EXPECT_EQ(*getUBCapacityBytes("Ascend910B"), 192 * 1024);
  EXPECT_EQ(*getUBCapacityBytes("Ascend910_95"), 256 * 1024);
  EXPECT_EQ(*getUBCapacityBytes("Ascend950"), 256 * 1024);
  EXPECT_FALSE(getUBCapacityBytes("future-chip").has_value());
}
```

- [ ] **Step 2: Run and confirm missing interfaces**

Run the Task 1 build command. Expected: compile failure naming `PipelineContractRegistry` and `getUBCapacityBytes`.

- [ ] **Step 3: Implement contract semantics**

```cpp
enum class ContractDisposition { Preserve, Transform, Invalidate, InternalError };

struct PipelineIdentity {
  std::string openSourcePipeline;
  std::string relevantOptionsJson;
  std::string targetArch;
  std::string tritonVersion;
  std::string cannVersionHash;
  std::string sha256;
};

struct PipelineStageContext {
  std::string stageName;
  StringMap<std::string> options;
};

class UBResourceContract {
public:
  virtual ~UBResourceContract() = default;
  virtual StringRef id() const = 0;
  virtual StringRef version() const = 0;
  virtual bool matches(const PipelineStageContext &) const = 0;
  virtual ContractDisposition apply(MandatoryUBResourceGraph &,
                                    const PipelineStageContext &) const = 0;
};
```

`applyOrInvalidateAll` must invalidate every currently valid resource when no contract matches. A matching contract returning `InternalError` returns `failure()` to the top-level analyzer.

- [ ] **Step 4: Add the empty production profile file**

```json
{
  "schema": "ttir-ub-lb-profile-v1",
  "profiles": []
}
```

Tests inject synthetic profiles through `addForTesting`; production code must not contain an `allow_unvalidated` option.

- [ ] **Step 5: Run tests and commit**

Run the graph test target and `git diff --check`. Expected: all tests pass.

Commit:

```bash
git add third_party/ascend/include/Analysis/TTIRUBLowerBound/UBResourceContract.h \
        third_party/ascend/lib/Analysis/TTIRUBLowerBound \
        third_party/ascend/unittest/TTIRUBLowerBoundTest.cpp \
        third_party/ascend/backend/ub_contract_profiles.json
git commit -s -m "feat: add UB resource contracts"
```

### Task 3: Strict TTIR direct-load source rule

**Files:**
- Create: `third_party/ascend/include/Analysis/TTIRUBLowerBound/TTIRUBLowerBound.h`
- Create: `third_party/ascend/lib/Analysis/TTIRUBLowerBound/TTIRUBLowerBound.cpp`
- Create: `third_party/ascend/lib/Analysis/TTIRUBLowerBound/DirectTensorLoadMaterialization.cpp`
- Modify: `third_party/ascend/lib/Analysis/TTIRUBLowerBound/CMakeLists.txt`
- Modify: `third_party/ascend/unittest/TTIRUBLowerBoundTest.cpp`

**Interfaces:**
- Consumes: MURG and registry from Tasks 1–2.
- Produces: `TTIRUBAnalysisOptions`, `TTIRUBAnalysisResult`, `analyzeTTIRUBLowerBound(ModuleOp, options, registry)`.

- [ ] **Step 1: Add a failing positive analyzer test**

Parse this fixture in the GTest with `parseSourceString<ModuleOp>` after loading `arith::ArithDialect` and `triton::TritonDialect`:

```mlir
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
```

Assert a synthetic all-preserve profile returns `lowerBoundBytes == 262144` and `decision == Reject` for `Ascend910B`.

- [ ] **Step 2: Add failing defer tests**

Create one mutation per case and assert `decision == Defer` with the named reason:

```text
masked-load
dynamic-shape
non-contiguous-pointer
load-not-reaching-store
nested-region
unsupported-op-reduction
sub-byte-element-type
unknown-pipeline-profile
```

Also test 49152 `f32` values on 192 KiB returns `defer` because equality does not overflow.

- [ ] **Step 3: Run and observe failures**

Run the GTest target. Expected: compile failure because `analyzeTTIRUBLowerBound` does not exist.

- [ ] **Step 4: Implement the minimal source matcher**

Define the public result types before implementing the matcher:

```cpp
enum class TTIRUBDecision { Reject, Defer };

struct TTIRUBAnalysisOptions {
  std::string targetArch;
  std::string compileMode;
  PipelineIdentity pipelineIdentity;
  SmallVector<PipelineStageContext> stages;
};

struct TTIRUBAnalysisResult {
  TTIRUBDecision decision = TTIRUBDecision::Defer;
  int64_t lowerBoundBytes = 0;
  std::optional<int64_t> capacityBytes;
  SmallVector<LowerBoundCertificate> certificates;
  SmallVector<std::string> unsupportedReasons;
  std::string contractVersion = "ttir-ub-lb-v1";
};
```

V1 accepts only this chain:

```text
function pointer block argument
  → tt.splat
  → tt.addptr(offset = tt.make_range(start=0, end=N))
  → unmasked ranked tt.load
  → direct unmasked tt.store
```

Require the load and store to be in the function entry block, reject any enclosing region other than `tt.func`, require one load result use, and accept only integer/float element widths in `{8, 16, 32, 64}`. Compute bytes as:

```cpp
numElements * llvm::divideCeil(elementBitWidth, 8u)
```

Use checked multiplication. Do not accept arithmetic elementwise chains in V1.

Build a `GMToUBLoad` singleton resource, apply every stage in the supplied registry, solve singleton LB, and return `Reject` only when `bytes > capacity`.

- [ ] **Step 5: Run tests and commit**

Run GTest and `git diff --check`. Expected: positive and all defer reason tests pass.

Commit:

```bash
git add third_party/ascend/include/Analysis/TTIRUBLowerBound/TTIRUBLowerBound.h \
        third_party/ascend/lib/Analysis/TTIRUBLowerBound \
        third_party/ascend/unittest/TTIRUBLowerBoundTest.cpp
git commit -s -m "feat: analyze mandatory TTIR UB loads"
```

### Task 4: pybind API, Python policy and unified capacity source

**Files:**
- Create: `third_party/ascend/ttir_ub_lower_bound_bindings.cc`
- Modify: `third_party/ascend/triton_ascend.cc`
- Modify: `third_party/ascend/CMakeLists.txt`
- Create: `third_party/ascend/backend/ub_lower_bound.py`
- Create: `third_party/ascend/backend/errors.py`
- Modify: `third_party/ascend/backend/runtime/utils.py`
- Create: `third_party/ascend/unittest/pytest_ut/test_ttir_ub_lower_bound.py`

**Interfaces:**
- Consumes: `analyzeTTIRUBLowerBound` and `getUBCapacityBytes`.
- Produces: `ascend.analysis.ttir_ub_lower_bound(module, options)` and `ascend.analysis.get_ub_capacity_bytes(arch)`.
- Produces: `apply_ub_lower_bound_policy(mod, metadata, opt, pipeline_identity)`.
- Produces: picklable `UBLowerBoundOverflow`.

- [ ] **Step 1: Write binding and policy tests with a mocked analysis result**

```python
def test_shadow_records_but_does_not_raise(monkeypatch):
    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", lambda *_: {
        "decision": "reject", "lower_bound_bytes": 262144,
        "capacity_bytes": 196608, "certificates": [{"kind": "singleton"}],
        "unsupported_reasons": [], "contract_version": "ttir-ub-lb-v1",
        "pipeline_identity": "test-id",
    })
    metadata = {"hash": "abc"}
    apply_ub_lower_bound_policy(object(), metadata, Options("shadow"), "test-id")
    assert metadata["ub_lower_bound_decision"] == "reject"

def test_enforce_raises_only_proven_reject(monkeypatch):
    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound", proven_reject)
    with pytest.raises(UBLowerBoundOverflow) as error:
        apply_ub_lower_bound_policy(object(), {}, Options("enforce"), "test-id")
    assert error.value.required == 262144
    assert error.value.limit == 196608

def test_analyzer_exception_fails_open(monkeypatch):
    monkeypatch.setattr(ascend.analysis, "ttir_ub_lower_bound",
                        lambda *_: (_ for _ in ()).throw(RuntimeError("bad")))
    metadata = {}
    apply_ub_lower_bound_policy(object(), metadata, Options("enforce"), "id")
    assert metadata["ub_lower_bound_decision"] == "defer"
    assert metadata["ub_lower_bound_unsupported_reasons"] == ["internal-error: bad"]
```

- [ ] **Step 2: Run and confirm import/interface failures**

Run:

```bash
python -m pytest third_party/ascend/unittest/pytest_ut/test_ttir_ub_lower_bound.py -q
```

Expected: import failure for `ub_lower_bound` or missing `ascend.analysis`.

- [ ] **Step 3: Implement binding serialization**

Register an `analysis` submodule from `init_triton_ascend` and return only JSON-serializable dict/list/string/integer values. Accept a `ModuleOp &` and a dict containing `arch`, `compile_mode`, `pipeline_identity`, `pipeline_stages`, and `contract_profile`.

Do not expose `allow_unvalidated`; profile lookup must come from packaged `ub_contract_profiles.json`.

- [ ] **Step 4: Implement Python policy and exception**

```python
class UBLowerBoundOverflow(OutOfResources):
    def __init__(self, required, limit, certificate, pipeline_identity):
        super().__init__(required, limit, "Ascend UB proven lower bound")
        self.certificate = certificate
        self.pipeline_identity = pipeline_identity

    def __reduce__(self):
        return (type(self), (self.required, self.limit,
                            self.certificate, self.pipeline_identity))
```

`apply_ub_lower_bound_policy` returns immediately in `off`, records a normalized metadata summary in `shadow/enforce`, catches analyzer exceptions as `defer`, and raises only in enforce plus reject.

- [ ] **Step 5: Replace Python capacity hardcoding**

Change `_init_npu_params()` to use:

```python
capacity = ascend.analysis.get_ub_capacity_bytes(target.arch)
if capacity is None:
    raise RuntimeError(f"Unknown Ascend UB capacity for {target.arch}")
ub_size_in_kbytes = capacity // 1024
```

Keep RF-size logic separate. Add tests for 192/256 KiB and unknown arch.

- [ ] **Step 6: Run tests and commit**

Run the GTest target, the new pytest file, and `git diff --check`. Expected: all pass.

Commit:

```bash
git add third_party/ascend/ttir_ub_lower_bound_bindings.cc \
        third_party/ascend/triton_ascend.cc third_party/ascend/CMakeLists.txt \
        third_party/ascend/backend/ub_lower_bound.py \
        third_party/ascend/backend/errors.py \
        third_party/ascend/backend/runtime/utils.py \
        third_party/ascend/unittest/pytest_ut/test_ttir_ub_lower_bound.py
git commit -s -m "feat: expose TTIR UB lower bound policy"
```

### Task 5: Real pipeline identity and compiler-stage integration

**Files:**
- Modify: `third_party/ascend/backend/compiler.py`
- Modify: `python/triton/compiler/compiler.py`
- Modify: `third_party/ascend/unittest/pytest_ut/test_ttir_ub_lower_bound.py`

**Interfaces:**
- Consumes: `apply_ub_lower_bound_policy`.
- Produces: `_build_ttir_to_linalg_pass_manager`, `_ttir_ub_pipeline_identity`, and `NPUOptions.ub_lower_bound_mode`.

- [ ] **Step 1: Add failing option and pipeline identity tests**

```python
def test_mode_validation():
    assert NPUOptions(ub_lower_bound_mode="off").ub_lower_bound_mode == "off"
    with pytest.raises(ValueError, match="ub_lower_bound_mode"):
        NPUOptions(ub_lower_bound_mode="probabilistic")

def test_pipeline_identity_changes_when_pass_order_changes(monkeypatch):
    first = _ttir_ub_pipeline_identity("pass-a,pass-b", metadata())
    second = _ttir_ub_pipeline_identity("pass-b,pass-a", metadata())
    assert first != second

def test_make_ttir_calls_policy_after_canonicalization(monkeypatch):
    calls = []
    monkeypatch.setattr(PM, "run", lambda self, mod: calls.append("pm"))
    monkeypatch.setattr(compiler, "apply_ub_lower_bound_policy",
                        lambda *args: calls.append("ub"))
    make_ttir(module, {}, Options("shadow"))
    assert calls == ["pm", "ub"]
```

- [ ] **Step 2: Refactor the pass manager without changing its pipeline string**

Move the current pass additions from `ttir_to_linalg` into:

```python
def _build_ttir_to_linalg_pass_manager(mod, metadata, opt):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    # Add the existing passes in exactly their current order.
    return pm
```

Before and after refactoring, capture `pm.get_pipeline_str()` for the existing default metadata fixture and assert byte-for-byte equality.

- [ ] **Step 3: Compute a normalized identity**

```python
def _ttir_ub_pipeline_identity(pipeline: str, metadata: dict) -> str:
    payload = {
        "pipeline": pipeline,
        "target": metadata["target"].arch,
        "triton_version": metadata["triton_version"],
        "cann_version_hash": get_cann_version_file_hash(),
        "options": {name: metadata.get(name) for name in UB_AFFECTING_OPTIONS},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()
```

`UB_AFFECTING_OPTIONS` must enumerate every option currently forwarded to BiSheng that can alter tiling, fusion, inplace, multi-buffer, CV or buffer reuse. A new forwarded option requires updating the tuple and its golden test.

- [ ] **Step 4: Invoke the policy after `make_ttir` passes**

Construct the future TTIR→Linalg PM only to obtain its normalized pipeline string; do not run it. Load the packaged profile matching the identity, then call the policy.

When `opt.debug` is true, write full JSON to `kernel.ttir.ub-lower-bound.json` through the existing dump manager.

- [ ] **Step 5: Preserve resource exceptions through the core stage wrapper**

In `python/triton/compiler/compiler.py`:

```python
        try:
            next_module = compile_ir(module, metadata)
        except OutOfResources:
            raise
        except Exception as e:
            # existing MLIRCompilationError wrapping remains unchanged
```

Add a unit test with a fake stage raising `UBLowerBoundOverflow` and assert the same exception type and fields reach the caller.

- [ ] **Step 6: Run tests and commit**

Run the new pytest file plus existing `test_debug_triton_opt_cmd.py`, then `git diff --check`.

Commit:

```bash
git add third_party/ascend/backend/compiler.py \
        python/triton/compiler/compiler.py \
        third_party/ascend/unittest/pytest_ut/test_ttir_ub_lower_bound.py
git commit -s -m "feat: run UB proof after TTIR canonicalization"
```

### Task 6: Parallel-safe Autotune filtering and telemetry

**Files:**
- Modify: `python/triton/runtime/_async_compile.py`
- Create: `python/test/unit/runtime/test_async_compile_context.py`
- Modify: `third_party/ascend/backend/ub_lower_bound.py`
- Modify: `third_party/ascend/backend/runtime/autotuner.py`
- Create: `third_party/ascend/unittest/autotune_ut/test_ub_lower_bound_filter.py`

**Interfaces:**
- Produces: `ub_filter_telemetry_session()` and `UBFilterStats`.
- Ensures async compiler workers inherit the active telemetry `ContextVar`.

- [ ] **Step 1: Add a failing generic context propagation test**

```python
def test_async_compile_propagates_contextvars():
    marker = ContextVar("marker", default="missing")
    marker.set("present")
    with ThreadPoolExecutor(max_workers=1) as pool:
        with AsyncCompileMode(pool) as mode:
            future = mode.submit("key", marker.get, lambda value: None)
            assert future.result() == "present"
```

- [ ] **Step 2: Implement context capture in `submit`**

```python
from contextvars import ContextVar, copy_context

context = copy_context()
future = self.executor.submit(context.run, compile_fn)
```

Run `python -m pytest python/test/unit/runtime/test_async_compile_context.py -q`; expected: pass.

- [ ] **Step 3: Add serial and parallel rejection tests**

Mock one config to raise `UBLowerBoundOverflow`, one to compile, and assert both `_batch_bench` paths retain only the compiling config. The parallel test must call `Future.result()` and must not let `AsyncCompileMode.__exit__` rethrow the already classified exception.

- [ ] **Step 4: Implement per-session telemetry**

```python
@dataclass
class UBFilterStats:
    analyzed: int = 0
    rejected: int = 0
    deferred: int = 0
    passed_to_backend: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

_active_stats = ContextVar("ascend_ub_filter_stats", default=None)
```

`apply_ub_lower_bound_policy` updates the active object under its lock. `_batch_bench` opens one session around serial or parallel compilation, catches `OutOfResources` in both paths, and prints one summary only when `TRITON_PRINT_AUTOTUNING=1`.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python -m pytest python/test/unit/runtime/test_async_compile_context.py -q
python -m pytest third_party/ascend/unittest/autotune_ut/test_ub_lower_bound_filter.py -q
python -m pytest third_party/ascend/unittest/autotune_ut/test_do_bench_compat.py -q
git diff --check
```

Expected: all pass; the telemetry test reports exactly one analyzed, one rejected and one passed-to-backend config.

Commit:

```bash
git add python/triton/runtime/_async_compile.py \
        python/test/unit/runtime/test_async_compile_context.py \
        third_party/ascend/backend/ub_lower_bound.py \
        third_party/ascend/backend/runtime/autotuner.py \
        third_party/ascend/unittest/autotune_ut/test_ub_lower_bound_filter.py
git commit -s -m "feat: filter UB overflow configs in autotune"
```

### Task 7: Real PlanMemory oracle and profile promotion gate

**Files:**
- Create: `third_party/ascend/tools/ttir_ub_oracle.py`
- Create: `third_party/ascend/unittest/ttir_ub_oracle/test_oracle.py`
- Create: `third_party/ascend/unittest/ttir_ub_oracle/fixtures/manifest.json`
- Create: `third_party/ascend/unittest/ttir_ub_oracle/fixtures/direct_copy.ttir.mlir`
- Create: `third_party/ascend/unittest/ttir_ub_oracle/fixtures/direct_copy.before_cvpipelining.mlir`
- Modify after successful certification: `third_party/ascend/backend/ub_contract_profiles.json`

**Interfaces:**
- Consumes: analyzer binding, canonical TTIR and before-CVPipelining fixture pairs.
- Consumes external real compiler: `bishengir-cvpipeline-suffix-compile` with `--ub-oracle-only` and `--plan-memory-seed`.
- Produces: machine-readable violation report and an auditable profile candidate JSON.

- [ ] **Step 1: Write oracle parser tests using captured stderr fixtures**

Test these rules:

```python
assert parse_planmemory_peak(success_stderr, attempt=0, scope="6") == 1572864
assert parse_overflow_scope(ub_overflow_stderr) == "UB"
assert parse_overflow_scope(l1_overflow_stderr) == "L1"
with pytest.raises(OracleUnavailable):
    classify_failure("generic parser error")
```

The parser must not classify a non-UB failure as UB overflow.

- [ ] **Step 2: Implement the manifest and subprocess runner**

The manifest schema is:

```json
{
  "schema": "ttir-ub-oracle-v1",
  "cases": [
    {
      "name": "direct-copy-f32-65536-a2",
      "ttir": "direct_copy.ttir.mlir",
      "before_cvpipelining": "direct_copy.before_cvpipelining.mlir",
      "arch": "Ascend910B",
      "options": {"compile_mode": "simd", "multibuffer": false},
      "expected_analyzer_decision": "reject"
    }
  ]
}
```

The runner obtains `pipeline_identity` from the analyzer result and writes it
to the profile candidate; a manifest must not provide or override that value.

For each case, call the analyzer once and the suffix compiler for seeds 0 through 19:

```text
bishengir-cvpipeline-suffix-compile INPUT \
  -o OUTPUT \
  --plan-memory-seed=N \
  --mlir-disable-threading \
  --ub-oracle-only
```

Set `BISHENGIR_DUMP_PLAN_MEMORY_ATTEMPTS=1`, parse scope `6`, and also run once with `--plan-memory-seed=-1` for retry behavior.

- [ ] **Step 3: Enforce the lower-bound assertions**

For every available seed result:

```python
assert lower_bound_bits <= actual_peak_bits
if analyzer_decision == "reject":
    assert oracle_status == "overflow"
    assert overflow_scope == "UB"
```

Exit code is `0` only when violations are zero; unavailable oracle cases use exit code `2`; proof violations use exit code `1`.

- [ ] **Step 4: Generate but do not automatically install a profile**

`--profile-candidate OUTPUT.json` writes the exact pipeline identity, contract version, target and CANN hash only when all 20 seeds and retry pass. It never edits the packaged profile file.

Run:

```bash
python third_party/ascend/tools/ttir_ub_oracle.py \
  --manifest third_party/ascend/unittest/ttir_ub_oracle/fixtures/manifest.json \
  --suffix-compiler /Users/sky/Code/AscendNPU-IR/.worktrees/cvpipeline-ub-post-model/build/bin/bishengir-cvpipeline-suffix-compile \
  --seeds 0-19 --check-retry \
  --profile-candidate /tmp/ttir-ub-profile-candidate.json
```

Expected: `violations=0`, `unavailable=0`, exit code `0`.

- [ ] **Step 5: Review and install the profile candidate**

Compare the candidate identity with the shadow metadata from the same binaries. Copy the complete candidate object into `ub_contract_profiles.json` only when the command in Step 4 succeeded. Re-run the oracle after installation and assert the analyzer now returns a production-valid certificate.

- [ ] **Step 6: Run unit tests and commit**

Run parser tests without hardware, then the full oracle in a CANN/BiSheng environment. Commit the tool and fixtures; include the profile only if certified.

```bash
git add third_party/ascend/tools/ttir_ub_oracle.py \
        third_party/ascend/unittest/ttir_ub_oracle
git add third_party/ascend/backend/ub_contract_profiles.json
git commit -s -m "test: validate TTIR UB lower bounds with PlanMemory"
```

### Task 8: End-to-end modes, documentation and release verification

**Files:**
- Create: `docs/zh/ttir_ub_conservative_filter.md`
- Create: `docs/en/ttir_ub_conservative_filter.md`
- Modify: `third_party/ascend/unittest/pytest_ut/test_ttir_ub_lower_bound.py`
- Modify: `third_party/ascend/unittest/autotune_ut/test_ub_lower_bound_filter.py`

**Interfaces:**
- Verifies the complete public behavior and documents activation/diagnostics/rollback.

- [ ] **Step 1: Add end-to-end mode tests**

Compile the same direct-copy TTIR with:

```text
off     → analyzer binding is not called
shadow  → compilation continues and metadata contains the result
enforce + no certified profile → defer and compilation continues
enforce + certified profile + LB == capacity → compilation continues
enforce + certified profile + LB > capacity → UBLowerBoundOverflow
```

Also assert the debug dump is valid JSON and contains the full contract trace.

- [ ] **Step 2: Write bilingual user/developer documentation**

Document this exact activation contract:

```python
triton.Config({"BLOCK_SIZE": 65536, "ub_lower_bound_mode": "shadow"})
triton.Config({"BLOCK_SIZE": 65536, "ub_lower_bound_mode": "enforce"})
```

Explain that `defer` is not proof of fit, `reject` is proof of overflow, default is `off`, and rollback is `ub_lower_bound_mode="off"`. Include MURG and `UBResourceContract` extension steps and the oracle promotion command from Task 7.

- [ ] **Step 3: Run focused verification**

```bash
cmake --build .build-ttir-ub --target TestAscendTTIRUBLowerBound -j8
ctest --test-dir .build-ttir-ub -R TestAscendTTIRUBLowerBound --output-on-failure
python -m pytest third_party/ascend/unittest/pytest_ut/test_ttir_ub_lower_bound.py -q
python -m pytest third_party/ascend/unittest/autotune_ut/test_ub_lower_bound_filter.py -q
python -m pytest third_party/ascend/unittest/autotune_ut/test_do_bench_compat.py -q
python -m pytest python/test/unit/runtime/test_async_compile_context.py -q
git diff --check
```

Expected: all tests pass and no whitespace errors.

- [ ] **Step 4: Run full release verification**

In the configured CANN/NPU environment:

```bash
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=ON" python setup.py bdist_wheel
python -m pytest -q third_party/ascend/unittest/autotune_ut
python -m pytest -q third_party/ascend/unittest/pytest_ut
cmake --build .build-ttir-ub --target check-triton-ascend-lit-tests -j8
python third_party/ascend/tools/ttir_ub_oracle.py \
  --manifest third_party/ascend/unittest/ttir_ub_oracle/fixtures/manifest.json \
  --suffix-compiler /Users/sky/Code/AscendNPU-IR/.worktrees/cvpipeline-ub-post-model/build/bin/bishengir-cvpipeline-suffix-compile \
  --seeds 0-19 --check-retry
```

Expected: build succeeds, test suites pass, oracle reports zero violations/unavailable cases.

- [ ] **Step 5: Measure analyzer overhead**

Run the fixture corpus 100 times in shadow mode, discard the first 10 iterations, and report p50/p95 from `time.perf_counter_ns()`. Acceptance: p95 is below 5 ms per module. Store the measurement summary in the final handoff, not in a generated repository artifact.

- [ ] **Step 6: Commit documentation and final test updates**

```bash
git add docs/zh/ttir_ub_conservative_filter.md \
        docs/en/ttir_ub_conservative_filter.md \
        third_party/ascend/unittest/pytest_ut/test_ttir_ub_lower_bound.py \
        third_party/ascend/unittest/autotune_ut/test_ub_lower_bound_filter.py
git commit -s -m "docs: document TTIR UB conservative filtering"
```

## Completion Checklist

- [ ] Production profile is either empty or backed by a checked-in zero-violation oracle report identity.
- [ ] No `allow_unvalidated`, confidence threshold, probability or heuristic reaches enforce policy.
- [ ] Every unsupported path produces a structured `defer` reason.
- [ ] `LB == capacity` is not rejected; only `LB > capacity` is rejected.
- [ ] Serial and parallel autotune preserve at least every non-UB-failing config.
- [ ] Pipeline pass order and UB-affecting options are included in the identity.
- [ ] Full certificate JSON is debug-only; normal metadata remains compact and JSON serializable.
- [ ] Existing capacity users consume the same C++ target contract.
- [ ] Focused tests, full tests, real oracle and `git diff --check` pass.
- [ ] All commits contain `Signed-off-by` trailers.
