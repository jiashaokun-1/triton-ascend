# HIVM MultiBuffer UB Model Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable C++ HIVM analysis that reports UB buffers, alias-aware lifetimes, MultiBuffer slots, shadow-planned peak bytes, conservative upper-bound bytes, and SAFE/REVIEW/OVERFLOW without reading `PlanMemory` output.

**Architecture:** Add focused analysis units under `bishengir/Dialect/HIVM/Analysis`: extraction, alias/liveness, slot expansion, planning, and reporting. Exercise the API through a test-only `bishengir-opt` pass and MLIR lit tests. Real `MarkMultiBuffer` and `PlanMemory` remain unchanged and become labels in a later calibration plan.

**Tech Stack:** C++17, MLIR/HIVM IR, MLIR `Liveness`, LLVM ADT and `llvm::json`, CMake/Ninja, `bishengir-opt`, llvm-lit/FileCheck.

## Global Constraints

- Implement in `/home/skj/code/AscendNPU-IR` inside container `sgl-skj` on `root@192.168.25.212`.
- Sync source offline from `/Users/sky/Code/AscendNPU-IR`; exclude `.git`, `build`, caches, and Torch-MLIR. Temporarily restore the repository-pinned LLVM submodule because the container has no LLVM/MLIR development package.
- Only `#hivm.address_space<ub>` contributes to UB bytes; GM/L1/L0 remain diagnostic metadata.
- Consume the second local `MarkMultiBuffer` checkpoint; never read `PlanMemory` pointer offsets as prediction input.
- Unknown UB-touching semantics, unresolved dynamic size, missing scope, or unmatched Mark source must never produce `SAFE`.
- Do not modify `MarkMultiBuffer.cpp`, `PlanMemory.cpp`, or production pipeline ordering in this plan.
- Use TDD and commit after every independently reviewable task.

---

## File Structure

Paths are relative to `/home/skj/code/AscendNPU-IR`.

- `bishengir/include/bishengir/Dialect/HIVM/Analysis/UBModel.h` — public options, result schema, decision, and facade.
- `bishengir/include/bishengir/Dialect/HIVM/Analysis/UBBufferGraph.h` — buffers, alias groups, operation points, conflicts, and slots.
- `bishengir/lib/Dialect/HIVM/Analysis/UBBufferGraph.cpp` — UB collection, size, stable IDs, aliases.
- `bishengir/lib/Dialect/HIVM/Analysis/UBLiveness.cpp` — gen/kill and structured control flow.
- `bishengir/lib/Dialect/HIVM/Analysis/UBPlanner.cpp` — slot expansion, live peak, placement, conservative bound.
- `bishengir/lib/Dialect/HIVM/Analysis/UBModel.cpp` — validation, orchestration, decision, JSON.
- `bishengir/test/lib/Dialect/HIVM/TestHIVMUBModel.cpp` — test-only command-line pass.
- `bishengir/test/Dialect/HIVM/Analysis/ub-model-*.mlir` — controlled behavior tests.

---

### Task 1: Prepare the Offline Workspace and Baseline

**Files:**
- Verify only: `/home/skj/code/AscendNPU-IR/build/bin/bishengir-opt`

**Interfaces:**
- Consumes: local source and SSH access.
- Produces: buildable server tree and a passing memory-pass baseline.

- [ ] **Step 1: Dry-run the filtered transfer**

```bash
rsync -an --stats --exclude=.git/ --exclude=build/ \
  --exclude='**/__pycache__/' --exclude='**/.pytest_cache/' \
  --exclude='third-party/llvm-project/' --exclude='third-party/torch-mlir/' \
  /Users/sky/Code/AscendNPU-IR/ \
  root@192.168.25.212:/home/skj/code/AscendNPU-IR/
```

Expected: dry-run succeeds and excludes local build artifacts, LLVM, and Torch-MLIR from the first transfer.

- [ ] **Step 2: Synchronize the filtered source**

```bash
rsync -a --delete-delay --exclude=.git/ --exclude=build/ \
  --exclude='**/__pycache__/' --exclude='**/.pytest_cache/' \
  --exclude='third-party/llvm-project/' --exclude='third-party/torch-mlir/' \
  /Users/sky/Code/AscendNPU-IR/ \
  root@192.168.25.212:/home/skj/code/AscendNPU-IR/
```

Expected: `CMakeLists.txt` is visible at the destination inside `sgl-skj`.

- [ ] **Step 3: Restore and transfer the repository-pinned LLVM source**

```bash
git -C /Users/sky/Code/AscendNPU-IR submodule update --init \
  third-party/llvm-project
rsync -a /Users/sky/Code/AscendNPU-IR/third-party/llvm-project/ \
  root@192.168.25.212:/home/skj/code/AscendNPU-IR/third-party/llvm-project/
```

Expected: the server has `third-party/llvm-project/llvm/CMakeLists.txt` at the commit recorded by the AscendNPU-IR superproject.

- [ ] **Step 4: Configure and build with the documented build script**

```bash
ssh root@192.168.25.212 \
  'docker exec sgl-skj bash --noprofile --norc -c "cd /home/skj/code/AscendNPU-IR && ./build-tools/build.sh -o ./build --build-type Release --apply-patches --fast-build -j 128"'
```

Expected: exit 0 and `build/bin/bishengir-opt` exists.

- [ ] **Step 5: Establish the baseline**

```bash
ssh root@192.168.25.212 \
  'docker exec sgl-skj bash --noprofile --norc -c "cd /home/skj/code/AscendNPU-IR/build && ./bin/llvm-lit -v ../bishengir/test/Dialect/HIVM/mark-multi-buffer.mlir ../bishengir/test/Dialect/HIVM/plan-memory.mlir"'
```

Expected: both pass. Stop and record any pre-existing failure before changing source.

- [ ] **Step 6: Release the temporary local LLVM working tree after the remote baseline passes**

```bash
git -C /Users/sky/Code/AscendNPU-IR submodule deinit -f -- \
  third-party/llvm-project
```

Expected: the remote LLVM source remains available for builds while local disk usage returns to its pre-transfer state.

---

### Task 2: Define the Contract and Collect Static UB Buffers

**Files:**
- Create: `bishengir/include/bishengir/Dialect/HIVM/Analysis/UBModel.h`
- Create: `bishengir/include/bishengir/Dialect/HIVM/Analysis/UBBufferGraph.h`
- Create: `bishengir/lib/Dialect/HIVM/Analysis/UBBufferGraph.cpp`
- Create: `bishengir/lib/Dialect/HIVM/Analysis/UBModel.cpp`
- Modify: `bishengir/lib/Dialect/HIVM/Analysis/CMakeLists.txt`
- Create: `bishengir/test/lib/Dialect/HIVM/TestHIVMUBModel.cpp`
- Modify: `bishengir/test/lib/Dialect/HIVM/CMakeLists.txt`
- Modify: `bishengir/test/lib/Dialect/Test/TestPasses.h`
- Create: `bishengir/test/Dialect/HIVM/Analysis/ub-model-buffer-collection.mlir`

**Interfaces:**
- Consumes: `func::FuncOp` and `UBModelOptions`.
- Produces: `FailureOr<UBModelResult> UBModel::analyze(func::FuncOp, const UBModelOptions &)`, stable buffer IDs, sizes, and validation issues.

- [ ] **Step 1: Write the failing UB-only collection test**

```mlir
// RUN: bishengir-opt %s -test-hivm-ub-model='ub-capacity-bytes=1024 alignment-bytes=32' | FileCheck %s
// CHECK: "raw_bytes":32
// CHECK: "aligned_bytes":32
// CHECK: "address_space":"ub"
// CHECK-NOT: "address_space":"gm"
func.func @collect_ub(%src: memref<8xf32, #hivm.address_space<gm>>) {
  %ub = memref.alloc() : memref<8xf32, #hivm.address_space<ub>>
  hivm.hir.load ins(%src : memref<8xf32, #hivm.address_space<gm>>)
                outs(%ub : memref<8xf32, #hivm.address_space<ub>>)
  return
}
```

- [ ] **Step 2: Verify the test fails because the pass is absent**

```bash
cd /home/skj/code/AscendNPU-IR/build
./bin/llvm-lit -v ../bishengir/test/Dialect/HIVM/Analysis/ub-model-buffer-collection.mlir
```

Expected: FAIL with unknown `test-hivm-ub-model`.

- [ ] **Step 3: Define the exact public result types**

```cpp
enum class UBDecision { Safe, Review, Overflow, Invalid };
enum class UBIssueKind { MissingScope, DynamicSizeWithoutBound,
  UnknownUBTouch, InvalidMultiBufferFactor, UnresolvedMarkSource };
struct UBModelOptions { uint64_t ubCapacityBytes = 0; uint64_t alignmentBytes = 0; };
struct UBBufferResult {
  std::string id; uint64_t rawBytes = 0; uint64_t alignedBytes = 0;
  uint32_t multiBufferFactor = 1; SmallVector<uint64_t> slotOffsets;
  int64_t genPoint = -1; int64_t killPoint = -1; std::string aliasGroup;
};
struct UBModelResult {
  UBDecision decision = UBDecision::Invalid;
  uint64_t livePeakBytes = 0, plannedBytes = 0, conservativeBytes = 0;
  SmallVector<UBBufferResult> buffers;
  SmallVector<std::pair<UBIssueKind, std::string>> issues;
  llvm::json::Object toJson() const;
};
class UBModel {
public:
  static FailureOr<UBModelResult> analyze(func::FuncOp,
                                          const UBModelOptions &);
};
```

- [ ] **Step 4: Implement static collection**

For static memrefs use:

```cpp
rawBytes = memrefType.getNumElements() *
           memrefType.getElementTypeBitWidth() / 8;
alignedBytes = llvm::alignTo(rawBytes, options.alignmentBytes);
id = (func.getSymName() + "/b" + Twine(allocOrdinal)).str();
```

Only UB allocs enter `buffers`. Dynamic shapes create `DynamicSizeWithoutBound` and prevent SAFE.

- [ ] **Step 5: Register the test pass and print deterministic JSON**

```cpp
llvm::outs() << llvm::formatv("{0}\n", llvm::json::Value(result.toJson()));
```

Register `TestHIVMUBModel` in `TestPasses.h`, add its source to `BiShengIRTestDialectHIVM`, and add all new analysis sources to `BiShengIRHIVMAnalysis`.

- [ ] **Step 6: Build, test, and commit**

```bash
cd /home/skj/code/AscendNPU-IR/build
ninja bishengir-opt BiShengIRTestDialectHIVM
./bin/llvm-lit -v ../bishengir/test/Dialect/HIVM/Analysis/ub-model-buffer-collection.mlir
cd ..
git add bishengir/include/bishengir/Dialect/HIVM/Analysis \
  bishengir/lib/Dialect/HIVM/Analysis \
  bishengir/test/lib/Dialect/HIVM bishengir/test/lib/Dialect/Test/TestPasses.h \
  bishengir/test/Dialect/HIVM/Analysis/ub-model-buffer-collection.mlir
git commit -m "feat(hivm): collect UB buffers for impact model"
```

Expected: focused test PASS and commit succeeds.

---

### Task 3: Build Alias Groups and PlanMemory-Style Lifetimes

**Files:**
- Modify: `bishengir/include/bishengir/Dialect/HIVM/Analysis/UBBufferGraph.h`
- Modify: `bishengir/lib/Dialect/HIVM/Analysis/UBBufferGraph.cpp`
- Create: `bishengir/lib/Dialect/HIVM/Analysis/UBLiveness.cpp`
- Modify: `bishengir/lib/Dialect/HIVM/Analysis/CMakeLists.txt`
- Create: `bishengir/test/Dialect/HIVM/Analysis/ub-model-alias-liveness.mlir`

**Interfaces:**
- Consumes: collected allocs, operation alias information, and MLIR `Liveness`.
- Produces: alias groups plus deterministic `[genPoint, killPoint]` and uncertainty reasons.

- [ ] **Step 1: Add failing tests for DPS, subview, loop yield, select, and unknown op**

```mlir
// CHECK: "alias_group":"g0"
// CHECK: "alias_member_count":4
// CHECK: "gen_point":1
// CHECK: "kill_point":3
// UNKNOWN: "decision":"REVIEW"
// UNKNOWN: "issue":"unknown_ub_touch"
```

Use separate `// -----` sections for a linear load/add/store chain, `scf.for` iter arg/yield, `scf.if`, `arith.select`, and an unregistered op with a UB operand.

- [ ] **Step 2: Verify missing alias/lifetime output fails**

```bash
./bin/llvm-lit -v ../bishengir/test/Dialect/HIVM/Analysis/ub-model-alias-liveness.mlir
```

Expected: FAIL on alias and gen/kill checks.

- [ ] **Step 3: Implement alias union-find**

```cpp
void unionAlias(Value lhs, Value rhs, bool conditional);
void collectOperationAliases(Operation *op);
void collectForAliases(scf::ForOp op);
void collectWhileAliases(scf::WhileOp op);
void collectIfAliases(scf::IfOp op);
void finalizeAliasGroups();
```

Use `getOperationAliasInfo`, DPS init/result, loop/while args and yields, if results, select, scope return, subview, and branch block arguments. Preserve a conditional bit where mutual exclusion is not proven.

- [ ] **Step 4: Implement gen/kill**

```cpp
LogicalResult computeLiveness(func::FuncOp func);
bool allAliasesDeadAfter(StringRef groupId, Operation *op,
                         const Liveness &liveness) const;
void extendToParentRegion(StringRef groupId, Operation *reasonOp,
                          UBIssueKind reason);
```

Number operations in deterministic pre-order. Treat load/DPS outputs as gen; kill only when all aliases are dead after the point. Extend loop-carried, yielded, preload, and unknown-touch intervals to the parent loop/region. Unknown UB touches force REVIEW.

- [ ] **Step 5: Build, test, and commit**

```bash
ninja bishengir-opt BiShengIRTestDialectHIVM
./bin/llvm-lit -v ../bishengir/test/Dialect/HIVM/Analysis/ub-model-alias-liveness.mlir
cd /home/skj/code/AscendNPU-IR
git add bishengir/include/bishengir/Dialect/HIVM/Analysis/UBBufferGraph.h \
  bishengir/lib/Dialect/HIVM/Analysis \
  bishengir/test/Dialect/HIVM/Analysis/ub-model-alias-liveness.mlir
git commit -m "feat(hivm): model UB aliases and lifetimes"
```

Expected: all alias/lifetime sections PASS.

---

### Task 4: Read MultiBuffer Marks and Expand Slots

**Files:**
- Modify: `bishengir/lib/Dialect/HIVM/Analysis/UBBufferGraph.cpp`
- Create: `bishengir/lib/Dialect/HIVM/Analysis/UBPlanner.cpp`
- Modify: `bishengir/lib/Dialect/HIVM/Analysis/CMakeLists.txt`
- Create: `bishengir/test/Dialect/HIVM/Analysis/ub-model-multibuffer.mlir`

**Interfaces:**
- Consumes: `annotation.mark` and base buffer graph.
- Produces: N physical slots for factor N, with sibling slots forced to conflict.

- [ ] **Step 1: Write failing factor 1/2/4/N and invalid-source tests**

```mlir
// FACTOR2: "multi_buffer_factor":2
// FACTOR2: "slot_count":2
// FACTOR4: "slot_count":4
// INVALID: "decision":"INVALID"
// UNRESOLVED: "decision":"REVIEW"
```

- [ ] **Step 2: Verify every buffer still has one slot**

```bash
./bin/llvm-lit -v ../bishengir/test/Dialect/HIVM/Analysis/ub-model-multibuffer.mlir
```

Expected: FAIL.

- [ ] **Step 3: Implement factor validation and expansion**

```cpp
struct UBSlot {
  std::string id, bufferId; uint32_t ordinal = 0; uint64_t size = 0;
  int64_t genPoint = -1, killPoint = -1;
  std::optional<uint64_t> offset;
};
LogicalResult applyMultiBufferMarks();
SmallVector<UBSlot> expandSlots() const;
```

Trace Mark source using the same alloc traceback utility used by production. Require integer factor `>=1`. Name slots `bufferId + ".slot" + ordinal`; sibling slots always conflict.

- [ ] **Step 4: Build, test, and commit**

```bash
ninja bishengir-opt BiShengIRTestDialectHIVM
./bin/llvm-lit -v ../bishengir/test/Dialect/HIVM/Analysis/ub-model-multibuffer.mlir
cd /home/skj/code/AscendNPU-IR
git add bishengir/lib/Dialect/HIVM/Analysis \
  bishengir/test/Dialect/HIVM/Analysis/ub-model-multibuffer.mlir
git commit -m "feat(hivm): expand multibuffer UB slots"
```

Expected: factor and error sections PASS.

---

### Task 5: Implement Shadow Placement, Live Peak, and Conservative Bound

**Files:**
- Modify: `bishengir/lib/Dialect/HIVM/Analysis/UBPlanner.cpp`
- Modify: `bishengir/lib/Dialect/HIVM/Analysis/UBModel.cpp`
- Create: `bishengir/test/Dialect/HIVM/Analysis/ub-model-planner.mlir`

**Interfaces:**
- Consumes: aligned slots, lifetime/sibling conflicts, UB capacity, and issues.
- Produces: offsets, `livePeakBytes`, `plannedBytes`, `conservativeBytes`, and decision.

- [ ] **Step 1: Add failing reuse, overlap, exact-boundary, and REVIEW tests**

```mlir
// REUSE: "planned_bytes":64
// REUSE: "slot_offsets":[0,0]
// OVERLAP: "planned_bytes":128
// EXACT: "decision":"SAFE"
// OVER: "decision":"OVERFLOW"
// REVIEW: "decision":"REVIEW"
```

- [ ] **Step 2: Verify the planner test fails**

```bash
./bin/llvm-lit -v ../bishengir/test/Dialect/HIVM/Analysis/ub-model-planner.mlir
```

Expected: FAIL on offsets and peaks.

- [ ] **Step 3: Implement deterministic first-fit placement**

```cpp
llvm::stable_sort(slots, [](const UBSlot &a, const UBSlot &b) {
  return std::tie(a.genPoint, a.killPoint, a.id) <
         std::tie(b.genPoint, b.killPoint, b.id);
});
bool conflicts(const UBSlot &a, const UBSlot &b) {
  bool life = a.genPoint <= b.killPoint && b.genPoint <= a.killPoint;
  bool sibling = a.bufferId == b.bufferId && a.id != b.id;
  return life || sibling;
}
```

Try aligned candidate offsets 0 and every placed interval end; choose the smallest non-overlapping address among conflicting slots. Set `plannedBytes = max(offset + size)`.

- [ ] **Step 4: Compute the two independent bounds and decision**

At every operation point sum live slot sizes for `livePeakBytes`. Sum every slot with no reuse for `conservativeBytes`. Apply:

```cpp
if (hasInvalidInput) decision = UBDecision::Invalid;
else if (plannedBytes > ubCapacityBytes) decision = UBDecision::Overflow;
else if (ubCapacityBytes == 0 || hasUncertainty ||
         conservativeBytes > ubCapacityBytes) decision = UBDecision::Review;
else decision = UBDecision::Safe;
```

- [ ] **Step 5: Build, test, and commit**

```bash
ninja bishengir-opt BiShengIRTestDialectHIVM
./bin/llvm-lit -v ../bishengir/test/Dialect/HIVM/Analysis/ub-model-planner.mlir
cd /home/skj/code/AscendNPU-IR
git add bishengir/lib/Dialect/HIVM/Analysis \
  bishengir/test/Dialect/HIVM/Analysis/ub-model-planner.mlir
git commit -m "feat(hivm): shadow-plan UB memory"
```

Expected: all planner decisions PASS.

---

### Task 6: Stabilize JSON and Test Mark-Before/After Pairs

**Files:**
- Modify: `bishengir/lib/Dialect/HIVM/Analysis/UBModel.cpp`
- Modify: `bishengir/test/lib/Dialect/HIVM/TestHIVMUBModel.cpp`
- Create: `bishengir/test/Dialect/HIVM/Analysis/ub-model-end-to-end.mlir`

**Interfaces:**
- Consumes: one complete HIVM snapshot.
- Produces: schema-version-1 JSON, stable enough for the later calibration harness to compare two snapshots.

- [ ] **Step 1: Write failing paired checks**

Use identical kernels except for a factor-2 Mark:

```mlir
// BEFORE: "planned_bytes":8192
// BEFORE: "multi_buffer_factor":1
// AFTER: "planned_bytes":16384
// AFTER: "multi_buffer_factor":2
// AFTER: "slot_offsets":[0,8192]
```

- [ ] **Step 2: Verify missing or unstable fields fail**

```bash
./bin/llvm-lit -v ../bishengir/test/Dialect/HIVM/Analysis/ub-model-end-to-end.mlir
```

Expected: FAIL.

- [ ] **Step 3: Emit the fixed schema**

Sort buffers by ID, slots by ordinal, issues by `(kind,message)`, and emit:

```json
{"schema_version":1,"function":"kernel","decision":"SAFE",
 "ub_capacity_bytes":262144,"live_peak_bytes":8192,
 "planned_bytes":8192,"conservative_bytes":8192,
 "buffers":[],"issues":[]}
```

Each buffer must include ID, raw/aligned bytes, alias group, gen/kill, factor, slot IDs, and offsets.

- [ ] **Step 4: Run every new test and commit**

```bash
cd /home/skj/code/AscendNPU-IR/build
ninja bishengir-opt BiShengIRTestDialectHIVM
./bin/llvm-lit -v ../bishengir/test/Dialect/HIVM/Analysis/ub-model-*.mlir
cd ..
git add bishengir/lib/Dialect/HIVM/Analysis/UBModel.cpp \
  bishengir/test/lib/Dialect/HIVM/TestHIVMUBModel.cpp \
  bishengir/test/Dialect/HIVM/Analysis/ub-model-end-to-end.mlir
git commit -m "test(hivm): cover UB impact model end to end"
```

Expected: all five model test files PASS.

---

### Task 7: Regression Gate and Document the Interface

**Files:**
- Create: `docs/source/en/developer_guide/features/UBImpactModel.md`
- Create: `docs/source/zh_cn/developer_guide/features/UBImpactModel.md`
- Modify: `docs/source/en/developer_guide/features/index.rst`
- Modify: `docs/source/zh_cn/developer_guide/features/index.rst`

**Interfaces:**
- Consumes: stable model API and JSON schema 1.
- Produces: documented input checkpoint, supported semantics, conservative policy, and handoff artifacts.

- [ ] **Step 1: Run memory-pass regressions**

```bash
cd /home/skj/code/AscendNPU-IR/build
./bin/llvm-lit -v \
  ../bishengir/test/Dialect/HIVM/mark-multi-buffer.mlir \
  ../bishengir/test/Dialect/HIVM/plan-memory.mlir \
  ../bishengir/test/Dialect/HIVM/enable-multi-buffer.mlir \
  ../bishengir/test/Dialect/HIVM/infer-hivm-mem-scope.mlir \
  ../bishengir/test/Dialect/HIVM/Analysis/ub-model-*.mlir
```

Expected: all PASS. Any failure blocks handoff.

- [ ] **Step 2: Write bilingual documentation**

Both documents must explicitly state:

```text
Input: HIVM after the second InferHIVMMemScope and before/after local MarkMultiBuffer.
Prediction must not consume PlanMemory output.
Only UB address space contributes bytes.
Unknown UB semantics force REVIEW.
Command: bishengir-opt input.mlir -test-hivm-ub-model='ub-capacity-bytes=<bytes>'.
Output: schema-version-1 JSON, one object per function.
```

Include all four decisions and a factor-2 before/after example.

- [ ] **Step 3: Verify formatting and analysis tests**

```bash
cd /home/skj/code/AscendNPU-IR
git diff --check
build/bin/llvm-lit -v bishengir/test/Dialect/HIVM/Analysis
```

Expected: no whitespace errors and all tests PASS.

- [ ] **Step 4: Commit documentation**

```bash
git add docs/source/en/developer_guide/features/UBImpactModel.md \
  docs/source/zh_cn/developer_guide/features/UBImpactModel.md \
  docs/source/en/developer_guide docs/source/zh_cn/developer_guide
git commit -m "docs: explain HIVM UB impact model"
```

- [ ] **Step 5: Produce the handoff pair**

```bash
git rev-parse HEAD
build/bin/bishengir-opt before_mark.mlir \
  -test-hivm-ub-model='ub-capacity-bytes=262144' > before.json
build/bin/bishengir-opt after_mark.mlir \
  -test-hivm-ub-model='ub-capacity-bytes=262144' > after.json
```

Expected: buffer identities match; only real Mark factors, slots, and derived peaks differ.

---

## Deferred Follow-up Plans

This plan ends with a working compiler-side model. After its API stabilizes, create separate plans for:

1. Calibration harness: capture Mark-before/after/PlanMemory, compare offsets and failures, compute recall/false positives/P50/P95, and attribute differences.
2. Real operators and FA: compile Vector operators and FA mix-CV configurations; run safe configurations on Ascend 910 for correctness and performance.
3. Autotuner mapping: map config/TTIR features to the stable model input and add early pruning without changing model semantics.
