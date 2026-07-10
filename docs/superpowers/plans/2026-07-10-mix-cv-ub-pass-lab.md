# Mix-CV UB Pass Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a reproducible mix-CV HIVM pass lab with before/after MLIR snapshots and a beginner-oriented UB-modeling guide.

**Architecture:** A minimal `Vector → Cube → Vector` input is kept under `docs/mix_cv_ub_pass_lab/`. Each `bishengir-opt` invocation consumes one numbered snapshot and produces the next. Focused upstream HIVM tests supplement the main sample only when it does not trigger a multi-buffer mechanism.

**Tech Stack:** MLIR/HIVM, `/Users/sky/Code/AscendNPU-IR/build/bin/bishengir-opt`, Markdown.

## Global Constraints

- Preserve a Vector → Cube → Vector data dependency.
- Every pass claim must cite an existing adjacent before/after snapshot.
- Every pass card explains trigger, internal mechanism, MLIR rewrite, and UB effect.
- GM workspace and Cube L1/L0 allocations are not counted as UB bytes.
- `MarkMultiBuffer` marks candidates; `PlanMemory` and `EnableMultiBuffer` materialize slots and rotation.

---

### Task 1: Establish primary mix-CV input

**Files:**

- Create: `docs/mix_cv_ub_pass_lab/00_source_linalg_mix_cv.mlir`
- Create: `docs/mix_cv_ub_pass_lab/01_hfusion_mix_cv.mlir`
- Create: `docs/mix_cv_ub_pass_lab/README.md`
- Source: `/Users/sky/Code/AscendNPU-IR/bishengir/test/Dialect/HFusion/OpFusion/test_mix_cv.mlir`

- [ ] Extract `testChain`, preserving `elemwise_unary → elemwise_unary → matmul → elemwise_unary`.
- [ ] Run and save HFusion:

```bash
/Users/sky/Code/AscendNPU-IR/build/bin/bishengir-opt --test-assign-fusion-kind --fusion-kind=MIX_CV -hfusion-fuse-ops='max-horizontal-fusion-size=-1' docs/mix_cv_ub_pass_lab/00_source_linalg_mix_cv.mlir -o docs/mix_cv_ub_pass_lab/01_hfusion_mix_cv.mlir
```

Expected: `matmul`, `elemwise_unary`, and MIX_CV evidence remain present.

- [ ] In README, record compiler path, every snapshot command, and the rule that failed passes are documented, not replaced by synthetic IR.
- [ ] Verify with `rg -n 'matmul|elemwise_unary|MIX_CV'` over snapshots 00 and 01.

### Task 2: Capture Tensor-level mix-CV passes and checkpoint A

**Files:**

- Create: `docs/mix_cv_ub_pass_lab/02_hivm_tensor.mlir`
- Create: `docs/mix_cv_ub_pass_lab/03_insert_load_store_for_mix_cv.mlir`
- Create: `docs/mix_cv_ub_pass_lab/04_insert_workspace_for_mix_cv.mlir`
- Create: `docs/mix_cv_ub_pass_lab/05_before_mark_multibuffer_1.mlir`
- Create: `docs/mix_cv_ub_pass_lab/06_mark_multibuffer_1.mlir`
- Create: `docs/mix_cv_ub_pass_lab/07_cv_pipelining.mlir`

- [ ] Convert Task 1 output with the smallest accepted ConvertToHIVM sequence and save `02_hivm_tensor.mlir`.
- [ ] Run:

```bash
/Users/sky/Code/AscendNPU-IR/build/bin/bishengir-opt docs/mix_cv_ub_pass_lab/02_hivm_tensor.mlir -hivm-insert-load-store-for-mix-cv -o docs/mix_cv_ub_pass_lab/03_insert_load_store_for_mix_cv.mlir
/Users/sky/Code/AscendNPU-IR/build/bin/bishengir-opt docs/mix_cv_ub_pass_lab/03_insert_load_store_for_mix_cv.mlir -insert-workspace-for-mix-cv -o docs/mix_cv_ub_pass_lab/04_insert_workspace_for_mix_cv.mlir
```

Expected: explicit bridge/workspace evidence. If not triggered, retain the valid primary output and separately save the upstream dedicated test result as a focused witness.

- [ ] Copy valid stage 04 to stage 05, then run:

```bash
/Users/sky/Code/AscendNPU-IR/build/bin/bishengir-opt docs/mix_cv_ub_pass_lab/05_before_mark_multibuffer_1.mlir -hivm-mark-multi-buffer='enable-auto=true' -o docs/mix_cv_ub_pass_lab/06_mark_multibuffer_1.mlir
/Users/sky/Code/AscendNPU-IR/build/bin/bishengir-opt docs/mix_cv_ub_pass_lab/06_mark_multibuffer_1.mlir -cv-pipelining -o docs/mix_cv_ub_pass_lab/07_cv_pipelining.mlir
```

Expected: marker/pipeline rewrites only where the input satisfies pass predicates.

- [ ] Diff 02→03, 03→04, 05→06, and 06→07; collect witnesses for `load|store|workspace|annotation.mark|hivm.multi_buffer|scf.for|scope`.

### Task 3: Capture Buffer-level checkpoint B and multi-buffer materialization

**Files:**

- Create: `docs/mix_cv_ub_pass_lab/08_bufferized.mlir`
- Create: `docs/mix_cv_ub_pass_lab/09_before_mark_multibuffer_2.mlir`
- Create: `docs/mix_cv_ub_pass_lab/10_mark_multibuffer_2.mlir`
- Create: `docs/mix_cv_ub_pass_lab/11_plan_memory_local.mlir`
- Create: `docs/mix_cv_ub_pass_lab/12_enable_multibuffer.mlir`
- Create: `docs/mix_cv_ub_pass_lab/aux_mark_multibuffer.mlir`
- Create: `docs/mix_cv_ub_pass_lab/aux_plan_memory_multibuffer.mlir`
- Create: `docs/mix_cv_ub_pass_lab/aux_enable_multibuffer.mlir`

- [ ] Run one-shot bufferization, canonicalization and ConvertToHIVM operations configured in `HIVMPipelines.cpp`; save stage 08.
- [ ] Run `-hivm-infer-mem-scope` and accepted size inference/set-size stages; save stage 09.
- [ ] Run:

```bash
/Users/sky/Code/AscendNPU-IR/build/bin/bishengir-opt docs/mix_cv_ub_pass_lab/09_before_mark_multibuffer_2.mlir -hivm-mark-multi-buffer='enable-auto=true limit-auto-multi-buffer-only-for-local-buffer=true' -o docs/mix_cv_ub_pass_lab/10_mark_multibuffer_2.mlir
/Users/sky/Code/AscendNPU-IR/build/bin/bishengir-opt docs/mix_cv_ub_pass_lab/10_mark_multibuffer_2.mlir -hivm-plan-memory -o docs/mix_cv_ub_pass_lab/11_plan_memory_local.mlir
/Users/sky/Code/AscendNPU-IR/build/bin/bishengir-opt docs/mix_cv_ub_pass_lab/11_plan_memory_local.mlir -hivm-enable-multi-buffer -o docs/mix_cv_ub_pass_lab/12_enable_multibuffer.mlir
```

Expected: alloc-to-pointer-cast planning and, when eligible, loop-derived slot selection.

- [ ] If primary snapshots lack a trigger, run the upstream `mark-multi-buffer.mlir`, `plan-memory.mlir`, and `enable-multi-buffer.mlir` fixtures one pass at a time and save `aux_*.mlir`, explicitly labeled “focused mechanism witness”.
- [ ] Verify with `rg -n 'address_space<ub>|memref.alloc|pointer_cast|hivm.multi_buffer|affine.apply|arith.select' docs/mix_cv_ub_pass_lab/*.mlir`.

### Task 4: Write and validate the beginner guide

**Files:**

- Create: `docs/mix_cv_ub_pass_lab.md`
- Modify: `docs/mix_cv_ub_pass_lab/README.md`

- [ ] Define tensor, SSA, memref, GM, UB, Cube, Vector, workspace, scope, loop, address space, bufferization, and multi-buffer before any raw MLIR.
- [ ] For `InsertLoadStoreForMixCV`, `InsertWorkspaceForMixCV`, Mark #1, `CVPipelining`, bufferization, `InferHIVMMemScope`/size, Mark #2, `PlanMemory`, and `EnableMultiBuffer`, show trigger, implementation entry point, before snippet, after snippet, actual diff, mechanism, and UB impact.
- [ ] Mark each auxiliary sample as distinct from the primary mix-CV compilation output.
- [ ] Derive:

```text
UB_peak(t) = Σ bytes(buffer) × multi_buffer_factor(buffer)
             for each UB buffer live at program point t
```

Separate checkpoint A (cause/context), B (actual UB candidates), Mark (factor), PlanMemory (layout truth), and EnableMultiBuffer (runtime rotation).

- [ ] Validate all snapshot references with `rg`, and parse every final snapshot with `bishengir-opt` before claiming it is valid.
- [ ] Commit only the lab, guide, and this plan after verification.
