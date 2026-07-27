# TTIR UB Multi-Resource Conservative Filter Implementation Plan

日期：2026-07-21

修订：2026-07-24

基准分支：`codex/ttir-ub-conservative-filter`

审计基线：`2782c69aabde56f73222ed3c78d2019623250787`

## 0. 计划目标与状态约定

本计划取代旧的 singleton-first 8-task 清单。新计划以多资源 MURG、
`CoexistenceWitness` 和 execution scope 为主；singleton 仅是退化证书。

状态：

- `[x]`：代码、测试和所需证据已完成；
- `[ ]`：尚未完成；
- `[~]`：代码已完成，但缺目标 CANN/PlanMemory 最终门禁；
- `DEFER`：有意不支持，并已有结构化 reason；
- `ROLLOUT`：实现完成但尚未安装 production profile。

正确性门禁：

```text
per-scope lower bound <= per-scope actual PlanMemory peak
reject iff proven lower bound > trusted allocator capacity
unknown / mismatch / error / malformed graph -> defer
```

production profile 必须保持空，除非同一 binary identity 的 seeds `0..19`、retry、
full compiler、PlanMemory 和 exact semantic replay 全部通过且经人工审核。

## 1. 当前总体进度

| 阶段 | 状态 | 完成边界 |
|---|---|---|
| P0 基础链路 | 完成 | MURG、contract registry、identity、policy、autotune、oracle |
| P1 direct memory | 完成切片 | direct-copy，candidate 未安装 |
| P2 elementwise/view | 完成切片 | binary-add、reshape-copy，多资源/alias 证书 |
| P3 reduction/loop/multibuffer | 完成切片 | sum、loop-carried-add、factor=2 |
| P4 Dynamic CV/MIX/irregular | 本地完成，服务器门禁待办 | dot→exp 分 scope；irregular 具名 defer |
| P5 dot/alignment/long-tail | 本地部分完成 | Alignment candidate primitive；general dot/atomic/custom fail-closed |
| Rollout | 未完成 | profile 为空，尚未 CI shadow 和 enforce 灰度 |

## 2. P0：多资源基础设施

### 2.1 MURG 与安全求解

- [x] `MandatoryUBResource` 支持 payload、instances、origin、lifetime、trace。
- [x] 支持 `mayAlias`、`mustAlias`、`mustDistinct`。
- [x] 支持显式 `CoexistenceWitness`。
- [x] `mustAlias` 使用 equivalence class 去重。
- [x] witness 只有 pairwise `mustDistinct` 才允许求和。
- [x] `minPayloadBytes * minInstances` 使用 checked arithmetic。
- [x] stable resource/witness ID overflow fail closed。
- [x] malformed relation、重复 ID、关系冲突 fail closed。
- [x] singleton 作为 witness 不成立时的安全退化证书。
- [x] `UBExecutionScope` 支持 `SingleCore/AIC/AIV`。
- [x] 跨 scope alias/relation/witness fail closed。

验收测试：

```bash
cmake --build .build-ttir-ub --target TestAscendTTIRUBLowerBound -j8
ctest --test-dir .build-ttir-ub \
  -R TestAscendTTIRUBLowerBound --output-on-failure
```

本地限制：当前 worktree 的完整 GTest link 受外部 MLIR/Triton revision mismatch 阻塞；
修改过的 UB C++ objects、GTest object 和 pybind object 已独立编译通过。最终 link/test
放入服务器门禁。

### 2.2 Contract registry 与 capacity

- [x] 精确 profile identity 和 ordered stage binding。
- [x] 每个 stage 必须恰好命中一个 versioned contract。
- [x] 未知 stage 使用 Invalidate，不默认 Preserve。
- [x] Preserve/Transform/Invalidate/InternalError 语义。
- [x] contract 参数漂移和 stage 顺序漂移 fail closed。
- [x] C++ capacity 是单一来源，Python runtime 使用 binding。
- [x] 未知 target 返回 `None/defer`。
- [x] 910_95/950 使用真实 PlanMemory allocator threshold 192 KiB。
- [x] packaged profile 初始并继续保持为空。
- [x] 无 `allow_unvalidated` 或概率拒绝入口。

### 2.3 Python policy、compiler 和 autotune

- [x] `off/shadow/enforce` option 校验，默认 `off`。
- [x] `off` 不构造 identity、不调用 analyzer。
- [x] policy 位于 canonical TTIR 后、未来 lowering 前。
- [x] `UBLowerBoundOverflow` 可 pickle，并穿过 compiler wrapper。
- [x] serial/parallel autotune 捕获资源异常并丢弃对应 config。
- [x] `ContextVar` 传入异步 worker。
- [x] `UBFilterStats` 提供 analyzed/rejected/deferred/passed-to-backend。
- [x] debug dump 保存完整 JSON；普通 metadata 保持紧凑。
- [x] Python 二次验证 capacity、identity、schema 和 `LB > capacity`。

## 3. PipelineIdentity

- [x] 绑定 normalized stage order。
- [x] 绑定 canonical TTIR SHA256。
- [x] 绑定 target、compile mode、core kind。
- [x] 绑定完整 UB-affecting options closed golden。
- [x] 绑定 compiler kind 和 content hash。
- [x] 绑定 CANN/BiSheng、Triton/Ascend、libdevice。
- [x] 绑定会改变 lowering 的 environment switches。
- [x] 任一 forwarded BiSheng flag 未进入 identity 时测试失败。
- [x] caller 不能通过自带 profile 或 identity 启用 rejection。

新增 option/pass 时必须同时：

1. 更新 `UB_AFFECTING_OPTIONS`；
2. 更新 forwarded flags closed golden；
3. 更新 identity 测试；
4. 使旧 profile 不再命中。

## 4. P1：Direct memory 多阶段闭环

### 4.1 已完成

- [x] strict contiguous `load → store` SSA matcher。
- [x] static ranked shape 和 element width checked payload。
- [x] mask、dynamic、descriptor、非连续 pointer、额外 use 具名 defer。
- [x] source resource 保存 elements/bit-width/consumer facts。
- [x] `direct-copy-preserve@1`。
- [x] `direct-copy-max-tiles@1`。
- [x] before-CVPipelining allocation bridge。
- [x] PlanMemory parser、seed/retry runner、candidate generator。
- [x] candidate 只生成不自动安装。

### 4.2 Rollout 状态

- [x] historical CANN seeds 0..19 + retry：zero violation/unavailable。
- [ ] production profile 人工审核与安装。`ROLLOUT`

安装前必须确认 candidate identity 与同一 binaries 的 shadow metadata 完全一致。

## 5. P2：Elementwise、view、alias 与 coexistence

### 5.1 Binary add

- [x] 严格匹配 `load(lhs) + load(rhs) → addf → store`。
- [x] 建立 lhs/rhs 两个 resource。
- [x] source 阶段为 pairwise `mayAlias`。
- [x] 建立同 program point witness。
- [x] materialization contract 验证两个 allocation 后精化为 `mustDistinct`。
- [x] witness certificate 对两个资源求和。
- [x] mayAlias、不同 witness、缺失 distinct 都不能求和。
- [x] full compiler、PlanMemory、semantic replay 历史门禁完成。

### 5.2 Reshape copy

- [x] 严格匹配受限 rank-1→rank-2→rank-1 inverse reshape。
- [x] 建立 load 和 logical view 两个 resource。
- [x] materialization contract 验证单 allocation 后精化为 `mustAlias`。
- [x] alias class 只计一次，禁止 view 重复计数。
- [x] broadcast/expand_dims/bitcast 保持 operation-specific defer。
- [x] full compiler、PlanMemory、semantic replay 历史门禁完成。

### 5.3 Rollout 状态

- [x] binary-add/reshape candidate 可生成。
- [ ] production profile 人工审核与安装。`ROLLOUT`

## 6. P3：Reduction、loop 与 multibuffer

### 6.1 Reduction sum

- [x] 严格匹配一维 contiguous f32 sum reduction。
- [x] 建立 input、scratch、accumulator 三个 resource。
- [x] 建立三资源 reduce-point witness。
- [x] source 阶段不提前声称 physical distinct。
- [x] suffix `extra-buffer` contract 后精化 pairwise `mustDistinct`。
- [x] solver 生成三资源 witness certificate。
- [x] 多 tile、其他 reducer、scan/argmin/argmax 具名 defer。
- [x] full compiler、PlanMemory、semantic replay 历史门禁完成。

### 6.2 Loop-carried add

- [x] 严格匹配固定 trip-count 的单 carried-tensor loop。
- [x] 建立 accumulator 和 loop-step-input。
- [x] accumulator lifetime 跨越 loop。
- [x] 建立 loop program point witness。
- [x] materialization 后精化两个 allocation 为 `mustDistinct`。
- [x] full compiler、PlanMemory、semantic replay 历史门禁完成。

### 6.3 MultiBuffer factor=2

- [x] `loop-carried-add-multibuffer@1` 只接受 exact factor=2。
- [x] step input 的 `minInstances` 从 1 提升为 2。
- [x] `num_stages=1` 不错误启用 multibuffer。
- [x] factor≠2、preload factor=4 和未知策略具名 defer。
- [x] full compiler、PlanMemory、semantic replay 历史门禁完成。

### 6.4 Rollout 状态

- [x] reduction/loop/multibuffer candidate 可生成。
- [ ] production profile 人工审核与安装。`ROLLOUT`

## 7. P4：Dynamic CV / MIX / irregular memory

### 7.1 Dynamic CV MIX dot→exp

- [x] fixture 使用真实 full-compiler canonical TTIR。
- [x] fixture 使用真实 `before-CVPipelining` boundary。
- [x] manifest 绑定两个输入 SHA256 和 full-compiler provenance。
- [x] strict matcher 校验 dot、exp、shape、type、property、use-def 和 operation 集。
- [x] ordered contract chain：
  `dynamic-cv-source-preserve@1`
  → `dynamic-cv-replay@1`
  → `dynamic-cv-result-preserve@1`。
- [x] source 投影为 AIC Fixpipe 和 AIV vector resource。
- [x] AIC/AIV 使用不同 `UBExecutionScope`。
- [x] 禁止跨 scope alias/coexistence 和求和。
- [x] analyzer 得到每 scope 512-byte singleton 下界。
- [x] default memory-space、legacy Fixpipe、runtime capacity 和 same-schema suffix
  compatibility patches 已实现并通过本地 `git apply --check`。
- [x] server 失联前 Dynamic seed 0 full PlanMemory 成功：
  AIC=4096 bits，AIV=4096 bits。

### 7.2 Irregular memory

- [x] 使用真实 indirect-add full-compiler boundary。
- [x] 识别 gather 前动态 `memref.alloc(%dim) : memref<?xf32>`。
- [x] 删除不安全的旧 96-byte irregular contracts/factories/schema。
- [x] 返回 `unsupported-irregular-source-extent`。
- [x] 不生成 certificate，不允许 profile promotion。

### 7.3 目标 CANN 最终门禁

以下任务必须在用户提供的 CANN 容器完成；本地开发阶段明确跳过：

- [~] 在同一环境重建 `libtriton`、suffix compiler、`cvpipeline_ub_model_cpp`：
  `libtriton` 已在 2026-07-27 CANN 9.0.0 环境重建；完整 LLVM/MLIR development
  build 也已成功，但 pinned LLVM 19.1.7 快照缺少官方 `0051` 补丁所需的
  `LinalgExtensions.cpp`，因此不能生成真实 `BiShengIRLinalgDialectExt`。后两个
  组件仍受 source/patch-version incompatibility 阻塞，不能用安装版 BiShengIR 替代。
- [~] 记录 binaries 和 semantic model 的 SHA256：已记录 `libtriton` 和安装版
  BiShengIR hash；suffix/model 因未能构建而无有效 hash。
- [~] 运行完整 C++ GTest 和 pybind/Python focused tests：focused suite 已在 rebuilt
  binary 上 `348 passed`；C++ GTest target 未构建。
- [ ] Dynamic seeds `0..19`（需要 same-schema suffix oracle）。
- [ ] Dynamic retry seed `-1`（需要 same-schema suffix oracle）。
- [ ] 对照 analyzer per-scope LB、full PlanMemory 和 exact semantic replay。
- [ ] 确认 auto-tile outcome、capacity、peak 和 status 一致。
- [ ] 最终报告 `violations=0`、`unavailable=0`。
- [ ] 人工审核是否生成/安装 Dynamic profile。`ROLLOUT`

服务器约束记录：

```text
host: root@192.168.25.217
container: sgl-sky
container code root: /home/sky/code
任何宿主机修改必须先交用户审核
```

2026-07-27 的可审计结果见
`../validation/2026-07-27-cann-9.0.0-server-validation.md`。

## 8. P5：General dot、alignment、atomic 与长尾

P5 必须继续以多资源 witness 为主，不得退回“按 op 查一个 UB 公式”。

### 8.1 General DotMaterializationContract

- [x] 从现有真实 MIX boundary 确认资源边界：A/B 位于 L1、accumulator 位于
  L0C，均不得进入 UB graph；跨核 Fixpipe 输出和后续 vector allocation 才属于 UB。
- [x] plain dot 与 dot_scaled 在正向 boundary 缺失时分别返回稳定
  `unsupported-dot-requires-full-boundary` /
  `unsupported-dot-scaled-requires-full-boundary`。
- [~] 已捕获 plain dot 的真实 full-compiler boundary；Fixpipe/UB physical
  materialization 仍待 same-schema suffix + PlanMemory 证明，保持具名 defer。
- [ ] 只为真实 boundary 中明确位于 UB 的 Fixpipe output、AIV consumer、
  workspace/scratch 建立资源清单。
- [ ] 为每个资源定义 execution scope 和 materialization kind。
- [ ] 严格匹配 dot/dot_scaled 的 shape、layout、precision 和 accumulator 类型。
- [ ] 建立初始 mayAlias、lifetime 和候选 witnesses。
- [ ] 从 real boundary 证明 physical mustAlias/mustDistinct。
- [ ] 为 tiling、SplitMix、Dynamic CV、bufferization 建立 ordered contracts。
- [ ] 验证 A/B/accumulator 是否在同 scope 同时存在；禁止跨核相加。
- [ ] 增加至少一个 supported fixture 和完整 defer mutation matrix。
- [ ] 完成 full compiler + PlanMemory + exact replay seeds/retry。

完成条件：不是“能识别 `tt.dot`”，而是至少一个 dot family 的完整多资源证书闭环。

### 8.2 AlignmentContract

- [x] 从 `PlanMemory.cpp` 确认每个 local allocation 的 `constBits` 会按
  `GetBufferSpaceInfo(scope).alignUnit` 独立向上取整。
- [x] 区分 payload rounding、allocation offset alignment 和 scope reservation：
  `alignedConstBits` 是逐 allocation size rounding；offset 由 aligned extent 排布；
  scope capacity 只是上限，不是额外加入下界的 reservation。
- [x] 新增 graph-level checked payload alignment primitive。
- [x] 新增 `UBAlignmentContract`，精确绑定 stage options、resource count 和
  alignment bytes。
- [x] Python profile loader 与 C++ binding 支持
  `<family-contract>+ub-alignment@1` composite schema，并拒绝 standalone
  alignment profile。
- [x] 新增 `SequentialContract`，在同一个真实 stage 内按
  family materialization → alignment 顺序执行，不伪造逻辑 stage。
- [x] 对 mustAlias class 只计物理 allocation 的最大 aligned payload。
- [x] 对 witness 中 mustDistinct allocations 分别对齐后再求和。
- [x] checked round-up，任何 overflow/未知 alignment fail closed。
- [x] 增加边界测试：差 1 byte、multiple instances、alias view、distinct witness。
- [ ] 证明具体 family 的 alignment 是最小强制增加量，而不是经验 padding。
- [x] 解决 suffix 内 tiling/materialization/alignment 多个 transfer function 的
  composition：一个 profile binding 装载有序 composite，证书 trace 展开为两个
  leaf contract ID。
- [ ] 用 PlanMemory offset/size ledger 验证 seeds/retry。

### 8.3 AtomicMemoryContract

- [x] 审计 `LoadStoreConverter`：atomic lowering 至少分 result-used/result-unused、
  hardware/software atomic、RMW kind、CAS、mask 和 target 支持路径；不能用单一
  `value tensor bytes` 公式覆盖。
- [x] `tt.atomic_rmw` / `tt.atomic_cas` 当前统一返回稳定
  `unsupported-op-atomic`。
- [x] 对合法 TTIR atomic_rmw/atomic_cas source form 增加零证书 defer 测试。
- [ ] 对 atomic_rmw 和 atomic_cas 分开建模。
- [ ] 识别 read-modify-write 输入、结果、compare/value 和临时资源。
- [ ] 建立副作用导致的不可消除事实。
- [ ] 证明输出与输入是 mustAlias、mayAlias 还是 distinct。
- [ ] 建立 atomic program point witness 和 lifetime。
- [x] 未知 memory ordering、scope、mask 或 converter 路径统一具名 defer，
  在分路径建模完成前不产生 atomic certificate。
- [~] 已捕获 RMW 和 CAS 的真实 full-compiler boundary；PlanMemory/replay fixture
  仍待 same-schema suffix oracle，保持 defer。

### 8.4 Custom ops 与 coverage long tail

- [x] `ttascend.*` / `hivm.custom_*` 未注册 custom path 返回稳定
  `unsupported-op-custom`。
- [x] 每个未注册 Ascend custom op 走统一 structured defer；未来逐 op 支持时再增加
  explicit materialization contract。
- [x] descriptor、layout transform、scan、复杂 control flow 保持 defer-first。
- [x] coverage matrix 已为 general dot、atomic、custom ops 增加 P5
  deliberate-defer 状态，并为 Alignment 标记 candidate primitive。
- [x] atomic、custom ops 有稳定 reason mutation 测试；general dot 的 full-boundary
  defer 已进入 coverage test。
- [x] coverage matrix 中每个 operation family 有：
  `candidate-supported | deliberate-defer | source-fact-only`。
- [x] 每个 deliberate-defer 都有稳定 reason，并由 classifier mutation 或
  machine-readable coverage test 覆盖。
- [x] generic unknown op 通过 `unsupported-op` 测试确认不会 silently Preserve。

## 9. Rollout：off → shadow → enforce

### 9.1 已实现的产品能力

- [x] 默认 `off`。
- [x] `shadow` 记录分析结果但继续编译。
- [x] `enforce` 只拒绝 production-valid certificate。
- [x] profile loader 拒绝 identity/schema/option/contract drift。
- [x] telemetry 支持 serial/parallel autotune。
- [x] 回退只需设为 `off`，不修改 TTIR。

### 9.2 待执行 rollout

- [ ] `[SERVER]` P4 server gate 完成。
- [ ] `[HUMAN]` 对 P1–P4 candidates 做人工 code/fixture/identity 审核。
- [ ] `[HUMAN]` 选择第一批最小 production profiles，不一次性全装。
- [ ] `[ROLLOUT]` 在 CI 开启 shadow，保存按 profile 分组的
  violation/unavailable 指标。
- [ ] `[ROLLOUT]` 在代表性 autotune workloads 开启 shadow。
- [ ] `[ROLLOUT]` 记录 analyzed/rejected/deferred/passed-to-backend 和编译耗时收益。
- [ ] `[ROLLOUT]` 连续观察窗口内保持 zero false reject 和 zero lower-bound violation。
- [ ] `[ROLLOUT]` 任一反例立即删除对应 profile identity。
- [ ] `[HUMAN]` 零违规门禁后允许用户显式 `enforce`。
- [x] 首版不改变默认模式。

这些未勾项不是剩余本地编码任务：`SERVER` 需要目标 CANN 环境，`HUMAN` 需要审核和
发布授权，`ROLLOUT` 需要 CI/代表性 workload 的真实观察窗口。production profile 为空时，
提前打开 shadow 不能验证任何 active candidate。

## 10. 本地验证清单

不需要 CANN server 的工作：

```bash
/opt/anaconda3/bin/python3 -m pytest -q \
  --confcutdir=third_party/ascend/unittest/ttir_ub_oracle \
  third_party/ascend/unittest/ttir_ub_oracle/test_oracle.py \
  third_party/ascend/unittest/ttir_ub_oracle/test_fixture_bundle.py \
  third_party/ascend/unittest/ttir_ub_oracle/test_prepare_same_schema_oracle.py \
  third_party/ascend/unittest/ttir_ub_oracle/test_coverage_matrix.py \
  third_party/ascend/unittest/ttir_ub_oracle/test_profile_schema.py

/opt/anaconda3/bin/python3 -m py_compile \
  third_party/ascend/tools/ttir_ub_oracle.py \
  third_party/ascend/tools/ttir_ub_fixture_bundle.py \
  third_party/ascend/backend/ub_lower_bound.py

git apply --check third_party/ascend/tools/patches/0001-feat-port-same-schema-CVPipeline-suffix-oracle.patch
git diff --check
```

已记录的本地结果：

- [x] oracle/fixture/helper/coverage/profile-schema tests：111 passed。
- [x] Python `py_compile`。
- [x] 五个外部源补丁分别通过对应 source tree 的 `git apply --check`。
- [x] UB implementation、GTest 和 pybind 修改对象独立编译。
- [x] HTML/JSON 解析和 `git diff --check`。

## 11. 服务器验证清单

服务器恢复后按一个 coherent build identity 执行：

1. 同步当前 commit 到容器 `/home/sky/code`。
2. 应用并记录 suffix/semantic compatibility patches。
3. 重建所有参与 identity 的 binaries。
4. 记录 SHA256 和版本。
5. 跑 C++ GTest、pybind/Python focused suites。
6. 跑 P4 Dynamic seeds 0..19 + retry。
7. 跑 exact semantic replay。
8. [x] 捕获 P5 plain dot/dot_scaled 和 atomic RMW/CAS 的真实 full-compiler
   boundary（2026-07-27；hash 见 validation record）。
9. 导出 Alignment 的 per-allocation `constBits/alignedConstBits/offset/scope peak` ledger。
10. 对获得正向模型的 P5 family 跑 seeds 0..19 + retry 与 exact replay。
11. 保存 machine-readable report。
12. 只有 `violations=0 && unavailable=0` 才允许生成 candidate。
13. candidate 仍需人工审核，不自动安装。

不得因服务器不可用把这些步骤勾成完成。

## 12. 每个新 family 的 Definition of Done

- [ ] strict matcher 覆盖完整 SSA pattern。
- [ ] MURG 包含所有必需资源，不漏 source/scratch/accumulator/index/mask。
- [ ] 每个资源有 payload、instances、scope、birth、last use。
- [ ] alias/distinct 关系完整且无冲突。
- [ ] 需要求和时有同 scope `CoexistenceWitness`。
- [ ] 全真实 stage 有精确 Preserve/Transform/Invalidate contract。
- [ ] pipeline identity 绑定所有影响 lowering 的输入。
- [ ] supported positive fixture。
- [ ] shape/type/attribute/use-def/stage drift defer mutations。
- [ ] full compiler boundary fixture 和 hashes。
- [ ] PlanMemory seeds 0..19 + retry。
- [ ] exact semantic replay。
- [ ] zero violation / zero unavailable。
- [ ] coverage matrix 更新。
- [ ] profile candidate 生成但不自动安装。

## 13. 完成标准

整个项目只有同时满足以下条件才算完成：

- [ ] P0–P5 的目标切片均达到各自 Definition of Done。
- [x] production profile 为空，或每个 identity 都有 checked-in zero-violation 报告。
- [x] 所有 unsupported path 都有结构化 defer reason。
- [x] multi-resource certificate 不重复计算 mustAlias。
- [x] mayAlias、不同 lifetime、不同 scope 的资源不相加。
- [x] `LB == capacity` 不拒绝。
- [x] serial/parallel autotune 不丢失非 UB 资源错误以外的合法 config。
- [x] full certificate 仅 debug dump；metadata JSON serializable。
- [x] capacity C++/Python 单一来源。
- [x] local focused tests、oracle 和 `git diff --check` 通过。
- [ ] `[SERVER]` server release tests 与真实 oracle gate 通过。
- [ ] rollout 经过 shadow 零违规窗口后才允许显式 enforce。
- [ ] 所有提交包含 `Signed-off-by`。
