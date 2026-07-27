# TTIR UB 多资源保守过滤器设计

日期：2026-07-21

修订：2026-07-24

状态：multi-resource-first 设计基准

本文取代同路径下早期的 singleton-first 版本。实现、测试、profile 审核和 rollout
都应以本文为准。历史汇报和交接材料只能作为证据索引，不能放宽本文的正确性门禁。

## 1. 背景与核心决策

Triton-Ascend autotune 会为同一个 kernel 生成多组 tile 和编译选项。部分配置最终在
HIVM `PlanMemory` 阶段因为 UB 容量不足而失败，但在失败前已经执行大量 lowering。

过滤器在每个 config 的 canonical TTIR 生成后、进入后端 lowering 前，构造该 config 的
`Mandatory UB Resource Graph`（MURG），并沿真实 pipeline 应用版本化
`UBResourceContract`。只有证书证明某个物理 UB execution scope 中必须同时存在的资源
下界已经超过该 scope 的 allocator capacity，才提前拒绝配置。

本方案以多资源证明为主：

- source matcher 提取一个语义 pattern 必然产生的**完整资源集合**，而不是只找最大的 tensor；
- alias、独立 allocation、生命周期和共存时刻都是证明的一部分；
- solver 的主要输入是显式 `CoexistenceWitness`；
- singleton 是合法的退化证书，用于只有一个必需资源或多资源关系尚未证明的情况；
- AIC 和 AIV 是不同物理 UB domain，跨 scope 资源不能相加。

正确性不变量按 execution scope 分别成立：

\[
LB_s(config) \le ActualUBPeak_s(config)
\]

\[
Reject(config) \iff
\exists s.\ Proven(LB_s) \land LB_s(config) > Capacity_s(target)
\]

任何不确定性都返回 `defer`，继续真实编译。

## 2. 目标与非目标

### 2.1 目标

- 在 per-config canonical TTIR 上运行确定性的静态分析。
- 对必需 UB payload、实例数、alias、lifetime、execution scope 建立可审核证明。
- 通过多资源 witness 捕获真实 overflow 的主要来源。
- 用精确 pipeline identity 防止旧合同被错误复用于新 pipeline。
- 用真实 full compiler、PlanMemory 和 exact semantic replay 共同验证下界。
- 支持 `off`、`shadow`、`enforce`，并保持 production profile 默认空。
- 未支持 operation、未知 stage、动态 extent 或合同漂移全部具名 `defer`。

### 2.2 非目标

- 不预测完整峰值，也不替代 `PlanMemory`。
- 不使用概率、回归或置信度作为硬拒绝依据。
- 不把所有 TTIR tensor 都当成物理 UB allocation。
- 不跨 AIC/AIV scope 汇总 UB。
- 不对 Cube L1/L0A/L0B/L0C 做容量规划。
- 不因扩大 matcher 而默认扩大合同适用范围。
- 不为了召回率自动安装未经审核的 production profile。

## 3. 决策语义

分析结果只有两种：

- `reject`：存在有效证书，且证书下界严格大于目标 allocator capacity；
- `defer`：过滤器不作决定，真实编译器继续执行。

不存在 `keep` 证明。`defer` 不表示一定可编译。

边界条件必须是：

```text
LB > capacity   → reject
LB == capacity  → defer
LB < capacity   → defer
```

分析器错误、整数溢出、图冲突、未知 target、profile 未命中都必须 fail-open 为 `defer`。

## 4. 总体架构

```text
Autotune config
    │
    ▼
canonical TTIR
    │
    ├─ mode=off ───────────────────────────────→ 原后端编译
    │
    ▼
PipelineIdentity
    │
    ▼
Strict source matcher
    │  为一个完整语义 pattern 提取必需资源和初始关系
    ▼
MandatoryUBResourceGraph
    │
    ▼
ordered UBResourceContract chain
    │  Preserve / Transform / Invalidate
    ▼
validated MURG
    │
    ▼
scope-aware multi-resource solver
    │
    ├─ 无有效证书 / LB ≤ capacity ─────────────→ defer
    └─ witness LB > capacity
         ├─ shadow ─────────────────────────────→ 记录并继续
         └─ enforce ─→ UBLowerBoundOverflow ────→ autotune 丢弃 config
```

模块职责：

```text
TTIRSourceAnalyzer
  识别完整 SSA 子图并建立候选资源、source facts 和初始关系

MandatoryUBResourceGraph
  保存资源、alias、distinct、lifetime、scope、witness 和 trace

PipelineContractRegistry
  按真实 stage 顺序验证或变换整张图

LowerBoundSolver
  以多资源 witness 为主，singleton 为退化路径，按 scope 求下界

PythonPolicy
  执行 off/shadow/enforce、metadata、debug dump 和异常策略

Oracle
  对照 full compiler、PlanMemory 和 exact semantic replay，生成候选 profile
```

## 5. MURG：多资源证明中间表示

### 5.1 为什么称为“证明图”

图中的节点和边不是对最终编译结果的猜测，而是当前合同链已经证明、且仍然有效的事实：

- 节点证明某个 payload 至少需要多少字节、至少有几个实例；
- alias 边证明资源是否共享存储；
- distinct 边证明资源必须落在不同 allocation；
- lifetime 和 witness 证明哪些资源在同一时刻必须存在；
- execution scope 证明这些资源属于哪个物理 UB domain；
- trace 证明每个事实经过了哪些精确 stage 合同。

图本身不是最终证书。solver 从仍有效的图事实中选择一个可验证子集，生成
`LowerBoundCertificate`。

### 5.2 资源节点

当前资源核心字段为：

```cpp
struct MandatoryUBResource {
  ResourceId id;
  std::string debugName;
  int64_t minPayloadBytes;
  int64_t minInstances;
  OperationOrigin origin;
  UBAddressSpace addressSpace;
  UBExecutionScope executionScope; // SingleCore / AIC / AIV
  MaterializationKind kind;
  ProgramPoint birth;
  ProgramPoint lastRequiredUse;
  SourceFacts sourceFacts;
  ContractTrace trace;
  ValidityState validity;
};
```

数值含义：

\[
bytes(r) = minPayloadBytes(r) \times minInstances(r)
\]

两项都必须是有证明的下界。乘法、加法、stable ID 分配均使用 checked arithmetic。

### 5.3 图关系

- `derives-from`：记录 TTIR value、逻辑资源和后续物理资源之间的来源关系；
- `mayAlias(a,b)`：当前不能证明共享，也不能证明独立；
- `mustAlias(a,b)`：两个逻辑资源确定落在同一 allocation；
- `mustDistinct(a,b)`：两个 alias class 确定落在不同 allocation；
- `CoexistenceWitness(point, resources)`：这些资源在同一个 program point 必须存活。

关系不能互相冲突。同一 pair 同时出现 `mustAlias` 与 `mustDistinct`、跨 execution scope
建立 alias/witness、重复或越界 ID，都会使图 malformed 并 `defer`。

### 5.4 生命周期

`birth` 和 `lastRequiredUse` 是资源的最小生存区间。source matcher 可以先记录保守事实，
后续 materialization/lifetime contract 再确认真实 lowering 中的 program point。

“两个资源都出现在 IR”不等于“必须同时存在”。多资源求和必须有显式 witness。

## 6. Scope-aware 多资源求解

solver 按以下顺序执行：

1. 丢弃 invalid resource，但保留其 trace 用于 defer 诊断。
2. 用 `mustAlias` 构造 alias equivalence class。
3. 每个 alias class 只取成员中的最大有效下界，禁止重复计数 view。
4. 检查 witness 中所有资源有效且属于同一 execution scope。
5. 检查 witness 中不同 alias class 两两具有 `mustDistinct`。
6. 对该 witness 的 alias classes 做 checked sum。
7. 在所有有效 witness 和 singleton 中取最大下界。

多资源 witness：

\[
LB_w =
\sum_{c \in AliasClasses(w)}
\max_{r \in c} bytes(r)
\]

有效条件：

\[
\forall c_i \ne c_j,\ MustDistinct(c_i,c_j)
\]

最终证书：

\[
LB = \max(\max_r bytes(r), \max_w LB_w)
\]

这里的 `max(singleton, witness)` 是安全兜底，不表示方案以 singleton 为主。

### 6.1 跨 scope 示例

Dynamic CV lowering 可能产生：

```text
AIC Fixpipe resource = 512 bytes
AIV vector resource  = 512 bytes
```

两个 core 各有独立 UB，因此正确下界是：

```text
LB_AIC = 512 bytes
LB_AIV = 512 bytes
```

不能生成 `1024 bytes` 的跨核 witness。任何跨 scope alias、distinct 或 coexistence
关系都应 fail closed。

## 7. Source matcher

matcher 的单位是一个受支持的 SSA 语义 pattern，不是一行 TTIR，也不是每个 operation
单独建立一套 identity。

输入是整个 canonical `ModuleOp`。matcher：

1. 先做结构和 dialect preflight；
2. 在完整 SSA use-def graph 中定位受支持 pattern；
3. 验证 operation 集合、shape、类型、属性、region、use 数量和控制流；
4. 为该 pattern 一次性建立完整候选资源集合；
5. 建立 source 阶段可知的 `mayAlias`、lifetime 和 witness；
6. 遇到额外 operation 或动态事实立即返回结构化 reason。

PipelineIdentity 属于整个 config/pipeline，不属于某个 operation。

### 7.1 当前已实现的安全切片

| Pattern | 候选资源 | 关键关系 |
|---|---|---|
| direct copy | GM→UB input | singleton |
| binary add | lhs input、rhs input | 初始 mayAlias；物化后 mustDistinct；同点 witness |
| reshape copy | load、logical view | 物化后 mustAlias；alias class 只计一次 |
| reduction sum | input、scratch、accumulator | suffix 后 pairwise mustDistinct；reduce witness |
| loop-carried add | accumulator、step input | 跨迭代 lifetime；loop witness |
| loop multibuffer factor=2 | accumulator、step input×2 | `minInstances=2`；loop witness |
| Dynamic CV MIX dot→exp | AIC Fixpipe、AIV vector | 分 scope singleton，禁止跨核相加 |

Irregular indirect-add 的真实 boundary 在 gather 前存在动态
`memref.alloc(%dim) : memref<?xf32>`。TTIR 无法证明 source extent，因此必须返回
`unsupported-irregular-source-extent`，不能只计算 index/value 两个小资源。

## 8. UBResourceContract

### 8.1 Transfer function 的含义

合同是一个精确 stage 对 MURG 事实的状态变换：

```text
MURG_before_stage
    │
    ├─ Preserve   → 事实和下界保持
    ├─ Transform  → 按已证明的单调规则更新 payload/instances/relations/lifetime/scope
    ├─ Invalidate → 相关事实失效
    └─ Error      → 整体 defer
    ▼
MURG_after_stage
```

它不是完整模拟 pass，也不是只返回一个经验公式。实现可以是：

- 精确结构检查；
- 有证明的整数传递公式；
- alias/lifetime/scope 关系精化；
- 对 exact semantic replay 结果的 identity-bound 使用；
- 无法证明时主动 Invalidate。

与完整语义重放的区别：

- 语义重放尝试模拟 pass 后的抽象 IR 状态；
- ResourceContract 只传递“拒绝证书所需的最小事实”；
- replay 可以作为某个合同的 oracle 或实现依据，但不能替代 pipeline identity 和
  production profile 审核。

### 8.2 合同必须绑定的信息

- 精确 stage 名称、序号和关键参数；
- operation family 和 source facts；
- target、compile mode、compiler kind/content hash；
- canonical TTIR hash；
- Triton/Ascend/CANN/BiSheng 版本；
- 前置不变量；
- 对资源、alias、distinct、lifetime、scope 的后置保证；
- 失效条件；
- unit test、fixture、PlanMemory 和 replay 证据。

### 8.3 当前合同族

生产 loader 当前认识以下已版本化切片：

- `invalidate-unmodeled-stage@1`
- `direct-copy-preserve@1`
- `direct-copy-max-tiles@1`
- `binary-add-preserve@1`
- `binary-add-max-tiles@1`
- `reshape-copy-preserve@1`
- `reshape-copy-max-tiles@1`
- `reduction-sum-preserve@1`
- `reduction-sum-max-tiles@1`
- `reduction-sum-extra-buffer@1`
- `loop-carried-add-preserve@1`
- `loop-carried-add-max-tiles@1`
- `loop-carried-add-multibuffer@1`
- `dynamic-cv-source-preserve@1`
- `dynamic-cv-replay@1`
- `dynamic-cv-result-preserve@1`
- `ub-alignment@1`（P5 leaf candidate primitive，不允许 standalone profile）
- `<family-contract>+ub-alignment@1`（同一真实 stage 内的有序 composite：
  先执行 family materialization，再逐物理资源 alignment；证书 trace 展开 leaf IDs）

未知 stage 或不匹配参数不得使用 family-wide 默认 Preserve。

PlanMemory 的三种 alignment 语义必须分开：

- payload rounding：`alignedConstBits = AlignUp(constBits, alignUnit)`，属于本方案
  可以建模的逐 allocation 强制下界；
- offset alignment：planner 用 aligned extent 排布/复用 offset，不代表再增加一份
  payload；
- scope reservation：UB capacity 是 planner 的上限检查，不作为额外资源加入 MURG。

## 9. PipelineIdentity

分析器绑定完整 pipeline，而不是为每个 operation 建立 identity：

```text
PipelineIdentity = SHA256(
  normalized pipeline/stage order
  + canonical TTIR SHA256
  + UB-affecting options
  + target arch / compile mode / core kind
  + Triton and Ascend versions
  + selected compiler kind and content hash
  + CANN/BiSheng hash
  + libdevice and effective environment switches
)
```

实际编译与分析器必须共享同一 pipeline builder 和规范化 option 集合。任何 pass
新增、删除、调序、默认值或二进制内容变化都必须改变 identity。profile 未精确命中时
只能 `defer`。

## 10. UB capacity

capacity 必须来自 C++ `getUBCapacityBytes(target)`，Python runtime 使用同一 binding。
未知 target 没有默认值。

当前 target contract：

| Target family | PlanMemory allocator capacity |
|---|---:|
| Ascend910B / 910_93 aliases | 192 KiB |
| Ascend310B1–B4 | 248 KiB |
| Ascend910_95 / Ascend950 aliases | 192 KiB |

910_95/950 的 nominal UB 为 256 KiB，但当前 identity 绑定的 PlanMemory 会保留 64 KiB，
真实拒绝阈值是 192 KiB。早期设计中的 256 KiB 已废止。

## 11. C++、binding 与 Python policy

公开分析结果：

```json
{
  "decision": "reject|defer",
  "lower_bound_bytes": 524288,
  "capacity_bytes": 196608,
  "certificates": [{
    "kind": "witness",
    "resource_ids": [0, 1],
    "bytes": 524288,
    "contract_trace": ["...", "..."]
  }],
  "unsupported_reasons": [],
  "defer_trace": [],
  "pipeline_identity": "...",
  "contract_version": "ttir-ub-lb-v1"
}
```

Python policy 必须二次验证 schema、capacity、identity、certificate 和严格大于关系。
`off` 不调用分析器；`shadow` 记录但不拒绝；`enforce` 只对经过 production profile
验证的 `reject` 抛出可 pickle 的 `UBLowerBoundOverflow`。

完整证书只进入 debug dump。普通 metadata 保持紧凑且 JSON serializable。

## 12. Oracle 与 profile promotion

每个候选 operation family 必须同时具备：

```text
canonical TTIR
  ├─ analyzer → MURG / certificate / LB
  ├─ full compiler → before-CVPipelining boundary
  ├─ suffix compiler → PlanMemory peak/overflow
  └─ exact semantic replay → expected scope/resource state
```

硬门禁：

\[
LB_s(config) \le
\min_{seed=0..19,retry} ActualUBPeak_s(config, seed)
\]

- 每个 analyzer reject 都必须是真实 UB overflow；
- overflow scope 必须是 UB，不能把 L1 或普通编译失败当 UB；
- violations 必须为 0；
- unavailable 必须为 0；
- binary hashes、fixture hashes、identity 和 auto-tile outcome 必须写入报告；
- deliberate-defer case 不得生成 production profile。

oracle 只能输出候选 profile，不能自动编辑 packaged profile。安装需要人工审核。

## 13. Operation family 覆盖策略

“全量 operation”指 coverage matrix 中必须有明确状态，不表示首版全部支持。

| Family | 当前策略 |
|---|---|
| direct memory | 受限 direct-copy candidate |
| elementwise | 受限 binary-add candidate；其他 defer |
| view | 受限 reshape-copy candidate；其他 defer |
| reduction | 受限 f32 sum candidate；其他 defer |
| control flow | 受限 loop-carried-add candidate；其他 defer |
| multibuffer | 受限 factor=2 candidate；其他 defer |
| Dynamic CV/MIX | dot→exp replay-shadow；等待 P4 server gate |
| descriptor / irregular | deliberate defer |
| layout transform | deliberate defer |
| dot general | P5；A/B 通常进入 L1、accumulator 进入 L0C，不能计入 UB；只有真实 lowering 证明的 Fixpipe/后续 AIV UB allocation 才能建资源 |
| atomic | P5；先 defer |
| alignment | P5；需证明 allocator 强制 padding 下界 |
| custom ops | P5；逐 op explicit contract，默认 defer |

扩展一个 family 的最小提交单元是：

1. strict matcher；
2. 完整 MURG resources；
3. alias/distinct/lifetime/scope/witness；
4. 全 stage contract chain；
5. full-compiler fixture；
6. PlanMemory + exact replay oracle；
7. coverage matrix 和 defer mutations。

缺少任一项不得扩大 matcher。

## 14. 性能、rollout 与回退

图构造和求解目标复杂度为 `O(V + E + W·K²)`；`K` 是单个 witness 的资源数，受支持
pattern 必须保持有界。代表性 corpus 的 p95 目标仍为每 module 小于 5 ms。

上线顺序：

1. 默认 `off`；
2. CI 和代表性 workload 开 `shadow`；
3. 持续比较每张多资源证书与真实 per-scope PlanMemory peak；
4. 任一反例立即撤销对应 identity/profile；
5. 零违规稳定后，仅允许用户显式 `enforce`；
6. 最后再评估默认模式，首版不自动切换。

紧急回退设置 `ub_lower_bound_mode=off`。过滤器不修改 TTIR，关闭后恢复旧编译路径。

## 15. 当前完成边界

- P0–P3 的代码切片已按多资源语义实现并有本地测试和历史 CANN oracle 证据；
- P4 Dynamic CV/MIX 和 irregular fail-closed 代码已完成；
- P4 仍缺目标 CANN 环境的 rebuilt binary hashes、seeds 0..19、retry 和 exact replay
  最终门禁；
- packaged production profile 仍为空；
- P5 的 Alignment primitive、同-stage composite、profile/oracle schema 和所有 long-tail
  structured defer 已完成本地实现；general dot/atomic 的正向资源模型仍等待真实 boundary；
- rollout 尚未开始 production promotion。

实施状态和可执行步骤以同目录
`../plans/2026-07-21-ttir-ub-conservative-filter-implementation.md` 为准。
