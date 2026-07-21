# TTIR UB 保守过滤器设计

日期：2026-07-21

## 1. 背景与决策

Triton-Ascend autotune 会为同一个 kernel 生成多组 tile 和编译选项。部分配置最终在
HIVM `PlanMemory` 阶段因为 UB 容量不足而失败，但在失败前已经执行 TTIR lowering、
CVPipeline、bufferization、PlanMemory 等大量后端工作。

本设计在每个 config 完成 `make_ttir` 规范化之后、进入 TTIR→Linalg/HIVM 之前，计算
UB 峰值的可证明下界。仅当下界已经超过物理 UB 容量时提前拒绝该 config：

\[
LB(config) \le ActualUBPeak(config)
\]

\[
Reject(config) \iff Proven(LB) \land LB(config) > Capacity(target)
\]

这是一条零误杀策略：允许把实际溢出的配置留给真实编译器，但不允许过滤一个真实可
编译的配置。任何不确定性都返回 `defer`，行为等价于继续真实编译。

2026-07-06 的 `ub-overflow-early-pruning-design.md` 中包含残差回归、overflow 概率和
阈值策略。该方案仍可用于排序、诊断和召回率研究，但概率输出不得作为本设计的硬拒绝
依据。本设计取代其硬拒绝合同。

## 2. 目标与非目标

### 2.1 目标

- 在 per-config canonicalized TTIR 上运行轻量、确定性的静态分析。
- 只依据可审核的证明证书提前过滤必然 UB overflow 的配置。
- 在分析不支持、pipeline 漂移或内部异常时 fail-open，即返回 `defer`。
- 把 TTIR 到最终 UB 物化之间的保证显式编码为版本化 `UBResourceContract`。
- 用 `Mandatory UB Resource Graph` 表达必需资源、物化来源、alias、独立性和共存关系。
- 使用真实 TTIR→HIVM→PlanMemory 作为 oracle，持续验证下界不变量。
- 支持 `off`、`shadow` 和 `enforce` 三种运行模式。

### 2.2 非目标

- 首版不预测完整 UB 峰值，也不替代真实 `PlanMemory`。
- 首版不为了提高召回率接受统计意义上的误杀概率。
- 首版不对 Cube 的 L1/L0A/L0B/L0C 做容量规划。
- 首版不覆盖 SIMT、MIX、dynamic shape、reduction、dot、atomic、复杂控制流和动态访问。
- 首版不把 Python AST 估算值或机器学习输出转换为拒绝证书。
- 首版不尝试在 `TileGenerator` 之前过滤；规则稳定后可另行前移。

## 3. 术语与正确性边界

### 3.1 下界与证书

`lower_bound_bytes` 不是预测值，而是证书覆盖的所有真实 lowering 都不能低于的 UB 峰值。
每个数值必须能追溯到：

- 哪个 TTIR operation/value 产生了资源；
- 为什么该资源必然物化在 UB；
- 为什么它不能被 scalarize、消除或进一步切成更小 tile；
- 哪些 pipeline stage 保留或变换了这个结论；
- 使用了哪个 target、编译器版本和 pass 选项合同。

测试和 oracle 用于发现证明实现的错误，但不能把“样本中没有反例”替代成证明。

### 3.2 决策状态

- `reject`：至少存在一张有效证书，且由证书得到的下界超过容量。
- `defer`：没有足以拒绝的证书，包括支持范围之外和证书在中途失效的情况。

不存在 `keep` 证明。`defer` 只表示过滤器不作决定，真实编译器仍是最终裁判。

### 3.3 UB 容量

首版使用目标芯片公开的物理 UB 容量：A2/A3 为 192 KiB，910_95/950 为 256 KiB。
使用物理上限而不是可能更小的 planner 可用量会降低召回率，但不会引入误杀。未知 target
必须 `defer`，不能套用默认容量。

## 4. 总体架构

```text
Autotune config
    ↓
make_ttir: inliner/combine/canonicalize/CSE/LICM/unroll
    ↓ canonicalized TTIR + compile options + pipeline identity
TTIR UB Lower-Bound Analyzer
    ↓
TTIR source rules → Mandatory UB Resource Graph
    ↓
UBResourceContract chain
    ↓
validated MURG → lower-bound certificates
    ├─ off/shadow/defer → ttir_to_linalg → BiShengIR → PlanMemory
    └─ enforce + LB > capacity → UBLowerBoundOverflow → discard config
```

模块分层：

```text
TTIRSourceAnalyzer
  只从 TTIR 建立候选资源和来源关系

MandatoryUBResourceGraph
  保存必需资源及其证明关系，不执行概率估算

PipelineContractRegistry
  按真实 pass 顺序应用 UBResourceContract

LowerBoundSolver
  只从仍有效的 singleton/coexistence witness 求下界

PythonPolicy
  执行 off/shadow/enforce，记录 metadata 或抛出资源异常
```

## 5. Mandatory UB Resource Graph

### 5.1 目的

`Mandatory UB Resource Graph`，简称 MURG，是证明中间表示。它不表示所有 TTIR tensor，
只表示已经证明在受支持 lowering 中必然占用 UB 的资源。没有证明的 tensor 不进入图。

MURG 解决三个常见的错误建模问题：

1. 一个 TTIR tensor 可能被融合、scalarize 或消除，不能看到 shape 就计入 UB。
2. 两个逻辑值可能 alias 同一物理 buffer，不能直接相加。
3. 两个独立 buffer 可能生命周期不重叠，也不能直接相加。

### 5.2 资源节点

每个 `MandatoryUBResource` 至少包含：

```cpp
struct MandatoryUBResource {
  ResourceId id;
  OperationOrigin origin;
  int64_t minPayloadBytes;
  int64_t minInstances;          // 下界，首版固定为 1
  AddressSpace addressSpace;     // 必须为 UB
  MaterializationKind kind;      // 例如 GMToUBLoad
  ProgramPoint birth;
  ProgramPoint lastRequiredUse;
  ContractTrace trace;
  ValidityState validity;
};
```

`minPayloadBytes` 首版不向上补 alignment。若 raw payload 已超过物理容量即可确定拒绝；
alignment 和 multi-buffer 只会增加真实使用量，暂时不计不会破坏下界。

### 5.3 图关系

MURG 包含四类关系：

- `derives-from`：资源从哪个 TTIR value 及后续哪个逻辑资源变换而来，用于 provenance。
- `may-alias` / `must-alias`：描述可能或确定共享存储的资源。
- `must-distinct`：证明两个资源不可能复用同一物理存储。
- `alive-at`：资源必须在某个已证明的 witness program point 存活。

多资源下界不能仅依赖两两 `must-coexist`。三个资源两两存在重叠，不代表三者同时存在。
因此 MURG 使用显式 `CoexistenceWitness`：

```cpp
struct CoexistenceWitness {
  WitnessId id;
  ProgramPoint point;
  SmallVector<ResourceId> resources;
  ContractTrace trace;
};
```

只有 witness 中所有资源都有效、两两 `must-distinct`，才能对它们求和。可能 alias 的资源
在首版不得参与组合证书。

### 5.4 下界求解

单资源证书：

\[
LB_{single} = bytes(r) \times minInstances(r)
\]

共存证书：

\[
LB_{witness} = \sum_{r \in witness} bytes(r) \times minInstances(r)
\]

最终结果：

\[
LB = \max(LB_{single}, LB_{witness})
\]

首版只启用 singleton witness。图结构预留多资源能力，但在 `must-distinct` 和 `alive-at`
合同完成前不能启用求和。

### 5.5 失效语义

资源证书在任一 stage 遇到以下情况即失效：

- pass 可能消除该资源；
- pass 可能把它切成更小 tile，但没有已证明的尺寸传递函数；
- address space 可能不再是 UB；
- alias 或生命周期关系变得未知；
- pipeline identity、target 或选项不在合同范围内。

失效资源可保留在调试 trace 中，但不能进入 `LowerBoundSolver`。

## 6. UBResourceContract

### 6.1 目的

`UBResourceContract` 描述一个 lowering rule 或 pipeline stage 对 MURG 结论的影响。分析器
并不在 TTIR 上复刻所有真实 pass，而是要求每个会影响证书的阶段给出可审核的传递保证。

```cpp
class UBResourceContract {
public:
  virtual StringRef id() const = 0;
  virtual StringRef version() const = 0;
  virtual bool matches(const PipelineStageContext &) const = 0;
  virtual ContractResult apply(MandatoryUBResourceGraph &,
                               const PipelineStageContext &) const = 0;
};
```

`ContractResult` 对每个受影响资源返回：

- `Preserve`：该阶段不会降低资源下界或破坏关系。
- `Transform`：使用已证明的单调传递函数更新最小字节数、实例数或 witness。
- `Invalidate`：无法证明保留，下游不得使用该证书。
- `InternalError`：分析实现异常；整个分析 fail-open 为 `defer`。

### 6.2 合同必须声明的信息

每个合同必须声明：

- 精确 stage/pass 名称及顺序位置；
- 适用的 operation 形态；
- target 和编译器版本范围；
- 相关编译选项和默认值；
- 前置不变量；
- 对 resource、alias、distinct、lifetime 的后置保证；
- 失效条件；
- 对应单元测试和真实 oracle fixture。

“当前实现看起来不会改变 buffer”不是合同。没有代码证据或真实 compiler oracle 支持的路径
必须 `Invalidate`。

### 6.3 Pipeline contract chain

`PipelineContractRegistry` 按真实编译顺序应用合同：

```text
canonicalized TTIR
  → AutoBlockify contract
  → TritonToStructure/Unstructure contract
  → TritonToHIVM/HFusion/Linalg materialization contract
  → DynamicCVPipeline contract
  → BiSheng tiling/canonicalization contract
  → SplitMix/InlineScope/TileAndBindSubBlock contract
  → bufferization/decompose/scope/alignment/multibuffer contract
  → PlanMemory interpretation contract
```

未知 pass 默认视为可能缩小或消除资源，并使相关证书失效。只有合同明确声明“只增加资源、
不缩小现有资源”时，才可以直接保留下界。

### 6.4 Pipeline identity

分析器必须绑定实际 pipeline，而不能维护一份容易漂移的静态 pass 名单。将
`ttir_to_linalg` 的 PassManager 构造重构为共享 builder，同时生成规范化 fingerprint：

```text
PipelineIdentity = hash(
  normalized open-source pass pipeline,
  relevant compile options,
  target arch,
  Triton/Ascend plugin version,
  CANN/BiSheng version hash
)
```

- Python 实际编译和 analyzer 使用同一 builder 的 pipeline 描述。
- fingerprint 未命中审核过的 contract profile 时，运行时 `defer`。
- CI 对 fingerprint 做 golden 检查；pass 新增、删除或调序必须触发审核。

## 7. 首版证明规则

### 7.1 DirectTensorLoadMaterialization

首版只为满足全部条件的 ranked `tt.load` 建立 singleton resource：

- static ranked tensor result；
- pure SIMD/AIV；
- 连续且可静态证明的 GM pointer 访问；
- 无 mask、boundary padding、descriptor、unstructured、discrete 或 deinterleave 路径；
- element type 大小静态已知；
- load 结果存在不可 DCE 的数据使用并最终到达 store；
- 无 reduction、dot、atomic、loop-carried value 和未知 region control flow；
- 所有可能 scalarize、消除或缩小 tile 的 stage 均有匹配合同。

源规则建立逻辑 load 候选。`TritonToLinalgLoadContract` 根据当前 converter 中 ranked load
创建同形状 `memref.alloc` 和 GM→local copy 的行为，证明 `GMToUBLoad` 物化。后续每个
合同继续传递这一结论。

如果 `auto_blockify`、sub-block tiling、CVPipeline 或 UB-saving 路径可能缩小资源，且没有
精确传递函数，则证书失效。首版宁可多数配置 `defer`，也不能拿原始 TTIR tile 大小直接
拒绝。

### 7.2 首版不启用的规则

- 两个输入 buffer 相加：需要 `must-distinct` 和同一 `CoexistenceWitness`。
- 输出 buffer：可能通过 DPS/inplace 复用输入。
- alignment：需要证明最终 planner 的强制对齐下界。
- multi-buffer 倍数：首版 `minInstances=1`。
- reduction accumulator/scratch：lowering 会产生 init/empty，后续仍可能改变。
- MIX/CV buffer：需要 SplitMix 和各 CV pass 的合同。

这些都作为后续独立合同增加，不能以 heuristic 填入 MURG。

## 8. C++ 与 Python 接口

### 8.1 建议目录

```text
third_party/ascend/include/Analysis/TTIRUBLowerBound/
  TTIRUBLowerBound.h
  MandatoryUBResourceGraph.h
  UBResourceContract.h
  PipelineContractRegistry.h

third_party/ascend/lib/Analysis/TTIRUBLowerBound/
  TTIRUBLowerBound.cpp
  MandatoryUBResourceGraph.cpp
  UBResourceContract.cpp
  DirectTensorLoadMaterialization.cpp

third_party/ascend/backend/
  ub_lower_bound.py
  errors.py
```

`include/Analysis` 和 `lib/Analysis` 分别加入 CMake。pybind 初始化放在独立源文件并由
`triton_ascend.cc` 注册：

```python
ascend.analysis.ttir_ub_lower_bound(module, options) -> dict
```

### 8.2 返回结构

```json
{
  "decision": "reject|defer",
  "lower_bound_bytes": 262144,
  "capacity_bytes": 196608,
  "certificates": [
    {
      "kind": "singleton",
      "resource": "gm_to_ub_load",
      "origin": "kernel.ttir.mlir:12:9",
      "bytes": 262144,
      "contract_trace": [
        "ttir-direct-load-v1",
        "triton-to-linalg-load-v1",
        "post-lowering-profile-v1"
      ]
    }
  ],
  "unsupported_reasons": [],
  "pipeline_identity": "...",
  "contract_version": "ttir-ub-lb-v1"
}
```

`lower_bound_bytes` 始终表示有效证书中的最大下界；没有有效证书时为 0，决策必须是
`defer`。调试估算如需保留，必须使用独立字段且不得传入拒绝策略。

## 9. 编译与 Autotune 集成

### 9.1 运行模式

在 `NPUOptions` 中增加 `ub_lower_bound_mode`：

- `off`：默认值，不调用分析器。
- `shadow`：运行分析并记录结果，不拒绝。
- `enforce`：仅对 `decision=reject` 抛出异常。

选项参与 compiler cache key。非法值在 option parsing 阶段报错。

### 9.2 接入位置

`make_ttir()` 完成现有 canonicalization/CSE/LICM/unroll 后调用 Python policy。分析发生在
真实 per-config TTIR 上，因此 config 的 constexpr tile 已经实例化。

这一位置仍需支付 TTIR 前端成本，但可以跳过：

- TTIR→Linalg/HIVM lowering；
- Dynamic CVPipeline；
- BiShengIR 后端 pass；
- bufferization 和 PlanMemory；
- 二进制生成及无效 config benchmark。

### 9.3 异常合同

新增可 pickle 的 `UBLowerBoundOverflow(OutOfResources)`：

- `required` 为证明下界；
- `limit` 为物理 UB 容量；
- 附带 primary certificate、origin、pipeline identity。

核心 compiler stage wrapper 必须原样重新抛出 `OutOfResources`，不能包装成普通
`MLIRCompilationError`。Ascend autotuner 的串行和并行 `_batch_bench` 都捕获该异常并
丢弃对应 config。

若所有 config 均被证明溢出，autotuner 报告 `No valid triton configs`，并展示最小下界
及其证书。不能为了保证至少一个候选而保留已证明溢出的 config。

### 9.4 Metadata 与日志

shadow 或成功通过的 enforce 编译把以下字段写入 metadata：

```text
ub_lower_bound_mode
ub_lower_bound_decision
ub_lower_bound_bytes
ub_capacity_bytes
ub_lower_bound_contract_version
ub_lower_bound_pipeline_identity
ub_lower_bound_certificate_count
ub_lower_bound_unsupported_reasons
```

完整证书只在 debug dump 中写入 `kernel.ttir.ub-lower-bound.json`。普通模式不逐配置输出，
`TRITON_PRINT_AUTOTUNING=1` 时只打印聚合的 analyzed/rejected/deferred/backend-compiled
数量。

分析器异常不传播到用户编译；policy 记录 `internal-error` reason 后返回 `defer`。

## 10. 容量单一来源

`UBResourceContract` 模块提供 `getUBCapacityBytes(target)`，C++ 分析器直接使用。现有
`third_party/ascend/backend/runtime/utils.py` 的 192/256 KiB 判断改为调用同一 pybind
接口，避免 TileGenerator 与过滤器对硬件容量理解不一致。

测试可以直接构造 target contract，不增加可由生产 config 覆盖容量的公开选项，避免
用户把错误容量变成误杀来源。

## 11. 测试设计

### 11.1 C++/MLIR 单元测试

- static direct load 的正向资源和完整 trace；
- `LB < capacity`、`LB == capacity`、`LB > capacity`；
- 192 KiB、256 KiB 和未知 target；
- shape 字节乘法溢出返回 `defer`；
- scalar、masked、dynamic、descriptor、unstructured、discrete、DCE load；
- reduction、dot、loop、atomic 和未知 op；
- 每个 contract 的 Preserve、Transform、Invalidate；
- MURG alias/distinct/witness 求解，尤其验证两两 overlap 不会被错误地三项求和；
- pipeline fingerprint mismatch 使证书失效。

### 11.2 Python/Autotune 测试

- `off` 不调用绑定；
- `shadow` 写 metadata 且继续编译；
- `enforce` 只拒绝证明溢出；
- analyzer error 和未知 target fail-open；
- `UBLowerBoundOverflow` 穿过 compiler wrapper；
- 串行和并行 autotune 都丢弃该 config；
- 所有 config 被拒绝时诊断完整；
- cache hit 不重复分析；
- off/shadow 不改变 best config。

### 11.3 真实 compiler oracle

对每个 TTIR/config 同时运行 lower-bound analyzer 和完整真实编译：

```text
canonicalized TTIR ─→ analyzer ─→ LB/certificate
         └──────────→ TTIR→HIVM→PlanMemory ─→ actual peak/overflow
```

oracle 必须从 `PlanMemory` 或 `--enable-print-memory-allocated-size` 获取 UB 数值。编译失败
时直接解析 stdout/stderr；不能把任意失败当作 UB overflow。

数据矩阵覆盖：

- vector add、elementwise chain、broadcast 和 DCE；
- threshold 附近多档静态 tile；
- FP16/BF16/FP32 和常见整数类型；
- A2/A3、910_95/950；
- multibuffer、`num_stages`、auto blockify、sub-block tiling；
- dynamic CV、MIX、SIMT、reduction、dot、mask 等 defer 路径；
- PlanMemory seed `0..19`。

硬门禁：

\[
LB(config) \le \min_{seed=0..19} ActualUBPeak(config, seed)
\]

- 每个 analyzer `reject` 必须由真实 PlanMemory 证明 overflow。
- 下界违规数必须为 0。
- 真实可编译 config 被拒绝数必须为 0。

## 12. 性能、上线与回退

分析器遍历 operation 和 MURG 边，目标复杂度为 `O(V+E)`。代表性 corpus 的 p95 分析
耗时目标小于 5 ms。

上线顺序：

1. 默认 `off`，完成单元测试、ASan/UBSan 和 oracle 工具。
2. CI 与代表性 autotune workload 开启 `shadow`。
3. 审核所有证书反例；任一反例立即使对应 contract profile 失效。
4. 满足零违规门禁后，允许用户显式开启 `enforce`。
5. 稳定后再评估默认模式，不在首版自动切换。

紧急回退只需设置 `ub_lower_bound_mode=off`。过滤器不会修改 TTIR，因此关闭后恢复现有
编译路径。

## 13. 演进路线

在保持同一证明合同的前提下逐步增加：

1. `TileAndBindSubBlock` 的最小 tile 传递函数；
2. 确定不 alias 且必须共存的双输入 elementwise witness；
3. reduction accumulator/scratch resource；
4. loop-carried resource 和 multi-buffer 的 `minInstances`；
5. SplitMix 后 AIV 投影以及 MIX/CVPipeline 合同；
6. 已稳定规则前移到 AST/TileGenerator，在生成 TTIR 前过滤同一批配置。

每项扩展必须新增 `UBResourceContract`、MURG 证明关系和真实 oracle fixture。不能只增加
一个经验公式或扩大 matcher。

## 14. 已知环境前置条件

设计 worktree 创建时仓库状态干净，但本机基线测试环境不完整：系统 Python 缺少
`pytest`，Anaconda Python 有 `pytest` 但没有安装 `triton`，仓库内也没有已有构建产物。
进入实现和验证阶段前需要按项目安装指南建立可导入当前 worktree `triton` 的构建环境。

该环境限制只影响本地基线验证，不改变本文的正确性合同和验收门禁。
