# Triton-Ascend 纯 Vector Config 的 UB Overflow 编译前预测设计

日期：2026-07-06

## 1. 背景

Triton-Ascend autotune 会为一个 kernel 生成多组 tile 和编译选项配置，然后逐项编译、运行和测量。部分配置会在 HIVM `PlanMemoryPass` 阶段因 UB 不足而编译失败。这类失败配置不会参与最终 benchmark，却消耗首次调优时间。

现有 `TileGenerator` 使用以下近似上界过滤候选：

```text
local_memory_bytes / max_dtype_bytes / min(pointer_count, 3)
```

它没有分析中间值生命周期、对齐、临时 buffer、inplace 复用或 MultiBuffer，因此不能充分预测真实 UB overflow。

MultiBuffer 相关真实编译链是：

```text
HIVM Tensor IR
  → MarkMultiBuffer #1（以 Workspace/Tensor 语义为主）
  → PlanMemory(GLOBAL_WORKSPACE_PLAN)
  → Bufferization
  → HIVM Buffer IR
  → MarkMultiBuffer #2（Local Buffer）
  → PlanMemory(LOCAL_MEM_PLAN)
```

两次 `MarkMultiBufferPass` 的输入都包含 HIVM op，但语义阶段不同：第一次以 tensor SSA 为主，第二次已物化为 memref buffer。本设计关注纯 Vector 算子的 UB，因此真实监督目标是第二次 Mark 后、Local PlanMemory 得到的 UB 结果。

## 2. 目标与非目标

### 2.1 目标

- 在候选进入编译前预测纯 Vector config 的 UB 使用量和 overflow 风险。
- 提前剪掉高置信度 overflow 配置，减少无效编译。
- 尽量保留真实可编译配置，允许少量漏判交给真实 PlanMemory 兜底。
- 显式建模 MultiBuffer 对 UB 的关键影响，但不在 Python 侧复刻完整 Mark Pass。
- 使用真实 PlanMemory 样本离线校准白盒估算误差。
- 在线推理足够轻量，可用于每个 autotune config。

### 2.2 非目标

- 第一版不覆盖 Cube 的 L1、L0A、L0B、L0C。
- 第一版不覆盖 CV Mix 和 GM Workspace。
- 不复刻 HIVM Bufferization、精确 liveness 或 PlanMemory 地址搜索。
- 不替代真实编译器的资源合法性检查。
- 不自动修改 config；模型只做 KEEP、REJECT 或 ABSTAIN 决策。
- 不对候选做性能排序；现有 `perf_model/top_k` 继续承担性能排序职责。

## 3. 方案选择

评估过三条路线：

1. 在 Python 侧完整复刻 HIVM、MarkMultiBuffer 和 PlanMemory。精度上限高，但等价于维护第二套编译器，漂移风险不可接受。
2. 直接用 AST/config 特征训练黑盒模型。在线简单，但严重依赖样本覆盖，对新 kernel 和编译器版本的泛化较弱。
3. 语义摘要 + 白盒基线 + 真实样本校准。

采用方案 3。它将可稳定观察的语义写入解析公式，将 Bufferization、临时 buffer、复用等不可观察差异交给离线残差校准。

## 4. 总体架构

### 4.1 在线路径

```text
Kernel AST + shape/dtype + config + device UB
  → SemanticSummaryBuilder（每个 kernel 一次，结果缓存）
  → FeatureEvaluator（每个 config 一次）
  → WhiteboxUBEstimator
  → CalibratedUBPredictor
  → UBPrunePolicy
  → KEEP / REJECT / ABSTAIN
```

### 4.2 离线路径

```text
kernel/shape/dtype/config 采样
  → 真实编译
  → 第二次 MarkMultiBuffer + LOCAL PlanMemory
  → 采集 actual UB bytes / overflow label
  → 与在线特征合并
  → 训练残差回归器和 overflow 概率校准器
  → 产出版本化模型包
```

真实编译是教师，不进入在线剪枝时延。

## 5. 组件设计

### 5.1 SemanticSummaryBuilder

输入是 `JITFunction.parse()` 返回的 kernel Python AST。输出是与具体 config 数值解耦的 `KernelSemanticSummary`。

需要识别：

- `tl.arange` 对应的 tile shape 表达式；
- `tl.load`、`tl.store` 和它们所在的循环层次；
- elementwise、cast、broadcast、where 和 reduction 操作；
- 逻辑值的定义点、使用点和最后使用点；
- dtype 传播关系；
- `for`、`if` 等控制流特征；
- split/tiling 参数与轴的映射；
- 无法稳定分析的动态或 alias 模式。

建议的数据接口：

```python
KernelSemanticSummary(
    values: list[ValueSummary],
    operations: OperationCounts,
    loops: list[LoopSummary],
    tile_axes: dict[str, SymbolicExpr],
    split_axes: dict[str, SymbolicExpr],
    unsupported_reasons: list[str],
    schema_version: str,
)
```

`ValueSummary` 至少包含：

```python
ValueSummary(
    name: str,
    role: str,              # load/result/temp/mask/reduction-temp
    shape: list[SymbolicExpr],
    dtype: DTypeExpr,
    birth: ProgramPoint,
    last_use: ProgramPoint,
    loop_depth: int,
    multibuffer_candidate: bool,
)
```

摘要按 kernel 源码/cache key 和 schema version 缓存。

### 5.2 FeatureEvaluator

对每个 config，把运行时 shape、dtype、`XBLOCK/XBLOCK_SUB` 等 meta 参数代入符号表达式，得到数值特征：

- 每个逻辑值的 tile 元素数和原始字节数；
- 32-byte 对齐后的字节数；
- 同时存活逻辑值的基础峰值；
- load/store/elementwise/reduction 数量；
- 循环层数和可求值的 trip count；
- multibuffer 开关及预测候选数；
- `UB_base / UB_capacity` 比值；
- unsupported 和置信度特征。

表达式无法求值时不抛出硬错误，而是生成 ABSTAIN 原因。

### 5.3 MultiBuffer 摘要模型

不复刻 `MarkMultiBufferPass` 的 MLIR Greedy Rewrite。仅保留与纯 Vector UB 相关的稳定语义：

```text
multibuffer=False
  → factor = 1

multibuffer=True
且 load/store 对应逻辑 Local Buffer 位于可流水循环
且可证明 trip count > 1
  → predicted factor = 2

preload、复杂控制流、alias 无法追踪
  → 不强行决定；写入低置信度特征，必要时 ABSTAIN
```

MultiBuffer 因子按逻辑 buffer 应用，不能简单把整个 kernel UB 乘 2。模型需区分 load 输入、store 输出、中间值和 mask。第一版不建模 Workspace MultiBuffer 和 preload 固定四缓冲场景。

### 5.4 WhiteboxUBEstimator

基础估算使用逻辑值峰值存活工作集：

\[
UB_{base} = \max_t \sum_{v \in Live(t)}
align_{32}(bytes(v)) \times mbFactor(v) + TempReserve
\]

其中：

- `Live(t)` 来自 AST 语句级近似生命周期，而非 MLIR 精确 liveness；
- `bytes(v)` 是 tile 元素数乘 dtype 字节数；
- `mbFactor(v)` 是 MultiBuffer 摘要预测；
- `TempReserve` 由 mask、规约和操作复杂度等特征构造，初始可为解析规则，后续由残差模型校准。

该结果必须可解释，至少能拆分为：

```text
input bytes
output bytes
intermediate bytes
mask/reduction reserve
alignment overhead
multibuffer increment
```

### 5.5 CalibratedUBPredictor

校准器包含两个输出头：

1. 残差回归：预测 `actual_ub_bytes - UB_base`；
2. overflow 分类：预测 `P(overflow)`。

第一版采用 CPU 推理快、可解释、易版本化的正则化模型或树模型，不使用神经网络。具体算法通过验证集比较确定，设计不绑定某一个库。

输出接口：

```python
UBPrediction(
    base_bytes: int,
    calibrated_bytes: int | None,
    overflow_probability: float | None,
    confidence: float,
    contributions: dict[str, float],
    reason_codes: list[str],
)
```

### 5.6 UBPrunePolicy

决策为三态：

- `KEEP`：风险低，进入真实编译；
- `REJECT`：高置信度 overflow，提前剪枝；
- `ABSTAIN`：不支持或不确定；在线行为等价于 KEEP。

阈值由验证集选择，以 Safe Retention 优先，不在实现中写死。若一次 autotune 的所有候选均被判为 REJECT，必须保留风险最低的一个候选交给真实编译验证。

## 6. Autotuner 集成

在线顺序：

```text
TileGenerator 生成 config
  → SIMD multibuffer True/False 扩展
  → UB Early Pruner
  → 通用 prune_configs
  → perf_model/top_k（若用户配置）
  → 真实编译和 benchmark
```

UB 模型必须位于 multibuffer 配置扩展之后，才能看到当前候选的实际开关。

UB 模型不复用 `perf_model` 语义。`perf_model` 负责性能排名；UB Early Pruner 负责资源风险。两者可以独立启停、组合运行。

建议将 UB Early Pruner 设计为 Ascend backend 自有的可选剪枝阶段，而不是要求用户为每个 autotune 装饰器提供 `early_config_prune` 回调。模型关闭时，行为必须与当前实现一致。

## 7. 标签采集与模型版本

### 7.1 样本维度

采样矩阵至少覆盖：

- 纯 Vector kernel 家族；
- 多档 shape 和 tile 大小；
- FP16/BF16/FP32 及常用整数 dtype；
- elementwise、broadcast、reduction、mask 和多级循环；
- multibuffer 开关；
- 支持的 Ascend 架构和 UB 容量。

### 7.2 标签字段

```text
kernel family / kernel hash
shape / dtype / config
hardware arch / UB capacity
compiler commit and option hash
semantic summary schema version
online numeric features
whitebox UB base
actual PlanMemory UB bytes（若可得）
overflow label
overflow required/limit（若可得）
```

成功样本训练字节残差和分类；无法取得精确 required bytes 的失败样本只参与 overflow 分类。

### 7.3 数据切分

训练、验证和测试必须按 kernel 家族分组。同一 kernel 的所有 shape/config 留在同一 split，防止相邻配置泄漏导致虚高指标。测试集必须包含完全未见的 kernel 家族。

### 7.4 模型包版本

模型包必须携带：

```text
feature schema version
compiler commit/compatibility id
hardware arch
UB capacity
training data version
model parameters
decision thresholds
```

模型缺失或版本不兼容时 fail-open。

## 8. 错误处理与可观测性

所有在线异常均不得阻断编译：

- AST 不支持或符号表达式无法求值：ABSTAIN；
- 模型文件缺失、损坏或版本不匹配：使用白盒结果作诊断，但不做高风险硬剪枝；
- 推理异常：记录原因并 KEEP；
- 所有候选被拒绝：保留风险最低者；
- 模型判定 KEEP 但真实 PlanMemory overflow：沿用现有编译失败过滤路径，并记录 false negative 样本。

调试输出应包含：

```text
config identity
base bytes and breakdown
predicted multibuffer factors
calibration residual
predicted bytes / capacity ratio
overflow probability / confidence
decision and reason codes
```

## 9. 测试与验收

### 9.1 单元测试

- AST 中 `tl.arange/load/store/for/reduce` 的摘要；
- dtype 和 shape 表达式求值；
- 语句级 birth/last-use；
- 32-byte 对齐；
- MultiBuffer factor 的正例、反例和 ABSTAIN；
- 模型缺失、版本不匹配和异常降级；
- 全部 REJECT 时保留最低风险 config。

### 9.2 Golden 测试

使用固定 kernel/shape/config 集合，对比：

- SemanticSummary；
- 白盒 UB breakdown；
- 第二次 Mark 的 Local Buffer 标记；
- PlanMemory 实际 UB bytes/overflow。

### 9.3 泛化测试

测试集按未见 kernel 家族评估，主要指标：

1. Safe Retention：真实可编译 config 的保留率；
2. Best-config Retention：未剪掉原始 autotune 最快 config 的比例；
3. Overflow Recall：真实 overflow config 的提前剪除率；
4. UB 字节预测 P50/P90 相对误差；
5. ABSTAIN 比例。

具体数值门槛由首轮数据分布和无模型基线确定，再固化为发布门槛；本设计不凭空指定未经数据验证的百分比。

### 9.4 端到端测试

- 模型关闭时与当前 autotune 行为一致；
- 模型开启后每个 key 至少有一个 config 进入编译；
- 比较无效编译数量和首次 autotune 总时延；
- 检查最终选择性能相对未剪枝基线没有不可接受退化；
- 在线每 config 推理时延满足 autotune 开销目标。

## 10. 分阶段交付

### 阶段 1：可解释白盒原型

- AST SemanticSummary；
- config 数值求值；
- 峰值工作集与 MultiBuffer 摘要；
- 只输出诊断，不剪枝；
- 与 PlanMemory 建立对照数据集。

### 阶段 2：离线校准

- 标签采集工具；
- kernel-family 数据切分；
- 残差回归和 overflow 分类；
- 模型版本与评估报告。

### 阶段 3：Shadow Mode

- 在线预测但不改变 config 集合；
- 记录预测与真实编译结果；
- 确认 Safe Retention、Best-config Retention 和运行稳定性。

### 阶段 4：受控剪枝

- 默认关闭或实验开关启用；
- 高置信度 REJECT；
- fail-open 和至少保留一个 config；
- 根据真实使用数据调整阈值。

## 11. 关键设计决策

- 在编译前运行，因此必须使用 AST 语义摘要，不能依赖已经生成的 HIVM IR。
- 对 MarkMultiBuffer 只建模纯 Vector Local Buffer 的关键效应，不逐行复刻 Pass。
- 用峰值存活工作集而不是简单 pointer 数或所有值总和。
- 用真实 PlanMemory 结果校准 AST 不可见的 Bufferization、临时 buffer 和复用误差。
- 选择 fail-open，因为目标允许少量漏判但要求尽量保留可用 config。
- UB 合法性剪枝与性能 `perf_model` 保持独立。

