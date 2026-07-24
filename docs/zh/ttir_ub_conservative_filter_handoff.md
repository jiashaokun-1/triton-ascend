# TTIR UB 保守过滤器开发交接

## 1. 文档目的

本文面向后续开发、评审和验证人员，说明 TTIR UB 保守过滤器的业务目标、整体方案、
核心数据结构、端到端代码流程、当前实现范围、验证状态、已知边界以及下一阶段工作。

当前开发基线：

- 仓库：`triton-ascend`
- 分支：`codex/ttir-ub-conservative-filter`
- 本次交接基线提交：`fcb3ffa98240e02569eb947ef8e9cd2e0a2084e0`
- 设计文档：`docs/superpowers/specs/2026-07-21-ttir-ub-conservative-filter-design.md`
- 实施计划：`docs/superpowers/plans/2026-07-21-ttir-ub-conservative-filter-implementation.md`
- 用户说明：`docs/zh/ttir_ub_conservative_filter.md`

## 2. 背景与目标

### 2.1 要解决的问题

autotune 会产生多组 tile、block、multi-buffer、pipeline depth 等配置。传统流程需要把
每个配置继续编译到 HIVM，并运行 PlanMemory 后，才能知道 UB 是否溢出。对于明显超过
UB 容量的配置，这部分完整编译成本是浪费。

本功能希望在 canonical TTIR 阶段建立一个计算成本较低的 UB 下界：

```text
若能够证明 MandatoryUB(config) > UBCapacity(target)
    则提前过滤该 config
否则
    继续走原有完整编译流程
```

它不是预测模型，也不尝试提前算出完整 UB peak。它只证明“至少需要多少 UB”。

### 2.2 正确性方向

过滤器采用单向、保守判定：

- `reject`：已经形成可检查的证明，说明必需 UB 下界严格大于容量；
- `defer`：当前规则不能证明，应继续真实编译；
- 不提供“fit”结论；
- `lower_bound == capacity` 不能过滤，只有 `lower_bound > capacity` 才能过滤；
- 未支持的 operation、shape、target、pipeline 或 contract 一律 `defer`。

因此正确性目标是避免错删真实可编译配置。漏掉一部分可过滤配置只影响收益，不影响
正确性。

### 2.3 为什么称为早期下界模型

这里的“模型”不是机器学习模型，而是对后续编译行为的解析规则和资源传递合同：

```text
TTIR 语义
  → 提取必然产生的 UB 资源
  → 按真实 lowering 阶段传递这些资源
  → 求一个有证明的 UB 下界
  → 与硬件容量比较
```

真实编译计算完整 buffer、alias、生存期、offset 和 peak；本模型只保留足以证明下界的
信息，所以执行成本应显著低于完整编译。

## 3. 整体架构

### 3.1 端到端流程

```text
Triton config
    │
    ▼
canonical TTIR
    │
    ├─ mode=off ───────────────────────────────→ 原有后端编译
    │
    ▼
构造精确 PipelineIdentity
    │
    ▼
严格 TTIR matcher
    │
    ▼
Mandatory UB Resource Graph（MURG）
    │
    ▼
逐阶段应用 UBResourceContract
    │
    ▼
求 LowerBoundCertificate
    │
    ▼
与 target UB capacity 比较
    │
    ├─ defer ──────────────────────────────────→ 原有后端编译
    ├─ reject + shadow ─────────────────────────→ 记录结果并继续编译
    └─ reject + enforce ─→ UBLowerBoundOverflow → autotune 丢弃该 config
```

### 3.2 四个核心层次

1. **语义识别层**：只识别能够严格证明的 TTIR 形态。
2. **资源图层**：用 MURG 表示必需 UB 资源及资源间关系。
3. **pipeline 合同层**：用 `UBResourceContract` 描述真实 pass 对资源下界的影响。
4. **策略与 autotune 层**：规范化结果、记录 telemetry，并在 enforce 模式过滤配置。

### 3.3 设计原则

- 证明与具体 SSA 名称解耦，资源使用稳定整数 ID。
- pipeline identity 和 contract version 是证书的一部分。
- matcher 先完整检查 applicability，再产生资源。
- 任何整数溢出、结构不完整或身份不匹配都使本次分析 `defer`。
- Python 不信任 binding 返回值，会再次检查 schema、类型、容量和证书一致性。
- 生产 profile 只能由真实 PlanMemory oracle 验证后人工加入，工具不自动修改 profile。

## 4. Mandatory UB Resource Graph（MURG）

### 4.1 作用

MURG 是早期模型的中间表示。它不保存完整 TTIR，也不直接模拟 PlanMemory，而是保存
形成下界证明所需的最小信息。

主要入口：

- `third_party/ascend/include/Analysis/TTIRUBLowerBound/MandatoryUBResourceGraph.h`
- `third_party/ascend/lib/Analysis/TTIRUBLowerBound/MandatoryUBResourceGraph.cpp`

### 4.2 MandatoryUBResource

每个资源包含：

- `minPayloadBytes`：单个实例必需的最小 payload；
- `minInstances`：必需实例数，为 multi-buffer 扩展预留；
- `addressSpace`：当前只建模 UB；
- `kind`：当前支持 `GMToUBLoad`；
- `birth` / `lastRequiredUse`：为后续生存期与共存证明预留；
- `validity` / `invalidReason`：合同无法继续证明时失效；
- `contractTrace`：资源由哪些 matcher/contract 得到；
- `origin` / `debugName`：诊断信息。

### 4.3 资源关系

MURG 已定义：

- `mayAlias`：可能共享物理存储，不能简单求和；
- `mustAlias`：确定是同一物理资源；
- `mustDistinct`：确定不能共享；
- `CoexistenceWitness`：证明一组资源必须同时存在，可用于求和下界。

当前生产分析只调用 `solveSingletonLowerBound()`，即取最大单资源下界。图层还实现了
`solveWitnessLowerBound()`，但尚未接入生产决策。后续支持双输入 elementwise 或明确
共存 scratch 时，才能在充分证明 distinct 和 overlap 后使用 witness。

### 4.4 算术和 ID 约束

- 大小使用 `int64_t`，乘法和累加需要显式检查溢出；
- `ResourceId` 和 `WitnessId` 使用稳定的 `uint32_t`；
- ID 耗尽、关系引用非法 ID、资源状态异常都会使图不可用于证明；
- graph API 只能把资源下界降低到已证明的新下界，不能凭估计提高数值。

## 5. UBResourceContract

### 5.1 为什么需要 contract

从 TTIR 看到的 tensor 不一定以原尺寸进入最终 UB。例如 tiling 可能把一个大 tensor
拆成小 tile，canonicalization 可能消除资源，multi-buffer 可能增加实例数。只在 TTIR
入口计算 `num_elements × element_bytes`，然后直接与 UB 容量比较，会把可编译配置错删。

因此必须对所有可能改变该证明的真实 pipeline 阶段建立传递合同。

入口文件：

- `third_party/ascend/include/Analysis/TTIRUBLowerBound/UBResourceContract.h`
- `third_party/ascend/lib/Analysis/TTIRUBLowerBound/UBResourceContract.cpp`

### 5.2 合同结果

`UBResourceContract::apply()` 返回：

- `Preserve`：该阶段不改变资源下界；
- `Transform`：按已证明的规则转换资源，例如降低为最小 tile payload；
- `Invalidate`：该阶段可能破坏证明，资源失效并最终 `defer`；
- `InternalError`：合同自身状态错误，整个分析 `defer`。

### 5.3 匹配规则

合同通过 `PipelineStageContext` 匹配：

- `stageName` 必须精确匹配；
- 影响传递函数的 options 必须进入匹配；
- 同一阶段不能有多个匹配合同；
- 未匹配阶段不能默认 Preserve，而是使资源失效；
- contract ID 和 version 必须进入 profile 和证书审计。

### 5.4 当前实现状态

P0 已实现 production profile → ordered registry 装载。P1 已新增可参数化的
`direct-copy-preserve@1` 与 `direct-copy-max-tiles@1` 候选合同；后者按
`ceil(inputPayload/maxTiles)` 传递下界。二者会核对资源数量、source elements、element
width、输入 payload、stage/options，任一漂移即 Invalidate。packaged profile 仍为空，
因此尚未安装经 oracle 认证的 Preserve/Transform 合同。

首个 fixture 把 `direct-copy-max-tiles@1` 绑定在真实的 `ttir.triton-to-linalg` stage；
before-CVPipelining snapshot 中的 local allocation 用来验证这一 stage 输出的 payload，
之后的 BiSheng suffix 只能 Preserve 或 Invalidate，不能反过来冒充 materialization stage。

候选与生产入口已经隔离：离线 oracle 只能通过
`ttir_ub_lower_bound_candidate_for_oracle` 试跑未认证的 active contracts；正常编译调用
`ttir_ub_lower_bound`，其中 Preserve/Transform profile 缺少完整认证元数据时按未知 profile
处理并 `defer`。只会使证明失效的 `invalidate-unmodeled-stage@1` 不需要认证。

## 6. TTIR matcher 与当前建模范围

### 6.1 代码入口

- `third_party/ascend/lib/Analysis/TTIRUBLowerBound/DirectTensorLoadMaterialization.cpp`
- `third_party/ascend/lib/Analysis/TTIRUBLowerBound/VerifierSafety.h`
- `third_party/ascend/lib/Analysis/TTIRUBLowerBound/TTIRUBLowerBound.cpp`

### 6.2 当前支持的形态

V1 matcher 只接受严格的 direct copy：

```text
public tt.func，且模块内只有一个函数
  → 单 block、无嵌套 region
  → tt.make_range [0, N)
  → tt.splat GM pointer
  → tt.addptr
  → 无 mask、无 boundary check 的 tt.load
  → load 结果只有一个 use
  → 无 mask 的 tt.store
  → tt.return
```

还要求：

- tensor 为一维静态 shape；
- element type 为 8/16/32/64 位整数或浮点；
- source/destination pointer chain 连续且无额外 use；
- load 和 store 位于入口 block，且顺序正确；
- load 的 tensor 形状、元素类型和 store 完全一致；
- 所有函数参数都只能被已匹配链使用；
- 模块中不能混入 matcher 未覆盖的 operation。

对该形态产生一个 `GMToUBLoad` 资源：

```text
minPayloadBytes = N × bytes(element_type)
minInstances = 1
contractTrace = ["ttir-direct-load-v1"]
```

### 6.3 当前主动 defer 的主要形态

- dynamic shape、非一维 tensor、sub-byte 或未支持 element type；
- masked load/store、boundary check、特殊 cache/eviction 语义；
- reduction、dot、atomic、算术链、loop、condition、nested region；
- pointer chain 不连续或有额外 use；
- load 没有到达 store，或者一个 load 有多个 use；
- 多函数模块、不完整函数结构、未知 operation；
- 非 AIV 路径；
- 未知 target、pipeline identity 或 stage contract。

这些限制是刻意的。增加覆盖时应先建立传递合同和真实 oracle，再放宽 matcher。

## 7. PipelineIdentity 与 profile

### 7.1 Identity 内容

生成入口位于 `third_party/ascend/backend/compiler.py`：

- canonicalized TTIR 后构造未来 TTIR→Linalg pipeline 字符串；
- 记录 pass 顺序；
- 收集影响 UB 的编译选项和环境开关；
- 记录 target arch、Triton version 和 CANN version hash；
- 记录实际选择的 compiler 类型及 compiler 文件内容 SHA256；
- 对完整 canonical TTIR 文本求 SHA256 并纳入 identity，不在 profile 中保存整份 IR；
- `auto_tile_and_bind_subblock` 使用 module-derived identity 规则；
- 对 canonical JSON 求 SHA256，作为证书 fingerprint。

身份中 canonical TTIR、pass 调序、UB 相关选项、compiler 类型或内容任一变化，都会产生不同 fingerprint。

### 7.2 Profile 文件

文件：`third_party/ascend/backend/ub_contract_profiles.json`

当前状态：

```text
schema = ttir-ub-lb-profile-v1
profiles = []
```

生产 profile 为空是当前最重要的运行边界。没有 profile 时分析器返回
`unknown-pipeline-profile`，不会形成 `reject`。

### 7.3 Binding 当前边界

文件：`third_party/ascend/ttir_ub_lower_bound_bindings.cc`

当前 binding 会：

- 解析并检查 arch、compile mode、pipeline identity 和 stage schema；
- 按完整 identity 选择唯一 profile；
- 按 stage ordinal、stage name、全部 options、contract ID/version 构造只读 registry；
- 拒绝 stage 删除、增加、调序、参数漂移、重复 identity 和未知 contract；
- 调用 C++ analyzer；
- 把结果序列化为 Python dict。

Python 不再固定传空 `pipeline_stages`。真实 PassManager builder 在添加每个 pass 的同时
记录 stage 及类型保持的 option 字符串，并在开源 lowering 之后加入由 compiler 内容哈希
约束的 `bisheng.ub-affecting-suffix` 原子边界。packaged profile 仍为空，因此真实编译仍会
得到 `unknown-pipeline-profile`；这是缺少已认证合同/profile，而不是执行链未接通。

## 8. Python policy 与运行模式

### 8.1 Policy 入口

文件：`third_party/ascend/backend/ub_lower_bound.py`

主要职责：

- `load_contract_profiles()`：加载并检查 packaged profile；
- `_pipeline_fingerprint()`：只接受有效字符串或 identity SHA256；
- `_normalize_result()`：重新验证 binding 返回值；
- `_record_metadata()`：写入紧凑 metadata；
- `apply_ub_lower_bound_policy()`：执行 off/shadow/enforce 行为；
- `ub_filter_telemetry_session()`：记录一次 autotune round 的统计。

后端 `simd` compile mode 会映射为 MURG 使用的 `aiv` core kind；其他模式保持原值，
由 C++ analyzer 决定是否 defer。

### 8.2 三种模式

| 模式 | 是否调用分析器 | reject 时行为 |
| --- | --- | --- |
| `off` | 否 | 保持原有编译流程 |
| `shadow` | 是 | 记录结果并继续编译 |
| `enforce` | 是 | 抛出 `UBLowerBoundOverflow` |

默认值是 `off`。紧急回退只需关闭该选项，不需要修改 TTIR 或清理编译缓存中的 IR。

### 8.3 Python 二次校验

Python 只接受精确 schema：

- decision 只能是 `defer` 或 `reject`；
- byte 数必须是非负 `int64`，不能是 bool；
- target capacity 必须等于 C++ capacity registry 的可信值；
- reject 必须满足 `lower_bound > capacity`；
- reject 必须只有一个合法 singleton certificate；
- certificate bytes 必须等于 lower bound；
- pipeline fingerprint 和 contract version 必须精确一致；
- reject 不能同时携带 unsupported reason。

任何异常返回统一规范化为 `defer`，避免错误结果进入过滤策略。

### 8.4 Metadata 与 debug dump

普通 metadata 记录：

- `ub_lower_bound_mode`
- `ub_lower_bound_decision`
- `ub_lower_bound_bytes`
- `ub_capacity_bytes`
- `ub_lower_bound_contract_version`
- `ub_lower_bound_pipeline_identity`
- `ub_lower_bound_certificate_count`
- `ub_lower_bound_unsupported_reasons`

开启 compiler debug 后，完整结果写入：

```text
kernel.ttir.ub-lower-bound.json
```

完整图和证书不进入普通 metadata，避免缓存和日志膨胀。

## 9. Autotune 与异步编译集成

### 9.1 过滤路径

`UBLowerBoundOverflow` 被视为资源不足类结果，与真实后端报告的资源不足配置一样，从本轮
候选中删除，其他 config 继续 benchmark。

主要文件：

- `third_party/ascend/backend/runtime/autotuner.py`
- `python/triton/runtime/_async_compile.py`
- `third_party/ascend/backend/errors.py`

### 9.2 串行与并行

实现同时覆盖：

- 串行 compile/benchmark；
- `AsyncCompileMode` 并行编译；
- 已完成 future；
- worker ContextVar 传播；
- worker 内清除 `active_mode`，避免嵌套提交回同一 executor；
- 已被调用方观察的 future exception 不在 context exit 时重复抛出；
- 未被观察的普通 exception 仍在 context exit 时报告。

### 9.3 Telemetry

每轮统计：

- `analyzed`
- `rejected`
- `deferred`
- `passed_to_backend`

在 `TRITON_PRINT_AUTOTUNING=1` 时输出一行 round summary，不逐 config 输出完整证书。

## 10. 真实 PlanMemory oracle

### 10.1 目标

单元测试只能证明模型内部一致，不能替代独立语义重放和真实 compiler 对照。oracle 同时运行三条路径：

```text
canonical TTIR ─────────→ lower-bound analyzer ─→ LB / certificate
before-CVPipelining IR ─→ cvpipeline_ub_model ───→ replay peak / overflow
before-CVPipelining IR ─→ real suffix compiler ──→ PlanMemory peak / overflow
```

必须满足：

```text
LB_bits <= ReplayUBPeak_bits == ActualUBPeak_bits
```

如果 analyzer 给出 reject，则每个 seed 的真实结果必须是 UB capacity failure，不能是 L1
或其他编译失败。

### 10.2 工具与 fixture

- 工具：`third_party/ascend/tools/ttir_ub_oracle.py`
- 测试：`third_party/ascend/unittest/ttir_ub_oracle/test_oracle.py`
- manifest：`third_party/ascend/unittest/ttir_ub_oracle/fixtures/manifest.json`
- TTIR fixture：`direct_copy.ttir.mlir`
- before-CVPipelining fixture：`direct_copy.before_cvpipelining.mlir`

工具能力：

- 严格 manifest schema 和 fixture 路径检查；
- manifest 不能注入 pipeline identity；
- 调用真实 `bishengir-cvpipeline-suffix-compile`；
- 调用独立 `cvpipeline_ub_model_cpp` 语义重放，并要求 `precision=exact`；
- 支持固定 seed 列表和 retry seed `-1`；
- 通过完成标记关联真实 PlanMemory attempt ID；
- 区分 UB、L1 等 memory scope；
- 解析 `PLANMEM_PEAK` 和 `PLANMEM_REQUIRED`；
- 输出机器可读 report；
- report 记录 TTIR fixture、before-CVP snapshot、suffix compiler 和 semantic model 二进制 SHA256；
- violation 返回码 1，oracle unavailable 返回码 2，完整通过返回码 0；
- 只在完整、零违规且存在非空证书时生成 profile candidate；
- oracle 使用 candidate-only analyzer 入口运行未安装的候选合同链，打破“空 profile 无法认证”的循环依赖；
- oracle 从真实 `post-TileAndBindSubBlock` stage snapshot 检测 `get_sub_block_idx`，manifest 标签本身不算 outcome 证据；
- 按 `operation_family` 严格解析配对 before-CVPipelining snapshot：direct-copy 必须有一个、binary-add 必须有两个静态 1-D local allocation；每个 allocation 都必须等于 materialization contract 推导的单资源 payload；
- 只有 seeds `0..19`、retry 对该精确 TTIR identity 的实际 outcome 全部验证一致，且
  identity 唯一时才生成可安装 candidate；
- 每个 seed 上语义重放与真实 suffix compiler 的 status、overflow scope、UB peak 和 capacity
  必须精确一致；缺失、非 exact 或不一致均禁止生成 candidate；
- candidate 携带 `semantic_model_sha256`；Python loader 和 C++ production binding 都要求该
  指纹是合法的小写 SHA256，避免重放模型变化后误用旧认证；
- 首个 direct-copy profile 只接受 `compile_mode=simd`、`multibuffer=false`；其他组合继续 defer；
- fixture 必须显式记录 `tile_mix_cube_loop` / `tile_mix_vector_loop`，oracle 将同一组参数同时
  传给真实 suffix compiler 和语义模型，禁止依赖两边可能漂移的默认值；
- 永远不自动编辑 packaged profile。
- `--profile-candidate` 必须同时提供 `--report`，保证 profile 中的 report hash 可审计。

### 10.3 单位

- analyzer 和 capacity registry 使用 bytes；
- PlanMemory 机器输出使用 bits；
- oracle 比较前执行 `lower_bound_bytes × 8`，不能混用单位。

### 10.4 当前验证结果

- 使用 LLVM `fad3272286528b8a491085183434c5ad4b59ab92` 完成原生 `libtriton.so` 构建和导入；
- UB/策略/oracle 聚焦 Python 测试：313 项通过；
- `TestAscendTTIRUBLowerBound` 原生 C++ GTest：92 项通过；
- 完整 identity-bound analyzer + 独立语义重放 + 真实 suffix compiler：
  direct-copy、binary-add、reshape-copy 三类 fixture 各执行 seed `0..19` 加 retry 共 21 次；
- analyzer contract LB 为 `4096 bytes`；21 次 semantic replay 与真实 PlanMemory peak 均为
  `32768 bits`，逐次精确相等，且下界不超过两者；
- 21 次均由真实 `post-TileAndBindSubBlock` snapshot 确认 auto-tile outcome 为 `false`；
- 联合 oracle report：violations 0，unavailable 0，并成功生成未安装的 candidate；
- 新 promotion gate 要求 suffix compiler 提供唯一的 `post-TileAndBindSubBlock` stage snapshot；缺失或歧义会明确 unavailable；
- identity 会把 `-cce-link-aicore-ll-module` 的绝对路径规范化为文件内容 SHA256；安装目录变化不再造成无意义漂移，文件内容变化仍会失配；
- 目标 CANN 环境已对 direct-copy 重跑同一联合门禁，21 次结果仍为 violations 0、unavailable 0；
- binary-add 的目标 CANN candidate 也已完成 seeds `0..19` + retry：analyzer 下界
  `524288 bytes`，真实 PlanMemory 与语义模型均为 UB overflow，required/peak 都是
  `4194304 bits`，capacity 为 `1572864 bits`，21 次逐次一致，auto-tile outcome 均为 `false`；
- reshape-copy 使用真实可编译的
  `1-D load -> 2-D value reshape -> inverse reshape -> 1-D store` canonical TTIR，避免对 pointer
  tensor 做 `tt.reshape`；真实 before-CVPipelining 边界只有一个 `262144-byte` local allocation，
  analyzer singleton 下界为 `262144 bytes`，真实 PlanMemory 与语义模型的 required/peak 均为
  `2097152 bits`，capacity 为 `1572864 bits`，21 次逐次一致，auto-tile outcome 均为 `false`；
- 三份 candidate 都只生成在构建目录供人工审核，没有写入 packaged profile；
- outcome=`true` 只属于未来 split MIX AIV profile，不是当前纯 AIV direct-copy profile 的认证前置条件。

## 11. 测试与质量状态

本次基线已执行：

- UB、autotune policy、async compile 和 oracle 聚焦 Python 测试：313 项通过；
- 精确 LLVM 原生构建：`libtriton.so` 构建并导入成功；
- 普通 C++ GTest：92 项通过；
- direct-copy、binary-add、reshape-copy 的 analyzer + semantic replay + 真实 suffix compiler
  联合 oracle：20 seeds + retry，0 violation / 0 unavailable；
- analyzer profile-miss 路径 100 次测量，去掉前 10 次后：
  - p50 `0.002542 ms`
  - p95 `0.003334 ms`
- `git diff --check` 通过；
- 本次所有提交均带 `Signed-off-by`。

2026-07-24 P4 修正后的本地增量验证：

- oracle、fixture bundle、same-schema helper、coverage matrix：103 项通过；
- 本次修改涉及的 4 个 UB C++ implementation object、GTest object 与 pybind object 均单独重编译成功；
- 三份 JSON evidence/schema 均可解析，HTML5 报告可由标准 parser 完整读取；
- 5 份外部 patch 分别通过目标 source tree 的 `git apply --check`；
- `git diff --check` 通过；
- 完整 GTest 链接被工作区既有 Triton/外部 MLIR header revision 不一致阻塞
  （`DiscardableAttributes.cpp` / `Ops.cpp`，不涉及本次 UB 文件）；本机也没有可导入的
  `triton._C`。因此运行态 GTest/pybind 与 CANN full-chain 不在本地伪装成已验证，统一留到
  服务器恢复后用同一 revision 重建执行。

环境限制：

- 4 个既有 autotune 文件依赖 `torch_npu`，本机无法收集；
- 本地开发机没有 CANN identity 环境和 NPU 硬件；真实 identity/PlanMemory 验证在目标 CANN
  容器完成；
- p95 数据是空生产 profile 的快速 defer 路径，不代表未来完整 matcher/contract 的最终开销；
- `.build-ttir-ub/` 是预存未跟踪构建目录，不纳入提交。

## 12. 当前已经实现的能力

### 12.1 已完成

- MURG 数据结构、alias/distinct/witness 关系和下界 solver；资源新增
  `UBExecutionScope::{SingleCore,AIC,AIV}`，不同物理 core 的资源禁止建立 relation/witness，
  因而不会把 AIC UB 与 AIV UB 错误相加；
- `UBResourceContract` 抽象、registry 和严格未匹配策略；
- verifier 前的结构预检；
- direct static GM→UB load/copy matcher；
- target UB capacity 单一 C++ 数据源；
- pipeline identity 和 UB 相关选项闭包检查；
- BiSheng linked-module 绝对路径按文件内容 SHA256 规范化，避免 profile 绑定安装目录；
- 与真实 PassManager builder 同源的 ordered pipeline stage manifest；
- packaged profile 严格 schema、唯一 identity 和 oracle promotion gate；
- profile → production registry 装载与 ordinal/id/version/options 精确匹配；
- fail-closed `invalidate-unmodeled-stage@1` 可执行合同；
- production/candidate analyzer API 分离，未认证 active contract 不能由生产入口装载；
- canonical TTIR 全文 SHA256 已进入 pipeline identity；
- 参数化 `direct-copy-preserve@1` / `direct-copy-max-tiles@1` 候选合同及严格 source-fact 校验；
- 严格的 `load(lhs) + load(rhs) -> arith.addf -> store` matcher，以及两条 GM→UB resource、
  mayAlias、CoexistenceWitness 和逐资源 lifetime facts；
- 参数化 `binary-add-preserve@1` / `binary-add-max-tiles@1` 候选合同；只有物化阶段验证
  source facts 后才能把同一 witness 内的 mayAlias 精化为 mustDistinct；
- `binary-add-preserve@1` / `reshape-copy-preserve@1` 可出现在 materialization 前后：前段只
  保持 matcher 已证明的 mayAlias，后段保持已精化的 mustDistinct/mustAlias；solver 仍只在
  mustDistinct 后求和、只在 mustAlias 后折叠 alias class，preserve 本身不会凭空加强关系；
- binary-add 合同同时验证 lifetime facts：两个资源必须有不同 birth、相同 lastRequiredUse，
  且 birth 都早于共同 use；任一漂移会 Invalidate；
- witness solver 只允许对同一 CoexistenceWitness 且 pairwise mustDistinct 的资源求和；
- 严格的无 reorder `load -> reshape -> [inverse reshape] -> store` matcher；真实认证 fixture
  使用 value tensor 的 rank-1→rank-2→rank-1 严格逆变换，让 pointer 保持在已验证的连续 1-D
  lowering 路径。load/view 先建 mayAlias，`reshape-copy-max-tiles@1` 在物化阶段证明单 allocation
  后才精化为 mustAlias，solver 按 alias class 计数一次；
- oracle 可构造未安装 candidate chain，且对 seed/retry/outcome/identity 做 promotion gate；
- oracle 已接入 `cvpipeline_ub_model_cpp` exact semantic replay；promotion 要求 replay 与真实
  PlanMemory 逐 seed 一致，并把 semantic model SHA256 写入 profile；
- oracle promotion 按 operation family 要求 singleton 或双资源 witness certificate 的
  bytes/resource ID 合法，且
  `contract_trace` 精确等于 source matcher 与 ordered candidate contracts；
- `ttir_ub_fixture_bundle.py` 要求同一 capture 目录内存在 canonical TTIR 与紧邻
  `createCVPipeliningPass` 之前的 `before_cvpipelining.mlir`，并核对 family-specific allocation
  数量和精确大小；raw `kernel.ttadapter.mlir`、缺文件或已有目标都会被拒绝；
- P4 fixture 已换成真实 full-compiler before-CVPipelining dump，并要求 target/device/core
  provenance；Dynamic MIX dot→exp 的两个逻辑资源分别属于 AIC 与 AIV，1024-byte source
  经已验证 projection 后各形成 512-byte 下界，最终是 512-byte singleton certificate，
  不是跨核相加的 2048-byte witness；
- irregular indirect-add 的真实 boundary 在 `gather_load` 前产生
  `memref.alloc(%dim) : memref<?xf32>`。TTIR 无法给出该 mandatory source buffer 的 extent，
  因此已删除旧 96-byte `irregular-memory-*` contract、profile schema 和 binding factory；
  strict matcher 只返回零证书的 `unsupported-irregular-source-extent`；
- oracle 会识别上述 irregular deliberate-defer，验证真实动态 source boundary 后跳过无意义的
  PlanMemory/promotion；profile candidate 只包含拥有有效 certificate 的 case；
- 同 schema suffix、真实 full-compiler boundary dump、默认 memory-space 兼容和 semantic runtime
  capacity 都有独立 patch；本地已验证 patch 可应用性；
- manifest loader 独立校验 `expected_input_payload_bytes = source_elements × element_bit_width / 8`
  及 int64 边界，不能通过手写 proposal 绕过 source-fact 一致性；
- 新 fixture 默认把 auto-tile outcome 留为 `null`，由 oracle 从所有 seed/retry 的真实
  `post-TileAndBindSubBlock` snapshot 推导；观察到不同 outcome 会产生 violation 并禁止晋升；
- 新 fixture 默认也不预设 analyzer decision；实际 `defer/reject` 由 analyzer 产生，但每个
  `reject` 仍必须在全部真实运行中得到 UB overflow。显式 golden expectation 漂移仍会产生 violation；
- C++ pybind API 与 Python 结果二次校验；
- off/shadow/enforce policy；
- debug certificate dump，并在每个 certificate 的 `contract_trace` 中保留实际参与
  下界计算的 matcher、witness 和逐 stage contract ID；
- 多资源 witness certificate 不再去重 contract ID，而是要求资源携带一致的有序 trace，
  原样保留每一个 pipeline stage；
- autotune 串行/并行过滤接线和 telemetry；
- PlanMemory seed/retry oracle、report 和 candidate gate；
- 双语用户文档和完整单元测试。

### 12.2 尚未形成生产收益的原因

虽然 P0 执行链路和 loader 已接通，但当前生产 profile 为空，且没有经认证的有效
Preserve/Transform 合同。因此：

- `off`：完全保持旧行为；
- `shadow`：调用分析器，但真实 pipeline identity 会得到 defer；
- `enforce`：当前也会 defer，不会提前过滤真实生产 config；
- 测试中的 reject 使用显式 synthetic profile/测试 registry 验证机制正确性；
- 不能把 synthetic reject 当作生产 profile 已上线。

这是有意保留的上线门禁，不应通过放宽 fingerprint 或提供 `allow_unvalidated` 之类开关
绕过。

## 13. 下一步工作

### P4：仅剩目标 CANN 最终门禁

服务器不可用前，Dynamic MIX seed 0 已通过真实 full PlanMemory：AIC 与 AIV scope peak 均为
`4096 bits`，与 analyzer 的 `512 bytes = 4096 bits` singleton 下界一致。当前仍不能声称 P4
promotion 完成，恢复服务器后只需执行以下环境验证，不再补本地建模代码：

1. 用当前源码重建 `libtriton`、同 schema suffix compiler 和 `cvpipeline_ub_model_cpp`，记录三者
   SHA256；
2. 对 Dynamic MIX 运行 seeds `0..19` + retry，逐 scope 比较 analyzer、真实 PlanMemory 与 exact
   semantic replay；
3. 确认 auto-tile outcome、status、capacity、peak 全部一致且 violations/unavailable 为 0；
4. irregular case 必须保持 `unsupported-irregular-source-extent`，不得生成 profile；
5. 只为有非空合法 certificate 的 case 生成 candidate，继续禁止自动安装 packaged profile。

### P1：为真实 pipeline 建立第一组有效合同

当前已完成候选合同实现、canonical TTIR 绑定、可执行 oracle candidate chain，并已在目标 CANN
环境完成 direct-copy 的 analyzer + semantic replay + PlanMemory 联合验证。下一步是人工审核
candidate，再决定是否写入 packaged profile；不能由 oracle 自动安装。`TileAndBindSubBlock` 的 true 分支属于 split MIX AIV，
应在未来 MIX profile 的独立 identity/fixture 中认证，不再阻塞 P1 的 false-outcome profile。
人工审核完成前，packaged profile 必须保持为空。

优先选择最小、可证明且能产生收益的路径，不要直接声明整个 pipeline Preserve。

建议顺序：

1. canonical TTIR 到 before-CVPipelining 之间的 stage manifest；
2. direct copy 路径中确定 Preserve 的 canonicalization；
3. `TileAndBindSubBlock` 的最小 tile transfer function；
4. multi-buffer disabled 的单实例合同；
5. suffix 中影响大小、倍数、alias 和生存期的阶段；
6. 每个合同分别准备正向 fixture 和 defer fixture。

任何阶段无法证明时，整份证书失效并 defer。

### P2：双输入 elementwise 与 alias/coexistence

binary-add 已完成目标 CANN 的 20 seeds + retry 联合验证并生成未安装 candidate：边界包含两个
`262144-byte` local allocations，analyzer witness 下界为 `524288 bytes`，真实 PlanMemory 与
语义模型都得到 `4194304-bit` UB overflow，且所有 auto-tile outcome 为 `false`。reshape-copy
也已完成相同门禁：canonical TTIR 采用
`1-D load -> 2-D value reshape -> inverse reshape -> 1-D store`，通过真实 open-source lowering；
同次编译捕获的 before-CVPipelining 边界只有一个 `262144-byte` allocation，analyzer 下界为
`262144 bytes`，真实 PlanMemory 与语义模型均得到 `2097152-bit` UB overflow，21 次结果精确一致，
auto-tile outcome 均为 `false`。这证明两个逻辑 resource 在物化后属于同一 mustAlias class，
solver 只计数一次。broadcast、expand_dims、bitcast 当前仍以具名 unsupported reason 明确 defer，
避免把尚未证明的 view/materialization 语义误当成已支持。人工审核完成前不得把三类 candidate
写入 packaged profile。下一步保存 broadcast/expand_dims/bitcast 的真实 lowering snapshot，确认各路径究竟
是 mustAlias view、独立 allocation 还是 materialized broadcast，再决定是否扩展合同；证据不足时继续 defer。

### P1：扩大真实 oracle corpus

当前已有 direct-copy、binary-add、reshape-copy 三个配对 fixture。下一步应把同一次真实编译中
自动保存 canonical TTIR 和 before-CVPipelining IR 的流程接入常规回归，继续避免两端配置漂移。

建议矩阵：

- threshold 附近不同 BLOCK_SIZE；
- FP16/BF16/FP32、常见整数类型；
- A2/A3、910_95/950；
- multi-buffer on/off、不同 `num_stages`；
- auto blockify 和 sub-block tiling 两种 outcome；
- vector add、elementwise chain、broadcast、DCE；
- reduction、dot、mask、dynamic shape、SIMT、MIX/CV 作为 defer 对照；
- 每个 fixture 固定 seeds `0..19` 和 retry。

CI 必须把 oracle unavailable 与 comparison violation 分开报告。

### P1：完善 Mandatory UB Resource Graph 求解

1. 接入 `minInstances`，准确表达 multi-buffer 必需实例数。
2. 为双输入 elementwise 建立 must-distinct + coexistence witness。
3. 证明 alias 后再合并资源，不能按 SSA value 数量求和。
4. 建模 reduction accumulator/scratch。
5. 建模 loop-carried resource 和跨 iteration 生命周期。
6. 将 `birth/lastRequiredUse` 与真实 stage transfer 结合，而不是仅保留字段。
7. 在生产 analyzer 中按证书类型选择 singleton/witness solver。

### P1：支持更多 TTIR 语义

扩展顺序建议：

1. 静态 elementwise unary；
2. 两输入、无 broadcast 的静态 elementwise binary；
3. 可证明的 broadcast；
4. reduction accumulator；
5. mask 和边界 tile；
6. loop 与 multi-buffer；
7. SplitMix 后 AIV 投影及 CV 融合路径。

每次扩展都必须同时提交 matcher、MURG 关系、stage contract、oracle fixture 和回归测试。

### P2：与 autotune config 建立更清晰的资源合同

目前 policy 在 canonical TTIR 后执行。后续可把 config 中影响 UB 的字段规范化为
`UBResourceContract` 输入，例如：

- `BLOCK_SIZE`
- `num_stages`
- multi-buffer 开关与策略
- auto blockify / sub-block tiling
- CV tile depth

同一配置规范化结果应同时参与 pipeline identity 和 cache key。不能出现分析器使用一组
options，而真实后端使用另一组 options。

### P2：上线策略

1. 默认继续 `off`；
2. CI 和代表性 workload 开 `shadow`；
3. 对每个证书与真实 peak 做持续对照；
4. 任一反例立即移除对应 profile/contract version；
5. 零违规稳定后，仅允许用户显式 `enforce`；
6. 收集 analyzed/rejected/deferred/passed-to-backend 和编译耗时收益；
7. 最后再评估是否调整默认模式。

## 14. 开发注意事项

### 14.1 不要做的事情

- 不要把 TTIR tensor 总大小直接当作最终 UB peak；
- 不要假设未知 pass 是 Preserve；
- 不要把 `defer` 显示为 fit；
- 不要根据概率、confidence 或经验阈值在 enforce 模式过滤；
- 不要让 manifest 或调用方覆盖 pipeline identity；
- 不要把任意编译失败归类为 UB overflow；
- 不要在没有 CANN/compiler 内容摘要的情况下复用 profile；
- 不要自动把 oracle candidate 写入 packaged profile。

### 14.2 增加规则的最小提交单元

一个完整规则至少应包含：

1. 明确输入合同；
2. MURG resource/relation；
3. 对受影响 stage 的 versioned contract；
4. unsupported/defer 诊断；
5. C++ 单元测试；
6. Python policy 测试；
7. 成对真实 compiler fixture；
8. seed/retry oracle 结果；
9. identity/profile version 更新；
10. 中英文文档更新。

## 15. 常用命令

### 15.1 Python 聚焦测试

```bash
python -m pytest -q --noconftest \
  third_party/ascend/unittest/pytest_ut/test_ttir_ub_lower_bound.py \
  third_party/ascend/unittest/autotune_ut/test_ub_lower_bound_filter.py \
  third_party/ascend/unittest/autotune_ut/test_do_bench_compat.py \
  python/test/unit/runtime/test_async_compile_context.py \
  third_party/ascend/unittest/ttir_ub_oracle/test_oracle.py \
  third_party/ascend/unittest/ttir_ub_oracle/test_fixture_bundle.py
```

### 15.2 Oracle parser 测试

```bash
python -m pytest -q --noconftest \
  third_party/ascend/unittest/ttir_ub_oracle/test_oracle.py
```

### 15.3 从同一次编译 dump 生成 fixture

从同一次真实编译保存 canonical `kernel.ttir.mlir`，并从 BiSheng stage dump 提取紧邻
`createCVPipeliningPass` 之前的 Generic IR，命名为 `before_cvpipelining.mlir`。把两者放在同一个
专用 capture 目录，再执行：

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

工具会先核对 family-specific allocation 数量与 `ceil(payload/max_tiles)` 大小，raw
`kernel.ttadapter.mlir` 不能通过，也不会覆盖已有 fixture。reshape 路径把
`--operation-family` 改成 `reshape-copy`。默认由 oracle
从所有真实 snapshot 推导 outcome；只有维护已知 golden fixture 时才显式传
`--auto-tile-outcome true|false` 增加预期值断言。analyzer decision 默认也不预设；只有维护
golden fixture 时才传 `--expected-analyzer-decision defer|reject`。

### 15.4 真实 PlanMemory 对照

先从 libtriton 使用的 pinned BiShengIR revision 构建同 schema suffix runner：

```bash
python third_party/ascend/tools/ttir_ub_prepare_same_schema_oracle.py \
  --apply-patch \
  --build-dir /path/to/triton-build \
  --cmake /path/to/cmake \
  --jobs 8
```

工具会拒绝错误 revision、既不能正向应用也不能反向验证的 patch，以及应用前已有修改的
AscendNPU-IR source。构建使用
`TRITON_ASCEND_BUILD_BISHENGIR_ORACLE_TOOLS=ON`，产物为
`/path/to/triton-build/bin/bishengir-cvpipeline-suffix-compile`。

当前 pipeline 的 Ascend950 nominal UB 是 256 KiB，但真实 PlanMemory 为 allocator
保留 64 KiB，判溢阈值是 192 KiB；analyzer 和 semantic replay 必须使用该真实阈值。
oracle 会把 analyzer 的逐 case capacity 通过 `--ub-capacity-bits` 显式传给 semantic
model，并核对返回值，禁止依赖可漂移默认值。

```bash
python third_party/ascend/tools/ttir_ub_oracle.py \
  --manifest third_party/ascend/unittest/ttir_ub_oracle/fixtures/manifest.json \
  --suffix-compiler /path/to/bishengir-cvpipeline-suffix-compile \
  --semantic-model /path/to/cvpipeline_ub_model \
  --seeds 0-19 \
  --check-retry \
  --report /tmp/ttir-ub-oracle-report.json \
  --profile-candidate /tmp/ttir-ub-profile-candidate.json
```

返回码：

- `0`：全部数据可用且 comparison violation 为 0；
- `1`：发现下界或 reject 证明违规；
- `2`：缺少 analyzer、compiler identity、PlanMemory 数据或其他必需环境。

### 15.5 提交前检查

```bash
git diff --check
git status --short
git log -1 --show-signature
```

## 16. 关键文件索引

| 模块 | 文件 |
| --- | --- |
| MURG API | `third_party/ascend/include/Analysis/TTIRUBLowerBound/MandatoryUBResourceGraph.h` |
| MURG 实现 | `third_party/ascend/lib/Analysis/TTIRUBLowerBound/MandatoryUBResourceGraph.cpp` |
| Contract API | `third_party/ascend/include/Analysis/TTIRUBLowerBound/UBResourceContract.h` |
| Contract 实现 | `third_party/ascend/lib/Analysis/TTIRUBLowerBound/UBResourceContract.cpp` |
| Analyzer API | `third_party/ascend/include/Analysis/TTIRUBLowerBound/TTIRUBLowerBound.h` |
| Analyzer 主流程 | `third_party/ascend/lib/Analysis/TTIRUBLowerBound/TTIRUBLowerBound.cpp` |
| Direct-load matcher | `third_party/ascend/lib/Analysis/TTIRUBLowerBound/DirectTensorLoadMaterialization.cpp` |
| Pybind | `third_party/ascend/ttir_ub_lower_bound_bindings.cc` |
| Pipeline identity | `third_party/ascend/backend/compiler.py` |
| Python policy | `third_party/ascend/backend/ub_lower_bound.py` |
| Capacity/runtime | `third_party/ascend/backend/runtime/utils.py` |
| Autotune | `third_party/ascend/backend/runtime/autotuner.py` |
| Async compile | `python/triton/runtime/_async_compile.py` |
| Profile | `third_party/ascend/backend/ub_contract_profiles.json` |
| Oracle | `third_party/ascend/tools/ttir_ub_oracle.py` |
| 同 schema suffix 准备/构建 | `third_party/ascend/tools/ttir_ub_prepare_same_schema_oracle.py` |
| 同 schema suffix patch | `third_party/ascend/tools/patches/0001-feat-port-same-schema-CVPipeline-suffix-oracle.patch` |
| Full compiler boundary dump patch | `third_party/ascend/tools/patches/cvpipeline_full_compiler_in_process_oracle.patch` |
| Suffix 默认 memory-space patch | `third_party/ascend/tools/patches/cvpipeline_suffix_default_memory_space.patch` |
| Legacy suffix Fixpipe patch | `third_party/ascend/tools/patches/cvpipeline_suffix_legacy_fixpipe.patch` |
| Semantic runtime capacity patch | `third_party/ascend/tools/patches/cvpipeline_ub_model_runtime_capacity.patch` |
| C++ 测试 | `third_party/ascend/unittest/TTIRUBLowerBoundTest.cpp` |
| Python 测试 | `third_party/ascend/unittest/pytest_ut/test_ttir_ub_lower_bound.py` |
| Autotune 测试 | `third_party/ascend/unittest/autotune_ut/test_ub_lower_bound_filter.py` |
| Oracle 测试 | `third_party/ascend/unittest/ttir_ub_oracle/test_oracle.py` |

## 17. 交接结论

当前代码已经完成“可审计的 TTIR UB 下界证明框架”和 autotune 接线：MURG、contract、
identity、policy、telemetry、异步过滤以及真实 PlanMemory oracle 都已具备，并有完整的
保守失败策略。

P0–P4 的本地开发已形成安全闭环：可证明切片拥有可执行合同链，无法证明的 family 拥有稳定、
机器可读的 deliberate-defer。P4 的关键修正是把 MIX AIC/AIV 建模为独立 UB execution scope，
并撤销遗漏动态 source allocation 的 irregular 96-byte 假证书。pinned revision 的同 schema
suffix、full-compiler capture、memory-space 兼容、可复现构建入口和真实 192 KiB allocator
threshold 对齐均已实现。当前版本仍不会对真实生产 config 给出有效 reject，因为 packaged
profile 为空。

服务器恢复后只剩 Dynamic MIX 的 20 seeds + retry suffix / semantic replay / analyzer 零违规
报告和 rebuilt binary hashes。irregular 必须继续具名 defer，除非未来能从 TTIR 或 config
获得可证明的 source extent 上界并重新提交 matcher、MURG、全 stage contracts 与 oracle。
任一 schema、capacity、stage、execution scope 或结果漂移都继续 fail closed。
