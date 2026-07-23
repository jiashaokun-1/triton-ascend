# MultiBuffer 对 UB 峰值影响的 HIVM 建模方案

## 1. 目标

本项目通过 HIVM 建模，评估第二次 `MarkMultiBufferPass` 对 UB 利用率和 UB 峰值的影响，并判断 kernel 是否可能发生 UB overflow。

第一阶段不接入 Autotuner，也不从 TTIR 或 Python AST 提前预测 HIVM。先以真实 HIVM 为输入，完成模型、解释报告和准确度校准。模型成熟后，再建立 Autotuner config 到 HIVM 特征的映射，将模型前移用于候选剪枝。

第一阶段的最高优先级是：**真实发生 UB overflow 的样本不得漏报**。允许暂时误报，并通过与真实 `PlanMemory` 的逐项校准降低误报率。

## 2. 范围

### 2.1 第一阶段包含

- 第二次 `MarkMultiBufferPass` 前后的 HIVM 对比；
- UB local buffer 的容量、alias、生命周期和地址冲突；
- `multi_buffer = 1/2/4/N` 的地址槽展开；
- 与 `PlanMemory` 一致的对齐和可复用地址规划语义；
- 普通 Vector 算子；
- mix-CV 中所有位于 UB 的 local buffer，以 FA（Flash Attention）作为真实 mix-CV 主回归算子；
- 影子规划结果、保守上界和真实 `PlanMemory` 标签的校准。

### 2.2 第一阶段不包含

- Autotuner config 到 HIVM 的映射；
- 基于 TTIR、Kernel AST 或 Python AST 的早期预测；
- 第一次 `MarkMultiBufferPass` 的 tensor/workspace 预测模型；
- GM workspace、L1、L0A、L0B、L0C 的 overflow 判定；
- 为了建模而修改 `MarkMultiBufferPass` 的决策规则。

Cube 自身的 L1/L0 占用不计入 UB，但 Cube 与 Vector 交接产生的 UB local buffer 必须纳入模型。

## 3. 为什么选择第二次 MarkMultiBuffer

HIVM local memory pipeline 的关键顺序是：

```text
Bufferization
  → 若干 HIVM 变换
  → AllocExtraBuffer
  → InferHIVMMemScope
  → MarkMultiBuffer（local-only）
  → PlanMemory（local memory）
  → Lower to loops
  → EnableMultiBuffer
```

在第二次 `MarkMultiBufferPass` 之前：

- tensor 已物化为 memref；
- 额外临时 buffer 已创建；
- 新 buffer 已经过 `InferHIVMMemScope`；
- UB、GM、L1、L0 等 address space 已可区分；
- shape、dtype、buffer size、use-def、loop/scope 结构可用于内存分析。

因此它是第一版 UB 精确模型最稳定的输入点。

## 4. 输入

模型接收同一个 kernel 的两个 HIVM 快照，并接收一个只用于校准的真实编译结果。

### 4.1 输入 A：第二次 MarkMultiBuffer 前 HIVM

输入 A 用于建立基础内存模型，至少需要提取：

- 函数、region、block 和 operation 的结构；
- `memref.alloc` 等 alloc-like 对象；
- memref shape、动态维度信息、dtype 和 buffer-size 属性；
- `#hivm.address_space<ub>` 等 memory scope；
- load、store 和 destination-style op 的输入输出；
- `scf.for`、`scf.while`、`scf.if`、scope、yield 和 branch；
- subview、select、循环参数、DPS init/result 等 alias 关系；
- inplace 候选和会限制 inplace 复用的语义；
- buffer 所属的父 loop、scope 和控制流路径。

只有 address space 为 UB 的对象进入 UB 容量计算。其他空间保留在诊断信息中，但不加入 UB 字节数。

### 4.2 输入 B：第二次 MarkMultiBuffer 后 HIVM

输入 B 与输入 A 的主体结构相同，新增的关键信息是：

```mlir
annotation.mark %buffer {
  hivm.multi_buffer = 2 : i32
}
```

模型沿 `annotation.mark` 的 source value 回溯到 alloc-like 对象，读取真实 `multi_buffer=N`。未标记的 buffer 使用 factor 1。

前后对象不能依赖 SSA 文本名称匹配。模型为每个对象生成结构身份：

```text
BufferId = function
         + parent region/block path
         + alloc ordinal
         + memref type/scope validation
```

若结构匹配失败，结果进入 `REVIEW`，不得静默采用 factor 1。

### 4.3 标签 C：PlanMemory 结果

标签 C 不参与模型推理，只用于准确度校准：

- 成功时：真实 `pointer_cast` offset、每个 memory scope 的最高地址、地址复用关系；
- 失败时：失败 scope、申请大小、硬件上限和诊断信息；
- 编译上下文：编译器版本、目标芯片、UB 容量、pass options。

这种隔离避免模型读取真实规划结果后再声称自己完成了预测。

## 5. 统一内存图

输入 A 和输入 B 使用同一个分析内核，转换为统一内存图。两次分析的主要差异只有 multi-buffer factor。

每个 `BufferRecord` 至少包含：

```text
buffer_id
defining_op
memref_type
address_space
raw_bytes
aligned_bytes
dynamic_size_state
alias_group_id
gen_point
kill_point
parent_control_flow
multi_buffer_factor
slot_ids
unknown_semantics
```

模型同时保留 operation 顺序、控制流关系和 buffer 冲突边。

## 6. Alias 分析

同一物理内存可能对应多个 SSA value。模型必须将这些 value 合并为 alias group，避免重复计数或提前结束生命周期。

第一阶段覆盖与 `PlanMemory` 相关的主要关系：

- operation alias interface；
- destination-style op 的 init/result；
- `scf.for` 的 init arg、region iter arg、yield 和 result；
- `scf.while` 的 before/after 参数、condition 和 yield；
- `scf.if` 两个分支的 yield/result；
- `arith.select`；
- scope 参数和返回值；
- `cf.br`、`cf.cond_br` 的目标 block argument；
- subview 和其他可证明共享底层存储的 view。

Alias group 的生命周期覆盖所有成员。条件分支 alias 需要记录条件属性；第一版不能证明互斥安全时按冲突处理。

## 7. 生命周期分析

生命周期采用接近 `PlanMemory` 的 gen/kill 语义，而不是简单使用 `alloc` 到最后一个直接 user。

### 7.1 Gen

buffer 从第一次需要稳定物理存储的位置开始存活。根据 HIVM op 语义，在 destination-style output、load output、store value 等位置建立 gen 信息。

### 7.2 Kill

只有当 buffer 及其所有 alias 在当前操作之后均死亡，且定义点对当前位置满足控制流支配关系时，才能建立 kill。

### 7.3 控制流

- buffer 跨 loop iter arg/yield 使用时，生命周期提升到循环边界；
- preload 或跨 scope 使用的对象按相应父循环边界处理；
- `if` 分支不能证明互斥复用时，保守地认为存在冲突；
- `while` 和非结构化 branch 沿 alias 与 block liveness 传播；
- 未知 op 接触 UB buffer 时，扩大其生命周期到父 region，并禁止相关可疑复用。

模型输出每个 buffer 的 gen、kill 和导致生命周期扩大的原因。

## 8. MultiBuffer 物化模型

对 factor 为 N 的 buffer，模型像 `PlanMemory::ExpandMultiBufferStorageEntry()` 一样展开 N 个地址槽：

```text
A, factor=2
  → A.slot0
  → A.slot1
```

每个槽继承原 buffer 的容量、对齐、alias group 和生命周期。同一 multi-buffer group 的槽位强制互相冲突，不能复用同一地址。

模型分别分析：

```text
Mark 前：未标记对象 factor=1
Mark 后：读取真实 hivm.multi_buffer=N
```

因此：

```text
multibuffer_peak_increment
  = UB_after_peak - UB_before_peak
```

## 9. 双轨内存规划

### 9.1 轨道 A：影子 PlanMemory

影子规划器追求接近真实 `PlanMemory`：

1. 按 memory scope 分组；
2. 对 raw size 应用目标 scope 的 alignment；
3. 根据生命周期、alias、inplace 和 multi-buffer group 建立冲突关系；
4. 在不冲突时允许复用 offset；
5. 模拟 placement，记录每个 slot 的 offset；
6. 使用最高 `offset + aligned_size` 作为 `planned_ub_bytes`；
7. 与目标 UB 上限比较，给出模拟规划是否失败。

### 9.2 轨道 B：保守安全上界

保守轨道用于保证不漏报：

- 不能证明可复用就不复用；
- 未知 UB op 相关对象不复用；
- 动态容量没有可靠上界时不判安全；
- 条件 alias 或分支互斥无法证明时按冲突处理；
- 模型前后对象匹配失败时不默认 factor 1。

输出 `safe_upper_bound_bytes` 和所有保守增加项。

### 9.3 决策

```text
SAFE
  影子规划和保守上界均未超限，且没有未知 UB 语义

OVERFLOW
  影子规划超过 UB 上限，或真实 PlanMemory 标签已明确失败

REVIEW
  影子规划安全但保守上界超限；
  或存在动态容量、未知 UB op、对象匹配失败等不确定信息
```

在离线评估数据集中可以用真实 PlanMemory 标签覆盖最终真值字段；在线模型自身的预测字段必须保持独立。

## 10. 输出

一次分析同时生成机器可读 JSON 和人可读报告，两者来自同一个结果对象。

### 10.1 摘要

- `UB_before_peak`；
- `UB_after_peak`；
- `multibuffer_peak_increment`；
- Mark 前后 UB 利用率；
- `shadow_planned_bytes`；
- `safe_upper_bound_bytes`；
- `SAFE / REVIEW / OVERFLOW`；
- 芯片 UB 上限和编译上下文。

### 10.2 峰值贡献

对峰值位置列出：

- buffer id 和定义位置；
- raw/aligned bytes；
- factor 和 slot 数；
- gen/kill；
- alias group；
- 模拟 offset；
- 与哪些对象冲突或复用；
- 对峰值的贡献；
- 被 Mark 的原因和源码位置。

### 10.3 校准差异

模型与真实 PlanMemory 不一致时，将差异归入：

- buffer set；
- size/alignment；
- alias；
- lifetime；
- inplace；
- multi-buffer expansion；
- placement/reuse；
- unknown semantics。

## 11. 实现架构

正式模型实现为 AscendNPU-IR 内的独立 C++ MLIR 分析工具或分析 pass：

- 直接消费已解析的 HIVM Module；
- 复用 HIVM 类型、dialect interface、MLIR Liveness 和稳定的内存空间信息；
- 不通过正则或纯文本解析实现核心语义；
- 不调用真实 PlanMemory 后读取答案作为模型结果；
- 模型组件保持独立：IR 提取、alias/liveness、slot expansion、shadow planner、conservative bound、reporter。

真实 `MarkMultiBuffer` 和 `PlanMemory` 作为校准基准。Python 只负责：

- 批量运行编译器和模型；
- 保存三个快照；
- 组织验证数据集；
- 计算准确率和误差指标；
- 生成人读报告。

## 12. 错误与保守处理

- 缺少 UB scope：输入无效，不得判 SAFE；
- 动态 shape 无上界：`REVIEW`，并报告缺失的维度；
- 非法 factor：输入错误；
- Mark source 无法回溯到 alloc：`REVIEW`；
- 前后 BufferId 无法匹配：`REVIEW`；
- 未知 op 接触 UB：扩大生命周期、禁止相关复用并记录；
- 真实 PlanMemory 崩溃或非资源类失败：与 UB overflow 分开标记为 compiler error，不能当作 overflow 标签；
- 目标芯片 UB 容量未知：只输出字节估算，不输出 SAFE。

## 13. 验证方案

### 13.1 可控 MLIR 小用例

覆盖：

- f16/f32 和不同 shape；
- 对齐前后一个单位的边界；
- 生命周期完全重叠、完全分离和首尾相接；
- subview、DPS、select、yield、loop iter arg；
- inplace 可复用与禁止复用；
- `for/while/if/scope`；
- factor 1、2、4 和一般 N；
- preload 和 loop load/store；
- 动态 shape、未知 UB op、缺 scope、非法 factor；
- 恰好等于 UB 上限和超过一个 alignment unit。

每个结构都生成 Mark 关闭/开启配对样本，验证 `Delta UB`。

### 13.2 真实算子回归

- 纯 Vector：elementwise、reduce、softmax、layernorm；
- 搬运密集：transpose、copy、gather/scatter；
- Cube 边界：matmul 及其 Vector 交接；
- mix-CV：以 FA 为主，覆盖 Cube + Vector、CV bridge、额外临时 buffer、循环和 MultiBuffer；
- shape、dtype、tile 和 multibuffer 配置组合。

每个样本保存 Mark 前 HIVM、Mark 后 HIVM、模型结果、PlanMemory 结果、编译器版本、芯片信息和 pass options。

### 13.3 指标

- overflow recall：硬门槛 100%；
- 误报率；
- planned peak 的 P50/P95/最大误差；
- MultiBuffer 峰值增量的绝对误差和相对误差；
- SAFE/REVIEW/OVERFLOW 混淆矩阵；
- 按 alias、life、alignment、placement 分类的差异数量。

## 14. NPU 验证环境

服务器：`${ASCEND_VALIDATION_HOST}`；容器：`${ASCEND_VALIDATION_CONTAINER}`；代码目录：`/home/skj/code`。

已确认：

- ARM64，640 CPU 核；
- 容器可访问 Ascend 910，设备健康；
- `/home` 剩余约 1.2 TB，但使用率已达 91%；
- 主机和容器没有 HTTP/HTTPS/SOCKS 代理；
- DNS 可以解析 GitHub，但 HTTPS 访问超时，不能依赖在线下载。

本地 `/Users/sky/Code/AscendNPU-IR` 是开发源码和 Git 变更的唯一真源；所有代码先在本地修改，便于直接审阅 diff 和保留提交历史。验证批次再离线增量同步到容器 `/home/skj/code/AscendNPU-IR`，排除 `.git`、本地 `build` 和缓存，在容器内构建与运行测试。服务器目录是可重建的验证副本，不能成为代码变更的唯一存储位置。

第一阶段的 compiler-side 校准不依赖 NPU。模型稳定后再用 NPU 验证：

- 模型判安全的 kernel 能运行且数值正确；
- MultiBuffer 的地址轮转和同步正确；
- MultiBuffer 的性能收益是否值得额外 UB 占用。

## 15. 阶段计划

1. 在 HIVM 上实现统一内存图、alias/liveness、槽位展开和双轨规划，并完成可控 MLIR 测试。
2. 与真实 MarkMultiBuffer/PlanMemory 批量校准，首先达到 overflow recall 100%，再降低误报。
3. 完成真实 Vector 算子和 FA mix-CV 回归，并在 Ascend 910 上验证正确性和性能。
4. 建立 Autotuner config 到 HIVM 特征和模型输入的映射，将成熟模型前移用于剪枝。

## 16. 验收标准

第一阶段完成需同时满足：

- 所有支持的 HIVM 输入均能生成可解释结果，未知语义不会被静默忽略；
- Mark 前后使用同一个分析内核，factor 是主要受控差异；
- 输出可以定位每个峰值 buffer、slot、生命周期和地址贡献；
- 可控用例和真实回归中的 overflow recall 为 100%；
- 与真实 PlanMemory 的差异可归因，而非只有一个不一致数字；
- FA mix-CV 样本覆盖真实 CV 交接和 UB local buffer；
- 模型接口不依赖 Autotuner，后续可以稳定接入 config 到 HIVM 映射层。
