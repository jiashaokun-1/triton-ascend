# TTIR UB 保守过滤器

## 目标

TTIR UB 保守过滤器只在证明某个 autotune 配置的必需 Unified Buffer（UB）用量超过
目标容量时，提前过滤该配置。它运行在 canonical TTIR 生成之后、昂贵的
TTIR→HIVM 编译之前。

该过滤器是单向判定：

- `reject` 表示已经证明该配置超过 UB 容量；
- `defer` 表示当前规则无法判定；
- `defer` 不表示该配置一定能放入 UB。

默认模式是 `off`，过滤器不会修改 TTIR。

## 开启与回退

在每个 autotune 配置中设置编译选项：

```python
triton.Config({"BLOCK_SIZE": 65536, "ub_lower_bound_mode": "shadow"})
triton.Config({"BLOCK_SIZE": 65536, "ub_lower_bound_mode": "enforce"})
```

三种模式的行为如下：

| 模式 | 是否分析 | 编译行为 |
| --- | --- | --- |
| `off` | 否 | 保持原有行为 |
| `shadow` | 是 | 记录结果，但始终继续编译 |
| `enforce` | 是 | 仅停止具有 `reject` 证明的配置 |

需要回退时设置 `ub_lower_bound_mode="off"`，或删除该选项即可。

## 整体架构

实现分为四层：

1. 严格 TTIR matcher 识别已支持的静态 direct load。
2. Mandatory UB Resource Graph（MURG）记录必需资源、identity/alias 关系、
   共存关系、最小 payload 和证明 trace。
3. `UBResourceContract` 描述真实 pipeline 中某个精确阶段如何 Preserve、Transform
   或 Invalidate 这些资源。
4. 策略层将已认证的下界与目标 UB 容量比较。只有
   `lower_bound_bytes > capacity_bytes` 才返回 `reject`；两者相等时继续编译。

每份证明都绑定精确 pipeline identity，其中包含有序 pass pipeline、影响 UB 的选项、
target、Triton 版本、CANN identity、实际选择的编译器类型及编译器内容摘要。遇到未知
operation、target、identity 或 contract 时，统一返回带结构化原因的 `defer`。

生产 profile 初始为空。只有真实编译器逐 seed 比较和 retry 校验全部通过后，才允许
评审并加入新 profile。

## Metadata 与诊断

`shadow` 和 `enforce` 会在 compiler metadata 中记录紧凑且可 JSON 序列化的字段，
包括模式、判定、下界、容量、pipeline fingerprint、证书数量和结构化原因。

开启 compiler debug 输出时，完整证书写入：

```text
kernel.ttir.ub-lower-bound.json
```

普通编译不会输出完整资源图和证书。

## 扩展建模范围

支持新的 TTIR pattern 或 lowering 阶段时，需要：

1. 在 MURG 中增加资源与关系，不依赖临时 SSA 名称。
2. 为每个受影响的 pipeline 阶段增加带版本的 `UBResourceContract`。
3. applicability 或传递规则不完整时返回 `defer`。
4. 增加 C++ matcher/graph 测试和 Python policy 测试。
5. 增加成对的 canonical TTIR 与 before-CVPipelining fixture。
6. 使用 seed `0..19` 和 retry 模式与真实 PlanMemory 比较，通过后再评审 profile。

为避免把不同编译产生的文件误配成一组，先开启 Triton kernel dump，并从同一个 cache
目录打包 `kernel.ttir.mlir` 和 `kernel.ttadapter.mlir`：

```bash
python third_party/ascend/tools/ttir_ub_fixture_bundle.py \
  --dump-dir /path/to/dumps/ONE_CACHE_KEY \
  --output-dir /tmp/binary-add-fixture \
  --name binary-add-f32-65536-a2 \
  --operation-family binary-add \
  --arch Ascend910B \
  --source-elements 65536 \
  --element-bit-width 32 \
  --max-tiles 64
```

打包器会自动推导 input payload、输出两个文件哈希，并拒绝覆盖已有 fixture。默认由 oracle
从所有 seed/retry 的真实 snapshot 推导 auto-tile outcome；结果不一致时禁止晋升。只有维护
已知 golden fixture 时才显式传 `--auto-tile-outcome true|false` 作为额外断言。

校验命令：

```bash
python third_party/ascend/tools/ttir_ub_oracle.py \
  --manifest third_party/ascend/unittest/ttir_ub_oracle/fixtures/manifest.json \
  --suffix-compiler /path/to/bishengir-cvpipeline-suffix-compile \
  --semantic-model /path/to/cvpipeline_ub_model \
  --seeds 0-19 --check-retry \
  --report /tmp/ttir-ub-oracle-report.json \
  --profile-candidate /tmp/ttir-ub-profile-candidate.json
```

工具不会修改 `ub_contract_profiles.json`。只有全部结果可用、比较违规为零且证书非空时
才生成候选文件；正式加入 profile 仍是显式评审步骤。
