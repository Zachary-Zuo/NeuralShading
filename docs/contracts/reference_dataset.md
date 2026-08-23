# 随机游走参考数据合同

## 目的

参考数据只描述 `MaterialProgram` 的散射响应和统计不确定性，不包含 K2、lobe、latent 或任何拟合后端参数。表示发生变化时，不应重新生成随机游走监督。

## 查询语义

一个方向响应样本由以下内容确定：

```text
ReferenceQuery
  material_program_id
  canonical_material_ir_id
  material_state_id
  view_direction_local
  light_directions_local[]
  sample_policy
  random_stream_id
  max_path_depth
```

持久化数据统一使用局部 shading frame。`wo` 是观察方向，`wi` 是入射光方向，两者指向远离表面。每个方向的监督量为：

```text
response_cos = f(wo, wi) * max(dot(Ns, wi), 0)
```

文件字段必须使用 `view_direction`、`light_direction` 和 `response_cos` 等角色明确的名称，不能直接用含义依赖调用方的 `wi/wo` 作为磁盘字段名。

## 统计结果

训练读取 API 必须返回：

```text
ReferenceStatistics
  mean
  variance 或 standard_error
  sample_count
  replica_mean_a        可选
  replica_mean_b        可选
```

训练代码不能从 `sample_count` 猜测方差。采集器内部使用 Welford 或等价的稳定累计方法。

- train 数据必须保存足以得到逐方向训练置信度的统计量；
- representation-ceiling、验证和 reference 回归数据必须额外保存 A/B 独立随机流；
- 磁盘可以使用 fp16 或压缩统计量，但 reader 必须输出 fp32 `mean` 和不确定性；
- 量化误差上界写入 manifest，并由 round-trip 测试验证。

v2 已确定采用 `mean/variance/replica_mean_a/replica_mean_b = float32`、`sample_count = uint32`，每个 RGB 方向 52 bytes。这里优先保证困难导体和深层栈长尾的二阶矩不因 fp16 溢出或下溢失真；后续若容量成为瓶颈，只能新增经过误差上界验证的磁盘 encoding，Python reader 的逻辑字段和物理语义不变。

新采集数据的 `variance` 是两个独立随机流合并后的逐样本总体方差，`standard_error = sqrt(variance / sample_count)`。由 v0 转换的数据没有逐样本二阶矩，manifest 将 `uncertainty_kind` 明确写为 `replica-mean-variance`；此时 reader 返回的 `standard_error = sqrt(variance)`，禁止按逐样本方差再次除以样本数。

## 数据集目录

目标逻辑布局：

```text
dataset/
  manifest.json
  material_programs.jsonl
  canonical_material_ir.bin
  material_states.npy
  family_splits.npy
  view_directions.npy
  light_directions.npy
  solid_angle_weights.npy
  shards/
    shard-00000.index.npy
    shard-00000.response.npy
    shard-00000.complete.json
```

实际大数组继续允许内存映射和分片。manifest 是唯一入口，reader 不通过文件名猜测格式。

## manifest 必需字段

```text
format_name
format_version
dataset_id
created_at
material_program_schema_version
canonical_ir_abi_version
scattering_contract_version
reference_implementation_id
reference_source_sha256
generator_git_commit
prior_id / prior_version
resolved_config_sha256
seed
direction_parameterization
response_measure
color_model
counts and shapes
statistics_encoding
quantization
split_policy
shards[]
content_hashes
```

manifest 不保存开发机绝对路径。所有文件 URI 相对于数据集根目录。

## family、state 和 tile

- `family`：共享结构和主参数的材质族，是 train/validation/test 划分的最小单位；
- `material_state`：同一 family 下的一组局部参数变化；
- `tile`：一个 material state 与一个 view direction 的全部入射方向响应。

同一 family 的全部 state 和 view 必须位于同一 split。writer 可以任意分片，但逻辑 tile ID 和 split 不依赖物理 shard。

## 方向集合和积分权重

方向参数化不是隐含常数。manifest 必须记录算法、版本、方向数组哈希和权重测度。

如果 tile 存储的已经是 `f*cos`，环境积分只能再乘光源辐亮度和立体角权重，不能重复乘余弦。reader 提供具有角色名称的字段，并在单元测试中用 Lambert 解析积分检查。

## reference 身份

`reference_implementation_id` 由以下内容共同确定：

- 随机游走 shader 及其包含文件哈希；
- MaterialProgram 节点语义版本；
- canonical IR ABI；
- max depth、RR、NEE、MIS 和介质采样策略；
- Falcor/Slang 版本；
- 已知适用范围和限制。

只记录一个顶层源码哈希不足以证明数据可复现。

## 生成要求

- 分片写入采用临时文件加原子完成标记；
- 支持按 shard 恢复，已完成 shard 必须验证哈希后复用；
- 固定 seed 和 stream ID 能重现统计分布及确定性测试规模的精确结果；
- 生成完成后执行有限值、非负、shape、hash、split、方向权重和噪声分布验证；
- D3D12 与未来 Vulkan 后端差异通过统计容差验证，不要求逐 bit 相同。

## v0 迁移

当前 `ncls-direction-tiles@1` 数据保留为 legacy evidence。新核心不长期支持两套 reader；提供一次性转换或隔离的 `legacy_v0` adapter：

- 原 A/B 均值转为 replica 字段；
- 因 v0 没有二阶矩，方差只能标记为 `replica-estimate`，不能伪装成逐样本方差；
- 保留原始 reference 哈希和格式限制；
- 转换不改变响应数值或重新命名其物理含义。

当前一次性转换入口为：

```powershell
conda run -n neural-shading python -m ncls.cli data convert-legacy-v0 <旧数据集> <新数据集>
conda run -n neural-shading python -m ncls.cli data validate <新数据集>
```
