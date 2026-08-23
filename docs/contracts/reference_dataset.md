# ReferenceDataset HDF5 合同

## 1. 定位

`ncls.reference-dataset@3` 是项目唯一的数据持久化合同。它保存某组源材质状态在一组明确查询上的 reference 响应，不规定源材质必须是层模型、OpenPBR、测量表或 MaterialX 图，也不包含 neural backend 的 latent、网络权重和私有 `ScatteringState`。

公共数据流只有一条：

```text
family-specific source state
        ↓ ReferenceProvider
(state, surface/footprint, wo, wi, proposal)
        ↓ family-specific Falcor reference
response + uncertainty
        ↓
one fixed HDF5 layout
```

旧目录、NPY shard 和旧 schema 不再读取或转换。数据必须用当前 source package、reference 和采集配置重新生成。

## 2. “HDF5 能恢复参数空间”的准确含义

HDF5 对已采样监督域是自包含的：仅打开文件即可得到所有已采样 state identity、原生状态描述 payload、split、空间/footprint 查询、`wo/wi`、采样 PDF、积分权重、随机流、响应和统计量。response-only evaluator、target-visible compression、监督审计与固定 query 评测不需要识别具体材质族。

它不能从有限样本恢复未采样的连续函数，也不重复嵌入数 GB 的 MERL 表或 4K MaterialX 纹理。`source_uri + source_sha256` 和原生 payload 共同锁定外部 source package；只有重新执行任意新 query、读取源纹理训练 source compiler，或构造新编辑状态时才需要这些锁定资源。

`source_sha256` 始终标识原始 source asset，不标识临时 adapter：MERL/OpenPBR 使用原始测量表或 `.mtlx` 的文件 SHA-256；MaterialX 使用文档及其已连接 base color、roughness、metalness、normal、displacement 纹理的确定性组合 SHA-256；LayerStack 使用规范化原生状态 identity。viewer scene 与 HDF5 使用同一规则。

因此项目区分两种恢复：

- 恢复已采样监督参数空间：只需 HDF5；
- 恢复源材质并求任意新点：需要 HDF5 中的 identity/hash 加 `assets/source-materials/` 对应 package。

## 3. Provider 接口

每个已复现材质族实现同一个 `ReferenceProvider`：

```text
descriptor
source_states()       枚举或采样原生状态
surface_samples()     给出 constant、UV 或 surface-point 位置与 footprint
query_plan()          给出 wo、wi、proposal PDF、立体角权重和 seed
evaluate()            调用该材质族的权威 reference
metadata()
close()
```

collector 不解析原生 payload，不认识材质参数，也没有按材质族分支。材质差异只存在于 provider 内。新增材质族时不得修改 HDF5 布局或公共 learning reader。

内存中的 `QueryPlan` 接受共享的 `wi[light, 3]`，也接受逐 `wo` 的 `wi[view, light, 3]`；共享表会在校验时显式广播。PDF 与积分权重相应为 `[view, light]`。这使 proposal 可以跟随移动的镜面峰，同时 HDF5 仍保持一个 query group 对应一个 `(state, surface, wo)` 的固定布局。

当前正式 provider：

| provider | family | reference | 查询域 |
|---|---|---|---|
| `layer-stack` | `ncls.layer-stack@1` | 多层随机游走 | 常量表面、上半球、Monte Carlo moments |
| `merl` | `merl.measured-brdf@1` | MERL 测量表 | 常量表面、上半球 |
| `openpbr` | `openpbr.surface@1.1.1` | Adobe OpenPBR BSDF | 常量表面、完整入射球、含透射/PDF |
| `materialx` | `materialx.textured-surface@1` | 原生纹理解析 + standard_surface | UV、footprint、normal map、上半球 |

pbrt coated package用于独立验证 LayerStack reference，不是一个独立训练 source family，因此不作为 provider 重复导出。

## 4. HDF5 固定布局

文件根属性记录格式、生成时间、Git 提交、query profile、采集配置、provider metadata、proposal code 表、计数和语义内容哈希。实现对应的机器可读清单位于 `src/ncls/data/schemas/reference_dataset_v3.layout.json`。

```text
/
  attrs
    format_name = ncls.reference-dataset
    format_version = 3
    dataset_id = SHA-256(全部语义内容)
    response_measure
    color_model
    query_profile_ids_json
    generation_config_json
    provider_metadata_json
    proposal_ids_json
    state_count / query_group_count / direction_count

  /states
    state_id, family_id, reference_id, asset_id
    split_group_id, split
    native_schema_id
    source_uri, source_sha256, parent_state_id
    payload_offsets, payload_blob

  /queries
    state_index, position_kind
    position, uv, uv_dx, uv_dy
    geometric_normal, geometric_tangent
    wo
    wi[direction_count]
    proposal_code, proposal_pdf[direction_count]
    solid_angle_weight[direction_count]
    rng_seed[direction_count]

  /responses
    mean[direction_count, RGB]
    variance[direction_count, RGB]
    replica_mean_a / replica_mean_b
    sample_count[direction_count]
    valid, event_flags, reference_pdf
```

一个 query group 固定一个 `(state, surface sample, wo)`，并携带 `direction_count` 个 `wi`。不同 group 的方向值和 proposal 可以不同，但一个文件内的方向数量固定，以保证 HDF5 高效随机读取和 batch 化。

## 5. 查询与响应语义

方向位于记录的局部几何 frame，均为指向远离表面的单位向量。`position_kind` 为：

- `constant`：材质无空间坐标；
- `uv`：`uv + uv_dx/uv_dy` 定义纹理位置和 footprint；
- `surface-point`：为以后需要真实表面点的 reference 保留。

持久化响应统一为：

```text
RGB f(wo, wi) × |dot(Ns, wi)|
```

其中 `Ns` 是 reference 实际使用的 shading normal；MaterialX 可由 normal map 和 footprint 得到它。OpenPBR 的透射方向因此同样使用绝对余弦。运行时公共 `evaluate()` 仍返回不含余弦的 `f`，模型输出参数化必须自行声明。

`proposal_pdf` 描述采集 `wi` 的分布，`solid_angle_weight` 描述当前离散积分权重，二者不能互相替代。固定 probe、均匀采样、microfacet/peak proposal 和自适应 query 都通过这两个字段保留真实语义。

版本化 E0 mixture `ncls.e0-peak-grazing-mixture@1` 是按 `wo` 构造的 uniform + 多尺度反射 peak + grazing 混合分布；完整球面时另含透射 peak。它使用可计算的归一化 PDF 和 `1/(N p)` 权重，目标是诊断 peak、掠射与透射覆盖，不预先宣告为最终训练分布。默认 uniform profile 和该 mixture 都对不同 state split 使用确定性独立方向；文件仍须由 supervision audit 检查实际方向哈希，不能只相信 profile 名称。

`rng_seed` 由 provider 返回，必须是 reference 实际执行该 query 使用的随机流 seed，writer 不得根据 query 行号再合成。确定性 reference 写 0；LayerStack 当前按 `(state, wo)` 使用一个 query-group seed，再在 shader 内结合 `wi` 索引派生随机流，因此同组各方向记录相同的 seed。

## 6. 状态、split 与原生语义

`native_payload` 由 `payload_offsets/payload_blob` 保存，解释方式只由 `native_schema_id` 选择。公共 reader 将其视为 opaque bytes。LayerStack learning adapter 可以解码 `MaterialProgram`；其他 family 可以拥有各自 adapter，但 response-only reader 不加载任何 adapter。

`split_group_id` 是不可跨 split 的最小语义单位：

- LayerStack：同一结构 family 及其编辑状态；
- MERL：同一物理样本；
- OpenPBR：同一原生资产及其派生编辑；
- MaterialX：同一图、纹理资产及其 crop/mip/query。

状态表保存 split，所有引用该 state 的 query 自动继承。训练只能读取 train，checkpoint 选择只能读取 validation，held-out test 由独立评测命令读取。

## 7. 统计与有效性

所有 response 数组在磁盘上使用 float32，`sample_count` 使用 uint32。随机 reference 保存合并后的逐样本总体方差和两个独立 replica mean：

```text
standard_error = sqrt(max(variance, 0) / max(sample_count, 1))
```

确定性 reference 使用零方差、`sample_count=1`，但仍写同一字段。`valid` 标明 query 是否有定义，`event_flags` 区分反射/透射等事件，`reference_pdf` 在 reference 暴露方向 PDF 时保存其值。

## 8. 完整性和写入规则

- writer 只写 `<target>.tmp`，完成 shape/语义校验与内容哈希后原子替换目标 HDF5；
- `dataset_id` 覆盖根语义属性和三个 group 的全部固定 dataset，包括字符串和 payload bytes；
- reader 检查格式版本、必需字段、shape、索引范围、split、payload offsets 和全部方向归一化；
- 不提供 resume、拼接旧 shard 或旧字段兜底。中断后删除临时文件并从锁定配置重跑；
- 相同 query 配置不等于相同 Monte Carlo bit pattern；可复现身份还包括 reference implementation hash、source hash、seed 和 Falcor/Slang 版本。

## 9. Learning 边界

`ReferenceQueryStore` 只提供公共 query/response，因此可用于任何 family 或混合 family 数据集。当前可部署 LayerStack baseline 使用显式的 `LayerStackReferenceStore` 解码原生 payload；这属于该模型的 source adapter，不属于数据合同。

任何新模型若只学习 `(state identity, query) → response`，都不应修改公共 reader。若 compiler 要读取原生图、参数或资源，则为对应 source family 注册 encoder/adapter，并继续以 HDF5 中的 state identity、payload 与 hash 作为训练样本身份。
