# Reference shard v5 合同

`reference-shard` v5 是 CorpusPlan 生成的矩形 HDF5 单元。一个 shard 固定一个结构 family、一个 query role 和一个 `direction_count`；完整语料由 `reference-corpus` manifest 组合。v1 表示完整 CorpusPlan，v2 在同一计划上额外冻结版本化 state selection。布局机读定义见 [`reference_dataset_v5.layout.json`](../../src/ncls/data/schemas/reference_dataset_v5.layout.json)、[`reference_corpus_v1.schema.json`](../../src/ncls/data/schemas/reference_corpus_v1.schema.json) 和 [`reference_corpus_v2.schema.json`](../../src/ncls/data/schemas/reference_corpus_v2.schema.json)。

## 顶层属性

- `format_name = reference-shard`，`format_version = 5`；
- `response_measure = rgb-bsdf-times-absolute-shading-normal-light-cosine`；
- `color_model = linear-srgb`；
- `sampling_name` 是可读采样方案名，例如 `peak-aware-v1`；
- `generation_config_json` 保存解析后的单 shard `CollectionConfig`；
- `provider_metadata_json` 保存 reference 实现 hash 和 provider 配置；
- `dataset_id` 是全部语义属性与 dataset 内容的 SHA-256。

文件不提供旧格式探测或转换。`ReferenceDataset.open()` 只接受这一合同。

## `states/`

每个 state 保存原生 payload 与身份字段：`state_id`、`family_id`、`reference_id`、`asset_id`、`split_group_id`、`native_schema_id`、`source_uri`、`source_sha256`、`parent_state_id`、`split`、`payload_offsets/payload_blob`。

P0 新增四个评测字段：

- `structure_family_id`：参数结构或资产工作流分组；
- `difficulty_class`：`W/G/S/unclassified`；
- `difficulty_tags_json`：`T/M` 的集合；
- `evaluation_cohort`：`train/validation/g2/g2s/workflow`。

同一个 `state_id` 在不同 role shard 中的全部 metadata 必须逐字段一致。

## `queries/`

每行是一个 `(state, surface, wo)` query group，含固定数量的 `wi`：

- state/role：`state_index`、`query_role`；role 为 `train/validation/test/adversarial_probe/dense_slice`；
- surface：`position_kind`、`position`、`uv`、`uv_dx`、`uv_dy`、`geometric_normal`、`geometric_tangent`；
- direction：`wo [3]`、`wi [direction_count,3]`；`wo.z` 必须为正，是否允许负 `wi.z` 由 reference incident domain 决定；
- proposal：`proposal_code`、`proposal_pdf`、`solid_angle_weight`；Monte Carlo 查询满足 `weight = 1/(N·pdf)`；
- provenance：逐方向 `rng_seed`。

不同 role 使用独立 seed 和方向表。manifest 验证会拒绝同一 state 跨 role 的方向 hash 碰撞。

## `responses/`

主查询保存：

- `mean/variance`；
- `replica_mean_a/replica_mean_b`；
- `sample_count`（两 replica 合并样本数）；
- `valid/event_flags/reference_pdf`。

reciprocal paired 查询保存 `reciprocal_mean`、`reciprocal_variance` 和 `reciprocal_sample_count`。它使用 canonical 交换：反射直接交换方向；透射交换后同时翻转两方向。quality-v1 用交叉乘 cosine 的形式计算 source-aware reciprocity deviation，不在 grazing 方向除以接近零的 cosine。reciprocal 只是带不确定度的 scorecard 诊断，可使用 CorpusPlan 中单独记录的诊断 target、上限和样本预算；train/validation/test 主 response 的硬性 reference 噪声门不得因此放宽。`adversarial_probe/dense_slice` 的主 response 同样属于诊断，可以使用单独版本化的噪声预算，但 variance/sample count 仍必须完整落盘。

## Corpus manifest

`reference-corpus` 内嵌完整 CorpusPlan 和 `plan_sha256`，逐 shard 保存 URI、role、结构 family、难度、state 集合、矩形尺寸、dataset ID、文件 SHA-256、采集状态、耗时及原始/reciprocal reference 样本支出。顶层 `totals` 汇总 wall-clock 与样本支出。v2 还内嵌 `corpus-selection` v1 与 `selection_sha256`；验证器会从基础 CorpusPlan 重新枚举并确认精确 state 集合和 shard 布局，而不是信任 manifest 自报 selection。

验证要求：全部 shard complete；URI、shard ID 唯一；文件与语义 hash 一致；实际 state/role/尺寸与计划一致；每个 state 同时具备五个 query role；source-test state 数满足计划；state metadata 不跨 shard 漂移。
