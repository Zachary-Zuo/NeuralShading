# Reference 语料

正式数据入口是 `CorpusPlan → reference shard → reference-corpus manifest`。`CorpusPlan` 把材质状态分布、采样密度、split、reference 噪声预算和 shard 策略放在一份可读配置中；采集器不再接受一组可任意拼接的历史 profile 参数。

当前第一份完整配置是 [`layer-stack-v1.json`](../configs/corpus/layer-stack-v1.json)。它生成 28 个结构 family、每族 10 个连续参数状态，共 280 个 state。每个结构 family 的参数由确定性 Latin hypercube 覆盖；4 个 family 整体留作 G2s，其余 family 各留 1 个 validation state 和 1 个 G2 test state。最终 source-test 共 64 个 state。

## 采样密度

train 查询按 response 难度选密度：

| 分级 | `wo` | 每个 `wo` 的 `wi` | proposal |
|---|---:|---:|---|
| W | 48 | 512 | 60% uniform、25% 三尺度反射峰、15% grazing |
| G | 64 | 1,024 | 40% uniform、40% 三尺度反射峰、20% grazing |
| S | 96 | 2,048 | 30% uniform、50% 三尺度反射峰、20% grazing |

`M`（移动峰或多峰）使用 S 密度。LayerStack provider 会先对同一 `wo` 做独立的 4,096 方向 reference probe，取实测 response 峰位，再围绕该中心生成三尺度 vMF proposal；固定的“可能峰位 patch”不属于当前合同。

`T`（含透射）在 train 中追加 32 个临界区相关的 canonical `wo.z > 0`，方向总数乘 1.5。整球 mixture 按顺序包含 uniform、反射峰、25% 透射峰、10% 临界角带和 grazing；实际权重、解析 PDF 与 `1/(N·p)` 都逐 query 落盘。

独立评测 role 固定为：validation `16×256` uniform、test `24×512` uniform、adversarial probe `16×128` 定向 mixture、dense slice `4×8192` uniform。dense slice 的审计若证明某个 state 的峰邻域分辨率不足，只把该 state 提升到 `4×16384`，并为它生成独立的矩形 shard；这两个值之外的密度不属于 v1。

## 生成与续采

### 单 state 冒烟采集

正式全量采集前，可以用可读的结构 family 与族内 state index 采一个 v5 shard。该文件使用 CorpusPlan 中对应 role 的正式密度和 adaptive reference budget，但只用于验证 reference、统计量、reciprocal pair 与 HDF5 链路，不冒充完整 corpus：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls.cli data collect-state `
  --config configs/corpus/layer-stack-v1.json `
  --structure-family layers-01-diffuse-variant-00 `
  --state-index 3 `
  --role validation `
  --output data/reference-responses/smoke/layer-stack-v1-validation.h5

conda run -n neural-shading python -m ncls.cli data validate `
  data/reference-responses/smoke/layer-stack-v1-validation.h5
```

同一路径已存在时采集器会拒绝覆盖。单-state shard 可以直接交给 learning reader 做工程诊断；正式训练和结论仍必须使用通过完整验证的 `reference-corpus` manifest。

### 完整 corpus

先只生成计划，检查 state、split、密度和输出路径：

```powershell
conda run -n neural-shading python -m ncls.cli data plan-corpus `
  --config configs/corpus/layer-stack-v1.json `
  --shard-root data/reference-responses `
  --output artifacts/corpus/layer-stack-v1-plan.json
```

LayerStack reference 依赖 Falcor Python，正式采集使用锁定环境入口：

```powershell
.\scripts\run_falcor_python.ps1 -m ncls.cli data collect-corpus `
  --config configs/corpus/layer-stack-v1.json `
  --shard-root data/reference-responses `
  --output artifacts/corpus/layer-stack-v1.json
```

采集按结构 family、query role 和实际密度拆成矩形 HDF5 shard。每个 shard 只有一个 `direction_count`，所以训练 batch 不需要 padding。已有文件只有在 HDF5 内容 hash、state 集合和计划完全一致时才会续用；不一致文件会直接报错，不会被覆盖。

完成后验证 manifest 与全部 shard：

```powershell
conda run -n neural-shading python -m ncls.cli data validate-corpus `
  artifacts/corpus/layer-stack-v1.json
```

manifest 内嵌完整 CorpusPlan、计划 hash、每个 shard 的 dataset/hash、原始与 reciprocal reference 总样本支出、wall-clock 汇总。`corpus_id` 由这些语义内容计算，不依赖文件名或长串拼接 ID。

首次完成 `4×8192` dense slice 后运行分辨率审计：

```powershell
conda run -n neural-shading python -m ncls.cli data audit-dense `
  artifacts/corpus/layer-stack-v1.json `
  --output artifacts/corpus/layer-stack-v1-dense-audit.json
```

审计只对 top 1% 方向承载至少 10% 能量的集中响应检查峰值最近邻间距；其 p95 超过 2° 的 state 会进入 `promote_state_ids`。把报告给出的 `corpus_plan_update.value` 写入 `sampling.dense_promotions`，重新生成计划并续采。晋升 ID 必须是当前计划真实存在的 state；采集器会拒绝未知 ID，且不会把 8,192/16,384 方向混进同一 shard。

## Reference 噪声与 reciprocal pair

LayerStack reference 的路径深度上限由 CorpusPlan 固定为 64。采样使用双 replica 自适应预算：从 1,024 个合并样本开始，全局目标 relative SE p95 为 0.04，单 query group 上限为 0.10，合并样本上限为 262,144。每个 GPU dispatch 不超过 4,096 条 query；达到预算仍超过单组上限时立即终止该 shard，而不是把不合格数据写入语料。

每条 `(wo, wi)` 还采集 canonical reciprocal pair：反射交换为 `(wi, wo)`，透射交换并同时翻转两方向，继续保证 `wo.z > 0`。HDF5 保存 reciprocal mean、variance 和 sample count，使 quality harness 可以计算 source-aware reciprocity deviation，而不假设所有 reference 本身严格互易。

## 扩展其他材质族

公共 `ReferenceProvider` 只暴露原生 state、surface query、direction query 和 reference response。新增材质族需要实现自己的 state 解析、难度分级和 proposal adapter，再增加对应 CorpusPlan；不得把源材质预先反演成 LayerStack。

P0 只冻结并完整生成 LayerStack。MERL 与 OpenPBR 接入沿用同一 shard/manifest 合同；MaterialX 的空间密度升级留到 spatial 阶段。迁移前 HDF5、profile、gate 和转换器不属于当前工作区接口；需要考察历史结果时使用对应 Git 提交。
