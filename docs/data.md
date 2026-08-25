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

### P1 代表性子语料

P1 先使用版本化 selection [`layer-stack-p1-v1.selection.json`](../configs/corpus/layer-stack-p1-v1.selection.json)。它只从 source-train 中冻结 30 个 state，按 `W/G/S × 无 M/有 M` 六个 strata 各取 5 个；这样 P1 可以读取同一 state 的 train/validation/test 方向而不提前污染 P2/P3 的 G2/G2s source holdout。

selection manifest 使用 `reference-corpus` v2：它同时内嵌完整 LayerStack v1 CorpusPlan 与精确 state selection/hash，仍要求每个入选 state 具备五种 query role、方向不跨 role 重合、全部 shard hash 有效。生成和采集命令为：

```powershell
conda run -n neural-shading python -m ncls.cli data plan-corpus `
  --config configs/corpus/layer-stack-v1.json `
  --selection configs/corpus/layer-stack-p1-v1.selection.json `
  --shard-root data/reference-responses `
  --output artifacts/corpus/layer-stack-p1-v1-plan.json

.\scripts\run_falcor_python.ps1 -m ncls.cli data collect-corpus `
  --config configs/corpus/layer-stack-v1.json `
  --selection configs/corpus/layer-stack-p1-v1.selection.json `
  --shard-root data/reference-responses `
  --output artifacts/corpus/layer-stack-p1-v1.json
```

P1 子语料不是降低单 state 的方向密度；它只减少 state 数量。dense audit 与逐 state 16,384 方向晋升规则保持不变。若某个 state 在 validation/test 的基础 262,144 合并样本上仍未通过 query-group relative SE 上限，必须把可复现的 state ID、适用 `query_roles`、更高的合并样本上限和基于实测的最终 group 上限登记到 `reference_budget.state_sample_promotions`；晋升只作用于声明的排名 role，不把 train、diagnostic role 或整个 corpus 的采集成本无条件翻倍。

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

LayerStack reference 的路径深度上限由 CorpusPlan 固定为 64。validation/test 主 response 使用双 replica 自适应预算：从 1,024 个合并样本开始，目标 relative SE p95 为 0.04，单 query group 上限为 0.10，基础合并样本上限为 262,144。train 拥有数量级更多的 peak-aware 方向，噪声无偏且逐 query 保存 SE，因此使用目标 0.06、最终 group p95 上限 0.25、最大 262,144；原 0.20 上限在困难 state 上实测为 0.243，继续把 19.6 万方向整体扩样的收益不足。train 不读取放宽后的数据作为排名 GT。每个 GPU dispatch 不超过 4,096 条 query；达到 state/role 对应预算仍超过上限时立即终止该 shard。

adversarial probe 与 dense slice 不参与训练、checkpoint 选择或 test 主排名。首轮 P1 在 state `bd6de2e…672db` 上实测：adversarial 的基础预算 p95 为 0.129，dense 即使 524,288 样本仍为 0.180；继续套用 0.10 会把一个诊断 state 推到约 200 万样本。两种 diagnostic role 因此独立使用目标 0.08、group p95 0.50 仅作报告参考、最大 262,144 合并样本，并完整落盘 variance/sample count。0.50 最初来自四层 diffuse state `17960657…fe54a` 的 adversarial cap 实测 p95 0.479239；随后 sheen dense 又实测到 0.765555，证明诊断主 response 也不应由任意阈值拒绝 shard。因此 adversarial/dense 达到 cap 后无条件落盘，继续提升诊断样本不会改善训练或排名结论。随后 state `bd6de2e…672db` 的正式 test 主 response 在 262,144 处为 0.155，确认排名 GT 需要晋升；按平方根估算需要约 2.4× 样本，因此它的 validation/test 上限提升到 1,048,576，最终 group 门仍为 0.10。四层 diffuse state `17960657…fe54a` 的 test 在 262,144 处最差 group p95 为 0.656138，而同 state dense 的 group p95 median 只有 0.0285、最差为 0.315；这说明是少数 view 的长尾，不是整 state 失真。该 state 在 1,048,576 的两次采集中最差 group p95 分别为 0.464690 和 0.510445，没有呈理想平方根收敛；validation/test 专属最终门因此定为 0.60，为已观测波动留余量且不再把预算推到 4M。它的 peak-aware train 在 262,144 时最差 group p95 为 0.673502，因此 train 单独登记 0.75 最终门但不增加 cap。sheen state `4ebd9258…cb1e7` 的 test 与 validation 在 262,144 时分别为 0.365615、0.545773，因此 validation/test 单独登记 0.65、train 预先登记 0.75，cap 都不增加。其他 train state 仍保持 0.25，其他 validation/test state 仍保持 0.10。reference SE 只保留为诊断，不临时改成训练优化目标。

reciprocal pair 只服务 source-aware reciprocity scorecard，不进入训练、方向 L1 或能量主指标。交换 grazing 方向会让 Monte Carlo 方差远高于原查询；P1 实测即使把 state 提升到 524,288，reciprocal group p95 仍曾达到 0.337，若强行套用主监督的 0.10 门需约 600 万样本。原 validation/test 配置在 262,144 cap 下也分别观测到 p95 0.535 和 0.830245，继续加码没有与主指标相称的收益。因此 validation/test/adversarial/dense reciprocal 统一使用目标 0.20、最大 65,536 合并样本、group p95 0.999 只作为报告参考线。32,768 与 65,536 的诊断 p95 曾分别实测为 0.938、0.866，四层 diffuse adversarial 在 65,536 时又达到 0.932538，未呈理想平方根收敛；65,536 仍比 262,144 减少 4×，并避免 dense 的 32,768 个交换查询产生约 80 亿样本支出。train reciprocal 更不参与任何训练目标，实测硬导体 state 按 65,536 上限运行 11 分钟、工作集增长到约 35 GB，已经会直接阻塞外观模型迭代；因此 train 单独使用目标 0.50、最大 4,096、报告参考线 0.999，只保留可审计的低成本诊断。4,096 样本时两个 state 又分别实测 p95 0.991491 和 0.999544，证明任何贴近 1 的硬门仍会拒绝有效主 response。reciprocal 现在达到 cap 后无条件落盘，所有 role 都完整保存 variance/sample count；高噪声 reciprocal 不得支撑模型质量结论。primary reference 的 role/state-specific 门仍是硬失败。早期已采 shard 使用更严格预算，仍是兼容的高质量数据；不因诊断预算放宽而重采。

每条 `(wo, wi)` 还采集 canonical reciprocal pair：反射交换为 `(wi, wo)`，透射交换并同时翻转两方向，继续保证 `wo.z > 0`。HDF5 保存 reciprocal mean、variance 和 sample count，使 quality harness 可以计算 source-aware reciprocity deviation，而不假设所有 reference 本身严格互易。

## 扩展其他材质族

公共 `ReferenceProvider` 只暴露原生 state、surface query、direction query 和 reference response。新增材质族需要实现自己的 state 解析、难度分级和 proposal adapter，再增加对应 CorpusPlan；不得把源材质预先反演成 LayerStack。

P0 只冻结并完整生成 LayerStack。MERL 与 OpenPBR 接入沿用同一 shard/manifest 合同；MaterialX 的空间密度升级留到 spatial 阶段。迁移前 HDF5、profile、gate 和转换器不属于当前工作区接口；需要考察历史结果时使用对应 Git 提交。
