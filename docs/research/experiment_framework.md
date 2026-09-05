# 实验框架：在线查询、拟合协议与检验流程

本文是当前研究的权威流程框架。它冻结source、online query、泛化轴、预算和评测规则，使候选方法在同一reference语义下比较。候选结构见[`model_candidates.md`](model_candidates.md)，reference query合同见[`../data.md`](../data.md)与[`../contracts/reference_query.md`](../contracts/reference_query.md)。

## 0. 框架定位

- **online reference优先**：训练batch由canonical reference在GPU上即时产生，不导出、读取或重放HDF5/shard/corpus。
- **原生source语义**：冻结的是source locator解析结果、`SourceSnapshot`、reference program与query recipe，不把不同材质族改写成LayerStackIR。
- **typed route**：evaluator与sampler使用不同batch合同；NVIDIA evaluator直接拟合线性`f`，learned sampler用自身proposal构造forward-KL。
- **基准先冻结**：formal前冻结source split、query recipe、seed、训练预算、checkpoint选择规则和评测指标。修改任一身份后产生新run，不能覆盖旧结果。
- **部署预算**：每个候选登记`C_eval`、`C_prepare`、`B_shared`、`B_asset`、state bytes与实测时间。硬约束见[`.trellis/spec/project/method-constraints.md`](../../.trellis/spec/project/method-constraints.md)；超软线结果可作capacity diagnostic，但不作为默认部署候选。

有限数据、模型和预算下的结果使用`optimized-code control`、`high-capacity teacher`或`best observed candidate`等限定表述，不称为“上界”。

## 1. Source 与 query 冻结

每个run identity至少覆盖：

1. source family/version与每个material locator；
2. locator解析后的`source_snapshot_ids`；
3. canonical reference descriptor/implementation identity；
4. method/source adaptation identity；
5. online query recipe、typed routes及各自seed stream；
6. training config、method implementation与checkpoint schema。

正式训练只保存上述identity和RNG state，不保存response batch。resume必须恢复各route generator与request count，继续同一确定性query stream。

### 1.1 两类training route

| route | 输入分布 | reference操作 | batch语义 |
|---|---|---|---|
| `reference-evaluator` | method recipe定义的surface、LOD、`wo/wi` | `prepare/evaluate` | conditioning、`wi`、`target_f` |
| `method-sampler` | 独立surface、LOD、`wo`与`sample_u` | 无 | conditioning、`sample_u` |

source `sample/pdf`仍必须由所有reference实现，用于path tracing、sample→PDF/weight恒等式和proposal审计；它们不是NVIDIA sampler监督route。

### 1.2 无效方向与随机reference

- material-local normal、transmission domain或其他source原生边界可令query返回`valid=false`。producer在GPU上压实valid行并继续补采，记录候选/拒绝/轮次数；不把invalid变成零GT。
- LayerStack random walk等stochastic reference通过`evaluation_samples`重复同一`evaluate`合同并平均`f`。不做亮度clamp、p95过滤或合法窄峰删除。
- 需要更低reference方差时，先以独立probe冻结`evaluation_samples`和预算，再修改recipe identity；不能根据正式结果临时加样本直到过门。

## 2. NVIDIA functional reproduction recipe

当前formal MaterialX配置冻结：

- evaluator与sampler各65,000条、独立seed stream；
- 300,000 global steps，100,000步从encoder bootstrap切换到latent finetune；
- evaluator使用half/difference方向和log-L1，输出/target均为线性`f`；
- sampler从当前GGX9 learned proposal取样，target density为`luminance(f)·|cosθi|`的detached估计；
- Adam、全局cosine schedule、前20,000步directional mollification，以及MaterialX spatial/LOD filtering由`configs/learning/nvidia-rta2024-materialx-formal.json`精确锁定。

smoke/profile可以缩小batch、step和filter sample，只证明协议、显存与生命周期；不得沿用formal结论身份。

## 3. 泛化合同

### 3.1 参数式材质族

| 轴 | 含义 | 冻结方式 |
|---|---|---|
| G1 | 同一state未见的`(wo,wi)` | 独立evaluation seed/query recipe |
| G2 | 同结构family未见连续参数state | source snapshot按family分层holdout |
| G2s | 未见层数/拓扑 | 完整结构family holdout，仅LayerStack等结构式source适用 |

### 3.2 资产式/测量式材质族

MERL、MaterialX与纹理OpenPBR中，一个资产本身就是压缩对象。主轴是G1和工作流稳健性W：同一pipeline、同一组超参能否批量处理冻结资产集合。跨资产shared decoder的收益表述为压缩摊销，不表述为零样本参数泛化。

### 3.3 跨族

跨族证据是所有family共用source/reference registry、dispatcher、typed batch、runner与评测入口；method没有相应source adaptation时必须fail closed。统一基础设施不等于一个method自动支持所有native feature domain。

## 4. 拟合与预算协议

| 路径 | 内容 | 记录 |
|---|---|---|
| 梯度训练 | 联合优化decoder、latent、encoder/compiler | step、query work units、optimizer/schedule、seed |
| direct fit | 聚类、最小二乘、VQ等无梯度/封闭解 | wall-clock、确定性配置、online query预算 |
| 混合 | encoder/direct fit初始化后有界精调 | 两阶段identity与各自预算 |

快速档只做≤30分钟smoke，不进正式比较；标准档使用冻结配置与共同deterministic seed，差距接近或轨迹不稳时才根据证据追加seed；冲刺档只用于已经晋级的候选。忠实实现但quality较低属于empirical outcome，不自动扩大预算或改结构。

## 5. 输出语义与指标

所有候选的公共输出都是线性RGB `f`。target transform只属于方法内部参数化；运行时不得从`f·cos`除以接近零的cosine恢复`f`。

指标分四层：

1. **hard sanity**：identity/split无泄漏、输出finite/nonnegative、checkpoint可恢复、query/batch合同成立；
2. **主指标**：方向域solid-angle weighted normalized L1与半球/球面积分能量相对误差，按source state报告median/p90/p95；
3. **结构scorecard**：log-domain error、峰位/峰高、top-energy recall、reciprocity、按family/难度/state分层；
4. **部署证据**：Python/Slang/package parity、真实GPU时间、显存与viewer capture，每研究阶段收尾执行一次。

bootstrap以冻结source state为重采样单位，≥1,000次；只有matched source/query/budget对照才能支撑“X优于Y”。observed quality参考线用于排序，不自动成为任务kill gate。

## 6. 评测数据流

`ncls learn evaluate <config> <checkpoint>`重新解析同一source locator并构造online producer，验证config、source snapshot、reference与query stream identity均与checkpoint一致。evaluation route使用独立名称/seed，不消费训练RNG；输出只写`artifacts/`报告。

若某次test结果驱动后续设计，下一阶段更换该考核轴的evaluation seed/split identity。单次诊断可固定一小组方向和source state，但它是versioned artifact，不是训练reader或新的offline GT。

## 7. 阶段路线

| 阶段 | 回答的问题 | 主要证据 |
|---|---|---|
| P0 | online source/query与评测协议是否完整 | 五family dispatcher、typed batch、resume、smoke |
| P1 | 哪个表示在部署预算内表达最难方向结构 | LayerStack G1 matched runs |
| P2 | shared decoder与状态latent如何取得 | LayerStack G2，autodecoder/encoder/direct fit对照 |
| P3 | pure source compiler与target-visible路径差距 | G2/G2s、bounded refinement |
| P4 | 资产式工作流是否稳健 | MERL/OpenPBR/MaterialX W分布 |
| P5+ | spatial、matched sampler、环境积分 | 胜出evaluator的扩展实验 |
| D | 每阶段部署收尾 | ScatteringPackage、Slang parity、viewer与成本 |

## 8. 实验记录

`docs/research/experiment_log.md` 每个正式 run 一行，记录 run identity、source/query recipe、预算、seed、两个主指标、成本、结论与 `outputs/<config-stem>/<run-id>/` 路径；独立研究临时报告可位于 artifacts。迁移前 HDF5/corpus 结果只作为明确标注的历史证据保留，不与当前 online run 作 matched 比较，也不为其恢复 reader/config。
