# Neural evaluator 建模、编译器训练与评测

训练、direct fit、独立评测和部署导出是项目的长期功能层。具体 representation、网络结构、latent 获取方式和 loss 通过版本化配置与注册实现替换；更换研究候选不会删除或重定义这些生命周期。

公共 runner 现在只依赖 `ncls.learning-pipeline@2`：它从 registry 取得 response reader、dataset partition policy、source adapter、feature/target transform、representation/model、latent inference、compiler、loss、metric suite 和 exporter 的版本化身份，再执行训练、validation、checkpoint 与独立评测。candidate-specific 实现负责把公共 response batch 变成自己的模型输入；runner 不再导入 LayerStack 或某个 backend 的预测函数。机器可读 pipeline 合同位于 `src/ncls/learning/schemas/learning_pipeline_v2.schema.json`；新训练配置使用 `training_config_v5.schema.json`。

TrainingConfig v5 把 `constant` 或 `cosine` 学习率策略、以及最终学习率相对初值的比例写入 manifest 与 checkpoint hash。它来自多界面 residual 在固定学习率后段振荡的实际证据，不是 architecture-specific 分支。v4 的语义保持为 constant，读取时不会把 v5 字段混入其 canonical payload，因此既有 run 仍可按原 hash 追溯；新配置不得再使用 v4。v4 兼容读取只用于复现实验，删除条件是所有被实验索引引用的 v4 配置和 checkpoint 已迁入带可执行环境的只读归档，且没有当前调用方再加载它。

当前正式 registry 同时包含 E1 的 linear、q90-scale `log1p` 和 train-only standardized `log1p` dense evaluator，以及部署回归项 `legacy-ltc-k2-p1-deployment-regression@1`。E1 pipeline 共用材质无关的 `ReferenceQueryStore`，固定一个 source state 后按 query role 划分 train、validation、test；standardized 版本的 channel scale/mean/std 只由最终 train query 拟合，并在训练时加入固定权重的互易性约束。q90 版本保留为 target-transform 对照，不因 standardized 版本存在就改写其既有语义。部署回归项只负责保留既有 compiler、MethodBundle、Slang 和 viewer 生命周期，不进入目标 neural evaluator 排名。legacy 适配的删除条件仍是没有 checkpoint/export/viewer 调用依赖其专用 feature/prediction 入口。

LayerStack 另注册 `analytic-core-neural-residual-standardized-e1@1` 和带能量/shape 联合监督的 `analytic-core-neural-residual-energy-shape-e1@1`：它们用 source adapter 精确求值顶层界面的直接散射，把剩余 response 以只由 train query 拟合的 standardized `asinh` 残差交给同一小型 neural evaluator。analytic core 是显式候选组成部分，不是公共输出词汇，也不能被其他 source family 强制提供。每个 run 额外报告 core-only normalized L1，防止单界面案例中“残差几乎为零”的结果被误写成多层 neural residual 已经成立。多界面容量实验中，energy/shape + GELU + cosine 的 64,603 参数候选通过冻结数值 gate；这仍只是 optimized-latent 单材质上界，不能据此宣称 shared decoder 或 source compiler 已成立。

E1 还注册了可复现淘汰用的 `plane-factorized-small-mlp-energy-shape-e1@1` 与 analytic-residual 变体。v1 每个 texel 只存一个 RGBA-width feature，prepare/evaluate 固定读取 4/20 个 plane texel；额外成本不会伪装成 MLP MAC。32² direct/residual 和 16² residual 都未通过 held-out query gate，因此 raw-direction 六成对 plane v1 已停止作为当前 Pareto 候选。保留这些版本化 pipeline/config 是为了复现实验，不代表 runner 维护第二套训练路径；删除条件是淘汰报告与 checkpoint 已进入可执行只读归档，且 registry/config 不再被复现入口引用。

E2 的第一条公共路径是 `dense-latent-shared-small-mlp-energy-shape-e2@1`。它复用材质无关的 response reader、方向编码、train-only standardized `log1p` transform、energy/shape loss 与 evaluator metric suite，但把单材质参数拆成每个 state 一条 optimized dense latent 和跨 state 共享的 `prepare/evaluate` MLP。模型分别报告单材质 `B_asset`、全部已拟合 latent 的总 bytes、`B_shared`、`C_prepare` 与 `C_eval`；公共 evaluator 同时输出 `by_state`、`by_family` 和 `by_source_split` 分布。该 pipeline 按 query role 划分生命周期，所有 selected state 的 train response 都可见，因此它是 target-visible autodecoder 压缩上界，不承担未见 source state 泛化；descriptor 的 `compiler_id=ncls.none-target-visible-capacity-study@1` 防止把它误写成 E3 source compiler。

LayerStack 的 E2 对照 `analytic-core-shared-neural-residual-energy-shape-e2@1` 使用同一个 latent table、shared decoder、partition、checkpoint 和 metric 生命周期，只把 source adapter 与 target transform 换成显式版本化的 direct-top core + signed standardized `asinh` residual。direct-top adapter 位于 `ncls.learning.source_adapters`，E1/E2 共用，不再从某个 pipeline 私有实现互相调用。它只解释 LayerStack native payload；公共 runner、response reader 与其他 source family 不需要提供层参数。

共享 residual 的首个容量实验暴露了跨 state 共用 transform 的具体缺陷：单界面 sheen 的 analytic core 已在 reference SE 内，而全局 residual scale 仍迫使 decoder 同时表达约 `1e-8` 与 `1e-1` 的残差。`analytic-core-shared-neural-residual-energy-shape-e2@2` 因而只用每个 state 的 train query 拟合各通道 `asinh` scale/mean/std，并把 9 个 float（36 bytes）显式计入每材质 `B_asset`；validation/test 不参与这些统计。旧 `@1` 保留用于复现已有 run。`@2` 同时提高已由 p95 长尾证明过弱的互易性 loss 权重，但不修改冻结 acceptance gate，也不把 target-visible 统计解释成 source compiler 输出。

`ncls.interface.sheen@1` 延续锁定 reference 的 terminator softening，源函数本身不满足严格交换互易。Falcor 固定交换方向 probe 显示两个 E2 sheen state 的 reference reciprocity p95 约为 `1.15/1.55`，而 direct-top core 复现同一数值，不能把忠实 evaluator 的绝对 reciprocity 当成额外模型误差。`analytic-core-shared-neural-residual-energy-shape-e2@3` 因而继续报告绝对 `reciprocity_relative_l1`，同时新增 `source_reciprocity_deviation_relative_l1`：对该单界面 sheen 扣除 reference/core 固有的有符号非互易项，对其余当前 LayerStack 状态仍以零为物理期望。`ncls.e2-shared-evaluator-acceptance@2` 只用后者替换旧 gate 的绝对互易性检查，阈值仍为 `0.05`，其他门槛不变。这是 source 语义适用范围修正，不是把非互易 reference 宣称为互易。

E2 多峰/平顶 response 还要求把 raw argmax 峰位与“峰支持集”分开。正式 reference 的两个独立 replica 在 test/adversarial 上已有约 `2.15°/2.55°` 的 raw peak angle p95；对 95% 峰高平台内两个近等值方向强行要求同一 argmax，会把 Monte Carlo 排序抖动写成模型峰位漂移。`analytic-core-shared-neural-residual-energy-shape-e2@4` 保留 raw peak angle，同时报告模型峰到 target 95% 峰值支持集的最近角距；`ncls.e2-shared-evaluator-acceptance@3` 对后者保持 `2°` 上限，并继续检查 peak ratio 与 top-energy recall。该 pipeline 另在既有 loss 上增加只读 train batch 的 reference-SE floor 项，直接压缩 `model_error_over_reference_standard_error` 长尾；floor 和权重属于版本化 loss，不从 validation/test 拟合。

## 三条路径的边界

Python 侧的工具生命周期仍分成三条路径，但必须记录拟合对象和结论范围：

- `direct-fit`：不经过 feed-forward compiler，直接优化候选表示的参数或 latent；当前基线逐 query group 拟合，它不产生通用编译器。
- `train`：训练共享 evaluator、材质 latent、compiler 或后续 sampler；只读取 train 数据，只用 validation 选择 best checkpoint。
- `evaluate`：从一个不可变 checkpoint 显式评测 validation 或 held-out test；训练循环不会读取 test 指标。

目标方法的小型 MLP 直接实现 `evaluate(wo, wi)`。`prepare()` 获取和过滤 latent，并编码 footprint、局部 frame 与 `wo`，供多个 `wi` 查询复用。当前已注册网络是端到端可部署基线，用来验证训练、导出、Slang compiler 和 viewer 生命周期；它不代表目标 neural evaluator 已经确定。

## 当前建模顺序

在模型结构未确定前，不做多灯、PT 方差或 UE 环境积分的系统 kill test。学习侧先按以下层次推进：

### 1. 单材质 neural evaluator 容量

对一个源材质状态覆盖多个 `wo` 和 `wi`，直接优化小型 evaluator 及其材质 latent，回答“给定网络和 latent 预算能否表达完整方向函数”。候选只覆盖少量可部署设计轴：

- material latent 的维数和精度；
- `wo/wi` 的局部方向编码；
- `prepare` shared trunk 与逐方向 evaluate head 的划分；
- MLP 深度、宽度、激活和输出参数化；
- direct response 与 analytic-core + neural-residual；
- 非负、动态范围、互易性和能量处理。

这个阶段不训练通用 compiler，也不声称未见材质泛化。

第一轮 LayerStack 是常量局部材质，`prepare` 原型可以只编码 material latent、局部 frame 和 `wo`；不能因为接口预留了 UV footprint/mip/LOD，就在没有 spatial supervision 时声称已学会纹理过滤。空间变化阶段再加入 latent texture fetch、mip/LOD 与 footprint，并使用版本化的 spatial query 数据。

### 2. 共享 decoder 与材质专属 latent

在多个材质状态间共享 evaluator 权重，只优化每个材质的 latent。它回答“统一 neural runtime 是否有足够容量”，并把网络共享能力与 compiler 泛化分开。

### 3. Feed-forward compiler

固定或联合微调共享 evaluator，让 compiler 从源材质的原生参数、图或资源产生 latent。train/validation/test 必须按材质 family/state 正确隔离，评测未见参数组合、编辑后的状态和跨源材质族适配。此阶段才回答“能否自动编译和保持编辑工作流”。

### 4. Slang 最小部署

只有 evaluator 在前述局部实验中成立后，才导出共享权重和 latent，验证 Python/Slang parity，并测量 `prepare`、单次 `evaluate`、state/latent bytes。此时解析方法作为 iso-time/iso-byte 对照。

### 5. Sampler 与 integration 扩展

evaluator 固定后再训练匹配的 proposal head。sampler 必须能生成方向并计算同一分布的 PDF；先验证 sample/PDF 与 MIS 语义，再比较固定时间下的方差。环境/面光 integration head 同样在 evaluator 之后研究，并与高样本 evaluator 积分对照。

推荐的逻辑分解是：

```text
h = PrepareTrunk(material_latent, footprint, frame, wo)
f = EvaluateHead(h, wi)

proposal_parameters = SamplerHead(h)
wi = Proposal.sample(proposal_parameters, ξ)
p  = Proposal.pdf(proposal_parameters, wi)
```

`SamplerHead` 可以在 prepare 时预计算并缓存 proposal parameters，也可以在第一次 sampling query 时执行；训练和成本报告必须说明选择。无论放置位置如何，`sample` 与 `pdf` 必须对应同一个 proposal，evaluator 与 sampler 必须共享同一个 `h` 和 material latent。

## TrainingConfig

所有新实验超参数先解析为 `ncls.training-config@5`，并将完整 JSON、pipeline contract、最终 dataset selection、train-only fitted state 与各自 SHA-256 写入 run。`dataset_selection` 只负责从 H5 选择明确的 state、asset 或 family，不改变落盘 split；E1 单材质 pipeline 要求选择后恰好只有一个 source state。E2 autodecoder 则为每个 selected state 建立一个有版本化 state-ID 映射的 latent slot；transform 和 slot 的优化只读取 train query，validation 只选 checkpoint，test 不进入训练或选择。它读取 source test state 的 train query 时必须在报告中标作 target-visible compression upper bound，不能作为 source-held-out 结果。最小示例：

```json
{
  "pipeline_id": "legacy-ltc-k2-p1-deployment-regression@1",
  "research_stage": "deployment-regression",
  "model_parameters": {"width": 64},
  "dataset_selection": {},
  "steps": 10000,
  "batch_size": 256,
  "learning_rate": 0.0003,
  "learning_rate_schedule": "constant",
  "final_learning_rate_fraction": 1.0,
  "weight_decay": 0.00001,
  "gradient_clip": 5.0,
  "validation_interval": 250,
  "checkpoint_interval": 250,
  "max_validation_query_groups": 4096,
  "seed": 20260822,
  "device": "cuda",
  "deterministic": true,
  "selection_metric": "relative_l1.median",
  "schema_name": "ncls.training-config",
  "schema_version": 5
}
```

运行：

```powershell
conda run -n neural-shading ncls learn train `
  --dataset data\reference-responses\layer-stack-v4.h5 `
  --run artifacts\runs\legacy-ltc-k2-p1-001 `
  --config configs\legacy-ltc-k2-p1.json
```

每个 run 包含：

```text
run/
  training_config.json
  run_manifest.json
  validation_history.json
  tensorboard/
  checkpoints/
    best.pt
    best.pt.sha256
    last.pt
    last.pt.sha256
```

checkpoint 保存 architecture、representation、feature contract、dataset ID、模型/optimizer/RNG 状态、只由最终 train query 拟合的 transform/codebook 状态及 validation 证据。loader 先验证 sidecar hash，再验证 training config、pipeline contract 和 fitted state 的内容哈希；validation/test 不会重新拟合这些状态。

## E0 supervision audit

训练前先对每个 H5 运行只读 audit；它只使用公共 `ReferenceDataset`，不会解码任何源材质 payload：

```powershell
conda run -n neural-shading ncls learn audit `
  --dataset data\reference-responses\layer-stack-v4.h5 `
  --output artifacts\research\supervision-audit\<dataset-id> `
  --gate configs\research\e0-supervision-gates-v6.json
```

输出包含 `audit.json`、中文 `report.md`、只由 source train × query train 拟合且带内容哈希的 `target_transform_statistics.json`，以及可选的 `gate_result.json`。`ncls.supervision-audit@6` 检查 source state/asset split 泄漏、四种 query role 的完整性与方向表复用、版本化 query profile、按 query role 的 peak/掠射/透射覆盖、reference standard error 与 replica 差异、response 长尾、积分能量和 top-energy 集中度。对 UV 数据，它从落盘的 `uv/uv_dx/uv_dy` 重算 footprint 尺度与旋转，并检查 U/V seam 两侧配对，不根据 proposal 文本猜测 spatial 覆盖。peak 位置按原始 response 幅值确定，top-energy 仍使用正确的积分权重，二者不能混用；2° peak gate 只验收 top-1% 积分能量占比至少 0.1 的集中 query，并要求至少存在 4 个这类 query，宽漫反射的任意离散最大值只保留为诊断。noise 除全局分位数外，还按 state、source split、query role 和积分能量分档，并把最坏 query-group relative SE 与 replica 差异作为独立 gate；正式采样预算应由这些分档决定，不能因总体 p95 较低就掩盖少数高方差状态，也不能无差别提高全部 H5 的样本数。validation/test/adversarial 不参与 transform scale、均值、方差或 codebook 统计。

## TensorBoard

训练记录 `train/loss`、gradient norm、learning rate，以及 validation loss 和各 evaluator 指标的 mean/median/p90/p95。指标包括立体角加权 normalized L1、linear/log error、积分能量、峰值比例、峰位角、top-energy recall、相对 reference standard error、互易性、finite 与非负比例；不会写入 `test/*` 或 adversarial tag。

```powershell
conda run -n neural-shading tensorboard `
  --logdir artifacts\runs\legacy-ltc-k2-p1-001\tensorboard
```

逐样本直接拟合也在其输出目录下记录 `tensorboard/`，便于比较不同表示与优化设置。

## 显式 held-out test

best checkpoint 确定后，单独执行：

```powershell
conda run -n neural-shading ncls learn evaluate `
  --dataset data\reference-responses\layer-stack-v4.h5 `
  --checkpoint artifacts\runs\legacy-ltc-k2-p1-001\checkpoints\best.pt `
  --split test `
  --output artifacts\runs\legacy-ltc-k2-p1-001\test_metrics.json
```

评测拒绝 dataset ID 或 feature contract 不一致的 checkpoint。test 结果是独立文件，不回写训练选择记录。

对抗性 query 同样只能在 checkpoint 固定后显式运行，并写到不同文件：

```powershell
conda run -n neural-shading ncls learn evaluate `
  --dataset data\reference-responses\layer-stack-e1-v4.h5 `
  --checkpoint artifacts\runs\e1-dense-001\checkpoints\best.pt `
  --split adversarial_probe `
  --output artifacts\runs\e1-dense-001\adversarial_metrics.json
```

`adversarial_probe` 不参与 checkpoint 选优。viewer capture 又是另一类独立证据，不能用这里的 query 指标代替。

E1 正式 run 在 test 与 adversarial 文件都生成后，用预先冻结的数值和静态成本 gate 复核：

```powershell
conda run -n neural-shading ncls learn gate-evaluator `
  --run-manifest artifacts\runs\e1-dense-001\run_manifest.json `
  --test-metrics artifacts\runs\e1-dense-001\test_metrics.json `
  --adversarial-metrics artifacts\runs\e1-dense-001\adversarial_metrics.json `
  --gate configs\research\e1-evaluator-gates-v1.json `
  --output artifacts\runs\e1-dense-001\gate_result.json
```

`ncls.e1-single-material-evaluator-acceptance@1` 在正式 sweep 前冻结 normalized L1、log、能量、peak ratio/角度、top-energy recall、相对 reference standard error、互易性、finite/nonnegative 和 `B_asset/C_prepare/C_eval` 上限。失败命令本身仍返回成功并写出逐项检查，表示可复现的研究淘汰证据；只有合同、hash 或输入不一致才抛出错误。E4 的真实 GPU 时间、显存、带宽和 viewer 视觉 gate 不能被这些静态 MAC/byte 上限替代。

E2 使用单独冻结的 `ncls.e2-shared-evaluator-acceptance@1`。它基于 v5 supervision noise floor 与 E1 optimized 单材质上界，放宽跨材质共享所需的质量 envelope，同时强制 `B_asset≤512 bytes`、`B_shared≤512 KiB` 以及 `C_prepare/C_eval≤65,536 MAC`。这个 gate 只筛选共享表示容量；E4 仍需 Python/Slang parity、真实 GPU 时间、显存/带宽和 viewer 对抗性视觉检查。

## Direct fit 的三种结论范围

direct fit 必须在 manifest 中记录 scope：

| scope | 优化对象 | 能支持的结论 |
|---|---|---|
| `direction-slice` | 单个 `(material, wo)` 的方向响应 | 某个 `wi` 切片的局部容量诊断 |
| `material-function` | 一个材质跨全部训练 `wo/wi` 的 latent/evaluator | 单材质完整方向函数的容量 |
| `shared-decoder` | 多材质共享 evaluator + 每材质 latent | 统一 neural representation 的容量 |

只有后二者直接服务当前 neural material program。随后把 feed-forward compiler 的预测结果与相同 evaluator 下的 optimized latent 比较，才能把“表示不够”与“compiler 学不到”分开。

下面的旧命令只重跑当前可部署基线的 `direction-slice` 诊断：

```powershell
conda run -n neural-shading ncls learn direct-fit `
  --dataset data\reference-responses\layer-stack-v4.h5 `
  --output artifacts\direct-fit\legacy-ltc-k2-test `
  --split test --family ltc --lobes 2 `
  --steps 800 --restarts 3
```

输出 `ncls.representation-ceiling@1` manifest、TensorBoard 和 `parameters.npz`。这里的参数是逐 query group 直接优化结果，只能判断一张方向切片，不能证明目标 view-conditioned evaluator，也不能当成通用材质编译器。新的 neural direct-fit manifest 必须新增并锁定上述 scope。

## MethodBundle

训练 run 与部署 artifact 分离。当前 exporter 会把显式 checkpoint、feature contract、后端 shader、validation 指标、固定 parity probe 和全部内容哈希写成一个新 bundle：

```powershell
conda run -n neural-shading ncls bundle export-legacy-ltc-k2 `
  --checkpoint artifacts\runs\legacy-ltc-k2-p1-001\checkpoints\best.pt `
  --run-manifest artifacts\runs\legacy-ltc-k2-p1-001\run_manifest.json `
  --output artifacts\bundles\legacy-ltc-k2-p1-001

conda run -n neural-shading ncls bundle validate `
  artifacts\bundles\legacy-ltc-k2-p1-001
```

P1 compiler 已有无 Python 运行时依赖的 Slang 实现：它读取 bundle 中按小端 float32 规范化的 `weights/model.bin`，执行与 PyTorch 相同的 token MLP、GRUCell、view 编码和参数 head，再生成私有的 `legacy-ltc-k2` scattering state。逐层 PyTorch/Slang parity、固定 bundle probe 和 viewer GPU parity 均已通过，因此 exporter 输出 `runtime_class=realtime`。

这条可部署链路让后续研究可以集中在 latent、共享 `prepare` state 和逐方向 neural evaluator 上。viewer 对所有方法使用公共 `prepare/evaluate` 和 capability；目标 evaluator、后续 matched `sample/pdf` 与 integration head 不改变主可见性或 MethodBundle 生命周期，但各自必须扩展准确的 feature、cost 和 capability 声明。
