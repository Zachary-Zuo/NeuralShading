# Neural evaluator 建模、编译器训练与评测

训练、direct fit、独立评测和部署导出是项目的长期功能层。具体 representation、网络结构、latent 获取方式和 loss 通过版本化配置与注册实现替换；更换研究候选不会删除或重定义这些生命周期。

## 三条路径的边界

Python 侧的工具生命周期仍分成三条路径，但必须记录拟合对象和结论范围：

- `direct-fit`：不经过 feed-forward compiler，直接优化候选表示的参数或 latent；它不自动等于“逐 tile 拟合”，也不产生通用编译器。
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

所有超参数先解析为 `ncls.training-config@1`，并将完整 JSON 与 SHA-256 写入 run。最小示例：

```json
{
  "architecture_id": "legacy-ltc-k2-p1@2",
  "representation_id": "legacy-ltc-k2@1",
  "width": 64,
  "steps": 10000,
  "batch_size": 256,
  "learning_rate": 0.0003,
  "weight_decay": 0.00001,
  "gradient_clip": 5.0,
  "validation_interval": 250,
  "checkpoint_interval": 250,
  "max_validation_tiles": 4096,
  "seed": 20260822,
  "device": "cuda",
  "deterministic": true,
  "schema_name": "ncls.training-config",
  "schema_version": 1
}
```

运行：

```powershell
conda run -n neural-shading ncls learn train `
  --dataset data\reference-v2 `
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

checkpoint 保存 architecture、representation、feature contract、dataset ID、模型/optimizer/RNG 状态和 validation 证据。loader 总是先验证 sidecar hash。

## TensorBoard

训练记录 `train/loss`、gradient norm、learning rate，以及 validation loss、median/p90/p95 relative-L1。不会写入 `test/*` tag。

```powershell
conda run -n neural-shading tensorboard `
  --logdir artifacts\runs\legacy-ltc-k2-p1-001\tensorboard
```

逐样本直接拟合也在其输出目录下记录 `tensorboard/`，便于比较不同表示与优化设置。

## 显式 held-out test

best checkpoint 确定后，单独执行：

```powershell
conda run -n neural-shading ncls learn evaluate `
  --dataset data\reference-v2 `
  --checkpoint artifacts\runs\legacy-ltc-k2-p1-001\checkpoints\best.pt `
  --split test `
  --output artifacts\runs\legacy-ltc-k2-p1-001\test_metrics.json
```

评测拒绝 dataset ID 或 feature contract 不一致的 checkpoint。test 结果是独立文件，不回写训练选择记录。

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
  --dataset data\reference-v2 `
  --output artifacts\direct-fit\legacy-ltc-k2-test `
  --split test --family ltc --lobes 2 `
  --steps 800 --restarts 3
```

输出 `ncls.representation-ceiling@1` manifest、TensorBoard 和 `parameters.npz`。这里的参数是逐 tile 直接优化结果，只能判断一张方向切片，不能证明目标 view-conditioned evaluator，也不能当成通用材质编译器。新的 neural direct-fit manifest 必须新增并锁定上述 scope。

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
