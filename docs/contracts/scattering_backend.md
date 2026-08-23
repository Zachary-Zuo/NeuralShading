# 散射后端合同

## 目的

本文定义实时光照、路径追踪、数据采集和学习评测共同使用的散射语义。接口以 `prepare + evaluate` 为基础，并通过 capability 增加 sampling、环境光、面光、透射等能力。目标 backend 用小型 MLP 在 `evaluate()` 中直接实现方向散射；`prepare()` 获取和过滤 latent，并形成同一着色点多次方向查询共享的 view-conditioned state。解析 backend 继续作为合同测试和性能基线。

Falcor 8.0/Slang 2024.1.34 的可编译合同位于 `shaders/ncls/contracts/`，Python 对应物位于 `src/ncls/core/scattering/`。本文冻结方向、测度和 capability 语义；物理 buffer 对齐与接口拆分的迁移由单独的生成 ABI 决定。完整问题与运行时图只在 `docs/realtime_material_compilation.md` 维护，本文不建立第二套研究定义。

## 整个材质接口与散射接口的边界

散射后端只是完整 deferred 材质接口的一部分。renderer 按以下稳定阶段处理材质：

```text
evaluate_visibility()      opacity / alpha test
evaluate_surface_frame()   shading normal、tangent、bitangent
evaluate_emission()        自发光
prepare_scattering()       生成不透明 ScatteringState
evaluate()                 已知入射方向时求局部表面散射
optional sample()/pdf()    由材质选择下一方向时使用
optional integrate_*()     面光、环境光等专用积分
```

位移在主可见性之前的几何阶段处理；interior/exterior medium 在路径跨越边界时交给 volume 阶段。它们都由 `MaterialProgram.outputs` 表达，但不能塞进 BSDF 返回值。

v1 中 opacity 固定为 1、emission 为 0、displacement 为空，surface frame 只来自几何切线和常量旋转。尽管如此，上述阶段边界从第一版就固定。法线图或 alpha texture 加入后只扩展公共材质求值器，不改变散射后端签名。

运行时 backend 只负责 `prepare_scattering` 及其后的散射操作。reference 和右侧方法必须共享 visibility、surface frame 和 emission 结果，避免把非散射差异误计为材质表示误差。

## 方向和数值约定

- `wo`：从着色点指向相机或上一条路径顶点的出射方向；
- `wi`：从着色点指向光源或下一条路径顶点的入射方向；
- 两者都指向远离着色点的一侧；
- 世界空间和局部空间转换只能通过 `NclsShadingFrame` 完成；
- `evaluate()` 返回不含几何余弦的 BSDF `f(wo, wi)`；
- renderer 在积分时恰好乘一次 `abs(dot(shadingNormal, wi))`；
- PDF 相对于立体角；delta 事件的连续 PDF 为 0，并通过 event flag 表达；
- RGB 值为线性工作空间，必须有限；非 delta 的物理散射值必须非负。

Falcor `IBSDF` 当前把余弦包含在返回值中，并使用不同的参数命名习惯。项目必须通过一个集中 adapter 转换，不能让 Falcor 约定泄漏到数据合同或 Python API。

## 公共结构

```text
NclsShadingFrame
  normal
  tangent
  bitangent

NclsSurfaceInteraction
  position
  geometricNormal
  shadingFrame
  uv
  uvDx / uvDy
  materialInstanceId
  primitiveId
  frontFacing

NclsScatteringContext
  surface
  woWorld
  transportMode          Radiance / Importance
  componentMask
```

`geometricNormal` 用于 sidedness 和防止 shading-normal 漏光，`shadingFrame.normal` 用于实际 BSDF。backend 不得假定 tangent 与某个世界轴对齐。

事件标志至少保留：

```text
Reflection
Transmission
Diffuse
Glossy
Delta
FrontSide
BackSide
VolumeBoundary
```

v1 backend 可以只返回 `Reflection`，但不能重新分配标志位。

## 求值结果

```text
NclsScatteringPdf
  forward
  reverse

NclsScatteringEval
  f
  pdf                   仅在 ScatteringSampling capability 下有意义
  eventFlags
  valid

NclsScatteringSample
  wiWorld
  weight
  pdf
  eta
  eventFlags
  valid
```

`sample().weight` 定义为该样本对路径吞吐的乘数：

```text
f(wo, wi) * abs(dot(Ns, wi)) / pdf.forward
```

delta 事件允许直接返回有限的离散事件权重。调用方不能通过重新求值 `f/pdf` 猜测 delta 权重。

`pdf.reverse` 为未来的双向方法、互易性诊断和可微渲染保留。无法提供反向 PDF 的 v1 backend 可以在 descriptor 中声明不支持；不能返回未经说明的近似值。

只声明方向求值能力的 backend 不需要制造 PDF；物理 ABI 暂时仍包含该字段时写 0，并由 descriptor 表明它不可用于 sampling 或 MIS。调用方必须先检查 capability，不能从数值猜测支持状态。

## Slang 语义接口

语义接口按能力分层，并使用 Slang associated type 和编译期 specialization。热路径不使用运行时虚调用；viewer 切换方法时切换 program/pipeline variant。

```c
interface INclsScatteringState
{
    NclsScatteringEval evaluate<S : ISampleGenerator>(
        float3 wiWorld,
        inout S sampleGenerator);
}

interface INclsSampleableScatteringState : INclsScatteringState
{
    bool sample<S : ISampleGenerator>(
        out NclsScatteringSample result,
        inout S sampleGenerator);

    NclsScatteringPdf pdf(float3 wiWorld);
}

interface INclsScatteringBackend
{
    associatedtype CompiledMaterial;
    associatedtype State : INclsScatteringState;

    State prepare(
        NclsScatteringContext context,
        CompiledMaterial material);
}
```

当前可编译 ABI 可以在迁移期间保留包含三个函数的 superset interface，但 `sample/pdf` 的可调用性只由 capability 决定。后续物理接口拆分不能改变已有方向和返回测度。

`CompiledMaterial` 和 `State` 都是 backend-specific associated type。目标 neural backend 的 `CompiledMaterial` 通常保存材质 code、spatial latent texture、LOD/过滤元数据和 provenance；共享网络权重由 backend 实例或 shader parameter block 绑定。公共合同不规定 latent 维数、权重布局或 state 字段。

`State` 捕获当前着色点、过滤后的 latent、局部 frame、footprint 和 `woWorld` 的可复用编码，所以后续逐灯或同一 ray hit 的查询不重复执行这些工作。它还可以缓存 matched sampler 的 proposal 参数，但实际随机方向生成仍由 `sample()` 完成。`evaluate()` 接收 sample generator 是为了让随机 reference 能在同一语义接口下做内部 Monte Carlo；确定性 neural evaluator 和解析基线忽略它。

目标 neural backend 的操作约束为：

- `prepare()` 读取 latent texture/material code，依据 UV footprint 选择和过滤 mip/LOD，处理 shading frame，编码 `woWorld` 并运行 shared trunk；它不消费用于选择下一路径方向的随机数；
- `evaluate()` 接收 `State` 和 `wiWorld`，由核心 evaluator MLP 直接返回不含余弦的 `f(wo, wi)`；
- `sample()` 使用 `State` 中的 proposal 参数或调用 `SamplerHead(State)`，再消费随机数生成实际方向和完整 sample result；
- `pdf()` 计算上述同一 proposal 的立体角密度，不能用另一个未匹配近似代替；
- `integrate_*()` 若存在，必须声明 light descriptor、预算和它所近似的 evaluator 积分。

`State` 只能在材质状态、surface/UV footprint、shading frame 和 `woWorld` 均未改变时复用。它可以跨多个 `wiWorld` 查询复用，不能跨不兼容的 pixel、ray hit 或观察方向复用。

如果 Slang 版本对某个泛型组合有限制，adapter 可以改写语法，但不能改变上述语义、方向和返回值。

## CPU/主机侧生命周期

每个运行时 backend 提供以下主机操作：

```text
describe() -> BackendDescriptor
compile_material(MaterialProgram, resources) -> CompiledMaterial
create_runtime(device, MethodBundle) -> BackendRuntime
prepare_frame(scene_revision, dimensions)
dispatch_prepare(gbuffer, output_state)
```

`CompiledMaterial` 是 view-independent 的静态结果。目标 neural 方法把源材质编译为材质 code、spatial latent、共享 decoder 所需元数据或这些内容的组合。`dispatch_prepare()` 依据 GBuffer 中的着色点、footprint 和观察方向获取/过滤 latent 并生成 per-pixel state；ray-hit shader 执行同样的逻辑。backend 也可以选择不落盘、直接在 lighting shader 中 inline prepare。

## BackendDescriptor

```text
BackendDescriptor
  backend_id
  backend_version
  scattering_contract_version
  supported_material_ir
  supported_capabilities
  state_storage_mode       inline / structured / raw
  state_stride
  state_alignment
  deterministic_eval
  bounded_execution
  shader_entry_points
  cost_model
```

`state_stride` 是方法描述，不是公共固定常数。renderer 依据 descriptor 分配资源，但从不解释内容。`inline` 模式允许 `state_stride=0`。

基础 capability 为：

```text
DirectionalEvaluation       prepare + evaluate
ScatteringSampling          sample + forward pdf
ReversePdf                  reverse pdf
PathTracingCompatible       完整 PT 所需的事件、权重与 sampling 组合
```

`PathTracingCompatible` 至少要求 `ScatteringSampling`，并要求 delta、透射和介质边界等实际声明能力的语义完整；它不能仅因为函数入口存在就自动成立。

`cost_model` 至少记录 CompiledMaterial/latent bytes、prepare 的纹理与编码成本、每次 evaluator MLP 的网络结构/精度/成本、可选 sample/pdf head、状态 bytes/pixel，以及每种专用积分器的查询预算。实际 GPU 时间由 viewer benchmark 补充。

能称为实时部署候选的 backend 必须声明 `bounded_execution=true`：latent/state 大小、网络结构、单次 evaluate 成本和内部循环都具有显式上界，不随源材质图深度、层间随机游走次数或未受控随机状态增长。整帧时间可以按可见像素、实际灯数和 renderer 配置的固定积分 query 预算增长；这些 scaling curve 在 evaluator 完成最小部署后进入 Pareto 报告。尚未满足这一点的方法可以作为建模或诊断 backend，但不能进入实时排名。

## 必需能力

所有可作为实时右侧方法的 backend 必须支持：

- `prepare`；
- `evaluate`；
- 各向异性 shading frame；
- 有限值、事件标志和 capability 查询。

`sample/pdf` 不是主 deferred lighting 的强制能力。声明 `ScatteringSampling` 或 `PathTracingCompatible` 的 neural backend 必须提供与 evaluator 共用 `State`、实际采样分布可计算且相互匹配的 `sample/pdf`。sampler head 可以预测一个可解析混合分布，也可以使用其他具有 tractable density 的方向模型；只输出方向而不能计算实际 PDF 不满足该 capability。没有专用重要性 sampler 时可以使用通用 proposal，但必须报告效率和适用事件。未声明 sampling 的 backend 仍可作为完整 deferred 方法，只是不能进入要求该 capability 的 renderer 路径。

## 可选光照积分能力

以下是 capability，不是基础状态字段：

```text
AnalyticPolygonIntegration
PrefilteredEnvironmentIntegration
NeuralEnvironmentIntegration
DeltaEvents
Transmission
HomogeneousVolume
```

对应的专用 lighting adapter 可以提供：

```text
integrate_polygon_light(...)
integrate_environment(...)
```

缺少专用能力时，renderer 可以用固定预算的方向集合调用 `evaluate`；若 backend 同时支持 `ScatteringSampling`，也可以使用 stochastic 积分。fallback 的质量和成本必须显式报告。专用积分器通过 capability 暴露，renderer 不读取其内部解析参数或 latent。

## 目标 neural material program

目标方法遵循以下语义分工：

```text
compile_material(source)
  -> material/spatial latent + decoder metadata

prepare(surface, footprint, wo, compiled)
  -> filtered latent + view-conditioned shared state h

evaluate(h, wi)
  -> EvaluateMLP 直接输出 f(wo, wi)
```

evaluator 可以在解析 physical core 上预测 residual，以改善动态范围、能量或极窄反射；但 residual 仍在逐方向 `evaluate()` 中执行，目标方法不退化为只在 `prepare()` 预测固定 closure 参数。

Path-tracing profile 在 evaluator 确定后增加：

```text
SamplerHead(h) -> tractable proposal parameters
sample(h, random) -> wi, weight, pdf
pdf(h, wi) -> the density of the same proposal
```

环境或面光 integration head 可以复用 `h`，但其语义必须是对当前 evaluator 的有界积分近似。若输出混入场景阴影或间接光，必须声明为独立 light-transport capability，不能仍称为局部材质散射。

MethodBundle 决定 shader、权重、latent 资源和 capability。renderer 只看到 descriptor 和接口实现。解析 closure backend 通过同一合同参与对照，但不约束 neural backend 的 latent、state 或输出词汇。

## reference adapter

每个源材质族的 reference adapter 实现同样的方向、事件和输出测度语义，但保留该源材质的原生参数、资源和求值算法。reference 至少提供方向求值；完整场景 reference path tracing 还要求该材质族提供 sampling capability。reference 可以是确定性的解析/查表实现，也可以像当前随机游走实现一样在 `evaluate()` 内部使用随机样本。

随机 reference 允许：

- `evaluate()` 内部使用随机样本；
- 每像素随时间累积；
- sample count 和方差由 reference 调用方记录；
- 运行成本不满足实时 backend 的固定预算。

reference 直接求值源材质 GT，不经过目标 neural material backend。它不生成通用 packet，也不参与 MethodBundle 的部署成本比较。不同材质族的 reference 不要求共享内部 IR、shader 或资源布局。

## 一致性测试

每个可导出的 evaluator backend 至少通过：

- Python/Slang 对相同物理状态的求值一致性；
- `prepare/evaluate` 的方向和测度一致性；
- 各向异性旋转测试；
- 有限、非负和事件标志测试；
- 白炉或能量诊断；
- 方法 bundle 加载后与训练端导出的响应一致；
- Falcor adapter 恰好处理一次余弦。

这些检查分阶段启用：建模原型先通过方向/测度、有限值和 reference 响应检查；形成 Slang 最小部署后再要求 Python/Slang、bundle 和 GPU parity。声明 `ScatteringSampling` 的 backend 额外通过 `evaluate/sample/pdf` 一致性、sampled distribution、delta 权重和 MIS 测度测试；声明专用环境或面光积分的 backend 额外与高样本 `evaluate` 积分对照。sampler 与 integration 测试不能先于 evaluator 模型定义。
